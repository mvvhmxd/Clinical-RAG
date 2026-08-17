"""Embedding providers for document indexing and query retrieval."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ng12_rag.config import Settings

LOGGER = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when no configured embedding provider can produce vectors."""


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Return row-normalised float32 vectors for cosine search via inner product."""

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(values / norms, dtype=np.float32)


@dataclass(frozen=True)
class EmbeddingDescriptor:
    provider: str
    model: str
    dimension: int
    normalised: bool


class EmbeddingProvider(ABC):
    """Minimal provider contract shared by indexing and retrieval."""

    provider_name: str
    model_name: str

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed passages in corpus order."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed one search query as a one-dimensional vector."""

    def descriptor(self, dimension: int) -> EmbeddingDescriptor:
        return EmbeddingDescriptor(
            provider=self.provider_name,
            model=self.model_name,
            dimension=dimension,
            normalised=True,
        )


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings with deterministic batching and explicit base URL support."""

    provider_name = "openai"

    def __init__(self, model: str, batch_size: int, timeout_seconds: float) -> None:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingError("OPENAI_API_KEY is not configured")
        base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        kwargs: dict[str, object] = {
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": 2,
        }
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        self.client = OpenAI(**kwargs)
        self.model_name = model
        self.batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError("Cannot embed an empty document sequence")
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            rows.extend(item.embedding for item in ordered)
        return l2_normalise(np.asarray(rows, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=[text])
        return l2_normalise(np.asarray(response.data[0].embedding))[0]


class LocalSentenceTransformerProvider(EmbeddingProvider):
    """Local BGE embeddings; no API key or network is needed after model caching."""

    provider_name = "local_sentence_transformer"

    def __init__(
        self,
        model: str,
        batch_size: int,
        query_prefix: str,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is required for the local embedding provider"
            ) from exc
        self.model_name = model
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.device = device or os.getenv("NG12_EMBEDDING_DEVICE", "cpu")
        LOGGER.info("Loading local embedding model %s on %s", model, self.device)
        self.model = SentenceTransformer(
            model,
            device=self.device,
            trust_remote_code=False,
        )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError("Cannot embed an empty text sequence")
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return l2_normalise(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]


def create_embedding_provider(
    settings: Settings,
    *,
    provider_override: str | None = None,
    model_override: str | None = None,
    probe: bool = True,
) -> EmbeddingProvider:
    """Create the requested provider, or probe OpenAI before a local fallback."""

    config = settings.section("embedding")
    provider = (provider_override or settings.embedding_provider).casefold()
    batch_size = int(config["batch_size"])
    timeout = float(config["request_timeout_seconds"])

    def openai_provider() -> OpenAIEmbeddingProvider:
        return OpenAIEmbeddingProvider(
            model=model_override or str(config["openai_model"]),
            batch_size=batch_size,
            timeout_seconds=timeout,
        )

    def local_provider() -> LocalSentenceTransformerProvider:
        return LocalSentenceTransformerProvider(
            model=model_override or str(config["local_model"]),
            batch_size=batch_size,
            query_prefix=str(config.get("local_query_prefix", "")),
        )

    if provider == "openai":
        return openai_provider()
    if provider in {"local", "local_sentence_transformer", "sentence_transformer"}:
        return local_provider()
    if provider != "auto":
        raise EmbeddingError(f"Unsupported embedding provider: {provider}")

    try:
        candidate = openai_provider()
        if probe:
            candidate.embed_query("NG12 embedding provider health check")
        LOGGER.info("Using OpenAI embedding model %s", candidate.model_name)
        return candidate
    except Exception as exc:
        LOGGER.warning(
            "OpenAI embeddings unavailable (%s); falling back to local model %s",
            exc,
            config["local_model"],
        )
        try:
            return local_provider()
        except Exception as local_exc:
            raise EmbeddingError(
                "Neither OpenAI nor local sentence-transformer embeddings are available"
            ) from local_exc
