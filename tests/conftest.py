from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from ng12_rag.config import Settings, load_settings
from ng12_rag.embeddings import EmbeddingProvider, l2_normalise
from ng12_rag.text import clinical_tokenize


class DeterministicEmbeddingProvider(EmbeddingProvider):
    provider_name = "deterministic_test"
    model_name = "token-hash-128"

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in clinical_tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return vector

    def embed_documents(self, texts):
        return l2_normalise(np.vstack([self._embed(text) for text in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalise(self._embed(text))[0]


@pytest.fixture()
def deterministic_provider() -> DeterministicEmbeddingProvider:
    return DeterministicEmbeddingProvider()


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings(project_root: Path) -> Settings:
    return load_settings(project_root / "config" / "config.yaml")


@pytest.fixture()
def isolated_settings(settings: Settings, tmp_path: Path) -> Settings:
    data = copy.deepcopy(settings.data)
    data["source"]["pdf_path"] = str(settings.path("source.pdf_path"))
    data["source"]["raw_recommendations_path"] = "data/processed/raw.jsonl"
    data["source"]["chunks_path"] = "data/processed/chunks.jsonl"
    data["source"]["corpus_manifest_path"] = "data/processed/manifest.json"
    data["negative_chunks"]["output_path"] = "data/processed/negative.jsonl"
    data["vector_store"]["index_path"] = "data/index/vectors.faiss"
    data["vector_store"]["metadata_path"] = "data/index/metadata.jsonl"
    data["vector_store"]["bm25_corpus_path"] = "data/index/bm25.json"
    data["vector_store"]["manifest_path"] = "data/index/index_manifest.json"
    data["retrieval"]["retrieval_log_path"] = "logs/retrieval.jsonl"
    data["reranker"]["provider"] = "deterministic"
    return Settings(data=data, project_root=tmp_path, config_path=tmp_path / "config.yaml")
