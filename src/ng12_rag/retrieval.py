"""Hybrid FAISS + BM25 retrieval with transparent score fusion and reranking."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ng12_rag.chunking import read_chunks
from ng12_rag.config import Settings
from ng12_rag.embeddings import EmbeddingProvider, create_embedding_provider
from ng12_rag.indexing import corpus_fingerprint
from ng12_rag.models import ComponentScores, RecommendationChunk, RetrievedChunk
from ng12_rag.text import clinical_tokenize, numeric_tokens, recommendation_ids

LOGGER = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """Raised when index artifacts are absent or incompatible."""


def _min_max(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _ordered_top_indices(scores: np.ndarray, limit: int) -> list[int]:
    if scores.ndim != 1:
        raise RetrievalError(f"Expected one-dimensional scores, got {scores.shape}")
    indices = np.argsort(-scores, kind="stable")[:limit]
    return [int(index) for index in indices if np.isfinite(scores[index])]


class ClinicalFeatureReranker:
    """Deterministic exact-feature score used alone or beside a cross-encoder."""

    ACTION_TERMS = {
        "refer",
        "referral",
        "offer",
        "consider",
        "investigation",
        "xray",
        "ct",
        "ultrasound",
        "endoscopy",
        "fit",
        "psa",
    }

    @staticmethod
    def score(query: str, chunk: RecommendationChunk) -> tuple[float, bool]:
        query_tokens = set(clinical_tokenize(query))
        document_tokens = set(clinical_tokenize(chunk.embedding_text))
        if not query_tokens:
            return 0.0, False

        overlap = len(query_tokens & document_tokens) / len(query_tokens)
        query_ids = recommendation_ids(query)
        id_match = bool(query_ids and chunk.metadata.recommendation_id in query_ids)

        query_numbers = numeric_tokens(query) - query_ids
        document_numbers = numeric_tokens(chunk.text) - {
            chunk.metadata.recommendation_id
        }
        numeric_match = bool(query_numbers and query_numbers.issubset(document_numbers))
        numeric_conflict = bool(
            query_numbers
            and document_numbers
            and not query_numbers.issubset(document_numbers)
        )
        exact_thresholds = set(
            re.findall(
                r"\b(?:exactly|equal\s+to|equals)\s+(\d+(?:\.\d+)?)\b",
                query.casefold(),
            )
        )
        document_lower = chunk.text.casefold()
        if any(
            re.search(rf"\bbelow\s+{re.escape(value)}\b", document_lower)
            for value in exact_thresholds
        ):
            numeric_conflict = True

        action_query = query_tokens & ClinicalFeatureReranker.ACTION_TERMS
        action_overlap = (
            len(action_query & document_tokens) / len(action_query)
            if action_query
            else 0.0
        )
        site_tokens = set(clinical_tokenize(chunk.metadata.cancer_site.replace("_", " ")))
        site_match = 1.0 if query_tokens & site_tokens else 0.0

        score = (
            0.45 * overlap
            + 0.20 * action_overlap
            + 0.15 * site_match
            + 0.10 * float(numeric_match)
            + 0.10 * float(id_match)
        )
        return min(max(score, 0.0), 1.0), numeric_conflict


class CrossEncoderReranker:
    """Optional semantic reranker with deterministic feature-only fallback."""

    def __init__(self, model_name: str, provider: str) -> None:
        self.model_name = model_name
        self.model = None
        if provider not in {"auto", "cross_encoder"}:
            return
        try:
            from sentence_transformers import CrossEncoder

            LOGGER.info("Loading cross-encoder reranker %s", model_name)
            self.model = CrossEncoder(model_name, trust_remote_code=False)
        except Exception as exc:
            if provider == "cross_encoder":
                raise RetrievalError(f"Could not load cross-encoder {model_name}: {exc}") from exc
            LOGGER.warning("Cross-encoder unavailable; using deterministic reranker: %s", exc)

    def score(self, query: str, chunks: Sequence[RecommendationChunk]) -> list[float] | None:
        if self.model is None or not chunks:
            return None
        pairs = [(query, chunk.embedding_text) for chunk in chunks]
        raw = self.model.predict(pairs, show_progress_bar=False)
        return [_sigmoid(float(value)) for value in np.asarray(raw).reshape(-1)]


class HybridRetriever:
    """Load persistent indexes and return auditable, reranked recommendations."""

    def __init__(
        self,
        settings: Settings,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.settings = settings
        self.config = settings.section("retrieval")
        self.negative_config = settings.section("negative_chunks")
        self.manifest = self._load_json(settings.path("vector_store.manifest_path"))
        self.chunks = read_chunks(settings.path("vector_store.metadata_path"))
        self.faiss_index = faiss.read_index(str(settings.path("vector_store.index_path")))
        self.bm25_payload = self._load_json(
            settings.path("vector_store.bm25_corpus_path")
        )
        self._validate_artifacts()
        tokenized_corpus = self.bm25_payload["tokenized_corpus"]
        self.bm25 = BM25Okapi(tokenized_corpus)

        embedding_manifest = self.manifest["embedding"]
        self.embedding_provider = embedding_provider or create_embedding_provider(
            settings,
            provider_override=str(embedding_manifest["provider"]),
            model_override=str(embedding_manifest["model"]),
            probe=False,
        )
        reranker_config = settings.section("reranker")
        self.cross_encoder = CrossEncoderReranker(
            str(reranker_config["cross_encoder_model"]),
            str(reranker_config["provider"]),
        )
        self.reranker_config = reranker_config

    @staticmethod
    def _load_json(path: Path) -> dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RetrievalError(f"Required index artifact is missing: {path}") from exc
        if not isinstance(payload, dict):
            raise RetrievalError(f"Expected JSON object in {path}")
        return payload

    def _validate_artifacts(self) -> None:
        expected_ids = [chunk.chunk_id for chunk in self.chunks]
        manifest_ids = self.manifest.get("ordered_chunk_ids")
        bm25_ids = self.bm25_payload.get("chunk_ids")
        if manifest_ids != expected_ids or bm25_ids != expected_ids:
            raise RetrievalError("FAISS, BM25, and metadata chunk ordering differ")
        fingerprint = corpus_fingerprint(self.chunks)
        if fingerprint != self.manifest.get("corpus_fingerprint"):
            raise RetrievalError("Metadata corpus fingerprint does not match index manifest")
        if fingerprint != self.bm25_payload.get("corpus_fingerprint"):
            raise RetrievalError("BM25 corpus fingerprint does not match index manifest")
        if self.faiss_index.ntotal != len(self.chunks):
            raise RetrievalError("FAISS vector count does not match metadata count")
        expected_dimension = int(self.manifest["embedding"]["dimension"])
        if self.faiss_index.d != expected_dimension:
            raise RetrievalError("FAISS dimension does not match embedding manifest")

    def _vector_results(self, query: str) -> tuple[dict[int, float], dict[int, int]]:
        query_vector = np.ascontiguousarray(
            self.embedding_provider.embed_query(query).reshape(1, -1),
            dtype=np.float32,
        )
        limit = min(int(self.config["vector_top_k"]), len(self.chunks))
        scores, indices = self.faiss_index.search(query_vector, limit)
        score_map: dict[int, float] = {}
        rank_map: dict[int, int] = {}
        for rank, (index, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if index < 0:
                continue
            score_map[int(index)] = float(score)
            rank_map[int(index)] = rank
        return score_map, rank_map

    def _bm25_results(self, query: str) -> tuple[dict[int, float], dict[int, int]]:
        tokens = clinical_tokenize(query)
        if not tokens:
            raise RetrievalError("Query contains no searchable terms")
        raw = np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)
        limit = min(int(self.config["bm25_top_k"]), len(self.chunks))
        indices = _ordered_top_indices(raw, limit)
        return (
            {index: float(raw[index]) for index in indices},
            {index: rank for rank, index in enumerate(indices, start=1)},
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        include_synthetic_negatives: bool = False,
    ) -> list[RetrievedChunk]:
        """Retrieve, fuse, rerank, log, and return recommendation chunks."""

        query = query.strip()
        if not query:
            raise RetrievalError("Query must not be empty")
        vector_scores, vector_ranks = self._vector_results(query)
        bm25_scores, bm25_ranks = self._bm25_results(query)
        candidates = set(vector_scores) | set(bm25_scores)
        rrf_k = float(self.config["reciprocal_rank_fusion_k"])
        vector_weight = float(self.config["vector_weight"])
        bm25_weight = float(self.config["bm25_weight"])
        exact_ids = recommendation_ids(query)
        query_numbers = numeric_tokens(query) - exact_ids

        components: dict[int, ComponentScores] = {}
        for index in candidates:
            rrf = 0.0
            if index in vector_ranks:
                rrf += vector_weight / (rrf_k + vector_ranks[index])
            if index in bm25_ranks:
                rrf += bm25_weight / (rrf_k + bm25_ranks[index])
            chunk = self.chunks[index]
            if chunk.metadata.recommendation_id in exact_ids:
                rrf += float(self.config["exact_recommendation_id_boost"])
            document_numbers = numeric_tokens(chunk.text) - {
                chunk.metadata.recommendation_id
            }
            if query_numbers and query_numbers.issubset(document_numbers):
                rrf += float(self.config["exact_numeric_match_boost"])
            feature_score, numeric_conflict = ClinicalFeatureReranker.score(query, chunk)
            if numeric_conflict:
                rrf -= float(self.config["numeric_conflict_penalty"])
            components[index] = ComponentScores(
                vector_score=vector_scores.get(index),
                vector_rank=vector_ranks.get(index),
                bm25_score=bm25_scores.get(index),
                bm25_rank=bm25_ranks.get(index),
                rrf_score=rrf,
                feature_score=feature_score,
                numeric_conflict=numeric_conflict,
            )

        fusion_limit = min(int(self.config["fusion_top_k"]), len(components))
        fused_indices = sorted(
            components,
            key=lambda index: (
                components[index].rrf_score,
                components[index].feature_score,
                self.chunks[index].chunk_id,
            ),
            reverse=True,
        )[:fusion_limit]

        rerank_limit = min(int(self.config["rerank_top_k"]), len(fused_indices))
        rerank_indices = fused_indices[:rerank_limit]
        rerank_chunks = [self.chunks[index] for index in rerank_indices]
        cross_scores = self.cross_encoder.score(query, rerank_chunks)
        normalised_fused = _min_max(
            {index: components[index].rrf_score for index in rerank_indices}
        )

        for position, index in enumerate(rerank_indices):
            feature = components[index].feature_score
            semantic = cross_scores[position] if cross_scores is not None else feature
            final_score = (
                float(self.reranker_config["cross_encoder_weight"]) * semantic
                + float(self.reranker_config["fused_score_weight"])
                * normalised_fused[index]
                + float(self.reranker_config["clinical_feature_weight"]) * feature
            )
            if self.chunks[index].metadata.is_synthetic_negative:
                final_score -= float(self.negative_config["rerank_penalty"])
            components[index] = components[index].model_copy(
                update={
                    "reranker_score": semantic,
                    "final_score": final_score,
                }
            )

        ranked_indices = sorted(
            rerank_indices,
            key=lambda index: (
                components[index].final_score,
                components[index].rrf_score,
                self.chunks[index].chunk_id,
            ),
            reverse=True,
        )
        minimum_score = float(self.config["minimum_fused_score"])
        ranked_indices = [
            index
            for index in ranked_indices
            if components[index].rrf_score >= minimum_score
            and (
                include_synthetic_negatives
                or not self.chunks[index].metadata.is_synthetic_negative
            )
        ]
        result_limit = top_k or int(self.config["final_top_k"])
        results = [
            RetrievedChunk(
                chunk=self.chunks[index],
                scores=components[index],
                rank=rank,
            )
            for rank, index in enumerate(ranked_indices[:result_limit], start=1)
        ]
        self._log_search(query, results)
        return results

    def _log_search(self, query: str, results: Sequence[RetrievedChunk]) -> None:
        if not bool(self.config.get("log_scores", True)):
            return
        path = self.settings.path("retrieval.retrieval_log_path")
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "query": query,
            "results": [
                {
                    "rank": result.rank,
                    "chunk_id": result.chunk.chunk_id,
                    "recommendation_id": result.chunk.metadata.recommendation_id,
                    "is_synthetic_negative": result.chunk.metadata.is_synthetic_negative,
                    "scores": result.scores.model_dump(mode="json"),
                }
                for result in results
            ],
        }
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
