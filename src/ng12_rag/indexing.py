"""Persistent FAISS and BM25 index construction for recommendation chunks."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from ng12_rag.chunking import read_chunks, write_chunks
from ng12_rag.config import Settings
from ng12_rag.embeddings import EmbeddingProvider, create_embedding_provider
from ng12_rag.models import RecommendationChunk
from ng12_rag.text import clinical_tokenize

LOGGER = logging.getLogger(__name__)


class IndexBuildError(RuntimeError):
    """Raised when corpus and index artifacts are inconsistent."""


def corpus_fingerprint(chunks: Iterable[RecommendationChunk]) -> str:
    """Hash ordered chunk content and metadata for compatibility checks."""

    digest = hashlib.sha256()
    for chunk in chunks:
        canonical = json.dumps(
            chunk.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_faiss(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(temporary))
    temporary.replace(path)


def validate_index_inputs(chunks: list[RecommendationChunk]) -> None:
    """Reject duplicate IDs and invalid negative-corpus labelling."""

    if not chunks:
        raise IndexBuildError("Cannot build an index from an empty corpus")
    ids = [chunk.chunk_id for chunk in chunks]
    duplicates = [identifier for identifier, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise IndexBuildError(f"Duplicate chunk IDs: {duplicates}")
    for chunk in chunks:
        if chunk.metadata.is_synthetic_negative and "SYNTHETIC NEGATIVE" not in chunk.text:
            raise IndexBuildError(
                f"Synthetic negative lacks visible safety label: {chunk.chunk_id}"
            )


def build_index(
    settings: Settings,
    *,
    chunks: list[RecommendationChunk] | None = None,
    negatives: list[RecommendationChunk] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, object]:
    """Build and persist aligned FAISS, chunk metadata, and BM25 corpus artifacts."""

    real_chunks = chunks or read_chunks(settings.path("source.chunks_path"))
    negative_chunks = negatives
    if negative_chunks is None:
        negative_path = settings.path("negative_chunks.output_path")
        negative_chunks = read_chunks(negative_path) if negative_path.exists() else []
    include_negatives = bool(settings.section("negative_chunks")["include_in_index"])
    indexed_chunks = real_chunks + (negative_chunks if include_negatives else [])
    validate_index_inputs(indexed_chunks)

    provider = embedding_provider or create_embedding_provider(settings)
    texts = [chunk.embedding_text for chunk in indexed_chunks]
    vectors = provider.embed_documents(texts)
    if vectors.ndim != 2 or vectors.shape[0] != len(indexed_chunks):
        raise IndexBuildError(
            f"Embedding matrix shape {vectors.shape} does not match "
            f"{len(indexed_chunks)} chunks"
        )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise IndexBuildError("Embedding matrix contains non-finite values")

    vector_store = settings.section("vector_store")
    if vector_store["metric"] != "inner_product":
        raise IndexBuildError("Only inner_product is supported for normalised vectors")
    faiss_index = faiss.IndexFlatIP(int(vectors.shape[1]))
    faiss_index.add(vectors)
    if faiss_index.ntotal != len(indexed_chunks):
        raise IndexBuildError("FAISS index size does not match the corpus")

    bm25_tokens = [clinical_tokenize(chunk.embedding_text) for chunk in indexed_chunks]
    if any(not tokens for tokens in bm25_tokens):
        empty = [
            indexed_chunks[index].chunk_id
            for index, tokens in enumerate(bm25_tokens)
            if not tokens
        ]
        raise IndexBuildError(f"BM25 tokenisation produced empty documents: {empty}")

    fingerprint = corpus_fingerprint(indexed_chunks)
    descriptor = provider.descriptor(int(vectors.shape[1]))
    _write_faiss(faiss_index, settings.path("vector_store.index_path"))
    write_chunks(indexed_chunks, settings.path("vector_store.metadata_path"))
    _atomic_json(
        settings.path("vector_store.bm25_corpus_path"),
        {
            "schema_version": "1.0.0",
            "tokenizer": "ng12_clinical_v1",
            "corpus_fingerprint": fingerprint,
            "chunk_ids": [chunk.chunk_id for chunk in indexed_chunks],
            "tokenized_corpus": bm25_tokens,
        },
    )
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "guideline_id": settings.section("project")["guideline_id"],
        "guideline_version": settings.section("project")["guideline_version"],
        "source_sha256": indexed_chunks[0].metadata.source_sha256,
        "corpus_fingerprint": fingerprint,
        "document_count": len(indexed_chunks),
        "real_document_count": len(real_chunks),
        "synthetic_negative_count": len(negative_chunks) if include_negatives else 0,
        "embedding": {
            "provider": descriptor.provider,
            "model": descriptor.model,
            "dimension": descriptor.dimension,
            "normalised": descriptor.normalised,
        },
        "faiss": {
            "index_type": "IndexFlatIP",
            "metric": "cosine_via_normalised_inner_product",
            "ntotal": int(faiss_index.ntotal),
        },
        "bm25": {
            "implementation": "rank_bm25.BM25Okapi",
            "tokenizer": "ng12_clinical_v1",
            "document_count": len(bm25_tokens),
        },
        "ordered_chunk_ids": [chunk.chunk_id for chunk in indexed_chunks],
    }
    _atomic_json(settings.path("vector_store.manifest_path"), manifest)
    LOGGER.info(
        "Built FAISS/BM25 index for %d documents using %s (%d dimensions)",
        len(indexed_chunks),
        descriptor.model,
        descriptor.dimension,
    )
    return manifest
