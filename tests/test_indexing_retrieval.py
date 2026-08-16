import json

import faiss

from ng12_rag.chunking import create_chunks, create_negative_chunks, read_chunks
from ng12_rag.indexing import build_index, corpus_fingerprint
from ng12_rag.ingestion import extract_recommendations
from ng12_rag.retrieval import HybridRetriever


def _build(isolated_settings, deterministic_provider):
    records, _ = extract_recommendations(
        isolated_settings.path("source.pdf_path"), isolated_settings
    )
    chunks = create_chunks(records, isolated_settings)
    negatives = create_negative_chunks(chunks, isolated_settings)
    manifest = build_index(
        isolated_settings,
        chunks=chunks,
        negatives=negatives,
        embedding_provider=deterministic_provider,
    )
    return chunks, negatives, manifest


def test_persisted_faiss_bm25_and_metadata_are_aligned(
    isolated_settings, deterministic_provider
):
    chunks, negatives, manifest = _build(isolated_settings, deterministic_provider)
    indexed = read_chunks(isolated_settings.path("vector_store.metadata_path"))
    faiss_index = faiss.read_index(str(isolated_settings.path("vector_store.index_path")))
    bm25 = json.loads(
        isolated_settings.path("vector_store.bm25_corpus_path").read_text(
            encoding="utf-8"
        )
    )

    assert len(indexed) == len(chunks) + len(negatives) == 39
    assert faiss_index.ntotal == len(indexed)
    assert bm25["chunk_ids"] == [chunk.chunk_id for chunk in indexed]
    assert manifest["corpus_fingerprint"] == corpus_fingerprint(indexed)
    assert manifest["real_document_count"] == 33
    assert manifest["synthetic_negative_count"] == 6


def test_hybrid_retrieval_returns_expected_recommendations_and_provenance(
    isolated_settings, deterministic_provider
):
    _build(isolated_settings, deterministic_provider)
    retriever = HybridRetriever(
        isolated_settings, embedding_provider=deterministic_provider
    )

    lung = retriever.search(
        "lung cancer 40 year old smoker with unexplained cough and weight loss", top_k=3
    )
    assert lung[0].chunk.metadata.recommendation_id == "1.1.2"
    assert lung[0].chunk.metadata.page_number == 9
    assert lung[0].scores.vector_score is not None
    assert lung[0].scores.bm25_score is not None

    renal = retriever.search(
        "45 year old unexplained visible haematuria without UTI renal cancer",
        top_k=3,
    )
    assert renal[0].chunk.metadata.recommendation_id == "1.6.6"
    assert renal[0].chunk.metadata.page_numbers == [23]


def test_exact_fit_boundary_prefers_referral_not_below_threshold_clause(
    isolated_settings, deterministic_provider
):
    _build(isolated_settings, deterministic_provider)
    retriever = HybridRetriever(
        isolated_settings, embedding_provider=deterministic_provider
    )
    results = retriever.search(
        "FIT result exactly 10 micrograms haemoglobin per gram faeces",
        top_k=5,
    )
    assert results[0].chunk.metadata.recommendation_id == "1.3.2"
    assert all(result.chunk.metadata.recommendation_id != "1.3.3" for result in results)


def test_synthetic_negative_chunks_are_never_returned_by_default(
    isolated_settings, deterministic_provider
):
    _build(isolated_settings, deterministic_provider)
    retriever = HybridRetriever(
        isolated_settings, embedding_provider=deterministic_provider
    )
    results = retriever.search(
        "30 year old smoker with unexplained cough and weight loss",
        top_k=10,
    )
    assert all(not result.chunk.metadata.is_synthetic_negative for result in results)


def test_retrieval_scores_are_logged_as_jsonl(isolated_settings, deterministic_provider):
    _build(isolated_settings, deterministic_provider)
    retriever = HybridRetriever(
        isolated_settings, embedding_provider=deterministic_provider
    )
    retriever.search("dysphagia oesophageal cancer referral", top_k=2)
    log_path = isolated_settings.path("retrieval.retrieval_log_path")
    event = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["query"] == "dysphagia oesophageal cancer referral"
    assert event["results"][0]["scores"]["final_score"] is not None
