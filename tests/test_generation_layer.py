from __future__ import annotations

from ng12_rag.generation import GenerationConfig, normalize_retrieved_chunks
from ng12_rag.guardrails import validate_response_sources
from ng12_rag.response_schema import (
    CLINICAL_DISCLAIMER,
    Citation,
    ConfidenceLevel,
    DocumentType,
    Evidence,
    FullResponse,
)


def test_typed_citation_formats_are_distinct() -> None:
    recommendation = Citation(
        document_type=DocumentType.NG12_SHORT,
        recommendation_id="1.3.2",
        chapter_number=None,
        chapter_title=None,
        section_title="Colorectal cancer",
        page_number=15,
        quoted_text="Refer adults using a suspected cancer pathway referral",
    )
    full_evidence = Citation(
        document_type=DocumentType.NG12_FULL,
        recommendation_id=None,
        chapter_number="9",
        chapter_title="Colorectal",
        section_title="Lower gastrointestinal tract cancers",
        page_number=115,
        quoted_text="The positive predictive value was reported in the study population",
    )

    assert recommendation.formatted_reference == "NG12, Recommendation 1.3.2, p.15"
    assert (
        full_evidence.formatted_reference
        == "NG12 Full Guideline, Chapter 9 (Colorectal), p.115"
    )


def test_normalizer_accepts_existing_nested_retrieval_shape() -> None:
    raw = {
        "chunk": {
            "chunk_id": "ng12-1-3-2",
            "text": "Refer adults using a suspected cancer pathway referral.",
            "metadata": {
                "recommendation_id": "1.3.2",
                "section_title": "Colorectal cancer",
                "page_number": 15,
                "cancer_site": "Colorectal",
            },
        },
        "scores": {"final_score": 0.88},
        "rank": 1,
    }

    result = normalize_retrieved_chunks([raw], config=GenerationConfig())

    assert result.rejected_reasons == []
    assert result.chunks[0]["document_type"] == "NG12_short"
    assert result.chunks[0]["recommendation_id"] == "1.3.2"
    assert result.chunks[0]["similarity_score"] == 0.88


def test_short_and_full_sources_pass_exact_provenance_validation() -> None:
    short_text = "Refer adults using a suspected cancer pathway referral for colorectal cancer."
    full_text = "The positive predictive value was reported in the study population."
    short_citation = Citation(
        document_type=DocumentType.NG12_SHORT,
        recommendation_id="1.3.2",
        chapter_number=None,
        chapter_title=None,
        section_title="Colorectal cancer",
        page_number=15,
        quoted_text="Refer adults using a suspected cancer pathway referral",
    )
    full_citation = Citation(
        document_type=DocumentType.NG12_FULL,
        recommendation_id=None,
        chapter_number="9",
        chapter_title="Colorectal",
        section_title="Lower gastrointestinal tract cancers",
        page_number=115,
        quoted_text="The positive predictive value was reported in the study population",
    )
    response = FullResponse(
        recommendation_summary=(
            "Referral is stated per NG12, Recommendation 1.3.2, p.15. Evidence context is "
            "reported per NG12 Full Guideline, Chapter 9 (Colorectal), p.115."
        ),
        evidence_list=[
            Evidence(
                claim=(
                    "Referral is stated per NG12, Recommendation 1.3.2, p.15."
                ),
                supporting_citations=[short_citation],
                confidence=ConfidenceLevel.HIGH,
            ),
            Evidence(
                claim=(
                    "The study context reports a positive predictive value per NG12 Full "
                    "Guideline, Chapter 9 (Colorectal), p.115."
                ),
                supporting_citations=[full_citation],
                confidence=ConfidenceLevel.MEDIUM,
            ),
        ],
        overall_confidence=ConfidenceLevel.MEDIUM,
        disclaimer=CLINICAL_DISCLAIMER,
        refusal_reason=None,
        clarifying_question=None,
    )
    chunks = [
        {
            "document_type": "NG12_short",
            "recommendation_id": "1.3.2",
            "chapter_number": None,
            "chapter_title": None,
            "section_title": "Colorectal cancer",
            "page_number": 15,
            "text": short_text,
        },
        {
            "document_type": "NG12_full",
            "recommendation_id": None,
            "chapter_number": "9",
            "chapter_title": "Colorectal",
            "section_title": "Lower gastrointestinal tract cancers",
            "page_number": 115,
            "text": full_text,
        },
    ]

    assert validate_response_sources(response, chunks) == []
