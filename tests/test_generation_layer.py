from __future__ import annotations

from ng12_rag.generation import GenerationConfig, normalize_retrieved_chunks
from ng12_rag.guardrails import validate_response_sources
from ng12_rag.response_schema import (
    CLINICAL_DISCLAIMER,
    Citation,
    ConditionEvaluation,
    ConditionLogic,
    ConditionStatus,
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
                condition_evaluations=[
                    ConditionEvaluation(
                        condition_text="a FIT result of at least 10 micrograms of haemoglobin",
                        stated_value="12 micrograms per gram",
                        status=ConditionStatus.MET,
                        at_boundary=False,
                        reasoning="12 is above the inclusive threshold of 10.",
                    )
                ],
                condition_logic=ConditionLogic.SINGLE,
                overall_conclusion=ConditionStatus.MET,
            ),
            Evidence(
                claim=(
                    "The study context reports a positive predictive value per NG12 Full "
                    "Guideline, Chapter 9 (Colorectal), p.115."
                ),
                supporting_citations=[full_citation],
                confidence=ConfidenceLevel.MEDIUM,
                # Rationale-only claims carry no referral conditions to evaluate.
                condition_evaluations=[],
                condition_logic=ConditionLogic.SINGLE,
                overall_conclusion=None,
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


def test_confidence_capping_preserves_condition_evaluations() -> None:
    """Capping rebuilt Evidence field by field and silently dropped the evaluation."""

    from ng12_rag.generation import _cap_answer_confidence

    citation = Citation(
        document_type=DocumentType.NG12_SHORT,
        recommendation_id="1.2.4",
        chapter_number=None,
        chapter_title=None,
        section_title="Upper gastrointestinal tract cancers",
        page_number=12,
        quoted_text="aged 40 and over and have jaundice",
    )
    evaluation = ConditionEvaluation(
        condition_text="aged 40 and over",
        stated_value="39",
        status=ConditionStatus.NOT_MET,
        at_boundary=False,
        reasoning="39 is one year below the threshold of 40.",
    )
    response = FullResponse(
        recommendation_summary=(
            "This patient does not meet the criterion per NG12, Recommendation 1.2.4, p.12."
        ),
        evidence_list=[
            Evidence(
                claim="The criterion is not met per NG12, Recommendation 1.2.4, p.12.",
                supporting_citations=[citation],
                confidence=ConfidenceLevel.HIGH,
                condition_evaluations=[evaluation],
                condition_logic=ConditionLogic.AND,
                overall_conclusion=ConditionStatus.NOT_MET,
            )
        ],
        overall_confidence=ConfidenceLevel.HIGH,
        disclaimer=CLINICAL_DISCLAIMER,
        refusal_reason=None,
        clarifying_question=None,
    )

    capped = _cap_answer_confidence(response, retrieval_cap=ConfidenceLevel.LOW)

    assert capped.evidence_list[0].confidence is ConfidenceLevel.LOW
    assert capped.evidence_list[0].condition_evaluations == [evaluation]
    assert capped.evidence_list[0].overall_conclusion is ConditionStatus.NOT_MET
    assert capped.evidence_list[0].condition_logic is ConditionLogic.AND
