"""The no-LLM path: same verdicts as the live path, and honest about which ran."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ng12_rag.deterministic import (
    ExecutionMode,
    evaluate_chunk_conditions,
    extract_stated_lab_value,
    generate_deterministic_response,
    infer_condition_logic,
)
from ng12_rag.generation import GroundedGenerator
from ng12_rag.response_schema import ConditionLogic, ConditionStatus

CORPUS = Path(__file__).resolve().parents[1] / "data/processed/chunks.jsonl"


@pytest.fixture(scope="module")
def chunks() -> dict[str, dict]:
    records = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]
    return {record["metadata"]["recommendation_id"]: record for record in records}


def as_context(record: dict) -> dict:
    metadata = record["metadata"]
    return {
        "chunk_id": record["chunk_id"],
        "text": record["text"],
        "metadata": metadata,
        "recommendation_id": metadata["recommendation_id"],
        "section_title": metadata["section_title"],
        "subsection_title": metadata.get("subsection_title"),
        "page_number": metadata["page_number"],
        "document_type": "NG12_short",
        "cancer_site": metadata["cancer_site"],
        "similarity": 0.9,
    }


@pytest.mark.parametrize(
    ("question", "recommendation_id", "expected"),
    [
        # The regression case, decided without any model involvement.
        ("A 39-year-old with jaundice.", "1.2.4", ConditionStatus.NOT_MET),
        ("A 40-year-old with jaundice.", "1.2.4", ConditionStatus.MET),
        ("A 41-year-old with jaundice.", "1.2.4", ConditionStatus.MET),
        ("Adult with a FIT result of 10 micrograms per gram.", "1.3.2", ConditionStatus.MET),
        ("Adult with a FIT result of 9 micrograms per gram.", "1.3.2", ConditionStatus.NOT_MET),
    ],
)
def test_conditions_are_evaluated_without_a_model(chunks, question, recommendation_id, expected):
    response, execution = generate_deterministic_response(
        question, [as_context(chunks[recommendation_id])]
    )
    assert execution.mode is ExecutionMode.FALLBACK
    assert response.evidence_list[0].overall_conclusion is expected


class TestGatedAlternatives:
    """1.6.6 is 'aged 45 and over AND (branch OR branch)', not a flat OR."""

    def test_age_gate_failure_defeats_a_satisfied_branch(self, chunks):
        question = "A 44-year-old with unexplained visible haematuria without urinary tract infection."
        response, _ = generate_deterministic_response(question, [as_context(chunks["1.6.6"])])
        evidence = response.evidence_list[0]
        assert evidence.overall_conclusion is ConditionStatus.NOT_MET
        assert evidence.condition_logic is ConditionLogic.AND

    def test_age_gate_met_with_a_satisfied_branch_passes(self, chunks):
        question = "A 50-year-old with unexplained visible haematuria without urinary tract infection."
        response, _ = generate_deterministic_response(question, [as_context(chunks["1.6.6"])])
        assert response.evidence_list[0].overall_conclusion is ConditionStatus.MET

    def test_gate_is_inclusive_at_its_threshold(self, chunks):
        question = "A 45-year-old with unexplained visible haematuria without urinary tract infection."
        response, _ = generate_deterministic_response(question, [as_context(chunks["1.6.6"])])
        assert response.evidence_list[0].overall_conclusion is ConditionStatus.MET


class TestMissingInformation:
    def test_missing_symptom_yields_unknown_not_a_guess(self, chunks):
        response, _ = generate_deterministic_response(
            "A 41-year-old patient.", [as_context(chunks["1.2.4"])]
        )
        assert response.evidence_list[0].overall_conclusion is ConditionStatus.UNKNOWN

    def test_no_usable_context_refuses_rather_than_inventing(self):
        response, execution = generate_deterministic_response("A 39-year-old with jaundice.", [])
        assert response.refusal_reason is not None
        assert response.evidence_list == []
        assert execution.mode is ExecutionMode.FALLBACK


class TestExecutionHonesty:
    def test_invalid_api_key_falls_back_and_says_so(self, chunks, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "deliberately-invalid")
        generator = GroundedGenerator()
        response, execution = generator.generate_with_execution(
            "A 39-year-old with jaundice. Do they meet pancreatic referral criteria?",
            [as_context(chunks["1.2.4"])],
        )
        assert execution.mode is ExecutionMode.FALLBACK
        assert execution.faithfulness_checked is False
        # The fallback must still reach the correct verdict, not merely avoid crashing.
        assert response.evidence_list[0].overall_conclusion is ConditionStatus.NOT_MET

    def test_operator_selected_mode_is_reported_distinctly(self, chunks, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "deliberately-invalid")
        generator = GroundedGenerator()
        _, execution = generator.generate_with_execution(
            "A 39-year-old with jaundice.",
            [as_context(chunks["1.2.4"])],
            deterministic=True,
        )
        assert execution.mode is ExecutionMode.DETERMINISTIC_SELECTED

    def test_guardrail_block_is_withheld_not_fallback(self, chunks, monkeypatch):
        """A refusal after checking is not the same as a failure to check."""

        monkeypatch.setenv("GEMINI_API_KEY", "deliberately-invalid")
        generator = GroundedGenerator()
        _, execution = generator.generate_with_execution(
            "Do I have cancer?", [as_context(chunks["1.2.4"])]
        )
        assert execution.mode is ExecutionMode.WITHHELD

    def test_fallback_confidence_never_claims_high(self, chunks):
        """No faithfulness check runs on this path, so it cannot claim top confidence."""

        response, _ = generate_deterministic_response(
            "A 39-year-old with jaundice.", [as_context(chunks["1.2.4"])]
        )
        assert response.overall_confidence.value != "High"

    def test_fallback_answer_discloses_its_own_origin(self, chunks):
        response, _ = generate_deterministic_response(
            "A 39-year-old with jaundice.", [as_context(chunks["1.2.4"])]
        )
        assert "deterministic extraction" in response.recommendation_summary


class TestHelpers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("if they are aged 40 and over and have jaundice", ConditionLogic.AND),
            ("if they are aged 45 and over and have: • a, or • b", ConditionLogic.OR),
        ],
    )
    def test_logic_is_read_from_the_source_wording(self, text, expected):
        assert infer_condition_logic(f"1.x.y {text}") is expected

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("FIT result of 10 micrograms per gram", 10.0),
            ("FIT was 12.5 ug/g", 12.5),
            ("a patient with jaundice", None),
        ],
    )
    def test_lab_value_extraction(self, question, expected):
        assert extract_stated_lab_value(question) == expected

    def test_evaluation_quotes_the_source_condition(self, chunks):
        evaluations, _, _ = evaluate_chunk_conditions(
            "A 39-year-old with jaundice.", as_context(chunks["1.2.4"])
        )
        assert any("aged 40 and over" in item.condition_text for item in evaluations)
