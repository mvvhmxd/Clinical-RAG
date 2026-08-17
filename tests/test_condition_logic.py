"""The condition evaluation this system exists to get right.

The headline case: a prior build retrieved the correct pancreatic recommendation for a
39-year-old with jaundice and quoted "aged 40 and over" verbatim, but never said the
criterion failed. Retrieval was right; the answer was misleading. These tests pin the
comparison itself, independently of any model.
"""

from __future__ import annotations

import pytest

from ng12_rag.condition_logic import (
    audit_age_claim,
    combine,
    evaluate_age,
    evaluate_threshold,
    extract_age_conditions,
    extract_stated_age,
    parse_age_condition,
)
from ng12_rag.response_schema import (
    ConditionEvaluation,
    ConditionLogic,
    ConditionStatus,
)

PANCREATIC = "Refer people using a suspected cancer pathway referral for pancreatic cancer if they are aged 40 and over and have jaundice."
FIT = "Refer adults ... if they have a FIT result of at least 10 micrograms of haemoglobin per gram of faeces."


class TestTheRegressionCase:
    def test_thirty_nine_year_old_does_not_meet_forty_and_over(self):
        condition = parse_age_condition(PANCREATIC)
        result = evaluate_age(condition, 39)
        assert result.status is ConditionStatus.NOT_MET
        assert "1 year below" in result.reasoning
        assert "39" in result.reasoning and "40" in result.reasoning

    def test_reasoning_names_both_threshold_and_stated_value(self):
        """A conclusion with no arithmetic is not auditable."""

        result = evaluate_age(parse_age_condition(PANCREATIC), 39)
        assert "aged 40 and over" in result.reasoning
        assert "NOT met" in result.reasoning


class TestBoundaries:
    @pytest.mark.parametrize(
        ("text", "age", "expected"),
        [
            # Inclusive wording: the threshold value itself qualifies.
            ("aged 40 and over", 40, ConditionStatus.MET),
            ("aged 40 or older", 40, ConditionStatus.MET),
            ("at least 40", 40, ConditionStatus.MET),
            # Exclusive wording: the threshold value does not.
            ("aged over 40", 40, ConditionStatus.NOT_MET),
            ("more than 40", 40, ConditionStatus.NOT_MET),
            # Upper bounds.
            ("aged under 50", 50, ConditionStatus.NOT_MET),
            ("aged under 50", 49, ConditionStatus.MET),
        ],
    )
    def test_inclusive_and_exclusive_wording_differ_at_the_threshold(self, text, age, expected):
        assert evaluate_age(parse_age_condition(text), age).status is expected

    def test_exact_boundary_is_flagged(self):
        assert evaluate_age(parse_age_condition("aged 45 and over"), 45).at_boundary is True
        assert evaluate_age(parse_age_condition("aged 45 and over"), 46).at_boundary is False

    def test_fit_threshold_is_inclusive_at_ten(self):
        assert evaluate_threshold(FIT, 10.0).status is ConditionStatus.MET
        assert evaluate_threshold(FIT, 9.9).status is ConditionStatus.NOT_MET
        assert evaluate_threshold(FIT, 10.0).at_boundary is True

    def test_more_than_threshold_is_exclusive(self):
        assert evaluate_threshold("more than 10 units", 10.0).status is ConditionStatus.NOT_MET
        assert evaluate_threshold("more than 10 units", 10.1).status is ConditionStatus.MET


class TestMissingInformation:
    def test_absent_age_is_unknown_not_guessed(self):
        result = evaluate_age(parse_age_condition(PANCREATIC), None)
        assert result.status is ConditionStatus.UNKNOWN
        assert "does not state" in result.reasoning

    def test_absent_lab_value_is_unknown(self):
        assert evaluate_threshold(FIT, None).status is ConditionStatus.UNKNOWN


class TestMultipleBands:
    def test_bladder_recommendation_exposes_both_age_bands(self):
        """1.6.4 gates two different branches on 45 and 60."""

        text = (
            "aged 45 and over and have unexplained visible haematuria, or aged 60 and over "
            "and have unexplained non-visible haematuria and either dysuria or a raised "
            "white cell count"
        )
        thresholds = [c.threshold for c in extract_age_conditions(text)]
        assert thresholds == [45, 60]


class TestCombination:
    def test_and_fails_when_any_condition_fails(self):
        statuses = [ConditionStatus.MET, ConditionStatus.NOT_MET]
        assert combine(statuses, ConditionLogic.AND) is ConditionStatus.NOT_MET

    def test_or_passes_when_any_branch_passes(self):
        statuses = [ConditionStatus.MET, ConditionStatus.NOT_MET]
        assert combine(statuses, ConditionLogic.OR) is ConditionStatus.MET

    def test_and_with_a_failure_beats_an_unknown(self):
        """No further information could rescue an already-failed AND."""

        statuses = [ConditionStatus.NOT_MET, ConditionStatus.UNKNOWN]
        assert combine(statuses, ConditionLogic.AND) is ConditionStatus.NOT_MET

    def test_and_is_unknown_when_only_unknowns_remain(self):
        statuses = [ConditionStatus.MET, ConditionStatus.UNKNOWN]
        assert combine(statuses, ConditionLogic.AND) is ConditionStatus.UNKNOWN

    def test_or_is_unknown_rather_than_not_met_when_a_branch_is_unevaluable(self):
        statuses = [ConditionStatus.NOT_MET, ConditionStatus.UNKNOWN]
        assert combine(statuses, ConditionLogic.OR) is ConditionStatus.UNKNOWN


class TestStatedAgeExtraction:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("A 39-year-old with jaundice", 39),
            ("a 39 year old presenting with jaundice", 39),
            ("patient aged 72 with haematuria", 72),
            ("Does a 45yo qualify?", 45),
            ("An adult with jaundice", None),
        ],
    )
    def test_reads_the_age_the_question_states(self, question, expected):
        assert extract_stated_age(question) == expected


class TestIndependentAudit:
    def test_audit_catches_a_wrong_conclusion(self):
        agrees, explanation = audit_age_claim(
            PANCREATIC, "A 39-year-old with jaundice", ConditionStatus.MET
        )
        assert agrees is False
        assert "disagrees" in explanation

    def test_audit_accepts_a_correct_conclusion(self):
        agrees, _ = audit_age_claim(
            PANCREATIC, "A 39-year-old with jaundice", ConditionStatus.NOT_MET
        )
        assert agrees is True


class TestSchemaEnforcement:
    def test_conclusion_must_follow_from_its_conditions(self):
        """A claim cannot assert MET while carrying a failed AND condition."""

        from ng12_rag.response_schema import Citation, ConfidenceLevel, DocumentType, Evidence

        citation = Citation(
            document_type=DocumentType.NG12_SHORT,
            recommendation_id="1.2.4",
            chapter_number=None,
            chapter_title=None,
            section_title="Upper gastrointestinal tract cancers",
            page_number=12,
            quoted_text="aged 40 and over and have jaundice",
        )
        with pytest.raises(ValueError, match="does not follow"):
            Evidence(
                claim="Criteria are met per NG12, Recommendation 1.2.4, p.12.",
                supporting_citations=[citation],
                confidence=ConfidenceLevel.HIGH,
                condition_evaluations=[
                    ConditionEvaluation(
                        condition_text="aged 40 and over",
                        stated_value="39",
                        status=ConditionStatus.NOT_MET,
                        at_boundary=False,
                        reasoning="39 is below 40.",
                    )
                ],
                condition_logic=ConditionLogic.AND,
                overall_conclusion=ConditionStatus.MET,
            )

    def test_a_condition_with_a_value_cannot_be_unknown(self):
        with pytest.raises(ValueError, match="cannot be UNKNOWN"):
            ConditionEvaluation(
                condition_text="aged 40 and over",
                stated_value="39",
                status=ConditionStatus.UNKNOWN,
                at_boundary=False,
                reasoning="unclear",
            )
