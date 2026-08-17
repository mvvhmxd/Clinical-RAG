"""Answer without an LLM, or say honestly that it cannot.

The generation layer must never be the only thing standing between a user and a safe answer.
When the model is unavailable, times out, exhausts its quota, or returns output that fails
validation, this module answers from the retrieved source text directly, using the same
condition logic the live path is required to apply.

It quotes rather than writes. Every sentence it emits is either the source's own wording or a
comparison it computed itself, so there is no step at which an unsupported claim could enter.
Where it cannot evaluate a condition, it says UNKNOWN rather than choosing a likely value.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .condition_logic import (
    combine,
    evaluate_age,
    evaluate_threshold,
    extract_age_conditions,
    extract_stated_age,
)
from .response_schema import (
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


class ExecutionMode(StrEnum):
    """How an answer was actually produced. Never adjusted for presentation."""

    LIVE = "live"
    """The model ran and its output passed the faithfulness check."""

    FALLBACK = "fallback"
    """The model was unavailable or invalid; deterministic extraction answered instead."""

    DETERMINISTIC_SELECTED = "deterministic_selected"
    """The operator explicitly asked for the no-LLM path."""

    WITHHELD = "withheld"
    """A guardrail blocked generation entirely."""


class ExecutionInfo(BaseModel):
    """The single source of truth for whether an answer used an LLM."""

    model_config = ConfigDict(extra="forbid")

    mode: ExecutionMode
    model: str | None = None
    reason: str | None = Field(
        default=None,
        description="Why the path was taken; required whenever the mode is not live.",
    )
    faithfulness_checked: bool = False


# NG12 marks sub-bullets with a fullwidth hyphen, so that character is matched deliberately.
_BRANCH_SEPARATOR = re.compile(r",\s*or\b|\u2022|\uff0d", re.IGNORECASE)
_LAB_UNITS = re.compile(r"micrograms?\s+of\s+haemoglobin[^,.;]*", re.IGNORECASE)
_STATED_LAB = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:µg|ug|micrograms?)\s*(?:/|per\s+)?\s*g", re.IGNORECASE
)


def infer_condition_logic(source_text: str) -> ConditionLogic:
    """Read the recommendation's own combining logic from its wording.

    A bulleted list joined by "or" is a set of alternative branches. Conditions joined by
    "and" must all hold. Guessing here would silently loosen or tighten a referral criterion,
    so the default is the stricter reading.
    """

    body = source_text.split(" ", 1)[-1]
    has_branches = bool(_BRANCH_SEPARATOR.search(body))
    if has_branches and re.search(r"\bor\b", body, re.IGNORECASE):
        return ConditionLogic.OR
    if re.search(r"\band\b", body, re.IGNORECASE):
        return ConditionLogic.AND
    return ConditionLogic.SINGLE


def extract_stated_lab_value(question: str) -> float | None:
    """Read a faecal-haemoglobin value from the question, if one is stated."""

    match = _STATED_LAB.search(question)
    return float(match.group(1)) if match else None


# The symptom_condition metadata field holds the whole conditional clause ("if they are aged
# 40 and over and have jaundice"), not discrete symptoms, so it cannot be split into
# conditions. Matching an explicit NG12 symptom vocabulary against the source text is both
# more accurate and inspectable. Longer phrases come first so "upper abdominal pain" is
# preferred over "abdominal pain".
_SYMPTOM_VOCABULARY: tuple[str, ...] = (
    "chest X-ray findings that suggest lung cancer",
    "unexplained visible haematuria without urinary tract infection",
    "iron-deficiency anaemia",
    "change in bowel habit",
    "upper abdominal pain",
    "raised white cell count",
    "non-visible haematuria",
    "visible haematuria",
    "abdominal mass",
    "abdominal pain",
    "rectal bleeding",
    "weight loss",
    "appetite loss",
    "shortness of breath",
    "finger clubbing",
    "night sweats",
    "chest pain",
    "thrombocytosis",
    "haemoptysis",
    "haematuria",
    "dysphagia",
    "dyspepsia",
    "jaundice",
    "dysuria",
    "vomiting",
    "nausea",
    "reflux",
    "fatigue",
    "cough",
)


def _symptom_terms(metadata: Mapping[str, Any], source_text: str) -> list[str]:
    """Return the symptom conditions this recommendation actually names."""

    haystack = f"{source_text} {metadata.get('symptom_condition') or ''}".lower()
    found: list[str] = []
    for term in _SYMPTOM_VOCABULARY:
        lowered = term.lower()
        if lowered not in haystack:
            continue
        # Skip a term already covered by a longer phrase that was matched first.
        if any(lowered in existing.lower() for existing in found):
            continue
        found.append(term)
    return found[:6]


def _first_branch_position(source_text: str) -> int | None:
    match = _BRANCH_SEPARATOR.search(source_text)
    return match.start() if match else None


def evaluate_chunk_conditions(
    question: str, chunk: Mapping[str, Any]
) -> tuple[list[ConditionEvaluation], ConditionLogic, ConditionStatus]:
    """Evaluate one recommendation's conditions against the question's stated values.

    Recommendations are rarely a flat list joined by a single operator. Renal 1.6.6 reads
    "aged 45 and over AND have: bullet OR bullet" -- the age is a gate over a group of
    alternatives, not one more alternative. Treating it as a peer branch would pass a
    44-year-old with haematuria. Lung 1.1.1 is the opposite: its age sits inside a branch
    ("or are aged 40 and over with unexplained haemoptysis") and must not gate the others.

    The two are told apart by position: an age stated before the first branch separator
    gates the group, one stated after belongs to its branch. The OR group is then reported
    as a single combined condition so the flat conclusion still follows from its parts.
    """

    source_text = str(chunk.get("text", ""))
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), Mapping) else chunk
    lowered_source = source_text.lower()
    branch_start = _first_branch_position(source_text)

    gates: list[ConditionEvaluation] = []
    branches: list[ConditionEvaluation] = []

    def place(evaluation: ConditionEvaluation, position: int | None) -> None:
        before_branches = branch_start is None or (position is not None and position < branch_start)
        (gates if before_branches else branches).append(evaluation)

    stated_age = extract_stated_age(question)
    age_source = str(metadata.get("age_condition") or "") or source_text
    for condition in extract_age_conditions(age_source)[:4]:
        result = evaluate_age(condition, stated_age)
        position = lowered_source.find(condition.source_text.lower())
        place(
            ConditionEvaluation(
                condition_text=condition.source_text,
                stated_value=str(stated_age) if stated_age is not None else None,
                status=result.status,
                at_boundary=result.at_boundary,
                reasoning=result.reasoning,
            ),
            position if position != -1 else None,
        )

    lab_text = str(metadata.get("lab_threshold") or "")
    if lab_text and re.search(r"\d", lab_text):
        stated_value = extract_stated_lab_value(question)
        result = evaluate_threshold(lab_text, stated_value, units=" µg/g")
        units_match = _LAB_UNITS.search(lab_text)
        place(
            ConditionEvaluation(
                condition_text=units_match.group(0) if units_match else lab_text[:200],
                stated_value=f"{stated_value:g}" if stated_value is not None else None,
                status=result.status,
                at_boundary=result.at_boundary,
                reasoning=result.reasoning,
            ),
            None if branch_start is None else branch_start + 1,
        )

    lowered_question = question.lower()
    for term in _symptom_terms(metadata, source_text):
        present = term.lower() in lowered_question
        place(
            ConditionEvaluation(
                condition_text=term,
                stated_value=term if present else None,
                status=ConditionStatus.MET if present else ConditionStatus.UNKNOWN,
                at_boundary=False,
                reasoning=(
                    f"The source requires {term!r}. The question states it."
                    if present
                    else f"The source requires {term!r}. The question does not mention it, "
                    "so this condition cannot be evaluated."
                ),
            ),
            lowered_source.find(term.lower()),
        )

    if not branches:
        evaluations = gates
        logic = ConditionLogic.AND if len(evaluations) > 1 else ConditionLogic.SINGLE
        return evaluations, logic, combine([e.status for e in evaluations], logic)

    branch_logic = (
        ConditionLogic.OR
        if infer_condition_logic(source_text) is ConditionLogic.OR
        else ConditionLogic.AND
    )
    branch_result = combine([e.status for e in branches], branch_logic)
    satisfied = [e.condition_text for e in branches if e.status is ConditionStatus.MET]
    joiner = " or " if branch_logic is ConditionLogic.OR else " and "
    branch_summary = ConditionEvaluation(
        condition_text=joiner.join(e.condition_text for e in branches)[:600],
        stated_value=", ".join(satisfied)[:200] if satisfied else None,
        status=branch_result,
        at_boundary=any(e.at_boundary for e in branches),
        reasoning=(
            f"The source lists {len(branches)} alternatives joined by "
            f"{'OR' if branch_logic is ConditionLogic.OR else 'AND'}. "
            + (
                f"Satisfied by: {', '.join(satisfied)}."
                if satisfied
                else "The question satisfies none of them explicitly."
            )
        ),
    )

    evaluations = [*gates, branch_summary]
    logic = ConditionLogic.AND if gates else ConditionLogic.SINGLE
    return evaluations, logic, combine([e.status for e in evaluations], logic)


def _citation_for(chunk: Mapping[str, Any]) -> Citation | None:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), Mapping) else chunk
    recommendation_id = metadata.get("recommendation_id")
    if not recommendation_id:
        return None
    quote = str(chunk.get("text", "")).strip()
    if not quote:
        return None
    return Citation(
        document_type=DocumentType.NG12_SHORT,
        recommendation_id=str(recommendation_id),
        chapter_number=None,
        chapter_title=None,
        section_title=str(metadata.get("section_title") or "NG12"),
        page_number=int(metadata.get("page_number") or 1),
        # Trimmed to the schema's quote ceiling while staying contiguous and verbatim.
        quoted_text=quote[:3900],
    )


_CONCLUSION_PHRASE = {
    ConditionStatus.MET: "This patient meets the referral criterion stated",
    ConditionStatus.NOT_MET: "This patient does not meet the referral criterion stated",
    ConditionStatus.UNKNOWN: (
        "This patient cannot be fully assessed against the referral criterion stated"
    ),
}


def generate_deterministic_response(
    question: str,
    chunks: Sequence[Mapping[str, Any]],
    *,
    mode: ExecutionMode = ExecutionMode.FALLBACK,
    reason: str = "model unavailable",
) -> tuple[FullResponse, ExecutionInfo]:
    """Produce a grounded answer, or an honest uncertain state, without calling a model."""

    execution = ExecutionInfo(mode=mode, model=None, reason=reason, faithfulness_checked=False)

    usable = [c for c in chunks if _citation_for(c) is not None]
    if not usable:
        return (
            FullResponse.refusal(
                reason=(
                    "No numbered NG12 recommendation was retrieved for this question, and "
                    "generation is unavailable, so no grounded answer can be given."
                )
            ),
            execution,
        )

    evidence: list[Evidence] = []
    summary_parts: list[str] = []

    for chunk in usable[:3]:
        citation = _citation_for(chunk)
        assert citation is not None
        evaluations, logic, conclusion = evaluate_chunk_conditions(question, chunk)

        claim = f"{_CONCLUSION_PHRASE[conclusion]} {citation.inline_reference}."
        # Lead with whatever decided the outcome: a failure under AND, otherwise the
        # condition that was satisfied, otherwise what could not be evaluated.
        decisive = next(
            (e for e in evaluations if e.status is ConditionStatus.NOT_MET),
            next(
                (e for e in evaluations if e.status is ConditionStatus.MET),
                next((e for e in evaluations if e.status is ConditionStatus.UNKNOWN), None),
            ),
        )
        summary_parts.append(claim + (f" {decisive.reasoning}" if decisive else ""))
        evidence.append(
            Evidence(
                claim=claim,
                supporting_citations=[citation],
                # Never High: this path performs no faithfulness verification.
                confidence=ConfidenceLevel.MEDIUM
                if conclusion is not ConditionStatus.UNKNOWN
                else ConfidenceLevel.LOW,
                condition_evaluations=evaluations,
                condition_logic=logic,
                overall_conclusion=conclusion,
            )
        )

    summary = (
        " ".join(summary_parts)
        + " This answer was produced by deterministic extraction from the source text because "
        "language-model generation was unavailable."
    )
    return (
        FullResponse(
            recommendation_summary=summary[:5900],
            evidence_list=evidence,
            overall_confidence=ConfidenceLevel.minimum(*(e.confidence for e in evidence)),
            disclaimer=CLINICAL_DISCLAIMER,
            refusal_reason=None,
            clarifying_question=None,
        ),
        execution,
    )
