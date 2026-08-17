"""Deterministic evaluation of guideline conditions against a question's stated values.

This module is the reference implementation of the comparison the system must never skip:
given "aged 40 and over" and a 39-year-old, it returns NOT_MET with the arithmetic spelled
out. It runs with no LLM, so the fallback path reaches the same conclusion as the live path.

Boundary handling is the highest-risk part and is handled explicitly rather than by
approximation:

* inclusive at the threshold -- "40 and over", "aged 40 or older", ">= 40", "at least 40"
* exclusive at the threshold -- "over 40", "more than 40", "above 40"
* inclusive upper bound      -- "under 50" is exclusive, "50 or under" is inclusive
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .response_schema import ConditionLogic, ConditionStatus

__all__ = [
    "AgeCondition",
    "ThresholdComparison",
    "audit_age_claim",
    "combine",
    "evaluate_age",
    "evaluate_threshold",
    "extract_age_conditions",
    "extract_stated_age",
    "parse_age_condition",
]


@dataclass(frozen=True)
class ThresholdComparison:
    """The outcome of comparing one stated value against one threshold."""

    status: ConditionStatus
    at_boundary: bool
    reasoning: str


@dataclass(frozen=True)
class AgeCondition:
    """A parsed age threshold and whether the threshold value itself qualifies."""

    threshold: int
    inclusive: bool
    direction: str  # "at_least" or "at_most"
    source_text: str

    def describe(self) -> str:
        if self.direction == "at_least":
            return f"age {'>=' if self.inclusive else '>'} {self.threshold}"
        return f"age {'<=' if self.inclusive else '<'} {self.threshold}"


# Ordered most specific first: "aged 40 and over" must not be caught by a bare "40" rule.
_AGE_PATTERNS: tuple[tuple[re.Pattern[str], str, bool], ...] = (
    (re.compile(r"aged?\s+(\d{1,3})\s+(?:years?\s+)?(?:and|or)\s+over", re.I), "at_least", True),
    (re.compile(r"aged?\s+(\d{1,3})\s+(?:years?\s+)?(?:and|or)\s+older", re.I), "at_least", True),
    (re.compile(r"(?:aged\s+)?at\s+least\s+(\d{1,3})", re.I), "at_least", True),
    (re.compile(r"aged?\s+(\d{1,3})\s+(?:years?\s+)?(?:and|or)\s+above", re.I), "at_least", True),
    (re.compile(r"[>≥]=?\s*(\d{1,3})", re.I), "at_least", True),
    (re.compile(r"aged?\s+over\s+(\d{1,3})", re.I), "at_least", False),
    (re.compile(r"(?:aged\s+)?more\s+than\s+(\d{1,3})", re.I), "at_least", False),
    (re.compile(r"aged?\s+under\s+(\d{1,3})", re.I), "at_most", False),
    (re.compile(r"(?:aged\s+)?younger\s+than\s+(\d{1,3})", re.I), "at_most", False),
    (re.compile(r"(?:aged\s+)?less\s+than\s+(\d{1,3})", re.I), "at_most", False),
    (re.compile(r"aged?\s+(\d{1,3})\s+(?:years?\s+)?(?:and|or)\s+under", re.I), "at_most", True),
    (re.compile(r"[<≤]=?\s*(\d{1,3})", re.I), "at_most", True),
)


def parse_age_condition(text: str) -> AgeCondition | None:
    """Parse the first age threshold in a condition string, or None if there is none."""

    for pattern, direction, inclusive in _AGE_PATTERNS:
        match = pattern.search(text)
        if match:
            return AgeCondition(
                threshold=int(match.group(1)),
                inclusive=inclusive,
                direction=direction,
                source_text=match.group(0).strip(),
            )
    return None


def extract_age_conditions(text: str) -> list[AgeCondition]:
    """Return every distinct age threshold in a condition string.

    Recommendations such as bladder 1.6.4 carry two age bands ("aged 45 and over" and
    "aged 60 and over"), each gating a different branch.
    """

    found: dict[tuple[int, bool, str], AgeCondition] = {}
    for pattern, direction, inclusive in _AGE_PATTERNS:
        for match in pattern.finditer(text):
            condition = AgeCondition(
                threshold=int(match.group(1)),
                inclusive=inclusive,
                direction=direction,
                source_text=match.group(0).strip(),
            )
            found.setdefault((condition.threshold, condition.inclusive, direction), condition)
    return sorted(found.values(), key=lambda c: c.threshold)


def evaluate_age(condition: AgeCondition, stated_age: int | None) -> ThresholdComparison:
    """Compare a stated age against one parsed age condition."""

    if stated_age is None:
        return ThresholdComparison(
            status=ConditionStatus.UNKNOWN,
            at_boundary=False,
            reasoning=(
                f"The source requires {condition.source_text!r}. The question does not state "
                "the patient's age, so this condition cannot be evaluated."
            ),
        )

    at_boundary = stated_age == condition.threshold
    if condition.direction == "at_least":
        met = stated_age >= condition.threshold if condition.inclusive else (
            stated_age > condition.threshold
        )
    else:
        met = stated_age <= condition.threshold if condition.inclusive else (
            stated_age < condition.threshold
        )

    if at_boundary:
        wording = "inclusive" if condition.inclusive else "exclusive"
        verdict = "MET" if met else "NOT MET"
        detail = (
            f"The stated age {stated_age} sits exactly on the threshold. "
            f"{condition.source_text!r} is {wording} at {condition.threshold}, so this is "
            f"{verdict}."
        )
    elif met:
        detail = (
            f"The source requires {condition.source_text!r} ({condition.describe()}). "
            f"The question states {stated_age}, which satisfies it."
        )
    else:
        gap = abs(stated_age - condition.threshold)
        year_word = "year" if gap == 1 else "years"
        side = "below" if stated_age < condition.threshold else "above"
        detail = (
            f"The source requires {condition.source_text!r} ({condition.describe()}). "
            f"The question states {stated_age}, which is {gap} {year_word} {side} the "
            f"threshold, so this condition is NOT met."
        )

    return ThresholdComparison(
        status=ConditionStatus.MET if met else ConditionStatus.NOT_MET,
        at_boundary=at_boundary,
        reasoning=detail,
    )


_INCLUSIVE_THRESHOLD = re.compile(r"at\s+least|no\s+less\s+than|[>≥]=|or\s+(?:more|above|over)", re.I)
_EXCLUSIVE_THRESHOLD = re.compile(r"more\s+than|greater\s+than|above|exceed", re.I)
_NUMBER = re.compile(r"(\d+(?:\.\d+)?)")


def evaluate_threshold(
    condition_text: str, stated_value: float | None, *, units: str = ""
) -> ThresholdComparison:
    """Compare a stated laboratory value against a threshold written in the source."""

    number = _NUMBER.search(condition_text)
    if number is None:
        return ThresholdComparison(
            status=ConditionStatus.UNKNOWN,
            at_boundary=False,
            reasoning=(
                f"No numeric threshold could be read from {condition_text!r}, so the value "
                "cannot be compared."
            ),
        )

    threshold = float(number.group(1))
    if stated_value is None:
        return ThresholdComparison(
            status=ConditionStatus.UNKNOWN,
            at_boundary=False,
            reasoning=(
                f"The source sets a threshold of {threshold:g}{units}. The question does not "
                "state a value, so this condition cannot be evaluated."
            ),
        )

    # "at least X" includes X; "more than X" does not. Default to inclusive only when the
    # source says so, because assuming inclusivity would loosen a referral criterion.
    inclusive = bool(_INCLUSIVE_THRESHOLD.search(condition_text)) and not _EXCLUSIVE_THRESHOLD.search(
        condition_text
    )
    at_boundary = stated_value == threshold
    met = stated_value >= threshold if inclusive else stated_value > threshold

    if at_boundary:
        wording = "inclusive" if inclusive else "exclusive"
        verdict = "MET" if met else "NOT MET"
        detail = (
            f"The stated value {stated_value:g}{units} sits exactly on the "
            f"{threshold:g}{units} threshold. The source wording is {wording} at that value, "
            f"so this is {verdict}."
        )
    else:
        relation = "at or above" if inclusive else "above"
        outcome = "satisfies" if met else "does not satisfy"
        detail = (
            f"The source requires a value {relation} {threshold:g}{units}. The question "
            f"states {stated_value:g}{units}, which {outcome} it."
        )

    return ThresholdComparison(
        status=ConditionStatus.MET if met else ConditionStatus.NOT_MET,
        at_boundary=at_boundary,
        reasoning=detail,
    )


_STATED_AGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(\d{1,3})\s*[-\s]?year[-\s]?old", re.I),
    re.compile(r"\bage[ds]?\s+(\d{1,3})\b", re.I),
    re.compile(r"\bis\s+(\d{1,3})\b(?!\s*(?:µg|ug|micro))", re.I),
    re.compile(r"\b(\d{1,3})\s*(?:yo|y/o|yrs?)\b", re.I),
)


def extract_stated_age(question: str) -> int | None:
    """Read the patient's age from the question, or None when it is not stated."""

    for pattern in _STATED_AGE_PATTERNS:
        match = pattern.search(question)
        if match:
            age = int(match.group(1))
            if 0 < age < 130:
                return age
    return None


def audit_age_claim(
    source_text: str, question: str, asserted: ConditionStatus | None
) -> tuple[bool, str]:
    """Independently recheck an age conclusion against the source and the question.

    Used to catch a generated answer that cites the right recommendation but reaches the
    wrong verdict. Returns (agrees, explanation); agreement is vacuously true when there is
    no single unambiguous age condition to check.
    """

    conditions = extract_age_conditions(source_text)
    if len(conditions) != 1 or asserted is None:
        return True, "no single age condition to audit"

    stated_age = extract_stated_age(question)
    computed = evaluate_age(conditions[0], stated_age)
    if computed.status is asserted:
        return True, computed.reasoning
    return False, (
        f"Deterministic check disagrees: computed {computed.status.value} but the answer "
        f"asserted {asserted.value}. {computed.reasoning}"
    )


def combine(statuses: list[ConditionStatus], logic: ConditionLogic) -> ConditionStatus:
    """Combine condition results using the source's own logical structure.

    Under OR one satisfied branch is enough. Under AND a single failure decides the outcome
    even when other conditions are unknown, because no additional information could rescue it.
    """

    if not statuses:
        return ConditionStatus.UNKNOWN

    if logic is ConditionLogic.OR:
        if ConditionStatus.MET in statuses:
            return ConditionStatus.MET
        if ConditionStatus.UNKNOWN in statuses:
            return ConditionStatus.UNKNOWN
        return ConditionStatus.NOT_MET

    if ConditionStatus.NOT_MET in statuses:
        return ConditionStatus.NOT_MET
    if ConditionStatus.UNKNOWN in statuses:
        return ConditionStatus.UNKNOWN
    return ConditionStatus.MET
