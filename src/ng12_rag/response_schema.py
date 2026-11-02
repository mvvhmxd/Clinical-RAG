"""Typed response contracts for the dual-source NG12 generation layer.

The response schema distinguishes clean numbered recommendations (``NG12_short``) from
rationale, evidence-review, and study-data chunks (``NG12_full``). Every clinical claim
must contain the complete human-readable citation for every source object that supports it.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CLINICAL_DISCLAIMER = (
    "This information supports recognition and referral decisions using the retrieved "
    "NICE NG12 recommendations; it does not diagnose cancer or replace clinical judgement. "
    "If symptoms are severe, rapidly worsening, or you think this may be an emergency, "
    "seek urgent medical help immediately."
)

_RECOMMENDATION_ID_RE = re.compile(r"^\d+(?:\.\d+)+$")
_CHAPTER_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*$")
_CITATION_REFERENCE_RE = re.compile(
    r"^(?:NG12, Recommendation \d+(?:\.\d+)+, p\.\d+|"
    r"NG12 Full Guideline, Chapter \d+(?:\.\d+)* \([^)]+\), p\.\d+)$"
)


def _normalise_reference(value: str) -> str:
    return " ".join(value.split()).casefold()


class DocumentType(StrEnum):
    """The two NG12 source-document families accepted by the generator."""

    NG12_SHORT = "NG12_short"
    NG12_FULL = "NG12_full"


class ConfidenceLevel(StrEnum):
    """Calibrated confidence labels exposed to callers and the user interface."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INSUFFICIENT = "Insufficient"

    @property
    def rank(self) -> int:
        """Return a monotonic rank suitable for conservative confidence capping."""

        return {
            ConfidenceLevel.INSUFFICIENT: 0,
            ConfidenceLevel.LOW: 1,
            ConfidenceLevel.MEDIUM: 2,
            ConfidenceLevel.HIGH: 3,
        }[self]

    @classmethod
    def minimum(cls, *levels: ConfidenceLevel) -> ConfidenceLevel:
        """Return the least-confident level, failing closed for an empty input."""

        if not levels:
            return cls.INSUFFICIENT
        return min(levels, key=lambda level: level.rank)


class ConditionStatus(StrEnum):
    """Whether the question's stated values satisfy one condition in the source text."""

    MET = "MET"
    NOT_MET = "NOT_MET"
    UNKNOWN = "UNKNOWN"


class ConditionLogic(StrEnum):
    """How a recommendation combines its conditions, per the source's own wording."""

    AND = "AND"
    OR = "OR"
    SINGLE = "SINGLE"


class ConditionEvaluation(BaseModel):
    """One discrete condition checked against the values stated in the question.

    Retrieving the right recommendation is not the same as answering the question. Quoting
    "aged 40 and over" for a 39-year-old without stating that the criterion fails is a
    misleading answer with a perfect citation, so every claim must carry this evaluation.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    condition_text: str = Field(
        ...,
        min_length=1,
        max_length=600,
        description="The condition as written in the source, quoted not paraphrased.",
    )
    stated_value: str | None = Field(
        ...,
        description="The value the question supplies for this condition, or null if absent.",
    )
    status: ConditionStatus
    at_boundary: bool = Field(
        ...,
        description="True when the stated value sits exactly on the threshold.",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        max_length=1200,
        description="The explicit comparison, naming both the threshold and the stated value.",
    )

    @model_validator(mode="after")
    def validate_unknown_has_no_value(self) -> ConditionEvaluation:
        if self.status is ConditionStatus.UNKNOWN and self.stated_value:
            raise ValueError(
                "A condition with a stated value cannot be UNKNOWN; evaluate it as MET or NOT_MET"
            )
        if self.status is not ConditionStatus.UNKNOWN and not self.stated_value:
            raise ValueError(
                "A condition can only be MET or NOT_MET when the question supplies a value"
            )
        return self


class Citation(BaseModel):
    """A typed source pointer bound to one retrieved NG12 chunk."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_type: DocumentType
    recommendation_id: str | None
    chapter_number: str | None
    chapter_title: str | None
    section_title: str = Field(..., min_length=1, max_length=300)
    page_number: int = Field(..., ge=1)
    quoted_text: str = Field(..., min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_source_identity(self) -> Citation:
        if self.document_type is DocumentType.NG12_SHORT:
            if not self.recommendation_id or not _RECOMMENDATION_ID_RE.fullmatch(
                self.recommendation_id
            ):
                raise ValueError(
                    "NG12_short citations require a specific recommendation_id such as 1.3.1"
                )
            if self.chapter_number is not None or self.chapter_title is not None:
                raise ValueError(
                    "NG12_short citations must set chapter_number and chapter_title to null"
                )
            return self

        if self.recommendation_id is not None:
            raise ValueError("NG12_full citations must set recommendation_id to null")
        if not self.chapter_number or not _CHAPTER_NUMBER_RE.fullmatch(self.chapter_number):
            raise ValueError(
                "NG12_full citations require a numeric chapter_number such as 9"
            )
        if not self.chapter_title:
            raise ValueError("NG12_full citations require chapter_title")
        return self

    @property
    def formatted_reference(self) -> str:
        """Return the exact user-facing citation format for this source type."""

        if self.document_type is DocumentType.NG12_SHORT:
            return (
                f"NG12, Recommendation {self.recommendation_id}, p.{self.page_number}"
            )
        return (
            f"NG12 Full Guideline, Chapter {self.chapter_number} "
            f"({self.chapter_title}), p.{self.page_number}"
        )

    @property
    def inline_reference(self) -> str:
        """Return the required claim-level inline citation text."""

        return f"per {self.formatted_reference}"

    @property
    def source_key(self) -> str:
        """Return a deterministic key used for exact verifier/source binding."""

        return _normalise_reference(self.formatted_reference)


class Evidence(BaseModel):
    """One atomic clinical claim and the retrieved sources supporting it."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim: str = Field(..., min_length=1, max_length=3000)
    supporting_citations: list[Citation] = Field(..., min_length=1, max_length=12)
    confidence: ConfidenceLevel
    # Required rather than defaulted: an optional field is one the model can silently omit,
    # which is exactly the failure this evaluation exists to prevent. Rationale-only claims
    # from the full guideline legitimately pass an empty list.
    condition_evaluations: list[ConditionEvaluation] = Field(
        ...,
        max_length=20,
        description=(
            "One entry per discrete condition in the cited recommendation. Required for every "
            "claim citing a numbered recommendation; empty only for full-guideline rationale."
        ),
    )
    condition_logic: ConditionLogic = Field(
        ...,
        description="How the source combines its conditions. Never assume AND where it says OR.",
    )
    overall_conclusion: ConditionStatus | None = Field(
        ...,
        description="The combined result of condition_evaluations under condition_logic.",
    )

    @model_validator(mode="after")
    def validate_conclusion_follows_conditions(self) -> Evidence:
        """The stated conclusion must follow from the parts, under the source's own logic."""

        if not self.condition_evaluations:
            return self

        statuses = [item.status for item in self.condition_evaluations]
        if self.condition_logic is ConditionLogic.OR:
            expected = (
                ConditionStatus.MET
                if ConditionStatus.MET in statuses
                else ConditionStatus.UNKNOWN
                if ConditionStatus.UNKNOWN in statuses
                else ConditionStatus.NOT_MET
            )
        else:
            expected = (
                ConditionStatus.NOT_MET
                if ConditionStatus.NOT_MET in statuses
                else ConditionStatus.UNKNOWN
                if ConditionStatus.UNKNOWN in statuses
                else ConditionStatus.MET
            )

        if self.overall_conclusion is None:
            self.overall_conclusion = expected
        elif self.overall_conclusion is not expected:
            raise ValueError(
                f"overall_conclusion {self.overall_conclusion} does not follow from "
                f"{[s.value for s in statuses]} combined with {self.condition_logic.value}"
            )
        return self

    @model_validator(mode="after")
    def validate_claim_binding(self) -> Evidence:
        if self.confidence is ConfidenceLevel.INSUFFICIENT:
            raise ValueError("An evidence claim cannot have Insufficient confidence")

        normalised_claim = _normalise_reference(self.claim)
        citation_keys: set[tuple[str, str]] = set()
        for citation in self.supporting_citations:
            key = (citation.source_key, citation.quoted_text)
            if key in citation_keys:
                raise ValueError("supporting_citations must not contain duplicates")
            citation_keys.add(key)

            if _normalise_reference(citation.inline_reference) not in normalised_claim:
                raise ValueError(
                    "Every citation must appear inline in the claim using its complete "
                    f"typed reference: '{citation.inline_reference}'"
                )
        return self


class FullResponse(BaseModel):
    """The complete user-facing answer, clarification, or refusal."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    recommendation_summary: str = Field(..., min_length=1, max_length=6000)
    evidence_list: list[Evidence] = Field(..., max_length=30)
    overall_confidence: ConfidenceLevel
    disclaimer: str = Field(..., min_length=1)
    refusal_reason: str | None = Field(
        ...,
        description="Non-null only for a hard refusal or safe fail-closed response.",
    )
    clarifying_question: str | None = Field(
        ...,
        description="Non-null only when one missing detail blocks a grounded answer.",
    )

    _DISCLAIMER: ClassVar[str] = CLINICAL_DISCLAIMER

    @field_validator("disclaimer")
    @classmethod
    def validate_disclaimer(cls, value: str) -> str:
        if value != CLINICAL_DISCLAIMER:
            raise ValueError("The fixed clinical disclaimer must be included verbatim")
        return value

    @model_validator(mode="after")
    def validate_response_state(self) -> FullResponse:
        is_refusal = self.refusal_reason is not None
        needs_clarification = self.clarifying_question is not None

        if is_refusal and needs_clarification:
            raise ValueError("A response cannot be both a refusal and a clarification")

        if is_refusal or needs_clarification:
            if self.evidence_list:
                raise ValueError("Refusal and clarification responses cannot contain evidence")
            if self.overall_confidence is not ConfidenceLevel.INSUFFICIENT:
                raise ValueError(
                    "Refusal and clarification responses must use Insufficient confidence"
                )
            return self

        if not self.evidence_list:
            raise ValueError("A grounded answer must contain at least one evidence claim")
        if self.overall_confidence is ConfidenceLevel.INSUFFICIENT:
            raise ValueError(
                "A response with Insufficient confidence must refuse or ask for clarification"
            )

        weakest_claim = ConfidenceLevel.minimum(
            *(evidence.confidence for evidence in self.evidence_list)
        )
        if self.overall_confidence.rank > weakest_claim.rank:
            raise ValueError(
                "overall_confidence cannot exceed the least-confident evidence claim"
            )
        return self

    @classmethod
    def refusal(cls, *, reason: str, summary: str | None = None) -> FullResponse:
        """Build a uniform hard-refusal response."""

        return cls(
            recommendation_summary=summary
            or "I cannot provide a grounded NG12 recommendation for this request.",
            evidence_list=[],
            overall_confidence=ConfidenceLevel.INSUFFICIENT,
            disclaimer=CLINICAL_DISCLAIMER,
            refusal_reason=reason,
            clarifying_question=None,
        )

    @classmethod
    def clarification(
        cls,
        *,
        question: str,
        summary: str = "More information is required before applying an NG12 referral rule.",
    ) -> FullResponse:
        """Build a uniform clarification response without guessing missing details."""

        return cls(
            recommendation_summary=summary,
            evidence_list=[],
            overall_confidence=ConfidenceLevel.INSUFFICIENT,
            disclaimer=CLINICAL_DISCLAIMER,
            refusal_reason=None,
            clarifying_question=question,
        )

    def with_disclaimer(self) -> FullResponse:
        """Return a copy with the immutable disclaimer restored."""

        return self.model_copy(update={"disclaimer": CLINICAL_DISCLAIMER})

    def to_markdown(self) -> str:
        """Render the response with visibly distinct short/full source citations."""

        sections = ["## Recommendation summary", self.recommendation_summary]

        if self.clarifying_question:
            sections.extend(["## Clarifying question", self.clarifying_question])
        elif self.refusal_reason:
            sections.extend(["## Refusal reason", self.refusal_reason])
        else:
            sections.append("## Supporting evidence")
            for evidence in self.evidence_list:
                sections.append(f"- **Claim:** {evidence.claim}")
                for citation in evidence.supporting_citations:
                    sections.append(
                        f"  > \"{citation.quoted_text}\" — "
                        f"{citation.formatted_reference}; {citation.section_title}"
                    )

            sections.append("## Citations")
            seen: set[tuple[str, str]] = set()
            for evidence in self.evidence_list:
                for citation in evidence.supporting_citations:
                    key = (citation.source_key, citation.quoted_text)
                    if key in seen:
                        continue
                    seen.add(key)
                    sections.append(
                        f"- **{citation.formatted_reference}** — {citation.section_title}"
                    )

        sections.extend(
            ["## Confidence", self.overall_confidence.value, "---", self.disclaimer]
        )
        return "\n\n".join(sections)


class ClaimVerification(BaseModel):
    """Faithfulness verdict for one evidence claim, indexed into the response."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_index: int = Field(..., ge=0)
    supported: bool
    supporting_citation_references: list[str] = Field(..., max_length=12)
    confidence: ConfidenceLevel
    explanation: str = Field(..., min_length=1, max_length=2000)

    @field_validator("supporting_citation_references")
    @classmethod
    def validate_supporting_references(cls, values: list[str]) -> list[str]:
        normalised = [_normalise_reference(value) for value in values]
        if len(normalised) != len(set(normalised)):
            raise ValueError("supporting_citation_references must be unique")
        for value in values:
            if not _CITATION_REFERENCE_RE.fullmatch(value):
                raise ValueError(f"Invalid typed citation reference: {value}")
        return values

    @model_validator(mode="after")
    def validate_verdict_confidence(self) -> ClaimVerification:
        if not self.supported and self.confidence is not ConfidenceLevel.INSUFFICIENT:
            raise ValueError("Unsupported claims must have Insufficient confidence")
        if self.supported and self.confidence is ConfidenceLevel.INSUFFICIENT:
            raise ValueError("Supported claims need Low, Medium, or High confidence")
        return self


class FaithfulnessReport(BaseModel):
    """Structured result from the post-generation faithfulness pass."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    all_claims_supported: bool
    summary_supported: bool
    summary_explanation: str = Field(..., min_length=1, max_length=2000)
    claim_verifications: list[ClaimVerification]
    unsupported_claims: list[str]
    overall_confidence: ConfidenceLevel

    @model_validator(mode="after")
    def validate_report_consistency(self) -> FaithfulnessReport:
        unsupported_verdicts = [
            verdict for verdict in self.claim_verifications if not verdict.supported
        ]
        has_unsupported_content = (not self.summary_supported) or bool(unsupported_verdicts)
        if self.all_claims_supported == has_unsupported_content:
            raise ValueError(
                "all_claims_supported must agree with the summary and claim verdicts"
            )
        if self.all_claims_supported and self.unsupported_claims:
            raise ValueError("unsupported_claims must be empty when every claim is supported")
        if not self.all_claims_supported and not self.unsupported_claims:
            raise ValueError("unsupported_claims must explain every failed faithfulness check")
        if not self.all_claims_supported and (
            self.overall_confidence is not ConfidenceLevel.INSUFFICIENT
        ):
            raise ValueError("A report with unsupported claims must be Insufficient")
        return self


__all__ = [
    "CLINICAL_DISCLAIMER",
    "Citation",
    "ClaimVerification",
    "ConfidenceLevel",
    "DocumentType",
    "Evidence",
    "FaithfulnessReport",
    "FullResponse",
]
