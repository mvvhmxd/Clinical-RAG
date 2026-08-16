"""Deterministic safety and scope guardrails for the NG12 RAG pipeline.

Hard safety rules run before retrieval and do not depend on an LLM. Post-generation checks
then bind every structured citation to the exact retrieved chunk. The module intentionally
fails closed whenever source identity or quote provenance cannot be established.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .response_schema import (
    CLINICAL_DISCLAIMER,
    Citation,
    ConfidenceLevel,
    DocumentType,
    FullResponse,
)

COVERED_SCOPE = (
    "Lung; Colorectal; Upper GI (oesophageal, gastric/stomach, pancreatic); and Renal & Bladder"
)


class InputClassification(StrEnum):
    """Pre-retrieval safety route."""

    ALLOWED = "Allowed"
    NEEDS_CAUTION = "Needs Caution"
    REFUSE_AND_REDIRECT = "Refuse+Redirect"


class GuardrailReason(StrEnum):
    """Machine-readable reason for a guardrail decision."""

    ALLOWED = "allowed"
    EMERGENCY = "emergency"
    DIAGNOSIS_REQUEST = "diagnosis_request"
    OUT_OF_SCOPE_SITE = "out_of_scope_cancer_site"
    UNDERSPECIFIED = "underspecified_query"
    NO_RETRIEVAL = "no_retrieved_chunks"
    LOW_RETRIEVAL_CONFIDENCE = "low_retrieval_confidence"
    INVALID_RETRIEVAL = "invalid_retrieval_metadata"
    UNSUPPORTED_GENERATION = "unsupported_generated_claim"


class GuardrailDecision(BaseModel):
    """Result of deterministic input classification."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    classification: InputClassification
    reason: GuardrailReason
    message: str = Field(..., min_length=1)
    clarifying_question: str | None = None
    detected_out_of_scope_sites: list[str] = Field(default_factory=list)

    def to_response(self) -> FullResponse | None:
        """Convert blocking decisions into the shared response schema."""

        if self.classification is InputClassification.ALLOWED:
            return None
        if self.classification is InputClassification.NEEDS_CAUTION:
            return FullResponse.clarification(
                question=self.clarifying_question or self.message,
                summary=self.message,
            )
        return FullResponse.refusal(reason=self.message)


class GuardrailConfig(BaseModel):
    """Thresholds and strictness controls for safety routing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    high_similarity_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    medium_similarity_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    minimum_similarity_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    require_similarity_scores: bool = False
    minimum_quote_characters: int = Field(default=12, ge=1, le=500)

    @field_validator("medium_similarity_threshold")
    @classmethod
    def validate_medium_threshold(cls, value: float, info: Any) -> float:
        high = info.data.get("high_similarity_threshold")
        if high is not None and value >= high:
            raise ValueError("medium_similarity_threshold must be below the high threshold")
        return value

    @field_validator("minimum_similarity_threshold")
    @classmethod
    def validate_minimum_threshold(cls, value: float, info: Any) -> float:
        medium = info.data.get("medium_similarity_threshold")
        if medium is not None and value >= medium:
            raise ValueError("minimum_similarity_threshold must be below the medium threshold")
        return value


class RetrievalAssessment(BaseModel):
    """Calibrated retrieval-confidence result."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    confidence: ConfidenceLevel
    top_score: float | None = Field(default=None, ge=0.0, le=1.0)
    should_answer: bool
    reason: str = Field(..., min_length=1)


class SourceValidationIssue(BaseModel):
    """A deterministic citation or provenance failure."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim_index: int | None = Field(default=None, ge=0)
    recommendation_id: str | None = None
    citation_reference: str | None = None
    document_type: DocumentType | None = None
    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


_OUT_OF_SCOPE_SITE_PATTERNS: dict[str, tuple[str, ...]] = {
    "Anal": (r"\banal\s+(?:cancer|tumou?r|lesion|mass|ulceration)\b",),
    "Breast": (r"\bbreast(?:\s+cancer|\s+lump|\s+tumou?r)?\b",),
    "Cervical": (r"\bcervi(?:cal|x)(?:\s+cancer|\s+tumou?r)?\b",),
    "Endometrial/Uterine": (r"\b(?:endometrial|uterine|womb)(?:\s+cancer|\s+tumou?r)?\b",),
    "Gallbladder/Biliary": (r"\b(?:gall\s*bladder|bile\s+duct|biliary|cholangiocarcinoma)\b",),
    "Gynaecological (other)": (r"\b(?:vulval|vulvar|vaginal)\b",),
    "Haematological": (r"\b(?:leukaemia|leukemia|lymphoma|myeloma|blood\s+cancer)\b",),
    "Head and neck": (
        r"\b(?:oral|mouth|laryngeal|larynx|pharyngeal|head\s+and\s+neck)\s+"
        r"(?:cancer|tumou?r)\b",
    ),
    "Liver": (r"\b(?:liver|hepatic)(?:\s+cancer|\s+tumou?r|\s+mass)?\b",),
    "Neurological/Brain": (r"\b(?:brain|cns|neurological)\s+(?:cancer|tumou?r)\b",),
    "Ovarian": (r"\bovari(?:an|y)(?:\s+cancer|\s+tumou?r|\s+mass)?\b",),
    "Prostate": (r"\bprostat(?:e|ic)(?:\s+cancer|\s+tumou?r|\s+symptom)?\b",),
    "Sarcoma/Bone": (r"\b(?:sarcoma|bone\s+cancer|bone\s+tumou?r)\b",),
    "Skin/Melanoma": (r"\b(?:melanoma|skin\s+cancer|suspicious\s+mole)\b",),
    "Testicular": (r"\btestic(?:le|ular)(?:\s+cancer|\s+tumou?r|\s+lump)?\b",),
    "Thyroid": (r"\bthyroid(?:\s+cancer|\s+tumou?r|\s+lump)?\b",),
}

_EMERGENCY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\b(?:cannot|can't|unable\s+to|struggling\s+to)\s+breathe\b",
        r"\bsevere\s+(?:difficulty\s+breathing|breathlessness|shortness\s+of\s+breath)\b",
        r"\b(?:collapsed|unconscious|not\s+responding|newly\s+confused)\b",
        r"\bsevere\s+chest\s+pain\b",
        r"\b(?:coughing|vomiting|bringing\s+up)\s+(?:a\s+)?large\s+amounts?\s+of\s+blood\b",
        r"\b(?:bleeding|blood\s+loss)\s+(?:will\s+not|won't|does\s+not)\s+stop\b",
        r"\b(?:vomiting|passing|coughing)\s+blood\b.{0,80}\b(?:faint|dizz|collapse|weak)\w*\b",
        r"\b(?:faint|dizz|collapse|weak)\w*\b.{0,80}\b(?:vomiting|passing|coughing)\s+blood\b",
    )
)

_DIAGNOSIS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bdo\s+i\s+have\s+cancer\b",
        r"\bhave\s+i\s+got\s+cancer\b",
        r"\b(?:does|do)\s+(?:this|these\s+symptoms?|it)\s+mean\s+(?:i|we|he|she|they)\s+"
        r"(?:have|has)\s+cancer\b",
        r"\bis\s+(?:this|it)\s+cancer\b",
        r"\btell\s+me\s+(?:whether|if)\s+i\s+have\s+cancer\b",
        r"\bdiagnos(?:e|is)\s+(?:me|my|this|the\s+patient)\b",
        r"\bwhat\s+(?:type\s+of\s+)?cancer\s+do\s+i\s+have\b",
        r"\bconfirm\s+(?:that\s+)?(?:i|the\s+patient)\s+(?:have|has)\s+cancer\b",
    )
)

_PATIENT_SPECIFIC_RE = re.compile(
    r"\b(?:i|me|my|we|our|patient|this\s+person|he|she|they|man|woman|person)\b",
    flags=re.IGNORECASE,
)
_AGE_RE = re.compile(
    r"\b(?:aged?\s*\d{1,3}|\d{1,3}\s*[- ]?years?[- ]?old|"
    r"(?:i\s+am|i'm|patient\s+is|person\s+is|he\s+is|she\s+is|they\s+are)\s+"
    r"\d{1,3})(?:\s*years?\s+old)?\b|"
    r"\b(?:child|teenager|young\s+person)\b",
    flags=re.IGNORECASE,
)
_SYMPTOM_RE = re.compile(
    r"\b(?:symptom|finding|cough|haemoptysis|hemoptysis|blood|bleed|haematuria|hematuria|"
    r"dysuria|urinary\s+tract\s+infection|uti|dysphagia|dyspepsia|reflux|jaundice|"
    r"weight\s+loss|appetite\s+loss|abdominal\s+pain|chest\s+pain|breathlessness|fatigue|"
    r"anaemia|anemia|thrombocytosis|vomit|nausea|rectal\s+mass|abdominal\s+mass|"
    r"change\s+in\s+bowel\s+habit|fit\s+(?:result|score|test)|x[- ]?ray|raised\s+white\s+"
    r"cell\s+count|wcc)\w*\b",
    flags=re.IGNORECASE,
)
_INLINE_CITATION_RE = re.compile(
    r"\bper\s+(\d+(?:\.\d+)+)(?!\.\d|\d)",
    flags=re.IGNORECASE,
)


def detect_out_of_scope_sites(query: str) -> list[str]:
    """Return explicitly mentioned cancer sites outside the configured scope."""

    detected: list[str] = []
    for site, patterns in _OUT_OF_SCOPE_SITE_PATTERNS.items():
        if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in patterns):
            detected.append(site)
    return detected


def contains_emergency_signal(query: str) -> bool:
    """Detect high-specificity acute danger language without over-triggering on NG12 terms."""

    return any(pattern.search(query) for pattern in _EMERGENCY_PATTERNS)


def is_diagnosis_request(query: str) -> bool:
    """Detect requests that ask the system to determine whether cancer is present."""

    return any(pattern.search(query) for pattern in _DIAGNOSIS_PATTERNS)


def _clarifying_question(query: str) -> str | None:
    """Return one concise question when a patient-specific request lacks key inputs."""

    if not _PATIENT_SPECIFIC_RE.search(query):
        return None
    has_age = bool(_AGE_RE.search(query))
    has_symptom = bool(_SYMPTOM_RE.search(query))
    if has_age and has_symptom:
        return None
    if not has_age and not has_symptom:
        return "What is the person's age, and what specific symptom, sign, or test result is being assessed?"
    if not has_age:
        return "What is the person's age?"
    return "What specific symptom, sign, or test result is being assessed?"


def classify_query(query: str) -> GuardrailDecision:
    """Apply deterministic pre-retrieval safety rules in strict precedence order."""

    cleaned = query.strip()
    if not cleaned:
        return GuardrailDecision(
            classification=InputClassification.NEEDS_CAUTION,
            reason=GuardrailReason.UNDERSPECIFIED,
            message="The request is empty, so no referral rule can be assessed safely.",
            clarifying_question=(
                "What cancer-site question, age, symptom, sign, or test result should be assessed?"
            ),
        )

    if contains_emergency_signal(cleaned):
        return GuardrailDecision(
            classification=InputClassification.REFUSE_AND_REDIRECT,
            reason=GuardrailReason.EMERGENCY,
            message=(
                "This assistant cannot safely assess a possible emergency. Seek immediate "
                "medical care now; in the UK, call 999 or go to A&E."
            ),
        )

    if is_diagnosis_request(cleaned):
        return GuardrailDecision(
            classification=InputClassification.REFUSE_AND_REDIRECT,
            reason=GuardrailReason.DIAGNOSIS_REQUEST,
            message=(
                "I cannot determine whether someone has cancer. I can instead explain "
                "whether a stated age, symptom, sign, or test result matches a retrieved "
                "NICE NG12 referral or investigation threshold."
            ),
        )

    out_of_scope_sites = detect_out_of_scope_sites(cleaned)
    if out_of_scope_sites:
        named_sites = ", ".join(out_of_scope_sites)
        return GuardrailDecision(
            classification=InputClassification.REFUSE_AND_REDIRECT,
            reason=GuardrailReason.OUT_OF_SCOPE_SITE,
            message=(
                f"The request mentions an out-of-scope cancer site: {named_sites}. "
                f"This system covers only {COVERED_SCOPE}."
            ),
            detected_out_of_scope_sites=out_of_scope_sites,
        )

    question = _clarifying_question(cleaned)
    if question:
        return GuardrailDecision(
            classification=InputClassification.NEEDS_CAUTION,
            reason=GuardrailReason.UNDERSPECIFIED,
            message=(
                "The patient-specific request is missing information needed to select the "
                "correct age- and symptom-dependent recommendation without guessing."
            ),
            clarifying_question=question,
        )

    return GuardrailDecision(
        classification=InputClassification.ALLOWED,
        reason=GuardrailReason.ALLOWED,
        message="The request is within the configured NG12 referral scope.",
    )


def _coerce_similarity_score(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= score <= 1.0:
        return None
    return score


def assess_retrieval_confidence(
    chunks: Sequence[Mapping[str, Any]],
    *,
    config: GuardrailConfig | None = None,
) -> RetrievalAssessment:
    """Map top similarity to a calibrated confidence gate.

    The function expects normalized similarities in ``[0, 1]`` where larger is better.
    Missing scores are conservatively downgraded to Low unless strict score presence is
    configured. Invalid or out-of-range scores never contribute to confidence.
    """

    policy = config or GuardrailConfig()
    if not chunks:
        return RetrievalAssessment(
            confidence=ConfidenceLevel.INSUFFICIENT,
            top_score=None,
            should_answer=False,
            reason="No relevant chunks were retrieved.",
        )

    scores = [
        score
        for score in (_coerce_similarity_score(chunk.get("similarity_score")) for chunk in chunks)
        if score is not None
    ]
    if not scores:
        if policy.require_similarity_scores:
            return RetrievalAssessment(
                confidence=ConfidenceLevel.INSUFFICIENT,
                top_score=None,
                should_answer=False,
                reason="Retrieved chunks did not include valid normalized similarity scores.",
            )
        return RetrievalAssessment(
            confidence=ConfidenceLevel.LOW,
            top_score=None,
            should_answer=True,
            reason=(
                "Similarity scores were unavailable; confidence was conservatively "
                "downgraded and clinician review is recommended."
            ),
        )

    top_score = max(scores)
    if top_score < policy.minimum_similarity_threshold:
        return RetrievalAssessment(
            confidence=ConfidenceLevel.INSUFFICIENT,
            top_score=top_score,
            should_answer=False,
            reason=(
                f"Top similarity {top_score:.3f} is below the minimum grounded-answer "
                f"threshold {policy.minimum_similarity_threshold:.3f}."
            ),
        )
    if top_score >= policy.high_similarity_threshold:
        confidence = ConfidenceLevel.HIGH
    elif top_score >= policy.medium_similarity_threshold:
        confidence = ConfidenceLevel.MEDIUM
    else:
        confidence = ConfidenceLevel.LOW

    return RetrievalAssessment(
        confidence=confidence,
        top_score=top_score,
        should_answer=True,
        reason=f"Top normalized similarity score is {top_score:.3f}.",
    )


def _normalise_whitespace(value: str) -> str:
    return " ".join(value.split())


def _normalise_title(value: str) -> str:
    return _normalise_whitespace(value).strip().casefold()


def _normalise_reference(value: str) -> str:
    return _normalise_whitespace(value).casefold()


_TYPED_REFERENCE_RE = re.compile(
    r"(?:NG12, Recommendation \d+(?:\.\d+)+, p\.\d+|"
    r"NG12 Full Guideline, Chapter \d+(?:\.\d+)* \([^)]+\), p\.\d+)",
    flags=re.IGNORECASE,
)


def _chunk_document_type(chunk: Mapping[str, Any]) -> DocumentType | None:
    raw = str(chunk.get("document_type") or "").strip()
    if not raw:
        raw = (
            DocumentType.NG12_SHORT.value
            if chunk.get("recommendation_id")
            else DocumentType.NG12_FULL.value
        )
    try:
        return DocumentType(raw)
    except ValueError:
        return None


def _source_identity_matches(citation: Citation, chunk: Mapping[str, Any]) -> bool:
    if _chunk_document_type(chunk) is not citation.document_type:
        return False
    if citation.document_type is DocumentType.NG12_SHORT:
        return str(chunk.get("recommendation_id") or "").strip() == citation.recommendation_id
    return (
        str(chunk.get("chapter_number") or "").strip() == citation.chapter_number
        and _normalise_title(str(chunk.get("chapter_title") or ""))
        == _normalise_title(citation.chapter_title or "")
    )


def _chunk_candidates(
    chunks: Sequence[Mapping[str, Any]], citation: Citation
) -> Iterable[Mapping[str, Any]]:
    for chunk in chunks:
        if _source_identity_matches(citation, chunk):
            yield chunk


def _citation_matches_chunk(citation: Citation, chunk: Mapping[str, Any]) -> bool:
    if not _source_identity_matches(citation, chunk):
        return False
    try:
        page_number = int(chunk.get("page_number"))
    except (TypeError, ValueError):
        return False
    if page_number != citation.page_number:
        return False
    if _normalise_title(str(chunk.get("section_title", ""))) != _normalise_title(
        citation.section_title
    ):
        return False
    source_text = _normalise_whitespace(str(chunk.get("text", "")))
    quote = _normalise_whitespace(citation.quoted_text)
    return bool(quote) and quote in source_text


def _typed_references(text: str) -> set[str]:
    return {_normalise_reference(match) for match in _TYPED_REFERENCE_RE.findall(text)}


def validate_response_sources(
    response: FullResponse,
    chunks: Sequence[Mapping[str, Any]],
    *,
    config: GuardrailConfig | None = None,
) -> list[SourceValidationIssue]:
    """Validate typed identity, metadata, exact quotes, and inline completeness."""

    if response.refusal_reason or response.clarifying_question:
        return []

    policy = config or GuardrailConfig()
    issues: list[SourceValidationIssue] = []
    structured_references = {
        _normalise_reference(citation.formatted_reference)
        for evidence in response.evidence_list
        for citation in evidence.supporting_citations
    }
    summary_references = _typed_references(response.recommendation_summary)
    if not summary_references:
        issues.append(
            SourceValidationIssue(
                code="summary_missing_inline_citation",
                message="The recommendation summary contains no complete typed citation.",
            )
        )
    for reference in sorted(summary_references - structured_references):
        issues.append(
            SourceValidationIssue(
                citation_reference=reference,
                code="summary_unknown_citation",
                message=f"The summary citation is not bound to evidence: {reference}.",
            )
        )

    for claim_index, evidence in enumerate(response.evidence_list):
        inline_references = _typed_references(evidence.claim)
        citation_references = {
            _normalise_reference(citation.formatted_reference)
            for citation in evidence.supporting_citations
        }
        for reference in sorted(inline_references - citation_references):
            issues.append(
                SourceValidationIssue(
                    claim_index=claim_index,
                    citation_reference=reference,
                    code="inline_citation_without_source",
                    message=(
                        f"Claim {claim_index} contains a typed citation without a matching "
                        f"Citation object: {reference}."
                    ),
                )
            )

        for citation in evidence.supporting_citations:
            reference = citation.formatted_reference
            candidates = list(_chunk_candidates(chunks, citation))
            if not candidates:
                issues.append(
                    SourceValidationIssue(
                        claim_index=claim_index,
                        recommendation_id=citation.recommendation_id,
                        citation_reference=reference,
                        document_type=citation.document_type,
                        code="source_not_retrieved",
                        message=f"The cited source was not retrieved: {reference}.",
                    )
                )
                continue

            if len(_normalise_whitespace(citation.quoted_text)) < policy.minimum_quote_characters:
                issues.append(
                    SourceValidationIssue(
                        claim_index=claim_index,
                        recommendation_id=citation.recommendation_id,
                        citation_reference=reference,
                        document_type=citation.document_type,
                        code="quote_too_short",
                        message=f"The quote for {reference} is too short to prove support.",
                    )
                )

            if not any(_citation_matches_chunk(citation, chunk) for chunk in candidates):
                issues.append(
                    SourceValidationIssue(
                        claim_index=claim_index,
                        recommendation_id=citation.recommendation_id,
                        citation_reference=reference,
                        document_type=citation.document_type,
                        code="citation_metadata_or_quote_mismatch",
                        message=(
                            f"{reference} does not match a retrieved chunk by type, source "
                            "identity, section, page, and exact quoted text."
                        ),
                    )
                )

    return issues


def detect_unsupported_claims(
    response: FullResponse,
    chunks: Sequence[Mapping[str, Any]],
    *,
    config: GuardrailConfig | None = None,
) -> list[str]:
    """Return concise unsupported-claim explanations for post-generation routing."""

    return [issue.message for issue in validate_response_sources(response, chunks, config=config)]


def ensure_fixed_disclaimer(response: FullResponse) -> FullResponse:
    """Restore the mandatory clinical disclaimer on every response path."""

    return response.model_copy(update={"disclaimer": CLINICAL_DISCLAIMER})


def unsupported_generation_response(reasons: Sequence[str]) -> FullResponse:
    """Fail closed when generated content cannot be proven against retrieved sources."""

    concise_reasons = "; ".join(dict.fromkeys(reason.strip() for reason in reasons if reason))
    if not concise_reasons:
        concise_reasons = "One or more generated claims could not be verified."
    return FullResponse.refusal(
        reason=(
            "I cannot provide the generated recommendation because the post-generation "
            f"grounding check failed: {concise_reasons}"
        )
    )


__all__ = [
    "COVERED_SCOPE",
    "GuardrailConfig",
    "GuardrailDecision",
    "GuardrailReason",
    "InputClassification",
    "RetrievalAssessment",
    "SourceValidationIssue",
    "assess_retrieval_confidence",
    "classify_query",
    "contains_emergency_signal",
    "detect_out_of_scope_sites",
    "detect_unsupported_claims",
    "ensure_fixed_disclaimer",
    "is_diagnosis_request",
    "unsupported_generation_response",
    "validate_response_sources",
]
