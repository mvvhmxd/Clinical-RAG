"""Typed domain models for the NG12 RAG pipeline.

The models in this module are deliberately strict: provenance and citation fields are
required so an apparently useful answer cannot silently lose its source binding.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ActionType(str, Enum):
    """Normalised clinical action represented by a recommendation."""

    REFER = "refer"
    CONSIDER_REFERRAL = "consider_referral"
    OFFER_INVESTIGATION = "offer_investigation"
    SAFETY_NETTING = "safety_netting"
    PATIENT_SUPPORT = "patient_support"
    DIAGNOSTIC_PROCESS = "diagnostic_process"
    CLINICAL_ASSESSMENT = "clinical_assessment"


class RuleType(str, Enum):
    """Logical shape of a recommendation."""

    SINGLE_CONDITION = "single_condition"
    MULTI_BRANCH = "multi_branch"
    PROBABILITY_BASED = "probability_based"
    THRESHOLD_BASED = "threshold_based"
    PROCESS = "process"


class ResponseStatus(str, Enum):
    """Top-level safety state returned by the pipeline."""

    ANSWERED = "answered"
    UNCERTAIN = "uncertain"
    REFUSED = "refused"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RawRecommendation(BaseModel):
    """A recommendation extracted from the PDF before semantic enrichment."""

    model_config = ConfigDict(extra="forbid")

    recommendation_id: str
    text: str = Field(min_length=1)
    section_id: str
    section_title: str
    subsection_title: str | None = None
    page_number: int = Field(ge=1)
    page_numbers: list[int] = Field(min_length=1)
    source_file: str
    source_sha256: str
    guideline_version: str

    @model_validator(mode="after")
    def validate_pages(self) -> "RawRecommendation":
        if self.page_numbers != sorted(set(self.page_numbers)):
            raise ValueError("page_numbers must be unique and sorted")
        if self.page_number != self.page_numbers[0]:
            raise ValueError("page_number must equal the first value in page_numbers")
        return self


class ChunkMetadata(BaseModel):
    """Retrieval metadata stored alongside every vector entry."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    guideline_id: str = "NG12"
    guideline_version: str
    document_title: str
    source_file: str
    source_sha256: str
    cancer_site: str
    recommendation_id: str
    page_number: int = Field(ge=1)
    page_numbers: list[int] = Field(min_length=1)
    section_id: str
    section_title: str
    subsection_title: str | None = None
    action_type: ActionType
    age_condition: str | None = None
    symptom_condition: str
    lab_threshold: str | None = None
    rule_type: RuleType
    revision_year: int = Field(ge=1900, le=2100)
    revision_history: list[int] = Field(min_length=1)
    is_synthetic_negative: bool = False
    synthetic_source_id: str | None = None
    synthetic_mutation: str | None = None

    @field_validator("recommendation_id")
    @classmethod
    def validate_recommendation_id(cls, value: str) -> str:
        parts = value.split(".")
        if len(parts) != 3 or any(not part.isdigit() for part in parts):
            raise ValueError("recommendation_id must look like '1.3.2'")
        return value

    @model_validator(mode="after")
    def validate_pages_and_negative_provenance(self) -> "ChunkMetadata":
        unique_pages = sorted(set(self.page_numbers))
        if unique_pages != self.page_numbers:
            raise ValueError("page_numbers must be unique and sorted")
        if self.page_number != self.page_numbers[0]:
            raise ValueError("page_number must equal the first value in page_numbers")
        if self.is_synthetic_negative and not self.synthetic_source_id:
            raise ValueError("synthetic negatives require synthetic_source_id")
        if not self.is_synthetic_negative and (
            self.synthetic_source_id or self.synthetic_mutation
        ):
            raise ValueError("real chunks cannot contain synthetic-negative provenance")
        return self


class RecommendationChunk(BaseModel):
    """One numbered NICE recommendation and its retrieval representation."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)
    metadata: ChunkMetadata

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not value.startswith("ng12-"):
            raise ValueError("chunk_id must start with 'ng12-'")
        return value


class ComponentScores(BaseModel):
    """Auditable scores emitted by hybrid retrieval and reranking."""

    model_config = ConfigDict(extra="forbid")

    vector_score: float | None = None
    vector_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    bm25_rank: int | None = Field(default=None, ge=1)
    rrf_score: float = 0.0
    feature_score: float = 0.0
    reranker_score: float | None = None
    final_score: float = 0.0
    numeric_conflict: bool = False


class RetrievedChunk(BaseModel):
    """A recommendation chunk plus all retrieval evidence."""

    model_config = ConfigDict(extra="forbid")

    chunk: RecommendationChunk
    scores: ComponentScores
    rank: int = Field(ge=1)


class SupportingEvidence(BaseModel):
    """A verbatim excerpt paired with its recommendation identifier."""

    model_config = ConfigDict(extra="forbid")

    quote: str = Field(min_length=1)
    recommendation_id: str


class Citation(BaseModel):
    """A complete source pointer suitable for display or audit."""

    model_config = ConfigDict(extra="forbid")

    document: str
    guideline_version: str
    recommendation_id: str
    section: str
    page: int = Field(ge=1)
    chunk_id: str


class GroundedClaim(BaseModel):
    """One generated claim with explicit recommendation-level support."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    recommendation_ids: list[str] = Field(min_length=1)


class FaithfulnessCheck(BaseModel):
    """Result of post-generation claim-to-source verification."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    citation_errors: list[str] = Field(default_factory=list)
    notes: str = ""


class RAGResponse(BaseModel):
    """Stable public response contract for the CLI and future UI/API."""

    model_config = ConfigDict(extra="forbid")

    status: ResponseStatus
    recommendation: str
    claims: list[GroundedClaim] = Field(default_factory=list)
    supporting_evidence: list[SupportingEvidence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence_level: ConfidenceLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    refusal_reason: str | None = None
    safety_notice: str
    faithfulness: FaithfulnessCheck | None = None

    @model_validator(mode="after")
    def validate_grounding_contract(self) -> "RAGResponse":
        if self.status == ResponseStatus.ANSWERED:
            if not self.claims or not self.supporting_evidence or not self.citations:
                raise ValueError(
                    "answered responses require claims, evidence, and citations"
                )
            cited = {citation.recommendation_id for citation in self.citations}
            for claim in self.claims:
                if not set(claim.recommendation_ids).issubset(cited):
                    raise ValueError("every claim recommendation_id must be cited")
        if self.status == ResponseStatus.REFUSED and not self.refusal_reason:
            raise ValueError("refused responses require refusal_reason")
        return self


GuardrailDecision = Literal["allow", "refuse_out_of_scope", "refuse_diagnosis", "clarify"]


class GuardrailResult(BaseModel):
    """Pre-retrieval scope and intent decision."""

    model_config = ConfigDict(extra="forbid")

    decision: GuardrailDecision
    reason: str
    user_message: str | None = None
