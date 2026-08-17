"""Grounded generation for the scoped NICE NG12 medical RAG system.

The public ``GroundedGenerator.generate`` method applies deterministic safety checks before
calling the model, requests a strict Pydantic-derived JSON schema, validates every citation
against the retrieved chunks, and performs a second model-assisted faithfulness pass. Any
unverifiable state returns a structured refusal rather than a best-effort clinical answer.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .llm_client import create_llm_client, default_generation_model
from .guardrails import (
    GuardrailConfig,
    assess_retrieval_confidence,
    classify_query,
    ensure_fixed_disclaimer,
    unsupported_generation_response,
    validate_response_sources,
)
from .prompts import (
    FAITHFULNESS_VERIFICATION_SYSTEM_PROMPT,
    GROUNDED_GENERATION_SYSTEM_PROMPT,
    build_faithfulness_user_prompt,
    build_generation_user_prompt,
)
from .response_schema import (
    ClaimVerification,
    ConfidenceLevel,
    DocumentType,
    Evidence,
    FaithfulnessReport,
    FullResponse,
)

logger = logging.getLogger(__name__)

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_RECOMMENDATION_ID_RE = re.compile(r"^\d+(?:\.\d+)+$")


class GenerationConfig(BaseModel):
    """Runtime configuration for generation and post-generation verification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(
        default_factory=lambda: os.getenv("NG12_GENERATION_MODEL") or default_generation_model()
    )
    verifier_model: str = Field(
        default_factory=lambda: os.getenv("NG12_VERIFICATION_MODEL") or default_generation_model()
    )
    max_completion_tokens: int = Field(default=5000, ge=500, le=30000)
    verifier_max_completion_tokens: int = Field(default=3500, ge=500, le=30000)
    request_timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0)
    max_context_chunks: int = Field(default=12, ge=1, le=50)
    max_chunk_characters: int = Field(default=12_000, ge=500, le=100_000)
    max_query_characters: int = Field(default=4000, ge=50, le=50_000)
    run_faithfulness_check: bool = True
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)


class ChunkNormalizationResult(BaseModel):
    """Validated chunks and non-fatal ingestion warnings."""

    model_config = ConfigDict(extra="forbid")

    chunks: list[dict[str, Any]]
    rejected_reasons: list[str]


def _mapping_from_chunk(chunk: Any) -> Mapping[str, Any] | None:
    if isinstance(chunk, Mapping):
        return chunk
    model_dump = getattr(chunk, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, Mapping) else None
    to_dict = getattr(chunk, "dict", None)
    if callable(to_dict):
        dumped = to_dict()
        return dumped if isinstance(dumped, Mapping) else None
    return None


def _first_present(chunk: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = chunk.get(key)
        if value is not None and value != "":
            return value
    return None


def _normalise_similarity(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 0.0 <= score <= 1.0 else None


def normalize_retrieved_chunks(
    chunks: Sequence[Any],
    *,
    config: GenerationConfig,
) -> ChunkNormalizationResult:
    """Normalize flat chunks or this repository's nested ``RetrievedChunk`` models.

    ``NG12_short`` requires a numbered recommendation. ``NG12_full`` requires chapter
    number/title provenance and must not carry a fabricated recommendation identifier.
    Invalid records are excluded from model context rather than repaired.
    """

    normalized: list[dict[str, Any]] = []
    rejected: list[str] = []
    seen: set[tuple[str, str, int, str]] = set()

    for index, raw_chunk in enumerate(chunks[: config.max_context_chunks]):
        outer = _mapping_from_chunk(raw_chunk)
        if outer is None:
            rejected.append(f"chunk {index}: expected a mapping or Pydantic model")
            continue

        chunk: Mapping[str, Any] = outer
        scores: Mapping[str, Any] = {}
        nested_chunk = _mapping_from_chunk(outer.get("chunk"))
        if nested_chunk is not None:
            metadata = _mapping_from_chunk(nested_chunk.get("metadata")) or {}
            chunk = {
                **metadata,
                "text": nested_chunk.get("text"),
                "chunk_id": nested_chunk.get("chunk_id"),
            }
            scores = _mapping_from_chunk(outer.get("scores")) or {}

        recommendation_id = str(
            _first_present(chunk, "recommendation_id", "recommendation", "reco_id") or ""
        ).strip()
        chapter_number = str(
            _first_present(chunk, "chapter_number", "chapter", "chapter_id") or ""
        ).strip()
        chapter_title = str(
            _first_present(chunk, "chapter_title", "chapter_name") or ""
        ).strip()
        raw_document_type = str(
            _first_present(chunk, "document_type", "source_type") or ""
        ).strip()
        if not raw_document_type:
            raw_document_type = (
                DocumentType.NG12_SHORT.value if recommendation_id else DocumentType.NG12_FULL.value
            )
        try:
            document_type = DocumentType(raw_document_type)
        except ValueError:
            rejected.append(f"chunk {index}: unsupported document_type {raw_document_type!r}")
            continue

        section_title = str(
            _first_present(chunk, "section_title", "section", "heading") or ""
        ).strip()
        source_text = str(
            _first_present(chunk, "text", "content", "chunk_text", "document") or ""
        ).strip()
        raw_page = _first_present(chunk, "page_number", "page", "source_page")

        if document_type is DocumentType.NG12_SHORT:
            if not _RECOMMENDATION_ID_RE.fullmatch(recommendation_id):
                rejected.append(f"chunk {index}: invalid or missing recommendation_id")
                continue
            chapter_number = ""
            chapter_title = ""
            source_identity = recommendation_id
        else:
            if recommendation_id:
                rejected.append(f"chunk {index}: NG12_full must not contain recommendation_id")
                continue
            if not chapter_number or not chapter_title:
                rejected.append(f"chunk {index}: NG12_full requires chapter_number and chapter_title")
                continue
            source_identity = f"{chapter_number}:{chapter_title}"

        if not section_title:
            rejected.append(f"chunk {index}: missing section_title")
            continue
        try:
            page_number = int(raw_page)
        except (TypeError, ValueError):
            rejected.append(f"chunk {index}: invalid or missing page_number")
            continue
        if page_number < 1:
            rejected.append(f"chunk {index}: page_number must be positive")
            continue
        if not source_text:
            rejected.append(f"chunk {index}: missing source text")
            continue
        if len(source_text) > config.max_chunk_characters:
            rejected.append(
                f"chunk {index}: source text exceeds {config.max_chunk_characters} characters"
            )
            continue

        key = (document_type.value, source_identity, page_number, source_text)
        if key in seen:
            continue
        seen.add(key)

        normalized_chunk: dict[str, Any] = {
            "document_type": document_type.value,
            "recommendation_id": recommendation_id or None,
            "chapter_number": chapter_number or None,
            "chapter_title": chapter_title or None,
            "section_title": section_title,
            "page_number": page_number,
            "text": source_text,
        }
        cancer_site = _first_present(chunk, "cancer_site", "site")
        if cancer_site is not None:
            normalized_chunk["cancer_site"] = str(cancer_site).strip()
        score = _normalise_similarity(
            _first_present(
                chunk,
                "similarity_score",
                "score",
                "relevance_score",
            )
        ) or _normalise_similarity(
            _first_present(scores, "final_score", "reranker_score", "vector_score")
        )
        if score is not None:
            normalized_chunk["similarity_score"] = score
        normalized.append(normalized_chunk)

    return ChunkNormalizationResult(chunks=normalized, rejected_reasons=rejected)


def _strict_response_format(model_type: type[BaseModel], *, name: str) -> dict[str, Any]:
    """Build the strict OpenAI-compatible JSON Schema response format."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": model_type.model_json_schema(),
        },
    }


def _model_token_limit(model: str, token_limit: int) -> dict[str, int]:
    """Select the token parameter accepted by the configured provider family."""

    lowered = model.casefold()
    if lowered.startswith("gemini") or lowered.startswith("gemma"):
        # Gemini's token budget travels inside generationConfig, set by the client adapter.
        return {}
    if lowered.startswith("gpt-"):
        return {"max_completion_tokens": token_limit}
    return {"max_tokens": token_limit}


def _extract_message_content(completion: Any) -> str:
    try:
        message = completion.choices[0].message
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("Model response did not contain a chat-completion message") from exc

    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError("The model declined the structured generation request")
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model response contained no structured JSON content")
    return content


def _parse_structured(content: str, model_type: type[_ModelT]) -> _ModelT:
    """Validate raw JSON without attempting to repair malformed model output."""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Model response was not valid JSON") from exc
    return model_type.model_validate(payload)


def _cap_answer_confidence(
    response: FullResponse,
    *,
    retrieval_cap: ConfidenceLevel,
    verifier_report: FaithfulnessReport | None = None,
) -> FullResponse:
    """Conservatively cap claim and overall confidence by retrieval and verification."""

    if response.refusal_reason or response.clarifying_question:
        return ensure_fixed_disclaimer(response)

    verifier_by_index: dict[int, ClaimVerification] = {}
    verifier_overall = retrieval_cap
    if verifier_report is not None:
        verifier_by_index = {
            verdict.claim_index: verdict for verdict in verifier_report.claim_verifications
        }
        verifier_overall = verifier_report.overall_confidence

    capped_evidence: list[Evidence] = []
    for index, evidence in enumerate(response.evidence_list):
        caps = [evidence.confidence, retrieval_cap]
        verdict = verifier_by_index.get(index)
        if verdict is not None:
            caps.append(verdict.confidence)
        # Copy-with-update rather than reconstructing field by field: an explicit rebuild
        # silently drops any field added later, which is how condition_evaluations went
        # missing from capped answers.
        capped_evidence.append(
            evidence.model_copy(update={"confidence": ConfidenceLevel.minimum(*caps)})
        )

    overall = ConfidenceLevel.minimum(
        response.overall_confidence,
        retrieval_cap,
        verifier_overall,
        *(evidence.confidence for evidence in capped_evidence),
    )
    return response.model_copy(
        update={
            "evidence_list": capped_evidence,
            "overall_confidence": overall,
            "refusal_reason": None,
            "clarifying_question": None,
        }
    )


def _validate_faithfulness_report(
    report: FaithfulnessReport,
    response: FullResponse,
) -> list[str]:
    """Verify report cardinality and cited-ID binding before trusting its verdicts."""

    issues: list[str] = []
    expected_indexes = set(range(len(response.evidence_list)))
    actual_indexes = {verdict.claim_index for verdict in report.claim_verifications}
    if len(report.claim_verifications) != len(actual_indexes):
        issues.append("The verifier returned duplicate claim indexes.")
    if actual_indexes != expected_indexes:
        issues.append("The verifier did not return exactly one verdict for every claim.")
    if not report.summary_supported:
        issues.append(
            "Verifier rejected the recommendation summary: "
            f"{report.summary_explanation}"
        )

    for verdict in report.claim_verifications:
        if verdict.claim_index not in expected_indexes:
            continue
        cited_references = {
            citation.formatted_reference
            for citation in response.evidence_list[verdict.claim_index].supporting_citations
        }
        verified_references = set(verdict.supporting_citation_references)
        if verdict.supported and verified_references != cited_references:
            issues.append(
                f"Verifier claim {verdict.claim_index} did not bind exactly to its typed "
                "citation references."
            )
        if not verdict.supported:
            issues.append(f"Verifier rejected claim {verdict.claim_index}: {verdict.explanation}")

    if not report.all_claims_supported:
        issues.extend(report.unsupported_claims)
    return list(dict.fromkeys(issues))


class GroundedGenerator:
    """Orchestrate safe query routing, generation, and faithfulness verification."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        config: GenerationConfig | None = None,
    ) -> None:
        self.config = config or GenerationConfig()
        # Gemini when GEMINI_API_KEY is present, OpenAI otherwise; injected clients win.
        self.client = client or create_llm_client()

    def _call_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_model: type[_ModelT],
        schema_name: str,
        max_tokens: int,
    ) -> _ModelT:
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=_strict_response_format(output_model, name=schema_name),
            timeout=self.config.request_timeout_seconds,
            **_model_token_limit(model, max_tokens),
        )
        return _parse_structured(_extract_message_content(completion), output_model)

    def _verify_faithfulness(
        self,
        *,
        query: str,
        response: FullResponse,
        chunks: Sequence[Mapping[str, Any]],
    ) -> FaithfulnessReport:
        return self._call_structured(
            model=self.config.verifier_model,
            system_prompt=FAITHFULNESS_VERIFICATION_SYSTEM_PROMPT,
            user_prompt=build_faithfulness_user_prompt(
                query=query,
                response=response.model_dump(mode="json"),
                chunks=chunks,
            ),
            output_model=FaithfulnessReport,
            schema_name="ng12_faithfulness_report",
            max_tokens=self.config.verifier_max_completion_tokens,
        )

    def generate(
        self,
        query: str,
        retrieved_chunks: Sequence[Any] | None,
    ) -> FullResponse:
        """Return a grounded answer, clarification, or safe structured refusal."""

        if not isinstance(query, str):
            return FullResponse.refusal(reason="The query must be provided as text.")
        if len(query) > self.config.max_query_characters:
            return FullResponse.refusal(
                reason=(
                    "The query is too long to assess safely. Please provide one concise "
                    "referral-threshold question."
                )
            )

        input_decision = classify_query(query)
        blocked_response = input_decision.to_response()
        if blocked_response is not None:
            return ensure_fixed_disclaimer(blocked_response)

        raw_chunks = list(retrieved_chunks or [])
        normalization = normalize_retrieved_chunks(raw_chunks, config=self.config)
        if not normalization.chunks:
            detail = (
                f" Source validation rejected the supplied context: "
                f"{'; '.join(normalization.rejected_reasons)}"
                if normalization.rejected_reasons
                else ""
            )
            return FullResponse.refusal(
                reason=(
                    "No valid retrieved NG12 recommendation chunks were available, so an "
                    f"answer would be ungrounded.{detail}"
                )
            )

        retrieval = assess_retrieval_confidence(
            normalization.chunks,
            config=self.config.guardrails,
        )
        if not retrieval.should_answer:
            return FullResponse.refusal(
                reason=(
                    "The retrieved evidence is not relevant enough for a safe answer. "
                    f"{retrieval.reason}"
                )
            )

        try:
            response = self._call_structured(
                model=self.config.model,
                system_prompt=GROUNDED_GENERATION_SYSTEM_PROMPT,
                user_prompt=build_generation_user_prompt(
                    query=query,
                    chunks=normalization.chunks,
                    retrieval_confidence=retrieval.confidence.value,
                ),
                output_model=FullResponse,
                schema_name="ng12_grounded_response",
                max_tokens=self.config.max_completion_tokens,
            )
        except (ValidationError, ValueError, TypeError, RuntimeError) as exc:
            logger.warning("Grounded generation failed validation: %s", exc)
            return FullResponse.refusal(
                reason=(
                    "The model did not return a structurally valid grounded response, so the "
                    "system failed closed."
                )
            )
        except Exception:
            logger.exception("Grounded generation request failed")
            return FullResponse.refusal(
                reason=(
                    "The grounded generation service was unavailable, so no clinical answer "
                    "was produced."
                )
            )

        response = ensure_fixed_disclaimer(response)
        if response.refusal_reason or response.clarifying_question:
            return response

        source_issues = validate_response_sources(
            response,
            normalization.chunks,
            config=self.config.guardrails,
        )
        if source_issues:
            return unsupported_generation_response([issue.message for issue in source_issues])

        report: FaithfulnessReport | None = None
        if self.config.run_faithfulness_check:
            try:
                report = self._verify_faithfulness(
                    query=query,
                    response=response,
                    chunks=normalization.chunks,
                )
            except (ValidationError, ValueError, TypeError, RuntimeError) as exc:
                logger.warning("Faithfulness verification failed validation: %s", exc)
                return FullResponse.refusal(
                    reason=(
                        "The post-generation faithfulness check did not return a valid verdict, "
                        "so the answer was withheld."
                    )
                )
            except Exception:
                logger.exception("Faithfulness verification request failed")
                return FullResponse.refusal(
                    reason=(
                        "The post-generation faithfulness check was unavailable, so the answer "
                        "was withheld."
                    )
                )

            faithfulness_issues = _validate_faithfulness_report(report, response)
            if faithfulness_issues:
                return unsupported_generation_response(faithfulness_issues)

        try:
            return _cap_answer_confidence(
                response,
                retrieval_cap=retrieval.confidence,
                verifier_report=report,
            )
        except ValidationError as exc:
            logger.warning("Confidence capping produced an invalid response: %s", exc)
            return FullResponse.refusal(
                reason=(
                    "The final confidence calibration was inconsistent, so the answer was withheld."
                )
            )


def generate_grounded_response(
    query: str,
    retrieved_chunks: Sequence[Any] | None,
    *,
    client: Any | None = None,
    config: GenerationConfig | None = None,
) -> FullResponse:
    """Convenience wrapper for one grounded generation request."""

    return GroundedGenerator(client=client, config=config).generate(
        query,
        retrieved_chunks,
    )


# Backwards-friendly alias for simple integrations.
generate_response = generate_grounded_response


__all__ = [
    "ChunkNormalizationResult",
    "GenerationConfig",
    "GroundedGenerator",
    "generate_grounded_response",
    "generate_response",
    "normalize_retrieved_chunks",
]
