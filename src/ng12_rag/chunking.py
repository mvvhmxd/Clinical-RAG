"""Recommendation-aware chunking and metadata enrichment for NG12.

One numbered recommendation is always one chunk. No recommendation is split or merged,
which keeps retrieval results directly citable and prevents clinical conditions from being
detached from their action.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from ng12_rag.config import Settings
from ng12_rag.models import (
    ActionType,
    ChunkMetadata,
    RawRecommendation,
    RecommendationChunk,
    RuleType,
)

LOGGER = logging.getLogger(__name__)

REVISION_CITATION_PATTERN = re.compile(
    r"\[((?:19|20)\d{2})(?:,\s*amended\s*((?:19|20)\d{2}))?\]",
    re.IGNORECASE,
)
AGE_PATTERN = re.compile(
    r"\baged\s+(?:under\s+\d+|\d+\s+(?:and\s+over|or\s+over|to\s+\d+))\b",
    re.IGNORECASE,
)
NUMERIC_THRESHOLD_PATTERNS = (
    re.compile(
        r"\b(?:at\s+least|below|more\s+than|greater\s+than|less\s+than)\s+"
        r"\d+(?:\.\d+)?(?:\s+[\wµμ/-]+){0,8}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bFIT\s+result\s+(?:of\s+)?(?:at\s+least|below)\s+\d+[^.;:]*",
        re.IGNORECASE,
    ),
    re.compile(r"\bPSA\s+levels?\s+are\s+above\s+the\s+threshold[^.;:]*", re.IGNORECASE),
)

SPECIFIC_CANCER_SITE: dict[str, str] = {
    "Lung cancer": "lung",
    "Mesothelioma": "mesothelioma",
    "Oesophageal cancer": "oesophageal",
    "Pancreatic cancer": "pancreatic",
    "Stomach cancer": "stomach",
    "Gall bladder cancer": "gall_bladder",
    "Liver cancer": "liver",
    "Colorectal cancer": "colorectal",
    "Anal cancer": "anal",
    "Prostate cancer": "prostate",
    "Bladder cancer": "bladder",
    "Renal cancer": "renal",
    "Testicular cancer": "testicular",
    "Penile cancer": "penile",
}

# Each mutation is intentionally adjacent to a real clause but clinically wrong. The
# source record remains linked in metadata and the text is unmistakably labelled.
NEGATIVE_MUTATIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "1.1.2",
        "aged 40 and over",
        "aged 30 and over",
        "age lowered from 40 to 30",
    ),
    (
        "1.2.1",
        "aged 55 and over",
        "aged 45 and over",
        "age lowered from 55 to 45",
    ),
    (
        "1.2.5",
        "aged 60 and over",
        "aged 50 and over",
        "age lowered from 60 to 50",
    ),
    (
        "1.3.2",
        "at least 10 micrograms",
        "at least 20 micrograms",
        "FIT threshold changed from 10 to 20 micrograms",
    ),
    (
        "1.6.4",
        "aged 45 and over",
        "aged 35 and over",
        "visible-haematuria age lowered from 45 to 35",
    ),
    (
        "1.6.6",
        "aged 45 and over",
        "aged 35 and over",
        "renal-cancer age lowered from 45 to 35",
    ),
)


class ChunkingError(RuntimeError):
    """Raised when a recommendation cannot be safely transformed into a chunk."""


def read_raw_recommendations(path: Path) -> list[RawRecommendation]:
    """Read strict raw recommendation records from deterministic JSON Lines."""

    records: list[RawRecommendation] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(RawRecommendation.model_validate_json(line))
            except Exception as exc:  # Pydantic supplies field-level context.
                raise ChunkingError(f"Invalid raw record at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ChunkingError(f"No raw recommendation records found in {path}")
    return records


def _body_without_id_or_revision(record: RawRecommendation) -> str:
    body = re.sub(
        rf"^\s*{re.escape(record.recommendation_id)}\s+", "", record.text, count=1
    )
    return REVISION_CITATION_PATTERN.sub("", body).strip()


def extract_revision_history(text: str) -> tuple[list[int], int]:
    """Return source and amendment years from the terminal NICE citation."""

    matches = list(REVISION_CITATION_PATTERN.finditer(text))
    if not matches:
        raise ChunkingError(f"Recommendation is missing a revision citation: {text[:90]!r}")
    match = matches[-1]
    years = [int(match.group(1))]
    if match.group(2):
        years.append(int(match.group(2)))
    return years, years[-1]


def extract_action_type(record: RawRecommendation) -> ActionType:
    """Normalise recommendation wording into an action category."""

    body = _body_without_id_or_revision(record).casefold()
    if record.recommendation_id == "1.3.3" or "safety netting" in body:
        return ActionType.SAFETY_NETTING
    if record.recommendation_id == "1.3.4" or "additional help" in body:
        return ActionType.PATIENT_SUPPORT
    if re.match(r"consider(?:\s+an?)?\s+.*referr|consider\s+referring", body):
        return ActionType.CONSIDER_REFERRAL
    if body.startswith("refer "):
        return ActionType.REFER
    investigation_terms = (
        "x-ray",
        "ct scan",
        "ultrasound",
        "endoscopy",
        "test",
        "testing",
        "digital rectal examination",
    )
    if body.startswith(("offer ", "consider ")) and any(
        term in body for term in investigation_terms
    ):
        return ActionType.OFFER_INVESTIGATION
    if "suspected cancer pathway referral" in body:
        return (
            ActionType.CONSIDER_REFERRAL
            if body.startswith("consider")
            else ActionType.REFER
        )
    return ActionType.CLINICAL_ASSESSMENT


def extract_age_condition(record: RawRecommendation) -> str | None:
    """Extract every explicit age branch while preserving source phrasing."""

    matches = [match.group(0) for match in AGE_PATTERN.finditer(record.text)]
    unique = list(dict.fromkeys(match.lower() for match in matches))
    if unique:
        return "; ".join(unique)
    if re.search(r"\badults\b", record.text, re.IGNORECASE):
        return "adults"
    return None


def extract_lab_threshold(record: RawRecommendation) -> str | None:
    """Extract numeric and named laboratory thresholds used by retrieval filters."""

    values: list[str] = []
    for pattern in NUMERIC_THRESHOLD_PATTERNS:
        values.extend(match.group(0).strip() for match in pattern.finditer(record.text))
    named_findings = (
        "low haemoglobin levels",
        "raised platelet count",
        "raised white cell count",
        "anaemia even in the absence of iron deficiency",
    )
    lowered = record.text.casefold()
    values.extend(term for term in named_findings if term in lowered)
    unique = list(dict.fromkeys(values))
    return "; ".join(unique) if unique else None


def extract_symptom_condition(record: RawRecommendation) -> str:
    """Return the condition-bearing clause without inventing a summary."""

    body = _body_without_id_or_revision(record)
    marker_patterns = (
        r"\bif\s+they\s*:\s*",
        r"\bif\s+they\s+",
        r"\bif\s+their\s+",
        r"\bin\s+people\s+",
        r"\bin\s+adults\s+",
        r"\bwho\s+have\s+",
        r"\bwith\s+",
    )
    starts: list[int] = []
    for pattern in marker_patterns:
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            starts.append(match.start())
    condition = body[min(starts) :] if starts else body
    # Preserve the complete branching logic; only flatten cosmetic whitespace.
    condition = re.sub(r"\s+", " ", condition).strip(" .")
    return condition or body


def infer_rule_type(record: RawRecommendation, lab_threshold: str | None) -> RuleType:
    """Classify the logical shape used for retrieval diagnostics and filtering."""

    body = _body_without_id_or_revision(record).casefold()
    if record.recommendation_id in {"1.3.3", "1.3.4"}:
        return RuleType.PROCESS
    if lab_threshold or re.search(r"\b(at least|below|above the threshold)\b", body):
        return RuleType.THRESHOLD_BASED
    branch_markers = (
        "any of the following",
        "2 or more",
        "1 or more",
        "either:",
        "following:",
        " or ",
    )
    if "•" in body or any(marker in body for marker in branch_markers):
        return RuleType.MULTI_BRANCH
    return RuleType.SINGLE_CONDITION


def _specific_site(record: RawRecommendation) -> str:
    if not record.subsection_title:
        raise ChunkingError(f"Missing subsection title for {record.recommendation_id}")
    try:
        return SPECIFIC_CANCER_SITE[record.subsection_title]
    except KeyError as exc:
        raise ChunkingError(
            f"No cancer-site mapping for subsection {record.subsection_title!r}"
        ) from exc


def build_embedding_text(chunk: RecommendationChunk) -> str:
    """Create a self-contained passage optimised for semantic and lexical search."""

    metadata = chunk.metadata
    fields = [
        "NICE NG12 suspected cancer recognition and referral.",
        f"Cancer site: {metadata.cancer_site.replace('_', ' ')}.",
        f"Section: {metadata.section_id} {metadata.section_title}.",
    ]
    if metadata.subsection_title:
        fields.append(f"Subsection: {metadata.subsection_title}.")
    fields.extend(
        [
            f"Recommendation {metadata.recommendation_id}.",
            f"Action: {metadata.action_type.value.replace('_', ' ')}.",
        ]
    )
    if metadata.age_condition:
        fields.append(f"Age condition: {metadata.age_condition}.")
    if metadata.lab_threshold:
        fields.append(f"Laboratory threshold: {metadata.lab_threshold}.")
    fields.append(f"Recommendation text: {chunk.text}")
    return " ".join(fields)


def create_chunk(record: RawRecommendation, settings: Settings) -> RecommendationChunk:
    """Transform one source recommendation into one retrieval chunk."""

    revision_history, revision_year = extract_revision_history(record.text)
    action_type = extract_action_type(record)
    age_condition = extract_age_condition(record)
    lab_threshold = extract_lab_threshold(record)
    rule_type = infer_rule_type(record, lab_threshold)
    guideline_version = settings.section("project")["guideline_version"]
    chunk_id = f"ng12-{guideline_version}-{record.recommendation_id}"
    metadata = ChunkMetadata(
        schema_version=settings.section("metadata")["schema_version"],
        guideline_version=guideline_version,
        document_title=settings.section("project")["guideline_title"],
        source_file=record.source_file,
        source_sha256=record.source_sha256,
        cancer_site=_specific_site(record),
        recommendation_id=record.recommendation_id,
        page_number=record.page_number,
        page_numbers=record.page_numbers,
        section_id=record.section_id,
        section_title=record.section_title,
        subsection_title=record.subsection_title,
        action_type=action_type,
        age_condition=age_condition,
        symptom_condition=extract_symptom_condition(record),
        lab_threshold=lab_threshold,
        rule_type=rule_type,
        revision_year=revision_year,
        revision_history=revision_history,
    )
    placeholder = RecommendationChunk(
        chunk_id=chunk_id,
        text=record.text,
        embedding_text=record.text,
        metadata=metadata,
    )
    return placeholder.model_copy(
        update={"embedding_text": build_embedding_text(placeholder)}
    )


def create_chunks(
    records: Iterable[RawRecommendation], settings: Settings
) -> list[RecommendationChunk]:
    """Create and validate a deterministic recommendation-level corpus."""

    chunks = [create_chunk(record, settings) for record in records]
    identifiers = [chunk.metadata.recommendation_id for chunk in chunks]
    duplicates = [key for key, count in Counter(identifiers).items() if count > 1]
    expected_count = int(settings.section("scope")["expected_recommendation_count"])
    if len(chunks) != expected_count or duplicates:
        raise ChunkingError(
            f"Chunk completeness failed: count={len(chunks)}, "
            f"expected={expected_count}, duplicates={duplicates}"
        )
    warning_limit = int(
        settings.section("chunking")["max_recommendation_characters_warning"]
    )
    for chunk in chunks:
        if len(chunk.text) > warning_limit:
            LOGGER.warning(
                "Long recommendation chunk %s has %d characters",
                chunk.chunk_id,
                len(chunk.text),
            )
    return chunks


def create_negative_chunks(
    chunks: Iterable[RecommendationChunk], settings: Settings
) -> list[RecommendationChunk]:
    """Create labelled near-miss clauses for retrieval robustness testing."""

    if not settings.section("negative_chunks").get("enabled", True):
        return []
    by_recommendation = {
        chunk.metadata.recommendation_id: chunk for chunk in chunks
    }
    negatives: list[RecommendationChunk] = []
    for index, (recommendation_id, source, replacement, description) in enumerate(
        NEGATIVE_MUTATIONS, start=1
    ):
        original = by_recommendation.get(recommendation_id)
        if original is None:
            raise ChunkingError(f"Negative mutation source not found: {recommendation_id}")
        if source not in original.text:
            raise ChunkingError(
                f"Negative mutation text {source!r} absent from {recommendation_id}"
            )
        mutated_text = original.text.replace(source, replacement, 1)
        mutated_text = (
            "[SYNTHETIC NEGATIVE — NOT NICE GUIDANCE] " + mutated_text
        )
        metadata = original.metadata.model_copy(
            update={
                "age_condition": extract_age_condition(
                    RawRecommendation(
                        recommendation_id=original.metadata.recommendation_id,
                        text=mutated_text,
                        section_id=original.metadata.section_id,
                        section_title=original.metadata.section_title,
                        subsection_title=original.metadata.subsection_title,
                        page_number=original.metadata.page_number,
                        page_numbers=original.metadata.page_numbers,
                        source_file=original.metadata.source_file,
                        source_sha256=original.metadata.source_sha256,
                        guideline_version=original.metadata.guideline_version,
                    )
                ),
                "lab_threshold": (
                    original.metadata.lab_threshold.replace(source, replacement, 1)
                    if original.metadata.lab_threshold
                    and source in original.metadata.lab_threshold
                    else original.metadata.lab_threshold
                ),
                "symptom_condition": original.metadata.symptom_condition.replace(
                    source, replacement, 1
                ),
                "is_synthetic_negative": True,
                "synthetic_source_id": original.chunk_id,
                "synthetic_mutation": description,
            }
        )
        negative = RecommendationChunk(
            chunk_id=f"{original.chunk_id}-negative-{index:03d}",
            text=mutated_text,
            embedding_text=mutated_text,
            metadata=metadata,
        )
        negatives.append(
            negative.model_copy(update={"embedding_text": build_embedding_text(negative)})
        )
    return negatives


def write_chunks(chunks: Iterable[RecommendationChunk], output_path: Path) -> None:
    """Atomically write chunks in stable JSON Lines order."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json(exclude_none=True) + "\n")
    temporary.replace(output_path)


def read_chunks(path: Path) -> list[RecommendationChunk]:
    """Read strict recommendation chunks from JSON Lines."""

    chunks: list[RecommendationChunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(RecommendationChunk.model_validate_json(line))
            except Exception as exc:
                raise ChunkingError(f"Invalid chunk at {path}:{line_number}: {exc}") from exc
    return chunks


def write_corpus_manifest(
    chunks: list[RecommendationChunk],
    negatives: list[RecommendationChunk],
    settings: Settings,
) -> None:
    """Write a compact manifest for downstream index compatibility checks."""

    manifest_path = settings.path("source.corpus_manifest_path")
    manifest = {
        "schema_version": settings.section("metadata")["schema_version"],
        "guideline_id": settings.section("project")["guideline_id"],
        "guideline_version": settings.section("project")["guideline_version"],
        "source_sha256": chunks[0].metadata.source_sha256,
        "chunking_strategy": settings.section("chunking")["strategy"],
        "recommendation_count": len(chunks),
        "negative_chunk_count": len(negatives),
        "scope_sections": list(settings.section("scope")["include_sections"].keys()),
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "counts_by_cancer_site": dict(
            sorted(Counter(chunk.metadata.cancer_site for chunk in chunks).items())
        ),
        "counts_by_action_type": dict(
            sorted(Counter(chunk.metadata.action_type.value for chunk in chunks).items())
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_chunking(
    settings: Settings,
    records: list[RawRecommendation] | None = None,
) -> tuple[list[RecommendationChunk], list[RecommendationChunk]]:
    """Create, persist, and audit real and synthetic-negative chunks."""

    source_records = records or read_raw_recommendations(
        settings.path("source.raw_recommendations_path")
    )
    chunks = create_chunks(source_records, settings)
    negatives = create_negative_chunks(chunks, settings)
    write_chunks(chunks, settings.path("source.chunks_path"))
    write_chunks(negatives, settings.path("negative_chunks.output_path"))
    write_corpus_manifest(chunks, negatives, settings)
    LOGGER.info(
        "Created %d real chunks and %d synthetic negative chunks",
        len(chunks),
        len(negatives),
    )
    return chunks, negatives
