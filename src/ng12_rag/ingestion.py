"""Page-aware ingestion for the pinned April 2026 NICE NG12 PDF.

The parser deliberately avoids fixed-size text splitting. It locates the canonical
section headings in the recommendation body, then extracts each numbered
recommendation as a source record while retaining every physical PDF page touched by
that recommendation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from ng12_rag.config import Settings
from ng12_rag.models import RawRecommendation

LOGGER = logging.getLogger(__name__)

TOP_LEVEL_SECTION_PATTERN = re.compile(
    r"(?m)^\s*(1\.\d{1,2})\s+[A-Z][^\n]*?\s*$"
)
HEADER_PATTERN = re.compile(
    r"^\s*Suspected cancer:\s*recognition and referral\s*\(NG12\)\s*$",
    re.IGNORECASE,
)
FOOTER_START_PATTERN = re.compile(
    r"^\s*(?:©\s*NICE\s+\d{4}\.|Subject to Notice of rights)",
    re.IGNORECASE,
)
# The footer wraps across a variable number of lines depending on where the PDF broke it:
#
#   © NICE 2026. All rights reserved. Subject to Notice of rights (https://...terms-and-
#   conditions#notice-of-rights).
#   Page 14 of
#   101
#
# On some pages the URL tail and "Page N of" share a line, on others they do not. Matching
# the block line by line handles both, where a single combined pattern did not and left
# "conditions#notice-of-rights). 101" embedded mid-recommendation.
FOOTER_TAIL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # The URL tail may be trailed by "Page N of" and/or the bare page total on the same
    # line, separated by column padding the extractor preserves as runs of spaces.
    re.compile(
        r"^conditions#notice-of-rights\)\.?(?:\s+Page\s+\d+\s+of)?(?:\s+\d{1,3})?\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^Page\s+\d+\s+of(?:\s+\d{1,3})?\s*$", re.IGNORECASE),
    re.compile(r"^\d{1,3}$"),
    re.compile(r"^All rights reserved\.?\s*$", re.IGNORECASE),
)

SUBSECTION_RANGES: dict[str, tuple[tuple[range, str], ...]] = {
    "1.1": (
        (range(1, 4), "Lung cancer"),
        (range(4, 7), "Mesothelioma"),
    ),
    "1.2": (
        (range(1, 4), "Oesophageal cancer"),
        (range(4, 6), "Pancreatic cancer"),
        (range(6, 10), "Stomach cancer"),
        (range(10, 11), "Gall bladder cancer"),
        (range(11, 12), "Liver cancer"),
    ),
    "1.3": (
        (range(1, 6), "Colorectal cancer"),
        (range(6, 7), "Anal cancer"),
    ),
    "1.6": (
        (range(1, 4), "Prostate cancer"),
        (range(4, 6), "Bladder cancer"),
        (range(6, 7), "Renal cancer"),
        (range(7, 9), "Testicular cancer"),
        (range(9, 11), "Penile cancer"),
    ),
}


class SourceValidationError(RuntimeError):
    """Raised when the input PDF does not match the pinned NG12 source."""


class IngestionError(RuntimeError):
    """Raised when the recommendation corpus cannot be extracted completely."""


@dataclass(frozen=True)
class PageSpan:
    """Character offsets for one page inside the assembled document text."""

    page_number: int
    start: int
    end: int


@dataclass(frozen=True)
class SourceValidationResult:
    """Auditable output of source validation."""

    sha256: str
    page_count: int
    title: str
    author: str
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def validate_source(pdf_path: Path, settings: Settings) -> SourceValidationResult:
    """Validate checksum, metadata, page count, and visible version marker."""

    source = settings.section("source")
    if not pdf_path.is_file():
        raise SourceValidationError(f"Source PDF does not exist: {pdf_path}")

    sha256 = file_sha256(pdf_path)
    reader = PdfReader(str(pdf_path))
    metadata = reader.metadata or {}
    title = str(metadata.get("/Title", "")).strip()
    author = str(metadata.get("/Author", "")).strip()
    first_page_text = reader.pages[0].extract_text() or ""

    errors: list[str] = []
    if sha256 != source["expected_sha256"]:
        errors.append(
            f"SHA-256 mismatch: expected {source['expected_sha256']}, got {sha256}"
        )
    if len(reader.pages) != int(source["expected_page_count"]):
        errors.append(
            "Page-count mismatch: expected "
            f"{source['expected_page_count']}, got {len(reader.pages)}"
        )
    if source["expected_title"].casefold() not in title.casefold():
        errors.append(f"Unexpected PDF title metadata: {title!r}")
    if source["expected_author"].casefold() not in author.casefold():
        errors.append(f"Unexpected PDF author metadata: {author!r}")
    if source["expected_last_updated_text"].casefold() not in first_page_text.casefold():
        errors.append("The expected 15 April 2026 update marker is absent from page 1")

    result = SourceValidationResult(
        sha256=sha256,
        page_count=len(reader.pages),
        title=title,
        author=author,
        errors=tuple(errors),
    )
    if errors and bool(source.get("strict_validation", True)):
        raise SourceValidationError("; ".join(errors))
    for error in errors:
        LOGGER.warning("Source validation warning: %s", error)
    return result


def _extract_page_text(page: object) -> str:
    """Extract page text in layout mode with a compatibility fallback."""

    try:
        text = page.extract_text(extraction_mode="layout")  # type: ignore[attr-defined]
    except TypeError:
        text = page.extract_text()  # type: ignore[attr-defined]
    return text or ""


def clean_page_text(text: str) -> str:
    """Remove repeated publication furniture while preserving clinical wording."""

    lines = text.replace("\u00ad", "").replace("\r\n", "\n").splitlines()
    cleaned: list[str] = []
    in_footer = False
    for line in lines:
        stripped = line.strip()
        if HEADER_PATTERN.match(stripped):
            in_footer = False
            continue
        if FOOTER_START_PATTERN.match(stripped):
            in_footer = True
            continue
        if in_footer:
            if not stripped or any(p.match(stripped) for p in FOOTER_TAIL_PATTERNS):
                continue
            # A line that is not part of the footer block means real content resumed.
            in_footer = False
        cleaned.append(line.rstrip())
    return "\n".join(cleaned).strip()


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract clean text for every physical PDF page using one-based numbering."""

    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        pages.append((page_number, clean_page_text(_extract_page_text(page))))
    if not pages:
        raise IngestionError("The PDF contained no pages")
    return pages


def assemble_document(pages: Iterable[tuple[int, str]]) -> tuple[str, list[PageSpan]]:
    """Assemble page text while recording exact character offsets per page."""

    pieces: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0
    for page_number, text in pages:
        separator = "\n\n" if pieces else ""
        pieces.append(separator)
        cursor += len(separator)
        start = cursor
        pieces.append(text)
        cursor += len(text)
        spans.append(PageSpan(page_number=page_number, start=start, end=cursor))
    return "".join(pieces), spans


def _normalise_recommendation_text(text: str) -> str:
    """Flatten layout whitespace without changing clinical terms or thresholds."""

    normalised = text.replace("\u2010", "-").replace("\u2011", "-")
    normalised = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "-", normalised)
    normalised = re.sub(r"\s+", " ", normalised).strip()
    normalised = re.sub(r"\s+([,.;:])", r"\1", normalised)
    return normalised


def expected_recommendation_ids(settings: Settings) -> list[str]:
    """Expand configured section ranges into the exact ordered ID allowlist."""

    identifiers: list[str] = []
    for item in settings.section("scope")["required_id_ranges"]:
        section = str(item["section"])
        identifiers.extend(
            f"{section}.{number}"
            for number in range(int(item["start"]), int(item["end"]) + 1)
        )
    return identifiers


def _section_span(document: str, section_id: str, title: str) -> tuple[int, int]:
    """Locate a body section and the beginning of the next top-level section."""

    heading = re.compile(
        rf"(?m)^\s*{re.escape(section_id)}\s+{re.escape(title)}\s*$",
        re.IGNORECASE,
    )
    candidates = list(heading.finditer(document))
    if not candidates:
        raise IngestionError(f"Could not locate section heading {section_id} {title!r}")

    # A contents entry contains dot leaders and therefore does not match this strict line.
    start_match = candidates[0]
    next_section = TOP_LEVEL_SECTION_PATTERN.search(document, start_match.end())
    end = next_section.start() if next_section else len(document)
    return start_match.end(), end


def _pages_for_span(
    document: str, spans: list[PageSpan], start: int, end: int
) -> list[int]:
    """Return every physical page containing non-whitespace text in a span."""

    pages: list[int] = []
    for span in spans:
        overlap_start = max(start, span.start)
        overlap_end = min(end, span.end)
        if overlap_start < overlap_end and document[overlap_start:overlap_end].strip():
            pages.append(span.page_number)
    if not pages:
        raise IngestionError(f"No page provenance found for character span {start}:{end}")
    return sorted(set(pages))


def _subsection_for(recommendation_id: str) -> str | None:
    section_id = ".".join(recommendation_id.split(".")[:2])
    number = int(recommendation_id.rsplit(".", 1)[1])
    for number_range, title in SUBSECTION_RANGES.get(section_id, ()):
        if number in number_range:
            return title
    return None


def extract_recommendations(
    pdf_path: Path, settings: Settings
) -> tuple[list[RawRecommendation], SourceValidationResult]:
    """Extract all and only the configured recommendation IDs from the PDF."""

    validation = validate_source(pdf_path, settings)
    pages = extract_pages(pdf_path)
    document, page_spans = assemble_document(pages)
    scope = settings.section("scope")
    project = settings.section("project")

    records: list[RawRecommendation] = []
    for section_id, section_config in scope["include_sections"].items():
        section_title = str(section_config["section_title"])
        section_start, section_end = _section_span(
            document=document,
            section_id=str(section_id),
            title=section_title,
        )
        section_text = document[section_start:section_end]
        recommendation_pattern = re.compile(
            rf"(?m)^\s*({re.escape(str(section_id))}\.\d+)\s+"
        )
        matches = list(recommendation_pattern.finditer(section_text))
        for index, match in enumerate(matches):
            recommendation_start = section_start + match.start()
            recommendation_end = (
                section_start + matches[index + 1].start()
                if index + 1 < len(matches)
                else section_end
            )
            raw_text = document[recommendation_start:recommendation_end]
            recommendation_id = match.group(1)
            subsection_title = _subsection_for(recommendation_id)
            for subsection_ranges in SUBSECTION_RANGES.values():
                for _, trailing_title in subsection_ranges:
                    raw_text = re.sub(
                        rf"\n\s*{re.escape(trailing_title)}\s*$",
                        "",
                        raw_text,
                        count=1,
                        flags=re.IGNORECASE,
                    )
            page_numbers = _pages_for_span(
                document, page_spans, recommendation_start, recommendation_end
            )
            records.append(
                RawRecommendation(
                    recommendation_id=recommendation_id,
                    text=_normalise_recommendation_text(raw_text),
                    section_id=str(section_id),
                    section_title=section_title,
                    subsection_title=subsection_title,
                    page_number=page_numbers[0],
                    page_numbers=page_numbers,
                    source_file=pdf_path.name,
                    source_sha256=validation.sha256,
                    guideline_version=str(project["guideline_version"]),
                )
            )

    expected = expected_recommendation_ids(settings)
    extracted = [record.recommendation_id for record in records]
    missing = sorted(set(expected) - set(extracted))
    unexpected = sorted(set(extracted) - set(expected))
    duplicates = sorted(
        identifier for identifier, count in Counter(extracted).items() if count > 1
    )
    if missing or unexpected or duplicates or len(extracted) != len(expected):
        raise IngestionError(
            "Recommendation completeness check failed: "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}, "
            f"expected_count={len(expected)}, extracted_count={len(extracted)}"
        )

    order = {identifier: index for index, identifier in enumerate(expected)}
    records.sort(key=lambda record: order[record.recommendation_id])
    return records, validation


def write_raw_recommendations(
    records: list[RawRecommendation], output_path: Path
) -> None:
    """Write validated source records in deterministic JSON Lines format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json(exclude_none=True) + "\n")
    temporary.replace(output_path)


def write_ingestion_manifest(
    *,
    records: list[RawRecommendation],
    validation: SourceValidationResult,
    output_path: Path,
    settings: Settings,
) -> None:
    """Write a machine-readable provenance and completeness manifest."""

    counts_by_section = Counter(record.section_id for record in records)
    manifest = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "guideline_id": settings.section("project")["guideline_id"],
        "guideline_version": settings.section("project")["guideline_version"],
        "source": {
            "file": settings.path("source.pdf_path").name,
            "sha256": validation.sha256,
            "page_count": validation.page_count,
            "title": validation.title,
            "author": validation.author,
            "validation_passed": validation.valid,
            "validation_errors": list(validation.errors),
        },
        "scope": list(settings.section("scope")["include_sections"].keys()),
        "recommendation_count": len(records),
        "recommendation_ids": [record.recommendation_id for record in records],
        "counts_by_section": dict(sorted(counts_by_section.items())),
        "page_range": [
            min(record.page_number for record in records),
            max(max(record.page_numbers) for record in records),
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def run_ingestion(settings: Settings) -> list[RawRecommendation]:
    """Validate, extract, audit, and persist the scoped raw corpus."""

    pdf_path = settings.path("source.pdf_path")
    records, validation = extract_recommendations(pdf_path, settings)
    write_raw_recommendations(
        records, settings.path("source.raw_recommendations_path")
    )
    manifest_path = settings.path("source.corpus_manifest_path").with_name(
        "ingestion_manifest.json"
    )
    write_ingestion_manifest(
        records=records,
        validation=validation,
        output_path=manifest_path,
        settings=settings,
    )
    LOGGER.info(
        "Extracted %d recommendations from %s (%s)",
        len(records),
        pdf_path,
        validation.sha256,
    )
    return records
