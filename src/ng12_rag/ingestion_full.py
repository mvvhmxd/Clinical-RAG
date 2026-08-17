"""Ingest the scoped chapters from the full 382-page NICE NG12 evidence guideline.

The module is intentionally additive: it does not change the existing short-guideline
pipeline. It extracts the four requested physical PDF page ranges with PyMuPDF, keeps
chapter/subsection provenance, packs text at paragraph and heading boundaries, and
writes JSON Lines using the existing chunk envelope: ``chunk_id``, ``text``,
``embedding_text``, and ``metadata``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

MIN_CHUNK_TOKENS = 400
TARGET_CHUNK_TOKENS = 650
MAX_CHUNK_TOKENS = 800
SOURCE_DOCUMENT = "NG12_full"
CONTENT_TYPE = "evidence_review"


@dataclass(frozen=True)
class ChapterSpec:
    number: str
    title: str
    first_page: int
    last_page: int
    cancer_site: str


TARGET_CHAPTERS: tuple[ChapterSpec, ...] = (
    ChapterSpec("7", "Lung and pleural cancers", 37, 53, "lung"),
    ChapterSpec("8", "Upper gastro-intestinal tract cancers", 54, 102, "upper_gastrointestinal"),
    ChapterSpec("9", "Lower gastrointestinal tract cancers", 103, 150, "colorectal"),
    ChapterSpec("12", "Urological cancers", 176, 214, "urological"),
)

_NUMBERED_HEADING = re.compile(
    r"^(?P<number>(?:7|8|9|12)\.\d+(?:\.\d+)*)\s+(?P<title>[^\n]+)$",
    re.IGNORECASE,
)

# The full guideline's cancer-site subsections, which correspond one-to-one with the short
# guideline's subsections. This is the link between the two tiers: a chapter number alone is
# too coarse, because chapter 8 spans five distinct sites and chapter 12 spans five more.
FULL_SUBSECTION_SITES: dict[str, str] = {
    "7.1": "lung",
    "7.2": "mesothelioma",
    "8.1": "oesophageal",
    "8.2": "pancreatic",
    "8.3": "stomach",
    "8.4": "small_intestinal",
    "8.5": "gall_bladder",
    "8.6": "liver",
    "9.1": "colorectal",
    "9.2": "anal",
    "12.1": "prostate",
    "12.2": "bladder",
    "12.3": "renal",
    "12.4": "testicular",
    "12.5": "penile",
}

# GRADE and diagnostic-accuracy tables contain rows such as "7.5 (6.6-8.5) 220/2930" and
# "9.03 (6.82-11.7) 52/576", which match the numbered-heading shape exactly. Treating them as
# subsection headings silently reassigns every following chunk to a fabricated subsection, so
# a heading must both carry a known subsection number and read like a title.
_HEADING_TITLE = re.compile(r"^[A-Za-z][A-Za-z \-/&,'()]{2,70}$")


def _valid_subsection_heading(number: str, title: str) -> bool:
    """Return True only for a real cancer-site subsection heading."""

    if number not in FULL_SUBSECTION_SITES:
        return False
    return bool(_HEADING_TITLE.fullmatch(title.strip()))
_RECOMMENDATION_ID = re.compile(r"\b1\.\d+\.\d+\b")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PAGE_FURNITURE = (
    re.compile(r"^Suspected cancer$", re.IGNORECASE),
    re.compile(r"^©\s*National Collaborating Centre for Cancer$", re.IGNORECASE),
    re.compile(r"^Update 2015$", re.IGNORECASE),
    re.compile(r"^\d{1,3}$"),
)
_STRUCTURAL_HEADINGS = {
    "clinical questions",
    "clinical evidence",
    "evidence statements",
    "recommendations",
    "recommendation",
    "signs and symptoms",
    "investigations",
    "investigation",
    "risk of bias in the included studies",
    "economic evidence",
    "health economic evidence",
    "evidence review",
}


@dataclass(frozen=True)
class TextUnit:
    text: str
    page_number: int
    subsection_title: str
    section_title: str
    is_heading: bool = False
    cancer_site: str | None = None


@dataclass
class DraftChunk:
    units: list[TextUnit]

    @property
    def token_count(self) -> int:
        return sum(_token_count(unit.text) for unit in self.units)


def _token_count(text: str) -> int:
    """Return a deterministic whitespace-token approximation."""

    return len(re.findall(r"\S+", text))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_block(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u00a0", " ").replace("\r", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    retained: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or any(pattern.fullmatch(line) for pattern in _PAGE_FURNITURE):
            continue
        retained.append(line)
    return "\n".join(retained).strip()


def _is_structural_heading(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text).strip().rstrip(":")
    lowered = compact.casefold()
    if lowered in _STRUCTURAL_HEADINGS:
        return True
    if len(compact) > 110 or "\n" in text:
        return False
    return bool(
        re.match(
            r"^(?:risk of bias|evidence|clinical|diagnostic|investigation|recommendation|"
            r"review question|included studies|excluded studies|summary of evidence)\b",
            compact,
            re.IGNORECASE,
        )
    )


def _split_piece(text: str, maximum: int = 180) -> list[str]:
    """Split an extracted block into small paragraph/sentence units for safe packing."""

    text = text.strip()
    if not text:
        return []
    if _token_count(text) <= maximum:
        return [text]

    candidates = [part.strip() for part in text.split("\n") if part.strip()]
    if len(candidates) == 1:
        candidates = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]

    output: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for candidate in candidates:
        words = candidate.split()
        if len(words) > maximum:
            if current:
                output.append(" ".join(current))
                current = []
                current_tokens = 0
            output.extend(
                " ".join(words[start : start + maximum])
                for start in range(0, len(words), maximum)
            )
            continue
        if current and current_tokens + len(words) > maximum:
            output.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(candidate)
        current_tokens += len(words)
    if current:
        output.append(" ".join(current))
    return output


def _page_blocks(page: Any) -> Iterator[str]:
    for block in page.get_text("blocks", sort=True):
        cleaned = _clean_block(str(block[4]))
        if cleaned:
            yield cleaned


def _extract_chapter_units(document: Any, spec: ChapterSpec) -> list[TextUnit]:
    units: list[TextUnit] = []
    subsection_title = f"{spec.number} {spec.title}"
    section_title = subsection_title
    cancer_site: str | None = None

    for page_number in range(spec.first_page, spec.last_page + 1):
        page = document[page_number - 1]
        for block in _page_blocks(page):
            compact = re.sub(r"\s+", " ", block).strip()
            numbered = _NUMBERED_HEADING.fullmatch(compact)
            heading = False
            if numbered and _valid_subsection_heading(
                numbered.group("number"), numbered.group("title")
            ):
                subsection_title = f"{numbered.group('number')} {numbered.group('title').strip()}"
                section_title = subsection_title
                cancer_site = FULL_SUBSECTION_SITES[numbered.group("number")]
                heading = True
            elif _is_structural_heading(block):
                section_title = compact.rstrip(":")
                heading = True

            pieces = [compact] if heading else _split_piece(block)
            for piece_index, piece in enumerate(pieces):
                units.append(
                    TextUnit(
                        text=piece,
                        page_number=page_number,
                        subsection_title=subsection_title,
                        section_title=section_title,
                        is_heading=heading and piece_index == 0,
                        cancer_site=cancer_site,
                    )
                )
    return units


def _rebalance_tail(chunks: list[DraftChunk]) -> list[DraftChunk]:
    if len(chunks) < 2 or chunks[-1].token_count >= MIN_CHUNK_TOKENS:
        return chunks

    previous = chunks[-2]
    tail = chunks[-1]
    if previous.token_count + tail.token_count <= MAX_CHUNK_TOKENS:
        previous.units.extend(tail.units)
        chunks.pop()
        return chunks

    while tail.token_count < MIN_CHUNK_TOKENS and len(previous.units) > 1:
        candidate = previous.units[-1]
        if previous.token_count - _token_count(candidate.text) < MIN_CHUNK_TOKENS:
            break
        previous.units.pop()
        tail.units.insert(0, candidate)
    return chunks


def _pack_units(units: Sequence[TextUnit]) -> list[DraftChunk]:
    chunks: list[DraftChunk] = []
    current: list[TextUnit] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _token_count(unit.text)
        if unit.is_heading and current_tokens >= MIN_CHUNK_TOKENS:
            chunks.append(DraftChunk(current))
            current = []
            current_tokens = 0
        if current and current_tokens + unit_tokens > MAX_CHUNK_TOKENS:
            chunks.append(DraftChunk(current))
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
        if current_tokens >= TARGET_CHUNK_TOKENS:
            chunks.append(DraftChunk(current))
            current = []
            current_tokens = 0

    if current:
        chunks.append(DraftChunk(current))
    return _rebalance_tail(chunks)


def _chunk_text(units: Iterable[TextUnit]) -> str:
    return "\n\n".join(unit.text for unit in units).strip()


def _build_chunk(
    draft: DraftChunk,
    spec: ChapterSpec,
    ordinal: int,
    source_file: str,
    source_sha256: str,
) -> dict[str, Any]:
    text = _chunk_text(draft.units)
    page_numbers = sorted({unit.page_number for unit in draft.units})
    headings = [unit.section_title for unit in draft.units if unit.is_heading]
    section_title = headings[-1] if headings else draft.units[0].section_title
    subsection_title = draft.units[0].subsection_title
    linked_ids = sorted(set(_RECOMMENDATION_ID.findall(text)))
    # Prefer the subsection-level site over the chapter-level fallback: chapter 8 alone spans
    # oesophageal, pancreatic, stomach, gall bladder and liver.
    sites = [unit.cancer_site for unit in draft.units if unit.cancer_site]
    cancer_site = sites[0] if sites else spec.cancer_site
    chunk_id = f"ng12-full-ch{int(spec.number):02d}-{ordinal:04d}"
    embedding_text = (
        "NICE NG12 full evidence guideline. "
        f"Cancer site: {cancer_site.replace('_', ' ')}. "
        f"Chapter {spec.number}: {spec.title}. "
        f"Subsection: {subsection_title}. Section: {section_title}. "
        f"Evidence: {text}"
    )
    return {
        "chunk_id": chunk_id,
        "text": text,
        "embedding_text": embedding_text,
        "metadata": {
            "source_document": SOURCE_DOCUMENT,
            "document_type": SOURCE_DOCUMENT,
            "source_file": source_file,
            "source_sha256": source_sha256,
            "cancer_site": cancer_site,
            "chapter": spec.number,
            "chapter_number": spec.number,
            "chapter_title": spec.title,
            "subsection_title": subsection_title,
            "section_title": section_title,
            "page_number": page_numbers[0],
            "page_numbers": page_numbers,
            "content_type": CONTENT_TYPE,
            "recommendation_id": None,
            "linked_recommendation_ids": linked_ids,
            "token_count": _token_count(text),
            "is_synthetic_negative": False,
        },
    }


def extract_full_guideline_chunks(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract and chunk only Chapters 7, 8, 9, and 12 from the full guideline."""

    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for full-guideline ingestion; install the project requirements"
        ) from exc

    source = Path(pdf_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Full guideline PDF not found: {source}")

    source_sha256 = _file_sha256(source)
    chunks: list[dict[str, Any]] = []
    with fitz.open(source) as document:
        if document.page_count < max(spec.last_page for spec in TARGET_CHAPTERS):
            raise ValueError(
                f"Expected at least 214 pages in the full guideline, found {document.page_count}"
            )
        ordinal = 1
        for spec in TARGET_CHAPTERS:
            chapter_units = _extract_chapter_units(document, spec)
            for _, subsection_group in groupby(
                chapter_units, key=lambda unit: unit.subsection_title
            ):
                for draft in _pack_units(list(subsection_group)):
                    chunks.append(
                        _build_chunk(
                            draft,
                            spec,
                            ordinal,
                            source.name,
                            source_sha256,
                        )
                    )
                    ordinal += 1
    return chunks


def write_full_chunks(chunks: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    """Atomically write full-guideline chunks as JSON Lines."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)
    return destination


def run_full_ingestion(
    pdf_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run the standalone full-guideline ingestion pipeline."""

    project_root = Path(__file__).resolve().parents[2]
    source = Path(pdf_path) if pdf_path else project_root / "data" / "raw" / "ng12_full.pdf"
    destination = (
        Path(output_path)
        if output_path
        else project_root / "data" / "processed" / "full_chunks.jsonl"
    )
    chunks = extract_full_guideline_chunks(source)
    write_full_chunks(chunks, destination)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        default=None,
        help="Path to the 382-page full NG12 evidence guideline PDF",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination JSONL path (default: data/processed/full_chunks.jsonl)",
    )
    args = parser.parse_args()
    chunks = run_full_ingestion(args.pdf, args.output)
    print(f"Wrote {len(chunks)} NG12_full chunks")


if __name__ == "__main__":
    main()
