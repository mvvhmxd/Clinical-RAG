"""Turn raw full-guideline extraction into a spec-compliant Tier 2 corpus.

The full evidence guideline answers a different question from the short guideline: not "what
are the referral criteria" but "why is this criterion set here". To be usable for that, every
retained chunk must point back at the specific Tier 1 recommendation it discusses, so a
rationale answer can still name the criterion it supports.

The 2015 full guideline never cites the short guideline's ``1.x.y`` numbering -- only 1 of 103
chunks mentions such an identifier -- so links cannot be read off the text. They are instead
derived from the two documents' shared cancer-site structure, then narrowed within a site by
weighted term overlap against each candidate recommendation.

Chunks that cannot be linked confidently are excluded rather than guessed at. A rationale
chunk attached to the wrong criterion is worse than one that is absent.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONTENT_TYPE = "rationale_evidence"
SOURCE_TIER = "secondary_full_guideline"

# Mirrors docs/scope.md. A site excluded from Tier 1 is excluded from Tier 2 identically.
LOCKED_CANCER_SITES: frozenset[str] = frozenset(
    {"lung", "oesophageal", "pancreatic", "stomach", "colorectal", "renal", "bladder"}
)

# A link is accepted only when the best candidate clears this score and is meaningfully
# ahead of the runner-up. Both floors matter: the first rejects chunks with no clinical
# signal (methodology, committee process), the second rejects chunks that discuss a site
# generally without being about one particular recommendation.
MIN_LINK_SCORE = 0.08
MIN_LINK_MARGIN = 1.15
MIN_CLINICAL_TOKENS = 40

_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "had", "has", "have", "if", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "were", "will", "with", "which", "who", "whom", "would", "people", "person", "adults", "patients", "using", "pathway", "suspected", "cancer", "refer", "referral", "offer", "consider", "assess", "assessment", "guideline", "recommendation", "recommendations", "evidence", "review", "study", "studies", "included", "patient", "primary", "care", "nice", "full", "short", "table", "figure", "appendix", "chapter", "section"]
)

_WORD = re.compile(r"[a-z][a-z\-]{2,}")
_NUMERIC = re.compile(r"\b\d+(?:\.\d+)?\b")

# Extraction noise specific to the evidence guideline: diagnostic-accuracy rows, confidence
# intervals, and reference markers that survive block extraction as standalone fragments.
_NOISE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Confidence-interval ranges in this document use en and em dashes as well as hyphens.
    re.compile(
        r"^\s*\d+(?:\.\d+)?\s*\(\s*\d+(?:\.\d+)?\s*[-\u2013\u2014]\s*\d+(?:\.\d+)?\s*\)\s*[\d/]*\s*$"
    ),
    re.compile(r"^\s*(?:TP|FP|FN|TN)\s*=.*$", re.IGNORECASE),
    re.compile(r"^\s*[\d\s./%()\u2013\u2014-]+\s*$"),
    re.compile(r"^\s*(?:Table|Figure)\s+\d+[\d.]*\s*:?\s*$", re.IGNORECASE),
    # Meta-analysis rows flatten into prose-looking text with a dangling sample size and a
    # run of author-year citations, e.g. "Haematuria All patients (N = Collins (2013), ...".
    re.compile(r"^.{0,60}\(\s*N\s*=\s*(?:\d[\d,\s]*)?(?:[A-Z][A-Za-z\-]+\s*\(\d{4}[^)]*\)[,;\s]*){2,}"),
)


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


def clean_tier2_text(text: str) -> str:
    """Remove statistics-table and reference noise without touching rationale prose."""

    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(p.match(stripped) for p in _NOISE_LINE_PATTERNS):
            continue
        kept.append(stripped)
    joined = "\n".join(kept)
    # Collapse the runs of spaces the two-column layout leaves behind mid-sentence.
    joined = re.sub(r"[ \t]{2,}", " ", joined)
    return joined.strip()


def _site_index(tier1: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    index: dict[str, list[Mapping[str, Any]]] = {}
    for record in tier1:
        site = record["metadata"]["cancer_site"]
        index.setdefault(site, []).append(record)
    return index


def _idf_weights(candidates: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Weight terms by how well they distinguish one recommendation from its site peers."""

    total = len(candidates)
    document_frequency: Counter[str] = Counter()
    for record in candidates:
        for term in set(_tokens(record["text"])):
            document_frequency[term] += 1
    return {
        term: math.log((total + 1) / (count + 0.5)) + 0.1
        for term, count in document_frequency.items()
    }


def _score(chunk_terms: Counter[str], record: Mapping[str, Any], weights: dict[str, float]) -> float:
    record_terms = set(_tokens(record["text"]))
    if not record_terms:
        return 0.0
    overlap = sum(weights.get(term, 0.1) for term in record_terms if term in chunk_terms)
    total = sum(weights.get(term, 0.1) for term in record_terms)
    return overlap / total if total else 0.0


def link_chunk(
    chunk: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    weights: dict[str, float],
) -> tuple[str | None, str]:
    """Return the best Tier 1 recommendation id for a chunk, plus the reason for the outcome."""

    if not candidates:
        return None, "no_tier1_recommendations_for_site"

    chunk_terms = Counter(_tokens(chunk["text"]))
    if sum(chunk_terms.values()) < MIN_CLINICAL_TOKENS:
        return None, "insufficient_clinical_content"

    scored = sorted(
        ((_score(chunk_terms, record, weights), record) for record in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    best_score, best = scored[0]
    if best_score < MIN_LINK_SCORE:
        return None, f"below_score_floor({best_score:.3f})"

    if len(scored) > 1:
        runner_up = scored[1][0]
        if runner_up > 0 and best_score / runner_up < MIN_LINK_MARGIN:
            return None, f"ambiguous_between_recommendations({best_score:.3f}/{runner_up:.3f})"

    return str(best["metadata"]["recommendation_id"]), "linked"


def build_tier2_corpus(
    raw_chunks: Iterable[Mapping[str, Any]],
    tier1_chunks: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (active Tier 2 chunks, excluded chunks with a recorded reason)."""

    by_site = _site_index(tier1_chunks)
    weights_by_site = {site: _idf_weights(records) for site, records in by_site.items()}

    active: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for raw in raw_chunks:
        record = json.loads(json.dumps(raw))  # defensive copy
        metadata = record["metadata"]
        site = metadata.get("cancer_site")

        cleaned = clean_tier2_text(record["text"])
        if not cleaned:
            metadata["exclusion_reason"] = "empty_after_cleaning"
            excluded.append(record)
            continue
        record["text"] = cleaned

        if site not in LOCKED_CANCER_SITES:
            metadata["exclusion_reason"] = f"out_of_locked_scope({site})"
            excluded.append(record)
            continue

        linked_id, reason = link_chunk(record, by_site.get(site, []), weights_by_site.get(site, {}))
        if linked_id is None:
            metadata["exclusion_reason"] = reason
            excluded.append(record)
            continue

        metadata["content_type"] = CONTENT_TYPE
        metadata["source_tier"] = SOURCE_TIER
        metadata["linked_recommendation_id"] = linked_id
        metadata.pop("linked_recommendation_ids", None)
        record["embedding_text"] = (
            "NICE NG12 full evidence guideline, supporting rationale for recommendation "
            f"{linked_id}. Cancer site: {site.replace('_', ' ')}. "
            f"Subsection: {metadata.get('subsection_title')}. "
            f"Evidence: {cleaned}"
        )
        active.append(record)

    return active, excluded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_tier2_build(
    *,
    raw_path: Path = Path("data/processed/full_chunks.jsonl"),
    tier1_path: Path = Path("data/processed/chunks.jsonl"),
    active_path: Path = Path("data/processed/full_evidence_chunks.jsonl"),
    excluded_path: Path = Path("data/excluded/tier2_excluded.jsonl"),
    manifest_path: Path = Path("data/processed/tier2_manifest.json"),
) -> dict[str, Any]:
    """Build the retrievable Tier 2 corpus and its excluded counterpart."""

    raw = _read_jsonl(raw_path)
    tier1 = _read_jsonl(tier1_path)
    active, excluded = build_tier2_corpus(raw, tier1)

    _write_jsonl(active, active_path)
    _write_jsonl(excluded, excluded_path)

    reasons = Counter(
        str(record["metadata"]["exclusion_reason"]).split("(")[0] for record in excluded
    )
    links = Counter(str(record["metadata"]["linked_recommendation_id"]) for record in active)
    manifest: dict[str, Any] = {
        "content_type": CONTENT_TYPE,
        "source_tier": SOURCE_TIER,
        "raw_chunk_count": len(raw),
        "active_chunk_count": len(active),
        "excluded_chunk_count": len(excluded),
        "locked_cancer_sites": sorted(LOCKED_CANCER_SITES),
        "linked_recommendation_counts": dict(sorted(links.items())),
        "exclusion_reason_counts": dict(sorted(reasons.items())),
        "link_thresholds": {
            "min_link_score": MIN_LINK_SCORE,
            "min_link_margin": MIN_LINK_MARGIN,
            "min_clinical_tokens": MIN_CLINICAL_TOKENS,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    manifest = run_tier2_build()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
