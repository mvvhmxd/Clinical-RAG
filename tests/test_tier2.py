"""Tier 2 corpus contract: scope lock, linkage, tagging, and cleaning."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ng12_rag.ingestion_full import FULL_SUBSECTION_SITES, _valid_subsection_heading
from ng12_rag.tier2 import (
    CONTENT_TYPE,
    LOCKED_CANCER_SITES,
    SOURCE_TIER,
    build_tier2_corpus,
    clean_tier2_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data/processed/full_chunks.jsonl"
TIER1_PATH = REPO_ROOT / "data/processed/chunks.jsonl"


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def corpora() -> tuple[list[dict], list[dict], list[dict]]:
    if not RAW_PATH.is_file():
        pytest.skip("full-guideline extraction has not been run")
    tier1 = _read(TIER1_PATH)
    active, excluded = build_tier2_corpus(_read(RAW_PATH), tier1)
    return tier1, active, excluded


def test_every_active_chunk_links_to_a_real_tier1_recommendation(corpora):
    tier1, active, _ = corpora
    valid_ids = {record["metadata"]["recommendation_id"] for record in tier1}
    assert active, "expected a non-empty Tier 2 corpus"
    for record in active:
        linked = record["metadata"].get("linked_recommendation_id")
        assert linked, f"{record['chunk_id']} has no linked_recommendation_id"
        assert linked in valid_ids, f"{record['chunk_id']} links to unknown id {linked}"


def test_active_chunks_are_tagged_as_tier2(corpora):
    _, active, _ = corpora
    for record in active:
        assert record["metadata"]["content_type"] == CONTENT_TYPE
        assert record["metadata"]["source_tier"] == SOURCE_TIER


def test_scope_lock_applies_identically_to_tier2(corpora):
    """A site excluded from Tier 1 must not be reachable through Tier 2."""

    _, active, excluded = corpora
    for record in active:
        assert record["metadata"]["cancer_site"] in LOCKED_CANCER_SITES

    excluded_sites = {
        record["metadata"]["cancer_site"]
        for record in excluded
        if str(record["metadata"].get("exclusion_reason", "")).startswith("out_of_locked_scope")
    }
    # These were present in the raw extraction and must have been filtered out.
    assert {"prostate", "mesothelioma", "anal"} <= excluded_sites


def test_a_linked_chunk_shares_its_site_with_its_recommendation(corpora):
    tier1, active, _ = corpora
    site_of = {r["metadata"]["recommendation_id"]: r["metadata"]["cancer_site"] for r in tier1}
    for record in active:
        linked = record["metadata"]["linked_recommendation_id"]
        assert site_of[linked] == record["metadata"]["cancer_site"]


def test_every_excluded_chunk_records_a_reason(corpora):
    _, _, excluded = corpora
    for record in excluded:
        assert record["metadata"].get("exclusion_reason")


def test_unlinkable_chunks_are_excluded_not_guessed(corpora):
    """Ambiguity must produce an exclusion, never an arbitrary link."""

    _, _, excluded = corpora
    reasons = [str(r["metadata"]["exclusion_reason"]) for r in excluded]
    assert any(r.startswith("ambiguous_between_recommendations") for r in reasons)


def test_grade_table_rows_are_not_treated_as_subsection_headings():
    """'7.5 (6.6-8.5) 220/2930' previously became a subsection title."""

    assert not _valid_subsection_heading("7.5", "(6.6-8.5) 220/2930")
    assert not _valid_subsection_heading("9.03", "(6.82-11.7) 52/576")
    assert not _valid_subsection_heading("8.37", "(6.12-11.1) 43/514")
    assert _valid_subsection_heading("7.1", "Lung cancer")
    assert _valid_subsection_heading("12.3", "Renal cancer")


def test_subsection_site_map_covers_tier1_locked_sites():
    mapped = set(FULL_SUBSECTION_SITES.values())
    assert mapped >= LOCKED_CANCER_SITES


def test_cleaning_removes_statistics_rows_but_keeps_prose():
    text = "\n".join(
        [
            "Jaundice is associated with pancreatic cancer in primary care.",
            "7.5 (6.6-8.5) 220/2930",
            "TP = True positives, FP = False positives.",
            "The committee agreed that age 40 was an appropriate threshold.",
        ]
    )
    cleaned = clean_tier2_text(text)
    assert "Jaundice is associated with pancreatic cancer" in cleaned
    assert "committee agreed that age 40" in cleaned
    assert "220/2930" not in cleaned
    assert "True positives" not in cleaned


def test_no_tier1_recommendation_ids_are_fabricated_in_active_text(corpora):
    """Any 1.x.y string appearing in Tier 2 text must be a real Tier 1 id."""

    tier1, active, _ = corpora
    valid_ids = {record["metadata"]["recommendation_id"] for record in tier1}
    pattern = re.compile(r"\b1\.\d{1,2}\.\d{1,2}\b")
    for record in active:
        for found in pattern.findall(record["text"]):
            assert found in valid_ids, f"{record['chunk_id']} cites unknown id {found}"
