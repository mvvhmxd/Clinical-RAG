"""Tier routing: criteria questions never answered from evidence prose, and vice versa."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ng12_rag.query_routing import QueryIntent, Tier2Retriever, classify_intent, route_query

TIER2_PATH = Path(__file__).resolve().parents[1] / "data/processed/full_evidence_chunks.jsonl"


@pytest.fixture(scope="module")
def tier2_chunks() -> list[dict]:
    if not TIER2_PATH.is_file():
        pytest.skip("Tier 2 corpus has not been built")
    return [json.loads(line) for line in TIER2_PATH.read_text().splitlines() if line.strip()]


class TestIntent:
    @pytest.mark.parametrize(
        "query",
        [
            "A 39-year-old with jaundice. Do they meet referral criteria for pancreatic cancer?",
            "What is the FIT threshold for colorectal referral?",
            "Should I refer a 50-year-old with visible haematuria?",
            "Does a 44-year-old with haematuria qualify for renal referral?",
        ],
    )
    def test_criteria_questions_route_to_tier1_only(self, query):
        decision = route_query(query)
        assert decision.intent is QueryIntent.CRITERIA_LOOKUP
        assert decision.search_tier1 is True
        assert decision.search_tier2 is False

    @pytest.mark.parametrize(
        "query",
        [
            "Why is the renal cancer age threshold set at 45?",
            "What evidence supports referring people with jaundice?",
            "How strong is the evidence for haematuria as a renal cancer symptom?",
        ],
    )
    def test_rationale_questions_route_to_tier2(self, query):
        decision = route_query(query)
        assert decision.intent is QueryIntent.RATIONALE_SEEKING
        assert decision.search_tier2 is True
        assert decision.search_tier1 is False

    def test_asking_for_both_keeps_both_tiers(self):
        decision = route_query(
            "What are the criteria for renal referral and why is the age threshold set there?"
        )
        assert decision.intent is QueryIntent.BOTH
        assert decision.search_tier1 and decision.search_tier2

    def test_why_about_a_specific_patient_is_still_a_criteria_question(self):
        """'Why would I refer this 50-year-old' applies a rule; it does not ask its origin."""

        decision = route_query("Why would I refer a 50-year-old with visible haematuria?")
        assert decision.intent is QueryIntent.CRITERIA_LOOKUP
        assert decision.search_tier2 is False

    def test_ambiguous_questions_default_to_the_primary_corpus(self):
        decision = route_query("Tell me about lung cancer referral")
        assert decision.search_tier1 is True
        assert decision.search_tier2 is False

    def test_decision_reports_the_signals_it_used(self):
        _, signals = classify_intent("Why is the threshold set at 45?")
        assert signals


class TestTier2Retrieval:
    def test_rationale_search_returns_linked_chunks(self, tier2_chunks):
        results = Tier2Retriever(tier2_chunks).search(
            "What evidence supports jaundice as a pancreatic cancer symptom?", top_k=3
        )
        assert results
        assert all(r[0]["metadata"]["linked_recommendation_id"] for r in results)

    def test_search_can_be_pinned_to_one_recommendation(self, tier2_chunks):
        results = Tier2Retriever(tier2_chunks).search(
            "why is this threshold set", top_k=5, linked_recommendation_ids=["1.6.6"]
        )
        assert all(r[0]["metadata"]["linked_recommendation_id"] == "1.6.6" for r in results)

    def test_every_returned_chunk_is_rationale_not_a_recommendation(self, tier2_chunks):
        """Tier 2 must never be presented as a numbered recommendation."""

        results = Tier2Retriever(tier2_chunks).search("evidence for haematuria", top_k=5)
        for chunk, _ in results:
            assert chunk["metadata"]["content_type"] == "rationale_evidence"
            assert chunk["metadata"]["source_tier"] == "secondary_full_guideline"

    def test_fit_threshold_rationale_is_genuinely_absent(self, tier2_chunks):
        """The 2015 evidence guideline predates the 2023 FIT recommendations.

        Recorded as a test so the gap stays visible: the corpus discusses faecal occult blood
        at length, which is adjacent enough to be conflated into a confident wrong rationale.
        """

        corpus = " ".join(c["text"].lower() for c in tier2_chunks)
        for absent in ("10 micrograms", "micrograms of haemoglobin", "hm-jackarc", "oc-sensor"):
            assert absent not in corpus
