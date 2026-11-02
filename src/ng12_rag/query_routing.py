"""Decide which corpus tier a question should be answered from.

The two tiers answer different questions and must not be pooled. "Does this patient meet the
criteria" is answered only from the numbered recommendations; "why is the threshold set here"
is answered from the evidence guideline, which is not normative and must never be presented as
a numbered recommendation.

When intent is genuinely unclear the router chooses Tier 1. Grounding safety on the criteria
path matters more than rationale coverage, and a criteria question answered from evidence
prose is the worse failure of the two.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "QueryIntent",
    "RoutingDecision",
    "Tier2Retriever",
    "classify_intent",
    "route_query",
]


class QueryIntent(StrEnum):
    """What the question is asking for."""

    CRITERIA_LOOKUP = "criteria_lookup"
    """Does this patient meet the criteria; what is the threshold."""

    RATIONALE_SEEKING = "rationale_seeking"
    """Why is the threshold set here; what evidence supports it."""

    BOTH = "both"
    """The criterion and its rationale are both explicitly requested."""


# "Why" alone is not decisive: "why would I refer this patient" is a criteria question.
# These phrases ask about the origin or strength of a rule rather than its application.
_RATIONALE_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwhy\s+(?:is|was|are|were|does|did|do)\b", re.I),
    re.compile(r"\bwhat\s+evidence\b", re.I),
    re.compile(r"\bhow\s+strong\s+is\s+the\s+evidence\b", re.I),
    re.compile(r"\b(?:rationale|justification|reasoning)\s+(?:for|behind)\b", re.I),
    re.compile(r"\bevidence\s+(?:base|supports?|supporting|behind|for)\b", re.I),
    re.compile(r"\b(?:basis|reason)\s+for\s+(?:the\s+)?(?:threshold|cut[- ]?off|age|criterion)\b", re.I),
    re.compile(r"\bhow\s+was\s+.{0,40}\b(?:chosen|derived|set|decided)\b", re.I),
    re.compile(r"\bstudies?\s+(?:support|underpin|inform)\b", re.I),
)

_CRITERIA_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:do|does|would|should)\s+(?:they|he|she|this|the)?\s*\w*\s*meet\b", re.I),
    re.compile(r"\bmeet(?:s)?\s+(?:the\s+)?(?:referral\s+)?criteri", re.I),
    re.compile(r"\bshould\s+(?:i|we|they)\s+refer\b", re.I),
    re.compile(r"\bwhat\s+is\s+the\s+(?:threshold|cut[- ]?off|age)\b", re.I),
    re.compile(r"\brefer(?:ral)?\s+criteri", re.I),
    # "criteria for renal referral" puts the words the other way round.
    re.compile(r"\bcriteri(?:a|on)\s+for\b", re.I),
    re.compile(r"\bwhat\s+are\s+the\s+criteri", re.I),
    re.compile(r"\bqualif(?:y|ies)\b", re.I),
)

# A stated patient value makes a question concrete, which points at criteria application.
_PATIENT_VALUE = re.compile(
    r"\b\d{1,3}\s*[-\s]?year[-\s]?old\b|\baged?\s+\d{1,3}\b|\b\d+(?:\.\d+)?\s*(?:µg|ug|micrograms?)\b",
    re.I,
)

_RECOMMENDATION_ID = re.compile(r"\b(1\.\d{1,2}\.\d{1,2})\b")


@dataclass(frozen=True)
class RoutingDecision:
    """Which tiers to search, and why."""

    intent: QueryIntent
    search_tier1: bool
    search_tier2: bool
    linked_recommendation_ids: tuple[str, ...] = ()
    reason: str = ""
    signals: tuple[str, ...] = field(default_factory=tuple)


def classify_intent(query: str) -> tuple[QueryIntent, list[str]]:
    """Return the question's intent and the phrases that decided it."""

    signals: list[str] = []
    rationale_hits = [p.pattern for p in _RATIONALE_MARKERS if p.search(query)]
    criteria_hits = [p.pattern for p in _CRITERIA_MARKERS if p.search(query)]
    has_patient_value = bool(_PATIENT_VALUE.search(query))

    if rationale_hits:
        signals.append(f"rationale_markers={len(rationale_hits)}")
    if criteria_hits:
        signals.append(f"criteria_markers={len(criteria_hits)}")
    if has_patient_value:
        signals.append("patient_value_stated")

    if rationale_hits and criteria_hits:
        return QueryIntent.BOTH, signals
    if rationale_hits and not has_patient_value:
        return QueryIntent.RATIONALE_SEEKING, signals
    # Everything else, including genuinely ambiguous phrasing, stays on the criteria path.
    return QueryIntent.CRITERIA_LOOKUP, signals


def route_query(query: str, *, tier1_recommendation_ids: Sequence[str] = ()) -> RoutingDecision:
    """Choose the corpus tiers for a question."""

    intent, signals = classify_intent(query)
    named_ids = tuple(
        rid for rid in _RECOMMENDATION_ID.findall(query) if not tier1_recommendation_ids or rid in tier1_recommendation_ids
    )

    if intent is QueryIntent.RATIONALE_SEEKING:
        return RoutingDecision(
            intent=intent,
            search_tier1=False,
            search_tier2=True,
            linked_recommendation_ids=named_ids,
            reason="asks why a rule exists, which the evidence guideline answers",
            signals=tuple(signals),
        )
    if intent is QueryIntent.BOTH:
        return RoutingDecision(
            intent=intent,
            search_tier1=True,
            search_tier2=True,
            linked_recommendation_ids=named_ids,
            reason="asks for the criterion and its rationale; they must stay separated",
            signals=tuple(signals),
        )
    return RoutingDecision(
        intent=intent,
        search_tier1=True,
        search_tier2=False,
        linked_recommendation_ids=named_ids,
        reason="criteria lookup, or ambiguous and defaulted to the primary corpus",
        signals=tuple(signals),
    )


_TOKEN = re.compile(r"[a-z][a-z\-]{2,}|\d+(?:\.\d+)?")
_TIER2_STOPWORDS = frozenset(
    ["the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "which", "have", "has", "been", "their", "they"]
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _TIER2_STOPWORDS]


class Tier2Retriever:
    """BM25-style search over the linked rationale corpus.

    Deliberately lexical and dependency-free: the rationale path must keep working when no
    embedding service is reachable, and this corpus is small enough that exact clinical terms
    carry most of the signal.
    """

    def __init__(self, chunks: Sequence[Mapping[str, Any]]) -> None:
        self.chunks = list(chunks)
        self._tokens = [_tokenize(str(c.get("text", ""))) for c in self.chunks]
        self._lengths = [len(t) or 1 for t in self._tokens]
        self._average_length = sum(self._lengths) / max(len(self._lengths), 1)
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))
        self._term_counts = [Counter(t) for t in self._tokens]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        linked_recommendation_ids: Sequence[str] = (),
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[tuple[Mapping[str, Any], float]]:
        """Return the highest-scoring rationale chunks, optionally pinned to a recommendation."""

        import math

        allowed = set(linked_recommendation_ids)
        query_tokens = _tokenize(query)
        total = len(self.chunks) or 1
        scored: list[tuple[Mapping[str, Any], float]] = []

        for index, chunk in enumerate(self.chunks):
            linked = str(chunk.get("metadata", {}).get("linked_recommendation_id") or "")
            if allowed and linked not in allowed:
                continue
            counts = self._term_counts[index]
            length = self._lengths[index]
            score = 0.0
            for term in query_tokens:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                idf = math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = frequency + k1 * (1 - b + b * length / self._average_length)
                score += idf * (frequency * (k1 + 1)) / denominator
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda pair: (-pair[1], str(pair[0].get("chunk_id"))))
        return scored[:top_k]
