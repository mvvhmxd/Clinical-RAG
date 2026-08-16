"""Clinical text normalisation shared by lexical indexing and retrieval."""

from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+){1,2}|\d+(?:\.\d+)?|[a-z]+")

CANONICAL_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("oesophageal", "esophageal"),
    ("haemoglobin", "hemoglobin"),
    ("haematuria", "hematuria"),
    ("faeces", "feces"),
    ("faecal", "fecal"),
    ("x-ray", "xray"),
    ("x ray", "xray"),
    ("computed tomography", "ct"),
    ("micrograms", "microgram"),
    ("microgrammes", "microgram"),
    ("μg", "microgram"),
    ("µg", "microgram"),
)

# Removing only high-frequency function words improves exact clinical term matching while
# retaining negation, comparison operators, age words, quantities, and action verbs.
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "using",
    "with",
}


def canonicalise(text: str) -> str:
    """Normalise British/US variants and measurement notation for retrieval."""

    value = text.casefold().replace("–", "-").replace("—", "-")
    for source, target in CANONICAL_REPLACEMENTS:
        value = value.replace(source, target)
    return value


def clinical_tokenize(text: str) -> list[str]:
    """Tokenise while preserving recommendation IDs, decimals, and thresholds."""

    tokens = TOKEN_PATTERN.findall(canonicalise(text))
    return [token for token in tokens if token not in STOPWORDS]


def numeric_tokens(text: str) -> set[str]:
    """Return explicit numeric values and recommendation identifiers."""

    return {
        token
        for token in TOKEN_PATTERN.findall(canonicalise(text))
        if token[0].isdigit()
    }


def recommendation_ids(text: str) -> set[str]:
    """Extract exact three-level NG12 recommendation identifiers."""

    return set(re.findall(r"\b1\.\d{1,2}\.\d{1,2}\b", text))
