"""Recommendation-aware RAG for the April 2026 NICE NG12 guideline."""

from ng12_rag.config import Settings, load_settings
from ng12_rag.models import RAGResponse, RecommendationChunk, RetrievedChunk

__all__ = [
    "RAGResponse",
    "RecommendationChunk",
    "RetrievedChunk",
    "Settings",
    "load_settings",
]

__version__ = "0.1.0"
