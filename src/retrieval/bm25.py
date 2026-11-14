"""BM25 Lexical Retrieval over Patient Events (Sections 7.1, 10.1, 12.2)."""

from __future__ import annotations

import re
from rank_bm25 import BM25Okapi

from src.schemas import Event


def tokenize_medical_text(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer preserving clinical terms, numbers, and hyphens."""
    if not text:
        return []
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return tokens if tokens else [""]


def build_event_search_text(event: Event) -> str:
    """Build the searchable string for an event from its concept, class, and text."""
    parts = [event.concept]
    if event.concept_class:
        parts.append(event.concept_class)
    parts.append(event.text)
    return " ".join(p for p in parts if p)


def bm25_scores(candidate_events: list[Event], query_text: str) -> dict[str, float]:
    """Compute raw BM25 scores for `candidate_events` against `query_text`."""
    if not candidate_events:
        return {}
    corpus_tokens = [tokenize_medical_text(build_event_search_text(ev)) for ev in candidate_events]
    query_tokens = tokenize_medical_text(query_text)
    bm25 = BM25Okapi(corpus_tokens)
    raw = bm25.get_scores(query_tokens)
    return {
        ev.event_id: float(score) for ev, score in zip(candidate_events, raw)
    }
