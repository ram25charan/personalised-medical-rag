"""Reciprocal-Rank Fusion (RRF) and Question-Relevance (Rq) Scoring (Sections 7.1, 12.2)."""

from __future__ import annotations

from typing import Optional

from src.retrieval.bm25 import bm25_scores
from src.retrieval.dense import dense_scores
from src.schemas import Event
from src.temporal.state import filter_candidate_pool


def min_max_normalise(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalise a dictionary of scores into `[0.0, 1.0]` within the candidate pool (Section 7.1 & 7.4).

    If all scores in the pool are identical, returns `0.0` if they are `<= 0` else `1.0`.
    """
    if not scores:
        return {}
    vals = list(scores.values())
    v_min = min(vals)
    v_max = max(vals)
    span = v_max - v_min
    if span <= 1e-12:
        fill = 1.0 if v_max > 0.0 else 0.0
        return {k: fill for k in scores}
    return {k: float((v - v_min) / span) for k, v in scores.items()}


def reciprocal_rank_fusion(
    score_dicts: list[dict[str, float]],
    rrf_k: int = 60,
) -> dict[str, float]:
    """Combine multiple score dictionaries via Reciprocal-Rank Fusion:
    `RRF(e) = sum_{m} 1 / (rrf_k + rank_m(e))` where rank_m(e) is 1-based.
    """
    if not score_dicts:
        return {}
    all_ids = sorted(set().union(*(d.keys() for d in score_dicts)))
    fused: dict[str, float] = {eid: 0.0 for eid in all_ids}

    for s_dict in score_dicts:
        # Sort descending by score, tie-break deterministically by event_id
        ranked = sorted(all_ids, key=lambda eid: (-s_dict.get(eid, 0.0), eid))
        for rank_idx, eid in enumerate(ranked, start=1):
            fused[eid] += 1.0 / float(rrf_k + rank_idx)

    return fused


def rq_scores(
    patient_events: list[Event],
    text: str,
    as_of_date: str,
    mode: str = "hybrid",
    rrf_k: int = 60,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    patient_id: Optional[str] = None,
) -> dict[str, float]:
    """Compute min-max normalised Question Relevance `Rq(e) in [0, 1]` over events available by `as_of_date`.

    Modes:
      - `"hybrid"`: BM25 + Dense combined via RRF, then min-max normalised (B1 / B2 / B3).
      - `"bm25_only"`: BM25 scores min-max normalised (RS1).
      - `"dense_only"`: Dense cosine similarity scores min-max normalised (RS2).
    """
    pool = filter_candidate_pool(patient_events, as_of_date, patient_id=patient_id)
    if not pool:
        return {}

    if mode == "bm25_only":
        raw_bm25 = bm25_scores(pool, text)
        return min_max_normalise(raw_bm25)

    if mode == "dense_only":
        raw_dense = dense_scores(pool, text, model_name=model_name)
        return min_max_normalise(raw_dense)

    # Default: hybrid BM25 + Dense via RRF
    raw_bm25 = bm25_scores(pool, text)
    raw_dense = dense_scores(pool, text, model_name=model_name)
    rrf = reciprocal_rank_fusion([raw_bm25, raw_dense], rrf_k=rrf_k)
    return min_max_normalise(rrf)
