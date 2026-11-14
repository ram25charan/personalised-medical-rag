"""Top-k Selection and Chronological Ordering under a Context Budget (Sections 7.4, 8.2, 12.2)."""

from __future__ import annotations

from typing import Optional

from src.fusion.fusion import compute_fused_scores
from src.schemas import Event


def rank_by_fused_score(
    rq: dict[str, float],
    rh: dict[str, float],
    rt: dict[str, float],
    weights: dict[str, float],
    k: int = 10,
    events_by_id: Optional[dict[str, Event]] = None,
) -> tuple[list[str], dict[str, float]]:
    """Select top-k event_ids ranked by fused score `S(e)` (descending).

    Ties are broken deterministically by `Rq(e)` descending, then `observed_time` descending, then `event_id`.
    """
    fused = compute_fused_scores(rq, rh, rt, weights)
    ev_map = events_by_id or {}

    ranked_ids = sorted(
        fused.keys(),
        key=lambda eid: (
            -fused[eid],
            -rq.get(eid, 0.0),
            (ev_map[eid].observed_time or "") if eid in ev_map else "",
            eid,
        ),
    )
    return ranked_ids[:k], fused


def select(
    rq: dict[str, float],
    rh: dict[str, float],
    rt: dict[str, float],
    weights: dict[str, float],
    k: int = 10,
) -> list[str]:
    """Contract function `select(rq, rh, rt, weights, k) -> list[event_id]` (Section 12.2)."""
    selected_ids, _ = rank_by_fused_score(rq, rh, rt, weights, k=k)
    return selected_ids


def select_recency_only(
    candidate_events: list[Event],
    rq: dict[str, float],
    k: int = 10,
    pool_size: int = 50,
) -> tuple[list[str], dict[str, float]]:
    """RS4 baseline (Section 8.2): take the top `pool_size` (50) by Rq, then sort by recency and select top-k."""
    ev_by_id = {e.event_id: e for e in candidate_events}
    top_rq_ids = sorted(
        ev_by_id.keys(),
        key=lambda eid: (-rq.get(eid, 0.0), eid),
    )[:pool_size]

    # Sort the top-50 pool by observed_time descending (unknown dates last), tie-breaking by Rq descending
    recency_ranked = sorted(
        top_rq_ids,
        key=lambda eid: (
            ev_by_id[eid].observed_time or "0000-00-00",
            rq.get(eid, 0.0),
            eid,
        ),
        reverse=True,
    )
    # Assign synthetic display score decreasing by rank in recency order
    scores: dict[str, float] = {eid: 0.0 for eid in ev_by_id}
    n = max(1, len(recency_ranked))
    for idx, eid in enumerate(recency_ranked):
        scores[eid] = round(1.0 - (idx / n) * 0.9, 6)

    return recency_ranked[:k], scores


def select_status_filter_only(
    candidate_events: list[Event],
    rq: dict[str, float],
    rt: dict[str, float],
    k: int = 10,
    min_rt: float = 0.2,
) -> tuple[list[str], dict[str, float]]:
    """RS5 baseline (Section 8.2): drop events whose Rt for the intent is <= 0.2, then rank by Rq."""
    ev_ids = [e.event_id for e in candidate_events]
    kept_ids = [eid for eid in ev_ids if rt.get(eid, 0.0) > min_rt]
    ranked_kept = sorted(
        kept_ids,
        key=lambda eid: (-rq.get(eid, 0.0), eid),
    )
    scores = {
        eid: (rq.get(eid, 0.0) if rt.get(eid, 0.0) > min_rt else 0.0)
        for eid in ev_ids
    }
    return ranked_kept[:k], scores


def order_chronologically(event_ids: list[str], events_by_id: dict[str, Event]) -> list[str]:
    """Global prompt rule (Section 7): selected events are always ordered chronologically in the prompt."""
    return sorted(
        event_ids,
        key=lambda eid: (
            (events_by_id[eid].observed_time or "0000-00-00") if eid in events_by_id else "0000-00-00",
            eid,
        ),
    )
