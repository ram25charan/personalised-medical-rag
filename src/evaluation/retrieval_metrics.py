"""Retrieval-Level Evaluation Metrics (Sections 8.6, 12.2).

Computes Recall@5, Recall@10, nDCG@10, MRR, Hit Rate@5, and Stale-Evidence Rate
(share of selected events that are labelled distractors in the gold).
"""

from __future__ import annotations

import math
from typing import Any


def recall_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Compute Recall@k. For insufficient-evidence questions (`len(gold_ids) == 0`),
    returns `1.0` if no events are selected else `0.0` (or is excluded when aggregating over answerable questions).
    """
    gold_set = set(gold_ids)
    if not gold_set:
        return 1.0 if len(ranked_ids[:k]) == 0 else 0.0
    top_k = set(ranked_ids[:k])
    return float(len(top_k & gold_set)) / float(len(gold_set))


def hit_rate_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Compute Hit Rate@k (`1.0` if at least one gold event appears in top-k, else `0.0`)."""
    gold_set = set(gold_ids)
    if not gold_set:
        return 1.0 if len(ranked_ids[:k]) == 0 else 0.0
    top_k = set(ranked_ids[:k])
    return 1.0 if len(top_k & gold_set) > 0 else 0.0


def mrr_score(ranked_ids: list[str], gold_ids: list[str], k: int | None = None) -> float:
    """Compute Mean Reciprocal Rank (MRR) of the first gold event."""
    gold_set = set(gold_ids)
    if not gold_set:
        return 1.0 if len(ranked_ids) == 0 else 0.0
    search_list = ranked_ids[:k] if k is not None else ranked_ids
    for idx, eid in enumerate(search_list, start=1):
        if eid in gold_set:
            return 1.0 / float(idx)
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Compute binary nDCG@k (`1 / log2(rank + 1)` for each gold hit in top-k, divided by ideal IDCG@k)."""
    gold_set = set(gold_ids)
    if not gold_set:
        return 1.0 if len(ranked_ids[:k]) == 0 else 0.0

    dcg = 0.0
    seen: set[str] = set()
    for idx, eid in enumerate(ranked_ids[:k], start=1):
        if eid in gold_set and eid not in seen:
            dcg += 1.0 / math.log2(idx + 1.0)
            seen.add(eid)

    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(idx + 1.0) for idx in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def stale_evidence_rate(selected_ids: list[str], distractor_ids: list[str], k: int | None = None) -> float:
    """Compute the Stale-Evidence Rate (Section 8.6):
    share of selected top-k events that are labelled distractors in the gold (`distractor_events`).
    """
    sel = selected_ids[:k] if k is not None else selected_ids
    if not sel:
        return 0.0
    dist_set = set(distractor_ids)
    hits = sum(1 for eid in sel if eid in dist_set)
    return float(hits) / float(len(sel))


def retrieval_metrics(
    ranked_ids: list[str],
    gold_ids: list[str],
    distractor_ids: list[str],
    k: int = 10,
) -> dict[str, float]:
    """Compute all retrieval metrics for a single question (Section 8.6 & 12.2)."""
    return {
        "recall_at_5": round(recall_at_k(ranked_ids, gold_ids, 5), 6),
        "recall_at_10": round(recall_at_k(ranked_ids, gold_ids, 10), 6),
        "recall_at_k": round(recall_at_k(ranked_ids, gold_ids, k), 6),
        "ndcg_at_10": round(ndcg_at_k(ranked_ids, gold_ids, 10), 6),
        "ndcg_at_k": round(ndcg_at_k(ranked_ids, gold_ids, k), 6),
        "mrr": round(mrr_score(ranked_ids, gold_ids), 6),
        "hit_rate_at_5": round(hit_rate_at_k(ranked_ids, gold_ids, 5), 6),
        "stale_evidence_rate": round(stale_evidence_rate(ranked_ids, distractor_ids, k=k), 6),
    }


def aggregate_retrieval_metrics(per_question_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Compute mean retrieval metrics across a list of per-question metric dicts."""
    if not per_question_metrics:
        return {}
    keys = list(per_question_metrics[0].keys())
    n = float(len(per_question_metrics))
    return {
        key: round(sum(m[key] for m in per_question_metrics) / n, 4)
        for key in keys
    }
