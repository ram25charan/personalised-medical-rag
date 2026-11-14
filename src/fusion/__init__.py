"""Package exports for `src.fusion` (Section 12.2)."""

from src.fusion.fusion import compute_fused_scores, validate_weights
from src.fusion.selection import (
    order_chronologically,
    rank_by_fused_score,
    select,
    select_recency_only,
    select_status_filter_only,
)

__all__ = [
    "compute_fused_scores",
    "order_chronologically",
    "rank_by_fused_score",
    "select",
    "select_recency_only",
    "select_status_filter_only",
    "validate_weights",
]
