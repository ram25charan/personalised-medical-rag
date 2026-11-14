"""Chronology-Aware Context Fusion Scoring (`S(e) = alpha * Rq(e) + beta * Rh(e) + gamma * Rt(e)`) (Sections 7.4, 12.2)."""

from __future__ import annotations

from src.retrieval.rrf import min_max_normalise


def validate_weights(weights: dict[str, float]) -> tuple[float, float, float]:
    """Validate that `alpha, beta, gamma >= 0` and `alpha + beta + gamma == 1.0` (within 1e-6)."""
    alpha = float(weights.get("alpha", 1.0))
    beta = float(weights.get("beta", 0.0))
    gamma = float(weights.get("gamma", 0.0))
    if alpha < -1e-9 or beta < -1e-9 or gamma < -1e-9:
        raise ValueError(f"Fusion weights must be non-negative, got {weights}")
    total = alpha + beta + gamma
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"Fusion weights must sum to 1.0, got {total} ({weights})")
    return alpha, beta, gamma


def compute_fused_scores(
    rq: dict[str, float],
    rh: dict[str, float],
    rt: dict[str, float],
    weights: dict[str, float],
    normalise_rq: bool = True,
) -> dict[str, float]:
    """Compute `S(e) = alpha * Rq(e) + beta * Rh(e) + gamma * Rt(e)` for all candidate events."""
    alpha, beta, gamma = validate_weights(weights)
    rq_norm = min_max_normalise(rq) if normalise_rq else rq
    all_ids = sorted(set(rq_norm.keys()) | set(rh.keys()) | set(rt.keys()))

    fused: dict[str, float] = {}
    for eid in all_ids:
        q_val = float(rq_norm.get(eid, 0.0))
        h_val = float(rh.get(eid, 0.0))
        t_val = float(rt.get(eid, 0.0))
        fused[eid] = round(alpha * q_val + beta * h_val + gamma * t_val, 8)
    return fused
