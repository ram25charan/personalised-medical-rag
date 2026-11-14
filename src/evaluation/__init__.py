"""Evaluation package for retrieval and generation metrics plus paired bootstrap CI."""

from src.evaluation.answer_metrics import (
    abstention_accuracy,
    answer_correctness,
    citation_precision_recall_f1,
    conflict_flag_accuracy,
    contradiction_rate,
    evaluate_single_answer,
    grounded_faithfulness,
    temporal_consistency,
    uncertainty_noted_accuracy,
)
from src.evaluation.bootstrap import bootstrap_mean_ci, paired_bootstrap
from src.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    hit_rate_at_k,
    mrr_score,
    ndcg_at_k,
    recall_at_k,
    retrieval_metrics,
    stale_evidence_rate,
)

__all__ = [
    "abstention_accuracy",
    "aggregate_retrieval_metrics",
    "answer_correctness",
    "bootstrap_mean_ci",
    "citation_precision_recall_f1",
    "conflict_flag_accuracy",
    "contradiction_rate",
    "evaluate_single_answer",
    "grounded_faithfulness",
    "hit_rate_at_k",
    "mrr_score",
    "ndcg_at_k",
    "paired_bootstrap",
    "recall_at_k",
    "retrieval_metrics",
    "stale_evidence_rate",
    "temporal_consistency",
    "uncertainty_noted_accuracy",
]
