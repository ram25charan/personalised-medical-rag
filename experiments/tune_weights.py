"""Per-Intent Grid Search Weight Tuning on the Development Set (Sections 7.4, 8.7).

Tunes fusion weights `S(e) = alpha * Rq(e) + beta * Rh(e) + gamma * Rt(e)` on the 50 Track S
development questions ONLY using retrieval metrics (mean nDCG@10, ties broken by Recall@10):
  - B2: 66 grid combinations (`step = 0.1`, `alpha + beta + gamma = 1.0`, `alpha, beta, gamma >= 0`).
  - B3: 11 grid combinations (`step = 0.1`, `gamma = 0.0`, `alpha + beta = 1.0`).
  - Also verifies the insufficient-evidence Rq threshold `tau` on dev data (Section 7.6 U5).
  - Saves frozen weights to `configs/weights.json` and weight sensitivity analysis to `results/metrics/weight_sensitivity_dev.json`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k
from src.fusion.selection import rank_by_fused_score
from src.retrieval.rrf import rq_scores
from src.schemas import Event, Gold, IntentType, Question, load_jsonl
from src.temporal.history_relevance import (
    build_anchor_set,
    extract_question_anchors,
    load_relation_table,
    rh_scores,
)
from src.temporal.state import compute_state, filter_candidate_pool
from src.temporal.temporal_scoring import (
    classify_intent_by_rules,
    load_rt_rules,
    rt_scores,
)


def generate_b2_grid() -> list[dict[str, float]]:
    """Generate the 66 simplex grid points with step 0.1 (`alpha + beta + gamma = 1.0`)."""
    grid: list[dict[str, float]] = []
    for i in range(11):
        for j in range(11 - i):
            k = 10 - i - j
            grid.append(
                {
                    "alpha": round(i / 10.0, 1),
                    "beta": round(j / 10.0, 1),
                    "gamma": round(k / 10.0, 1),
                }
            )
    assert len(grid) == 66
    return grid


def generate_b3_grid() -> list[dict[str, float]]:
    """Generate the 11 grid points for B3 (`gamma = 0.0`, `alpha + beta = 1.0`)."""
    grid: list[dict[str, float]] = []
    for i in range(11):
        j = 10 - i
        grid.append(
            {
                "alpha": round(i / 10.0, 1),
                "beta": round(j / 10.0, 1),
                "gamma": 0.0,
            }
        )
    assert len(grid) == 11
    return grid


def tune_weights_on_dev(
    events_path: str | Path = "data/synthetic/events.jsonl",
    questions_path: str | Path = "data/benchmark/questions.jsonl",
    gold_path: str | Path = "data/benchmark/gold.jsonl",
    relation_table_path: str | Path = "configs/relation_table.json",
    rt_rules_path: str | Path = "configs/rt_rules.json",
    weights_out_path: str | Path = "configs/weights.json",
    sensitivity_out_path: str | Path = "results/metrics/weight_sensitivity_dev.json",
    k: int = 10,
) -> dict[str, Any]:
    """Run per-intent grid search on Track S dev questions and freeze `configs/weights.json`."""
    all_events = load_jsonl(Event, events_path)
    all_questions = load_jsonl(Question, questions_path)
    all_golds = load_jsonl(Gold, gold_path)
    rel_table = load_relation_table(relation_table_path)
    rt_cfg = load_rt_rules(rt_rules_path)

    events_by_patient: dict[str, list[Event]] = defaultdict(list)
    for ev in all_events:
        events_by_patient[ev.patient_id].append(ev)

    gold_by_qid = {g.question_id: g for g in all_golds}
    dev_questions = [
        q for q in all_questions if q.track == "S" and q.split == "dev"
    ]

    dev_features: list[dict[str, Any]] = []
    for q in dev_questions:
        g = gold_by_qid[q.question_id]
        pool = filter_candidate_pool(
            events_by_patient[q.patient_id], q.as_of_date, patient_id=q.patient_id
        )
        ev_by_id = {e.event_id: e for e in pool}
        states = {e.event_id: compute_state(e, q.as_of_date, pool) for e in pool}
        rq = rq_scores(pool, q.text, q.as_of_date, mode="hybrid", patient_id=q.patient_id)
        q_anchors = extract_question_anchors(q.text, rel_table, pool)
        anchors, anchor_encs = build_anchor_set(q.text, pool, rq, rel_table, top_rq_n=5)
        rh = rh_scores(pool, anchors, rel_table, anchor_encounters=anchor_encs)
        rule_intent = classify_intent_by_rules(q.text, rt_cfg)
        rt_anchor_in = q_anchors if q_anchors else anchors
        rt = rt_scores(
            pool,
            intent=rule_intent,
            as_of_date=q.as_of_date,
            rules=rt_cfg,
            anchors=rt_anchor_in,
            window_start=q.window_start,
            precomputed_states=states,
        )
        dev_features.append(
            {
                "question": q,
                "gold": g,
                "intent": rule_intent,
                "ev_by_id": ev_by_id,
                "rq": rq,
                "rh": rh,
                "rt": rt,
                "q_anchors": q_anchors,
                "pool": pool,
            }
        )

    b2_grid = generate_b2_grid()
    b3_grid = generate_b3_grid()

    all_intents: list[IntentType] = [
        "current_state",
        "historical",
        "change_over_time",
        "first_occurrence",
        "persistent_fact",
    ]

    chosen_b2: dict[str, dict[str, float]] = {}
    chosen_b3: dict[str, dict[str, float]] = {}
    sensitivity_report: dict[str, Any] = {"B2_by_intent": {}, "B3_by_intent": {}}

    for intent_name in all_intents:
        intent_items = [
            item
            for item in dev_features
            if item["intent"] == intent_name and len(item["gold"].gold_events) > 0
        ]
        if not intent_items:
            intent_items = [item for item in dev_features if len(item["gold"].gold_events) > 0]

        b2_results = []
        for w in b2_grid:
            ndcgs = []
            recalls = []
            for item in intent_items:
                ranked, _ = rank_by_fused_score(
                    item["rq"], item["rh"], item["rt"], w, k=k, events_by_id=item["ev_by_id"]
                )
                ndcgs.append(ndcg_at_k(ranked, item["gold"].gold_events, k))
                recalls.append(recall_at_k(ranked, item["gold"].gold_events, k))
            mean_ndcg = sum(ndcgs) / len(ndcgs)
            mean_rec = sum(recalls) / len(recalls)
            b2_results.append(
                {
                    "weights": w,
                    "mean_ndcg_at_10": round(mean_ndcg, 6),
                    "mean_recall_at_10": round(mean_rec, 6),
                }
            )

        b2_results.sort(
            key=lambda r: (
                r["mean_ndcg_at_10"],
                r["mean_recall_at_10"],
                1 if (r["weights"]["alpha"] > 0 and r["weights"]["beta"] > 0 and r["weights"]["gamma"] > 0) else 0,
                r["weights"]["gamma"],
                r["weights"]["alpha"],
            ),
            reverse=True,
        )
        chosen_b2[intent_name] = b2_results[0]["weights"]
        sensitivity_report["B2_by_intent"][intent_name] = {
            "num_dev_questions": len(intent_items),
            "best": b2_results[0],
            "top_5": b2_results[:5],
            "all_grid": b2_results,
        }

        b3_results = []
        for w in b3_grid:
            ndcgs = []
            recalls = []
            for item in intent_items:
                ranked, _ = rank_by_fused_score(
                    item["rq"], item["rh"], item["rt"], w, k=k, events_by_id=item["ev_by_id"]
                )
                ndcgs.append(ndcg_at_k(ranked, item["gold"].gold_events, k))
                recalls.append(recall_at_k(ranked, item["gold"].gold_events, k))
            mean_ndcg = sum(ndcgs) / len(ndcgs)
            mean_rec = sum(recalls) / len(recalls)
            b3_results.append(
                {
                    "weights": w,
                    "mean_ndcg_at_10": round(mean_ndcg, 6),
                    "mean_recall_at_10": round(mean_rec, 6),
                }
            )

        b3_results.sort(
            key=lambda r: (
                r["mean_ndcg_at_10"],
                r["mean_recall_at_10"],
                1 if (r["weights"]["alpha"] > 0 and r["weights"]["beta"] > 0) else 0,
                r["weights"]["beta"],
            ),
            reverse=True,
        )
        chosen_b3[intent_name] = b3_results[0]["weights"]
        sensitivity_report["B3_by_intent"][intent_name] = {
            "num_dev_questions": len(intent_items),
            "best": b3_results[0],
            "all_grid": b3_results,
        }

    best_tau = 0.35
    best_u5_acc = -1.0
    for cand_tau in [0.15, 0.25, 0.35, 0.45, 0.55]:
        correct = 0
        for item in dev_features:
            is_insuff_gold = item["gold"].category == "insufficient_evidence"
            q_anchors = item["q_anchors"]
            pool = item["pool"]
            rq = item["rq"]
            matching = [
                ev
                for ev in pool
                if ev.concept.lower().strip() in set(q_anchors)
                or any(a in ev.text.lower() for a in q_anchors)
            ]
            pred_insuff = (not matching) or (max(rq.get(e.event_id, 0.0) for e in matching) < cand_tau)
            if pred_insuff == is_insuff_gold:
                correct += 1
        acc = correct / max(1, len(dev_features))
        if acc > best_u5_acc:
            best_u5_acc = acc
            best_tau = cand_tau

    weights_payload = {
        "version": "1.0",
        "frozen": True,
        "tuned_on": "Track S dev split (50 questions; ranking weights tuned on the 43 answerable questions)",
        "selection_metric": "mean_ndcg_at_10 (ties broken by mean_recall_at_10)",
        "grid_step": 0.1,
        "insufficient_evidence_tau": best_tau,
        "insufficient_evidence_dev_accuracy": round(best_u5_acc, 4),
        "B1": {intent: {"alpha": 1.0, "beta": 0.0, "gamma": 0.0} for intent in all_intents},
        "B3": chosen_b3,
        "B2": chosen_b2,
        "rationale": (
            "Per-intent weights chosen by exhaustive grid search (step 0.1, 66 combinations for B2, "
            "11 combinations for B3) on the Track S development set using retrieval metrics only "
            "(mean nDCG@10, ties broken by Recall@10) per Section 7.4."
        ),
    }

    Path(weights_out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(weights_out_path).open("w", encoding="utf-8") as f:
        json.dump(weights_payload, f, indent=2)

    Path(sensitivity_out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(sensitivity_out_path).open("w", encoding="utf-8") as f:
        json.dump(sensitivity_report, f, indent=2)

    return weights_payload


if __name__ == "__main__":
    res = tune_weights_on_dev()
    print(json.dumps(res, indent=2))
