"""Retrieval-Level Evaluation Runner (`experiments/run_retrieval.py`) (Sections 8.2, 8.4, 8.6, 8.7).

Runs all 8 retrieval-level systems (RS1, RS2, B1=RS3, RS4, RS5, B3, B2, ORACLE) across:
  - Track S Dev (50 questions)
  - Track S Test (100 questions)
  - Track R Test (30 questions)

Also runs:
  - Context-budget study (E1): k in {5, 10, 20}
  - Oracle-intent vs Rule-intent comparison (Section 7.3 & 8.4)
  - Paired bootstrap confidence intervals (1,000 resamples) for H1 (B3 vs B1, B2 vs B1),
    H2 (B2 vs B3), and baseline comparisons (B2 vs RS4, B2 vs RS5)
  - Per-track, per-intent, and per-category breakdowns (E4), never pooling tracks into a single score.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from src.evaluation.bootstrap import bootstrap_mean_ci, paired_bootstrap
from src.evaluation.retrieval_metrics import retrieval_metrics
from src.pipeline import load_json_config, run_retrieval_stage
from src.retrieval.rrf import rq_scores
from src.schemas import Event, Gold, Question, ResultRecord, load_jsonl, save_jsonl
from src.temporal.history_relevance import load_relation_table
from src.temporal.state import filter_candidate_pool
from src.temporal.temporal_scoring import load_rt_rules

RETRIEVAL_SYSTEMS = ["RS1", "RS2", "B1", "RS4", "RS5", "B3", "B2", "ORACLE"]
PRIMARY_METRICS = [
    "recall_at_5",
    "recall_at_10",
    "ndcg_at_10",
    "mrr",
    "hit_rate_at_5",
    "stale_evidence_rate",
]


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _compute_config_hash(*objs: Any) -> str:
    blob = json.dumps(objs, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _summarize_split_results(
    records_by_system: dict[str, list[dict[str, Any]]],
    n_resamples: int = 1000,
) -> dict[str, Any]:
    """Compute per-system bootstrap means + CIs, hypothesis paired tests, and intent/category breakdowns."""
    system_summaries: dict[str, Any] = {}
    for sys_id, recs in records_by_system.items():
        ans_recs = [r for r in recs if len(r["gold_events"]) > 0]
        target_pool = ans_recs if ans_recs else recs

        sys_metrics: dict[str, Any] = {}
        for m_name in PRIMARY_METRICS:
            pool_to_use = recs if m_name == "stale_evidence_rate" else target_pool
            vals = [float(r["metrics"][m_name]) for r in pool_to_use]
            sys_metrics[m_name] = bootstrap_mean_ci(vals, n_resamples=n_resamples)

        by_intent: dict[str, dict[str, float]] = {}
        intent_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in target_pool:
            intent_groups[r["gold_intent"]].append(r)
        for intent_name, grp in sorted(intent_groups.items()):
            by_intent[intent_name] = {
                "count": len(grp),
                "ndcg_at_10": round(sum(x["metrics"]["ndcg_at_10"] for x in grp) / len(grp), 4),
                "recall_at_10": round(sum(x["metrics"]["recall_at_10"] for x in grp) / len(grp), 4),
                "mrr": round(sum(x["metrics"]["mrr"] for x in grp) / len(grp), 4),
            }

        by_cat: dict[str, dict[str, float]] = {}
        cat_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in recs:
            cat_groups[r["category"]].append(r)
        for cat_name, grp in sorted(cat_groups.items()):
            by_cat[cat_name] = {
                "count": len(grp),
                "ndcg_at_10": round(sum(x["metrics"]["ndcg_at_10"] for x in grp) / len(grp), 4),
                "recall_at_10": round(sum(x["metrics"]["recall_at_10"] for x in grp) / len(grp), 4),
                "stale_evidence_rate": round(
                    sum(x["metrics"]["stale_evidence_rate"] for x in grp) / len(grp), 4
                ),
            }

        system_summaries[sys_id] = {
            "num_questions_total": len(recs),
            "num_questions_answerable": len(ans_recs),
            "metrics": sys_metrics,
            "by_intent": by_intent,
            "by_category": by_cat,
        }

    comparisons_spec = [
        ("H1_history_alone_B3_vs_B1", "B3", "B1"),
        ("H1_combined_B2_vs_B1", "B2", "B1"),
        ("H2_temporal_gain_B2_vs_B3", "B2", "B3"),
        ("Baseline_B2_vs_RS4_recency", "B2", "RS4"),
        ("Baseline_B2_vs_RS5_status_filter", "B2", "RS5"),
        ("Lexical_vs_Dense_B1_vs_RS1", "B1", "RS1"),
        ("Lexical_vs_Dense_B1_vs_RS2", "B1", "RS2"),
    ]
    paired_tests: dict[str, Any] = {}
    for comp_label, sys_a, sys_b in comparisons_spec:
        if sys_a not in records_by_system or sys_b not in records_by_system:
            continue
        recs_a = records_by_system[sys_a]
        recs_b = records_by_system[sys_b]
        ans_idx = [i for i, r in enumerate(recs_a) if len(r["gold_events"]) > 0]

        comp_res: dict[str, Any] = {}
        for m_name in PRIMARY_METRICS:
            idxs = range(len(recs_a)) if m_name == "stale_evidence_rate" else ans_idx
            vals_a = [float(recs_a[i]["metrics"][m_name]) for i in idxs]
            vals_b = [float(recs_b[i]["metrics"][m_name]) for i in idxs]
            comp_res[m_name] = paired_bootstrap(vals_a, vals_b, n_resamples=n_resamples)
        paired_tests[comp_label] = comp_res

    return {
        "systems": system_summaries,
        "paired_bootstrap_tests": paired_tests,
    }


def run_retrieval_experiments(
    k_default: int = 10,
    budget_ks: list[int] = [5, 10, 20],
) -> dict[str, Any]:
    """Run all Phase 3 retrieval-level experiments, ablations, and bootstrap analyses."""
    r_events = load_jsonl(Event, "data/processed/events.jsonl")
    s_events = load_jsonl(Event, "data/synthetic/events.jsonl")
    questions = load_jsonl(Question, "data/benchmark/questions.jsonl")
    golds = load_jsonl(Gold, "data/benchmark/gold.jsonl")

    weights_cfg = load_json_config("configs/weights.json")
    systems_cfg = load_json_config("configs/systems.json")
    rel_table = load_relation_table("configs/relation_table.json")
    rt_rules = load_rt_rules("configs/rt_rules.json")

    git_commit = _get_git_commit()
    cfg_hash = _compute_config_hash(weights_cfg, systems_cfg, rel_table, rt_rules)
    run_date_str = date.today().isoformat()

    events_by_patient: dict[str, list[Event]] = defaultdict(list)
    for ev in r_events + s_events:
        events_by_patient[ev.patient_id].append(ev)

    gold_by_qid = {g.question_id: g for g in golds}

    precomputed_rq: dict[str, dict[str, dict[str, float]]] = {}
    for q in questions:
        pool = filter_candidate_pool(
            events_by_patient[q.patient_id], q.as_of_date, patient_id=q.patient_id
        )
        precomputed_rq[q.question_id] = {
            "hybrid": rq_scores(pool, q.text, q.as_of_date, mode="hybrid", patient_id=q.patient_id),
            "bm25": rq_scores(pool, q.text, q.as_of_date, mode="bm25_only", patient_id=q.patient_id),
            "dense": rq_scores(pool, q.text, q.as_of_date, mode="dense_only", patient_id=q.patient_id),
        }

    splits = {
        "S_dev": [q for q in questions if q.track == "S" and q.split == "dev"],
        "S_test": [q for q in questions if q.track == "S" and q.split == "test"],
        "R_test": [q for q in questions if q.track == "R" and q.split == "test"],
    }

    all_result_records: list[ResultRecord] = []
    full_metrics_report: dict[str, Any] = {
        "schema_version": "1.0",
        "git_commit": git_commit,
        "config_hash": cfg_hash,
        "run_date": run_date_str,
        "default_k": k_default,
        "splits": {},
        "budget_study_E1": {},
        "intent_ablation_oracle_vs_rules": {},
    }

    for split_name, split_qs in splits.items():
        records_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sys_id in RETRIEVAL_SYSTEMS:
            for q in split_qs:
                g = gold_by_qid[q.question_id]
                p_evs = events_by_patient[q.patient_id]
                rq_cache = precomputed_rq[q.question_id]

                pipe_out = run_retrieval_stage(
                    patient_id=q.patient_id,
                    question_text=q.text,
                    as_of_date=q.as_of_date,
                    patient_events=p_evs,
                    system_id=sys_id,
                    k=k_default,
                    window_start=q.window_start,
                    intent_override=None,
                    oracle_gold_ids=g.gold_events if sys_id == "ORACLE" else None,
                    weights_config=weights_cfg,
                    systems_config=systems_cfg,
                    relation_table=rel_table,
                    rt_rules=rt_rules,
                    precomputed_rq_hybrid=rq_cache["hybrid"],
                    precomputed_rq_bm25=rq_cache["bm25"],
                    precomputed_rq_dense=rq_cache["dense"],
                )

                m = retrieval_metrics(
                    pipe_out.selected_event_ids,
                    g.gold_events,
                    g.distractor_events,
                    k=k_default,
                )
                records_by_system[sys_id].append(
                    {
                        "question_id": q.question_id,
                        "category": g.category,
                        "gold_intent": g.intent,
                        "pred_intent": pipe_out.intent,
                        "gold_events": g.gold_events,
                        "selected_event_ids": pipe_out.selected_event_ids,
                        "metrics": m,
                    }
                )

                w_used = (
                    weights_cfg.get(sys_id, {}).get(pipe_out.intent, {})
                    if sys_id in ("B1", "B2", "B3")
                    else {}
                )
                all_result_records.append(
                    ResultRecord(
                        run_id=f"phase3_{split_name}_{sys_id}_k{k_default}",
                        git_commit=git_commit,
                        config_hash=cfg_hash,
                        system_id=sys_id,
                        track=q.track,
                        question_id=q.question_id,
                        model_id=None,
                        run_date=run_date_str,
                        temperature=0.0,
                        k=k_default,
                        weights=w_used,
                        intent_source="rules",
                        prompt_hash=None,
                        raw_output=None,
                        parsed_answer=None,
                        metrics=m,
                    )
                )

        full_metrics_report["splits"][split_name] = _summarize_split_results(records_by_system)

    budget_systems = ["B1", "RS4", "RS5", "B3", "B2"]
    for split_name in ["S_test", "R_test"]:
        split_qs = splits[split_name]
        budget_table: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

        for k_val in budget_ks:
            for sys_id in budget_systems:
                recalls_k = []
                ndcgs_k = []
                stales_k = []
                for q in split_qs:
                    g = gold_by_qid[q.question_id]
                    rq_cache = precomputed_rq[q.question_id]
                    pipe_out = run_retrieval_stage(
                        patient_id=q.patient_id,
                        question_text=q.text,
                        as_of_date=q.as_of_date,
                        patient_events=events_by_patient[q.patient_id],
                        system_id=sys_id,
                        k=k_val,
                        window_start=q.window_start,
                        weights_config=weights_cfg,
                        systems_config=systems_cfg,
                        relation_table=rel_table,
                        rt_rules=rt_rules,
                        precomputed_rq_hybrid=rq_cache["hybrid"],
                        precomputed_rq_bm25=rq_cache["bm25"],
                        precomputed_rq_dense=rq_cache["dense"],
                    )
                    m = retrieval_metrics(
                        pipe_out.selected_event_ids,
                        g.gold_events,
                        g.distractor_events,
                        k=k_val,
                    )
                    stales_k.append(m["stale_evidence_rate"])
                    if len(g.gold_events) > 0:
                        recalls_k.append(m["recall_at_k"])
                        ndcgs_k.append(m["ndcg_at_k"])

                budget_table[sys_id][f"k={k_val}"] = {
                    "recall_at_k": round(sum(recalls_k) / max(1, len(recalls_k)), 4),
                    "ndcg_at_k": round(sum(ndcgs_k) / max(1, len(ndcgs_k)), 4),
                    "stale_evidence_rate": round(sum(stales_k) / max(1, len(stales_k)), 4),
                }
        full_metrics_report["budget_study_E1"][split_name] = dict(budget_table)

    for split_name in ["S_test", "R_test"]:
        split_qs = splits[split_name]
        intent_matches = 0
        rule_ndcgs = []
        oracle_ndcgs = []
        for q in split_qs:
            g = gold_by_qid[q.question_id]
            rq_cache = precomputed_rq[q.question_id]
            out_rule = run_retrieval_stage(
                patient_id=q.patient_id,
                question_text=q.text,
                as_of_date=q.as_of_date,
                patient_events=events_by_patient[q.patient_id],
                system_id="B2",
                k=k_default,
                window_start=q.window_start,
                intent_override=None,
                weights_config=weights_cfg,
                systems_config=systems_cfg,
                relation_table=rel_table,
                rt_rules=rt_rules,
                precomputed_rq_hybrid=rq_cache["hybrid"],
            )
            out_oracle = run_retrieval_stage(
                patient_id=q.patient_id,
                question_text=q.text,
                as_of_date=q.as_of_date,
                patient_events=events_by_patient[q.patient_id],
                system_id="B2",
                k=k_default,
                window_start=q.window_start,
                intent_override=g.intent,
                weights_config=weights_cfg,
                systems_config=systems_cfg,
                relation_table=rel_table,
                rt_rules=rt_rules,
                precomputed_rq_hybrid=rq_cache["hybrid"],
            )
            if out_rule.intent == g.intent:
                intent_matches += 1
            if len(g.gold_events) > 0:
                m_rule = retrieval_metrics(
                    out_rule.selected_event_ids, g.gold_events, g.distractor_events, k=k_default
                )
                m_orac = retrieval_metrics(
                    out_oracle.selected_event_ids, g.gold_events, g.distractor_events, k=k_default
                )
                rule_ndcgs.append(m_rule["ndcg_at_10"])
                oracle_ndcgs.append(m_orac["ndcg_at_10"])

        full_metrics_report["intent_ablation_oracle_vs_rules"][split_name] = {
            "intent_classification_accuracy": round(intent_matches / max(1, len(split_qs)), 4),
            "rule_intent_mean_ndcg_at_10": round(sum(rule_ndcgs) / max(1, len(rule_ndcgs)), 4),
            "oracle_intent_mean_ndcg_at_10": round(
                sum(oracle_ndcgs) / max(1, len(oracle_ndcgs)), 4
            ),
        }

    save_jsonl(all_result_records, "results/outputs/retrieval_runs.jsonl")
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    with Path("results/metrics/retrieval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(full_metrics_report, f, indent=2)

    _write_phase3_markdown_report(full_metrics_report, weights_cfg)
    return full_metrics_report


def _write_phase3_markdown_report(report: dict[str, Any], weights_cfg: dict[str, Any]) -> None:
    """Write human-readable tables of Phase 3 retrieval results to `docs/retrieval_evaluation_phase3.md`."""
    md = [
        "# Phase 3 Retrieval-Level Evaluation Report",
        "",
        f"- **Git commit:** `{report['git_commit']}`",
        f"- **Config hash:** `{report['config_hash']}`",
        f"- **Run date:** `{report['run_date']}`",
        f"- **Default context budget:** `k = {report['default_k']}`",
        "",
        "## 1. Frozen Per-Intent Fusion Weights (Tuned on Track S Dev Only)",
        "",
        "| Intent | B2 ($\\alpha, \\beta, \\gamma$) | B3 ($\\alpha, \\beta, \\gamma=0$) |",
        "|---|---|---|",
    ]
    for intent_name, w2 in weights_cfg.get("B2", {}).items():
        w3 = weights_cfg.get("B3", {}).get(intent_name, {})
        md.append(
            f"| `{intent_name}` | `({w2.get('alpha')}, {w2.get('beta')}, {w2.get('gamma')})` "
            f"| `({w3.get('alpha')}, {w3.get('beta')}, 0.0)` |"
        )

    for split_key, title in [
        ("S_test", "2. Track S Test Set Results (100 Questions; 86 Answerable, 14 Abstain)"),
        ("R_test", "3. Track R Real-Data Test Set Results (30 Questions; 26 Answerable, 4 Abstain) — Sim-to-Real Transfer (E2)"),
        ("S_dev", "4. Track S Development Set Results (50 Questions; 43 Answerable, 7 Abstain)"),
    ]:
        sp_data = report["splits"][split_key]["systems"]
        md.extend(
            [
                "",
                f"## {title}",
                "",
                "| System | Recall@5 | Recall@10 | nDCG@10 (95% CI) | MRR | HitRate@5 | Stale-Evidence Rate |",
                "|---|---:|---:|---|---:|---:|---:|",
            ]
        )
        for sys_id in RETRIEVAL_SYSTEMS:
            m = sp_data[sys_id]["metrics"]
            ndcg = m["ndcg_at_10"]
            md.append(
                f"| **{sys_id}** | {m['recall_at_5']['mean']:.4f} | {m['recall_at_10']['mean']:.4f} "
                f"| {ndcg['mean']:.4f} [{ndcg['ci_lower']:.4f}, {ndcg['ci_upper']:.4f}] "
                f"| {m['mrr']['mean']:.4f} | {m['hit_rate_at_5']['mean']:.4f} "
                f"| {m['stale_evidence_rate']['mean']:.4f} |"
            )

        pt = report["splits"][split_key]["paired_bootstrap_tests"]
        md.extend(
            [
                "",
                f"### Paired Bootstrap Hypothesis Tests (`{split_key}`, 1,000 resamples)",
                "| Comparison | $\\Delta$ nDCG@10 [95% CI] | $\\Delta$ Recall@10 [95% CI] | $\\Delta$ Stale Rate [95% CI] |",
                "|---|---|---|---|",
            ]
        )
        for comp_name, c_metrics in pt.items():
            dn = c_metrics["ndcg_at_10"]
            dr = c_metrics["recall_at_10"]
            ds = c_metrics["stale_evidence_rate"]
            md.append(
                f"| `{comp_name}` | `{dn['diff_mean']:+.4f}` [{dn['ci_lower']:+.4f}, {dn['ci_upper']:+.4f}] "
                f"| `{dr['diff_mean']:+.4f}` [{dr['ci_lower']:+.4f}, {dr['ci_upper']:+.4f}] "
                f"| `{ds['diff_mean']:+.4f}` [{ds['ci_lower']:+.4f}, {ds['ci_upper']:+.4f}] |"
            )

    md.extend(
        [
            "",
            "## 5. Exploratory Analysis E1: Context Budget Curve ($k \\in \\{5, 10, 20\\}$)",
            "",
            "| Track | System | Recall@5 (k=5) | Recall@10 (k=10) | Recall@20 (k=20) | nDCG@5 (k=5) | nDCG@10 (k=10) | nDCG@20 (k=20) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split_key in ["S_test", "R_test"]:
        b_tbl = report["budget_study_E1"][split_key]
        for sys_id in ["B1", "RS4", "RS5", "B3", "B2"]:
            k5 = b_tbl[sys_id]["k=5"]
            k10 = b_tbl[sys_id]["k=10"]
            k20 = b_tbl[sys_id]["k=20"]
            md.append(
                f"| `{split_key}` | **{sys_id}** | {k5['recall_at_k']:.4f} | {k10['recall_at_k']:.4f} "
                f"| {k20['recall_at_k']:.4f} | {k5['ndcg_at_k']:.4f} | {k10['ndcg_at_k']:.4f} | {k20['ndcg_at_k']:.4f} |"
            )

    md.extend(
        [
            "",
            "## 6. Intent Classification Ablation (Rule Intent vs. Oracle Intent on B2)",
            "",
            "| Split | Intent Classification Accuracy | Rule-Intent B2 nDCG@10 | Oracle-Intent B2 nDCG@10 |",
            "|---|---:|---:|---:|",
        ]
    )
    for split_key, ab_info in report["intent_ablation_oracle_vs_rules"].items():
        md.append(
            f"| `{split_key}` | {ab_info['intent_classification_accuracy']:.4f} "
            f"| {ab_info['rule_intent_mean_ndcg_at_10']:.4f} "
            f"| {ab_info['oracle_intent_mean_ndcg_at_10']:.4f} |"
        )

    Path("docs/retrieval_evaluation_phase3.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    rep = run_retrieval_experiments()
    print("Completed Phase 3 retrieval evaluation. Summary keys:", list(rep.keys()))
