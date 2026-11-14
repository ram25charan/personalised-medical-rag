"""Unified Generation & Retrieval Evaluation Runner (`src/evaluation/runner.py`) (Sections 8.3, 8.4, 8.6, 8.7).

Supports:
  - Priority-ordered generation evaluation across `{B1, B2, B1+, B2-, B4, ORACLE}`
  - Precomputed hybrid Rq caching in memory so running multiple systems on the same split
    computes dense embeddings only once per question
  - Automatic LLM disk response caching (`results/cache/`) and 4-key rotation (`MultiKeyLLMClient`)
  - Full deterministic answer metrics + retrieval metrics per question
  - 1,000-resample bootstrap means, 95% CIs, paired hypothesis tests (`H1/H2`, `H3`, `E3` 2x2 factorial),
    and per-intent / per-category breakdowns (`E4`).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Optional

from src.evaluation.answer_metrics import evaluate_single_answer
from src.evaluation.bootstrap import bootstrap_mean_ci, paired_bootstrap
from src.evaluation.retrieval_metrics import retrieval_metrics
from src.generation.groq_client import MultiKeyLLMClient
from src.generation.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from src.pipeline import load_json_config, run_retrieval_stage
from src.retrieval.rrf import rq_scores
from src.schemas import Event, Gold, Question, ResultRecord
from src.temporal.history_relevance import load_relation_table
from src.temporal.state import compute_state, filter_candidate_pool
from src.temporal.temporal_scoring import load_rt_rules

GENERATION_SYSTEMS_PRIORITY = ["B1", "B2", "B1+", "B2-", "B4", "ORACLE"]

GENERATION_METRICS = [
    "answer_correctness",
    "temporal_consistency",
    "contradiction_rate",
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "abstention_accuracy",
    "conflict_flag_accuracy",
    "uncertainty_noted_accuracy",
    "grounded_faithfulness",
]


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def compute_config_hash(*objs: Any) -> str:
    blob = json.dumps(objs, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def summarize_generation_split(
    records_by_system: dict[str, list[dict[str, Any]]],
    n_resamples: int = 1000,
) -> dict[str, Any]:
    """Compute bootstrap means + 95% CIs, paired bootstrap hypothesis tests, 2x2 factorial summary,
    and per-intent / per-category breakdowns for a split.
    """
    system_summaries: dict[str, Any] = {}
    for sys_id, recs in records_by_system.items():
        if not recs:
            continue
        ans_recs = [r for r in recs if r["expected_behavior"] != "abstain"]

        sys_metrics: dict[str, Any] = {}
        for m_name in GENERATION_METRICS:
            vals = [float(r["metrics"][m_name]) for r in recs]
            sys_metrics[m_name] = bootstrap_mean_ci(vals, n_resamples=n_resamples)

        # Also compute answer_correctness specifically on answerable questions vs abstain questions
        if ans_recs:
            sys_metrics["answer_correctness_answerable_only"] = bootstrap_mean_ci(
                [float(r["metrics"]["answer_correctness"]) for r in ans_recs],
                n_resamples=n_resamples,
            )

        # Breakdown by intent (E4)
        by_intent: dict[str, dict[str, Any]] = {}
        intent_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in recs:
            intent_groups[r["gold_intent"]].append(r)
        for intent_name, grp in sorted(intent_groups.items()):
            by_intent[intent_name] = {
                "count": len(grp),
                "answer_correctness": round(
                    sum(x["metrics"]["answer_correctness"] for x in grp) / len(grp), 4
                ),
                "temporal_consistency": round(
                    sum(x["metrics"]["temporal_consistency"] for x in grp) / len(grp), 4
                ),
                "contradiction_rate": round(
                    sum(x["metrics"]["contradiction_rate"] for x in grp) / len(grp), 4
                ),
                "citation_f1": round(
                    sum(x["metrics"]["citation_f1"] for x in grp) / len(grp), 4
                ),
            }

        # Breakdown by category (E4)
        by_category: dict[str, dict[str, Any]] = {}
        cat_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in recs:
            cat_groups[r["category"]].append(r)
        for cat_name, grp in sorted(cat_groups.items()):
            by_category[cat_name] = {
                "count": len(grp),
                "answer_correctness": round(
                    sum(x["metrics"]["answer_correctness"] for x in grp) / len(grp), 4
                ),
                "temporal_consistency": round(
                    sum(x["metrics"]["temporal_consistency"] for x in grp) / len(grp), 4
                ),
                "contradiction_rate": round(
                    sum(x["metrics"]["contradiction_rate"] for x in grp) / len(grp), 4
                ),
                "citation_f1": round(
                    sum(x["metrics"]["citation_f1"] for x in grp) / len(grp), 4
                ),
                "conflict_flag_accuracy": round(
                    sum(x["metrics"]["conflict_flag_accuracy"] for x in grp) / len(grp), 4
                ),
                "uncertainty_noted_accuracy": round(
                    sum(x["metrics"]["uncertainty_noted_accuracy"] for x in grp) / len(grp), 4
                ),
                "abstention_accuracy": round(
                    sum(x["metrics"]["abstention_accuracy"] for x in grp) / len(grp), 4
                ),
            }

        system_summaries[sys_id] = {
            "num_questions_total": len(recs),
            "num_questions_answerable": len(ans_recs),
            "metrics": sys_metrics,
            "by_intent": by_intent,
            "by_category": by_category,
        }

    # Paired bootstrap comparisons (H1/H2, H3, E3 2x2 factorial, and ORACLE upper bound)
    comparisons_spec = [
        ("H1_H2_full_B2_vs_B1", "B2", "B1"),
        ("H3_flags_B2_vs_B4", "B2", "B4"),
        ("E3_selection_at_A0_B2minus_vs_B1", "B2-", "B1"),
        ("E3_selection_at_A2_B2_vs_B1plus", "B2", "B1+"),
        ("E3_annotation_on_B1_B1plus_vs_B1", "B1+", "B1"),
        ("E3_annotation_on_B2_B2_vs_B2minus", "B2", "B2-"),
        ("E3_status_A1_vs_raw_A0_B4_vs_B2minus", "B4", "B2-"),
        ("Upper_bound_ORACLE_vs_B2", "ORACLE", "B2"),
    ]
    paired_tests: dict[str, Any] = {}
    for label, sys_a, sys_b in comparisons_spec:
        if sys_a not in records_by_system or sys_b not in records_by_system:
            continue
        recs_a = records_by_system[sys_a]
        recs_b = records_by_system[sys_b]
        if len(recs_a) != len(recs_b):
            continue
        paired_tests[label] = {}
        for m_name in GENERATION_METRICS:
            vals_a = [float(r["metrics"][m_name]) for r in recs_a]
            vals_b = [float(r["metrics"][m_name]) for r in recs_b]
            paired_tests[label][m_name] = paired_bootstrap(
                vals_a, vals_b, n_resamples=n_resamples
            )

    # E3 2x2 Factorial Matrix summary ({B1, B1+, B2-, B2})
    factorial_e3: dict[str, Any] = {}
    if all(s in system_summaries for s in ["B1", "B1+", "B2-", "B2"]):
        for m_name in [
            "answer_correctness",
            "temporal_consistency",
            "contradiction_rate",
            "citation_f1",
            "conflict_flag_accuracy",
            "uncertainty_noted_accuracy",
        ]:
            b1_v = system_summaries["B1"]["metrics"][m_name]["mean"]
            b1p_v = system_summaries["B1+"]["metrics"][m_name]["mean"]
            b2m_v = system_summaries["B2-"]["metrics"][m_name]["mean"]
            b2_v = system_summaries["B2"]["metrics"][m_name]["mean"]
            factorial_e3[m_name] = {
                "B1_selB1_annA0": b1_v,
                "B1plus_selB1_annA2": b1p_v,
                "B2minus_selB2_annA0": b2m_v,
                "B2_selB2_annA2": b2_v,
                "main_effect_selection": round(0.5 * ((b2m_v - b1_v) + (b2_v - b1p_v)), 4),
                "main_effect_annotation": round(0.5 * ((b1p_v - b1_v) + (b2_v - b2m_v)), 4),
                "interaction_effect": round((b2_v - b2m_v) - (b1p_v - b1_v), 4),
            }

    return {
        "systems": system_summaries,
        "paired_hypothesis_tests": paired_tests,
        "factorial_2x2_E3": factorial_e3,
    }


def run_generation_evaluation_on_questions(
    questions: list[Question],
    gold_by_qid: dict[str, Gold],
    events_by_patient: dict[str, list[Event]],
    system_ids: list[str],
    provider: str = "groq",
    model_id: Optional[str] = None,
    k: int = 10,
    temperature: float = 0.0,
    use_cache: bool = True,
    min_request_interval_sec: float = 0.8,
) -> tuple[dict[str, list[dict[str, Any]]], list[ResultRecord]]:
    """Run generation evaluation across `system_ids` on `questions`."""
    sys_cfg = load_json_config("configs/systems.json")
    weights_cfg = load_json_config("configs/weights.json")
    rel_tbl = load_relation_table("configs/relation_table.json")
    rt_cfg = load_rt_rules("configs/rt_rules.json")

    client = MultiKeyLLMClient(
        provider=provider,
        model_id=model_id,
        temperature=temperature,
        min_request_interval_sec=min_request_interval_sec,
    )
    active_model_id = client.model_id
    git_commit = get_git_commit()
    cfg_hash = compute_config_hash(
        sys_cfg, weights_cfg, rel_tbl, rt_cfg, PROMPT_VERSION, active_model_id
    )
    run_date_str = date.today().isoformat()

    # Precompute Rq hybrid and temporal states per question once so all 6 systems share them
    precomputed_rq: dict[str, dict[str, float]] = {}
    precomputed_pools: dict[str, list[Event]] = {}
    precomputed_states: dict[str, dict[str, Any]] = {}

    for q in questions:
        p_evs = events_by_patient.get(q.patient_id, [])
        pool = filter_candidate_pool(p_evs, q.as_of_date, patient_id=q.patient_id)
        precomputed_pools[q.question_id] = pool
        precomputed_states[q.question_id] = {
            ev.event_id: compute_state(ev, q.as_of_date, pool) for ev in pool
        }
        precomputed_rq[q.question_id] = rq_scores(
            pool, q.text, q.as_of_date, mode="hybrid", patient_id=q.patient_id
        )

    records_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    result_records: list[ResultRecord] = []

    total_calls = len(system_ids) * len(questions)
    done_calls = 0

    for sys_id in system_ids:
        sys_recs_temp: list[dict[str, Any]] = []
        sys_res_temp: list[ResultRecord] = []
        sys_aborted = False
        for q in questions:
            done_calls += 1
            gold = gold_by_qid[q.question_id]
            p_evs = events_by_patient.get(q.patient_id, [])
            pool = precomputed_pools[q.question_id]
            states_by_id = precomputed_states[q.question_id]

            pipe_out = run_retrieval_stage(
                patient_id=q.patient_id,
                question_text=q.text,
                as_of_date=q.as_of_date,
                patient_events=p_evs,
                system_id=sys_id,
                k=k,
                window_start=q.window_start,
                oracle_gold_ids=gold.gold_events if sys_id in ("ORACLE", "ORACLE-G") else None,
                weights_config=weights_cfg,
                systems_config=sys_cfg,
                relation_table=rel_tbl,
                rt_rules=rt_cfg,
                precomputed_rq_hybrid=precomputed_rq[q.question_id],
            )

            user_prompt = build_user_prompt(
                patient_id=q.patient_id,
                question_text=q.text,
                as_of_date=q.as_of_date,
                context_text=pipe_out.context_text,
                window_start=q.window_start,
            )

            try:
                parsed_ans, raw_out, prompt_hash, call_meta = client.generate_from_prompts(
                    SYSTEM_PROMPT,
                    user_prompt,
                    use_cache=use_cache,
                )
            except RuntimeError as exc:
                print(
                    f"[WARNING] API quota exhausted during system {sys_id} at {q.question_id}: {exc}. "
                    f"Preserving earlier completed priority systems per Section 8.3.",
                    flush=True,
                )
                sys_aborted = True
                break

            ret_m = retrieval_metrics(
                pipe_out.selected_event_ids,
                gold.gold_events,
                gold.distractor_events,
                k=k,
            )
            ans_m = evaluate_single_answer(
                parsed_ans,
                gold,
                pipe_out.selected_event_ids,
                pool,
                states_by_id,
            )
            combined_metrics = {**ret_m, **ans_m}

            # Determine active weights for logging
            mode = sys_cfg["systems"].get(sys_id, {}).get("mode", "fusion_b2")
            if mode == "fusion_b2":
                w_used = weights_cfg.get("B2", {}).get(
                    pipe_out.intent, {"alpha": 0.4, "beta": 0.2, "gamma": 0.4}
                )
            elif mode == "fusion_b3":
                w_used = weights_cfg.get("B3", {}).get(
                    pipe_out.intent, {"alpha": 0.6, "beta": 0.4, "gamma": 0.0}
                )
            else:
                w_used = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}

            rec = ResultRecord(
                run_id=f"gen_{q.track}_{q.split}_{sys_id}_{q.question_id}",
                git_commit=git_commit,
                config_hash=cfg_hash,
                system_id=sys_id,
                track=q.track,
                question_id=q.question_id,
                model_id=active_model_id,
                run_date=run_date_str,
                temperature=temperature,
                k=k,
                weights=w_used,
                intent_source=pipe_out.intent_source,
                prompt_hash=prompt_hash,
                raw_output=raw_out,
                parsed_answer=parsed_ans,
                metrics=combined_metrics,
            )
            sys_res_temp.append(rec)

            sys_recs_temp.append(
                {
                    "question_id": q.question_id,
                    "patient_id": q.patient_id,
                    "question_text": q.text,
                    "as_of_date": q.as_of_date,
                    "gold_intent": gold.intent,
                    "predicted_intent": pipe_out.intent,
                    "category": gold.category,
                    "expected_behavior": gold.expected_behavior,
                    "expected_answer": gold.expected_answer.model_dump(mode="json"),
                    "expected_conflict": gold.expected_conflict,
                    "expected_flags": gold.expected_flags,
                    "gold_events": gold.gold_events,
                    "selected_event_ids": pipe_out.selected_event_ids,
                    "context_text": pipe_out.context_text,
                    "parsed_answer": parsed_ans.model_dump(mode="json"),
                    "metrics": combined_metrics,
                    "call_meta": call_meta,
                }
            )

            if done_calls % 10 == 0 or done_calls == total_calls:
                cached_tag = "CACHE" if call_meta.get("cached") else call_meta.get("key_label", "LIVE")
                print(
                    f"[{done_calls}/{total_calls}] {sys_id} | {q.question_id} ({cached_tag}) "
                    f"-> corr={ans_m['answer_correctness']:.2f}, cite_f1={ans_m['citation_f1']:.2f}",
                    flush=True,
                )

        if not sys_aborted:
            records_by_system[sys_id].extend(sys_recs_temp)
            result_records.extend(sys_res_temp)
        else:
            break

    return records_by_system, result_records
