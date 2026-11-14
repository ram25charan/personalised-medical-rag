"""Generation-Level Evaluation Runner (`experiments/run_generation.py`) (Sections 8.3, 8.6, 8.7).

Runs the 6 generation-level configurations in priority order (`B1`, `B2`, `B1+`, `B2-`, `B4`, `ORACLE`)
on the benchmark splits using a single pinned `model_id` per experiment table (with automatic
intra-provider API key rotation across the user's configured keys and persistent disk caching).

Also generates:
  - `results/metrics/generation_metrics.json`: Per-system bootstrap means, 95% CIs (1,000 resamples),
    paired hypothesis tests (`H1/H2: B2 vs B1`, `H3: B2 vs B4`, `E3: 2x2 factorial {B1, B1+, B2-, B2}`),
    and per-intent / per-category breakdowns (`E4`).
  - `results/outputs/generation_runs.jsonl`: All validated `ResultRecord` entries.
  - `results/metrics/manual_review_20.json`: Stratified manual error analysis & faithfulness audit
    sample of 20 responses (`B1` vs `B2`) per Section 8.6.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.runner import (
    GENERATION_SYSTEMS_PRIORITY,
    get_git_commit,
    run_generation_evaluation_on_questions,
    summarize_generation_split,
)
from src.schemas import Event, Gold, Question, ResultRecord, load_jsonl, save_jsonl


def build_manual_review_sample(
    records_by_system: dict[str, list[dict[str, Any]]],
    sample_size: int = 20,
) -> dict[str, Any]:
    """Select a stratified 20-question sample across all 7 clinical categories from Track S test
    comparing B1 and B2 side-by-side for manual faithfulness and error taxonomy review (Section 8.6).
    """
    b2_recs = {r["question_id"]: r for r in records_by_system.get("B2", [])}
    b1_recs = {r["question_id"]: r for r in records_by_system.get("B1", [])}
    if not b2_recs:
        return {"sample_size": 0, "items": []}

    by_cat: dict[str, list[str]] = defaultdict(list)
    for qid, rec in sorted(b2_recs.items()):
        by_cat[rec["category"]].append(qid)

    selected_qids: list[str] = []
    # Round-robin across categories until we reach sample_size (20)
    cats = sorted(by_cat.keys())
    idx = 0
    while len(selected_qids) < sample_size:
        added_any = False
        for c in cats:
            if idx < len(by_cat[c]) and len(selected_qids) < sample_size:
                selected_qids.append(by_cat[c][idx])
                added_any = True
        if not added_any:
            break
        idx += 1

    items: list[dict[str, Any]] = []
    for qid in selected_qids:
        r2 = b2_recs[qid]
        r1 = b1_recs.get(qid, {})
        items.append(
            {
                "question_id": qid,
                "patient_id": r2["patient_id"],
                "category": r2["category"],
                "intent": r2["gold_intent"],
                "question_text": r2["question_text"],
                "as_of_date": r2["as_of_date"],
                "gold_events": r2["gold_events"],
                "expected_answer": r2["expected_answer"],
                "expected_behavior": r2["expected_behavior"],
                "B1": {
                    "selected_event_ids": r1.get("selected_event_ids", []),
                    "parsed_answer": r1.get("parsed_answer", {}),
                    "metrics": r1.get("metrics", {}),
                },
                "B2": {
                    "selected_event_ids": r2.get("selected_event_ids", []),
                    "parsed_answer": r2.get("parsed_answer", {}),
                    "metrics": r2.get("metrics", {}),
                },
            }
        )

    return {
        "sample_size": len(items),
        "description": "Stratified 20-question manual review sample across all 7 categories (Section 8.6).",
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 Generation Evaluation.")
    parser.add_argument("--provider", type=str, default="groq", choices=["groq", "gemini"])
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument(
        "--systems",
        type=str,
        default=",".join(GENERATION_SYSTEMS_PRIORITY),
        help="Comma-separated list of systems to evaluate.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="S_test,R_test",
        help="Comma-separated splits to evaluate (e.g. S_dev,S_test,R_test).",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.8)
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="results/metrics/generation_metrics.json",
    )
    parser.add_argument(
        "--output-runs",
        type=str,
        default="results/outputs/generation_runs.jsonl",
    )
    parser.add_argument(
        "--manual-review-out",
        type=str,
        default="results/metrics/manual_review_20.json",
    )
    args = parser.parse_args()

    system_ids = [s.strip() for s in args.systems.split(",") if s.strip()]
    splits_requested = [s.strip() for s in args.splits.split(",") if s.strip()]

    questions = load_jsonl(Question, "data/benchmark/questions.jsonl")
    golds = load_jsonl(Gold, "data/benchmark/gold.jsonl")
    gold_by_qid = {g.question_id: g for g in golds}

    s_events = load_jsonl(Event, "data/synthetic/events.jsonl")
    r_events = load_jsonl(Event, "data/processed/events.jsonl")
    events_by_patient: dict[str, list[Event]] = defaultdict(list)
    for ev in s_events:
        events_by_patient[ev.patient_id].append(ev)
    for ev in r_events:
        events_by_patient[ev.patient_id].append(ev)

    split_map = {
        "S_dev": [q for q in questions if q.track == "S" and q.split == "dev"],
        "S_test": [q for q in questions if q.track == "S" and q.split == "test"],
        "R_test": [q for q in questions if q.track == "R" and q.split == "test"],
    }

    all_result_records: list[ResultRecord] = []
    full_metrics_report: dict[str, Any] = {
        "git_commit": get_git_commit(),
        "provider": args.provider,
        "model_id": args.model_id,
        "k": args.k,
        "systems_evaluated": system_ids,
        "splits": {},
    }

    s_test_records_by_sys: dict[str, list[dict[str, Any]]] = {}

    for split_name in splits_requested:
        q_list = split_map.get(split_name, [])
        if not q_list:
            continue
        print(
            f"\n=== Running Generation Evaluation on {split_name} ({len(q_list)} questions, systems={system_ids}) ===",
            flush=True,
        )
        recs_by_sys, res_records = run_generation_evaluation_on_questions(
            questions=q_list,
            gold_by_qid=gold_by_qid,
            events_by_patient=events_by_patient,
            system_ids=system_ids,
            provider=args.provider,
            model_id=args.model_id,
            k=args.k,
            min_request_interval_sec=args.interval,
        )
        all_result_records.extend(res_records)
        if res_records and not full_metrics_report["model_id"]:
            full_metrics_report["model_id"] = res_records[0].model_id

        full_metrics_report["splits"][split_name] = summarize_generation_split(
            recs_by_sys, n_resamples=1000
        )
        if split_name == "S_test":
            s_test_records_by_sys = recs_by_sys

    out_metrics_path = Path(args.output_metrics)
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with out_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(full_metrics_report, f, indent=2, ensure_ascii=False)
    print(f"Saved generation metrics report to {out_metrics_path}")

    save_jsonl(all_result_records, args.output_runs)
    print(f"Saved {len(all_result_records)} generation ResultRecords to {args.output_runs}")

    if s_test_records_by_sys and args.manual_review_out:
        review_sample = build_manual_review_sample(s_test_records_by_sys, sample_size=20)
        review_path = Path(args.manual_review_out)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with review_path.open("w", encoding="utf-8") as f:
            json.dump(review_sample, f, indent=2, ensure_ascii=False)
        print(f"Saved 20-question manual review sample to {review_path}")


if __name__ == "__main__":
    main()
