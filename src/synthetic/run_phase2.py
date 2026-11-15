"""Phase 2 runner: Generates Track S synthetic patients/events and Track S + Track R benchmark questions and gold."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.schemas import Event, load_jsonl, save_jsonl
from src.synthetic.generator import generate


def run_phase2(
    config_path: str | Path = "configs/synthetic_generator.json",
    track_r_events_path: str | Path = "data/processed/events.jsonl",
    synthetic_dir: str | Path = "data/synthetic",
    benchmark_dir: str | Path = "data/benchmark",
    docs_dir: str | Path = "docs",
) -> dict:
    """Execute Phase 2 synthetic patient generation and benchmark construction."""
    syn_out = Path(synthetic_dir)
    syn_out.mkdir(parents=True, exist_ok=True)
    bench_out = Path(benchmark_dir)
    bench_out.mkdir(parents=True, exist_ok=True)
    doc_out = Path(docs_dir)
    doc_out.mkdir(parents=True, exist_ok=True)

    track_r_events = load_jsonl(Event, track_r_events_path)
    s_events, questions, golds = generate(
        config_path=config_path,
        seed=None,
        track_r_events=track_r_events,
    )

    save_jsonl(s_events, syn_out / "events.jsonl")
    save_jsonl(questions, bench_out / "questions.jsonl")
    save_jsonl(golds, bench_out / "gold.jsonl")

    # Compute summary statistics
    s_patients = sorted({e.patient_id for e in s_events})
    dev_patients = sorted({q.patient_id for q in questions if q.track == "S" and q.split == "dev"})
    test_s_patients = sorted({q.patient_id for q in questions if q.track == "S" and q.split == "test"})

    q_by_id = {q.question_id: q for q in questions}
    cat_counts_s_dev = Counter(
        g.category for g in golds if q_by_id[g.question_id].track == "S" and q_by_id[g.question_id].split == "dev"
    )
    cat_counts_s_test = Counter(
        g.category for g in golds if q_by_id[g.question_id].track == "S" and q_by_id[g.question_id].split == "test"
    )
    cat_counts_r_test = Counter(
        g.category for g in golds if q_by_id[g.question_id].track == "R"
    )
    intent_counts = Counter(g.intent for g in golds)

    summary = {
        "schema_version": "1.0",
        "synthetic_patients_count": len(s_patients),
        "synthetic_events_count": len(s_events),
        "avg_events_per_synthetic_patient": round(len(s_events) / max(1, len(s_patients)), 2),
        "disjoint_patient_splits": len(set(dev_patients) & set(test_s_patients)) == 0,
        "questions_total": len(questions),
        "questions_by_track_and_split": {
            "S_dev": sum(1 for q in questions if q.track == "S" and q.split == "dev"),
            "S_test": sum(1 for q in questions if q.track == "S" and q.split == "test"),
            "R_test": sum(1 for q in questions if q.track == "R" and q.split == "test"),
        },
        "categories_S_dev": dict(cat_counts_s_dev),
        "categories_S_test": dict(cat_counts_s_test),
        "categories_R_test": dict(cat_counts_r_test),
        "intents_total": dict(intent_counts),
    }

    with (bench_out / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write generator & benchmark documentation in docs/synthetic_and_benchmark_phase2.md
    md = [
        "# Phase 2 Synthetic Patient Generator & Benchmark Specification",
        "",
        "## 1. Synthetic Longitudinal Patient Dataset (Track S)",
        "",
        f"- **Generator version:** `1.0` (`configs/synthetic_generator.json`, seed `42`)",
        f"- **Total synthetic patients:** {summary['synthetic_patients_count']} (`S_001` - `S_100`)",
        f"- **Total synthetic events (`data/synthetic/events.jsonl`):** {summary['synthetic_events_count']} (mean {summary['avg_events_per_synthetic_patient']} events/patient)",
        f"- **Patient split:** 34 dev patients (`S_001` - `S_034`), 66 test patients (`S_035` - `S_100`). Disjoint check: `{summary['disjoint_patient_splits']}`",
        "",
        "### Embedded Clinical Scenarios per Patient",
        "1. **Persistent Condition & Allergy:** Documented drug/material allergy (`allergy`) and chronic hypertension (`condition`) with initial onset and follow-up visits.",
        "2. **Medication Lifecycle:** Initial antihypertensive prescription (`medication_order`), subsequent dose escalation, discontinuation, and replacement with a new active agent, plus an active glycemic medication with open end date (`U1`).",
        "3. **Longitudinal Lab Trend:** 4 sequential lab measurements (`lab_result` for HbA1c, LDL, serum creatinine, or TSH) across ~2.5 years, plus 1 post-`as_of_date` future measurement to verify the `observed_time <= as_of_date` availability filter.",
        "4. **Missing & Ambiguous Dates:** Statin prescription with month-only precision (`U2`) and historical GERD condition with missing onset date (`U3`).",
        "5. **Conflicting Records:** Alternating `C1` status conflict (outpatient active vs external pharmacy ended on overlapping interval) and `C2` dose conflict (primary care vs specialist conflicting active dose).",
        "6. **Irrelevant History / Distractors:** 45-165 routine outpatient visits, vital signs, and short-course unrelated medications per patient.",
        "",
        "## 2. Benchmark Questions & Programmatic Gold (`data/benchmark/`)",
        "",
        f"- **Total questions (`questions.jsonl` & `gold.jsonl`):** {summary['questions_total']}",
        f"  - **Track S Dev:** {summary['questions_by_track_and_split']['S_dev']}",
        f"  - **Track S Test:** {summary['questions_by_track_and_split']['S_test']}",
        f"  - **Track R Test:** {summary['questions_by_track_and_split']['R_test']}",
        "",
        "### Category Breakdown",
        "| Category | Track S Dev | Track S Test | Track R Test |",
        "|---|---:|---:|---:|",
    ]
    for cat in [
        "temporal_ordering",
        "current_vs_historical",
        "longitudinal_change",
        "irrelevant_history",
        "missing_or_ambiguous_dates",
        "conflicting_information",
        "insufficient_evidence",
    ]:
        md.append(
            f"| `{cat}` | {cat_counts_s_dev.get(cat, 0)} | {cat_counts_s_test.get(cat, 0)} | {cat_counts_r_test.get(cat, 0)} |"
        )

    md.extend(
        [
            "",
            "### Question Intent Breakdown (All Tracks)",
            "| Intent | Count |",
            "|---|---:|",
        ]
    )
    for intent_name, cnt in sorted(intent_counts.items()):
        md.append(f"| `{intent_name}` | {cnt} |")

    with (doc_out / "synthetic_and_benchmark_phase2.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    return summary


if __name__ == "__main__":
    res = run_phase2()
    print(json.dumps(res, indent=2))
