"""Phase 1 runner: Ingests MIMIC-IV-ED demo v2.2, builds Track R events, timelines, and reports."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.events import (
    build_coverage_report,
    build_events,
    build_patient_timelines,
)
from src.data.ingestion import assess_data_quality, load_raw
from src.schemas import save_jsonl


def run_phase1(
    raw_dir: str | Path = "data/raw/mimic_iv_ed_demo",
    processed_dir: str | Path = "data/processed",
    docs_dir: str | Path = "docs",
) -> dict:
    """Execute the full Phase 1 ingestion, event construction, and reporting pipeline."""
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_path = Path(docs_dir)
    doc_path.mkdir(parents=True, exist_ok=True)

    raw = load_raw(raw_dir)
    quality_report = assess_data_quality(raw, raw_dir)
    if not quality_report["validation"]["valid"]:
        raise RuntimeError(
            f"Raw table validation failed: {quality_report['validation']['mismatches']}"
        )

    events = build_events(raw)
    timelines = build_patient_timelines(events)
    coverage_report = build_coverage_report(events)

    save_jsonl(events, out_dir / "events.jsonl")

    with (out_dir / "quality_report.json").open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    with (out_dir / "timelines.json").open("w", encoding="utf-8") as f:
        json.dump(timelines, f, indent=2)

    with (out_dir / "coverage_report.json").open("w", encoding="utf-8") as f:
        json.dump(coverage_report, f, indent=2)

    ls = quality_report["longitudinal_summary"]
    le = coverage_report["longitudinal_evidence_counts"]
    md_lines = [
        "# Phase 1 Data Quality & Real-Data Coverage Audit (Track R)",
        "",
        "## 1. Raw Table Validation (MIMIC-IV-ED Demo v2.2)",
        "",
        "| Table | Expected Rows | Actual Rows | Exact Duplicates | Orphan Stays |",
        "|---|---:|---:|---:|---:|",
    ]
    for tname, exp_cnt in quality_report["validation"]["row_counts"].items():
        dup = quality_report["exact_duplicate_rows"][tname]
        orph = quality_report["relational_integrity"]["orphan_stays"].get(tname, 0)
        md_lines.append(f"| `{tname}` | {exp_cnt} | {exp_cnt} | {dup} | {orph} |")

    md_lines.extend(
        [
            "",
            "## 2. Longitudinal Structure Summary",
            "",
            f"- **Unique patients with ED stays:** {ls['total_patients']}",
            f"- **Total ED visits (`edstays`):** {ls['total_stays']}",
            f"- **Patients with >= 2 visits:** {ls['patients_with_2plus_stays']}",
            f"- **Patients with >= 3 visits:** {ls['patients_with_3plus_stays']}",
            f"- **Maximum visits for a single patient:** {ls['max_stays_per_patient']}",
            f"- **Median time span across multi-visit patients:** {ls['median_multi_visit_span_days']} days",
            f"- **Visits with medication reconciliation (`medrecon`):** {ls['stays_with_medrecon']}",
            f"- **Patients with >= 2 visits having `medrecon`:** {ls['patients_with_2plus_medrecon_visits']}",
            f"- **Multi-visit stays without `medrecon`:** {ls['multi_visit_stays_without_medrecon']}",
            f"- **Patients with both ICD-9 and ICD-10 diagnosis codes:** {ls['patients_with_both_icd9_and_icd10']}",
            "",
            "## 3. Normalised Track R Event Counts (`data/processed/events.jsonl`)",
            "",
            f"- **Total normalised events:** {coverage_report['total_events']}",
            "",
            "| Event Type | Count | Status Basis | Source Table |",
            "|---|---:|---|---|",
            f"| `visit` | {coverage_report['event_type_counts'].get('visit', 0)} | `interval` | `edstays` |",
            f"| `triage_observation` | {coverage_report['event_type_counts'].get('triage_observation', 0)} | `interval` | `triage` |",
            f"| `diagnosis_at_visit` | {coverage_report['event_type_counts'].get('diagnosis_at_visit', 0)} | `interval` | `diagnosis` |",
            f"| `reported_home_medication` | {coverage_report['event_type_counts'].get('reported_home_medication', 0)} | `snapshot` | `medrecon` (intra-stay duplicates collapsed) |",
            f"| `ed_administered_medication` | {coverage_report['event_type_counts'].get('ed_administered_medication', 0)} | `interval` | `pyxis` |",
            f"| `vital_sign` | {coverage_report['event_type_counts'].get('vital_sign', 0)} | `interval` | `vitalsign` |",
            "",
            "## 4. Scenario & Benchmark Category Coverage on Track R",
            "",
            f"- **Consecutive `medrecon` visit pairs:** {le['consecutive_medrecon_visit_pairs']}",
            f"- **Visit pairs with added or dropped home medications:** {le['visit_pairs_with_med_add_or_drop']}",
            f"- **Patients with medication list changes across visits:** {le['patients_with_med_list_changes']}",
            f"- **Patients with recurring diagnoses across visits:** {le['patients_with_recurring_diagnoses']}",
            f"- **Patients with >= 3 vital sign measurements in a visit:** {le['patients_with_3plus_vitals_in_a_visit']}",
            "",
            "| Category | Supported on Track R? | Notes |",
            "|---|---|---|",
        ]
    )
    for cat_key, cat_info in coverage_report["category_support"].items():
        md_lines.append(
            f"| `{cat_key}` | `{cat_info['supported']}` | {cat_info['basis']} |"
        )

    with (doc_path / "data_audit_phase1.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return {
        "events_count": len(events),
        "patients_count": len(timelines),
        "quality_valid": quality_report["validation"]["valid"],
    }


if __name__ == "__main__":
    summary = run_phase1()
    print(json.dumps(summary, indent=2))
