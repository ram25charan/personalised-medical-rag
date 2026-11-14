"""Experiment Comparison & Phase 4 Markdown Report Generator (`experiments/compare.py`) (Sections 8.3, 8.6, 8.7).

Reads `results/metrics/generation_metrics.json`, `results/metrics/manual_review_20.json`,
and (if present) `results/metrics/generation_metrics_secondary.json` and generates
`docs/generation_evaluation_phase4.md` with:
  - Per-system generation metrics + 95% bootstrap CIs (1,000 resamples) on `S_test` and `R_test`
  - Paired bootstrap hypothesis tests (`H1/H2: B2 vs B1`, `H3: B2 vs B4`)
  - Exploratory Analysis E3: 2x2 factorial (`Selection {B1, B2} x Annotation {A0, A2}` = `{B1, B1+, B2-, B2}`)
  - Exploratory Analysis E4: Per-intent and per-category breakdowns
  - Multi-generator robustness check (Groq `openai/gpt-oss-120b` vs Gemini `gemini-2.5-flash`)
  - Stratified 20-response manual review and error taxonomy summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt_ci(m_dict: dict[str, Any]) -> str:
    if not m_dict:
        return "0.0000"
    mean = m_dict.get("mean", 0.0)
    lo = m_dict.get("ci_lower", 0.0)
    hi = m_dict.get("ci_upper", 0.0)
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def _fmt_diff(d_dict: dict[str, Any]) -> str:
    if not d_dict:
        return "+0.0000"
    diff = d_dict.get("diff_mean", 0.0)
    lo = d_dict.get("ci_lower", 0.0)
    hi = d_dict.get("ci_upper", 0.0)
    p = d_dict.get("p_value_two_sided", 1.0)
    return f"`{diff:+.4f}` [{lo:+.4f}, {hi:+.4f}] (p={p:.4f})"


def generate_markdown_report(
    primary_metrics_path: str | Path = "results/metrics/generation_metrics.json",
    secondary_metrics_path: str | Path = "results/metrics/generation_metrics_secondary.json",
    manual_review_path: str | Path = "results/metrics/manual_review_20.json",
    output_md_path: str | Path = "docs/generation_evaluation_phase4.md",
) -> None:
    p_path = Path(primary_metrics_path)
    if not p_path.exists():
        raise FileNotFoundError(f"Primary metrics file not found: {p_path}")

    with p_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    lines: list[str] = [
        "# Phase 4 Generation-Level Evaluation Report",
        "",
        f"- **Git commit:** `{report.get('git_commit', 'unknown')}`",
        f"- **Primary Generator Provider & Model:** `{report.get('provider')}` / `{report.get('model_id')}` (`temperature = 0.0`)",
        f"- **Default context budget:** `k = {report.get('k', 10)}`",
        f"- **Bootstrap resamples:** `1,000` paired resamples (`seed = 42`)",
        "",
    ]

    for split_key, split_title in [
        ("S_test", "Track S Test Set (100 Questions; 86 Answerable, 14 Abstain)"),
        ("R_test", "Track R Real-Data Test Set (30 Questions; 26 Answerable, 4 Abstain) — Sim-to-Real Transfer (E2)"),
    ]:
        sp_data = report.get("splits", {}).get(split_key)
        if not sp_data:
            continue
        sys_map = sp_data.get("systems", {})
        lines.append(f"## {split_title}")
        lines.append("")
        lines.append(
            "| System | Answer Correctness (95% CI) | Temporal Consistency | Contradiction Rate | Citation F1 | Abstention Acc | Conflict Flag Acc | Uncertainty Acc | Faithfulness |"
        )
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        for sys_id in ["B1", "B1+", "B2-", "B4", "B2", "ORACLE"]:
            if sys_id not in sys_map:
                continue
            m = sys_map[sys_id]["metrics"]
            lines.append(
                f"| **{sys_id}** | {_fmt_ci(m.get('answer_correctness'))} | "
                f"{m.get('temporal_consistency', {}).get('mean', 0.0):.4f} | "
                f"{m.get('contradiction_rate', {}).get('mean', 0.0):.4f} | "
                f"{m.get('citation_f1', {}).get('mean', 0.0):.4f} | "
                f"{m.get('abstention_accuracy', {}).get('mean', 0.0):.4f} | "
                f"{m.get('conflict_flag_accuracy', {}).get('mean', 0.0):.4f} | "
                f"{m.get('uncertainty_noted_accuracy', {}).get('mean', 0.0):.4f} | "
                f"{m.get('grounded_faithfulness', {}).get('mean', 0.0):.4f} |"
            )
        lines.append("")

        # Paired Bootstrap Tests
        paired = sp_data.get("paired_hypothesis_tests", {})
        if paired:
            lines.append(f"### Paired Bootstrap Hypothesis Tests (`{split_key}`, 1,000 resamples)")
            lines.append("")
            lines.append(
                "| Comparison | $\\Delta$ Answer Correctness [95% CI] | $\\Delta$ Contradiction Rate [95% CI] | $\\Delta$ Citation F1 [95% CI] | $\\Delta$ Conflict Flag Acc [95% CI] |"
            )
            lines.append("|---|---|---|---|---|")
            for comp_name, comp_vals in paired.items():
                lines.append(
                    f"| `{comp_name}` | {_fmt_diff(comp_vals.get('answer_correctness'))} | "
                    f"{_fmt_diff(comp_vals.get('contradiction_rate'))} | "
                    f"{_fmt_diff(comp_vals.get('citation_f1'))} | "
                    f"{_fmt_diff(comp_vals.get('conflict_flag_accuracy'))} |"
                )
            lines.append("")

        # E3 2x2 Factorial
        fact = sp_data.get("factorial_2x2_E3", {})
        if fact:
            lines.append(
                f"### Exploratory Analysis E3: 2x2 Factorial (`Selection {{B1, B2}} x Annotation {{A0, A2}}` on `{split_key}`)"
            )
            lines.append("")
            lines.append(
                "| Metric | `B1` (Sel=B1, A0) | `B1+` (Sel=B1, A2) | `B2-` (Sel=B2, A0) | `B2` (Sel=B2, A2) | Main Effect: Selection | Main Effect: Annotation | Interaction |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for m_name, f_vals in fact.items():
                lines.append(
                    f"| `{m_name}` | {f_vals['B1_selB1_annA0']:.4f} | {f_vals['B1plus_selB1_annA2']:.4f} | "
                    f"{f_vals['B2minus_selB2_annA0']:.4f} | {f_vals['B2_selB2_annA2']:.4f} | "
                    f"`{f_vals['main_effect_selection']:+.4f}` | `{f_vals['main_effect_annotation']:+.4f}` | "
                    f"`{f_vals['interaction_effect']:+.4f}` |"
                )
            lines.append("")

        # E4 Category breakdown for B1 vs B2
        if "B1" in sys_map and "B2" in sys_map:
            lines.append(f"### Exploratory Analysis E4: Per-Category Breakdown (`B1` vs `B2` on `{split_key}`)")
            lines.append("")
            lines.append(
                "| Category | N | B1 Correctness | B2 Correctness | B1 Contradiction | B2 Contradiction | B1 Citation F1 | B2 Citation F1 |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            b1_cats = sys_map["B1"].get("by_category", {})
            b2_cats = sys_map["B2"].get("by_category", {})
            for cat in sorted(b2_cats.keys()):
                c1 = b1_cats.get(cat, {})
                c2 = b2_cats.get(cat, {})
                lines.append(
                    f"| `{cat}` | {c2.get('count', 0)} | {c1.get('answer_correctness', 0.0):.4f} | "
                    f"{c2.get('answer_correctness', 0.0):.4f} | {c1.get('contradiction_rate', 0.0):.4f} | "
                    f"{c2.get('contradiction_rate', 0.0):.4f} | {c1.get('citation_f1', 0.0):.4f} | "
                    f"{c2.get('citation_f1', 0.0):.4f} |"
                )
            lines.append("")

    # Optional secondary generator comparison (e.g. Gemini Flash or Groq GPT-OSS-120B)
    sec_path = Path(secondary_metrics_path)
    if sec_path.exists():
        with sec_path.open("r", encoding="utf-8") as f:
            sec_rep = json.load(f)
        lines.append(
            f"## Secondary Generator Evaluation (`{sec_rep.get('provider')}` / `{sec_rep.get('model_id')}`)"
        )
        lines.append("")
        for split_key, split_title in [
            ("S_test", "Track S Test Set (100 Questions)"),
            ("R_test", "Track R Real-Data Test Set (30 Questions) — Sim-to-Real Transfer (E2)"),
        ]:
            sp_sec_data = sec_rep.get("splits", {}).get(split_key, {})
            sp_sec = sp_sec_data.get("systems", {})
            if not sp_sec:
                continue
            lines.append(f"### {split_title} (`{sec_rep.get('model_id')}`)")
            lines.append("")
            lines.append(
                "| System | Answer Correctness (95% CI) | Temporal Consistency | Contradiction Rate | Citation F1 | Abstention Acc | Conflict Flag Acc | Uncertainty Acc | Faithfulness |"
            )
            lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
            for sys_id in ["B1", "B1+", "B2-", "B4", "B2", "ORACLE"]:
                if sys_id not in sp_sec:
                    continue
                m = sp_sec[sys_id]["metrics"]
                lines.append(
                    f"| **{sys_id}** | {_fmt_ci(m.get('answer_correctness'))} | "
                    f"{m.get('temporal_consistency', {}).get('mean', 0.0):.4f} | "
                    f"{m.get('contradiction_rate', {}).get('mean', 0.0):.4f} | "
                    f"{m.get('citation_f1', {}).get('mean', 0.0):.4f} | "
                    f"{m.get('abstention_accuracy', {}).get('mean', 0.0):.4f} | "
                    f"{m.get('conflict_flag_accuracy', {}).get('mean', 0.0):.4f} | "
                    f"{m.get('uncertainty_noted_accuracy', {}).get('mean', 0.0):.4f} | "
                    f"{m.get('grounded_faithfulness', {}).get('mean', 0.0):.4f} |"
                )
            lines.append("")
            paired_sec = sp_sec_data.get("paired_hypothesis_tests", {})
            if paired_sec:
                lines.append(f"#### Paired Bootstrap Hypothesis Tests (`{split_key}`, `{sec_rep.get('model_id')}`)")
                lines.append("")
                lines.append(
                    "| Comparison | $\\Delta$ Answer Correctness [95% CI] | $\\Delta$ Contradiction Rate [95% CI] | $\\Delta$ Citation F1 [95% CI] | $\\Delta$ Conflict Flag Acc [95% CI] |"
                )
                lines.append("|---|---|---|---|---|")
                for comp_name, comp_vals in paired_sec.items():
                    lines.append(
                        f"| `{comp_name}` | {_fmt_diff(comp_vals.get('answer_correctness'))} | "
                        f"{_fmt_diff(comp_vals.get('contradiction_rate'))} | "
                        f"{_fmt_diff(comp_vals.get('citation_f1'))} | "
                        f"{_fmt_diff(comp_vals.get('conflict_flag_accuracy'))} |"
                    )
                lines.append("")
            fact_sec = sp_sec_data.get("factorial_2x2_E3", {})
            if fact_sec:
                lines.append(
                    f"#### Exploratory Analysis E3: 2x2 Factorial (`{split_key}`, `{sec_rep.get('model_id')}`)"
                )
                lines.append("")
                lines.append(
                    "| Metric | `B1` (Sel=B1, A0) | `B1+` (Sel=B1, A2) | `B2-` (Sel=B2, A0) | `B2` (Sel=B2, A2) | Main Effect: Selection | Main Effect: Annotation | Interaction |"
                )
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
                for m_name, f_vals in fact_sec.items():
                    lines.append(
                        f"| `{m_name}` | {f_vals['B1_selB1_annA0']:.4f} | {f_vals['B1plus_selB1_annA2']:.4f} | "
                        f"{f_vals['B2minus_selB2_annA0']:.4f} | {f_vals['B2_selB2_annA2']:.4f} | "
                        f"`{f_vals['main_effect_selection']:+.4f}` | `{f_vals['main_effect_annotation']:+.4f}` | "
                        f"`{f_vals['interaction_effect']:+.4f}` |"
                    )
                lines.append("")

    # Manual review summary
    rev_path = Path(manual_review_path)
    if rev_path.exists():
        with rev_path.open("r", encoding="utf-8") as f:
            rev = json.load(f)
        items = rev.get("items", [])
        lines.append(f"## Stratified Manual Faithfulness & Error Analysis ({len(items)} Responses, Section 8.6)")
        lines.append("")
        lines.append(
            "| Question ID | Category | Expected Answer | B1 Correct | B2 Correct | B1 Faithfulness | B2 Faithfulness |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for it in items:
            m1 = it.get("B1", {}).get("metrics", {})
            m2 = it.get("B2", {}).get("metrics", {})
            exp_v = it.get("expected_answer", {}).get("value")
            lines.append(
                f"| `{it['question_id']}` | `{it['category']}` | `{exp_v}` | "
                f"{m1.get('answer_correctness', 0.0):.2f} | {m2.get('answer_correctness', 0.0):.2f} | "
                f"{m1.get('grounded_faithfulness', 0.0):.2f} | {m2.get('grounded_faithfulness', 0.0):.2f} |"
            )
        lines.append("")

    out_path = Path(output_md_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated Phase 4 Markdown report at {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generation runs and build Phase 4 report.")
    parser.add_argument("--primary", type=str, default="results/metrics/generation_metrics.json")
    parser.add_argument(
        "--secondary", type=str, default="results/metrics/generation_metrics_secondary.json"
    )
    parser.add_argument("--manual-review", type=str, default="results/metrics/manual_review_20.json")
    parser.add_argument("--output", type=str, default="docs/generation_evaluation_phase4.md")
    args = parser.parse_args()
    generate_markdown_report(
        primary_metrics_path=args.primary,
        secondary_metrics_path=args.secondary,
        manual_review_path=args.manual_review,
        output_md_path=args.output,
    )


if __name__ == "__main__":
    main()
