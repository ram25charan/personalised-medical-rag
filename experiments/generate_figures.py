"""Publication Figure Generator (`experiments/generate_figures.py`) (Section 8.9 & Phase 6).

Generates 300-DPI IEEE-ready PNG figures in `results/figures/` directly from measured
experiment outputs (`results/metrics/retrieval_metrics.json`, `weight_sensitivity_dev.json`,
`generation_metrics.json`, and `generation_metrics_gemini.json`):
  1. `fig1_budget_curve_E1.png`       — Context budget study (k in {5, 10, 20}) on S_test & R_test
  2. `fig2_weight_sensitivity.png`    — Dev-set sensitivity of nDCG@10 to temporal weight gamma by intent
  3. `fig3_generation_comparison.png` — Multi-generator system comparison (Answer Correctness + Contradiction Rate)
  4. `fig4_category_breakdown_E4.png` — Per-category B1 vs B2 breakdown on S_test (7 challenge categories)
  5. `fig5_factorial_E3.png`          — 2x2 Selection x Annotation factorial interaction plots
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

OUT_DIR = Path("results/figures")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_budget_curve_e1(ret_metrics: dict[str, Any], out_dir: Path) -> Path:
    """Figure 1: Context-budget study (k in {5, 10, 20}) across S_test and R_test."""
    budget = ret_metrics["budget_study_E1"]
    ks = [5, 10, 20]
    systems = [
        ("B2", "B2 (Rq+Rh+Rt)", "#0F62FE", "-o", 2.3),
        ("B3", "B3 (Rq+Rh)", "#198038", "--s", 1.8),
        ("B1", "B1 (Hybrid Rq)", "#525252", "-.^", 1.8),
        ("RS5", "RS5 (Status Filter)", "#8A3FFC", ":d", 1.6),
        ("RS4", "RS4 (Pure Recency)", "#DA1E28", ":x", 1.5),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, (sp_key, sp_title) in zip(
        axes, [("S_test", "Track S Test (100 Questions)"), ("R_test", "Track R Test (30 Questions)")]
    ):
        sp_data = budget[sp_key]
        for sys_id, label, color, fmt, lw in systems:
            vals = [sp_data[sys_id][f"k={k}"]["ndcg_at_k"] for k in ks]
            ax.plot(ks, vals, fmt, label=label, color=color, linewidth=lw, markersize=5.5)
        ax.set_title(sp_title)
        ax.set_xlabel("Context Budget (top-k events)")
        ax.set_xticks(ks)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_ylabel("nDCG@k")
    axes[1].legend(loc="lower right", frameon=True, framealpha=0.92)
    fig.suptitle("Context Budget Sensitivity (E1): Retrieval nDCG@k across k ∈ {5, 10, 20}", y=1.02, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "fig1_budget_curve_E1.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_weight_sensitivity(ws_data: dict[str, Any], out_dir: Path) -> Path:
    """Figure 2: Mean dev-set nDCG@10 as a function of temporal weight gamma per intent."""
    intents = [
        ("current_state", "current_state", "#0F62FE", "-o"),
        ("historical", "historical", "#198038", "--s"),
        ("change_over_time", "change_over_time", "#8A3FFC", "-.^"),
        ("first_occurrence", "first_occurrence", "#B28600", "-d"),
        ("persistent_fact", "persistent_fact", "#DA1E28", ":v"),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for intent_key, label, color, fmt in intents:
        grid = ws_data["B2_by_intent"][intent_key]["all_grid"]
        by_gamma: dict[float, list[float]] = defaultdict(list)
        for row in grid:
            g = round(float(row["weights"]["gamma"]), 2)
            by_gamma[g].append(float(row["mean_ndcg_at_10"]))
        gammas = sorted(by_gamma.keys())
        max_ndcg = [max(by_gamma[g]) for g in gammas]
        ax.plot(gammas, max_ndcg, fmt, label=label, color=color, linewidth=1.9, markersize=5)

    ax.set_title("Dev-Set Weight Sensitivity: Best nDCG@10 vs. Temporal Weight (γ) by Intent")
    ax.set_xlabel("Temporal Relevance Weight (γ = w_t)")
    ax.set_ylabel("Best Dev nDCG@10 (over valid α, β)")
    ax.set_ylim(0.50, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, ncol=2)
    fig.tight_layout()
    out_path = out_dir / "fig2_weight_sensitivity.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_generation_comparison(
    groq_metrics: dict[str, Any],
    gemini_metrics: dict[str, Any],
    out_dir: Path,
) -> Path:
    """Figure 3: Multi-generator comparison of Answer Correctness (95% CI) and Contradiction Rate."""
    sys_order = ["B1", "B1+", "B2-", "B4", "B2"]
    panels = [
        ("Groq openai/gpt-oss-120b (S_test, n=100)", groq_metrics["splits"]["S_test"]["systems"]),
        ("Gemini gemini-flash-lite-latest (S_test, n=100)", gemini_metrics["splits"]["S_test"]["systems"]),
        ("Gemini gemini-flash-lite-latest (R_test, n=30)", gemini_metrics["splits"]["R_test"]["systems"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.6), sharex=True)
    x = np.arange(len(sys_order))
    colors = ["#8D8D8D", "#6FDC8C", "#78A9FF", "#4589FF", "#0F62FE"]

    for col_idx, (title, sys_map) in enumerate(panels):
        # Row 0: Answer Correctness with 95% CI error bars
        ax_top = axes[0, col_idx]
        means = [sys_map[s]["metrics"]["answer_correctness"]["mean"] for s in sys_order]
        err_lo = [m - sys_map[s]["metrics"]["answer_correctness"]["ci_lower"] for s, m in zip(sys_order, means)]
        err_hi = [sys_map[s]["metrics"]["answer_correctness"]["ci_upper"] - m for s, m in zip(sys_order, means)]
        bars = ax_top.bar(x, means, yerr=[err_lo, err_hi], capsize=3.5, color=colors, edgecolor="#161616", linewidth=0.6)
        ax_top.set_title(title, fontsize=9.5)
        ax_top.set_ylim(0.45, 1.09)
        ax_top.grid(True, axis="y", linestyle="--", alpha=0.35)
        for b, m, hi in zip(bars, means, err_hi):
            ax_top.text(b.get_x() + b.get_width() / 2, m + hi + 0.014, f"{m:.2f}", ha="center", va="bottom", fontsize=7.8)

        # Row 1: Contradiction Rate (lower is better)
        ax_bot = axes[1, col_idx]
        c_rates = [sys_map[s]["metrics"]["contradiction_rate"]["mean"] for s in sys_order]
        bars_c = ax_bot.bar(x, c_rates, color=colors, edgecolor="#161616", linewidth=0.6)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(sys_order)
        ax_bot.set_ylim(0.0, 0.18)
        ax_bot.grid(True, axis="y", linestyle="--", alpha=0.35)
        for b, c in zip(bars_c, c_rates):
            ax_bot.text(b.get_x() + b.get_width() / 2, c + 0.005, f"{c:.2f}", ha="center", va="bottom", fontsize=7.8)

    axes[0, 0].set_ylabel("Answer Correctness (↑)")
    axes[1, 0].set_ylabel("Contradiction Rate (↓)")
    fig.suptitle("Generation-Level Performance Across Systems and Models (95% Bootstrap CIs)", y=1.01, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "fig3_generation_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_category_breakdown_e4(groq_metrics: dict[str, Any], out_dir: Path) -> Path:
    """Figure 4: Per-category Answer Correctness and Contradiction Rate (B1 vs B2, S_test)."""
    sys_map = groq_metrics["splits"]["S_test"]["systems"]
    b1_cats = sys_map["B1"]["by_category"]
    b2_cats = sys_map["B2"]["by_category"]
    cats = sorted(b2_cats.keys())
    short_labels = [c.replace("_", "\n") for c in cats]

    x = np.arange(len(cats))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

    b1_corr = [b1_cats[c]["answer_correctness"] for c in cats]
    b2_corr = [b2_cats[c]["answer_correctness"] for c in cats]
    axes[0].bar(x - width / 2, b1_corr, width, label="B1 (Conventional RAG)", color="#A8B2BD", edgecolor="#161616", linewidth=0.6)
    axes[0].bar(x + width / 2, b2_corr, width, label="B2 (Chronology-Aware)", color="#0F62FE", edgecolor="#161616", linewidth=0.6)
    axes[0].set_title("Answer Correctness by Category (↑)")
    axes[0].set_ylabel("Mean Score")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels, fontsize=7.5)
    axes[0].set_ylim(0.0, 1.12)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(loc="lower right", frameon=True)

    b1_cont = [b1_cats[c]["contradiction_rate"] for c in cats]
    b2_cont = [b2_cats[c]["contradiction_rate"] for c in cats]
    axes[1].bar(x - width / 2, b1_cont, width, label="B1 (Conventional RAG)", color="#A8B2BD", edgecolor="#161616", linewidth=0.6)
    axes[1].bar(x + width / 2, b2_cont, width, label="B2 (Chronology-Aware)", color="#DA1E28", edgecolor="#161616", linewidth=0.6)
    axes[1].set_title("Contradiction Rate by Category (↓)")
    axes[1].set_ylabel("Mean Rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, fontsize=7.5)
    axes[1].set_ylim(0.0, 0.72)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(loc="upper right", frameon=True)

    fig.suptitle("Exploratory Category Analysis (E4): Groq openai/gpt-oss-120b on S_test (n=100)", y=1.02, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "fig4_category_breakdown_E4.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_factorial_e3(groq_metrics: dict[str, Any], out_dir: Path) -> Path:
    """Figure 5: 2x2 Factorial (Selection {B1, B2} x Annotation {A0, A2}) interaction plots."""
    fact = groq_metrics["splits"]["S_test"]["factorial_2x2_E3"]
    metrics_to_plot = [
        ("answer_correctness", "Answer Correctness (↑)", (0.78, 0.94)),
        ("contradiction_rate", "Contradiction Rate (↓)", (0.00, 0.13)),
        ("uncertainty_noted_accuracy", "Uncertainty Flag Accuracy (↑)", (0.60, 1.02)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
    for ax, (m_key, title, ylim) in zip(axes, metrics_to_plot):
        fv = fact[m_key]
        # Selection = B1: A0 -> A2; Selection = B2: A0 -> A2
        ax.plot([0, 1], [fv["B1_selB1_annA0"], fv["B1plus_selB1_annA2"]], "--o", color="#525252", label="Selection = B1", linewidth=1.9)
        ax.plot([0, 1], [fv["B2minus_selB2_annA0"], fv["B2_selB2_annA2"]], "-s", color="#0F62FE", label="Selection = B2", linewidth=2.2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["A0 (Raw)", "A2 (State+Flags)"])
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.annotate(
            f"Sel: {fv['main_effect_selection']:+.3f}\nAnn: {fv['main_effect_annotation']:+.3f}",
            xy=(0.05, 0.82),
            xycoords="axes fraction",
            fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F4F6F8", edgecolor="#D0D7DE"),
        )

    axes[0].legend(loc="lower right", frameon=True)
    fig.suptitle("2×2 Factorial Decomposition (E3): Selection × Annotation on S_test (Groq)", y=1.03, fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / "fig5_factorial_E3.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ret_metrics = _load_json("results/metrics/retrieval_metrics.json")
    ws_data = _load_json("results/metrics/weight_sensitivity_dev.json")
    groq_metrics = _load_json("results/metrics/generation_metrics.json")
    gemini_metrics = _load_json("results/metrics/generation_metrics_gemini.json")

    paths = [
        plot_budget_curve_e1(ret_metrics, OUT_DIR),
        plot_weight_sensitivity(ws_data, OUT_DIR),
        plot_generation_comparison(groq_metrics, gemini_metrics, OUT_DIR),
        plot_category_breakdown_e4(groq_metrics, OUT_DIR),
        plot_factorial_e3(groq_metrics, OUT_DIR),
    ]
    for p in paths:
        print(f"Generated {p}")


if __name__ == "__main__":
    main()
