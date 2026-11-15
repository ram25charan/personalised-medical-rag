"""Single-Page Clinical Record Viewer & Chronology-Aware RAG Interface (`app/main.py`) (Section 9 & Phase 5 Plan).

Design principles enforced:
  - Record viewer, not a chatbot (no avatars, no chat bubbles, no gradients, no emojis as icons).
  - Restrained clinical design system: sans-serif body + monospace for IDs (`[E1]`, `S_001_0012`), dates, and scores.
  - Verbatim pipeline status and flag names (`ACTIVE`, `ENDED`, `SUPERSEDED`, `PERSISTENT`, `UNKNOWN`,
    `LISTED`, `NOT_LISTED_LATEST`; `C1`–`C3`; `U1`–`U5`) with consistent colour + symbol labels across
    timeline, badges, and evidence records.
  - Progressive disclosure: Patient, `as_of_date`, System (`B2`, `B1`, `Compare B1 vs B2`), View Mode (`Patient` / `Clinician`),
    and Question up front; Provider/Model, `k`, `window_start`, and Track R axis mode in an "Advanced" expander.
  - Single pinned model per provider (no multi-key rotation in UI), disk cache first, and Gemini blocked on Track R.
  - Immediate zero-LLM selection diff for `Compare B1 vs B2`, with on-demand answer generation per system.
  - Separate General Information box (`[C1]`, `[C2]`) with relevance threshold and strict separation check.
"""

from __future__ import annotations

import html
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.generation.cache import load_cached_response, save_cached_response
from src.generation.groq_client import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    _call_gemini_http,
    _call_groq_http,
    load_api_keys,
    parse_and_validate_answer_json,
)
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt, compute_prompt_hash
from src.pipeline import load_json_config, run_retrieval_stage
from src.retrieval.corpus_retrieval import (
    assert_no_corpus_citation_in_claims,
    retrieve_corpus_passages,
)
from src.schemas import AnswerJSON, Event, Gold, PipelineOutput, Question, load_jsonl
from src.temporal.history_relevance import load_relation_table
from src.temporal.state import compute_state, filter_candidate_pool
from src.temporal.temporal_scoring import load_rt_rules

# ---------------------------------------------------------------------------
# Design System: Verbatim Status Styling (Colour + Accessible Pattern Prefix)
# ---------------------------------------------------------------------------
STATUS_STYLE_MAP: dict[str, dict[str, str]] = {
    "ACTIVE": {
        "bg": "#DEFBE6",
        "fg": "#0E6027",
        "border": "#24A148",
        "plot_color": "#198038",
        "symbol": "●",
        "opacity": "1.0",
    },
    "PERSISTENT": {
        "bg": "#E0F2FE",
        "fg": "#075985",
        "border": "#0284C7",
        "plot_color": "#0284C7",
        "symbol": "◆",
        "opacity": "1.0",
    },
    "LISTED": {
        "bg": "#DEFBE6",
        "fg": "#0E6027",
        "border": "#24A148",
        "plot_color": "#198038",
        "symbol": "●",
        "opacity": "1.0",
    },
    "ENDED": {
        "bg": "#EBEBEB",
        "fg": "#393939",
        "border": "#8D8D8D",
        "plot_color": "#8D8D8D",
        "symbol": "○",
        "opacity": "0.58",
    },
    "NOT_LISTED_LATEST": {
        "bg": "#EBEBEB",
        "fg": "#393939",
        "border": "#8D8D8D",
        "plot_color": "#8D8D8D",
        "symbol": "○",
        "opacity": "0.58",
    },
    "SUPERSEDED": {
        "bg": "#FDF6DD",
        "fg": "#684E00",
        "border": "#D2A106",
        "plot_color": "#B28600",
        "symbol": "⊘",
        "opacity": "0.62",
    },
    "UNKNOWN": {
        "bg": "#F6F2FF",
        "fg": "#491D8B",
        "border": "#8A3FFC",
        "plot_color": "#8A3FFC",
        "symbol": "?",
        "opacity": "0.75",
    },
}

ACCENT_BLUE = "#0F62FE"

EVENT_LANE_ORDER = [
    "medication_order",
    "reported_home_medication",
    "ed_administered_medication",
    "condition",
    "diagnosis_at_visit",
    "lab_result",
    "allergy",
    "vital_sign",
    "triage_observation",
    "visit",
]


def get_status_style(state_str: str) -> dict[str, str]:
    return STATUS_STYLE_MAP.get(
        state_str,
        {
            "bg": "#F4F6F8",
            "fg": "#161616",
            "border": "#8D8D8D",
            "plot_color": "#6F6F6F",
            "symbol": "▪",
            "opacity": "0.8",
        },
    )


def render_status_badge_html(state_str: str) -> str:
    st_info = get_status_style(state_str)
    safe_state = html.escape(state_str)
    return (
        f'<span class="status-badge" style="background:{st_info["bg"]};color:{st_info["fg"]};'
        f'border:1px solid {st_info["border"]};">'
        f'{st_info["symbol"]} {safe_state}</span>'
    )


def render_provenance_badge_html(track: str, source_dataset: str) -> str:
    if track == "R":
        return (
            '<span class="prov-badge prov-real">'
            "REAL RECORD &middot; MIMIC-IV-ED v2.2 (DE-IDENTIFIED, SHIFTED DATES)</span>"
        )
    return (
        '<span class="prov-badge prov-synth">'
        f"SYNTHETIC RECORD &middot; TRACK S ({html.escape(source_dataset)})</span>"
    )


def inject_design_system_css(view_mode: str) -> None:
    body_size = "1.04rem" if view_mode == "Patient" else "0.94rem"
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }}
        .prototype-banner {{
            background-color: #F4F6F8;
            border-left: 4px solid #161616;
            border-top: 1px solid #E0E6ED;
            border-right: 1px solid #E0E6ED;
            border-bottom: 1px solid #E0E6ED;
            padding: 8px 14px;
            font-size: 0.84rem;
            color: #161616;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .mono-id {{
            font-family: "IBM Plex Mono", "Consolas", "SFMono-Regular", monospace;
            font-size: 0.86em;
            background: #F4F6F8;
            border: 1px solid #D0D7DE;
            padding: 1px 6px;
            color: #161616;
        }}
        .cite-tag {{
            font-family: "IBM Plex Mono", "Consolas", monospace;
            font-size: 0.84em;
            font-weight: 600;
            background: #EDF5FF;
            color: #0043CE;
            border: 1px solid #0F62FE;
            padding: 1px 6px;
            margin-right: 4px;
        }}
        .corpus-tag {{
            font-family: "IBM Plex Mono", "Consolas", monospace;
            font-size: 0.84em;
            font-weight: 600;
            background: #F4F6F8;
            color: #393939;
            border: 1px solid #8D8D8D;
            padding: 1px 6px;
            margin-right: 4px;
        }}
        .status-badge {{
            font-family: "IBM Plex Mono", "Consolas", monospace;
            font-size: 0.78rem;
            font-weight: 600;
            padding: 2px 8px;
            display: inline-block;
            margin-right: 6px;
            letter-spacing: 0.01em;
        }}
        .flag-badge-conflict {{
            font-family: "IBM Plex Mono", "Consolas", monospace;
            font-size: 0.76rem;
            font-weight: 600;
            background: #FFF1F1;
            color: #A2191F;
            border: 1px solid #DA1E28;
            padding: 2px 7px;
            display: inline-block;
            margin-right: 6px;
        }}
        .flag-badge-unc {{
            font-family: "IBM Plex Mono", "Consolas", monospace;
            font-size: 0.76rem;
            font-weight: 600;
            background: #FDF6DD;
            color: #684E00;
            border: 1px solid #D2A106;
            padding: 2px 7px;
            display: inline-block;
            margin-right: 6px;
        }}
        .diff-badge-b2 {{
            font-family: "IBM Plex Mono", "Consolas", monospace;
            font-size: 0.75rem;
            font-weight: 600;
            background: #EDF5FF;
            color: #0043CE;
            border: 1px solid #0F62FE;
            padding: 1px 6px;
        }}
        .diff-badge-b1 {{
            font-family: "IBM Plex Mono", "Consolas", monospace;
            font-size: 0.75rem;
            font-weight: 600;
            background: #F4F6F8;
            color: #525252;
            border: 1px solid #8D8D8D;
            padding: 1px 6px;
        }}
        .prov-badge {{
            font-family: "IBM Plex Mono", "Consolas", monospace;
            font-size: 0.76rem;
            font-weight: 600;
            padding: 3px 8px;
            display: inline-block;
        }}
        .prov-real {{
            background: #FFF8E1;
            color: #5C4300;
            border: 1px solid #D2A106;
        }}
        .prov-synth {{
            background: #EDF5FF;
            color: #0043CE;
            border: 1px solid #0F62FE;
        }}
        .answer-prose {{
            font-size: {body_size};
            line-height: 1.62;
            color: #161616;
            padding: 6px 0;
        }}
        .record-header {{
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #525252;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached Data Loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_events_and_questions() -> dict[str, Any]:
    real_events: list[Event] = load_jsonl(Event, ROOT_DIR / "data/processed/events.jsonl")
    synth_events: list[Event] = load_jsonl(Event, ROOT_DIR / "data/synthetic/events.jsonl")
    questions: list[Question] = load_jsonl(Question, ROOT_DIR / "data/benchmark/questions.jsonl")

    by_patient: dict[str, list[Event]] = {}
    for ev in synth_events + real_events:
        by_patient.setdefault(ev.patient_id, []).append(ev)

    for pid in by_patient:
        by_patient[pid].sort(key=lambda e: (e.observed_time or e.valid_start or "0000", e.event_id))

    q_by_patient: dict[str, list[Question]] = {}
    for q in questions:
        q_by_patient.setdefault(q.patient_id, []).append(q)

    synth_pids = sorted({e.patient_id for e in synth_events})
    real_pids = sorted({e.patient_id for e in real_events})

    return {
        "by_patient": by_patient,
        "q_by_patient": q_by_patient,
        "synth_pids": synth_pids,
        "real_pids": real_pids,
    }


@st.cache_data(show_spinner=False)
def load_optional_gold_lookup() -> dict[str, dict[str, Any]]:
    """Loaded separately and ONLY accessed if user explicitly opens the post-generation verification expander."""
    golds: list[Gold] = load_jsonl(Gold, ROOT_DIR / "data/benchmark/gold.jsonl")
    return {g.question_id: g.model_dump(mode="json") for g in golds}


@st.cache_data(show_spinner=False)
def load_static_configs() -> dict[str, Any]:
    return {
        "systems": load_json_config(ROOT_DIR / "configs/systems.json"),
        "weights": load_json_config(ROOT_DIR / "configs/weights.json"),
        "relation_table": load_relation_table(ROOT_DIR / "configs/relation_table.json"),
        "rt_rules": load_rt_rules(ROOT_DIR / "configs/rt_rules.json"),
    }


# ---------------------------------------------------------------------------
# Single-Key Pinned Generator Call (No Multi-Key Rotation, Cache-First)
# ---------------------------------------------------------------------------
def generate_single_pinned_answer(
    pipe_out: PipelineOutput,
    question_text: str,
    window_start: Optional[str],
    provider: str,
    model_id: str,
    track: str,
) -> tuple[Optional[AnswerJSON], dict[str, Any]]:
    """Run generation with a single pinned model and primary key, checking disk cache first.

    Blocks Gemini on Track R per Phase 5 data-governance rule.
    Parses HTTP 429 retry time into a structured user-facing status if rate-limited.
    """
    if track == "R" and provider.lower() == "gemini":
        return None, {
            "error_type": "blocked_provider",
            "message": "Gemini free tier is blocked for Track R (MIMIC-IV-ED) records because free-tier prompts may be used by Google for product improvement. Please use the pinned Groq model (openai/gpt-oss-120b).",
        }

    user_prompt = build_user_prompt(
        patient_id=pipe_out.patient_id,
        question_text=question_text,
        as_of_date=pipe_out.as_of_date,
        context_text=pipe_out.context_text,
        window_start=window_start,
    )
    prompt_hash = compute_prompt_hash(SYSTEM_PROMPT, user_prompt)

    cached = load_cached_response(
        model_id, prompt_hash, cache_dir=ROOT_DIR / "results/cache"
    )
    if cached is not None:
        parsed = parse_and_validate_answer_json(
            cached["raw_output"], context_prompt=user_prompt
        )
        assert_no_corpus_citation_in_claims(parsed)
        return parsed, {
            "cached": True,
            "provider": provider,
            "model_id": model_id,
            "prompt_hash": prompt_hash,
        }

    keys_map = load_api_keys(ROOT_DIR / ".env")
    prov_keys = keys_map.get(provider.lower(), [])
    if not prov_keys:
        return None, {
            "error_type": "missing_key",
            "message": f"No API key configured in .env for provider '{provider}'. Cached responses remain available.",
        }

    # Use only the first/primary key for the pinned provider (no multi-key rotation)
    primary_label, primary_key = prov_keys[0]
    try:
        if provider.lower() == "groq":
            raw_out, meta = _call_groq_http(
                primary_key,
                model_id,
                SYSTEM_PROMPT,
                user_prompt,
                temperature=0.0,
            )
        else:
            raw_out, meta = _call_gemini_http(
                primary_key,
                model_id,
                SYSTEM_PROMPT,
                user_prompt,
                temperature=0.0,
            )
        parsed = parse_and_validate_answer_json(raw_out, context_prompt=user_prompt)
        assert_no_corpus_citation_in_claims(parsed)
        meta["cached"] = False
        meta["model_id"] = model_id
        meta["key_label"] = primary_label
        meta["prompt_hash"] = prompt_hash
        save_cached_response(
            model_id,
            prompt_hash,
            raw_out,
            parsed.model_dump(mode="json"),
            metadata=meta,
            cache_dir=ROOT_DIR / "results/cache",
        )
        return parsed, meta
    except Exception as exc:
        err_str = str(exc)
        m_wait = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
        retry_sec = float(m_wait.group(1)) if m_wait else None
        if "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower():
            return None, {
                "error_type": "rate_limit",
                "retry_seconds": retry_sec,
                "message": (
                    f"Rate limit reached on {provider} ({model_id}). "
                    + (f"Please wait {retry_sec:.1f} seconds before retrying." if retry_sec else "Please wait ~60 seconds for the token window to reset.")
                ),
            }
        return None, {
            "error_type": "api_error",
            "message": f"API request failed: {err_str[:240]}",
        }


# ---------------------------------------------------------------------------
# Timeline Centrepiece (Swimlanes + Interval Bars + Point Events + Selected Outline)
# ---------------------------------------------------------------------------
def _parse_iso_day(d_str: Optional[str]) -> Optional[date]:
    if not d_str:
        return None
    clean = d_str.strip()
    try:
        if len(clean) == 10:
            return date.fromisoformat(clean)
        if len(clean) == 7:
            return date.fromisoformat(f"{clean}-01")
        if len(clean) == 4:
            return date.fromisoformat(f"{clean}-01-01")
        return date.fromisoformat(clean[:10])
    except Exception:
        return None


def build_timeline_figure(
    patient_events: list[Event],
    as_of_date: str,
    selected_rank_map: dict[str, int],
    states_at_tq: dict[str, str],
    track: str,
    use_relative_days: bool = False,
    window_start: Optional[str] = None,
) -> go.Figure:
    """Build swimlane timeline chart with validity bars for intervals, dots for point events,
    muted styling for ended/superseded states, accent outline for selected top-k [E#],
    and vertical reference line at `as_of_date`.
    """
    tq_date = _parse_iso_day(as_of_date) or date(2025, 1, 1)

    all_parsed_dates = [
        d
        for ev in patient_events
        for d in (_parse_iso_day(ev.valid_start), _parse_iso_day(ev.observed_time))
        if d is not None
    ]
    day_zero = min(all_parsed_dates) if all_parsed_dates else tq_date

    def to_x(d_obj: date) -> Any:
        if use_relative_days:
            return (d_obj - day_zero).days
        return d_obj.isoformat()

    fig = go.Figure()

    # Present lanes in clinical order
    present_types = [t for t in EVENT_LANE_ORDER if any(e.event_type == t for e in patient_events)]
    for e in patient_events:
        if e.event_type not in present_types:
            present_types.append(e.event_type)

    # Draw interval bars for medication orders / conditions with valid_start and valid_end
    for ev in patient_events:
        d_start = _parse_iso_day(ev.valid_start or ev.observed_time)
        if d_start is None:
            continue
        is_after_tq = d_start > tq_date
        state_str = "AFTER_AS_OF_DATE" if is_after_tq else states_at_tq.get(ev.event_id, "UNKNOWN")
        st_info = get_status_style(state_str) if not is_after_tq else {
            "plot_color": "#C6C6C6",
            "symbol": "◌",
            "opacity": "0.30",
        }
        is_selected = ev.event_id in selected_rank_map
        e_rank = selected_rank_map.get(ev.event_id)

        d_end = _parse_iso_day(ev.valid_end)
        if ev.status_basis == "interval" and not is_after_tq:
            eff_end = min(d_end, tq_date) if d_end else tq_date
            if eff_end > d_start:
                line_color = ACCENT_BLUE if is_selected else st_info["plot_color"]
                line_width = 7 if is_selected else 4
                fig.add_trace(
                    go.Scatter(
                        x=[to_x(d_start), to_x(eff_end)],
                        y=[ev.event_type, ev.event_type],
                        mode="lines",
                        line=dict(
                            color=line_color,
                            width=line_width,
                            dash="solid" if state_str in ("ACTIVE", "PERSISTENT") else "dot",
                        ),
                        opacity=0.95 if is_selected else float(st_info["opacity"]),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # Group point markers by status so legend shows verbatim status names + symbols
    grouped: dict[str, list[tuple[Event, date, bool]]] = {}
    for ev in patient_events:
        d_pt = _parse_iso_day(ev.observed_time or ev.valid_start)
        if d_pt is None:
            continue
        is_after_tq = d_pt > tq_date
        st_key = "AFTER_AS_OF_DATE" if is_after_tq else states_at_tq.get(ev.event_id, "UNKNOWN")
        grouped.setdefault(st_key, []).append((ev, d_pt, ev.event_id in selected_rank_map))

    for st_key, items in grouped.items():
        st_info = (
            get_status_style(st_key)
            if st_key != "AFTER_AS_OF_DATE"
            else {"plot_color": "#C6C6C6", "symbol": "◌", "opacity": "0.35"}
        )
        xs = [to_x(d_pt) for _, d_pt, _ in items]
        ys = [ev.event_type for ev, _, _ in items]
        texts = [
            f"[E{selected_rank_map[ev.event_id]}]" if is_sel else ""
            for ev, _, is_sel in items
        ]
        marker_sizes = [13 if is_sel else 8 for _, _, is_sel in items]
        line_colors = [ACCENT_BLUE if is_sel else "#FFFFFF" for _, _, is_sel in items]
        line_widths = [2.6 if is_sel else 0.8 for _, _, is_sel in items]
        hover_texts = [
            (
                f"<b>{f'[E{selected_rank_map[ev.event_id]}] ' if is_sel else ''}{html.escape(ev.concept)}</b>"
                f"<br>ID: {ev.event_id} | Type: {ev.event_type}"
                f"<br>Status at as_of_date: {st_key}"
                f"<br>Observed/Start: {ev.observed_time or ev.valid_start or 'unknown'}"
                f"{f' -> End: {ev.valid_end}' if ev.valid_end else ''}"
                f"<br>Precision: {ev.date_precision}"
                f"<br>Detail: {html.escape(ev.text[:90])}"
            )
            for ev, _, is_sel in items
        ]
        legend_name = (
            f"{st_info['symbol']} {st_key}"
            if st_key != "AFTER_AS_OF_DATE"
            else "◌ AFTER as_of_date (Excluded)"
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name=legend_name,
                text=texts,
                textposition="top center",
                textfont=dict(
                    family="IBM Plex Mono, Consolas, monospace",
                    size=10,
                    color=ACCENT_BLUE,
                ),
                marker=dict(
                    size=marker_sizes,
                    color=st_info["plot_color"],
                    opacity=float(st_info["opacity"]),
                    line=dict(color=line_colors, width=line_widths),
                    symbol="diamond" if st_key == "PERSISTENT" else ("circle-open" if st_key in ("ENDED", "NOT_LISTED_LATEST") else "circle"),
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
            )
        )

    # Vertical line at as_of_date
    x_tq = to_x(tq_date)
    fig.add_vline(
        x=x_tq,
        line_width=2,
        line_dash="dash",
        line_color="#161616",
    )
    fig.add_annotation(
        x=x_tq,
        y=1.06,
        yref="paper",
        text=f"as_of_date: {f'Day +{(tq_date - day_zero).days}' if use_relative_days else as_of_date}",
        showarrow=False,
        font=dict(family="IBM Plex Mono, Consolas, monospace", size=11, color="#161616"),
        bgcolor="#F4F6F8",
        bordercolor="#161616",
        borderwidth=1,
    )

    if window_start:
        w_date = _parse_iso_day(window_start)
        if w_date:
            fig.add_vline(
                x=to_x(w_date),
                line_width=1.5,
                line_dash="dot",
                line_color=ACCENT_BLUE,
            )

    x_title = (
        "Days Since First Recorded Visit (Day 0 = First Visit)"
        if use_relative_days
        else ("Shifted Calendar Date (Track R)" if track == "R" else "Clinical Timeline Date")
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=34, b=28),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor="#EDF1F5",
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=list(reversed(present_types)),
            showgrid=True,
            gridcolor="#F4F6F8",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="left",
            x=0.0,
            font=dict(family="IBM Plex Mono, Consolas, monospace", size=11),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Structured Clinical History Summary (Clinician View Stretch Item, Section 9.2)
# ---------------------------------------------------------------------------
def render_structured_clinical_summary(
    pool_events: list[Event],
    states_map: dict[str, str],
) -> None:
    active_meds: list[str] = []
    ended_meds: list[str] = []
    persistent_items: list[str] = []
    latest_labs: dict[str, Event] = {}

    for ev in pool_events:
        st_val = states_map.get(ev.event_id, "UNKNOWN")
        if ev.event_type in ("medication_order", "reported_home_medication"):
            dose = ev.attributes.get("dose", "")
            label = f"{ev.concept} ({dose})" if dose else ev.concept
            if st_val in ("ACTIVE", "LISTED") and label not in active_meds:
                active_meds.append(label)
            elif st_val in ("ENDED", "SUPERSEDED", "NOT_LISTED_LATEST") and label not in ended_meds:
                ended_meds.append(label)
        elif ev.event_type in ("condition", "diagnosis_at_visit", "allergy"):
            if st_val == "PERSISTENT" and ev.concept not in persistent_items:
                persistent_items.append(f"{ev.concept} [{ev.event_type}]")
        elif ev.event_type == "lab_result":
            prev = latest_labs.get(ev.concept)
            if prev is None or (ev.observed_time or "") >= (prev.observed_time or ""):
                latest_labs[ev.concept] = ev

    with st.container(border=True):
        st.markdown('<div class="record-header">Structured Longitudinal State Summary at as_of_date</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption("ACTIVE / LISTED MEDICATIONS")
            st.markdown(", ".join(f"`{m}`" for m in active_meds[:8]) if active_meds else "*None recorded*")
        with c2:
            st.caption("PERSISTENT CONDITIONS & ALLERGIES")
            st.markdown(", ".join(f"`{p}`" for p in persistent_items[:8]) if persistent_items else "*None recorded*")
        with c3:
            st.caption("LATEST LABORATORY VALUES")
            if latest_labs:
                lab_strs = []
                for k_concept, l_ev in list(latest_labs.items())[:5]:
                    val = l_ev.attributes.get("value", "")
                    unit = l_ev.attributes.get("unit", "")
                    lab_strs.append(f"`{k_concept}: {val} {unit}` ({l_ev.observed_time})")
                st.markdown("<br>".join(lab_strs), unsafe_allow_html=True)
            else:
                st.markdown("*No lab results in Track R ED snapshot*")
        with c4:
            st.caption("ENDED / SUPERSEDED MEDICATIONS")
            st.markdown(", ".join(f"`{m}`" for m in ended_meds[:8]) if ended_meds else "*None recorded*")


# ---------------------------------------------------------------------------
# Conflict & Uncertainty Presentation (Section 9.6)
# ---------------------------------------------------------------------------
def render_conflicts_and_uncertainties(
    pipe_out: PipelineOutput,
    events_by_id: dict[str, Event],
    rank_map: dict[str, int],
) -> None:
    if pipe_out.flags.insufficient_evidence:
        with st.container(border=True):
            st.markdown(
                '<span class="flag-badge-unc">U5 &middot; INSUFFICIENT_EVIDENCE</span> '
                "<strong>Insufficient Evidence in Patient Record:</strong> "
                "No candidate events cleared the query relevance threshold at the specified <code>as_of_date</code>. "
                "The system abstains rather than guessing.",
                unsafe_allow_html=True,
            )

    if pipe_out.flags.conflicts:
        for cf in pipe_out.flags.conflicts:
            with st.container(border=True):
                e_tags = [
                    f"[E{rank_map[eid]}] ({eid})" if eid in rank_map else eid
                    for eid in cf.event_ids
                ]
                st.markdown(
                    f'<span class="flag-badge-conflict">CONFLICT &middot; {html.escape(cf.rule)}</span> '
                    f"<strong>Conflicting Records Detected ({', '.join(e_tags)}):</strong> "
                    f"{html.escape(cf.description or '')}",
                    unsafe_allow_html=True,
                )
                if len(cf.event_ids) >= 2:
                    cols = st.columns(min(3, len(cf.event_ids)))
                    for idx, eid in enumerate(cf.event_ids[:3]):
                        ev = events_by_id.get(eid)
                        if ev:
                            with cols[idx]:
                                r_lbl = f"[E{rank_map[eid]}]" if eid in rank_map else eid
                                st.markdown(
                                    f"<span class='cite-tag'>{r_lbl}</span> "
                                    f"<span class='mono-id'>{ev.event_id}</span><br>"
                                    f"<strong>{html.escape(ev.concept)}</strong> &mdash; {html.escape(ev.text)}<br>"
                                    f"<small>Interval: <code>{ev.valid_start or ev.observed_time}</code> &rarr; "
                                    f"<code>{ev.valid_end or 'ongoing'}</code> | Source: <code>{ev.source_table or ev.source_dataset}</code></small>",
                                    unsafe_allow_html=True,
                                )

    if pipe_out.flags.provenance_notes:
        for pn in pipe_out.flags.provenance_notes:
            with st.container(border=True):
                st.markdown(
                    f'<span class="mono-id">PROVENANCE &middot; {html.escape(pn.note_type)}</span> '
                    f"{html.escape(pn.description)}",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Answer Presentation
# ---------------------------------------------------------------------------
def format_answer_text_with_citations(
    answer_text: str,
    cited_event_ids: list[str],
    rank_map: dict[str, int],
) -> str:
    escaped = html.escape(answer_text)
    # Highlight [E#] tags
    escaped = re.sub(
        r"\[(E\d+)\]",
        r'<span class="cite-tag">[\1]</span>',
        escaped,
    )
    # Also replace raw event_ids if mentioned in prose with their [E#] tag + mono-id
    for eid in cited_event_ids:
        if eid in rank_map and eid in escaped:
            tag = f'<span class="cite-tag">[E{rank_map[eid]}]</span><span class="mono-id">{eid}</span>'
            escaped = escaped.replace(eid, tag)
    return f'<div class="answer-prose">{escaped}</div>'


def render_answer_panel(
    ans: Optional[AnswerJSON],
    meta: dict[str, Any],
    pipe_out: PipelineOutput,
    rank_map: dict[str, int],
    system_label: str,
) -> None:
    with st.container(border=True):
        header_cols = st.columns([3, 2])
        with header_cols[0]:
            st.markdown(
                f'<div class="record-header">Grounded Response &middot; {html.escape(system_label)} '
                f'(Intent: {html.escape(pipe_out.intent)})</div>',
                unsafe_allow_html=True,
            )
        with header_cols[1]:
            if meta.get("model_id"):
                src_badge = "DISK CACHE" if meta.get("cached") else "LIVE API"
                st.markdown(
                    f'<div style="text-align:right;"><span class="mono-id">{html.escape(str(meta.get("model_id")))} &middot; {src_badge}</span></div>',
                    unsafe_allow_html=True,
                )

        if ans is None:
            err_type = meta.get("error_type", "")
            msg = meta.get("message", "Answer has not been generated yet.")
            if err_type == "rate_limit":
                st.warning(msg)
            elif err_type == "blocked_provider":
                st.error(msg)
            else:
                st.info(msg)
            return

        if ans.abstain or pipe_out.flags.insufficient_evidence:
            st.markdown(
                '<span class="flag-badge-unc">ABSTAIN &middot; U5 INSUFFICIENT EVIDENCE</span> '
                "<strong>The patient's records up to the specified date do not contain sufficient evidence to answer this question.</strong>",
                unsafe_allow_html=True,
            )
            st.markdown(
                format_answer_text_with_citations(ans.answer_text, ans.cited_event_ids, rank_map),
                unsafe_allow_html=True,
            )
            return

        st.markdown(
            format_answer_text_with_citations(ans.answer_text, ans.cited_event_ids, rank_map),
            unsafe_allow_html=True,
        )

        # Status claims & cited event IDs bar
        b_col1, b_col2 = st.columns([3, 2])
        with b_col1:
            if ans.status_claims:
                claims_html = " ".join(
                    f'<span class="mono-id">{html.escape(sc.concept)}</span> {render_status_badge_html(sc.status)}'
                    for sc in ans.status_claims
                )
                st.markdown(f"<small><strong>Status Claims:</strong></small> {claims_html}", unsafe_allow_html=True)
        with b_col2:
            if ans.cited_event_ids:
                cites_html = " ".join(
                    f'<span class="cite-tag">[E{rank_map[cid]}]</span><span class="mono-id">{html.escape(cid)}</span>'
                    if cid in rank_map
                    else f'<span class="mono-id">{html.escape(cid)}</span>'
                    for cid in ans.cited_event_ids
                )
                st.markdown(f"<small><strong>Supporting Events:</strong></small> {cites_html}", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Evidence List Presentation (Patient Mode vs Clinician Mode Density)
# ---------------------------------------------------------------------------
def render_evidence_records(
    pipe_out: PipelineOutput,
    events_by_id: dict[str, Event],
    view_mode: str,
    diff_tags_by_eid: Optional[dict[str, str]] = None,
) -> None:
    cand_by_id = {c.event_id: c for c in pipe_out.candidates}
    unc_by_eid: dict[str, list[str]] = {}
    for uf in pipe_out.flags.uncertainty:
        for eid in uf.event_ids:
            unc_by_eid.setdefault(eid, []).append(uf.flag)

    conf_by_eid: dict[str, list[str]] = {}
    for cf in pipe_out.flags.conflicts:
        for eid in cf.event_ids:
            conf_by_eid.setdefault(eid, []).append(cf.rule)

    if view_mode == "Patient":
        for rank_idx, eid in enumerate(pipe_out.selected_event_ids, start=1):
            ev = events_by_id.get(eid)
            cand = cand_by_id.get(eid)
            if not ev or not cand:
                continue
            with st.container(border=True):
                u_badges = " ".join(
                    f'<span class="flag-badge-unc">{u}</span>' for u in unc_by_eid.get(eid, [])
                )
                c_badges = " ".join(
                    f'<span class="flag-badge-conflict">{c}</span>' for c in conf_by_eid.get(eid, [])
                )
                diff_html = diff_tags_by_eid.get(eid, "") if diff_tags_by_eid else ""
                date_str = ev.valid_start or ev.observed_time or "unknown"
                if ev.valid_end:
                    date_str = f"{date_str} to {ev.valid_end}"
                st.markdown(
                    f'<span class="cite-tag">[E{rank_idx}]</span> '
                    f'<span class="mono-id">{html.escape(ev.event_id)}</span> '
                    f'{render_status_badge_html(cand.state)} {c_badges} {u_badges} {diff_html}'
                    f'<div style="margin-top:6px;font-size:1.02rem;">'
                    f'<strong>{html.escape(ev.concept)}</strong> &mdash; {html.escape(ev.text)}</div>'
                    f'<div style="margin-top:4px;font-size:0.84rem;color:#525252;">'
                    f'Type: <code>{ev.event_type}</code> &middot; Date: <code>{date_str}</code> '
                    f'(precision: <code>{ev.date_precision}</code>) &middot; '
                    f'Provenance: <code>{ev.source_type.upper()} / {ev.source_table or ev.source_dataset}</code></div>',
                    unsafe_allow_html=True,
                )
    else:
        # Clinician Mode: Compact high-density table with Rq, Rh, Rt, Fused Score, and verbatim statuses
        rows = []
        for rank_idx, eid in enumerate(pipe_out.selected_event_ids, start=1):
            ev = events_by_id.get(eid)
            cand = cand_by_id.get(eid)
            if not ev or not cand:
                continue
            flags_list = conf_by_eid.get(eid, []) + unc_by_eid.get(eid, [])
            st_sym = get_status_style(cand.state)["symbol"]
            row_dict: dict[str, Any] = {
                "Rank": f"[E{rank_idx}]",
                "Event ID": ev.event_id,
                "Status": f"{st_sym} {cand.state}",
                "Type": ev.event_type,
                "Concept": ev.concept,
                "Clinical Text": ev.text,
                "Valid Start / Obs": ev.valid_start or ev.observed_time or "unknown",
                "Valid End": ev.valid_end or "-",
                "Precision": ev.date_precision,
                "Rq": round(cand.rq, 3),
                "Rh": round(cand.rh, 3),
                "Rt": round(cand.rt, 3),
                "Fused S": round(cand.score, 3),
                "Flags": ", ".join(flags_list) if flags_list else "-",
                "Source Table": ev.source_table or ev.source_dataset,
            }
            if diff_tags_by_eid and eid in diff_tags_by_eid:
                row_dict["Selection Diff"] = (
                    "ONLY B2" if "ONLY B2" in diff_tags_by_eid[eid] else ("ONLY B1" if "ONLY B1" in diff_tags_by_eid[eid] else "SHARED")
                )
            rows.append(row_dict)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# General Medical Information Box ([C1], [C2]) (Section 9.5 & Part 1)
# ---------------------------------------------------------------------------
def render_general_information_box(
    question_text: str,
    selected_events: list[Event],
    insufficient_evidence: bool,
) -> None:
    passages = retrieve_corpus_passages(
        question_text=question_text,
        selected_events=selected_events,
        insufficient_evidence=insufficient_evidence,
        top_k=2,
    )
    if not passages:
        # Part 1 Rule 5: "If nothing clears the threshold, show no background box."
        return

    with st.container(border=True):
        st.markdown(
            '<div class="record-header">General Medical Background (Demo Reference Corpus &middot; Separate from Patient Record)</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Educational reference passages retrieved from public-domain MedlinePlus and paraphrased FDA DailyMed class notes. "
            "These [C#] passages never enter patient event [E#] scoring and never support patient-specific claims."
        )
        cols = st.columns(len(passages))
        for idx, item in enumerate(passages):
            ch = item["chunk"]
            tag = item["citation_tag"]
            with cols[idx]:
                st.markdown(
                    f'<span class="corpus-tag">{tag}</span> '
                    f'<span class="mono-id">{html.escape(ch.chunk_id)} &middot; {html.escape(ch.chunk_type)}</span> '
                    f'<strong>{html.escape(ch.title)}</strong>'
                    f'<p style="font-size:0.88rem;line-height:1.5;color:#262626;margin-top:6px;">{html.escape(ch.text)}</p>'
                    f'<div style="font-size:0.76rem;color:#525252;">'
                    f'<strong>Source:</strong> <a href="{html.escape(ch.url)}" target="_blank">{html.escape(ch.source_name)}</a> '
                    f'(Accessed: <code>{html.escape(ch.accessed)}</code>)<br>'
                    f'<strong>Matched Concepts:</strong> <code>{", ".join(item["matched_concepts"])}</code><br>'
                    f'<em>{html.escape(ch.license_note)}</em></div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Main Single-Page Application
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Chronology-Aware Clinical Record Viewer",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    data_bundle = load_all_events_and_questions()
    configs = load_static_configs()

    # Top bar: Title + View Mode Switch
    top_col1, top_col2 = st.columns([4, 1])
    with top_col1:
        st.markdown("### Chronology-Aware Clinical Record & Evidence Viewer")
    with top_col2:
        view_mode = st.radio(
            "Display Density Mode",
            options=["Patient", "Clinician"],
            horizontal=True,
            index=1,
        )

    inject_design_system_css(view_mode)

    # Permanent Prototype Disclaimer Notice (Section 9.5)
    st.markdown(
        '<div class="prototype-banner">'
        '<span><strong>NOTICE:</strong> Research prototype. Not medical advice. Synthetic and de-identified demo data.</span>'
        '<span class="mono-id">Schema v1.0 &middot; Prompt v1.0</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Primary Control Bar (Progressive Disclosure: Patient, Date, System, Question up front)
    with st.container(border=True):
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.3, 1.4, 1.3, 2.0])
        with ctrl1:
            track_choice = st.selectbox(
                "Data Track",
                options=["S (Synthetic Longitudinal)", "R (MIMIC-IV-ED Demo)"],
                index=0,
            )
            active_track = "S" if track_choice.startswith("S") else "R"

        pids = data_bundle["synth_pids"] if active_track == "S" else data_bundle["real_pids"]
        with ctrl2:
            default_pid_idx = pids.index("S_020") if ("S_020" in pids and active_track == "S") else 0
            selected_pid = st.selectbox("Patient ID", options=pids, index=default_pid_idx)

        patient_events: list[Event] = data_bundle["by_patient"].get(selected_pid, [])
        patient_questions: list[Question] = data_bundle["q_by_patient"].get(selected_pid, [])

        # Determine latest date in patient's record
        obs_dates = sorted(
            {
                (e.observed_time or e.valid_start)[:10]
                for e in patient_events
                if (e.observed_time or e.valid_start) and len(e.observed_time or e.valid_start or "") >= 10
            }
        )
        default_as_of = patient_questions[0].as_of_date if patient_questions else (obs_dates[-1] if obs_dates else "2025-09-01")

        with ctrl4:
            preset_options = ["Custom clinical question..."] + [
                f"[{q.question_id}] {q.text} (as_of: {q.as_of_date})"
                for q in patient_questions
            ]
            selected_preset = st.selectbox(
                "Preset Benchmark Questions (loads question & as_of_date only)",
                options=preset_options,
                index=1 if len(preset_options) > 1 else 0,
            )

        active_preset_q: Optional[Question] = None
        if selected_preset != "Custom clinical question...":
            q_idx = preset_options.index(selected_preset) - 1
            active_preset_q = patient_questions[q_idx]
            default_as_of = active_preset_q.as_of_date

        with ctrl3:
            as_of_date_str = st.text_input(
                "Query Date (as_of_date, YYYY-MM-DD)",
                value=default_as_of,
            ).strip()

        q_col1, q_col2 = st.columns([3.5, 1.5])
        with q_col1:
            question_text = st.text_input(
                "Clinical Question",
                value=active_preset_q.text if active_preset_q else "What medications is the patient currently taking?",
            ).strip()
        with q_col2:
            system_mode = st.selectbox(
                "Pipeline Configuration",
                options=[
                    "Compare B1 vs B2 (Selection Diff + On-Demand LLM)",
                    "B2 (Chronology-Aware Fusion + A2)",
                    "B1 (Conventional RAG Baseline)",
                ],
                index=0,
            )

        # Progressive Disclosure: Advanced Controls Expander
        with st.expander("Advanced Configuration (Provider, Top-k Budget, Window Filter, Timeline Axis)"):
            adv1, adv2, adv3, adv4 = st.columns(4)
            with adv1:
                if active_track == "R":
                    provider_choice = st.selectbox(
                        "LLM Provider (Track R Policy)",
                        options=["groq (openai/gpt-oss-120b) — Pinned"],
                        index=0,
                        help="Gemini free tier is disabled on Track R because free-tier prompts may be used by Google to improve products.",
                    )
                    active_provider = "groq"
                    active_model_id = DEFAULT_GROQ_MODEL
                else:
                    provider_choice = st.selectbox(
                        "Pinned Generator Model",
                        options=[
                            f"groq ({DEFAULT_GROQ_MODEL})",
                            f"gemini ({DEFAULT_GEMINI_MODEL})",
                        ],
                        index=0,
                    )
                    active_provider = "groq" if provider_choice.startswith("groq") else "gemini"
                    active_model_id = DEFAULT_GROQ_MODEL if active_provider == "groq" else DEFAULT_GEMINI_MODEL
            with adv2:
                k_budget = st.slider("Context Budget (top-k events)", min_value=5, max_value=20, value=10, step=1)
            with adv3:
                default_win = active_preset_q.window_start if (active_preset_q and active_preset_q.window_start) else ""
                window_start_in = st.text_input("Optional window_start (YYYY-MM-DD)", value=default_win).strip()
                window_start_val = window_start_in if window_start_in else None
            with adv4:
                use_rel_days = st.checkbox(
                    "Use 'Days Since First Visit' Timeline Axis",
                    value=(active_track == "R"),
                    help="Recommended for Track R (MIMIC-IV-ED) because real dates are shifted.",
                )

    # Provenance header bar
    sample_src_ds = patient_events[0].source_dataset if patient_events else "custom_generator"
    st.markdown(
        f'{render_provenance_badge_html(active_track, sample_src_ds)} '
        f'<span class="mono-id">Patient: {html.escape(selected_pid)}</span> '
        f'<span class="mono-id">Total Events: {len(patient_events)}</span> '
        f'<span class="mono-id">as_of_date: {html.escape(as_of_date_str)}</span>',
        unsafe_allow_html=True,
    )

    # Empty question guard
    if not question_text:
        st.info("Enter a clinical question above or select a preset question to inspect retrieval and temporal state fusion.")
        return

    # Execute Retrieval & Temporal Fusion Stages (Deterministic, <0.1s, Zero LLM Calls)
    pipe_b2 = run_retrieval_stage(
        patient_id=selected_pid,
        question_text=question_text,
        as_of_date=as_of_date_str,
        patient_events=patient_events,
        system_id="B2",
        k=k_budget,
        window_start=window_start_val,
        weights_config=configs["weights"],
        systems_config=configs["systems"],
        relation_table=configs["relation_table"],
        rt_rules=configs["rt_rules"],
    )
    pipe_b1 = run_retrieval_stage(
        patient_id=selected_pid,
        question_text=question_text,
        as_of_date=as_of_date_str,
        patient_events=patient_events,
        system_id="B1",
        k=k_budget,
        window_start=window_start_val,
        weights_config=configs["weights"],
        systems_config=configs["systems"],
        relation_table=configs["relation_table"],
        rt_rules=configs["rt_rules"],
    )

    primary_pipe = pipe_b1 if system_mode.startswith("B1") else pipe_b2
    events_by_id = {e.event_id: e for e in patient_events}
    pool_at_tq = filter_candidate_pool(patient_events, as_of_date_str, patient_id=selected_pid)
    states_at_tq = {e.event_id: compute_state(e, as_of_date_str, pool_at_tq) for e in pool_at_tq}
    rank_map_primary = {eid: idx + 1 for idx, eid in enumerate(primary_pipe.selected_event_ids)}

    # 1. Timeline Centrepiece
    with st.container(border=True):
        tl_label = "B1 Top-k Highlighted" if system_mode.startswith("B1") else "B2 Top-k Highlighted"
        st.markdown(
            f'<div class="record-header">Longitudinal Patient Timeline ({tl_label} in Accent Blue &middot; Verbatim Statuses at as_of_date)</div>',
            unsafe_allow_html=True,
        )
        fig = build_timeline_figure(
            patient_events=patient_events,
            as_of_date=as_of_date_str,
            selected_rank_map=rank_map_primary,
            states_at_tq=states_at_tq,
            track=active_track,
            use_relative_days=use_rel_days,
            window_start=window_start_val,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Clinician Mode Stretch Summary
    if view_mode == "Clinician":
        render_structured_clinical_summary(pool_at_tq, states_at_tq)

    # Conflict & Uncertainty Alerts (from B2's temporal/conflict analysis)
    render_conflicts_and_uncertainties(pipe_b2, events_by_id, {eid: i + 1 for i, eid in enumerate(pipe_b2.selected_event_ids)})

    # 2. Mode-Specific Results: Side-by-Side B1 vs B2 vs Single System
    state_key_prefix = f"{selected_pid}_{as_of_date_str}_{question_text}_{active_provider}_{active_model_id}_{k_budget}"

    if system_mode.startswith("Compare"):
        set_b1 = set(pipe_b1.selected_event_ids)
        set_b2 = set(pipe_b2.selected_event_ids)
        only_b2 = [eid for eid in pipe_b2.selected_event_ids if eid not in set_b1]
        only_b1 = [eid for eid in pipe_b1.selected_event_ids if eid not in set_b2]
        shared_eids = [eid for eid in pipe_b2.selected_event_ids if eid in set_b1]

        diff_tags_b2 = {
            eid: ('<span class="diff-badge-b2">ONLY B2 (Promoted by Rh/Rt)</span>' if eid in only_b2 else '<span class="mono-id">SHARED</span>')
            for eid in pipe_b2.selected_event_ids
        }
        diff_tags_b1 = {
            eid: ('<span class="diff-badge-b1">ONLY B1 (Dropped in B2)</span>' if eid in only_b1 else '<span class="mono-id">SHARED</span>')
            for eid in pipe_b1.selected_event_ids
        }

        with st.container(border=True):
            st.markdown(
                '<div class="record-header">Immediate Selection Comparison: B1 (Hybrid Rq Only, A0) vs B2 (Chronology-Aware Fusion, A2)</div>',
                unsafe_allow_html=True,
            )
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Detected Intent", pipe_b2.intent)
            m2.metric("Shared Top-k Events", f"{len(shared_eids)} / {k_budget}")
            m3.metric("Picked ONLY by B2", len(only_b2))
            m4.metric("Picked ONLY by B1", len(only_b1))
            m5.metric("B2 Flags (Conflicts / Unc.)", f"{len(pipe_b2.flags.conflicts)}C / {len(pipe_b2.flags.uncertainty)}U")

            btn_c1, btn_c2, btn_c3 = st.columns(3)
            with btn_c1:
                gen_b2_clicked = st.button("Generate B2 Answer", type="primary", use_container_width=True)
            with btn_c2:
                gen_b1_clicked = st.button("Generate B1 Answer", use_container_width=True)
            with btn_c3:
                gen_both_clicked = st.button("Generate Both (B1 & B2)", use_container_width=True)

            if gen_b2_clicked or gen_both_clicked:
                with st.status("Running B2 pipeline stages...", expanded=False) as status_box:
                    st.write("1. Candidate pool filtered at `as_of_date` (`E_<=tq`).")
                    st.write(f"2. Intent `{pipe_b2.intent}` classified; fused `Rq + Rh + Rt` scores computed.")
                    st.write(f"3. Attached A2 annotations ({len(pipe_b2.flags.conflicts)} conflict, {len(pipe_b2.flags.uncertainty)} uncertainty flags).")
                    st.write(f"4. Querying pinned model `{active_model_id}` (checking disk cache first)...")
                    ans_b2, meta_b2 = generate_single_pinned_answer(
                        pipe_b2, question_text, window_start_val, active_provider, active_model_id, active_track
                    )
                    st.session_state[f"ans_B2_{state_key_prefix}"] = (ans_b2, meta_b2)
                    status_box.update(label="B2 execution complete", state="complete")

            if gen_b1_clicked or gen_both_clicked:
                with st.status("Running B1 baseline pipeline stages...", expanded=False) as status_box:
                    st.write("1. Candidate pool filtered at `as_of_date` (`E_<=tq`).")
                    st.write("2. Ranked by hybrid `Rq` only with raw unannotated context (`A0`).")
                    st.write(f"3. Querying pinned model `{active_model_id}` (checking disk cache first)...")
                    ans_b1, meta_b1 = generate_single_pinned_answer(
                        pipe_b1, question_text, window_start_val, active_provider, active_model_id, active_track
                    )
                    st.session_state[f"ans_B1_{state_key_prefix}"] = (ans_b1, meta_b1)
                    status_box.update(label="B1 execution complete", state="complete")

        col_left, col_right = st.columns(2)
        with col_left:
            b2_stored = st.session_state.get(f"ans_B2_{state_key_prefix}", (None, {"message": "Click 'Generate B2 Answer' above to run LLM generation (selection shown below)."}))
            render_answer_panel(
                b2_stored[0],
                b2_stored[1],
                pipe_b2,
                {eid: i + 1 for i, eid in enumerate(pipe_b2.selected_event_ids)},
                "B2 (Proposed: Chronology-Aware + A2)",
            )
            st.markdown('<div class="record-header">B2 Selected Evidence ([E1]–[Ek])</div>', unsafe_allow_html=True)
            render_evidence_records(pipe_b2, events_by_id, view_mode, diff_tags_by_eid=diff_tags_b2)

        with col_right:
            b1_stored = st.session_state.get(f"ans_B1_{state_key_prefix}", (None, {"message": "Click 'Generate B1 Answer' above to run LLM generation (selection shown below)."}))
            render_answer_panel(
                b1_stored[0],
                b1_stored[1],
                pipe_b1,
                {eid: i + 1 for i, eid in enumerate(pipe_b1.selected_event_ids)},
                "B1 (Baseline: Hybrid Rq Only + A0)",
            )
            st.markdown('<div class="record-header">B1 Selected Evidence ([E1]–[Ek])</div>', unsafe_allow_html=True)
            render_evidence_records(pipe_b1, events_by_id, view_mode, diff_tags_by_eid=diff_tags_b1)

        any_generated = (
            st.session_state.get(f"ans_B2_{state_key_prefix}", (None, {}))[0] is not None
            or st.session_state.get(f"ans_B1_{state_key_prefix}", (None, {}))[0] is not None
        )

    else:
        sys_code = "B1" if system_mode.startswith("B1") else "B2"
        sys_title = (
            "B1 (Conventional RAG Baseline, A0)"
            if sys_code == "B1"
            else "B2 (Chronology-Aware Fusion + A2)"
        )
        run_clicked = st.button(f"Generate {sys_code} Answer", type="primary")
        if run_clicked:
            with st.status(f"Executing {sys_code} pipeline stages...", expanded=False) as status_box:
                st.write("1. Candidate pool filtered at `as_of_date` (`E_<=tq`).")
                st.write(f"2. Intent `{primary_pipe.intent}` classified and top-{k_budget} events selected.")
                st.write(f"3. Querying pinned model `{active_model_id}` (checking disk cache first)...")
                ans_single, meta_single = generate_single_pinned_answer(
                    primary_pipe, question_text, window_start_val, active_provider, active_model_id, active_track
                )
                st.session_state[f"ans_{sys_code}_{state_key_prefix}"] = (ans_single, meta_single)
                status_box.update(label=f"{sys_code} execution complete", state="complete")

        stored_single = st.session_state.get(
            f"ans_{sys_code}_{state_key_prefix}",
            (None, {"message": f"Click 'Generate {sys_code} Answer' above to generate the grounded response."}),
        )
        render_answer_panel(
            stored_single[0],
            stored_single[1],
            primary_pipe,
            rank_map_primary,
            sys_title,
        )
        st.markdown(f'<div class="record-header">{sys_code} Selected Evidence Records ([E1]–[E{len(primary_pipe.selected_event_ids)}])</div>', unsafe_allow_html=True)
        render_evidence_records(primary_pipe, events_by_id, view_mode)
        any_generated = stored_single[0] is not None

    # Optional Post-Generation Benchmark Verification Expander (Hidden by default, shown only after generation)
    if active_preset_q is not None and any_generated:
        with st.expander("Show Expected Benchmark Answer (Demo Verification Only — Never Used by Pipeline)", expanded=False):
            gold_lookup = load_optional_gold_lookup()
            g_item = gold_lookup.get(active_preset_q.question_id)
            if g_item:
                st.markdown(
                    f"- **Benchmark Question ID:** `{active_preset_q.question_id}` (`{g_item.get('category')}` | `{g_item.get('intent')}`)\n"
                    f"- **Expected Behavior:** `{g_item.get('expected_behavior')}`\n"
                    f"- **Expected Answer Value:** `{g_item.get('expected_answer', {}).get('value')}`\n"
                    f"- **Gold Supporting Events:** `{', '.join(g_item.get('gold_events', []))}`\n"
                    f"- **Distractor (Stale/Superseded) Events:** `{', '.join(g_item.get('distractor_events', [])) or 'None'}`"
                )

    # 3. Separate General Medical Background Box ([C1], [C2])
    selected_ev_objs = [
        events_by_id[eid]
        for eid in pipe_b2.selected_event_ids
        if eid in events_by_id
    ]
    render_general_information_box(
        question_text=question_text,
        selected_events=selected_ev_objs,
        insufficient_evidence=pipe_b2.flags.insufficient_evidence,
    )


if __name__ == "__main__":
    main()
