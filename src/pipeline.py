"""Unified Chronology-Aware RAG Pipeline (`src/pipeline.py`) (Sections 5.2, 7.7, 7.8, 12.2).

Serves both the evaluation runner and the Streamlit demonstration application (AD3).
Supports all retrieval-level and generation-level system configurations (RS1-RS5, B1, B1+, B2-, B2, B3, B4, ORACLE)
and annotation levels A0, A1, and A2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from src.flags.conflict_detection import detect_conflicts
from src.flags.uncertainty import uncertainty_flags
from src.fusion.selection import (
    order_chronologically,
    rank_by_fused_score,
    select_recency_only,
    select_status_filter_only,
)
from src.retrieval.rrf import rq_scores
from src.schemas import (
    AnswerJSON,
    CandidateScore,
    Event,
    FlagsContainer,
    IntentType,
    PipelineOutput,
    TemporalStateType,
)
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


def load_json_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def assemble_context_text(
    chronological_event_ids: list[str],
    events_by_id: dict[str, Event],
    states_by_id: dict[str, TemporalStateType],
    flags: FlagsContainer,
    annotation_level: str = "A2",
) -> str:
    """Assemble the grounded prompt context string according to annotation level (Section 7.7):
      - `A0`: events only (ID, observed time, text).
      - `A1`: `A0` plus computed temporal status labels (`state`, validity interval).
      - `A2`: `A1` plus conflict and uncertainty flags (`C1-C3`, `U1-U5`, provenance notes).
    """
    lines: list[str] = []
    for idx, eid in enumerate(chronological_event_ids, start=1):
        ev = events_by_id.get(eid)
        if ev is None:
            continue
        obs_str = ev.observed_time if ev.observed_time is not None else "UNKNOWN_DATE"
        if annotation_level == "A0":
            lines.append(f"[E{idx}] id={ev.event_id} | date={obs_str} | {ev.text}")
        else:
            st = states_by_id.get(eid, "UNKNOWN")
            v_start = ev.valid_start or "UNKNOWN"
            v_end = ev.valid_end or "OPEN"
            lines.append(
                f"[E{idx}] id={ev.event_id} | date={obs_str} | validity=[{v_start} to {v_end}] "
                f"| status={st} | {ev.text}"
            )

    if annotation_level == "A2":
        if flags.conflicts:
            lines.append("\n[DETECTED CONFLICTS]")
            for c in flags.conflicts:
                lines.append(
                    f"- Rule {c.rule} on events {', '.join(c.event_ids)}: {c.description or ''}"
                )
        if flags.uncertainty:
            lines.append("\n[UNCERTAINTY FLAGS]")
            for u in flags.uncertainty:
                lines.append(
                    f"- Flag {u.flag} on events {', '.join(u.event_ids)}: {u.description or ''}"
                )
        if flags.insufficient_evidence:
            lines.append(
                "\n[INSUFFICIENT EVIDENCE WARNING (U5)]: No candidate record reaches the relevance threshold for the queried concept."
            )
        if flags.provenance_notes:
            lines.append("\n[PROVENANCE NOTES]")
            for p in flags.provenance_notes:
                lines.append(f"- {p.note_type} ({', '.join(p.event_ids)}): {p.description}")

    return "\n".join(lines)


def run_retrieval_stage(
    patient_id: str,
    question_text: str,
    as_of_date: str,
    patient_events: list[Event],
    system_id: str = "B2",
    k: int = 10,
    window_start: Optional[str] = None,
    intent_override: Optional[IntentType] = None,
    oracle_gold_ids: Optional[list[str]] = None,
    weights_config: Optional[dict[str, Any]] = None,
    systems_config: Optional[dict[str, Any]] = None,
    relation_table: Optional[dict[str, Any]] = None,
    rt_rules: Optional[dict[str, Any]] = None,
    precomputed_rq_hybrid: Optional[dict[str, float]] = None,
    precomputed_rq_bm25: Optional[dict[str, float]] = None,
    precomputed_rq_dense: Optional[dict[str, float]] = None,
) -> PipelineOutput:
    """Execute the retrieval, scoring, fusion, and flag detection stages for one question (Section 5.2 & 12.2)."""
    sys_cfg_all = systems_config or load_json_config("configs/systems.json")
    rel_tbl = relation_table or load_relation_table("configs/relation_table.json")
    rt_cfg = rt_rules or load_rt_rules("configs/rt_rules.json")

    sys_spec = sys_cfg_all.get("systems", {}).get(
        system_id,
        {"mode": "fusion_b2", "annotation": "A2"},
    )
    mode = sys_spec.get("mode", "fusion_b2")
    annotation_level = sys_spec.get("annotation", "A2")
    tau = float(sys_cfg_all.get("insufficient_evidence_tau", 0.35))

    pool = filter_candidate_pool(patient_events, as_of_date, patient_id=patient_id)
    ev_by_id = {e.event_id: e for e in pool}

    if intent_override is not None:
        intent: IntentType = intent_override
        intent_source: Literal["rules", "oracle"] = "oracle"
    else:
        intent = classify_intent_by_rules(question_text, rt_cfg)
        intent_source = "rules"

    states_by_id: dict[str, TemporalStateType] = {
        ev.event_id: compute_state(ev, as_of_date, pool) for ev in pool
    }

    if mode == "bm25_only":
        rq = (
            precomputed_rq_bm25
            if precomputed_rq_bm25 is not None
            else rq_scores(pool, question_text, as_of_date, mode="bm25_only", patient_id=patient_id)
        )
    elif mode == "dense_only":
        rq = (
            precomputed_rq_dense
            if precomputed_rq_dense is not None
            else rq_scores(pool, question_text, as_of_date, mode="dense_only", patient_id=patient_id)
        )
    else:
        rq = (
            precomputed_rq_hybrid
            if precomputed_rq_hybrid is not None
            else rq_scores(pool, question_text, as_of_date, mode="hybrid", patient_id=patient_id)
        )

    question_only_anchors = extract_question_anchors(question_text, rel_tbl, pool)
    anchors, anchor_encounters = build_anchor_set(
        question_text, pool, rq, rel_tbl, top_rq_n=5
    )
    rh = rh_scores(pool, anchors, rel_tbl, anchor_encounters=anchor_encounters)

    rt_anchor_input = question_only_anchors if question_only_anchors else anchors
    rt = rt_scores(
        pool,
        intent=intent,
        as_of_date=as_of_date,
        rules=rt_cfg,
        anchors=rt_anchor_input,
        window_start=window_start,
        precomputed_states=states_by_id,
    )

    if mode in ("bm25_only", "dense_only", "hybrid_rq_only"):
        weights = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}
        selected_ranked_ids, fused_scores = rank_by_fused_score(
            rq, rh, rt, weights, k=k, events_by_id=ev_by_id
        )
    elif mode == "recency_only":
        pool_sz = int(sys_cfg_all.get("recency_pool_size", 50))
        selected_ranked_ids, fused_scores = select_recency_only(
            pool, rq, k=k, pool_size=pool_sz
        )
    elif mode == "status_filter_only":
        min_rt = float(sys_cfg_all.get("status_filter_min_rt", 0.2))
        selected_ranked_ids, fused_scores = select_status_filter_only(
            pool, rq, rt, k=k, min_rt=min_rt
        )
    elif mode == "fusion_b3":
        w_cfg = (weights_config or {}).get("B3", {}).get(
            intent, {"alpha": 0.6, "beta": 0.4, "gamma": 0.0}
        )
        selected_ranked_ids, fused_scores = rank_by_fused_score(
            rq, rh, rt, w_cfg, k=k, events_by_id=ev_by_id
        )
    elif mode == "fusion_b2":
        w_cfg = (weights_config or {}).get("B2", {}).get(
            intent, {"alpha": 0.4, "beta": 0.2, "gamma": 0.4}
        )
        selected_ranked_ids, fused_scores = rank_by_fused_score(
            rq, rh, rt, w_cfg, k=k, events_by_id=ev_by_id
        )
    elif mode == "oracle":
        gold_list = [eid for eid in (oracle_gold_ids or []) if eid in ev_by_id]
        selected_ranked_ids = gold_list[:k]
        fused_scores = {
            eid: (1.0 if eid in set(selected_ranked_ids) else 0.0) for eid in ev_by_id
        }
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    q_anchor_set = {a.lower().strip() for a in question_only_anchors if a.strip()}
    if q_anchor_set:
        relevant_concepts = set(q_anchor_set)
        concepts_map = rel_tbl.get("concepts", {})
        q_low = question_text.lower()
        asks_meds_or_tx = any(w in q_low for w in ["medication", "medications", "prescribed", "taking", "treat", "treating", "dose", "dosage"])
        for anc in q_anchor_set:
            spec = concepts_map.get(anc, {})
            is_cond = any(
                e.concept.lower().strip() == anc and e.event_type in {"condition", "diagnosis_at_visit"}
                for e in pool
            )
            if is_cond and asks_meds_or_tx:
                for rel_c in spec.get("related_concepts", []):
                    relevant_concepts.add(rel_c.lower().strip())
    else:
        relevant_concepts = set()

    conflicts, prov_notes = detect_conflicts(
        selected_ranked_ids, pool, as_of_date, relevant_concepts=q_anchor_set
    )
    u_flags, insuff_ev = uncertainty_flags(
        selected_ranked_ids,
        pool,
        rq,
        question_only_anchors,
        tau=tau,
        question_text=question_text,
        relevant_concepts=relevant_concepts,
    )
    if insuff_ev:
        conflicts = []
        u_flags = []
    flags_container = FlagsContainer(
        conflicts=conflicts,
        uncertainty=u_flags,
        insufficient_evidence=insuff_ev,
        provenance_notes=prov_notes,
    )

    selected_set = set(selected_ranked_ids)
    candidates_list: list[CandidateScore] = []
    for eid in sorted(
        ev_by_id.keys(),
        key=lambda x: (-fused_scores.get(x, 0.0), -rq.get(x, 0.0), x),
    ):
        candidates_list.append(
            CandidateScore(
                event_id=eid,
                state=states_by_id[eid],
                rq=round(float(rq.get(eid, 0.0)), 4),
                rh=round(float(rh.get(eid, 0.0)), 4),
                rt=round(float(rt.get(eid, 0.0)), 4),
                score=round(float(fused_scores.get(eid, 0.0)), 4),
                selected=(eid in selected_set),
            )
        )

    chrono_ids = order_chronologically(selected_ranked_ids, ev_by_id)
    context_text = assemble_context_text(
        chrono_ids,
        ev_by_id,
        states_by_id,
        flags_container,
        annotation_level=annotation_level,
    )

    return PipelineOutput(
        patient_id=patient_id,
        as_of_date=as_of_date,
        system_id=system_id,
        intent=intent,
        intent_source=intent_source,
        candidates=candidates_list,
        selected_event_ids=selected_ranked_ids,
        flags=flags_container,
        context_text=context_text,
        background_chunk_ids=[],
    )


def run_pipeline(
    patient_id: str,
    question: str,
    as_of_date: str,
    system_config: dict[str, Any] | str = "B2",
    patient_events: Optional[list[Event]] = None,
    generate_answer: bool = False,
) -> tuple[PipelineOutput, Optional[AnswerJSON]]:
    """Shared top-level pipeline contract `run_pipeline(patient_id, question, as_of, system_config)` (Section 12.2)."""
    if isinstance(system_config, str):
        sys_id = system_config
        cfg_dict: dict[str, Any] = {"system_id": sys_id}
    else:
        cfg_dict = system_config
        sys_id = str(cfg_dict.get("system_id", "B2"))

    k = int(cfg_dict.get("k", 10))
    window_start = cfg_dict.get("window_start")
    intent_override = cfg_dict.get("intent_override")
    oracle_gold_ids = cfg_dict.get("oracle_gold_ids")

    if patient_events is None:
        from src.schemas import load_jsonl

        events_path = (
            "data/processed/events.jsonl"
            if patient_id.startswith("R")
            else "data/synthetic/events.jsonl"
        )
        all_evs = load_jsonl(Event, events_path)
        patient_events = [e for e in all_evs if e.patient_id == patient_id]

    weights_path = Path("configs/weights.json")
    weights_cfg = load_json_config(weights_path) if weights_path.exists() else None

    pipe_out = run_retrieval_stage(
        patient_id=patient_id,
        question_text=question,
        as_of_date=as_of_date,
        patient_events=patient_events,
        system_id=sys_id,
        k=k,
        window_start=window_start,
        intent_override=intent_override,
        oracle_gold_ids=oracle_gold_ids,
        weights_config=weights_cfg,
    )

    ans_json: Optional[AnswerJSON] = None
    if generate_answer:
        from src.generation.groq_client import generate

        gen_cfg = {
            "provider": cfg_dict.get("provider", "groq"),
            "model_id": cfg_dict.get("model_id"),
            "patient_id": patient_id,
            "question_text": question,
            "as_of_date": as_of_date,
            "window_start": window_start,
            "preferred_key_label": cfg_dict.get("preferred_key_label"),
            "allow_cross_provider_fallback": cfg_dict.get("allow_cross_provider_fallback", False),
        }
        ans_json = generate(pipe_out.context_text, gen_cfg)

    return pipe_out, ans_json
