"""Phase 5 Unit & Integration Tests (`tests/test_phase5.py`) (Sections 9, 10.6, 11.5 & Phase 5 Plan).

Tests:
  1. Demo corpus validation (`data/corpus/demo_corpus.jsonl`): schema, 100-200 word bounds,
     known concept keys, MedlinePlus/DailyMed provenance, and zero dosing/advice terms.
  2. Corpus retrieval (`src/retrieval/corpus_retrieval.py`) across 10 demo questions + off-topic
     and insufficient-evidence queries returning `[]`, with `top_k = 2`.
  3. Strict separation check (`assert_no_corpus_citation_in_claims`): blocks `[C#]` / `C_###` in patient claims.
  4. Streamlit interface components (`app/main.py`): end-to-end execution on both a synthetic (Track S)
     and a real (Track R) patient, timeline figure generation (date and relative-day axes),
     verbatim status rendering, and Track R Gemini provider blocking.
"""

from __future__ import annotations

import pytest

from app.main import (
    STATUS_STYLE_MAP,
    build_timeline_figure,
    generate_single_pinned_answer,
    load_all_events_and_questions,
    load_static_configs,
    render_status_badge_html,
)
from src.pipeline import run_retrieval_stage
from src.retrieval.corpus_retrieval import (
    assert_no_corpus_citation_in_claims,
    retrieve_corpus_passages,
)
from src.retrieval.validate_corpus import validate_demo_corpus
from src.schemas import AnswerJSON, StatusClaim
from src.temporal.state import compute_state, filter_candidate_pool


def test_demo_corpus_schema_concepts_provenance_and_advice_filter() -> None:
    report = validate_demo_corpus("data/corpus/demo_corpus.jsonl")
    assert report["valid"] is True
    assert 20 <= report["num_chunks"] <= 30
    assert report["min_words"] >= 100
    assert report["max_words"] <= 200
    assert report["advice_flags"] == []
    assert report["unknown_concepts"] == []
    assert report["by_type"]["condition_overview"] >= 5
    assert report["by_type"]["test_explainer"] >= 4
    assert report["by_type"]["drug_class_note"] >= 8


def test_corpus_retrieval_10_demo_questions_and_off_topic_threshold() -> None:
    clinical_questions = [
        ("Is the patient currently taking lisinopril or amlodipine for hypertension?", "lisinopril"),
        ("How has the patient HbA1c changed over time on metformin?", "metformin"),
        ("What was the trend in LDL cholesterol after starting atorvastatin?", "atorvastatin"),
        ("What are the most recent serum creatinine and eGFR results for chronic kidney disease?", "serum creatinine"),
        ("Is the patient taking levothyroxine for hypothyroidism and what is the TSH?", "levothyroxine"),
        ("Did the patient switch from warfarin to apixaban for atrial fibrillation?", "apixaban"),
        ("Is the patient currently prescribed furosemide for congestive heart failure?", "furosemide"),
        ("Does the patient take omeprazole or pantoprazole for gastroesophageal reflux disease?", "omeprazole"),
        ("Does the patient have any documented penicillin allergy or sulfa allergy?", "penicillin allergy"),
    ]
    for q_text, expected_concept in clinical_questions:
        results = retrieve_corpus_passages(q_text, top_k=2)
        assert 1 <= len(results) <= 2, f"Expected 1-2 passages for '{q_text}', got {len(results)}"
        assert results[0]["citation_tag"] == "[C1]"
        all_concepts = {c for r in results for c in r["chunk"].concepts}
        assert expected_concept in all_concepts, f"Expected '{expected_concept}' in retrieved concepts {all_concepts}"

    # 10th question: off-topic query must return empty list so no background box is rendered
    off_topic = retrieve_corpus_passages(
        "What is the weather forecast in Tokyo and how do you bake sourdough bread?",
        top_k=2,
    )
    assert off_topic == []

    # Insufficient evidence (U5) must also suppress the background box
    u5_res = retrieve_corpus_passages(
        "Is the patient taking metformin?",
        insufficient_evidence=True,
        top_k=2,
    )
    assert u5_res == []


def test_corpus_separation_guard() -> None:
    valid_ans = AnswerJSON(
        answer_text="The patient is currently taking metformin [E1].",
        status_claims=[StatusClaim(concept="metformin", status="ACTIVE")],
        cited_event_ids=["S_001_0005"],
        abstain=False,
        conflict_flagged=False,
        uncertainty_noted=False,
    )
    assert assert_no_corpus_citation_in_claims(valid_ans) is True

    invalid_id_ans = AnswerJSON(
        answer_text="Metformin is used for diabetes.",
        status_claims=[StatusClaim(concept="metformin", status="ACTIVE")],
        cited_event_ids=["C_005"],
    )
    with pytest.raises(ValueError, match="Separation violation"):
        assert_no_corpus_citation_in_claims(invalid_id_ans)

    invalid_text_ans = AnswerJSON(
        answer_text="The patient is taking metformin [C1].",
        status_claims=[],
        cited_event_ids=["S_001_0005"],
    )
    with pytest.raises(ValueError, match="Separation violation"):
        assert_no_corpus_citation_in_claims(invalid_text_ans)


def test_app_end_to_end_real_and_synthetic_patients_and_track_r_gemini_block() -> None:
    bundle = load_all_events_and_questions()
    configs = load_static_configs()

    # 1. Synthetic patient (Track S) end-to-end retrieval + timeline + cached generation
    s_qs = [q for qlist in bundle["q_by_patient"].values() for q in qlist if q.track == "S" and q.split == "test"]
    assert s_qs
    sq = s_qs[0]
    s_events = bundle["by_patient"][sq.patient_id]

    pipe_s_b2 = run_retrieval_stage(
        patient_id=sq.patient_id,
        question_text=sq.text,
        as_of_date=sq.as_of_date,
        patient_events=s_events,
        system_id="B2",
        k=10,
        window_start=sq.window_start,
        weights_config=configs["weights"],
        systems_config=configs["systems"],
        relation_table=configs["relation_table"],
        rt_rules=configs["rt_rules"],
    )
    pool_s = filter_candidate_pool(s_events, sq.as_of_date, patient_id=sq.patient_id)
    states_s = {e.event_id: compute_state(e, sq.as_of_date, pool_s) for e in pool_s}
    fig_s = build_timeline_figure(
        patient_events=s_events,
        as_of_date=sq.as_of_date,
        selected_rank_map={eid: i + 1 for i, eid in enumerate(pipe_s_b2.selected_event_ids)},
        states_at_tq=states_s,
        track="S",
        use_relative_days=False,
    )
    assert len(fig_s.data) >= 1

    # Verify cached generation works on Track S
    ans_s, meta_s = generate_single_pinned_answer(
        pipe_s_b2,
        sq.text,
        sq.window_start,
        provider="groq",
        model_id="openai/gpt-oss-120b",
        track="S",
    )
    assert ans_s is not None
    assert meta_s.get("cached") is True

    # 2. Real patient (Track R) end-to-end retrieval + relative-day timeline + Gemini block
    r_qs = [q for qlist in bundle["q_by_patient"].values() for q in qlist if q.track == "R"]
    assert r_qs
    rq = r_qs[0]
    r_events = bundle["by_patient"][rq.patient_id]

    pipe_r_b2 = run_retrieval_stage(
        patient_id=rq.patient_id,
        question_text=rq.text,
        as_of_date=rq.as_of_date,
        patient_events=r_events,
        system_id="B2",
        k=10,
        window_start=rq.window_start,
        weights_config=configs["weights"],
        systems_config=configs["systems"],
        relation_table=configs["relation_table"],
        rt_rules=configs["rt_rules"],
    )
    pool_r = filter_candidate_pool(r_events, rq.as_of_date, patient_id=rq.patient_id)
    states_r = {e.event_id: compute_state(e, rq.as_of_date, pool_r) for e in pool_r}
    fig_r = build_timeline_figure(
        patient_events=r_events,
        as_of_date=rq.as_of_date,
        selected_rank_map={eid: i + 1 for i, eid in enumerate(pipe_r_b2.selected_event_ids)},
        states_at_tq=states_r,
        track="R",
        use_relative_days=True,
    )
    assert len(fig_r.data) >= 1

    # Verify Gemini is blocked on Track R
    ans_r_blocked, meta_r_blocked = generate_single_pinned_answer(
        pipe_r_b2,
        rq.text,
        rq.window_start,
        provider="gemini",
        model_id="gemini-flash-lite-latest",
        track="R",
    )
    assert ans_r_blocked is None
    assert meta_r_blocked.get("error_type") == "blocked_provider"

    # 3. Verbatim status badge check
    for st_name in ("ACTIVE", "ENDED", "SUPERSEDED", "PERSISTENT", "UNKNOWN", "LISTED", "NOT_LISTED_LATEST"):
        assert st_name in STATUS_STYLE_MAP
        badge_html = render_status_badge_html(st_name)
        assert st_name in badge_html
