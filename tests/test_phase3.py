"""Phase 3 unit tests: Temporal state computation, Rt intent rows, Rh relation scoring,
availability rule (`observed_time <= as_of_date`), patient scoping, fusion normalisation and
weight validation, C1-C3 conflict rules, U1-U5 uncertainty flags, and retrieval metrics
on hand-computed examples (Sections 10.6 & 11.5).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.evaluation.bootstrap import paired_bootstrap
from src.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    mrr_score,
    ndcg_at_k,
    recall_at_k,
    stale_evidence_rate,
)
from src.flags.conflict_detection import detect_conflicts
from src.flags.uncertainty import uncertainty_flags
from src.fusion.fusion import compute_fused_scores, validate_weights
from src.fusion.selection import order_chronologically, select
from src.pipeline import run_pipeline
from src.retrieval.rrf import min_max_normalise, rq_scores
from src.schemas import Event
from src.temporal.history_relevance import load_relation_table, rh_scores
from src.temporal.state import compute_state, filter_candidate_pool
from src.temporal.temporal_scoring import classify_intent_by_rules, load_rt_rules, rt_scores


def _make_event(
    event_id: str,
    patient_id: str = "S_999",
    event_type: str = "medication_order",
    concept: str = "metformin",
    concept_class: str | None = "Biguanide Antidiabetic",
    text: str = "Metformin 500 mg",
    observed_time: str | None = "2024-01-10",
    valid_start: str | None = "2024-01-10",
    valid_end: str | None = None,
    date_precision: str = "day",
    status_basis: str = "interval",
    source_table: str | None = None,
    encounter_id: str | None = "ENC_1",
    attributes: dict | None = None,
) -> Event:
    return Event(
        event_id=event_id,
        patient_id=patient_id,
        track="S" if patient_id.startswith("S") else "R",
        event_type=event_type,  # type: ignore[arg-type]
        concept=concept,
        concept_class=concept_class,
        text=text,
        observed_time=observed_time,
        valid_start=valid_start,
        valid_end=valid_end,
        date_precision=date_precision,  # type: ignore[arg-type]
        time_is_shifted=patient_id.startswith("R"),
        status_basis=status_basis,  # type: ignore[arg-type]
        source_type="synthetic" if patient_id.startswith("S") else "real",
        source_dataset="test",
        source_table=source_table,
        source_rows=[],
        encounter_id=encounter_id,
        confidence=1.0,
        attributes=attributes or {},
    )


def test_temporal_state_all_states_and_edge_cases():
    """Verify ACTIVE, ENDED, SUPERSEDED, PERSISTENT, UNKNOWN, LISTED, and NOT_LISTED_LATEST (Section 7.3 & 10.6)."""
    as_of = "2025-06-01"
    e_active = _make_event("E1", concept="metformin", valid_start="2024-01-10", valid_end=None)
    e_ended = _make_event(
        "E2", concept="amlodipine", valid_start="2023-01-01", valid_end="2024-01-01"
    )
    e_lab_old = _make_event(
        "E3",
        event_type="lab_result",
        concept="hba1c",
        observed_time="2024-01-01",
        valid_start="2024-01-01",
        valid_end="2024-01-01",
    )
    e_lab_new = _make_event(
        "E4",
        event_type="lab_result",
        concept="hba1c",
        observed_time="2025-01-01",
        valid_start="2025-01-01",
        valid_end="2025-01-01",
    )
    e_allergy = _make_event(
        "E5", event_type="allergy", concept="penicillin allergy", valid_start="2021-03-01"
    )
    e_unknown_prec = _make_event(
        "E6", concept="atorvastatin", observed_time="2024-06", valid_start="2024-06", date_precision="month"
    )
    e_missing_start = _make_event(
        "E7", concept="gerd", observed_time=None, valid_start=None, date_precision="unknown"
    )

    pool = [e_active, e_ended, e_lab_old, e_lab_new, e_allergy, e_unknown_prec, e_missing_start]
    assert compute_state(e_active, as_of, pool) == "ACTIVE"
    assert compute_state(e_ended, as_of, pool) == "ENDED"
    assert compute_state(e_lab_old, as_of, pool) == "SUPERSEDED"
    assert compute_state(e_lab_new, as_of, pool) == "ENDED"
    assert compute_state(e_allergy, as_of, pool) == "PERSISTENT"
    assert compute_state(e_unknown_prec, as_of, pool) == "UNKNOWN"
    assert compute_state(e_missing_start, as_of, pool) == "UNKNOWN"

    # Snapshot basis (Track R medrecon): LISTED vs NOT_LISTED_LATEST
    r_old = _make_event(
        "R1_1",
        patient_id="R1",
        event_type="reported_home_medication",
        concept="digoxin",
        observed_time="2150-01-01",
        valid_start="2150-01-01",
        status_basis="snapshot",
        encounter_id="STAY_1",
    )
    r_new = _make_event(
        "R1_2",
        patient_id="R1",
        event_type="reported_home_medication",
        concept="aspirin",
        observed_time="2150-06-01",
        valid_start="2150-06-01",
        status_basis="snapshot",
        encounter_id="STAY_2",
    )
    r_pool = [r_old, r_new]
    assert compute_state(r_old, "2150-06-02", r_pool) == "NOT_LISTED_LATEST"
    assert compute_state(r_new, "2150-06-02", r_pool) == "LISTED"


def test_availability_and_patient_scoping():
    """Verify `observed_time <= as_of_date` and strict patient scoping (no cross-patient leak)."""
    e_past = _make_event("E1", patient_id="S_1", observed_time="2024-01-01")
    e_future = _make_event("E2", patient_id="S_1", observed_time="2025-12-01")
    e_other_patient = _make_event("E3", patient_id="S_2", observed_time="2024-01-01")

    filtered = filter_candidate_pool([e_past, e_future, e_other_patient], "2025-01-01", patient_id="S_1")
    assert [e.event_id for e in filtered] == ["E1"]

    scores = rq_scores(
        [e_past, e_future, e_other_patient],
        "Metformin",
        "2025-01-01",
        mode="bm25_only",
        patient_id="S_1",
    )
    assert list(scores.keys()) == ["E1"]


def test_rh_relation_table_scoring():
    """Verify Rh gives 1.0 for same concept, 0.7 for related concept/class, 0.5 for same encounter, 0.0 for unrelated."""
    rel_table = load_relation_table("configs/relation_table.json")
    e_same = _make_event("E1", concept="metformin", encounter_id="ENC_A")
    e_related = _make_event(
        "E2", event_type="lab_result", concept="hba1c", concept_class="Glycemic Control Laboratory Test", encounter_id="ENC_B"
    )
    e_same_enc = _make_event(
        "E3", event_type="medication_order", concept="acetaminophen", concept_class="Analgesics", encounter_id="ENC_A"
    )
    e_unrelated = _make_event(
        "E4", event_type="medication_order", concept="cetirizine", concept_class="Antihistamines", encounter_id="ENC_C"
    )

    scores = rh_scores(
        [e_same, e_related, e_same_enc, e_unrelated],
        anchors=["metformin"],
        relation_table=rel_table,
        anchor_encounters={"ENC_A"},
    )
    assert scores["E1"] == 1.0
    assert scores["E2"] == 0.7
    assert scores["E3"] == 0.5
    assert scores["E4"] == 0.0


def test_rt_scoring_all_five_intents():
    """Verify every intent row of the Rt table in Section 7.3."""
    as_of = "2025-06-01"
    e1 = _make_event(
        "E1",
        event_type="lab_result",
        concept="hba1c",
        observed_time="2023-01-01",
        valid_start="2023-01-01",
        valid_end="2023-01-01",
    )
    e2 = _make_event(
        "E2",
        event_type="lab_result",
        concept="hba1c",
        observed_time="2024-01-01",
        valid_start="2024-01-01",
        valid_end="2024-01-01",
    )
    e3 = _make_event(
        "E3",
        event_type="lab_result",
        concept="hba1c",
        observed_time="2025-01-01",
        valid_start="2025-01-01",
        valid_end="2025-01-01",
    )
    e_active = _make_event("E4", concept="metformin", valid_start="2024-05-01", valid_end=None)
    e_persist = _make_event("E5", event_type="allergy", concept="penicillin allergy", valid_start="2022-01-01")
    e_ended = _make_event("E6", concept="lisinopril", valid_start="2022-01-01", valid_end="2023-01-01")
    pool = [e1, e2, e3, e_active, e_persist, e_ended]

    # 1. current_state
    rt_curr = rt_scores(pool, "current_state", as_of, anchors=["metformin"])
    assert rt_curr["E4"] == 1.0  # ACTIVE
    assert rt_curr["E5"] == 1.0  # PERSISTENT
    assert rt_curr["E1"] == 0.2  # SUPERSEDED
    assert rt_curr["E6"] == 0.1  # ENDED

    # 2. change_over_time (E1 is earliest, E2 intermediate, E3 latest of hba1c series)
    rt_cot = rt_scores(pool, "change_over_time", as_of, anchors=["hba1c"])
    assert rt_cot["E1"] == 1.0
    assert rt_cot["E2"] == 0.7
    assert rt_cot["E3"] == 1.0
    assert rt_cot["E4"] == 0.1

    # 3. first_occurrence (E1 is earliest of hba1c, E2 & E3 are later)
    rt_fo = rt_scores(pool, "first_occurrence", as_of, anchors=["hba1c"])
    assert rt_fo["E1"] == 1.0
    assert rt_fo["E2"] == 0.3
    assert rt_fo["E3"] == 0.3
    assert rt_fo["E4"] == 0.1

    # 4. persistent_fact
    rt_pf = rt_scores(pool, "persistent_fact", as_of)
    assert rt_pf["E5"] == 1.0
    assert rt_pf["E4"] == 0.5
    assert rt_pf["E6"] == 0.2

    # 5. historical
    rt_hist = rt_scores(pool, "historical", as_of, window_start="2024-01-01")
    assert rt_hist["E2"] == 1.0
    assert rt_hist["E6"] == 0.1  # ended in 2023 before window_start 2024-01-01


def test_fusion_normalisation_and_weights():
    """Verify fusion min-max normalisation, weight sum checks, and chronological prompt ordering."""
    with pytest.raises(ValueError):
        validate_weights({"alpha": 0.5, "beta": 0.5, "gamma": 0.5})
    with pytest.raises(ValueError):
        validate_weights({"alpha": 1.2, "beta": -0.2, "gamma": 0.0})

    rq = {"E1": 10.0, "E2": 20.0}
    rh = {"E1": 1.0, "E2": 0.0}
    rt = {"E1": 1.0, "E2": 0.1}
    fused = compute_fused_scores(rq, rh, rt, {"alpha": 0.4, "beta": 0.2, "gamma": 0.4})
    # E1: rq_norm = 0.0 -> 0.4*0 + 0.2*1.0 + 0.4*1.0 = 0.6
    # E2: rq_norm = 1.0 -> 0.4*1.0 + 0.2*0 + 0.4*0.1 = 0.44
    assert math.isclose(fused["E1"], 0.6, abs_tol=1e-6)
    assert math.isclose(fused["E2"], 0.44, abs_tol=1e-6)
    assert select(rq, rh, rt, {"alpha": 0.4, "beta": 0.2, "gamma": 0.4}, k=1) == ["E1"]


def test_conflict_rules_c1_c2_c3_and_track_r_provenance():
    """Verify C1 status conflict, C2 dose conflict, C3 date conflict, and Track R provenance note."""
    as_of = "2025-06-01"
    # C1: active vs ended on overlapping validity
    c1_a = _make_event(
        "C1_A",
        concept="apixaban",
        valid_start="2024-01-01",
        valid_end=None,
        attributes={"dose": "5 mg", "reported_status": "active", "clinical_source": "clinic"},
    )
    c1_b = _make_event(
        "C1_B",
        concept="apixaban",
        valid_start="2024-01-01",
        valid_end="2024-01-10",
        attributes={"dose": "5 mg", "reported_status": "ended", "clinical_source": "pharmacy"},
    )
    conf_c1, _ = detect_conflicts(["C1_A"], [c1_a, c1_b], as_of)
    assert any(c.rule == "C1" for c in conf_c1)

    # C2: different doses on overlapping active intervals
    c2_a = _make_event(
        "C2_A",
        concept="furosemide",
        valid_start="2024-02-01",
        valid_end=None,
        attributes={"dose": "20 mg", "clinical_source": "pcp"},
    )
    c2_b = _make_event(
        "C2_B",
        concept="furosemide",
        valid_start="2024-02-05",
        valid_end=None,
        attributes={"dose": "40 mg", "clinical_source": "cardiology"},
    )
    conf_c2, _ = detect_conflicts(["C2_A"], [c2_a, c2_b], as_of)
    assert any(c.rule == "C2" for c in conf_c2)

    # C3: dates differing by > 30 days across sources
    c3_a = _make_event(
        "C3_A",
        event_type="condition",
        concept="hypothyroidism",
        observed_time="2023-01-01",
        valid_start="2023-01-01",
        valid_end="2023-01-01",
        attributes={"clinical_source": "hospital_a"},
    )
    c3_b = _make_event(
        "C3_B",
        event_type="condition",
        concept="hypothyroidism",
        observed_time="2023-04-15",
        valid_start="2023-04-15",
        valid_end="2023-04-15",
        attributes={"clinical_source": "hospital_b"},
    )
    conf_c3, _ = detect_conflicts(["C3_A"], [c3_a, c3_b], as_of)
    assert any(c.rule == "C3" for c in conf_c3)

    # Track R medrecon vs pyxis -> informational provenance note, NOT conflict
    r_med = _make_event(
        "R_M",
        patient_id="R1",
        event_type="reported_home_medication",
        concept="aspirin",
        status_basis="snapshot",
        source_table="medrecon",
    )
    r_pyx = _make_event(
        "R_P",
        patient_id="R1",
        event_type="ed_administered_medication",
        concept="aspirin",
        status_basis="interval",
        source_table="pyxis",
    )
    conf_r, notes_r = detect_conflicts(["R_M", "R_P"], [r_med, r_pyx], "2150-01-01")
    assert len(conf_r) == 0
    assert len(notes_r) == 1
    assert notes_r[0].note_type == "medrecon_vs_pyxis"


def test_uncertainty_flags_u1_to_u5():
    """Verify U1 (open med end), U2 (coarse date), U3 (missing date), U4 (snapshot), and U5 (insufficient evidence)."""
    e_u1 = _make_event("U1_E", event_type="medication_order", concept="metformin", valid_end=None)
    e_u2 = _make_event(
        "U2_E", event_type="condition", concept="asthma", observed_time="2023-05", valid_start="2023-05", date_precision="month"
    )
    e_u3 = _make_event(
        "U3_E", event_type="condition", concept="gerd", observed_time=None, valid_start=None, date_precision="unknown"
    )
    e_u4 = _make_event(
        "U4_E",
        patient_id="R1",
        event_type="reported_home_medication",
        concept="lisinopril",
        status_basis="snapshot",
        source_table="medrecon",
    )
    pool = [e_u1, e_u2, e_u3, e_u4]
    rq = {"U1_E": 0.9, "U2_E": 0.5, "U3_E": 0.4, "U4_E": 0.8}

    flags, insuff = uncertainty_flags(
        ["U1_E", "U2_E", "U3_E", "U4_E"],
        pool,
        rq,
        question_anchors=["metformin"],
        tau=0.35,
    )
    flag_codes = {f.flag for f in flags}
    assert flag_codes == {"U1", "U2", "U3", "U4"}
    assert insuff is False

    # U5: Querying an unrecorded concept ("tacrolimus") not in pool
    _, insuff_u5 = uncertainty_flags(
        ["U1_E"], pool, rq, question_anchors=["tacrolimus"], tau=0.35
    )
    assert insuff_u5 is True


def test_retrieval_metrics_hand_computed():
    """Verify Recall@k, HitRate@k, MRR, nDCG@k, and Stale-Evidence Rate on hand-computed examples (Section 10.6)."""
    ranked = ["E_dist1", "E_gold1", "E_other", "E_gold2", "E_dist2"]
    gold = ["E_gold1", "E_gold2"]
    distractors = ["E_dist1", "E_dist2"]

    # Recall@2: 1 of 2 gold found -> 0.5; Recall@5: 2 of 2 found -> 1.0
    assert math.isclose(recall_at_k(ranked, gold, 2), 0.5)
    assert math.isclose(recall_at_k(ranked, gold, 5), 1.0)
    assert math.isclose(hit_rate_at_k(ranked, gold, 2), 1.0)
    # First gold is at rank 2 -> MRR = 1/2 = 0.5
    assert math.isclose(mrr_score(ranked, gold), 0.5)
    # nDCG@5: hits at rank 2 (1/log2(3)) and rank 4 (1/log2(5)); ideal hits at rank 1 (1.0) and rank 2 (1/log2(3))
    expected_dcg = (1.0 / math.log2(3.0)) + (1.0 / math.log2(5.0))
    expected_idcg = 1.0 + (1.0 / math.log2(3.0))
    assert math.isclose(ndcg_at_k(ranked, gold, 5), expected_dcg / expected_idcg, abs_tol=1e-6)
    # Stale-evidence rate: 2 distractors out of 5 selected -> 0.4
    assert math.isclose(stale_evidence_rate(ranked, distractors, k=5), 0.4)

    # Paired bootstrap check
    boot = paired_bootstrap([1.0, 1.0, 0.8, 0.9], [0.5, 0.5, 0.4, 0.5], n_resamples=500, seed=42)
    assert boot["diff_mean"] > 0.35
    assert boot["ci_lower"] > 0.3
