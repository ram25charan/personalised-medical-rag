"""Phase 1 unit tests: Schema validation, raw ingestion counts, medication normalisation,
event construction, provenance, intra-stay duplicate collapsing, and timelines (Section 10.6 & 11.5).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data.events import (
    build_coverage_report,
    build_events,
    build_patient_timelines,
)
from src.data.ingestion import (
    EXPECTED_TABLE_ROWS,
    EXPECTED_UNIQUE_PATIENTS,
    assess_data_quality,
    load_raw,
    validate_raw_counts,
)
from src.data.normalisation import (
    build_gsn_canonical_map,
    clean_medication_string,
    normalise_diagnosis,
    normalise_medication,
)
from src.schemas import (
    AnswerJSON,
    CorpusChunk,
    Event,
    Gold,
    PipelineOutput,
    Question,
    ResultRecord,
    load_jsonl,
)


@pytest.fixture(scope="module")
def raw_data():
    return load_raw("data/raw/mimic_iv_ed_demo")


@pytest.fixture(scope="module")
def track_r_events(raw_data):
    return build_events(raw_data)


def test_raw_table_counts_match_spec(raw_data):
    """Verify all six tables load and match Section 6.2 row and patient counts."""
    val = validate_raw_counts(raw_data)
    assert val["valid"] is True
    assert val["unique_patients"] == EXPECTED_UNIQUE_PATIENTS
    for table_name, expected_rows in EXPECTED_TABLE_ROWS.items():
        assert len(raw_data[table_name]) == expected_rows


def test_data_quality_audit_matches_longitudinal_facts(raw_data):
    """Verify longitudinal metrics from Section 6.2."""
    report = assess_data_quality(raw_data)
    ls = report["longitudinal_summary"]
    assert ls["total_patients"] == 64
    assert ls["total_stays"] == 222
    assert ls["patients_with_2plus_stays"] == 35
    assert ls["patients_with_3plus_stays"] == 24
    assert ls["max_stays_per_patient"] == 23
    assert ls["stays_with_medrecon"] == 181
    assert ls["patients_with_2plus_medrecon_visits"] == 30
    assert ls["patients_with_both_icd9_and_icd10"] == 14


def test_medication_normalisation_brand_and_brackets(raw_data):
    """Verify brand names and bracketed names resolve via GSN / cleaning rules."""
    assert clean_medication_string("digoxin [Digitek]") == "digoxin"
    assert clean_medication_string("*depakote DR") == "depakote dr"
    assert clean_medication_string("cholecalciferol (vitamin D3) [Vitamin D3]") == "cholecalciferol"

    gsn_to_concept, gsn_to_class = build_gsn_canonical_map(
        raw_data["medrecon"], raw_data["pyxis"]
    )
    concept_386, class_386 = normalise_medication(
        "Vasotec", 386, None, gsn_to_concept, gsn_to_class
    )
    assert concept_386 == "enalapril maleate"
    assert class_386 == "ACE Inhibitors"

    concept_18, _ = normalise_medication(
        "digoxin [Digitek]", 18, None, gsn_to_concept, gsn_to_class
    )
    assert concept_18 == "digoxin"


def test_diagnosis_normalisation():
    concept, cclass = normalise_diagnosis("Essential (primary) hypertension", "I10", 10)
    assert concept == "essential (primary) hypertension"
    assert cclass == "ICD-10"


def test_event_construction_and_provenance(raw_data, track_r_events):
    """Verify event schema compliance, provenance tracking, and intra-stay duplicate collapsing."""
    assert len(track_r_events) == 5471

    event_ids = set()
    medrecon_rows_covered = set()

    for ev in track_r_events:
        assert isinstance(ev, Event)
        assert ev.track == "R"
        assert ev.source_type == "real"
        assert ev.time_is_shifted is True
        assert ev.source_dataset == "mimic_iv_ed_demo_v2.2"
        assert ev.source_table in EXPECTED_TABLE_ROWS
        assert len(ev.source_rows) >= 1
        assert ev.event_id not in event_ids
        event_ids.add(ev.event_id)

        if ev.event_type == "reported_home_medication":
            assert ev.status_basis == "snapshot"
            assert ev.valid_end is None
            for r_idx in ev.source_rows:
                medrecon_rows_covered.add(r_idx)
        else:
            assert ev.status_basis == "interval"

    assert len(medrecon_rows_covered) == 2764

    stay_med_pairs = set()
    for ev in track_r_events:
        if ev.event_type == "reported_home_medication":
            pair = (ev.encounter_id, ev.concept)
            assert pair not in stay_med_pairs
            stay_med_pairs.add(pair)


def test_timelines_and_coverage_report(track_r_events):
    """Verify per-patient chronological timelines and coverage report counts."""
    timelines = build_patient_timelines(track_r_events)
    assert len(timelines) == 64

    for pid, tl in timelines.items():
        assert tl["patient_id"] == pid
        assert tl["visit_count"] >= 1
        rel_days = [
            e["relative_day"] for e in tl["events"] if e["relative_day"] is not None
        ]
        assert rel_days == sorted(rel_days)

    cov = build_coverage_report(track_r_events)
    assert cov["longitudinal_evidence_counts"]["consecutive_medrecon_visit_pairs"] == 132
    assert cov["category_support"]["1_temporal_ordering"]["supported"] is True
    assert cov["category_support"]["6_conflicting_information"]["supported"] is False


def test_saved_jsonl_roundtrip():
    """Verify `data/processed/events.jsonl` loads cleanly through `load_jsonl(Event, ...)`."""
    events_path = Path("data/processed/events.jsonl")
    assert events_path.exists()
    loaded = load_jsonl(Event, events_path)
    assert len(loaded) == 5471


def test_all_contract_schemas_validate():
    """Verify Question, Gold, CorpusChunk, PipelineOutput, AnswerJSON, and ResultRecord contracts."""
    q = Question(
        question_id="Q_S_0031",
        patient_id="S_007",
        track="S",
        split="test",
        text="What medications is the patient currently taking?",
        as_of_date="2025-09-01",
        window_start=None,
    )
    assert q.track == "S"

    g = Gold(
        question_id="Q_S_0031",
        intent="current_state",
        category="current_vs_historical",
        anchor_concepts=["metformin"],
        gold_events=["S_007_0150"],
        distractor_events=["S_007_0091"],
        expected_answer={"type": "set", "value": ["metformin"]},
        expected_behavior="answer",
        expected_conflict=False,
        expected_flags=["U1"],
    )
    assert g.expected_answer.type == "set"

    c = CorpusChunk(
        chunk_id="C_012",
        title="HbA1c test",
        text="An HbA1c test measures blood glucose...",
        chunk_type="test_explainer",
        source_name="MedlinePlus, National Library of Medicine",
        url="https://medlineplus.gov/lab-tests/hemoglobin-a1c-hba1c-test/",
        accessed="2026-11-14",
        license_note="Public domain; acknowledge MedlinePlus",
        concepts=["hba1c", "diabetes"],
    )
    assert c.chunk_type == "test_explainer"

    po = PipelineOutput(
        patient_id="S_007",
        as_of_date="2025-09-01",
        system_id="B2",
        intent="current_state",
        intent_source="rules",
        candidates=[
            {
                "event_id": "S_007_0150",
                "state": "ACTIVE",
                "rq": 0.82,
                "rh": 1.0,
                "rt": 1.0,
                "score": 0.93,
                "selected": True,
            }
        ],
        selected_event_ids=["S_007_0150"],
        flags={
            "conflicts": [{"rule": "C1", "event_ids": ["S_007_0150", "S_007_0188"]}],
            "uncertainty": [{"flag": "U1", "event_ids": ["S_007_0150"]}],
            "insufficient_evidence": False,
        },
        context_text="[E1] S_007_0150 ...",
        background_chunk_ids=[],
    )
    assert po.selected_event_ids == ["S_007_0150"]

    ans = AnswerJSON(
        answer_text="The patient is currently taking metformin [S_007_0150].",
        status_claims=[{"concept": "metformin", "status": "ACTIVE"}],
        cited_event_ids=["S_007_0150"],
        abstain=False,
        conflict_flagged=False,
        uncertainty_noted=True,
    )
    assert ans.uncertainty_noted is True

    rr = ResultRecord(
        run_id="run_001",
        git_commit="3968708",
        config_hash="abc1234",
        system_id="B2",
        track="S",
        question_id="Q_S_0031",
        model_id="llama-3.3-70b-versatile",
        run_date="2026-11-14",
        temperature=0.0,
        k=10,
        weights={"alpha": 0.4, "beta": 0.2, "gamma": 0.4},
        intent_source="rules",
        prompt_hash="def5678",
        raw_output="{}",
        parsed_answer=ans,
        metrics={"recall_at_10": 1.0},
    )
    assert rr.k == 10

    with pytest.raises(ValidationError):
        Question(
            question_id="Q_1",
            patient_id="S_1",
            track="S",
            split="dev",
            text="?",
            as_of_date="2025-01-01",
            unknown_extra_field="forbidden",
        )
