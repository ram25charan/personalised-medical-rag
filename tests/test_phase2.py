"""Phase 2 unit tests: Generator determinism, schema validation of synthetic events,
questions, and gold labels, disjoint dev/test patient splits, gold/distractor integrity,
and relation table validation (Sections 10.6 & 11.5).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.schemas import Event, Gold, Question, load_jsonl
from src.synthetic.generator import generate


def test_generator_determinism_fixed_seed():
    """Verify `generate(config, seed=42)` produces identical outputs across runs (Section 11.5)."""
    r_events = load_jsonl(Event, "data/processed/events.jsonl")
    ev1, q1, g1 = generate("configs/synthetic_generator.json", seed=42, track_r_events=r_events)
    ev2, q2, g2 = generate("configs/synthetic_generator.json", seed=42, track_r_events=r_events)

    assert len(ev1) == len(ev2) > 10000
    assert len(q1) == len(q2) == 180
    assert len(g1) == len(g2) == 180

    assert [e.model_dump() for e in ev1[:50]] == [e.model_dump() for e in ev2[:50]]
    assert [q.model_dump() for q in q1] == [q.model_dump() for q in q2]
    assert [g.model_dump() for g in g1] == [g.model_dump() for g in g2]


def test_different_seed_changes_synthetic_output():
    """Verify changing the seed changes generated synthetic timelines."""
    ev_42, _, _ = generate("configs/synthetic_generator.json", seed=42, track_r_events=None)
    ev_99, _, _ = generate("configs/synthetic_generator.json", seed=99, track_r_events=None)
    assert len(ev_42) != len(ev_99) or ev_42[0].observed_time != ev_99[0].observed_time


def test_disjoint_patient_splits_and_counts():
    """Verify dev and test patient sets are strictly disjoint and match target sizes (DD4, 11.5)."""
    questions = load_jsonl(Question, "data/benchmark/questions.jsonl")
    golds = load_jsonl(Gold, "data/benchmark/gold.jsonl")

    assert len(questions) == 180
    assert len(golds) == 180

    s_dev_q = [q for q in questions if q.track == "S" and q.split == "dev"]
    s_test_q = [q for q in questions if q.track == "S" and q.split == "test"]
    r_test_q = [q for q in questions if q.track == "R" and q.split == "test"]

    assert len(s_dev_q) == 50
    assert len(s_test_q) == 100
    assert len(r_test_q) == 30

    dev_patients = {q.patient_id for q in s_dev_q}
    test_patients = {q.patient_id for q in s_test_q}
    assert len(dev_patients & test_patients) == 0


def test_gold_and_distractor_event_integrity():
    """Verify every gold_event and distractor_event exists in that patient's event pool on or before as_of_date."""
    r_events = load_jsonl(Event, "data/processed/events.jsonl")
    s_events = load_jsonl(Event, "data/synthetic/events.jsonl")
    questions = load_jsonl(Question, "data/benchmark/questions.jsonl")
    golds = load_jsonl(Gold, "data/benchmark/gold.jsonl")

    events_by_id = {e.event_id: e for e in (r_events + s_events)}
    q_by_id = {q.question_id: q for q in questions}

    for g in golds:
        q = q_by_id[g.question_id]
        assert len(set(g.gold_events) & set(g.distractor_events)) == 0

        if g.category == "insufficient_evidence":
            assert len(g.gold_events) == 0
            assert g.expected_behavior == "abstain"
        else:
            assert len(g.gold_events) >= 1

        for eid in g.gold_events + g.distractor_events:
            assert eid in events_by_id, f"Missing event_id {eid} in events pool"
            ev = events_by_id[eid]
            assert ev.patient_id == q.patient_id, (
                f"Cross-patient leak in gold: {ev.patient_id} != {q.patient_id}"
            )
            if ev.observed_time is not None:
                assert ev.observed_time <= q.as_of_date, (
                    f"Future event in gold/distractors: {ev.observed_time} > {q.as_of_date}"
                )


def test_relation_table_frozen_and_valid():
    """Verify `configs/relation_table.json` is versioned, frozen, and structurally valid (Section 7.2 & 11.5)."""
    rel_path = Path("configs/relation_table.json")
    assert rel_path.exists()
    with rel_path.open("r", encoding="utf-8") as f:
        rel = json.load(f)

    assert rel.get("version") == "1.0"
    assert rel.get("frozen") is True
    concepts = rel.get("concepts", {})
    assert len(concepts) >= 35
    for c_name, c_info in concepts.items():
        assert c_name == c_name.lower()
        assert "concept_class" in c_info
        assert isinstance(c_info.get("synonyms"), list)
        assert isinstance(c_info.get("related_concepts"), list)
