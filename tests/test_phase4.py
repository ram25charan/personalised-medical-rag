"""Phase 4 Unit Tests (`tests/test_phase4.py`) (Sections 7.5-7.7, 8.6, 11 Phase 4).

Tests:
  1. JSON answer parsing (`parse_and_validate_answer_json`) for valid, fenced, and malformed outputs.
  2. Disk cache round-trip (`save_cached_response`, `load_cached_response`) and `MultiKeyLLMClient` cache hits.
  3. Automatic intra-provider API key rotation (`groq-1 -> groq-2`) on simulated HTTP 429 rate limit.
  4. Context assembly across annotation levels `A0`, `A1`, and `A2`.
  5. Deterministic generation-level answer metrics (`answer_correctness`, `temporal_consistency`,
     `contradiction_rate`, `citation_precision_recall_f1`, `abstention_accuracy`,
     `conflict_flag_accuracy`, `uncertainty_noted_accuracy`, `grounded_faithfulness`).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.evaluation.answer_metrics import (
    abstention_accuracy,
    answer_correctness,
    citation_precision_recall_f1,
    conflict_flag_accuracy,
    contradiction_rate,
    evaluate_single_answer,
    grounded_faithfulness,
    temporal_consistency,
    uncertainty_noted_accuracy,
)
from src.generation.cache import load_cached_response, save_cached_response
from src.generation.groq_client import MultiKeyLLMClient, parse_and_validate_answer_json
from src.pipeline import assemble_context_text
from src.schemas import (
    AnswerJSON,
    ConflictFlag,
    Event,
    ExpectedAnswer,
    FlagsContainer,
    Gold,
    StatusClaim,
    UncertaintyFlag,
)


def _make_dummy_event(
    eid: str = "S_001_0001",
    concept: str = "amlodipine",
    obs: str = "2022-06-11",
    v_start: str = "2022-06-11",
    v_end: str | None = "2022-09-09",
    text: str = "amlodipine 10 mg discontinued on 2022-09-09.",
) -> Event:
    return Event(
        event_id=eid,
        patient_id="S_001",
        track="S",
        event_type="medication_order",
        concept=concept,
        concept_class="Calcium Channel Blockers",
        text=text,
        observed_time=obs,
        valid_start=v_start,
        valid_end=v_end,
        date_precision="day",
        time_is_shifted=False,
        status_basis="interval",
        source_type="synthetic",
        source_dataset="custom_generator",
    )


def test_parse_and_validate_answer_json_variants() -> None:
    raw_clean = (
        '{"answer_text": "Amlodipine was discontinued on 2022-09-09.", '
        '"status_claims": [{"concept": "Amlodipine", "status": "ended"}], '
        '"cited_event_ids": ["[S_001_0001]", "S_001_0001"], '
        '"abstain": false, "conflict_flagged": false, "uncertainty_noted": false}'
    )
    parsed = parse_and_validate_answer_json(raw_clean)
    assert parsed.answer_text == "Amlodipine was discontinued on 2022-09-09."
    assert len(parsed.status_claims) == 1
    assert parsed.status_claims[0].concept == "amlodipine"
    assert parsed.status_claims[0].status == "ENDED"
    assert parsed.cited_event_ids == ["S_001_0001"]
    assert parsed.abstain is False

    raw_fenced = f"Here is the JSON response:\n```json\n{raw_clean}\n```\nThank you."
    parsed_fenced = parse_and_validate_answer_json(raw_fenced)
    assert parsed_fenced.cited_event_ids == ["S_001_0001"]

    with pytest.raises(ValueError):
        parse_and_validate_answer_json("This is not JSON at all.")

    with pytest.raises(ValueError):
        parse_and_validate_answer_json("")


def test_cache_round_trip(tmp_path: Path) -> None:
    ans = AnswerJSON(
        answer_text="Cached test answer.",
        status_claims=[StatusClaim(concept="lisinopril", status="ACTIVE")],
        cited_event_ids=["S_001_0002"],
        abstain=False,
        conflict_flagged=False,
        uncertainty_noted=True,
    )
    save_cached_response(
        model_id="openai/gpt-oss-120b",
        prompt_hash="abc123def456",
        raw_output='{"answer_text": "Cached test answer."}',
        parsed_answer=ans.model_dump(mode="json"),
        metadata={"key_label": "groq-1"},
        cache_dir=tmp_path,
    )
    loaded = load_cached_response("openai/gpt-oss-120b", "abc123def456", cache_dir=tmp_path)
    assert loaded is not None
    assert loaded["parsed_answer"]["answer_text"] == "Cached test answer."
    assert loaded["metadata"]["key_label"] == "groq-1"


def test_multi_key_rotation_on_429(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GROQ_API_KEY_1=gsk_fake_key_1\nGROQ_API_KEY_2=gsk_fake_key_2\n",
        encoding="utf-8",
    )
    client = MultiKeyLLMClient(
        provider="groq",
        model_id="openai/gpt-oss-120b",
        cache_dir=tmp_path / "cache",
        env_path=env_file,
        min_request_interval_sec=0.0,
    )

    call_log: list[str] = []

    def fake_groq_http(api_key: str, model_id: str, sys_p: str, usr_p: str, temperature: float = 0.0):
        call_log.append(api_key)
        if api_key == "gsk_fake_key_1":
            raise RuntimeError("Groq HTTP 429: Rate limit reached for key 1")
        return (
            '{"answer_text": "Rotated to key 2", "status_claims": [], "cited_event_ids": ["S_001_0001"], '
            '"abstain": false, "conflict_flagged": false, "uncertainty_noted": false}',
            {"provider": "groq", "usage": {"total_tokens": 100}},
        )

    with patch("src.generation.groq_client._call_groq_http", side_effect=fake_groq_http):
        with patch("time.sleep", return_value=None):
            parsed, _, _, meta = client.generate_from_prompts("sys", "user", use_cache=True)

    assert parsed.answer_text == "Rotated to key 2"
    assert meta["key_label"] == "groq-2"
    assert call_log == ["gsk_fake_key_1", "gsk_fake_key_2"]


def test_context_assembly_annotation_levels() -> None:
    ev = _make_dummy_event()
    events_by_id = {ev.event_id: ev}
    states_by_id = {ev.event_id: "ENDED"}
    flags = FlagsContainer(
        conflicts=[ConflictFlag(rule="C1", event_ids=["S_001_0001", "S_001_0002"], description="Conflict")],
        uncertainty=[UncertaintyFlag(flag="U1", event_ids=["S_001_0001"], description="No end date")],
        insufficient_evidence=False,
    )

    ctx_a0 = assemble_context_text(["S_001_0001"], events_by_id, states_by_id, flags, "A0")
    assert "status=ENDED" not in ctx_a0
    assert "[DETECTED CONFLICTS]" not in ctx_a0

    ctx_a1 = assemble_context_text(["S_001_0001"], events_by_id, states_by_id, flags, "A1")
    assert "status=ENDED" in ctx_a1
    assert "[DETECTED CONFLICTS]" not in ctx_a1

    ctx_a2 = assemble_context_text(["S_001_0001"], events_by_id, states_by_id, flags, "A2")
    assert "status=ENDED" in ctx_a2
    assert "[DETECTED CONFLICTS]" in ctx_a2
    assert "[UNCERTAINTY FLAGS]" in ctx_a2


def test_deterministic_answer_metrics_status_and_contradiction() -> None:
    ev = _make_dummy_event()
    gold = Gold(
        question_id="Q_TEST_1",
        intent="current_state",
        category="current_vs_historical",
        anchor_concepts=["amlodipine"],
        gold_events=["S_001_0001"],
        distractor_events=["S_001_0000"],
        expected_answer=ExpectedAnswer(type="status", value="ENDED"),
        expected_behavior="answer",
        expected_conflict=False,
        expected_flags=[],
    )

    # Correct answer (B2 style)
    good_ans = AnswerJSON(
        answer_text="Amlodipine was discontinued on 2022-09-09.",
        status_claims=[StatusClaim(concept="amlodipine", status="ENDED")],
        cited_event_ids=["S_001_0001"],
        abstain=False,
        conflict_flagged=False,
        uncertainty_noted=False,
    )
    m_good = evaluate_single_answer(
        good_ans, gold, ["S_001_0001"], [ev], {"S_001_0001": "ENDED"}
    )
    assert m_good["answer_correctness"] == 1.0
    assert m_good["temporal_consistency"] == 1.0
    assert m_good["contradiction_rate"] == 0.0
    assert m_good["citation_f1"] == 1.0
    assert m_good["grounded_faithfulness"] == 1.0

    # Contradictory answer claiming ENDED drug is ACTIVE (B1 failure mode)
    bad_ans = AnswerJSON(
        answer_text="The patient is currently taking amlodipine 5 mg daily.",
        status_claims=[StatusClaim(concept="amlodipine", status="ACTIVE")],
        cited_event_ids=["S_001_0000"],
        abstain=False,
        conflict_flagged=False,
        uncertainty_noted=False,
    )
    m_bad = evaluate_single_answer(
        bad_ans, gold, ["S_001_0000", "S_001_0001"], [ev], {"S_001_0001": "ENDED"}
    )
    assert m_bad["answer_correctness"] == 0.0
    assert m_bad["temporal_consistency"] == 0.0
    assert m_bad["contradiction_rate"] == 1.0
    assert m_bad["citation_f1"] == 0.0


def test_deterministic_answer_metrics_set_ordered_and_abstain() -> None:
    # 1. Ordered values
    gold_ord = Gold(
        question_id="Q_ORD",
        intent="change_over_time",
        category="longitudinal_change",
        anchor_concepts=["tsh"],
        gold_events=["S_001_0010", "S_001_0020"],
        distractor_events=[],
        expected_answer=ExpectedAnswer(type="ordered_values", value=[8.4, 2.2]),
        expected_behavior="answer",
        expected_conflict=False,
        expected_flags=[],
    )
    ans_ord = AnswerJSON(
        answer_text="TSH decreased from 8.4 mIU/L initially to 2.2 mIU/L on the latest test.",
        status_claims=[],
        cited_event_ids=["S_001_0010", "S_001_0020"],
    )
    assert answer_correctness(ans_ord, gold_ord) == 1.0
    p, r, f1 = citation_precision_recall_f1(ans_ord.cited_event_ids, gold_ord.gold_events)
    assert (p, r, f1) == (1.0, 1.0, 1.0)

    # 2. Set F1 on conflict question
    gold_conf = Gold(
        question_id="Q_CONF",
        intent="current_state",
        category="conflicting_information",
        anchor_concepts=["levothyroxine"],
        gold_events=["S_001_0030", "S_001_0031"],
        distractor_events=[],
        expected_answer=ExpectedAnswer(type="set", value=["ACTIVE", "ENDED"]),
        expected_behavior="flag_conflict",
        expected_conflict=True,
        expected_flags=["C1", "U1"],
    )
    ans_conf = AnswerJSON(
        answer_text="Outpatient clinic lists levothyroxine as active, while external pharmacy states it was discontinued.",
        status_claims=[
            StatusClaim(concept="levothyroxine", status="ACTIVE"),
            StatusClaim(concept="levothyroxine", status="ENDED"),
        ],
        cited_event_ids=["S_001_0030", "S_001_0031"],
        abstain=False,
        conflict_flagged=True,
        uncertainty_noted=True,
    )
    assert answer_correctness(ans_conf, gold_conf) == 1.0
    assert conflict_flag_accuracy(ans_conf, gold_conf) == 1.0
    assert uncertainty_noted_accuracy(ans_conf, gold_conf) == 1.0

    # 3. Abstention on insufficient evidence
    gold_abs = Gold(
        question_id="Q_ABS",
        intent="current_state",
        category="insufficient_evidence",
        anchor_concepts=["tacrolimus"],
        gold_events=[],
        distractor_events=["S_001_0001"],
        expected_answer=ExpectedAnswer(type="none", value=None),
        expected_behavior="abstain",
        expected_conflict=False,
        expected_flags=["U5"],
    )
    ans_abs = AnswerJSON(
        answer_text="Insufficient evidence: tacrolimus is not documented in the patient's records.",
        status_claims=[],
        cited_event_ids=[],
        abstain=True,
        conflict_flagged=False,
        uncertainty_noted=True,
    )
    assert answer_correctness(ans_abs, gold_abs) == 1.0
    assert abstention_accuracy(ans_abs, gold_abs) == 1.0
    assert grounded_faithfulness(ans_abs, ["S_001_0001"], gold_abs) == 1.0
