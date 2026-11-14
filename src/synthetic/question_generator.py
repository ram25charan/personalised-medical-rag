"""Template-based Benchmark Question and Programmatic Gold Generator (Sections 6.8, 8.5, 12.1).

Generates questions and exact programmatic gold labels across:
- Track S: 50 dev questions (from the 34 dev patients) and 100 test questions (from the 66 test patients),
  balanced across all 7 benchmark categories and covering all 5 question intents.
- Track R: 30 test questions on real MIMIC-IV-ED demo patients anchored to visit `outtime`,
  following snapshot semantics ("recorded on the list at visit N", Section 6.8 & DD5).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from src.schemas import Event, ExpectedAnswer, Gold, Question

ALL_CATEGORIES = [
    "temporal_ordering",
    "current_vs_historical",
    "longitudinal_change",
    "irrelevant_history",
    "missing_or_ambiguous_dates",
    "conflicting_information",
    "insufficient_evidence",
]


def _make_track_s_question_and_gold(
    bp: dict[str, Any],
    category: str,
    q_id: str,
    variant: int = 0,
) -> tuple[Question, Gold]:
    """Construct one `(Question, Gold)` pair for a synthetic patient blueprint and category."""
    pid = bp["patient_id"]
    split = bp["split"]
    as_of = bp["as_of_date"]
    win_start = bp["window_start"]
    sc = bp["scenarios"]
    dist_ids = bp["distractor_event_ids"]

    if category == "temporal_ordering":
        if variant % 2 == 0:
            # First occurrence of chronic condition
            c_info = sc["condition_htn"]
            q_text = f"When was {c_info['concept']} first diagnosed for this patient?"
            intent = "first_occurrence"
            anchors = [c_info["concept"]]
            gold_ids = [c_info["first_id"]]
            distractors = [c_info["followup_id"]] + dist_ids[:4]
            exp_ans = ExpectedAnswer(type="value", value=c_info["onset_date"])
            exp_beh = "answer"
            exp_conf = False
            exp_flags: list[str] = []
        else:
            # Sequence of hypertension medications (historical ordering)
            m_info = sc["med_lifecycle"]
            q_text = (
                f"In what chronological order were the antihypertensive medications "
                f"{m_info['old_med']} and {m_info['new_med']} prescribed up to {as_of}?"
            )
            intent = "historical"
            anchors = [m_info["old_med"], m_info["new_med"], "essential (primary) hypertension"]
            gold_ids = [m_info["start_id"], m_info["new_active_id"]]
            distractors = dist_ids[:5]
            exp_ans = ExpectedAnswer(
                type="ordered_values", value=[m_info["old_med"], m_info["new_med"]]
            )
            exp_beh = "answer"
            exp_conf = False
            exp_flags = ["U1"]

    elif category == "current_vs_historical":
        m_info = sc["med_lifecycle"]
        if variant % 2 == 0:
            # Ask about the discontinued medication vs current active replacement
            q_text = (
                f"Is the patient currently taking {m_info['old_med']} as of {as_of}, "
                f"or has it been stopped and replaced?"
            )
            intent = "current_state"
            anchors = [m_info["old_med"], m_info["new_med"]]
            gold_ids = [m_info["dose_change_id"], m_info["new_active_id"]]
            distractors = [m_info["start_id"]] + dist_ids[:4]
            exp_ans = ExpectedAnswer(type="status", value="ENDED")
            exp_beh = "answer"
            exp_conf = False
            exp_flags = ["U1"]
        else:
            # Ask which antihypertensive is currently active
            q_text = (
                f"What antihypertensive medication is the patient currently taking as of {as_of}?"
            )
            intent = "current_state"
            anchors = ["essential (primary) hypertension", m_info["new_med"], m_info["old_med"]]
            gold_ids = [m_info["new_active_id"]]
            distractors = [m_info["start_id"], m_info["dose_change_id"]] + dist_ids[:3]
            exp_ans = ExpectedAnswer(type="set", value=[m_info["new_med"]])
            exp_beh = "answer"
            exp_conf = False
            exp_flags = ["U1"]

    elif category == "longitudinal_change":
        if variant % 2 == 0:
            l_info = sc["lab_trend"]
            q_text = (
                f"How has the patient's {l_info['concept']} changed from the earliest "
                f"to the latest recorded test up to {as_of}?"
            )
            intent = "change_over_time"
            anchors = [l_info["concept"]]
            # Minimal gold for earliest-to-latest change is the full series (or first & last)
            gold_ids = [l_info["gold_ids"][0], l_info["gold_ids"][-1]]
            distractors = dist_ids[:5]
            exp_ans = ExpectedAnswer(
                type="ordered_values",
                value=[l_info["values"][0], l_info["values"][-1]],
            )
            exp_beh = "answer"
            exp_conf = False
            exp_flags = []
        else:
            m_info = sc["med_lifecycle"]
            q_text = f"How did the patient's {m_info['old_med']} dose change over time before it was stopped?"
            intent = "change_over_time"
            anchors = [m_info["old_med"]]
            gold_ids = [m_info["start_id"], m_info["dose_change_id"]]
            distractors = dist_ids[:5]
            exp_ans = ExpectedAnswer(
                type="ordered_values",
                value=[m_info["old_dose_1"], m_info["old_dose_2"]],
            )
            exp_beh = "answer"
            exp_conf = False
            exp_flags = []

    elif category == "irrelevant_history":
        if variant % 2 == 0:
            a_info = sc["allergy"]
            q_text = f"Does the patient have a documented {a_info['concept']} in their record?"
            intent = "persistent_fact"
            anchors = [a_info["concept"]]
            gold_ids = a_info["gold_ids"]
            distractors = dist_ids[:6]
            exp_ans = ExpectedAnswer(type="status", value="PERSISTENT")
            exp_beh = "answer"
            exp_conf = False
            exp_flags = []
        else:
            m_info = sc["med_lifecycle"]
            q_text = f"What is the current status and dose of {m_info['dm_med']} as of {as_of}?"
            intent = "current_state"
            anchors = [m_info["dm_med"]]
            gold_ids = [m_info["dm_active_id"]]
            distractors = dist_ids[:6]
            exp_ans = ExpectedAnswer(type="value", value=m_info["dm_dose"])
            exp_beh = "answer"
            exp_conf = False
            exp_flags = ["U1"]

    elif category == "missing_or_ambiguous_dates":
        md_info = sc["missing_dates"]
        if variant % 2 == 0:
            q_text = (
                f"When did the patient start taking {md_info['statin_concept']}, "
                f"and is the exact start day known?"
            )
            intent = "first_occurrence"
            anchors = [md_info["statin_concept"]]
            gold_ids = [md_info["statin_id"]]
            distractors = dist_ids[:5]
            exp_ans = ExpectedAnswer(type="value", value=md_info["statin_month"])
            exp_beh = "flag_uncertainty"
            exp_conf = False
            exp_flags = ["U1", "U2"]
        else:
            q_text = (
                f"When was {md_info['gerd_concept']} first recorded in the patient's history?"
            )
            intent = "first_occurrence"
            anchors = [md_info["gerd_concept"]]
            gold_ids = [md_info["gerd_id"]]
            distractors = dist_ids[:5]
            exp_ans = ExpectedAnswer(type="value", value="unknown")
            exp_beh = "flag_uncertainty"
            exp_conf = False
            exp_flags = ["U3"]

    elif category == "conflicting_information":
        c_info = sc["conflict"]
        q_text = (
            f"What is the patient's current status and dosage for {c_info['concept']} as of {as_of}?"
        )
        intent = "current_state"
        anchors = [c_info["concept"]]
        gold_ids = c_info["gold_ids"]
        distractors = dist_ids[:5]
        if c_info["rule"] == "C1":
            exp_ans = ExpectedAnswer(type="set", value=["ACTIVE", "ENDED"])
        else:
            exp_ans = ExpectedAnswer(
                type="set", value=[c_info["dose_a"], c_info["dose_b"]]
            )
        exp_beh = "flag_conflict"
        exp_conf = True
        exp_flags = [c_info["rule"], "U1"]

    elif category == "insufficient_evidence":
        ie_info = sc["insufficient_evidence"]
        q_text = (
            f"Is the patient currently taking {ie_info['concept']} as of {as_of}?"
        )
        intent = "current_state"
        anchors = [ie_info["concept"]]
        gold_ids = []
        distractors = dist_ids[:6]
        exp_ans = ExpectedAnswer(type="none", value=None)
        exp_beh = "abstain"
        exp_conf = False
        exp_flags = ["U5"]

    else:
        raise ValueError(f"Unknown category: {category}")

    q_obj = Question(
        question_id=q_id,
        patient_id=pid,
        track="S",
        split=split,
        text=q_text,
        as_of_date=as_of,
        window_start=win_start if intent in ("historical", "change_over_time") else None,
    )
    g_obj = Gold(
        question_id=q_id,
        intent=intent,
        category=category,
        anchor_concepts=anchors,
        gold_events=gold_ids,
        distractor_events=distractors,
        expected_answer=exp_ans,
        expected_behavior=exp_beh,
        expected_conflict=exp_conf,
        expected_flags=exp_flags,
    )
    return q_obj, g_obj


def generate_track_r_questions(
    track_r_events: list[Event],
    target_count: int = 30,
) -> tuple[list[Question], list[Gold]]:
    """Deterministically generate ~30 test questions on Track R (real MIMIC-IV-ED demo) anchored to visit `outtime`.

    Follows Section 6.8 & DD5:
    - Real-track gold means "recorded on the list at visit N", not clinical truth.
    - Covers Categories 1-5 and 7 (Category 6 has no real conflicts per coverage report).
    """
    by_patient: dict[str, list[Event]] = defaultdict(list)
    for ev in track_r_events:
        by_patient[ev.patient_id].append(ev)

    questions: list[Question] = []
    golds: list[Gold] = []
    q_counter = 1

    def add_pair(
        pid: str,
        text: str,
        as_of: str,
        win_start: Optional[str],
        intent: str,
        category: str,
        anchors: list[str],
        gold_ids: list[str],
        dist_ids: list[str],
        ans_type: str,
        ans_val: Any,
        behavior: str = "answer",
        flags: Optional[list[str]] = None,
    ) -> None:
        nonlocal q_counter
        qid = f"Q_R_{q_counter:04d}"
        q_counter += 1
        questions.append(
            Question(
                question_id=qid,
                patient_id=pid,
                track="R",
                split="test",
                text=text,
                as_of_date=as_of,
                window_start=win_start,
            )
        )
        golds.append(
            Gold(
                question_id=qid,
                intent=intent,  # type: ignore[arg-type]
                category=category,  # type: ignore[arg-type]
                anchor_concepts=anchors,
                gold_events=gold_ids,
                distractor_events=dist_ids,
                expected_answer=ExpectedAnswer(type=ans_type, value=ans_val),  # type: ignore[arg-type]
                expected_behavior=behavior,  # type: ignore[arg-type]
                expected_conflict=False,
                expected_flags=flags or [],
            )
        )

    # Sort patients deterministically
    sorted_pids = sorted(by_patient.keys())

    # 1. Category 2 (current_vs_historical): 6 questions on multi-visit patients where a drug was on an earlier visit list
    #    or listed on the latest visit
    cat2_count = 0
    for pid in sorted_pids:
        if cat2_count >= 6:
            break
        p_evs = by_patient[pid]
        visits = [e for e in p_evs if e.event_type == "visit"]
        if len(visits) < 2:
            continue
        meds_by_stay: dict[str, dict[str, Event]] = defaultdict(dict)
        for e in p_evs:
            if e.event_type == "reported_home_medication" and e.encounter_id:
                meds_by_stay[e.encounter_id][e.concept] = e
        med_visits = [v for v in visits if v.encounter_id in meds_by_stay]
        if len(med_visits) < 2:
            continue
        v_prev, v_last = med_visits[-2], med_visits[-1]
        s_prev = v_prev.encounter_id or ""
        s_last = v_last.encounter_id or ""
        dropped = sorted(set(meds_by_stay[s_prev].keys()) - set(meds_by_stay[s_last].keys()))
        kept = sorted(set(meds_by_stay[s_last].keys()))
        if dropped and kept:
            target_drug = dropped[0] if (cat2_count % 2 == 0) else kept[0]
            as_of = v_last.valid_end or v_last.observed_time or ""
            if target_drug in dropped:
                gold_ev = meds_by_stay[s_prev][target_drug]
                dist_evs = [e.event_id for e in meds_by_stay[s_last].values()][:4]
                add_pair(
                    pid=pid,
                    text=f"Was {target_drug} recorded on the patient's latest home medication list at the visit ending {as_of}?",
                    as_of=as_of,
                    win_start=None,
                    intent="current_state",
                    category="current_vs_historical",
                    anchors=[target_drug],
                    gold_ids=[gold_ev.event_id],
                    dist_ids=dist_evs,
                    ans_type="status",
                    ans_val="NOT_LISTED_LATEST",
                    behavior="flag_uncertainty",
                    flags=["U4"],
                )
            else:
                gold_ev = meds_by_stay[s_last][target_drug]
                dist_evs = [
                    e.event_id
                    for k, e in meds_by_stay[s_prev].items()
                    if k != target_drug
                ][:4]
                add_pair(
                    pid=pid,
                    text=f"Was {target_drug} recorded on the patient's home medication list at the visit ending {as_of}?",
                    as_of=as_of,
                    win_start=None,
                    intent="current_state",
                    category="current_vs_historical",
                    anchors=[target_drug],
                    gold_ids=[gold_ev.event_id],
                    dist_ids=dist_evs,
                    ans_type="status",
                    ans_val="LISTED",
                    behavior="answer",
                    flags=["U4"],
                )
            cat2_count += 1

    # 2. Category 3 (longitudinal_change): 6 questions on medications added between consecutive visits
    cat3_count = 0
    for pid in sorted_pids:
        if cat3_count >= 6:
            break
        p_evs = by_patient[pid]
        visits = [e for e in p_evs if e.event_type == "visit"]
        meds_by_stay = defaultdict(dict)
        for e in p_evs:
            if e.event_type == "reported_home_medication" and e.encounter_id:
                meds_by_stay[e.encounter_id][e.concept] = e
        med_visits = [v for v in visits if v.encounter_id in meds_by_stay]
        if len(med_visits) < 2:
            continue
        v1, v2 = med_visits[0], med_visits[1]
        s1, s2 = v1.encounter_id or "", v2.encounter_id or ""
        added = sorted(set(meds_by_stay[s2].keys()) - set(meds_by_stay[s1].keys()))
        if added:
            as_of = v2.valid_end or v2.observed_time or ""
            win_start = v1.valid_start or v1.observed_time
            gold_ids = [meds_by_stay[s2][c].event_id for c in added]
            dist_ids = [e.event_id for e in meds_by_stay[s1].values()][:4]
            add_pair(
                pid=pid,
                text=f"Which home medications were newly listed at the visit on {as_of} compared with the earlier visit on {v1.valid_start}?",
                as_of=as_of,
                win_start=win_start,
                intent="change_over_time",
                category="longitudinal_change",
                anchors=added[:3],
                gold_ids=gold_ids,
                dist_ids=dist_ids,
                ans_type="set",
                ans_val=added,
                behavior="answer",
                flags=["U4"],
            )
            cat3_count += 1

    # 3. Category 1 (temporal_ordering): 6 questions on earliest recorded diagnosis across visits
    cat1_count = 0
    for pid in sorted_pids:
        if cat1_count >= 6:
            break
        p_evs = by_patient[pid]
        visits = [e for e in p_evs if e.event_type == "visit"]
        diags = [e for e in p_evs if e.event_type == "diagnosis_at_visit"]
        if len(visits) >= 2 and len(diags) >= 2:
            first_diag = diags[0]
            last_visit = visits[-1]
            as_of = last_visit.valid_end or last_visit.observed_time or ""
            later_diags = [d.event_id for d in diags[1:] if d.concept != first_diag.concept][:4]
            add_pair(
                pid=pid,
                text=f"When was {first_diag.concept} first recorded as an ED visit diagnosis for this patient?",
                as_of=as_of,
                win_start=None,
                intent="first_occurrence",
                category="temporal_ordering",
                anchors=[first_diag.concept],
                gold_ids=[first_diag.event_id],
                dist_ids=later_diags,
                ans_type="value",
                ans_val=first_diag.observed_time,
                behavior="answer",
                flags=[],
            )
            cat1_count += 1

    # 4. Category 4 (irrelevant_history): 5 questions on recurring diagnosis in multi-visit patients
    cat4_count = 0
    for pid in sorted_pids:
        if cat4_count >= 5:
            break
        p_evs = by_patient[pid]
        visits = [e for e in p_evs if e.event_type == "visit"]
        diags_by_concept: dict[str, list[Event]] = defaultdict(list)
        for e in p_evs:
            if e.event_type == "diagnosis_at_visit":
                diags_by_concept[e.concept].append(e)
        recurring = [
            (c, evs) for c, evs in sorted(diags_by_concept.items()) if len(evs) >= 2
        ]
        if recurring:
            concept, evs = recurring[0]
            last_visit = visits[-1]
            as_of = last_visit.valid_end or last_visit.observed_time or ""
            vitals = [e.event_id for e in p_evs if e.event_type == "vital_sign"][:5]
            add_pair(
                pid=pid,
                text=f"Has the patient had recurring ED diagnoses of {concept} across multiple visits up to {as_of}?",
                as_of=as_of,
                win_start=None,
                intent="persistent_fact",
                category="irrelevant_history",
                anchors=[concept],
                gold_ids=[e.event_id for e in evs],
                dist_ids=vitals,
                ans_type="status",
                ans_val="PERSISTENT",
                behavior="answer",
                flags=[],
            )
            cat4_count += 1

    # 5. Category 5 (missing_or_ambiguous_dates): 3 questions on stays where home medication start/stop dates are absent (snapshot U1/U4)
    cat5_count = 0
    for pid in sorted_pids:
        if cat5_count >= 3:
            break
        p_evs = by_patient[pid]
        visits = [e for e in p_evs if e.event_type == "visit"]
        meds = [e for e in p_evs if e.event_type == "reported_home_medication"]
        if meds and visits:
            m_ev = meds[0]
            last_visit = visits[-1]
            as_of = last_visit.valid_end or last_visit.observed_time or ""
            vitals = [e.event_id for e in p_evs if e.event_type == "vital_sign"][:4]
            add_pair(
                pid=pid,
                text=f"Does the ED record specify when the patient originally started or stopped taking the home medication {m_ev.concept}?",
                as_of=as_of,
                win_start=None,
                intent="historical",
                category="missing_or_ambiguous_dates",
                anchors=[m_ev.concept],
                gold_ids=[m_ev.event_id],
                dist_ids=vitals,
                ans_type="value",
                ans_val="unknown",
                behavior="flag_uncertainty",
                flags=["U1", "U4"],
            )
            cat5_count += 1

    # 6. Category 7 (insufficient_evidence): 4 questions asking about unrecorded medications/labs on Track R
    cat7_count = 0
    unrecorded_Pool = ["methotrexate", "tacrolimus", "lithium carbonate", "hba1c"]
    for pid in sorted_pids:
        if cat7_count >= 4 or len(questions) >= target_count:
            break
        p_evs = by_patient[pid]
        visits = [e for e in p_evs if e.event_type == "visit"]
        concepts_present = {e.concept for e in p_evs}
        q_concept = unrecorded_Pool[cat7_count % len(unrecorded_Pool)]
        if q_concept not in concepts_present and visits:
            last_visit = visits[-1]
            as_of = last_visit.valid_end or last_visit.observed_time or ""
            dist_ids = [e.event_id for e in p_evs[:5]]
            add_pair(
                pid=pid,
                text=f"Is there any record of {q_concept} in this patient's ED history up to {as_of}?",
                as_of=as_of,
                win_start=None,
                intent="current_state",
                category="insufficient_evidence",
                anchors=[q_concept],
                gold_ids=[],
                dist_ids=dist_ids,
                ans_type="none",
                ans_val=None,
                behavior="abstain",
                flags=["U5"],
            )
            cat7_count += 1

    return questions[:target_count], golds[:target_count]


def generate_benchmark_questions(
    blueprints: list[dict[str, Any]],
    config: dict[str, Any],
    track_r_events: Optional[list[Event]] = None,
    seed: Optional[int] = None,
) -> tuple[list[Question], list[Gold]]:
    """Generate Track S (50 dev, 100 test) and Track R (30 test) benchmark questions and gold labels."""
    targets = config.get("target_questions", {})
    n_s_dev = int(targets.get("track_s_dev", 50))
    n_s_test = int(targets.get("track_s_test", 100))
    n_r_test = int(targets.get("track_r_test", 30))

    dev_bps = [bp for bp in blueprints if bp["split"] == "dev"]
    test_bps = [bp for bp in blueprints if bp["split"] == "test"]

    questions: list[Question] = []
    golds: list[Gold] = []
    s_counter = 1

    # Generate Track S dev questions (50 questions balanced across the 7 categories)
    for i in range(n_s_dev):
        bp = dev_bps[i % len(dev_bps)]
        category = ALL_CATEGORIES[i % len(ALL_CATEGORIES)]
        qid = f"Q_S_{s_counter:04d}"
        s_counter += 1
        q, g = _make_track_s_question_and_gold(bp, category, qid, variant=i // len(ALL_CATEGORIES))
        questions.append(q)
        golds.append(g)

    # Generate Track S test questions (100 questions balanced across the 7 categories)
    for i in range(n_s_test):
        bp = test_bps[i % len(test_bps)]
        category = ALL_CATEGORIES[i % len(ALL_CATEGORIES)]
        qid = f"Q_S_{s_counter:04d}"
        s_counter += 1
        q, g = _make_track_s_question_and_gold(bp, category, qid, variant=i // len(ALL_CATEGORIES))
        questions.append(q)
        golds.append(g)

    # Generate Track R test questions if Track R events are provided
    if track_r_events:
        r_questions, r_golds = generate_track_r_questions(track_r_events, target_count=n_r_test)
        questions.extend(r_questions)
        golds.extend(r_golds)

    return questions, golds
