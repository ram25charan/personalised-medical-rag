"""Deterministic Generation-Level Answer Evaluation Metrics (`src/evaluation/answer_metrics.py`) (Section 8.6).

Computes all generation-level metrics from the structured JSON output (`AnswerJSON`) against
programmatic `Gold` annotations and patient event states without using an LLM judge:
  1. `answer_correctness`: exact match for single-valued (`status`, `value`), set F1 for `set`,
     ordered match for `ordered_values`, and abstention match for `none`.
  2. `temporal_consistency`: share of status claims in `status_claims` that agree with the
     query-time temporal state of the concept/events.
  3. `contradiction_rate`: 1.0 if an answer states an `ENDED`, `HISTORICAL`, or `NOT_LISTED_LATEST`
     concept as currently `ACTIVE` (or vice versa), else 0.0.
  4. `citation_precision_recall_f1`: precision, recall, and F1 of `cited_event_ids` against `gold_events`.
  5. `abstention_accuracy`: agreement with `expected_behavior == "abstain"` across all questions.
  6. `conflict_flag_accuracy`: agreement of `conflict_flagged` with `gold.expected_conflict`.
  7. `uncertainty_noted_accuracy`: agreement of `uncertainty_noted` with `len(gold.expected_flags) > 0`.
  8. `grounded_faithfulness`: 1.0 if all `cited_event_ids` exist in the retrieved prompt context
     (no hallucinated citations) and answerable responses cite at least one context event.
"""

from __future__ import annotations

import re
from typing import Any

from src.schemas import AnswerJSON, Event, Gold, TemporalStateType


_MONTH_NAMES = {
    "01": "january",
    "02": "february",
    "03": "march",
    "04": "april",
    "05": "may",
    "06": "june",
    "07": "july",
    "08": "august",
    "09": "september",
    "10": "october",
    "11": "november",
    "12": "december",
}


def _norm(text: Any) -> str:
    s = str(text or "").strip().lower()
    # Normalise Unicode narrow no-break spaces (\u202f), non-breaking spaces (\u00a0), and multi-spaces
    return re.sub(r"\s+", " ", s)


def _build_concept_states_map(
    pool_events: list[Event],
    states_by_id: dict[str, TemporalStateType],
    gold: Gold,
) -> dict[str, set[str]]:
    """Map each lowercase concept name (and anchor concept) to the set of temporal states
    of its most recent/relevant events at `as_of_date`.
    """
    by_concept: dict[str, list[tuple[str, str]]] = {}
    for ev in pool_events:
        c = _norm(ev.concept)
        if not c:
            continue
        obs = ev.observed_time or "0000-00-00"
        st = states_by_id.get(ev.event_id, "UNKNOWN")
        by_concept.setdefault(c, []).append((obs, st))

    concept_states: dict[str, set[str]] = {}
    for c, items in by_concept.items():
        # If this concept has a C1 conflict in gold, both ACTIVE and ENDED are valid states
        if gold.expected_conflict and c in {_norm(a) for a in gold.anchor_concepts}:
            concept_states[c] = {st for _, st in items}
        else:
            # Find latest observed_time for this concept
            max_obs = max(obs for obs, _ in items)
            latest_states = {st for obs, st in items if obs == max_obs}
            # Include PERSISTENT if any record for this concept is persistent
            if any(st == "PERSISTENT" for _, st in items):
                latest_states.add("PERSISTENT")
            concept_states[c] = latest_states
    return concept_states


def _status_matches(claimed_status: str, target_status: str) -> bool:
    cs = claimed_status.strip().upper()
    ts = target_status.strip().upper()
    if cs == ts:
        return True
    equiv_groups = [
        {"ACTIVE", "CURRENT", "ONGOING"},
        {"ENDED", "DISCONTINUED", "STOPPED", "INACTIVE"},
        {"PERSISTENT", "ACTIVE", "CHRONIC"},
        {"LISTED", "ACTIVE", "DOCUMENTED_HOME_MED"},
        {"NOT_LISTED_LATEST", "OMITTED", "NOT_LISTED", "ENDED", "HISTORICAL"},
        {"HISTORICAL", "PAST", "PRIOR"},
    ]
    for grp in equiv_groups:
        if ts in grp and cs in grp:
            # Note: do not conflate ACTIVE with ENDED or NOT_LISTED_LATEST
            if ts in {"ENDED", "NOT_LISTED_LATEST"} and cs == "ACTIVE":
                return False
            if ts == "ACTIVE" and cs in {"ENDED", "NOT_LISTED_LATEST", "HISTORICAL"}:
                return False
            return True
    return False


def answer_correctness(parsed: AnswerJSON, gold: Gold) -> float:
    """Deterministic `answer_correctness` (Section 8.6)."""
    exp_type = gold.expected_answer.type
    exp_val = gold.expected_answer.value
    ans_lower = _norm(parsed.answer_text)

    # 1. Insufficient evidence (`none`)
    if exp_type == "none" or gold.expected_behavior == "abstain":
        return 1.0 if parsed.abstain else 0.0

    # If the question is answerable but the model abstained, correctness is 0.0
    if parsed.abstain:
        return 0.0

    # 2. Single-valued `status`
    if exp_type == "status":
        target_st = str(exp_val).strip().upper()
        primary_anchor = _norm(gold.anchor_concepts[0]) if gold.anchor_concepts else ""

        # Check structured status_claims first
        matching_claims = [
            c
            for c in parsed.status_claims
            if not primary_anchor
            or primary_anchor in _norm(c.concept)
            or _norm(c.concept) in primary_anchor
        ]
        if matching_claims:
            # If any claim on the primary anchor directly contradicts target_st, return 0.0
            for c in matching_claims:
                if target_st in {"ENDED", "NOT_LISTED_LATEST"} and c.status.upper() == "ACTIVE":
                    return 0.0
                if target_st in {"ACTIVE", "PERSISTENT", "LISTED"} and c.status.upper() in {
                    "ENDED",
                    "NOT_LISTED_LATEST",
                }:
                    return 0.0
            if any(_status_matches(c.status, target_st) for c in matching_claims):
                return 1.0

        # Fallback check on answer_text when status_claims omitted the anchor
        if target_st == "ENDED":
            if any(
                w in ans_lower
                for w in ["discontinued", "stopped", "ended", "no longer", "switched from"]
            ):
                return 1.0
        elif target_st == "PERSISTENT":
            if any(
                w in ans_lower
                for w in ["persistent", "chronic", "active", "documented", "allergy", "yes"]
            ):
                return 1.0
        elif target_st == "LISTED":
            if (
                any(w in ans_lower for w in ["listed", "active", "taking", "documented", "present"])
                and "not listed" not in ans_lower
            ):
                return 1.0
        elif target_st == "NOT_LISTED_LATEST":
            if any(
                w in ans_lower
                for w in [
                    "not listed",
                    "omitted",
                    "not present",
                    "absent",
                    "not on the latest",
                    "only listed in",
                    "prior visit",
                ]
            ):
                return 1.0
        return 0.0

    # 3. Single-valued `value` (date, dose, or "unknown")
    if exp_type == "value":
        target_str = _norm(exp_val)
        if target_str == "unknown":
            unknown_cues = [
                "unknown",
                "not specified",
                "unspecified",
                "missing",
                "not documented",
                "no date",
                "no exact date",
                "no start date",
                "unrecorded",
                "ambiguous",
            ]
            return 1.0 if (any(w in ans_lower for w in unknown_cues) or parsed.uncertainty_noted) else 0.0

        # Exact substring or date prefix match (e.g. "2180-05-06 19:17:00" -> "2180-05-06")
        if target_str in ans_lower:
            return 1.0
        if len(target_str) >= 10 and re.match(r"^\d{4}-\d{2}-\d{2}", target_str):
            date_prefix = target_str[:10]
            if date_prefix in ans_lower:
                return 1.0
        # Month-precision date match (e.g. "2022-06" -> "june 2022" or "jun 2022")
        m_ym = re.match(r"^(\d{4})-(\d{2})$", target_str)
        if m_ym:
            yr, mo = m_ym.group(1), m_ym.group(2)
            m_name = _MONTH_NAMES.get(mo, "")
            if m_name and (f"{m_name} {yr}" in ans_lower or f"{m_name[:3]} {yr}" in ans_lower):
                return 1.0
        return 0.0

    # 4. Set-valued `set` (Set F1)
    if exp_type == "set":
        raw_items = exp_val if isinstance(exp_val, list) else [exp_val]
        exp_set = {_norm(x) for x in raw_items if _norm(x)}
        if not exp_set:
            return 1.0

        # Subcase 4a: expected set is {"active", "ended"} (C1 conflict question)
        if exp_set == {"active", "ended"}:
            claimed_statuses = {c.status.strip().upper() for c in parsed.status_claims}
            has_active = "ACTIVE" in claimed_statuses or "active" in ans_lower
            has_ended = (
                "ENDED" in claimed_statuses
                or any(w in ans_lower for w in ["ended", "discontinued", "stopped"])
            )
            pred_set: set[str] = set()
            if has_active:
                pred_set.add("active")
            if has_ended:
                pred_set.add("ended")
            tp = len(exp_set & pred_set)
            if tp == 0:
                return 0.0
            prec = tp / len(pred_set)
            rec = tp / len(exp_set)
            return round(2 * prec * rec / (prec + rec), 4)

        # Subcase 4b: current_vs_historical condition-level query where anchor_concepts lists
        # [condition, active_med, discontinued_med] and exp_set == {active_med}
        candidate_meds = [_norm(c) for c in gold.anchor_concepts[1:]] if len(gold.anchor_concepts) > 1 else []
        if candidate_meds and exp_set.issubset(set(candidate_meds)):
            pred_active: set[str] = set()
            # Check status_claims first
            for c in parsed.status_claims:
                c_name = _norm(c.concept)
                for med in candidate_meds:
                    if med in c_name or c_name in med:
                        if c.status.strip().upper() in {"ACTIVE", "LISTED", "PERSISTENT"}:
                            pred_active.add(med)
            # If status_claims didn't populate pred_active, inspect answer_text
            if not pred_active:
                for med in candidate_meds:
                    if med in ans_lower:
                        # Check that it isn't only mentioned as discontinued/stopped
                        med_discontinued_pattern = rf"{re.escape(med)}[^.;]*?(?:discontinued|stopped|ended|historical|prior)"
                        if not re.search(med_discontinued_pattern, ans_lower):
                            pred_active.add(med)
            if not pred_active:
                return 0.0
            tp = len(exp_set & pred_active)
            if tp == 0:
                return 0.0
            prec = tp / len(pred_active)
            rec = tp / len(exp_set)
            return round(2 * prec * rec / (prec + rec), 4)

        # Subcase 4c: general set of values/doses/medications (e.g. ["20 mg", "40 mg"] or med list)
        matched = {item for item in exp_set if item in ans_lower}
        tp = len(matched)
        if tp == 0:
            return 0.0
        prec = 1.0  # all matched items are in exp_set
        rec = tp / len(exp_set)
        return round(2 * prec * rec / (prec + rec), 4)

    # 5. Ordered values `ordered_values`
    if exp_type == "ordered_values":
        raw_list = exp_val if isinstance(exp_val, list) else [exp_val]
        exp_list = [_norm(x) for x in raw_list if _norm(x)]
        if not exp_list:
            return 1.0

        positions: list[int] = []
        for item in exp_list:
            idx = ans_lower.find(item)
            if idx == -1:
                # Try numeric string without trailing .0 if float
                if item.endswith(".0") and item[:-2] in ans_lower:
                    idx = ans_lower.find(item[:-2])
                else:
                    return 0.0
            positions.append(idx)

        # Verify chronological order in text (or all values present when explicitly describing transition)
        if all(positions[i] <= positions[i + 1] for i in range(len(positions) - 1)):
            return 1.0
        # Also allow if both values are present and status_claims orders HISTORICAL before ACTIVE
        return 0.5

    return 0.0


def temporal_consistency(
    parsed: AnswerJSON,
    gold: Gold,
    concept_states: dict[str, set[str]],
) -> float:
    """Share of status claims in `status_claims` that agree with the computed query-time
    temporal state of the concept (Section 8.6).
    """
    if gold.expected_behavior == "abstain":
        return 1.0 if parsed.abstain else 0.0
    if parsed.abstain:
        return 0.0

    if not parsed.status_claims:
        # Fall back to whether the answer's primary status/value claim is temporally consistent
        return 1.0 if answer_correctness(parsed, gold) >= 0.99 else 0.0

    consistent_count = 0
    evaluated_count = 0
    for claim in parsed.status_claims:
        c_norm = _norm(claim.concept)
        c_status = claim.status.strip().upper()
        matched_states: set[str] = set()
        for known_concept, st_set in concept_states.items():
            if c_norm == known_concept or c_norm in known_concept or known_concept in c_norm:
                matched_states |= st_set

        if not matched_states:
            # If the concept is the primary anchor and gold specifies a status, compare against gold
            if gold.expected_answer.type == "status":
                evaluated_count += 1
                if _status_matches(c_status, str(gold.expected_answer.value)):
                    consistent_count += 1
            continue

        evaluated_count += 1
        if any(_status_matches(c_status, true_st) for true_st in matched_states):
            consistent_count += 1

    if evaluated_count == 0:
        return 1.0 if answer_correctness(parsed, gold) >= 0.99 else 0.0

    return round(consistent_count / evaluated_count, 4)


def contradiction_rate(
    parsed: AnswerJSON,
    gold: Gold,
    concept_states: dict[str, set[str]],
) -> float:
    """Returns 1.0 if the answer states an `ENDED`, `HISTORICAL`, or `NOT_LISTED_LATEST`
    concept as currently `ACTIVE` (or vice versa), else 0.0 (Section 8.6).
    """
    if gold.expected_behavior == "abstain":
        # Claiming a non-existent medication is active on an abstain question is a contradiction
        if not parsed.abstain and any(
            c.status.strip().upper() in {"ACTIVE", "LISTED", "PERSISTENT"}
            for c in parsed.status_claims
        ):
            return 1.0
        return 0.0

    # Check explicit status_claims against known concept states
    for claim in parsed.status_claims:
        c_norm = _norm(claim.concept)
        c_status = claim.status.strip().upper()
        matched_states: set[str] = set()
        for known_concept, st_set in concept_states.items():
            if c_norm == known_concept or c_norm in known_concept or known_concept in c_norm:
                matched_states |= st_set

        if not matched_states:
            continue

        # If there is a known C1 conflict (both ACTIVE and ENDED present), neither claim alone is a contradiction
        if "ACTIVE" in matched_states and "ENDED" in matched_states:
            continue

        if c_status in {"ACTIVE", "LISTED"} and matched_states.issubset(
            {"ENDED", "HISTORICAL", "NOT_LISTED_LATEST"}
        ):
            return 1.0
        if c_status in {"ENDED", "NOT_LISTED_LATEST"} and matched_states.issubset(
            {"ACTIVE", "PERSISTENT", "LISTED"}
        ):
            return 1.0

    # Also check current_vs_historical questions when status_claims is empty or incomplete
    if gold.category == "current_vs_historical" and answer_correctness(parsed, gold) < 0.99:
        return 1.0

    return 0.0


def citation_precision_recall_f1(
    cited_event_ids: list[str],
    gold_events: list[str],
) -> tuple[float, float, float]:
    """Compute `(precision, recall, f1)` of `cited_event_ids` against `gold_events` (Section 8.6)."""
    cited_set = {str(x).strip() for x in cited_event_ids if str(x).strip()}
    gold_set = {str(x).strip() for x in gold_events if str(x).strip()}

    if not gold_set:
        # For insufficient_evidence questions where gold_events is empty:
        if not cited_set:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    if not cited_set:
        return 0.0, 0.0, 0.0

    tp = len(cited_set & gold_set)
    if tp == 0:
        return 0.0, 0.0, 0.0

    prec = tp / len(cited_set)
    rec = tp / len(gold_set)
    f1 = 2 * prec * rec / (prec + rec)
    return round(prec, 4), round(rec, 4), round(f1, 4)


def abstention_accuracy(parsed: AnswerJSON, gold: Gold) -> float:
    """1.0 if `parsed.abstain == (gold.expected_behavior == 'abstain')`, else 0.0."""
    expected_abstain = gold.expected_behavior == "abstain"
    return 1.0 if bool(parsed.abstain) == expected_abstain else 0.0


def conflict_flag_accuracy(parsed: AnswerJSON, gold: Gold) -> float:
    """1.0 if `parsed.conflict_flagged == gold.expected_conflict`, else 0.0."""
    return 1.0 if bool(parsed.conflict_flagged) == bool(gold.expected_conflict) else 0.0


def uncertainty_noted_accuracy(parsed: AnswerJSON, gold: Gold) -> float:
    """1.0 if `parsed.uncertainty_noted` matches whether uncertainty/conflict/abstain flags are expected."""
    expected_uncertainty = (
        len(gold.expected_flags) > 0
        or gold.expected_behavior in {"flag_uncertainty", "flag_conflict", "abstain"}
    )
    return 1.0 if bool(parsed.uncertainty_noted) == expected_uncertainty else 0.0


def grounded_faithfulness(
    parsed: AnswerJSON,
    selected_event_ids: list[str],
    gold: Gold,
) -> float:
    """Deterministic faithfulness check (Section 8.6):
    Returns 1.0 if all `cited_event_ids` belong to the provided context (`selected_event_ids`)
    and non-abstaining answers cite at least one context event.
    """
    selected_set = set(selected_event_ids)
    cited_set = {str(x).strip() for x in parsed.cited_event_ids if str(x).strip()}

    # Any hallucinated event ID outside the prompt context violates faithfulness
    if not cited_set.issubset(selected_set):
        return 0.0

    if gold.expected_behavior == "abstain" or parsed.abstain:
        return 1.0

    return 1.0 if len(cited_set) > 0 else 0.0


def evaluate_single_answer(
    parsed: AnswerJSON,
    gold: Gold,
    selected_event_ids: list[str],
    pool_events: list[Event],
    states_by_id: dict[str, TemporalStateType],
) -> dict[str, float]:
    """Compute the full dictionary of deterministic generation metrics for one question."""
    concept_states = _build_concept_states_map(pool_events, states_by_id, gold)
    c_prec, c_rec, c_f1 = citation_precision_recall_f1(parsed.cited_event_ids, gold.gold_events)
    return {
        "answer_correctness": answer_correctness(parsed, gold),
        "temporal_consistency": temporal_consistency(parsed, gold, concept_states),
        "contradiction_rate": contradiction_rate(parsed, gold, concept_states),
        "citation_precision": c_prec,
        "citation_recall": c_rec,
        "citation_f1": c_f1,
        "abstention_accuracy": abstention_accuracy(parsed, gold),
        "conflict_flag_accuracy": conflict_flag_accuracy(parsed, gold),
        "uncertainty_noted_accuracy": uncertainty_noted_accuracy(parsed, gold),
        "grounded_faithfulness": grounded_faithfulness(parsed, selected_event_ids, gold),
    }
