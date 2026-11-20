"""Intent Classification and Intent-Dependent Temporal Relevance Scoring (Rt) (Sections 7.3, 12.2).

Never imports gold or synthetic generator internals (AD6).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from src.schemas import Event, IntentType, TemporalStateType
from src.temporal.state import compute_state


def load_rt_rules(path: str | Path = "configs/rt_rules.json") -> dict[str, Any]:
    """Load the frozen Rt rules and keyword intent rules."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def classify_intent_by_rules(
    question_text: str,
    rules: Optional[dict[str, Any]] = None,
) -> IntentType:
    """Determine question intent from keyword rules (Section 7.3).

    Checks ordered priority: `first_occurrence` -> `change_over_time` -> `persistent_fact`
    -> `historical` -> `current_state` (default).
    """
    q_low = question_text.lower()
    kw_map = (rules or {}).get("intent_keywords", {})

    priority_order: list[IntentType] = [
        "first_occurrence",
        "change_over_time",
        "persistent_fact",
        "historical",
        "current_state",
    ]
    for intent_name in priority_order:
        phrases = kw_map.get(intent_name, [])
        for phrase in phrases:
            if phrase.lower() in q_low:
                return intent_name
    return "current_state"


def _overlaps_window(event: Event, window_start: Optional[str], as_of_date: str) -> bool:
    """Check whether an event's validity interval overlaps `[window_start, as_of_date]`."""
    start = event.valid_start or event.observed_time
    if start is None:
        return False
    end = event.valid_end or as_of_date
    win_s = window_start or "0000-01-01"
    return start <= as_of_date and end >= win_s


def rt_scores(
    patient_events: list[Event],
    intent: IntentType,
    as_of_date: str,
    rules: Optional[dict[str, Any]] = None,
    anchors: Optional[list[str]] = None,
    window_start: Optional[str] = None,
    precomputed_states: Optional[dict[str, TemporalStateType]] = None,
) -> dict[str, float]:
    """Compute Temporal Relevance `Rt(e) in [0, 1]` for each candidate event (Section 7.3 Table & 12.2).

    Follows the exact Rt table in Section 7.3:
      - `current_state`:
          ACTIVE / LISTED / PERSISTENT -> 1.0
          UNKNOWN -> 0.5
          SUPERSEDED -> 0.2
          ENDED / NOT_LISTED_LATEST -> 0.1
      - `historical`:
          Interval overlaps window up to as_of (and matches anchor concept/series if anchors exist) -> 1.0
          UNKNOWN -> 0.5
          Otherwise -> 0.1
      - `change_over_time`:
          Earliest and latest events of the anchor series in the window -> 1.0
          Intermediate events of the anchor series -> 0.7
          Other events -> 0.1
      - `first_occurrence`:
          Earliest event of the anchor concept -> 1.0
          Later events of that concept -> 0.3
          Other events -> 0.1
      - `persistent_fact`:
          PERSISTENT -> 1.0
          ACTIVE / LISTED -> 0.5
          Others -> 0.2
    """
    states = precomputed_states or {
        ev.event_id: compute_state(ev, as_of_date, patient_events)
        for ev in patient_events
    }
    anchor_set = {a.lower().strip() for a in (anchors or []) if a.strip()}

    by_concept: dict[str, list[Event]] = defaultdict(list)
    for ev in patient_events:
        by_concept[ev.concept.lower().strip()].append(ev)
    for c_key in by_concept:
        by_concept[c_key].sort(key=lambda e: (e.observed_time or "0000-00-00", e.event_id))

    scores: dict[str, float] = {}

    if intent == "current_state":
        for ev in patient_events:
            st = states[ev.event_id]
            if st in ("ACTIVE", "LISTED", "PERSISTENT"):
                scores[ev.event_id] = 1.0
            elif st == "UNKNOWN":
                scores[ev.event_id] = 0.5
            elif st == "SUPERSEDED":
                scores[ev.event_id] = 0.2
            else:
                scores[ev.event_id] = 0.1

    elif intent == "historical":
        for ev in patient_events:
            st = states[ev.event_id]
            if st == "UNKNOWN":
                scores[ev.event_id] = 0.5
            elif _overlaps_window(ev, window_start, as_of_date):
                scores[ev.event_id] = 1.0
            else:
                scores[ev.event_id] = 0.1

    elif intent == "change_over_time":
        boundary_ids: set[str] = set()
        intermediate_ids: set[str] = set()
        target_concepts = anchor_set if anchor_set else set(by_concept.keys())

        for c_key in target_concepts:
            series = [
                e for e in by_concept.get(c_key, []) if _overlaps_window(e, window_start, as_of_date)
            ]
            if not series:
                series = by_concept.get(c_key, [])
            if len(series) == 1:
                boundary_ids.add(series[0].event_id)
            elif len(series) >= 2:
                boundary_ids.add(series[0].event_id)
                boundary_ids.add(series[-1].event_id)
                for mid_ev in series[1:-1]:
                    intermediate_ids.add(mid_ev.event_id)

        for ev in patient_events:
            if ev.event_id in boundary_ids:
                scores[ev.event_id] = 1.0
            elif ev.event_id in intermediate_ids:
                scores[ev.event_id] = 0.7
            else:
                scores[ev.event_id] = 0.1

    elif intent == "first_occurrence":
        earliest_ids: set[str] = set()
        later_ids: set[str] = set()
        target_concepts = anchor_set if anchor_set else set(by_concept.keys())

        for c_key in target_concepts:
            series = by_concept.get(c_key, [])
            if series:
                earliest_ids.add(series[0].event_id)
                for lat_ev in series[1:]:
                    later_ids.add(lat_ev.event_id)

        for ev in patient_events:
            if ev.event_id in earliest_ids:
                scores[ev.event_id] = 1.0
            elif ev.event_id in later_ids:
                scores[ev.event_id] = 0.3
            else:
                scores[ev.event_id] = 0.1

    elif intent == "persistent_fact":
        for ev in patient_events:
            st = states[ev.event_id]
            if st == "PERSISTENT":
                scores[ev.event_id] = 1.0
            elif st in ("ACTIVE", "LISTED"):
                scores[ev.event_id] = 0.5
            else:
                scores[ev.event_id] = 0.2

    else:
        raise ValueError(f"Unsupported intent: {intent}")

    return scores
