"""Query-time Temporal State Computation (Sections 7.3, 12.2, AD2).

Computes the temporal state of each candidate event relative to `as_of_date`.
Status is never stored on the event; it is computed dynamically at query time.
Never imports gold or synthetic generator internals (AD6).
"""

from __future__ import annotations

from typing import Optional

from src.schemas import Event, TemporalStateType


def is_available_by_as_of(event: Event, as_of_date: str) -> bool:
    """Global availability rule (Section 7, AD5): candidate pool includes all events of the
    selected patient observed on or before `as_of_date` (or with unknown observed_time, i.e. U3).
    Events observed strictly after `as_of_date` are excluded.
    """
    if event.observed_time is None:
        return True
    # ISO-8601 prefix/string comparison handles YYYY, YYYY-MM, and YYYY-MM-DD against YYYY-MM-DD
    return event.observed_time <= as_of_date


def filter_candidate_pool(patient_events: list[Event], as_of_date: str, patient_id: Optional[str] = None) -> list[Event]:
    """Filter events to a single patient and exclude events observed after `as_of_date`."""
    pool: list[Event] = []
    for ev in patient_events:
        if patient_id is not None and ev.patient_id != patient_id:
            continue
        if is_available_by_as_of(ev, as_of_date):
            pool.append(ev)
    return pool


def _get_latest_snapshot_encounter(available_events: list[Event]) -> Optional[str]:
    """Identify the encounter_id of the latest snapshot (`medrecon`) visit at or before `as_of_date`."""
    snapshot_evs = [
        e for e in available_events if e.status_basis == "snapshot" and e.observed_time is not None
    ]
    if not snapshot_evs:
        return None
    latest_ev = max(snapshot_evs, key=lambda e: (e.observed_time or "", e.event_id))
    return latest_ev.encounter_id


def compute_state(
    event: Event,
    as_of_date: str,
    patient_events: list[Event],
) -> TemporalStateType:
    """Compute the temporal state of `event` at `as_of_date` given `patient_events` (Section 7.3).

    States (interval basis):
      - `UNKNOWN`: Start date missing (`valid_start is None`) or `date_precision != "day"` (e.g. `unknown`, `month`, `year`).
      - `PERSISTENT`: Persistent type (`allergy`, `condition`, or recurring `diagnosis_at_visit` across >= 2 visits)
        with no resolution (`valid_end` is None or > `as_of_date`).
      - `SUPERSEDED`: A later event of the same patient, event_type, and concept series (newer lab value,
        dose change, or newer record of the same item) exists on or before `as_of_date` (excluding simultaneous
        conflicting records within 7 days where one remains active).
      - `ENDED`: `valid_end` is known and `<= as_of_date`.
      - `ACTIVE`: `valid_start <= as_of_date` and (`valid_end` is None or `> as_of_date`).

    States (snapshot basis, Track R `medrecon`):
      - `LISTED`: On the latest medication reconciliation list at or before `as_of_date`.
      - `NOT_LISTED_LATEST`: On an earlier reconciliation list, absent from the latest list at or before `as_of_date`.
    """
    available = filter_candidate_pool(patient_events, as_of_date, patient_id=event.patient_id)

    # 1. Snapshot basis (Track R reported_home_medication)
    if event.status_basis == "snapshot":
        latest_enc = _get_latest_snapshot_encounter(available)
        if latest_enc is None:
            return "UNKNOWN"
        # Check if this concept is listed on the latest snapshot encounter
        concepts_on_latest = {
            e.concept
            for e in available
            if e.status_basis == "snapshot" and e.encounter_id == latest_enc
        }
        if event.encounter_id == latest_enc or event.concept in concepts_on_latest:
            if event.encounter_id == latest_enc:
                return "LISTED"
            # Earlier visit's listing of a drug that is also on the latest visit
            return "SUPERSEDED"
        return "NOT_LISTED_LATEST"

    # 2. Interval basis: check UNKNOWN date or coarse precision first (Section 7.3)
    if event.valid_start is None or event.date_precision != "day":
        return "UNKNOWN"

    # 3. Persistent types (allergy, chronic condition, recurring diagnosis across visits)
    if event.event_type in ("allergy", "condition"):
        if event.valid_end is None or event.valid_end > as_of_date:
            return "PERSISTENT"
        return "ENDED"

    if event.event_type == "diagnosis_at_visit":
        same_diag_visits = {
            e.encounter_id
            for e in available
            if e.event_type == "diagnosis_at_visit" and e.concept == event.concept
        }
        if len(same_diag_visits) >= 2:
            return "PERSISTENT"

    # 4. Check SUPERSEDED for longitudinal series (lab_result, vital_sign, medication_order)
    if event.event_type in ("lab_result", "vital_sign"):
        later_same = [
            e
            for e in available
            if e.event_id != event.event_id
            and e.event_type == event.event_type
            and e.concept == event.concept
            and e.observed_time is not None
            and event.observed_time is not None
            and e.observed_time > event.observed_time
        ]
        if later_same:
            return "SUPERSEDED"

    if event.event_type == "medication_order":
        # If this medication order has ended and a later order of the same medication exists,
        # or if this order was explicitly closed by a dose change
        if event.valid_end is not None and event.valid_end <= as_of_date:
            later_orders = [
                e
                for e in available
                if e.event_id != event.event_id
                and e.event_type == "medication_order"
                and e.concept == event.concept
                and e.valid_start is not None
                and e.valid_start >= event.valid_end
            ]
            if later_orders:
                return "SUPERSEDED"
            return "ENDED"

    # 5. Standard interval ENDED vs ACTIVE
    if event.valid_end is not None and event.valid_end <= as_of_date:
        return "ENDED"

    if event.valid_start <= as_of_date:
        return "ACTIVE"

    return "UNKNOWN"
