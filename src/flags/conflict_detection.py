"""Rule-based Conflict Detection Module (C1-C3) (Sections 7.5, 12.2).

Examines the selected events and all other available events of the patient that share a concept
with a selected event:
  - C1 status conflict: Same concept, overlapping validity, one source indicates active and another ended / discontinued.
  - C2 dose conflict: Same medication, overlapping validity, different doses (while both active/ongoing).
  - C3 date conflict: Same concept and type reported with dates differing by more than 30 days across sources.

On Track R, a mismatch between reconciliation (`medrecon`) and administration (`pyxis`) records
is reported as an informational provenance note, not as a conflict (Section 7.5).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Optional

from src.schemas import ConflictFlag, Event, ProvenanceNote
from src.temporal.state import filter_candidate_pool


def _parse_iso_day(date_str: Optional[str]) -> Optional[date]:
    if not date_str or len(date_str) < 10:
        return None
    try:
        return date.fromisoformat(date_str[:10])
    except ValueError:
        return None


def _intervals_overlap(ev1: Event, ev2: Event, as_of_date: str) -> bool:
    """Check whether two events' validity intervals overlap on or before `as_of_date`."""
    s1 = ev1.valid_start or ev1.observed_time
    s2 = ev2.valid_start or ev2.observed_time
    if not s1 or not s2:
        return False
    e1 = ev1.valid_end or as_of_date
    e2 = ev2.valid_end or as_of_date
    return s1 <= e2 and s2 <= e1


def detect_conflicts(
    selected_event_ids: list[str],
    patient_events: list[Event],
    as_of_date: str,
) -> tuple[list[ConflictFlag], list[ProvenanceNote]]:
    """Detect C1, C2, and C3 conflicts and Track R provenance notes (Sections 7.5, 12.2)."""
    available = filter_candidate_pool(patient_events, as_of_date)
    ev_by_id = {e.event_id: e for e in available}

    selected_evs = [ev_by_id[eid] for eid in selected_event_ids if eid in ev_by_id]
    selected_concepts = {
        e.concept.lower().strip()
        for e in selected_evs
        if e.concept.lower().strip() not in {"vital_signs", "ed_visit", "outpatient_visit", "triage_assessment"}
    }

    # Group all available patient events sharing a selected concept
    by_concept: dict[str, list[Event]] = defaultdict(list)
    for ev in available:
        c_low = ev.concept.lower().strip()
        if c_low in selected_concepts:
            by_concept[c_low].append(ev)

    conflicts: list[ConflictFlag] = []
    provenance_notes: list[ProvenanceNote] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    for concept, evs in sorted(by_concept.items()):
        if len(evs) < 2:
            continue

        # Track R check: mismatch between medrecon and pyxis is an informational provenance note
        has_medrecon = [e for e in evs if e.source_table == "medrecon"]
        has_pyxis = [e for e in evs if e.source_table == "pyxis"]
        if has_medrecon and has_pyxis:
            note_ids = sorted({e.event_id for e in (has_medrecon + has_pyxis)})
            provenance_notes.append(
                ProvenanceNote(
                    note_type="medrecon_vs_pyxis",
                    event_ids=note_ids,
                    description=(
                        f"Concept '{concept}' appears in both home medication reconciliation (medrecon) "
                        f"and ED dispensing (pyxis); treated as informational provenance per Section 7.5."
                    ),
                )
            )

        # Pairwise conflict check on interval-basis records
        n = len(evs)
        for i in range(n):
            for j in range(i + 1, n):
                e1, e2 = evs[i], evs[j]
                # Skip Track R medrecon vs pyxis pairs
                if {e1.source_table, e2.source_table} == {"medrecon", "pyxis"}:
                    continue
                if e1.status_basis == "snapshot" or e2.status_basis == "snapshot":
                    continue

                pair_ids = sorted([e1.event_id, e2.event_id])
                src1 = e1.attributes.get("clinical_source") or e1.source_table
                src2 = e2.attributes.get("clinical_source") or e2.source_table
                rep1 = e1.attributes.get("reported_status")
                rep2 = e2.attributes.get("reported_status")
                action1 = e1.attributes.get("action")
                action2 = e2.attributes.get("action")

                # Ignore normal sequential dose escalation where one order ends at the exact date the next starts
                if (
                    e1.valid_end is not None
                    and e2.valid_start is not None
                    and e1.valid_end <= e2.valid_start
                    and (action2 == "dose_change" or action1 == "dose_change")
                ):
                    continue
                if (
                    e2.valid_end is not None
                    and e1.valid_start is not None
                    and e2.valid_end <= e1.valid_start
                    and (action1 == "dose_change" or action2 == "dose_change")
                ):
                    continue

                overlap = _intervals_overlap(e1, e2, as_of_date)

                # C1: Status conflict (overlapping validity, one active/open and another ended/discontinued across sources)
                is_active_1 = (e1.valid_end is None or e1.valid_end > as_of_date) or rep1 == "active"
                is_ended_1 = (e1.valid_end is not None and e1.valid_end <= as_of_date) or rep1 == "ended"
                is_active_2 = (e2.valid_end is None or e2.valid_end > as_of_date) or rep2 == "active"
                is_ended_2 = (e2.valid_end is not None and e2.valid_end <= as_of_date) or rep2 == "ended"

                if overlap and ((is_active_1 and is_ended_2) or (is_active_2 and is_ended_1)):
                    # Check that they are concurrent conflicting reports (e.g. different clinical_source or overlapping open end)
                    if src1 != src2 or (e1.valid_end is None or e2.valid_end is None):
                        key = ("C1", pair_ids[0], pair_ids[1])
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            conflicts.append(
                                ConflictFlag(
                                    rule="C1",
                                    event_ids=pair_ids,
                                    description=f"Status conflict on '{concept}': overlapping records disagree on active vs ended status.",
                                )
                            )
                            continue

                # C2: Dose conflict (same medication, overlapping active validity, different doses)
                if e1.event_type == "medication_order" and e2.event_type == "medication_order":
                    dose1 = e1.attributes.get("dose")
                    dose2 = e2.attributes.get("dose")
                    if overlap and dose1 and dose2 and str(dose1).strip().lower() != str(dose2).strip().lower():
                        # Both open or concurrent overlapping active intervals
                        if (e1.valid_end is None and e2.valid_end is None) or (src1 and src2 and src1 != src2):
                            key = ("C2", pair_ids[0], pair_ids[1])
                            if key not in seen_pairs:
                                seen_pairs.add(key)
                                conflicts.append(
                                    ConflictFlag(
                                        rule="C2",
                                        event_ids=pair_ids,
                                        description=f"Dose conflict on '{concept}': overlapping records specify '{dose1}' vs '{dose2}'.",
                                    )
                                )
                                continue

                # C3: Date conflict (same concept and type reported across different sources with dates > 30 days apart)
                if e1.event_type == e2.event_type and src1 and src2 and src1 != src2:
                    d1 = _parse_iso_day(e1.observed_time)
                    d2 = _parse_iso_day(e2.observed_time)
                    if d1 and d2 and abs((d2 - d1).days) > 30:
                        key = ("C3", pair_ids[0], pair_ids[1])
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            conflicts.append(
                                ConflictFlag(
                                    rule="C3",
                                    event_ids=pair_ids,
                                    description=f"Date conflict on '{concept}': reported dates differ by {abs((d2 - d1).days)} days across sources.",
                                )
                            )

    return conflicts, provenance_notes
