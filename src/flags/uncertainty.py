"""Rule-based Uncertainty-Aware Evidence Handling (U1-U5) (Sections 7.6, 12.2).

Flags:
  - U1: Medication with no recorded end date (status inferred).
  - U2: Date precision coarser than a day (`month` or `year`).
  - U3: Date missing (`observed_time is None` or `date_precision == "unknown"`).
  - U4: Snapshot or reconciliation source (a listing is not proof of an active prescription).
  - U5: Insufficient evidence: no event reaches the Rq threshold `tau` for the anchor concept
        (or no candidate event matches the question's specific queried concept).
"""

from __future__ import annotations

from typing import Optional

from src.schemas import Event, UncertaintyFlag


def uncertainty_flags(
    selected_event_ids: list[str],
    candidate_events: list[Event],
    rq_scores: dict[str, float],
    question_anchors: list[str],
    tau: float = 0.35,
    question_text: Optional[str] = None,
    relevant_concepts: Optional[set[str]] = None,
) -> tuple[list[UncertaintyFlag], bool]:
    """Compute U1-U4 flags on selected events and U5 (`insufficient_evidence`) (Sections 7.6, 12.2)."""
    ev_by_id = {e.event_id: e for e in candidate_events}
    selected_evs = [
        ev_by_id[eid]
        for eid in selected_event_ids
        if eid in ev_by_id
        and (
            relevant_concepts is None
            or ev_by_id[eid].concept.lower().strip() in relevant_concepts
        )
    ]

    u1_ids: list[str] = []
    u2_ids: list[str] = []
    u3_ids: list[str] = []
    u4_ids: list[str] = []

    for ev in selected_evs:
        if ev.event_type in ("medication_order", "reported_home_medication") and ev.valid_end is None:
            u1_ids.append(ev.event_id)

        if ev.date_precision in ("month", "year"):
            u2_ids.append(ev.event_id)

        if ev.observed_time is None or ev.valid_start is None or ev.date_precision == "unknown":
            u3_ids.append(ev.event_id)

        if ev.status_basis == "snapshot" or ev.source_table == "medrecon":
            u4_ids.append(ev.event_id)

    flags: list[UncertaintyFlag] = []
    if u1_ids:
        flags.append(
            UncertaintyFlag(
                flag="U1",
                event_ids=u1_ids,
                description="Medication has no recorded end date; active status is inferred.",
            )
        )
    if u2_ids:
        flags.append(
            UncertaintyFlag(
                flag="U2",
                event_ids=u2_ids,
                description="Date precision is coarser than a day (month or year only).",
            )
        )
    if u3_ids:
        flags.append(
            UncertaintyFlag(
                flag="U3",
                event_ids=u3_ids,
                description="Observation or start date is missing/unknown.",
            )
        )
    if u4_ids:
        flags.append(
            UncertaintyFlag(
                flag="U4",
                event_ids=u4_ids,
                description="Snapshot/reconciliation source; listing is not proof of an active prescription or clinical discontinuation.",
            )
        )

    insufficient_evidence = False
    anchor_set = {a.lower().strip() for a in question_anchors if a.strip()}
    if not candidate_events:
        insufficient_evidence = True
    elif anchor_set:
        matching_anchor_events = [
            ev
            for ev in candidate_events
            if ev.concept.lower().strip() in anchor_set
            or any(a in ev.text.lower() for a in anchor_set)
        ]
        if not matching_anchor_events:
            insufficient_evidence = True
        else:
            max_anchor_rq = max(rq_scores.get(ev.event_id, 0.0) for ev in matching_anchor_events)
            if max_anchor_rq < tau:
                insufficient_evidence = True
    else:
        if question_text:
            stop_words = {
                "what", "when", "where", "which", "who", "how", "is", "are", "was", "were",
                "the", "patient", "currently", "taking", "as", "of", "in", "this", "their",
                "history", "record", "recorded", "up", "to", "there", "any", "ed", "visit",
                "status", "medication", "medications", "for", "has", "had", "been", "does",
                "did", "first", "latest", "ending", "compared", "with", "earlier", "or", "and",
            }
            import re
            q_words = [
                w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", question_text.lower())
                if w not in stop_words and not w.isdigit() and len(w) >= 4
            ]
            if q_words:
                corpus_blob = " ".join(
                    f"{e.concept} {e.concept_class or ''} {e.text}".lower() for e in candidate_events
                )
                if not any(w in corpus_blob for w in q_words):
                    insufficient_evidence = True

    return flags, insufficient_evidence
