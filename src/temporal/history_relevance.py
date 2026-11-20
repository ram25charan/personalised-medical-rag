"""Patient-History-Aware Relevance Module (Rh) (Sections 7.2, 12.2).

Computes how closely each candidate event relates to the question's anchor concepts:
  - Anchor concepts = (a) concepts found in the question via the concept lexicon,
    plus (b) the concepts of the top 5 events by Rq (for non-generic event types).
  - Relatedness of event e to anchor a (maximum over anchors):
      1.0 -> Same concept
      0.7 -> Same concept class or a link in `configs/relation_table.json`
      0.5 -> Same encounter (`encounter_id`) as an anchor event
      0.0 -> Unrelated (so history relevance never boosts information merely because it belongs to the same patient).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.schemas import Event

_GENERIC_CONCEPTS = {"vital_signs", "ed_visit", "outpatient_visit", "triage_assessment"}


def load_relation_table(path: str | Path = "configs/relation_table.json") -> dict[str, Any]:
    """Load the frozen clinical concept relation table and lexicon."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_question_anchors(
    question_text: str,
    relation_table: dict[str, Any],
    candidate_events: Optional[list[Event]] = None,
) -> list[str]:
    """Extract clinical anchor concepts from `question_text` using the lexicon and candidate concepts."""
    q_low = question_text.lower()
    found: list[str] = []
    concepts_dict = relation_table.get("concepts", {})

    for canonical, info in concepts_dict.items():
        synonyms = [canonical] + [s.lower() for s in info.get("synonyms", [])]
        for syn in synonyms:
            if syn and syn in q_low:
                if canonical not in found:
                    found.append(canonical)
                break

    if candidate_events:
        for ev in candidate_events:
            c_low = ev.concept.lower().strip()
            if (
                c_low
                and c_low not in _GENERIC_CONCEPTS
                and len(c_low) >= 4
                and c_low in q_low
                and c_low not in found
            ):
                found.append(c_low)

    return found


def build_anchor_set(
    question_text: str,
    candidate_events: list[Event],
    rq_scores: dict[str, float],
    relation_table: dict[str, Any],
    top_rq_n: int = 5,
) -> tuple[list[str], set[str]]:
    """Return `(anchor_concepts, anchor_encounter_ids)` per Section 7.2:
    (a) concepts found in the question via the lexicon, plus
    (b) concepts of the top 5 events by Rq (excluding generic routine vital/visit placeholders).
    """
    lexicon_anchors = extract_question_anchors(question_text, relation_table, candidate_events)
    anchors = list(lexicon_anchors)

    sorted_evs = sorted(
        candidate_events,
        key=lambda e: (rq_scores.get(e.event_id, 0.0), e.observed_time or ""),
        reverse=True,
    )
    top_evs = sorted_evs[:top_rq_n]
    for ev in top_evs:
        c = ev.concept.lower().strip()
        if c not in _GENERIC_CONCEPTS and c not in anchors:
            anchors.append(c)

    primary_set = set(lexicon_anchors) if lexicon_anchors else {
        e.concept.lower().strip()
        for e in top_evs[:2]
        if e.concept.lower().strip() not in _GENERIC_CONCEPTS
    }
    anchor_encounters: set[str] = set()
    for ev in candidate_events:
        if ev.concept.lower().strip() in primary_set and ev.encounter_id:
            anchor_encounters.add(ev.encounter_id)

    return anchors, anchor_encounters


def rh_scores(
    patient_events: list[Event],
    anchors: list[str],
    relation_table: dict[str, Any],
    anchor_encounters: Optional[set[str]] = None,
) -> dict[str, float]:
    """Compute History Relevance `Rh(e) in [0, 1]` for each event in `patient_events` (Section 7.2 & 12.2)."""
    concepts_dict = relation_table.get("concepts", {})
    anchor_list = [a.lower().strip() for a in anchors if a.strip()]
    anchor_set = set(anchor_list)

    related_to_anchors: set[str] = set()
    anchor_classes: set[str] = set()

    for a in anchor_list:
        if a in concepts_dict:
            info = concepts_dict[a]
            if info.get("concept_class"):
                anchor_classes.add(str(info["concept_class"]).lower().strip())
            for rel_c in info.get("related_concepts", []):
                related_to_anchors.add(rel_c.lower().strip())

    for c_name, info in concepts_dict.items():
        rels = {r.lower().strip() for r in info.get("related_concepts", [])}
        if rels & anchor_set:
            related_to_anchors.add(c_name.lower().strip())

    for ev in patient_events:
        c_low = ev.concept.lower().strip()
        if c_low in anchor_set and ev.concept_class and c_low not in _GENERIC_CONCEPTS:
            c_cls = ev.concept_class.lower().strip()
            if not c_cls.startswith("icd-"):
                anchor_classes.add(c_cls)

    enc_set = anchor_encounters or set()
    scores: dict[str, float] = {}

    for ev in patient_events:
        c_low = ev.concept.lower().strip()
        c_cls = (ev.concept_class or "").lower().strip()

        if c_low in anchor_set and c_low not in _GENERIC_CONCEPTS:
            scores[ev.event_id] = 1.0
        elif (
            c_low in related_to_anchors
            or (c_cls and not c_cls.startswith("icd-") and c_cls in anchor_classes and c_low not in _GENERIC_CONCEPTS)
        ):
            scores[ev.event_id] = 0.7
        elif ev.encounter_id and ev.encounter_id in enc_set:
            scores[ev.event_id] = 0.5
        else:
            scores[ev.event_id] = 0.0

    return scores
