"""Track R Event Construction and Timeline Builder (Sections 6.5, 6.6, 6.7, 12.2).

Converts the six raw MIMIC-IV-ED demo tables into the unified `Event` representation,
collapses duplicate normalised medications within the same stay while retaining all source rows
in provenance, orders events chronologically per patient, and produces relative timelines and
the real-data scenario coverage report.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import pandas as pd

from src.data.normalisation import (
    NORMALISATION_VERSION,
    build_gsn_canonical_map,
    normalise_diagnosis,
    normalise_medication,
)
from src.schemas import Event


def _clean_val(val: Any) -> Any:
    """Convert pandas NaN/NaT to None and numpy scalars to native Python types."""
    if val is None or pd.isna(val):
        return None
    if hasattr(val, "item"):
        return val.item()
    return val


def _format_vitals_summary(row: pd.Series) -> tuple[str, dict[str, Any]]:
    """Build a human-readable vital sign summary string and structured attributes dict."""
    fields = [
        ("temperature", "Temp", "F"),
        ("heartrate", "HR", "bpm"),
        ("resprate", "RR", "/min"),
        ("o2sat", "SpO2", "%"),
        ("sbp", "SBP", "mmHg"),
        ("dbp", "DBP", "mmHg"),
        ("rhythm", "Rhythm", ""),
        ("pain", "Pain", ""),
        ("acuity", "Acuity", ""),
    ]
    parts: list[str] = []
    attrs: dict[str, Any] = {}
    for col, label, unit in fields:
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
            val = _clean_val(row[col])
            attrs[col] = val
            unit_str = f" {unit}" if unit else ""
            parts.append(f"{label}: {val}{unit_str}")
    return (", ".join(parts) if parts else "No vital values recorded"), attrs


def build_events(raw: dict[str, pd.DataFrame]) -> list[Event]:
    """Construct normalised Track R `Event` objects from raw MIMIC-IV-ED demo tables.

    Follows Section 6.5 Table 5:
    - `edstays` -> `visit` (`intime`, `outtime`, interval basis)
    - `triage` -> `triage_observation` (visit `intime`, interval basis)
    - `medrecon` -> `reported_home_medication` (`charttime`, snapshot basis; intra-stay duplicates collapsed)
    - `pyxis` -> `ed_administered_medication` (`charttime`, interval basis; administration, not prescription)
    - `diagnosis` -> `diagnosis_at_visit` (visit `intime`, interval basis; keeps `seq_num`, `icd_code`, `icd_version`)
    - `vitalsign` -> `vital_sign` (`charttime`, interval basis)
    """
    edstays = raw["edstays"]
    diagnosis = raw["diagnosis"]
    medrecon = raw["medrecon"]
    pyxis = raw["pyxis"]
    triage = raw["triage"]
    vitalsign = raw["vitalsign"]

    gsn_to_concept, gsn_to_class = build_gsn_canonical_map(medrecon, pyxis)

    stay_lookup: dict[int, dict[str, Any]] = {}
    for idx, row in edstays.iterrows():
        stay_id = int(row["stay_id"])
        stay_lookup[stay_id] = {
            "subject_id": int(row["subject_id"]),
            "hadm_id": _clean_val(row.get("hadm_id")),
            "intime": str(row["intime"]).strip(),
            "outtime": str(row["outtime"]).strip(),
            "disposition": _clean_val(row.get("disposition")),
            "arrival_transport": _clean_val(row.get("arrival_transport")),
            "gender": _clean_val(row.get("gender")),
            "race": _clean_val(row.get("race")),
            "row_idx": int(idx),
        }

    patient_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for stay_id, info in stay_lookup.items():
        pid = f"R{info['subject_id']}"
        disp = info["disposition"] or "UNKNOWN"
        transport = info["arrival_transport"] or "UNKNOWN"
        text = (
            f"ED visit (stay {stay_id}): admitted {info['intime']}, "
            f"discharged {info['outtime']}, disposition {disp}, arrival via {transport}"
        )
        patient_records[pid].append(
            {
                "event_type": "visit",
                "concept": "ed_visit",
                "concept_class": "Encounter",
                "text": text,
                "observed_time": info["intime"],
                "valid_start": info["intime"],
                "valid_end": info["outtime"],
                "date_precision": "day",
                "status_basis": "interval",
                "source_table": "edstays",
                "source_rows": [info["row_idx"]],
                "encounter_id": str(stay_id),
                "attributes": {
                    "hadm_id": info["hadm_id"],
                    "disposition": info["disposition"],
                    "arrival_transport": info["arrival_transport"],
                    "gender": info["gender"],
                    "race": info["race"],
                },
            }
        )

    for idx, row in triage.iterrows():
        stay_id = int(row["stay_id"])
        subject_id = int(row["subject_id"])
        pid = f"R{subject_id}"
        stay_info = stay_lookup.get(stay_id, {})
        intime = stay_info.get("intime")
        vitals_str, vitals_attrs = _format_vitals_summary(row)
        cc = _clean_val(row.get("chiefcomplaint"))
        vitals_attrs["chiefcomplaint"] = cc
        cc_str = f"Chief complaint: {cc}. " if cc else ""
        text = f"Triage observation at ED arrival: {cc_str}{vitals_str}"
        patient_records[pid].append(
            {
                "event_type": "triage_observation",
                "concept": (str(cc).strip().lower() if cc else "triage_assessment"),
                "concept_class": "Triage",
                "text": text,
                "observed_time": intime,
                "valid_start": intime,
                "valid_end": intime,
                "date_precision": "day" if intime else "unknown",
                "status_basis": "interval",
                "source_table": "triage",
                "source_rows": [int(idx)],
                "encounter_id": str(stay_id),
                "attributes": vitals_attrs,
            }
        )

    medrecon_groups: dict[tuple[int, str], list[tuple[int, pd.Series, Optional[str]]]] = defaultdict(list)
    for idx, row in medrecon.iterrows():
        stay_id = int(row["stay_id"])
        raw_name = str(row["name"]).strip()
        gsn_val = _clean_val(row.get("gsn"))
        etc_desc = _clean_val(row.get("etcdescription"))
        concept, concept_class = normalise_medication(
            raw_name, gsn_val, etc_desc, gsn_to_concept, gsn_to_class
        )
        medrecon_groups[(stay_id, concept)].append((int(idx), row, concept_class))

    for (stay_id, concept), entries in medrecon_groups.items():
        _, first_row, first_class = entries[0]
        subject_id = int(first_row["subject_id"])
        pid = f"R{subject_id}"
        source_rows = [e[0] for e in entries]
        raw_names = sorted({str(e[1]["name"]).strip() for e in entries})
        gsns = sorted({int(e[1]["gsn"]) for e in entries if pd.notna(e[1].get("gsn"))})
        etccodes = sorted(
            {int(float(e[1]["etccode"])) for e in entries if pd.notna(e[1].get("etccode"))}
        )
        etcdescs = sorted(
            {
                str(e[1]["etcdescription"]).strip()
                for e in entries
                if pd.notna(e[1].get("etcdescription")) and str(e[1]["etcdescription"]).strip()
            }
        )
        ndcs = sorted({str(e[1]["ndc"]).strip() for e in entries if pd.notna(e[1].get("ndc"))})
        charttimes = sorted(
            {str(e[1]["charttime"]).strip() for e in entries if pd.notna(e[1].get("charttime"))}
        )
        obs_time = charttimes[0] if charttimes else stay_lookup.get(stay_id, {}).get("intime")
        chosen_class = etcdescs[0] if etcdescs else first_class
        class_note = f" ({chosen_class})" if chosen_class else ""
        raw_note = f" [raw: {'; '.join(raw_names)}]" if raw_names else ""
        text = f"Reported home medication on reconciliation list: {concept}{class_note}{raw_note}"

        patient_records[pid].append(
            {
                "event_type": "reported_home_medication",
                "concept": concept,
                "concept_class": chosen_class,
                "text": text,
                "observed_time": obs_time,
                "valid_start": obs_time,
                "valid_end": None,
                "date_precision": "day" if obs_time else "unknown",
                "status_basis": "snapshot",
                "source_table": "medrecon",
                "source_rows": source_rows,
                "encounter_id": str(stay_id),
                "attributes": {
                    "raw_names": raw_names,
                    "gsns": gsns,
                    "etccodes": etccodes,
                    "etcdescriptions": etcdescs,
                    "ndcs": ndcs,
                    "collapsed_row_count": len(source_rows),
                    "normalisation_version": NORMALISATION_VERSION,
                },
            }
        )

    for idx, row in pyxis.iterrows():
        stay_id = int(row["stay_id"])
        subject_id = int(row["subject_id"])
        pid = f"R{subject_id}"
        raw_name = str(row["name"]).strip()
        gsn_val = _clean_val(row.get("gsn"))
        concept, concept_class = normalise_medication(
            raw_name, gsn_val, None, gsn_to_concept, gsn_to_class
        )
        obs_time = str(row["charttime"]).strip() if pd.notna(row.get("charttime")) else None
        class_note = f" ({concept_class})" if concept_class else ""
        text = f"ED medication dispensed (Pyxis): {concept}{class_note} [raw: {raw_name}]"
        patient_records[pid].append(
            {
                "event_type": "ed_administered_medication",
                "concept": concept,
                "concept_class": concept_class,
                "text": text,
                "observed_time": obs_time,
                "valid_start": obs_time,
                "valid_end": obs_time,
                "date_precision": "day" if obs_time else "unknown",
                "status_basis": "interval",
                "source_table": "pyxis",
                "source_rows": [int(idx)],
                "encounter_id": str(stay_id),
                "attributes": {
                    "raw_name": raw_name,
                    "gsn": int(float(gsn_val)) if gsn_val is not None else None,
                    "med_rn": _clean_val(row.get("med_rn")),
                    "gsn_rn": _clean_val(row.get("gsn_rn")),
                    "normalisation_version": NORMALISATION_VERSION,
                },
            }
        )

    for idx, row in diagnosis.iterrows():
        stay_id = int(row["stay_id"])
        subject_id = int(row["subject_id"])
        pid = f"R{subject_id}"
        stay_info = stay_lookup.get(stay_id, {})
        intime = stay_info.get("intime")
        outtime = stay_info.get("outtime")
        icd_code = str(row["icd_code"]).strip()
        icd_version = int(row["icd_version"])
        icd_title = str(row["icd_title"]).strip()
        seq_num = int(row["seq_num"])
        concept, concept_class = normalise_diagnosis(icd_title, icd_code, icd_version)
        text = (
            f"ED visit diagnosis (seq {seq_num}, ICD-{icd_version} {icd_code}): {icd_title}"
        )
        patient_records[pid].append(
            {
                "event_type": "diagnosis_at_visit",
                "concept": concept,
                "concept_class": concept_class,
                "text": text,
                "observed_time": intime,
                "valid_start": intime,
                "valid_end": outtime,
                "date_precision": "day" if intime else "unknown",
                "status_basis": "interval",
                "source_table": "diagnosis",
                "source_rows": [int(idx)],
                "encounter_id": str(stay_id),
                "attributes": {
                    "seq_num": seq_num,
                    "icd_code": icd_code,
                    "icd_version": icd_version,
                    "icd_title": icd_title,
                },
            }
        )

    for idx, row in vitalsign.iterrows():
        stay_id = int(row["stay_id"])
        subject_id = int(row["subject_id"])
        pid = f"R{subject_id}"
        obs_time = str(row["charttime"]).strip() if pd.notna(row.get("charttime")) else None
        vitals_str, vitals_attrs = _format_vitals_summary(row)
        text = f"ED vital signs measurement: {vitals_str}"
        patient_records[pid].append(
            {
                "event_type": "vital_sign",
                "concept": "vital_signs",
                "concept_class": "Vital Signs",
                "text": text,
                "observed_time": obs_time,
                "valid_start": obs_time,
                "valid_end": obs_time,
                "date_precision": "day" if obs_time else "unknown",
                "status_basis": "interval",
                "source_table": "vitalsign",
                "source_rows": [int(idx)],
                "encounter_id": str(stay_id),
                "attributes": vitals_attrs,
            }
        )

    type_priority = {
        "visit": 0,
        "triage_observation": 1,
        "diagnosis_at_visit": 2,
        "reported_home_medication": 3,
        "vital_sign": 4,
        "ed_administered_medication": 5,
    }

    all_events: list[Event] = []
    for pid in sorted(patient_records.keys()):
        records = patient_records[pid]
        records.sort(
            key=lambda r: (
                r["observed_time"] or "9999-12-31",
                type_priority.get(r["event_type"], 99),
                r["concept"],
                r["source_rows"][0] if r["source_rows"] else 0,
            )
        )
        for seq_idx, rec in enumerate(records, start=1):
            event_id = f"{pid}_{seq_idx:04d}"
            all_events.append(
                Event(
                    event_id=event_id,
                    patient_id=pid,
                    track="R",
                    event_type=rec["event_type"],
                    concept=rec["concept"],
                    concept_class=rec["concept_class"],
                    text=rec["text"],
                    observed_time=rec["observed_time"],
                    valid_start=rec["valid_start"],
                    valid_end=rec["valid_end"],
                    date_precision=rec["date_precision"],
                    time_is_shifted=True,
                    status_basis=rec["status_basis"],
                    source_type="real",
                    source_dataset="mimic_iv_ed_demo_v2.2",
                    source_table=rec["source_table"],
                    source_rows=rec["source_rows"],
                    encounter_id=rec["encounter_id"],
                    confidence=1.0,
                    attributes=rec["attributes"],
                )
            )

    return all_events


def build_patient_timelines(events: list[Event]) -> dict[str, Any]:
    """Build per-patient chronological timelines using relative time offsets (Section 6.5.6).

    Because MIMIC dates are shifted per patient, relative offsets in days/hours from the
    patient's earliest timestamp are computed alongside encounter anchors (`outtime` as_of_date).
    """
    by_patient: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        by_patient[ev.patient_id].append(ev)

    timelines: dict[str, Any] = {}
    for pid in sorted(by_patient.keys()):
        p_events = by_patient[pid]
        parsed_times = [
            pd.to_datetime(ev.observed_time, errors="coerce")
            for ev in p_events
            if ev.observed_time is not None
        ]
        valid_times = [t for t in parsed_times if pd.notna(t)]
        t0 = min(valid_times) if valid_times else None
        t_max = max(valid_times) if valid_times else None
        span_days = (
            round((t_max - t0).total_seconds() / 86400.0, 3)
            if (t0 is not None and t_max is not None)
            else 0.0
        )

        visits = [ev for ev in p_events if ev.event_type == "visit"]
        visit_anchors = []
        for v_idx, v in enumerate(visits, start=1):
            v_in = pd.to_datetime(v.valid_start, errors="coerce")
            v_out = pd.to_datetime(v.valid_end, errors="coerce")
            rel_day_in = (
                round((v_in - t0).total_seconds() / 86400.0, 3)
                if (t0 is not None and pd.notna(v_in))
                else 0.0
            )
            rel_day_out = (
                round((v_out - t0).total_seconds() / 86400.0, 3)
                if (t0 is not None and pd.notna(v_out))
                else 0.0
            )
            visit_anchors.append(
                {
                    "visit_number": v_idx,
                    "stay_id": v.encounter_id,
                    "intime": v.valid_start,
                    "outtime_as_of_date": v.valid_end,
                    "relative_day_start": rel_day_in,
                    "relative_day_end": rel_day_out,
                    "disposition": v.attributes.get("disposition"),
                }
            )

        timeline_events = []
        for ev in p_events:
            t_ev = pd.to_datetime(ev.observed_time, errors="coerce")
            rel_days = (
                round((t_ev - t0).total_seconds() / 86400.0, 4)
                if (t0 is not None and pd.notna(t_ev))
                else None
            )
            timeline_events.append(
                {
                    "event_id": ev.event_id,
                    "event_type": ev.event_type,
                    "concept": ev.concept,
                    "observed_time": ev.observed_time,
                    "relative_day": rel_days,
                    "encounter_id": ev.encounter_id,
                    "status_basis": ev.status_basis,
                }
            )

        timelines[pid] = {
            "patient_id": pid,
            "track": "R",
            "total_events": len(p_events),
            "visit_count": len(visits),
            "span_days": span_days,
            "visit_anchors": visit_anchors,
            "events": timeline_events,
        }

    return timelines


def build_coverage_report(events: list[Event]) -> dict[str, Any]:
    """Analyze which benchmark categories and scenarios Track R supports (Sections 6.5.8, 6.8, 8.5)."""
    by_patient: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        by_patient[ev.patient_id].append(ev)

    event_type_counts: dict[str, int] = defaultdict(int)
    for ev in events:
        event_type_counts[ev.event_type] += 1

    patients_with_med_changes = 0
    consecutive_visit_pairs = 0
    pairs_with_added_or_dropped_meds = 0
    patients_with_recurring_diagnoses = 0
    patients_with_within_visit_vital_series = 0
    stays_with_medrecon_pyxis_overlap = 0

    for p_events in by_patient.values():
        visits = [ev for ev in p_events if ev.event_type == "visit"]
        med_by_stay: dict[str, set[str]] = defaultdict(set)
        pyxis_by_stay: dict[str, set[str]] = defaultdict(set)
        diag_by_stay: dict[str, set[str]] = defaultdict(set)
        vitals_by_stay: dict[str, int] = defaultdict(int)

        for ev in p_events:
            sid = ev.encounter_id or ""
            if ev.event_type == "reported_home_medication":
                med_by_stay[sid].add(ev.concept)
            elif ev.event_type == "ed_administered_medication":
                pyxis_by_stay[sid].add(ev.concept)
            elif ev.event_type == "diagnosis_at_visit":
                diag_by_stay[sid].add(ev.concept)
            elif ev.event_type == "vital_sign":
                vitals_by_stay[sid] += 1

        med_stays = [
            v.encounter_id for v in visits if v.encounter_id in med_by_stay
        ]
        if len(med_stays) >= 2:
            patient_changed = False
            for i in range(len(med_stays) - 1):
                consecutive_visit_pairs += 1
                s1, s2 = med_stays[i], med_stays[i + 1]
                added = med_by_stay[s2] - med_by_stay[s1]
                dropped = med_by_stay[s1] - med_by_stay[s2]
                if added or dropped:
                    pairs_with_added_or_dropped_meds += 1
                    patient_changed = True
            if patient_changed:
                patients_with_med_changes += 1

        diag_counts: dict[str, int] = defaultdict(int)
        for sid, concepts in diag_by_stay.items():
            for c in concepts:
                diag_counts[c] += 1
        if any(cnt >= 2 for cnt in diag_counts.values()):
            patients_with_recurring_diagnoses += 1

        if any(cnt >= 3 for cnt in vitals_by_stay.values()):
            patients_with_within_visit_vital_series += 1

        for sid in set(med_by_stay.keys()) | set(pyxis_by_stay.keys()):
            if med_by_stay.get(sid) and pyxis_by_stay.get(sid):
                stays_with_medrecon_pyxis_overlap += 1

    return {
        "schema_version": "1.0",
        "total_events": len(events),
        "total_patients": len(by_patient),
        "event_type_counts": dict(event_type_counts),
        "longitudinal_evidence_counts": {
            "consecutive_medrecon_visit_pairs": consecutive_visit_pairs,
            "visit_pairs_with_med_add_or_drop": pairs_with_added_or_dropped_meds,
            "patients_with_med_list_changes": patients_with_med_changes,
            "patients_with_recurring_diagnoses": patients_with_recurring_diagnoses,
            "patients_with_3plus_vitals_in_a_visit": patients_with_within_visit_vital_series,
            "stays_with_both_medrecon_and_pyxis": stays_with_medrecon_pyxis_overlap,
        },
        "category_support": {
            "1_temporal_ordering": {
                "supported": True,
                "basis": "Sequence of ED visits, diagnoses across visits, and within-visit vital/Pyxis timestamps.",
            },
            "2_current_vs_historical": {
                "supported": True,
                "basis": "Snapshot-basis: medications LISTED on visit N vs NOT_LISTED_LATEST on visit N after appearing on an earlier visit.",
                "limitation": "Snapshot list only; absence from a later list is not proof of clinical discontinuation.",
            },
            "3_longitudinal_change": {
                "supported": True,
                "basis": f"{pairs_with_added_or_dropped_meds} consecutive visit pairs have medications added or dropped; within-visit vital sign series.",
            },
            "4_irrelevant_history": {
                "supported": True,
                "basis": "Multi-visit patients with up to 23 visits and dense vital/medication records around a specific query concept.",
            },
            "5_missing_or_ambiguous_dates": {
                "supported": "limited",
                "basis": "25 stays of multi-visit patients lack medrecon lists; home medications have no prescription start/end dates (U1/U4 uncertainty).",
            },
            "6_conflicting_information": {
                "supported": False,
                "basis": "No explicit active-vs-stopped or dose-conflict records exist in MIMIC-IV-ED demo (medrecon vs pyxis is an informational provenance note per Section 7.5). Fully tested in Track S.",
            },
            "7_insufficient_evidence": {
                "supported": True,
                "basis": "Constructed by querying unrecorded medications, labs, or conditions not present in the patient's ED history.",
            },
        },
    }
