"""Deterministic, seeded synthetic longitudinal patient generator (Sections 6.3, 6.8, 12.2).

No LLM writes patient histories. Generates ~100 synthetic patients (Track S) with 70-180 events
each, embedding realistic longitudinal clinical scenarios:
  - Chronic medication lifecycles with 6-8 routine refill/dispense records before discontinuation
    and switch to a new active medication (so budget k=5/10/20 faces real competition from stale refills).
  - Condition-linked regimens (e.g. Type 2 diabetes mellitus -> metformin + HbA1c series;
    Essential hypertension -> lisinopril/amlodipine) where individual medication/lab notes record
    the drug or test without repeating the disease name verbatim, testing History Relevance (Rh).
  - Serial lab trends with 8 longitudinal measurements over 3 years.
  - Persistent conditions and allergies, missing/ambiguous dates (U2/U3), and C1/C2 conflicts.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from src.schemas import Event


def _fmt_date(d: date) -> str:
    return d.isoformat()


class _PatientEventCollector:
    """Helper to accumulate raw event dicts for one synthetic patient before sorting & ID assignment."""

    def __init__(self, patient_id: str, generator_version: str, seed: int) -> None:
        self.patient_id = patient_id
        self.generator_version = generator_version
        self.seed = seed
        self.raw_events: list[dict[str, Any]] = []

    def add(
        self,
        tag: str,
        event_type: str,
        concept: str,
        concept_class: Optional[str],
        text: str,
        observed_time: Optional[str],
        valid_start: Optional[str],
        valid_end: Optional[str],
        date_precision: str = "day",
        status_basis: str = "interval",
        encounter_id: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> str:
        attrs = dict(attributes or {})
        attrs["generator_version"] = self.generator_version
        attrs["seed"] = self.seed
        self.raw_events.append(
            {
                "_tag": tag,
                "event_type": event_type,
                "concept": concept,
                "concept_class": concept_class,
                "text": text,
                "observed_time": observed_time,
                "valid_start": valid_start,
                "valid_end": valid_end,
                "date_precision": date_precision,
                "status_basis": status_basis,
                "encounter_id": encounter_id,
                "attributes": attrs,
            }
        )
        return tag

    def finalize(self) -> tuple[list[Event], dict[str, str]]:
        """Sort events chronologically, assign stable `<patient_id>_<seq>` IDs, and return tag -> event_id map."""
        type_priority = {
            "visit": 0,
            "triage_observation": 1,
            "condition": 2,
            "allergy": 3,
            "medication_order": 4,
            "lab_result": 5,
            "vital_sign": 6,
            "symptom": 7,
        }
        indexed = list(enumerate(self.raw_events))
        indexed.sort(
            key=lambda item: (
                item[1]["observed_time"] if item[1]["observed_time"] is not None else "0000-00-00",
                type_priority.get(item[1]["event_type"], 99),
                item[1]["concept"],
                item[0],
            )
        )
        events: list[Event] = []
        tag_to_id: dict[str, str] = {}
        for seq_idx, (_, rec) in enumerate(indexed, start=1):
            eid = f"{self.patient_id}_{seq_idx:04d}"
            tag_to_id[rec["_tag"]] = eid
            events.append(
                Event(
                    event_id=eid,
                    patient_id=self.patient_id,
                    track="S",
                    event_type=rec["event_type"],
                    concept=rec["concept"],
                    concept_class=rec["concept_class"],
                    text=rec["text"],
                    observed_time=rec["observed_time"],
                    valid_start=rec["valid_start"],
                    valid_end=rec["valid_end"],
                    date_precision=rec["date_precision"],
                    time_is_shifted=False,
                    status_basis=rec["status_basis"],
                    source_type="synthetic",
                    source_dataset="custom_generator",
                    source_table=None,
                    source_rows=[],
                    encounter_id=rec["encounter_id"],
                    confidence=1.0,
                    attributes=rec["attributes"],
                )
            )
        return events, tag_to_id


_DIABETES_DRUGS = [
    ("metformin", "Biguanide Antidiabetic", "500 mg", "1000 mg"),
    ("glipizide", "Sulfonylurea Antidiabetic", "5 mg", "10 mg"),
]
_HTN_PATTERNS = [
    (
        "lisinopril",
        "ACE Inhibitors",
        "10 mg",
        "20 mg",
        "amlodipine",
        "Calcium Channel Blockers",
        "5 mg",
    ),
    (
        "enalapril maleate",
        "ACE Inhibitors",
        "5 mg",
        "10 mg",
        "hydrochlorothiazide",
        "Thiazide Diuretics",
        "25 mg",
    ),
    (
        "amlodipine",
        "Calcium Channel Blockers",
        "5 mg",
        "10 mg",
        "metoprolol",
        "Beta-Adrenergic Blocking Agents",
        "50 mg",
    ),
]
_STATIN_PATTERNS = [
    ("atorvastatin", "Antihyperlipidemic - HMG-CoA Reductase Inhibitors (Statins)", "20 mg", "40 mg"),
    ("rosuvastatin", "Antihyperlipidemic - HMG-CoA Reductase Inhibitors (Statins)", "10 mg", "20 mg"),
    ("simvastatin", "Antihyperlipidemic - HMG-CoA Reductase Inhibitors (Statins)", "20 mg", "40 mg"),
]
_ALLERGY_PATTERNS = [
    ("penicillin allergy", "Drug Allergy", "hives and facial swelling"),
    ("sulfa allergy", "Drug Allergy", "maculopapular rash"),
    ("latex allergy", "Material Allergy", "contact urticaria"),
]
_LAB_TREND_PATTERNS = [
    ("hba1c", "Glycemic Control Laboratory Test", "%", [9.1, 8.7, 8.4, 8.0, 7.7, 7.4, 7.1, 6.8]),
    ("ldl cholesterol", "Lipid Panel Laboratory Test", "mg/dL", [174, 162, 150, 138, 126, 114, 104, 92]),
    ("serum creatinine", "Renal Function Laboratory Test", "mg/dL", [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.9]),
    ("tsh", "Thyroid Function Laboratory Test", "mIU/L", [8.4, 7.2, 6.1, 5.0, 4.2, 3.5, 2.9, 2.2]),
]
_CONFLICT_PATTERNS = [
    ("apixaban", "Direct Factor Xa Inhibitors", "5 mg", "2.5 mg"),
    ("furosemide", "Loop Diuretics", "20 mg", "40 mg"),
    ("levothyroxine", "Thyroid Hormones", "50 mcg", "75 mcg"),
    ("omeprazole", "Proton-Pump Inhibitors", "20 mg", "40 mg"),
]


def generate_synthetic_patients(
    config: dict[str, Any],
    seed: Optional[int] = None,
) -> tuple[list[Event], list[dict[str, Any]]]:
    """Deterministically generate synthetic patients, their events, and embedded scenario metadata."""
    actual_seed = int(seed if seed is not None else config.get("seed", 42))
    rng = random.Random(actual_seed)

    gen_version = str(config.get("generator_version", "1.0"))
    num_patients = int(config.get("num_patients", 100))
    dev_patients = int(config.get("dev_patients", 34))
    min_events = int(config.get("min_events_per_patient", 80))
    max_events = int(config.get("max_events_per_patient", 180))
    distractor_meds = config.get("distractor_medications", [])
    unrecorded_pool = config.get("unrecorded_query_concepts", [])

    all_events: list[Event] = []
    patient_blueprints: list[dict[str, Any]] = []

    for p_idx in range(1, num_patients + 1):
        pid = f"S_{p_idx:03d}"
        split = "dev" if p_idx <= dev_patients else "test"
        target_event_count = rng.randint(max(85, min_events), max_events)
        collector = _PatientEventCollector(pid, gen_version, actual_seed)

        base_date = date(2021, 1, 15) + timedelta(days=rng.randint(0, 90))
        as_of_dt = base_date + timedelta(days=rng.randint(1200, 1360))
        as_of_str = _fmt_date(as_of_dt)

        allergy_concept, allergy_class, allergy_reaction = _ALLERGY_PATTERNS[
            p_idx % len(_ALLERGY_PATTERNS)
        ]
        allergy_dt = base_date + timedelta(days=rng.randint(5, 25))
        collector.add(
            tag="allergy_1",
            event_type="allergy",
            concept=allergy_concept,
            concept_class=allergy_class,
            text=f"Documented {allergy_concept}: reaction {allergy_reaction} (persistent allergy).",
            observed_time=_fmt_date(allergy_dt),
            valid_start=_fmt_date(allergy_dt),
            valid_end=None,
            encounter_id=f"{pid}_ENC_01",
            attributes={"reaction": allergy_reaction, "persistent": True},
        )

        htn_pat = _HTN_PATTERNS[p_idx % len(_HTN_PATTERNS)]
        old_med, old_med_class, old_dose_1, old_dose_2, new_med, new_med_class, new_dose = htn_pat

        cond_dt_1 = base_date + timedelta(days=rng.randint(30, 50))
        collector.add(
            tag="cond_htn_first",
            event_type="condition",
            concept="essential (primary) hypertension",
            concept_class="Cardiovascular Disorder",
            text="Diagnosis recorded: essential (primary) hypertension.",
            observed_time=_fmt_date(cond_dt_1),
            valid_start=_fmt_date(cond_dt_1),
            valid_end=None,
            encounter_id=f"{pid}_ENC_02",
            attributes={"onset": _fmt_date(cond_dt_1), "status": "chronic"},
        )

        htn_followup_tags: list[str] = []
        for f_i in range(1, 9):
            f_dt = cond_dt_1 + timedelta(days=f_i * 120 + rng.randint(0, 15))
            if f_dt < as_of_dt - timedelta(days=20):
                f_tag = f"cond_htn_followup_{f_i}"
                htn_followup_tags.append(f_tag)
                collector.add(
                    tag=f_tag,
                    event_type="condition",
                    concept="essential (primary) hypertension",
                    concept_class="Cardiovascular Disorder",
                    text="Outpatient diagnosis check: essential (primary) hypertension.",
                    observed_time=_fmt_date(f_dt),
                    valid_start=_fmt_date(cond_dt_1),
                    valid_end=None,
                    encounter_id=f"{pid}_ENC_HTN_{f_i}",
                    attributes={"onset": _fmt_date(cond_dt_1), "status": "chronic"},
                )

        dm_diag_dt = base_date + timedelta(days=rng.randint(60, 85))
        collector.add(
            tag="cond_dm_first",
            event_type="condition",
            concept="type 2 diabetes mellitus",
            concept_class="Endocrine / Metabolic Disorder",
            text="Condition recorded: type 2 diabetes mellitus.",
            observed_time=_fmt_date(dm_diag_dt),
            valid_start=_fmt_date(dm_diag_dt),
            valid_end=None,
            encounter_id=f"{pid}_ENC_03",
            attributes={"onset": _fmt_date(dm_diag_dt), "status": "chronic"},
        )

        med_t1 = cond_dt_1 + timedelta(days=5)
        refill_tags: list[str] = []
        prev_dt = med_t1
        next_dt = prev_dt + timedelta(days=60)
        collector.add(
            tag="med_lifecycle_start",
            event_type="medication_order",
            concept=old_med,
            concept_class=old_med_class,
            text=f"Prescription order: {old_med} {old_dose_1} oral daily.",
            observed_time=_fmt_date(prev_dt),
            valid_start=_fmt_date(prev_dt),
            valid_end=_fmt_date(next_dt),
            encounter_id=f"{pid}_ENC_02",
            attributes={"dose": old_dose_1, "action": "start"},
        )
        prev_dt = next_dt

        for r_i in range(1, 7):
            r_end = prev_dt + timedelta(days=60)
            r_tag = f"med_lifecycle_refill_{r_i}"
            refill_tags.append(r_tag)
            collector.add(
                tag=r_tag,
                event_type="medication_order",
                concept=old_med,
                concept_class=old_med_class,
                text=f"Medication refill order: {old_med} {old_dose_1} oral daily (refill #{r_i}).",
                observed_time=_fmt_date(prev_dt),
                valid_start=_fmt_date(prev_dt),
                valid_end=_fmt_date(r_end),
                encounter_id=f"{pid}_ENC_REF_{r_i}",
                attributes={"dose": old_dose_1, "action": "refill"},
            )
            prev_dt = r_end

        med_t2 = prev_dt
        med_t3_stop = med_t2 + timedelta(days=90)
        collector.add(
            tag="med_lifecycle_dose_change",
            event_type="medication_order",
            concept=old_med,
            concept_class=old_med_class,
            text=f"Prescription dose change: {old_med} increased to {old_dose_2} oral daily; discontinued {old_dose_1} and stopped on {_fmt_date(med_t3_stop)}.",
            observed_time=_fmt_date(med_t2),
            valid_start=_fmt_date(med_t2),
            valid_end=_fmt_date(med_t3_stop),
            encounter_id=f"{pid}_ENC_04",
            attributes={"dose": old_dose_2, "action": "dose_change", "stopped": True},
        )

        med_t4_new = med_t3_stop + timedelta(days=10)
        collector.add(
            tag="med_lifecycle_new_active",
            event_type="medication_order",
            concept=new_med,
            concept_class=new_med_class,
            text=f"Active prescription order: {new_med} {new_dose} oral once daily.",
            observed_time=_fmt_date(med_t4_new),
            valid_start=_fmt_date(med_t4_new),
            valid_end=None,
            encounter_id=f"{pid}_ENC_06",
            attributes={"dose": new_dose, "action": "start"},
        )

        dm_med, dm_class, dm_dose1, dm_dose2 = _DIABETES_DRUGS[p_idx % len(_DIABETES_DRUGS)]
        dm_curr_dt = dm_diag_dt + timedelta(days=5)
        dm_old_refill_tags: list[str] = []
        for dr_i in range(1, 7):
            dr_end = dm_curr_dt + timedelta(days=75)
            dr_tag = f"med_dm_refill_{dr_i}"
            dm_old_refill_tags.append(dr_tag)
            collector.add(
                tag=dr_tag,
                event_type="medication_order",
                concept=dm_med,
                concept_class=dm_class,
                text=f"Medication order disp: {dm_med} {dm_dose1} twice daily (course #{dr_i}).",
                observed_time=_fmt_date(dm_curr_dt),
                valid_start=_fmt_date(dm_curr_dt),
                valid_end=_fmt_date(dr_end),
                encounter_id=f"{pid}_ENC_DM_{dr_i}",
                attributes={"dose": dm_dose1, "action": "refill"},
            )
            dm_curr_dt = dr_end

        collector.add(
            tag="med_chronic_active",
            event_type="medication_order",
            concept=dm_med,
            concept_class=dm_class,
            text=f"Current ongoing prescription: {dm_med} {dm_dose2} oral twice daily.",
            observed_time=_fmt_date(dm_curr_dt),
            valid_start=_fmt_date(dm_curr_dt),
            valid_end=None,
            encounter_id=f"{pid}_ENC_03",
            attributes={"dose": dm_dose2, "action": "start"},
        )

        lab_concept, lab_class, lab_unit, base_vals = _LAB_TREND_PATTERNS[
            p_idx % len(_LAB_TREND_PATTERNS)
        ]
        jitter = round((rng.random() - 0.5) * 0.2, 1)
        lab_vals = [round(v + jitter, 1) for v in base_vals]
        lab_tags: list[str] = []
        for l_i, l_val in enumerate(lab_vals, start=1):
            l_dt = base_date + timedelta(days=80 + l_i * 125)
            t_tag = f"lab_trend_{l_i}"
            lab_tags.append(t_tag)
            collector.add(
                tag=t_tag,
                event_type="lab_result",
                concept=lab_concept,
                concept_class=lab_class,
                text=f"Laboratory test result: {lab_concept} measured at {l_val} {lab_unit}.",
                observed_time=_fmt_date(l_dt),
                valid_start=_fmt_date(l_dt),
                valid_end=_fmt_date(l_dt),
                encounter_id=f"{pid}_ENC_LAB_{l_i}",
                attributes={"value": l_val, "unit": lab_unit, "series_index": l_i},
            )

        statin_concept, statin_class, statin_dose, _ = _STATIN_PATTERNS[
            p_idx % len(_STATIN_PATTERNS)
        ]
        ambig_month_str = f"{base_date.year + 1}-06"
        collector.add(
            tag="med_ambig_month",
            event_type="medication_order",
            concept=statin_concept,
            concept_class=statin_class,
            text=f"Patient reported starting {statin_concept} {statin_dose} in {ambig_month_str} (exact day unknown).",
            observed_time=ambig_month_str,
            valid_start=ambig_month_str,
            valid_end=None,
            date_precision="month",
            encounter_id=f"{pid}_ENC_AMBIG",
            attributes={"dose": statin_dose, "action": "start"},
        )
        collector.add(
            tag="cond_missing_date",
            event_type="condition",
            concept="gastroesophageal reflux disease",
            concept_class="Gastrointestinal Disorder",
            text="History of gastroesophageal reflux disease reported by patient; onset date unknown.",
            observed_time=None,
            valid_start=None,
            valid_end=None,
            date_precision="unknown",
            encounter_id=f"{pid}_ENC_AMBIG",
            attributes={"onset": None, "status": "historical_unverified"},
        )

        conf_med, conf_class, conf_dose_a, conf_dose_b = _CONFLICT_PATTERNS[
            p_idx % len(_CONFLICT_PATTERNS)
        ]
        conf_dt_a = base_date + timedelta(days=780)
        conf_dt_b = conf_dt_a + timedelta(days=5)
        conflict_type = "C1" if (p_idx % 2 == 0) else "C2"

        if conflict_type == "C1":
            collector.add(
                tag="conflict_a",
                event_type="medication_order",
                concept=conf_med,
                concept_class=conf_class,
                text=f"Outpatient clinic medication order: {conf_med} {conf_dose_a} daily is ACTIVE and ongoing.",
                observed_time=_fmt_date(conf_dt_a),
                valid_start=_fmt_date(conf_dt_a),
                valid_end=None,
                encounter_id=f"{pid}_ENC_CONF_1",
                attributes={"dose": conf_dose_a, "reported_status": "active", "clinical_source": "outpatient_clinic"},
            )
            collector.add(
                tag="conflict_b",
                event_type="medication_order",
                concept=conf_med,
                concept_class=conf_class,
                text=f"External pharmacy reconciliation record: {conf_med} {conf_dose_a} was discontinued / ended on {_fmt_date(conf_dt_b)}.",
                observed_time=_fmt_date(conf_dt_b),
                valid_start=_fmt_date(conf_dt_a),
                valid_end=_fmt_date(conf_dt_b),
                encounter_id=f"{pid}_ENC_CONF_2",
                attributes={"dose": conf_dose_a, "reported_status": "ended", "clinical_source": "external_pharmacy"},
            )
        else:
            collector.add(
                tag="conflict_a",
                event_type="medication_order",
                concept=conf_med,
                concept_class=conf_class,
                text=f"Primary care prescription record: {conf_med} {conf_dose_a} once daily (active).",
                observed_time=_fmt_date(conf_dt_a),
                valid_start=_fmt_date(conf_dt_a),
                valid_end=None,
                encounter_id=f"{pid}_ENC_CONF_1",
                attributes={"dose": conf_dose_a, "reported_status": "active", "clinical_source": "primary_care"},
            )
            collector.add(
                tag="conflict_b",
                event_type="medication_order",
                concept=conf_med,
                concept_class=conf_class,
                text=f"Specialist clinic medication list: {conf_med} {conf_dose_b} once daily (active, conflicting dose).",
                observed_time=_fmt_date(conf_dt_b),
                valid_start=_fmt_date(conf_dt_b),
                valid_end=None,
                encounter_id=f"{pid}_ENC_CONF_2",
                attributes={"dose": conf_dose_b, "reported_status": "active", "clinical_source": "specialist_clinic"},
            )

        future_dt = as_of_dt + timedelta(days=rng.randint(15, 60))
        collector.add(
            tag="future_event_1",
            event_type="lab_result",
            concept=lab_concept,
            concept_class=lab_class,
            text=f"Future post-reference laboratory result for {lab_concept}: 5.9 {lab_unit}.",
            observed_time=_fmt_date(future_dt),
            valid_start=_fmt_date(future_dt),
            valid_end=_fmt_date(future_dt),
            encounter_id=f"{pid}_ENC_FUTURE",
            attributes={"value": 5.9, "unit": lab_unit, "is_future_relative_to_default_as_of": True},
        )

        distractor_tags: list[str] = []
        current_count = len(collector.raw_events)
        needed = max(25, target_event_count - current_count)

        for d_i in range(1, needed + 1):
            d_offset = rng.randint(10, (as_of_dt - base_date).days - 10)
            d_dt = base_date + timedelta(days=d_offset)
            d_str = _fmt_date(d_dt)
            enc_id = f"{pid}_DIST_ENC_{d_i:03d}"

            choice = d_i % 3
            if choice == 0 and distractor_meds:
                d_med = distractor_meds[(p_idx + d_i) % len(distractor_meds)]
                end_dt = d_dt + timedelta(days=rng.randint(5, 14))
                end_str = _fmt_date(min(end_dt, as_of_dt - timedelta(days=2)))
                d_tag = f"dist_med_{d_i}"
                distractor_tags.append(d_tag)
                collector.add(
                    tag=d_tag,
                    event_type="medication_order",
                    concept=d_med["concept"],
                    concept_class=d_med["concept_class"],
                    text=f"Short-course prescription order for {d_med['concept']} {d_med['dose']} oral from {d_str} to {end_str}.",
                    observed_time=d_str,
                    valid_start=d_str,
                    valid_end=end_str,
                    encounter_id=enc_id,
                    attributes={"dose": d_med["dose"], "action": "short_course"},
                )
            elif choice == 1:
                hr = rng.randint(64, 92)
                sbp = rng.randint(114, 146)
                dbp = rng.randint(68, 90)
                temp = round(97.8 + rng.random() * 1.2, 1)
                d_tag = f"dist_vital_{d_i}"
                distractor_tags.append(d_tag)
                collector.add(
                    tag=d_tag,
                    event_type="vital_sign",
                    concept="vital_signs",
                    concept_class="Vital Signs",
                    text=f"Routine clinic vital signs: Temp: {temp} F, HR: {hr} bpm, SBP: {sbp} mmHg, DBP: {dbp} mmHg, SpO2: 98 %.",
                    observed_time=d_str,
                    valid_start=d_str,
                    valid_end=d_str,
                    encounter_id=enc_id,
                    attributes={"temperature": temp, "heartrate": hr, "sbp": sbp, "dbp": dbp, "o2sat": 98},
                )
            else:
                d_tag = f"dist_visit_{d_i}"
                distractor_tags.append(d_tag)
                collector.add(
                    tag=d_tag,
                    event_type="visit",
                    concept="outpatient_visit",
                    concept_class="Encounter",
                    text=f"Routine outpatient follow-up visit on {d_str}; disposition HOME.",
                    observed_time=d_str,
                    valid_start=d_str,
                    valid_end=d_str,
                    encounter_id=enc_id,
                    attributes={"disposition": "HOME"},
                )

        p_events, tag_map = collector.finalize()
        all_events.extend(p_events)

        unrecorded_item = unrecorded_pool[p_idx % len(unrecorded_pool)]
        patient_blueprints.append(
            {
                "patient_id": pid,
                "split": split,
                "as_of_date": as_of_str,
                "window_start": _fmt_date(base_date),
                "tag_map": tag_map,
                "distractor_event_ids": [tag_map[t] for t in distractor_tags[:10]],
                "scenarios": {
                    "allergy": {
                        "concept": allergy_concept,
                        "reaction": allergy_reaction,
                        "gold_ids": [tag_map["allergy_1"]],
                    },
                    "condition_htn": {
                        "concept": "essential (primary) hypertension",
                        "onset_date": _fmt_date(cond_dt_1),
                        "first_id": tag_map["cond_htn_first"],
                        "followup_ids": [tag_map[t] for t in htn_followup_tags],
                    },
                    "condition_dm": {
                        "concept": "type 2 diabetes mellitus",
                        "diag_id": tag_map["cond_dm_first"],
                    },
                    "med_lifecycle": {
                        "old_med": old_med,
                        "old_dose_1": old_dose_1,
                        "old_dose_2": old_dose_2,
                        "new_med": new_med,
                        "new_dose": new_dose,
                        "dm_med": dm_med,
                        "dm_dose": dm_dose2,
                        "start_id": tag_map["med_lifecycle_start"],
                        "refill_ids": [tag_map[t] for t in refill_tags],
                        "dose_change_id": tag_map["med_lifecycle_dose_change"],
                        "new_active_id": tag_map["med_lifecycle_new_active"],
                        "dm_active_id": tag_map["med_chronic_active"],
                        "dm_refill_ids": [tag_map[t] for t in dm_old_refill_tags],
                    },
                    "lab_trend": {
                        "concept": lab_concept,
                        "unit": lab_unit,
                        "values": lab_vals,
                        "gold_ids": [tag_map[t] for t in lab_tags],
                    },
                    "missing_dates": {
                        "statin_concept": statin_concept,
                        "statin_month": ambig_month_str,
                        "statin_id": tag_map["med_ambig_month"],
                        "gerd_concept": "gastroesophageal reflux disease",
                        "gerd_id": tag_map["cond_missing_date"],
                    },
                    "conflict": {
                        "rule": conflict_type,
                        "concept": conf_med,
                        "dose_a": conf_dose_a,
                        "dose_b": conf_dose_b,
                        "gold_ids": [tag_map["conflict_a"], tag_map["conflict_b"]],
                    },
                    "insufficient_evidence": {
                        "concept": unrecorded_item["concept"],
                        "concept_class": unrecorded_item["concept_class"],
                    },
                },
            }
        )

    return all_events, patient_blueprints


def generate(
    config_path: str | Path = "configs/synthetic_generator.json",
    seed: Optional[int] = None,
    track_r_events: Optional[list[Event]] = None,
) -> tuple[list[Event], list[Any], list[Any]]:
    """Top-level interface required by Section 12.2: `generate(config, seed) -> (events, questions, gold)`."""
    from src.synthetic.question_generator import generate_benchmark_questions

    if isinstance(config_path, dict):
        config = config_path
    else:
        with Path(config_path).open("r", encoding="utf-8") as f:
            config = json.load(f)

    s_events, blueprints = generate_synthetic_patients(config, seed=seed)
    questions, gold = generate_benchmark_questions(
        blueprints=blueprints,
        config=config,
        track_r_events=track_r_events,
        seed=seed,
    )
    return s_events, questions, gold
