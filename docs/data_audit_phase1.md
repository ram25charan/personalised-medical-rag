# Phase 1 Data Quality & Real-Data Coverage Audit (Track R)

## 1. Raw Table Validation (MIMIC-IV-ED Demo v2.2)

| Table | Expected Rows | Actual Rows | Exact Duplicates | Orphan Stays |
|---|---:|---:|---:|---:|
| `edstays` | 222 | 222 | 0 | 0 |
| `diagnosis` | 545 | 545 | 0 | 0 |
| `medrecon` | 2764 | 2764 | 0 | 0 |
| `pyxis` | 1082 | 1082 | 0 | 0 |
| `triage` | 222 | 222 | 0 | 0 |
| `vitalsign` | 1038 | 1038 | 0 | 0 |

## 2. Longitudinal Structure Summary

- **Unique patients with ED stays:** 64
- **Total ED visits (`edstays`):** 222
- **Patients with >= 2 visits:** 35
- **Patients with >= 3 visits:** 24
- **Maximum visits for a single patient:** 23
- **Median time span across multi-visit patients:** 828.16 days
- **Visits with medication reconciliation (`medrecon`):** 181
- **Patients with >= 2 visits having `medrecon`:** 30
- **Multi-visit stays without `medrecon`:** 27
- **Patients with both ICD-9 and ICD-10 diagnosis codes:** 14

## 3. Normalised Track R Event Counts (`data/processed/events.jsonl`)

- **Total normalised events:** 5471

| Event Type | Count | Status Basis | Source Table |
|---|---:|---|---|
| `visit` | 222 | `interval` | `edstays` |
| `triage_observation` | 222 | `interval` | `triage` |
| `diagnosis_at_visit` | 545 | `interval` | `diagnosis` |
| `reported_home_medication` | 2362 | `snapshot` | `medrecon` (intra-stay duplicates collapsed) |
| `ed_administered_medication` | 1082 | `interval` | `pyxis` |
| `vital_sign` | 1038 | `interval` | `vitalsign` |

## 4. Scenario & Benchmark Category Coverage on Track R

- **Consecutive `medrecon` visit pairs:** 132
- **Visit pairs with added or dropped home medications:** 117
- **Patients with medication list changes across visits:** 29
- **Patients with recurring diagnoses across visits:** 18
- **Patients with >= 3 vital sign measurements in a visit:** 51

| Category | Supported on Track R? | Notes |
|---|---|---|
| `1_temporal_ordering` | `True` | Sequence of ED visits, diagnoses across visits, and within-visit vital/Pyxis timestamps. |
| `2_current_vs_historical` | `True` | Snapshot-basis: medications LISTED on visit N vs NOT_LISTED_LATEST on visit N after appearing on an earlier visit. |
| `3_longitudinal_change` | `True` | 117 consecutive visit pairs have medications added or dropped; within-visit vital sign series. |
| `4_irrelevant_history` | `True` | Multi-visit patients with up to 23 visits and dense vital/medication records around a specific query concept. |
| `5_missing_or_ambiguous_dates` | `limited` | 25 stays of multi-visit patients lack medrecon lists; home medications have no prescription start/end dates (U1/U4 uncertainty). |
| `6_conflicting_information` | `False` | No explicit active-vs-stopped or dose-conflict records exist in MIMIC-IV-ED demo (medrecon vs pyxis is an informational provenance note per Section 7.5). Fully tested in Track S. |
| `7_insufficient_evidence` | `True` | Constructed by querying unrecorded medications, labs, or conditions not present in the patient's ED history. |
