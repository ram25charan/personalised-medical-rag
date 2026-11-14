# Phase 2 Synthetic Patient Generator & Benchmark Specification

## 1. Synthetic Longitudinal Patient Dataset (Track S)

- **Generator version:** `1.0` (`configs/synthetic_generator.json`, seed `42`)
- **Total synthetic patients:** 100 (`S_001` - `S_100`)
- **Total synthetic events (`data/synthetic/events.jsonl`):** 12184 (mean 121.84 events/patient)
- **Patient split:** 34 dev patients (`S_001` - `S_034`), 66 test patients (`S_035` - `S_100`). Disjoint check: `True`

### Embedded Clinical Scenarios per Patient
1. **Persistent Condition & Allergy:** Documented drug/material allergy (`allergy`) and chronic hypertension (`condition`) with initial onset and follow-up visits.
2. **Medication Lifecycle:** Initial antihypertensive prescription (`medication_order`), subsequent dose escalation, discontinuation, and replacement with a new active agent, plus an active glycemic medication with open end date (`U1`).
3. **Longitudinal Lab Trend:** 4 sequential lab measurements (`lab_result` for HbA1c, LDL, serum creatinine, or TSH) across ~2.5 years, plus 1 post-`as_of_date` future measurement to verify the `observed_time <= as_of_date` availability filter.
4. **Missing & Ambiguous Dates:** Statin prescription with month-only precision (`U2`) and historical GERD condition with missing onset date (`U3`).
5. **Conflicting Records:** Alternating `C1` status conflict (outpatient active vs external pharmacy ended on overlapping interval) and `C2` dose conflict (primary care vs specialist conflicting active dose).
6. **Irrelevant History / Distractors:** 45-165 routine outpatient visits, vital signs, and short-course unrelated medications per patient.

## 2. Benchmark Questions & Programmatic Gold (`data/benchmark/`)

- **Total questions (`questions.jsonl` & `gold.jsonl`):** 180
  - **Track S Dev:** 50
  - **Track S Test:** 100
  - **Track R Test:** 30

### Category Breakdown
| Category | Track S Dev | Track S Test | Track R Test |
|---|---:|---:|---:|
| `temporal_ordering` | 8 | 15 | 6 |
| `current_vs_historical` | 7 | 15 | 6 |
| `longitudinal_change` | 7 | 14 | 6 |
| `irrelevant_history` | 7 | 14 | 5 |
| `missing_or_ambiguous_dates` | 7 | 14 | 3 |
| `conflicting_information` | 7 | 14 | 0 |
| `insufficient_evidence` | 7 | 14 | 4 |

### Question Intent Breakdown (All Tracks)
| Intent | Count |
|---|---:|
| `change_over_time` | 27 |
| `current_state` | 84 |
| `first_occurrence` | 39 |
| `historical` | 14 |
| `persistent_fact` | 16 |
