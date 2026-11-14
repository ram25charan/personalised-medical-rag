# PROJECT MASTER SPECIFICATION

**Chronology-Aware Context Fusion for Personalized Medical Retrieval-Augmented Generation**

---

## 0. Document Control

### 0.1 Document Version

| Field | Value |
|---|---|
| Version | 1.0 (initial master specification) |
| Status | Authoritative. Implementation may begin at Phase 1 |
| Project type | Undergraduate college capstone: a working demonstration plus a research paper |

### 0.2 Last Updated

2026-11-14

### 0.3 Document Authority and Precedence

3. Any change to a decision in Section 14 must be recorded in the Decision Change History (14.6) before it is implemented.
4. Anything marked **STRETCH** is optional. It is attempted only after every non-stretch item in its phase is complete.
5. This is an undergraduate-scale project. When two valid options exist, choose the simpler one and record the choice.

---

## 1. Project Overview

### 1.1 Project Title

Chronology-Aware Context Fusion for Personalized Medical Retrieval-Augmented Generation

### 1.2 Project Summary

The project investigates whether a retrieval-augmented generation (RAG) system answers questions about a patient's recorded history better when it considers three things together:

1. **Question relevance (Rq):** how well a patient record entry matches the question.
2. **History relevance (Rh):** how closely the entry relates to the other clinically relevant events in the patient's history.
3. **Temporal relevance (Rt):** whether the entry is still valid, ended, superseded or persistent, relative to the question's reference date.

**What is retrieved.** The retrieval corpus is the **patient's own structured events** (medications, diagnoses, observations, labs). This is patient-record RAG for temporal questions. A small **demo clinical corpus** adds general background text in the demo only (Section 6.4).

**Contribution.** A chronology-aware selection method, a small benchmark with programmatically derived gold labels, and an honest analysis of when temporal fusion helps, fails or adds nothing. The demonstration application is a thin layer over the same pipeline.

### 1.3 Project Vision

A reproducible, modest research prototype. A user can select a demonstration patient, ask a question about the recorded history, and see which events were selected, why, what is current or historical, where records conflict or are uncertain, and the grounded answer with citations.

This is **not** a clinical decision-support product and makes no clinical-effectiveness or safety claims.

### 1.4 Intended Users

| User | Role |
|---|---|
| Capstone evaluators and paper reviewers | Assess the method, the experiments and the demonstration |
| Project team | Develop and run the experiments |
| Simulated patient / simulated clinician (demo only) | Two presentation modes of one interface (Section 9). No real users, no real authentication |

---

## 2. Research Foundation

### 2.1 Background and Motivation

Conventional RAG ranks passages by semantic similarity to the question. Patient histories are different: medications start, change and stop, diagnoses persist or resolve, results are superseded, dates are missing and records can disagree. For a question such as "what medication am I currently taking?", the semantically best-matching record may be an old, discontinued prescription. Recency alone is also wrong, because an allergy recorded years ago stays relevant.

Related work exists. Examples are medical-RAG benchmarks (MIRAGE), LLM temporal reasoning over longitudinal clinical records (Kruse et al., TIMER) and event- and time-aware retrieval for long-horizon structured EHR prediction (EHR-RAG, Cao et al., 2026). This project does **not** claim to be the first to apply time-aware retrieval to patient events. Its angle is as-of-date status questions, question-intent-dependent temporal logic, conflict and uncertainty handling, and a controlled ablation design.

### 2.2 Problem Statement

Given a patient's time-stamped events, a question and a reference date (`as_of_date`), select the events that are relevant to the question, relevant to the patient's history and correctly interpreted in time, so that a language model's answer is grounded, temporally consistent and honest about conflicts and uncertainty.

### 2.3 Research Question

> Does incorporating patient-history relevance and temporal relevance into event selection and context assembly improve retrieval quality and answer consistency, compared with question-only retrieval and history-aware retrieval without temporal weighting, under a fixed context budget?

### 2.4 Research Objectives

1. Represent patient histories as structured, time-aware events with provenance.
2. Build conventional and simple temporal baselines.
3. Implement history relevance (Rh), intent-dependent temporal relevance (Rt) and a configurable fusion score.
4. Implement rule-based conflict detection and uncertainty flags.
5. Build a benchmark with programmatic gold labels across real and synthetic tracks.
6. Compare systems with retrieval metrics (no LLM) and a smaller set of LLM-generation experiments.
7. Separate the effect of **event selection** from the effect of **status annotations**.
8. Deliver a working single-page demonstration and a research paper.

### 2.5 Expected Research Contributions

Planned contributions, not promises of positive results:

1. A structured, provenance-aware patient event representation with a question-time (`as_of_date`) status computation.
2. An intent-dependent temporal relevance function and a three-term fusion score with per-intent weights tuned on development data.
3. A small benchmark (real, synthetic and optionally perturbed-real tracks) with programmatically derived gold labels and a documented generator.
4. A controlled comparison of question-only, history-aware and chronology-aware selection, including simple strong baselines (recency-only, status-filter-only).
5. An analysis of when temporal fusion helps, fails or adds no measurable benefit, including a context-budget curve and a synthetic-to-real transfer check.

### 2.6 Research Hypothesis

Formal hypotheses (all untested until results exist):

| ID | Hypothesis | Comparison | Primary measures |
|---|---|---|---|
| H1 | History-aware selection improves retrieval over question-only selection | B3 vs B1 (history alone); B2 vs B1 (combined) | Recall@k, nDCG@k, MRR, stale-evidence rate |
| H2 | Temporal relevance adds value beyond history-aware selection | B2 vs B3 | Same, plus temporal answer correctness |
| H3 | Conflict and uncertainty flags improve handling of conflict, uncertainty and insufficient-evidence cases | B2 vs B4 | Conflict-flag accuracy, abstention accuracy, contradiction rate |

Exploratory analyses (no formal hypothesis):

- **E1:** accuracy as a function of the context budget k.
- **E2:** synthetic-to-real transfer (weights tuned on synthetic dev data, tested on Tracks S, R and optionally P).
- **E3:** attribution of gains to event selection versus status annotation (2x2 design, Section 8.3).
- **E4:** per-track, per-intent and per-category breakdowns.

Negative and inconclusive results are reported in full.

---

## 3. Project Scope and Boundaries

### 3.1 In-Scope Features

- Ingestion and normalisation of the MIMIC-IV-ED demo into typed events.
- A deterministic, seeded synthetic patient generator and template-based question generator.
- Patient-scoped event retrieval: BM25 + dense (all-MiniLM-L6-v2) with reciprocal-rank fusion.
- Rh, Rt, fusion and budgeted selection, with per-intent weights tuned on dev data.
- Rule-based conflict detection and uncertainty flags.
- Grounded generation through the Groq API with structured JSON output and cited event IDs.
- Retrieval-level and generation-level experiments, with bootstrap confidence intervals.
- A small demo clinical corpus (about 20-30 passages), used only in the demo.
- A single-page Streamlit demonstration with a patient/clinician view-mode switch.
- A research paper in IEEE format.

### 3.2 Out-of-Scope Features

- Real authentication, authorisation or patient-data isolation.
- Separate patient and doctor applications; a database server; SQLite.
- Fine-tuning, GPU training, LangChain or LlamaIndex.
- Evaluating the demo clinical corpus; full guideline retrieval (future work).
- Clinical validation, diagnosis, treatment or dosing recommendations.
- LLM-as-judge scoring (replaced by deterministic checks plus a small manual review).
- The credentialed MIMIC-IV clinical database (not available; the clinical demo could not be obtained).

**STRETCH items:** perturbed-real Track P; a second generator model; full-history (B0) baseline on short histories; a learned ranker instead of grid search; a doctor-specific summary mode.

### 3.3 Assumptions and Constraints

| Item | Constraint |
|---|---|
| Data | MIMIC-IV-ED demo only: no labs, no prescriptions with start/stop, no free text. Labs and medication timing exist only in synthetic data |
| Dates | MIMIC dates are shifted per patient (years run 2112-2201). Only intervals within a patient are meaningful |
| LLM | Groq free tier. Published sample limits for text models (read Sept 2026): 30 requests/min, 1,000 requests/day, 8,000 tokens/min, 200,000 tokens/day, per organisation and per model. The console limits page is authoritative. Tokens, not requests, are the binding limit |
| Prompt size | A single prompt may not exceed the per-minute token limit, so long full-history prompts are not feasible |
| Compute | CPU-only local machine is sufficient |
| Team size | Undergraduate capstone. No deadline has been fixed (Section 14.5) |

### 3.4 Non-Negotiable Research Principles

1. **No fabricated results.** Report only measured numbers, including negative ones.
2. **No fabricated data presented as real.** Every event carries provenance (real, synthetic, perturbed).
3. **Recency is not relevance.** Temporal logic must depend on question intent and event type.
4. **Fair comparison.** Same questions, same patient data, same prompts and generation settings across systems, except the factor under test.
5. **No tuning on test data.** Development and test sets are split by patient. Weights, rules and prompts are frozen before test evaluation.
6. **Report separately by track.** Never publish a single pooled score.
7. **Uncertainty is surfaced, not silently resolved.** Missing dates, conflicts and insufficient evidence must be reported.
8. **Keep it simple.** Implement the minimum complete pipeline before any optional feature.

---

## 4. System Requirements

### 4.1 Patient-Facing Requirements

Implemented as the **Patient view mode** of the single demo page (Section 9). The user can:

- Select a demonstration patient and set the question's reference date (`as_of_date`).
- View a chronological timeline of that patient's events.
- Ask a question about the recorded history.
- See the answer in plain language, the supporting events with citations, any conflict or uncertainty warning, and a clear statement when the records cannot answer the question.

### 4.2 Doctor-Facing Requirements

Implemented as the **Clinician view mode** of the same page. It uses the same pipeline and shows, in addition:

- Event-level detail: type, source table, provenance, validity interval, date precision and computed status.
- Per-event component scores (Rq, Rh, Rt, fused score) for inspection.
- Detected conflicts and uncertainty flags in structured form.

A doctor-specific structured-summary generator is **STRETCH**. The view mode changes presentation only, never retrieval.

### 4.3 Shared System Requirements

1. One backend pipeline serves both view modes and all experiments.
2. Every pipeline stage is a separate, independently testable module.
3. The pipeline can be run in a "system configuration" given by a config file (B1, B2, etc.).
4. Every answer contains cited event IDs. Patient-specific claims must come only from patient events.
5. The demo clinical corpus is switched off in all experiments.

### 4.4 Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Ingest the six MIMIC-IV-ED demo tables and produce a normalised event table with provenance |
| FR2 | Generate synthetic patients and benchmark questions deterministically from a seed and a config file |
| FR3 | Compute, for every event and question, the temporal state at `as_of_date` (ACTIVE, ENDED, SUPERSEDED, PERSISTENT, UNKNOWN, or snapshot-basis states) |
| FR4 | Retrieve candidates within one patient with BM25 and dense retrieval, combined by reciprocal-rank fusion, with inspectable scores |
| FR5 | Compute Rh, Rt and the fused score S = αRq + βRh + γRt with configurable weights |
| FR6 | Select the top-k events under a context budget and order them chronologically for the prompt |
| FR7 | Detect conflicts and uncertainty with the rules in Section 7.5 and 7.6 |
| FR8 | Generate a structured JSON answer through the Groq API with response caching and resumable runs |
| FR9 | Compute retrieval and answer metrics, with paired bootstrap confidence intervals |
| FR10 | Log every experiment run with configuration, model ID, prompts, seeds and outputs |
| FR11 | Provide a Streamlit demo with timeline, selection display, flags, answer and background box |

### 4.5 Non-Functional Requirements

- **Reproducibility:** a fixed seed and logged configuration reproduce data, retrieval and metrics. LLM outputs are cached.
- **Modularity:** modules communicate only through the contracts in Section 12.
- **Performance:** retrieval over one patient's events must be interactive (seconds) on CPU.
- **Safety:** every user-facing answer shows a "research prototype, not medical advice" notice.
- **Rate-limit tolerance:** the experiment runner handles HTTP 429 with backoff and can resume after interruption.

---

## 5. System Architecture

### 5.1 High-Level Architecture

```text
 MIMIC-IV-ED demo (Track R)         Synthetic generator (Track S)    [Perturbation (Track P, STRETCH)]
        |                                    |                                  |
        +------------ normalise ------------>+<---------------------------------+
                                  |
                      UNIFIED PATIENT EVENT TABLE (+ provenance)
                                  |
              question generator + as_of_date + programmatic gold
                                  |
              patient-scoped candidate events
                    |
        +-----------+-------------+
        |           |             |
   Rq: BM25+dense   Rh: relatedness  Rt: intent x event type x as_of
        |           |             |
        +-----------+-------------+
                    |
        FUSION S = a*Rq + b*Rh + g*Rt  ->  top-k under budget
                    |
        conflict detection + uncertainty flags
                    |
        CONTEXT ASSEMBLY (events [E#], optional corpus [C#] in demo only)
                    |
                 GROQ LLM  ->  structured JSON answer + citations
                    |
          EVALUATION (retrieval metrics; answer checks)   /   DEMO UI
```

### 5.2 End-to-End System Workflow

1. **Build time:** ingest data, generate synthetic data, build the event table, generate questions and gold labels, split dev/test by patient.
2. **Index time:** per patient, build a BM25 index and dense embeddings over event text.
3. **Query time:** take (patient, question, `as_of_date`); determine intent; get candidates; score Rq, Rh, Rt; fuse; select top-k; flag conflicts and uncertainty; assemble context; call the LLM; return a structured answer.
4. **Evaluation time:** run retrieval-only comparisons (no LLM) for all systems; run LLM generation for the defined subset of systems on the test set; compute metrics and intervals.

### 5.3 Core Component Responsibilities

| Component | Responsibility |
|---|---|
| Ingestion | Load raw files unchanged; assess quality; build normalised events |
| Synthetic generator | Produce patients, events and scenario-embedded evidence plus distractors |
| Question generator | Produce template questions, intents, `as_of_date` and gold labels |
| Timeline / state module | Compute temporal state per event at `as_of_date` |
| Retrieval | BM25, dense, RRF; patient-scoped candidates |
| History relevance | Rh from the anchor concepts |
| Temporal scoring | Rt from intent, event type and state |
| Fusion | Combine normalised scores; select top-k |
| Conflict / uncertainty | Rule-based flags |
| Generation | Prompt construction, Groq client, caching, JSON parsing |
| Evaluation | Metrics, bootstrap, tables and figures |
| App | Streamlit page over the same pipeline |

### 5.4 Patient and Doctor Interface Architecture

One Streamlit page. A view-mode switch changes wording and level of detail only. Both modes call the identical `run_pipeline()` function. There are no separate retrieval systems and no authentication (patient selection simulates access to a record).

### 5.5 Component Interaction and Data Flow

Modules exchange only the data contracts in Section 12 (Event, Question, Gold, PipelineOutput, AnswerJSON, ResultRecord, CorpusChunk). Retrieval never reads gold labels. The gold-label generator and the Rt scoring code are separate modules; Rt must not import gold or generator internals (see 8.10).

---

## 6. Dataset Strategy and Data Specification

### 6.1 Overall Dataset Strategy

Three tracks, always labelled and reported separately.

| Track | Source | Gold labels | Role |
|---|---|---|---|
| **R** (real) | MIMIC-IV-ED demo v2.2 | Snapshot-derived (weaker) | Small real-data check; sim-to-real test |
| **S** (synthetic) | Rule-based generator | Exact, from construction | Main dev/test sets; scale |
| **P** (perturbed real) | Real histories plus injected events | Injected truth | **STRETCH** |

Synthetic data supplies what the ED demo lacks: labs, explicit medication start/stop/dose, conflicts, missing dates and long histories. It must never be presented as real-world validation.

### 6.2 Original Dataset: MIMIC-IV-ED Demo

**License:** Open Data Commons Open Database License v1.0 (ODbL). Cite Johnson et al. (2023), MIMIC-IV-ED v2.2, PhysioNet, https://doi.org/10.13026/5ntk-km72.

**Contents (verified on the uploaded archive):**

| Table | Rows | Notes |
|---|---:|---|
| `edstays` | 222 | ED visits: intime, outtime, disposition |
| `diagnosis` | 545 | Per-visit ICD-9/10 codes; about 2.5 per stay |
| `medrecon` | 2,764 | Medication reconciliation list per visit |
| `pyxis` | 1,082 | ED medication dispensing records |
| `triage` | 222 | Triage vitals and chief complaint |
| `vitalsign` | 1,038 | Time-stamped vitals within a visit |

**Longitudinal structure:** 64 patients (the demo page mentions 100 subject IDs, but only 64 have ED stays). 35 patients have 2 or more stays; 24 have 3 or more; the maximum is 23. Among multi-visit patients the median span is about 827 days. 181 of 222 stays have a medication list. 30 patients have medication lists at 2 or more visits (132 consecutive visit pairs).

**Known limitations:**

1. `medrecon` is a medication list per visit, not a start/stop record. Absence from a later list is **not** proof of discontinuation.
2. Medication names are noisy (brand/generic; 344 duplicate normalised names within stays). Normalise using `gsn` / `etccode`; version the rules.
3. ED diagnoses are per-visit codes, not onset dates. Persistence is inferred only from repeated codes across visits. 14 patients have both ICD-9 and ICD-10 codes (coding noise, not clinical conflict).
4. 25 stays of multi-visit patients have no medication list.
5. No labs, no free-text notes, no outpatient history.
6. Dates are shifted per patient.

### 6.3 Synthetic Longitudinal Patient Data

**Generator requirements:** deterministic (seeded), config-driven, rule-based. **No LLM writes patient histories.** An LLM may be used only to paraphrase question wording, and the gold labels never depend on it.

**Size (targets):** about 100 synthetic patients, each with roughly 50-300 events. Core evidence events are embedded among unrelated distractors (routine vitals, unrelated medications and diagnoses) so that selection under a budget matters. Dev/test split by patient (roughly one third dev, two thirds test).

**Distributions sampled from real data:** medication names and classes (from `medrecon`), chronic-condition combinations (from `diagnosis`) and inter-visit spacing (from `edstays`).

**Scenario templates (each maps to benchmark categories in 8.5):**

| Scenario | Content |
|---|---|
| Medication lifecycle | Start, dose change, stop, later new medication |
| Lab trend | Repeated values of one test over time (e.g. HbA1c) |
| Conflicting records | Two sources disagree on status or dose |
| Persistent condition | Old diagnosis with no resolution; allergy |
| Missing / ambiguous dates | Events with unknown or month-only dates |
| Irrelevant history | Long unrelated history around a narrow question |
| Insufficient evidence | Question about something never recorded |

**Track P (STRETCH):** 20-30 real patients (from the 30 with two or more medication lists) with 1-2 injected events each (a discontinuation, a conflicting record, a missing-date event), provenance `perturbed`.

### 6.4 Clinical Evidence Corpus

A **demo-only** corpus of about 20-30 short passages (100-200 words each).

| Passage type | Example topics |
|---|---|
| Condition overviews | Type 2 diabetes, hypertension, high cholesterol |
| Test explainers | HbA1c, kidney-function tests |
| Drug-class notes | Statins, loop diuretics, antiplatelets, thyroid hormone |

**Rules:**

- General, non-personal information only. No dosing, no treatment advice.
- **Sources:** MedlinePlus health-topic summaries and medical-test information (public domain per MedlinePlus; acknowledge with "Source: MedlinePlus, National Library of Medicine"). **Do not copy MedlinePlus drug monographs** (copyright ASHP). Write drug-class notes in your own words from the FDA label on DailyMed, and verify DailyMed's reuse terms. No paywalled references. No Wikipedia (share-alike).
- Format: one JSONL record per passage (Section 12.1).
- **Not evaluated and not used in experiments.** Use in the demo only (Section 7.7).

### 6.5 Data Preprocessing and Standardization

1. **Ingestion:** load the six raw files unchanged; keep raw files read-only.
2. **Quality assessment:** missing values, duplicates, invalid timestamps, unusual values. Flag outliers; do not delete them.
3. **Relational integration:** join on `subject_id` and `stay_id`; validate counts.
4. **Normalisation:** medication names via `gsn`/`etccode`; keep the original string; version the rules.
5. **Event construction (Track R):**

| Source | Event type | Time | Notes |
|---|---|---|---|
| `medrecon` | `reported_home_medication` | `charttime` | Snapshot basis. Collapse duplicate rows of one drug in a stay and keep all source rows in provenance |
| `pyxis` | `ed_administered_medication` | `charttime` | Administration, not a prescription |
| `diagnosis` | `diagnosis_at_visit` | visit `intime` | Per-visit code, not onset; keep `seq_num`, ICD code and version |
| `vitalsign`, `triage` | `vital_sign`, `triage_observation` | `charttime` / visit `intime` | Within-visit observations |
| `edstays` | `visit` | `intime`, `outtime` | Encounter anchor with disposition |

6. **Timelines:** order events by time within each patient using relative offsets.
7. **Real-track `as_of_date`:** questions on Track R are anchored to a visit, with `as_of_date` = that visit's `outtime`.
8. **Outputs:** a data-quality report, the normalised event table, per-patient timelines and a coverage report listing which scenarios the real data supports.

### 6.6 Unified Patient Event Representation

Every event, real or synthetic, has these fields (full contract in 12.1):

`event_id`, `patient_id`, `track`, `event_type`, `concept` (normalised), `concept_class`, `text`, `observed_time`, `valid_start`, `valid_end`, `date_precision` (day / month / year / unknown), `status_basis` (`interval` or `snapshot`), `source_type` (real / synthetic / perturbed), `source_dataset`, `source_table`, `source_rows`, `encounter_id`, `confidence`, `time_is_shifted`, `attributes`.

**Status is not stored.** It is computed at query time from the validity interval (or snapshot) and `as_of_date`. Date, validity interval and status are distinct concepts and are never conflated. Missing values stay missing; nothing is invented.

**Event types:** `reported_home_medication`, `ed_administered_medication`, `medication_order` (synthetic), `diagnosis_at_visit`, `condition` (synthetic, with onset/resolution), `allergy`, `lab_result`, `symptom`, `vital_sign`, `triage_observation`, `visit`.

### 6.7 Data Provenance and Source Tracking

- Event IDs have the form `<track>_<patient>_<seq>` (e.g. `R10000032_0042`, `S_007_0150`).
- Real events keep `source_table` and `source_rows`. Synthetic events record `source_dataset = "custom_generator"` with generator version and seed.
- Synthetic records are labelled in every output, including the demo.
- Retrieval is patient-scoped: one patient's events can never be retrieved for another patient.

### 6.8 Benchmark Dataset Strategy

**Questions:** template-based, one template family per (intent x event type). Five question intents: `current_state`, `historical`, `change_over_time`, `first_occurrence`, `persistent_fact`. Each question has an `as_of_date`.

**Target sizes (approximate):**

| Track | Dev | Test |
|---|---:|---:|
| S (synthetic) | about 50 | about 100 |
| R (real) | none (small check only) | about 30 |
| P (STRETCH) | none | about 20-30 |

**Gold labels (programmatic, from construction):**

- `gold_events`: the **minimal** set of events needed to answer.
- `distractor_events`: stale, superseded, ended or irrelevant events that must not be selected (used for the stale-evidence rate).
- `expected_answer`: structured (see 12.1).
- `expected_behavior`: `answer`, `abstain`, `flag_conflict` or `flag_uncertainty`.

**Real-track semantics:** gold means "recorded on the list at visit N", not clinical truth. Supported question types: medications listed at visit N; what was added or dropped between visits; whether drug X was listed at visit N; diagnoses recorded at visit N; recurring diagnoses across visits; a small within-visit vitals category. A manually verified sample (about 20 questions) checks the real-track gold.

**Splits:** by patient, not by question. Test data is never used to tune weights, rules or prompts.

### 6.9 Dataset Limitations and Usage Constraints

- Track R is small (64 patients), hospital/ED-centric and snapshot-based. Findings are exploratory.
- Synthetic data is authored by the project team and cannot establish real-world performance.
- **ODbL:** attribution is required. A publicly released database derived from MIMIC data may have to remain under ODbL (share-alike). If the benchmark is released, release the MIMIC-derived part under ODbL with attribution or release only the synthetic part plus scripts. Check the license terms before release.
- **Groq and real rows:** Track R rows are de-identified and openly licensed, but confirm that sending them to the Groq API is acceptable before doing so. If not confirmed, Track R is evaluated at the retrieval level only (no LLM generation).
- Never send restricted data (e.g. credentialed MIMIC data) to any external API.

---

## 7. Proposed Research Methodology

**Global rules for all systems**

- **Candidate pool:** all events of the selected patient observed on or before `as_of_date`. Events after `as_of_date` are excluded for every system (records that do not yet exist cannot be retrieved). The pool is at most a few hundred events, so everything is scored directly.
- **Budget:** top-k events are passed to the generator. Default k = 10; the budget study uses k in {5, 10, 20}.
- **Prompt order:** selected events are always ordered chronologically in the prompt, for every system.

### 7.1 Conventional RAG Baseline

Question-only hybrid retrieval: BM25 (`rank-bm25`) and dense retrieval (`sentence-transformers/all-MiniLM-L6-v2`, cosine similarity) over the patient's event text, combined by reciprocal-rank fusion. Rq is the fused score, min-max normalised within the pool. The top-k by Rq are selected. This is **B1**. It uses no history relevance and no temporal information (beyond the `as_of_date` availability rule).

### 7.2 Patient-History-Aware Retrieval

**Rh** measures how related an event is to the **anchor concepts** of the question. Anchor concepts = (a) concepts found in the question via a concept lexicon, plus (b) the concepts of the top 5 events by Rq.

Relatedness of event e to an anchor a (take the maximum over anchors):

| Relation | Rh |
|---|---:|
| Same concept | 1.0 |
| Same concept class (e.g. same drug class), or a link in the relation table | 0.7 |
| Same encounter as an anchor event | 0.5 |
| None | 0.0 |

- **Relation table:** a small, hand-built, versioned table (e.g. diabetes - insulin / metformin / HbA1c; hypertension - antihypertensive and diuretic classes; hyperlipidaemia - statins; kidney problems - creatinine). It is written **before** looking at test questions and frozen. It is shared with the demo corpus mapping (Section 6.4).
- Unrelated events get Rh = 0, so history relevance never boosts information merely because it belongs to the same patient.
- STRETCH ablation: Rh from embedding similarity instead of the relation table.

### 7.3 Temporal Relevance Modeling

**Temporal state** of an event at `as_of_date`:

| State | Rule (interval basis) |
|---|---|
| `ACTIVE` | `valid_start` <= as_of and `valid_end` is open or unknown |
| `ENDED` | `valid_end` known and <= as_of |
| `SUPERSEDED` | A later event of the same patient and concept series (newer lab value, dose change, newer record of the same item) on or before as_of |
| `PERSISTENT` | Persistent type (allergy, chronic condition, recurring diagnosis) with no resolution on or before as_of |
| `UNKNOWN` | Start date missing or date precision unknown |

Snapshot basis (Track R `medrecon`): `LISTED` (on the latest list at or before as_of) and `NOT_LISTED_LATEST` (on an earlier list, absent from the latest list). `NOT_LISTED_LATEST` is **not** "discontinued".

**Intent** is determined by keyword rules (for example "current", "now", "when first", "changed", "ever", "allergic"). Every question also stores its true intent, so an **oracle-intent** run reports how much intent errors cost.

**Rt by intent** (initial values; frozen before tuning; any change must be logged):

| Intent | Rt = 1.0 | 0.7 | 0.5 | 0.3 | 0.2 | 0.1 | 0 |
|---|---|---|---|---|---|---|---|
| `current_state` | ACTIVE, LISTED, PERSISTENT | | UNKNOWN | | SUPERSEDED | ENDED, NOT_LISTED_LATEST | (future events are already excluded) |
| `historical` (window up to as_of) | Interval overlaps the window | | UNKNOWN | | | Otherwise | |
| `change_over_time` | Earliest and latest events of the anchor series in the window | Intermediate events of the series | | | | Other events | |
| `first_occurrence` | Earliest event of the anchor concept | | | Later events of that concept | | Other events | |
| `persistent_fact` | PERSISTENT | | ACTIVE | | Others | | |

Recency alone is never used as relevance.

### 7.4 Chronology-Aware Context Fusion

```
S(e) = α · Rq(e) + β · Rh(e) + γ · Rt(e)      with α, β, γ >= 0 and α + β + γ = 1
```

- Rq is min-max normalised within the pool. Rh and Rt already lie in [0, 1].
- **Weights are per intent**, chosen by grid search (step 0.1, 66 combinations) on the **development set** using retrieval metrics only (mean nDCG@10, ties broken by Recall@10). No LLM tokens are used for tuning. The weights are then frozen.
- **B3** uses γ = 0 (α + β = 1, tuned separately). **B1** uses α = 1.
- Report a weight-sensitivity analysis.
- Document the rationale of the chosen weights; arbitrary weights are not allowed.

### 7.5 Conflict Detection

A separate, testable module. It examines the selected events and all other events of the patient that share a concept with a selected event.

| Rule | Condition |
|---|---|
| C1 status conflict | Same concept, overlapping validity, one source indicates active and another ended / discontinued |
| C2 dose conflict | Same medication, overlapping validity, different doses |
| C3 date conflict | Same concept and type reported with dates differing by more than 30 days across sources |

Output: a list of flags with the conflicting event IDs. The generator must present both records and must not arbitrarily choose one. On Track R, a mismatch between reconciliation and administration records is reported as an informational provenance note, not as a conflict.

### 7.6 Uncertainty-Aware Evidence Handling

| Flag | Condition |
|---|---|
| U1 | Medication with no recorded end date (status inferred) |
| U2 | Date precision coarser than a day |
| U3 | Date missing |
| U4 | Snapshot or reconciliation source (a listing is not proof of an active prescription) |
| U5 | Insufficient evidence: no event reaches the Rq threshold τ for the anchor concept. τ is chosen on dev data |

### 7.7 Evidence-Grounded Generation

- **LLM:** Groq API, one pinned model ID recorded with its run date. Temperature 0. The same prompt template for every system.
- **Annotation levels:**
  - **A0:** events only (text and time).
  - **A1:** A0 plus computed status labels.
  - **A2:** A1 plus conflict and uncertainty flags.
- **Prompt rules:** patient-specific facts only from the provided events `[E#]`; cite event IDs for every patient-specific claim; say when the records are insufficient; mention conflicts; never invent medications, doses, dates or diagnoses; no medical advice.
- **Structured output (JSON):** `answer_text`, `status_claims` (list of concept and status), `cited_event_ids`, `abstain`, `conflict_flagged`, `uncertainty_noted`.
- **Demo only:** the prompt gains a separate `[C#]` background section from the demo corpus (top 1-2 passages retrieved from the concepts of the selected events plus the question). Background passages are general information. They never support patient-specific claims.
- **Reliability:** cache every response keyed by (model, prompt hash); retry with exponential backoff on HTTP 429; resume after interruption; parse and validate the JSON, and log failures.

### 7.8 Complete Proposed Method

**B2 = hybrid Rq + Rh + Rt fusion, per-intent frozen weights, top-k selection, chronological ordering, conflict and uncertainty flags (A2), structured grounded generation.**

---

## 8. Experimental Design and Evaluation

### 8.1 Experimental Objectives

1. Test H1-H3 (Section 2.6) with the controlled comparisons below.
2. Quantify how much of any gain comes from event selection versus status annotation (E3).
3. Show behaviour across context budgets (E1) and across tracks, including synthetic-to-real transfer (E2).
4. Report where the method fails.

### 8.2 Baseline Configurations

**Retrieval-level systems** (no LLM; run on **all** test questions in every track):

| ID | System |
|---|---|
| RS1 | BM25 only |
| RS2 | Dense only |
| RS3 = **B1** | Hybrid question-only (RRF) |
| RS4 | Recency-only: take the top 50 by Rq, then sort by recency and select top-k |
| RS5 | Status-filter-only: drop events whose Rt for the intent is <= 0.2, then rank by Rq |
| **B3** | Rq + Rh (γ = 0) |
| **B2** | Rq + Rh + Rt |
| ORACLE | Gold events (upper bound) |

RS4 and RS5 answer the reviewer question "is this just a recency sort or a status filter?".

### 8.3 Proposed Method Configurations

**Generation-level configurations** (LLM; test set):

| ID | Selection | Annotation | Purpose |
|---|---|---|---|
| **B1** | Hybrid question-only | A0 | Conventional baseline |
| **B1+** | Hybrid question-only | A2 | Effect of annotation alone |
| **B2-** | B2 selection | A0 | Effect of selection alone |
| **B2** | B2 selection | A2 | Proposed method |
| **B4** | B2 selection | A1 (no conflict / uncertainty flags) | Isolates conflict and uncertainty handling |
| **ORACLE-G** | Gold events | A2 | Reference upper bound |

The 2x2 {B1, B1+, B2-, B2} separates selection from annotation (E3). **STRETCH:** B3 with the LLM; B0 (history-only / full-history generation) on short histories only, because of the Groq token limits.

**Token budget (estimate, not measured):** about 1.4K tokens per call; 6 configurations x roughly 130-160 test questions gives about 1.0-1.3 million tokens, i.e. several days of free-tier quota on one model. Run configurations in priority order: B1, B2, then B1+, B2-, B4, ORACLE-G. If quota is a problem, run B4 and ORACLE-G on a subset, or spread runs across models and days.

### 8.4 Ablation Study Design

| Comparison | Isolates |
|---|---|
| B3 vs B1 | History relevance (H1) |
| B2 vs B3 | Temporal relevance (H2) |
| B2 vs B4 | Conflict and uncertainty handling (H3) |
| 2x2 selection x annotation | Selection vs annotation (E3) |
| Rule intent vs oracle intent | Cost of intent errors |
| k in {5, 10, 20} | Budget curve (E1) |
| Rh by relation table vs embeddings | STRETCH |

B2 vs B1 changes history and time together, so it must be read together with B3 vs B1 and B2 vs B3.

### 8.5 Benchmark Categories

Seven categories, each with template questions:

1. Temporal ordering
2. Current vs historical status
3. Longitudinal change
4. Irrelevant patient history
5. Missing or ambiguous dates
6. Conflicting information
7. Insufficient evidence

Roughly equal numbers per category on Track S. Track R covers the categories the real data supports (1-5 in limited form; 7 can be constructed; 6 only if found). Question templates and the generator are documented so the benchmark can be regenerated.

### 8.6 Evaluation Metrics

**Retrieval (no LLM):** Recall@5, Recall@10, nDCG@10, MRR, Hit Rate@5, and the **stale-evidence rate** (share of selected events that are labelled distractors in the gold, i.e. stale, superseded or irrelevant).

**Answer (deterministic, from structured JSON vs structured gold):**

| Metric | Definition |
|---|---|
| Answer correctness | Exact match for single-valued answers; set F1 for list answers; ordered match for change-over-time |
| Temporal consistency | `status_claims` agree with the computed gold state |
| Contradiction rate | Share of answers with a status claim contradicting the gold |
| Citation precision / recall | Cited event IDs versus `gold_events` |
| Abstention accuracy | Abstain exactly when `expected_behavior = abstain` |
| Conflict-flag accuracy | Flag raised exactly when a conflict exists |
| Faithfulness (deterministic) | Cited IDs exist, and the claimed concept appears in the cited events |

**Manual review:** about 20 responses reviewed by hand for faithfulness and sanity. No LLM judge.

**Statistics:** paired bootstrap (at least 1,000 resamples) for 95% confidence intervals on paired differences; per-track, per-intent and per-category breakdowns; absolute differences reported with uncertainty.

### 8.7 Evaluation Protocol

1. Freeze ingestion and the generator; generate data and the benchmark; split dev/test by patient.
2. Run all retrieval-level systems on dev; tune per-intent weights by grid search on dev retrieval metrics only; fix τ.
3. Iterate prompts and intent rules on at most about 20 dev questions, then freeze everything (configuration hash recorded).
4. Run retrieval-level systems on test (all tracks).
5. Run generation-level configurations on test in priority order (Track R generation only if the Groq / license check in 6.9 passes).
6. Compute metrics and intervals; manual review; failure analysis.
7. No changes after step 4. Any deviation is documented in the paper.

### 8.8 Reproducibility Requirements

- Fixed seeds; versioned generator, relation table, normalisation rules, Rt table, prompts and config files.
- Every result record stores: git commit, config hash, track, model ID and run date, temperature, k, weights, intent source, prompt hash and raw output.
- Cached LLM responses are stored with the results; `requirements.txt` pins package versions; data files are hashed.
- Hosted models can change or be retired; the model ID and date are always reported.

### 8.9 Result Analysis and Reporting

- Tables with 95% confidence intervals, one per track; never a single pooled score.
- Figures: budget curve (E1), weight sensitivity, per-intent and per-category comparisons.
- Failure analysis: a sample of errors per category with the cause (retrieval miss, wrong state, intent error, generation error).
- Negative and inconclusive results are reported. The paper's wording stays within what the measurements show.

### 8.10 Threats to Validity

| Threat | Mitigation / statement |
|---|---|
| Circularity: the team writes both the generator and the Rt rules, so B2 may win by construction | Gold labels come from the generator's construction, in a module separate from scoring; Rt never imports gold or generator internals; the relation table is frozen before test questions are written; Track R provides a check not authored by the team |
| Small samples (64 real patients, about 100 synthetic test questions) | Bootstrap intervals; findings framed as exploratory |
| Weak real-track labels (snapshots, not start/stop) | Real-track gold stated as "recorded on list"; manual check of a sample |
| Intent classification errors | Oracle-intent comparison |
| Hand-built relation table biases Rh | Documented, frozen and versioned |
| Date shifting | Only within-patient intervals used |
| ED data is not outpatient longitudinal care | Stated limitation |
| LLM / hosted-model drift and free-tier limits | Pinned model ID and date; cached outputs; priority run order |
| Medication-name normalisation affects gold | Rules versioned; sample verified |
| Synthetic data may differ stylistically from real data | Same text templates for rendering both; separate reporting |

---

## 9. Application Design

### 9.1 Patient Interface

The Patient view mode of the single Streamlit page: patient selector, `as_of_date` input, timeline, question box, plain-language answer, supporting events with IDs, and warnings. Provenance badges (Real / Synthetic) are always visible.

### 9.2 Doctor Interface

The Clinician view mode of the same page. It adds the per-event detail: type, source table, validity interval, date precision, computed status and component scores (Rq, Rh, Rt, fused), plus the structured conflict and uncertainty flags. **STRETCH:** an automatic structured history summary.

### 9.3 Shared RAG Backend

Both modes call the same `run_pipeline(patient_id, question, as_of_date, config)`. A system selector (B1 or B2) lets the user compare the conventional and proposed pipelines on the same question. There are no two retrieval systems.

### 9.4 Patient Timeline Visualization

A chronological timeline of the patient's events, coloured by event type, with the events selected for the current question highlighted and ended or superseded events visually distinguished. Use Streamlit-native charting or Plotly, whichever is installed.

### 9.5 Evidence and Source Presentation

- Selected events listed with `[E#]` IDs, provenance badge and source table.
- A separate **General information** box for demo-corpus passages `[C#]` with their sources and the MedlinePlus acknowledgement where applicable.
- Citations in the answer link to the listed events.
- A permanent notice: "Research prototype. Not medical advice. Synthetic and de-identified demo data."

### 9.6 Conflict and Uncertainty Presentation

- Conflicts: a visible warning naming the conflicting events, showing both records side by side.
- Uncertainty flags (U1-U4): short labels beside the affected events.
- Insufficient evidence (U5): the answer area states that the records cannot answer the question instead of guessing.

---

## 10. Technology Stack and Engineering Standards

### 10.1 Selected Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11, 3.12, or 3.13 |
| Frontend | Streamlit (single page) |
| LLM | **Groq API** (hosted open models). Exact model ID chosen in Phase 4 and recorded (Section 14.5) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| Lexical retrieval | `rank-bm25` |
| Dense retrieval | Sentence Transformers embeddings + cosine similarity (numpy) |
| Data processing | pandas, numpy |
| Evaluation | Python, numpy / scikit-learn where helpful, custom metrics, paired bootstrap |
| Storage | JSON / JSONL / Parquet files. **No database** |
| Testing | pytest |
| Version control | Git and GitHub |

**Not used:** LangChain, LlamaIndex, SQLite, any GPU training.

### 10.2 Development Environment

VS Code; a virtual environment; CPU-only machine; the repository runs end-to-end from a fresh clone using the README instructions.

### 10.3 Project Repository Structure

```text
personalized-medical-rag/
├── app/
│   └── main.py                    # single Streamlit page (view-mode switch)
├── data/
│   ├── raw/mimic_iv_ed_demo/      # original files, read-only
│   ├── processed/                 # events.jsonl, timelines, quality reports
│   ├── synthetic/                 # generated patients/events
│   ├── benchmark/                 # questions.jsonl, gold.jsonl
│   └── corpus/                    # demo_corpus.jsonl (demo only)
├── src/
│   ├── data/                      # ingestion.py, normalisation.py, events.py
│   ├── synthetic/                 # generator.py, question_generator.py, scenarios/
│   ├── retrieval/                 # bm25.py, dense.py, rrf.py, corpus_retrieval.py
│   ├── temporal/                  # state.py, temporal_scoring.py, history_relevance.py
│   ├── fusion/                    # fusion.py, selection.py
│   ├── flags/                     # conflict_detection.py, uncertainty.py
│   ├── generation/                # groq_client.py, prompts.py, cache.py
│   ├── pipeline.py                # run_pipeline()
│   └── evaluation/                # retrieval_metrics.py, answer_metrics.py, bootstrap.py, runner.py
├── experiments/                   # run_retrieval.py, tune_weights.py, run_generation.py, compare.py
├── configs/                       # systems.json, weights.json, relation_table.json, rt_rules.json
├── results/                       # outputs/, metrics/, figures/, cache/
├── tests/
├── docs/                          # this specification, data audit, paper drafts
├── .env                           # GROQ_API_KEY (never committed)
├── .gitignore
├── requirements.txt
└── README.md
```

### 10.4 Coding and Modularity Standards

- Small, typed functions; one responsibility per module; no hidden global state.
- Modules interact only through the contracts in Section 12.
- Retrieval, Rh, Rt, fusion, flags, generation and evaluation are separately importable and testable.
- The gold-label generator and the Rt / scoring modules must not import each other.
- Docstrings state purpose and units; comments explain decisions, not syntax.

### 10.5 Configuration and Environment Management

- All tunables (k, weights, thresholds, model ID, temperature, seeds, paths) live in `configs/` files, never hard-coded.
- The Groq API key is read from the `GROQ_API_KEY` environment variable (`.env` locally). Never commit it.
- Groq limits are read from the console limits page. The runner enforces a configurable requests-per-minute and tokens-per-minute pace and a daily token counter, and it resumes after interruption.

### 10.6 Testing Standards

Unit tests are required for: state computation for each state and edge case (missing end date, missing start date, superseded series, snapshot basis); each Rt intent row; Rh relations; each conflict rule C1-C3; each uncertainty flag; the `observed_time <= as_of_date` availability rule; patient scoping (no cross-patient retrieval); fusion normalisation and weight handling; generator determinism for a fixed seed; metric functions on hand-computed examples; JSON answer parsing and failure handling. Phase completion requires the phase's tests to pass.

### 10.7 Logging and Experiment Tracking

Structured logs per run (JSONL). Each result record follows the `ResultRecord` contract (12.1). Raw LLM outputs are kept in the cache directory. A run manifest lists the config hash, git commit, package versions and data hashes.

### 10.8 Version Control

Git with small, descriptive commits. Branch per phase (merge when the phase is complete). Tag the commit used for the final reported results. `data/raw/` and `.env` are never modified or committed (large or restricted files go in `.gitignore`).

---

## 11. Implementation Roadmap

No calendar dates are fixed (Section 14.5). Phases are ordered by dependency.

### 11.1 Phase Overview

| Phase | Name |
|---|---|
| 1 | Foundation and real-data ingestion |
| 2 | Synthetic data and benchmark |
| 3 | Retrieval, scoring, fusion and retrieval-level evaluation (first usable result) |
| 4 | Conflict / uncertainty handling, generation and generation-level evaluation |
| 5 | Demo corpus and application |
| 6 | Paper and final documentation |

### 11.2 Phase Dependencies

```text
1 -> 2 -> 3 -> 4 -> 6
          3 -> 5 -> (uses 4 for generation)
5 and 4 can overlap once Phase 3 is complete; the paper skeleton can start after Phase 3.
```

### 11.3 Phase Objectives

1. **Phase 1:** environment, repository and schemas; ingest the ED demo; quality report; normalised event table and timelines; real-data coverage report.
2. **Phase 2:** deterministic synthetic generator; question generator with `as_of_date`, intents and programmatic gold; concept lexicon and relation table v1; dev/test split by patient. STRETCH: Track P.
3. **Phase 3:** state module, BM25 + dense + RRF, Rh, Rt, fusion, per-intent weight tuning on dev, retrieval metrics and bootstrap; run all retrieval-level systems. This phase alone already yields results for the paper.
4. **Phase 4:** conflict and uncertainty modules, prompts, Groq client with cache and resumable runner; run generation-level configurations in priority order; answer metrics; manual review.
5. **Phase 5:** build the demo corpus; corpus retrieval; the Streamlit page (timeline, selection display, flags, answer, background box, view-mode and system toggles).
6. **Phase 6:** tables and figures, paper in IEEE format, README and reproducibility package.

### 11.4 Phase Inputs and Expected Outputs

| Phase | Inputs | Outputs |
|---|---|---|
| 1 | ED demo zip, this specification | `events.jsonl` (Track R), quality report, timelines, coverage report, schemas |
| 2 | Event schema, real-data distributions | Synthetic events, `questions.jsonl`, `gold.jsonl`, relation table, lexicon, generator docs |
| 3 | Events, questions, gold | `pipeline.py` retrieval stages, tuned `weights.json`, retrieval-level metrics with intervals |
| 4 | Phase 3 outputs, Groq key | Flags modules, prompts, cached generations, answer metrics, manual-review sheet |
| 5 | Phases 3-4, corpus sources | `demo_corpus.jsonl`, `app/main.py` |
| 6 | All results | Paper, figures, README |

### 11.5 Phase Completion Criteria

| Phase | Done when |
|---|---|
| 1 | All six tables load; counts match Section 6.2; event table validates against the schema; coverage report written; tests pass |
| 2 | Generator reproduces identical output for a fixed seed; questions and gold validate; dev/test patient sets are disjoint; relation table frozen and versioned |
| 3 | All retrieval-level systems run on dev and test; weights frozen; metrics and intervals produced; tests pass |
| 4 | B1 and B2 (minimum) generated and scored on the test set; remaining configurations run as quota allows; manual review done |
| 5 | Demo runs end-to-end on at least one real and one synthetic patient; corpus passages carry source and license notes |
| 6 | Paper reflects only measured results; README reproduces the pipeline |

### 11.6 Cross-Phase Integration Requirements

- A change to a schema in Section 12 requires a version bump and updating every dependent module and test in the same change.
- The demo (Phase 5) imports `run_pipeline()` and adds no retrieval logic of its own.
- Results must be reproducible from the committed configs and cached outputs.

### 11.7 Phase Handoff Protocol

At the end of each phase: update Section 15 (Project Status), record any new or changed decisions in Section 14, tag the commit, and list open issues. The next phase begins from that tagged state.

---

## 12. Interface and Data Contracts

### 12.1 Shared Data Schemas

**Event** (one JSON object per line):

```json
{
  "event_id": "S_007_0150",
  "patient_id": "S_007",
  "track": "S",
  "event_type": "medication_order",
  "concept": "metformin",
  "concept_class": "Biguanide antidiabetic",
  "text": "Metformin prescribed, 500 mg",
  "observed_time": "2025-01-10",
  "valid_start": "2025-01-10",
  "valid_end": null,
  "date_precision": "day",
  "time_is_shifted": false,
  "status_basis": "interval",
  "source_type": "synthetic",
  "source_dataset": "custom_generator",
  "source_table": null,
  "source_rows": [],
  "encounter_id": null,
  "confidence": 1.0,
  "attributes": {"dose": "500 mg", "generator_version": "0.1", "seed": 42}
}
```

For Track R, `time_is_shifted` is `true`, `source_type` is `real`, and `source_table` / `source_rows` reference the original file and rows. (`time_is_shifted` is part of the contract in addition to the fields listed in 6.6.)

**Question:**

```json
{
  "question_id": "Q_S_0031",
  "patient_id": "S_007",
  "track": "S",
  "split": "test",
  "text": "What medications is the patient currently taking?",
  "as_of_date": "2025-09-01",
  "window_start": null
}
```

**Gold** (stored separately; never read by retrieval or generation):

```json
{
  "question_id": "Q_S_0031",
  "intent": "current_state",
  "category": "current_vs_historical",
  "anchor_concepts": ["metformin"],
  "gold_events": ["S_007_0150", "S_007_0188"],
  "distractor_events": ["S_007_0091", "S_007_0120"],
  "expected_answer": {"type": "set", "value": ["metformin"]},
  "expected_behavior": "answer",
  "expected_conflict": false,
  "expected_flags": ["U1"]
}
```

`expected_answer.type` is one of `status`, `set`, `value`, `ordered_values` or `none`. `expected_behavior` is one of `answer`, `abstain`, `flag_conflict`, `flag_uncertainty`.

**CorpusChunk** (demo corpus):

```json
{
  "chunk_id": "C_012",
  "title": "HbA1c test",
  "text": "...",
  "chunk_type": "test_explainer",
  "source_name": "MedlinePlus, National Library of Medicine",
  "url": "https://medlineplus.gov/...",
  "accessed": "2026-11-20",
  "license_note": "Public domain; acknowledge MedlinePlus",
  "concepts": ["hba1c", "diabetes"]
}
```

**PipelineOutput:**

```json
{
  "patient_id": "S_007",
  "as_of_date": "2025-09-01",
  "system_id": "B2",
  "intent": "current_state",
  "intent_source": "rules",
  "candidates": [
    {"event_id": "S_007_0150", "state": "ACTIVE", "rq": 0.82, "rh": 1.0, "rt": 1.0, "score": 0.93, "selected": true}
  ],
  "selected_event_ids": ["S_007_0150"],
  "flags": {
    "conflicts": [{"rule": "C1", "event_ids": ["S_007_0150", "S_007_0188"]}],
    "uncertainty": [{"flag": "U1", "event_ids": ["S_007_0150"]}],
    "insufficient_evidence": false
  },
  "context_text": "...",
  "background_chunk_ids": []
}
```

**AnswerJSON** (the model's required output):

```json
{
  "answer_text": "...",
  "status_claims": [{"concept": "metformin", "status": "ACTIVE"}],
  "cited_event_ids": ["S_007_0150"],
  "abstain": false,
  "conflict_flagged": false,
  "uncertainty_noted": true
}
```

**ResultRecord:** `run_id`, `git_commit`, `config_hash`, `system_id`, `track`, `question_id`, `model_id`, `run_date`, `temperature`, `k`, `weights`, `intent_source`, `prompt_hash`, `raw_output`, `parsed_answer`, `metrics`.

### 12.2 Module Responsibilities and Interfaces

| Module | Interface (informal) |
|---|---|
| `data.ingestion` | `load_raw(path) -> dict[str, DataFrame]` |
| `data.events` | `build_events(raw) -> list[Event]` |
| `synthetic.generator` | `generate(config, seed) -> (events, questions, gold)` |
| `temporal.state` | `compute_state(event, as_of, patient_events) -> State` |
| `retrieval` | `rq_scores(patient_events, text, as_of) -> dict[event_id, float]` |
| `temporal.history_relevance` | `rh_scores(patient_events, anchors, relation_table) -> dict` |
| `temporal.temporal_scoring` | `rt_scores(patient_events, intent, as_of, rules) -> dict` |
| `fusion` | `select(rq, rh, rt, weights, k) -> list[event_id]` |
| `flags` | `detect_conflicts(...)`, `uncertainty_flags(...)` |
| `generation` | `generate(context, config) -> AnswerJSON` (cached) |
| `pipeline` | `run_pipeline(patient_id, question, as_of, system_config) -> PipelineOutput, AnswerJSON` |
| `evaluation` | `retrieval_metrics(...)`, `answer_metrics(...)`, `paired_bootstrap(...)` |

### 12.3 Input and Output Specifications

- All files are UTF-8 JSON or JSONL; dates are ISO-8601 strings (month- or year-only dates keep `date_precision`).
- IDs are stable and never reused. Intermediate files carry the schema version.
- Retrieval functions never receive `Gold`. Only the ORACLE systems and the evaluators read it.

### 12.4 Cross-Phase Compatibility Requirements

New optional fields may be added without breaking consumers. Renaming or removing a field, or changing its meaning, is a breaking change and follows 12.5.

### 12.5 Contract Change Management

1. Propose the change and its reason in Section 14.6.
2. Bump the schema version.
3. Update all producers, consumers, tests and sample files in the same commit.
4. Regenerate any affected data and rerun the affected tests. Results computed under the old schema are re-run, not patched.

---

## 13. Research and Engineering Constraints

### 13.1 Data Privacy and Security

- Use only open or synthetic data for this project. Do not use credentialed datasets.
- Treat Track R as de-identified but still sensitive: do not publish raw patient rows beyond what the ODbL permits and the paper requires.
- Confirm that sending Track R rows to Groq is acceptable (Section 6.9) before doing so; otherwise evaluate Track R at the retrieval level only.
- Keep API keys in environment variables only; never commit them.

### 13.2 Clinical Safety Boundaries

- The system must not diagnose, prescribe, recommend doses or replace professional judgement.
- A visible "research prototype, not medical advice" notice appears in the app and the paper.
- The demo corpus contains general information only.
- A successful experiment on synthetic or small real data does not establish clinical validity or safety.

### 13.3 Evidence and Citation Requirements

- Every patient-specific claim in an answer cites event IDs.
- Background claims cite corpus passage IDs and are visually separated.
- The paper cites related work, the datasets and the demo-corpus sources with their licenses.
- The status of an event as "current" or "historical" is always reported relative to the stated `as_of_date`.

### 13.4 Reproducibility and Traceability

- Everything in Section 8.8. Any number in the paper can be traced to a result record, a config hash and a commit.

### 13.5 Prohibited Shortcuts and Invalid Claims

- No fabricated results, no invented data presented as real, no tuning on test data, no changing the test set after seeing results.
- No pooled single score across tracks.
- No claim of novelty beyond what the literature review supports; no claim of clinical benefit.
- No LLM-generated patient histories or gold labels.
- No treating recency as relevance, no treating a medication-list entry as a prescription, no treating "not listed" as "discontinued".
- No leaking gold labels, intent or anchor concepts into non-oracle systems.
- No copying copyrighted drug monographs into the demo corpus.

---

## 14. Decision Register

### 14.1 Confirmed Architectural Decisions

| ID | Decision |
|---|---|
| AD1 | The ranked candidates are **patient events**, retrieved within one patient (patient-scoped). Guideline passages are never ranked with Rt |
| AD2 | Event status is **computed at query time** from the validity interval (or snapshot) and the question's `as_of_date`. It is not stored |
| AD3 | One shared pipeline (`run_pipeline`) serves the demo and all experiments. The demo is a **single Streamlit page** with a patient/clinician view-mode switch |
| AD4 | Modular implementation; no LangChain or LlamaIndex; file-based storage (JSON/JSONL/Parquet), no database |
| AD5 | Candidate pool = events observed on or before `as_of_date` for every system; default budget k = 10 |
| AD6 | Gold-label generation and scoring code are separate modules that do not import each other |

### 14.2 Confirmed Research Decisions

| ID | Decision |
|---|---|
| RD1 | Formal hypotheses H1-H3; exploratory analyses E1-E4 |
| RD2 | Fusion S = αRq + βRh + γRt, per-intent weights tuned by grid search on dev retrieval metrics only, then frozen |
| RD3 | Rh = relatedness to anchor concepts (same concept / class or relation-table link / same encounter), relation table frozen before test questions |
| RD4 | Rt is rule-based by intent x temporal state; intent from keyword rules, with an oracle-intent comparison |
| RD5 | Selection and annotation are separated with a 2x2 design (B1, B1+, B2-, B2), plus B4 and an oracle |
| RD6 | Strong simple baselines are included: recency-only, status-filter-only, BM25-only, dense-only, oracle |
| RD7 | Deterministic answer metrics; no LLM judge; manual review of about 20 responses |
| RD8 | Paired bootstrap intervals; results reported per track, never pooled |
| RD9 | Conflict detection and uncertainty flags are rule-based (C1-C3, U1-U5) |
| RD10 | Experiments use patient events only; the demo corpus is not evaluated |

### 14.3 Confirmed Dataset Decisions

| ID | Decision |
|---|---|
| DD1 | Track R = MIMIC-IV-ED demo v2.2 (64 patients, 222 stays), ODbL license, attribution required |
| DD2 | Track S = about 100 deterministic synthetic patients with roughly 50-300 events each; question targets about 50 dev / 100 test |
| DD3 | Track P (perturbed real) is a STRETCH goal |
| DD4 | Dev/test split by patient |
| DD5 | Real-track gold means "recorded on the list at visit N", not clinical truth; "not listed" is not "discontinued" |
| DD6 | No full MIMIC-IV and no MIMIC-IV clinical demo (the clinical demo could not be obtained); labs and medication timing are synthetic only |
| DD7 | Demo corpus: about 20-30 general passages; MedlinePlus health-topic and test summaries (public domain, acknowledged); no copied drug monographs; drug-class notes paraphrased from FDA labels (terms to verify) |
| DD8 | Track R rows go to Groq only after the license / terms check in 6.9; otherwise Track R is retrieval-level only |

### 14.4 Confirmed Technology Decisions

| ID | Decision |
|---|---|
| TD1 | **Groq** replaces Gemini for generation |
| TD2 | `all-MiniLM-L6-v2` + `rank-bm25` + reciprocal-rank fusion |
| TD3 | Temperature 0, structured JSON output, response cache, resumable runner |
| TD4 | Python 3.11/3.12/3.13, Streamlit, pandas, numpy, pytest, Git/GitHub |
| TD5 | Plan around Groq free-tier token limits (tokens/day and tokens/min are the binding limits); run generation configurations in priority order |

### 14.5 Deferred Decisions

| ID | Open item | Decide |
|---|---|---|
| DF1 | Exact Groq model ID and its limits | Phase 4, after checking the console limits page |
| DF2 | Project deadline and calendar schedule (none fixed) | When known; update Section 11 |
| DF3 | Whether to attempt STRETCH items (Track P, second generator, B0 on short histories, learned ranker, clinician summary) | After Phase 3 |
| DF4 | Final paper venue and template details | Phase 6 (IEEE format assumed) |
| DF5 | Whether Track R may be sent to Groq (license / terms check) | Before Phase 4 |
| DF6 | Final sizes (patients, questions) | After a feasibility check in Phase 2 |
| DF7 | Novelty wording: read EHR-RAG (Cao et al., 2026) fully and verify the abstract of reference 4 | Before writing the paper |

### 14.6 Decision Change History

| Date | Version | Change |
|---|---|---|
| 2026-11-14 | 1.0 | Initial master specification. Consolidated changes against the original specification and slide deck: Gemini replaced by Groq; candidates are patient events, not evidence chunks; `as_of_date` and status computed at query time added; three data tracks added (R, S, P-stretch); baselines revised to B1-B4 plus retrieval-level baselines and oracles, B5 dropped; 2x2 selection x annotation added; two Streamlit interfaces replaced by one page with a view-mode switch; SQLite and LLM-as-judge removed; personalization coverage and safety-issue rate dropped; guideline corpus reduced to a demo-only corpus; hypothesis H4 demoted to exploratory analysis |
| 2026-11-14 | 1.0.1 | Updated TD4 to include Python 3.13 (local development environment runs Python 3.13.5). |

---

## 15. Project Status and Next Milestone

### 15.1 Completed Work

- Proposal slide deck (17 slides) and the original project specification.
- Review of the original specification and design negotiation; this master specification.
- Audit of the MIMIC-IV-ED demo archive (table sizes, longitudinal structure, limitations as recorded in 6.2).
- License check for demo-corpus sources (MedlinePlus public-domain vs ASHP-copyrighted content).
- Check of Groq free-tier sample limits.
- Review of the `temporal-rag` repository as design inspiration (MIT license).
- **Phase 1 (Foundation and Real-Data Ingestion):**
  - Repository structure, Git initialization (`main` and `phase-1`), and shared Pydantic schemas (`src/schemas.py`).
  - Read-only ingestion and quality audit of the 6 MIMIC-IV-ED demo v2.2 tables (`src/data/ingestion.py`, `data/processed/quality_report.json`).
  - Versioned (`v1.0`) medication and diagnosis normalisation (`src/data/normalisation.py`).
  - Construction of 5,471 normalised Track R events (`data/processed/events.jsonl`), per-patient relative timelines (`data/processed/timelines.json`), and real-data scenario coverage report (`data/processed/coverage_report.json`, `docs/data_audit_phase1.md`).
  - Phase 1 `pytest` unit test suite (`tests/test_phase1.py`, 8/8 passing).

### 15.2 Current Implementation State

Phase 1 is **complete** and verified (`v0.1-phase1`). Ready to start Phase 2 (Synthetic data and benchmark).

### 15.3 Outstanding Work

Phases 2-6 (Section 11), then STRETCH items if time permits.

### 15.4 Known Issues and Blockers

- The MIMIC-IV clinical demo (labs, prescriptions) could not be downloaded (PhysioNet login failed); the project proceeds with the ED demo plus synthetic data. A `wget` download may still work, but is optional.
- No deadline fixed (DF2).
- Groq model ID not chosen (DF1); free-tier limits may force multi-day generation runs.
- License / terms confirmation for sending Track R rows to Groq and for releasing derived data (DF5, 6.9).
- Reference 4 (Najafabadipour 2020): abstract not yet reviewed.
- EHR-RAG (Cao et al., 2026) is close related work and must be read fully before novelty is discussed.
- The `ClinQuery` and `EHR-QA-Skill` repositories were not reviewed.
- Track R is small and its labels are weak by construction.

### 15.5 Next Recommended Action

Start **Phase 2**: implement the deterministic, seeded synthetic patient generator and template-based question generator with programmatic gold labels; create the versioned concept lexicon and relation table v1; generate Track S and Track R benchmark questions and disjoint patient splits.

---

## 16. Instructions for AI Implementation Assistants

### 16.1 Role and Responsibilities

Act as a senior AI engineer and research collaborator for an undergraduate capstone. Implement this specification step by step, produce complete executable code where practical, explain the purpose of each substantial component before introducing it, and keep the scope small.

### 16.2 Master Specification Authority

This document is authoritative (0.3). If a user request, an older document or your own judgement conflicts with it, say so and propose a change through Section 14.6 instead of silently deviating. Do not change the research methodology silently. Do not introduce new research claims without explaining their implications.

### 16.3 Phase-Specific Execution Rules

- Work on one phase at a time (Section 11). Do not start a phase until the previous phase's completion criteria (11.5) are met, except where 11.2 allows overlap.
- Do not jump to the frontend before the retrieval pipeline and its evaluation work.
- Do not attempt STRETCH items until all non-stretch items of the current phase are done.
- Prefer the simplest implementation that satisfies the contract.

### 16.4 Existing Code and Data Inspection Rules

- Before writing code, inspect the working directory, existing files, configs, data and tests.
- Reuse what exists. Do not assume a component exists because this document mentions it.
- Preserve working code unless there is a clear reason to change it. State the reason when you do.

### 16.5 Decision-Making Rules

- When a design decision is unclear: describe the ambiguity, list the technically sound options, recommend one with justification, and ask for clarification only if the decision materially affects architecture or research validity.
- Record every new decision in Section 14.
- Never use test data to make a decision (weights, rules, prompts, thresholds).

### 16.6 Testing and Validation Requirements

- Write the unit tests listed in 10.6 for the code you add.
- Validate outputs at each stage (schema validation, counts, spot checks).
- When something fails: diagnose the root cause, explain it, fix the cause rather than working around it, and verify the fix with a test.
- Never fabricate experimental results or sample outputs presented as real.

### 16.7 Documentation and Handoff Requirements

- After each phase: update Section 15, record decisions in Section 14, list open issues and tag the commit (11.7).
- Keep configs, prompts, seeds and model IDs versioned (8.8).
- Keep a short implementation log in `docs/`.

---

## 17. References and Supporting Documents

### 17.1 Dataset Documentation

- Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Mark, R., & Horng, S. (2023). *MIMIC-IV-ED* (version 2.2). PhysioNet. https://doi.org/10.13026/5ntk-km72 (demo subset used; license: Open Data Commons Open Database License v1.0).
- Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*. https://doi.org/10.1038/s44360-026-00096-z (standard PhysioNet citation).
- MIMIC-IV Clinical Database Demo (not used; could not be obtained): https://physionet.org/content/mimic-iv-demo/

### 17.2 Research References

1. Xiong, G., Jin, Q., Lu, Z., & Zhang, A. (2024). Benchmarking Retrieval-Augmented Generation for Medicine. *Findings of ACL 2024*, 6233-6251. https://doi.org/10.18653/v1/2024.findings-acl.372
2. Kruse, M., Hu, S., Derby, N., Wu, Y., Stonbraker, S., Yao, B., Wang, D., Goldberg, E. M., & Gao, Y. (2025). Large Language Models with Temporal Reasoning for Longitudinal Clinical Summarization and Prediction. *Findings of EMNLP 2025*, 20715-20735. https://doi.org/10.18653/v1/2025.findings-emnlp.1128
3. Cui, H., Unell, A., Chen, B., Fries, J. A., Alsentzer, E., Koyejo, S., & Shah, N. H. (2025). TIMER: temporal instruction modeling and evaluation for longitudinal clinical records. *npj Digital Medicine*, 8, 577. https://doi.org/10.1038/s41746-025-01965-9
4. Najafabadipour, M., Zanin, M., Rodriguez-Gonzalez, A., Torrente, M., Nunez Garcia, B., Cruz-Bermudez, J. L., Provencio, M., & Menasalvas, E. (2020). Reconstructing the patient's natural history from electronic health records. *Artificial Intelligence in Medicine*, 105, 101860. https://doi.org/10.1016/j.artmed.2020.101860 **[abstract not yet reviewed; verify relevance]**
5. Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAs: Automated Evaluation of Retrieval Augmented Generation. *EACL 2024 System Demonstrations*, 150-158. https://aclanthology.org/2024.eacl-demo.16/ (related evaluation work; its LLM-judge approach is not used here)
6. Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024). ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. *NAACL-HLT 2024*, 338-354. https://doi.org/10.18653/v1/2024.naacl-long.20 (related evaluation work; not used)
7. Cao, L., Chen, Q., & Guo, Y. (2026). EHR-RAG: Bridging Long-Horizon Structured Electronic Health Records and Large Language Models via Enhanced Retrieval-Augmented Generation. arXiv:2601.21340. **Closest related work found so far; read fully before any novelty claim.**

### 17.3 Technical Documentation

- Groq rate limits: the limits page in the Groq console is authoritative (sample limits were read in Sept 2026).
- `sentence-transformers/all-MiniLM-L6-v2` model card; `rank-bm25`; Streamlit documentation.
- MedlinePlus copyright and reuse terms: https://medlineplus.gov/copyright.html and https://medlineplus.gov/faq/copyrightfaq.html
- DailyMed (FDA labels) for drug-class background; verify reuse terms.
- `Emmimal/temporal-rag` (MIT): https://github.com/Emmimal/temporal-rag. Design inspiration for validity filtering and time decay only; not a dependency.
