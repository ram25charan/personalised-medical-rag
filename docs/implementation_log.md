# Implementation Log

## Phase 1: Foundation and Real-Data Ingestion (2026-11-14)

- **Repository & Environment**:
  - Initialized Git repository with `main` and `phase-1` branches; added remote `https://github.com/ram25charan/personalised-medical-rag.git`.
  - Created directory structure matching Section 10.3 (`src/`, `data/`, `configs/`, `experiments/`, `results/`, `tests/`, `docs/`).
  - Recorded `TD4` update in `docs/master_specification.md` (Section 14.6) to include Python 3.13.5.
- **Data Contracts (`src/schemas.py`)**:
  - Implemented strict Pydantic v2 models (`extra="forbid"`) for `Event`, `Question`, `Gold`, `CorpusChunk`, `PipelineOutput`, `AnswerJSON`, and `ResultRecord` (Section 12.1), plus `save_jsonl` and `load_jsonl` helpers.
- **Ingestion & Quality Audit (`src/data/ingestion.py`)**:
  - Implemented read-only loader `load_raw()` and `assess_data_quality()` over the six MIMIC-IV-ED demo v2.2 tables (`edstays`: 222, `diagnosis`: 545, `medrecon`: 2,764, `pyxis`: 1,082, `triage`: 222, `vitalsign`: 1,038; 64 unique patients).
  - Verified relational integrity (0 orphan stays/subjects), SHA-256 hashes, and physiological outlier flags (preserved without deletion).
- **Normalisation & Event Builder (`src/data/normalisation.py`, `src/data/events.py`)**:
  - Implemented versioned (`v1.0`) medication normalisation via GSN and cleaned generic names, collapsing intra-stay `medrecon` duplicates (2,764 raw rows -> 2,362 unique stay-concept events) while preserving all original `source_rows` and raw names in provenance.
  - Built 5,471 normalised Track R events (`data/processed/events.jsonl`), per-patient relative timelines (`data/processed/timelines.json`), quality report (`data/processed/quality_report.json`), and real-data scenario coverage report (`data/processed/coverage_report.json` and `docs/data_audit_phase1.md`).
- **Verification (`tests/test_phase1.py`)**:
  - Added 8 unit tests covering raw table counts, longitudinal statistics, brand/generic medication normalisation, event provenance, duplicate collapsing, timeline chronological ordering, and contract schema validation.

## Phase 2: Synthetic Data and Benchmark (2026-11-14)

- **Relation Table & Concept Lexicon (`configs/relation_table.json`)**:
  - Created and froze `v1.0` clinical concept relation table and synonym lexicon (41 core concepts across diabetes, hypertension, hyperlipidemia, CKD, thyroid, arrhythmia, anticoagulation, GERD, asthma, and drug/material allergies) prior to generating test questions (Section 7.2 & 8.10).
- **Deterministic Synthetic Patient Generator (`src/synthetic/generator.py`, `configs/synthetic_generator.json`)**:
  - Generated 100 synthetic patients (`S_001`–`S_100`, 34 dev / 66 test) with 12,184 normalised events (`data/synthetic/events.jsonl`, mean 121.84 events/patient, range 60–180).
  - Embedded all required longitudinal clinical scenarios (persistent condition & allergy, medication lifecycle with dose change and discontinuation/replacement, 4-point lab trends + post-`as_of_date` future check, month-only `U2` and missing `U3` dates, `C1` status and `C2` dose conflicts, and realistic distractor visits/vitals/short-course medications).
- **Benchmark Question & Programmatic Gold Generator (`src/synthetic/question_generator.py`, `src/synthetic/run_phase2.py`)**:
  - Generated 180 benchmark questions (`data/benchmark/questions.jsonl`) and programmatic gold labels (`data/benchmark/gold.jsonl`): 50 Track S dev questions, 100 Track S test questions, and 30 Track R test questions anchored to visit `outtime`.
  - Documented the generator and benchmark statistics in `data/benchmark/benchmark_summary.json` and `docs/synthetic_and_benchmark_phase2.md`.
- **Verification (`tests/test_phase2.py`)**:
  - Added 5 unit tests verifying fixed-seed determinism, seed sensitivity, disjoint dev/test patient splits, gold/distractor event existence and `observed_time <= as_of_date` integrity, and relation table structure.

## Phase 3: Retrieval, Scoring, Fusion, and Retrieval-Level Evaluation (2026-11-15)

- **Temporal State & Scoring (`src/temporal/state.py`, `src/temporal/history_relevance.py`, `src/temporal/temporal_scoring.py`, `configs/rt_rules.json`)**:
  - Implemented query-time temporal state computation (`ACTIVE`, `ENDED`, `SUPERSEDED`, `PERSISTENT`, `UNKNOWN`, and snapshot states `LISTED`, `NOT_LISTED_LATEST`) and `observed_time <= as_of_date` availability filtering.
  - Implemented anchor extraction and history relevance ($R_h \in \{1.0, 0.7, 0.5, 0.0\}$) using the frozen relation table.
  - Implemented keyword-based intent classification and intent-dependent temporal relevance ($R_t$) across all 5 intents.
- **Hybrid Retrieval & Fusion (`src/retrieval/`, `src/fusion/`, `src/flags/`, `src/pipeline.py`, `configs/systems.json`)**:
  - Implemented BM25 (`rank-bm25`), Dense (`sentence-transformers/all-MiniLM-L6-v2` on CPU with in-memory embedding cache and offline-first loading), and Reciprocal-Rank Fusion (`RRF`) with min-max pool normalisation.
  - Implemented fused score selection ($S = \alpha R_q + \beta R_h + \gamma R_t$), recency-only (`RS4`), status-filter-only (`RS5`), rule-based conflict detection (`C1`–`C3` + Track R provenance notes), uncertainty flags (`U1`–`U5`), and `run_pipeline()`.
- **Dev Weight Tuning & Retrieval Evaluation (`experiments/tune_weights.py`, `experiments/run_retrieval.py`, `src/evaluation/`)**:
  - Tuned per-intent fusion weights on the 50 Track S dev questions (`configs/weights.json`, `results/metrics/weight_sensitivity_dev.json`).
  - Evaluated all 8 retrieval systems (`RS1`, `RS2`, `B1`, `RS4`, `RS5`, `B3`, `B2`, `ORACLE`) on `S_dev`, `S_test`, and `R_test` with 1,000-resample paired bootstrap 95% confidence intervals, context budget study ($k \in \{5, 10, 20\}$), and rule-intent vs. oracle-intent ablation (`results/metrics/retrieval_metrics.json`, `results/outputs/retrieval_runs.jsonl`, `docs/retrieval_evaluation_phase3.md`).
- **Verification (`tests/test_phase3.py`)**:
  - Added 8 unit tests covering all temporal states, availability and patient scoping, $R_h$, $R_t$, fusion normalisation and weight validation, `C1`–`C3` conflicts, `U1`–`U5` uncertainty flags, and hand-computed retrieval metrics.
