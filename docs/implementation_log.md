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
