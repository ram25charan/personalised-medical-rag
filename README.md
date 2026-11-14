# Chronology-Aware Context Fusion for Personalized Medical Retrieval-Augmented Generation

Research prototype and evaluation suite for longitudinal patient-record Retrieval-Augmented Generation (RAG) with question relevance ($R_q$), history relevance ($R_h$), and intent-dependent temporal relevance ($R_t$).

> **Notice:** Research prototype. Not medical advice. Uses synthetic data and the openly licensed, de-identified MIMIC-IV-ED Demo v2.2 dataset.

## Repository Structure

```text
personalized-medical-rag/
├── app/
│   └── main.py                    # Single Streamlit page (Patient / Clinician view-mode switch)
├── data/
│   ├── raw/mimic_iv_ed_demo/      # Original MIMIC-IV-ED demo v2.2 files (read-only, ignored by git)
│   ├── processed/                 # events.jsonl, timelines, quality & coverage reports
│   ├── synthetic/                 # Generated synthetic patients and events (Track S)
│   ├── benchmark/                 # questions.jsonl, gold.jsonl
│   └── corpus/                    # demo_corpus.jsonl (demo only)
├── src/
│   ├── schemas.py                 # Shared data contracts and validators (Section 12.1)
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
├── tests/                         # Unit tests per phase
└── docs/                          # Master specification, data audit, implementation logs, paper drafts
```

## Setup Instructions

1. Create and activate the virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Place the MIMIC-IV-ED Demo v2.2 files inside `data/raw/mimic_iv_ed_demo/`.
3. Run tests:
   ```powershell
   pytest
   ```

## Dataset Attribution

- **MIMIC-IV-ED Demo v2.2**: Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Mark, R., & Horng, S. (2023). *MIMIC-IV-ED* (version 2.2). PhysioNet. https://doi.org/10.13026/5ntk-km72 (Open Data Commons Open Database License v1.0).
- Pollard, T., et al. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*. https://doi.org/10.1038/s44360-026-00096-z.
