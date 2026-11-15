# Chronology-Aware Context Fusion for Personalized Medical Retrieval-Augmented Generation

A longitudinal patient-record Retrieval-Augmented Generation (RAG) system and evaluation suite that models **query-time temporal validity**, **clinical history relevance**, and **intent-dependent temporal scoring** to prevent stale, ended, or superseded records from corrupting clinical question answering.

> **Clinical & Research Notice:** Research prototype. Not medical advice. Uses deterministic synthetic longitudinal records (Track S) and the openly licensed, de-identified MIMIC-IV-ED Demo v2.2 dataset (Track R).

---

## Table of Contents

1. [Problem Motivation & Core Idea](#1-problem-motivation--core-idea)
2. [System Architecture & Pipeline](#2-system-architecture--pipeline)
3. [Datasets, Benchmark & Demo Corpus](#3-datasets--benchmark--demo-corpus)
4. [Experimental Configurations & Evaluation Results](#4-experimental-configurations--evaluation-results)
5. [Interactive Clinical Record Viewer (Streamlit)](#5-interactive-clinical-record-viewer-streamlit)
6. [Repository Structure & Configuration](#6-repository-structure--configuration)
7. [Installation & Reproducibility Guide](#7-installation--reproducibility-guide)
8. [Test Suite Summary](#8-test-suite-summary)
9. [Dataset Attributions, Licenses & Known Limitations](#9-dataset-attributions-licenses--known-limitations)

---

## 1. Problem Motivation & Core Idea

Standard medical RAG pipelines treat a patient's electronic health record (EHR) as a flat bag of text chunks ranked solely by semantic or lexical similarity ($R_q$). In longitudinal care, this fails systematically:

- **Stale / Superseded Therapy Confusion:** A discontinued medication (e.g., *lisinopril* stopped in 2023 and replaced by *amlodipine*) often has high lexical overlap with *"What medications is the patient currently taking?"*, causing the LLM to report inactive drugs as active.
- **Temporal Leakage:** Events occurring *after* a query reference date (`as_of_date`) must be strictly excluded when reconstructing historical state.
- **Intent Blindness:** A query about *current status* needs `ACTIVE` / `PERSISTENT` events, whereas a query about *longitudinal change* (`change_over_time`) or *first occurrence* (`first_occurrence`) needs historical and superseded records across time.
- **Unflagged Conflicts & Missing Evidence:** Overlapping prescriptions, date conflicts, imprecise dates (`month` / `year` / `unknown`), and insufficient evidence require explicit structured warnings rather than hallucinated certainty.

To address this, the framework computes **temporal state at query time (`as_of_date`)** and ranks candidate patient events using a frozen, per-intent linear fusion:

$$S(e, q) = w_q(\text{intent}) \cdot \tilde{R}_q(e, q) + w_h(\text{intent}) \cdot R_h(e, q) + w_t(\text{intent}) \cdot R_t(e, \text{state}(e, t_q), \text{intent})$$

followed by **chronological ordering** and **explicit temporal/conflict/uncertainty annotation (`A2`)** in the prompt context.

---

## 2. System Architecture & Pipeline

```text
  Patient Events (JSONL) + Query (text, as_of_date, optional window_start)
         │
         ▼
  [1] Temporal Availability Filter (E_<=tq) & Query-Time State Computation
      └─► States: ACTIVE | ENDED | SUPERSEDED | PERSISTENT | UNKNOWN | LISTED | NOT_LISTED_LATEST
         │
         ▼
  [2] Multi-Signal Event Scoring (per candidate event e in E_<=tq)
      ├─► Rq : Hybrid Lexical (BM25) + Dense (all-MiniLM-L6-v2) via RRF (k=60)
      ├─► Rh : Clinical History Relevance via frozen relation table (configs/relation_table.json)
      └─► Rt : Intent-Dependent Temporal Relevance (configs/rt_rules.json)
         │
         ▼
  [3] Chronology-Aware Fusion & Selection (configs/weights.json)
      └─► Top-k selection (default k=10) + chronological ordering (oldest → newest)
         │
         ▼
  [4] Conflict & Uncertainty Detection (question-concept scoped)
      ├─► Conflicts   : C1 (overlapping active meds), C2 (therapeutic duplication), C3 (date conflict)
      └─► Uncertainty : U1 (imprecise date), U2 (low confidence), U3 (stale lab >180d),
                        U4 (snapshot ambiguity), U5 (insufficient_evidence, tau=0.15)
         │
         ├──────────────────────────────────────────────────────────┐
         ▼                                                          ▼
  [5a] Grounded LLM Generation (temp=0.0, JSON)             [5b] Demo Background Corpus (UI only)
       └─► [E#] event citations + status claims                  └─► Separate BM25+MiniLM+RRF index
           + abstain / conflict / uncertainty flags                  over 25 MedlinePlus/DailyMed chunks
           (Groq openai/gpt-oss-120b or Gemini Flash)                ([C1], [C2]; never mixed into [E#])
```

### Key Design Mechanics

- **7 Verbatim Temporal States (`src/temporal/state.py`):**
  - *Interval-based (Track S):* `ACTIVE`, `ENDED`, `SUPERSEDED`, `PERSISTENT`, `UNKNOWN`.
  - *Snapshot-based (Track R ED visits):* `LISTED`, `NOT_LISTED_LATEST`.
- **5 Clinical Query Intents (`configs/rt_rules.json`):** `current_state`, `historical`, `change_over_time`, `first_occurrence`, and `persistent_fact`.
- **3 Context Annotation Levels:**
  - `A0`: Raw event text and dates only.
  - `A1`: Event text + dates + computed temporal state label.
  - `A2`: `A1` + explicit conflict (`C1`–`C3`) and uncertainty (`U1`–`U5`) flags.
- **Frozen Per-Intent Fusion Weights (`configs/weights.json`, tuned on Track S Dev only):**
  - `current_state`: `B2` $(\alpha=0.8, \beta=0.1, \gamma=0.1)$
  - `historical`, `change_over_time`, `first_occurrence`, `persistent_fact`: `B2` $(\alpha=0.1, \beta=0.1, \gamma=0.8)$

---

## 3. Datasets & Benchmark Suite

| Artifact | Track / Scope | Size & Composition | Location |
|---|---|---|---|
| **Track R Events** | Real MIMIC-IV-ED Demo v2.2 | **5,471 events** across **64 patients** (ED visits, triage, vitals, diagnoses, `medrecon`, `pyxis`; relative shifted dates) | `data/processed/events.jsonl` |
| **Track S Events** | Deterministic Synthetic (`seed=42`) | **13,116 events** across **100 patients** (`S_001`–`S_100`; disjoint **34 dev** / **66 test** patient splits) | `data/synthetic/events.jsonl` |
| **Benchmark Questions & Gold** | 7 clinical challenge categories | **180 questions** total: **50 `S_dev`**, **100 `S_test`** (86 answerable, 14 abstain), **30 `R_test`** (26 answerable, 4 abstain) | `data/benchmark/questions.jsonl`, `data/benchmark/gold.jsonl` |
| **Demo Clinical Corpus** | UI background box only | **25 passages (`C_001`–`C_025`)**, 100–143 words each: 9 MedlinePlus conditions, 6 MedlinePlus lab explainers, 10 paraphrased FDA DailyMed drug-class notes | `data/corpus/demo_corpus.jsonl` |

### 7 Benchmark Challenge Categories
`temporal_ordering` · `current_vs_historical` · `longitudinal_change` · `irrelevant_history` · `missing_or_ambiguous_dates` · `conflicting_information` · `insufficient_evidence`

---

## 4. Experimental Configurations & Evaluation Results

All experiments use a fixed context budget of $k = 10$ (with budget ablation $k \in \{5, 10, 20\}$), `temperature = 0.0`, frozen prompt template `v1.0`, and **1,000-resample paired bootstrap** confidence intervals (`seed = 42`). Each experimental table is strictly pinned to a single generator model ID.

### 4.1 Evaluated Systems
- **Retrieval Baselines:** `RS1` (BM25 only), `RS2` (Dense `all-MiniLM-L6-v2` only), `RS4` (Pure recency), `RS5` (Hard status filter + hybrid), `B3` ($R_q + R_h$ without $R_t$).
- **Generation Systems ($2 \times 2$ Selection $\times$ Annotation + Baselines):**
  - `B1`: Hybrid $R_q$ selection + `A0` raw context (Conventional RAG baseline)
  - `B1+`: Hybrid $R_q$ selection + `A2` annotations
  - `B2-`: Chronology-aware selection ($R_q + R_h + R_t$) + `A0` raw context
  - `B4`: Chronology-aware selection + `A1` status labels only (no conflict/uncertainty flags)
  - `B2`: **Proposed Full System** — Chronology-aware selection + `A2` status, conflict, and uncertainty flags
  - `ORACLE`: Gold-event selection upper bound + `A2`

### 4.2 Retrieval-Level Results (`k = 10`)

#### Track S Test Set (100 Questions; 86 Answerable)

| System | Recall@5 | Recall@10 | nDCG@10 (95% CI) | MRR | HitRate@5 | Stale-Evidence Rate |
|---|---:|---:|---|---:|---:|---:|
| **RS1** (BM25) | 0.7267 | 0.8372 | 0.7109 [0.6324, 0.7829] | 0.6928 | 0.7674 | 0.2210 |
| **RS2** (Dense) | 0.8081 | 0.8895 | 0.8050 [0.7280, 0.8716] | 0.7948 | 0.8372 | 0.2600 |
| **B1** (Hybrid RRF) | 0.8256 | 0.8605 | 0.7870 [0.7088, 0.8528] | 0.7814 | 0.8372 | 0.2540 |
| **RS4** (Recency) | 0.0291 | 0.1105 | 0.0445 [0.0222, 0.0691] | 0.0267 | 0.0581 | 0.0460 |
| **RS5** (Status Filter) | 0.6919 | 0.7267 | 0.6839 [0.5971, 0.7648] | 0.7174 | 0.7442 | 0.1761 |
| **B3** ($R_q + R_h$) | 0.8256 | 0.8953 | 0.7986 [0.7254, 0.8593] | 0.7872 | 0.8372 | 0.2550 |
| **B2** ($R_q + R_h + R_t$) | **0.8372** | **0.9070** | **0.8589 [0.7889, 0.9220]** | **0.8465** | **0.8372** | 0.2530 |

- **Paired gains on `S_test`:** `B2` vs `B1` improves nDCG@10 by **`+0.0719`** `[+0.0467, +0.1008]` and Recall@10 by **`+0.0465`** `[+0.0116, +0.0872]`. `B2` vs `B3` (isolating $R_t$) improves nDCG@10 by **`+0.0603`** `[+0.0369, +0.0881]`.
- **Track R Real-Data Transfer (`R_test`, 30 Questions):** `B2` improves nDCG@10 from `0.7453` (`B1`) to **`0.7940`** `[0.6591, 0.9161]`, Recall@10 from `0.8462` to **`0.8750`**, and MRR from `0.7288` to **`0.7840`**.

### 4.3 Generation-Level Results (`S_test` and `R_test`)

#### Primary Generator: Groq `openai/gpt-oss-120b` on `S_test` (100 Questions)

| System | Answer Correctness (95% CI) | Temporal Consistency | Contradiction Rate | Citation F1 | Abstention Acc | Conflict Flag Acc | Uncertainty Acc | Faithfulness |
|---|---|---|---|---:|---:|---:|---:|---:|
| **B1** | 0.8267 [0.7567, 0.8968] | 0.8500 | 0.1000 | 0.8633 | 0.9100 | 0.9500 | 0.6900 | 1.0000 |
| **B1+** | 0.8800 [0.8200, 0.9400] | 0.9000 | 0.0500 | 0.8800 | 0.9100 | 1.0000 | 0.9500 | 1.0000 |
| **B2-** | 0.8500 [0.7833, 0.9167] | 0.9000 | 0.0700 | 0.9100 | 0.9800 | 0.9700 | 0.6800 | 0.9900 |
| **B4** | 0.9000 [0.8400, 0.9500] | **0.9700** | **0.0300** | 0.9167 | **0.9800** | 0.9900 | 0.7900 | 0.9900 |
| **B2** | **0.9000 [0.8400, 0.9500]** | 0.9600 | **0.0300** | **0.9267** | **0.9800** | **1.0000** | **0.9700** | **1.0000** |

- **Paired Hypothesis Tests (`S_test`, Groq `openai/gpt-oss-120b`):**
  - `H1/H2 (B2 vs B1)`: Answer Correctness **`+0.0733`** `[+0.0299, +0.1201]` ($p < 0.0001$), Contradiction Rate **`-0.0700`** `[-0.1200, -0.0200]` ($p = 0.0020$, **70% reduction**), Citation F1 **`+0.0633`** ($p < 0.0001$), Conflict Flag Accuracy **`+0.0500`** ($p = 0.0100$).
  - `current_vs_historical` category: Answer Correctness doubles from **`0.4000` (`B1`) $\to$ `0.8000` (`B2`)**, while Contradiction Rate drops from **`0.6000` $\to$ `0.2000`**.
  - `2x2` Factorial (`E3`): Selection main effect = `+0.0217` correctness (`+0.0467` citation F1); Annotation (`A2` vs `A0`) main effect = `+0.0517` correctness (`-0.0450` contradiction rate, `+0.2750` uncertainty accuracy).

#### Secondary Generator: Gemini `gemini-flash-lite-latest` (`S_test` & `R_test`)

| Split | System | Answer Correctness (95% CI) | Temporal Consistency | Contradiction Rate | Citation F1 | Abstention Acc | Uncertainty Acc |
|---|---|---|---:|---:|---:|---:|---:|
| **`S_test`** (100 Q) | **B1** | 0.8500 [0.7800, 0.9200] | 0.5700 | 0.1200 | 0.8800 | 0.8900 | 0.8400 |
| **`S_test`** (100 Q) | **B2** | **0.9000 [0.8400, 0.9500]** | **0.8850** | **0.0300** | **0.9267** | **1.0000** | **0.8900** |
| **`S_test`** (100 Q) | **ORACLE** | 1.0000 [1.0000, 1.0000] | 0.7400 | 0.0000 | 1.0000 | 1.0000 | 0.9100 |
| **`R_test`** (30 Q) | **B1** | 0.7111 [0.5556, 0.8667] | 0.6933 | 0.0667 | 0.7911 | 0.8333 | 0.8667 |
| **`R_test`** (30 Q) | **B2** | **0.8022 [0.6622, 0.9223]** | **0.9000** | **0.0000** | **0.8340** | **0.9000** | **0.9000** |
| **`R_test`** (30 Q) | **ORACLE** | 0.9667 [0.9000, 1.0000] | 0.8000 | 0.1000 | 1.0000 | 1.0000 | 1.0000 |

- **Error Taxonomy (`S_test`, Section 8.9):** On both generators, `B2` eliminates **100%** of stale/superseded state confusion errors (`0/100`), abstention/hallucination errors (`0/100`), and extraction errors (`0/100`), leaving only `10` retrieval misses at $k=10$ (down from `20` total errors in `B1` on Groq and `15` on Gemini).

---

## 5. Interactive Clinical Record Viewer (Streamlit)

The single-page clinical record viewer ([app/main.py](file:///c:/Users/DELL/Desktop/Personalised-medical-rag/app/main.py)) calls `run_retrieval_stage()` directly and adds no separate retrieval logic:

- **Patient vs. Clinician Density Switch:** Patient mode renders clean evidence cards with `[E#]` citations and plain-language answers; Clinician mode renders a compact scored table (`Rq`, `Rh`, `Rt`, `Fused S`), provenance (`source_table`), and an automatic structured longitudinal summary.
- **Longitudinal Swimlane Timeline:** Plotly timeline grouped by `event_type`, with horizontal validity bars for interval events, muted markers for `ENDED` / `SUPERSEDED` events, accent-blue outlines + `[E#]` labels for selected top-$k$ events, a dashed vertical line at `as_of_date`, and a **"Days Since First Recorded Visit"** axis for date-shifted Track R records.
- **Immediate `B1 vs B2` Selection Diff:** Shows `ONLY B2 (Promoted by Rh/Rt)`, `ONLY B1 (Dropped in B2)`, and `SHARED` events immediately with zero LLM calls; LLM generation runs only on explicit button click using a single pinned model and disk cache first.
- **Data Governance:** Gemini is automatically disabled when viewing Track R (`MIMIC-IV-ED`) records.
- **Separate General Background Box:** Displays at most 2 thresholded `[C1]`/`[C2]` MedlinePlus or FDA DailyMed passages, suppressed automatically on off-topic or `U5` (`insufficient_evidence`) queries.

---

## 6. Repository Structure & Configuration

```text
personalized-medical-rag/
├── .streamlit/config.toml         # Light clinical theme (#0F62FE accent)
├── app/
│   └── main.py                    # Single-page Streamlit Clinical Record Viewer
├── configs/
│   ├── relation_table.json        # Frozen clinical concept lexicon & relations (v1.0)
│   ├── rt_rules.json              # Intent classification & Rt scoring rules
│   ├── synthetic_generator.json   # Seeded synthetic cohort configuration
│   ├── systems.json               # System definitions (RS1-RS5, B1-B4, B1+, B2-, ORACLE)
│   └── weights.json               # Frozen dev-tuned per-intent fusion weights
├── data/
│   ├── raw/mimic_iv_ed_demo/      # Original MIMIC-IV-ED Demo v2.2 CSVs (read-only, gitignored)
│   ├── processed/                 # Track R events.jsonl (5,471), timelines, quality & coverage reports
│   ├── synthetic/                 # Track S events.jsonl (13,116 events, 100 patients)
│   ├── benchmark/                 # questions.jsonl & gold.jsonl (180 questions)
│   └── corpus/demo_corpus.jsonl   # 25 validated demo background passages (C_001-C_025)
├── docs/                          # Master spec, phase reports, data audit, implementation log
├── experiments/
│   ├── tune_weights.py            # Dev-set grid search for per-intent weights
│   ├── run_retrieval.py           # Full retrieval benchmark & budget/ablation studies
│   ├── run_generation.py          # Resumable priority-ordered generation runner
│   └── compare.py                 # Markdown report & failure taxonomy generator
├── results/
│   ├── cache/                     # Disk-backed SHA-256 LLM response cache
│   ├── metrics/                   # Retrieval & generation JSON metrics + manual review sheets
│   ├── outputs/                   # Per-question JSONL run logs
│   └── figures/                   # Publication figures
├── src/
│   ├── schemas.py                 # Pydantic v2 data contracts (v1.0)
│   ├── pipeline.py                # Unified run_retrieval_stage() & run_pipeline()
│   ├── data/                      # MIMIC-IV-ED ingestion, normalisation, event builder
│   ├── synthetic/                 # Synthetic patient & benchmark question/gold generators
│   ├── temporal/                  # State computation, Rh relation scoring, Rt intent scoring
│   ├── retrieval/                 # BM25, dense MiniLM, RRF, corpus retrieval & validator
│   ├── fusion/                    # Linear fusion & chronological ordering
│   ├── flags/                     # Conflict (C1-C3) & uncertainty (U1-U5) detectors
│   ├── generation/                # Prompt template v1.0, disk cache, MultiKeyLLMClient
│   └── evaluation/                # Retrieval/answer metrics, bootstrap CIs, evaluation runner
└── tests/                         # 31 unit & integration tests (test_phase1.py .. test_phase5.py)
```

---

## 7. Installation & Reproducibility Guide

### 7.1 Environment Setup

1. **Create and activate a virtual environment (Python 3.11–3.13; verified on 3.13.5):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   *(On CPU-only machines, install CPU PyTorch before `sentence-transformers` if needed.)*

2. **Configure API keys (for live LLM generation only; cached runs and retrieval/UI work without keys):**
   Copy `.env.example` to `.env` and set your Groq and/or Gemini keys:
   ```dotenv
   GROQ_API_KEY_1=...
   GROQ_API_KEY_2=...
   GEMINI_API_KEY_1=...
   GEMINI_API_KEY_2=...
   ```

3. **Place MIMIC-IV-ED Demo v2.2 CSVs** (optional if using committed `data/processed/events.jsonl`) in `data/raw/mimic_iv_ed_demo/`.

### 7.2 Reproducing Data, Experiments & Application

- **Rebuild Track R events & audit reports:**
  ```powershell
  python -m src.data.run_phase1
  ```
- **Rebuild Track S synthetic cohort (100 patients) & benchmark (180 questions):**
  ```powershell
  python -m src.synthetic.run_phase2
  ```
- **Tune dev weights & run full retrieval evaluation (`RS1`–`RS5`, `B1`, `B3`, `B2`, `ORACLE` + budget study):**
  ```powershell
  python -m experiments.tune_weights
  python -m experiments.run_retrieval
  ```
- **Run generation evaluation (cached responses replay deterministically; uncached queries call the API):**
  ```powershell
  python -m experiments.run_generation --provider groq --model-id openai/gpt-oss-120b --systems B1,B2,B1+,B2-,B4 --splits S_test
  python -m experiments.run_generation --provider gemini --model-id gemini-flash-lite-latest --systems B1,B2,B1+,B2-,B4,ORACLE --splits S_test,R_test --output-metrics results/metrics/generation_metrics_gemini.json --output-runs results/outputs/generation_runs_gemini.jsonl --manual-review-out results/metrics/manual_review_20_gemini.json
  python -m experiments.compare
  ```
- **Validate the demo clinical corpus:**
  ```powershell
  python -m src.retrieval.validate_corpus
  ```
- **Launch the Streamlit Clinical Record Viewer:**
  ```powershell
  python -m streamlit run app/main.py
  ```

---

## 8. Test Suite Summary

Run the full automated test suite from the repository root:

```powershell
python -m pytest -v
```

All **31 / 31** tests pass:

| Test File | Tests | What Is Verified |
|---|---:|---|
| [tests/test_phase1.py](file:///c:/Users/DELL/Desktop/Personalised-medical-rag/tests/test_phase1.py) | 8 | MIMIC-IV-ED table counts, data quality audit, medication/diagnosis normalisation, Track R event provenance, timelines, JSONL roundtrip, Pydantic schema validation |
| [tests/test_phase2.py](file:///c:/Users/DELL/Desktop/Personalised-medical-rag/tests/test_phase2.py) | 5 | Fixed-seed (`seed=42`) determinism, seed sensitivity, disjoint dev/test patient splits, gold/distractor integrity, frozen relation table |
| [tests/test_phase3.py](file:///c:/Users/DELL/Desktop/Personalised-medical-rag/tests/test_phase3.py) | 8 | All 7 temporal states, `as_of_date` availability & patient scoping, $R_h$ scoring, $R_t$ intent scoring, fusion normalisation/weights, `C1`–`C3` conflicts, `U1`–`U5` uncertainty flags, retrieval metrics |
| [tests/test_phase4.py](file:///c:/Users/DELL/Desktop/Personalised-medical-rag/tests/test_phase4.py) | 6 | `AnswerJSON` parsing & `[E#]` citation resolution, disk cache roundtrip, 429 key rotation, `A0`/`A1`/`A2` context assembly, deterministic answer metrics (`answer_correctness`, `contradiction_rate`, `abstention_accuracy`) |
| [tests/test_phase5.py](file:///c:/Users/DELL/Desktop/Personalised-medical-rag/tests/test_phase5.py) | 4 | Demo corpus schema/word-count/concept/advice validation, 10-question + off-topic/`U5` corpus retrieval, `[C#]` separation guard, end-to-end UI execution on Track S & Track R + Track R Gemini block |

---

## 9. Dataset Attributions, Licenses & Known Limitations

- **MIMIC-IV-ED Demo v2.2:** Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Mark, R., & Horng, S. (2023). *MIMIC-IV-ED* (version 2.2). PhysioNet. https://doi.org/10.13026/5ntk-km72 (Open Data Commons Open Database License v1.0).
- **PhysioNet Citation:** Pollard, T., et al. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*. https://doi.org/10.1038/s44360-026-00096-z.
- **MedlinePlus (U.S. National Library of Medicine):** Public-domain condition and laboratory test summaries used in `data/corpus/demo_corpus.jsonl` with source URLs and NLM attribution notices.
- **DailyMed (U.S. FDA Prescribing Information):** Drug-class background notes paraphrased from public-domain FDA labels; no ASHP/MedlinePlus drug monographs are used.
- **Known Limitations:** Track R is derived from emergency-department snapshots (`medrecon`/`pyxis`) rather than outpatient prescription intervals, and its test split is small (`30` questions) with weak labels by construction.
