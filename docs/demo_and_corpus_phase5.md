# Phase 5 Documentation: Demo Clinical Corpus & Streamlit Record Viewer

- **Phase Milestone:** `v0.5-phase5`
- **Specification Reference:** Sections 6.4, 9.1–9.6, 11.3–11.7 of `docs/master_specification.md` and the Phase 5 Plan.

---

## 1. Demo Clinical Corpus (`data/corpus/demo_corpus.jsonl`)

### 1.1 Data-Driven Topic Selection & Provenance
Topics were selected directly by querying the normalised concepts appearing in `data/processed/events.jsonl` (Track R), `data/synthetic/events.jsonl` (Track S), and `configs/relation_table.json`. The corpus contains **25 passages (`C_001`–`C_025`)** across **11 clinical topic groups**:

| Topic Group | Passages | Types | Pipeline Concept Keys |
|---|---|---|---|
| Hypertension & Antihypertensives | `C_001`, `C_002` | `condition_overview`, `drug_class_note` | `essential (primary) hypertension`, `lisinopril`, `enalapril maleate`, `amlodipine` |
| Type 2 Diabetes & Glycemic Control | `C_003`, `C_004`, `C_005` | `condition_overview`, `test_explainer`, `drug_class_note` | `type 2 diabetes mellitus`, `hba1c`, `fasting glucose`, `metformin`, `glipizide`, `insulin glargine` |
| Hyperlipidemia & Statins | `C_006`, `C_007`, `C_008` | `condition_overview`, `test_explainer`, `drug_class_note` | `hyperlipidemia`, `ldl cholesterol`, `atorvastatin`, `rosuvastatin`, `simvastatin` |
| Chronic Kidney Disease & Electrolytes | `C_009`, `C_010`, `C_011` | `condition_overview`, `test_explainer`, `test_explainer` | `chronic kidney disease`, `serum creatinine`, `egfr`, `potassium`, `potassium chloride`, `furosemide`, `hydrochlorothiazide`, `lisinopril` |
| Hypothyroidism & Thyroid Replacement | `C_012`, `C_013`, `C_014` | `condition_overview`, `test_explainer`, `drug_class_note` | `hypothyroidism`, `tsh`, `levothyroxine` |
| Atrial Fibrillation & Anticoagulants / Rate Control | `C_015`, `C_016`, `C_017` | `condition_overview`, `drug_class_note`, `drug_class_note` | `atrial fibrillation`, `apixaban`, `warfarin`, `metoprolol`, `digoxin` |
| Heart Failure & Diuretics | `C_018`, `C_019` | `condition_overview`, `drug_class_note` | `congestive heart failure`, `furosemide`, `hydrochlorothiazide` |
| GERD & Proton-Pump Inhibitors | `C_020`, `C_021` | `condition_overview`, `drug_class_note` | `gastroesophageal reflux disease`, `omeprazole`, `pantoprazole` |
| Asthma & Bronchodilators | `C_022`, `C_023` | `condition_overview`, `drug_class_note` | `asthma`, `albuterol` |
| Antiplatelet & Nitrate Therapy | `C_024` | `drug_class_note` | `aspirin`, `clopidogrel`, `nitroglycerin` |
| Drug & Material Allergies | `C_025` | `test_explainer` | `penicillin allergy`, `sulfa allergy`, `latex allergy` |

### 1.2 Automated Corpus Validation (`src/retrieval/validate_corpus.py`)
- **Schema Validation:** All 25 passages validate against `CorpusChunk` (`src/schemas.py`).
- **Word-Count Bounds:** Every passage is strictly between **100 and 200 words** (`min = 100`, `max = 143`, `avg = 114.5`).
- **Provenance & Licensing:**
  - `condition_overview` (9) and `test_explainer` (6) passages use public-domain **MedlinePlus (U.S. National Library of Medicine)** health-topic and lab-test summaries with URLs, access dates (`2026-11-15`), and NLM attribution notices.
  - `drug_class_note` (10) passages paraphrase the **Indications and Usage** section of U.S. FDA labels on **DailyMed**. Zero ASHP/MedlinePlus drug monograph URLs (`medlineplus.gov/druginfo`) are used.
- **Zero Advice / Dosing Filter:** Verified zero matches for `mg`, `mcg`, `dose`, `dosing`, `take`, `twice daily`, `once daily`, or `you should`.
- **Concept Alignment:** 100% of concept keys exist in `configs/relation_table.json` and the normalised Track R / Track S event tables.

---

## 2. Corpus Retrieval & Strict Separation (`src/retrieval/corpus_retrieval.py`)

1. **Hybrid BM25 + MiniLM + RRF with Relevance Threshold:**
   - Queries over `data/corpus/demo_corpus.jsonl` combine the clinical question with relevant selected event concepts (`[E#]`).
   - Returns at most **`top_k = 2`** passages (`[C1]`, `[C2]`).
   - Enforces a relevance gate (`min_relevance_threshold = 0.38` plus concept match or high direct question similarity) and suppresses the background box (`returns []`) when `insufficient_evidence` (`U5`) is `True` or when the question is off-topic.
2. **10-Question Retrieval Verification:**
   - Verified across 9 clinical questions (covering hypertension, diabetes/HbA1c, hyperlipidemia/LDL, CKD/creatinine/eGFR, hypothyroidism/TSH, AFib/anticoagulation, CHF/furosemide, GERD/PPIs, and drug allergies) plus 1 off-topic question (`"What is the weather forecast in Tokyo and how do you bake sourdough bread?"` $\to$ returns `[]`).
3. **Separation Guard (`assert_no_corpus_citation_in_claims`):**
   - Verifies that no `[C#]` or `C_###` ID ever appears in `AnswerJSON.cited_event_ids` or `AnswerJSON.answer_text`.

---

## 3. Single-Page Streamlit Record Viewer (`app/main.py` & `.streamlit/config.toml`)

1. **Clinical Record Viewer Design System:**
   - Restrained neutral palette (`#FFFFFF`, `#F4F6F8`, `#161616`) with a single primary accent colour (`#0F62FE`) in `.streamlit/config.toml`.
   - Monospace typography (`IBM Plex Mono` / `Consolas`) for IDs (`[E1]`, `S_020_0014`, `[C1]`), dates, and component scores (`Rq`, `Rh`, `Rt`, `Fused S`).
   - Verbatim pipeline status badges (`● ACTIVE`, `◆ PERSISTENT`, `● LISTED`, `○ ENDED`, `○ NOT_LISTED_LATEST`, `⊘ SUPERSEDED`, `? UNKNOWN`) with matching colours and colour-blind-accessible geometric symbols across the timeline, status claims, and evidence list.
2. **Longitudinal Timeline Centrepiece:**
   - Swimlanes by `event_type`, horizontal bars for validity intervals (`valid_start` $\to$ `valid_end`), markers for point events, muted opacity for ended/superseded events, accent-blue outlines + `[E#]` labels for selected top-$k$ events, and a vertical dashed reference line at `as_of_date`.
   - Defaults to `"Days Since First Recorded Visit (Day 0, Day +N)"` when viewing Track R (`MIMIC-IV-ED`) records due to date shifting.
3. **Immediate `B1 vs B2` Selection Diff & On-Demand Generation:**
   - Displays `B1` vs `B2` top-$k$ selection differences immediately (`ONLY B2 (Promoted by Rh/Rt)`, `ONLY B1 (Dropped in B2)`, `SHARED`) with zero LLM calls, and triggers LLM generation only when the user clicks `Generate B2 Answer`, `Generate B1 Answer`, or `Generate Both`.
4. **Single Pinned Provider & Track R Governance:**
   - Uses a single pinned model per provider (`openai/gpt-oss-120b` or `gemini-flash-lite-latest` on Track S) with disk cache lookup first and parsed retry-wait messages on HTTP 429.
   - Automatically blocks Gemini on Track R (`MIMIC-IV-ED`) records to comply with data-protection governance.
