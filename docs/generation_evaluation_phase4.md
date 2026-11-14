# Phase 4 Generation-Level Evaluation Report

- **Git commit:** `3e3593f`
- **Primary Generator Provider & Model:** `groq` / `openai/gpt-oss-120b` (`temperature = 0.0`)
- **Default context budget:** `k = 10`
- **Bootstrap resamples:** `1,000` paired resamples (`seed = 42`)

## Track S Test Set (100 Questions; 86 Answerable, 14 Abstain)

| System | Answer Correctness (95% CI) | Temporal Consistency | Contradiction Rate | Citation F1 | Abstention Acc | Conflict Flag Acc | Uncertainty Acc | Faithfulness |
|---|---|---|---|---:|---:|---:|---:|---:|
| **B1** | 0.8267 [0.7567, 0.8968] | 0.8500 | 0.1000 | 0.8633 | 0.9100 | 0.9500 | 0.6900 | 1.0000 |
| **B1+** | 0.8800 [0.8200, 0.9400] | 0.9000 | 0.0500 | 0.8800 | 0.9100 | 1.0000 | 0.9500 | 1.0000 |
| **B2-** | 0.8500 [0.7833, 0.9167] | 0.9000 | 0.0700 | 0.9100 | 0.9800 | 0.9700 | 0.6800 | 0.9900 |
| **B4** | 0.9000 [0.8400, 0.9500] | 0.9700 | 0.0300 | 0.9167 | 0.9800 | 0.9900 | 0.7900 | 0.9900 |
| **B2** | 0.9000 [0.8400, 0.9500] | 0.9600 | 0.0300 | 0.9267 | 0.9800 | 1.0000 | 0.9700 | 1.0000 |

### Paired Bootstrap Hypothesis Tests (`S_test`, 1,000 resamples)

| Comparison | $\Delta$ Answer Correctness [95% CI] | $\Delta$ Contradiction Rate [95% CI] | $\Delta$ Citation F1 [95% CI] | $\Delta$ Conflict Flag Acc [95% CI] |
|---|---|---|---|---|
| `H1_H2_full_B2_vs_B1` | `+0.0733` [+0.0299, +0.1201] (p=0.0000) | `-0.0700` [-0.1200, -0.0200] (p=0.0020) | `+0.0633` [+0.0267, +0.1067] (p=0.0000) | `+0.0500` [+0.0100, +0.1000] (p=0.0100) |
| `H3_flags_B2_vs_B4` | `+0.0000` [+0.0000, +0.0000] (p=1.0000) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) | `+0.0100` [+0.0000, +0.0267] (p=0.2780) | `+0.0100` [+0.0000, +0.0300] (p=0.7800) |
| `E3_selection_at_A0_B2minus_vs_B1` | `+0.0233` [-0.0167, +0.0667] (p=0.2740) | `-0.0300` [-0.0700, +0.0100] (p=0.2320) | `+0.0467` [+0.0133, +0.0867] (p=0.0000) | `+0.0200` [-0.0200, +0.0600] (p=0.4660) |
| `E3_selection_at_A2_B2_vs_B1plus` | `+0.0200` [+0.0000, +0.0500] (p=0.2720) | `-0.0200` [-0.0500, +0.0000] (p=0.2720) | `+0.0467` [+0.0133, +0.0867] (p=0.0000) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |
| `E3_annotation_on_B1_B1plus_vs_B1` | `+0.0533` [+0.0167, +0.0967] (p=0.0000) | `-0.0500` [-0.0900, -0.0100] (p=0.0060) | `+0.0167` [+0.0033, +0.0333] (p=0.0100) | `+0.0500` [+0.0100, +0.1000] (p=0.0100) |
| `E3_annotation_on_B2_B2_vs_B2minus` | `+0.0500` [+0.0133, +0.0934] (p=0.0000) | `-0.0400` [-0.0800, -0.0100] (p=0.0360) | `+0.0167` [+0.0033, +0.0367] (p=0.0320) | `+0.0300` [+0.0000, +0.0700] (p=0.0900) |
| `E3_status_A1_vs_raw_A0_B4_vs_B2minus` | `+0.0500` [+0.0133, +0.0934] (p=0.0000) | `-0.0400` [-0.0800, -0.0100] (p=0.0360) | `+0.0067` [+0.0000, +0.0167] (p=0.2460) | `+0.0200` [+0.0000, +0.0500] (p=0.2460) |

### Exploratory Analysis E3: 2x2 Factorial (`Selection {B1, B2} x Annotation {A0, A2}` on `S_test`)

| Metric | `B1` (Sel=B1, A0) | `B1+` (Sel=B1, A2) | `B2-` (Sel=B2, A0) | `B2` (Sel=B2, A2) | Main Effect: Selection | Main Effect: Annotation | Interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `answer_correctness` | 0.8267 | 0.8800 | 0.8500 | 0.9000 | `+0.0217` | `+0.0517` | `-0.0033` |
| `temporal_consistency` | 0.8500 | 0.9000 | 0.9000 | 0.9600 | `+0.0550` | `+0.0550` | `+0.0100` |
| `contradiction_rate` | 0.1000 | 0.0500 | 0.0700 | 0.0300 | `-0.0250` | `-0.0450` | `+0.0100` |
| `citation_f1` | 0.8633 | 0.8800 | 0.9100 | 0.9267 | `+0.0467` | `+0.0167` | `-0.0000` |
| `conflict_flag_accuracy` | 0.9500 | 1.0000 | 0.9700 | 1.0000 | `+0.0100` | `+0.0400` | `-0.0200` |
| `uncertainty_noted_accuracy` | 0.6900 | 0.9500 | 0.6800 | 0.9700 | `+0.0050` | `+0.2750` | `+0.0300` |

### Exploratory Analysis E4: Per-Category Breakdown (`B1` vs `B2` on `S_test`)

| Category | N | B1 Correctness | B2 Correctness | B1 Contradiction | B2 Contradiction | B1 Citation F1 | B2 Citation F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `conflicting_information` | 14 | 0.9048 | 1.0000 | 0.0000 | 0.0000 | 0.8810 | 1.0000 |
| `current_vs_historical` | 15 | 0.4000 | 0.8000 | 0.6000 | 0.2000 | 0.6667 | 0.8000 |
| `insufficient_evidence` | 14 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| `irrelevant_history` | 14 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| `longitudinal_change` | 14 | 1.0000 | 1.0000 | 0.0714 | 0.0000 | 1.0000 | 1.0000 |
| `missing_or_ambiguous_dates` | 14 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| `temporal_ordering` | 15 | 0.5333 | 0.5333 | 0.0000 | 0.0000 | 0.5333 | 0.7111 |

## Secondary Generator Evaluation (`gemini` / `gemini-flash-lite-latest`)

### Track S Test Set (100 Questions) (`gemini-flash-lite-latest`)

| System | Answer Correctness (95% CI) | Temporal Consistency | Contradiction Rate | Citation F1 | Abstention Acc | Conflict Flag Acc | Uncertainty Acc | Faithfulness |
|---|---|---|---|---:|---:|---:|---:|---:|
| **B1** | 0.8500 [0.7800, 0.9200] | 0.5700 | 0.1200 | 0.8800 | 0.8900 | 1.0000 | 0.8400 | 0.9600 |
| **B1+** | 0.8800 [0.8200, 0.9400] | 0.8550 | 0.0500 | 0.8800 | 0.9600 | 1.0000 | 0.9400 | 1.0000 |
| **B2-** | 0.8667 [0.7967, 0.9268] | 0.6300 | 0.1300 | 0.9233 | 0.9400 | 0.9900 | 0.8100 | 1.0000 |
| **B4** | 0.8900 [0.8300, 0.9500] | 0.8750 | 0.0300 | 0.9267 | 0.9900 | 1.0000 | 0.8800 | 1.0000 |
| **B2** | 0.9000 [0.8400, 0.9500] | 0.8850 | 0.0300 | 0.9267 | 1.0000 | 1.0000 | 0.8900 | 1.0000 |
| **ORACLE** | 1.0000 [1.0000, 1.0000] | 0.7400 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9100 | 1.0000 |

#### Paired Bootstrap Hypothesis Tests (`S_test`, `gemini-flash-lite-latest`)

| Comparison | $\Delta$ Answer Correctness [95% CI] | $\Delta$ Contradiction Rate [95% CI] | $\Delta$ Citation F1 [95% CI] | $\Delta$ Conflict Flag Acc [95% CI] |
|---|---|---|---|---|
| `H1_H2_full_B2_vs_B1` | `+0.0500` [+0.0100, +0.1000] (p=0.0240) | `-0.0900` [-0.1500, -0.0400] (p=0.0000) | `+0.0467` [+0.0133, +0.0867] (p=0.0000) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |
| `H3_flags_B2_vs_B4` | `+0.0100` [+0.0000, +0.0300] (p=0.7580) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |
| `E3_selection_at_A0_B2minus_vs_B1` | `+0.0167` [-0.0300, +0.0667] (p=0.5260) | `+0.0100` [-0.0400, +0.0500] (p=0.8720) | `+0.0433` [+0.0100, +0.0833] (p=0.0000) | `-0.0100` [-0.0300, +0.0000] (p=0.7020) |
| `E3_selection_at_A2_B2_vs_B1plus` | `+0.0200` [+0.0000, +0.0500] (p=0.2720) | `-0.0200` [-0.0500, +0.0000] (p=0.2720) | `+0.0467` [+0.0133, +0.0867] (p=0.0000) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |
| `E3_annotation_on_B1_B1plus_vs_B1` | `+0.0300` [+0.0000, +0.0700] (p=0.1140) | `-0.0700` [-0.1200, -0.0300] (p=0.0000) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |
| `E3_annotation_on_B2_B2_vs_B2minus` | `+0.0333` [+0.0033, +0.0733] (p=0.0360) | `-0.1000` [-0.1600, -0.0500] (p=0.0000) | `+0.0033` [+0.0000, +0.0100] (p=0.7020) | `+0.0100` [+0.0000, +0.0300] (p=0.7020) |
| `E3_status_A1_vs_raw_A0_B4_vs_B2minus` | `+0.0233` [-0.0100, +0.0667] (p=0.2800) | `-0.1000` [-0.1600, -0.0500] (p=0.0000) | `+0.0033` [+0.0000, +0.0100] (p=0.7020) | `+0.0100` [+0.0000, +0.0300] (p=0.7020) |
| `Upper_bound_ORACLE_vs_B2` | `+0.1000` [+0.0500, +0.1600] (p=0.0000) | `-0.0300` [-0.0700, +0.0000] (p=0.0780) | `+0.0733` [+0.0300, +0.1267] (p=0.0000) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |

#### Exploratory Analysis E3: 2x2 Factorial (`S_test`, `gemini-flash-lite-latest`)

| Metric | `B1` (Sel=B1, A0) | `B1+` (Sel=B1, A2) | `B2-` (Sel=B2, A0) | `B2` (Sel=B2, A2) | Main Effect: Selection | Main Effect: Annotation | Interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `answer_correctness` | 0.8500 | 0.8800 | 0.8667 | 0.9000 | `+0.0184` | `+0.0317` | `+0.0033` |
| `temporal_consistency` | 0.5700 | 0.8550 | 0.6300 | 0.8850 | `+0.0450` | `+0.2700` | `-0.0300` |
| `contradiction_rate` | 0.1200 | 0.0500 | 0.1300 | 0.0300 | `-0.0050` | `-0.0850` | `-0.0300` |
| `citation_f1` | 0.8800 | 0.8800 | 0.9233 | 0.9267 | `+0.0450` | `+0.0017` | `+0.0034` |
| `conflict_flag_accuracy` | 1.0000 | 1.0000 | 0.9900 | 1.0000 | `-0.0050` | `+0.0050` | `+0.0100` |
| `uncertainty_noted_accuracy` | 0.8400 | 0.9400 | 0.8100 | 0.8900 | `-0.0400` | `+0.0900` | `-0.0200` |

### Track R Real-Data Test Set (30 Questions) — Sim-to-Real Transfer (E2) (`gemini-flash-lite-latest`)

| System | Answer Correctness (95% CI) | Temporal Consistency | Contradiction Rate | Citation F1 | Abstention Acc | Conflict Flag Acc | Uncertainty Acc | Faithfulness |
|---|---|---|---|---:|---:|---:|---:|---:|
| **B1** | 0.7111 [0.5556, 0.8667] | 0.6933 | 0.0667 | 0.7911 | 0.8333 | 1.0000 | 0.8667 | 1.0000 |
| **B1+** | 0.7778 [0.6331, 0.9111] | 0.8933 | 0.0333 | 0.8278 | 0.9000 | 1.0000 | 0.9000 | 0.9667 |
| **B2-** | 0.7356 [0.5822, 0.8800] | 0.6333 | 0.0667 | 0.8022 | 0.8333 | 1.0000 | 0.9000 | 1.0000 |
| **B4** | 0.7689 [0.6222, 0.9000] | 0.8667 | 0.0000 | 0.8467 | 0.8667 | 1.0000 | 0.8667 | 1.0000 |
| **B2** | 0.8022 [0.6622, 0.9223] | 0.9000 | 0.0000 | 0.8340 | 0.9000 | 1.0000 | 0.9000 | 0.9667 |
| **ORACLE** | 0.9667 [0.9000, 1.0000] | 0.8000 | 0.1000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

#### Paired Bootstrap Hypothesis Tests (`R_test`, `gemini-flash-lite-latest`)

| Comparison | $\Delta$ Answer Correctness [95% CI] | $\Delta$ Contradiction Rate [95% CI] | $\Delta$ Citation F1 [95% CI] | $\Delta$ Conflict Flag Acc [95% CI] |
|---|---|---|---|---|
| `H1_H2_full_B2_vs_B1` | `+0.0911` [-0.0089, +0.2156] (p=0.1020) | `-0.0667` [-0.1667, +0.0000] (p=0.2520) | `+0.0429` [-0.0254, +0.1222] (p=0.3420) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |
| `H3_flags_B2_vs_B4` | `+0.0333` [+0.0000, +0.1000] (p=0.6740) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) | `-0.0127` [-0.0714, +0.0222] (p=0.7620) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |
| `E3_selection_at_A0_B2minus_vs_B1` | `+0.0244` [-0.0267, +0.1000] (p=0.7200) | `+0.0000` [-0.1000, +0.1000] (p=1.0000) | `+0.0111` [-0.0956, +0.1111] (p=0.8180) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |
| `E3_selection_at_A2_B2_vs_B1plus` | `+0.0244` [-0.0267, +0.1000] (p=0.7200) | `-0.0333` [-0.1000, +0.0000] (p=0.7040) | `+0.0062` [-0.0592, +0.0669] (p=0.9300) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |
| `E3_annotation_on_B1_B1plus_vs_B1` | `+0.0667` [+0.0000, +0.1667] (p=0.2460) | `-0.0333` [-0.1000, +0.0000] (p=0.6920) | `+0.0367` [-0.0111, +0.1011] (p=0.2860) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |
| `E3_annotation_on_B2_B2_vs_B2minus` | `+0.0667` [+0.0000, +0.1667] (p=0.2460) | `-0.0667` [-0.1667, +0.0000] (p=0.2700) | `+0.0317` [-0.0238, +0.1095] (p=0.5220) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |
| `E3_status_A1_vs_raw_A0_B4_vs_B2minus` | `+0.0333` [+0.0000, +0.1000] (p=0.7540) | `-0.0667` [-0.1667, +0.0000] (p=0.2700) | `+0.0444` [-0.0222, +0.1444] (p=0.2660) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |
| `Upper_bound_ORACLE_vs_B2` | `+0.1644` [+0.0111, +0.3200] (p=0.0360) | `+0.1000` [+0.0000, +0.2333] (p=0.0980) | `+0.1660` [+0.0555, +0.2899] (p=0.0000) | `+0.0000` [+0.0000, +0.0000] (p=1.0000) |

#### Exploratory Analysis E3: 2x2 Factorial (`R_test`, `gemini-flash-lite-latest`)

| Metric | `B1` (Sel=B1, A0) | `B1+` (Sel=B1, A2) | `B2-` (Sel=B2, A0) | `B2` (Sel=B2, A2) | Main Effect: Selection | Main Effect: Annotation | Interaction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `answer_correctness` | 0.7111 | 0.7778 | 0.7356 | 0.8022 | `+0.0245` | `+0.0667` | `-0.0001` |
| `temporal_consistency` | 0.6933 | 0.8933 | 0.6333 | 0.9000 | `-0.0267` | `+0.2334` | `+0.0667` |
| `contradiction_rate` | 0.0667 | 0.0333 | 0.0667 | 0.0000 | `-0.0167` | `-0.0500` | `-0.0333` |
| `citation_f1` | 0.7911 | 0.8278 | 0.8022 | 0.8340 | `+0.0086` | `+0.0342` | `-0.0049` |
| `conflict_flag_accuracy` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `+0.0000` | `+0.0000` | `+0.0000` |
| `uncertainty_noted_accuracy` | 0.8667 | 0.9000 | 0.9000 | 0.9000 | `+0.0166` | `+0.0166` | `-0.0333` |

## Stratified Manual Faithfulness & Error Analysis (20 Responses, Section 8.6)

| Question ID | Category | Expected Answer | B1 Correct | B2 Correct | B1 Faithfulness | B2 Faithfulness |
|---|---|---|---:|---:|---:|---:|
| `Q_S_0056` | `conflicting_information` | `['ACTIVE', 'ENDED']` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0052` | `current_vs_historical` | `ENDED` | 0.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0057` | `insufficient_evidence` | `None` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0054` | `irrelevant_history` | `PERSISTENT` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0053` | `longitudinal_change` | `[174.0, 92.0]` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0055` | `missing_or_ambiguous_dates` | `2022-06` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0051` | `temporal_ordering` | `2021-03-24` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0063` | `conflicting_information` | `['20 mg', '40 mg']` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0059` | `current_vs_historical` | `['hydrochlorothiazide']` | 0.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0064` | `insufficient_evidence` | `None` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0061` | `irrelevant_history` | `10 mg` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0060` | `longitudinal_change` | `['5 mg', '10 mg']` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0062` | `missing_or_ambiguous_dates` | `unknown` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0058` | `temporal_ordering` | `['lisinopril', 'amlodipine']` | 0.00 | 0.00 | 1.00 | 1.00 |
| `Q_S_0070` | `conflicting_information` | `['ACTIVE', 'ENDED']` | 0.67 | 1.00 | 1.00 | 1.00 |
| `Q_S_0066` | `current_vs_historical` | `ENDED` | 0.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0071` | `insufficient_evidence` | `None` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0068` | `irrelevant_history` | `PERSISTENT` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0067` | `longitudinal_change` | `[8.4, 2.2]` | 1.00 | 1.00 | 1.00 | 1.00 |
| `Q_S_0069` | `missing_or_ambiguous_dates` | `2022-06` | 1.00 | 1.00 | 1.00 | 1.00 |

## Failure Taxonomy & Root-Cause Error Analysis (`openai/gpt-oss-120b`, Section 8.9)

### Track S Test (`openai/gpt-oss-120b`) Error Breakdown (`answer_correctness < 1.0`)

| System | Total Questions | Total Errors | Retrieval Miss at k=10 | Stale/Superseded State Confusion | Abstention / Hallucination Error | Partial Set / Value Extraction |
|---|---:|---:|---:|---:|---:|---:|
| **B1** | 100 | 20 | 12 | 4 | 0 | 4 |
| **B1+** | 100 | 12 | 12 | 0 | 0 | 0 |
| **B2-** | 100 | 17 | 10 | 4 | 0 | 3 |
| **B4** | 100 | 10 | 10 | 0 | 0 | 0 |
| **B2** | 100 | 10 | 10 | 0 | 0 | 0 |

## Failure Taxonomy & Root-Cause Error Analysis (`gemini-flash-lite-latest`, Section 8.9)

### Track S Test (`gemini-flash-lite-latest`) Error Breakdown (`answer_correctness < 1.0`)

| System | Total Questions | Total Errors | Retrieval Miss at k=10 | Stale/Superseded State Confusion | Abstention / Hallucination Error | Partial Set / Value Extraction |
|---|---:|---:|---:|---:|---:|---:|
| **B1** | 100 | 15 | 15 | 0 | 0 | 0 |
| **B1+** | 100 | 12 | 12 | 0 | 0 | 0 |
| **B2-** | 100 | 14 | 13 | 0 | 0 | 1 |
| **B4** | 100 | 11 | 11 | 0 | 0 | 0 |
| **B2** | 100 | 10 | 10 | 0 | 0 | 0 |
| **ORACLE** | 100 | 0 | 0 | 0 | 0 | 0 |

### Track R Real-Data Test (`gemini-flash-lite-latest`) Error Breakdown (`answer_correctness < 1.0`)

| System | Total Questions | Total Errors | Retrieval Miss at k=10 | Stale/Superseded State Confusion | Abstention / Hallucination Error | Partial Set / Value Extraction |
|---|---:|---:|---:|---:|---:|---:|
| **B1** | 30 | 10 | 9 | 0 | 1 | 0 |
| **B1+** | 30 | 8 | 8 | 0 | 0 | 0 |
| **B2-** | 30 | 9 | 8 | 0 | 1 | 0 |
| **B4** | 30 | 8 | 8 | 0 | 0 | 0 |
| **B2** | 30 | 7 | 7 | 0 | 0 | 0 |
| **ORACLE** | 30 | 1 | 0 | 1 | 0 | 0 |
