# SilentWindow 🪟
> **"An early-warning clinical decision support system that knows when NOT to speak."**

[![Research Prototype](https://img.shields.io/badge/Status-Research%20Prototype-amber.svg)](#disclaimer)
[![Leakage Free](https://img.shields.io/badge/Data%20Leakage-Zero%20(Patient--Level%20Split)-emerald.svg)](#strict-no-leakage-guarantee)
[![Architecture](https://img.shields.io/badge/Architecture-3--Layer%20Trust%20CDS-blue.svg)](#architecture)
[![License](https://img.shields.io/badge/Dataset-PhysioNet%20Challenge%202012-cyan.svg)](#dataset--inspection)

---

## ⚠️ Important Scientific & Safety Disclaimer
> **RESEARCH PROTOTYPE ONLY — RETROSPECTIVE DATA ONLY.**
> This software is an academic research and clinical decision-support demonstration prototype. It is **NOT** a certified medical device, does not provide clinical diagnoses or treatment recommendations, and has not undergone prospective clinical trials.
> 
> *Notice on Event Timestamps:*
> **"Lead time is estimated relative to the available outcome/proxy because an exact event timestamp is unavailable."**
> In the retrospective challenge dataset, exact physiological collapse timestamps are not recorded; all early warning lead times are strictly measured relative to telemetry duration and hospital discharge/mortality proxy.

---

## 1. The Core Problem: Alarm Fatigue & Transient Spikes

In Intensive Care Units (ICUs), clinical monitors emit hundreds of audible alarms per bed every day. Over **85% to 99% of these alarms are clinically non-actionable**, caused by:
- Temporary sensor artifacts (loose leads, patient repositioning)
- Irregular sampling intervals and missing measurements
- Physiologically implausible spikes (e.g., heart rate jumping 80 → 220 → 82 in 10 minutes)
- Rigid population threshold triggers (e.g., standard NEWS2 / single threshold models)

This causes **alarm fatigue**, leading healthcare providers to experience sensory overload, desensitization, and delayed response to true acute patient deterioration.

### The SilentWindow Principle
> **Do not simply predict risk. Decide WHEN there is enough reliable, persistent evidence to justify an alert.**

---

## 2. Conceptual Architecture

```mermaid
flowchart TD
    RAW["Raw ICU Telemetry Stream<br/>(Irregular intervals, missingness, sensor noise)"] --> L1["LAYER 1: Trust-Aware Signal Processing<br/>• Plausibility check (dataset-aware ranges)<br/>• Sudden jump detection<br/>• Multi-vital discordance assessment<br/>• Missingness indicators & staleness clocks"]
    L1 --> L2["LAYER 2: Personal Baseline & Trajectory<br/>• Patient-specific baseline (first 6h window)<br/>• Deviation & % change from personal baseline<br/>• Rolling multi-scale trajectory (1h, 3h, 6h slopes)<br/>• Multi-variable deterioration patterns"]
    L2 --> ML["XGBoost Risk Model + Platt Calibrator<br/>• Tabular past-only feature vector<br/>• Calibrated empirical event probability p(t)"]
    ML --> L3["LAYER 3: Sequential Evidence Accumulator<br/>• CUSUM-style sequential accumulator E(t)<br/>• Credibility-weighted evidence increment<br/>• Exponential decay on normalizing risk<br/>• Refractory period & alert budget ceiling"]
    L3 --> STATES{"Alert State Engine"}
    STATES -->|E(t) < 0.40| STABLE["🟢 STABLE<br/>Silent Monitoring"]
    STATES -->|0.40 ≤ E(t) < 0.75| WATCH["🟡 WATCH<br/>Dashboard advisory (No alarm)"]
    STATES -->|E(t) ≥ 0.75| ALERT["🔴 ALERT<br/>High-confidence alert with SHAP rationale"]
    ALERT --> EXP["Explainability & Timeline Audit<br/>• Top SHAP factor waterfall<br/>• Layer 1 downweighted reading provenance"]
```

![SilentWindow architecture overview](https://private-us-east-1.manuscdn.com/sessionFile/PqNYiyZEI5wUl9WdFrIM7z/sandbox/SEbiLfexsHSNxCUYCd8xIs-images_1790680019170_na1fn_L2hvbWUvdWJ1bnR1L0F1cmEtRVdTL2RvY3MvYXJjaGl0ZWN0dXJl.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvUHFOWWl5WkVJNXdVbDlXZEZySU03ei9zYW5kYm94L1NFYmlMZmV4c0hTTnhDVVlDZDh4SXMtaW1hZ2VzXzE3OTA2ODAwMTkxNzBfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwwRjFjbUV0UlZkVEwyUnZZM012WVhKamFHbDBaV04wZFhKbC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTIwMjI0MDB9fX1dfQ__&Key-Pair-Id=K2QY5QTL8JSY6C&Signature=MEYCIQCld-WgsKqrdUzdKNvtAICPtBR4~HtXfPfSv-~xIG2pDgIhAL895EVEJt31Htv4pFQ3Uyy2DUv4-CxmfAQuM4DV4Rdi)

The system is strictly divided into three decoupled layers:
1. **Layer 1 — Trust-Aware Signal Processing (`ml/trust_layer.py`)**: Assesses data credibility ($0.05 \le c_t \le 1.0$) without deleting records.
2. **Layer 2 — Personal Baseline + Trajectory (`ml/features.py`)**: Detects patient-specific deviation relative to their own admission state.
3. **Layer 3 — Sequential Evidence Accumulator (`ml/evidence_accumulator.py`)**: Filters transient spikes and enforces persistence before firing.

---

## 3. Dataset & Data Inspection

The system operates on the real **PhysioNet / Computing in Cardiology Challenge 2012 (Set A)** dataset:
- **Total Patients:** 3,200 ICU admissions with 48 hours of time-series observations.
- **Vitals & Labs:** HR, SysABP, DiasABP, MAP, RespRate, Temp, SaO2, GCS, Urine, Glucose, BUN, Creatinine, Platelets, WBC, Lactate, etc.
- **Ground Truth Target:** Acute in-hospital clinical deterioration / mortality (`In-hospital_death`: 13.84% positive class).
- **Sampling Behavior:** Irregular sampling intervals (median 428 observations per patient over 48 hours).
- **Inspection Summary:** Automatically cataloged in [`results/data_summary.json`](results/data_summary.json).

### Outcome and timestamp limits

The challenge outcomes provide in-hospital mortality, survival, scores, and length of stay, but **do not provide a timestamp for physiological collapse or death**. Consequently, “lead time” is only the time from the first alert to the available telemetry/stay endpoint. It is not time-to-event and cannot validate whether an alert preceded a clinical deterioration.

For robustness, the pipeline also computes an optional secondary label: **in-hospital death OR ICU stay under 48 hours** (`died_or_short_stay_48h`). This is not a replacement ground truth; it is a sensitivity analysis that mixes two different phenomena and should be interpreted descriptively.

---

## 4. Strict No-Leakage Guarantee

Patient data leakage renders clinical AI models useless in the real world. SilentWindow enforces two non-negotiable mathematical guarantees:

1. **Patient-Level Splitting:**
   - Observations for any given patient are isolated exclusively to one split: Train (70%), Validation (15%), or Test (15%).
   - Verified by automated unit test `test_patient_level_splitting`: zero patient ID intersection between splits.

2. **Strict Past-Only Causality ($t \le T$):**
   - At evaluation timestamp $T$, only observations with timestamp $\le T$ are visible.
   - Future data ($t > T$) is never revealed or interpolated.
   - Verified by automated unit test `test_no_future_leakage_assertion`: appending future catastrophic events at $T + 5\text{h}$ produces bit-for-bit identical features at time $T$.

---

## 5. Layer 1 — Trust-Aware Signal Processing

Rather than silently imputing or deleting noisy readings, Layer 1 generates an explicit assessment tuple for every observation:
$$\text{Assessment} = \langle \text{reading\_value}, \text{reading\_validity}, \text{credibility\_score}, \text{reason\_for\_downweighting} \rangle$$

- **Physiological Plausibility:** Checks conservative dataset percentiles (e.g. HR $\notin [30, 220]$, SysABP $\notin [40, 240]$).
- **Sudden Jump Detection:** Compares the rate of change against recent history ($\Delta v / \Delta t$). E.g., HR jumping $+50\text{ bpm}$ in 10 minutes receives a severe credibility penalty ($c_t < 0.40$).
- **Multi-Vital Discordance:** Checks cross-parameter physical consistency (e.g., SysABP $\le$ DiasABP).
- **Staleness Clock:** Tracks elapsed hours since the last measurement for each parameter (`time_since_last_measurement`) alongside explicit missingness flags.

---

## 6. Layer 2 — Personal Baseline & Trajectory

- **Personal Baseline:** Calibrated from the patient's earliest valid observations (default: first 6 hours of ICU stay).
  $$\Delta_{\text{baseline}}(t) = v(t) - \bar{v}_{\text{baseline}}$$
- **Trajectory Slopes:** Past-only rolling window statistics (1h, 3h, 6h rolling means, rolling standard deviation, and regression slope).
- **Multi-Variable Patterns:** Learns compounded shock trajectories (e.g., HR accelerating while blood pressure drops and respiratory rate increases).
- **Time outside normal range:** For each configured vital, accumulates capped exposure hours outside the dataset-aware plausible range using only observations available at time $T$.
- **Worst value so far:** Tracks the largest deviation beyond the plausible range seen so far.
- **Recent minus baseline:** Compares the recent 3-hour mean with the patient-specific baseline.
- **Static context:** Includes age, gender, weight, and ICU type from the admission record. Age and ICU type are also used for descriptive subgroup checks; they are not treated as outcomes.

---

## 7. Machine Learning Model & Calibration

- **Primary Classifier:** XGBoost tabular gradient boosting with class-weight rebalancing.
- **Probability Calibration:** Platt scaling (sigmoid calibration) fitted on validation patients, transforming raw logits into well-calibrated empirical event frequencies ($P(\text{deterioration} \mid X_t)$).
- **Validation metrics** (the model-training validation split, not the held-out replay below):
  - ROC-AUC: `0.6990`
  - PR-AUC: `0.2803`
  - Brier Score: Raw `0.1275` $\rightarrow$ Calibrated `0.1080` (significant calibration improvement)

On the **152-patient chronological test replay**, the stored evaluation artifact reports ROC-AUC `0.6407` and PR-AUC `0.2443`. These are modest discrimination results and should not be read as evidence of clinical-grade performance.

Running `python run.py --train` writes `results/calibration_curve.json` and, when Matplotlib is available, `results/calibration_plot.png`. The plot compares raw and Platt-calibrated predicted risk with observed validation frequency. The held-out evaluation also writes age and ICU-type subgroup summaries to `results/evaluation_summary.json` so performance differences are visible rather than averaged away.

---

## 8. Layer 3 — Sequential Evidence Accumulator

At chronological time step $t$:
Let $r_t$ be calibrated risk, $r_0 = 0.22$ be the neutral baseline, and $c_t$ be Layer 1 mean credibility:
$$s_t = r_t - r_0$$

When risk is elevated ($s_t > 0$):
$$\Delta E_t = s_t \times c_t \times \left[1.0 + \min(1.0, 0.25 \times (k_t - 1))\right]$$
$$E_t = E_{t-1} + \Delta E_t$$
where $k_t$ is the consecutive persistence count.

When risk normalizes ($s_t \le 0$):
$$E_t = \max\left(0.0, E_{t-1} \times \lambda - |s_t| \times 0.15\right) \quad (\lambda = 0.88)$$

### Two Operational Alert Levels
- **🟡 WATCH ($0.40 \le E_t < 0.75$):** Evidence of deterioration is rising. Visible on the Ward Monitor; does not fire an audible pager alarm.
- **🔴 ALERT ($E_t \ge 0.75$):** Accumulated evidence has crossed the high-confidence threshold.
- **Refractory Window (6.0 hours):** Suppresses repeat alarms for the same clinical episode to prevent alert fatigue.
- **Alert Budget (Max 3 alerts/patient):** Enforces alert ceiling.

---

## 9. Baseline Comparison & Evaluation Results

Evaluated across **152 unseen test patients (22 positive / 130 negative; 7,095 total monitoring hours)**. Counts below are patient-level unless explicitly labelled as alert events. “False-alert events” counts every alert emitted for a negative patient, so it can exceed the number of false-positive patients.

| Metric | Full SilentWindow | Plain ML (Direct $p \ge 0.50$) | Simplified Threshold |
| :--- | :---: | :---: | :---: |
| **Sensitivity (Recall; TP / 22)** | **13.6% (3)** | 4.5% (1) | **40.9% (9)** |
| **False-positive patients (FP / 130)** | 17 | 2 | 32 |
| **PPV / Precision (TP / alerted patients)** | **15.0%** | 33.3% | 22.0% |
| **False-alert events** | **51** | 24 | **120** |
| **Alerts / Patient-Day** | **0.203** | 0.203 | **0.504 (+148% burden)** |
| **Median proxy lead time** | **37.0 hours** | 41.0 hours | 41.0 hours |

*What these numbers actually show:*
- With only **22 positives**, the full system detects **3 patients**, while the simple threshold baseline detects **9**. The full system is therefore trading away most detection for fewer alert events; it is not a sensitivity improvement over the threshold baseline.
- The full system emits **51 false-alert events for 3 true-positive patients**. Its patient-level PPV is **15.0%**, close to the **13.8%** test-set base rate; the current result does not establish that alerts are better than a simple prevalence-informed strategy.
- SilentWindow reduces alert burden by **59.7% versus the threshold baseline** (0.203 vs 0.504 alerts/patient-day), but this comparison must be read alongside its lower sensitivity.
- The **37-hour figure is a proxy lead time**, measured from alert to the available stay/outcome endpoint—not from alert to a recorded physiological collapse. It should not be presented as validated clinical warning time.

The evaluation artifact also reports episode-oriented quantities for each system: total alerts, alerts per patient, the number and share of patients who ever alerted, median time from the start of replay to the first alert, and the distribution of first-alert times. These are alert-process metrics, not clinical event-time metrics.

---

## 10. Architectural Ablation Study

Empirical results from systematic ablation across the test cohort:

| System Configuration | Sensitivity | PPV | False-alert events | Alerts / Pt-Day | Median Lead Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A. Full SilentWindow** | **13.6%** | 15.0% | **51** | **0.203** | **37.0h** |
| **B. Without Trust Layer** | 13.6% | 15.0% | 51 | 0.203 | 37.0h |
| **C. Without Personal Baseline** | 13.6% | **75.0%** | **11** | **0.154** | 37.0h |
| **D. Without Accumulator** | 4.5% | 33.3% | 24 | 0.203 | 41.0h |

*Interpretation and limitation:*
- Removing the Evidence Accumulator changes the operating point substantially: sensitivity falls from 13.6% to 4.5%.
- The “Without Trust Layer” row is **identical to the full system** on this cohort, so this experiment provides no evidence that the Trust Layer improved these metrics.
- Removing the Personal Baseline actually produces **fewer false-alert events (11 vs 51) and higher PPV (75% vs 15%) at the same sensitivity**. This contradicts any claim that every layer improves this test result. It may reflect threshold interactions or instability from the small sample and requires further evaluation.

---

## 11. Sensor Noise Stress Lab

Tested on **30 patients** under synthetic 50% sensor corruption (missing dropouts + motion spikes + jitter). This is a small, synthetic stress test and is not evidence of real-world robustness:

```
Clean vs Corrupted Telemetry:
• Simplified Threshold: False Alerts 7 -> 16 (+128.6%)
• Plain ML (Instant):   False Alerts 0 -> 2  (+200.0% increase)
• SilentWindow:         False Alerts 3 -> 3  (+0.0% in this run)
```

The threshold baseline’s increase from 7 to 16 false-alert events is the clearest signal in this experiment. SilentWindow did not gain false-alert events in this particular run, but the sample is too small to justify “immune,” “rock solid,” or general resilience claims; its noisy sensitivity also changed from 13.6% to 27.3%.

---

## 12. SHAP Explainability & Trust Provenance

For every patient alert, the system generates:
1. **SHAP Feature Contributions:** Exact tree contributions calculated via `shap.TreeExplainer`.
2. **Clinical Decision Summary:** Bulleted synthesis explaining which vitals changed and why.
3. **Data Quality & Trust Audit Table:** Lists every downweighted observation, its measured value, credibility score, and clinical downweighting rationale.

---

## 13. Dashboard Features

The prototype includes a responsive healthcare web console (`http://localhost:8000`):
- **Page 1 — Overview:** Top-level cohort metrics, active ward state, and architectural flow.
- **Page 2 — Ward Monitor:** Real-time bed table with 🟢 Stable, 🟡 Watch, and 🔴 Alert badges, risk meters, and quick navigation.
- **Page 3 — Patient Timeline & Simulation Mode:**
  - Hour-by-hour chronological replay with **[Play]**, **[Pause]**, **[Reset]**, and **Speed controls (1x, 2x, 5x)**.
  - Strict display banner: **"Data available up to: XX:00"** (demonstrates future data is never used).
  - 3 synchronized multi-track charts (Vitals + Downweighted Artifact Markers, Calibrated Risk, Evidence Score + Watch/Alert triggers).
- **Page 4 — Alert Explanation:** Live SHAP factor waterfall and downweighted telemetry table.
- **Page 5 — Model Performance:** Interactive ROC/PR curves, confusion matrices, and lead-time trade-off chart.
- **Page 6 — Ablation Study:** Live comparative metrics table across the 4 architectural variants.
- **Page 7 — Noise Lab:** Interactive noise intensity slider (0–100%) and stress-test runner.

---

## 14. Installation & Quickstart

### Prerequisites
- Python 3.10–3.12 (the dependency lock is tested against this range)
- Web browser (Chrome, Firefox, Edge, Safari)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add the raw PhysioNet data

Place the challenge files at:

```text
data/raw/Outcomes-train.txt
data/raw/set-a/<RecordID>.txt
```

The raw challenge data is intentionally not bundled in this repository. The default configuration samples up to 1,000 patients; set `data.max_patients` to `null` in `configs/config.yaml` to use all available admissions.

### 3. One-command reproducibility path
```bash
python run.py --reproduce
```

This runs the unit tests, rebuilds the feature cache when the schema changes, trains/calibrates the model, evaluates the chronological test cohort, runs the ablation, and runs the synthetic noise test. Outputs are written to `models/` and `results/`.

### 4. Run the Full Prototype (Server & Dashboard)
```bash
python run.py
```
*The dashboard will automatically open at `http://127.0.0.1:8000`.*

### 5. Run Pipeline Components Individually
```bash
# Run Pytest Unit Test Suite (8/8 tests verifying leakage and layers)
python run.py --test

# Train XGBoost Model and Fit Probability Calibration
python run.py --train

# Run Chronological Cohort Evaluation
python run.py --evaluate

# Run Ablation Study
python run.py --ablation

# Run Sensor Noise Stress Testing
python run.py --noise-test

# Run entire pipeline end-to-end and launch server
python run.py --all
```

---

## 15. Limitations & Future Work

- **Retrospective only:** This is an offline replay on PhysioNet Challenge 2012 data, not a prospective or external validation.
- **Mortality proxy:** In-hospital mortality is not the same as a timestamped deterioration event; the secondary short-stay label has the same limitation.
- **Small held-out cohort:** The reported test set has 152 patients and 22 deaths, so patient-level sensitivity and PPV are highly unstable.
- **No clinical utility claim:** Alert burden, calibration, subgroup checks, and synthetic noise tests are descriptive and do not establish safety or benefit.
- **Future work:** Add timestamped clinical events, validate on an external ICU cohort, pre-register thresholds, report confidence intervals, expand subgroup analysis, and run a prospective silent trial with an auditable real-time stream.

---

## 16. Project Directory Structure

```
silentwindow/
├── backend/
│   ├── main.py                  # FastAPI app entry point & static file server
│   └── api/
│       └── router.py            # REST endpoints (/overview, /ward, /timeline, /explanation, /performance, /ablation, /noise-lab)
├── ml/
│   ├── data_inspection.py       # Dataset audit & summary generator
│   ├── preprocessing.py         # Ingestion & patient-level split generator
│   ├── trust_layer.py           # Layer 1: Credibility assessment & artifact downweighting
│   ├── features.py              # Layer 2: Personal baseline & past-only trajectory features
│   ├── labeling.py              # Ground-truth outcome association
│   ├── train.py                 # Model training & probability calibration
│   ├── calibration.py           # Platt sigmoid / isotonic probability calibrator
│   ├── baselines.py             # Simplified Threshold & Plain ML baselines
│   ├── evidence_accumulator.py  # Layer 3: Sequential CUSUM accumulator & refractory logic
│   ├── replay.py                # Strict chronological patient replay engine
│   ├── evaluation.py            # Chronological cohort metrics & trade-off analysis
│   ├── ablation.py              # 4-variant systematic ablation experiment
│   ├── noise_test.py            # In-memory sensor corruption & stress test engine
│   └── explainability.py        # SHAP TreeExplainer & trust provenance generator
├── frontend/
│   ├── index.html               # Modern healthcare dashboard SPA (7 pages)
│   └── app.js                   # Interactive client logic, Chart.js multi-track charts, & simulation player
├── configs/
│   └── config.yaml              # Configurable thresholds, windows, and parameters
├── tests/
│   └── test_silentwindow.py     # Rigorous unit tests (Leakage, Trust, Accumulator, Splits)
├── data/
│   ├── raw/                     # PhysioNet Challenge 2012 raw telemetry files
│   └── processed/               # Patient splits & cached feature parquets
├── models/                      # Trained model, calibrator, and feature names
├── results/                     # Evaluation summaries, ablation results, and noise test data
├── docs/                        # GitHub-friendly architecture diagram source and PNG
├── requirements.txt             # Python package dependencies
├── README.md                    # System documentation and scientific report
└── run.py                       # Unified CLI and application runner
```

---

## 17. Hackathon / Demo Walkthrough Script (3–5 Minutes)

1. **Step 1 — Overview & Ward Monitor:** Open the dashboard at `http://127.0.0.1:8000`. Show the disclaimer and review the 152 test patients in the **Ward Monitor**. Filter by `🔴 Alert` to show patients with persistent deterioration.
2. **Step 2 — Patient Timeline & Simulation:** Click on Patient `138123` or select from the dropdown. Press **[▶ Play]** in the simulation control deck. Point out the banner: `"Data available up to: XX:00"` to show that no future readings are ever used.
3. **Step 3 — Trust Downweighting:** Point to the purple diamond markers on Track 1 (Vitals). Explain: *"Here, a sudden jump artifact was received. Instead of firing an alarm, Layer 1 downweighted its credibility to 0.15."*
4. **Step 4 — Watch vs Alert:** Watch Track 3 as evidence accumulates. Show how the system transitions into **🟡 WATCH** first (providing situational awareness) before crossing the threshold to **🔴 ALERT** only after sustained physiological deviation.
5. **Step 5 — Explainability:** Switch to the **Alert Explanation** tab. Show the top SHAP features (e.g. respiratory rate trajectory and blood pressure drop) and the Layer 1 trust table explaining downweighted readings.
6. **Step 6 — Noise Lab:** Switch to the **Noise Lab**. Click **[Run Stress Test]**. Describe the threshold result (+128.6% false-alert events) as the clearest stress-test signal, and describe SilentWindow’s 0.0% change as a result from this small 30-patient run—not proof of immunity.
7. **Step 7 — Ablation & Performance:** Conclude on the **Ablation Study** and **Performance** pages. Point out that the accumulator changes sensitivity, while the trust-layer ablation is null and the personal-baseline ablation is favorable on this cohort; do not claim that every layer contributes positively.

---

## 18. AI Copilot, Supabase & Vercel

The **AI Copilot** page uses a server-side Gemini call through `POST /api/ai/ask`. Secrets are never placed in HTML or browser JavaScript. If Supabase is configured, generated question/answer pairs are best-effort persisted to `public.ai_chat_messages`; the page remains usable if persistence is not configured.

### Local configuration

```bash
cp .env.example .env
# Edit .env with a newly generated Gemini API key and the Supabase project URL/key.
```

Run [`supabase/schema.sql`](supabase/schema.sql) in the Supabase SQL editor to create the optional persistence table and its insert-only RLS policy. The database connection string is not needed by this app and must never contain a placeholder password in deployment configuration.

### Vercel configuration

The repository includes [`vercel.json`](vercel.json) and [`api/index.py`](api/index.py). In Vercel Project Settings → Environment Variables, add:

- `GEMINI_API_KEY`
- `GEMINI_MODEL` (optional; defaults to `gemini-2.0-flash`)
- `SUPABASE_URL`
- `SUPABASE_PUBLISHABLE_KEY`

Then deploy with the Vercel CLI or by importing the GitHub repository. Because the Gemini key was pasted into a chat message, rotate it and use the replacement key in Vercel rather than reusing the exposed key.
