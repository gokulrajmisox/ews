# SilentWindow Model Accuracy and Real-Time Readiness

**Generated:** 2026-09-29  
**Repository commit:** `cc20b28dc43228ca7587228dfdaf2ffa0bbc674b`

## Executive conclusion

The model is **partially validated offline but is not validated on real-time clinical data**. On the available patient-level held-out cohort, the calibrated risk score has **ROC-AUC 0.641** and **PR-AUC 0.244**. At the current SilentWindow alert policy, accuracy is **76.3%**, but sensitivity is only **13.6%** and precision is **15.0%**.

The deployed production URL currently returns `FUNCTION_INVOCATION_FAILED`, and the API code exposes cached retrospective endpoints rather than a POST/WebSocket telemetry-ingestion endpoint. Therefore, a genuine live-data accuracy check cannot yet be performed.

> This is a research prototype using retrospective data. It must not be used for clinical decision-making.

## Offline held-out evaluation

- **Cohort:** 152 patients; 22 positive outcomes and 130 negative outcomes.
- **Time points:** 6,943 chronological records across the saved test evaluation.
- **Monitoring horizon:** up to 48 hours per patient.
- **Validation metrics saved during training:** ROC-AUC `0.699`, PR-AUC `0.2803`, raw Brier `0.1275`, calibrated Brier `0.1080`, log loss `0.3638`.
- **Independent reconstruction from saved test records:** ROC-AUC `0.6407`, PR-AUC `0.2443`, Brier `0.1268`, log loss `0.4132`.

### Alert-policy metrics

| Policy | Accuracy | Precision | Recall / sensitivity | F1 | Confusion matrix (TN, FP, FN, TP) |
|---|---:|---:|---:|---:|---|
| SilentWindow | 76.3% | 15.0% | 13.6% | 0.143 | `[113, 17, 19, 3]` |
| Plain ML | 84.9% | 33.3% | 4.5% | 0.080 | `[128, 2, 21, 1]` |
| Threshold baseline | 70.4% | 22.0% | 40.9% | 0.286 | `[98, 32, 13, 9]` |

Accuracy is not sufficient here because the outcome prevalence is only **14.5%**. The SilentWindow policy misses 19 of 22 positive patients in this saved test cohort.

## Calibration check

The independent patient-level calibration bins were:

| Risk bin | Patients | Mean predicted risk | Observed positive rate |
|---|---:|---:|---:|
| (0.0248, 0.0927] | 31 | 0.066 | 0.065 |
| (0.0927, 0.136] | 30 | 0.115 | 0.100 |
| (0.136, 0.205] | 30 | 0.171 | 0.133 |
| (0.205, 0.258] | 30 | 0.228 | 0.200 |
| (0.258, 0.564] | 31 | 0.361 | 0.226 |

The highest-risk bin averaged **0.361** predicted risk but observed **0.226**, indicating overprediction in this small test cohort. Calibration needs a larger independent temporal validation set.

## Robustness under corrupted streams

The saved stress test injects 50% intensity missingness, spikes, and timestamp jitter into 30 evaluated patients. It is a robustness experiment, not real-time validation:

- SilentWindow sensitivity changed from **13.64%** clean to **27.27%** noisy; false alerts stayed at **3** in the saved run.
- Plain ML sensitivity changed from **4.55%** to **0%**; noisy false alerts increased from **0** to **2**.
- Threshold baseline sensitivity changed from **40.91%** to **81.82%**, while false alarms increased from **7** to **16**.

These results show behavior under injected perturbations, not performance on naturally arriving hospital telemetry.

## Real-time readiness findings

1. **No live input route:** `backend/api/router.py` defines cached GET routes (`/overview`, `/ward`, `/patient/...`, `/performance`, `/noise-lab`) and no POST/WebSocket endpoint for new vital signs.
2. **No live ground truth:** Accuracy requires timestamped outcomes or labels after the prediction window; none is connected to the deployed service.
3. **Production health issue:** `https://ews-brown.vercel.app` currently returns HTTP 500 / `FUNCTION_INVOCATION_FAILED`, so live scoring cannot be exercised.
4. **Retrospective-only disclaimer:** The saved evaluation explicitly states that it uses retrospective data and estimated lead time.

## What is needed for a real-time accuracy check

- Add a versioned telemetry ingestion endpoint, for example `POST /api/v1/patients/{patient_id}/observations`, with schema validation, timestamps, units, and missing-value handling.
- Persist patient state between events; the current Vercel function is not a durable stream processor.
- Add an authenticated source of outcome labels and define an evaluation window (for example, deterioration within 24 hours).
- Log each prediction with model version, feature timestamp, risk, alert state, and data-quality flags.
- Run prospective shadow mode first; calculate sensitivity, specificity, precision, PR-AUC, calibration, alert burden, latency, and subgroup metrics after labels mature.
- Fix the production function invocation error before accepting any real telemetry.
