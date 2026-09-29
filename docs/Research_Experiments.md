# SilentWindow research experiments

This document records the five requested modeling tracks. The experiments are **non-destructive**: they write research outputs under `results/research/` and never replace the checked-in files under `models/` or change the deployed runtime configuration.

## Run everything

```bash
python -m experiments.run_research
# or
make research
```

The orchestrator writes:

| Track | Result |
|---|---|
| Expanded alert-policy sweep | Runs against `results/test_eval_records.parquet` already included in the repository |
| Class-weight and XGBoost tuning | Requires raw data or cached train/validation features |
| Feature-group and top-k ablations | Requires raw data or cached train/validation features |
| Future-window labels | Requires raw patient records and outcomes |
| Optional XGBoost/logistic/LightGBM ensemble | Requires raw data or cached train/validation features |

## 1. Expanded policy sweep

`ml/policy_sweep.py` evaluates the Cartesian product of:

- Watch thresholds: `0.30, 0.35, 0.40, 0.45`
- Alert thresholds: `0.55, 0.60, 0.65, 0.70, 0.75`
- Evidence decay: `0.80, 0.85, 0.88, 0.92, 0.95`
- Persistence boost: `1.00, 1.10, 1.25, 1.40, 1.60`

Invalid policies where the watch threshold is greater than or equal to the alert threshold are excluded. The output also computes a simple non-dominated frontier using sensitivity, precision, and false-alert events.

The current checked-in replay artifact produced **500 evaluated policies**. The current configuration remains:

| Watch | Alert | Decay | Persistence | Sensitivity | Precision | False-alert events | Alerts/patient-day |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.75 | 0.88 | 1.25 | 13.64% | 15.00% | 51 | 0.203 |

One low-burden frontier point in this retrospective cohort used watch `0.30`, alert `0.75`, decay `0.85`, and persistence `1.00`; it produced sensitivity `13.64%`, precision `15.00%`, 48 false-alert events, and 0.186 alerts per patient-day. This is an exploratory operating point, **not a recommendation for clinical deployment**.

Output: [`results/research/policy_sweep_expanded.json`](../results/research/policy_sweep_expanded.json).

## 2. Class-weight and XGBoost tuning

`experiments/tune_xgboost.py` evaluates deterministic combinations of depth, learning rate, estimators, child weight, subsampling, column subsampling, gamma, and `scale_pos_weight` values including `3.0`, `5.0`, `6.0`, and the empirical training ratio. Ranking uses PR-AUC first, then recall at a documented probability threshold; accuracy is not the optimization target.

No tuning conclusion is reported in this checkout because `data/raw/` is intentionally absent. The output records the missing prerequisites rather than fabricating a benchmark.

## 3. Feature and ablation experiments

`experiments/feature_ablation.py` compares the full feature set against:

- Without GCS features
- Without ICU type
- Vitals and trajectory features without static context
- Vitals-only features
- Top 30, 50, 75, 100, and all ranked features

Each variant is retrained on the training split and scored on validation data. The production model is not overwritten.

## 4. Future-window labels

`ml/labeling.py` now provides `assign_future_window_proxy_labels`. The experiment evaluates 6-, 12-, 18-, and 24-hour windows using the available telemetry/outcome endpoint as an explicit proxy.

This is not a true deterioration label: the challenge data does not provide an exact physiological collapse or death timestamp. The generated field is therefore named `future_window_proxy_target` and carries a disclaimer in every labeled frame.

## 5. Optional ensemble

`experiments/ensemble.py` benchmarks XGBoost and class-balanced logistic regression, and uses LightGBM when it is installed. The default blend is `0.6 XGBoost + 0.4 logistic regression`; when LightGBM is available it uses `0.5 XGBoost + 0.3 LightGBM + 0.2 logistic regression`. These weights are benchmark defaults, not tuned clinical operating points.

## Outputs and interpretation

The manifest is [`results/research/experiment_manifest.json`](../results/research/experiment_manifest.json). In the current checkout, the policy sweep is completed and the four raw-data-dependent tracks are marked `blocked_missing_data`.

To run the blocked tracks, obtain the dataset under its own terms and place it at:

```text
data/raw/Outcomes-train.txt
data/raw/set-a/<RecordID>.txt
```

Then run `make research`. Review the resulting validation and locked-test metrics before considering any model or policy promotion. No research artifact should be described as prospective or clinically validated.
