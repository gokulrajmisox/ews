# SilentWindow Model Improvement Experiment

**Date:** 2026-09-29  
**Status:** Offline benchmark only; not deployed

## Summary

A new XGBoost benchmark was trained on the recoverable derived ICU feature table using **patient-grouped train/validation/test splits**. Outcome-related fields (`Survival`, `SAPS-I`, `SOFA`, `Length_of_stay`) and identifiers were excluded from the features. The alert threshold was selected on validation patients only.

This is a stronger benchmark than the current deployed artifact, but it is **not yet a drop-in replacement** because the deployed application uses a different 127-feature `FeatureExtractor` pipeline and the original raw patient files are no longer available in the repository.

## Results

| Evaluation | Current saved model | Grouped retraining benchmark |
|---|---:|---:|
| Patient-level ROC-AUC | 0.641 | **0.819** |
| Patient-level PR-AUC | 0.244 | **0.371** |
| Accuracy at selected threshold | 76.3% | **78.3%** |
| Precision | 15.0% | **35.2%** |
| Recall / sensitivity | 13.6% | **68.2%** |
| F1 | 0.143 | **0.464** |

The benchmark’s threshold was **0.51**, selected using the validation cohort to maximize F1. On a more realistic first-24-hour window, the benchmark achieved:

- ROC-AUC: **0.793**
- PR-AUC: **0.330**
- Accuracy: **82.5%**
- Precision: **39.5%**
- Recall: **51.5%**
- F1: **0.447**

## Method

- Data: recoverable `final_ml_dataset.csv` from the initial repository commit.
- Patients: 3,197 total; 443 positive patients.
- Split: 70% train / 15% validation / 15% test by patient identifier, preventing rows from the same patient appearing across splits.
- Model: XGBoost histogram training, 350 estimators, depth 4, learning rate 0.04, subsampling 0.85, column subsampling 0.85, `min_child_weight=4`, `reg_lambda=2`, and class weight 1.8.
- Missing values: median imputation with missingness indicators.
- Evaluation: patient-level maximum predicted risk; threshold selected on validation only.

## Important limitation

The original raw files expected by `ml/train.py` (`data/raw/set-a/*.txt` and `Outcomes-train.txt`) are absent. The existing production model uses a different feature schema and cannot safely load this benchmark artifact without an adapter and a full chronological validation.

Therefore, the benchmark has **not** been copied into `models/`, committed as the production model, or deployed.

## Recommended next step

Restore the original raw patient files or provide the current live telemetry schema. Then:

1. Rebuild the existing chronological feature pipeline.
2. Add the improved regularization and class-weight settings.
3. Select the alert threshold on validation patients only.
4. Evaluate on a locked temporal test set, especially a first-24-hour window.
5. Run prospective shadow mode before enabling clinical alerts.
