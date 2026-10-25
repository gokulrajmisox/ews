"""
train.py - Leakage-Free Model Training and Probability Calibration.

Trains XGBoost deterioration classifier on patient-level training split,
calibrates risk estimates on patient-level validation split,
and evaluates discrimination and calibration metrics.

Training target (configs/config.yaml, `labeling` section): the default is
the future-window proxy label — an hourly row is positive only when the
patient eventually dies in hospital AND the end-of-telemetry endpoint falls
within `horizon_hours` after the current eval time (a near-term early-warning
target; exact event timestamps are unavailable in the challenge dataset).
Set `labeling.target: eventual_outcome` to train on the legacy label (final
in-hospital outcome broadcast to every hourly row) for comparison.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

from ml.preprocessing import load_config, get_patient_splits, load_patient_raw
from ml.trust_layer import TrustAwareSignalProcessor
from ml.features import FeatureExtractor
from ml.labeling import OutcomeLabeler
from ml.calibration import ModelCalibrator

def get_labeling_config(config: dict) -> Tuple[str, float]:
    """Read the training-target configuration.

    Returns (target_kind, horizon_hours). Defaults to the future-window proxy
    label with a 12-hour horizon; "eventual_outcome" restores the legacy
    final-outcome-per-row label.
    """
    labeling_cfg = config.get("labeling", {})
    target_kind = labeling_cfg.get("target", "future_window_proxy")
    horizon_hours = float(labeling_cfg.get("horizon_hours", 12.0))
    return target_kind, horizon_hours

def build_split_dataset(
    patient_ids: List[int],
    config: dict,
    records_dir: str,
    trust_proc: TrustAwareSignalProcessor,
    extractor: FeatureExtractor,
    labeler: OutcomeLabeler,
    step_hours: float = 2.0
) -> pd.DataFrame:
    """Builds tabular feature matrix for a list of patient IDs."""
    target_kind, horizon_hours = get_labeling_config(config)
    all_dfs = []
    for pid in patient_ids:
        filepath = os.path.join(records_dir, f"{pid}.txt")
        if not os.path.exists(filepath):
            continue
        static_info, df_obs = load_patient_raw(filepath)
        assessments = trust_proc.process_patient_stream(df_obs)
        df_p = extractor.generate_patient_chronological_features(
            assessments, static_info, pid, start_hour=4.0, end_hour=48.0, step_hours=step_hours
        )
        # End-of-telemetry endpoint proxy for the future-window label — the
        # same convention as replay_patient (ml/replay.py): the configured
        # max horizon capped by the patient's last observed hour.
        max_time = float(df_obs["time_hours"].max()) if not df_obs.empty else 48.0
        event_proxy_hours = min(48.0, max_time)
        df_labeled = labeler.assign_training_labels(
            df_p, pid,
            target_kind=target_kind,
            horizon_hours=horizon_hours,
            event_proxy_hours=event_proxy_hours,
        )
        all_dfs.append(df_labeled)

    if not all_dfs:
        raise ValueError("No data extracted for split!")
    return pd.concat(all_dfs, ignore_index=True)

def train_and_calibrate(config_path: str = "configs/config.yaml") -> Dict:
    config = load_config(config_path)
    records_dir = config["data"]["records_dir"]
    models_dir = "models"
    processed_dir = config["data"]["processed_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    splits_file = os.path.join(processed_dir, "splits.json")
    if os.path.exists(splits_file):
        with open(splits_file, "r") as f:
            splits = json.load(f)
        train_ids = splits["train"]
        val_ids = splits["val"]
        test_ids = splits["test"]
    else:
        train_ids, val_ids, test_ids, _ = get_patient_splits(config)

    trust_proc = TrustAwareSignalProcessor(config)
    extractor = FeatureExtractor(config)
    labeler = OutcomeLabeler(config["data"]["outcomes_file"])

    target_kind, horizon_hours = get_labeling_config(config)
    # Cache filenames encode the target definition so a config change never
    # silently reuses feature frames labeled under a different target.
    target_suffix = f"{target_kind}_h{int(horizon_hours)}"
    train_cache = os.path.join(processed_dir, f"train_features_{target_suffix}.parquet")
    val_cache = os.path.join(processed_dir, f"val_features_{target_suffix}.parquet")

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print("Loading cached train and validation feature sets...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
    else:
        print(f"Building training features for {len(train_ids)} patients...")
        df_train = build_split_dataset(train_ids, config, records_dir, trust_proc, extractor, labeler, step_hours=2.0)
        df_train.to_parquet(train_cache, index=False)
        print(f"Building validation features for {len(val_ids)} patients...")
        df_val = build_split_dataset(val_ids, config, records_dir, trust_proc, extractor, labeler, step_hours=2.0)
        df_val.to_parquet(val_cache, index=False)

    meta_cols = {
        "patient_id", "eval_time", "target", "sofa", "saps", "length_of_stay", "survival",
        # Future-window proxy label metadata — never model features (the
        # disclaimer column is a string and must stay out of XGBoost input).
        "future_window_proxy_target", "future_window_horizon_hours",
        "future_window_proxy_disclaimer",
    }
    feature_cols = [c for c in df_train.columns if c not in meta_cols]

    X_train = df_train[feature_cols].copy()
    y_train = df_train["target"].values
    X_val = df_val[feature_cols].copy()
    y_val = df_val["target"].values

    print(f"Dataset shapes: Train={X_train.shape} (Pos={y_train.sum()}), Val={X_val.shape} (Pos={y_val.sum()})")
    print(f"Training target: {target_kind}"
          + (f" (horizon {horizon_hours}h, end-of-telemetry endpoint proxy)"
             if target_kind == "future_window_proxy" else ""))

    # Train XGBoost
    import xgboost as xgb
    xgb_params = config["model"].get("xgboost_params", {})
    pos_weight = float((len(y_train) - y_train.sum()) / max(1, y_train.sum()))
    scale_pos = xgb_params.get("scale_pos_weight", min(pos_weight, 4.0))

    model = xgb.XGBClassifier(
        max_depth=xgb_params.get("max_depth", 4),
        learning_rate=xgb_params.get("learning_rate", 0.05),
        n_estimators=xgb_params.get("n_estimators", 150),
        subsample=xgb_params.get("subsample", 0.85),
        colsample_bytree=xgb_params.get("colsample_bytree", 0.85),
        scale_pos_weight=scale_pos,
        random_state=xgb_params.get("random_state", 42),
        eval_metric="logloss"
    )

    print("Fitting XGBoost model...")
    model.fit(X_train, y_train)

    # Raw predictions on validation
    raw_val_probs = model.predict_proba(X_val)[:, 1]

    # Fit probability calibrator
    calib_method = config["model"].get("calibration_method", "sigmoid")
    calibrator = ModelCalibrator(method=calib_method)
    calibrator.fit(raw_val_probs, y_val)
    calib_val_probs = calibrator.predict_calibrated(raw_val_probs)

    # Compute discrimination and calibration metrics
    val_roc = float(roc_auc_score(y_val, raw_val_probs))
    val_pr = float(average_precision_score(y_val, raw_val_probs))
    val_brier_raw = float(brier_score_loss(y_val, raw_val_probs))
    val_brier_cal = float(brier_score_loss(y_val, calib_val_probs))
    val_logloss = float(log_loss(y_val, calib_val_probs))

    print(f"Validation Metrics:\n  ROC-AUC: {val_roc:.4f}\n  PR-AUC: {val_pr:.4f}")
    print(f"  Brier Raw: {val_brier_raw:.4f} -> Calibrated: {val_brier_cal:.4f}\n  Log Loss: {val_logloss:.4f}")

    # Feature importances
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    top_features = [{"feature": feature_cols[i], "importance": float(importances[i])} for i in top_indices]

    # Save artifacts
    model_path = os.path.join(models_dir, "xgboost_model.joblib")
    calibrator_path = os.path.join(models_dir, "calibrator.joblib")
    features_path = os.path.join(models_dir, "feature_names.json")
    metrics_path = os.path.join(models_dir, "metrics.json")

    joblib.dump(model, model_path)
    joblib.dump(calibrator, calibrator_path)
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    results_summary = {
        "model_type": "xgboost",
        "calibration_method": calib_method,
        "target_definition": {
            "kind": target_kind,
            "horizon_hours": horizon_hours,
            "endpoint_proxy": "end_of_telemetry",
            "disclaimer": (
                "Rows are positive only when the patient dies in hospital AND the "
                "end-of-telemetry endpoint falls within the horizon after the "
                "current eval time. An exact deterioration/death timestamp is "
                "unavailable in the challenge dataset."
            ),
        },
        "n_features": len(feature_cols),
        "validation_metrics": {
            "roc_auc": round(val_roc, 4),
            "pr_auc": round(val_pr, 4),
            "brier_score_raw": round(val_brier_raw, 4),
            "brier_score_calibrated": round(val_brier_cal, 4),
            "log_loss": round(val_logloss, 4)
        },
        "top_features": top_features
    }

    with open(metrics_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("Model and calibration artifacts successfully saved to models/")
    return results_summary

if __name__ == "__main__":
    train_and_calibrate()
