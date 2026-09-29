"""
train.py - Leakage-Free Model Training and Probability Calibration.

Trains XGBoost deterioration classifier on patient-level training split,
calibrates risk estimates on patient-level validation split,
and evaluates discrimination and calibration metrics.
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
        df_labeled = labeler.assign_hourly_labels(df_p, pid)
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

    train_cache = os.path.join(processed_dir, "train_features.parquet")
    val_cache = os.path.join(processed_dir, "val_features.parquet")

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

    meta_cols = {"patient_id", "eval_time", "target", "sofa", "saps", "length_of_stay", "survival"}
    feature_cols = [c for c in df_train.columns if c not in meta_cols]

    X_train = df_train[feature_cols].copy()
    y_train = df_train["target"].values
    X_val = df_val[feature_cols].copy()
    y_val = df_val["target"].values

    print(f"Dataset shapes: Train={X_train.shape} (Pos={y_train.sum()}), Val={X_val.shape} (Pos={y_val.sum()})")

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
