"""
explainability.py - Trust-Aware Clinical Decision Support & SHAP Explanation Engine.

Generates:
1. Exact SHAP feature contributions from the trained tree model.
2. Human-interpretable clinical trajectory factor mappings.
3. Trust layer breakdown: down-weighted readings, credibility scores, and physical audit logs.
4. Synthesizes a factual, data-driven alert explanation summary.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

import shap
from ml.preprocessing import load_config, load_patient_raw
from ml.trust_layer import TrustAwareSignalProcessor
from ml.features import FeatureExtractor

FEATURE_DISPLAY_MAP = {
    "HR_dev_from_baseline": "Heart rate deviation from personal baseline",
    "SysABP_dev_from_baseline": "Systolic blood pressure drop from personal baseline",
    "RespRate_dev_from_baseline": "Respiratory rate elevation from personal baseline",
    "shock_index": "Elevated Shock Index (HR / SysABP)",
    "multi_var_deterioration_signal": "Multi-vital composite deterioration pattern",
    "RespRate_slope_3h": "Upward trajectory in Respiratory Rate over past 3h",
    "SysABP_slope_3h": "Declining trajectory in Systolic Blood Pressure over past 3h",
    "HR_slope_3h": "Accelerating Heart Rate over past 3h",
    "Temp_dev_from_baseline": "Temperature deviation from baseline",
    "GCS_last_value": "Depressed neurological status (Glasgow Coma Scale)",
    "Urine_last_value": "Diminished urine output / oliguria indicator",
    "SaO2_last_value": "Oxygen desaturation (SaO2)",
    "mean_recent_credibility": "Overall vital telemetry trust score",
    "HR_rolling_std_3h": "Elevated heart rate variability",
    "RespRate_rolling_mean_3h": "Sustained tachypnea (elevated 3h mean RR)"
}

class ClinicalExplainer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.models_dir = "models"
        
        model_path = os.path.join(self.models_dir, "xgboost_model.joblib")
        features_path = os.path.join(self.models_dir, "feature_names.json")
        calib_path = os.path.join(self.models_dir, "calibrator.joblib")

        self.model = joblib.load(model_path)
        self.calibrator = joblib.load(calib_path)
        with open(features_path, "r") as f:
            self.feature_cols = json.load(f)

        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)
        self.trust_proc = TrustAwareSignalProcessor(self.config)
        self.extractor = FeatureExtractor(self.config)

    def explain_patient_at_time(
        self,
        patient_id: int,
        eval_time_hours: float
    ) -> Dict[str, Any]:
        """
        Generates full SHAP and Trust Layer explanation for a patient at specific hour.
        Strictly uses only data revealed up to eval_time_hours.
        """
        records_dir = self.config["data"]["records_dir"]
        filepath = os.path.join(records_dir, f"{patient_id}.txt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Patient file not found: {filepath}")

        static_info, df_obs_full = load_patient_raw(filepath)
        
        # 1. Reveal data <= eval_time_hours
        df_revealed = df_obs_full[df_obs_full["time_hours"] <= eval_time_hours].copy()

        # 2. Layer 1: Trust assessment
        assessments = self.trust_proc.process_patient_stream(df_revealed)
        suspicious_list = []
        for a in assessments:
            if a.credibility_score < 0.70:
                suspicious_list.append({
                    "time_hours": round(a.time_hours, 2),
                    "parameter": a.parameter,
                    "value": round(a.original_value, 2),
                    "credibility_score": round(a.credibility_score, 3),
                    "reason": a.reason or "Downweighted telemetry reading"
                })

        # 3. Layer 2: Feature extraction
        features = self.extractor.extract_patient_features_at_time(
            eval_time_hours, assessments, static_info, patient_id
        )
        feat_df = pd.DataFrame([features])[self.feature_cols]

        # 4. Model prediction
        raw_prob = float(self.model.predict_proba(feat_df)[:, 1][0])
        calib_prob = float(self.calibrator.predict_calibrated(np.array([raw_prob]))[0])

        # 5. SHAP values
        shap_values = self.explainer.shap_values(feat_df)
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0]

        # Rank features by positive contribution towards deterioration
        ranked_indices = np.argsort(sv)[::-1]
        top_factors = []
        for idx in ranked_indices:
            feat_name = self.feature_cols[idx]
            val = float(feat_df[feat_name].iloc[0])
            contrib = float(sv[idx])
            
            # Skip missing or negative contribution factors for alert justification
            if contrib <= 0.001 or np.isnan(val):
                continue

            display_name = FEATURE_DISPLAY_MAP.get(
                feat_name,
                feat_name.replace("_", " ").title()
            )

            top_factors.append({
                "feature": feat_name,
                "display_name": display_name,
                "current_value": round(val, 2),
                "shap_impact": round(contrib, 4),
                "direction": "increases risk" if contrib > 0 else "reduces risk"
            })
            if len(top_factors) >= 6:
                break

        # Generate structured factual bullets
        bullet_points = []
        for tf in top_factors[:4]:
            bullet_points.append(f"• {tf['display_name']} ({tf['current_value']}) actively increases deterioration risk")

        if len(suspicious_list) > 0:
            bullet_points.append(f"• {len(suspicious_list)} telemetry artifact(s) downweighted by Trust Layer (preventing false alarms)")
        else:
            bullet_points.append("• Telemetry stream verified consistent and physiologically credible")

        total_measurements = len(df_revealed)
        missing_count = sum(1 for v in ["HR", "SysABP", "RespRate", "Temp", "SaO2", "GCS"] if features.get(f"{v}_is_missing", 0) == 1)

        return {
            "patient_id": patient_id,
            "eval_time_hours": round(eval_time_hours, 2),
            "calibrated_risk": round(calib_prob, 4),
            "raw_risk": round(raw_prob, 4),
            "top_factors": top_factors,
            "summary_bullets": bullet_points,
            "data_quality": {
                "total_measurements_processed": total_measurements,
                "suspicious_readings_count": len(suspicious_list),
                "missing_vital_parameters": missing_count,
                "overall_stream_credibility": round(float(features.get("mean_recent_credibility", 1.0)), 3)
            },
            "suspicious_readings": suspicious_list[::-1][:15]  # most recent first
        }
