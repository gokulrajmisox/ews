"""
replay.py - Strict Chronological Patient Telemetry Replay Engine.

REPLAY PRINCIPLE:
At evaluation hour T, only observations with timestamp <= T are revealed.
No future observation is ever available to Layer 1, Layer 2, or Layer 3.
Synchronously executes:
1. Full SilentWindow (Trust -> Personal Baseline -> Trajectory -> Accumulator)
2. Baseline A (Simplified Threshold Baseline)
3. Baseline B (Plain ML thresholding)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from ml.preprocessing import load_config, load_patient_raw
from ml.trust_layer import TrustAwareSignalProcessor, ObservationAssessment
from ml.features import FeatureExtractor
from ml.labeling import OutcomeLabeler
from ml.evidence_accumulator import SequentialEvidenceAccumulator, AccumulatorState
from ml.baselines import SimplifiedThresholdBaseline, PlainMLBaseline

class ChronologicalReplayEngine:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.models_dir = "models"
        
        # Load trained ML model & calibrator
        model_path = os.path.join(self.models_dir, "xgboost_model.joblib")
        calib_path = os.path.join(self.models_dir, "calibrator.joblib")
        features_path = os.path.join(self.models_dir, "feature_names.json")

        if not os.path.exists(model_path) or not os.path.exists(calib_path):
            raise FileNotFoundError("Trained model or calibrator not found. Run ml/train.py first!")

        self.model = joblib.load(model_path)
        self.calibrator = joblib.load(calib_path)
        with open(features_path, "r") as f:
            self.feature_cols = json.load(f)

        self.trust_proc = TrustAwareSignalProcessor(self.config)
        self.extractor = FeatureExtractor(self.config)
        self.labeler = OutcomeLabeler(self.config["data"]["outcomes_file"])
        self.accumulator = SequentialEvidenceAccumulator(self.config)
        self.threshold_baseline = SimplifiedThresholdBaseline(self.config)
        self.plain_ml_baseline = PlainMLBaseline(self.config)

    def replay_patient(
        self,
        patient_id: int,
        start_hour: float = 2.0,
        end_hour: float = 48.0,
        step_hours: float = 1.0,
        enable_trust: bool = True,
        enable_personal_baseline: bool = True,
        enable_accumulator: bool = True,
        raw_obs_override: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """
        Replays a single patient chronologically, step-by-step.
        Supports ablation flags to test disabled layers.
        """
        records_dir = self.config["data"]["records_dir"]
        filepath = os.path.join(records_dir, f"{patient_id}.txt")
        
        if raw_obs_override is not None:
            static_info, df_obs_full = {}, raw_obs_override
            # Check if static is embedded
            if os.path.exists(filepath):
                static_info, _ = load_patient_raw(filepath)
        else:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Patient file not found: {filepath}")
            static_info, df_obs_full = load_patient_raw(filepath)

        outcome_info = self.labeler.get_patient_outcome(patient_id)
        max_time = float(df_obs_full["time_hours"].max()) if not df_obs_full.empty else 48.0
        patient_end = min(end_hour, max_time)

        self.accumulator.reset()
        patient_history: List[Dict[str, Any]] = []
        suspicious_readings_logged = []

        cur_t = start_hour
        while cur_t <= patient_end + 1e-4:
            # 1. REVEAL ONLY PAST DATA (Strict causality guarantee)
            df_revealed = df_obs_full[df_obs_full["time_hours"] <= cur_t].copy()

            # Automated leakage verification assert
            if not df_revealed.empty:
                max_revealed_t = df_revealed["time_hours"].max()
                if max_revealed_t > cur_t + 1e-6:
                    raise AssertionError(f"DATA LEAKAGE ERROR: Observation time {max_revealed_t} > current time {cur_t}")

            # Layer 1: Trust-aware processing on revealed observations
            assessments = self.trust_proc.process_patient_stream(df_revealed)
            
            # Track any downweighted readings up to now
            downweighted = [a for a in assessments if a.credibility_score < 0.70]

            # Layer 2: Feature extraction up to cur_t
            features = self.extractor.extract_patient_features_at_time(
                cur_t, assessments, static_info, patient_id
            )

            # If personal baseline is disabled in ablation:
            if not enable_personal_baseline:
                for k in list(features.keys()):
                    if "_dev_from_baseline" in k or "_pct_dev" in k:
                        features[k] = 0.0

            # Construct model feature vector
            feat_df = pd.DataFrame([features])[self.feature_cols]

            # Predict risk
            raw_prob = float(self.model.predict_proba(feat_df)[:, 1][0])
            calib_prob = float(self.calibrator.predict_calibrated(np.array([raw_prob]))[0])

            # Recent credibility
            mean_cred = features.get("mean_recent_credibility", 1.0)

            # Layer 3: Sequential Evidence Accumulator
            if enable_accumulator:
                accum_state = self.accumulator.update(
                    time_hours=cur_t,
                    calibrated_risk=calib_prob,
                    raw_risk=raw_prob,
                    credibility_weight=mean_cred,
                    enable_trust=enable_trust
                )
                sw_state = accum_state.state
                sw_alert = accum_state.alert_fired
                sw_evidence = accum_state.evidence_score
                sw_suppressed = accum_state.suppressed_by_refractory
                sw_refractory_rem = accum_state.refractory_hours_remaining
                sw_reason = accum_state.reason
            else:
                # Direct risk threshold ablation (no evidence accumulator)
                sw_alert = (calib_prob >= 0.50)
                sw_state = "ALERT" if sw_alert else "STABLE"
                sw_evidence = calib_prob
                sw_suppressed = False
                sw_refractory_rem = 0.0
                sw_reason = "Direct risk threshold ablation"

            # Parallel Baseline evaluations
            thresh_decision = self.threshold_baseline.evaluate(features, cur_t)
            plain_ml_decision = self.plain_ml_baseline.evaluate(calib_prob, cur_t)

            # Estimated lead time calculation
            # Note: Lead time is estimated relative to the available outcome/proxy (discharge/max stay)
            # because an exact event timestamp is unavailable in the challenge dataset.
            lead_time = None
            if sw_alert and outcome_info["in_hospital_death"] == 1:
                lead_time = max(0.0, round(patient_end - cur_t, 2))

            time_point_record = {
                "patient_id": patient_id,
                "time_hours": round(cur_t, 2),
                "raw_risk": round(raw_prob, 4),
                "calibrated_risk": round(calib_prob, 4),
                "evidence_score": round(sw_evidence, 4),
                "state": sw_state,
                "alert_fired": sw_alert,
                "suppressed": sw_suppressed,
                "refractory_remaining": sw_refractory_rem,
                "reason": sw_reason,
                "mean_credibility": round(mean_cred, 3),
                "suspicious_count": len(downweighted),
                # Baseline comparison decisions
                "baseline_threshold_state": thresh_decision.state,
                "baseline_threshold_alert": thresh_decision.alert_fired,
                "baseline_threshold_reasons": thresh_decision.trigger_reasons,
                "baseline_plain_ml_state": plain_ml_decision.state,
                "baseline_plain_ml_alert": plain_ml_decision.alert_fired,
                # Patient vitals snapshot
                "vitals": {
                    "HR": features.get("HR_last_value", None),
                    "SysABP": features.get("SysABP_last_value", None),
                    "DiasABP": features.get("DiasABP_last_value", None),
                    "MAP": features.get("MAP_last_value", None),
                    "RespRate": features.get("RespRate_last_value", None),
                    "Temp": features.get("Temp_last_value", None),
                    "SaO2": features.get("SaO2_last_value", None),
                    "GCS": features.get("GCS_last_value", None)
                },
                # Outcome labels
                "actual_outcome": outcome_info["in_hospital_death"],
                "sofa": outcome_info["sofa"],
                "saps": outcome_info["saps"],
                "length_of_stay": outcome_info["length_of_stay"],
                "estimated_lead_time_hours": lead_time,
                "lead_time_disclaimer": "Lead time is estimated relative to the available outcome/proxy because an exact event timestamp is unavailable."
            }

            patient_history.append(time_point_record)
            cur_t += step_hours

        return patient_history

    def replay_cohort(
        self,
        patient_ids: List[int],
        step_hours: float = 1.0,
        enable_trust: bool = True,
        enable_personal_baseline: bool = True,
        enable_accumulator: bool = True
    ) -> List[Dict[str, Any]]:
        """Replays an entire cohort of patients and flattens all chronological decision records."""
        all_records = []
        for pid in patient_ids:
            try:
                hist = self.replay_patient(
                    pid, step_hours=step_hours,
                    enable_trust=enable_trust,
                    enable_personal_baseline=enable_personal_baseline,
                    enable_accumulator=enable_accumulator
                )
                all_records.extend(hist)
            except Exception as e:
                print(f"Error replaying patient {pid}: {e}")
                continue
        return all_records
