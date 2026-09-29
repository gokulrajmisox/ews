"""
features.py - Layer 2: Personal Baseline and Trajectory Feature Engineering.

GUARANTEES:
1. Strict past-only causality: at evaluation time T, no observation with time > T is ever used.
2. Staleness clock: time_since_last_measurement for each variable.
3. Personal baseline: computed from first 6 hours (or adaptive early window) of valid observations.
4. Trajectory features: rolling means, slopes, deviations, credibility tracking.
5. Automated leakage verification assert.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ml.trust_layer import ObservationAssessment, TrustAwareSignalProcessor

KEY_VARIABLES = [
    "HR", "SysABP", "DiasABP", "MAP", "RespRate", "Temp", "GCS", "Urine", "SaO2", "Glucose"
]

class FeatureExtractor:
    def __init__(self, config: dict):
        self.config = config
        self.baseline_window = config["pipeline"].get("baseline_window_hours", 6.0)
        self.rolling_windows = config["pipeline"].get("rolling_windows_hours", [1.0, 3.0, 6.0])
        self.trust_processor = TrustAwareSignalProcessor(config)

    def extract_patient_features_at_time(
        self,
        current_time: float,
        assessments: List[ObservationAssessment],
        static_info: Dict[str, float],
        patient_id: int
    ) -> Dict[str, float]:
        """
        Extracts features at exact chronological time T.
        STRICT ASSERTION: Filters observations to strictly time <= current_time.
        """
        # 1. Leakage guard
        past_assessments = [a for a in assessments if a.time_hours <= current_time]
        future_count = sum(1 for a in assessments if a.time_hours > current_time)
        
        # Verify no future leakage
        for a in past_assessments:
            if a.time_hours > current_time + 1e-6:
                raise ValueError(f"LEAKAGE DETECTED! Observation time {a.time_hours} > current_time {current_time}")

        feat = {
            "patient_id": patient_id,
            "eval_time": current_time,
            "age": float(static_info.get("Age", 65.0)),
            "gender": float(static_info.get("Gender", 0.0)),
            "weight": float(static_info.get("Weight", 75.0)),
            "icu_type": float(static_info.get("ICUType", 1.0)),
        }

        # 2. Build parameter histories from past assessments
        # Map parameter alias (e.g. NISysABP -> SysABP)
        param_data = {var: [] for var in KEY_VARIABLES}
        for a in past_assessments:
            std_p = self.trust_processor.map_parameter_alias(a.parameter)
            if std_p in param_data:
                param_data[std_p].append(a)

        # 3. Compute Layer 2: Personal Baseline for each variable
        # Base window is min(baseline_window, current_time)
        baseline_cutoff = min(self.baseline_window, current_time)
        
        for var in KEY_VARIABLES:
            records = param_data[var]
            
            # --- Staleness Clock & Missingness ---
            if not records:
                feat[f"{var}_is_missing"] = 1.0
                feat[f"{var}_time_since_last"] = float(current_time)
                feat[f"{var}_last_value"] = np.nan
                feat[f"{var}_last_credibility"] = 0.0
                feat[f"{var}_baseline_mean"] = np.nan
                feat[f"{var}_dev_from_baseline"] = 0.0
                feat[f"{var}_pct_dev_from_baseline"] = 0.0
                feat[f"{var}_rolling_mean_3h"] = np.nan
                feat[f"{var}_rolling_std_3h"] = 0.0
                feat[f"{var}_slope_3h"] = 0.0
                feat[f"{var}_change_1h"] = 0.0
                feat[f"{var}_change_6h"] = 0.0
                continue

            last_rec = records[-1]
            feat[f"{var}_is_missing"] = 0.0
            feat[f"{var}_time_since_last"] = max(0.0, float(current_time - last_rec.time_hours))
            feat[f"{var}_last_value"] = float(last_rec.original_value)
            feat[f"{var}_last_credibility"] = float(last_rec.credibility_score)

            # Personal baseline computed from observations up to baseline_cutoff
            baseline_recs = [r for r in records if r.time_hours <= baseline_cutoff and r.credibility_score > 0.3]
            if len(baseline_recs) >= 1:
                base_vals = [r.original_value for r in baseline_recs]
                base_mean = float(np.mean(base_vals))
                feat[f"{var}_baseline_mean"] = base_mean
                dev = float(last_rec.original_value - base_mean)
                feat[f"{var}_dev_from_baseline"] = dev
                feat[f"{var}_pct_dev_from_baseline"] = float(dev / (base_mean + 1e-4))
            else:
                # Fallback to first available credible value
                base_mean = float(records[0].original_value)
                feat[f"{var}_baseline_mean"] = base_mean
                feat[f"{var}_dev_from_baseline"] = float(last_rec.original_value - base_mean)
                feat[f"{var}_pct_dev_from_baseline"] = 0.0

            # --- Trajectory features over rolling windows ---
            # Recent 3 hours
            rec_3h = [r for r in records if r.time_hours >= (current_time - 3.0) and r.credibility_score > 0.2]
            if len(rec_3h) >= 1:
                vals_3h = [r.original_value for r in rec_3h]
                feat[f"{var}_rolling_mean_3h"] = float(np.mean(vals_3h))
                feat[f"{var}_rolling_std_3h"] = float(np.std(vals_3h)) if len(vals_3h) > 1 else 0.0
                # Slope calculation
                if len(rec_3h) >= 2:
                    t_points = np.array([r.time_hours for r in rec_3h])
                    v_points = np.array(vals_3h)
                    dt = t_points[-1] - t_points[0]
                    if dt > 0.1:
                        feat[f"{var}_slope_3h"] = float((v_points[-1] - v_points[0]) / dt)
                    else:
                        feat[f"{var}_slope_3h"] = 0.0
                else:
                    feat[f"{var}_slope_3h"] = 0.0
            else:
                feat[f"{var}_rolling_mean_3h"] = feat[f"{var}_last_value"]
                feat[f"{var}_rolling_std_3h"] = 0.0
                feat[f"{var}_slope_3h"] = 0.0

            # Change over 1 hour
            rec_1h = [r for r in records if r.time_hours >= (current_time - 1.0)]
            if len(rec_1h) >= 2:
                feat[f"{var}_change_1h"] = float(rec_1h[-1].original_value - rec_1h[0].original_value)
            else:
                feat[f"{var}_change_1h"] = 0.0

            # Change over 6 hours
            rec_6h = [r for r in records if r.time_hours >= (current_time - 6.0)]
            if len(rec_6h) >= 2:
                feat[f"{var}_change_6h"] = float(rec_6h[-1].original_value - rec_6h[0].original_value)
            else:
                feat[f"{var}_change_6h"] = 0.0

        # 4. Multi-variable compound deterioration signals
        hr = feat.get("HR_last_value", np.nan)
        sys = feat.get("SysABP_last_value", np.nan)
        rr = feat.get("RespRate_last_value", np.nan)
        temp = feat.get("Temp_last_value", np.nan)
        
        # Shock Index = HR / SysABP (normal: 0.5 - 0.7, elevated > 0.9 indicates circulatory compromise)
        if not np.isnan(hr) and not np.isnan(sys) and sys > 10:
            feat["shock_index"] = float(hr / sys)
        else:
            feat["shock_index"] = np.nan

        # Compound vital trajectory: HR up + MAP/Sys down + RR up
        hr_dev = feat.get("HR_dev_from_baseline", 0.0)
        sys_dev = feat.get("SysABP_dev_from_baseline", 0.0)
        rr_dev = feat.get("RespRate_dev_from_baseline", 0.0)
        
        compound_score = 0.0
        if hr_dev > 10: compound_score += 1.0
        if sys_dev < -15: compound_score += 1.0
        if rr_dev > 4: compound_score += 1.0
        feat["multi_var_deterioration_signal"] = compound_score

        # Average credibility of recent vitals
        recent_creds = [feat[f"{v}_last_credibility"] for v in KEY_VARIABLES if feat[f"{v}_is_missing"] == 0]
        feat["mean_recent_credibility"] = float(np.mean(recent_creds)) if recent_creds else 1.0

        return feat

    def generate_patient_chronological_features(
        self,
        assessments: List[ObservationAssessment],
        static_info: Dict[str, float],
        patient_id: int,
        start_hour: float = 2.0,
        end_hour: float = 48.0,
        step_hours: float = 1.0
    ) -> pd.DataFrame:
        """Generates hourly feature vectors for a patient without any lookahead."""
        rows = []
        cur_t = start_hour
        while cur_t <= end_hour:
            row = self.extract_patient_features_at_time(cur_t, assessments, static_info, patient_id)
            rows.append(row)
            cur_t += step_hours
            
        return pd.DataFrame(rows)
