"""
labeling.py - Target Definition and Ground-Truth Association.

TARGET SPECIFICATION:
Primary target: In-hospital mortality / acute clinical deterioration (In-hospital_death: 1 vs 0).
Secondary metrics: SOFA score, SAPS-I score, and Length of Stay.

IMPORTANT SCIENTIFIC NOTE:
Lead time is estimated relative to the available outcome/proxy (end of telemetry / ICU stay)
because an exact sub-hourly event timestamp is unavailable in the challenge dataset.
Never pretend an unavailable event timestamp exists.
"""

import os
import pandas as pd
from typing import Dict, Tuple

class OutcomeLabeler:
    def __init__(self, outcomes_file: str = "data/raw/Outcomes-train.txt"):
        self.outcomes_file = outcomes_file
        self.df_outcomes = pd.read_csv(outcomes_file)
        self.outcomes_by_id = self.df_outcomes.set_index("RecordID").to_dict(orient="index")

    def get_patient_outcome(self, patient_id: int) -> Dict[str, float]:
        """Retrieves ground truth outcome dictionary for a patient."""
        rec = self.outcomes_by_id.get(patient_id, None)
        if rec is None:
            return {
                "in_hospital_death": 0,
                "sofa": -1.0,
                "saps": -1.0,
                "length_of_stay": -1.0,
                "survival": -1.0
            }
        return {
            "in_hospital_death": int(rec.get("In-hospital_death", 0)),
            "sofa": float(rec.get("SOFA", -1.0)),
            "saps": float(rec.get("SAPS-I", -1.0)),
            "length_of_stay": float(rec.get("Length_of_stay", -1.0)),
            "survival": float(rec.get("Survival", -1.0))
        }

    def assign_hourly_labels(self, df_patient_features: pd.DataFrame, patient_id: int) -> pd.DataFrame:
        """
        Assigns the ground-truth outcome to the hourly feature rows of a patient.
        """
        outcome = self.get_patient_outcome(patient_id)
        df = df_patient_features.copy()
        df["target"] = outcome["in_hospital_death"]
        df["sofa"] = outcome["sofa"]
        df["saps"] = outcome["saps"]
        df["length_of_stay"] = outcome["length_of_stay"]
        return df

    def assign_future_window_proxy_labels(
        self,
        df_patient_features: pd.DataFrame,
        patient_id: int,
        event_proxy_hours: float,
        horizon_hours: float,
    ) -> pd.DataFrame:
        """Assign a research-only future-window label using an endpoint proxy.

        The challenge data does not expose an exact deterioration/death timestamp.
        Therefore ``event_proxy_hours`` must be the available telemetry/outcome
        endpoint, and the result is explicitly a proxy target rather than a true
        early-warning label. Rows are positive only when a death outcome exists
        and the proxy endpoint falls strictly after the current time and within
        the requested horizon.
        """
        if horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")
        if "eval_time" not in df_patient_features.columns:
            raise ValueError("Future-window labels require an eval_time column")
        outcome = self.get_patient_outcome(patient_id)
        df = df_patient_features.copy()
        df["future_window_proxy_target"] = 0
        if outcome["in_hospital_death"] == 1:
            delta = float(event_proxy_hours) - df["eval_time"].astype(float)
            df.loc[(delta > 0.0) & (delta <= float(horizon_hours)), "future_window_proxy_target"] = 1
        df["future_window_horizon_hours"] = float(horizon_hours)
        df["future_window_proxy_disclaimer"] = (
            "Proxy endpoint only; exact deterioration timestamp unavailable."
        )
        return df
