"""
labeling.py - Target Definition and Ground-Truth Association.

TARGET SPECIFICATION:

Default training target (configs/config.yaml: labeling.target =
"future_window_proxy"): a near-term early-warning horizon label. An hourly
row is positive only when the patient eventually dies in hospital AND the
available telemetry endpoint proxy falls strictly after the current eval
time and within ``labeling.horizon_hours``. The production training path
(build_split_dataset in ml/train.py) uses the patient's end-of-telemetry
time as the endpoint proxy — the same convention as replay_patient in
ml/replay.py. The assigned column is also aliased to ``target`` so training
code reads a stable column name regardless of configuration.

Legacy target (labeling.target = "eventual_outcome"): the patient's final
in-hospital outcome broadcast to every hourly row (assign_hourly_labels).
Kept as an explicit opt-in for comparing against the horizon label.

Secondary metadata attached to every labeled frame: sofa, saps,
length_of_stay. These are outcome-adjacent and are excluded from the model
feature set in ml/train.py (meta_cols).

IMPORTANT SCIENTIFIC NOTE:
Lead time is estimated relative to the available outcome/proxy (end of telemetry / ICU stay)
because an exact sub-hourly event timestamp is unavailable in the challenge dataset.
Never pretend an unavailable event timestamp exists.
"""

import os
import pandas as pd
from typing import Dict, Optional, Tuple

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

    def assign_training_labels(
        self,
        df_patient_features: pd.DataFrame,
        patient_id: int,
        target_kind: str = "future_window_proxy",
        horizon_hours: float = 12.0,
        event_proxy_hours: Optional[float] = None,
    ) -> pd.DataFrame:
        """Assign the configured training target for one patient's feature rows.

        target_kind="future_window_proxy" (default): builds the near-term
        horizon label via :meth:`assign_future_window_proxy_labels`, using
        ``event_proxy_hours`` as the endpoint proxy (the production training
        path passes the patient's end-of-telemetry time, the same convention
        as ``replay_patient`` in ml/replay.py). The proxy label is aliased to
        the ``target`` column so training code reads a stable column name, and
        the eventual-outcome metadata (sofa/saps/length_of_stay) is attached
        for reference — both stay out of the model feature set (see meta_cols
        in ml/train.py).

        target_kind="eventual_outcome": legacy behaviour — the patient's final
        in-hospital outcome broadcast to every hourly row
        (:meth:`assign_hourly_labels`).
        """
        if target_kind == "future_window_proxy":
            if event_proxy_hours is None:
                raise ValueError("future_window_proxy target requires event_proxy_hours")
            df = self.assign_future_window_proxy_labels(
                df_patient_features, patient_id, event_proxy_hours, horizon_hours
            )
            outcome = self.get_patient_outcome(patient_id)
            df["target"] = df["future_window_proxy_target"]
            df["sofa"] = outcome["sofa"]
            df["saps"] = outcome["saps"]
            df["length_of_stay"] = outcome["length_of_stay"]
            return df
        if target_kind == "eventual_outcome":
            return self.assign_hourly_labels(df_patient_features, patient_id)
        raise ValueError(f"Unknown labeling target_kind: {target_kind!r}")
