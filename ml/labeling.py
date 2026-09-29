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
