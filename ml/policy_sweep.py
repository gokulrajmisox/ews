"""Alert-policy operating-point analysis over chronological replay records."""

from __future__ import annotations

import copy
import os
from typing import Iterable

import pandas as pd

from ml.evidence_accumulator import SequentialEvidenceAccumulator
from ml.preprocessing import load_config


def sweep_alert_thresholds(
    records_path: str = "results/test_eval_records.parquet",
    thresholds: Iterable[float] = (0.55, 0.65, 0.75, 0.85, 0.95),
    config_path: str = "configs/config.yaml",
) -> list[dict]:
    """Replay the saved cohort at multiple alert thresholds.

    This is an operating-point analysis, not threshold selection. Thresholds must
    be selected on validation data before using a single value for test reporting.
    """
    if not os.path.exists(records_path):
        raise FileNotFoundError(f"Replay records not found: {records_path}")
    df = pd.read_parquet(records_path)
    required = {"patient_id", "time_hours", "calibrated_risk", "raw_risk", "mean_credibility", "actual_outcome"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Replay records are missing columns: {sorted(missing)}")

    base_config = load_config(config_path)
    total_patient_days = max(1e-9, float(df.groupby("patient_id")["time_hours"].max().sum()) / 24.0)
    results = []
    for threshold in thresholds:
        threshold = float(threshold)
        if not 0.0 < threshold <= 2.0:
            raise ValueError("Alert thresholds must be in the interval (0, 2].")
        config = copy.deepcopy(base_config)
        config.setdefault("evidence_accumulator", {})["alert_threshold"] = threshold
        patient_alerted: dict[int, bool] = {}
        patient_outcome: dict[int, int] = {}
        total_alerts = 0
        false_alert_events = 0
        for patient_id, patient_df in df.groupby("patient_id"):
            accumulator = SequentialEvidenceAccumulator(config)
            patient_df = patient_df.sort_values("time_hours")
            for row in patient_df.itertuples(index=False):
                snapshot = accumulator.update(
                    time_hours=float(row.time_hours),
                    calibrated_risk=float(row.calibrated_risk),
                    raw_risk=float(row.raw_risk),
                    credibility_weight=float(row.mean_credibility),
                )
                if snapshot.alert_fired:
                    total_alerts += 1
                    patient_alerted[int(patient_id)] = True
                    if int(row.actual_outcome) == 0:
                        false_alert_events += 1
            patient_outcome[int(patient_id)] = int(patient_df["actual_outcome"].iloc[-1])

        positives = sum(value == 1 for value in patient_outcome.values())
        detected_positive = sum(patient_alerted.get(pid, False) and outcome == 1 for pid, outcome in patient_outcome.items())
        alerted_patients = sum(patient_alerted.values())
        precision = detected_positive / alerted_patients if alerted_patients else 0.0
        results.append(
            {
                "alert_threshold": round(threshold, 3),
                "total_alerts": total_alerts,
                "false_alert_events": false_alert_events,
                "alerted_patients": alerted_patients,
                "positive_patients": positives,
                "positive_patients_detected": detected_positive,
                "sensitivity": round(detected_positive / positives, 4) if positives else 0.0,
                "precision": round(precision, 4),
                "alerts_per_patient_day": round(total_alerts / total_patient_days, 4),
            }
        )
    return results
