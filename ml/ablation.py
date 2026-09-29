"""
ablation.py - Systematic Ablation Experiment for SilentWindow Architecture.

Evaluates 4 concrete configurations on test cohort:
A. Full SilentWindow (Trust + Personal Baseline + Trajectory + Accumulator)
B. Without Trust Layer (credibility forced to 1.0; no downweighting of noise)
C. Without Personal Baseline (population absolute features only, no delta from baseline)
D. Without Evidence Accumulator (direct probability thresholding >= 0.50)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from ml.preprocessing import load_config
from ml.replay import ChronologicalReplayEngine
from ml.evidence_accumulator import SequentialEvidenceAccumulator
from ml.evaluation import compute_system_metrics

def run_ablation_study(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    config = load_config(config_path)
    records_file = os.path.join("results", "test_eval_records.parquet")
    
    if not os.path.exists(records_file):
        raise FileNotFoundError("test_eval_records.parquet not found. Run ml/evaluation.py first!")

    df_records = pd.read_parquet(records_file)
    unique_patients = df_records["patient_id"].unique().tolist()
    print(f"Loaded {len(df_records)} evaluation timepoints across {len(unique_patients)} test patients.")

    accumulator = SequentialEvidenceAccumulator(config)
    total_hours_sum = float(df_records.groupby("patient_id")["time_hours"].max().sum())

    # --- Variant A: Full SilentWindow ---
    summaries_a = []
    for pid, pdf in df_records.groupby("patient_id"):
        outcome = int(pdf["actual_outcome"].iloc[0])
        max_t = float(pdf["time_hours"].max())
        alerts = int(pdf["alert_fired"].sum())
        first_alert = pdf[pdf["alert_fired"] == True]
        lead = max(0.0, max_t - first_alert["time_hours"].iloc[0]) if not first_alert.empty else None
        summaries_a.append({"patient_id": pid, "actual_outcome": outcome, "alerts": alerts, "alerts_lead_time": lead})

    metrics_a = compute_system_metrics(summaries_a, "alerts", total_hours_sum)

    # --- Variant B: Without Trust Layer (credibility = 1.0) ---
    summaries_b = []
    for pid, pdf in df_records.groupby("patient_id"):
        outcome = int(pdf["actual_outcome"].iloc[0])
        max_t = float(pdf["time_hours"].max())
        pdf_sorted = pdf.sort_values("time_hours")
        
        accumulator.reset()
        alerts_b = 0
        first_alert_t = None
        for _, row in pdf_sorted.iterrows():
            st = accumulator.update(
                time_hours=row["time_hours"],
                calibrated_risk=row["calibrated_risk"],
                raw_risk=row["raw_risk"],
                credibility_weight=1.0,
                enable_trust=False
            )
            if st.alert_fired:
                alerts_b += 1
                if first_alert_t is None:
                    first_alert_t = row["time_hours"]

        lead = max(0.0, max_t - first_alert_t) if first_alert_t is not None else None
        summaries_b.append({"patient_id": pid, "actual_outcome": outcome, "alerts": alerts_b, "alerts_lead_time": lead})

    metrics_b = compute_system_metrics(summaries_b, "alerts", total_hours_sum)

    # --- Variant D: Without Evidence Accumulator (Direct Instant Risk >= 0.50) ---
    summaries_d = []
    for pid, pdf in df_records.groupby("patient_id"):
        outcome = int(pdf["actual_outcome"].iloc[0])
        max_t = float(pdf["time_hours"].max())
        alerts_d = int(pdf["baseline_plain_ml_alert"].sum())
        first_alert = pdf[pdf["baseline_plain_ml_alert"] == True]
        lead = max(0.0, max_t - first_alert["time_hours"].iloc[0]) if not first_alert.empty else None
        summaries_d.append({"patient_id": pid, "actual_outcome": outcome, "alerts": alerts_d, "alerts_lead_time": lead})

    metrics_d = compute_system_metrics(summaries_d, "alerts", total_hours_sum)

    # --- Variant C: Without Personal Baseline ---
    # Run on representative test sample (35 patients) to assess the delta
    print("Evaluating Without Personal Baseline on test cohort sample...")
    replay_engine = ChronologicalReplayEngine(config_path)
    sample_ids = unique_patients[:40]
    records_c = replay_engine.replay_cohort(sample_ids, step_hours=1.0, enable_personal_baseline=False)
    df_c = pd.DataFrame(records_c)
    
    summaries_c = []
    hours_c = 0.0
    for pid, pdf in df_c.groupby("patient_id"):
        outcome = int(pdf["actual_outcome"].iloc[0])
        max_t = float(pdf["time_hours"].max())
        hours_c += max_t
        alerts_c = int(pdf["alert_fired"].sum())
        first_alert = pdf[pdf["alert_fired"] == True]
        lead = max(0.0, max_t - first_alert["time_hours"].iloc[0]) if not first_alert.empty else None
        summaries_c.append({"patient_id": pid, "actual_outcome": outcome, "alerts": alerts_c, "alerts_lead_time": lead})

    metrics_c_sample = compute_system_metrics(summaries_c, "alerts", hours_c)
    
    # Scale alerts for cohort comparability
    scale_factor = len(unique_patients) / len(sample_ids)
    metrics_c = {
        "sensitivity_recall": metrics_c_sample["sensitivity_recall"],
        "false_alerts": int(round(metrics_c_sample["false_alerts"] * scale_factor)),
        "alerts_per_patient_day": metrics_c_sample["alerts_per_patient_day"],
        "median_lead_time_hours": metrics_c_sample["median_lead_time_hours"],
        "precision": metrics_c_sample["precision"],
        "total_alerts": int(round(metrics_c_sample["total_alerts"] * scale_factor))
    }

    results_table = [
        {
            "variant_id": "full_silentwindow",
            "system_name": "Full SilentWindow",
            "description": "Trust Layer + Personal Baseline + Trajectory + Evidence Accumulator",
            "sensitivity": metrics_a["sensitivity_recall"],
            "false_alarms": metrics_a["false_alerts"],
            "alerts_per_patient_day": metrics_a["alerts_per_patient_day"],
            "median_lead_time": metrics_a["median_lead_time_hours"] if metrics_a["median_lead_time_hours"] is not None else 0.0,
            "precision": metrics_a["precision"],
            "total_alerts": metrics_a["total_alerts"]
        },
        {
            "variant_id": "no_trust",
            "system_name": "Without Trust Layer",
            "description": "Personal Baseline + Trajectory + Evidence Accumulator (No Credibility Weighting)",
            "sensitivity": metrics_b["sensitivity_recall"],
            "false_alarms": metrics_b["false_alerts"],
            "alerts_per_patient_day": metrics_b["alerts_per_patient_day"],
            "median_lead_time": metrics_b["median_lead_time_hours"] if metrics_b["median_lead_time_hours"] is not None else 0.0,
            "precision": metrics_b["precision"],
            "total_alerts": metrics_b["total_alerts"]
        },
        {
            "variant_id": "no_personal_baseline",
            "system_name": "Without Personal Baseline",
            "description": "Trust Layer + Trajectory + Evidence Accumulator (No Patient-Specific Delta)",
            "sensitivity": metrics_c["sensitivity_recall"],
            "false_alarms": metrics_c["false_alerts"],
            "alerts_per_patient_day": metrics_c["alerts_per_patient_day"],
            "median_lead_time": metrics_c["median_lead_time_hours"] if metrics_c["median_lead_time_hours"] is not None else 0.0,
            "precision": metrics_c["precision"],
            "total_alerts": metrics_c["total_alerts"]
        },
        {
            "variant_id": "no_accumulator",
            "system_name": "Without Evidence Accumulator",
            "description": "Trust Layer + Personal Baseline + Trajectory + Direct Instantaneous Threshold",
            "sensitivity": metrics_d["sensitivity_recall"],
            "false_alarms": metrics_d["false_alerts"],
            "alerts_per_patient_day": metrics_d["alerts_per_patient_day"],
            "median_lead_time": metrics_d["median_lead_time_hours"] if metrics_d["median_lead_time_hours"] is not None else 0.0,
            "precision": metrics_d["precision"],
            "total_alerts": metrics_d["total_alerts"]
        }
    ]

    ablation_summary = {
        "disclaimer": "Research prototype — retrospective data only. Not for clinical decision-making. Lead time estimated relative to outcome/proxy.",
        "results_table": results_table
    }

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "ablation_results.json"), "w") as f:
        json.dump(ablation_summary, f, indent=2)

    print("\n--- ABLATION RESULTS TABLE ---")
    df_res = pd.DataFrame(results_table)
    print(df_res[["system_name", "sensitivity", "false_alarms", "alerts_per_patient_day", "median_lead_time"]].to_string(index=False))

    return ablation_summary

if __name__ == "__main__":
    run_ablation_study()
