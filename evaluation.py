"""
evaluation.py - Comprehensive Chronological Evaluation and Comparative Benchmarking.

Evaluates test cohort on:
- Sensitivity / Recall (patient-level deterioration detection)
- False Alarm Rate / False Positive Rate
- Alerts per patient
- Alerts per patient-day
- Median Lead Time (with clear proxy disclaimer)
- Precision, Confusion Matrix, PR-AUC, ROC-AUC
- Lead-time vs Alert-burden trade-off across SilentWindow vs Plain ML vs Simplified Threshold.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, roc_auc_score, average_precision_score

from ml.preprocessing import load_config, load_patient_raw
from ml.replay import ChronologicalReplayEngine

def compute_system_metrics(
    patient_summaries: List[Dict[str, Any]],
    alert_col: str,
    total_hours_sum: float,
    outcome_key: str = "actual_outcome"
) -> Dict[str, Any]:
    """Computes clinical early-warning metrics for an alerting strategy."""
    y_true = np.array([p[outcome_key] for p in patient_summaries])
    y_pred = np.array([1 if p[alert_col] > 0 else 0 for p in patient_summaries])
    total_alerts = sum(p[alert_col] for p in patient_summaries)
    n_patients = len(patient_summaries)
    n_pos = int(y_true.sum())
    n_neg = int(n_patients - n_pos)

    # Confusion matrix elements (patient level)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    sensitivity = round(float(tp / max(1, n_pos)), 4)
    specificity = round(float(tn / max(1, n_neg)), 4)
    precision = round(float(tp / max(1, tp + fp)), 4)
    false_positive_rate = round(float(fp / max(1, n_neg)), 4)
    
    # False alarms per non-deteriorating patient
    false_alerts = sum(p[alert_col] for p in patient_summaries if p[outcome_key] == 0)
    
    # Alerts per patient and patient-day
    alerts_per_patient = round(float(total_alerts / max(1, n_patients)), 3)
    total_patient_days = max(1e-4, total_hours_sum / 24.0)
    alerts_per_patient_day = round(float(total_alerts / total_patient_days), 3)

    # Lead times for true positives
    lead_times = [p[f"{alert_col}_lead_time"] for p in patient_summaries if p[outcome_key] == 1 and p.get(f"{alert_col}_lead_time") is not None]
    median_lead_time = round(float(np.median(lead_times)), 2) if lead_times else None
    mean_lead_time = round(float(np.mean(lead_times)), 2) if lead_times else None
    first_time_col = f"{alert_col}_first_time"
    first_alert_times = [p[first_time_col] for p in patient_summaries if p.get(first_time_col) is not None]
    median_first_alert = round(float(np.median(first_alert_times)), 2) if first_alert_times else None

    return {
        "n_patients": n_patients,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "total_alerts": total_alerts,
        "n_patients_ever_alert": int(np.sum(y_pred)),
        "share_patients_ever_alert": round(float(np.mean(y_pred)), 4) if n_patients else 0.0,
        "median_time_to_first_alert_hours": median_first_alert,
        "time_to_first_alert_hours": first_alert_times,
        "false_alerts": false_alerts,
        "alerts_per_patient": alerts_per_patient,
        "alerts_per_patient_day": alerts_per_patient_day,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "false_positive_rate": false_positive_rate,
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn
        },
        "median_lead_time_hours": median_lead_time,
        "mean_lead_time_hours": mean_lead_time,
        "lead_times": lead_times
    }

def run_evaluation(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    config = load_config(config_path)
    splits_file = os.path.join(config["data"]["processed_dir"], "splits.json")
    with open(splits_file, "r") as f:
        splits = json.load(f)

    test_ids = splits["test"]
    print(f"Running chronological test evaluation on {len(test_ids)} patients...")
    replay_engine = ChronologicalReplayEngine(config_path)
    
    all_records = replay_engine.replay_cohort(test_ids, step_hours=1.0)
    df_eval = pd.DataFrame(all_records)

    # Save raw records for dashboard timeline inspection
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    df_eval.to_parquet(os.path.join(results_dir, "test_eval_records.parquet"), index=False)

    # Aggregate by patient
    patient_summaries = []
    total_telemetry_hours = 0.0

    for pid in test_ids:
        pdf = df_eval[df_eval["patient_id"] == pid].sort_values("time_hours")
        if pdf.empty:
            continue
        
        outcome = int(pdf["actual_outcome"].iloc[0])
        secondary_outcome = int(pdf["secondary_outcome"].iloc[0])
        patient_file = os.path.join(config["data"]["records_dir"], f"{pid}.txt")
        static_info, _ = load_patient_raw(patient_file)
        max_t = float(pdf["time_hours"].max())
        total_telemetry_hours += max_t

        # 1. SilentWindow
        sw_alerts = int(pdf["alert_fired"].sum())
        sw_first_alert = pdf[pdf["alert_fired"] == True]
        sw_first_time = float(sw_first_alert["time_hours"].iloc[0]) if not sw_first_alert.empty else None
        sw_lead = max(0.0, max_t - sw_first_alert["time_hours"].iloc[0]) if not sw_first_alert.empty else None

        # 2. Threshold Baseline
        th_alerts = int(pdf["baseline_threshold_alert"].sum())
        th_first_alert = pdf[pdf["baseline_threshold_alert"] == True]
        th_first_time = float(th_first_alert["time_hours"].iloc[0]) if not th_first_alert.empty else None
        th_lead = max(0.0, max_t - th_first_alert["time_hours"].iloc[0]) if not th_first_alert.empty else None

        # 3. Plain ML
        pml_alerts = int(pdf["baseline_plain_ml_alert"].sum())
        pml_first_alert = pdf[pdf["baseline_plain_ml_alert"] == True]
        pml_first_time = float(pml_first_alert["time_hours"].iloc[0]) if not pml_first_alert.empty else None
        pml_lead = max(0.0, max_t - pml_first_alert["time_hours"].iloc[0]) if not pml_first_alert.empty else None

        patient_summaries.append({
            "patient_id": pid,
            "age": float(static_info.get("Age", np.nan)),
            "icu_type": int(static_info.get("ICUType", -1)),
            "actual_outcome": outcome,
            "secondary_outcome": secondary_outcome,
            "max_hours": max_t,
            "sw_alerts": sw_alerts,
            "sw_alerts_lead_time": sw_lead,
            "sw_alerts_first_time": sw_first_time,
            "th_alerts": th_alerts,
            "th_alerts_lead_time": th_lead,
            "th_alerts_first_time": th_first_time,
            "pml_alerts": pml_alerts,
            "pml_alerts_lead_time": pml_lead,
            "pml_alerts_first_time": pml_first_time,
            "max_risk": float(pdf["calibrated_risk"].max()),
            "max_evidence": float(pdf["evidence_score"].max())
        })

    # Compute metrics for each system
    metrics_sw = compute_system_metrics(patient_summaries, "sw_alerts", total_telemetry_hours)
    metrics_th = compute_system_metrics(patient_summaries, "th_alerts", total_telemetry_hours)
    metrics_pml = compute_system_metrics(patient_summaries, "pml_alerts", total_telemetry_hours)

    # Compute continuous curves (PR and ROC) using patient-level max calibrated risk
    y_true_all = [p["actual_outcome"] for p in patient_summaries]
    y_scores_all = [p["max_risk"] for p in patient_summaries]

    roc_auc = float(roc_auc_score(y_true_all, y_scores_all))
    pr_auc = float(average_precision_score(y_true_all, y_scores_all))

    # ROC curve coordinates
    fpr_arr, tpr_arr, _ = roc_curve(y_true_all, y_scores_all)
    roc_curve_data = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr_arr, tpr_arr)]

    # PR curve coordinates
    prec_arr, rec_arr, _ = precision_recall_curve(y_true_all, y_scores_all)
    pr_curve_data = [{"recall": round(float(r), 4), "precision": round(float(p), 4)} for r, p in zip(rec_arr, prec_arr)]
    subgroup_metrics = {}
    for group_name, group_fn in {
        "age_under_65": lambda p: p["age"] < 65,
        "age_65_or_older": lambda p: p["age"] >= 65,
    }.items():
        subset = [p for p in patient_summaries if not np.isnan(p["age"]) and group_fn(p)]
        if subset:
            subgroup_metrics[group_name] = {
                "n_patients": len(subset),
                "silent_window": compute_system_metrics(subset, "sw_alerts", sum(p["max_hours"] for p in subset)),
            }
    for icu_type in sorted({p["icu_type"] for p in patient_summaries if p["icu_type"] >= 0}):
        subset = [p for p in patient_summaries if p["icu_type"] == icu_type]
        subgroup_metrics[f"icu_type_{icu_type}"] = {
            "n_patients": len(subset),
            "silent_window": compute_system_metrics(subset, "sw_alerts", sum(p["max_hours"] for p in subset)),
        }

    summary = {
        "disclaimer": "Research prototype — retrospective data only. Not for clinical decision-making. Lead time is estimated relative to the available outcome/proxy because an exact event timestamp is unavailable.",
        "test_patients_count": len(patient_summaries),
        "total_monitoring_hours": round(total_telemetry_hours, 1),
        "auc_metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4)
        },
        "systems": {
            "silent_window": metrics_sw,
            "plain_ml": metrics_pml,
            "threshold_baseline": metrics_th
        },
        "secondary_robustness": {
            "label": "in-hospital death OR ICU stay under 48 hours",
            "positive_patients": int(sum(p["secondary_outcome"] for p in patient_summaries)),
            "systems": {
                "silent_window": compute_system_metrics(patient_summaries, "sw_alerts", total_telemetry_hours, "secondary_outcome"),
                "plain_ml": compute_system_metrics(patient_summaries, "pml_alerts", total_telemetry_hours, "secondary_outcome"),
                "threshold_baseline": compute_system_metrics(patient_summaries, "th_alerts", total_telemetry_hours, "secondary_outcome")
            }
        },
        "subgroup_metrics": subgroup_metrics,
        "curves": {
            "roc": roc_curve_data,
            "pr": pr_curve_data
        }
    }

    summary_file = os.path.join(results_dir, "evaluation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- TEST EVALUATION COMPLETE ---")
    print(f"SilentWindow: Sens={metrics_sw['sensitivity_recall']}, Alerts/pt-day={metrics_sw['alerts_per_patient_day']}, False Alerts={metrics_sw['false_alerts']}")
    print(f"Plain ML:     Sens={metrics_pml['sensitivity_recall']}, Alerts/pt-day={metrics_pml['alerts_per_patient_day']}, False Alerts={metrics_pml['false_alerts']}")
    print(f"Threshold:    Sens={metrics_th['sensitivity_recall']}, Alerts/pt-day={metrics_th['alerts_per_patient_day']}, False Alerts={metrics_th['false_alerts']}")

    return summary

if __name__ == "__main__":
    run_evaluation()
