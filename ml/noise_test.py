"""
noise_test.py - Synthetic Telemetry Noise Stress Testing & Noise Lab.

CRITICAL RULE:
The noise generator NEVER modifies original disk files.
It operates purely on in-memory deep copies of observation DataFrames.
Injects:
- Random sensor dropouts (missingness)
- Transient motion / loose-lead spikes (e.g. HR=250, SysABP=35)
- Irregular sampling gaps
"""

import os
import copy
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from ml.preprocessing import load_config, load_patient_raw
from ml.replay import ChronologicalReplayEngine
from ml.evaluation import compute_system_metrics

class NoiseStressTester:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.replay_engine = ChronologicalReplayEngine(config_path)

    def inject_noise_to_stream(
        self,
        df_obs: pd.DataFrame,
        intensity: float = 0.5,
        inject_missingness: bool = True,
        inject_spikes: bool = True,
        inject_jitter: bool = True,
        seed: int = 99
    ) -> pd.DataFrame:
        """
        Creates a synthetically corrupted in-memory copy of an observation stream.
        Original DataFrame is never modified.
        """
        if df_obs.empty:
            return df_obs.copy()

        np.random.seed(seed)
        df_noisy = df_obs.copy()

        # 1. Missingness injection (dropout)
        if inject_missingness and intensity > 0.01:
            drop_prob = min(0.40, intensity * 0.35)
            mask = np.random.rand(len(df_noisy)) > drop_prob
            df_noisy = df_noisy[mask].copy()

        # 2. Transient Spike injection
        if inject_spikes and intensity > 0.01 and not df_noisy.empty:
            spike_prob = min(0.15, intensity * 0.12)
            for idx in df_noisy.index:
                if np.random.rand() < spike_prob:
                    param = df_noisy.at[idx, "parameter"]
                    val = df_noisy.at[idx, "value"]
                    if param == "HR":
                        # Extreme tachy spike or zero drop
                        df_noisy.at[idx, "value"] = 230.0 if np.random.rand() > 0.3 else 15.0
                    elif param in ["SysABP", "NISysABP"]:
                        # Severe spike or cuff loss
                        df_noisy.at[idx, "value"] = 35.0 if np.random.rand() > 0.5 else 235.0
                    elif param == "RespRate":
                        df_noisy.at[idx, "value"] = 48.0 if np.random.rand() > 0.5 else 3.0
                    elif param == "Temp":
                        df_noisy.at[idx, "value"] = 41.5 if np.random.rand() > 0.5 else 32.0

        # 3. Timing irregularity & jitter
        if inject_jitter and intensity > 0.01 and not df_noisy.empty:
            jitter_hours = (np.random.rand(len(df_noisy)) - 0.5) * (intensity * 0.6)
            df_noisy["time_hours"] = np.clip(df_noisy["time_hours"] + jitter_hours, 0.0, 48.0)
            df_noisy = df_noisy.sort_values("time_hours").reset_index(drop=True)

        return df_noisy

    def run_stress_test(
        self,
        patient_subset: Optional[List[int]] = None,
        intensity: float = 0.5,
        inject_missingness: bool = True,
        inject_spikes: bool = True,
        inject_jitter: bool = True,
        max_patients: int = 40
    ) -> Dict[str, Any]:
        """Runs comparative stress test across original vs noisy streams."""
        if patient_subset is None:
            splits_file = os.path.join(self.config["data"]["processed_dir"], "splits.json")
            with open(splits_file, "r") as f:
                splits = json.load(f)
            patient_subset = splits["test"][:max_patients]

        print(f"Running Noise Stress Test on {len(patient_subset)} patients (Intensity: {intensity*100:.0f}%)...")
        records_dir = self.config["data"]["records_dir"]

        orig_summaries_sw = []
        orig_summaries_pml = []
        orig_summaries_th = []
        total_hours_orig = 0.0

        noisy_summaries_sw = []
        noisy_summaries_pml = []
        noisy_summaries_th = []
        total_hours_noisy = 0.0

        for pid in patient_subset:
            filepath = os.path.join(records_dir, f"{pid}.txt")
            if not os.path.exists(filepath):
                continue
            static_info, df_obs_orig = load_patient_raw(filepath)
            outcome = self.replay_engine.labeler.get_patient_outcome(pid)["in_hospital_death"]
            max_t = float(df_obs_orig["time_hours"].max()) if not df_obs_orig.empty else 48.0

            # 1. Clean Replay
            hist_orig = self.replay_engine.replay_patient(pid, step_hours=1.0)
            total_hours_orig += max_t
            
            sw_alerts_o = sum(1 for r in hist_orig if r["alert_fired"])
            pml_alerts_o = sum(1 for r in hist_orig if r["baseline_plain_ml_alert"])
            th_alerts_o = sum(1 for r in hist_orig if r["baseline_threshold_alert"])

            orig_summaries_sw.append({"patient_id": pid, "actual_outcome": outcome, "alerts": sw_alerts_o, "alerts_lead_time": None})
            orig_summaries_pml.append({"patient_id": pid, "actual_outcome": outcome, "alerts": pml_alerts_o, "alerts_lead_time": None})
            orig_summaries_th.append({"patient_id": pid, "actual_outcome": outcome, "alerts": th_alerts_o, "alerts_lead_time": None})

            # 2. Noisy Replay
            df_obs_noisy = self.inject_noise_to_stream(
                df_obs_orig,
                intensity=intensity,
                inject_missingness=inject_missingness,
                inject_spikes=inject_spikes,
                inject_jitter=inject_jitter,
                seed=pid
            )
            hist_noisy = self.replay_engine.replay_patient(pid, step_hours=1.0, raw_obs_override=df_obs_noisy)
            total_hours_noisy += max_t

            sw_alerts_n = sum(1 for r in hist_noisy if r["alert_fired"])
            pml_alerts_n = sum(1 for r in hist_noisy if r["baseline_plain_ml_alert"])
            th_alerts_n = sum(1 for r in hist_noisy if r["baseline_threshold_alert"])

            noisy_summaries_sw.append({"patient_id": pid, "actual_outcome": outcome, "alerts": sw_alerts_n, "alerts_lead_time": None})
            noisy_summaries_pml.append({"patient_id": pid, "actual_outcome": outcome, "alerts": pml_alerts_n, "alerts_lead_time": None})
            noisy_summaries_th.append({"patient_id": pid, "actual_outcome": outcome, "alerts": th_alerts_n, "alerts_lead_time": None})

        # Calculate metrics
        m_sw_orig = compute_system_metrics(orig_summaries_sw, "alerts", total_hours_orig)
        m_sw_noisy = compute_system_metrics(noisy_summaries_sw, "alerts", total_hours_noisy)

        m_pml_orig = compute_system_metrics(orig_summaries_pml, "alerts", total_hours_orig)
        m_pml_noisy = compute_system_metrics(noisy_summaries_pml, "alerts", total_hours_noisy)

        m_th_orig = compute_system_metrics(orig_summaries_th, "alerts", total_hours_orig)
        m_th_noisy = compute_system_metrics(noisy_summaries_th, "alerts", total_hours_noisy)

        comparison = {
            "noise_settings": {
                "intensity_pct": round(intensity * 100, 1),
                "inject_missingness": inject_missingness,
                "inject_spikes": inject_spikes,
                "inject_jitter": inject_jitter,
                "evaluated_patients": len(orig_summaries_sw)
            },
            "comparison": [
                {
                    "system": "SilentWindow",
                    "clean_alerts_per_day": m_sw_orig["alerts_per_patient_day"],
                    "noisy_alerts_per_day": m_sw_noisy["alerts_per_patient_day"],
                    "clean_false_alerts": m_sw_orig["false_alerts"],
                    "noisy_false_alerts": m_sw_noisy["false_alerts"],
                    "false_alert_increase_pct": round(
                        ((m_sw_noisy["false_alerts"] - m_sw_orig["false_alerts"]) / max(1, m_sw_orig["false_alerts"])) * 100, 1
                    ),
                    "clean_sensitivity": m_sw_orig["sensitivity_recall"],
                    "noisy_sensitivity": m_sw_noisy["sensitivity_recall"]
                },
                {
                    "system": "Plain ML (Instant Risk)",
                    "clean_alerts_per_day": m_pml_orig["alerts_per_patient_day"],
                    "noisy_alerts_per_day": m_pml_noisy["alerts_per_patient_day"],
                    "clean_false_alerts": m_pml_orig["false_alerts"],
                    "noisy_false_alerts": m_pml_noisy["false_alerts"],
                    "false_alert_increase_pct": round(
                        ((m_pml_noisy["false_alerts"] - m_pml_orig["false_alerts"]) / max(1, m_pml_orig["false_alerts"])) * 100, 1
                    ),
                    "clean_sensitivity": m_pml_orig["sensitivity_recall"],
                    "noisy_sensitivity": m_pml_noisy["sensitivity_recall"]
                },
                {
                    "system": "Simplified Threshold",
                    "clean_alerts_per_day": m_th_orig["alerts_per_patient_day"],
                    "noisy_alerts_per_day": m_th_noisy["alerts_per_patient_day"],
                    "clean_false_alerts": m_th_orig["false_alerts"],
                    "noisy_false_alerts": m_th_noisy["false_alerts"],
                    "false_alert_increase_pct": round(
                        ((m_th_noisy["false_alerts"] - m_th_orig["false_alerts"]) / max(1, m_th_orig["false_alerts"])) * 100, 1
                    ),
                    "clean_sensitivity": m_th_orig["sensitivity_recall"],
                    "noisy_sensitivity": m_th_noisy["sensitivity_recall"]
                }
            ]
        }

        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "noise_stress_results.json"), "w") as f:
            json.dump(comparison, f, indent=2)

        return comparison

if __name__ == "__main__":
    tester = NoiseStressTester()
    res = tester.run_stress_test(intensity=0.5, max_patients=30)
    print("\n--- NOISE STRESS TEST RESULTS ---")
    for r in res["comparison"]:
        print(f"{r['system']}: False Alerts {r['clean_false_alerts']} -> {r['noisy_false_alerts']} ({r['false_alert_increase_pct']:+}% increase)")
