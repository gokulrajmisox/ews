"""
data_inspection.py - Inspects the ICU telemetry dataset (PhysioNet Challenge 2012)
Outputs detailed summary:
- Patient files, record count
- Columns, data types
- Vitals and lab parameter frequencies and missingness
- Observation time ranges
- Outcome distributions and class balance
- Output a comprehensive inspection report
"""

import os
import glob
import json
import pandas as pd
import numpy as np

def inspect_dataset(data_dir="data/raw", output_report="results/data_summary.json"):
    outcomes_file = os.path.join(data_dir, "Outcomes-train.txt")
    records_dir = os.path.join(data_dir, "set-a")
    
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    
    report = {
        "dataset_name": "PhysioNet / Computing in Cardiology Challenge 2012 (Set A)",
        "source_files": {
            "outcomes_file": outcomes_file,
            "records_dir": records_dir
        },
        "patients": {},
        "outcomes": {},
        "parameters": {},
        "sampling": {},
        "notes": []
    }
    
    # 1. Outcomes inspection
    if os.path.exists(outcomes_file):
        df_outcomes = pd.read_csv(outcomes_file)
        report["outcomes"]["columns"] = list(df_outcomes.columns)
        report["outcomes"]["num_patients"] = len(df_outcomes)
        death_counts = df_outcomes["In-hospital_death"].value_counts().to_dict()
        report["outcomes"]["in_hospital_death_counts"] = {str(k): int(v) for k, v in death_counts.items()}
        mortality_rate = float(df_outcomes["In-hospital_death"].mean())
        report["outcomes"]["mortality_rate"] = round(mortality_rate, 4)
        report["outcomes"]["length_of_stay_summary"] = {
            "min": float(df_outcomes["Length_of_stay"].min()),
            "median": float(df_outcomes["Length_of_stay"].median()),
            "max": float(df_outcomes["Length_of_stay"].max())
        }
        report["outcomes"]["sofa_summary"] = {
            "median": float(df_outcomes["SOFA"].median()),
            "mean": float(df_outcomes["SOFA"].mean())
        }
    else:
        report["notes"].append("Outcomes-train.txt not found")
        
    # 2. Inspect patient files
    patient_files = glob.glob(os.path.join(records_dir, "*.txt"))
    report["patients"]["total_patient_files"] = len(patient_files)
    
    # Sample up to 200 patient files to get statistical distribution of parameters & sampling
    param_counts = {}
    param_values = {}
    time_points_per_patient = []
    max_time_per_patient = []
    
    sample_files = patient_files[:250] if len(patient_files) > 250 else patient_files
    
    for f in sample_files:
        try:
            df_p = pd.read_csv(f)
            # Parse time HH:MM into minutes/hours
            def parse_time(t_str):
                parts = str(t_str).split(":")
                return int(parts[0]) * 60 + int(parts[1])
            
            times = df_p["Time"].apply(parse_time)
            time_points_per_patient.append(len(times))
            max_time_per_patient.append(times.max() / 60.0) # hours
            
            for param, val in zip(df_p["Parameter"], df_p["Value"]):
                param_counts[param] = param_counts.get(param, 0) + 1
                if param not in param_values:
                    param_values[param] = []
                if len(param_values[param]) < 1000:
                    try:
                        param_values[param].append(float(val))
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            continue
            
    # Compile parameter stats
    param_stats = {}
    for p, vals in param_values.items():
        if len(vals) > 0:
            arr = np.array(vals)
            param_stats[p] = {
                "count_in_sample": param_counts.get(p, 0),
                "min": float(np.min(arr)),
                "p01": float(np.percentile(arr, 1)),
                "p50": float(np.percentile(arr, 50)),
                "p99": float(np.percentile(arr, 99)),
                "max": float(np.max(arr)),
            }
            
    report["parameters"] = param_stats
    report["sampling"]["mean_records_per_patient"] = float(np.mean(time_points_per_patient)) if time_points_per_patient else 0
    report["sampling"]["median_records_per_patient"] = float(np.median(time_points_per_patient)) if time_points_per_patient else 0
    report["sampling"]["mean_max_hours"] = float(np.mean(max_time_per_patient)) if max_time_per_patient else 0
    report["sampling"]["median_max_hours"] = float(np.median(max_time_per_patient)) if max_time_per_patient else 0
    
    report["notes"].append("Lead time is estimated relative to the available outcome/proxy because an exact event timestamp is unavailable.")
    
    with open(output_report, "w") as out:
        json.dump(report, out, indent=2)
        
    print(f"Data inspection complete! Report written to {output_report}")
    print(f"Total patient files: {len(patient_files)}")
    if "outcomes" in report and "num_patients" in report["outcomes"]:
        print(f"Total outcome rows: {report['outcomes']['num_patients']}")
        print(f"Mortality rate: {report['outcomes']['mortality_rate'] * 100:.1f}%")
    print(f"Top parameters tracked: {list(param_stats.keys())[:15]}")
    return report

if __name__ == "__main__":
    inspect_dataset()
