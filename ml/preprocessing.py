"""
preprocessing.py - Data ingestion, patient splitting, and raw observation sequencing.

STRICT RULE: Patient-level splitting. No patient's observations can appear in multiple splits.
"""

import os
import glob
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_time_to_hours(time_str: str) -> float:
    """Converts 'HH:MM' string to float hours."""
    try:
        parts = str(time_str).strip().split(":")
        return float(parts[0]) + float(parts[1]) / 60.0
    except Exception:
        return 0.0

def load_patient_raw(file_path: str) -> Tuple[Dict, pd.DataFrame]:
    """
    Parses a single patient's raw telemetry text file.
    Returns:
      static_info: dict of {Age, Gender, Height, Weight, ICUType, RecordID}
      df_obs: DataFrame with columns [time_hours, parameter, value]
    """
    df = pd.read_csv(file_path)
    
    static_params = {"RecordID", "Age", "Gender", "Height", "ICUType", "Weight"}
    static_info = {}
    obs_records = []
    
    for _, row in df.iterrows():
        param = str(row["Parameter"]).strip()
        val_str = str(row["Value"]).strip()
        try:
            val = float(val_str)
        except ValueError:
            continue
            
        t_hours = parse_time_to_hours(row["Time"])
        
        if param in static_params and t_hours == 0.0 and param not in static_info:
            static_info[param] = val
        else:
            obs_records.append({
                "time_hours": t_hours,
                "parameter": param,
                "value": val
            })
            
    df_obs = pd.DataFrame(obs_records)
    if not df_obs.empty:
        df_obs = df_obs.sort_values(by="time_hours").reset_index(drop=True)
    else:
        df_obs = pd.DataFrame(columns=["time_hours", "parameter", "value"])
        
    return static_info, df_obs

def get_patient_splits(config: dict) -> Tuple[List[int], List[int], List[int], pd.DataFrame]:
    """
    Performs patient-level train/val/test splitting with zero row leakage.
    Saves split manifest to data/processed/splits.json.
    """
    outcomes_file = config["data"]["outcomes_file"]
    df_outcomes = pd.read_csv(outcomes_file)
    
    max_patients = config["data"].get("max_patients", None)
    if max_patients and max_patients < len(df_outcomes):
        # Stratified sampling to preserve mortality rate
        df_pos = df_outcomes[df_outcomes["In-hospital_death"] == 1]
        df_neg = df_outcomes[df_outcomes["In-hospital_death"] == 0]
        pos_n = int(max_patients * len(df_pos) / len(df_outcomes))
        neg_n = max_patients - pos_n
        
        sample_pos = df_pos.sample(n=pos_n, random_state=config["data"]["random_seed"])
        sample_neg = df_neg.sample(n=neg_n, random_state=config["data"]["random_seed"])
        df_outcomes = pd.concat([sample_pos, sample_neg]).sample(frac=1.0, random_state=config["data"]["random_seed"]).reset_index(drop=True)
        
    all_patients = df_outcomes["RecordID"].tolist()
    labels = df_outcomes["In-hospital_death"].tolist()
    
    # Stratified split by patient
    np.random.seed(config["data"]["random_seed"])
    pos_ids = [pid for pid, l in zip(all_patients, labels) if l == 1]
    neg_ids = [pid for pid, l in zip(all_patients, labels) if l == 0]
    
    np.random.shuffle(pos_ids)
    np.random.shuffle(neg_ids)
    
    train_frac = config["data"]["train_split"]
    val_frac = config["data"]["val_split"]
    
    def split_group(p_list):
        n = len(p_list)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_p = p_list[:n_train]
        val_p = p_list[n_train:n_train + n_val]
        test_p = p_list[n_train + n_val:]
        return train_p, val_p, test_p
        
    tr_pos, val_pos, te_pos = split_group(pos_ids)
    tr_neg, val_neg, te_neg = split_group(neg_ids)
    
    train_ids = tr_pos + tr_neg
    val_ids = val_pos + val_neg
    test_ids = te_pos + te_neg
    
    # Rigorous check: no patient in multiple splits
    set_tr = set(train_ids)
    set_val = set(val_ids)
    set_te = set(test_ids)
    
    assert len(set_tr.intersection(set_val)) == 0, "DATA LEAK: Patient in both train and val!"
    assert len(set_tr.intersection(set_te)) == 0, "DATA LEAK: Patient in both train and test!"
    assert len(set_val.intersection(set_te)) == 0, "DATA LEAK: Patient in both val and test!"
    
    os.makedirs(config["data"]["processed_dir"], exist_ok=True)
    manifest = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_test": len(test_ids),
        "mortality_train": float(df_outcomes[df_outcomes["RecordID"].isin(train_ids)]["In-hospital_death"].mean()),
        "mortality_val": float(df_outcomes[df_outcomes["RecordID"].isin(val_ids)]["In-hospital_death"].mean()),
        "mortality_test": float(df_outcomes[df_outcomes["RecordID"].isin(test_ids)]["In-hospital_death"].mean()),
    }
    
    import json
    with open(os.path.join(config["data"]["processed_dir"], "splits.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    return train_ids, val_ids, test_ids, df_outcomes

if __name__ == "__main__":
    cfg = load_config()
    tr, val, te, df_o = get_patient_splits(cfg)
    print(f"Patient-level split complete: Train={len(tr)}, Val={len(val)}, Test={len(te)}")
    print(f"Mortality in test set: {df_o[df_o['RecordID'].isin(te)]['In-hospital_death'].mean() * 100:.2f}%")
