"""
router.py - FastAPI API Endpoints for SilentWindow Clinical Decision Support.
"""

import os
import json
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np

from ml.preprocessing import load_config
from ml.explainability import ClinicalExplainer
from ml.noise_test import NoiseStressTester

router = APIRouter(prefix="/api")

# Lazy-loaded singletons
_explainer = None
_noise_tester = None

def get_explainer():
    global _explainer
    if _explainer is None:
        _explainer = ClinicalExplainer("configs/config.yaml")
    return _explainer

def get_noise_tester():
    global _noise_tester
    if _noise_tester is None:
        _noise_tester = NoiseStressTester("configs/config.yaml")
    return _noise_tester

@router.get("/overview")
def get_overview():
    """Returns top-level cohort overview metrics and clinical disclaimer."""
    eval_file = os.path.join("results", "evaluation_summary.json")
    if not os.path.exists(eval_file):
        raise HTTPException(status_code=404, detail="Evaluation results not found. Run ml/evaluation.py first.")
    
    with open(eval_file, "r") as f:
        data = json.load(f)

    # Load test patient states
    records_file = os.path.join("results", "test_eval_records.parquet")
    df = pd.read_parquet(records_file)
    
    # Latest state per patient
    latest_rows = df.sort_values("time_hours").groupby("patient_id").last().reset_index()
    watch_count = int((latest_rows["state"] == "WATCH").sum())
    alert_count = int((latest_rows["state"] == "ALERT").sum())
    stable_count = int((latest_rows["state"] == "STABLE").sum())

    return {
        "disclaimer": data["disclaimer"],
        "total_patients": data["test_patients_count"],
        "total_monitoring_hours": data["total_monitoring_hours"],
        "current_states": {
            "stable": stable_count,
            "watch": watch_count,
            "alert": alert_count
        },
        "systems": data["systems"],
        "auc_metrics": data["auc_metrics"]
    }

@router.get("/ward")
def get_ward_monitor():
    """Returns ward monitor table for all test patients."""
    records_file = os.path.join("results", "test_eval_records.parquet")
    if not os.path.exists(records_file):
        raise HTTPException(status_code=404, detail="Evaluation records not found.")

    df = pd.read_parquet(records_file)
    ward_rows = []

    for pid, pdf in df.groupby("patient_id"):
        pdf_sorted = pdf.sort_values("time_hours")
        last_row = pdf_sorted.iloc[-1]
        
        # Determine trend over last 3 hours
        if len(pdf_sorted) >= 3:
            r_prev = pdf_sorted.iloc[-3]["calibrated_risk"]
            r_now = last_row["calibrated_risk"]
            diff = r_now - r_prev
            if diff > 0.05:
                trend = "increasing"
            elif diff < -0.05:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        alert_count = int(pdf["alert_fired"].sum())

        ward_rows.append({
            "patient_id": int(pid),
            "state": str(last_row["state"]),
            "risk": round(float(last_row["calibrated_risk"]), 3),
            "raw_risk": round(float(last_row["raw_risk"]), 3),
            "evidence_score": round(float(last_row["evidence_score"]), 3),
            "last_update_hours": round(float(last_row["time_hours"]), 1),
            "trend": trend,
            "total_alerts": alert_count,
            "actual_outcome": int(last_row["actual_outcome"]),
            "sofa": float(last_row["sofa"]),
            "saps": float(last_row["saps"]),
            "length_of_stay": float(last_row["length_of_stay"]),
            "mean_credibility": round(float(last_row["mean_credibility"]), 2),
            "suspicious_readings": int(last_row["suspicious_count"])
        })

    # Sort: Alerts first, then Watches, then by highest risk
    state_order = {"ALERT": 0, "WATCH": 1, "STABLE": 2}
    ward_rows.sort(key=lambda x: (state_order.get(x["state"], 3), -x["evidence_score"], -x["risk"]))

    return {"patients": ward_rows}

@router.get("/patient/{patient_id}/timeline")
def get_patient_timeline(patient_id: int):
    """Returns chronological timeline for a specific patient."""
    records_file = os.path.join("results", "test_eval_records.parquet")
    if not os.path.exists(records_file):
        raise HTTPException(status_code=404, detail="Evaluation records not found.")

    df = pd.read_parquet(records_file)
    pdf = df[df["patient_id"] == patient_id].sort_values("time_hours")
    if pdf.empty:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")

    timeline = []
    for _, row in pdf.iterrows():
        raw_v = row["vitals"] if isinstance(row["vitals"], dict) else {}
        clean_v = {k: (round(float(v), 2) if pd.notna(v) else None) for k, v in raw_v.items()}
        
        reason = str(row["reason"]) if pd.notna(row["reason"]) else None
        lead_time = float(row["estimated_lead_time_hours"]) if pd.notna(row["estimated_lead_time_hours"]) else None

        timeline.append({
            "time_hours": round(float(row["time_hours"]), 2),
            "raw_risk": round(float(row["raw_risk"]), 4),
            "calibrated_risk": round(float(row["calibrated_risk"]), 4),
            "evidence_score": round(float(row["evidence_score"]), 4),
            "state": str(row["state"]),
            "alert_fired": bool(row["alert_fired"]),
            "suppressed": bool(row["suppressed"]),
            "refractory_remaining": round(float(row["refractory_remaining"]), 2) if pd.notna(row["refractory_remaining"]) else 0.0,
            "reason": reason,
            "mean_credibility": round(float(row["mean_credibility"]), 3),
            "suspicious_count": int(row["suspicious_count"]) if pd.notna(row["suspicious_count"]) else 0,
            "baseline_threshold_alert": bool(row["baseline_threshold_alert"]),
            "baseline_plain_ml_alert": bool(row["baseline_plain_ml_alert"]),
            "vitals": clean_v,
            "estimated_lead_time_hours": lead_time
        })

    last_row = pdf.iloc[-1]
    return {
        "patient_id": patient_id,
        "actual_outcome": int(last_row["actual_outcome"]),
        "sofa": float(last_row["sofa"]) if pd.notna(last_row["sofa"]) else None,
        "saps": float(last_row["saps"]) if pd.notna(last_row["saps"]) else None,
        "length_of_stay": float(last_row["length_of_stay"]) if pd.notna(last_row["length_of_stay"]) else None,
        "total_monitoring_hours": float(pdf["time_hours"].max()),
        "total_alerts": int(pdf["alert_fired"].sum()),
        "threshold_baseline_alerts": int(pdf["baseline_threshold_alert"].sum()),
        "plain_ml_alerts": int(pdf["baseline_plain_ml_alert"].sum()),
        "lead_time_disclaimer": "Lead time is estimated relative to the available outcome/proxy because an exact event timestamp is unavailable.",
        "timeline": timeline
    }

@router.get("/patient/{patient_id}/explanation")
def get_patient_explanation(patient_id: int, eval_time: Optional[float] = None):
    """Returns SHAP feature breakdown and Layer 1 trust audit for patient."""
    try:
        explainer = get_explainer()
        
        # If eval_time not provided or invalid, use time of first alert or max time
        try:
            eval_time = float(eval_time) if eval_time is not None else None
        except (ValueError, TypeError):
            eval_time = None

        if eval_time is None:
            records_file = os.path.join("results", "test_eval_records.parquet")
            df = pd.read_parquet(records_file)
            pdf = df[df["patient_id"] == patient_id]
            if not pdf.empty:
                alerts = pdf[pdf["alert_fired"] == True]
                if not alerts.empty:
                    eval_time = float(alerts["time_hours"].iloc[0])
                else:
                    eval_time = float(pdf["time_hours"].max())
            else:
                eval_time = 24.0

        explanation = explainer.explain_patient_at_time(patient_id, eval_time)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
def get_performance_metrics():
    """Returns evaluation metrics, ROC/PR curves, and baseline trade-offs."""
    eval_file = os.path.join("results", "evaluation_summary.json")
    if not os.path.exists(eval_file):
        raise HTTPException(status_code=404, detail="Evaluation results not found.")
    
    with open(eval_file, "r") as f:
        data = json.load(f)
    return data

@router.get("/ablation")
def get_ablation_results():
    """Returns results of the 4-component ablation study."""
    abl_file = os.path.join("results", "ablation_results.json")
    if not os.path.exists(abl_file):
        raise HTTPException(status_code=404, detail="Ablation results not found.")
    
    with open(abl_file, "r") as f:
        data = json.load(f)
    return data

class NoiseStressRequest(BaseModel):
    intensity: float = 0.5
    inject_missingness: bool = True
    inject_spikes: bool = True
    inject_jitter: bool = True
    max_patients: int = 25

@router.get("/noise-lab")
def get_noise_results():
    """Returns cached noise stress test results."""
    noise_file = os.path.join("results", "noise_stress_results.json")
    if os.path.exists(noise_file):
        with open(noise_file, "r") as f:
            return json.load(f)
    # If not run yet, run default
    tester = get_noise_tester()
    return tester.run_stress_test(intensity=0.5, max_patients=25)

@router.post("/noise-lab/run")
def run_noise_lab(req: NoiseStressRequest):
    """Runs interactive noise stress test with user-selected parameters."""
    tester = get_noise_tester()
    results = tester.run_stress_test(
        intensity=req.intensity,
        inject_missingness=req.inject_missingness,
        inject_spikes=req.inject_spikes,
        inject_jitter=req.inject_jitter,
        max_patients=min(req.max_patients, 35)
    )
    return results


from fastapi import UploadFile, File, Form
from google import genai
from google.genai import types
from google.genai.errors import APIError

@router.post("/chat")
async def chat_with_ai(
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
        if not api_key:
            return {"reply": "Error: GEMINI_API_KEY is not set in the environment."}
            
        client = genai.Client(api_key=api_key)
        
        prompt = ""
        if file:
            content = await file.read()
            csv_text = content.decode('utf-8')
            prompt += f"Here is the patient CSV data:\n{csv_text}\n\n"
            
        if message:
            prompt += message
        else:
            prompt += "Please analyze this patient data."
            
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return {"reply": response.text}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"reply": f"An error occurred: {str(e)}"}


import requests

@router.post("/telegram_alert")
async def send_telegram_alert(patient_id: int):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        return {"success": False, "message": "Telegram credentials not configured in environment."}
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    text = f"dY"" CRITICAL ALERT: Patient {patient_id} has entered the ALERT state! Immediate review recommended."
    
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": text})
        if resp.status_code == 200:
            return {"success": True, "message": "Alert sent via Telegram."}
        else:
            return {"success": False, "message": f"Telegram API error: {resp.text}"}
    except Exception as e:
        return {"success": False, "message": str(e)}
