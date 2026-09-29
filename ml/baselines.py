"""
baselines.py - Transparent Baseline Comparators for Clinical Deterioration.

Baseline A: Simplified Threshold Baseline (NEWS2-inspired, fixed threshold rules)
Notice: Labeled strictly as "Simplified threshold baseline" rather than official NEWS2
because full NEWS2 parameters (AVPU, Room Air, supplemental O2 scale) are not completely present.

Baseline B: Plain ML (Instantaneous ML probability thresholding)
Uses identical features and XGBoost model but triggers an alert whenever risk > threshold,
demonstrating alert storms and spike vulnerability without evidence accumulation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class BaselineDecision:
    time_hours: float
    state: str  # "STABLE" or "ALERT"
    alert_fired: bool
    score: float
    trigger_reasons: List[str]

class SimplifiedThresholdBaseline:
    """
    Fixed-threshold vital signs comparator.
    Fires an alert if >= 2 vital signs cross abnormal physiological boundaries,
    or if any single vital crosses an extreme critical boundary.
    """
    def __init__(self, config: dict):
        base_cfg = config.get("baselines", {}).get("thresholds", {})
        self.hr_high = base_cfg.get("HR_high", 110.0)
        self.hr_low = base_cfg.get("HR_low", 50.0)
        self.sys_low = base_cfg.get("SysABP_low", 90.0)
        self.rr_high = base_cfg.get("RespRate_high", 25.0)
        self.rr_low = base_cfg.get("RespRate_low", 8.0)
        self.temp_high = base_cfg.get("Temp_high", 38.5)
        self.temp_low = base_cfg.get("Temp_low", 35.5)

    def evaluate(self, features: dict, time_hours: float) -> BaselineDecision:
        reasons = []
        abnormal_count = 0

        # HR check
        hr = features.get("HR_last_value", np.nan)
        if not np.isnan(hr):
            if hr > self.hr_high:
                abnormal_count += 1
                reasons.append(f"Tachycardia (HR {hr:.0f} > {self.hr_high:.0f})")
            elif hr < self.hr_low:
                abnormal_count += 1
                reasons.append(f"Bradycardia (HR {hr:.0f} < {self.hr_low:.0f})")

        # SysABP check
        sys = features.get("SysABP_last_value", np.nan)
        if not np.isnan(sys):
            if sys < self.sys_low:
                abnormal_count += 1
                reasons.append(f"Hypotension (SysABP {sys:.0f} < {self.sys_low:.0f})")

        # RespRate check
        rr = features.get("RespRate_last_value", np.nan)
        if not np.isnan(rr):
            if rr > self.rr_high:
                abnormal_count += 1
                reasons.append(f"Tachypnea (RR {rr:.0f} > {self.rr_high:.0f})")
            elif rr < self.rr_low:
                abnormal_count += 1
                reasons.append(f"Bradypnea (RR {rr:.0f} < {self.rr_low:.0f})")

        # Temp check
        temp = features.get("Temp_last_value", np.nan)
        if not np.isnan(temp):
            if temp > self.temp_high:
                abnormal_count += 1
                reasons.append(f"Hyperthermia (Temp {temp:.1f} > {self.temp_high:.1f})")
            elif temp < self.temp_low:
                abnormal_count += 1
                reasons.append(f"Hypothermia (Temp {temp:.1f} < {self.temp_low:.1f})")

        is_alert = (abnormal_count >= 2)
        state = "ALERT" if is_alert else "STABLE"

        return BaselineDecision(
            time_hours=time_hours,
            state=state,
            alert_fired=is_alert,
            score=float(abnormal_count),
            trigger_reasons=reasons
        )

class PlainMLBaseline:
    """
    Direct probabilistic threshold baseline without sequential evidence accumulation.
    Alerts immediately whenever risk probability > threshold (default 0.50).
    """
    def __init__(self, config: dict):
        base_cfg = config.get("baselines", {})
        self.risk_threshold = base_cfg.get("plain_ml_risk_threshold", 0.50)

    def evaluate(self, calibrated_risk: float, time_hours: float) -> BaselineDecision:
        is_alert = (calibrated_risk >= self.risk_threshold)
        reasons = [f"Risk ({calibrated_risk:.2f}) >= threshold ({self.risk_threshold:.2f})"] if is_alert else []

        return BaselineDecision(
            time_hours=time_hours,
            state="ALERT" if is_alert else "STABLE",
            alert_fired=is_alert,
            score=float(calibrated_risk),
            trigger_reasons=reasons
        )
