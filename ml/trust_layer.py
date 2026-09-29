"""
trust_layer.py - Layer 1: Trust-Aware Signal Processing.

Assesses telemetry observations for sensor artifacts, implausibility, and sudden spikes
WITHOUT deleting data. Produces:
- reading_value
- reading_validity
- credibility_score [0.0, 1.0]
- reason_for_downweighting
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class ObservationAssessment:
    time_hours: float
    parameter: str
    original_value: float
    valid: bool
    credibility_score: float
    reason: Optional[str] = None

class TrustAwareSignalProcessor:
    def __init__(self, config: dict):
        self.config = config
        trust_cfg = config.get("trust_layer", {})
        self.plausible_ranges = trust_cfg.get("plausible_ranges", {})
        self.max_jumps = trust_cfg.get("max_jump_per_hour", {})
        self.penalties = trust_cfg.get("penalties", {
            "implausible": 0.85,
            "jump": 0.60,
            "discordant": 0.40
        })

    def map_parameter_alias(self, param: str) -> str:
        """Maps non-invasive / invasive blood pressure aliases to standard names."""
        mapping = {
            "NISysABP": "SysABP",
            "NIDiasABP": "DiasABP",
            "NIMAP": "MAP"
        }
        return mapping.get(param, param)

    def assess_observation(
        self,
        time_hours: float,
        parameter: str,
        value: float,
        history: List[Tuple[float, float]], # List of (time_hours, value)
        cross_param_context: Optional[Dict[str, float]] = None # contemporaneous other vitals
    ) -> ObservationAssessment:
        """
        Assesses credibility of a single observation based on:
        1. Range plausibility
        2. Temporal jump vs recent history
        3. Multi-parameter physiological consistency
        """
        std_param = self.map_parameter_alias(parameter)
        credibility = 1.0
        reasons = []
        is_valid = True

        # 1. Physiological plausibility check
        if std_param in self.plausible_ranges:
            low, high = self.plausible_ranges[std_param]
            if value < low or value > high:
                is_valid = False
                penalty = self.penalties.get("implausible", 0.85)
                credibility *= (1.0 - penalty)
                reasons.append(f"Implausible range: {value:.1f} outside [{low}, {high}]")

        # 2. Sudden jump artifact detection
        if history and std_param in self.max_jumps:
            last_t, last_val = history[-1]
            dt = max(0.05, time_hours - last_t)  # avoid div by zero
            # Only test jumps if last reading was within recent 3 hours
            if dt <= 3.0:
                jump_rate = abs(value - last_val) / dt
                max_allowed = self.max_jumps[std_param]
                if jump_rate > max_allowed:
                    penalty = self.penalties.get("jump", 0.60)
                    credibility *= (1.0 - penalty)
                    reasons.append(f"Sudden jump: {last_val:.1f} -> {value:.1f} ({jump_rate:.1f}/hr > {max_allowed}/hr)")

        # 3. Cross-parameter physiological consistency
        if cross_param_context:
            if std_param == "SysABP" and "DiasABP" in cross_param_context:
                dias = cross_param_context["DiasABP"]
                if value <= dias:
                    credibility *= (1.0 - self.penalties.get("discordant", 0.40))
                    reasons.append(f"Discordant vitals: SysABP ({value:.1f}) <= DiasABP ({dias:.1f})")
            elif std_param == "DiasABP" and "SysABP" in cross_param_context:
                sys = cross_param_context["SysABP"]
                if value >= sys:
                    credibility *= (1.0 - self.penalties.get("discordant", 0.40))
                    reasons.append(f"Discordant vitals: DiasABP ({value:.1f}) >= SysABP ({sys:.1f})")

        # Ensure bounds [0.05, 1.0]
        credibility = float(np.clip(credibility, 0.05, 1.0))
        reason_str = " | ".join(reasons) if reasons else None

        return ObservationAssessment(
            time_hours=time_hours,
            parameter=parameter,
            original_value=value,
            valid=is_valid,
            credibility_score=round(credibility, 3),
            reason=reason_str
        )

    def process_patient_stream(self, df_obs) -> List[ObservationAssessment]:
        """Processes a patient's full observation stream chronologically."""
        assessments = []
        # Maintain history per parameter: dict of param -> list of (t, val)
        param_histories = {}
        
        # Sort chronologically
        df_sorted = df_obs.sort_values(by="time_hours").reset_index(drop=True)
        
        # Group near-contemporaneous observations (within 0.1 hour / 6 min) for cross-consistency
        for idx, row in df_sorted.iterrows():
            t = float(row["time_hours"])
            param = str(row["parameter"])
            val = float(row["value"])
            std_param = self.map_parameter_alias(param)
            
            # Find contemporaneous vitals in recent 0.1 hr
            cross_context = {}
            for other_p in ["SysABP", "DiasABP", "MAP", "HR"]:
                if other_p != std_param and other_p in param_histories and param_histories[other_p]:
                    last_t, last_v = param_histories[other_p][-1]
                    if abs(t - last_t) <= 0.15:
                        cross_context[other_p] = last_v

            hist = param_histories.get(std_param, [])
            assessment = self.assess_observation(t, param, val, hist, cross_context)
            assessments.append(assessment)
            
            # Update history only with valid or moderately credible readings to prevent noise corruption
            if assessment.credibility_score > 0.30:
                if std_param not in param_histories:
                    param_histories[std_param] = []
                param_histories[std_param].append((t, val))

        return assessments
