"""
evidence_accumulator.py - Layer 3: Sequential Evidence Accumulator.

Principles:
- Do not trigger alerts purely based on instantaneous risk probability.
- Sequentially accumulate evidence over time weighted by credibility.
- Require persistent evidence of deterioration across multiple time steps.
- Provide two distinct operational levels: WATCH (yellow) and ALERT (red).
- Enforce alert budget and refractory period to combat alert fatigue.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class AccumulatorState:
    time_hours: float
    raw_risk: float
    calibrated_risk: float
    credibility_weight: float
    evidence_score: float
    persistence_count: int
    state: str  # "STABLE", "WATCH", "ALERT"
    alert_fired: bool
    suppressed_by_refractory: bool = False
    refractory_hours_remaining: float = 0.0
    reason: Optional[str] = None

class SequentialEvidenceAccumulator:
    """
    CUSUM-inspired sequential evidence accumulator with credibility weighting
    and clinical refractory control.
    """
    def __init__(self, config: dict):
        accum_cfg = config.get("evidence_accumulator", {})
        self.r0 = accum_cfg.get("neutral_risk_threshold", 0.22)
        self.decay_rate = accum_cfg.get("decay_rate", 0.88)
        self.persistence_boost = accum_cfg.get("persistence_boost_factor", 1.25)
        self.watch_threshold = accum_cfg.get("watch_threshold", 0.40)
        self.alert_threshold = accum_cfg.get("alert_threshold", 0.75)
        self.refractory_period = accum_cfg.get("refractory_period_hours", 6.0)
        self.max_alerts = accum_cfg.get("max_alerts_per_patient", 3)

        self.reset()

    def reset(self):
        """Resets accumulator for a new patient stream."""
        self.evidence_score: float = 0.0
        self.persistence_count: int = 0
        self.last_alert_time: Optional[float] = None
        self.total_alerts_fired: int = 0
        self.history: List[AccumulatorState] = []

    def update(
        self,
        time_hours: float,
        calibrated_risk: float,
        raw_risk: float,
        credibility_weight: float = 1.0,
        enable_trust: bool = True
    ) -> AccumulatorState:
        """
        Updates evidence accumulator for a single chronological time step.

        Parameters:
            time_hours: current evaluation timestamp in hours
            calibrated_risk: calibrated risk probability p(t)
            raw_risk: raw risk probability
            credibility_weight: Layer 1 trust score in [0.05, 1.0]
            enable_trust: ablation flag (if False, credibility_weight treated as 1.0)
        """
        eff_cred = credibility_weight if enable_trust else 1.0
        eff_cred = float(np.clip(eff_cred, 0.05, 1.0))
        risk_signal = calibrated_risk - self.r0

        # Evidence increment calculation
        if risk_signal > 0:
            self.persistence_count += 1
            # Progressive persistence factor: 1.0 at k=1, scaling up with sustained elevated risk
            p_factor = 1.0 + min(1.0, (self.persistence_boost - 1.0) * (self.persistence_count - 1))
            delta_e = risk_signal * eff_cred * p_factor
            self.evidence_score += delta_e
        else:
            self.persistence_count = 0
            # Exponential decay towards 0
            self.evidence_score = max(0.0, self.evidence_score * self.decay_rate - abs(risk_signal) * 0.15)

        self.evidence_score = round(float(self.evidence_score), 4)

        # Refractory period evaluation
        refractory_active = False
        remaining_refractory = 0.0
        if self.last_alert_time is not None:
            time_since_alert = time_hours - self.last_alert_time
            if time_since_alert < self.refractory_period:
                refractory_active = True
                remaining_refractory = round(self.refractory_period - time_since_alert, 2)

        # State determination
        state = "STABLE"
        alert_fired = False
        suppressed = False
        reason = None

        if self.evidence_score >= self.alert_threshold:
            if refractory_active:
                state = "ALERT"
                suppressed = True
                reason = f"Alert suppressed: within refractory period ({remaining_refractory:.1f}h remaining)"
            elif self.total_alerts_fired >= self.max_alerts:
                state = "ALERT"
                suppressed = True
                reason = f"Alert suppressed: max budget reached ({self.max_alerts} alerts fired)"
            else:
                state = "ALERT"
                alert_fired = True
                self.last_alert_time = time_hours
                self.total_alerts_fired += 1
                reason = f"High-confidence alert: evidence ({self.evidence_score:.2f}) >= {self.alert_threshold:.2f}"
        elif self.evidence_score >= self.watch_threshold:
            state = "WATCH"
            reason = f"Watch advisory: evidence ({self.evidence_score:.2f}) >= {self.watch_threshold:.2f}"
        else:
            state = "STABLE"

        snapshot = AccumulatorState(
            time_hours=time_hours,
            raw_risk=round(float(raw_risk), 4),
            calibrated_risk=round(float(calibrated_risk), 4),
            credibility_weight=round(eff_cred, 3),
            evidence_score=self.evidence_score,
            persistence_count=self.persistence_count,
            state=state,
            alert_fired=alert_fired,
            suppressed_by_refractory=suppressed,
            refractory_hours_remaining=remaining_refractory,
            reason=reason
        )
        self.history.append(snapshot)
        return snapshot
