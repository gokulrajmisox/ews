"""Incremental inference for the live-style demonstration API.

This module intentionally keeps state in memory. It is suitable for a local demo and
shadow-mode prototype, not a multi-worker production deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.evidence_accumulator import SequentialEvidenceAccumulator
from ml.features import FeatureExtractor
from ml.preprocessing import load_config
from ml.trust_layer import ObservationAssessment, TrustAwareSignalProcessor


class StreamingInferenceError(RuntimeError):
    """Raised when incremental inference cannot be completed safely."""


@dataclass
class PatientStreamState:
    patient_id: int
    static_info: dict[str, float]
    assessments: list[ObservationAssessment] = field(default_factory=list)
    histories: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    accumulator: SequentialEvidenceAccumulator | None = None
    last_time_hours: float | None = None
    received_observations: int = 0


class StreamingInferenceService:
    """Run the existing model one chronological observation batch at a time."""

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        self.trust_processor = TrustAwareSignalProcessor(self.config)
        self.extractor = FeatureExtractor(self.config)
        self._states: dict[int, PatientStreamState] = {}
        self._lock = RLock()
        self._loaded = False
        self._model = None
        self._calibrator = None
        self._feature_names: list[str] = []

    def _load_artifacts(self) -> None:
        if self._loaded:
            return
        root = Path(__file__).resolve().parents[1]
        model_path = root / "models" / "xgboost_model.joblib"
        calibrator_path = root / "models" / "calibrator.joblib"
        features_path = root / "models" / "feature_names.json"
        missing = [str(p) for p in (model_path, calibrator_path, features_path) if not p.exists()]
        if missing:
            raise StreamingInferenceError(
                "Model artifacts are missing: " + ", ".join(missing) + ". Run the training pipeline first."
            )
        import json

        self._model = joblib.load(model_path)
        self._calibrator = joblib.load(calibrator_path)
        self._feature_names = json.loads(features_path.read_text())
        if not self._feature_names:
            raise StreamingInferenceError("models/feature_names.json is empty.")
        self._loaded = True

    def _get_or_create(self, patient_id: int, static_info: dict[str, float] | None) -> PatientStreamState:
        state = self._states.get(patient_id)
        if state is None:
            state = PatientStreamState(
                patient_id=patient_id,
                static_info={k: float(v) for k, v in (static_info or {}).items()},
                accumulator=SequentialEvidenceAccumulator(self.config),
            )
            self._states[patient_id] = state
        elif static_info:
            state.static_info.update({k: float(v) for k, v in static_info.items()})
        return state

    def _assess_batch(
        self,
        state: PatientStreamState,
        time_hours: float,
        observations: dict[str, float],
    ) -> list[ObservationAssessment]:
        assessments: list[ObservationAssessment] = []
        cross_context: dict[str, float] = {}
        for param, value in observations.items():
            standard = self.trust_processor.map_parameter_alias(param)
            history = state.histories.get(standard, [])
            assessment = self.trust_processor.assess_observation(
                time_hours=time_hours,
                parameter=param,
                value=float(value),
                history=history,
                cross_param_context=cross_context,
            )
            assessments.append(assessment)
            cross_context[standard] = float(value)
            if assessment.credibility_score > 0.30:
                state.histories.setdefault(standard, []).append((time_hours, float(value)))
        return assessments

    def update(
        self,
        patient_id: int,
        time_hours: float,
        observations: dict[str, float],
        static_info: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Score one chronological observation batch and return an auditable snapshot."""
        if not observations:
            raise StreamingInferenceError("At least one observation is required.")
        if not np.isfinite(time_hours) or time_hours < 0:
            raise StreamingInferenceError("time_hours must be a finite non-negative number.")
        max_hours = float(self.config.get("pipeline", {}).get("max_telemetry_hours", 48.0))
        if time_hours > max_hours:
            raise StreamingInferenceError(f"time_hours cannot exceed configured horizon ({max_hours:g}h).")
        if any(not np.isfinite(float(value)) for value in observations.values()):
            raise StreamingInferenceError("Observation values must be finite numbers.")

        with self._lock:
            self._load_artifacts()
            state = self._get_or_create(patient_id, static_info)
            if state.last_time_hours is not None and time_hours < state.last_time_hours:
                raise StreamingInferenceError(
                    f"Out-of-order batch: {time_hours:g}h arrived after {state.last_time_hours:g}h."
                )

            new_assessments = self._assess_batch(state, time_hours, observations)
            state.assessments.extend(new_assessments)
            state.received_observations += len(new_assessments)
            state.last_time_hours = time_hours

            features = self.extractor.extract_patient_features_at_time(
                current_time=time_hours,
                assessments=state.assessments,
                static_info=state.static_info,
                patient_id=patient_id,
            )
            frame = pd.DataFrame([features])
            for name in self._feature_names:
                if name not in frame:
                    frame[name] = np.nan
            raw_risk = float(self._model.predict_proba(frame[self._feature_names])[:, 1][0])
            calibrated_risk = float(self._calibrator.predict_calibrated(np.array([raw_risk]))[0])
            credibility = float(np.mean([a.credibility_score for a in new_assessments]))
            snapshot = state.accumulator.update(
                time_hours=time_hours,
                calibrated_risk=calibrated_risk,
                raw_risk=raw_risk,
                credibility_weight=credibility,
            )

            downweighted = [a for a in new_assessments if a.credibility_score < 0.70 or not a.valid]
            return {
                "patient_id": patient_id,
                "time_hours": round(time_hours, 3),
                "raw_risk": round(raw_risk, 4),
                "calibrated_risk": round(calibrated_risk, 4),
                "evidence_score": snapshot.evidence_score,
                "persistence_count": snapshot.persistence_count,
                "state": snapshot.state,
                "alert_fired": snapshot.alert_fired,
                "suppressed_by_refractory": snapshot.suppressed_by_refractory,
                "refractory_hours_remaining": snapshot.refractory_hours_remaining,
                "reason": snapshot.reason,
                "data_quality": {
                    "mean_credibility": round(credibility, 3),
                    "observations_received": len(new_assessments),
                    "observations_seen_for_patient": state.received_observations,
                    "downweighted": [
                        {
                            "parameter": a.parameter,
                            "value": a.original_value,
                            "credibility": a.credibility_score,
                            "reason": a.reason,
                        }
                        for a in downweighted
                    ],
                },
                "model": {
                    "feature_count": len(self._feature_names),
                    "config": self.config_path,
                },
                "disclaimer": "Research prototype; not for clinical decision-making.",
            }

    def reset(self, patient_id: int) -> bool:
        with self._lock:
            return self._states.pop(patient_id, None) is not None

    def state_summary(self, patient_id: int) -> dict[str, Any] | None:
        with self._lock:
            state = self._states.get(patient_id)
            if state is None:
                return None
            return {
                "patient_id": patient_id,
                "last_time_hours": state.last_time_hours,
                "observations_seen": state.received_observations,
                "history_points": len(state.assessments),
                "alert_count": state.accumulator.total_alerts_fired if state.accumulator else 0,
            }
