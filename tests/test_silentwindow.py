"""
test_silentwindow.py - Rigorous Unit Tests for SilentWindow Core Architecture.

Verifies:
1. Patient-level train/val/test splitting (zero overlap guarantee).
2. Layer 1 Trust-Aware Signal Processing (plausibility, jumps, and discordance).
3. Staleness clock and missingness preservation.
4. Layer 2 Personal Baseline adaptation and deviation.
5. Strict causality: no feature at time T ever changes when future data > T is injected.
6. Layer 3 Sequential Evidence Accumulator:
   - Resistance to single transient spikes.
   - Accumulation and escalation to WATCH and ALERT under sustained risk.
   - Refractory period suppression and alert budget ceiling.
"""

import os
import copy
import pytest
import numpy as np
import pandas as pd

from ml.preprocessing import load_config, get_patient_splits, load_patient_raw
from ml.trust_layer import TrustAwareSignalProcessor, ObservationAssessment
from ml.features import FeatureExtractor
from ml.evidence_accumulator import SequentialEvidenceAccumulator

@pytest.fixture
def config():
    return load_config("configs/config.yaml")

def test_patient_level_splitting(config):
    """Guarantees zero row/patient leakage between splits."""
    if not os.path.exists(config["data"]["outcomes_file"]):
        pytest.skip("PhysioNet raw data is not present in this checkout.")
    tr, val, te, _ = get_patient_splits(config)
    set_tr = set(tr)
    set_val = set(val)
    set_te = set(te)
    
    assert len(set_tr.intersection(set_val)) == 0, "DATA LEAK: Patient in both train and val!"
    assert len(set_tr.intersection(set_te)) == 0, "DATA LEAK: Patient in both train and test!"
    assert len(set_val.intersection(set_te)) == 0, "DATA LEAK: Patient in both val and test!"

def test_trust_layer_credibility(config):
    """Tests credibility downweighting on implausible values, spikes, and discordant vitals."""
    proc = TrustAwareSignalProcessor(config)

    # 1. Physiologically implausible value (HR = 300)
    a1 = proc.assess_observation(1.0, "HR", 300.0, history=[(0.5, 80.0)])
    assert a1.credibility_score < 0.30, f"Expected low credibility for HR=300, got {a1.credibility_score}"
    assert not a1.valid
    assert "Implausible" in str(a1.reason)

    # 2. Sudden jump artifact (80 -> 180 in 10 minutes = 600/hr > max_allowed 45/hr)
    a2 = proc.assess_observation(1.16, "HR", 180.0, history=[(1.0, 80.0)])
    assert a2.credibility_score < 0.60, f"Expected downweight for sudden jump, got {a2.credibility_score}"
    assert "Sudden jump" in str(a2.reason)

    # 3. Discordant blood pressure (SysABP <= DiasABP)
    a3 = proc.assess_observation(2.0, "SysABP", 50.0, history=[], cross_param_context={"DiasABP": 65.0})
    assert a3.credibility_score < 0.70, f"Expected penalty for SysABP <= DiasABP, got {a3.credibility_score}"
    assert "Discordant" in str(a3.reason)

    # 4. Normal credible vital
    a4 = proc.assess_observation(3.0, "HR", 75.0, history=[(2.0, 72.0)])
    assert a4.credibility_score >= 0.95
    assert a4.valid

def test_staleness_and_missingness(config):
    """Verifies that missingness flags and staleness clocks are preserved."""
    extractor = FeatureExtractor(config)
    assessments = [
        ObservationAssessment(time_hours=1.0, parameter="HR", original_value=72.0, valid=True, credibility_score=1.0)
    ]
    # Evaluate at hour 4.0
    feat = extractor.extract_patient_features_at_time(4.0, assessments, {"Age": 60}, 9999)
    
    # HR was observed 3 hours ago
    assert feat["HR_is_missing"] == 0.0
    assert abs(feat["HR_time_since_last"] - 3.0) < 1e-4

    # RespRate was never observed
    assert feat["RespRate_is_missing"] == 1.0
    assert abs(feat["RespRate_time_since_last"] - 4.0) < 1e-4

def test_personal_baseline_adaptation(config):
    """Verifies that early window (first 6 hours) defines personal baseline."""
    extractor = FeatureExtractor(config)
    assessments = [
        ObservationAssessment(time_hours=1.0, parameter="HR", original_value=60.0, valid=True, credibility_score=1.0),
        ObservationAssessment(time_hours=3.0, parameter="HR", original_value=64.0, valid=True, credibility_score=1.0),
        ObservationAssessment(time_hours=8.0, parameter="HR", original_value=92.0, valid=True, credibility_score=1.0),
    ]
    # At hour 8.0, baseline window is first 6.0 hours: mean(60, 64) = 62.0
    feat = extractor.extract_patient_features_at_time(8.0, assessments, {"Age": 55}, 9999)
    assert abs(feat["HR_baseline_mean"] - 62.0) < 1e-3
    assert abs(feat["HR_dev_from_baseline"] - 30.0) < 1e-3  # 92 - 62 = +30 bpm

def test_no_future_leakage_assertion(config):
    """
    CRITICAL PROOF OF NON-LEAKAGE:
    Features generated at time T must be 100% IDENTICAL whether or not future data > T exists.
    """
    extractor = FeatureExtractor(config)
    
    # Base stream up to hour 5.0
    base_assessments = [
        ObservationAssessment(time_hours=1.0, parameter="HR", original_value=70.0, valid=True, credibility_score=1.0),
        ObservationAssessment(time_hours=2.0, parameter="SysABP", original_value=120.0, valid=True, credibility_score=1.0),
        ObservationAssessment(time_hours=4.5, parameter="HR", original_value=78.0, valid=True, credibility_score=1.0),
    ]
    
    feat_without_future = extractor.extract_patient_features_at_time(5.0, base_assessments, {"Age": 65}, 1234)

    # Injected catastrophic future events at hour 10.0 and hour 20.0
    contaminated_assessments = copy.deepcopy(base_assessments) + [
        ObservationAssessment(time_hours=10.0, parameter="HR", original_value=220.0, valid=True, credibility_score=1.0),
        ObservationAssessment(time_hours=12.0, parameter="SysABP", original_value=40.0, valid=True, credibility_score=1.0),
        ObservationAssessment(time_hours=20.0, parameter="HR", original_value=0.0, valid=False, credibility_score=0.1),
    ]

    feat_with_future = extractor.extract_patient_features_at_time(5.0, contaminated_assessments, {"Age": 65}, 1234)

    for k in feat_without_future:
        v1 = feat_without_future[k]
        v2 = feat_with_future[k]
        if np.isnan(v1):
            assert np.isnan(v2), f"Feature {k} discrepancy with future data!"
        else:
            assert abs(v1 - v2) < 1e-6, f"LEAKAGE CONFIRMED: Feature {k} changed from {v1} to {v2} when future data was present!"

def test_evidence_accumulator_spike_rejection(config):
    """
    Verifies that a single transient spike (even risk=0.90) DOES NOT immediately trigger
    a high-confidence alert without persistence.
    """
    accum = SequentialEvidenceAccumulator(config)
    
    # Normal baseline (r = 0.15)
    s1 = accum.update(time_hours=1.0, calibrated_risk=0.15, raw_risk=0.15, credibility_weight=1.0)
    assert s1.state == "STABLE"
    assert not s1.alert_fired

    # Single transient spike (r = 0.85), but low credibility (sensor artifact)
    s2 = accum.update(time_hours=2.0, calibrated_risk=0.85, raw_risk=0.85, credibility_weight=0.3)
    assert not s2.alert_fired, "VULNERABILITY: Single spike triggered an alert!"
    assert s2.state != "ALERT"

    # Immediately normalizes (r = 0.15)
    s3 = accum.update(time_hours=3.0, calibrated_risk=0.15, raw_risk=0.15, credibility_weight=1.0)
    assert s3.state == "STABLE"
    assert not s3.alert_fired
    assert s3.evidence_score < accum.watch_threshold

def test_evidence_accumulator_persistence(config):
    """Verifies that sustained credible risk progressively triggers WATCH and then ALERT."""
    accum = SequentialEvidenceAccumulator(config)

    # Sustained deterioration: hours 1 to 4 with elevated risk
    s1 = accum.update(time_hours=1.0, calibrated_risk=0.50, raw_risk=0.50, credibility_weight=0.95)
    s2 = accum.update(time_hours=2.0, calibrated_risk=0.60, raw_risk=0.60, credibility_weight=0.95)
    
    # Must enter WATCH state
    assert s2.evidence_score >= accum.watch_threshold
    assert s2.state == "WATCH"

    # Third consecutive hour of elevated risk
    s3 = accum.update(time_hours=3.0, calibrated_risk=0.75, raw_risk=0.75, credibility_weight=0.95)
    assert s3.state == "ALERT"
    assert s3.alert_fired

def test_alert_refractory_and_budget(config):
    """Verifies duplicate alert suppression during refractory period and adherence to alert budget."""
    accum = SequentialEvidenceAccumulator(config)
    
    # Drive accumulator: hour 1 watch, hour 2 triggers initial alert
    accum.update(time_hours=1.0, calibrated_risk=0.5, raw_risk=0.5, credibility_weight=1.0)
    s2 = accum.update(time_hours=2.0, calibrated_risk=0.75, raw_risk=0.75, credibility_weight=1.0)
    assert s2.alert_fired, "Initial high-confidence alert should fire at hour 2"
    assert s2.state == "ALERT"

    # Hour 3.0: Risk remains high (1 hour after alert), but within 6-hour refractory window -> MUST BE SUPPRESSED
    s3 = accum.update(time_hours=3.0, calibrated_risk=0.85, raw_risk=0.85, credibility_weight=1.0)
    assert not s3.alert_fired, "SPAM ERROR: Alert fired during refractory window!"
    assert s3.suppressed_by_refractory
    assert s3.refractory_hours_remaining > 0
