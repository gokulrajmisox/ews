"""
train.py - Leakage-Free Model Training and Probability Calibration.

Trains XGBoost deterioration classifier on patient-level training split,
calibrates risk estimates on patient-level validation split,
and evaluates discrimination and calibration metrics.

Training target (configs/config.yaml, `labeling` section): the default is
the future-window proxy label — an hourly row is positive only when the
patient eventually dies in hospital AND the end-of-telemetry endpoint falls
within `horizon_hours` after the current eval time (a near-term early-warning
target; exact event timestamps are unavailable in the challenge dataset).
Set `labeling.target: eventual_outcome` to train on the legacy label (final
in-hospital outcome broadcast to every hourly row) for comparison.
"""
