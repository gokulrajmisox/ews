"""
calibration.py - Probability Calibration (Platt / Isotonic).

Ensures model risk probabilities reflect true empirical event frequencies.
Provides both raw and calibrated probabilities.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class ModelCalibrator:
    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.calibrator = None

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray):
        """Fits calibration curve on validation patient predictions."""
        # Clean inputs
        raw_probs = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
            self.calibrator.fit(raw_probs, y_true)
        else: # Platt sigmoid
            # Logistic regression on logit of raw probabilities
            logits = np.log(raw_probs / (1.0 - raw_probs)).reshape(-1, 1)
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(logits, y_true)

    def predict_calibrated(self, raw_probs: np.ndarray) -> np.ndarray:
        """Transforms raw probabilities into calibrated probabilities."""
        if self.calibrator is None:
            return raw_probs
        raw_probs = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.calibrator.predict(raw_probs), 0.01, 0.99)
        else:
            logits = np.log(raw_probs / (1.0 - raw_probs)).reshape(-1, 1)
            try:
                return self.calibrator.predict_proba(logits)[:, 1]
            except AttributeError:
                # Older/newer scikit-learn versions have changed fitted
                # LogisticRegression attributes. The binary Platt model only
                # needs coef_ and intercept_, so evaluate it directly when a
                # persisted artifact cannot call predict_proba.
                coef = np.asarray(self.calibrator.coef_).reshape(-1)
                intercept = float(np.asarray(self.calibrator.intercept_).reshape(-1)[0])
                scores = logits.reshape(-1) * float(coef[0]) + intercept
                return 1.0 / (1.0 + np.exp(-scores))
