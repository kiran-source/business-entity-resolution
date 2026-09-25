"""
ProbabilityCalibrator: Implements Platt scaling and Isotonic regression
to calibrate classifier probabilities if validation justifies it.
"""

from typing import Optional
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    """Post-hoc probability calibration module."""

    def __init__(self, method: str = "sigmoid"):
        """method: 'sigmoid' (Platt scaling) or 'isotonic'"""
        self.method = method
        self.calibrator = None
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray):
        """Fit calibrator on validation probabilities and labels."""
        if len(y_true) < 10 or len(np.unique(y_true)) < 2:
            return

        # Ensure 2D for logistic regression
        p = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        if self.method == "sigmoid":
            # Platt scaling: fit logistic regression on log-odds
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            lr = LogisticRegression(C=1.0, solver="lbfgs")
            lr.fit(logits, y_true)
            self.calibrator = lr
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y_true)
            self.calibrator = iso
        self.is_fitted = True

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrate probabilities."""
        if not self.is_fitted or self.calibrator is None:
            return raw_probs

        p = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        if self.method == "sigmoid":
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            return self.calibrator.predict_proba(logits)[:, 1]
        elif self.method == "isotonic":
            return self.calibrator.predict(p)
        return raw_probs
