"""
ThresholdOptimizer: Searches for the probability threshold that maximizes
macro F0.5 on held-out Source-1 validation entities.
"""

from typing import Dict, List, Set, Tuple, Any
import numpy as np
from .f05 import compute_macro_f05


class ThresholdOptimizer:
    """Finds optimal decision threshold maximizing macro F0.5."""

    def __init__(
        self,
        min_thresh: float = 0.10,
        max_thresh: float = 0.90,
        step: float = 0.02,
    ):
        self.thresholds = np.arange(min_thresh, max_thresh + step / 2.0, step)

    def optimize(
        self,
        candidate_probabilities: Dict[str, Dict[str, float]],
        ground_truth: Dict[str, Set[str]],
    ) -> Dict[str, Any]:
        """
        candidate_probabilities: s1_id -> {candidate_target_id: match_probability}
        ground_truth: s1_id -> set(true_match_ids)
        """
        best_f05 = -1.0
        best_threshold = 0.50
        best_metrics: Dict[str, Any] = {}
        grid_history = []

        for thresh in self.thresholds:
            thresh_f = round(float(thresh), 4)
            # Make predictions at this threshold
            predictions: Dict[str, Set[str]] = {}
            for s1_id in ground_truth.keys():
                cand_probs = candidate_probabilities.get(s1_id, {})
                selected = {
                    target_id
                    for target_id, prob in cand_probs.items()
                    if prob >= thresh_f
                }
                predictions[s1_id] = selected

            metrics = compute_macro_f05(predictions, ground_truth)
            f05 = metrics["macro_f05"]
            grid_history.append({"threshold": thresh_f, **metrics})

            if f05 > best_f05:
                best_f05 = f05
                best_threshold = thresh_f
                best_metrics = metrics

        return {
            "optimal_threshold": best_threshold,
            "best_macro_f05": best_f05,
            "best_macro_precision": best_metrics.get("macro_precision", 0.0),
            "best_macro_recall": best_metrics.get("macro_recall", 0.0),
            "singleton_accuracy": best_metrics.get("singleton_accuracy", 0.0),
            "grid_history": grid_history,
        }
