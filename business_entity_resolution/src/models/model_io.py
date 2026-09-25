"""
ModelIO: Serializes and loads trained classifier models, calibrators, and feature configs.
"""

import os
import json
import pickle
from typing import Dict, Any, Optional, Tuple
import lightgbm as lgb


class ModelIO:
    """Handles serialization and deserialization of ER models."""

    @staticmethod
    def save_model(
        model: lgb.LGBMClassifier,
        output_path: str,
        threshold: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Save LightGBM model and associated threshold and metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        model_file = output_path if output_path.endswith(".txt") else f"{output_path}.txt"
        meta_file = f"{os.path.splitext(model_file)[0]}_meta.json"

        # Save LightGBM booster text format
        model.booster_.save_model(model_file)

        meta = {
            "optimal_threshold": threshold,
            "feature_names": model.feature_name_,
            "n_features": len(model.feature_name_),
            "license": "MIT License (LightGBM)",
            **(metadata or {}),
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load_model(model_path: str) -> Tuple[lgb.Booster, float, Dict[str, Any]]:
        """Load LightGBM booster, optimal threshold, and metadata."""
        model_file = model_path if model_path.endswith(".txt") else f"{model_path}.txt"
        meta_file = f"{os.path.splitext(model_file)[0]}_meta.json"

        booster = lgb.Booster(model_file=model_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        threshold = meta.get("optimal_threshold", 0.5)
        return booster, threshold, meta
