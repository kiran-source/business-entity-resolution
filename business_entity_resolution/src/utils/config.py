"""
Config: YAML-based configuration loader and validator.
"""

import os
from typing import Dict, Any, Optional
import yaml


class ConfigLoader:
    """Loads and merges configuration dictionary from YAML file."""

    DEFAULT_CONFIG = {
        "random_seed": 42,
        "data": {
            "train_dir": "student_resource/dataset/train",
            "test_dir": "student_resource/dataset/test",
            "val_fraction": 0.15,
        },
        "blocking": {
            "max_candidates_per_s1": 35,
            "enable_tfidf": True,
            "tfidf_min_similarity": 0.28,
            "tfidf_top_k": 15,
        },
        "model": {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "num_leaves": 31,
            "min_child_samples": 20,
            "neg_pos_ratio": 3.0,
        },
        "threshold": {
            "min_thresh": 0.15,
            "max_thresh": 0.85,
            "step": 0.02,
        },
        "output": {
            "output_dir": "output",
            "matching_file": "output/matching_results.tsv",
            "candidate_file": "output/candidate_pairs.tsv",
            "submission_zip": "team_submission.zip",
        },
    }

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load YAML configuration or fall back to defaults."""
        config = dict(cls.DEFAULT_CONFIG)
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            # Deep update
            for section, values in user_cfg.items():
                if isinstance(values, dict) and section in config:
                    config[section].update(values)
                else:
                    config[section] = values
        return config
