"""
Reproducibility: Enforces strict random seeds across Python, NumPy, and ML frameworks.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42):
    """Seed all pseudo-random number generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
