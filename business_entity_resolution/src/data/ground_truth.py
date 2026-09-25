"""
GroundTruthHandler: Parses ground truth TSV, handles zero/one/many matches,
and performs strict Source-1 entity level train/validation splitting to guarantee zero leakage.
"""

import os
from typing import Dict, Set, List, Tuple, Generator, Optional
import random


class GroundTruthHandler:
    """Manages ground truth annotations and leakage-free entity splitting."""

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self._matches: Dict[str, Set[str]] = {}

    def load(self, filepath: Optional[str] = None) -> Dict[str, Set[str]]:
        """Load ground truth mappings: s1_id -> set of matched_ids."""
        path = filepath or self.filepath
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth file not found: {path}")

        matches: Dict[str, Set[str]] = {}
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                if not line.strip():
                    continue
                s1_id, tab, rest = line.partition("\t")
                s1_id = s1_id.strip()
                if not tab:
                    matches[s1_id] = set()
                    continue
                mids_str = rest.rstrip("\r\n").strip()
                if mids_str:
                    mids = {m.strip() for m in mids_str.split(",") if m.strip()}
                    matches[s1_id] = mids
                else:
                    matches[s1_id] = set()
        self._matches = matches
        return self._matches

    def get_matches(self, s1_id: str) -> Set[str]:
        """Return the set of ground truth match IDs for a given S1 entity ID."""
        return self._matches.get(s1_id, set())

    @property
    def matches(self) -> Dict[str, Set[str]]:
        return self._matches

    def split_s1_entities(
        self,
        s1_ids: List[str],
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """
        Split Source 1 entities strictly by entity_id to avoid data leakage.
        Ensures the same S1 entity NEVER appears in both train and validation.
        """
        rng = random.Random(seed)
        shuffled = list(s1_ids)
        rng.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1.0 - val_fraction))
        train_ids = shuffled[:split_idx]
        val_ids = shuffled[split_idx:]
        return train_ids, val_ids
