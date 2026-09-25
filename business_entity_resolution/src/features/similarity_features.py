"""
SimilarityFeatures: High-performance string similarity primitives.
Uses RapidFuzz (C++ accelerated) with pure Python fallbacks.
"""

from typing import List, Set
import rapidfuzz.distance.Levenshtein as lev
import rapidfuzz.distance.JaroWinkler as jw
import rapidfuzz.fuzz as fuzz


class SimilarityPrimitives:
    """Core fuzzy and set-theoretic similarity metrics."""

    @staticmethod
    def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def token_overlap_ratio(tokens_a: List[str], tokens_b: List[str]) -> float:
        """Compute intersection over minimum length."""
        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0
        set_a, set_b = set(tokens_a), set(tokens_b)
        min_len = min(len(set_a), len(set_b))
        if min_len == 0:
            return 0.0
        return len(set_a & set_b) / min_len

    @staticmethod
    def normalized_levenshtein(str_a: str, str_b: str) -> float:
        """Compute normalized Levenshtein similarity in [0, 1]."""
        if not str_a and not str_b:
            return 1.0
        if not str_a or not str_b:
            return 0.0
        return lev.normalized_similarity(str_a, str_b)

    @staticmethod
    def jaro_winkler(str_a: str, str_b: str) -> float:
        """Compute Jaro-Winkler similarity in [0, 1]."""
        if not str_a and not str_b:
            return 1.0
        if not str_a or not str_b:
            return 0.0
        return jw.similarity(str_a, str_b)

    @staticmethod
    def token_sort_ratio(str_a: str, str_b: str) -> float:
        """Compute fuzzy token sort ratio in [0, 1]."""
        if not str_a and not str_b:
            return 1.0
        if not str_a or not str_b:
            return 0.0
        return fuzz.token_sort_ratio(str_a, str_b) / 100.0

    @staticmethod
    def prefix_similarity(str_a: str, str_b: str) -> float:
        """Compute common prefix length ratio."""
        if not str_a or not str_b:
            return 0.0
        min_l = min(len(str_a), len(str_b))
        max_l = max(len(str_a), len(str_b))
        common = 0
        for i in range(min_l):
            if str_a[i] == str_b[i]:
                common += 1
            else:
                break
        return common / max_l if max_l > 0 else 0.0
