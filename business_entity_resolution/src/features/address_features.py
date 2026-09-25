"""
AddressFeatures: Extracts 13 similarity, component, and numeric features for addresses.
Handles missing/empty address fields safely.
"""

from typing import Dict, List, Set
from ..preprocessing.address_normalizer import NormalizedAddress
from .similarity_features import SimilarityPrimitives


class AddressFeatures:
    """Feature extractor for business address pairs."""

    FEATURE_NAMES = [
        "addr_both_empty",
        "addr_one_empty",
        "addr_exact_clean",
        "addr_exact_sorted",
        "addr_exact_alpha",
        "addr_token_jaccard",
        "addr_token_overlap",
        "addr_common_tokens_count",
        "addr_levenshtein",
        "addr_numeric_overlap_count",
        "addr_postal_code_match",
        "addr_building_num_match",
        "addr_len_diff_ratio",
    ]

    @classmethod
    def extract(cls, a1: NormalizedAddress, a2: NormalizedAddress) -> Dict[str, float]:
        """Extract address similarity features, safely handling empty strings."""
        if a1.is_empty and a2.is_empty:
            return {
                "addr_both_empty": 1.0,
                "addr_one_empty": 0.0,
                "addr_exact_clean": 1.0,
                "addr_exact_sorted": 1.0,
                "addr_exact_alpha": 1.0,
                "addr_token_jaccard": 1.0,
                "addr_token_overlap": 1.0,
                "addr_common_tokens_count": 0.0,
                "addr_levenshtein": 1.0,
                "addr_numeric_overlap_count": 0.0,
                "addr_postal_code_match": 0.5,
                "addr_building_num_match": 0.5,
                "addr_len_diff_ratio": 0.0,
            }

        if a1.is_empty or a2.is_empty:
            return {
                "addr_both_empty": 0.0,
                "addr_one_empty": 1.0,
                "addr_exact_clean": 0.0,
                "addr_exact_sorted": 0.0,
                "addr_exact_alpha": 0.0,
                "addr_token_jaccard": 0.0,
                "addr_token_overlap": 0.0,
                "addr_common_tokens_count": 0.0,
                "addr_levenshtein": 0.0,
                "addr_numeric_overlap_count": 0.0,
                "addr_postal_code_match": 0.0,
                "addr_building_num_match": 0.0,
                "addr_len_diff_ratio": 1.0,
            }

        tokens1_set = set(a1.tokens)
        tokens2_set = set(a2.tokens)
        common_tokens = len(tokens1_set & tokens2_set)

        # Numeric tokens overlap (building/house numbers, sector numbers)
        num_set1 = set(a1.numeric_tokens)
        num_set2 = set(a2.numeric_tokens)
        num_overlap = len(num_set1 & num_set2)

        # Postal code match
        if a1.postal_code and a2.postal_code:
            postal_match = 1.0 if a1.postal_code == a2.postal_code else 0.0
        else:
            postal_match = 0.5  # Neutral when postal code is absent

        # Building number match
        if a1.building_number and a2.building_number:
            bldg_match = 1.0 if a1.building_number == a2.building_number else 0.0
        else:
            bldg_match = 0.5  # Neutral when building number is absent

        len1, len2 = len(a1.cleaned), len(a2.cleaned)
        max_len = max(len1, len2, 1)
        len_diff_ratio = abs(len1 - len2) / max_len

        return {
            "addr_both_empty": 0.0,
            "addr_one_empty": 0.0,
            "addr_exact_clean": 1.0 if a1.cleaned == a2.cleaned else 0.0,
            "addr_exact_sorted": 1.0 if a1.sorted_tokens == a2.sorted_tokens else 0.0,
            "addr_exact_alpha": 1.0 if a1.alphanumeric == a2.alphanumeric else 0.0,
            "addr_token_jaccard": SimilarityPrimitives.jaccard_similarity(tokens1_set, tokens2_set),
            "addr_token_overlap": SimilarityPrimitives.token_overlap_ratio(a1.tokens, a2.tokens),
            "addr_common_tokens_count": float(common_tokens),
            "addr_levenshtein": SimilarityPrimitives.normalized_levenshtein(a1.cleaned, a2.cleaned),
            "addr_numeric_overlap_count": float(num_overlap),
            "addr_postal_code_match": postal_match,
            "addr_building_num_match": bldg_match,
            "addr_len_diff_ratio": len_diff_ratio,
        }
