"""
NameFeatures: Extracts 14 high-signal similarity and distance features for business names.
"""

from typing import Dict, List, Set
from ..preprocessing.name_normalizer import NormalizedName
from .similarity_features import SimilarityPrimitives


class NameFeatures:
    """Feature extractor for business name pairs."""

    FEATURE_NAMES = [
        "name_exact_clean",
        "name_exact_core",
        "name_exact_sorted",
        "name_exact_alpha",
        "name_token_jaccard",
        "name_token_overlap",
        "name_common_tokens_count",
        "name_char_jaccard",
        "name_levenshtein",
        "name_jaro_winkler",
        "name_token_sort_ratio",
        "name_prefix_sim",
        "name_len_diff_ratio",
        "name_token_count_diff",
    ]

    @classmethod
    def extract(cls, n1: NormalizedName, n2: NormalizedName) -> Dict[str, float]:
        """Extract all name similarity features between two normalized names."""
        tokens1_set = set(n1.tokens)
        tokens2_set = set(n2.tokens)
        common_tokens = len(tokens1_set & tokens2_set)

        char_ngrams1 = set(n1.char_ngrams)
        char_ngrams2 = set(n2.char_ngrams)

        len1, len2 = len(n1.core_name), len(n2.core_name)
        max_len = max(len1, len2, 1)
        len_diff_ratio = abs(len1 - len2) / max_len
        token_count_diff = abs(len(n1.tokens) - len(n2.tokens))

        return {
            "name_exact_clean": 1.0 if n1.cleaned == n2.cleaned else 0.0,
            "name_exact_core": 1.0 if n1.core_name == n2.core_name else 0.0,
            "name_exact_sorted": 1.0 if n1.sorted_tokens == n2.sorted_tokens else 0.0,
            "name_exact_alpha": 1.0 if n1.alphanumeric == n2.alphanumeric else 0.0,
            "name_token_jaccard": SimilarityPrimitives.jaccard_similarity(tokens1_set, tokens2_set),
            "name_token_overlap": SimilarityPrimitives.token_overlap_ratio(n1.tokens, n2.tokens),
            "name_common_tokens_count": float(common_tokens),
            "name_char_jaccard": SimilarityPrimitives.jaccard_similarity(char_ngrams1, char_ngrams2),
            "name_levenshtein": SimilarityPrimitives.normalized_levenshtein(n1.core_name, n2.core_name),
            "name_jaro_winkler": SimilarityPrimitives.jaro_winkler(n1.core_name, n2.core_name),
            "name_token_sort_ratio": SimilarityPrimitives.token_sort_ratio(n1.cleaned, n2.cleaned),
            "name_prefix_sim": SimilarityPrimitives.prefix_similarity(n1.core_name, n2.core_name),
            "name_len_diff_ratio": len_diff_ratio,
            "name_token_count_diff": float(token_count_diff),
        }
