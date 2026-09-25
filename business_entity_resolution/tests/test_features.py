"""
Unit tests for feature engineering modules.
"""

import pytest
import numpy as np
from business_entity_resolution.src.preprocessing.name_normalizer import NameNormalizer
from business_entity_resolution.src.preprocessing.address_normalizer import AddressNormalizer
from business_entity_resolution.src.blocking.candidate_generator import CandidatePair
from business_entity_resolution.src.features.similarity_features import SimilarityPrimitives
from business_entity_resolution.src.features.name_features import NameFeatures
from business_entity_resolution.src.features.address_features import AddressFeatures
from business_entity_resolution.src.features.pair_features import PairFeatures


def test_similarity_primitives():
    assert SimilarityPrimitives.jaccard_similarity({"a", "b"}, {"a", "c"}) == 1.0 / 3.0
    assert SimilarityPrimitives.normalized_levenshtein("acme corp", "acme corp") == 1.0
    assert SimilarityPrimitives.jaro_winkler("acme corp", "acme corp") == 1.0
    assert SimilarityPrimitives.token_overlap_ratio(["a", "b"], ["a", "c"]) == 0.5
    assert SimilarityPrimitives.prefix_similarity("apple", "application") == 4.0 / 11.0


def test_pair_features_extraction():
    nn = NameNormalizer()
    an = AddressNormalizer()

    n1 = nn.normalize("Roberts Titan Inc")
    a1 = an.normalize("13834 Willowtwist Street, Houston, TX")
    c1 = "US"

    n2 = nn.normalize("Roberts Titan Inc")
    a2 = an.normalize("13834-A Willowtwist Saint, Houston, Texas")
    c2 = "US"

    cand_pair = CandidatePair(s1_id="S1-1", target_id="S2-1", gen_exact=True, gen_count=1)

    feats = PairFeatures.extract_features_dict(n1, a1, c1, n2, a2, c2, cand_pair)
    assert len(feats) == len(PairFeatures.ALL_FEATURE_NAMES)
    assert feats["name_exact_clean"] == 1.0
    assert feats["name_exact_core"] == 1.0
    assert feats["country_match"] == 1.0
    assert feats["gen_exact"] == 1.0
    assert feats["addr_levenshtein"] > 0.7

    arr = PairFeatures.dict_to_array(feats)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (len(PairFeatures.ALL_FEATURE_NAMES),)
    assert not np.isnan(arr).any()
