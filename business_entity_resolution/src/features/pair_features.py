"""
PairFeatures: Orchestrates name, address, interaction, and candidate-provenance features
into a standardized feature vector for gradient boosted classification.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from ..preprocessing.name_normalizer import NormalizedName, NameNormalizer
from ..preprocessing.address_normalizer import NormalizedAddress, AddressNormalizer
from ..blocking.candidate_generator import CandidatePair
from .name_features import NameFeatures
from .address_features import AddressFeatures


class PairFeatures:
    """Computes pairwise feature vectors between S1 and candidate target entities."""

    INTERACTION_FEATURE_NAMES = [
        "name_x_addr",
        "name_add_addr",
        "name_jacc_x_addr_jacc",
        "harmonic_sim",
        "both_exact_clean",
        "name_addr_disagreement",
        "country_match",
    ]

    PROVENANCE_FEATURE_NAMES = [
        "gen_exact",
        "gen_token",
        "gen_tfidf",
        "gen_address",
        "gen_count",
    ]

    ALL_FEATURE_NAMES = (
        NameFeatures.FEATURE_NAMES
        + AddressFeatures.FEATURE_NAMES
        + INTERACTION_FEATURE_NAMES
        + PROVENANCE_FEATURE_NAMES
    )

    def __init__(self):
        self.name_normalizer = NameNormalizer()
        self.addr_normalizer = AddressNormalizer()

    @classmethod
    def extract_features_dict(
        cls,
        n1: NormalizedName,
        a1: NormalizedAddress,
        c1: str,
        n2: NormalizedName,
        a2: NormalizedAddress,
        c2: str,
        cand_pair: Optional[CandidatePair] = None,
    ) -> Dict[str, float]:
        """Extract full feature dictionary for a single entity pair."""
        feats = {}

        # 1. Name features (14)
        name_feats = NameFeatures.extract(n1, n2)
        feats.update(name_feats)

        # 2. Address features (13)
        addr_feats = AddressFeatures.extract(a1, a2)
        feats.update(addr_feats)

        # 3. Cross-field interaction features (7)
        nl = name_feats["name_levenshtein"]
        al = addr_feats["addr_levenshtein"]
        nj = name_feats["name_token_jaccard"]
        aj = addr_feats["addr_token_jaccard"]

        feats["name_x_addr"] = nl * al
        feats["name_add_addr"] = (nl + al) / 2.0
        feats["name_jacc_x_addr_jacc"] = nj * aj
        feats["harmonic_sim"] = (2.0 * nl * al) / (nl + al + 1e-6)
        feats["both_exact_clean"] = name_feats["name_exact_clean"] * addr_feats["addr_exact_clean"]
        feats["name_addr_disagreement"] = abs(nl - al)
        feats["country_match"] = 1.0 if c1.lower() == c2.lower() else 0.0

        # 4. Candidate provenance features (5)
        if cand_pair is not None:
            feats["gen_exact"] = float(cand_pair.gen_exact)
            feats["gen_token"] = float(cand_pair.gen_token)
            feats["gen_tfidf"] = float(cand_pair.gen_tfidf)
            feats["gen_address"] = float(cand_pair.gen_address)
            feats["gen_count"] = float(cand_pair.gen_count)
        else:
            feats["gen_exact"] = 0.0
            feats["gen_token"] = 0.0
            feats["gen_tfidf"] = 0.0
            feats["gen_address"] = 0.0
            feats["gen_count"] = 0.0

        return feats

    @classmethod
    def dict_to_array(cls, feat_dict: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to float32 numpy array in fixed column order."""
        return np.array([feat_dict[k] for k in cls.ALL_FEATURE_NAMES], dtype=np.float32)
