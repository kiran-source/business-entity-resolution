"""
Predict: Efficient batched pairwise inference over candidate sets.
Outputs calibrated match probabilities and applies optimal F0.5 decision thresholds.
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import lightgbm as lgb

from ..preprocessing.name_normalizer import NormalizedName, NameNormalizer
from ..preprocessing.address_normalizer import NormalizedAddress, AddressNormalizer
from ..preprocessing.country_normalizer import CountryNormalizer
from ..blocking.candidate_generator import CandidatePair
from ..features.pair_features import PairFeatures


class PairPredictor:
    """Performs batched inference on candidate pairs."""

    def __init__(
        self,
        booster: lgb.Booster,
        threshold: float = 0.50,
        batch_size: int = 50000,
    ):
        self.booster = booster
        self.threshold = threshold
        self.batch_size = batch_size
        self.name_normalizer = NameNormalizer()
        self.addr_normalizer = AddressNormalizer()

    def predict_candidates(
        self,
        s1_records: List[Dict[str, str]],
        candidate_pairs_map: Dict[str, Dict[str, CandidatePair]],
        target_records: Dict[str, Dict[str, str]],
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, float]]]:
        """
        Evaluate candidate pairs and return:
        1. final_matches: s1_id -> set of predicted matched target IDs
        2. candidate_probabilities: s1_id -> {target_id: probability}
        """
        final_matches: Dict[str, Set[str]] = {rec["entity_id"]: set() for rec in s1_records}
        cand_probs: Dict[str, Dict[str, float]] = {rec["entity_id"]: {} for rec in s1_records}

        # Normalize S1 records
        norm_s1 = {}
        for rec in s1_records:
            sid = rec["entity_id"]
            cty = CountryNormalizer.normalize(rec["country"])
            norm_s1[sid] = (
                self.name_normalizer.normalize(rec["business_name"]),
                self.addr_normalizer.normalize(rec["business_address"], cty),
                cty,
            )

        # Batch feature accumulation
        batch_features = []
        batch_refs = []

        def process_batch():
            if not batch_features:
                return
            X = np.vstack(batch_features)
            probs = self.booster.predict(X)
            for (sid, tid), p in zip(batch_refs, probs):
                p_val = float(p)
                cand_probs[sid][tid] = p_val
                if p_val >= self.threshold:
                    final_matches[sid].add(tid)
            batch_features.clear()
            batch_refs.clear()

        for rec in s1_records:
            sid = rec["entity_id"]
            n1, a1, cty1 = norm_s1[sid]
            cands = candidate_pairs_map.get(sid, {})

            for tid, cpair in cands.items():
                if tid not in target_records:
                    continue
                t_rec = target_records[tid]
                cty2 = CountryNormalizer.normalize(t_rec["country"])
                n2 = self.name_normalizer.normalize(t_rec["business_name"])
                a2 = self.addr_normalizer.normalize(t_rec["business_address"], cty2)

                feats = PairFeatures.extract_features_dict(n1, a1, cty1, n2, a2, cty2, cpair)
                batch_features.append(PairFeatures.dict_to_array(feats))
                batch_refs.append((sid, tid))

                if len(batch_features) >= self.batch_size:
                    process_batch()

        # Process any remaining items
        process_batch()

        return final_matches, cand_probs
