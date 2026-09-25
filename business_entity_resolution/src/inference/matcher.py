"""
Matcher: Coordinates candidate generation, ML scoring, and zero/one/many decision logic.
Guarantees that final predictions are a strict subset of candidate_pairs.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from ..blocking.candidate_generator import CandidateGenerator, CandidatePair
from ..models.predict import PairPredictor


class EntityMatcher:
    """End-to-end entity matcher coordinating candidate blocking and scoring."""

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        predictor: PairPredictor,
    ):
        self.candidate_generator = candidate_generator
        self.predictor = predictor

    def match(
        self,
        s1_records: List[Dict[str, str]],
        target_records: Dict[str, Dict[str, str]],
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Execute blocking + ML scoring for S1 records against target records.
        Returns:
        1. final_predictions: s1_id -> set of predicted matched target IDs
        2. candidate_pairs: s1_id -> set of candidate target IDs passed to model
        """
        # 1. Blocking / candidate generation
        cand_pairs_map = self.candidate_generator.generate_candidates_batch(s1_records)

        # 2. Extract final candidate pairs set (strictly before ML pruning)
        candidate_pairs_set: Dict[str, Set[str]] = {
            rec["entity_id"]: set(cand_pairs_map.get(rec["entity_id"], {}).keys())
            for rec in s1_records
        }

        # 3. Model inference over the candidate pairs
        final_matches, _ = self.predictor.predict_candidates(
            s1_records=s1_records,
            candidate_pairs_map=cand_pairs_map,
            target_records=target_records,
        )

        # 4. Strict assertion: predictions MUST be a subset of candidates
        for sid, matched_ids in final_matches.items():
            cands = candidate_pairs_set.get(sid, set())
            # Enforce subset
            final_matches[sid] = matched_ids & cands

        return final_matches, candidate_pairs_set
