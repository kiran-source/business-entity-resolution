"""
Resolver: Partitioned entity resolution processor across countries and sources.
Supports open-set countries (US, India, France, etc.) and memory-efficient streaming.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from ..preprocessing.country_normalizer import CountryNormalizer
from ..blocking.candidate_generator import CandidateGenerator
from ..models.predict import PairPredictor
from .matcher import EntityMatcher


class EntityResolver:
    """Manages country-partitioned batch resolution to ensure scalability and memory safety."""

    def __init__(
        self,
        predictor: PairPredictor,
        max_candidates_per_s1: int = 35,
        enable_tfidf: bool = True,
    ):
        self.predictor = predictor
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.enable_tfidf = enable_tfidf

    def resolve_partition(
        self,
        country: str,
        s1_records: List[Dict[str, str]],
        target_records: Dict[str, Dict[str, str]],
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """Resolve all S1 entities within a single country partition."""
        if not s1_records:
            return {}, {}

        # 1. Build country-specific candidate generator
        cg = CandidateGenerator(
            max_candidates_per_s1=self.max_candidates_per_s1,
            enable_tfidf=self.enable_tfidf,
        )

        for tid, trec in target_records.items():
            cg.index_target_record(
                entity_id=tid,
                business_name=trec["business_name"],
                business_address=trec["business_address"],
                country=trec["country"],
            )
        cg.finalize_indexing()

        # 2. Run matcher
        matcher = EntityMatcher(candidate_generator=cg, predictor=self.predictor)
        matches, candidates = matcher.match(s1_records, target_records)
        return matches, candidates
