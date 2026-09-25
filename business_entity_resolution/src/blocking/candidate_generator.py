"""
CandidateGenerator: Coordinates multi-strategy candidate generation, records provenance,
and enforces candidate set constraints for ML ranking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from .exact_blocking import ExactBlocker
from .token_blocking import TokenBlocker
from .tfidf_blocking import TfidfBlocker
from .address_blocking import AddressBlocker
from ..preprocessing.name_normalizer import NormalizedName, NameNormalizer
from ..preprocessing.address_normalizer import NormalizedAddress, AddressNormalizer
from ..preprocessing.country_normalizer import CountryNormalizer


@dataclass(slots=True)
class CandidatePair:
    s1_id: str
    target_id: str
    gen_exact: bool = False
    gen_token: bool = False
    gen_tfidf: bool = False
    gen_address: bool = False
    gen_count: int = 0


class CandidateGenerator:
    """Multi-stage blocking orchestrator with provenance tracking."""

    def __init__(
        self,
        max_candidates_per_s1: int = 40,
        enable_tfidf: bool = True,
        tfidf_min_similarity: float = 0.30,
        tfidf_top_k: int = 15,
    ):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.enable_tfidf = enable_tfidf
        self.exact_blocker = ExactBlocker()
        self.token_blocker = TokenBlocker()
        self.address_blocker = AddressBlocker()
        self.tfidf_blocker = TfidfBlocker(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_similarity=tfidf_min_similarity,
            top_k=tfidf_top_k,
        )

        self.name_normalizer = NameNormalizer()
        self.addr_normalizer = AddressNormalizer()

        # Target storage for TF-IDF building: country -> (list of ids, list of texts)
        self._target_ids: List[str] = []
        self._target_texts: List[str] = []

    def index_target_record(
        self,
        entity_id: str,
        business_name: str,
        business_address: str,
        country: str,
    ):
        """Index a single target record (S2 or S3)."""
        cty = CountryNormalizer.normalize(country)
        norm_name = self.name_normalizer.normalize(business_name)
        norm_addr = self.addr_normalizer.normalize(business_address, cty)

        # 1. Exact blocker
        self.exact_blocker.index_target(entity_id, cty, norm_name)
        # 2. Token blocker
        self.token_blocker.index_target(entity_id, cty, norm_name)
        # 3. Address blocker
        self.address_blocker.index_target(entity_id, cty, norm_name, norm_addr)

        # Accumulate for TF-IDF index
        if self.enable_tfidf:
            self._target_ids.append(entity_id)
            # Combine name and address tokens
            text = f"{norm_name.core_name} {norm_addr.cleaned}"
            self._target_texts.append(text)

    def finalize_indexing(self):
        """Build global TF-IDF sparse matrix after streaming all targets."""
        if self.enable_tfidf and self._target_texts:
            self.tfidf_blocker.fit_and_index_targets(self._target_ids, self._target_texts)
            # Free raw text arrays to save RAM
            self._target_texts = []

    def generate_candidates_batch(
        self,
        s1_records: List[Dict[str, str]],
    ) -> Dict[str, Dict[str, CandidatePair]]:
        """
        Generate candidate pairs with provenance for a batch of S1 records.
        Returns: s1_id -> {target_id: CandidatePair}
        """
        results: Dict[str, Dict[str, CandidatePair]] = {rec["entity_id"]: {} for rec in s1_records}

        s1_ids = []
        s1_texts = []
        s1_meta = []

        for rec in s1_records:
            sid = rec["entity_id"]
            cty = CountryNormalizer.normalize(rec["country"])
            norm_name = self.name_normalizer.normalize(rec["business_name"])
            norm_addr = self.addr_normalizer.normalize(rec["business_address"], cty)

            s1_ids.append(sid)
            s1_meta.append((cty, norm_name, norm_addr))
            s1_texts.append(f"{norm_name.core_name} {norm_addr.cleaned}")

            # 1. Exact candidates
            exact_cand = self.exact_blocker.block(sid, cty, norm_name)
            for cid in exact_cand:
                if cid not in results[sid]:
                    results[sid][cid] = CandidatePair(s1_id=sid, target_id=cid)
                pair = results[sid][cid]
                pair.gen_exact = True

            # 2. Token candidates
            token_cand = self.token_blocker.block(sid, cty, norm_name)
            for cid in token_cand:
                if cid not in results[sid]:
                    results[sid][cid] = CandidatePair(s1_id=sid, target_id=cid)
                pair = results[sid][cid]
                pair.gen_token = True

            # 3. Address candidates
            addr_cand = self.address_blocker.block(sid, cty, norm_name, norm_addr)
            for cid in addr_cand:
                if cid not in results[sid]:
                    results[sid][cid] = CandidatePair(s1_id=sid, target_id=cid)
                pair = results[sid][cid]
                pair.gen_address = True

        # 4. TF-IDF candidates in batch
        if self.enable_tfidf and self.tfidf_blocker.is_fitted:
            tfidf_cands = self.tfidf_blocker.block_batch(s1_ids, s1_texts)
            for sid, cands in tfidf_cands.items():
                for cid in cands:
                    if cid not in results[sid]:
                        results[sid][cid] = CandidatePair(s1_id=sid, target_id=cid)
                    pair = results[sid][cid]
                    pair.gen_tfidf = True

        # Compute provenance counts and prune to max_candidates_per_s1
        pruned_results: Dict[str, Dict[str, CandidatePair]] = {}
        for sid, cand_map in results.items():
            for pair in cand_map.values():
                pair.gen_count = (
                    int(pair.gen_exact)
                    + int(pair.gen_token)
                    + int(pair.gen_tfidf)
                    + int(pair.gen_address)
                )

            # Sort candidates by generation provenance strength (exact > token > tfidf > address)
            sorted_pairs = sorted(
                cand_map.values(),
                key=lambda p: (p.gen_count, p.gen_exact, p.gen_token, p.gen_tfidf),
                reverse=True,
            )
            # Prune to max_candidates_per_s1
            pruned_pairs = sorted_pairs[: self.max_candidates_per_s1]
            pruned_results[sid] = {p.target_id: p for p in pruned_pairs}

        return pruned_results

    @staticmethod
    def measure_candidate_recall(
        candidates_map: Dict[str, Set[str]],
        ground_truth: Dict[str, Set[str]],
        total_targets: int,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate recall and reduction ratio.
        """
        total_gt_pairs = 0
        recalled_gt_pairs = 0
        total_candidates = 0

        for sid, gt_mids in ground_truth.items():
            cands = candidates_map.get(sid, set())
            total_candidates += len(cands)
            for mid in gt_mids:
                total_gt_pairs += 1
                if mid in cands:
                    recalled_gt_pairs += 1

        recall = recalled_gt_pairs / max(1, total_gt_pairs)
        n_s1 = len(ground_truth)
        cartesian_space = max(1, n_s1 * total_targets)
        reduction_ratio = 1.0 - (total_candidates / cartesian_space)

        return {
            "total_s1": n_s1,
            "total_gt_pairs": total_gt_pairs,
            "recalled_gt_pairs": recalled_gt_pairs,
            "candidate_recall": round(recall, 5),
            "candidate_recall_pct": round(recall * 100, 2),
            "total_candidates_generated": total_candidates,
            "avg_candidates_per_s1": round(total_candidates / max(1, n_s1), 2),
            "reduction_ratio": round(reduction_ratio, 6),
        }
