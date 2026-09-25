"""
Controlled Candidate Recall Test:
Selects S1 entities whose true matched S2/S3 targets are explicitly included
among the indexed targets, plus thousands of negative targets.
"""

import os
import json
import time
from typing import Dict, Set

from business_entity_resolution.src.data.loader import DataLoader
from business_entity_resolution.src.data.ground_truth import GroundTruthHandler
from business_entity_resolution.src.blocking.candidate_generator import CandidateGenerator

BASE_TRAIN = r"c:\Users\kiran\OneDrive\Documents\Desktop\AWS_PROJ\student_resource\dataset\train"


def test_controlled_recall(n_s1: int = 500, n_noise_targets: int = 20000):
    print(f"=== Controlled Candidate Recall Test ({n_s1} S1s with their true targets + {n_noise_targets} noise) ===")

    loader = DataLoader()
    gt_handler = GroundTruthHandler(os.path.join(BASE_TRAIN, "train_ground_truth.tsv"))
    gt_map = gt_handler.load()

    # 1. Collect S1 records and their target IDs
    s1_records = []
    target_ids_needed = set()
    eval_gt: Dict[str, Set[str]] = {}

    for rec in loader.iter_records(os.path.join(BASE_TRAIN, "train_source1.tsv"), max_records=n_s1):
        sid = rec["entity_id"]
        mids = gt_map.get(sid, set())
        if mids:
            s1_records.append(rec)
            eval_gt[sid] = mids
            target_ids_needed.update(mids)

    print(f"Loaded {len(s1_records)} S1 entities with {len(target_ids_needed)} true matched target IDs.")

    # 2. Index the needed targets + noise targets
    cg = CandidateGenerator(max_candidates_per_s1=40, enable_tfidf=True)
    indexed_needed = 0
    total_indexed = 0

    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(BASE_TRAIN, src_file)
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                eid, bname, baddr, bcty = parts[0], parts[1], parts[2], parts[3]
                is_needed = eid in target_ids_needed
                if is_needed or i < (n_noise_targets // 2):
                    cg.index_target_record(eid, bname, baddr, bcty)
                    total_indexed += 1
                    if is_needed:
                        indexed_needed += 1
                if indexed_needed >= len(target_ids_needed) and total_indexed >= n_noise_targets:
                    break

    cg.finalize_indexing()
    print(f"Indexed {total_indexed} total targets (found {indexed_needed}/{len(target_ids_needed)} true targets).")

    # 3. Generate candidates
    cand_pairs_map = cg.generate_candidates_batch(s1_records)

    exact_cands = {sid: {cid for cid, p in pmap.items() if p.gen_exact} for sid, pmap in cand_pairs_map.items()}
    token_cands = {sid: {cid for cid, p in pmap.items() if p.gen_token} for sid, pmap in cand_pairs_map.items()}
    tfidf_cands = {sid: {cid for cid, p in pmap.items() if p.gen_tfidf} for sid, pmap in cand_pairs_map.items()}
    union_cands = {sid: set(pmap.keys()) for sid, pmap in cand_pairs_map.items()}

    res_exact = CandidateGenerator.measure_candidate_recall(exact_cands, eval_gt, total_indexed)
    res_token = CandidateGenerator.measure_candidate_recall(token_cands, eval_gt, total_indexed)
    res_tfidf = CandidateGenerator.measure_candidate_recall(tfidf_cands, eval_gt, total_indexed)
    res_union = CandidateGenerator.measure_candidate_recall(union_cands, eval_gt, total_indexed)

    print("\n--- Controlled Recall Results ---")
    print(f"Exact Name Recall:   {res_exact['candidate_recall_pct']}%")
    print(f"Token Block Recall:  {res_token['candidate_recall_pct']}%")
    print(f"TF-IDF Block Recall: {res_tfidf['candidate_recall_pct']}%")
    print(f"UNION Recall:        {res_union['candidate_recall_pct']}% (Target recall when in pool)")
    print(f"Avg Candidates / S1: {res_union['avg_candidates_per_s1']}")
    print(f"Reduction Ratio:     {res_union['reduction_ratio'] * 100:.4f}%\n")


if __name__ == "__main__":
    test_controlled_recall()
