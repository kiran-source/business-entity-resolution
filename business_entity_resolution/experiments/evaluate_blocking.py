"""
Phase 6: Candidate Recall and Reduction Ratio Evaluation.
Compares individual blocking strategies and their union on real challenge training data.
"""

import os
import sys
import json
import time
from typing import Dict, Set, List

from business_entity_resolution.src.data.loader import DataLoader
from business_entity_resolution.src.data.ground_truth import GroundTruthHandler
from business_entity_resolution.src.blocking.candidate_generator import CandidateGenerator

BASE_TRAIN = r"c:\Users\kiran\OneDrive\Documents\Desktop\AWS_PROJ\student_resource\dataset\train"
EXP_DIR = r"c:\Users\kiran\OneDrive\Documents\Desktop\AWS_PROJ\experiments"
os.makedirs(EXP_DIR, exist_ok=True)


def evaluate_blocking(n_s1_eval: int = 1500, n_targets: int = 60000):
    print(f"=== Evaluating Candidate Generation on {n_s1_eval} S1s and {n_targets} S2/S3 Targets ===")
    t0 = time.time()

    loader = DataLoader()
    gt_handler = GroundTruthHandler(os.path.join(BASE_TRAIN, "train_ground_truth.tsv"))
    gt_map = gt_handler.load()

    # 1. Load S1 validation records
    s1_records = []
    eval_gt: Dict[str, Set[str]] = {}
    for rec in loader.iter_records(os.path.join(BASE_TRAIN, "train_source1.tsv"), max_records=n_s1_eval):
        eid = rec["entity_id"]
        s1_records.append(rec)
        eval_gt[eid] = gt_map.get(eid, set())

    # 2. Index S2 and S3 target records
    cg = CandidateGenerator(max_candidates_per_s1=40, enable_tfidf=True)
    target_count = 0
    t_index_start = time.time()
    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(BASE_TRAIN, src_file)
        for rec in loader.iter_records(path, max_records=n_targets // 2):
            cg.index_target_record(
                entity_id=rec["entity_id"],
                business_name=rec["business_name"],
                business_address=rec["business_address"],
                country=rec["country"],
            )
            target_count += 1
    cg.finalize_indexing()
    index_time = time.time() - t_index_start
    print(f"Indexed {target_count} target records in {index_time:.2f}s")

    # 3. Generate candidates for S1
    t_gen_start = time.time()
    cand_pairs_map = cg.generate_candidates_batch(s1_records)
    gen_time = time.time() - t_gen_start
    print(f"Candidate generation completed in {gen_time:.2f}s")

    # Extract ID sets for each strategy to compare recall
    exact_cands: Dict[str, Set[str]] = {}
    token_cands: Dict[str, Set[str]] = {}
    tfidf_cands: Dict[str, Set[str]] = {}
    addr_cands: Dict[str, Set[str]] = {}
    union_cands: Dict[str, Set[str]] = {}

    for sid, pmap in cand_pairs_map.items():
        exact_cands[sid] = {cid for cid, p in pmap.items() if p.gen_exact}
        token_cands[sid] = {cid for cid, p in pmap.items() if p.gen_token}
        tfidf_cands[sid] = {cid for cid, p in pmap.items() if p.gen_tfidf}
        addr_cands[sid] = {cid for cid, p in pmap.items() if p.gen_address}
        union_cands[sid] = set(pmap.keys())

    stats_exact = CandidateGenerator.measure_candidate_recall(exact_cands, eval_gt, target_count)
    stats_token = CandidateGenerator.measure_candidate_recall(token_cands, eval_gt, target_count)
    stats_tfidf = CandidateGenerator.measure_candidate_recall(tfidf_cands, eval_gt, target_count)
    stats_addr = CandidateGenerator.measure_candidate_recall(addr_cands, eval_gt, target_count)
    stats_union = CandidateGenerator.measure_candidate_recall(union_cands, eval_gt, target_count)

    total_time = time.time() - t0

    print("\n--- Strategy Comparison ---")
    print(f"Exact Name:    Recall = {stats_exact['candidate_recall_pct']}%, Avg Pairs/S1 = {stats_exact['avg_candidates_per_s1']}")
    print(f"Token Index:   Recall = {stats_token['candidate_recall_pct']}%, Avg Pairs/S1 = {stats_token['avg_candidates_per_s1']}")
    print(f"Address Index: Recall = {stats_addr['candidate_recall_pct']}%, Avg Pairs/S1 = {stats_addr['avg_candidates_per_s1']}")
    print(f"TF-IDF:        Recall = {stats_tfidf['candidate_recall_pct']}%, Avg Pairs/S1 = {stats_tfidf['avg_candidates_per_s1']}")
    print(f"FINAL UNION:   Recall = {stats_union['candidate_recall_pct']}%, Avg Pairs/S1 = {stats_union['avg_candidates_per_s1']}")
    print(f"Reduction Ratio: {stats_union['reduction_ratio'] * 100:.4f}% reduction")
    print(f"Total Evaluation Time: {total_time:.2f}s\n")

    results = {
        "s1_eval_count": n_s1_eval,
        "target_count": target_count,
        "exact_blocking": stats_exact,
        "token_blocking": stats_token,
        "address_blocking": stats_addr,
        "tfidf_blocking": stats_tfidf,
        "final_union": stats_union,
        "runtime_seconds": round(total_time, 2),
    }

    out_file = os.path.join(EXP_DIR, "candidate_recall_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    evaluate_blocking()
