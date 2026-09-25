"""
Baseline Evaluation: Exact normalized name + address within country.
Measures baseline candidate recall, precision, and macro F0.5.
"""

import os
import sys
import json
import time
from collections import defaultdict
from typing import Dict, Set, List

from business_entity_resolution.src.data.loader import DataLoader
from business_entity_resolution.src.data.ground_truth import GroundTruthHandler
from business_entity_resolution.src.preprocessing.name_normalizer import NameNormalizer
from business_entity_resolution.src.preprocessing.address_normalizer import AddressNormalizer
from business_entity_resolution.src.preprocessing.country_normalizer import CountryNormalizer
from business_entity_resolution.src.evaluation.f05 import compute_macro_f05

BASE_TRAIN = r"c:\Users\kiran\OneDrive\Documents\Desktop\AWS_PROJ\student_resource\dataset\train"
EXP_DIR = r"c:\Users\kiran\OneDrive\Documents\Desktop\AWS_PROJ\experiments"
os.makedirs(EXP_DIR, exist_ok=True)


def run_baseline(n_s1_eval: int = 2000, n_targets: int = 100000):
    print(f"Running Exact Match Baseline on {n_s1_eval} S1 validation entities against {n_targets} S2/S3 targets...")
    t0 = time.time()

    loader = DataLoader()
    gt_handler = GroundTruthHandler(os.path.join(BASE_TRAIN, "train_ground_truth.tsv"))
    gt_map = gt_handler.load()

    name_norm = NameNormalizer()
    addr_norm = AddressNormalizer()

    # 1. Load evaluation S1 records
    s1_records = []
    eval_gt: Dict[str, Set[str]] = {}
    for rec in loader.iter_records(os.path.join(BASE_TRAIN, "train_source1.tsv"), max_records=n_s1_eval):
        eid = rec["entity_id"]
        s1_records.append(rec)
        eval_gt[eid] = gt_map.get(eid, set())

    # Build exact matching index from S2 and S3 targets
    # key: (country, core_name, cleaned_addr) -> list of target entity_ids
    index = defaultdict(list)
    total_targets = 0

    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(BASE_TRAIN, src_file)
        for rec in loader.iter_records(path, max_records=n_targets // 2):
            total_targets += 1
            cty = CountryNormalizer.normalize(rec["country"])
            nm = name_norm.normalize(rec["business_name"]).core_name
            ad = addr_norm.normalize(rec["business_address"]).cleaned
            key = (cty, nm, ad)
            index[key].append(rec["entity_id"])

    # 2. Predict matches for S1 using exact match index
    predictions: Dict[str, Set[str]] = {}
    total_candidates = 0

    for rec in s1_records:
        eid = rec["entity_id"]
        cty = CountryNormalizer.normalize(rec["country"])
        nm = name_norm.normalize(rec["business_name"]).core_name
        ad = addr_norm.normalize(rec["business_address"]).cleaned
        key = (cty, nm, ad)
        matched_targets = set(index.get(key, []))
        predictions[eid] = matched_targets
        total_candidates += len(matched_targets)

    # 3. Evaluate F0.5
    eval_results = compute_macro_f05(predictions, eval_gt)
    elapsed = time.time() - t0

    result = {
        "experiment_name": "baseline_exact_match",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "s1_eval_count": n_s1_eval,
        "targets_indexed": total_targets,
        "total_predicted_matches": total_candidates,
        "metrics": eval_results,
        "runtime_seconds": round(elapsed, 2),
    }

    out_file = os.path.join(EXP_DIR, "baseline_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== Baseline Results ===")
    print(f"Macro Precision: {eval_results['macro_precision']:.4f}")
    print(f"Macro Recall:    {eval_results['macro_recall']:.4f}")
    print(f"Macro F0.5:      {eval_results['macro_f05']:.4f}")
    print(f"Singleton Acc:   {eval_results['singleton_accuracy']:.4f}")
    print(f"Runtime:         {elapsed:.2f}s")
    print(f"Results saved to: {out_file}\n")
    return result


if __name__ == "__main__":
    run_baseline()
