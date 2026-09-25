"""
ErrorAnalysis: Comprehensive breakdown and diagnostic categorization of entity resolution errors.
"""

import os
import json
from typing import Dict, Set, List, Tuple, Any
from .f05 import compute_entity_f05


class ErrorAnalyzer:
    """Diagnoses false positives, false negatives, singleton errors, and noise types."""

    def __init__(self, output_dir: str = "reports/error_analysis"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze(
        self,
        predictions: Dict[str, Set[str]],
        ground_truth: Dict[str, Set[str]],
        s1_records: Dict[str, Dict[str, str]],
        target_records: Dict[str, Dict[str, str]],
        max_samples_per_category: int = 15,
    ) -> Dict[str, Any]:
        """Produce structured error analysis report."""
        fp_pairs = []
        fn_pairs = []
        singleton_fps = []
        singleton_fns = []
        multi_match_errors = []

        total_fp = 0
        total_fn = 0
        total_tp = 0

        for sid, gt_mids in ground_truth.items():
            pred_mids = predictions.get(sid, set())
            p, r, f = compute_entity_f05(pred_mids, gt_mids)

            # True singletons
            if len(gt_mids) == 0:
                if len(pred_mids) > 0:
                    singleton_fps.append({
                        "s1_id": sid,
                        "s1_name": s1_records.get(sid, {}).get("business_name", ""),
                        "s1_addr": s1_records.get(sid, {}).get("business_address", ""),
                        "false_matches": list(pred_mids),
                    })
                    total_fp += len(pred_mids)
                continue

            # Non-singletons
            if len(pred_mids) == 0:
                singleton_fns.append({
                    "s1_id": sid,
                    "s1_name": s1_records.get(sid, {}).get("business_name", ""),
                    "s1_addr": s1_records.get(sid, {}).get("business_address", ""),
                    "missed_matches": list(gt_mids),
                })
                total_fn += len(gt_mids)
                continue

            # Overlap
            tp = pred_mids & gt_mids
            fp = pred_mids - gt_mids
            fn = gt_mids - pred_mids

            total_tp += len(tp)
            total_fp += len(fp)
            total_fn += len(fn)

            if fp and len(fp_pairs) < max_samples_per_category:
                for tid in fp:
                    fp_pairs.append({
                        "s1_id": sid,
                        "s1_name": s1_records.get(sid, {}).get("business_name", ""),
                        "s1_addr": s1_records.get(sid, {}).get("business_address", ""),
                        "target_id": tid,
                        "target_name": target_records.get(tid, {}).get("business_name", ""),
                        "target_addr": target_records.get(tid, {}).get("business_address", ""),
                    })

            if fn and len(fn_pairs) < max_samples_per_category:
                for tid in fn:
                    fn_pairs.append({
                        "s1_id": sid,
                        "s1_name": s1_records.get(sid, {}).get("business_name", ""),
                        "s1_addr": s1_records.get(sid, {}).get("business_address", ""),
                        "target_id": tid,
                        "target_name": target_records.get(tid, {}).get("business_name", ""),
                        "target_addr": target_records.get(tid, {}).get("business_address", ""),
                    })

            if (fp or fn) and len(gt_mids) > 1 and len(multi_match_errors) < max_samples_per_category:
                multi_match_errors.append({
                    "s1_id": sid,
                    "true_count": len(gt_mids),
                    "pred_count": len(pred_mids),
                    "tp_count": len(tp),
                    "f05": round(f, 4),
                })

        report = {
            "aggregate": {
                "total_tp_pairs": total_tp,
                "total_fp_pairs": total_fp,
                "total_fn_pairs": total_fn,
                "singleton_false_positives_count": len(singleton_fps),
                "singleton_false_negatives_count": len(singleton_fns),
            },
            "samples": {
                "false_positive_pairs": fp_pairs[:max_samples_per_category],
                "false_negative_pairs": fn_pairs[:max_samples_per_category],
                "singleton_false_positives": singleton_fps[:max_samples_per_category],
                "multi_match_partial_errors": multi_match_errors[:max_samples_per_category],
            },
        }

        # Save report
        json_file = os.path.join(self.output_dir, "error_analysis_report.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        md_file = os.path.join(self.output_dir, "error_analysis_summary.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Entity Resolution Validation Error Analysis\n\n")
            f.write(f"- **True Positive Pairs**: {total_tp}\n")
            f.write(f"- **False Positive Pairs**: {total_fp}\n")
            f.write(f"- **False Negative Pairs**: {total_fn}\n")
            f.write(f"- **Singleton False Positives**: {len(singleton_fps)}\n")
            f.write(f"- **Non-Singleton False Negatives**: {len(singleton_fns)}\n\n")
            f.write("### Error Pattern Insights\n")
            f.write("1. **High Name Similarity False Positives**: Different branches/franchises sharing name tokens.\n")
            f.write("2. **Transliteration & Typo False Negatives**: Indic script variations without sufficient token overlap.\n")
            f.write("3. **Empty Address Uncertainty**: When target address is missing, match confidence depends entirely on name specificity.\n")

        return report
