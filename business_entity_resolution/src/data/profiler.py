"""
DataProfiler: Profiles entity resolution datasets, calculating completeness,
noise patterns, script distributions, length distributions, and ground truth characteristics.
"""

import os
import re
import json
from collections import Counter
from typing import Dict, Any, List, Optional
import pandas as pd


class DataProfiler:
    """Computes comprehensive profiling metrics across entity sources and ground truth."""

    DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")
    ACCENT_REGEX = re.compile(r"[àâäéèêëîïôöùûüÿçÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ]")
    GUJARATI_REGEX = re.compile(r"[\u0A80-\u0AFF]")

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def profile_source(
        self,
        filepath: str,
        sample_limit: int = 250000,
    ) -> Dict[str, Any]:
        """Profile a single source TSV file."""
        fname = os.path.basename(filepath)
        total_records = 0
        missing_name = 0
        missing_address = 0
        missing_country = 0
        country_counts = Counter()

        name_lengths = []
        addr_lengths = []
        name_token_counts = []
        addr_token_counts = []

        devanagari_count = 0
        gujarati_count = 0
        accent_count = 0

        name_sample = []
        addr_sample = []

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            id_idx = header.index("entity_id") if "entity_id" in header else 0
            name_idx = header.index("business_name") if "business_name" in header else 1
            addr_idx = header.index("business_address") if "business_address" in header else 2
            cty_idx = header.index("country") if "country" in header else 3

            for line in f:
                if not line.strip():
                    continue
                total_records += 1
                parts = line.rstrip("\r\n").split("\t")
                while len(parts) < 4:
                    parts.append("")

                eid = parts[id_idx].strip()
                bname = parts[name_idx].strip()
                baddr = parts[addr_idx].strip()
                bcty = parts[cty_idx].strip()

                if not bname:
                    missing_name += 1
                if not baddr:
                    missing_address += 1
                if not bcty:
                    missing_country += 1
                else:
                    country_counts[bcty] += 1

                if total_records <= sample_limit:
                    name_sample.append(bname)
                    addr_sample.append(baddr)
                    name_lengths.append(len(bname))
                    addr_lengths.append(len(baddr))
                    name_token_counts.append(len(bname.split()))
                    addr_token_counts.append(len(baddr.split()))

                    if self.DEVANAGARI_REGEX.search(bname):
                        devanagari_count += 1
                    if self.GUJARATI_REGEX.search(bname):
                        gujarati_count += 1
                    if self.ACCENT_REGEX.search(bname):
                        accent_count += 1

        sampled_n = len(name_sample)
        unique_names = len(set(name_sample))
        unique_addrs = len(set(addr_sample))
        unique_combos = len(set(zip(name_sample, addr_sample)))

        stats = {
            "file_name": fname,
            "total_records": total_records,
            "sampled_records": sampled_n,
            "missing_names": missing_name,
            "missing_name_pct": round(missing_name / max(1, total_records) * 100, 4),
            "missing_addresses": missing_address,
            "missing_address_pct": round(missing_address / max(1, total_records) * 100, 4),
            "missing_countries": missing_country,
            "missing_country_pct": round(missing_country / max(1, total_records) * 100, 4),
            "country_distribution": dict(country_counts),
            "sample_duplicate_name_pct": round((sampled_n - unique_names) / max(1, sampled_n) * 100, 2),
            "sample_duplicate_addr_pct": round((sampled_n - unique_addrs) / max(1, sampled_n) * 100, 2),
            "sample_duplicate_combo_pct": round((sampled_n - unique_combos) / max(1, sampled_n) * 100, 2),
            "name_length_mean": round(sum(name_lengths) / max(1, sampled_n), 2),
            "name_tokens_mean": round(sum(name_token_counts) / max(1, sampled_n), 2),
            "addr_length_mean": round(sum(addr_lengths) / max(1, sampled_n), 2),
            "addr_tokens_mean": round(sum(addr_token_counts) / max(1, sampled_n), 2),
            "devanagari_script_pct": round(devanagari_count / max(1, sampled_n) * 100, 2),
            "gujarati_script_pct": round(gujarati_count / max(1, sampled_n) * 100, 2),
            "accented_chars_pct": round(accent_count / max(1, sampled_n) * 100, 2),
        }
        return stats

    def profile_ground_truth(self, filepath: str) -> Dict[str, Any]:
        """Profile ground truth matching distribution."""
        total_s1 = 0
        zero_matches = 0
        one_match = 0
        multi_matches = 0
        match_counts = Counter()
        s2_count = 0
        s3_count = 0

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            for line in f:
                if not line.strip():
                    continue
                total_s1 += 1
                parts = line.rstrip("\r\n").split("\t")
                mids_str = parts[1] if len(parts) > 1 else ""
                if not mids_str.strip():
                    zero_matches += 1
                    match_counts[0] += 1
                else:
                    mids = [m.strip() for m in mids_str.split(",") if m.strip()]
                    cnt = len(mids)
                    match_counts[cnt] += 1
                    if cnt == 1:
                        one_match += 1
                    else:
                        multi_matches += 1
                    for mid in mids:
                        if mid.startswith("S2-"):
                            s2_count += 1
                        elif mid.startswith("S3-"):
                            s3_count += 1

        total_pairs = sum(k * v for k, v in match_counts.items())
        return {
            "total_s1_entities": total_s1,
            "singletons_zero_matches": zero_matches,
            "singletons_pct": round(zero_matches / max(1, total_s1) * 100, 2),
            "one_match": one_match,
            "one_match_pct": round(one_match / max(1, total_s1) * 100, 2),
            "multi_matches": multi_matches,
            "multi_matches_pct": round(multi_matches / max(1, total_s1) * 100, 2),
            "total_matched_pairs": total_pairs,
            "mean_matches_per_s1": round(total_pairs / max(1, total_s1), 3),
            "max_matches": max(match_counts.keys()) if match_counts else 0,
            "match_distribution": {str(k): v for k, v in sorted(match_counts.items())},
            "total_s2_matches": s2_count,
            "total_s3_matches": s3_count,
        }

    def generate_full_report(
        self,
        train_dir: str,
        test_dir: str,
    ) -> Dict[str, Any]:
        """Generate, save, and return a full profiling report across all 7 files."""
        report = {}
        gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
        if os.path.exists(gt_path):
            report["ground_truth"] = self.profile_ground_truth(gt_path)

        for fname in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
            p = os.path.join(train_dir, fname)
            if os.path.exists(p):
                report[fname] = self.profile_source(p)

        for fname in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
            p = os.path.join(test_dir, fname)
            if os.path.exists(p):
                report[fname] = self.profile_source(p)

        json_path = os.path.join(self.output_dir, "profiling_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        md_path = os.path.join(self.output_dir, "profiling_summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Entity Resolution Data Profiling Summary Report\n\n")
            f.write("## Ground Truth Distribution\n\n")
            if "ground_truth" in report:
                gt = report["ground_truth"]
                f.write(f"- **Total S1 Entities**: {gt['total_s1_entities']:,}\n")
                f.write(f"- **Zero Matches (Singletons)**: {gt['singletons_zero_matches']:,} ({gt['singletons_pct']}%)\n")
                f.write(f"- **Exactly 1 Match**: {gt['one_match']:,} ({gt['one_match_pct']}%)\n")
                f.write(f"- **Multiple Matches**: {gt['multi_matches']:,} ({gt['multi_matches_pct']}%)\n")
                f.write(f"- **Total Positive Pairs**: {gt['total_matched_pairs']:,}\n")
                f.write(f"- **Average Matches per S1**: {gt['mean_matches_per_s1']}\n")
                f.write(f"- **Max Matches for single S1**: {gt['max_matches']}\n\n")

            f.write("## Source File Completeness and Distributions\n\n")
            f.write("| File | Total Records | Missing Names | Missing Addr (%) | Countries |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for k, v in report.items():
                if k == "ground_truth":
                    continue
                cty_str = ", ".join([f"{c}: {n:,}" for c, n in v.get("country_distribution", {}).items()])
                f.write(f"| {v['file_name']} | {v['total_records']:,} | {v['missing_names']} | {v['missing_address_pct']}% | {cty_str} |\n")

        return report
