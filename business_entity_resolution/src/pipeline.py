"""
Pipeline: Main orchestrator for the Business Entity Resolution Challenge.
Coordinates profiling, training, validation, threshold optimization, inference,
validation checks, and submission packaging.
"""

import os
import sys
import time
import json
import subprocess
from typing import Dict, List, Set, Any, Optional

from .utils.logging_utils import setup_logger
from .utils.reproducibility import set_seed
from .utils.config import ConfigLoader
from .data.loader import DataLoader
from .data.ground_truth import GroundTruthHandler
from .data.profiler import DataProfiler
from .preprocessing.country_normalizer import CountryNormalizer
from .blocking.candidate_generator import CandidateGenerator
from .models.train import PairDatasetBuilder, ModelTrainer
from .models.predict import PairPredictor
from .evaluation.f05 import compute_macro_f05
from .evaluation.error_analysis import ErrorAnalyzer
from .output.matching_results import MatchingResultsWriter
from .output.candidate_pairs import CandidatePairsWriter
from .output.submission import SubmissionPackager


class EntityResolutionPipeline:
    """End-to-end reproducible pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigLoader.load(config_path)
        self.logger = setup_logger("EntityResolutionPipeline")
        set_seed(self.config.get("random_seed", 42))

        self.loader = DataLoader()
        self.trainer: Optional[ModelTrainer] = None
        self.optimal_threshold: float = 0.50

    def run_profiling(self):
        """Phase 2: Generate data profiling report."""
        self.logger.info("Running Data Profiling...")
        train_dir = self.config["data"]["train_dir"]
        test_dir = self.config["data"]["test_dir"]
        reports_dir = self.config["output"]["reports_dir"]

        profiler = DataProfiler(output_dir=reports_dir)
        report = profiler.generate_full_report(train_dir=train_dir, test_dir=test_dir)
        self.logger.info(
            f"Profiling complete. Ground truth S1 count: {report.get('ground_truth', {}).get('total_s1_entities', 0)}"
        )

    def train_and_validate(self) -> Dict[str, Any]:
        """Phases 5-11: Train candidate blocker, build pairs, train ML model, optimize threshold."""
        t0 = time.time()
        self.logger.info("Starting Training and Validation Phase...")

        gt_path = self.config["data"]["train_ground_truth"]
        gt_handler = GroundTruthHandler(gt_path)
        gt_map = gt_handler.load()

        sample_s1_n = self.config["data"].get("sample_size_train", 12000)
        sample_tgt_n = self.config["data"].get("sample_size_targets", 120000)

        # 1. Load S1 entities
        s1_records_map: Dict[str, Dict[str, str]] = {}
        target_ids_needed: Set[str] = set()
        s1_file = self.config["data"]["train_source1"]

        for rec in self.loader.iter_records(s1_file, max_records=sample_s1_n):
            sid = rec["entity_id"]
            s1_records_map[sid] = rec
            mids = gt_map.get(sid, set())
            target_ids_needed.update(mids)

        s1_ids = list(s1_records_map.keys())
        val_fraction = self.config["data"].get("val_fraction", 0.15)
        train_s1_ids, val_s1_ids = gt_handler.split_s1_entities(
            s1_ids, val_fraction=val_fraction, seed=self.config.get("random_seed", 42)
        )

        train_s1_records = {sid: s1_records_map[sid] for sid in train_s1_ids}
        val_s1_records = {sid: s1_records_map[sid] for sid in val_s1_ids}
        val_gt = {sid: gt_map.get(sid, set()) for sid in val_s1_ids}

        self.logger.info(
            f"Split {len(s1_ids)} S1 records into {len(train_s1_ids)} train and {len(val_s1_ids)} validation (zero leakage)."
        )

        # 2. Index targets: true targets + background noise targets
        cg = CandidateGenerator(
            max_candidates_per_s1=self.config["blocking"]["max_candidates_per_s1"],
            enable_tfidf=self.config["blocking"]["enable_tfidf"],
            tfidf_min_similarity=self.config["blocking"]["tfidf_min_similarity"],
            tfidf_top_k=self.config["blocking"]["tfidf_top_k"],
        )

        target_records: Dict[str, Dict[str, str]] = {}
        indexed_needed = 0
        noise_budget = sample_tgt_n // 2

        for src_file in [self.config["data"]["train_source2"], self.config["data"]["train_source3"]]:
            with open(src_file, "r", encoding="utf-8") as f:
                next(f)
                noise_count = 0
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) < 4:
                        continue
                    eid, bname, baddr, bcty = parts[0], parts[1], parts[2], parts[3]
                    is_needed = eid in target_ids_needed
                    if is_needed or noise_count < noise_budget:
                        trec = {
                            "entity_id": eid,
                            "business_name": bname,
                            "business_address": baddr,
                            "country": bcty,
                        }
                        target_records[eid] = trec
                        cg.index_target_record(eid, bname, baddr, bcty)
                        if is_needed:
                            indexed_needed += 1
                        else:
                            noise_count += 1
                    if indexed_needed >= len(target_ids_needed) and noise_count >= noise_budget:
                        break

        cg.finalize_indexing()
        self.logger.info(
            f"Indexed {len(target_records)} target records ({indexed_needed}/{len(target_ids_needed)} true targets indexed)."
        )

        # 3. Generate candidate pairs
        self.logger.info("Generating candidate pairs for train and validation S1 entities...")
        train_cand_pairs = cg.generate_candidates_batch(list(train_s1_records.values()))
        val_cand_pairs = cg.generate_candidates_batch(list(val_s1_records.values()))

        # Evaluate candidate recall on validation set
        val_cand_id_sets = {sid: set(cand_map.keys()) for sid, cand_map in val_cand_pairs.items()}
        cand_recall_stats = CandidateGenerator.measure_candidate_recall(
            val_cand_id_sets, val_gt, total_targets=len(target_records)
        )
        self.logger.info(
            f"Validation Candidate Recall: {cand_recall_stats['candidate_recall_pct']}% "
            f"(Avg candidates/S1: {cand_recall_stats['avg_candidates_per_s1']}, Reduction: {cand_recall_stats['reduction_ratio'] * 100:.3f}%)"
        )

        # 4. Build feature datasets with hard negatives
        self.logger.info("Building pairwise feature matrices with hard negative mining...")
        pair_builder = PairDatasetBuilder(
            neg_pos_ratio=self.config["model"]["neg_pos_ratio"],
            random_seed=self.config.get("random_seed", 42),
        )

        X_train, y_train, _ = pair_builder.build_pairs_data(
            train_s1_records, target_records, train_cand_pairs, gt_map
        )
        X_val, y_val, _ = pair_builder.build_pairs_data(
            val_s1_records, target_records, val_cand_pairs, val_gt
        )
        self.logger.info(f"Train pairs shape: {X_train.shape} (positives: {int(y_train.sum())}, negatives: {len(y_train) - int(y_train.sum())})")
        self.logger.info(f"Val pairs shape: {X_val.shape} (positives: {int(y_val.sum())}, negatives: {len(y_val) - int(y_val.sum())})")

        # 5. Train LightGBM classifier
        self.logger.info("Training LightGBM pairwise match classifier...")
        trainer = ModelTrainer(
            learning_rate=self.config["model"]["learning_rate"],
            n_estimators=self.config["model"]["n_estimators"],
            num_leaves=self.config["model"]["num_leaves"],
            min_child_samples=self.config["model"]["min_child_samples"],
            random_state=self.config.get("random_seed", 42),
        )
        fit_metrics = trainer.train(X_train, y_train, X_val, y_val)
        self.logger.info(f"Fit metrics: {fit_metrics}")

        # 6. Optimize F0.5 threshold on validation S1 entities
        self.logger.info("Optimizing decision threshold to maximize macro F0.5...")
        opt_results = trainer.optimize_threshold(
            val_candidate_pairs=val_cand_pairs,
            val_s1_records=val_s1_records,
            target_records=target_records,
            val_ground_truth=val_gt,
        )

        self.optimal_threshold = opt_results["optimal_threshold"]
        self.logger.info(
            f"=== OPTIMAL THRESHOLD FOUND: {self.optimal_threshold} ===\n"
            f"  Macro F0.5:      {opt_results['best_macro_f05']:.4f}\n"
            f"  Macro Precision: {opt_results['best_macro_precision']:.4f}\n"
            f"  Macro Recall:    {opt_results['best_macro_recall']:.4f}\n"
            f"  Singleton Acc:   {opt_results['singleton_accuracy']:.4f}"
        )

        # 7. Save model
        model_save_path = self.config["model"]["save_model_path"]
        trainer.save(model_save_path, metadata={"opt_results": opt_results, "recall_stats": cand_recall_stats})
        self.trainer = trainer

        # 8. Error Analysis on validation set
        self.logger.info("Running validation error analysis...")
        analyzer = ErrorAnalyzer(output_dir=os.path.join(self.config["output"]["reports_dir"], "error_analysis"))
        # Predict at optimal threshold
        predictor = PairPredictor(booster=trainer.model.booster_, threshold=self.optimal_threshold)
        val_preds, _ = predictor.predict_candidates(
            s1_records=list(val_s1_records.values()),
            candidate_pairs_map=val_cand_pairs,
            target_records=target_records,
        )
        err_report = analyzer.analyze(val_preds, val_gt, val_s1_records, target_records)
        self.logger.info(f"Error report generated: {err_report['aggregate']}")

        elapsed = time.time() - t0
        res = {
            "candidate_recall": cand_recall_stats,
            "threshold_optimization": opt_results,
            "fit_metrics": fit_metrics,
            "runtime_seconds": round(elapsed, 2),
        }

        # Save experiment results
        exp_file = os.path.join(self.config["output"]["experiments_dir"], "training_experiment.json")
        os.makedirs(os.path.dirname(exp_file), exist_ok=True)
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        return res

    def run_inference(self, test_s1_limit: Optional[int] = None):
        """Phases 16-17: Run test inference across US, India, and France."""
        self.logger.info("Running Test Inference...")
        if self.trainer is None:
            raise RuntimeError("Pipeline has not been trained yet!")

        test_s1_file = self.config["data"]["test_source1"]
        test_s2_file = self.config["data"]["test_source2"]
        test_s3_file = self.config["data"]["test_source3"]

        # 1. Read required test S1 IDs
        required_s1_ids = []
        s1_by_country: Dict[str, List[Dict[str, str]]] = {}
        for rec in self.loader.iter_records(test_s1_file, max_records=test_s1_limit):
            sid = rec["entity_id"]
            required_s1_ids.append(sid)
            cty = CountryNormalizer.normalize(rec["country"])
            if cty not in s1_by_country:
                s1_by_country[cty] = []
            s1_by_country[cty].append(rec)

        self.logger.info(
            f"Loaded {len(required_s1_ids)} test S1 entities across countries: {list(s1_by_country.keys())}"
        )

        all_matches: Dict[str, Set[str]] = {sid: set() for sid in required_s1_ids}
        all_candidates: Dict[str, Set[str]] = {sid: set() for sid in required_s1_ids}

        predictor = PairPredictor(
            booster=self.trainer.model.booster_,
            threshold=self.optimal_threshold,
        )

        # 2. Process country partitions (US, India, France)
        for country, s1_records in s1_by_country.items():
            self.logger.info(f"Processing partition: Country={country} ({len(s1_records)} S1 entities)...")

            # Load S2 and S3 targets for this country
            target_records: Dict[str, Dict[str, str]] = {}
            cg = CandidateGenerator(
                max_candidates_per_s1=self.config["blocking"]["max_candidates_per_s1"],
                enable_tfidf=self.config["blocking"]["enable_tfidf"],
                tfidf_min_similarity=self.config["blocking"]["tfidf_min_similarity"],
                tfidf_top_k=self.config["blocking"]["tfidf_top_k"],
            )

            for target_file in [test_s2_file, test_s3_file]:
                # Stream targets for this country (e.g. France, US, India)
                # Sample a sufficient target pool per country partition
                target_budget = 40000
                count = 0
                for trec in self.loader.iter_records(target_file, target_country=country):
                    eid = trec["entity_id"]
                    target_records[eid] = trec
                    cg.index_target_record(eid, trec["business_name"], trec["business_address"], country)
                    count += 1
                    if count >= target_budget:
                        break

            cg.finalize_indexing()
            self.logger.info(f"Indexed {len(target_records)} {country} targets. Running blocking and inference...")

            # Run candidate generation in chunks to conserve memory
            chunk_size = 1000
            for i in range(0, len(s1_records), chunk_size):
                chunk_s1 = s1_records[i : i + chunk_size]
                chunk_cands_map = cg.generate_candidates_batch(chunk_s1)

                # Store candidate pairs (strictly before ML pruning)
                for rec in chunk_s1:
                    sid = rec["entity_id"]
                    cands = set(chunk_cands_map.get(sid, {}).keys())
                    all_candidates[sid] = cands

                # Predict matches
                chunk_matches, _ = predictor.predict_candidates(
                    s1_records=chunk_s1,
                    candidate_pairs_map=chunk_cands_map,
                    target_records=target_records,
                )

                for sid, mids in chunk_matches.items():
                    # Guarantee subset
                    all_matches[sid] = mids & all_candidates[sid]

        # 3. Write outputs
        matching_out = self.config["output"]["matching_file"]
        candidate_out = self.config["output"]["candidate_file"]

        self.logger.info(f"Writing matching results to {matching_out}...")
        MatchingResultsWriter.write(matching_out, required_s1_ids, all_matches)

        self.logger.info(f"Writing candidate pairs to {candidate_out}...")
        CandidatePairsWriter.write(candidate_out, required_s1_ids, all_candidates)

        # 4. Pre-validate
        self.logger.info("Running pre-validation checks...")
        valid, errors = SubmissionPackager.pre_validate(
            matching_out, candidate_out, set(required_s1_ids)
        )
        if not valid:
            self.logger.error(f"Pre-validation failed: {errors}")
            raise ValueError(f"Output validation failed: {errors}")
        self.logger.info("Pre-validation PASSED!")

        # 5. Run official challenge validator
        validator_script = os.path.join(os.path.dirname(self.config["data"]["train_dir"]), "utils", "validate_submission.py")
        if os.path.exists(validator_script):
            self.logger.info(f"Running official validator: {validator_script}")
            cmd = [
                sys.executable,
                validator_script,
                "--matching", matching_out,
                "--candidate", candidate_out,
                "--test-dir", self.config["data"]["test_dir"],
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.logger.info(f"Official Validator Output:\n{proc.stdout}")
            if proc.returncode != 0:
                self.logger.error(f"Official validator failed with code {proc.returncode}!\n{proc.stderr}")

        # 6. Package final submission ZIP
        zip_path = self.config["output"]["submission_zip"]
        code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        doc_path = os.path.join(code_dir, "Documentation_template.md")

        self.logger.info(f"Creating submission zip: {zip_path}...")
        SubmissionPackager.create_submission_zip(
            zip_path=zip_path,
            output_dir=self.config["output"]["output_dir"],
            code_dir=code_dir,
            documentation_path=doc_path,
        )
        self.logger.info(f"Submission ZIP successfully packaged: {zip_path}")
        return zip_path
