"""
Train: Pairwise dataset builder with hard-negative mining,
LightGBM classifier training, validation scoring, and optimal threshold selection.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

from ..preprocessing.name_normalizer import NormalizedName, NameNormalizer
from ..preprocessing.address_normalizer import NormalizedAddress, AddressNormalizer
from ..preprocessing.country_normalizer import CountryNormalizer
from ..blocking.candidate_generator import CandidatePair
from ..features.pair_features import PairFeatures
from ..evaluation.threshold import ThresholdOptimizer
from .calibration import ProbabilityCalibrator
from .model_io import ModelIO


class PairDatasetBuilder:
    """Builds pairwise training features with hard-negative mining."""

    def __init__(
        self,
        neg_pos_ratio: float = 3.0,
        random_seed: int = 42,
    ):
        self.neg_pos_ratio = neg_pos_ratio
        self.rng = np.random.RandomState(random_seed)
        self.name_normalizer = NameNormalizer()
        self.addr_normalizer = AddressNormalizer()

    def build_pairs_data(
        self,
        s1_records: Dict[str, Dict[str, str]],
        target_records: Dict[str, Dict[str, str]],
        candidate_pairs_map: Dict[str, Dict[str, CandidatePair]],
        ground_truth: Dict[str, Set[str]],
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
        """
        Construct feature matrix X and label vector y.
        Incorporates true positive matches and mines hard negatives from blocking candidates.
        """
        # Precompute normalized representations for fast feature generation
        norm_s1_names = {}
        norm_s1_addrs = {}
        for sid, rec in s1_records.items():
            cty = CountryNormalizer.normalize(rec["country"])
            norm_s1_names[sid] = self.name_normalizer.normalize(rec["business_name"])
            norm_s1_addrs[sid] = self.addr_normalizer.normalize(rec["business_address"], cty)

        norm_tgt_names = {}
        norm_tgt_addrs = {}
        for tid, rec in target_records.items():
            cty = CountryNormalizer.normalize(rec["country"])
            norm_tgt_names[tid] = self.name_normalizer.normalize(rec["business_name"])
            norm_tgt_addrs[tid] = self.addr_normalizer.normalize(rec["business_address"], cty)

        rows = []
        labels = []
        pair_ids = []

        for sid, rec in s1_records.items():
            cty1 = CountryNormalizer.normalize(rec["country"])
            n1 = norm_s1_names[sid]
            a1 = norm_s1_addrs[sid]
            gt_mids = ground_truth.get(sid, set())
            cand_map = candidate_pairs_map.get(sid, {})

            positives = []
            negatives = []

            # 1. Process candidate pairs
            for tid, cpair in cand_map.items():
                if tid not in target_records:
                    continue
                is_match = tid in gt_mids
                if is_match:
                    positives.append((tid, cpair))
                else:
                    negatives.append((tid, cpair))

            # 2. Also ensure all ground truth matches in target_records are included
            for tid in gt_mids:
                if tid in target_records and tid not in cand_map:
                    # Ground truth match not caught by blocking (missed candidate)
                    cpair = CandidatePair(s1_id=sid, target_id=tid)
                    positives.append((tid, cpair))

            # Sample hard negatives based on neg_pos_ratio
            n_pos = len(positives)
            if n_pos > 0:
                max_negs = max(1, int(n_pos * self.neg_pos_ratio))
                if len(negatives) > max_negs:
                    # Sort negatives by gen_count (hardest negatives first: high similarity/blocking frequency)
                    negatives.sort(key=lambda x: x[1].gen_count, reverse=True)
                    sampled_negs = negatives[:max_negs]
                else:
                    sampled_negs = negatives
            else:
                # Singleton S1: keep top 2 hardest candidate negatives
                negatives.sort(key=lambda x: x[1].gen_count, reverse=True)
                sampled_negs = negatives[:2]

            # If no negatives were generated for this S1, pick a random target as a negative
            if not sampled_negs and target_records:
                for random_tid in target_records.keys():
                    if random_tid not in gt_mids:
                        cpair = CandidatePair(s1_id=sid, target_id=random_tid)
                        sampled_negs.append((random_tid, cpair))
                        break

            # Build feature rows
            for tid, cpair in positives:
                n2 = norm_tgt_names[tid]
                a2 = norm_tgt_addrs[tid]
                cty2 = CountryNormalizer.normalize(target_records[tid]["country"])
                feats = PairFeatures.extract_features_dict(n1, a1, cty1, n2, a2, cty2, cpair)
                rows.append(PairFeatures.dict_to_array(feats))
                labels.append(1)
                pair_ids.append((sid, tid))

            for tid, cpair in sampled_negs:
                n2 = norm_tgt_names[tid]
                a2 = norm_tgt_addrs[tid]
                cty2 = CountryNormalizer.normalize(target_records[tid]["country"])
                feats = PairFeatures.extract_features_dict(n1, a1, cty1, n2, a2, cty2, cpair)
                rows.append(PairFeatures.dict_to_array(feats))
                labels.append(0)
                pair_ids.append((sid, tid))

        if not rows:
            return (
                np.empty((0, len(PairFeatures.ALL_FEATURE_NAMES)), dtype=np.float32),
                np.empty(0, dtype=np.int32),
                [],
            )

        X = np.vstack(rows)
        y = np.array(labels, dtype=np.int32)
        return X, y, pair_ids


class ModelTrainer:
    """Trains and optimizes the gradient boosted pairwise match classifier."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        n_estimators: int = 250,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        random_state: int = 42,
    ):
        self.model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            importance_type="gain",
        )
        self.calibrator = ProbabilityCalibrator(method="sigmoid")
        self.optimal_threshold: float = 0.50

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Fit model on training pairs with validation monitoring."""
        has_val = (
            X_val is not None
            and y_val is not None
            and len(X_val) > 0
            and len(np.unique(y_val)) > 1
        )
        eval_set = [(X_val, y_val)] if has_val else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)] if eval_set else None,
        )

        metrics: Dict[str, Any] = {}
        if X_val is not None and y_val is not None and len(y_val) > 0:
            val_probs = self.model.predict_proba(X_val)[:, 1]
            try:
                metrics["val_roc_auc"] = round(roc_auc_score(y_val, val_probs), 4)
                metrics["val_pr_auc"] = round(average_precision_score(y_val, val_probs), 4)
            except Exception:
                pass

            # Fit calibrator on validation predictions
            self.calibrator.fit(val_probs, y_val)

        return metrics

    def optimize_threshold(
        self,
        val_candidate_pairs: Dict[str, Dict[str, CandidatePair]],
        val_s1_records: Dict[str, Dict[str, str]],
        target_records: Dict[str, Dict[str, str]],
        val_ground_truth: Dict[str, Set[str]],
    ) -> Dict[str, Any]:
        """
        Evaluate candidate probabilities across candidate pairs and find optimal threshold
        maximizing official Macro F0.5.
        """
        name_norm = NameNormalizer()
        addr_norm = AddressNormalizer()

        # Generate predictions for all validation candidate pairs
        cand_probs: Dict[str, Dict[str, float]] = {sid: {} for sid in val_s1_records.keys()}

        feature_batch = []
        pair_refs = []

        for sid, rec in val_s1_records.items():
            cty1 = CountryNormalizer.normalize(rec["country"])
            n1 = name_norm.normalize(rec["business_name"])
            a1 = addr_norm.normalize(rec["business_address"], cty1)

            cands = val_candidate_pairs.get(sid, {})
            for tid, cpair in cands.items():
                if tid not in target_records:
                    continue
                t_rec = target_records[tid]
                cty2 = CountryNormalizer.normalize(t_rec["country"])
                n2 = name_norm.normalize(t_rec["business_name"])
                a2 = addr_norm.normalize(t_rec["business_address"], cty2)

                feats = PairFeatures.extract_features_dict(n1, a1, cty1, n2, a2, cty2, cpair)
                feature_batch.append(PairFeatures.dict_to_array(feats))
                pair_refs.append((sid, tid))

        if feature_batch:
            X_eval = np.vstack(feature_batch)
            raw_probs = self.model.predict_proba(X_eval)[:, 1]
            cal_probs = self.calibrator.predict_proba(raw_probs)

            for (sid, tid), p in zip(pair_refs, cal_probs):
                cand_probs[sid][tid] = float(p)

        optimizer = ThresholdOptimizer(min_thresh=0.15, max_thresh=0.85, step=0.02)
        opt_results = optimizer.optimize(cand_probs, val_ground_truth)
        self.optimal_threshold = opt_results["optimal_threshold"]
        return opt_results

    def save(self, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model and optimal threshold."""
        ModelIO.save_model(
            self.model,
            filepath,
            threshold=self.optimal_threshold,
            metadata=metadata,
        )
