"""
Official Challenge Metric: Macro-averaged F_beta (beta = 0.5).
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Correctly accounts for singletons (empty ground truth vs empty predictions = 1.0).
"""

from typing import Dict, Set, List, Tuple, Any


def compute_entity_f05(
    predicted_ids: Set[str],
    ground_truth_ids: Set[str],
    beta: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F_0.5 for a single Source 1 entity.
    Returns: (precision, recall, f05)
    """
    pred_empty = len(predicted_ids) == 0
    gt_empty = len(ground_truth_ids) == 0

    if gt_empty and pred_empty:
        # Correctly predicted singleton: full credit
        return 1.0, 1.0, 1.0
    if gt_empty and not pred_empty:
        # False merge on singleton: precision 0, recall 1 (or 0), F0.5 = 0.0
        return 0.0, 0.0, 0.0
    if not gt_empty and pred_empty:
        # Missed match entirely: recall 0, precision 1 (or 0), F0.5 = 0.0
        return 0.0, 0.0, 0.0

    tp = len(predicted_ids & ground_truth_ids)
    precision = tp / len(predicted_ids)
    recall = tp / len(ground_truth_ids)

    beta_sq = beta ** 2  # 0.25
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        f05 = 0.0
    else:
        f05 = ((1.0 + beta_sq) * precision * recall) / denom

    return precision, recall, f05


def compute_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Compute macro-averaged Precision, Recall, and F_0.5 across all S1 entities.
    Every S1 entity in ground_truth is evaluated.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f05": 0.0, "total_entities": 0}

    precisions = []
    recalls = []
    f05s = []

    singleton_count = 0
    singleton_correct = 0
    multi_count = 0
    multi_f05s = []

    for s1_id, gt_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        p, r, f = compute_entity_f05(pred_ids, gt_ids)
        precisions.append(p)
        recalls.append(r)
        f05s.append(f)

        if len(gt_ids) == 0:
            singleton_count += 1
            if len(pred_ids) == 0:
                singleton_correct += 1
        else:
            multi_count += 1
            multi_f05s.append(f)

    return {
        "macro_precision": round(sum(precisions) / total_entities, 5),
        "macro_recall": round(sum(recalls) / total_entities, 5),
        "macro_f05": round(sum(f05s) / total_entities, 5),
        "total_entities": total_entities,
        "singleton_count": singleton_count,
        "singleton_accuracy": round(singleton_correct / max(1, singleton_count), 5),
        "non_singleton_count": multi_count,
        "non_singleton_f05": round(sum(multi_f05s) / max(1, multi_count), 5),
    }
