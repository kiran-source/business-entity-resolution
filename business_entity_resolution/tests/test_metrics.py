"""
Unit tests for official F0.5 challenge evaluation metric.
"""

import pytest
from business_entity_resolution.src.evaluation.f05 import compute_entity_f05, compute_macro_f05


def test_f05_official_example():
    # From README:
    # pred: [S2-00047, S2-00193, S3-00812]
    # gt: [S2-00047, S3-00812]
    # F0.5 should be approx 0.714
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    gt = {"S2-00047", "S3-00812"}
    p, r, f05 = compute_entity_f05(pred, gt)
    assert round(p, 3) == 0.667
    assert round(r, 3) == 1.000
    assert round(f05, 3) == 0.714


def test_f05_singletons():
    # Correct singleton
    p, r, f05 = compute_entity_f05(set(), set())
    assert (p, r, f05) == (1.0, 1.0, 1.0)

    # False merge on singleton
    p, r, f05 = compute_entity_f05({"S2-999"}, set())
    assert (p, r, f05) == (0.0, 0.0, 0.0)

    # Missed match on non-singleton
    p, r, f05 = compute_entity_f05(set(), {"S2-999"})
    assert (p, r, f05) == (0.0, 0.0, 0.0)


def test_macro_f05():
    gt = {
        "S1-1": {"S2-101", "S3-201"},
        "S1-2": set(),  # singleton
        "S1-3": {"S2-103"},
    }
    preds = {
        "S1-1": {"S2-101", "S3-201"},  # Perfect match -> 1.0
        "S1-2": set(),  # Perfect singleton -> 1.0
        "S1-3": set(),  # Missed match -> 0.0
    }
    res = compute_macro_f05(preds, gt)
    assert res["total_entities"] == 3
    assert res["singleton_count"] == 1
    assert res["singleton_accuracy"] == 1.0
    # Macro F0.5 = (1.0 + 1.0 + 0.0) / 3 = 0.66667
    assert round(res["macro_f05"], 4) == 0.6667
