"""
Unit tests for data loader, profiler, and ground truth handler.
"""

import os
import tempfile
import pytest
from business_entity_resolution.src.data.loader import DataLoader
from business_entity_resolution.src.data.ground_truth import GroundTruthHandler
from business_entity_resolution.src.data.profiler import DataProfiler


@pytest.fixture
def sample_tsv_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        s1_path = os.path.join(tmpdir, "train_source1.tsv")
        with open(s1_path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-001\tAcme Corp\t123 Main St, Austin, TX\tUS\n")
            f.write("S1-002\tApex Infotech Pvt Ltd\t45 MG Road, Bangalore\tIndia\n")
            f.write("S1-003\tSolo Enterprise\t99 Ocean Ave, Miami, FL\tUS\n")

        gt_path = os.path.join(tmpdir, "train_ground_truth.tsv")
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-001\tS2-101,S3-201\n")
            f.write("S1-002\tS2-102\n")
            f.write("S1-003\t\n")

        yield tmpdir, s1_path, gt_path


def test_data_loader_iter(sample_tsv_data):
    tmpdir, s1_path, _ = sample_tsv_data
    loader = DataLoader()
    records = list(loader.iter_records(s1_path))
    assert len(records) == 3
    assert records[0]["entity_id"] == "S1-001"
    assert records[0]["country"] == "US"
    assert records[1]["business_name"] == "Apex Infotech Pvt Ltd"

    # Test country filter
    us_records = list(loader.iter_records(s1_path, target_country="US"))
    assert len(us_records) == 2


def test_ground_truth_handler(sample_tsv_data):
    tmpdir, _, gt_path = sample_tsv_data
    handler = GroundTruthHandler(gt_path)
    matches = handler.load()
    assert len(matches) == 3
    assert matches["S1-001"] == {"S2-101", "S3-201"}
    assert matches["S1-002"] == {"S2-102"}
    assert matches["S1-003"] == set()  # Singleton has empty set

    train_ids, val_ids = handler.split_s1_entities(list(matches.keys()), val_fraction=0.33, seed=42)
    assert set(train_ids).isdisjoint(set(val_ids))
    assert len(train_ids) + len(val_ids) == 3


def test_data_profiler(sample_tsv_data):
    tmpdir, s1_path, gt_path = sample_tsv_data
    profiler = DataProfiler(output_dir=os.path.join(tmpdir, "reports"))
    src_profile = profiler.profile_source(s1_path)
    assert src_profile["total_records"] == 3
    assert src_profile["missing_names"] == 0
    assert src_profile["country_distribution"] == {"US": 2, "India": 1}

    gt_profile = profiler.profile_ground_truth(gt_path)
    assert gt_profile["total_s1_entities"] == 3
    assert gt_profile["singletons_zero_matches"] == 1
    assert gt_profile["one_match"] == 1
    assert gt_profile["multi_matches"] == 1
