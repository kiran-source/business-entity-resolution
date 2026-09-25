"""
Integration test for full end-to-end entity resolution pipeline.
"""

import os
import tempfile
import zipfile
import pytest
import yaml

from business_entity_resolution.src.pipeline import EntityResolutionPipeline


@pytest.fixture
def mini_challenge_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = os.path.join(tmpdir, "train")
        test_dir = os.path.join(tmpdir, "test")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # Train S1
        with open(os.path.join(train_dir, "train_source1.tsv"), "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-1\tAcme Corp\t123 Main St, Austin, TX\tUS\n")
            f.write("S1-2\tApex Infotech Pvt Ltd\t45 MG Road, Bangalore\tIndia\n")
            f.write("S1-3\tSolo Enterprise\t99 Ocean Ave, Miami, FL\tUS\n")
            f.write("S1-4\tGlobal Tech Inc\t500 5th Ave, New York, NY\tUS\n")
            f.write("S1-5\tMumbai Clinic\t12 SV Road, Mumbai, Maharashtra\tIndia\n")
            f.write("S1-6\tDelta Systems\t10 Elm St, Boston, MA\tUS\n")

        # Train S2
        with open(os.path.join(train_dir, "train_source2.tsv"), "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-10\tAcme Corporation\t123 Main Street, Austin, Texas\tUS\n")
            f.write("S2-20\tApex Infotech\t45 MG Rd, Bangalore, Karnataka\tIndia\n")
            f.write("S2-40\tGlobal Tech\t500 Fifth Avenue, NY\tUS\n")
            f.write("S2-50\tMumbai Clinic Pvt Ltd\t12 SV Rd, Mumbai\tIndia\n")
            f.write("S2-99\tUnrelated Co\t100 Wall St, New York\tUS\n")

        # Train S3
        with open(os.path.join(train_dir, "train_source3.tsv"), "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-10\tAcme Incorporated\t123 Main St, Austin\tUS\n")
            f.write("S3-40\tGlobal Tech Systems\t500 5th Ave, New York\tUS\n")
            f.write("S3-99\tNoise Corp\t200 Broadway, New York\tUS\n")

        # Ground truth
        with open(os.path.join(train_dir, "train_ground_truth.tsv"), "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS2-10,S3-10\n")
            f.write("S1-2\tS2-20\n")
            f.write("S1-3\t\n")
            f.write("S1-4\tS2-40,S3-40\n")
            f.write("S1-5\tS2-50\n")
            f.write("S1-6\t\n")

        # Test S1 (includes France!)
        with open(os.path.join(test_dir, "test_source1.tsv"), "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S1-T1\tAcme Corp\t123 Main St, Austin, TX\tUS\n")
            f.write("S1-T2\tMarina Ecole France SARL\t63 Rue de Dieppe, Lille\tFrance\n")
            f.write("S1-T3\tSolo Inc\t10 Ocean Dr, Miami\tUS\n")

        # Test S2
        with open(os.path.join(test_dir, "test_source2.tsv"), "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S2-T10\tAcme Corporation\t123 Main Street, Austin, Texas\tUS\n")
            f.write("S2-T20\tMarina Ecole France\t63 R. DE DIEPPE, LILLE\tFrance\n")

        # Test S3
        with open(os.path.join(test_dir, "test_source3.tsv"), "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.write("S3-T10\tAcme Inc\t123 Main St, Austin\tUS\n")
            f.write("S3-T20\tOther France Club\t10 Paris Rd\tFrance\n")

        # Dummy doc template
        doc_path = os.path.join(tmpdir, "Documentation_template.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("# Documentation\n")

        cfg = {
            "random_seed": 42,
            "data": {
                "train_dir": train_dir,
                "test_dir": test_dir,
                "train_source1": os.path.join(train_dir, "train_source1.tsv"),
                "train_source2": os.path.join(train_dir, "train_source2.tsv"),
                "train_source3": os.path.join(train_dir, "train_source3.tsv"),
                "train_ground_truth": os.path.join(train_dir, "train_ground_truth.tsv"),
                "test_source1": os.path.join(test_dir, "test_source1.tsv"),
                "test_source2": os.path.join(test_dir, "test_source2.tsv"),
                "test_source3": os.path.join(test_dir, "test_source3.tsv"),
                "val_fraction": 0.33,
                "sample_size_train": 10,
                "sample_size_targets": 20,
            },
            "blocking": {
                "max_candidates_per_s1": 10,
                "enable_tfidf": False,  # disable tfidf on tiny 3-row test
                "tfidf_min_similarity": 0.2,
                "tfidf_top_k": 5,
            },
            "model": {
                "learning_rate": 0.1,
                "n_estimators": 10,
                "num_leaves": 7,
                "min_child_samples": 1,
                "neg_pos_ratio": 2.0,
                "save_model_path": os.path.join(tmpdir, "model.txt"),
            },
            "output": {
                "output_dir": out_dir,
                "matching_file": os.path.join(out_dir, "matching_results.tsv"),
                "candidate_file": os.path.join(out_dir, "candidate_pairs.tsv"),
                "submission_zip": os.path.join(tmpdir, "team_submission.zip"),
                "reports_dir": os.path.join(tmpdir, "reports"),
                "experiments_dir": os.path.join(tmpdir, "experiments"),
            },
        }

        cfg_path = os.path.join(tmpdir, "config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        yield tmpdir, cfg_path


def test_full_pipeline_run(mini_challenge_env):
    tmpdir, cfg_path = mini_challenge_env

    pipeline = EntityResolutionPipeline(config_path=cfg_path)
    pipeline.run_profiling()

    train_res = pipeline.train_and_validate()
    assert "threshold_optimization" in train_res
    assert pipeline.optimal_threshold > 0.0

    zip_path = pipeline.run_inference()
    assert os.path.exists(zip_path)

    # Verify zip content
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "output/matching_results.tsv" in namelist
        assert "output/candidate_pairs.tsv" in namelist

    # Verify matching_results.tsv formatting
    matching_file = pipeline.config["output"]["matching_file"]
    with open(matching_file, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\r\n").split("\t") for l in f]

    assert lines[0] == ["source1_entity_id", "matched_entity_ids"]
    assert len(lines) == 4  # Header + 3 test S1 entities
    s1_ids_in_output = [l[0] for l in lines[1:]]
    assert set(s1_ids_in_output) == {"S1-T1", "S1-T2", "S1-T3"}
