"""
Unit tests for output formatting, pre-validation, and submission packaging.
"""

import os
import tempfile
import pytest
from business_entity_resolution.src.output.matching_results import MatchingResultsWriter
from business_entity_resolution.src.output.candidate_pairs import CandidatePairsWriter
from business_entity_resolution.src.output.submission import SubmissionPackager


@pytest.fixture
def output_tempdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_matching_results_writer(output_tempdir):
    out_path = os.path.join(output_tempdir, "matching_results.tsv")
    req_ids = ["S1-1", "S1-2", "S1-3"]
    matches = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": set(),  # singleton
        "S1-3": {"S2-30"},
    }

    MatchingResultsWriter.write(out_path, req_ids, matches)
    assert os.path.exists(out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\r\n").split("\t") for line in f]

    assert lines[0] == ["source1_entity_id", "matched_entity_ids"]
    assert lines[1] == ["S1-1", "S2-10,S3-20"]
    assert lines[2] == ["S1-2", ""]  # Empty singleton
    assert lines[3] == ["S1-3", "S2-30"]


def test_candidate_pairs_writer(output_tempdir):
    out_path = os.path.join(output_tempdir, "candidate_pairs.tsv")
    req_ids = ["S1-1", "S1-2"]
    candidates = {
        "S1-1": {"S2-10", "S3-20", "S2-99"},
        "S1-2": set(),
    }

    CandidatePairsWriter.write(out_path, req_ids, candidates)
    assert os.path.exists(out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\r\n").split("\t") for line in f]

    assert lines[0] == ["source1_entity_id", "candidate_entity_ids"]
    assert lines[1] == ["S1-1", "S2-10,S2-99,S3-20"]


def test_submission_pre_validation(output_tempdir):
    match_file = os.path.join(output_tempdir, "matching_results.tsv")
    cand_file = os.path.join(output_tempdir, "candidate_pairs.tsv")

    req_ids = {"S1-1", "S1-2"}
    matches = {"S1-1": {"S2-10"}, "S1-2": set()}
    cands = {"S1-1": {"S2-10", "S3-99"}, "S1-2": set()}

    MatchingResultsWriter.write(match_file, list(req_ids), matches)
    CandidatePairsWriter.write(cand_file, list(req_ids), cands)

    valid, errors = SubmissionPackager.pre_validate(match_file, cand_file, req_ids)
    assert valid is True
    assert len(errors) == 0

    # Test error when match is NOT in candidate pairs
    bad_cands = {"S1-1": {"S3-99"}, "S1-2": set()}
    CandidatePairsWriter.write(cand_file, list(req_ids), bad_cands)
    valid_bad, errors_bad = SubmissionPackager.pre_validate(match_file, cand_file, req_ids)
    assert valid_bad is False
    assert any("not in candidate_pairs" in e for e in errors_bad)
