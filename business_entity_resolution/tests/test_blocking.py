"""
Unit tests for blocking strategies and candidate generation.
"""

import pytest
from business_entity_resolution.src.preprocessing.name_normalizer import NameNormalizer
from business_entity_resolution.src.preprocessing.address_normalizer import AddressNormalizer
from business_entity_resolution.src.blocking.exact_blocking import ExactBlocker
from business_entity_resolution.src.blocking.token_blocking import TokenBlocker
from business_entity_resolution.src.blocking.tfidf_blocking import TfidfBlocker
from business_entity_resolution.src.blocking.address_blocking import AddressBlocker
from business_entity_resolution.src.blocking.candidate_generator import CandidateGenerator


def test_exact_blocker():
    nn = NameNormalizer()
    eb = ExactBlocker()

    target_name = nn.normalize("Roberts Titan Inc")
    eb.index_target("S2-001", "US", target_name)

    query_name = nn.normalize("Roberts Titan")
    cands = eb.block("S1-100", "US", query_name)
    assert "S2-001" in cands

    # Country mismatch should return empty
    cands_wrong_cty = eb.block("S1-100", "India", query_name)
    assert len(cands_wrong_cty) == 0


def test_token_blocker():
    nn = NameNormalizer()
    tb = TokenBlocker()

    target_name = nn.normalize("Moncada Learning Center")
    tb.index_target("S3-002", "US", target_name)

    query_name = nn.normalize("LLC Moncada Enterprises")
    cands = tb.block("S1-200", "US", query_name)
    assert "S3-002" in cands


def test_tfidf_blocker():
    tb = TfidfBlocker(top_k=2, min_similarity=0.2)
    ids = ["S2-10", "S2-20", "S2-30"]
    texts = ["roberts titan houston", "apex infotech bangalore", "global logistics chicago"]
    tb.fit_and_index_targets(ids, texts)

    queries = ["roberts titan inc texas"]
    res = tb.block_batch(["S1-1"], queries)
    assert "S1-1" in res
    assert "S2-10" in res["S1-1"]


def test_address_blocker():
    nn = NameNormalizer()
    an = AddressNormalizer()
    ab = AddressBlocker()

    t_name = nn.normalize("Apex Technologies")
    t_addr = an.normalize("123 Main St, Austin, TX 78701")
    ab.index_target("S2-55", "US", t_name, t_addr)

    q_name = nn.normalize("Apex Labs")
    q_addr = an.normalize("Suite 200, 123 Main Street, TX 78701")
    cands = ab.block("S1-99", "US", q_name, q_addr)
    assert "S2-55" in cands


def test_candidate_generator_integration():
    cg = CandidateGenerator(max_candidates_per_s1=10)
    cg.index_target_record("S2-1", "Roberts Titan Inc", "13834 Willowtwist St, Houston, TX", "US")
    cg.index_target_record("S3-1", "Apex Infotech Pvt Ltd", "45 MG Road, Bangalore", "India")
    cg.finalize_indexing()

    s1_records = [
        {"entity_id": "S1-1", "business_name": "Roberts Titan", "business_address": "Houston, Texas", "country": "US"}
    ]
    cand_map = cg.generate_candidates_batch(s1_records)
    assert "S1-1" in cand_map
    assert "S2-1" in cand_map["S1-1"]
    pair = cand_map["S1-1"]["S2-1"]
    assert pair.gen_count >= 1

    gt = {"S1-1": {"S2-1"}}
    cand_ids = {"S1-1": set(cand_map["S1-1"].keys())}
    recall_stats = CandidateGenerator.measure_candidate_recall(cand_ids, gt, total_targets=2)
    assert recall_stats["candidate_recall"] == 1.0
