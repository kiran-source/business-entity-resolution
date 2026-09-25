"""
Candidate generation and blocking strategies for scalable entity resolution.
"""
from .exact_blocking import ExactBlocker
from .token_blocking import TokenBlocker
from .tfidf_blocking import TfidfBlocker
from .address_blocking import AddressBlocker
from .candidate_generator import CandidateGenerator, CandidatePair

__all__ = [
    "ExactBlocker",
    "TokenBlocker",
    "TfidfBlocker",
    "AddressBlocker",
    "CandidateGenerator",
    "CandidatePair",
]
