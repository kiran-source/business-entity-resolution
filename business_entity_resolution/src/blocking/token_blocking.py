"""
TokenBlocker: Inverted token index blocking with frequency filtering to prevent explosion.
Retrieves candidate pairs sharing rare, highly discriminative business tokens within country.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
from ..preprocessing.name_normalizer import NormalizedName


class TokenBlocker:
    """Inverted token index with stop-word / frequency pruning."""

    COMMON_BUSINESS_TOKENS = {
        "and", "the", "of", "in", "for", "with", "at", "by", "from",
        "inc", "corp", "llc", "ltd", "pvt", "limited", "company", "co",
        "services", "solutions", "enterprises", "associates", "group",
        "technologies", "international", "global", "national", "management",
        "consulting", "holding", "holdings", "industries", "systems", "trading",
    }

    def __init__(self, max_postings: int = 150):
        self.max_postings = max_postings
        # (country, token) -> list of entity_ids
        self.inverted_index = defaultdict(list)

    def index_target(self, entity_id: str, country: str, norm_name: NormalizedName):
        """Index discriminative tokens from a target record."""
        cty = country.lower()
        for tok in norm_name.tokens:
            if len(tok) >= 4 and tok not in self.COMMON_BUSINESS_TOKENS:
                # Cap posting list to avoid memory/time explosion on frequent words
                postings = self.inverted_index[(cty, tok)]
                if len(postings) < self.max_postings:
                    postings.append(entity_id)

    def block(self, s1_id: str, country: str, norm_name: NormalizedName) -> Set[str]:
        """Retrieve candidates sharing discriminative tokens with S1."""
        candidates = set()
        cty = country.lower()
        for tok in norm_name.tokens:
            if len(tok) >= 4 and tok not in self.COMMON_BUSINESS_TOKENS:
                candidates.update(self.inverted_index.get((cty, tok), []))
        return candidates
