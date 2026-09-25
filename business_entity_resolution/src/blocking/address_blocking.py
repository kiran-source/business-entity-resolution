"""
AddressBlocker: Generates candidate pairs based on composite address keys
(postal code + name prefix, building number + street token) to capture matches
with severe name variation or transliteration.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
from ..preprocessing.name_normalizer import NormalizedName
from ..preprocessing.address_normalizer import NormalizedAddress


class AddressBlocker:
    """Composite address key blocking within country."""

    def __init__(self, max_postings: int = 100):
        self.max_postings = max_postings
        # (country, postal_code, name_prefix) -> list of target entity_ids
        self.postal_name_index = defaultdict(list)
        # (country, building_num, street_token) -> list of target entity_ids
        self.building_street_index = defaultdict(list)

    def index_target(
        self,
        entity_id: str,
        country: str,
        norm_name: NormalizedName,
        norm_addr: NormalizedAddress,
    ):
        """Index a single target record."""
        if norm_addr.is_empty:
            return

        cty = country.lower()
        # 1. Postal code + name prefix (first 2 chars)
        if norm_addr.postal_code and norm_name.core_name:
            prefix = norm_name.core_name[:2].lower()
            key = (cty, norm_addr.postal_code, prefix)
            plist = self.postal_name_index[key]
            if len(plist) < self.max_postings:
                plist.append(entity_id)

        # 2. Building number + street token
        if norm_addr.building_number and len(norm_addr.tokens) >= 2:
            st_tok = norm_addr.tokens[1] if norm_addr.tokens[0].isdigit() else norm_addr.tokens[0]
            if len(st_tok) >= 3:
                key2 = (cty, norm_addr.building_number, st_tok)
                plist2 = self.building_street_index[key2]
                if len(plist2) < self.max_postings:
                    plist2.append(entity_id)

    def block(
        self,
        s1_id: str,
        country: str,
        norm_name: NormalizedName,
        norm_addr: NormalizedAddress,
    ) -> Set[str]:
        """Retrieve candidate matches for an S1 entity."""
        if norm_addr.is_empty:
            return set()

        candidates = set()
        cty = country.lower()

        # 1. Postal code + name prefix
        if norm_addr.postal_code and norm_name.core_name:
            prefix = norm_name.core_name[:2].lower()
            key = (cty, norm_addr.postal_code, prefix)
            candidates.update(self.postal_name_index.get(key, []))

        # 2. Building number + street token
        if norm_addr.building_number and len(norm_addr.tokens) >= 2:
            st_tok = norm_addr.tokens[1] if norm_addr.tokens[0].isdigit() else norm_addr.tokens[0]
            if len(st_tok) >= 3:
                key2 = (cty, norm_addr.building_number, st_tok)
                candidates.update(self.building_street_index.get(key2, []))

        return candidates
