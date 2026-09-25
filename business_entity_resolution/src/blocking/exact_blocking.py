"""
ExactBlocker: Generates candidate matches using exact normalized name keys,
including core name, sorted tokens, and alphanumeric representations within country.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
from ..preprocessing.name_normalizer import NormalizedName


class ExactBlocker:
    """Indexes target entities by exact normalized name keys."""

    def __init__(self):
        # (country, key_str) -> list of target entity_ids
        self.core_name_index = defaultdict(list)
        self.sorted_tokens_index = defaultdict(list)
        self.alphanumeric_index = defaultdict(list)

    def index_target(self, entity_id: str, country: str, norm_name: NormalizedName):
        """Index a single target record (Source 2 or Source 3)."""
        cty = country.lower()
        if norm_name.core_name and len(norm_name.core_name) >= 3:
            self.core_name_index[(cty, norm_name.core_name)].append(entity_id)

        if norm_name.sorted_tokens and len(norm_name.tokens) >= 2:
            self.sorted_tokens_index[(cty, norm_name.sorted_tokens)].append(entity_id)

        if norm_name.alphanumeric and len(norm_name.alphanumeric) >= 4:
            self.alphanumeric_index[(cty, norm_name.alphanumeric)].append(entity_id)

    def block(self, s1_id: str, country: str, norm_name: NormalizedName) -> Set[str]:
        """Retrieve candidate entity IDs for a Source 1 entity."""
        candidates = set()
        cty = country.lower()

        if norm_name.core_name:
            candidates.update(self.core_name_index.get((cty, norm_name.core_name), []))

        if norm_name.sorted_tokens:
            candidates.update(self.sorted_tokens_index.get((cty, norm_name.sorted_tokens), []))

        if norm_name.alphanumeric:
            candidates.update(self.alphanumeric_index.get((cty, norm_name.alphanumeric), []))

        return candidates
