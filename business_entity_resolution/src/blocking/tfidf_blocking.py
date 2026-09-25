"""
TfidfBlocker: Sparse matrix TF-IDF vector retrieval for sub-linear nearest-neighbor blocking.
Supports character n-grams (3-4 char_wb) and word n-grams with top-K candidate extraction.
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfBlocker:
    """Computes sparse cosine similarity over character or word n-grams."""

    def __init__(
        self,
        analyzer: str = "char_wb",
        ngram_range: Tuple[int, int] = (3, 4),
        min_df: int = 1,
        top_k: int = 15,
        min_similarity: float = 0.30,
    ):
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            dtype=np.float32,
            sublinear_tf=True,
        )
        self.target_ids: List[str] = []
        self.target_matrix: Optional[sparse.csr_matrix] = None
        self.is_fitted = False

    def fit_and_index_targets(self, target_ids: List[str], target_texts: List[str]):
        """Fit vectorizer on targets and create sparse target matrix."""
        if not target_texts:
            return
        self.target_ids = target_ids
        self.target_matrix = self.vectorizer.fit_transform(target_texts)
        self.is_fitted = True

    def block_batch(
        self,
        s1_ids: List[str],
        s1_texts: List[str],
    ) -> Dict[str, Set[str]]:
        """
        Batch query against target matrix using sparse matrix dot product.
        Returns dict: s1_id -> set of candidate target_ids.
        """
        if not self.is_fitted or self.target_matrix is None or not s1_texts:
            return {sid: set() for sid in s1_ids}

        query_matrix = self.vectorizer.transform(s1_texts)
        # Sparse matrix multiplication: shape (N_queries, N_targets)
        sim_matrix = query_matrix.dot(self.target_matrix.T)

        results: Dict[str, Set[str]] = {}
        for row_idx, s1_id in enumerate(s1_ids):
            row = sim_matrix.getrow(row_idx)
            # Find non-zero indices and scores
            cols = row.indices
            data = row.data
            if len(data) == 0:
                results[s1_id] = set()
                continue

            # Filter by min_similarity
            mask = data >= self.min_similarity
            valid_cols = cols[mask]
            valid_data = data[mask]

            if len(valid_data) == 0:
                results[s1_id] = set()
                continue

            # Top-K
            if len(valid_data) > self.top_k:
                top_idx = np.argpartition(valid_data, -self.top_k)[-self.top_k :]
                selected_cols = valid_cols[top_idx]
            else:
                selected_cols = valid_cols

            candidates = {self.target_ids[c] for c in selected_cols}
            results[s1_id] = candidates

        return results
