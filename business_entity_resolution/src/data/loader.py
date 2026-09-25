"""
DataLoader: Robust, memory-efficient loader for tab-separated entity datasets.
Handles chunked streaming, validation, and country partitioning without loading
entire massive tables into memory at once.
"""

import os
from typing import Generator, Dict, Any, List, Optional, Tuple, Iterable
import pandas as pd


class DataLoader:
    """Handles loading and streaming of entity resolution TSV data."""

    EXPECTED_SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
    EXPECTED_GT_COLS = ["source1_entity_id", "matched_entity_ids"]

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or ""

    def _resolve_path(self, path: str) -> str:
        if self.base_dir and not os.path.isabs(path):
            return os.path.join(self.base_dir, path)
        return path

    def iter_records(
        self,
        filepath: str,
        target_country: Optional[str] = None,
        max_records: Optional[int] = None,
    ) -> Generator[Dict[str, str], None, None]:
        """
        Stream records one by one with minimal memory footprint.
        Yields dict with keys: entity_id, business_name, business_address, country.
        """
        resolved_path = self._resolve_path(filepath)
        count = 0
        with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
            header_line = f.readline().rstrip("\r\n")
            headers = [h.strip() for h in header_line.split("\t")]
            
            # Map header indices
            id_idx = headers.index("entity_id") if "entity_id" in headers else 0
            name_idx = headers.index("business_name") if "business_name" in headers else 1
            addr_idx = headers.index("business_address") if "business_address" in headers else 2
            cty_idx = headers.index("country") if "country" in headers else 3

            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\r\n").split("\t")
                # Pad parts if trailing empty columns were stripped
                while len(parts) < 4:
                    parts.append("")
                
                cty = parts[cty_idx].strip()
                if target_country and cty.lower() != target_country.lower():
                    continue

                yield {
                    "entity_id": parts[id_idx].strip(),
                    "business_name": parts[name_idx].strip(),
                    "business_address": parts[addr_idx].strip(),
                    "country": cty,
                }
                count += 1
                if max_records and count >= max_records:
                    break

    def load_df(
        self,
        filepath: str,
        nrows: Optional[int] = None,
        columns: Optional[List[str]] = None,
        country: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load a TSV file into a pandas DataFrame with strict sep='\t' and optimized dtypes.
        """
        resolved_path = self._resolve_path(filepath)
        df = pd.read_csv(
            resolved_path,
            sep="\t",
            nrows=nrows,
            usecols=columns,
            dtype={"entity_id": "string", "business_name": "string", "business_address": "string", "country": "string"},
            keep_default_na=False,
            na_values=[],
            encoding="utf-8",
        )
        if country:
            df = df[df["country"].str.lower() == country.lower()].reset_index(drop=True)
        return df

    def iter_chunks(
        self,
        filepath: str,
        chunksize: int = 50000,
        columns: Optional[List[str]] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """Stream chunks of DataFrame for batch processing."""
        resolved_path = self._resolve_path(filepath)
        for chunk in pd.read_csv(
            resolved_path,
            sep="\t",
            chunksize=chunksize,
            usecols=columns,
            dtype={"entity_id": "string", "business_name": "string", "business_address": "string", "country": "string"},
            keep_default_na=False,
            na_values=[],
            encoding="utf-8",
        ):
            yield chunk
