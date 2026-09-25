"""
TextUtils: Core low-level text normalization and tokenization utilities.
Provides unicode normalization, accent stripping, n-gram generation,
and safe alphanumeric extraction.
"""

import re
import unicodedata
from typing import List, Set, Tuple


class TextUtils:
    """Core string normalization primitives."""

    PUNCT_REGEX = re.compile(r"[^\w\s]")
    WHITESPACE_REGEX = re.compile(r"\s+")
    DIGITS_REGEX = re.compile(r"\b\d+\b")

    @classmethod
    def strip_accents(cls, text: str) -> str:
        """Strip accents and diacritics via Unicode NFKD decomposition."""
        if not text:
            return ""
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(c for c in decomposed if not unicodedata.combining(c))

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Standard cleaning:
        1. Strip accents
        2. Lowercase
        3. Replace punctuation with space
        4. Collapse multiple spaces
        """
        if not text:
            return ""
        s = cls.strip_accents(text).lower()
        s = cls.PUNCT_REGEX.sub(" ", s)
        return cls.WHITESPACE_REGEX.sub(" ", s).strip()

    @classmethod
    def alphanumeric_only(cls, text: str) -> str:
        """Extract only lowercase alphanumeric characters."""
        if not text:
            return ""
        cleaned = cls.strip_accents(text).lower()
        return re.sub(r"[^a-z0-9]", "", cleaned)

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenize cleaned text into distinct non-empty words."""
        cleaned = cls.clean_text(text)
        return [t for t in cleaned.split() if t]

    @classmethod
    def sorted_tokens(cls, text: str) -> str:
        """Return alphabetically sorted unique tokens joined by space."""
        tokens = sorted(set(cls.tokenize(text)))
        return " ".join(tokens)

    @classmethod
    def char_ngrams(cls, text: str, n: int = 3) -> List[str]:
        """Generate character n-grams from alphanumeric text with padding."""
        s = f" {cls.clean_text(text)} "
        if len(s) < n:
            return [s]
        return [s[i : i + n] for i in range(len(s) - n + 1)]

    @classmethod
    def extract_numbers(cls, text: str) -> List[str]:
        """Extract all numeric sequences from text."""
        return cls.DIGITS_REGEX.findall(text)
