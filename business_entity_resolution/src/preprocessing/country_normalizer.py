"""
CountryNormalizer: Standardizes country labels and supports open-set country representations.
Treats country as an open-set string label while standardizing case and common aliases.
"""

from typing import Optional


class CountryNormalizer:
    """Standardizes country labels across open sets."""

    COUNTRY_MAP = {
        "us": "US",
        "usa": "US",
        "united states": "US",
        "united states of america": "US",
        "india": "India",
        "ind": "India",
        "in": "India",
        "france": "France",
        "fr": "France",
    }

    @classmethod
    def normalize(cls, country: Optional[str]) -> str:
        """Normalize country name, falling back to title-cased string for unseen countries."""
        if not country:
            return "UNKNOWN"
        raw = country.strip()
        lowered = raw.lower()
        if lowered in cls.COUNTRY_MAP:
            return cls.COUNTRY_MAP[lowered]
        return raw.title()
