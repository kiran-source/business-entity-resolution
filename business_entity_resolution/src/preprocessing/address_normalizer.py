"""
AddressNormalizer: Multi-representation normalization engine for business addresses.
Extracts street components, building numbers, postal codes, state abbreviations,
standardizes street suffix abbreviations, strips noise landmarks, and handles missing fields.
"""

import re
from dataclasses import dataclass
from typing import List, Set, Optional, Tuple
from .text_utils import TextUtils
from .transliteration import IndicTransliterator


@dataclass(slots=True)
class NormalizedAddress:
    original: str
    cleaned: str
    alphanumeric: str
    tokens: List[str]
    sorted_tokens: str
    numeric_tokens: List[str]
    postal_code: str
    building_number: str
    is_empty: bool


class AddressNormalizer:
    """Produces multi-faceted normalized views of a business address."""

    # Standard street suffix abbreviations (multilingual: US, India, France)
    STREET_SUFFIX_MAP = {
        # US & General
        "st": "street", "saint": "street", "str": "street",
        "rd": "road",
        "ave": "avenue", "av": "avenue",
        "blvd": "boulevard", "bvd": "boulevard", "bd": "boulevard",
        "dr": "drive",
        "ln": "lane",
        "ct": "court",
        "pl": "place",
        "cir": "circle",
        "pkwy": "parkway", "pky": "parkway",
        "hwy": "highway",
        "expy": "expressway",
        "ste": "suite",
        "apt": "apartment",
        "bldg": "building",
        "fl": "floor", "flr": "floor",
        "unit": "unit",
        "dept": "department",
        # Indian
        "ngr": "nagar",
        "mrg": "marg",
        "sec": "sector",
        "ph": "phase",
        "nr": "near",
        "opp": "opposite",
        "dist": "district",
        "tq": "taluk",
        # French
        "r": "rue",
        "all": "allee",
        "imp": "impasse",
    }

    # US State names to 2-letter codes mapping
    US_STATE_MAP = {
        "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
        "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
        "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
        "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
        "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
        "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
        "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
        "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
        "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
        "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
        "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    }

    POSTAL_CODE_REGEX = re.compile(r"\b(?:\d{5}(?:-\d{4})?|\d{6})\b")
    BUILDING_NUM_REGEX = re.compile(r"^\s*#?\s*(\d+[a-z]?|\d+-\w+)\b", re.IGNORECASE)

    def __init__(self):
        pass

    def normalize(self, address: str, country: Optional[str] = None) -> NormalizedAddress:
        """Construct multi-faceted normalized views of a business address."""
        orig = address or ""
        if not orig.strip():
            return NormalizedAddress(
                original="",
                cleaned="",
                alphanumeric="",
                tokens=[],
                sorted_tokens="",
                numeric_tokens=[],
                postal_code="",
                building_number="",
                is_empty=True,
            )

        # Transliterate if Indic characters present
        text = IndicTransliterator.transliterate(orig) if IndicTransliterator.has_indic_script(orig) else orig

        # Postal code candidate extraction
        postal_codes = self.POSTAL_CODE_REGEX.findall(text)
        postal_code = postal_codes[0] if postal_codes else ""

        # Building number extraction
        building_match = self.BUILDING_NUM_REGEX.search(text)
        building_num = building_match.group(1).lower() if building_match else ""

        # Standard cleaning
        clean_text = TextUtils.clean_text(text)

        # Standardize tokens
        raw_tokens = clean_text.split()
        norm_tokens = []
        numeric_tokens = []

        for tok in raw_tokens:
            # Map street abbreviations
            tok_norm = self.STREET_SUFFIX_MAP.get(tok, tok)
            # Map US state full names to codes if country is US or unspecified
            if tok_norm in self.US_STATE_MAP:
                tok_norm = self.US_STATE_MAP[tok_norm]

            norm_tokens.append(tok_norm)
            if tok_norm.isdigit():
                # Strip leading zeros e.g. 00303 -> 303
                numeric_tokens.append(str(int(tok_norm)))

        cleaned = " ".join(norm_tokens)
        alphanumeric = TextUtils.alphanumeric_only(cleaned)
        sorted_tokens = " ".join(sorted(set(norm_tokens)))

        return NormalizedAddress(
            original=orig,
            cleaned=cleaned,
            alphanumeric=alphanumeric,
            tokens=norm_tokens,
            sorted_tokens=sorted_tokens,
            numeric_tokens=numeric_tokens,
            postal_code=postal_code,
            building_number=building_num,
            is_empty=False,
        )
