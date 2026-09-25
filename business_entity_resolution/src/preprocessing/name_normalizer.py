"""
NameNormalizer: Multi-representation normalization engine for business names.
Handles legal suffix normalization, DBA/formerly clauses, domain-style names,
punctuation stripping, accent removal, token sorting, and core-name extraction.
"""

import re
from dataclasses import dataclass
from typing import List, Set, Optional
from .text_utils import TextUtils
from .transliteration import IndicTransliterator


@dataclass(slots=True)
class NormalizedName:
    original: str
    cleaned: str
    alphanumeric: str
    tokens: List[str]
    sorted_tokens: str
    deduped_tokens: str
    core_name: str
    legal_suffix: str
    char_ngrams: List[str]
    has_indic: bool


class NameNormalizer:
    """Produces multi-faceted normalized views of a business name."""

    # Legal suffixes across US, India, France
    # Ordered longest first to avoid partial matches
    LEGAL_SUFFIX_MAP = {
        # India
        "private limited": "pvt ltd",
        "pvt limited": "pvt ltd",
        "private ltd": "pvt ltd",
        "pvt ltd": "pvt ltd",
        "pvtltd": "pvt ltd",
        "private": "pvt",
        "pvt": "pvt",
        "limited liability partnership": "llp",
        "limited": "ltd",
        "ltd": "ltd",
        "llp": "llp",
        # US
        "incorporated": "inc",
        "corporation": "corp",
        "corp": "corp",
        "inc": "inc",
        "limited liability company": "llc",
        "llc": "llc",
        "pllc": "pllc",
        "company": "co",
        "co": "co",
        "l p": "lp",
        "lp": "lp",
        # France
        "societe par actions simplifiee unipersonnelle": "sasu",
        "societe par actions simplifiee": "sas",
        "societe a responsabilite limitee": "sarl",
        "societe civile immobiliere": "sci",
        "societe anonyme": "sa",
        "entreprise unipersonnelle a responsabilite limitee": "eurl",
        "sasu": "sasu",
        "sas": "sas",
        "sarl": "sarl",
        "sci": "sci",
        "eurl": "eurl",
        "s a s": "sas",
        "s a r l": "sarl",
        "s a": "sa",
        "sa": "sa",
        "et fils": "fils",
        "freres": "freres",
    }

    # Regex for stripping prefixes like DBA, formerly, aka, the, m/s
    DBA_REGEX = re.compile(
        r"^(.*?)\s+(?:dba|fka|formerly|aka|d/b/a|t/a|trading as)\s*[:\-]?\s*(.*)$",
        re.IGNORECASE,
    )
    DOMAIN_SUFFIX_REGEX = re.compile(r"\.(?:com|org|net|in|co\.in|fr|io|biz|info)$", re.IGNORECASE)

    # Precompiled regex for legal suffix removal at start or end
    LEGAL_PATTERN = re.compile(
        r"\b(?:pvt ltd|private limited|private ltd|pvt limited|incorporated|corporation|"
        r"limited liability company|llc|pllc|inc|corp|ltd|llp|sarl|sasu|sas|eurl|sci|sa)\b",
        re.IGNORECASE,
    )

    # Multi-word and acronym legal patterns ordered by specificity
    MULTI_WORD_PATTERNS = [
        # France
        (re.compile(r"\b(?:societe par actions simplifiee unipersonnelle|sasu)\b", re.I), "sasu"),
        (re.compile(r"\b(?:societe par actions simplifiee|s a s|sas)\b", re.I), "sas"),
        (re.compile(r"\b(?:societe a responsabilite limitee|s a r l|sarl)\b", re.I), "sarl"),
        (re.compile(r"\b(?:societe civile immobiliere|sci)\b", re.I), "sci"),
        (re.compile(r"\b(?:societe anonyme|s a|sa)\b", re.I), "sa"),
        (re.compile(r"\b(?:entreprise unipersonnelle a responsabilite limitee|eurl)\b", re.I), "eurl"),
        (re.compile(r"\b(?:groupement d interet economique|gie)\b", re.I), "gie"),
        (re.compile(r"\b(?:societe en nom collectif|snc)\b", re.I), "snc"),
        (re.compile(r"\b(?:entreprise individuelle|ei)\b", re.I), "ei"),
        # India
        (re.compile(r"\b(?:private limited|pvt limited|private ltd|pvt ltd|pvtltd)\b", re.I), "pvt ltd"),
        (re.compile(r"\b(?:limited liability partnership|llp)\b", re.I), "llp"),
        (re.compile(r"\b(?:public limited|public ltd)\b", re.I), "ltd"),
        # US
        (re.compile(r"\b(?:limited liability company|llc|pllc)\b", re.I), "llc"),
        (re.compile(r"\b(?:incorporated|inc)\b", re.I), "inc"),
        (re.compile(r"\b(?:corporation|corp)\b", re.I), "corp"),
        (re.compile(r"\b(?:limited partnership|lp|l p)\b", re.I), "lp"),
        (re.compile(r"\b(?:company|co)\b", re.I), "co"),
    ]

    def __init__(self):
        pass

    def normalize(self, name: str) -> NormalizedName:
        """Construct multi-faceted normalized views of a business name."""
        orig = name or ""
        has_indic = IndicTransliterator.has_indic_script(orig)
        
        # 1. Transliterate if Indic script is present
        translit = IndicTransliterator.transliterate(orig) if has_indic else orig

        # 2. Check DBA / formerly patterns (e.g. "Zetaflux DBA: Fowler Peak Optics")
        dba_match = self.DBA_REGEX.match(translit)
        if dba_match:
            # Use the part after DBA if present, or both
            p1, p2 = dba_match.group(1).strip(), dba_match.group(2).strip()
            translit = p2 if p2 else p1

        # 3. Handle domain names (e.g. "siiainvestments.com" -> "siia investments")
        clean_name = self.DOMAIN_SUFFIX_REGEX.sub("", translit)
        if clean_name.startswith("www."):
            clean_name = clean_name[4:]

        # 4. Standard clean
        cleaned = TextUtils.clean_text(clean_name)

        # 5. Multi-phrase and token-level legal suffix extraction
        found_suffix = []
        core_str = cleaned
        for pat, canonical_suffix in self.MULTI_WORD_PATTERNS:
            if pat.search(core_str):
                found_suffix.append(canonical_suffix)
                core_str = pat.sub(" ", core_str)

        tokens = core_str.split()
        core_tokens = []
        for tok in tokens:
            if tok in self.LEGAL_SUFFIX_MAP:
                sfx = self.LEGAL_SUFFIX_MAP[tok]
                if sfx not in found_suffix:
                    found_suffix.append(sfx)
            else:
                core_tokens.append(tok)

        core_name = " ".join(core_tokens).strip()
        # Fallback if core name was entirely stripped
        if not core_name:
            core_name = cleaned

        legal_suffix = " ".join(found_suffix)

        # 6. Alphanumeric
        alphanumeric = TextUtils.alphanumeric_only(cleaned)

        # 7. Token views
        all_tokens = tokens
        sorted_tokens = " ".join(sorted(all_tokens))
        
        # Deduplicated tokens preserving first appearance
        seen = set()
        deduped = []
        for t in all_tokens:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        deduped_tokens = " ".join(deduped)

        # 8. Character n-grams
        char_ngrams = TextUtils.char_ngrams(core_name, n=3)

        return NormalizedName(
            original=orig,
            cleaned=cleaned,
            alphanumeric=alphanumeric,
            tokens=all_tokens,
            sorted_tokens=sorted_tokens,
            deduped_tokens=deduped_tokens,
            core_name=core_name,
            legal_suffix=legal_suffix,
            char_ngrams=char_ngrams,
            has_indic=has_indic,
        )
