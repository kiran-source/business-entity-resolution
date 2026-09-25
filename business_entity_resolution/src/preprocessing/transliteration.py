"""
IndicTransliterator: Offline transliterator for Devanagari and Gujarati scripts.
Converts Indic Unicode text to Romanized phonetic representations
strictly offline without external APIs or internet lookups.
"""

import re
from typing import Dict


class IndicTransliterator:
    """Offline phonetic mapping for Devanagari and Gujarati unicode characters."""

    # Consonants
    DEVANAGARI_MAP: Dict[str, str] = {
        "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
        "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
        "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
        "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
        "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
        "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
        "ष": "sh", "स": "s", "ह": "h", "क्ष": "ksh", "त्र": "tr", "ज्ञ": "gy",
        # Vowels
        "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
        "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
        # Matras (vowel signs)
        "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
        "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
        # Modifiers
        "ं": "n", "ँ": "n", "ः": "h", "्": "",
        # Common Hindi corporate terms in devanagari
        "प्राइवेट": "private", "लिमिटेड": "limited", "एलएलपी": "llp",
        "कंपनी": "company", "इंटरप्राइजेज": "enterprises", "ट्रेडर्स": "traders",
    }

    GUJARATI_MAP: Dict[str, str] = {
        "ક": "k", "ખ": "kh", "ગ": "g", "ઘ": "gh", "ઙ": "ng",
        "ચ": "ch", "છ": "chh", "જ": "j", "ઝ": "jh", "ઞ": "ny",
        "ટ": "t", "ઠ": "th", "ડ": "d", "ઢ": "dh", "ણ": "n",
        "ત": "t", "થ": "th", "દ": "d", "ધ": "dh", "ન": "n",
        "પ": "p", "ફ": "ph", "બ": "b", "ભ": "bh", "મ": "m",
        "ય": "y", "ર": "r", "લ": "l", "વ": "v", "શ": "sh",
        "ષ": "sh", "સ": "s", "હ": "h", "ળ": "l",
        # Vowels and Matras
        "અ": "a", "આ": "aa", "ઇ": "i", "ઈ": "ee", "ઉ": "u", "ઊ": "oo",
        "એ": "e", "ઐ": "ai", "ઓ": "o", "ઔ": "au",
        "ા": "a", "િ": "i", "ી": "i", "ુ": "u", "ૂ": "u",
        "ે": "e", "ૈ": "ai", "ો": "o", "ૌ": "au",
        "ં": "n", "્": "",
        # Common terms
        "કન્સ્ટ્રક્શન્સ": "constructions", "લિમિટેડ": "limited",
    }

    INDIC_CHAR_REGEX = re.compile(r"[\u0900-\u097F\u0A80-\u0AFF]")

    @classmethod
    def has_indic_script(cls, text: str) -> bool:
        """Check if string contains any Indic unicode characters."""
        return bool(cls.INDIC_CHAR_REGEX.search(text))

    @classmethod
    def transliterate(cls, text: str) -> str:
        """
        Transliterate Indic scripts to Romanized characters.
        If text contains no Indic characters, returns cleaned original.
        """
        if not text or not cls.has_indic_script(text):
            return text

        result = text
        # First replace whole multi-char corporate words
        for k, v in cls.DEVANAGARI_MAP.items():
            if len(k) > 1 and k in result:
                result = result.replace(k, f" {v} ")
        for k, v in cls.GUJARATI_MAP.items():
            if len(k) > 1 and k in result:
                result = result.replace(k, f" {v} ")

        # Then character by character replacement
        chars = []
        for ch in result:
            if ch in cls.DEVANAGARI_MAP:
                chars.append(cls.DEVANAGARI_MAP[ch])
            elif ch in cls.GUJARATI_MAP:
                chars.append(cls.GUJARATI_MAP[ch])
            else:
                chars.append(ch)

        res = "".join(chars)
        # Collapse multiple spaces
        return re.sub(r"\s+", " ", res).strip()
