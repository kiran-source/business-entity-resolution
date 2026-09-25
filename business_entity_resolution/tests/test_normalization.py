"""
Unit tests for text utilities, transliteration, and normalizers.
"""

import pytest
from business_entity_resolution.src.preprocessing.text_utils import TextUtils
from business_entity_resolution.src.preprocessing.transliteration import IndicTransliterator
from business_entity_resolution.src.preprocessing.name_normalizer import NameNormalizer
from business_entity_resolution.src.preprocessing.address_normalizer import AddressNormalizer
from business_entity_resolution.src.preprocessing.country_normalizer import CountryNormalizer


def test_text_utils():
    # Accent stripping
    assert TextUtils.strip_accents("Léarning Center Àmicale") == "Learning Center Amicale"
    # Clean text
    assert TextUtils.clean_text("<< Team   Ecole! >>") == "team ecole"
    # Alphanumeric
    assert TextUtils.alphanumeric_only("Roberts Titan, Inc. #123") == "robertstitaninc123"
    # Tokenize
    tokens = TextUtils.tokenize("Prime Money LLC")
    assert tokens == ["prime", "money", "llc"]
    # Sorted tokens
    assert TextUtils.sorted_tokens("Zeta Alpha Beta") == "alpha beta zeta"
    # Char ngrams
    ngrams = TextUtils.char_ngrams("Acme", n=3)
    assert len(ngrams) > 0


def test_transliteration():
    # Devanagari
    hindi_text = "राम मार्केटिंग प्राइवेट लिमिटेड"
    translit = IndicTransliterator.transliterate(hindi_text)
    assert "private" in translit
    assert "limited" in translit
    assert "ram" in translit.lower()

    # Gujarati
    gujarati_text = "વિજય કન્સ્ટ્રક્શન્સ"
    translit_guj = IndicTransliterator.transliterate(gujarati_text)
    assert "constructions" in translit_guj


def test_name_normalizer():
    normalizer = NameNormalizer()

    # Legal suffix removal and core name extraction
    res1 = normalizer.normalize("Acme Technologies Pvt Ltd")
    assert res1.core_name == "acme technologies"
    assert "pvt ltd" in res1.legal_suffix

    # US legal suffix at start
    res2 = normalizer.normalize("LLC Moncada Learning Center")
    assert "moncada learning center" in res2.core_name

    # DBA pattern
    res3 = normalizer.normalize("Zetaflux DBA: Fowler Peak Optics")
    assert "fowler peak optics" in res3.core_name

    # Domain name format
    res4 = normalizer.normalize("siiainvestments.com")
    assert res4.core_name == "siiainvestments"

    # French entity
    res5 = normalizer.normalize("Marina Ecole France SARL")
    assert res5.core_name == "marina ecole france"
    assert "sarl" in res5.legal_suffix


def test_address_normalizer():
    normalizer = AddressNormalizer()

    # Street suffix standardization
    res1 = normalizer.normalize("123 Main St, Houston, TX")
    assert "street" in res1.tokens
    assert "tx" in res1.tokens
    assert "123" in res1.numeric_tokens

    # State full name to code
    res2 = normalizer.normalize("5780 Fawn Ct, Fort Worth, Texas")
    assert "tx" in res2.tokens
    assert "court" in res2.tokens

    # Postal code extraction
    res3 = normalizer.normalize("KH NO. -570/13, New Delhi, DL 110001")
    assert res3.postal_code == "110001"

    # Empty address handling
    res_empty = normalizer.normalize("")
    assert res_empty.is_empty is True
    assert res_empty.cleaned == ""


def test_country_normalizer():
    assert CountryNormalizer.normalize("US") == "US"
    assert CountryNormalizer.normalize("usa") == "US"
    assert CountryNormalizer.normalize("India") == "India"
    assert CountryNormalizer.normalize("france") == "France"
    # Unseen country (open-set)
    assert CountryNormalizer.normalize("germany") == "Germany"
