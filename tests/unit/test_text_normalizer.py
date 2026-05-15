"""Tests for ingestion/text_normalizer.py — TextNormalizer."""

import unicodedata

from slm_hindi.config.settings import QualityFilterConfig
from slm_hindi.ingestion.text_normalizer import TextNormalizer


def _make_config(**kwargs) -> QualityFilterConfig:
    return QualityFilterConfig(**kwargs)


def _normalizer(**kwargs) -> TextNormalizer:
    return TextNormalizer(_make_config(**kwargs))


def test_unicode_nfc_applied():
    norm = _normalizer()
    # NFD: 'अ' decomposed
    nfd_char = unicodedata.normalize("NFD", "भारत")
    result = norm.normalize(nfd_char)
    assert unicodedata.is_normalized("NFC", result)


def test_multiple_spaces_collapsed():
    norm = _normalizer()
    result = norm.normalize("हिंदी   पाठ   है।")
    assert "  " not in result


def test_multiple_newlines_collapsed_to_two():
    norm = _normalizer()
    result = norm.normalize("पंक्ति एक।\n\n\n\n\nपंक्ति दो।")
    assert "\n\n\n" not in result


def test_danda_preserved():
    norm = _normalizer()
    text = "यह वाक्य है। यह दूसरा वाक्य है।"
    result = norm.normalize(text)
    assert "।" in result


def test_url_removed_when_enabled():
    norm = _normalizer(remove_urls=True)
    result = norm.normalize("देखें https://example.com पर।")
    assert "https://" not in result


def test_url_preserved_when_disabled():
    norm = _normalizer(remove_urls=False)
    result = norm.normalize("देखें https://example.com पर।")
    assert "https://example.com" in result


def test_repeated_lines_deduplicated():
    norm = _normalizer()
    # 4 identical lines should be reduced to 1
    text = "यह शीर्षक है।\n" * 4 + "असली सामग्री।"
    result = norm.normalize(text)
    lines = [ln for ln in result.splitlines() if "यह शीर्षक है।" in ln]
    assert len(lines) == 1


def test_decorative_symbols_removed():
    norm = _normalizer()
    result = norm.normalize("★ यह पाठ है ☆ और यहाँ ✓ समाप्त होता है।")
    assert "★" not in result
    assert "☆" not in result


def test_leading_trailing_whitespace_stripped():
    norm = _normalizer()
    result = norm.normalize("  \n  हिंदी पाठ।  \n  ")
    assert result == result.strip()


def test_normalize_records_updates_final_text():
    from slm_hindi.schema.corpus_record import CorpusRecord
    norm = _normalizer()
    record = CorpusRecord(
        record_id="r1", document_id="d1",
        source_type="huggingface_dataset", source_name="test",
        raw_text="  हिंदी   पाठ।  ",
        cleaned_text="  हिंदी   पाठ।  ",
    )
    [updated] = norm.normalize_records([record])
    assert updated.final_text == "हिंदी पाठ।"
    assert updated.char_count == len("हिंदी पाठ।")
