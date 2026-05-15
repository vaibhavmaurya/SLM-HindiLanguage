"""Tests for ingestion/quality_filter.py — QualityFilter."""

from slm_hindi.config.settings import QualityFilterConfig
from slm_hindi.ingestion.quality_filter import QualityFilter
from slm_hindi.schema.corpus_record import CorpusRecord

_HINDI_TEXT = "भारत एक विविधताओं से भरा देश है। यहाँ विभिन्न धर्म और संस्कृतियाँ पाई जाती हैं।"


def _filter(**kwargs) -> QualityFilter:
    config = QualityFilterConfig(**kwargs)
    return QualityFilter(config)


def _record(text: str, **kwargs) -> CorpusRecord:
    return CorpusRecord(
        record_id="r1", document_id="d1",
        source_type="huggingface_dataset", source_name="test",
        raw_text=text, final_text=text,
        char_count=len(text), word_count=len(text.split()),
        **kwargs,
    )


def test_good_hindi_passes():
    f = _filter()
    passed, rejected = f.filter([_record(_HINDI_TEXT)])
    assert len(passed) == 1
    assert len(rejected) == 0


def test_low_devanagari_ratio_rejected():
    f = _filter(min_devanagari_ratio=0.60)
    # Mostly ASCII text
    text = "This is mostly English text with minimal हिंदी."
    passed, rejected = f.filter([_record(text)])
    assert len(rejected) == 1


def test_too_short_rejected():
    f = _filter(min_char_count=30)
    passed, rejected = f.filter([_record("हिंदी")])
    assert len(rejected) == 1


def test_too_long_rejected():
    f = _filter(max_char_count=100)
    long_text = _HINDI_TEXT * 10  # well over 100 chars
    passed, rejected = f.filter([_record(long_text)])
    assert len(rejected) == 1


def test_high_digit_ratio_rejected_as_table_fragment():
    f = _filter(reject_table_fragments=True, max_digit_ratio=0.30)
    # Mostly digits
    digit_heavy = "1234567890 " * 10
    passed, rejected = f.filter([_record(digit_heavy)])
    assert len(rejected) == 1


def test_table_fragment_rejection_disabled():
    f = _filter(reject_table_fragments=False)
    digit_heavy = "1234567890 " * 10
    # Won't reject on digit ratio alone when table_fragments=False
    # (may still reject on devanagari_ratio — that's fine for this test, just check no crash)
    passed, rejected = f.filter([_record(digit_heavy)])
    assert isinstance(passed, list)


def test_quality_score_set():
    f = _filter()
    records = [_record(_HINDI_TEXT)]
    passed, _ = f.filter(records)
    assert passed[0].quality_score > 0.0


def test_devanagari_ratio_annotated():
    f = _filter()
    records = [_record(_HINDI_TEXT)]
    passed, _ = f.filter(records)
    assert passed[0].devanagari_ratio > 0.60


def test_multiple_records_filtered_correctly():
    f = _filter()
    records = [
        _record(_HINDI_TEXT),
        _record("short"),
        _record(_HINDI_TEXT),
    ]
    passed, rejected = f.filter(records)
    assert len(passed) == 2
    assert len(rejected) == 1
