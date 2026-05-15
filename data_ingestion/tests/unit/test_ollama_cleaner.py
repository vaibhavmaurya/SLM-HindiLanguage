"""Tests for ingestion/ollama_cleaner.py — OllamaCleaner."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from slm_hindi.config.settings import ModelCleaningConfig
from slm_hindi.ingestion.ollama_cleaner import OllamaCleaner, OllamaError
from slm_hindi.schema.corpus_record import CorpusRecord


def _make_config(**kwargs) -> ModelCleaningConfig:
    return ModelCleaningConfig(**kwargs)


def _make_record(text: str = "यह हिंदी पाठ है।") -> CorpusRecord:
    return CorpusRecord(
        record_id="rec_001", document_id="doc_001",
        source_type="pdf", source_name="test",
        raw_text=text,
    )


def _mock_response(text: str = "साफ हिंदी पाठ।") -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"response": text}
    resp.raise_for_status.return_value = None
    return resp


def test_clean_sets_cleaned_text():
    config = _make_config()
    cleaner = OllamaCleaner(config)
    record = _make_record()
    with patch("requests.post", return_value=_mock_response("साफ पाठ।")):
        results = cleaner.clean([record])
    assert results[0].cleaned_text == "साफ पाठ।"


def test_clean_calls_api_once_for_short_text():
    config = _make_config()
    cleaner = OllamaCleaner(config)
    record = _make_record("छोटा पाठ।")
    with patch("requests.post", return_value=_mock_response()) as mock_post:
        cleaner.clean([record])
    assert mock_post.call_count == 1


def test_long_text_produces_multiple_chunks():
    config = _make_config()
    config.chunking.max_input_chars = 100
    config.chunking.overlap_chars = 10
    cleaner = OllamaCleaner(config)
    long_text = "हिंदी पाठ। " * 50  # ~550 chars
    record = _make_record(long_text)
    with patch("requests.post", return_value=_mock_response()) as mock_post:
        cleaner.clean([record])
    assert mock_post.call_count > 1


def test_timeout_triggers_retry():
    config = _make_config(max_retries=3, retry_backoff_base_seconds=0)
    cleaner = OllamaCleaner(config)
    record = _make_record()
    with patch("requests.post", side_effect=requests.Timeout("timeout")):
        with pytest.raises(OllamaError, match="timed out"):
            cleaner.clean([record])


def test_http_error_raises_ollama_error():
    config = _make_config(max_retries=1, retry_backoff_base_seconds=0)
    cleaner = OllamaCleaner(config)
    record = _make_record()
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(OllamaError, match="HTTP error"):
            cleaner.clean([record])


def test_disabled_cleaning_returns_raw_text():
    config = _make_config(enabled=False)
    cleaner = OllamaCleaner(config)
    record = _make_record("मूल पाठ।")
    results = cleaner.clean([record])
    assert results[0].cleaned_text == "मूल पाठ।"


def test_chunk_overlap_included():
    config = _make_config()
    config.chunking.max_input_chars = 50
    config.chunking.overlap_chars = 10
    config.chunking.split_on_paragraph = False
    cleaner = OllamaCleaner(config)
    text = "A" * 100
    chunks = cleaner._chunk_text(text)
    assert len(chunks) > 1
    # Overlap: last 10 chars of chunk 0 should appear at start of chunk 1
    assert chunks[0][-10:] == chunks[1][:10]
