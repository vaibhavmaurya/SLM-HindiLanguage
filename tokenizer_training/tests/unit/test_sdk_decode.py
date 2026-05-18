"""Tests for SDK decode — Phase 8."""

from __future__ import annotations

import pytest

from hindi_tokenizer.sdk.decode import decode_ids
from hindi_tokenizer.sdk.encode import encode_text


@pytest.fixture()
def mock_tokenizer(mocker):
    tok = mocker.MagicMock()
    tok.decode.return_value = "भारत है।"
    tok.__call__ = mocker.Mock(return_value={"input_ids": [2, 42, 43, 3], "attention_mask": [1, 1, 1, 1]})
    return tok


def test_decode_returns_string(mock_tokenizer):
    result = decode_ids(mock_tokenizer, [2, 100, 200, 3])
    assert isinstance(result, str)


def test_decode_skips_special_tokens(mock_tokenizer):
    decode_ids(mock_tokenizer, [2, 100, 200, 3], skip_special_tokens=True)
    call_kwargs = mock_tokenizer.decode.call_args.kwargs
    assert call_kwargs.get("skip_special_tokens") is True


def test_decode_empty_ids(mock_tokenizer):
    mock_tokenizer.decode.return_value = ""
    result = decode_ids(mock_tokenizer, [])
    assert result == ""


def test_decode_roundtrip(mocker):
    tok = mocker.MagicMock()
    original = "भारत है।"
    tok.__call__ = mocker.Mock(return_value={"input_ids": [2, 42, 43, 3], "attention_mask": [1, 1, 1, 1]})
    tok.decode.return_value = original
    encoded = encode_text(tok, original, add_special_tokens=True)
    decoded = decode_ids(tok, encoded["input_ids"], skip_special_tokens=True)
    assert decoded == original
