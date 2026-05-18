"""Tests for SDK encode — Phase 8."""

from __future__ import annotations

import pytest

from hindi_tokenizer.sdk.encode import encode_text


@pytest.fixture()
def mock_tokenizer(mocker):
    tok = mocker.MagicMock()
    tok.bos_token_id = 2
    tok.eos_token_id = 3
    tok.return_value = {"input_ids": [2, 42, 43, 3], "attention_mask": [1, 1, 1, 1]}
    return tok


def test_encode_returns_input_ids(mock_tokenizer):
    result = encode_text(mock_tokenizer, "भारत है।")
    assert "input_ids" in result


def test_encode_returns_attention_mask(mock_tokenizer):
    result = encode_text(mock_tokenizer, "भारत है।")
    assert "attention_mask" in result


def test_encode_with_special_tokens_true(mock_tokenizer):
    encode_text(mock_tokenizer, "भारत है।", add_special_tokens=True)
    call_kwargs = mock_tokenizer.call_args.kwargs
    assert call_kwargs.get("add_special_tokens") is True


def test_encode_with_special_tokens_false(mock_tokenizer):
    encode_text(mock_tokenizer, "भारत है।", add_special_tokens=False)
    call_kwargs = mock_tokenizer.call_args.kwargs
    assert call_kwargs.get("add_special_tokens") is False


def test_encode_empty_string(mocker):
    tok = mocker.MagicMock()
    tok.return_value = {"input_ids": [], "attention_mask": []}
    result = encode_text(tok, "", add_special_tokens=False)
    assert result["input_ids"] == []
