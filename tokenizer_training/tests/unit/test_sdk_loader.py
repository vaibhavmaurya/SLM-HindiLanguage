"""Tests for SDK loader — Phase 8."""

from __future__ import annotations

import pytest

from hindi_tokenizer.sdk.loader import load_hindi_slm_tokenizer


def test_load_from_local_path(tmp_path, mocker):
    mocker.patch(
        "hindi_tokenizer.sdk.loader.AutoTokenizer.from_pretrained",
        return_value=mocker.MagicMock(),
    )
    result = load_hindi_slm_tokenizer(str(tmp_path))
    assert result is not None


def test_load_calls_auto_tokenizer_from_pretrained(tmp_path, mocker):
    mock_from_pretrained = mocker.patch(
        "hindi_tokenizer.sdk.loader.AutoTokenizer.from_pretrained",
        return_value=mocker.MagicMock(),
    )
    load_hindi_slm_tokenizer(str(tmp_path))
    mock_from_pretrained.assert_called_once_with(str(tmp_path))


def test_load_raises_on_missing_path(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(OSError):
        load_hindi_slm_tokenizer(str(missing))
