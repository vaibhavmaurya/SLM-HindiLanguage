"""Tests for TokenizerValidator — Phase 5."""

from __future__ import annotations

import json

import pytest
from transformers import AutoTokenizer

from hindi_tokenizer.training.tokenizer_trainer import TokenizerTrainer
from hindi_tokenizer.validation.tokenizer_validator import TokenizerValidator

_SMALL_VOCAB = 500
_SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
_SPECIAL_TOKEN_IDS = {
    "<pad>": 0,
    "<unk>": 1,
    "<s>": 2,
    "</s>": 3,
    "<|system|>": 4,
    "<|user|>": 5,
    "<|assistant|>": 6,
    "<|end|>": 7,
}


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory, small_corpus_path):
    out = tmp_path_factory.mktemp("validator_trained")
    TokenizerTrainer(vocab_size=_SMALL_VOCAB).train(corpus_file=small_corpus_path, output_dir=out)
    return out


@pytest.fixture(scope="module")
def validation_sentences(validation_sentences_path):
    return [line for line in validation_sentences_path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def validated_report_path(tmp_path_factory, trained_dir, validation_sentences):
    """Run validate() once; share result across module tests."""
    out = tmp_path_factory.mktemp("validator_reports")
    report_path = out / "report.json"
    TokenizerValidator(
        artifact_dir=trained_dir,
        validation_sentences=validation_sentences,
        variant_name="test_v500",
        tokenizer_version="v001",
    ).validate(report_path=report_path)
    return report_path


@pytest.fixture(scope="module")
def report_dict(validated_report_path):
    return json.loads(validated_report_path.read_text(encoding="utf-8"))


def _make_mock_tokenizer(mocker, *, encode_side_effect=None, decode_return="भारत है।", vocab_size=500):
    mock_tok = mocker.MagicMock()
    mock_tok.unk_token_id = 1
    mock_tok.pad_token_id = 0
    mock_tok.bos_token_id = 2
    mock_tok.eos_token_id = 3
    mock_tok.__len__ = mocker.Mock(return_value=vocab_size)
    if encode_side_effect is not None:
        mock_tok.encode.side_effect = encode_side_effect
    mock_tok.decode.return_value = decode_return
    return mock_tok


def _patch_auto_tokenizer(mocker, mock_tok):
    return mocker.patch(
        "hindi_tokenizer.validation.tokenizer_validator.AutoTokenizer.from_pretrained",
        return_value=mock_tok,
    )


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------


def test_validator_loads_tokenizer_via_auto_tokenizer(trained_dir, validation_sentences, tmp_path, mocker):
    spy = mocker.patch(
        "hindi_tokenizer.validation.tokenizer_validator.AutoTokenizer.from_pretrained",
        wraps=AutoTokenizer.from_pretrained,
    )
    TokenizerValidator(artifact_dir=trained_dir, validation_sentences=validation_sentences).validate(
        tmp_path / "report.json"
    )
    spy.assert_called_once_with(str(trained_dir))


# ---------------------------------------------------------------------------
# UNK rate
# ---------------------------------------------------------------------------


def test_unk_rate_computed_correctly(trained_dir, tmp_path, mocker):
    sentence = "भारत है।"
    # 1 UNK + 999 regular tokens → unk_rate = 0.001
    token_ids = [1] + [42] * 999

    def encode_side_effect(text, add_special_tokens=False):
        if text in _SPECIAL_TOKEN_IDS:
            return [_SPECIAL_TOKEN_IDS[text]]
        return token_ids

    mock_tok = _make_mock_tokenizer(mocker, encode_side_effect=encode_side_effect, decode_return=sentence)
    _patch_auto_tokenizer(mocker, mock_tok)

    report = TokenizerValidator(artifact_dir=trained_dir, validation_sentences=[sentence]).validate(
        tmp_path / "report.json"
    )
    assert report.unk_rate == pytest.approx(0.001)


def test_unk_rate_low_for_validation_sentences(trained_dir, validation_sentences, tmp_path):
    report = TokenizerValidator(
        artifact_dir=trained_dir, validation_sentences=validation_sentences
    ).validate(tmp_path / "report.json")
    assert report.unk_rate < 0.05


# ---------------------------------------------------------------------------
# Other scalar metrics
# ---------------------------------------------------------------------------


def test_chars_per_token_is_positive_float(trained_dir, validation_sentences, tmp_path):
    report = TokenizerValidator(
        artifact_dir=trained_dir, validation_sentences=validation_sentences
    ).validate(tmp_path / "report.json")
    assert report.chars_per_token > 0.0


def test_tokens_per_word_is_positive_float(trained_dir, validation_sentences, tmp_path):
    report = TokenizerValidator(
        artifact_dir=trained_dir, validation_sentences=validation_sentences
    ).validate(tmp_path / "report.json")
    assert report.tokens_per_word > 0.0


def test_roundtrip_success_rate_above_threshold(trained_dir, validation_sentences, tmp_path):
    report = TokenizerValidator(
        artifact_dir=trained_dir, validation_sentences=validation_sentences
    ).validate(tmp_path / "report.json")
    assert report.roundtrip_success_rate >= 0.50


def test_roundtrip_compares_stripped_text(trained_dir, tmp_path, mocker):
    sentence = "भारत है।"

    def encode_side_effect(text, add_special_tokens=False):
        if text in _SPECIAL_TOKEN_IDS:
            return [_SPECIAL_TOKEN_IDS[text]]
        return [42, 43]

    # Decoder adds surrounding whitespace — should still count as a match after strip
    mock_tok = _make_mock_tokenizer(
        mocker, encode_side_effect=encode_side_effect, decode_return="  " + sentence + "  "
    )
    _patch_auto_tokenizer(mocker, mock_tok)

    report = TokenizerValidator(artifact_dir=trained_dir, validation_sentences=[sentence]).validate(
        tmp_path / "report.json"
    )
    assert report.roundtrip_success_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Special token integrity
# ---------------------------------------------------------------------------


def test_special_token_integrity_no_failures(trained_dir, validation_sentences, tmp_path):
    report = TokenizerValidator(
        artifact_dir=trained_dir, validation_sentences=validation_sentences
    ).validate(tmp_path / "report.json")
    assert report.special_token_split_failures == 0


def test_special_token_integrity_detects_split(trained_dir, tmp_path, mocker):
    sentence = "भारत है।"

    def encode_side_effect(text, add_special_tokens=False):
        if text == "<pad>":
            return [100, 101, 102]  # split into 3 pieces — failure
        if text in _SPECIAL_TOKEN_IDS:
            return [_SPECIAL_TOKEN_IDS[text]]
        return [42, 43]

    mock_tok = _make_mock_tokenizer(
        mocker, encode_side_effect=encode_side_effect, decode_return=sentence
    )
    _patch_auto_tokenizer(mocker, mock_tok)

    report = TokenizerValidator(artifact_dir=trained_dir, validation_sentences=[sentence]).validate(
        tmp_path / "report.json"
    )
    assert report.special_token_split_failures >= 1


# ---------------------------------------------------------------------------
# JSON report structure
# ---------------------------------------------------------------------------


_REQUIRED_KEYS = {
    "unk_rate",
    "chars_per_token",
    "tokens_per_word",
    "roundtrip_success_rate",
    "special_token_failures",
    "vocab_size",
    "pad_token_id",
    "unk_token_id",
    "bos_token_id",
    "eos_token_id",
    "passes_thresholds",
}


def test_validation_report_has_all_required_keys(report_dict):
    assert _REQUIRED_KEYS <= set(report_dict.keys())


def test_validation_report_saved_to_json_file(validated_report_path):
    assert validated_report_path.exists()


def test_validation_report_json_is_valid(validated_report_path):
    data = json.loads(validated_report_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# passes_thresholds
# ---------------------------------------------------------------------------


def test_passes_thresholds_true_when_all_met(trained_dir, tmp_path, mocker):
    sentences = ["भारत है।"] * 20

    # 2 regular tokens per sentence, 8 chars → chars_per_token = 4.0 > 3.0
    # No UNK → unk_rate = 0.0
    # Decode matches → roundtrip = 1.0
    def encode_side_effect(text, add_special_tokens=False):
        if text in _SPECIAL_TOKEN_IDS:
            return [_SPECIAL_TOKEN_IDS[text]]
        return [42, 43]

    mock_tok = _make_mock_tokenizer(
        mocker, encode_side_effect=encode_side_effect, decode_return="भारत है।"
    )
    _patch_auto_tokenizer(mocker, mock_tok)

    report = TokenizerValidator(artifact_dir=trained_dir, validation_sentences=sentences).validate(
        tmp_path / "report.json"
    )
    assert report.passes_thresholds is True


def test_passes_thresholds_false_on_high_unk_rate(trained_dir, tmp_path, mocker):
    sentence = "भारत है।"

    def encode_side_effect(text, add_special_tokens=False):
        if text in _SPECIAL_TOKEN_IDS:
            return [_SPECIAL_TOKEN_IDS[text]]
        return [1] * 50 + [42] * 50  # 50 UNKs out of 100 → unk_rate = 0.5

    mock_tok = _make_mock_tokenizer(mocker, encode_side_effect=encode_side_effect, decode_return=sentence)
    _patch_auto_tokenizer(mocker, mock_tok)

    report = TokenizerValidator(artifact_dir=trained_dir, validation_sentences=[sentence]).validate(
        tmp_path / "report.json"
    )
    assert report.passes_thresholds is False


def test_passes_thresholds_false_on_low_roundtrip(trained_dir, tmp_path, mocker):
    sentence = "भारत है।"

    def encode_side_effect(text, add_special_tokens=False):
        if text in _SPECIAL_TOKEN_IDS:
            return [_SPECIAL_TOKEN_IDS[text]]
        return [42, 43]

    mock_tok = _make_mock_tokenizer(
        mocker, encode_side_effect=encode_side_effect, decode_return="completely different text"
    )
    _patch_auto_tokenizer(mocker, mock_tok)

    report = TokenizerValidator(artifact_dir=trained_dir, validation_sentences=[sentence]).validate(
        tmp_path / "report.json"
    )
    assert report.passes_thresholds is False


def test_passes_thresholds_false_on_special_token_failure(trained_dir, tmp_path, mocker):
    sentence = "भारत है।"

    def encode_side_effect(text, add_special_tokens=False):
        if text == "<pad>":
            return [100, 101, 102]  # split
        if text in _SPECIAL_TOKEN_IDS:
            return [_SPECIAL_TOKEN_IDS[text]]
        return [42, 43]

    mock_tok = _make_mock_tokenizer(
        mocker, encode_side_effect=encode_side_effect, decode_return=sentence
    )
    _patch_auto_tokenizer(mocker, mock_tok)

    report = TokenizerValidator(artifact_dir=trained_dir, validation_sentences=[sentence]).validate(
        tmp_path / "report.json"
    )
    assert report.passes_thresholds is False


# ---------------------------------------------------------------------------
# Output directory creation
# ---------------------------------------------------------------------------


def test_validator_creates_parent_dirs(trained_dir, validation_sentences, tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "report.json"
    TokenizerValidator(artifact_dir=trained_dir, validation_sentences=validation_sentences).validate(
        report_path=nested
    )
    assert nested.exists()


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_validator_logs_started_event(trained_dir, validation_sentences, tmp_path, mocker):
    mock_logger = mocker.Mock()
    TokenizerValidator(
        artifact_dir=trained_dir, validation_sentences=validation_sentences
    ).validate(tmp_path / "report.json", run_logger=mock_logger)
    statuses = [c.kwargs.get("status") for c in mock_logger.log_event.call_args_list]
    assert "started" in statuses


def test_validator_logs_completed_event(trained_dir, validation_sentences, tmp_path, mocker):
    mock_logger = mocker.Mock()
    TokenizerValidator(
        artifact_dir=trained_dir, validation_sentences=validation_sentences
    ).validate(tmp_path / "report.json", run_logger=mock_logger)
    statuses = [c.kwargs.get("status") for c in mock_logger.log_event.call_args_list]
    assert "completed" in statuses or "failed" in statuses


def test_validator_logs_failed_event_on_thresholds(trained_dir, tmp_path, mocker):
    sentence = "भारत है।"

    def encode_side_effect(text, add_special_tokens=False):
        if text in _SPECIAL_TOKEN_IDS:
            return [_SPECIAL_TOKEN_IDS[text]]
        return [1] * 100  # all UNK → unk_rate = 1.0 → fails thresholds

    mock_tok = _make_mock_tokenizer(
        mocker, encode_side_effect=encode_side_effect, decode_return="wrong"
    )
    _patch_auto_tokenizer(mocker, mock_tok)

    mock_logger = mocker.Mock()
    TokenizerValidator(artifact_dir=trained_dir, validation_sentences=[sentence]).validate(
        tmp_path / "report.json", run_logger=mock_logger
    )
    statuses = [c.kwargs.get("status") for c in mock_logger.log_event.call_args_list]
    assert "failed" in statuses
