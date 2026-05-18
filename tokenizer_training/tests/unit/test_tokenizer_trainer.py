"""Tests for TokenizerTrainer — Phase 4."""

from __future__ import annotations

import json

import pytest
from transformers import AutoTokenizer

from hindi_tokenizer.training.tokenizer_trainer import TokenizerTrainer

_SMALL_VOCAB = 500  # fast training on the 177-line fixture corpus

_SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory, small_corpus_path):
    """Train once and reuse across module tests — training is the expensive step."""
    out = tmp_path_factory.mktemp("trainer_out")
    TokenizerTrainer(vocab_size=_SMALL_VOCAB).train(corpus_file=small_corpus_path, output_dir=out)
    return out


# ---------------------------------------------------------------------------
# Artifact existence
# ---------------------------------------------------------------------------


def test_train_creates_tokenizer_json(trained_dir):
    assert (trained_dir / "tokenizer.json").exists()


def test_train_creates_tokenizer_config_json(trained_dir):
    assert (trained_dir / "tokenizer_config.json").exists()


def test_train_creates_special_tokens_map_json(trained_dir):
    assert (trained_dir / "special_tokens_map.json").exists()


def test_train_creates_metadata_json(trained_dir):
    assert (trained_dir / "tokenizer_metadata.json").exists()


# ---------------------------------------------------------------------------
# Metadata content
# ---------------------------------------------------------------------------


def test_train_metadata_contains_algorithm(trained_dir):
    data = json.loads((trained_dir / "tokenizer_metadata.json").read_text(encoding="utf-8"))
    assert data["algorithm"] == "unigram"


def test_train_metadata_contains_vocab_size(trained_dir):
    data = json.loads((trained_dir / "tokenizer_metadata.json").read_text(encoding="utf-8"))
    assert data["vocab_size"] == _SMALL_VOCAB


def test_train_metadata_contains_normalizer(trained_dir):
    data = json.loads((trained_dir / "tokenizer_metadata.json").read_text(encoding="utf-8"))
    assert data["normalizer"] == "NFKC"


def test_train_metadata_contains_pre_tokenizer(trained_dir):
    data = json.loads((trained_dir / "tokenizer_metadata.json").read_text(encoding="utf-8"))
    assert data["pre_tokenizer"] == "Metaspace"


# ---------------------------------------------------------------------------
# Vocab size behaviour
# ---------------------------------------------------------------------------


def test_train_vocab_size_32k(small_corpus_path, tmp_path):
    """Metadata records the requested vocab_size; tokenizer loads successfully."""
    TokenizerTrainer(vocab_size=32000).train(corpus_file=small_corpus_path, output_dir=tmp_path)
    data = json.loads((tmp_path / "tokenizer_metadata.json").read_text(encoding="utf-8"))
    assert data["vocab_size"] == 32000
    tok = AutoTokenizer.from_pretrained(str(tmp_path))
    assert len(tok) > 0


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------


def test_train_special_tokens_present(trained_dir):
    tok = AutoTokenizer.from_pretrained(str(trained_dir))
    for token in _SPECIAL_TOKENS:
        assert token in tok.all_special_tokens, f"{token!r} missing from tokenizer"


def test_train_pad_token_id_is_zero(trained_dir):
    tok = AutoTokenizer.from_pretrained(str(trained_dir))
    assert tok.pad_token_id == 0


def test_train_unk_token_id_is_one(trained_dir):
    tok = AutoTokenizer.from_pretrained(str(trained_dir))
    assert tok.unk_token_id == 1


def test_train_bos_token_id_is_two(trained_dir):
    tok = AutoTokenizer.from_pretrained(str(trained_dir))
    assert tok.bos_token_id == 2


def test_train_eos_token_id_is_three(trained_dir):
    tok = AutoTokenizer.from_pretrained(str(trained_dir))
    assert tok.eos_token_id == 3


# ---------------------------------------------------------------------------
# Happy-path round-trips
# ---------------------------------------------------------------------------


def test_train_loads_with_auto_tokenizer(trained_dir):
    tok = AutoTokenizer.from_pretrained(str(trained_dir))
    assert tok is not None


def test_train_small_fixture_corpus(small_corpus_path, tmp_path):
    TokenizerTrainer(vocab_size=_SMALL_VOCAB).train(corpus_file=small_corpus_path, output_dir=tmp_path)


def test_train_creates_output_dir_if_missing(small_corpus_path, tmp_path):
    out = tmp_path / "nested" / "output"
    TokenizerTrainer(vocab_size=_SMALL_VOCAB).train(corpus_file=small_corpus_path, output_dir=out)
    assert out.exists()


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_train_logs_started_event(small_corpus_path, tmp_path, mocker):
    mock_logger = mocker.Mock()
    TokenizerTrainer(vocab_size=_SMALL_VOCAB).train(
        corpus_file=small_corpus_path, output_dir=tmp_path, run_logger=mock_logger
    )
    statuses = [c.kwargs.get("status") for c in mock_logger.log_event.call_args_list]
    assert "started" in statuses


def test_train_logs_completed_event(small_corpus_path, tmp_path, mocker):
    mock_logger = mocker.Mock()
    TokenizerTrainer(vocab_size=_SMALL_VOCAB).train(
        corpus_file=small_corpus_path, output_dir=tmp_path, run_logger=mock_logger
    )
    statuses = [c.kwargs.get("status") for c in mock_logger.log_event.call_args_list]
    assert "completed" in statuses


def test_train_registers_tokenizer_json(small_corpus_path, tmp_path, mocker):
    mock_registry = mocker.Mock()
    TokenizerTrainer(vocab_size=_SMALL_VOCAB).train(
        corpus_file=small_corpus_path, output_dir=tmp_path, file_registry=mock_registry
    )
    mock_registry.register_file.assert_called()
    registered_paths = [str(c.kwargs.get("path", "")) for c in mock_registry.register_file.call_args_list]
    assert any("tokenizer.json" in p for p in registered_paths)
