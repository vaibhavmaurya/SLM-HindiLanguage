"""Integration tests: TokenizerTrainer + ExperimentRunner pipeline — Phase 10."""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from hindi_tokenizer.training.experiment_runner import ExperimentRunner
from hindi_tokenizer.training.tokenizer_trainer import TokenizerTrainer

_SMALL_VOCAB = 500


@pytest.fixture(scope="module")
def trained_artifact_dir(tmp_path_factory, small_corpus_path):
    out = tmp_path_factory.mktemp("trained")
    TokenizerTrainer(vocab_size=_SMALL_VOCAB).train(corpus_file=small_corpus_path, output_dir=out)
    return out


def test_trainer_on_fixture_small_corpus(trained_artifact_dir):
    assert (trained_artifact_dir / "tokenizer.json").exists()


def test_trained_artifact_loads_with_auto_tokenizer(trained_artifact_dir):
    tok = AutoTokenizer.from_pretrained(str(trained_artifact_dir))
    assert tok is not None


def test_trained_tokenizer_encodes_hindi(trained_artifact_dir):
    tok = AutoTokenizer.from_pretrained(str(trained_artifact_dir))
    result = tok("भारत है।")
    assert len(result["input_ids"]) > 0


def test_trained_tokenizer_roundtrip(trained_artifact_dir, validation_sentences_path):
    tok = AutoTokenizer.from_pretrained(str(trained_artifact_dir))
    sentences = [ln for ln in validation_sentences_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    successes = 0
    for sentence in sentences[:10]:
        ids = tok.encode(sentence, add_special_tokens=False)
        decoded = tok.decode(ids).strip()
        if decoded == sentence.strip():
            successes += 1
    assert successes >= 1


def test_experiment_runner_creates_three_artifacts(tmp_path_factory, small_corpus_path):
    base = tmp_path_factory.mktemp("experiment")
    vocab_sizes = [300, 400, 500]
    ExperimentRunner(
        corpus_file=small_corpus_path,
        output_base_dir=base,
        vocab_sizes=vocab_sizes,
    ).run()
    for vs in vocab_sizes:
        assert (base / f"vocab_{vs}" / "tokenizer.json").exists()
