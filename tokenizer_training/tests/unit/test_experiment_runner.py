"""Tests for ExperimentRunner — Phase 4."""

from __future__ import annotations

from pathlib import Path

from hindi_tokenizer.training.experiment_runner import ExperimentRunner

_VOCAB_SIZES = [200, 300, 400]  # small for fast unit-test training


def _make_runner(small_corpus_path: Path, output_base_dir: Path, **kwargs) -> ExperimentRunner:
    defaults = {
        "corpus_file": small_corpus_path,
        "output_base_dir": output_base_dir,
        "vocab_sizes": _VOCAB_SIZES,
    }
    defaults.update(kwargs)
    return ExperimentRunner(**defaults)


def test_experiment_runner_trains_all_variants(small_corpus_path, tmp_path):
    dirs = _make_runner(small_corpus_path, tmp_path).run()
    assert len(dirs) == 3
    for d in dirs:
        assert (d / "tokenizer.json").exists()


def test_experiment_runner_uses_configured_vocab_sizes(small_corpus_path, tmp_path):
    runner = _make_runner(small_corpus_path, tmp_path, vocab_sizes=[200, 300])
    dirs = runner.run()
    assert len(dirs) == 2
    assert all((d / "tokenizer.json").exists() for d in dirs)


def test_experiment_runner_uses_same_input_file(small_corpus_path, tmp_path):
    runner = _make_runner(small_corpus_path, tmp_path, vocab_sizes=[200, 300])
    assert runner.corpus_file == small_corpus_path
    dirs = runner.run()
    assert len(dirs) == 2
    assert all((d / "tokenizer.json").exists() for d in dirs)


def test_experiment_runner_logs_each_variant(small_corpus_path, tmp_path, mocker):
    mock_logger = mocker.Mock()
    _make_runner(small_corpus_path, tmp_path, vocab_sizes=[200, 300]).run(run_logger=mock_logger)
    assert mock_logger.log_event.call_count >= 2


def test_experiment_runner_skips_existing_artifact(small_corpus_path, tmp_path):
    """When artifact exists and force_retrain=False, the training step is skipped."""
    runner = _make_runner(small_corpus_path, tmp_path, vocab_sizes=[200])
    dirs = runner.run()
    # Sentinel file proves the output dir is not wiped on a second run
    sentinel = dirs[0] / "_sentinel.txt"
    sentinel.write_text("skip_marker", encoding="utf-8")
    runner.run(force_retrain=False)
    assert sentinel.exists(), "Sentinel removed — training ran again when it should have been skipped"


def test_experiment_runner_force_retrain_overwrites(small_corpus_path, tmp_path):
    """When force_retrain=True, the existing artifact directory is wiped before retraining."""
    runner = _make_runner(small_corpus_path, tmp_path, vocab_sizes=[200])
    dirs = runner.run()
    sentinel = dirs[0] / "_sentinel.txt"
    sentinel.write_text("force_marker", encoding="utf-8")
    runner.run(force_retrain=True)
    assert not sentinel.exists(), "Sentinel still exists — force_retrain did not wipe the directory"


def test_experiment_runner_returns_artifact_dirs(small_corpus_path, tmp_path):
    dirs = _make_runner(small_corpus_path, tmp_path, vocab_sizes=[200]).run()
    assert len(dirs) == 1
    assert isinstance(dirs[0], Path)
    assert (dirs[0] / "tokenizer.json").exists()
