"""Unit tests for the run_tokenizer CLI — Phase 10."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from hindi_tokenizer.orchestration.run_tokenizer import app

runner = CliRunner()

_REL_CONFIG = "tests/fixtures/sample_configs/test_tokenizer_training_config.yaml"


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_exits_without_writing_data(tmp_path, test_config_path):
    result = runner.invoke(app, ["--config", str(test_config_path), "--dry-run"])
    assert result.exit_code == 0
    assert not (tmp_path / "data").exists()


def test_dry_run_prints_config_summary(test_config_path):
    result = runner.invoke(app, ["--config", str(test_config_path), "--dry-run"])
    assert result.exit_code == 0
    assert "hindi-slm-tokenizer" in result.output
    assert str(test_config_path) in result.output


# ---------------------------------------------------------------------------
# Individual step dispatch
# ---------------------------------------------------------------------------


def test_step_sample_calls_corpus_sampler(mocker):
    mock_sampler_cls = mocker.patch("hindi_tokenizer.orchestration.run_tokenizer.CorpusSampler")
    mock_sampler_cls.return_value.sample.return_value = mocker.MagicMock()
    result = runner.invoke(app, ["--config", _REL_CONFIG, "--step", "sample"])
    assert result.exit_code == 0
    mock_sampler_cls.return_value.sample.assert_called_once()


def test_step_train_calls_experiment_runner(mocker):
    mock_runner_cls = mocker.patch("hindi_tokenizer.orchestration.run_tokenizer.ExperimentRunner")
    mock_runner_cls.return_value.run.return_value = []
    result = runner.invoke(app, ["--config", _REL_CONFIG, "--step", "train"])
    assert result.exit_code == 0
    mock_runner_cls.return_value.run.assert_called_once()


def test_step_validate_calls_tokenizer_validator(mocker):
    mock_val_cls = mocker.patch("hindi_tokenizer.orchestration.run_tokenizer.TokenizerValidator")
    mock_report = mocker.MagicMock()
    mock_report.passes_thresholds = True
    mock_val_cls.return_value.validate.return_value = mock_report
    result = runner.invoke(app, ["--config", _REL_CONFIG, "--step", "validate"])
    assert result.exit_code == 0
    assert mock_val_cls.return_value.validate.call_count >= 1


def test_step_compare_calls_tokenizer_comparator(mocker, tmp_path, monkeypatch, test_config_path):
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / "data" / "reports"
    reports_dir.mkdir(parents=True)
    for vs in [24000, 32000, 48000]:
        (reports_dir / f"validation_vocab_{vs}.json").write_text('{"passes_thresholds": true}', encoding="utf-8")

    mock_cmp_cls = mocker.patch("hindi_tokenizer.orchestration.run_tokenizer.TokenizerComparator")
    mock_result = mocker.MagicMock()
    mock_result.recommended_variant = "hindi_unigram_32k_v001"
    mock_cmp_cls.return_value.compare.return_value = mock_result

    runner.invoke(app, ["--config", str(test_config_path), "--step", "compare"])
    mock_cmp_cls.return_value.compare.assert_called_once()


def test_step_package_calls_artifact_packager(mocker, tmp_path, monkeypatch, test_config_path):
    monkeypatch.chdir(tmp_path)
    mock_pkg_cls = mocker.patch("hindi_tokenizer.orchestration.run_tokenizer.ArtifactPackager")
    mock_pkg_cls.return_value.package.return_value = tmp_path / "data" / "final"
    mocker.patch("hindi_tokenizer.orchestration.run_tokenizer.ChecksumGenerator")
    result = runner.invoke(app, ["--config", str(test_config_path), "--step", "package"])
    assert result.exit_code == 0
    mock_pkg_cls.return_value.package.assert_called_once()


def test_step_publish_calls_tokenizer_publisher(mocker, tmp_path, monkeypatch, test_config_path):
    monkeypatch.chdir(tmp_path)
    mock_pub_cls = mocker.patch("hindi_tokenizer.orchestration.run_tokenizer.TokenizerPublisher")
    mock_pub_cls.return_value.publish.return_value = None
    result = runner.invoke(app, ["--config", str(test_config_path), "--step", "publish"])
    assert result.exit_code == 0
    mock_pub_cls.return_value.publish.assert_called_once()


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------


def test_creates_output_directories_at_startup(mocker, tmp_path, monkeypatch, test_config_path):
    monkeypatch.chdir(tmp_path)
    mocker.patch("hindi_tokenizer.orchestration.run_tokenizer.CorpusSampler").return_value.sample.return_value = (
        mocker.MagicMock()
    )
    runner.invoke(app, ["--config", str(test_config_path), "--step", "sample"])
    assert (tmp_path / "data" / "reports").exists()
    assert (tmp_path / "data" / "artifacts").exists()


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------


def test_run_id_is_uuid(mocker, tmp_path, monkeypatch, test_config_path):
    monkeypatch.chdir(tmp_path)
    mocker.patch("hindi_tokenizer.orchestration.run_tokenizer.CorpusSampler").return_value.sample.return_value = (
        mocker.MagicMock()
    )
    result = runner.invoke(app, ["--config", str(test_config_path), "--step", "sample"])
    uuid_pattern = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    assert uuid_pattern.search(result.output), f"No UUID found in output: {result.output!r}"


# ---------------------------------------------------------------------------
# Smoke test flag
# ---------------------------------------------------------------------------


def test_smoke_test_flag_uses_smoke_sample(mocker, tmp_path, monkeypatch, test_config_path):
    monkeypatch.chdir(tmp_path)
    mock_sampler_cls = mocker.patch("hindi_tokenizer.orchestration.run_tokenizer.CorpusSampler")
    mock_sampler_cls.return_value.sample.return_value = mocker.MagicMock()
    runner.invoke(app, ["--config", str(test_config_path), "--step", "sample", "--smoke-test"])
    call_kwargs = mock_sampler_cls.return_value.sample.call_args.kwargs
    output_file = str(call_kwargs.get("output_file", ""))
    assert "smoke_test" in output_file
