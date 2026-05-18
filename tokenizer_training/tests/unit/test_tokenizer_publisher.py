"""Tests for TokenizerPublisher — Phase 8."""

from __future__ import annotations

import pytest

from hindi_tokenizer.publishing.tokenizer_publisher import TokenizerPublisher

_REPO_ID = "vaibhavmaurya/hindi-slm-tokenizer-v001"


@pytest.fixture()
def artifact_dir(tmp_path):
    d = tmp_path / "artifact"
    d.mkdir()
    (d / "tokenizer.json").write_text('{"model": "unigram"}', encoding="utf-8")
    (d / "VERSION").write_text("hindi_slm_tokenizer_v001", encoding="utf-8")
    return d


@pytest.fixture()
def publisher(artifact_dir):
    return TokenizerPublisher(source_dir=artifact_dir, repo_id=_REPO_ID)


# ---------------------------------------------------------------------------
# Core HF Hub calls
# ---------------------------------------------------------------------------


def test_publisher_calls_create_repo(publisher, mocker):
    mock_api = mocker.MagicMock()
    mocker.patch("hindi_tokenizer.publishing.tokenizer_publisher.HfApi", return_value=mock_api)
    publisher.publish()
    mock_api.create_repo.assert_called_once()
    call_kwargs = mock_api.create_repo.call_args
    assert call_kwargs.kwargs.get("repo_id") == _REPO_ID or _REPO_ID in call_kwargs.args


def test_publisher_calls_upload_folder(publisher, mocker):
    mock_api = mocker.MagicMock()
    mocker.patch("hindi_tokenizer.publishing.tokenizer_publisher.HfApi", return_value=mock_api)
    publisher.publish()
    mock_api.upload_folder.assert_called_once()
    call_kwargs = mock_api.upload_folder.call_args
    assert call_kwargs.kwargs.get("repo_id") == _REPO_ID or _REPO_ID in call_kwargs.args


def test_publisher_create_repo_called_before_upload(publisher, mocker):
    call_order = []
    mock_api = mocker.MagicMock()
    mock_api.create_repo.side_effect = lambda **kw: call_order.append("create_repo")
    mock_api.upload_folder.side_effect = lambda **kw: call_order.append("upload_folder")
    mocker.patch("hindi_tokenizer.publishing.tokenizer_publisher.HfApi", return_value=mock_api)
    publisher.publish()
    assert call_order.index("create_repo") < call_order.index("upload_folder")


def test_publisher_exist_ok_true_on_create(publisher, mocker):
    mock_api = mocker.MagicMock()
    mocker.patch("hindi_tokenizer.publishing.tokenizer_publisher.HfApi", return_value=mock_api)
    publisher.publish()
    call_kwargs = mock_api.create_repo.call_args.kwargs
    assert call_kwargs.get("exist_ok") is True


def test_publisher_upload_folder_path_is_artifact_dir(artifact_dir, mocker):
    mock_api = mocker.MagicMock()
    mocker.patch("hindi_tokenizer.publishing.tokenizer_publisher.HfApi", return_value=mock_api)
    TokenizerPublisher(source_dir=artifact_dir, repo_id=_REPO_ID).publish()
    call_kwargs = mock_api.upload_folder.call_args.kwargs
    assert str(artifact_dir) == call_kwargs.get("folder_path") or artifact_dir == call_kwargs.get("folder_path")


# ---------------------------------------------------------------------------
# Token & config
# ---------------------------------------------------------------------------


def test_publisher_uses_hf_token_from_env(artifact_dir, mocker, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token-xyz")
    mock_api_cls = mocker.patch("hindi_tokenizer.publishing.tokenizer_publisher.HfApi")
    TokenizerPublisher(source_dir=artifact_dir, repo_id=_REPO_ID).publish()
    mock_api_cls.assert_called_once_with(token="test-token-xyz")


def test_publisher_uses_repo_id_from_config(artifact_dir, mocker):
    custom_repo = "vaibhavmaurya/custom-repo"
    mock_api = mocker.MagicMock()
    mocker.patch("hindi_tokenizer.publishing.tokenizer_publisher.HfApi", return_value=mock_api)
    TokenizerPublisher(source_dir=artifact_dir, repo_id=custom_repo).publish()
    call_kwargs = mock_api.create_repo.call_args.kwargs
    assert call_kwargs.get("repo_id") == custom_repo


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_publisher_raises_on_missing_artifact_dir(tmp_path, mocker):
    mocker.patch("hindi_tokenizer.publishing.tokenizer_publisher.HfApi")
    p = TokenizerPublisher(source_dir=tmp_path / "nonexistent", repo_id=_REPO_ID)
    with pytest.raises(FileNotFoundError):
        p.publish()


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_publisher_dry_run_skips_upload(publisher, mocker):
    mock_api_cls = mocker.patch("hindi_tokenizer.publishing.tokenizer_publisher.HfApi")
    publisher.publish(dry_run=True)
    mock_api_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_publisher_logs_started_event(publisher, mocker):
    mock_api = mocker.MagicMock()
    mocker.patch("hindi_tokenizer.publishing.tokenizer_publisher.HfApi", return_value=mock_api)
    mock_logger = mocker.Mock()
    publisher.publish(run_logger=mock_logger)
    statuses = [c.kwargs.get("status") for c in mock_logger.log_event.call_args_list]
    assert "started" in statuses


def test_publisher_logs_completed_event(publisher, mocker):
    mock_api = mocker.MagicMock()
    mocker.patch("hindi_tokenizer.publishing.tokenizer_publisher.HfApi", return_value=mock_api)
    mock_logger = mocker.Mock()
    publisher.publish(run_logger=mock_logger)
    statuses = [c.kwargs.get("status") for c in mock_logger.log_event.call_args_list]
    assert "completed" in statuses
