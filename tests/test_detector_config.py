"""Detector configuration: model path resolution and confidence threshold."""

import importlib

import pytest

import detector_service.config as cfg


@pytest.fixture
def reload_config(monkeypatch):
    """Reload detector_service.config with given env vars, restoring the original afterwards."""
    def _reload(**env):
        for key in ("MODEL_PATH", "CONF_THRESHOLD"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(cfg)

    yield _reload
    monkeypatch.undo()
    importlib.reload(cfg)


def test_default_model_path_is_inside_repo():
    assert cfg.resolve_model_path(None) == cfg.REPO_ROOT / "models" / "yolov8n.pt"
    assert cfg.resolve_model_path("") == cfg.DEFAULT_MODEL_PATH


@pytest.mark.parametrize("value", ["models/yolov8n.pt", "models\\yolov8n.pt"])
def test_relative_path_resolves_from_repo_root_not_cwd(value, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # CWD must not matter
    assert cfg.resolve_model_path(value) == cfg.REPO_ROOT / "models" / "yolov8n.pt"


def test_absolute_path_kept(tmp_path):
    weights = tmp_path / "custom.pt"
    assert cfg.resolve_model_path(str(weights)) == weights


def test_conf_threshold_default_unchanged(reload_config):
    assert reload_config().CONF_THRESHOLD == 0.03


def test_conf_threshold_from_env(reload_config):
    assert reload_config(CONF_THRESHOLD="0.25").CONF_THRESHOLD == 0.25


@pytest.mark.parametrize("bad", ["0", "-0.1", "1.5", "abc"])
def test_invalid_conf_threshold_rejected(reload_config, bad):
    with pytest.raises(ValueError):
        reload_config(CONF_THRESHOLD=bad)
