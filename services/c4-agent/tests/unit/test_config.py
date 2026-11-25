from pathlib import Path

import pytest
from pydantic import ValidationError

from c4.common.config import PROJECT_ROOT, Settings, load_yaml, resolve_path


def _env_keys() -> set[str]:
    lines = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    return {line.split("=")[0] for line in lines if "=" in line and not line.startswith("#")}


def test_env_example_matches_settings():
    assert _env_keys() == {name.upper() for name in Settings.model_fields}


def test_env_example_values_load():
    settings = Settings(_env_file=PROJECT_ROOT / ".env.example")
    assert settings.model_dump() == Settings().model_dump()


def test_environment_overrides_default(monkeypatch):
    monkeypatch.setenv("USE_MOCKS", "false")
    monkeypatch.setenv("TOOL_TIMEOUT_S", "2.5")
    settings = Settings()
    assert settings.use_mocks is False
    assert settings.tool_timeout_s == 2.5


def test_invalid_value_rejected(monkeypatch):
    monkeypatch.setenv("TOOL_TIMEOUT_S", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_secrets_are_not_printed(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy_value")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:pw@host/db")
    settings = Settings()
    assert "hf_dummy_value" not in repr(settings)
    assert "pw@host" not in repr(settings)
    assert settings.hf_token.get_secret_value() == "hf_dummy_value"


def test_models_yaml_has_the_four_study_models():
    config = load_yaml("configs/models.yaml")
    assert [m["hf_id"] for m in config["study_models"]] == [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
    ]
    assert config["temperature"] == 0
    assert config["seed"] == 42


def test_yaml_must_be_a_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(path)


def test_relative_paths_use_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert resolve_path("configs/models.yaml").is_file()
    assert resolve_path(tmp_path) == Path(tmp_path)
