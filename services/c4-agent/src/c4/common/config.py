"""Settings from environment/.env, plus the YAML config loader."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    vllm_base_url: str = "http://localhost:8000/v1"
    deployed_model_config: Path = Path("configs/deployed_model.yaml")
    hf_token: SecretStr = SecretStr("")

    use_mocks: bool = True
    tool1_url: str = "http://localhost:8090/review-insights"
    tool2_url: str = "http://localhost:8090/recommend-hotels"
    tool3_url: str = "http://localhost:8090/plan-itinerary"
    tool_timeout_s: float = Field(default=10, gt=0)

    kb_backend: str = "chroma"
    kb_path: Path = Path("./data/kb/chroma")
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    web_search_provider: str = "none"
    web_search_api_key: SecretStr = SecretStr("")
    web_search_allowlist: Path = Path("configs/web_allowlist.txt")

    database_url: SecretStr = SecretStr("")
    log_dir: Path = Path("./logs")
    mock_delay_ms: int = Field(default=0, ge=0, le=60_000)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_path(path: Path | str) -> Path:
    """Resolve a relative path against the project root, not the current directory."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = resolve_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data
