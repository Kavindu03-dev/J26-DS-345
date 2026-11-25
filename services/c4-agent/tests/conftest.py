import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from c4.common.config import PROJECT_ROOT, Settings, get_settings

SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
START_TIMEOUT_S = 900  # the first run downloads the model


@pytest.fixture(autouse=True)
def isolated_settings(request, monkeypatch):
    """Non-GPU tests must not depend on the developer's .env or environment."""
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setitem(Settings.model_config, "env_file", None)
        for name in Settings.model_fields:
            monkeypatch.delenv(name.upper(), raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _served_models(base_url: str) -> list[str] | None:
    try:
        reply = httpx.get(f"{base_url}/models", timeout=5)
        reply.raise_for_status()
    except httpx.HTTPError:
        return None
    return [m["id"] for m in reply.json()["data"]]


@pytest.fixture(scope="session")
def vllm_server(tmp_path_factory):
    """Use a running vLLM server, or start one with scripts/serve_vllm.sh."""
    base_url = Settings().vllm_base_url.rstrip("/")
    served = _served_models(base_url)
    if served is not None:
        assert SMOKE_MODEL in served, f"{base_url} serves {served}, expected {SMOKE_MODEL}"
        yield base_url
        return

    log = Path(tmp_path_factory.mktemp("vllm")) / "server.log"
    with log.open("w", encoding="utf-8") as out:
        process = subprocess.Popen(
            ["bash", "scripts/serve_vllm.sh", SMOKE_MODEL],
            cwd=PROJECT_ROOT,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        try:
            deadline = time.monotonic() + START_TIMEOUT_S
            while _served_models(base_url) is None:
                tail = log.read_text(encoding="utf-8", errors="replace")[-3000:]
                assert process.poll() is None, f"vLLM stopped while starting:\n{tail}"
                assert time.monotonic() < deadline, f"vLLM did not start in time:\n{tail}"
                time.sleep(3)
            yield base_url
        finally:
            _stop(process)


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        import signal

        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
