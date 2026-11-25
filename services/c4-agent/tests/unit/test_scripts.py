from c4.common.config import PROJECT_ROOT
from scripts.check_gpu import build_warnings, parse_cuda, parse_query
from scripts.smoke_generate import build_request


def test_parse_gpu_query():
    text = "NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB, 610.78\n"
    assert parse_query(text) == [
        {"name": "NVIDIA GeForce RTX 4050 Laptop GPU", "vram_mb": 6141, "driver": "610.78"}
    ]
    assert parse_query("") == []


def test_parse_cuda_old_and_new_header():
    assert parse_cuda("| CUDA Version: 12.4 |") == "12.4"
    assert parse_cuda("| CUDA UMD Version: 13.3 |") == "13.3"
    assert parse_cuda("") is None


def test_warnings():
    big = {"name": "T4", "vram_mb": 16000, "driver": "1"}
    small = {"name": "RTX 4050", "vram_mb": 6141, "driver": "1"}
    assert build_warnings([big], "Linux") == []
    assert "below 8 GB" in build_warnings([small], "Linux")[0]
    assert "WSL2" in build_warnings([big], "Windows")[0]
    assert "No NVIDIA GPU" in build_warnings([], "Linux")[0]


def test_smoke_request_is_greedy_and_raw():
    request = build_request("m", "p")
    assert request["temperature"] == 0
    assert request["seed"] == 42
    assert request["add_special_tokens"] is False
    assert request["skip_special_tokens"] is False


def test_shell_files_have_lf_endings():
    for name in ("scripts/serve_vllm.sh", "Makefile"):
        assert b"\r" not in (PROJECT_ROOT / name).read_bytes(), name
