# Run inside Linux/WSL2. Windows equivalents are in PLAN.md.
UV    ?= uv
MODEL ?= Qwen/Qwen2.5-0.5B-Instruct

# On WSL2 keep the venv on the Linux disk: /mnt/e is slow and may hold a Windows .venv.
ifneq (,$(findstring icrosoft,$(shell uname -r)))
export UV_PROJECT_ENVIRONMENT ?= $(HOME)/.venvs/c4-agent
endif

.PHONY: setup setup-gpu test test-gpu lint mocks serve smoke check-gpu requirements

setup:
	$(UV) sync
setup-gpu:
	$(UV) sync --group gpu
test:
	$(UV) run --no-sync python -m pytest -m "not gpu and not network"
test-gpu: check-gpu
	$(UV) run --no-sync python -m pytest -m gpu
lint:
	$(UV) run --no-sync ruff check .
	$(UV) run --no-sync ruff format --check .
mocks:
	$(UV) run --no-sync uvicorn c4.mocks.app:app --port 8090
serve:
	bash scripts/serve_vllm.sh $(MODEL)
smoke:
	$(UV) run --no-sync python scripts/smoke_generate.py --model $(MODEL)
check-gpu:
	$(UV) run --no-sync python scripts/check_gpu.py
requirements:  # CI installs with pip from this file
	$(UV) export --group dev --no-group gpu --no-hashes --no-emit-project -o requirements.txt
