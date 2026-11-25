# C4 plan and progress

Phase numbers follow `README.md` section 8. The Claude doc "Component 4 Process Plan" numbers
them differently (its Phase 0 = README 0+1, its Phase 1 = README 2).

Scope: backend only, and only inside `services/c4-agent`. Phase 7 (frontend) is deferred.

Status: `[x]` done and tested, `[~]` built but not verified, `[ ]` not started.

## Phase 0 - Foundation and mocks

- [x] 0.1 Project config: `pyproject.toml`, `uv.lock`, `.env.example`, `Makefile`, `CLAUDE.md`
- [x] 0.2 Tool schemas v1 for Tools 1-5, loader and validator
- [x] 0.3 Settings and YAML loader, ids, JSONL logger with PII scrubbing
- [x] 0.4 Mock Tools 1-3 on port 8090 (404, 422, delay)
- [x] 0.5 `check_gpu.py` (run on Windows; rerun in WSL2 so `results/env.json` records Linux)
- [~] 0.6 `serve_vllm.sh` and `smoke_generate.py`: need WSL2
- [x] 0.7 Unit and integration tests, lint
- [ ] 0.8 `make lint test` inside WSL2, then `make test-gpu`

## Phase 1 - Parser, decoding modes, evaluation

- [ ] 1.1 Record the exact raw output of Qwen2.5 and Llama-3.2, then plan for approval
- [ ] 1.2 Strict parser and adversarial tests (no repair)
- [ ] 1.3 vLLM client (TTFT, total time, tokens) and prompt rendering
- [ ] 1.4 Grammar "tool call(s) OR plain answer", with accept/reject tests
- [ ] 1.5 Modes a, b, c, d behind one interface
- [ ] 1.6 Metrics and tests
- [ ] 1.7 BFCL handler
- [ ] 1.8 `validate_pipeline.py`: reference score within +/-3 points

## Phase 2 - Data

- [ ] 2.1 Training set: sample, >= 15% no-call, BFCL dedup, render, split
- [ ] 2.2 Test-set tooling: draft, validate, kappa (gold labels are the student's)
- [ ] 2.3 Agent test set (`knowledge_base`, `clarify`)
- [ ] 2.4 Knowledge base: ingest, chunk, embed, retrieve, recall@4
- [ ] 2.5 Freeze and verify, after the student confirms

## Phase 3 - QLoRA

- [ ] 3.1 `qlora.yaml`, `qlora.py`, dry-run test
- [ ] 3.2 0.5B pilot, then freeze the config after approval
- [ ] 3.3 Train 1.5B, Llama-1B, 3B with the same config
- [ ] 3.4 Merge, serve and sanity check

## Phase 4 - Experiments

- [ ] 4.1 Statistics and report code on synthetic data
- [ ] 4.2 `run_matrix.py` (resumable) and keyword baseline
- [ ] 4.3 20-item dry run, then the full run
- [ ] 4.4 Report H1-H4, error sample, configuration choice

## Phase 5 - Agent core

- [ ] 5.1 State and fake LLM client
- [ ] 5.2 Nodes: guardrails, router, clarify, executor, synthesiser
- [ ] 5.3 Graph (max 4 tool rounds, memory) and tool registry
- [ ] 5.4 FastAPI `/chat` with SSE, `/health`, logging
- [ ] 5.5 Scenario tests, live end-to-end test

## Phase 6 - Knowledge base, web search, guardrails

- [ ] 6.1 Injection test suite first
- [ ] 6.2 Tool 5 and Tool 4 (`none` and `cached` providers)
- [ ] 6.3 Guardrails, labels, source badges data
- [ ] 6.4 Add a tool by schema only

## Phase 7 - Frontend: deferred (backend only for now)

## Phase 8 - Integration

- [ ] 8.1 Load the chosen model and mode
- [ ] 8.2 Switch Tools 1-3 to real services one at a time, with contract tests
- [ ] 8.3 Shared database, Dockerfile, deployment, smoke tests

## Phase 9 - System evaluation

- [ ] 9.1 Agent evaluation, load test, requirement report
- [ ] 9.2 User-study materials and SUS scorer

## Phase 10 - Release

- [ ] 10.1 Model cards, benchmark package, `make reproduce-small`, thesis assets

## Dummy things in use

| Waiting for | Dummy used now | Where | Replaced in |
|---|---|---|---|
| C1 reviews | Fictional scores and facts | `src/c4/mocks` | 8.2 |
| C2 hotels | 15 fictional hotels and prices | `src/c4/mocks` | 8.2 |
| C3 itinerary | Simple greedy days, fictional restaurants | `src/c4/mocks` | 8.2 |
| Shared database | `DATABASE_URL` empty | `.env.example` | 8.3 |
| Web search provider | `none` | `.env.example` | 6.2 |

## Open issues

1. **Schemas differ from the team's code. Settle before the Phase 2 freeze**, because gold labels
   use these argument names.
   - C1: shared contract is `GET /hotels/{hotel_id}/aspect-scores`, with no `aspects` filter and
     no `weak_aspect_alert`.
   - C2: shared contract uses `check_out`, `adults`, `children`, nested `budget_lkr` and returns
     `ranked_hotels`. C2 itself only has `/health` so far. Per-night or whole-stay price is undecided.
   - C3: the real `POST /plan-itinerary` takes a full hotel object, trip hours and 0-1 preference
     weights, not only `hotel_id`.
   - Our schemas follow README 6.1. `travel_purpose` is free text because its list is unfinished.
2. **6 GB GPU.** Qwen2.5-3B at 16-bit needs about 6 GB for weights alone, so it will not serve
   here. Ground rule 4 needs one machine for all 16 configurations, so the Phase 4 matrix should
   run on one 16 GB GPU (for example Kaggle). This laptop is for development and the small models.
3. **Docker.** `docker-compose.yml` builds this folder, and there is no Dockerfile until the agent
   API exists. `docker compose up --build` fails for the team if this is merged before then.
   Compose also expects C4 on port 8004; README says 8080.
4. **Supervisor sign-off** for local GPU, LangGraph, RAG and Next.js is required by the process
   plan before that work starts (Phases 5-7).
5. **PII names.** The scrubber finds introduced names ("I'm ...", "my name is ...", titles) and
   known names. A bare name in free text is not detected.

## Running the tests

In WSL2, from this folder: `make setup lint test`, then `make setup-gpu test-gpu`.

On Windows (no make; Smart App Control blocks uv's Python and `pytest.exe`), CPU tests only:

```powershell
py install 3.11
py -m uv sync --python (py -V:3.11 -c "import sys;print(sys.executable)")
py -m uv run --no-sync ruff check .
py -m uv run --no-sync python -m pytest
```
