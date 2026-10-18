# Step 4: training data preparation

Builds the supervised fine-tuning set described in the proposal (Section 3):
10,000 xLAM-60k tool-call examples plus at least 15% no-call examples,
decontaminated against the BFCL test set.

## Run it (Windows PowerShell, from `services/c4-agent`)

```powershell
.venv\Scripts\activate
pip install -r requirements-training.txt
$env:HF_TOKEN = "hf_..."          # your Hugging Face token (only for this terminal)
python -m training.prepare_data --push-to-hub Kavindu03dev/c4-fc-sft
```

Takes about a minute (mostly the 100 MB download). Output goes to `<repo>/data/c4_fc/`
(ignored by Git) and, with `--push-to-hub`, to a **private** Hugging Face dataset that the
Colab training notebook loads.

| File | What it is |
|------|------------|
| `train.jsonl`, `val.jsonl` | training and validation examples (95% / 5%, stratified by kind) |
| `stats.json` | every count: rows dropped and why, final mix, no-call share, seed, BFCL commit |
| `preview.md` | a few examples of each kind, readable |

## What the pipeline does

1. **Convert** each xLAM row to chat format (`tools` as JSON Schema + `messages` with `tool_calls`),
   which the Qwen2.5 and Llama-3.2 chat templates both accept.
2. **Drop, never repair**, malformed rows: calls to unknown tools, unknown arguments,
   missing required arguments.
3. **Drop duplicates** and anything sharing a 13-word sequence with a BFCL v4 single-turn
   question (`bfcl_v4_reference.json`, 3,491 items, built by `build_bfcl_reference.py`).
4. **Sample** 10,000 tool-call examples with a fixed seed (42), so the run is reproducible.
5. **Add no-call examples** (15% of the final set):
   - 80% *irrelevance*: a real query paired with an unrelated example's tools (no shared keywords),
     answered in text. Queries come from rows not used for tool-call training.
   - 20% *conversational*: greetings, thanks, capability and off-topic requests.
   Tourism wording is deliberately excluded so the tourism test set stays unseen.

## Files

| File | Purpose |
|------|---------|
| `xlam_convert.py` | xLAM row -> chat example; type mapping; quality filter |
| `no_call.py` | irrelevance and conversational no-call examples |
| `decontaminate.py` | BFCL 13-gram overlap filter |
| `prepare_data.py` | the command-line pipeline above |
| `build_bfcl_reference.py` | rebuilds `bfcl_v4_reference.json` from the Gorilla repo |

Tests: `tests/test_training_data.py` (uses `tests/fixtures/xlam_sample.json`, a small synthetic file
in xLAM's exact format with deliberately broken rows).
