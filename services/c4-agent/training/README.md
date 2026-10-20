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

---

# Step 5: QLoRA fine-tuning

Notebook: `notebooks/01_finetune_qlora.ipynb` (generated from `fc_format.py` and `train_qlora.py`
by `python -m training.build_notebook`; a test fails if the notebook is out of date).

## Settings (identical for every model, proposal Table 4)

| Setting | Value | Why |
|---|---|---|
| Base model | 4-bit NF4, double quantisation, frozen | QLoRA: fits a 16 GB T4 |
| LoRA | r = 16, alpha = 32, dropout 0.05, all attention + MLP projections | standard QLoRA setup |
| Optimiser | AdamW (full precision), LR 2e-4, cosine, 3% warm-up | only ~2% of parameters train, so 8-bit states save ~nothing; avoids a bitsandbytes crash on T4 |
| Attention | standard (eager) on T4, SDPA on newer GPUs | SDPA fp16 kernels crashed (illegal memory access) on T4 |
| Batch | 1 x 16 gradient accumulation = effective batch 16 | Qwen's 152k vocabulary makes larger micro-batches run out of T4 memory |
| Epochs | 1 (about 700 steps) | 11k examples; avoids over-fitting the format |
| Max length | 2,048 tokens | > 99% of examples fit |
| Loss | assistant answer only (prompt tokens masked with -100) | the model learns to answer, not to repeat tools |
| Precision | fp16 on T4 (no native bf16), bf16 on GPUs with compute capability >= 8 | emulated bf16 on a T4 is slow; note: the first 0.5B run used emulated bf16 |
| Seed | 42 | reproducible |
| Gradient checkpointing | on with eager attention (T4), off otherwise | memory only (identical maths) |
| Progress evaluation | 200 validation examples every 100 steps; final loss on all 593 | speed only |

## Outputs (Google Drive: `tourism-agent/checkpoints/<model>/`)

- `final_adapter/`: LoRA adapter + tokenizer (also uploaded privately to `Kavindu03dev/c4-<model>-fc-lora`)
- `train_summary.json`: settings, data counts, token lengths, training time, losses, quick check
- `quick_eval_samples.json`: 50 validation answers with correct / wrong
- Weights & Biases project `c4-function-calling`: loss curves

The quick check is greedy free decoding on 50 validation examples - a sanity check only.
The study's evaluation (four decoding modes, BFCL, tourism set) is Steps 7-8.
