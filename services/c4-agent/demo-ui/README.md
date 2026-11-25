# C4 offline demo (React)

A fast, offline demo of the Component 4 tourism agent for presentations.
**Open `index.html` (or `dist/index.html`) by double-clicking it** – it runs in any browser with
no internet, no server and no GPU.

## What is real and what is scripted

| Part | Source |
|---|---|
| Router (the fine-tuned model's tool choice) | **Scripted** – the test set's correct calls, so no GPU is needed |
| Strict parsing and schema validation | Real C4 code |
| Hotel name → ID, C1 / C2 / C3 calls | Real C4 executor + mock services (C1 uses Component 1's published data; C2 prices and C3 restaurants are mock data) |
| The final answers | Real C4 synthesiser output |
| Decoding-modes view | **Illustrative** model outputs, scored by the real parser, validator and scorer |

The app says this on screen ("Offline demo · scripted router · real C4 outputs").
The live model runs in `notebooks/03_demo.ipynb`.

## Views
- **Chat** – Claude-style chat: tool-call cards (click to see the strict check and
  the raw tool result), answers, multi-turn follow-ups, example requests.
- **Decoding modes** – one question under the four decoding modes (constraint tax).
- **How it works** – pipeline, tools and key numbers.

## Rebuild (only if you change something)
```bash
# from services/c4-agent
python demo-ui/tools/export_scenarios.py   # regenerates src/scenarios.json from the real C4 code
cd demo-ui
npm install
npm run build                              # writes dist/index.html (one self-contained file)
```
