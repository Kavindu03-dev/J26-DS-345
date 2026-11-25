# C4 front-end demo (Next.js)

Claude-style chat front end for the Component 4 tourism agent. Runs offline on a
laptop – no GPU, no Colab.

## Run it
1. Install **Node.js LTS** (https://nodejs.org) if `node -v` doesn't work.
2. In this folder:
   ```bash
   npm install
   npm run dev
   ```
3. Open **http://localhost:3000**

Do `npm install` **before** the presentation (it needs internet once).
After that, `npm run dev` works offline.

## What is real and what is scripted
- **Scripted:** the router's tool choice (the test set's correct calls), so no GPU is needed.
- **Real C4 code output:** strict checks, hotel-name resolution, C1/C2/C3 mock results
  (C1 uses Component 1's published data) and the synthesised answers
  (`data/scenarios.json`, made by `../demo-ui/tools/export_scenarios.py`).
- **Illustrative:** the four model outputs on the *Decoding modes* tab (scored by the real scorer).

The live model runs in `../notebooks/03_demo.ipynb`.

## Files
- `app/page.jsx`, `app/layout.jsx` – Next.js app router pages
- `components/TourismAgentDemo.jsx` – the whole UI (chat, decoding modes, how it works)
- `app/globals.css` – styles
- `data/scenarios.json` – real outputs of the C4 agent code
