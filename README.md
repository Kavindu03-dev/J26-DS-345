# J26-DS-345: Tourism Review Analysis and Recommendation System for Sri Lankan Hotels

SLIIT B.Sc. (Hons) IT, Data Science research project. Four components built by four people, run together as one platform.

| Component | Folder | Owner |
|-----------|--------|-------|
| C1: Trust-aware review analysis and aspect sentiment engine | `services/c1-review-engine` | Dissanayake L.G.D.P.M. |
| C2: Personalized multi-criteria hotel recommendation and price intelligence | `services/c2-hotel-recommender` | Hansada S.A.B. |
| C3: Spatial-temporal itinerary optimization and route-based restaurant recommendation | `services/c3-itinerary-planner` | Bandara H.A.N.D. |
| C4: LLM agent with function calling (calls C1, C2 and C3 as tools) | `services/c4-agent` | Gajanayaka K.R.K.V. |

Supervisor: Prof. Samantha Thelijjagoda. Co-supervisor: Dr. Junius Anjana.

## Repository layout

```
services/      one folder per component (own code, tests, requirements, Dockerfile)
contracts/     shared API formats between components (change only via pull request)
frontend/      web UI
data/          local datasets (not committed to Git)
docs/          proposals, diagrams, reports
.github/       CI and notification workflows
```

## Branch workflow

1. `main` always runs. Nobody pushes to it directly.
2. Work on a short-lived branch named after your component, e.g. `c4/function-calling` or `c2/price-model`.
3. Open a pull request into `main` every few days and get one teammate to review it.
4. After every merge, pull the latest `main` into your own branch.
5. Stay inside your own `services/` folder. Changes to `contracts/` or `docker-compose.yml` need all four of you to agree.

## Run everything

```
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---------|-----|
| C1 | http://localhost:8001/health |
| C2 | http://localhost:8002/health |
| C3 | http://localhost:8003/health |
| C4 | http://localhost:8004/health |

## Run one service locally

```
cd services/c4-agent
pip install -r requirements.txt
pytest
uvicorn src.main:app --reload --port 8004
```

## Secrets and large files

- Never commit API keys. Copy `.env.example` to `.env` (ignored by Git).
- Never commit datasets or model checkpoints. Share them through Git LFS, DVC, or a Drive / Hugging Face link.
