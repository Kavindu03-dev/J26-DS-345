from fastapi import FastAPI

app = FastAPI(title="C1 Trust-Aware Review Engine")


@app.get("/health")
def health():
    return {"service": "c1-review-engine", "status": "ok"}
