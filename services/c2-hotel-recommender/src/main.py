from fastapi import FastAPI

app = FastAPI(title="C2 Personalized Hotel Recommender")


@app.get("/health")
def health():
    return {"service": "c2-hotel-recommender", "status": "ok"}
