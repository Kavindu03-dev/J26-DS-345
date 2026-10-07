from fastapi import FastAPI

app = FastAPI(title="C3 Itinerary Planner")


@app.get("/health")
def health():
    return {"service": "c3-itinerary-planner", "status": "ok"}
