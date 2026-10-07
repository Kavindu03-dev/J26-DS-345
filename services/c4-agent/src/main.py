from fastapi import FastAPI

app = FastAPI(title="C4 Tourism Agent")


@app.get("/health")
def health():
    return {"service": "c4-agent", "status": "ok"}
