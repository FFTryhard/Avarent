from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.service import service, start_csv_loop #app, was before (maybe do that later is cleaner)
from app.auth import verify_key

app = FastAPI(title="MGA Quant API")


# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    start_csv_loop()


@app.get("/")
def root():
    return {"status": "online"}


@app.get("/v1/latest")
def latest(client=Depends(verify_key)):
    data = service.get_latest()

    if data is None:
        return {"status": "warming_up"}

    return data


@app.get("/v1/health")
def health():
    return {
        "server": "ok",
        "engine_running": service.running,
        "has_output": service.get_latest() is not None
    }