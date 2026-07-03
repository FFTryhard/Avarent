"""
main.py
FastAPI application — Avarent Regime API v2

Endpoints
---------
GET  /                      health ping
GET  /v1/health             engine + feed status
GET  /v1/latest             Mode A hosted signal (or BYOD if client has one)
GET  /v1/history            Mode A hosted signal history
POST /v1/ingest             Mode B — push your own data row
GET  /v1/client/latest      Mode B — latest snapshot for caller's engine
GET  /v1/client/history     Mode B — history for caller's engine

Environment variables
---------------------
FEED_MODE       "csv" (default) | "live"
CSV_PATH        path to the ETF log-returns CSV (used when FEED_MODE=csv)
PLAYBACK_DELAY  seconds per row for CSV feed (default 0.0 = instant warmup)
CPP_BUILD_PATH  path to compiled mga_module (default: cpp/build)
API_KEYS        comma-separated keys, optionally key:tier
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from slowapi.errors import RateLimitExceeded
# pyrefly: ignore [missing-import]
from slowapi import _rate_limit_exceeded_handler

from avarent.service import EngineService, ClientEngineRegistry
from avarent.feed import CSVDataFeed, YFinanceFeed
from avarent.auth import verify_key
from avarent.ratelimit import limiter, _dynamic_limit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HEADERS = ["GLD", "HYG", "IWM", "QQQ", "SPY", "TLT", "XLE", "XLF", "XLK", "XLV"]

FEED_MODE = os.getenv("FEED_MODE", "csv")
CSV_PATH  = os.getenv(
    "CSV_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "etf_daily_log_returns.csv"),
)
PLAYBACK_DELAY = float(os.getenv("PLAYBACK_DELAY", "0.0"))

# ---------------------------------------------------------------------------
# Build the hosted data feed (Mode A)
# ---------------------------------------------------------------------------

if FEED_MODE == "live":
    _feed = YFinanceFeed(expected_headers=HEADERS)
    logger.info("main: using YFinanceFeed (live daily EOD)")
else:
    _feed = CSVDataFeed(
        expected_headers=HEADERS,
        csv_path=CSV_PATH,
        playback_delay_sec=PLAYBACK_DELAY,
    )
    logger.info("main: using CSVDataFeed (%s)", CSV_PATH)

# ---------------------------------------------------------------------------
# Engine instances
# ---------------------------------------------------------------------------

hosted_service  = EngineService(headers=HEADERS, client_id="hosted", data_feed=_feed)
client_registry = ClientEngineRegistry()

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    hosted_service.start()
    yield
    hosted_service.stop()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Avarent Regime API",
    version="2.0.0",
    description=(
        "Real-time market regime detection. "
        "Mode A: hosted ETF signal. "
        "Mode B: bring your own data."
    ),
    lifespan=lifespan,
)

# Attach rate limiter — must happen right after app creation
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root():
    return {"status": "online", "version": "2.0.0"}


@app.get("/v1/health", tags=["meta"])
def health():
    return {
        "server":         "ok",
        "feed_mode":      FEED_MODE,
        "engine_running": hosted_service.running,
        "has_output":     hosted_service.get_latest() is not None,
    }

# ---------------------------------------------------------------------------
# Mode A — Hosted signal (10-ETF universe, updated daily)
# ---------------------------------------------------------------------------

@app.get("/v1/latest", tags=["hosted"])
@limiter.limit(_dynamic_limit)
def latest(request: Request, client: dict = Depends(verify_key)):
    """
    Returns the most recent regime snapshot.
    If the caller has a BYOD engine, that snapshot is returned instead.
    Falls back to the hosted signal otherwise.
    """
    # Check if this client has a BYOD engine with output
    byod = client_registry.get(client["client_id"])
    if byod:
        data = byod.get_latest()
        if data:
            return {**data, "source": "byod"}

    # Hosted signal
    data = hosted_service.get_latest()
    if data is None:
        return {"status": "warming_up"}
    return {**data, "source": "hosted"}


@app.get("/v1/history", tags=["hosted"])
@limiter.limit(_dynamic_limit)
def history(
    request: Request,
    client: dict = Depends(verify_key),
    days: int = Query(default=90, ge=1, le=365, description="Number of snapshots to return"),
):
    """Returns up to `days` historical snapshots for the hosted signal, newest-first."""
    return hosted_service.get_history(limit=days)

# ---------------------------------------------------------------------------
# Mode B — BYOD (client supplies their own asset universe + data rows)
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    timestamp: str
    headers:   list[str]
    values:    list[float]


@app.post("/v1/ingest", tags=["byod"])
@limiter.limit(_dynamic_limit)
def ingest(request: Request, body: IngestRequest, client: dict = Depends(verify_key)):
    """
    Push one row of log returns for your own asset universe.

    Rules:
    - `headers` and `values` must be the same length.
    - `headers` are locked after the first call — send the same list every time.
    - Maximum 50 assets.
    - Engine requires 250 rows before producing output (warmup).

    Returns the latest regime snapshot once the engine is warm, otherwise
    returns {"status": "warming_up"} with a progress indicator.
    """
    if len(body.headers) != len(body.values):
        raise HTTPException(status_code=422, detail="headers and values must have the same length")

    try:
        engine = client_registry.get_or_create(client["client_id"], body.headers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    engine.ingest_row(body.timestamp, body.values)

    data = engine.get_latest()
    if data is None:
        return {
            "status":  "warming_up",
            "message": "Engine needs 250 rows before producing output. Keep ingesting.",
        }
    return {**data, "source": "byod"}


@app.get("/v1/client/latest", tags=["byod"])
@limiter.limit(_dynamic_limit)
def client_latest(request: Request, client: dict = Depends(verify_key)):
    """Returns the latest snapshot for this client's BYOD engine."""
    engine = client_registry.get(client["client_id"])
    if engine is None:
        raise HTTPException(
            status_code=404,
            detail="No BYOD engine found. POST rows to /v1/ingest first.",
        )
    data = engine.get_latest()
    if data is None:
        return {"status": "warming_up"}
    return {**data, "source": "byod"}


@app.get("/v1/client/history", tags=["byod"])
@limiter.limit(_dynamic_limit)
def client_history(
    request: Request,
    client: dict = Depends(verify_key),
    days: int = Query(default=90, ge=1, le=365),
):
    """Returns up to `days` historical snapshots for this client's BYOD engine, newest-first."""
    engine = client_registry.get(client["client_id"])
    if engine is None:
        raise HTTPException(
            status_code=404,
            detail="No BYOD engine found. POST rows to /v1/ingest first.",
        )
    return engine.get_history(limit=days)
