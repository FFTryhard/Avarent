"""
service.py
Engine orchestration layer.

EngineService      — wraps the C++ engine, handles ingestion, stores history.
ClientEngineRegistry — manages isolated per-client BYOD engine instances.

Key changes from v1:
  - window_stride overridden to 1 (daily signal update after warmup)
  - Payload formatting delegated to regimes.format_snapshot()
  - Every snapshot persisted to SQLite via SnapshotStore
  - No longer coupled to MarketDataFeed: feed is optional (None for BYOD clients)
"""

import threading
import sys
import os
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Locate and import the compiled C++ pybind11 module
cpp_build_path = os.getenv(
    "CPP_BUILD_PATH",
    os.path.join(os.path.dirname(__file__), "..", "cpp", "build"),
)
sys.path.append(os.path.abspath(cpp_build_path))

# pyrefly: ignore [missing-import]
import mga_module  # noqa: E402 — must come after sys.path update

from avarent.feed import MarketDataFeed
from avarent.regimes import format_snapshot
from avarent.store import SnapshotStore


# ---------------------------------------------------------------------------
# EngineService
# ---------------------------------------------------------------------------

class EngineService:
    """
    Wraps one C++ Engine instance.

    Used in two modes:
      Mode A (hosted)  — data_feed is provided; feed calls ingest_row on each tick.     
      Mode B (BYOD)    — data_feed is None; API endpoint calls ingest_row directly.
    """

    def __init__(
        self,
        headers: list[str],
        client_id: str = "hosted",
        data_feed: MarketDataFeed | None = None,
        profile=mga_module.EngineProfile.LONG_SHORT_EQUITY,
    ):
        self.lock      = threading.Lock()
        self.client_id = client_id
        self.data_feed = data_feed
        self.headers   = list(headers)
        self.store     = SnapshotStore()

        # Build config from profile then override stride → 1 for daily signal
        config = mga_module.profile_defaults(profile)
        config.window_stride = 1  # produce output on every row after warmup

        self.engine = mga_module.Engine(config)
        self.engine.setHeaders(self.headers)

        self.latest_output: dict | None = None

    # ------------------------------------------------------------------
    # Lifecycle (only relevant for Mode A / hosted)
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.data_feed.is_running if self.data_feed else True

    def start(self) -> None:
        """Start the data feed (Mode A only)."""
        if self.data_feed:
            self.data_feed.start(self.ingest_row)

    def stop(self) -> None:
        """Stop the data feed (Mode A only)."""
        if self.data_feed:
            self.data_feed.stop()

    # ------------------------------------------------------------------
    # Core ingestion
    # ------------------------------------------------------------------

    def ingest_row(self, timestamp: str, row: list[float]) -> None:
        """
        Feed one row into the engine. Thread-safe.
        When the engine produces output, format and persist the snapshot.
        """
        with self.lock:
            self.engine.ingest(row)

            if self.engine.ready():
                out      = self.engine.get_snapshot()
                snapshot = format_snapshot(timestamp, out)

                self.latest_output = snapshot
                self.store.save(self.client_id, snapshot)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_latest(self) -> dict | None:
        with self.lock:
            return self.latest_output

    def get_history(self, limit: int = 90) -> list[dict]:
        return self.store.get_history(self.client_id, limit)


# ---------------------------------------------------------------------------
# ClientEngineRegistry  (Mode B — BYOD)
# ---------------------------------------------------------------------------

class ClientEngineRegistry:
    """
    Maintains one isolated EngineService per BYOD client.

    Engines are created on the first POST /v1/ingest call.
    Headers are locked at creation — a different universe requires a new API key.
    Hard cap: 50 assets per client.
    """

    MAX_ASSETS = 50

    def __init__(self):
        self._engines: dict[str, EngineService] = {}
        self._lock = threading.Lock()

    def get_or_create(self, client_id: str, headers: list[str]) -> EngineService:
        """Return the client's engine, creating it on first call."""
        with self._lock:
            if client_id not in self._engines:
                if len(headers) > self.MAX_ASSETS:
                    raise ValueError(
                        f"Maximum {self.MAX_ASSETS} assets per client "
                        f"(got {len(headers)})"
                    )
                self._engines[client_id] = EngineService(
                    headers=headers,
                    client_id=f"byod_{client_id}",
                    data_feed=None,
                )
                logger.info(
                    "ClientEngineRegistry: created engine for '%s' (%d assets: %s)",
                    client_id, len(headers), headers,
                )
            else:
                # Validate headers match the initialised engine
                existing_headers = self._engines[client_id].headers
                if existing_headers != list(headers):
                    raise ValueError(
                        f"Header mismatch for client '{client_id}': "
                        f"engine initialised with {existing_headers}, received {list(headers)}. "
                        "Use a different API key for a different asset universe."
                    )

            return self._engines[client_id]

    def get(self, client_id: str) -> EngineService | None:
        """Return an existing engine or None if the client hasn't ingested yet."""
        with self._lock:
            return self._engines.get(client_id)