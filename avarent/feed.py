"""
feed.py
Market data ingestion layer.

Two concrete implementations:
  CSVDataFeed   — row-by-row replay from a static CSV (dev / testing)
  YFinanceFeed  — live daily EOD feed via yfinance + APScheduler (production)

Both implement the MarketDataFeed ABC so EngineService is decoupled from the source.
"""
#Purely mode A, i can choose what type of feed it is

from abc import ABC, abstractmethod
from typing import Callable, List
import threading
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketDataFeed(ABC):
    """
    Abstract interface for market data ingestion.
    When new data arrives, call: ingest_callback(timestamp_str, [val1, val2, ...])
    The list must follow the exact order of self.expected_headers.
    """

    def __init__(self, expected_headers: List[str]):
        self.expected_headers = expected_headers
        self.is_running = False

    @abstractmethod
    def start(self, ingest_callback: Callable[[str, List[float]], None]) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
# CSV Feed (dev / testing)
# ---------------------------------------------------------------------------

class CSVDataFeed(MarketDataFeed):
    """
    Reads from a static CSV file and plays it back row by row.

    playback_delay_sec:
      0.0  → instant warmup (testing)
      1.0  → one row per second (live demo feel)
    """

    def __init__(
        self,
        expected_headers: List[str],
        csv_path: str,
        playback_delay_sec: float = 0.0,
    ):
        super().__init__(expected_headers)
        self.csv_path = csv_path
        self.playback_delay_sec = playback_delay_sec
        self._thread: threading.Thread | None = None

    def start(self, ingest_callback: Callable[[str, List[float]], None]) -> None:
        self.is_running = True

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            logger.error("CSVDataFeed: failed to load %s: %s", self.csv_path, e)
            self.is_running = False
            return

        logger.info("CSVDataFeed: %d rows from %s", len(df), self.csv_path)
        logger.info("CSVDataFeed: playback speed = %ss/row", self.playback_delay_sec)

        def _run():
            try:
                for i in range(len(df)):
                    if not self.is_running:
                        break
                    timestamp = df.iloc[i, 0]
                    row = df.iloc[i, 1:].values.tolist()
                    ingest_callback(str(timestamp), row)
                    if self.playback_delay_sec > 0:
                        import time
                        time.sleep(self.playback_delay_sec)
                logger.info("CSVDataFeed: playback complete.")
            except Exception as e:
                logger.error("CSVDataFeed thread error: %s", e, exc_info=True)
            finally:
                self.is_running = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# YFinance Feed (production — daily EOD)
# ---------------------------------------------------------------------------

class YFinanceFeed(MarketDataFeed):
    """
    Live daily feed via yfinance.

    On start():
      1. Pulls `warmup_years` years of historical daily closes.
      2. Feeds all historical rows immediately (fast warmup, no delay).
      3. Schedules a daily APScheduler job at 16:15 EST to pull the latest close.

    Log returns are computed as ln(close_t / close_{t-1}).
    Column order is locked to expected_headers on every pull.
    """

    def __init__(
        self,
        expected_headers: List[str],
        warmup_years: int = 2,
    ):
        super().__init__(expected_headers)
        self.warmup_years = warmup_years
        self._ingest_callback: Callable | None = None
        self._warmup_thread: threading.Thread | None = None

    # ------------------------------------------------------------------

    def start(self, ingest_callback: Callable[[str, List[float]], None]) -> None:
        self.is_running = True
        self._ingest_callback = ingest_callback

        # Kick off historical warmup in background (non-blocking)
        self._warmup_thread = threading.Thread(
            target=self._run_warmup, daemon=True
        )
        self._warmup_thread.start()

    def stop(self) -> None:
        self.is_running = False
        from avarent.scheduler import stop_scheduler
        stop_scheduler()

    # ------------------------------------------------------------------

    def _run_warmup(self) -> None:
        """Pull historical data and feed all rows, then register the daily job."""
        try:
            import yfinance as yf

            logger.info(
                "YFinanceFeed: downloading %dy history for %s",
                self.warmup_years,
                self.expected_headers,
            )
            df = yf.download(
                self.expected_headers,
                period=f"{self.warmup_years}y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )["Close"]

            # Lock column order to expected_headers
            df = df[self.expected_headers]

            # Daily log returns — drops the first NaN row
            log_ret = np.log(df / df.shift(1)).dropna()

            logger.info("YFinanceFeed: feeding %d warmup rows...", len(log_ret))

            for idx, row_data in log_ret.iterrows():
                if not self.is_running:
                    break
                self._ingest_callback(str(idx.date()), row_data.tolist())

            logger.info("YFinanceFeed: warmup complete.")

        except Exception as e:
            logger.error("YFinanceFeed warmup failed: %s", e, exc_info=True)
            self.is_running = False
            return

        # Register the recurring daily job only after warmup succeeds
        try:
            from avarent.scheduler import start_scheduler, add_daily_job
            start_scheduler()
            add_daily_job(self._pull_and_ingest_latest, job_id="daily_yfinance")
        except Exception as e:
            logger.error("YFinanceFeed: failed to register daily job: %s", e)

    def _pull_and_ingest_latest(self) -> None:
        """Called by APScheduler every weekday at 16:15 EST."""
        try:
            import yfinance as yf

            # Pull last 5 days to ensure we always have at least 2 close prices
            df = yf.download(
                self.expected_headers,
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )["Close"][self.expected_headers].dropna()

            if len(df) < 2:
                logger.warning("YFinanceFeed: not enough rows for daily log return")
                return

            latest   = df.iloc[-1]
            previous = df.iloc[-2]
            log_ret  = np.log(latest / previous).tolist()
            timestamp = str(df.index[-1].date())

            self._ingest_callback(timestamp, log_ret)
            logger.info("YFinanceFeed: daily ingest done for %s", timestamp)

        except Exception as e:
            logger.error("YFinanceFeed daily pull failed: %s", e, exc_info=True)
