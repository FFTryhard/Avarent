"""
scheduler.py
Thin APScheduler wrapper for the daily yfinance pull (Mode A).
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def get_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(timezone="America/New_York")
    return _scheduler


def start_scheduler() -> None:
    s = get_scheduler()
    if not s.running:
        s.start()
        logger.info("Scheduler: started (tz=America/New_York)")


def stop_scheduler() -> None:
    s = get_scheduler()
    if s.running:
        s.shutdown(wait=False)
        logger.info("Scheduler: stopped")


def add_daily_job(
    func,
    hour: int = 16,
    minute: int = 15,
    job_id: str = "daily_yfinance",
) -> None:
    """Register `func` to run every weekday at HH:MM Eastern."""
    s = get_scheduler()
    s.add_job(
        func,
        CronTrigger(day_of_week="mon-fri", hour=hour, minute=minute),
        id=job_id,
        replace_existing=True,
        misfire_grace_time=300,
    )
    logger.info("Scheduler: daily job '%s' at %02d:%02d EST", job_id, hour, minute)
