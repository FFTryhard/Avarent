"""
store.py
SQLite-backed snapshot history.
One table: snapshots(id, client_id, timestamp, payload, created_at)
"""

import sqlite3
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# DB lives at project root
DB_PATH = Path(__file__).parent.parent / "avarent.db"


class SnapshotStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id  TEXT    NOT NULL,
                    timestamp  TEXT    NOT NULL,
                    payload    TEXT    NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_client_ts "
                "ON snapshots(client_id, created_at)"
            )
        logger.debug("SnapshotStore: DB ready at %s", self.db_path)

    def save(self, client_id: str, snapshot: dict) -> None:
        """Persist one snapshot. Called after every engine output."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO snapshots (client_id, timestamp, payload) VALUES (?, ?, ?)",
                (client_id, snapshot.get("timestamp", ""), json.dumps(snapshot)),
            )

    def get_latest(self, client_id: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload FROM snapshots "
                "WHERE client_id=? ORDER BY created_at DESC LIMIT 1",
                (client_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def get_history(self, client_id: str, limit: int = 90) -> list[dict]:
        """Return up to `limit` snapshots, newest-first."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT payload FROM snapshots "
                "WHERE client_id=? ORDER BY created_at DESC LIMIT ?",
                (client_id, limit),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]
