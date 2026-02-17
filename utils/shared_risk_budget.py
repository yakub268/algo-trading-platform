"""
Shared Risk Budget — Cross-System Kalshi Exposure Tracking
Tracks combined Kalshi exposure across Trading Bot and Event Platform.
Both systems write trades here and check before entering.
"""

import os
import sqlite3
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger('SharedRiskBudget')

DB_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "trading_bot", "data", "live", "shared_risk.db")


class SharedRiskBudget:
    """Cross-system risk coordination for Kalshi markets."""

    def __init__(self, max_daily_loss: float = 50.0, max_total_exposure: float = 200.0):
        self.max_daily_loss = max_daily_loss
        self.max_total_exposure = max_total_exposure
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system TEXT NOT NULL,
                market_id TEXT NOT NULL,
                ticker TEXT,
                side TEXT,
                quantity INTEGER,
                entry_price REAL,
                cost REAL,
                entry_time TEXT,
                status TEXT DEFAULT 'OPEN'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system TEXT NOT NULL,
                date TEXT NOT NULL,
                realized_pnl REAL DEFAULT 0,
                trade_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def has_position(self, market_id: str) -> bool:
        """Check if ANY system has an open position in this market."""
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT COUNT(*) FROM shared_positions WHERE market_id = ? AND status = 'OPEN'",
                (market_id,)
            ).fetchone()
            conn.close()
            return row[0] > 0

    def get_total_exposure(self) -> float:
        """Get total open exposure across all systems."""
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT COALESCE(SUM(cost), 0) FROM shared_positions WHERE status = 'OPEN'"
            ).fetchone()
            conn.close()
            return row[0]

    def get_daily_pnl(self) -> float:
        """Get combined daily PnL across all systems."""
        today = datetime.now().strftime('%Y-%m-%d')
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0) FROM shared_pnl WHERE date = ?",
                (today,)
            ).fetchone()
            conn.close()
            return row[0]

    def can_enter(self, market_id: str, cost: float) -> tuple:
        """Check if a new position is allowed. Returns (allowed, reason)."""
        if self.has_position(market_id):
            return False, f"Position already exists in {market_id}"

        exposure = self.get_total_exposure()
        if exposure + cost > self.max_total_exposure:
            return False, f"Would exceed max exposure: ${exposure + cost:.2f} > ${self.max_total_exposure:.2f}"

        daily_pnl = self.get_daily_pnl()
        if daily_pnl <= -self.max_daily_loss:
            return False, f"Daily loss limit hit: ${daily_pnl:.2f}"

        return True, "OK"

    def record_entry(self, system: str, market_id: str, ticker: str,
                     side: str, quantity: int, entry_price: float):
        """Record a new position entry."""
        cost = quantity * entry_price / 100.0
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO shared_positions (system, market_id, ticker, side, quantity, entry_price, cost, entry_time) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (system, market_id, ticker, side, quantity, entry_price, cost, datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
        logger.info(f"[SharedRisk] {system} entered {market_id}: {side} x{quantity} @ {entry_price}c (${cost:.2f})")

    def record_exit(self, system: str, market_id: str, pnl: float):
        """Record a position exit and update daily PnL."""
        today = datetime.now().strftime('%Y-%m-%d')
        with self._lock:
            conn = self._connect()
            conn.execute(
                "UPDATE shared_positions SET status = 'CLOSED' WHERE system = ? AND market_id = ? AND status = 'OPEN'",
                (system, market_id)
            )
            # Upsert daily PnL
            existing = conn.execute(
                "SELECT id FROM shared_pnl WHERE system = ? AND date = ?",
                (system, today)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE shared_pnl SET realized_pnl = realized_pnl + ?, trade_count = trade_count + 1 WHERE id = ?",
                    (pnl, existing[0])
                )
            else:
                conn.execute(
                    "INSERT INTO shared_pnl (system, date, realized_pnl, trade_count) VALUES (?, ?, ?, 1)",
                    (system, today, pnl)
                )
            conn.commit()
            conn.close()
        logger.info(f"[SharedRisk] {system} exited {market_id}: PnL=${pnl:+.2f}")

    def get_status(self) -> Dict:
        """Get shared risk status summary."""
        return {
            'total_exposure': self.get_total_exposure(),
            'daily_pnl': self.get_daily_pnl(),
            'max_daily_loss': self.max_daily_loss,
            'max_total_exposure': self.max_total_exposure,
        }
