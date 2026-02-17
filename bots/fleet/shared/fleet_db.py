"""
Fleet Database — SQLite storage for all fleet trades and bot stats.

Tables:
- fleet_trades: Unified trade log across all 19 bots
- fleet_bot_stats: Daily per-bot performance (for Thompson Sampling + dashboard)
- fleet_thompson: Thompson Sampling alpha/beta state
- fleet_risk_state: Daily risk tracking
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger('Fleet.DB')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DB_DIR = os.path.join(_PROJECT_ROOT, 'data', 'fleet')
DB_PATH = os.path.join(DB_DIR, 'fleet_trades.db')


class FleetDB:
    """SQLite database for fleet trading system."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        conn = self._get_conn()
        with conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS fleet_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    bot_name TEXT NOT NULL,
                    bot_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    position_size_usd REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    pnl REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    confidence REAL DEFAULT 0,
                    edge REAL DEFAULT 0,
                    reason TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS fleet_bot_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    trades_count INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    sharpe_ratio REAL,
                    win_rate REAL,
                    avg_edge REAL,
                    capital_allocated REAL DEFAULT 0,
                    UNIQUE(bot_name, date)
                );

                CREATE TABLE IF NOT EXISTS fleet_thompson (
                    bot_name TEXT PRIMARY KEY,
                    alpha REAL DEFAULT 1.0,
                    beta REAL DEFAULT 1.0,
                    total_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    last_sample REAL,
                    last_allocation REAL,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS fleet_risk_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    daily_pnl REAL DEFAULT 0,
                    daily_trades INTEGER DEFAULT 0,
                    open_positions INTEGER DEFAULT 0,
                    total_exposure REAL DEFAULT 0,
                    max_drawdown_today REAL DEFAULT 0,
                    risk_blocked INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_fleet_trades_bot ON fleet_trades(bot_name);
                CREATE INDEX IF NOT EXISTS idx_fleet_trades_status ON fleet_trades(status);
                CREATE INDEX IF NOT EXISTS idx_fleet_trades_symbol ON fleet_trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_fleet_trades_entry_time ON fleet_trades(entry_time);
            ''')
        conn.close()
        logger.info(f"Fleet DB initialized at {self.db_path}")

    # =========================================================================
    # TRADE OPERATIONS
    # =========================================================================

    def log_trade(self, signal_dict: Dict[str, Any]) -> bool:
        """Log a new trade from a FleetSignal dict."""
        try:
            conn = self._get_conn()
            with conn:
                conn.execute('''
                    INSERT OR IGNORE INTO fleet_trades
                    (trade_id, bot_name, bot_type, symbol, side, entry_price,
                     quantity, position_size_usd, entry_time, confidence, edge, reason, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_dict['trade_id'],
                    signal_dict['bot_name'],
                    signal_dict['bot_type'],
                    signal_dict['symbol'],
                    signal_dict['side'],
                    signal_dict['entry_price'],
                    signal_dict.get('quantity', 0),
                    signal_dict.get('position_size_usd', 0),
                    signal_dict.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    signal_dict.get('confidence', 0),
                    signal_dict.get('edge', 0),
                    signal_dict.get('reason', ''),
                    json.dumps(signal_dict.get('metadata', {})),
                ))
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            return False

    def close_trade(self, trade_id: str, exit_price: float, pnl: float, pnl_pct: float) -> bool:
        """Close an open trade."""
        try:
            conn = self._get_conn()
            with conn:
                conn.execute('''
                    UPDATE fleet_trades
                    SET exit_price = ?, exit_time = ?, pnl = ?, pnl_pct = ?, status = 'closed'
                    WHERE trade_id = ? AND status = 'open'
                ''', (exit_price, datetime.now(timezone.utc).isoformat(), pnl, pnl_pct, trade_id))
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
            return False

    def get_open_positions(self, bot_name: str = None) -> List[Dict]:
        """Get open positions, optionally filtered by bot."""
        conn = self._get_conn()
        if bot_name:
            rows = conn.execute(
                "SELECT * FROM fleet_trades WHERE status = 'open' AND bot_name = ? ORDER BY entry_time DESC",
                (bot_name,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM fleet_trades WHERE status = 'open' ORDER BY entry_time DESC"
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_recent_trades(self, bot_name: str = None, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get recent trades with pagination."""
        conn = self._get_conn()
        if bot_name:
            rows = conn.execute(
                "SELECT * FROM fleet_trades WHERE bot_name = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (bot_name, limit, offset)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM fleet_trades ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_open_position_count(self) -> int:
        conn = self._get_conn()
        count = conn.execute("SELECT COUNT(*) FROM fleet_trades WHERE status = 'open'").fetchone()[0]
        conn.close()
        return count

    def get_total_exposure(self) -> float:
        conn = self._get_conn()
        result = conn.execute(
            "SELECT COALESCE(SUM(position_size_usd), 0) FROM fleet_trades WHERE status = 'open'"
        ).fetchone()[0]
        conn.close()
        return result

    def has_open_position(self, symbol: str, bot_name: str = None) -> bool:
        conn = self._get_conn()
        if bot_name:
            row = conn.execute(
                "SELECT 1 FROM fleet_trades WHERE symbol = ? AND bot_name = ? AND status = 'open' LIMIT 1",
                (symbol, bot_name)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT 1 FROM fleet_trades WHERE symbol = ? AND status = 'open' LIMIT 1",
                (symbol,)
            ).fetchone()
        conn.close()
        return row is not None

    def get_today_trades(self, bot_name: str = None) -> List[Dict]:
        """Get all trades from today."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn = self._get_conn()
        if bot_name:
            rows = conn.execute(
                "SELECT * FROM fleet_trades WHERE bot_name = ? AND entry_time >= ? ORDER BY entry_time DESC",
                (bot_name, today)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM fleet_trades WHERE entry_time >= ? ORDER BY entry_time DESC",
                (today,)
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_today_pnl(self) -> float:
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn = self._get_conn()
        result = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM fleet_trades WHERE entry_time >= ? AND status = 'closed'",
            (today,)
        ).fetchone()[0]
        conn.close()
        return result

    # =========================================================================
    # BOT STATS
    # =========================================================================

    def update_bot_stats(self, bot_name: str, pnl: float, won: bool, edge: float = 0, capital: float = 0):
        """Update daily bot stats after a trade closes."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn = self._get_conn()
        with conn:
            conn.execute('''
                INSERT INTO fleet_bot_stats (bot_name, date, trades_count, wins, losses, total_pnl, avg_edge, capital_allocated)
                VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(bot_name, date) DO UPDATE SET
                    trades_count = trades_count + 1,
                    wins = wins + ?,
                    losses = losses + ?,
                    total_pnl = total_pnl + ?,
                    avg_edge = (avg_edge * trades_count + ?) / (trades_count + 1),
                    capital_allocated = MAX(capital_allocated, ?)
            ''', (
                bot_name, today, int(won), int(not won), pnl, edge, capital,
                int(won), int(not won), pnl, edge, capital,
            ))
        conn.close()

    def get_bot_performance(self, bot_name: str = None) -> List[Dict]:
        """Get performance stats, optionally filtered by bot."""
        conn = self._get_conn()
        if bot_name:
            rows = conn.execute(
                "SELECT * FROM fleet_bot_stats WHERE bot_name = ? ORDER BY date DESC LIMIT 30",
                (bot_name,)
            ).fetchall()
        else:
            rows = conn.execute('''
                SELECT bot_name,
                       SUM(trades_count) as total_trades,
                       SUM(wins) as total_wins,
                       SUM(losses) as total_losses,
                       SUM(total_pnl) as total_pnl,
                       AVG(avg_edge) as avg_edge,
                       CASE WHEN SUM(trades_count) > 0
                            THEN ROUND(CAST(SUM(wins) AS REAL) / SUM(trades_count), 4)
                            ELSE 0 END as win_rate
                FROM fleet_bot_stats
                GROUP BY bot_name
                ORDER BY total_pnl DESC
            ''').fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # =========================================================================
    # THOMPSON SAMPLING
    # =========================================================================

    def get_thompson_state(self, bot_name: str) -> Dict:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM fleet_thompson WHERE bot_name = ?", (bot_name,)).fetchone()
        conn.close()
        if row:
            return dict(row)
        return {'bot_name': bot_name, 'alpha': 1.0, 'beta': 1.0, 'total_trades': 0, 'total_pnl': 0}

    def update_thompson(self, bot_name: str, won: bool, pnl: float):
        conn = self._get_conn()
        with conn:
            conn.execute('''
                INSERT INTO fleet_thompson (bot_name, alpha, beta, total_trades, total_pnl)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(bot_name) DO UPDATE SET
                    alpha = alpha + ?,
                    beta = beta + ?,
                    total_trades = total_trades + 1,
                    total_pnl = total_pnl + ?,
                    updated_at = datetime('now')
            ''', (
                bot_name,
                1.0 + (1.0 if won else 0.0),
                1.0 + (0.0 if won else 1.0),
                pnl,
                1.0 if won else 0.0,
                0.0 if won else 1.0,
                pnl,
            ))
        conn.close()

    def get_all_thompson_states(self) -> List[Dict]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM fleet_thompson ORDER BY bot_name").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # =========================================================================
    # RISK STATE
    # =========================================================================

    def get_risk_state(self) -> Dict:
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM fleet_risk_state WHERE date = ?", (today,)).fetchone()
        conn.close()
        if row:
            return dict(row)
        return {
            'date': today, 'daily_pnl': 0, 'daily_trades': 0,
            'open_positions': 0, 'total_exposure': 0,
            'max_drawdown_today': 0, 'risk_blocked': 0,
        }

    def update_risk_state(self, **kwargs):
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn = self._get_conn()
        with conn:
            conn.execute('''
                INSERT INTO fleet_risk_state (date, daily_pnl, daily_trades, open_positions, total_exposure, risk_blocked)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    daily_pnl = ?,
                    daily_trades = ?,
                    open_positions = ?,
                    total_exposure = ?,
                    risk_blocked = risk_blocked + ?,
                    updated_at = datetime('now')
            ''', (
                today,
                kwargs.get('daily_pnl', 0),
                kwargs.get('daily_trades', 0),
                kwargs.get('open_positions', 0),
                kwargs.get('total_exposure', 0),
                kwargs.get('risk_blocked', 0),
                kwargs.get('daily_pnl', 0),
                kwargs.get('daily_trades', 0),
                kwargs.get('open_positions', 0),
                kwargs.get('total_exposure', 0),
                kwargs.get('risk_blocked', 0),
            ))
        conn.close()
