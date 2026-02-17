"""
Kalshi Infrastructure Layer

Shared infrastructure for all Kalshi bots:
- KalshiPositionManager: Persistent position tracking + API reconciliation
- KalshiFillTracker: Real fill polling from Kalshi API
- KalshiSettlementCollector: Tracks settlement results + P&L
- KalshiRiskManager: Daily loss limits, position caps, contradictory position blocker

All Kalshi bots share a single instance of each component, injected by the orchestrator.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import re
import time
import sqlite3
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger('KalshiInfrastructure')


# =============================================================================
# DATABASE
# =============================================================================

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'live', 'kalshi_positions.db'
)


def _get_db(db_path: str = None) -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode for concurrent reads."""
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_schema(db_path: str = None):
    """Create all tables if they don't exist."""
    conn = _get_db(db_path)
    with conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL DEFAULT 0,
                avg_price_cents INTEGER NOT NULL DEFAULT 0,
                market_title TEXT,
                opened_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                UNIQUE(ticker, side)
            );

            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fill_id TEXT UNIQUE,
                order_id TEXT,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                action TEXT NOT NULL,
                price_cents INTEGER NOT NULL,
                count INTEGER NOT NULL,
                created_time TEXT NOT NULL,
                recorded_at TEXT NOT NULL DEFAULT (datetime('now')),
                bot_name TEXT
            );

            CREATE TABLE IF NOT EXISTS settlements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT UNIQUE NOT NULL,
                market_title TEXT,
                result TEXT,
                settled_at TEXT,
                our_side TEXT,
                our_quantity INTEGER,
                our_avg_price_cents INTEGER,
                pnl_cents INTEGER,
                checked_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS daily_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                realized_pnl_cents INTEGER NOT NULL DEFAULT 0,
                num_fills INTEGER NOT NULL DEFAULT 0,
                num_settlements INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                action TEXT NOT NULL,
                price_cents INTEGER NOT NULL,
                count INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                bot_name TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
            CREATE INDEX IF NOT EXISTS idx_fills_ticker ON fills(ticker);
            CREATE INDEX IF NOT EXISTS idx_settlements_ticker ON settlements(ticker);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
        ''')
    conn.close()
    logger.info(f"Kalshi infrastructure DB initialized: {db_path or DB_PATH}")


# =============================================================================
# POSITION MANAGER
# =============================================================================

class KalshiPositionManager:
    """
    Persistent position tracking backed by SQLite.
    Syncs with Kalshi API on startup and provides position queries for all bots.
    """

    def __init__(self, client, db_path: str = None):
        self.client = client
        self.db_path = db_path or DB_PATH
        self._lock = threading.Lock()
        init_schema(self.db_path)

    def sync_from_api(self) -> Dict:
        """
        Sync positions from Kalshi API into local DB.
        Returns stats: {synced, added, updated, removed}
        """
        stats = {'synced': 0, 'added': 0, 'updated': 0, 'removed': 0}

        try:
            api_positions = self.client.get_positions()
        except Exception as e:
            logger.error(f"Failed to fetch positions from Kalshi API: {e}")
            return stats

        if not isinstance(api_positions, list):
            logger.warning(f"Unexpected positions response: {type(api_positions)}")
            return stats

        now = datetime.now(timezone.utc).isoformat()
        api_tickers = set()

        conn = _get_db(self.db_path)
        with conn:
            for pos in api_positions:
                ticker = pos.get('ticker', '')
                if not ticker:
                    continue

                # Kalshi API returns market_positions with yes/no quantities
                yes_qty = pos.get('market_exposure', {}).get('yes', 0) if isinstance(pos.get('market_exposure'), dict) else 0
                no_qty = pos.get('market_exposure', {}).get('no', 0) if isinstance(pos.get('market_exposure'), dict) else 0

                # Fallback: direct position fields
                if yes_qty == 0 and no_qty == 0:
                    yes_qty = pos.get('yes_count', pos.get('position', 0))
                    no_qty = pos.get('no_count', 0)

                for side, qty in [('yes', yes_qty), ('no', no_qty)]:
                    if qty <= 0:
                        continue

                    api_tickers.add((ticker, side))
                    avg_price = pos.get('average_price', 0)
                    # Kalshi may return price as decimal 0-1 or cents 1-99
                    if isinstance(avg_price, float) and 0 < avg_price < 1:
                        avg_price = int(avg_price * 100)

                    existing = conn.execute(
                        "SELECT id, quantity FROM positions WHERE ticker=? AND side=?",
                        (ticker, side)
                    ).fetchone()

                    if existing:
                        if existing['quantity'] != qty:
                            conn.execute(
                                "UPDATE positions SET quantity=?, avg_price_cents=?, updated_at=?, status='open' WHERE id=?",
                                (qty, avg_price, now, existing['id'])
                            )
                            stats['updated'] += 1
                    else:
                        conn.execute(
                            "INSERT INTO positions (ticker, side, quantity, avg_price_cents, opened_at, updated_at, status) VALUES (?,?,?,?,?,?,?)",
                            (ticker, side, qty, avg_price, now, now, 'open')
                        )
                        stats['added'] += 1

                    stats['synced'] += 1

            # Mark positions not in API as closed
            db_open = conn.execute(
                "SELECT id, ticker, side FROM positions WHERE status='open'"
            ).fetchall()

            for row in db_open:
                if (row['ticker'], row['side']) not in api_tickers:
                    conn.execute(
                        "UPDATE positions SET status='closed', updated_at=? WHERE id=?",
                        (now, row['id'])
                    )
                    stats['removed'] += 1

        conn.close()
        logger.info(f"Position sync: {stats}")
        return stats

    def reconcile(self) -> List[str]:
        """
        Compare DB state to API state, log and auto-correct discrepancies.
        Returns list of discrepancy descriptions.
        """
        discrepancies = []
        try:
            api_positions = self.client.get_positions()
        except Exception as e:
            discrepancies.append(f"API fetch failed: {e}")
            return discrepancies

        conn = _get_db(self.db_path)
        db_positions = conn.execute(
            "SELECT ticker, side, quantity FROM positions WHERE status='open'"
        ).fetchall()
        conn.close()

        db_map = {(r['ticker'], r['side']): r['quantity'] for r in db_positions}
        api_map = {}

        for pos in (api_positions if isinstance(api_positions, list) else []):
            ticker = pos.get('ticker', '')
            yes_qty = pos.get('yes_count', pos.get('position', 0))
            no_qty = pos.get('no_count', 0)
            if yes_qty > 0:
                api_map[(ticker, 'yes')] = yes_qty
            if no_qty > 0:
                api_map[(ticker, 'no')] = no_qty

        # Check for mismatches
        all_keys = set(db_map.keys()) | set(api_map.keys())
        for key in all_keys:
            db_qty = db_map.get(key, 0)
            api_qty = api_map.get(key, 0)
            if db_qty != api_qty:
                discrepancies.append(f"{key[0]} {key[1]}: DB={db_qty} API={api_qty}")

        if discrepancies:
            logger.warning(f"Position discrepancies found ({len(discrepancies)}): {discrepancies}")
            # Auto-correct by re-syncing
            self.sync_from_api()
        else:
            logger.info("Position reconciliation: all positions match")

        return discrepancies

    def get_position(self, ticker: str, side: str = None) -> Optional[Dict]:
        """Get position for a ticker (optionally filtered by side)."""
        conn = _get_db(self.db_path)
        if side:
            row = conn.execute(
                "SELECT * FROM positions WHERE ticker=? AND side=? AND status='open'",
                (ticker, side)
            ).fetchone()
            conn.close()
            return dict(row) if row else None
        else:
            rows = conn.execute(
                "SELECT * FROM positions WHERE ticker=? AND status='open'",
                (ticker,)
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows] if rows else None

    def get_all_positions(self) -> List[Dict]:
        """Get all open positions."""
        conn = _get_db(self.db_path)
        rows = conn.execute(
            "SELECT * FROM positions WHERE status='open' ORDER BY ticker"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def update_position(self, ticker: str, side: str, qty_delta: int, price_cents: int, market_title: str = None):
        """
        Update a position after a fill.
        qty_delta: positive for buys, negative for sells.
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = _get_db(self.db_path)

        with conn:
            existing = conn.execute(
                "SELECT id, quantity, avg_price_cents FROM positions WHERE ticker=? AND side=? AND status='open'",
                (ticker, side)
            ).fetchone()

            if existing:
                new_qty = existing['quantity'] + qty_delta
                if new_qty <= 0:
                    conn.execute(
                        "UPDATE positions SET quantity=0, status='closed', updated_at=? WHERE id=?",
                        (now, existing['id'])
                    )
                else:
                    # Weighted average price for buys
                    if qty_delta > 0:
                        old_cost = existing['quantity'] * existing['avg_price_cents']
                        new_cost = qty_delta * price_cents
                        new_avg = (old_cost + new_cost) // new_qty
                    else:
                        new_avg = existing['avg_price_cents']

                    conn.execute(
                        "UPDATE positions SET quantity=?, avg_price_cents=?, updated_at=? WHERE id=?",
                        (new_qty, new_avg, now, existing['id'])
                    )
            elif qty_delta > 0:
                conn.execute(
                    "INSERT INTO positions (ticker, side, quantity, avg_price_cents, market_title, opened_at, updated_at, status) VALUES (?,?,?,?,?,?,?,?)",
                    (ticker, side, qty_delta, price_cents, market_title, now, now, 'open')
                )

        conn.close()

    def get_open_position_count(self) -> int:
        """Get count of unique tickers with open positions."""
        conn = _get_db(self.db_path)
        row = conn.execute(
            "SELECT COUNT(DISTINCT ticker) as cnt FROM positions WHERE status='open' AND quantity > 0"
        ).fetchone()
        conn.close()
        return row['cnt'] if row else 0


# =============================================================================
# FILL TRACKER
# =============================================================================

class KalshiFillTracker:
    """
    Polls Kalshi API for fills and updates position manager.
    Tracks the last fill ID/timestamp to avoid reprocessing.
    """

    def __init__(self, client, position_manager: KalshiPositionManager, db_path: str = None):
        self.client = client
        self.position_manager = position_manager
        self.db_path = db_path or DB_PATH
        self._last_fill_id = self._get_last_fill_id()
        self._lock = threading.Lock()

    def _get_last_fill_id(self) -> Optional[str]:
        """Get the most recent fill_id from DB to avoid reprocessing."""
        conn = _get_db(self.db_path)
        row = conn.execute(
            "SELECT fill_id FROM fills ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return row['fill_id'] if row else None

    def poll_fills(self, bot_name: str = None) -> List[Dict]:
        """
        Poll Kalshi API for new fills and record them.
        Returns list of new fills processed.
        """
        new_fills = []

        try:
            api_fills = self.client.get_fills(limit=100)
        except Exception as e:
            logger.error(f"Failed to fetch fills from Kalshi API: {e}")
            return new_fills

        if not isinstance(api_fills, list):
            return new_fills

        conn = _get_db(self.db_path)
        now = datetime.now(timezone.utc).isoformat()
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        # Collect fills to process, then update positions outside the transaction
        fills_to_process = []

        with conn:
            for fill in api_fills:
                fill_id = fill.get('trade_id', fill.get('fill_id', ''))
                if not fill_id:
                    continue

                # Check if already recorded
                existing = conn.execute(
                    "SELECT id FROM fills WHERE fill_id=?", (fill_id,)
                ).fetchone()
                if existing:
                    continue

                ticker = fill.get('ticker', '')
                side = fill.get('side', '')
                action = fill.get('action', 'buy')
                price_cents = fill.get('yes_price', fill.get('no_price', fill.get('price', 0)))
                count = fill.get('count', 1)
                created_time = fill.get('created_time', now)

                # Record fill
                conn.execute(
                    "INSERT OR IGNORE INTO fills (fill_id, order_id, ticker, side, action, price_cents, count, created_time, recorded_at, bot_name) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (fill_id, fill.get('order_id', ''), ticker, side, action, price_cents, count, created_time, now, bot_name)
                )

                # Update daily P&L tracking
                conn.execute('''
                    INSERT INTO daily_pnl (date, realized_pnl_cents, num_fills)
                    VALUES (?, 0, 1)
                    ON CONFLICT(date) DO UPDATE SET num_fills = num_fills + 1
                ''', (today,))

                fills_to_process.append({
                    'fill_id': fill_id,
                    'ticker': ticker,
                    'side': side,
                    'action': action,
                    'price_cents': price_cents,
                    'count': count,
                })

        conn.close()

        # Update positions AFTER releasing the DB connection (avoids "database is locked")
        for fill_info in fills_to_process:
            qty_delta = fill_info['count'] if fill_info['action'] == 'buy' else -fill_info['count']
            self.position_manager.update_position(
                fill_info['ticker'], fill_info['side'], qty_delta, fill_info['price_cents']
            )
            logger.info(f"New fill: {fill_info['action'].upper()} {fill_info['count']} {fill_info['side'].upper()} {fill_info['ticker']} @ {fill_info['price_cents']}c")

        new_fills = fills_to_process

        if new_fills:
            logger.info(f"Processed {len(new_fills)} new fills")

        return new_fills

    def get_fills_today(self) -> List[Dict]:
        """Get all fills from today."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn = _get_db(self.db_path)
        rows = conn.execute(
            "SELECT * FROM fills WHERE DATE(created_time) = ? ORDER BY created_time DESC",
            (today,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


# =============================================================================
# SETTLEMENT COLLECTOR
# =============================================================================

class KalshiSettlementCollector:
    """
    Tracks settlement results for all tickers with open/closed positions.
    Ported from super_bowl/settlement_collector.py, adapted for trading_bot.
    """

    API_DELAY = 0.15  # seconds between API calls

    def __init__(self, client, position_manager: KalshiPositionManager, db_path: str = None):
        self.client = client
        self.position_manager = position_manager
        self.db_path = db_path or DB_PATH

    def collect(self, limit: int = 0) -> Dict:
        """
        Check all tickers with positions for settlement.
        Returns stats: {checked, settled, errors}
        """
        stats = {'checked': 0, 'settled': 0, 'errors': 0}
        conn = _get_db(self.db_path)

        # Get all tickers that we have/had positions in, minus already-settled
        rows = conn.execute('''
            SELECT DISTINCT p.ticker, p.side, p.quantity, p.avg_price_cents
            FROM positions p
            LEFT JOIN settlements s ON p.ticker = s.ticker
            WHERE s.id IS NULL
        ''').fetchall()

        if limit > 0:
            rows = rows[:limit]

        conn.close()

        if not rows:
            logger.debug("No unsettled positions to check")
            return stats

        logger.info(f"Checking {len(rows)} tickers for settlement...")
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        for row in rows:
            ticker = row['ticker']
            try:
                market = self.client.get_market(ticker)
                status = (market.get('status') or '').lower()
                stats['checked'] += 1

                if status in ('settled', 'closed', 'finalized'):
                    result = (market.get('result') or '').lower()
                    if result not in ('yes', 'no'):
                        result = 'voided'

                    # Calculate P&L
                    pnl_cents = self._calculate_pnl(
                        result, row['side'], row['quantity'], row['avg_price_cents']
                    )

                    conn2 = _get_db(self.db_path)
                    with conn2:
                        conn2.execute('''
                            INSERT OR REPLACE INTO settlements
                            (ticker, market_title, result, settled_at, our_side, our_quantity, our_avg_price_cents, pnl_cents, checked_at)
                            VALUES (?,?,?,?,?,?,?,?,?)
                        ''', (
                            ticker,
                            market.get('title', ''),
                            result,
                            market.get('close_time', datetime.now(timezone.utc).isoformat()),
                            row['side'],
                            row['quantity'],
                            row['avg_price_cents'],
                            pnl_cents,
                            datetime.now(timezone.utc).isoformat()
                        ))

                        # Update daily P&L
                        conn2.execute('''
                            INSERT INTO daily_pnl (date, realized_pnl_cents, num_settlements)
                            VALUES (?, ?, 1)
                            ON CONFLICT(date) DO UPDATE SET
                                realized_pnl_cents = realized_pnl_cents + ?,
                                num_settlements = num_settlements + 1
                        ''', (today, pnl_cents, pnl_cents))

                    conn2.close()
                    stats['settled'] += 1

                    pnl_dollars = pnl_cents / 100
                    logger.info(f"Settlement: {ticker} → {result} | Our: {row['side']} x{row['quantity']} @ {row['avg_price_cents']}c | P&L: ${pnl_dollars:+.2f}")

                time.sleep(self.API_DELAY)

            except Exception as e:
                logger.warning(f"Error checking {ticker}: {e}")
                stats['errors'] += 1

        logger.info(f"Settlement collection: {stats}")
        return stats

    def _calculate_pnl(self, result: str, our_side: str, quantity: int, avg_price_cents: int) -> int:
        """
        Calculate P&L in cents for a settled position.

        If result matches our side: we win (100 - price) * qty
        If result doesn't match: we lose price * qty
        Voided: 0 P&L (money returned)
        """
        if result == 'voided':
            return 0

        if result == our_side:
            # Won: payout is 100c per contract, we paid avg_price
            return (100 - avg_price_cents) * quantity
        else:
            # Lost: we lose what we paid
            return -avg_price_cents * quantity

    def get_settlement_stats(self) -> Dict:
        """Get settlement statistics."""
        conn = _get_db(self.db_path)
        total = conn.execute("SELECT COUNT(*) as cnt FROM settlements").fetchone()['cnt']
        wins = conn.execute("SELECT COUNT(*) as cnt FROM settlements WHERE pnl_cents > 0").fetchone()['cnt']
        losses = conn.execute("SELECT COUNT(*) as cnt FROM settlements WHERE pnl_cents < 0").fetchone()['cnt']
        total_pnl = conn.execute("SELECT COALESCE(SUM(pnl_cents), 0) as total FROM settlements").fetchone()['total']
        conn.close()

        return {
            'total_settled': total,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / total if total > 0 else 0,
            'total_pnl_cents': total_pnl,
            'total_pnl_dollars': total_pnl / 100,
        }


# =============================================================================
# RISK MANAGER
# =============================================================================

class KalshiRiskManager:
    """
    Kalshi-specific risk controls:
    - Daily loss limit
    - Max open positions
    - Max position per market
    - Contradictory position blocker
    """

    def __init__(
        self,
        position_manager: KalshiPositionManager,
        db_path: str = None,
        daily_loss_limit_cents: int = 2000,  # $20
        max_open_positions: int = 12,
        max_contracts_per_market: int = 25,
    ):
        self.position_manager = position_manager
        self.db_path = db_path or DB_PATH
        self.daily_loss_limit_cents = daily_loss_limit_cents
        self.max_open_positions = max_open_positions
        self.max_contracts_per_market = max_contracts_per_market
        self._paused = False
        self._lock = threading.Lock()

    def check_trade_allowed(self, ticker: str, side: str, count: int) -> Tuple[bool, str]:
        """
        Check if a trade is allowed by all risk rules.
        Returns (allowed, reason).
        """
        if self._paused:
            return False, "Trading paused by risk manager"

        # Check daily loss limit
        allowed, reason = self._check_daily_loss()
        if not allowed:
            return False, reason

        # Check max open positions
        allowed, reason = self._check_max_positions()
        if not allowed:
            return False, reason

        # Check max contracts per market
        allowed, reason = self._check_market_limit(ticker, side, count)
        if not allowed:
            return False, reason

        # Check contradictory positions
        allowed, reason = self._check_contradictory(ticker, side)
        if not allowed:
            return False, reason

        return True, "OK"

    def _check_daily_loss(self) -> Tuple[bool, str]:
        """Check if daily loss limit has been hit."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn = _get_db(self.db_path)
        row = conn.execute(
            "SELECT realized_pnl_cents FROM daily_pnl WHERE date=?", (today,)
        ).fetchone()
        conn.close()

        if row and row['realized_pnl_cents'] <= -self.daily_loss_limit_cents:
            return False, f"Daily loss limit hit: ${row['realized_pnl_cents']/100:.2f} (limit: ${self.daily_loss_limit_cents/100:.2f})"

        return True, "OK"

    def _check_max_positions(self) -> Tuple[bool, str]:
        """Check if max open positions limit has been hit."""
        count = self.position_manager.get_open_position_count()
        if count >= self.max_open_positions:
            return False, f"Max open positions reached: {count}/{self.max_open_positions}"
        return True, "OK"

    def _check_market_limit(self, ticker: str, side: str, count: int) -> Tuple[bool, str]:
        """Check if adding count contracts would exceed per-market limit."""
        pos = self.position_manager.get_position(ticker, side)
        current_qty = pos['quantity'] if pos else 0

        if current_qty + count > self.max_contracts_per_market:
            return False, f"Per-market limit: {ticker} {side} would be {current_qty + count}/{self.max_contracts_per_market}"
        return True, "OK"

    def _check_contradictory(self, ticker: str, side: str) -> Tuple[bool, str]:
        """
        Block trades that would create contradictory positions on the same underlying.
        Ported from super_bowl's _has_conflicting_position logic.
        """
        underlying = self._extract_underlying(ticker)
        if not underlying:
            return True, "OK"  # Can't parse — allow

        opposite = 'no' if side == 'yes' else 'yes'

        # Check if we have a position on the same underlying but opposite side
        # or same underlying different threshold
        conn = _get_db(self.db_path)
        rows = conn.execute(
            "SELECT ticker, side, quantity FROM positions WHERE status='open' AND quantity > 0 AND ticker LIKE ?",
            (f"{underlying}%",)
        ).fetchall()
        conn.close()

        for row in rows:
            if row['ticker'] == ticker and row['side'] == opposite and row['quantity'] > 0:
                return False, f"CONFLICT: Already have {opposite.upper()} position on {ticker}"

            # Different threshold on same underlying
            if row['ticker'] != ticker:
                return False, f"CONFLICT: Position on {row['ticker']} ({row['side']}) conflicts with {ticker} ({side})"

        return True, "OK"

    @staticmethod
    def _extract_underlying(ticker: str) -> Optional[str]:
        """
        Extract the underlying event from a Kalshi ticker.
        E.g., KXHIGHCHI-26FEB09-B38.5 → KXHIGHCHI-26FEB09
        Handles B/T prefixes, negative thresholds, decimal-only.
        """
        # Pattern: SERIES-DATE-THRESHOLD
        match = re.match(r'^([A-Z0-9]+-\d{2}[A-Z]{3}\d{2})', ticker)
        if match:
            return match.group(1)

        # Fallback: strip last segment after final dash if it starts with B/T or is numeric
        parts = ticker.rsplit('-', 1)
        if len(parts) == 2 and (parts[1][0] in ('B', 'T') or parts[1].replace('.', '').replace('-', '').isdigit()):
            return parts[0]

        return None

    def pause(self):
        """Pause all Kalshi trading."""
        with self._lock:
            self._paused = True
        logger.warning("Kalshi trading PAUSED by risk manager")

    def resume(self):
        """Resume Kalshi trading."""
        with self._lock:
            self._paused = False
        logger.info("Kalshi trading RESUMED")

    def get_daily_pnl_cents(self) -> int:
        """Get today's realized P&L in cents."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn = _get_db(self.db_path)
        row = conn.execute(
            "SELECT realized_pnl_cents FROM daily_pnl WHERE date=?", (today,)
        ).fetchone()
        conn.close()
        return row['realized_pnl_cents'] if row else 0

    def get_status(self) -> Dict:
        """Get risk manager status summary."""
        open_positions = self.position_manager.get_open_position_count()
        daily_pnl = self.get_daily_pnl_cents()

        return {
            'paused': self._paused,
            'open_positions': open_positions,
            'max_positions': self.max_open_positions,
            'daily_pnl_cents': daily_pnl,
            'daily_pnl_dollars': daily_pnl / 100,
            'daily_loss_limit_dollars': self.daily_loss_limit_cents / 100,
            'max_contracts_per_market': self.max_contracts_per_market,
        }


# =============================================================================
# INFRASTRUCTURE BUNDLE
# =============================================================================

class KalshiInfrastructure:
    """
    Convenience class that bundles all Kalshi infrastructure components.
    Instantiated once by the orchestrator and passed to all Kalshi bots.
    """

    def __init__(self, client, db_path: str = None, config: Dict = None):
        """
        Args:
            client: KalshiClient instance
            db_path: Path to SQLite database
            config: Optional config overrides {daily_loss_limit_cents, max_open_positions, max_contracts_per_market}
        """
        config = config or {}
        self.db_path = db_path or DB_PATH
        self.client = client

        self.position_manager = KalshiPositionManager(client, self.db_path)
        self.fill_tracker = KalshiFillTracker(client, self.position_manager, self.db_path)
        self.settlement_collector = KalshiSettlementCollector(client, self.position_manager, self.db_path)
        self.risk_manager = KalshiRiskManager(
            self.position_manager,
            self.db_path,
            daily_loss_limit_cents=config.get('daily_loss_limit_cents', 2000),
            max_open_positions=config.get('max_open_positions', 12),
            max_contracts_per_market=config.get('max_contracts_per_market', 25),
        )

        logger.info("KalshiInfrastructure initialized")

    def startup_reconciliation(self):
        """Run on startup: sync positions from API and reconcile."""
        logger.info("Running startup reconciliation...")
        self.position_manager.sync_from_api()
        discrepancies = self.position_manager.reconcile()
        if discrepancies:
            logger.warning(f"Startup reconciliation found {len(discrepancies)} discrepancies (auto-corrected)")
        else:
            logger.info("Startup reconciliation: clean")

    def poll_fills(self, bot_name: str = None) -> List[Dict]:
        """Poll for new fills — call every 60s from orchestrator."""
        return self.fill_tracker.poll_fills(bot_name)

    def collect_settlements(self) -> Dict:
        """Collect settlements — call every 30 min from orchestrator."""
        return self.settlement_collector.collect()

    def cancel_all_orders(self):
        """Cancel all open orders on Kalshi — called on shutdown."""
        try:
            orders = self.client.get_orders(status='resting')
            if not orders:
                orders = self.client.get_orders(status='open')
        except Exception as e:
            logger.error(f"Failed to fetch open orders for cancellation: {e}")
            return

        cancelled = 0
        for order in (orders or []):
            order_id = order.get('order_id', '')
            if order_id:
                try:
                    self.client.cancel_order(order_id)
                    cancelled += 1
                except Exception as e:
                    logger.warning(f"Failed to cancel order {order_id}: {e}")

        if cancelled:
            logger.info(f"Cancelled {cancelled} open Kalshi orders on shutdown")

    def get_status(self) -> Dict:
        """Get combined infrastructure status."""
        risk_status = self.risk_manager.get_status()
        settlement_stats = self.settlement_collector.get_settlement_stats()

        return {
            **risk_status,
            **settlement_stats,
            'fills_today': len(self.fill_tracker.get_fills_today()),
        }
