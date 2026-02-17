"""
Trade Logger - SQLite Database for Trade History

Stores all trades across platforms for analysis and reporting.

Author: Jacob
Created: January 2026
"""

import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TradeLogger')

# Default database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'trades.db')


class TradeLogger:
    """
    SQLite-based trade logging system.

    Features:
    - Store trades from all platforms
    - Query trade history
    - Calculate statistics
    - Export to CSV
    """

    def __init__(self, db_path: str = DB_PATH):
        """
        Initialize trade logger.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_exists()
        self._init_tables()

    def _ensure_db_exists(self):
        """Ensure database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_tables(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    platform TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    size REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    status TEXT DEFAULT 'open',
                    strategy TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Daily summaries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE NOT NULL,
                    starting_balance REAL,
                    ending_balance REAL,
                    total_pnl REAL,
                    trade_count INTEGER,
                    win_count INTEGER,
                    loss_count INTEGER,
                    win_rate REAL,
                    alpaca_pnl REAL,
                    kalshi_pnl REAL,
                    oanda_pnl REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_platform ON trades(platform)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")

            logger.info("Database tables initialized")

    def log_trade_open(
        self,
        trade_id: str,
        platform: str,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Log a new trade opening.

        Args:
            trade_id: Unique trade identifier
            platform: Trading platform
            symbol: Trading symbol
            side: Trade direction
            entry_price: Entry price
            size: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy: Strategy name
            notes: Additional notes

        Returns:
            True if logged successfully
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trades (
                        trade_id, platform, symbol, side, entry_price, size,
                        stop_loss, take_profit, strategy, notes, entry_time, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
                """, (
                    trade_id, platform, symbol, side, entry_price, size,
                    stop_loss, take_profit, strategy, notes, datetime.now()
                ))

            logger.info(f"Logged trade open: {trade_id}")
            return True

        except sqlite3.IntegrityError:
            logger.warning(f"Trade {trade_id} already exists")
            return False
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            return False

    def log_trade_close(
        self,
        trade_id: str,
        exit_price: float,
        notes: Optional[str] = None
    ) -> Optional[float]:
        """
        Log a trade closing.

        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            notes: Additional notes

        Returns:
            P&L amount or None if failed
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get trade details
                cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
                trade = cursor.fetchone()

                if not trade:
                    logger.warning(f"Trade {trade_id} not found")
                    return None

                # Calculate P&L
                entry_price = trade['entry_price']
                size = trade['size']
                side = trade['side']

                if side.lower() in ('long', 'buy', 'yes'):
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price

                pnl = size * pnl_pct

                # Update trade
                cursor.execute("""
                    UPDATE trades SET
                        exit_price = ?,
                        pnl = ?,
                        pnl_pct = ?,
                        exit_time = ?,
                        status = 'closed',
                        notes = COALESCE(notes || ' | ', '') || ?
                    WHERE trade_id = ?
                """, (exit_price, pnl, pnl_pct, datetime.now(), notes or '', trade_id))

            logger.info(f"Logged trade close: {trade_id}, P&L: ${pnl:.2f}")
            return pnl

        except Exception as e:
            logger.error(f"Failed to close trade: {e}")
            return None

    def get_trade(self, trade_id: str) -> Optional[Dict]:
        """Get a specific trade by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_open_trades(self, platform: Optional[str] = None) -> List[Dict]:
        """Get all open trades, optionally filtered by platform."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if platform:
                cursor.execute(
                    "SELECT * FROM trades WHERE status = 'open' AND platform = ?",
                    (platform,)
                )
            else:
                cursor.execute("SELECT * FROM trades WHERE status = 'open'")
            return [dict(row) for row in cursor.fetchall()]

    def get_trades(
        self,
        platform: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Query trades with filters.

        Args:
            platform: Filter by platform
            symbol: Filter by symbol
            start_date: Start date filter
            end_date: End date filter
            status: Filter by status (open/closed)
            limit: Maximum results

        Returns:
            List of trade dictionaries
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if platform:
            query += " AND platform = ?"
            params.append(platform)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_statistics(
        self,
        platform: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate trading statistics.

        Args:
            platform: Filter by platform (optional)
            days: Number of days to analyze

        Returns:
            Dictionary with statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build query
            query = """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl) as total_pnl,
                    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                    ABS(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END)) as gross_loss,
                    AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                    AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                    MAX(pnl) as largest_win,
                    MIN(pnl) as largest_loss
                FROM trades
                WHERE status = 'closed'
                AND entry_time >= datetime('now', ?)
            """
            params = [f'-{days} days']

            if platform:
                query += " AND platform = ?"
                params.append(platform)

            cursor.execute(query, params)
            row = cursor.fetchone()

            if not row or row['total_trades'] == 0:
                return {}

            stats = dict(row)

            # Calculate derived metrics
            stats['win_rate'] = stats['wins'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
            stats['profit_factor'] = stats['gross_profit'] / stats['gross_loss'] if stats['gross_loss'] > 0 else float('inf')

            return stats

    def save_daily_summary(
        self,
        date: datetime,
        starting_balance: float,
        ending_balance: float,
        platform_pnl: Dict[str, float]
    ) -> bool:
        """
        Save daily summary.

        Args:
            date: Summary date
            starting_balance: Starting balance
            ending_balance: Ending balance
            platform_pnl: P&L by platform

        Returns:
            True if saved successfully
        """
        try:
            # Get today's trade stats
            stats = self.get_statistics(days=1)

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO daily_summaries (
                        date, starting_balance, ending_balance, total_pnl,
                        trade_count, win_count, loss_count, win_rate,
                        alpaca_pnl, kalshi_pnl, oanda_pnl
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date.date(),
                    starting_balance,
                    ending_balance,
                    ending_balance - starting_balance,
                    stats.get('total_trades', 0),
                    stats.get('wins', 0),
                    stats.get('losses', 0),
                    stats.get('win_rate', 0),
                    platform_pnl.get('alpaca', 0),
                    platform_pnl.get('kalshi', 0),
                    platform_pnl.get('oanda', 0)
                ))

            logger.info(f"Saved daily summary for {date.date()}")
            return True

        except Exception as e:
            logger.error(f"Failed to save daily summary: {e}")
            return False

    def export_to_csv(self, filepath: str, days: int = 30) -> bool:
        """
        Export trades to CSV file.

        Args:
            filepath: Output CSV path
            days: Number of days to export

        Returns:
            True if exported successfully
        """
        try:
            import csv

            trades = self.get_trades(
                start_date=datetime.now() - timedelta(days=days),
                limit=10000
            )

            if not trades:
                logger.warning("No trades to export")
                return False

            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                writer.writeheader()
                writer.writerows(trades)

            logger.info(f"Exported {len(trades)} trades to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export: {e}")
            return False


# Import timedelta for export function
from datetime import timedelta


# Example usage
if __name__ == "__main__":
    logger = TradeLogger()

    # Log a trade
    logger.log_trade_open(
        trade_id="test_001",
        platform="alpaca",
        symbol="BTC/USD",
        side="long",
        entry_price=42000,
        size=100,
        stop_loss=40000,
        take_profit=46000,
        strategy="EMARSIStrategy"
    )

    # Close the trade
    pnl = logger.log_trade_close("test_001", exit_price=43500)
    print(f"Trade P&L: ${pnl:.2f}")

    # Get statistics
    stats = logger.get_statistics(days=30)
    print(f"Statistics: {stats}")
