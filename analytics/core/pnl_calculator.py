"""
Real-Time P&L Calculator
========================

Comprehensive P&L tracking and calculation system for all trading strategies
and markets including stocks, crypto, forex, and prediction markets.

Features:
- Real-time P&L calculation
- Multi-strategy and multi-market tracking
- Position-level P&L attribution
- Unrealized and realized P&L separation
- Currency conversion support
- Time-weighted returns

Author: Trading Bot System
Created: February 2026
"""

import logging
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, asdict
from collections import defaultdict
import yfinance as yf
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Individual position tracking"""
    symbol: str
    strategy: str
    platform: str
    side: str  # 'long', 'short', 'yes', 'no'
    entry_price: Decimal
    current_price: Decimal
    quantity: Decimal
    entry_time: datetime
    last_update: datetime
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    currency: str = 'USD'

    def update_unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Update unrealized P&L based on current price"""
        self.current_price = current_price
        self.last_update = datetime.now()

        if self.side.lower() in ['long', 'buy', 'yes']:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # short, sell, no
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

        return self.unrealized_pnl


@dataclass
class PnLSnapshot:
    """Point-in-time P&L snapshot"""
    timestamp: datetime
    total_pnl: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    daily_pnl: Decimal
    mtd_pnl: Decimal
    ytd_pnl: Decimal
    position_count: int
    by_strategy: Dict[str, Decimal]
    by_platform: Dict[str, Decimal]
    by_market: Dict[str, Decimal]


class PnLCalculator:
    """
    Real-time P&L calculator with comprehensive tracking capabilities.

    Features:
    - Real-time position tracking
    - Multi-strategy P&L attribution
    - Platform-specific calculations
    - Historical P&L snapshots
    - Currency conversion
    - Performance metrics integration
    """

    def __init__(self, db_path: str = None):
        """
        Initialize P&L calculator.

        Args:
            db_path: Database path for persistence
        """
        self.db_path = db_path or "data/pnl_tracker.db"
        self.positions: Dict[str, Position] = {}
        self.pnl_history: List[PnLSnapshot] = []
        self.starting_capital = Decimal('10000')  # Default starting capital
        self.daily_starting_balance = Decimal('0')

        # Thread safety
        self._lock = threading.RLock()

        # Price cache for real-time updates
        self._price_cache: Dict[str, Tuple[Decimal, datetime]] = {}
        self._price_update_interval = 30  # seconds

        # Currency conversion rates cache
        self._fx_rates: Dict[str, Decimal] = {'USD': Decimal('1')}

        self._init_database()
        self._load_existing_positions()

        logger.info("PnL Calculator initialized")

    def _init_database(self):
        """Initialize SQLite database for persistence"""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    strategy TEXT,
                    platform TEXT,
                    side TEXT,
                    entry_price REAL,
                    current_price REAL,
                    quantity REAL,
                    entry_time TEXT,
                    last_update TEXT,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    currency TEXT,
                    status TEXT DEFAULT 'open'
                )
            ''')

            # P&L snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pnl_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    total_pnl REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    daily_pnl REAL,
                    mtd_pnl REAL,
                    ytd_pnl REAL,
                    position_count INTEGER,
                    by_strategy TEXT,
                    by_platform TEXT,
                    by_market TEXT
                )
            ''')

            # Realized trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS realized_trades (
                    trade_id TEXT PRIMARY KEY,
                    position_id TEXT,
                    symbol TEXT,
                    strategy TEXT,
                    platform TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    realized_pnl REAL,
                    currency TEXT
                )
            ''')

            conn.commit()

    def _load_existing_positions(self):
        """Load existing open positions from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM positions WHERE status = 'open'
                ''')

                for row in cursor.fetchall():
                    position = Position(
                        symbol=row[1],
                        strategy=row[2],
                        platform=row[3],
                        side=row[4],
                        entry_price=Decimal(str(row[5])),
                        current_price=Decimal(str(row[6])),
                        quantity=Decimal(str(row[7])),
                        entry_time=datetime.fromisoformat(row[8]),
                        last_update=datetime.fromisoformat(row[9]),
                        unrealized_pnl=Decimal(str(row[10])),
                        realized_pnl=Decimal(str(row[11])),
                        currency=row[12]
                    )
                    self.positions[row[0]] = position

            logger.info(f"Loaded {len(self.positions)} existing positions")
        except Exception as e:
            logger.error(f"Error loading positions: {e}")

    def open_position(
        self,
        position_id: str,
        symbol: str,
        strategy: str,
        platform: str,
        side: str,
        entry_price: Union[float, Decimal],
        quantity: Union[float, Decimal],
        currency: str = 'USD'
    ) -> bool:
        """
        Open a new position.

        Args:
            position_id: Unique position identifier
            symbol: Trading symbol
            strategy: Strategy name
            platform: Trading platform
            side: Position side (long/short/yes/no)
            entry_price: Entry price
            quantity: Position quantity
            currency: Currency

        Returns:
            True if position opened successfully
        """
        with self._lock:
            try:
                if position_id in self.positions:
                    logger.warning(f"Position {position_id} already exists")
                    return False

                position = Position(
                    symbol=symbol,
                    strategy=strategy,
                    platform=platform,
                    side=side.lower(),
                    entry_price=Decimal(str(entry_price)),
                    current_price=Decimal(str(entry_price)),
                    quantity=Decimal(str(quantity)),
                    entry_time=datetime.now(),
                    last_update=datetime.now(),
                    currency=currency
                )

                self.positions[position_id] = position
                self._save_position(position_id, position)

                logger.info(f"Opened position: {position_id} - {symbol} {side} @ {entry_price}")
                return True

            except Exception as e:
                logger.error(f"Error opening position {position_id}: {e}")
                return False

    def close_position(
        self,
        position_id: str,
        exit_price: Union[float, Decimal],
        quantity: Optional[Union[float, Decimal]] = None
    ) -> Optional[Decimal]:
        """
        Close a position (full or partial).

        Args:
            position_id: Position to close
            exit_price: Exit price
            quantity: Quantity to close (None for full close)

        Returns:
            Realized P&L or None if error
        """
        with self._lock:
            try:
                if position_id not in self.positions:
                    logger.warning(f"Position {position_id} not found")
                    return None

                position = self.positions[position_id]
                close_qty = Decimal(str(quantity)) if quantity else position.quantity
                exit_price_decimal = Decimal(str(exit_price))

                # Calculate realized P&L
                if position.side in ['long', 'buy', 'yes']:
                    realized_pnl = (exit_price_decimal - position.entry_price) * close_qty
                else:
                    realized_pnl = (position.entry_price - exit_price_decimal) * close_qty

                # Record realized trade
                self._record_realized_trade(
                    position_id, position, exit_price_decimal, close_qty, realized_pnl
                )

                # Update or remove position
                if close_qty >= position.quantity:
                    # Full close
                    del self.positions[position_id]
                    self._update_position_status(position_id, 'closed')
                else:
                    # Partial close
                    position.quantity -= close_qty
                    position.realized_pnl += realized_pnl
                    self._save_position(position_id, position)

                logger.info(f"Closed position: {position_id} - P&L: {realized_pnl}")
                return realized_pnl

            except Exception as e:
                logger.error(f"Error closing position {position_id}: {e}")
                return None

    def update_price(self, symbol: str, price: Union[float, Decimal]):
        """Update current price for a symbol"""
        with self._lock:
            price_decimal = Decimal(str(price))
            self._price_cache[symbol] = (price_decimal, datetime.now())

            # Update all positions with this symbol
            for position_id, position in self.positions.items():
                if position.symbol == symbol:
                    position.update_unrealized_pnl(price_decimal)
                    self._save_position(position_id, position)

    def update_all_prices(self):
        """Update prices for all positions using live market data"""
        symbols = set(position.symbol for position in self.positions.values())

        for symbol in symbols:
            try:
                # Try to get live price
                price = self._fetch_live_price(symbol)
                if price:
                    self.update_price(symbol, price)
            except Exception as e:
                logger.error(f"Error updating price for {symbol}: {e}")

    def _fetch_live_price(self, symbol: str) -> Optional[Decimal]:
        """Fetch live price for a symbol"""
        try:
            # Check cache first (avoid excessive API calls)
            if symbol in self._price_cache:
                cached_price, cached_time = self._price_cache[symbol]
                if (datetime.now() - cached_time).seconds < self._price_update_interval:
                    return cached_price

            # Fetch from appropriate source based on symbol type
            if '/' in symbol and symbol.upper().endswith('/USD'):
                # Crypto pair
                ticker = yf.Ticker(symbol.replace('/', '-') + '=X')
            elif symbol.upper() in ['EUR/USD', 'GBP/USD', 'USD/JPY']:
                # Forex pair
                ticker = yf.Ticker(symbol.upper() + '=X')
            else:
                # Stock/ETF
                ticker = yf.Ticker(symbol.upper())

            hist = ticker.history(period="1d", interval="1m")
            if len(hist) > 0:
                return Decimal(str(hist['Close'].iloc[-1]))

        except Exception as e:
            logger.debug(f"Failed to fetch price for {symbol}: {e}")

        return None

    def get_current_pnl(self) -> PnLSnapshot:
        """Get current P&L snapshot"""
        with self._lock:
            now = datetime.now()

            # Calculate unrealized P&L
            total_unrealized = Decimal('0')
            by_strategy = defaultdict(lambda: Decimal('0'))
            by_platform = defaultdict(lambda: Decimal('0'))
            by_market = defaultdict(lambda: Decimal('0'))

            for position in self.positions.values():
                total_unrealized += position.unrealized_pnl
                by_strategy[position.strategy] += position.unrealized_pnl
                by_platform[position.platform] += position.unrealized_pnl

                # Classify by market
                market = self._classify_market(position.symbol, position.platform)
                by_market[market] += position.unrealized_pnl

            # Get realized P&L for today
            realized_today = self._get_realized_pnl_for_period('today')
            realized_mtd = self._get_realized_pnl_for_period('mtd')
            realized_ytd = self._get_realized_pnl_for_period('ytd')

            total_realized = self._get_realized_pnl_for_period('all')
            total_pnl = total_unrealized + total_realized
            daily_pnl = total_unrealized + realized_today

            snapshot = PnLSnapshot(
                timestamp=now,
                total_pnl=total_pnl,
                unrealized_pnl=total_unrealized,
                realized_pnl=total_realized,
                daily_pnl=daily_pnl,
                mtd_pnl=total_unrealized + realized_mtd,
                ytd_pnl=total_unrealized + realized_ytd,
                position_count=len(self.positions),
                by_strategy=dict(by_strategy),
                by_platform=dict(by_platform),
                by_market=dict(by_market)
            )

            # Save snapshot
            self._save_pnl_snapshot(snapshot)
            self.pnl_history.append(snapshot)

            # Keep only recent snapshots in memory (last 1000)
            if len(self.pnl_history) > 1000:
                self.pnl_history = self.pnl_history[-1000:]

            return snapshot

    def _classify_market(self, symbol: str, platform: str) -> str:
        """Classify symbol into market category"""
        if platform.lower() in ['kalshi', 'polymarket']:
            return 'prediction'
        elif '/' in symbol or platform.lower() in ['binance', 'coinbase']:
            return 'crypto'
        elif platform.lower() in ['oanda', 'fxcm']:
            return 'forex'
        else:
            return 'stocks'

    def _get_realized_pnl_for_period(self, period: str) -> Decimal:
        """Get realized P&L for a specific period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if period == 'today':
                    start_date = datetime.now().date()
                    query = '''
                        SELECT SUM(realized_pnl) FROM realized_trades
                        WHERE date(exit_time) = ?
                    '''
                    cursor.execute(query, (start_date,))
                elif period == 'mtd':
                    start_date = datetime.now().replace(day=1).date()
                    query = '''
                        SELECT SUM(realized_pnl) FROM realized_trades
                        WHERE date(exit_time) >= ?
                    '''
                    cursor.execute(query, (start_date,))
                elif period == 'ytd':
                    start_date = datetime.now().replace(month=1, day=1).date()
                    query = '''
                        SELECT SUM(realized_pnl) FROM realized_trades
                        WHERE date(exit_time) >= ?
                    '''
                    cursor.execute(query, (start_date,))
                else:  # all
                    query = 'SELECT SUM(realized_pnl) FROM realized_trades'
                    cursor.execute(query)

                result = cursor.fetchone()[0]
                return Decimal(str(result)) if result else Decimal('0')

        except Exception as e:
            logger.error(f"Error getting realized P&L for {period}: {e}")
            return Decimal('0')

    def get_strategy_attribution(self) -> Dict[str, Dict[str, Union[Decimal, int]]]:
        """Get P&L attribution by strategy"""
        with self._lock:
            attribution = defaultdict(lambda: {
                'unrealized_pnl': Decimal('0'),
                'realized_pnl': Decimal('0'),
                'total_pnl': Decimal('0'),
                'position_count': 0
            })

            # Unrealized P&L from open positions
            for position in self.positions.values():
                strategy = position.strategy
                attribution[strategy]['unrealized_pnl'] += position.unrealized_pnl
                attribution[strategy]['position_count'] += 1

            # Realized P&L from closed trades
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT strategy, SUM(realized_pnl)
                        FROM realized_trades
                        GROUP BY strategy
                    ''')

                    for strategy, realized_pnl in cursor.fetchall():
                        attribution[strategy]['realized_pnl'] = Decimal(str(realized_pnl))
            except Exception as e:
                logger.error(f"Error getting realized P&L by strategy: {e}")

            # Calculate totals
            for strategy_data in attribution.values():
                strategy_data['total_pnl'] = (
                    strategy_data['unrealized_pnl'] + strategy_data['realized_pnl']
                )

            return dict(attribution)

    def get_rolling_returns(self, window_days: int = 30) -> pd.DataFrame:
        """Get rolling returns for the specified window"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, total_pnl
                    FROM pnl_snapshots
                    WHERE timestamp >= datetime('now', ?)
                    ORDER BY timestamp
                '''
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(f'-{window_days} days',),
                    parse_dates=['timestamp']
                )

                if len(df) == 0:
                    return pd.DataFrame()

                df['returns'] = df['total_pnl'].pct_change()
                df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1

                return df[['timestamp', 'total_pnl', 'returns', 'cumulative_returns']]

        except Exception as e:
            logger.error(f"Error calculating rolling returns: {e}")
            return pd.DataFrame()

    def _save_position(self, position_id: str, position: Position):
        """Save position to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO positions
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
                ''', (
                    position_id,
                    position.symbol,
                    position.strategy,
                    position.platform,
                    position.side,
                    float(position.entry_price),
                    float(position.current_price),
                    float(position.quantity),
                    position.entry_time.isoformat(),
                    position.last_update.isoformat(),
                    float(position.unrealized_pnl),
                    float(position.realized_pnl),
                    position.currency
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving position {position_id}: {e}")

    def _update_position_status(self, position_id: str, status: str):
        """Update position status in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE positions SET status = ? WHERE position_id = ?',
                    (status, position_id)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating position status {position_id}: {e}")

    def _record_realized_trade(
        self,
        position_id: str,
        position: Position,
        exit_price: Decimal,
        quantity: Decimal,
        realized_pnl: Decimal
    ):
        """Record a realized trade"""
        try:
            trade_id = f"{position_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO realized_trades VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id,
                    position_id,
                    position.symbol,
                    position.strategy,
                    position.platform,
                    position.side,
                    float(position.entry_price),
                    float(exit_price),
                    float(quantity),
                    position.entry_time.isoformat(),
                    datetime.now().isoformat(),
                    float(realized_pnl),
                    position.currency
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error recording realized trade: {e}")

    def _save_pnl_snapshot(self, snapshot: PnLSnapshot):
        """Save P&L snapshot to database"""
        try:
            import json

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO pnl_snapshots
                    (timestamp, total_pnl, unrealized_pnl, realized_pnl,
                     daily_pnl, mtd_pnl, ytd_pnl, position_count,
                     by_strategy, by_platform, by_market)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.timestamp.isoformat(),
                    float(snapshot.total_pnl),
                    float(snapshot.unrealized_pnl),
                    float(snapshot.realized_pnl),
                    float(snapshot.daily_pnl),
                    float(snapshot.mtd_pnl),
                    float(snapshot.ytd_pnl),
                    snapshot.position_count,
                    json.dumps({k: float(v) for k, v in snapshot.by_strategy.items()}),
                    json.dumps({k: float(v) for k, v in snapshot.by_platform.items()}),
                    json.dumps({k: float(v) for k, v in snapshot.by_market.items()})
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving P&L snapshot: {e}")

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        current_pnl = self.get_current_pnl()
        attribution = self.get_strategy_attribution()

        return {
            'timestamp': current_pnl.timestamp.isoformat(),
            'total_pnl': float(current_pnl.total_pnl),
            'unrealized_pnl': float(current_pnl.unrealized_pnl),
            'realized_pnl': float(current_pnl.realized_pnl),
            'daily_pnl': float(current_pnl.daily_pnl),
            'mtd_pnl': float(current_pnl.mtd_pnl),
            'ytd_pnl': float(current_pnl.ytd_pnl),
            'position_count': current_pnl.position_count,
            'by_strategy': {k: float(v['total_pnl']) for k, v in attribution.items()},
            'by_platform': {k: float(v) for k, v in current_pnl.by_platform.items()},
            'by_market': {k: float(v) for k, v in current_pnl.by_market.items()}
        }


# Example usage
if __name__ == "__main__":
    calculator = PnLCalculator()

    # Open some test positions
    calculator.open_position(
        "test_001", "AAPL", "momentum", "alpaca", "long", 150.0, 100
    )
    calculator.open_position(
        "test_002", "BTC/USD", "crypto_momentum", "binance", "long", 45000.0, 0.1
    )

    # Update prices
    calculator.update_price("AAPL", 155.0)
    calculator.update_price("BTC/USD", 47000.0)

    # Get current P&L
    pnl = calculator.get_current_pnl()
    print(f"Total P&L: ${pnl.total_pnl}")
    print(f"Unrealized: ${pnl.unrealized_pnl}")
    print(f"By Strategy: {pnl.by_strategy}")