"""
Sector Rotation Strategy Bot

Based on Mebane Faber's research - beats S&P 500 by 3.6% annually over 15 years.
Monthly rebalancing of top sectors by 12-month momentum with trend filter.

Strategy Rules:
1. Rank 11 sector ETFs by 12-month total return
2. Hold top 3 sectors IF price > 10-month SMA (trend filter)
3. Move to cash (or bonds) if sector fails trend filter
4. Rebalance monthly on first trading day

Expected Returns: 8-15% annually with lower drawdowns than buy-and-hold
Complexity: 2/5 (very simple to implement)
Time Commitment: 1 hour per month

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import volume-price divergence filter (available but not used for monthly rotation)
try:
    from filters.volume_price_divergence import VolumePriceDivergenceFilter
    DIVERGENCE_FILTER_AVAILABLE = True
except ImportError:
    DIVERGENCE_FILTER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SectorRotation')


class SectorETF(Enum):
    """S&P 500 Sector ETFs"""
    XLK = "Technology"
    XLF = "Financials"
    XLV = "Healthcare"
    XLE = "Energy"
    XLI = "Industrials"
    XLY = "Consumer Discretionary"
    XLP = "Consumer Staples"
    XLU = "Utilities"
    XLRE = "Real Estate"
    XLC = "Communication Services"
    XLB = "Materials"


@dataclass
class SectorSignal:
    """Signal for a sector ETF"""
    symbol: str
    sector_name: str
    momentum_12m: float  # 12-month return
    momentum_rank: int   # 1 = best
    current_price: float
    sma_10m: float
    above_trend: bool
    signal: str  # 'hold', 'buy', 'sell', 'avoid'
    weight: float  # Portfolio weight (0.0 to 1.0)
    timestamp: datetime


@dataclass
class PortfolioState:
    """Current portfolio state"""
    holdings: Dict[str, float]  # symbol -> weight
    cash_weight: float
    last_rebalance: datetime
    total_value: float


class SectorRotationBot:
    """
    Sector Rotation Strategy Implementation
    
    Key Features:
    - Monthly rebalancing
    - 12-month momentum ranking
    - 10-month SMA trend filter
    - Automatic position sizing
    - Paper trading support
    
    Usage:
        bot = SectorRotationBot(paper_mode=True)
        signals = bot.generate_signals()
        
        # On first of month:
        if bot.is_rebalance_day():
            orders = bot.get_rebalance_orders(account_value=10000)
    """
    
    SECTOR_ETFS = [e.name for e in SectorETF]
    
    # Strategy parameters
    TOP_N_SECTORS = 3           # Number of sectors to hold
    MOMENTUM_PERIOD = 252       # Trading days for 12-month momentum
    TREND_SMA_PERIOD = 200      # ~10 months in trading days
    MIN_WEIGHT = 0.10           # Minimum position weight
    MAX_WEIGHT = 0.40           # Maximum position weight
    CASH_ETF = "SHY"            # Short-term treasury for cash position
    
    def __init__(self, paper_mode: bool = None, db_path: str = None):
        # Safe default: read from environment, default to PAPER if not set
        if paper_mode is None:
            paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
        self.paper_mode = paper_mode
        # Use mode-specific path
        try:
            from utils.data_paths import get_db_path
            self.db_path = db_path or get_db_path("sector_rotation.db")
        except ImportError:
            self.db_path = db_path or "data/sector_rotation.db"
        self.alpaca_client = None

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()
        self._init_alpaca()
        
        logger.info(f"SectorRotationBot initialized (paper={paper_mode})")
    
    def _init_database(self):
        """Initialize SQLite database for tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sector_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                momentum_12m REAL,
                momentum_rank INTEGER,
                current_price REAL,
                sma_10m REAL,
                above_trend INTEGER,
                signal TEXT,
                weight REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rebalance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT,
                quantity REAL,
                price REAL,
                value REAL,
                weight REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                holdings TEXT,
                cash_weight REAL,
                total_value REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _init_alpaca(self):
        """Initialize Alpaca client"""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')
            
            if api_key and api_secret:
                self.trading_client = TradingClient(api_key, api_secret, paper=self.paper_mode)
                self.data_client = StockHistoricalDataClient(api_key, api_secret)
                logger.info("[OK] Alpaca clients initialized")
            else:
                logger.warning("Alpaca credentials not found - using yfinance fallback")
                self.trading_client = None
                self.data_client = None
        except ImportError:
            logger.warning("Alpaca SDK not installed - using yfinance")
            self.trading_client = None
            self.data_client = None
    
    def get_sector_data(self, symbol: str, period_days: int = 300) -> Optional[Dict]:
        """
        Get historical data for a sector ETF.
        
        Returns:
            Dict with 'prices', 'momentum_12m', 'sma_10m', 'current_price'
        """
        try:
            import yfinance as yf
            import pandas as pd
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y")
            
            if len(df) < self.MOMENTUM_PERIOD:
                logger.warning(f"{symbol}: Insufficient data ({len(df)} bars)")
                return None
            
            # Calculate 12-month momentum (total return)
            current_price = df['Close'].iloc[-1]
            price_12m_ago = df['Close'].iloc[-self.MOMENTUM_PERIOD] if len(df) >= self.MOMENTUM_PERIOD else df['Close'].iloc[0]
            momentum_12m = (current_price - price_12m_ago) / price_12m_ago
            
            # Calculate 10-month SMA (trend filter)
            sma_10m = df['Close'].rolling(window=self.TREND_SMA_PERIOD).mean().iloc[-1]
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'momentum_12m': momentum_12m,
                'sma_10m': sma_10m,
                'above_trend': current_price > sma_10m,
                'prices': df['Close'].tolist()[-60:],  # Last 60 days for charting
            }
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def generate_signals(self) -> List[SectorSignal]:
        """
        Generate rotation signals for all sector ETFs.
        
        Returns:
            List of SectorSignal objects, sorted by momentum rank
        """
        logger.info("Generating sector rotation signals...")
        
        # Gather data for all sectors
        sector_data = []
        for symbol in self.SECTOR_ETFS:
            data = self.get_sector_data(symbol)
            if data:
                sector_data.append(data)
        
        if not sector_data:
            logger.error("No sector data available")
            return []
        
        # Sort by 12-month momentum (descending)
        sector_data.sort(key=lambda x: x['momentum_12m'], reverse=True)
        
        # Generate signals
        signals = []
        now = datetime.now(timezone.utc)
        
        for rank, data in enumerate(sector_data, 1):
            symbol = data['symbol']
            sector_name = SectorETF[symbol].value if symbol in SectorETF.__members__ else symbol
            
            # Determine signal
            if rank <= self.TOP_N_SECTORS and data['above_trend']:
                signal = 'buy'
                weight = 1.0 / self.TOP_N_SECTORS  # Equal weight among top sectors
            elif rank <= self.TOP_N_SECTORS and not data['above_trend']:
                signal = 'avoid'  # Would be top sector but fails trend filter
                weight = 0.0
            else:
                signal = 'sell' if rank > self.TOP_N_SECTORS else 'hold'
                weight = 0.0
            
            sector_signal = SectorSignal(
                symbol=symbol,
                sector_name=sector_name,
                momentum_12m=data['momentum_12m'],
                momentum_rank=rank,
                current_price=data['current_price'],
                sma_10m=data['sma_10m'],
                above_trend=data['above_trend'],
                signal=signal,
                weight=weight,
                timestamp=now
            )
            signals.append(sector_signal)
            
            # Log signal
            trend_emoji = "📈" if data['above_trend'] else "📉"
            signal_emoji = {"buy": "🟢", "sell": "🔴", "avoid": "⚠️", "hold": "⏸️"}.get(signal, "⚪")
            logger.info(f"  #{rank} {symbol} ({sector_name}): {data['momentum_12m']:+.1%} {trend_emoji} → {signal_emoji} {signal.upper()}")
        
        # Store signals in database
        self._store_signals(signals)
        
        return signals
    
    def _store_signals(self, signals: List[SectorSignal]):
        """Store signals in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for sig in signals:
            cursor.execute("""
                INSERT INTO sector_signals 
                (timestamp, symbol, momentum_12m, momentum_rank, current_price, sma_10m, above_trend, signal, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sig.timestamp.isoformat(),
                sig.symbol,
                sig.momentum_12m,
                sig.momentum_rank,
                sig.current_price,
                sig.sma_10m,
                1 if sig.above_trend else 0,
                sig.signal,
                sig.weight
            ))
        
        conn.commit()
        conn.close()
    
    def is_rebalance_day(self) -> bool:
        """
        Check if today is a rebalance day.
        
        Rebalance on first trading day of the month.
        """
        today = datetime.now()
        
        # Simple check: first 5 days of month (to catch market holidays)
        if today.day <= 5:
            # Check if we already rebalanced this month
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT MAX(timestamp) FROM rebalance_history
                WHERE strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now')
            """)
            
            row = cursor.fetchone()
            conn.close()
            
            if row[0] is None:
                return True  # Haven't rebalanced this month
        
        return False
    
    def get_rebalance_orders(self, account_value: float, current_holdings: Dict[str, float] = None) -> List[Dict]:
        """
        Calculate orders needed to rebalance portfolio.
        
        Args:
            account_value: Total account value
            current_holdings: Dict of symbol -> current value
            
        Returns:
            List of order dicts with 'symbol', 'side', 'quantity', 'value'
        """
        signals = self.generate_signals()
        
        if not signals:
            return []
        
        current_holdings = current_holdings or {}
        orders = []
        
        # Calculate target positions
        target_positions = {}
        for sig in signals:
            if sig.weight > 0:
                target_value = account_value * sig.weight
                target_positions[sig.symbol] = {
                    'target_value': target_value,
                    'target_shares': int(target_value / sig.current_price),
                    'price': sig.current_price
                }
        
        # Calculate sells (positions to exit)
        for symbol, current_value in current_holdings.items():
            if symbol not in target_positions and symbol in self.SECTOR_ETFS:
                # Need to sell entire position
                orders.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': 'all',
                    'current_value': current_value,
                    'reason': 'No longer in top sectors'
                })
        
        # Calculate buys/adjustments
        for symbol, target in target_positions.items():
            current_value = current_holdings.get(symbol, 0)
            diff = target['target_value'] - current_value
            
            if abs(diff) > account_value * 0.02:  # 2% threshold to avoid small trades
                if diff > 0:
                    orders.append({
                        'symbol': symbol,
                        'side': 'buy',
                        'quantity': int(diff / target['price']),
                        'value': diff,
                        'reason': 'Increase to target weight'
                    })
                else:
                    orders.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'quantity': int(abs(diff) / target['price']),
                        'value': abs(diff),
                        'reason': 'Reduce to target weight'
                    })
        
        return orders
    
    def execute_rebalance(self, account_value: float) -> Dict:
        """
        Execute monthly rebalance.
        
        Returns:
            Dict with rebalance results
        """
        if self.paper_mode:
            logger.info("[PAPER] Executing sector rotation rebalance...")
        else:
            logger.info("[LIVE] Executing sector rotation rebalance...")
        
        orders = self.get_rebalance_orders(account_value)
        
        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'account_value': account_value,
            'orders_planned': len(orders),
            'orders_executed': 0,
            'orders': []
        }
        
        for order in orders:
            if self.paper_mode:
                # Simulate execution
                logger.info(f"[PAPER] {order['side'].upper()} {order['quantity']} {order['symbol']}")
                order['status'] = 'filled'
                order['fill_price'] = order.get('value', 0) / max(order.get('quantity', 1), 1)
                results['orders_executed'] += 1
            else:
                # Real execution via Alpaca
                try:
                    from alpaca.trading.client import TradingClient
                    from alpaca.trading.requests import MarketOrderRequest
                    from alpaca.trading.enums import OrderSide, TimeInForce

                    # Load Alpaca credentials
                    api_key = os.getenv('ALPACA_API_KEY')
                    secret_key = os.getenv('ALPACA_SECRET_KEY')

                    if not api_key or not secret_key:
                        raise ValueError("Alpaca credentials not configured")

                    client = TradingClient(api_key, secret_key, paper=False)

                    # Create market order
                    side = OrderSide.BUY if order['side'].lower() == 'buy' else OrderSide.SELL
                    order_request = MarketOrderRequest(
                        symbol=order['symbol'],
                        qty=order['quantity'],
                        side=side,
                        time_in_force=TimeInForce.DAY
                    )

                    # Submit order
                    trade = client.submit_order(order_request)
                    order['status'] = 'submitted'
                    order['order_id'] = str(trade.id)
                    results['orders_executed'] += 1
                    logger.info(f"[LIVE] {order['side'].upper()} {order['quantity']} {order['symbol']} - Order ID: {trade.id}")

                except Exception as e:
                    logger.error(f"Order execution failed: {e}")
                    order['status'] = 'failed'
                    order['error'] = str(e)
            
            results['orders'].append(order)
            
            # Log to database
            self._log_rebalance(order)
        
        return results
    
    def _log_rebalance(self, order: Dict):
        """Log rebalance order to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO rebalance_history 
            (timestamp, action, symbol, side, quantity, price, value, weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            'rebalance',
            order['symbol'],
            order['side'],
            order.get('quantity'),
            order.get('fill_price'),
            order.get('value'),
            order.get('weight')
        ))
        
        conn.commit()
        conn.close()
    
    def get_current_allocation(self) -> Dict:
        """Get current portfolio allocation from last signals"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT symbol, weight, momentum_rank, signal
            FROM sector_signals
            WHERE timestamp = (SELECT MAX(timestamp) FROM sector_signals)
            AND weight > 0
            ORDER BY momentum_rank
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        allocation = {}
        for row in rows:
            allocation[row[0]] = {
                'weight': row[1],
                'rank': row[2],
                'signal': row[3]
            }
        
        return allocation
    
    def get_performance_summary(self) -> Dict:
        """Get strategy performance summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count rebalances
        cursor.execute("SELECT COUNT(DISTINCT strftime('%Y-%m', timestamp)) FROM rebalance_history")
        months_active = cursor.fetchone()[0]
        
        # Get unique symbols held
        cursor.execute("SELECT DISTINCT symbol FROM rebalance_history WHERE side = 'buy'")
        symbols_held = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'months_active': months_active,
            'symbols_held': symbols_held,
            'strategy': 'Faber Sector Rotation',
            'top_n_sectors': self.TOP_N_SECTORS,
            'rebalance_frequency': 'Monthly'
        }
    
    def get_dashboard_data(self) -> Dict:
        """Get data for dashboard display"""
        signals = self.generate_signals()
        
        return {
            'strategy_name': 'Sector Rotation',
            'last_update': datetime.now(timezone.utc).isoformat(),
            'signals': [
                {
                    'symbol': s.symbol,
                    'sector': s.sector_name,
                    'momentum': f"{s.momentum_12m:+.1%}",
                    'rank': s.momentum_rank,
                    'trend': 'Above' if s.above_trend else 'Below',
                    'signal': s.signal.upper(),
                    'weight': f"{s.weight:.0%}" if s.weight > 0 else '-'
                }
                for s in signals
            ],
            'allocation': self.get_current_allocation(),
            'is_rebalance_day': self.is_rebalance_day(),
            'next_rebalance': 'First trading day of next month',
            'performance': self.get_performance_summary()
        }


if __name__ == "__main__":
    print("=" * 60)
    print("SECTOR ROTATION BOT - TEST")
    print("=" * 60)
    
    bot = SectorRotationBot(paper_mode=True)
    
    # Generate signals
    print("\nGenerating sector rotation signals...")
    signals = bot.generate_signals()
    
    print("\n" + "=" * 60)
    print("SECTOR RANKINGS")
    print("=" * 60)
    
    for sig in signals:
        trend = "✅" if sig.above_trend else "❌"
        weight = f"{sig.weight:.0%}" if sig.weight > 0 else "-"
        print(f"#{sig.momentum_rank:2d} {sig.symbol:5s} ({sig.sector_name:25s}): "
              f"{sig.momentum_12m:+7.1%} | Trend: {trend} | Signal: {sig.signal:5s} | Weight: {weight}")
    
    # Check rebalance
    print(f"\nIs rebalance day? {bot.is_rebalance_day()}")
    
    # Get rebalance orders
    print("\n" + "=" * 60)
    print("REBALANCE ORDERS (for $10,000 account)")
    print("=" * 60)
    
    orders = bot.get_rebalance_orders(account_value=10000)
    
    if orders:
        for order in orders:
            print(f"  {order['side'].upper():4s} {order.get('quantity', 'all'):>5} {order['symbol']:5s} "
                  f"(${order.get('value', 0):,.0f}) - {order.get('reason', '')}")
    else:
        print("  No orders needed")
    
    # Dashboard data
    print("\n" + "=" * 60)
    print("DASHBOARD DATA")
    print("=" * 60)
    
    dashboard = bot.get_dashboard_data()
    print(f"Strategy: {dashboard['strategy_name']}")
    print(f"Last Update: {dashboard['last_update']}")
    print(f"Rebalance Day: {dashboard['is_rebalance_day']}")
