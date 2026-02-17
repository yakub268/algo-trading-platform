"""
MULTI-COIN MOMENTUM BOT
=======================

Rotates capital into the top performing coins daily.
This is a momentum-based strategy that chases strength.

Strategy:
- Every 4 hours, scan all USDC pairs
- Rank by 24h performance (price change %)
- Hold top 3 performers
- When a coin drops out of top 3, sell it and buy the new top performer
- No stop-loss (momentum strategy), but max 24h hold
- Position size: split available capital equally among 3 positions

Risk Warning:
- Momentum chasing can result in buying tops
- No stop-loss means unlimited downside
- High turnover = high fees
- Only use capital you can afford to lose

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Add parent directories for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

# Import base class (now uses AlpacaCryptoClient internally)
from bots.aggressive.base_aggressive_bot import BaseAggressiveBot
from bots.alpaca_crypto_client import AlpacaCryptoClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MultiMomentum')


@dataclass
class CoinPerformance:
    """Track a coin's performance metrics"""
    product_id: str
    symbol: str
    current_price: float
    price_24h_ago: float
    change_24h_pct: float
    volume_24h: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            'product_id': self.product_id,
            'symbol': self.symbol,
            'current_price': self.current_price,
            'price_24h_ago': self.price_24h_ago,
            'change_24h_pct': self.change_24h_pct,
            'volume_24h': self.volume_24h,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp
        }


@dataclass
class MomentumHolding:
    """Track a momentum position"""
    product_id: str
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    entry_change_pct: float  # The 24h change when we entered
    current_price: float = 0.0
    current_change_pct: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0

    def update(self, current_price: float, current_change_pct: float):
        """Update current values and P&L"""
        self.current_price = current_price
        self.current_change_pct = current_change_pct
        self.pnl = (current_price - self.entry_price) * self.quantity
        self.pnl_pct = (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0

    def to_dict(self) -> dict:
        return {
            'product_id': self.product_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if isinstance(self.entry_time, datetime) else self.entry_time,
            'entry_change_pct': self.entry_change_pct,
            'current_price': self.current_price,
            'current_change_pct': self.current_change_pct,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct
        }


class MultiMomentumBot(BaseAggressiveBot):
    """
    Multi-Coin Momentum Bot

    Rotates capital into top 3 performing USDC coins every 4 hours.
    Pure momentum strategy - rides winners, dumps losers.
    """

    # Strategy parameters
    TOP_N = 3  # Number of top performers to hold
    SCAN_INTERVAL_HOURS = 4  # Scan every 4 hours
    MAX_HOLD_HOURS = 24  # Maximum hold time before forced rotation
    MIN_VOLUME_USD = 100000  # Minimum $100k 24h volume in USD
    STOP_LOSS_PCT = 0.05     # 5% stop-loss (was missing entirely)

    # Exclude stablecoins and wrapped tokens
    EXCLUDED_SYMBOLS = {
        'USDT', 'DAI', 'BUSD', 'TUSD', 'USDP', 'GUSD', 'PAX',  # Stablecoins
        'WBTC', 'WETH', 'STETH',  # Wrapped tokens
        'USDC'  # Our quote currency
    }

    def __init__(
        self,
        capital: float = 300.0,
        paper_mode: bool = None,
        holdings_file: str = None
    ):
        """
        Initialize Multi-Momentum Bot.

        Args:
            capital: Total capital to deploy across 3 positions
            paper_mode: Paper trading mode
            holdings_file: Path to JSON file for persisting holdings
        """
        super().__init__(capital=capital, paper_mode=paper_mode)

        # Holdings file for persistence
        if holdings_file is None:
            bot_dir = os.path.dirname(os.path.abspath(__file__))
            holdings_file = os.path.join(bot_dir, 'multi_momentum_holdings.json')
        self.holdings_file = holdings_file

        # Current holdings
        self.holdings: Dict[str, MomentumHolding] = {}

        # Performance rankings
        self.rankings: List[CoinPerformance] = []

        # Last scan time
        self.last_scan_time: Optional[datetime] = None

        # Load existing holdings
        self._load_holdings()

        # Trades list (separate from base class history)
        self.trades: List[Dict] = []

        # Central DB for orchestrator integration
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.central_db_path = os.path.join(base_dir, 'data', 'live', 'trading_master.db')

        # Price cache
        self.price_cache: Dict[str, Dict] = {}

        logger.info(
            f"MultiMomentumBot initialized - Capital: ${capital}, "
            f"Top {self.TOP_N}, Scan every {self.SCAN_INTERVAL_HOURS}h, "
            f"Current holdings: {len(self.holdings)}"
        )

    def get_usdc_pairs(self) -> List[str]:
        """
        Get available crypto /USD pairs for Alpaca.

        Returns:
            List of product IDs like ['BTC/USD', 'ETH/USD', ...]
        """
        return self._get_default_usdc_pairs()

    def _get_default_usdc_pairs(self) -> List[str]:
        """Default crypto /USD pairs for Alpaca - Core 6 prioritized first"""
        return [
            # CORE UNIVERSE - Prioritized for best liquidity/volatility
            'BTC/USD',   # Core - largest, most liquid
            'ETH/USD',   # Core - second largest
            'SOL/USD',   # Core - best volatility/liquidity ratio
            'XRP/USD',   # High volume, clear levels
            'DOGE/USD',  # High volatility for momentum signals
            'LINK/USD',  # Steady oracle leader
            # EXTENDED UNIVERSE - Secondary scanning
            'AVAX/USD', 'DOT/USD', 'SHIB/USD',
            'UNI/USD', 'ATOM/USD', 'LTC/USD', 'ADA/USD', 'NEAR/USD',
            'APT/USD', 'ARB/USD', 'OP/USD',
        ]

    def get_prices(self, product_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch current prices for multiple products via Alpaca.

        Args:
            product_ids: List of product IDs (e.g., ['BTC/USD', 'ETH/USD'])

        Returns:
            Dict of product_id -> {bid, ask, mid}
        """
        prices = {}
        alpaca = getattr(self, 'alpaca', None) or getattr(self, 'coinbase', None)

        if not alpaca or not alpaca._initialized:
            if not self.paper_mode:
                logger.error("LIVE MODE: Alpaca not initialized — refusing to use mock prices")
                return {}
            return self._get_mock_prices(product_ids)

        try:
            results = alpaca.get_best_bid_ask(product_ids)
            for item in results:
                product_id = item.get('product_id', '')
                bid = item.get('bid', 0)
                ask = item.get('ask', 0)
                mid = item.get('mid', (bid + ask) / 2 if bid and ask else 0)
                prices[product_id] = {'bid': bid, 'ask': ask, 'mid': mid}

        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            prices = self._get_mock_prices(product_ids)

        self.price_cache.update(prices)
        return prices

    def _get_mock_prices(self, product_ids: List[str]) -> Dict[str, Dict]:
        """Generate mock prices for paper trading"""
        import random

        # Base prices for common coins
        base_prices = {
            'BTC/USD': 95000, 'ETH/USD': 3200, 'SOL/USD': 180,
            'DOGE/USD': 0.32, 'XRP/USD': 2.10, 'AVAX/USD': 35,
            'LINK/USD': 22, 'DOT/USD': 7, 'SHIB/USD': 0.000022,
            'UNI/USD': 12, 'ATOM/USD': 8,
            'LTC/USD': 110, 'ADA/USD': 0.95, 'NEAR/USD': 5,
            'APT/USD': 9, 'ARB/USD': 0.85, 'OP/USD': 2.1,
        }

        prices = {}
        for product_id in product_ids:
            base = base_prices.get(product_id, 1.0)
            spread_pct = random.uniform(0.001, 0.003)
            mid = base * (1 + random.uniform(-0.02, 0.02))
            bid = mid * (1 - spread_pct / 2)
            ask = mid * (1 + spread_pct / 2)

            prices[product_id] = {
                'bid': bid,
                'ask': ask,
                'mid': mid
            }

        return prices

    def _load_holdings(self):
        """Load holdings from JSON file"""
        try:
            if os.path.exists(self.holdings_file):
                with open(self.holdings_file, 'r') as f:
                    data = json.load(f)

                for h in data.get('holdings', []):
                    entry_time = datetime.fromisoformat(h['entry_time']) if isinstance(h['entry_time'], str) else h['entry_time']
                    holding = MomentumHolding(
                        product_id=h['product_id'],
                        symbol=h['symbol'],
                        quantity=h['quantity'],
                        entry_price=h['entry_price'],
                        entry_time=entry_time,
                        entry_change_pct=h.get('entry_change_pct', 0)
                    )
                    self.holdings[h['product_id']] = holding

                self.last_scan_time = datetime.fromisoformat(data['last_scan_time']) if data.get('last_scan_time') else None
                logger.info(f"Loaded {len(self.holdings)} holdings from {self.holdings_file}")

        except Exception as e:
            logger.warning(f"Could not load holdings: {e}")
            self.holdings = {}

    def _save_holdings(self):
        """Save holdings to JSON file"""
        try:
            data = {
                'holdings': [h.to_dict() for h in self.holdings.values()],
                'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
                'updated': datetime.now(timezone.utc).isoformat()
            }

            with open(self.holdings_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self.holdings)} holdings to {self.holdings_file}")

        except Exception as e:
            logger.error(f"Could not save holdings: {e}")

    def _log_trade_to_central_db(self, signal: Dict, is_close: bool = False):
        """Write trade to central trading_master.db for orchestrator visibility"""
        try:
            conn = sqlite3.connect(self.central_db_path)
            cursor = conn.cursor()

            if is_close:
                # Close existing open trade
                symbol = signal.get('symbol', signal.get('product_id', ''))
                exit_price = signal.get('price', 0)
                pnl = signal.get('pnl', 0)
                pnl_pct = signal.get('pnl_pct', 0)
                now = datetime.now(timezone.utc).isoformat()

                cursor.execute('''
                    SELECT trade_id, entry_price FROM trades
                    WHERE bot_name = 'Multi-Momentum' AND symbol = ? AND status = 'open'
                    ORDER BY entry_time DESC LIMIT 1
                ''', (symbol,))
                row = cursor.fetchone()
                if row:
                    trade_id = row[0]
                    cursor.execute('''
                        UPDATE trades SET exit_price = ?, exit_time = ?, pnl = ?, pnl_pct = ?, status = 'closed'
                        WHERE trade_id = ?
                    ''', (exit_price, now, pnl, pnl_pct, trade_id))
                    logger.info(f"[DB] Closed trade {trade_id} in central DB | PnL: ${pnl:.2f}")
                else:
                    logger.warning(f"[DB] No open trade found for Multi-Momentum {symbol}")
            else:
                # Open new trade
                symbol = signal.get('symbol', signal.get('product_id', ''))
                trade_id = f"Multi-Momentum_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
                now = datetime.now(timezone.utc).isoformat()

                cursor.execute('''
                    INSERT OR IGNORE INTO trades (trade_id, bot_name, market, symbol, side, entry_price,
                        quantity, entry_time, pnl, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (trade_id, 'Multi-Momentum', 'crypto', symbol,
                      signal.get('action', 'buy'), signal.get('price', 0),
                      signal.get('quantity', 0), now, 0, 'open'))
                logger.info(f"[DB] Opened trade {trade_id} in central DB")

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[DB] Failed to log trade to central DB: {e}")

    def get_24h_performance(self) -> List[CoinPerformance]:
        """
        Scan all USDC pairs and rank by 24h performance.

        Returns:
            List of CoinPerformance sorted by change_24h_pct descending
        """
        usdc_pairs = self.get_usdc_pairs()
        performances = []

        logger.info(f"Scanning {len(usdc_pairs)} USDC pairs for momentum...")

        # Get current prices
        current_prices = self.get_prices(usdc_pairs)

        for product_id in usdc_pairs:
            symbol = product_id.replace('/USD', '').replace('-USDC', '')

            # Skip excluded symbols
            if symbol in self.EXCLUDED_SYMBOLS:
                continue

            price_data = current_prices.get(product_id)
            if not price_data:
                continue

            current_price = price_data['mid']
            if current_price <= 0:
                continue

            # Get 24h candle data for historical price
            try:
                candle_data = self._get_24h_candle(product_id)
                if candle_data:
                    price_24h_ago = candle_data['open']
                    volume_24h = candle_data['volume']

                    # Alpaca returns volume in coin units, convert to USD
                    volume_24h_usd = volume_24h * current_price

                    # Filter by minimum volume
                    if volume_24h_usd < self.MIN_VOLUME_USD:
                        continue

                    change_24h_pct = (current_price - price_24h_ago) / price_24h_ago if price_24h_ago > 0 else 0

                    performances.append(CoinPerformance(
                        product_id=product_id,
                        symbol=symbol,
                        current_price=current_price,
                        price_24h_ago=price_24h_ago,
                        change_24h_pct=change_24h_pct,
                        volume_24h=volume_24h_usd
                    ))

            except Exception as e:
                logger.debug(f"Could not get 24h data for {product_id}: {e}")
                continue

        # Sort by 24h change descending
        performances.sort(key=lambda x: x.change_24h_pct, reverse=True)
        self.rankings = performances

        logger.info(f"Ranked {len(performances)} coins by 24h performance")

        # Log top 10 for debugging
        for i, p in enumerate(performances[:10]):
            logger.info(f"  #{i+1} {p.symbol}: {p.change_24h_pct:+.2%} (${p.current_price:.6f})")

        return performances

    def _get_24h_candle(self, product_id: str) -> Optional[Dict]:
        """
        Get 24h candle data for a product via Alpaca.

        Returns:
            Dict with 'open', 'close', 'high', 'low', 'volume' or None
        """
        alpaca = getattr(self, 'alpaca', None) or getattr(self, 'coinbase', None)
        if not alpaca or not alpaca._initialized:
            if not self.paper_mode:
                logger.error(f"LIVE MODE: Alpaca not initialized — refusing to use mock candle data for {product_id}")
                return None
            # Return mock data for paper trading only
            import random
            mock_price = self.price_cache.get(product_id, {}).get('mid', 1.0)
            change = random.uniform(-0.15, 0.25)  # -15% to +25% mock change
            return {
                'open': mock_price / (1 + change),
                'close': mock_price,
                'high': mock_price * 1.05,
                'low': mock_price * 0.95,
                'volume': random.uniform(100000, 10000000)
            }

        try:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame

            alpaca_sym = AlpacaCryptoClient.to_alpaca_symbol(product_id)
            data_client = alpaca.data_client or CryptoHistoricalDataClient()

            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=25)

            request = CryptoBarsRequest(
                symbol_or_symbols=alpaca_sym,
                timeframe=TimeFrame.Day,
                start=start_time,
                end=end_time
            )
            bars = data_client.get_crypto_bars(request)

            if hasattr(bars, 'df') and len(bars.df) > 0:
                df = bars.df.reset_index()
                row = df.iloc[-1]
                return {
                    'open': float(row['open']),
                    'close': float(row['close']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'volume': float(row['volume'])
                }

        except Exception as e:
            logger.debug(f"Candle fetch error for {product_id}: {e}")

        return None

    def _should_rotate(self) -> bool:
        """Check if it's time to rotate positions"""
        if self.last_scan_time is None:
            return True

        hours_since_scan = (datetime.now(timezone.utc) - self.last_scan_time).total_seconds() / 3600
        return hours_since_scan >= self.SCAN_INTERVAL_HOURS

    def _check_max_hold_time(self) -> List[str]:
        """Check for positions that have exceeded max hold time"""
        expired = []
        now = datetime.now(timezone.utc)

        for product_id, holding in self.holdings.items():
            hours_held = (now - holding.entry_time).total_seconds() / 3600
            if hours_held >= self.MAX_HOLD_HOURS:
                expired.append(product_id)
                logger.info(f"{holding.symbol} exceeded {self.MAX_HOLD_HOURS}h max hold time")

        return expired

    def run_scan(self) -> List[Dict]:
        """
        Main scan method - rotates into top performers.

        Returns:
            List of trade signals (buys and sells)
        """
        signals = []

        logger.info("=" * 60)
        logger.info("MULTI-COIN MOMENTUM SCAN")
        logger.info("=" * 60)

        # Check if we should rotate
        if not self._should_rotate():
            hours_since = (datetime.now(timezone.utc) - self.last_scan_time).total_seconds() / 3600 if self.last_scan_time else 0
            logger.info(f"Rotation not due yet. Last scan: {hours_since:.1f}h ago. Next in {self.SCAN_INTERVAL_HOURS - hours_since:.1f}h")

            # Still update current holdings with latest prices + check stop-losses
            stop_loss_exits = self._update_holdings_prices()
            if stop_loss_exits:
                signals.extend(stop_loss_exits)
            return signals

        # Get 24h performance rankings
        rankings = self.get_24h_performance()

        if not rankings:
            logger.warning("No coins to rank - skipping rotation")
            return signals

        # Get top N performers
        top_performers = rankings[:self.TOP_N]
        top_product_ids = {p.product_id for p in top_performers}

        logger.info(f"\nTop {self.TOP_N} performers:")
        for i, p in enumerate(top_performers):
            logger.info(f"  #{i+1} {p.symbol}: {p.change_24h_pct:+.2%}")

        # Check for max hold time violations
        expired_positions = self._check_max_hold_time()

        # Determine sells: holdings not in top N OR expired
        positions_to_sell = []
        for product_id, holding in self.holdings.items():
            if product_id not in top_product_ids or product_id in expired_positions:
                positions_to_sell.append(product_id)

        # Execute sells first to free up capital
        for product_id in positions_to_sell:
            holding = self.holdings[product_id]
            sell_signal = self._execute_sell(holding)
            if sell_signal:
                signals.append(sell_signal)

        # Determine buys: top N not currently held
        positions_to_buy = []
        for perf in top_performers:
            if perf.product_id not in self.holdings:
                positions_to_buy.append(perf)

        # Calculate position size for new buys
        available_capital = self._get_available_capital()
        num_positions_needed = self.TOP_N - len(self.holdings) + len(positions_to_sell)

        if num_positions_needed > 0 and positions_to_buy:
            position_size = available_capital / min(num_positions_needed, len(positions_to_buy))
            position_size = min(position_size, self.capital / self.TOP_N)  # Cap at equal split

            for perf in positions_to_buy[:num_positions_needed]:
                buy_signal = self._execute_buy(perf, position_size)
                if buy_signal:
                    signals.append(buy_signal)

        # Update scan time
        self.last_scan_time = datetime.now(timezone.utc)

        # Save holdings
        self._save_holdings()

        # Log summary
        logger.info(f"\nRotation complete: {len([s for s in signals if s['action'] == 'sell'])} sells, "
                   f"{len([s for s in signals if s['action'] == 'buy'])} buys")
        logger.info(f"Current holdings: {len(self.holdings)}")

        return signals

    def _get_available_capital(self) -> float:
        """Calculate available capital for new positions"""
        if self.paper_mode:
            # In paper mode, track capital minus current position values
            positions_value = sum(
                h.quantity * h.current_price
                for h in self.holdings.values()
                if h.current_price > 0
            )
            # Use entry prices if current prices not set
            if positions_value == 0:
                positions_value = sum(
                    h.quantity * h.entry_price
                    for h in self.holdings.values()
                )
            return max(0, self.capital - positions_value)
        else:
            alpaca = getattr(self, 'alpaca', None) or getattr(self, 'coinbase', None)
            return alpaca.get_usdc_balance() if alpaca else 0.0

    def _update_holdings_prices(self) -> List[Dict]:
        """Update current prices and P&L for all holdings. Returns stop-loss exit signals."""
        stop_loss_exits = []
        if not self.holdings:
            return stop_loss_exits

        product_ids = list(self.holdings.keys())
        prices = self.get_prices(product_ids)

        for product_id, holding in list(self.holdings.items()):
            price_data = prices.get(product_id)
            if price_data:
                # Get current 24h change
                candle = self._get_24h_candle(product_id)
                current_change = 0
                if candle:
                    current_change = (price_data['mid'] - candle['open']) / candle['open'] if candle['open'] > 0 else 0

                holding.update(price_data['mid'], current_change)

                # CHECK STOP-LOSS: exit if position is down more than STOP_LOSS_PCT
                if holding.pnl_pct <= -self.STOP_LOSS_PCT:
                    logger.warning(
                        f"STOP-LOSS HIT: {holding.symbol} at {holding.pnl_pct:+.2%} "
                        f"(threshold: -{self.STOP_LOSS_PCT:.0%}) — closing position"
                    )
                    sell_signal = self._execute_sell(holding)
                    if sell_signal:
                        sell_signal['reason'] = f'Stop-loss: {holding.pnl_pct:+.2%} <= -{self.STOP_LOSS_PCT:.0%}'
                        stop_loss_exits.append(sell_signal)

        self._save_holdings()
        return stop_loss_exits

    def _execute_buy(self, perf: CoinPerformance, amount_usd: float) -> Optional[Dict]:
        """Execute a buy for a top performer"""
        if amount_usd < 1:
            logger.warning(f"Insufficient capital to buy {perf.symbol}")
            return None

        quantity = amount_usd / perf.current_price

        signal = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'buy',  # lowercase for master_orchestrator compatibility
            'product_id': perf.product_id,
            'symbol': perf.symbol,
            'price': perf.current_price,
            'quantity': quantity,
            'amount_usd': amount_usd,
            'change_24h_pct': perf.change_24h_pct,
            'reason': f'Top {self.TOP_N} performer ({perf.change_24h_pct:+.2%})',
            'paper': self.paper_mode,
            'status': 'pending'
        }

        if self.paper_mode:
            signal['status'] = 'filled'

            # Create holding
            self.holdings[perf.product_id] = MomentumHolding(
                product_id=perf.product_id,
                symbol=perf.symbol,
                quantity=quantity,
                entry_price=perf.current_price,
                entry_time=datetime.now(timezone.utc),
                entry_change_pct=perf.change_24h_pct,
                current_price=perf.current_price,
                current_change_pct=perf.change_24h_pct
            )

            logger.info(
                f"[PAPER] BUY {perf.symbol}: {quantity:.8f} @ ${perf.current_price:.6f} "
                f"(${amount_usd:.2f}) | 24h: {perf.change_24h_pct:+.2%}"
            )

        else:
            try:
                alpaca = getattr(self, 'alpaca', None) or getattr(self, 'coinbase', None)
                order = alpaca.create_market_order(
                    product_id=perf.product_id,
                    side='BUY',
                    quote_size=str(round(amount_usd, 2))
                )

                if order and order.get('success', True):
                    signal['order_id'] = order.get('order_id', str(order))
                    signal['status'] = 'submitted'

                    self.holdings[perf.product_id] = MomentumHolding(
                        product_id=perf.product_id,
                        symbol=perf.symbol,
                        quantity=quantity,
                        entry_price=perf.current_price,
                        entry_time=datetime.now(timezone.utc),
                        entry_change_pct=perf.change_24h_pct,
                        current_price=perf.current_price,
                        current_change_pct=perf.change_24h_pct
                    )

                    logger.info(f"[LIVE] BUY {perf.symbol}: Order submitted")
                else:
                    signal['status'] = 'failed'

            except Exception as e:
                signal['status'] = 'error'
                signal['error'] = str(e)
                logger.error(f"Buy order error for {perf.symbol}: {e}")

        self.trades.append(signal)

        # NOTE: Central DB writes handled by orchestrator's _log_trade_from_signal()
        # Removed _log_trade_to_central_db() call here to prevent duplicate DB entries

        return signal

    def _execute_sell(self, holding: MomentumHolding) -> Optional[Dict]:
        """Execute a sell to rotate out of a position"""
        # Get current price
        prices = self.get_prices([holding.product_id])
        price_data = prices.get(holding.product_id)

        if not price_data:
            logger.error(f"Could not get price for {holding.product_id}")
            return None

        current_price = price_data['bid']  # Use bid for selling
        holding.update(current_price, holding.current_change_pct)

        signal = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'sell',  # lowercase for master_orchestrator compatibility
            'product_id': holding.product_id,
            'symbol': holding.symbol,
            'price': current_price,
            'quantity': holding.quantity,
            'amount_usd': holding.quantity * current_price,
            'entry_price': holding.entry_price,
            'pnl': holding.pnl,
            'pnl_pct': holding.pnl_pct,
            'reason': 'Dropped out of top performers / rotation',
            'paper': self.paper_mode,
            'status': 'pending'
        }

        if self.paper_mode:
            signal['status'] = 'filled'

            logger.info(
                f"[PAPER] SELL {holding.symbol}: {holding.quantity:.8f} @ ${current_price:.6f} | "
                f"P&L: ${holding.pnl:.2f} ({holding.pnl_pct:+.2%})"
            )

            # Remove holding
            del self.holdings[holding.product_id]

        else:
            try:
                alpaca = getattr(self, 'alpaca', None) or getattr(self, 'coinbase', None)
                order = alpaca.create_market_order(
                    product_id=holding.product_id,
                    side='SELL',
                    base_size=str(round(holding.quantity, 8))
                )

                if order and order.get('success', True):
                    signal['order_id'] = order.get('order_id', str(order))
                    signal['status'] = 'submitted'

                    del self.holdings[holding.product_id]
                    logger.info(f"[LIVE] SELL {holding.symbol}: Order submitted")
                else:
                    signal['status'] = 'failed'

            except Exception as e:
                signal['status'] = 'error'
                signal['error'] = str(e)
                logger.error(f"Sell order error for {holding.symbol}: {e}")

        self.trades.append(signal)

        # NOTE: Central DB writes handled by orchestrator's _log_trade_from_signal()
        # Removed _log_trade_to_central_db() call here to prevent duplicate DB entries

        return signal

    def get_status(self) -> Dict:
        """Get bot status"""
        # Update holdings with current prices
        self._update_holdings_prices()

        total_pnl = sum(h.pnl for h in self.holdings.values())
        total_value = sum(h.quantity * h.current_price for h in self.holdings.values() if h.current_price > 0)

        # Use entry prices if current not set
        if total_value == 0:
            total_value = sum(h.quantity * h.entry_price for h in self.holdings.values())

        holdings_info = {}
        for product_id, h in self.holdings.items():
            holdings_info[h.symbol] = {
                'quantity': h.quantity,
                'entry_price': h.entry_price,
                'current_price': h.current_price,
                'entry_change_pct': h.entry_change_pct,
                'current_change_pct': h.current_change_pct,
                'pnl': h.pnl,
                'pnl_pct': h.pnl_pct,
                'hours_held': (datetime.now(timezone.utc) - h.entry_time).total_seconds() / 3600
            }

        hours_since_scan = 0
        if self.last_scan_time:
            hours_since_scan = (datetime.now(timezone.utc) - self.last_scan_time).total_seconds() / 3600

        return {
            'name': 'MultiMomentumBot',
            'capital': self.capital,
            'paper_mode': self.paper_mode,
            'alpaca_connected': self.alpaca._initialized if hasattr(self, 'alpaca') and self.alpaca else False,
            'holdings_count': len(self.holdings),
            'holdings': holdings_info,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'available_capital': self._get_available_capital(),
            'last_scan': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'hours_since_scan': hours_since_scan,
            'next_scan_in': max(0, self.SCAN_INTERVAL_HOURS - hours_since_scan),
            'trades_today': len(self.trades),
            'top_performers': [
                {'symbol': p.symbol, 'change_24h': p.change_24h_pct}
                for p in self.rankings[:5]
            ] if self.rankings else []
        }


def main():
    """Test the Multi-Momentum bot"""
    logger.info("=" * 60)
    logger.info("MULTI-COIN MOMENTUM BOT")
    logger.info("=" * 60)
    logger.info("\nStrategy: Rotate into top 3 performing coins every 4 hours")
    logger.info("Risk: HIGH - No stop-loss, momentum chasing")
    logger.info("=" * 60)

    bot = MultiMomentumBot(capital=300.0, paper_mode=True)

    logger.info(f"\nInitial Status:")
    status = bot.get_status()
    logger.info(f"  Capital: ${status['capital']}")
    logger.info(f"  Holdings: {status['holdings_count']}")
    logger.info(f"  Paper Mode: {status['paper_mode']}")

    logger.info("\n--- Running Momentum Scan ---")
    signals = bot.run_scan()

    logger.info(f"\nGenerated {len(signals)} signals:")
    for signal in signals:
        action = signal['action']
        symbol = signal['symbol']
        price = signal['price']
        amount = signal.get('amount_usd', signal.get('quantity', 0) * price)
        reason = signal.get('reason', '')

        if action == 'buy':
            logger.info(f"  BUY {symbol}: ${amount:.2f} @ ${price:.6f} - {reason}")
        else:
            pnl = signal.get('pnl', 0)
            pnl_pct = signal.get('pnl_pct', 0)
            logger.info(f"  SELL {symbol}: ${amount:.2f} @ ${price:.6f} | P&L: ${pnl:.2f} ({pnl_pct:+.2%})")

    logger.info("\n--- Final Status ---")
    status = bot.get_status()
    logger.info(f"  Holdings: {status['holdings_count']}")
    logger.info(f"  Total Value: ${status['total_value']:.2f}")
    logger.info(f"  Total P&L: ${status['total_pnl']:.2f}")

    if status['holdings']:
        logger.info("\n  Current Positions:")
        for symbol, info in status['holdings'].items():
            logger.info(f"    {symbol}: {info['quantity']:.8f} @ ${info['entry_price']:.6f} "
                  f"({info['entry_change_pct']:+.2%} at entry)")

    if status['top_performers']:
        logger.info("\n  Top 5 Performers:")
        for i, p in enumerate(status['top_performers']):
            logger.info(f"    #{i+1} {p['symbol']}: {p['change_24h']:+.2%}")

    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()
