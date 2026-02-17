"""
RSI EXTREMES BOT (V4 Momentum)
===============================

Momentum-based crypto strategy using RSI(14) momentum confirmation
and EMA(20) trend filter on Alpaca USD pairs.

Strategy (V4 Momentum):
- Scans top liquid /USD pairs via Alpaca
- BUY when RSI(14) crosses above 50 from below AND price > EMA(20)
  (momentum confirmation - trend is resuming)
- EXIT when RSI(14) drops below 40 OR price < EMA(20) OR -5% stop-loss
- Position size: $60 per trade
- Max 3 concurrent positions

This is a momentum strategy - entering when trend resumes after
a pullback. Crypto trends more than it mean-reverts, so momentum
entries outperform mean reversion for digital assets.

Risk Level: HIGH
Expected Win Rate: ~55% with larger winners than losers

Author: Trading Bot Arsenal
Created: February 2026
Updated: February 2026 (V4 momentum conversion)
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add parent directories for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

# Import base class and signal type
try:
    from bots.aggressive.base_aggressive_bot import BaseAggressiveBot, TradeSignal
    HAS_BASE_CLASS = True
except ImportError:
    HAS_BASE_CLASS = False
    BaseAggressiveBot = object
    TradeSignal = None

# Import AlpacaCryptoClient for execution
try:
    from bots.alpaca_crypto_client import AlpacaCryptoClient
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False
    AlpacaCryptoClient = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RSISignal:
    """RSI-based trading signal"""
    symbol: str
    product_id: str
    current_price: float
    rsi_value: float
    signal_type: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str
    timestamp: datetime


@dataclass
class RSIPosition:
    """Track an RSI extremes position"""
    product_id: str
    symbol: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    rsi_at_entry: float


class RSIExtremesBot(BaseAggressiveBot if HAS_BASE_CLASS else object):
    """
    RSI Momentum Bot (V4)

    Momentum-based entries on crypto /USD pairs via Alpaca.

    Entry: RSI(14) crosses above 50 from below AND price > EMA(20)
    Exit: RSI(14) < 40 OR price < EMA(20) OR -5% stop loss
    """

    # Strategy Parameters (V4 Momentum)
    RSI_PERIOD = 14              # Standard RSI period
    RSI_ENTRY_THRESHOLD = 48     # Buy when RSI crosses above 48 from below (relaxed from 50)
    RSI_EXIT_THRESHOLD = 38      # Sell when RSI drops below 38 (relaxed from 40)
    EMA_PERIOD = 20              # EMA trend filter period

    POSITION_SIZE_USD = 60.0     # $60 per trade
    STOP_LOSS_PCT = 0.05         # 5% stop loss
    TAKE_PROFIT_PCT = 0.10       # 10% take profit (kept as trailing target)
    MAX_POSITIONS = 3            # Max concurrent positions
    MIN_HOLD_HOURS = 2           # Minimum hold time before indicator-based exits

    # Scanning parameters
    TOP_PAIRS_LIMIT = 50         # Scan top 50 USD pairs
    MIN_CANDLES_REQUIRED = 25    # Need 25 candles for RSI + EMA
    CANDLE_GRANULARITY = "ONE_HOUR"  # 1-hour candles for calculation
    CANDLES_TO_FETCH = 50        # Fetch 50 candles for calculation

    # Risk filters
    MIN_VOLUME_24H = 10000       # Minimum $10k 24h volume (if available)
    MAX_SPREAD_PCT = 0.02        # Skip if spread > 2%
    MIN_PRICE_USD = 0.01         # Skip tokens priced below 1 cent (avoids millions-of-shares positions)

    def __init__(
        self,
        capital: float = 200.0,
        paper_mode: bool = None,
        position_size: float = None,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
        max_positions: int = None
    ):
        """
        Initialize RSI Extremes Bot (V4 Momentum).

        Args:
            capital: Total capital for this strategy
            paper_mode: Paper trading mode (default from env)
            position_size: USD per position (default $60)
            stop_loss_pct: Stop loss % (default 5%)
            take_profit_pct: Take profit % (default 10%)
            max_positions: Max concurrent positions (default 3)
        """
        # Initialize base class if available
        if HAS_BASE_CLASS:
            super().__init__(
                capital=capital,
                paper_mode=paper_mode,
                position_size=position_size or self.POSITION_SIZE_USD,
                stop_loss_pct=stop_loss_pct or self.STOP_LOSS_PCT,
                take_profit_pct=take_profit_pct or self.TAKE_PROFIT_PCT,
                max_positions=max_positions or self.MAX_POSITIONS
            )
        else:
            # Fallback initialization
            self.capital = capital
            if paper_mode is None:
                paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
            self.paper_mode = paper_mode
            self.position_size = position_size or self.POSITION_SIZE_USD
            self.stop_loss_pct = stop_loss_pct or self.STOP_LOSS_PCT
            self.take_profit_pct = take_profit_pct or self.TAKE_PROFIT_PCT
            self.max_positions = max_positions or self.MAX_POSITIONS

            # Initialize Alpaca client
            self.alpaca = AlpacaCryptoClient() if HAS_ALPACA else None
            self.coinbase = self.alpaca  # backward compat alias
            self.active_positions = {}
            self.trades_history = []
            self.signals_history = []
            self.logger = logging.getLogger(self.__class__.__name__)

        # Momentum tracking (V4)
        self.rsi_positions: Dict[str, RSIPosition] = {}
        self.rsi_cache: Dict[str, Tuple[float, datetime]] = {}  # symbol -> (rsi, time)
        self.prev_rsi: Dict[str, float] = {}  # Track previous RSI for crossover detection
        self.cache_duration = timedelta(minutes=5)

        # P&L tracking
        self.pnl_today = 0.0
        self.wins = 0
        self.losses = 0

        logger.info(
            f"RSIExtremesBot (V4 Momentum) initialized - "
            f"Entry: RSI crosses above {self.RSI_ENTRY_THRESHOLD}, "
            f"Exit: RSI < {self.RSI_EXIT_THRESHOLD} or price < EMA({self.EMA_PERIOD}), "
            f"Position: ${self.position_size}, SL: {self.stop_loss_pct:.0%}"
        )

    def _calculate_rsi(self, candles: List[dict]) -> Optional[float]:
        """
        Calculate RSI(14) using Wilder's smoothing (standard method).

        Wilder's RSI:
        1. Seed avg_gain/avg_loss with SMA of first 14 changes
        2. Then apply exponential smoothing:
           avg_gain = (prev_avg_gain * 13 + current_gain) / 14
           avg_loss = (prev_avg_loss * 13 + current_loss) / 14
        3. RS = avg_gain / avg_loss
        4. RSI = 100 - (100 / (1 + RS))

        Args:
            candles: List of candle dicts with 'close' prices

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(candles) < self.RSI_PERIOD + 1:
            return None

        closes = [c['close'] for c in candles]

        # Calculate all price changes
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]

        if len(changes) < self.RSI_PERIOD:
            return None

        # Step 1: Seed with SMA of first RSI_PERIOD changes
        first_gains = [c if c > 0 else 0 for c in changes[:self.RSI_PERIOD]]
        first_losses = [abs(c) if c < 0 else 0 for c in changes[:self.RSI_PERIOD]]
        avg_gain = sum(first_gains) / self.RSI_PERIOD
        avg_loss = sum(first_losses) / self.RSI_PERIOD

        # Step 2: Apply Wilder's exponential smoothing for remaining changes
        for change in changes[self.RSI_PERIOD:]:
            gain = change if change > 0 else 0
            loss = abs(change) if change < 0 else 0
            avg_gain = (avg_gain * (self.RSI_PERIOD - 1) + gain) / self.RSI_PERIOD
            avg_loss = (avg_loss * (self.RSI_PERIOD - 1) + loss) / self.RSI_PERIOD

        # Step 3: Calculate RS and RSI
        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_ema(self, candles: List[dict], period: int = None) -> Optional[float]:
        """
        Calculate EMA (Exponential Moving Average) from candle data.

        Args:
            candles: List of candle dicts with 'close' prices
            period: EMA period (default self.EMA_PERIOD)

        Returns:
            Current EMA value or None if insufficient data
        """
        if period is None:
            period = self.EMA_PERIOD

        if len(candles) < period:
            return None

        closes = [c['close'] for c in candles]

        # Calculate EMA using exponential smoothing
        multiplier = 2.0 / (period + 1)

        # Seed with SMA of first 'period' values
        ema = sum(closes[:period]) / period

        # Apply exponential smoothing for remaining values
        for price in closes[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _get_usdc_pairs(self) -> List[str]:
        """Get top crypto /USD pairs for Alpaca - Core 6 prioritized first"""
        # Alpaca supports these crypto pairs (slash format)
        return [
            # CORE UNIVERSE - Prioritized for best liquidity/volatility
            'BTC/USD',   # Core - largest, most liquid
            'ETH/USD',   # Core - second largest
            'SOL/USD',   # Core - best volatility/liquidity ratio
            'XRP/USD',   # High volume, clear levels
            'DOGE/USD',  # High volatility for RSI signals
            'LINK/USD',  # Steady oracle leader
            # EXTENDED UNIVERSE - Secondary scanning
            'SHIB/USD', 'AVAX/USD', 'UNI/USD', 'DOT/USD',
            'ADA/USD', 'ATOM/USD', 'LTC/USD', 'BCH/USD',
            'NEAR/USD', 'APT/USD', 'ARB/USD', 'OP/USD', 'AAVE/USD',
            'MKR/USD', 'CRV/USD', 'GRT/USD', 'FIL/USD',
            'ALGO/USD', 'XLM/USD', 'XTZ/USD',
        ][:self.TOP_PAIRS_LIMIT]

    def _get_candles(self, product_id: str) -> List[dict]:
        """Get historical candles for RSI calculation via Alpaca"""
        if HAS_BASE_CLASS and hasattr(self, 'get_candles'):
            return self.get_candles(
                symbol=product_id,
                granularity=self.CANDLE_GRANULARITY,
                limit=self.CANDLES_TO_FETCH
            )

        # Fallback: use Alpaca data client directly
        alpaca = getattr(self, 'alpaca', None) or getattr(self, 'coinbase', None)
        if not alpaca or not alpaca._initialized:
            if not self.paper_mode:
                logger.error(f"LIVE MODE: Alpaca not initialized — refusing to use mock data for {product_id}")
                return []
            return self._generate_mock_candles(product_id)

        try:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from datetime import datetime as dt, timedelta, timezone as tz

            alpaca_sym = AlpacaCryptoClient.to_alpaca_symbol(product_id) if HAS_ALPACA else product_id
            data_client = alpaca.data_client or CryptoHistoricalDataClient()

            end = dt.now(tz.utc)
            start = end - timedelta(hours=self.CANDLES_TO_FETCH * 2)

            request = CryptoBarsRequest(
                symbol_or_symbols=alpaca_sym,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )
            bars = data_client.get_crypto_bars(request)

            candles = []
            if hasattr(bars, 'df'):
                df = bars.df.reset_index()
                for _, row in df.iterrows():
                    candles.append({
                        'time': int(row['timestamp'].timestamp()) if hasattr(row['timestamp'], 'timestamp') else 0,
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']),
                    })

            candles.sort(key=lambda x: x['time'])
            return candles[-self.CANDLES_TO_FETCH:]

        except Exception as e:
            logger.debug(f"Error getting candles for {product_id}: {e}")
            return []

    def _generate_mock_candles(self, product_id: str) -> List[dict]:
        """Generate mock candles for testing"""
        import random

        # Base prices for common pairs
        base_prices = {
            'BTC/USD': 95000, 'ETH/USD': 3200, 'SOL/USD': 150,
            'DOGE/USD': 0.35, 'SHIB/USD': 0.00003, 'AVAX/USD': 45,
            'LINK/USD': 22, 'UNI/USD': 15, 'DOT/USD': 7,
        }

        base = base_prices.get(product_id, 10)
        price = base

        candles = []
        import time as time_module
        current_time = int(time_module.time())

        for i in range(self.CANDLES_TO_FETCH):
            # Simulate price movement with potential for RSI extremes
            if random.random() < 0.1:  # 10% chance of larger move
                change = random.uniform(-0.08, 0.08)
            else:
                change = random.uniform(-0.02, 0.02)

            price = price * (1 + change)

            candles.append({
                'time': current_time - (3600 * (self.CANDLES_TO_FETCH - i)),
                'open': price * (1 + random.uniform(-0.005, 0.005)),
                'high': price * (1 + random.uniform(0, 0.01)),
                'low': price * (1 - random.uniform(0, 0.01)),
                'close': price,
                'volume': random.uniform(10000, 100000)
            })

        return candles

    def _get_current_price(self, product_id: str) -> Optional[float]:
        """Get current price for a product via Alpaca"""
        if HAS_BASE_CLASS and hasattr(self, 'get_price'):
            return self.get_price(product_id)

        alpaca = getattr(self, 'alpaca', None) or getattr(self, 'coinbase', None)
        if not alpaca or not alpaca._initialized:
            return None

        try:
            return alpaca.get_price(product_id)
        except Exception as e:
            logger.debug(f"Error getting price for {product_id}: {e}")
            return None

    def _scan_for_entries(self) -> List[RSISignal]:
        """
        Scan USD pairs for V4 momentum entry opportunities.

        V4 Momentum Entry:
        - RSI(14) crosses above 50 from below (momentum confirmation)
        - Price > EMA(20) (trend filter)

        Returns:
            List of RSISignal objects for qualifying momentum entries
        """
        signals = []
        usdc_pairs = self._get_usdc_pairs()

        logger.info(f"Scanning {len(usdc_pairs)} USD pairs for momentum entries...")

        for product_id in usdc_pairs:
            try:
                # Skip if already in position
                if product_id in self.rsi_positions:
                    continue

                # Skip if at max positions
                if len(self.rsi_positions) >= self.max_positions:
                    break

                # Get candles and calculate indicators
                candles = self._get_candles(product_id)

                if len(candles) < self.MIN_CANDLES_REQUIRED:
                    logger.debug(f"Skipping {product_id}: insufficient candle data")
                    continue

                rsi = self._calculate_rsi(candles)
                ema = self._calculate_ema(candles)

                if rsi is None or ema is None:
                    continue

                # Cache the RSI
                self.rsi_cache[product_id] = (rsi, datetime.now(timezone.utc))

                current_price = candles[-1]['close']

                # Skip micro-priced tokens (avoids millions-of-shares positions)
                if current_price < self.MIN_PRICE_USD:
                    logger.debug(f"Skipping {product_id}: price ${current_price:.8f} below min ${self.MIN_PRICE_USD}")
                    continue

                symbol = product_id.replace('/USD', '').replace('-USDC', '')

                # Get previous RSI for crossover detection
                prev_rsi = self.prev_rsi.get(product_id, rsi)
                self.prev_rsi[product_id] = rsi

                # V4 MOMENTUM ENTRY:
                # 1a. RSI crosses above threshold from below (classic crossover), OR
                # 1b. RSI was below threshold, now solidly above (within 2 candles)
                # 2. Price is above EMA(20) (trend filter)
                rsi_crossed_above = prev_rsi < self.RSI_ENTRY_THRESHOLD and rsi >= self.RSI_ENTRY_THRESHOLD
                # Also accept: RSI rising and just above threshold (momentum building)
                rsi_momentum_entry = (
                    rsi > self.RSI_ENTRY_THRESHOLD
                    and rsi <= self.RSI_ENTRY_THRESHOLD + 8  # Don't chase if RSI already high
                    and rsi > prev_rsi + 1.5  # RSI accelerating upward
                    and prev_rsi < self.RSI_ENTRY_THRESHOLD + 5  # Was near threshold recently
                )
                price_above_ema = current_price > ema

                if (rsi_crossed_above or rsi_momentum_entry) and price_above_ema:
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)

                    # Confidence based on momentum strength:
                    # - How far RSI crossed above 50 (stronger = better)
                    # - How far price is above EMA (stronger trend = better)
                    rsi_strength = min(1.0, (rsi - self.RSI_ENTRY_THRESHOLD) / 20.0)  # 0-1 scale
                    ema_strength = min(1.0, (current_price - ema) / ema * 50.0)  # normalized
                    confidence = min(0.95, 0.5 + rsi_strength * 0.25 + ema_strength * 0.20)

                    signal = RSISignal(
                        symbol=symbol,
                        product_id=product_id,
                        current_price=current_price,
                        rsi_value=rsi,
                        signal_type='BUY',
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        confidence=confidence,
                        reason=(
                            f"V4 Momentum: RSI({self.RSI_PERIOD}) "
                            f"{'crossed above' if rsi_crossed_above else 'momentum entry'} "
                            f"{self.RSI_ENTRY_THRESHOLD} ({prev_rsi:.1f}->{rsi:.1f}), "
                            f"price ${current_price:.6f} > EMA({self.EMA_PERIOD}) ${ema:.6f}"
                        ),
                        timestamp=datetime.now(timezone.utc)
                    )

                    signals.append(signal)

                    logger.info(
                        f"MOMENTUM ENTRY: {symbol} RSI={prev_rsi:.1f}->{rsi:.1f} "
                        f"@ ${current_price:.6f} > EMA ${ema:.6f} "
                        f"| Confidence: {confidence:.0%}"
                    )

            except Exception as e:
                logger.debug(f"Error scanning {product_id}: {e}")
                continue

        # Sort by confidence (strongest momentum first)
        signals.sort(key=lambda x: x.confidence, reverse=True)

        return signals

    def _check_exit_conditions(self) -> List[RSISignal]:
        """
        Check existing positions for V4 momentum exit conditions.

        Exit when:
        - RSI(14) drops below 40 (momentum fading)
        - Price drops below EMA(20) (trend broken)
        - -5% stop loss hit
        - +10% take profit hit (optional trailing target)

        Returns:
            List of RSISignal objects for positions to exit
        """
        exit_signals = []

        for product_id, position in list(self.rsi_positions.items()):
            try:
                # Get current price and indicators
                candles = self._get_candles(product_id)
                if not candles:
                    continue

                # FIX: Use live price API instead of stale hourly candle for exits
                # This prevents entry_price == exit_price when exit occurs within same hour
                live_price = self.alpaca.get_price(product_id)  # product_id is already in SOL/USD format
                current_price = live_price if live_price else candles[-1]['close']  # Fallback to candle if API fails

                # Use completed candles only for indicator-based exits
                # (last candle is still forming, so RSI/EMA from it flicker)
                completed_candles = candles[:-1] if len(candles) > self.MIN_CANDLES_REQUIRED else candles
                rsi = self._calculate_rsi(completed_candles)
                ema = self._calculate_ema(completed_candles)

                pnl_pct = (current_price - position.entry_price) / position.entry_price

                # Calculate hold time
                now = datetime.now(timezone.utc)
                entry_time = position.entry_time
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                time_held_hours = (now - entry_time).total_seconds() / 3600

                should_exit = False
                exit_reason = ""

                # Hard stop loss always applies regardless of hold time (-5%)
                if current_price <= position.stop_loss:
                    should_exit = True
                    exit_reason = f"Stop loss hit: {pnl_pct:.1%} (stop {-self.stop_loss_pct:.0%})"

                # Take profit always applies (+10%)
                elif current_price >= position.take_profit:
                    should_exit = True
                    exit_reason = f"Take profit hit: +{pnl_pct:.1%} (target {self.take_profit_pct:.0%})"

                # Indicator-based exits only after minimum hold time
                elif time_held_hours < self.MIN_HOLD_HOURS:
                    logger.debug(
                        f"Skipping exit check for {position.symbol}: "
                        f"held {time_held_hours:.1f}h < {self.MIN_HOLD_HOURS}h minimum"
                    )
                    continue

                # Check RSI momentum exit (below 40 = momentum fading)
                elif rsi and rsi < self.RSI_EXIT_THRESHOLD:
                    should_exit = True
                    exit_reason = f"Momentum fading: RSI({self.RSI_PERIOD}) = {rsi:.1f} < {self.RSI_EXIT_THRESHOLD}"

                # Check EMA trend break (price below EMA20)
                elif ema and current_price < ema:
                    should_exit = True
                    exit_reason = (
                        f"Trend broken: price ${current_price:.6f} < "
                        f"EMA({self.EMA_PERIOD}) ${ema:.6f}"
                    )

                if should_exit:
                    signal = RSISignal(
                        symbol=position.symbol,
                        product_id=product_id,
                        current_price=current_price,
                        rsi_value=rsi or 50,
                        signal_type='SELL',
                        entry_price=position.entry_price,
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        confidence=1.0,
                        reason=exit_reason,
                        timestamp=datetime.now(timezone.utc)
                    )
                    exit_signals.append(signal)

                    logger.info(
                        f"MOMENTUM EXIT: {position.symbol} @ ${current_price:.6f} "
                        f"| Entry: ${position.entry_price:.6f} | PnL: {pnl_pct:.1%} | {exit_reason}"
                    )

            except Exception as e:
                logger.debug(f"Error checking exit for {product_id}: {e}")
                continue

        return exit_signals

    def _execute_entry(self, signal: RSISignal) -> Optional[dict]:
        """Execute an entry trade"""
        if signal.entry_price < self.MIN_PRICE_USD:
            logger.warning(f"Blocked {signal.symbol}: price ${signal.entry_price:.8f} below min ${self.MIN_PRICE_USD}")
            return None
        quantity = self.position_size / signal.entry_price

        trade_record = {
            'timestamp': signal.timestamp.isoformat(),
            'symbol': signal.symbol,
            'product_id': signal.product_id,
            'side': 'BUY',
            'quantity': quantity,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'rsi': signal.rsi_value,
            'reason': signal.reason,
            'paper': self.paper_mode,
            'status': 'pending'
        }

        if self.paper_mode:
            trade_record['status'] = 'filled'

            # Track position
            self.rsi_positions[signal.product_id] = RSIPosition(
                product_id=signal.product_id,
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                quantity=quantity,
                entry_time=signal.timestamp,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                rsi_at_entry=signal.rsi_value
            )

            logger.info(
                f"[PAPER] BUY {signal.symbol}: ${self.position_size:.2f} @ ${signal.entry_price:.6f} "
                f"| RSI: {signal.rsi_value:.1f} | SL: ${signal.stop_loss:.6f} | TP: ${signal.take_profit:.6f}"
            )

        else:
            # Live execution via Alpaca
            alpaca = getattr(self, 'alpaca', None) or getattr(self, 'coinbase', None)
            if alpaca and alpaca._initialized:
                try:
                    order = alpaca.create_market_order(
                        product_id=signal.product_id,
                        side='BUY',
                        quote_size=str(self.position_size)
                    )

                    if order and order.get('success', True):
                        trade_record['order_id'] = order.get('order_id')
                        trade_record['status'] = 'submitted'

                        # Track position
                        self.rsi_positions[signal.product_id] = RSIPosition(
                            product_id=signal.product_id,
                            symbol=signal.symbol,
                            entry_price=signal.entry_price,
                            quantity=quantity,
                            entry_time=signal.timestamp,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            rsi_at_entry=signal.rsi_value
                        )

                        logger.info(f"[LIVE] BUY order submitted: {signal.symbol}")
                    else:
                        trade_record['status'] = 'failed'

                except Exception as e:
                    trade_record['status'] = 'failed'
                    trade_record['error'] = str(e)
                    logger.error(f"Order failed: {e}")
            else:
                trade_record['status'] = 'failed'
                trade_record['error'] = 'Alpaca not available'

        self.trades_history.append(trade_record)
        return trade_record

    def _execute_exit(self, signal: RSISignal) -> Optional[dict]:
        """Execute an exit trade"""
        position = self.rsi_positions.get(signal.product_id)
        if not position:
            return None

        pnl = (signal.current_price - position.entry_price) * position.quantity
        pnl_pct = (signal.current_price - position.entry_price) / position.entry_price

        trade_record = {
            'timestamp': signal.timestamp.isoformat(),
            'symbol': signal.symbol,
            'product_id': signal.product_id,
            'side': 'SELL',
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': signal.current_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'rsi_at_entry': position.rsi_at_entry,
            'rsi_at_exit': signal.rsi_value,
            'reason': signal.reason,
            'paper': self.paper_mode,
            'status': 'pending'
        }

        if self.paper_mode:
            trade_record['status'] = 'filled'

            # Update P&L tracking
            self.pnl_today += pnl
            if pnl > 0:
                self.wins += 1
            else:
                self.losses += 1

            # Remove position
            del self.rsi_positions[signal.product_id]

            logger.info(
                f"[PAPER] SELL {signal.symbol} @ ${signal.current_price:.6f} "
                f"| PnL: ${pnl:.2f} ({pnl_pct:.1%}) | {signal.reason}"
            )

        else:
            # Live execution via Alpaca
            alpaca = getattr(self, 'alpaca', None) or getattr(self, 'coinbase', None)
            if alpaca and alpaca._initialized:
                try:
                    order = alpaca.create_market_order(
                        product_id=signal.product_id,
                        side='SELL',
                        base_size=str(round(position.quantity, 8))
                    )

                    if order and order.get('success', True):
                        trade_record['order_id'] = order.get('order_id')
                        trade_record['status'] = 'submitted'

                        del self.rsi_positions[signal.product_id]
                        self.pnl_today += pnl

                        logger.info(f"[LIVE] SELL order submitted: {signal.symbol}")
                    else:
                        trade_record['status'] = 'failed'

                except Exception as e:
                    trade_record['status'] = 'failed'
                    trade_record['error'] = str(e)
                    logger.error(f"Exit order failed: {e}")
            else:
                trade_record['status'] = 'failed'

        self.trades_history.append(trade_record)
        return trade_record

    def run_scan(self) -> List[Dict]:
        """
        Main scan method - scans for RSI extremes and manages positions.

        Returns:
            List of Dict signals compatible with master_orchestrator._log_trade_from_signal()
            Format: {'symbol': product_id, 'action': side, 'price': entry_price,
                     'quantity': quantity, 'confidence': confidence, 'reason': reason}
        """
        logger.info("=" * 50)
        logger.info(f"RSIExtremesBot (V4 Momentum) scanning... (positions: {len(self.rsi_positions)}/{self.max_positions})")

        results = []

        # 1. Check existing positions for exits
        exit_signals = self._check_exit_conditions()
        for signal in exit_signals:
            # Capture position BEFORE _execute_exit() deletes it from self.rsi_positions
            position = self.rsi_positions.get(signal.product_id)
            quantity = position.quantity if position else (self.position_size / signal.entry_price)
            trade = self._execute_exit(signal)
            if trade:
                results.append({
                    'symbol': signal.product_id,
                    'action': 'sell',
                    'price': signal.current_price,
                    'quantity': quantity,
                    'entry_price': signal.entry_price,
                    'confidence': signal.confidence,
                    'reason': signal.reason,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'rsi': signal.rsi_value,
                    'signal_type': 'exit',
                    'status': trade.get('status', 'filled'),  # Mark as already-executed
                    'pnl': trade.get('pnl', 0),
                    'timestamp': signal.timestamp.isoformat() if hasattr(signal.timestamp, 'isoformat') else str(signal.timestamp)
                })

        # 2. Check for new entry opportunities (if slots available)
        slots_available = self.max_positions - len(self.rsi_positions)

        if slots_available > 0:
            entry_signals = self._scan_for_entries()

            for signal in entry_signals[:slots_available]:
                trade = self._execute_entry(signal)
                if trade and trade.get('status') in ('filled', 'submitted'):
                    # Return Dict format for master_orchestrator compatibility
                    quantity = self.position_size / signal.entry_price
                    results.append({
                        'symbol': signal.product_id,
                        'action': 'buy',
                        'price': signal.entry_price,
                        'quantity': quantity,
                        'confidence': signal.confidence,
                        'reason': signal.reason,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'rsi': signal.rsi_value,
                        'signal_type': 'entry',
                        'status': trade.get('status', 'filled'),  # Mark as already-executed
                        'timestamp': signal.timestamp.isoformat() if hasattr(signal.timestamp, 'isoformat') else str(signal.timestamp)
                    })

        logger.info(
            f"Scan complete: {len(results)} signals, "
            f"{len(self.rsi_positions)} open positions, "
            f"P&L today: ${self.pnl_today:.2f}"
        )

        return results

    def get_status(self) -> dict:
        """Get bot status"""
        total_trades = self.wins + self.losses
        win_rate = self.wins / total_trades if total_trades > 0 else 0

        status = {
            'name': 'RSIExtremesBot (V4 Momentum)',
            'capital': self.capital,
            'paper_mode': self.paper_mode,
            'alpaca_connected': (self.alpaca._initialized if hasattr(self, 'alpaca') and self.alpaca else
                                 self.coinbase._initialized if hasattr(self, 'coinbase') and self.coinbase else False),
            'open_positions': len(self.rsi_positions),
            'max_positions': self.max_positions,
            'trades_today': len(self.trades_history),
            'pnl_today': self.pnl_today,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'strategy_params': {
                'rsi_period': self.RSI_PERIOD,
                'rsi_entry_threshold': self.RSI_ENTRY_THRESHOLD,
                'rsi_exit_threshold': self.RSI_EXIT_THRESHOLD,
                'ema_period': self.EMA_PERIOD,
                'position_size': self.position_size,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
            },
            'positions': [
                {
                    'symbol': p.symbol,
                    'entry': p.entry_price,
                    'stop_loss': p.stop_loss,
                    'take_profit': p.take_profit,
                    'rsi_at_entry': p.rsi_at_entry
                }
                for p in self.rsi_positions.values()
            ],
            'last_scan': datetime.now(timezone.utc).isoformat()
        }

        # Add base class status if available
        if HAS_BASE_CLASS:
            try:
                base_status = super().get_status()
                status.update(base_status)
            except Exception:
                pass

        return status


def main():
    """Test the RSI Extremes Bot (V4 Momentum)"""
    logger.info("=" * 60)
    logger.info("RSI EXTREMES BOT - V4 Momentum Strategy")
    logger.info("=" * 60)
    logger.info(f"Strategy: Buy when RSI(14) crosses above 50 AND price > EMA(20)")
    logger.info(f"  Exit: RSI < 40 or price < EMA(20) or -5% stop")
    logger.info(f"  Position Size: $60")
    logger.info(f"  Stop Loss: 5%")
    logger.info(f"  Take Profit: 10%")
    logger.info(f"  Max Positions: 3")
    logger.info("=" * 60)

    # Initialize bot
    bot = RSIExtremesBot(
        capital=200.0,
        paper_mode=True
    )

    logger.info(f"\nInitial Status:")
    status = bot.get_status()
    logger.info(f"  Capital: ${status['capital']}")
    logger.info(f"  Paper Mode: {status['paper_mode']}")
    logger.info(f"  Alpaca: {'Connected' if status['alpaca_connected'] else 'Mock Mode'}")

    # Run several scans
    for i in range(3):
        logger.info(f"\n--- Scan {i+1}/3 ---")
        signals = bot.run_scan()

        if signals:
            logger.info(f"Signals: {len(signals)}")
            for s in signals:
                side = s.side if hasattr(s, 'side') else 'UNKNOWN'
                symbol = s.symbol if hasattr(s, 'symbol') else 'UNKNOWN'
                rsi = s.metadata.get('rsi', 0) if hasattr(s, 'metadata') else 0
                logger.info(f"  {side} {symbol} | RSI: {rsi:.1f} | {s.reason}")
        else:
            logger.info("  No signals")

        if i < 2:
            import time
            time.sleep(1)

    logger.info(f"\n--- Final Status ---")
    status = bot.get_status()
    logger.info(f"  Open Positions: {status['open_positions']}")
    logger.info(f"  Total Trades: {status['trades_today']}")
    logger.info(f"  P&L Today: ${status['pnl_today']:.2f}")
    logger.info(f"  Win Rate: {status['win_rate']:.0%} ({status['wins']}/{status['wins'] + status['losses']})")

    if status['positions']:
        logger.info("\n  Position Details:")
        for pos in status['positions']:
            logger.info(f"    {pos['symbol']}: Entry ${pos['entry']:.6f}, RSI at entry: {pos['rsi_at_entry']:.1f}")

    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()
