"""
ALPACA CRYPTO EMA MOMENTUM BOT (V4)
====================================
EMA crossover momentum strategy on BTC/USD and ETH/USD.

STRATEGY:
- EMA(12) crosses ABOVE EMA(26) + volume confirmation = BUY
- EMA(12) crosses BELOW EMA(26) = SELL
- 15-minute timeframe for faster signals than RSI-2
- Volume must be >1.5x 20-period average to confirm breakout
- MACD histogram slope as trend strength filter
- 4% stop loss (tighter than RSI bot — faster timeframe)
- Regime-based position sizing

REPLACES: RSI-2 mean reversion (mean reversion doesn't work on crypto)

Research: EMA(12/26) crossover captures crypto trends earlier than RSI(14).
Volume confirmation filters ~40% of false crossovers.

Author: Trading Bot Arsenal
Created: February 2026 (V4)
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: Alpaca not available")

# Import regime detector
try:
    from filters.regime_detector import RegimeDetector, MarketRegime
    REGIME_DETECTOR_AVAILABLE = True
except ImportError:
    REGIME_DETECTOR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AlpacaCryptoEMAMomentum')


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Regime(Enum):
    BULL = "bull"
    BULL_VOLATILE = "bull_volatile"
    SIDEWAYS = "sideways"
    CORRECTION = "correction"
    CRASH = "crash"


REGIME_POSITION_MULTIPLIERS = {
    Regime.BULL: 1.0,
    Regime.BULL_VOLATILE: 0.85,
    Regime.SIDEWAYS: 0.50,
    Regime.CORRECTION: 0.30,
    Regime.CRASH: 0.15,
}


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    ema_fast_at_entry: float
    ema_slow_at_entry: float
    regime_at_entry: Regime


@dataclass
class Signal:
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    ema_fast: float
    ema_slow: float
    macd_histogram: float
    volume_ratio: float
    entropy: float
    regime: Regime
    confidence: float
    stop_loss: Optional[float] = None
    reason: str = ""


class AlpacaCryptoMomentumBot:
    """
    V4 EMA Crossover Momentum Crypto Trading Bot.

    Strategy:
    - EMA(12) crosses ABOVE EMA(26) + volume > 1.5x avg = BUY
    - EMA(12) crosses BELOW EMA(26) = SELL
    - 15-minute timeframe
    - 4% stop loss
    - MACD histogram slope confirms trend strength
    - Regime-based position sizing
    - Entropy filter (skip if > 0.8)
    """

    SYMBOLS = [
        'BTC/USD',   # Core - largest, most liquid
        'ETH/USD',   # Core - second largest
        'SOL/USD',   # Core - best volatility/liquidity ratio
        'XRP/USD',   # Added - high volume, clear levels
        'DOGE/USD',  # Satellite - high volatility for signals
        'LINK/USD',  # Satellite - steady oracle leader
    ]

    # EMA parameters
    EMA_FAST_PERIOD = 12
    EMA_SLOW_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9

    # Volume confirmation
    VOLUME_AVG_PERIOD = 20
    VOLUME_CONFIRMATION_RATIO = 1.5  # Volume must be 1.5x average

    # Timeframe: 15-minute bars
    TIMEFRAME_MINUTES = 15

    # Risk parameters
    STOP_LOSS_PCT = 0.04        # 4% stop (tighter than RSI bot's 5%)
    RISK_PER_TRADE = 0.02       # 2% risk per trade
    MAX_POSITION_PCT = 0.30     # 30% max per position

    # Entropy filter
    ENTROPY_THRESHOLD = 0.8

    def __init__(
        self,
        capital: float = 135.0,
        paper_mode: bool = None
    ):
        self.capital = capital
        if paper_mode is None:
            paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
        self.paper_mode = paper_mode
        self.available_capital = capital

        # Initialize clients
        self.trading_client = None
        self.data_client = None
        self.regime_detector = None

        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')

        if ALPACA_AVAILABLE:
            try:
                self.data_client = CryptoHistoricalDataClient()
                logger.info("Alpaca crypto data client initialized")

                if api_key and api_secret:
                    self.trading_client = TradingClient(
                        api_key=api_key,
                        secret_key=api_secret,
                        paper=paper_mode
                    )
                    logger.info(f"Alpaca trading client initialized (paper={paper_mode})")
            except Exception as e:
                logger.warning(f"Alpaca client init failed: {e}")

        if REGIME_DETECTOR_AVAILABLE:
            try:
                self.regime_detector = RegimeDetector()
                logger.info("Regime detector initialized")
            except Exception as e:
                logger.warning(f"Regime detector init failed: {e}")

        # State tracking
        self.positions: Dict[str, Position] = {}
        self.trades_today: List[Dict] = []
        self.pnl_today: float = 0.0
        self.consecutive_losses: int = 0
        self.daily_loss: float = 0.0

        # Previous EMA values for crossover detection
        self.prev_ema_fast: Dict[str, float] = {}
        self.prev_ema_slow: Dict[str, float] = {}

        # Track consecutive declining MACD histogram bars per symbol
        self.histogram_decline_count: Dict[str, int] = {}
        self.prev_histogram: Dict[str, float] = {}
        self.MACD_DECLINE_BARS_REQUIRED = 3  # Require 3 consecutive declining bars

        logger.info(f"AlpacaCryptoEMAMomentum V4 initialized - Capital: ${capital}")

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate MACD components.
        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line
        Histogram = MACD Line - Signal Line
        """
        ema_fast = self.calculate_ema(prices, self.EMA_FAST_PERIOD)
        ema_slow = self.calculate_ema(prices, self.EMA_SLOW_PERIOD)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.MACD_SIGNAL_PERIOD, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram,
        }

    def calculate_entropy(self, prices: pd.Series, window: int = 20) -> float:
        """
        Shannon entropy of price changes.
        High entropy (>0.8) = random/noisy = skip.
        """
        returns = prices.pct_change().dropna().tail(window)
        if len(returns) < window:
            return 0.5

        bins = [-float('inf'), -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, float('inf')]
        hist, _ = np.histogram(returns, bins=bins)

        probs = hist / hist.sum()
        probs = probs[probs > 0]

        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(bins) - 1)
        return entropy / max_entropy if max_entropy > 0 else 0

    def check_volume_confirmation(self, volume: pd.Series) -> float:
        """
        Check if current volume confirms the signal.
        Returns ratio of current volume to average volume.
        A ratio >= 1.5 confirms the breakout.
        """
        if len(volume) < self.VOLUME_AVG_PERIOD + 1:
            return 1.0  # Not enough data, neutral

        avg_volume = volume.iloc[-(self.VOLUME_AVG_PERIOD + 1):-1].mean()
        current_volume = volume.iloc[-1]

        if avg_volume <= 0:
            return 1.0

        return current_volume / avg_volume

    def detect_regime(self, df: pd.DataFrame) -> Regime:
        """Detect market regime for position sizing"""
        if self.regime_detector and REGIME_DETECTOR_AVAILABLE:
            try:
                result = self.regime_detector.detect(df)
                regime_map = {
                    MarketRegime.BULL: Regime.BULL,
                    MarketRegime.BEAR: Regime.CORRECTION,
                    MarketRegime.SIDEWAYS: Regime.SIDEWAYS,
                    MarketRegime.HIGH_VOLATILITY: Regime.BULL_VOLATILE,
                    MarketRegime.LOW_VOLATILITY: Regime.BULL,
                }
                return regime_map.get(result.current_regime, Regime.SIDEWAYS)
            except Exception as e:
                logger.warning(f"Regime detection failed: {e}")

        # Fallback: simple rule-based
        if len(df) < 50:
            return Regime.SIDEWAYS

        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1]
        trend = df['close'].iloc[-1] / df['close'].iloc[-20] - 1

        if volatility > 0.05:
            if trend < -0.10:
                return Regime.CRASH
            return Regime.BULL_VOLATILE
        elif trend > 0.05:
            return Regime.BULL
        elif trend < -0.05:
            return Regime.CORRECTION
        else:
            return Regime.SIDEWAYS

    def get_market_data(self, symbol: str, bars: int = 200) -> Optional[pd.DataFrame]:
        """
        Get 15-minute bars from Alpaca.
        Returns None if data unavailable (never uses mock data).
        """
        if not self.data_client:
            logger.error(f"No data client for {symbol} - skipping")
            return None

        try:
            end = datetime.now(timezone.utc)
            # Fetch enough raw data for 200 15-min bars
            start = end - timedelta(minutes=bars * self.TIMEFRAME_MINUTES * 2)

            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end
            )

            bars_data = self.data_client.get_crypto_bars(request)

            if hasattr(bars_data, 'df'):
                df = bars_data.df
            else:
                df = pd.DataFrame(bars_data[symbol])

            if len(df) < 30:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} bars")
                return None

            # Resample to 15-minute bars
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            df_15m = df.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            return df_15m.tail(bars)

        except Exception as e:
            logger.error(f"Alpaca API error for {symbol}: {e}")
            return None

    def analyze(self, symbol: str) -> Optional[Signal]:
        """
        Analyze a symbol and generate EMA crossover signal.

        BUY: EMA(12) crosses above EMA(26) AND volume > 1.5x average
        SELL: EMA(12) crosses below EMA(26) (no volume req for exits)
        """
        df = self.get_market_data(symbol)
        if df is None or len(df) < self.EMA_SLOW_PERIOD + 10:
            return None

        # Calculate indicators
        macd_data = self.calculate_macd(df['close'])
        ema_fast = macd_data['ema_fast']
        ema_slow = macd_data['ema_slow']
        histogram = macd_data['histogram']

        current_ema_fast = ema_fast.iloc[-1]
        current_ema_slow = ema_slow.iloc[-1]
        prev_ema_fast = ema_fast.iloc[-2]
        prev_ema_slow = ema_slow.iloc[-2]

        current_histogram = histogram.iloc[-1]
        prev_histogram = histogram.iloc[-2]
        histogram_slope = current_histogram - prev_histogram

        price = df['close'].iloc[-1]

        # Volume confirmation
        volume_ratio = self.check_volume_confirmation(df['volume'])

        # Entropy filter
        entropy = self.calculate_entropy(df['close'])

        # Regime detection
        regime = self.detect_regime(df)

        # Skip if entropy too high
        if entropy > self.ENTROPY_THRESHOLD:
            logger.info(f"{symbol}: Entropy {entropy:.2f} > {self.ENTROPY_THRESHOLD} - skipping")
            return None

        # Check for existing position -> check exit
        if symbol in self.positions:
            return self._check_exit(
                symbol, price, current_ema_fast, current_ema_slow,
                prev_ema_fast, prev_ema_slow, histogram_slope,
                volume_ratio, entropy, regime
            )

        # --- ENTRY LOGIC ---
        # Bullish crossover: fast EMA crosses above slow EMA
        fast_crossed_above = (prev_ema_fast <= prev_ema_slow and
                              current_ema_fast > current_ema_slow)

        if fast_crossed_above:
            # Volume confirmation required for entries
            if volume_ratio < self.VOLUME_CONFIRMATION_RATIO:
                logger.info(
                    f"{symbol}: EMA crossover detected but volume ratio "
                    f"{volume_ratio:.2f} < {self.VOLUME_CONFIRMATION_RATIO} - skipping"
                )
                return None

            # MACD histogram should be turning positive (momentum building)
            if histogram_slope <= 0:
                logger.info(
                    f"{symbol}: EMA crossover but histogram slope negative "
                    f"({histogram_slope:.4f}) - weak momentum, skipping"
                )
                return None

            stop_loss = price * (1 - self.STOP_LOSS_PCT)

            # Confidence: higher volume ratio + stronger histogram = higher confidence
            confidence = min(
                0.5 + (volume_ratio - 1.0) * 0.15 + abs(histogram_slope) * 10,
                1.0
            )

            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                price=price,
                ema_fast=current_ema_fast,
                ema_slow=current_ema_slow,
                macd_histogram=current_histogram,
                volume_ratio=volume_ratio,
                entropy=entropy,
                regime=regime,
                confidence=confidence,
                stop_loss=stop_loss,
                reason=(
                    f"EMA({self.EMA_FAST_PERIOD}) crossed above "
                    f"EMA({self.EMA_SLOW_PERIOD}) | "
                    f"Vol: {volume_ratio:.1f}x avg | "
                    f"MACD hist slope: {histogram_slope:+.4f}"
                )
            )

        return None

    def _check_exit(
        self, symbol: str, price: float,
        current_ema_fast: float, current_ema_slow: float,
        prev_ema_fast: float, prev_ema_slow: float,
        histogram_slope: float, volume_ratio: float,
        entropy: float, regime: Regime
    ) -> Optional[Signal]:
        """Check if we should exit an existing position"""
        position = self.positions[symbol]

        should_exit = False
        reason = ""

        # Bearish crossover: fast EMA crosses below slow EMA
        fast_crossed_below = (prev_ema_fast >= prev_ema_slow and
                              current_ema_fast < current_ema_slow)

        if fast_crossed_below:
            should_exit = True
            reason = (
                f"EMA({self.EMA_FAST_PERIOD}) crossed below "
                f"EMA({self.EMA_SLOW_PERIOD})"
            )

        # Stop loss
        elif price <= position.stop_loss:
            should_exit = True
            reason = f"Stop loss hit (${position.stop_loss:,.2f})"

        # Crash regime emergency exit
        elif regime == Regime.CRASH and price < position.entry_price:
            should_exit = True
            reason = "Crash regime detected - emergency exit"

        # MACD histogram divergence: 3 consecutive declining bars while in profit
        elif price > position.entry_price * 1.02:
            # Track consecutive declining histogram bars
            if histogram_slope < 0:
                self.histogram_decline_count[symbol] = self.histogram_decline_count.get(symbol, 0) + 1
            else:
                self.histogram_decline_count[symbol] = 0

            if self.histogram_decline_count.get(symbol, 0) >= self.MACD_DECLINE_BARS_REQUIRED:
                should_exit = True
                reason = (
                    f"MACD histogram declining {self.MACD_DECLINE_BARS_REQUIRED} consecutive bars "
                    f"(slope {histogram_slope:+.4f}) - locking in profit"
                )
                self.histogram_decline_count[symbol] = 0  # Reset counter

        if should_exit:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                price=price,
                ema_fast=current_ema_fast,
                ema_slow=current_ema_slow,
                macd_histogram=0.0,
                volume_ratio=volume_ratio,
                entropy=entropy,
                regime=regime,
                confidence=1.0,
                reason=reason
            )

        return None

    def calculate_position_size(self, signal: Signal) -> float:
        """Calculate position size based on risk and regime"""
        risk_amount = self.available_capital * self.RISK_PER_TRADE
        stop_distance = abs(signal.price - signal.stop_loss)

        if stop_distance == 0:
            return 0

        base_quantity = risk_amount / stop_distance

        # Regime multiplier
        regime_multiplier = REGIME_POSITION_MULTIPLIERS.get(signal.regime, 0.5)
        quantity = base_quantity * regime_multiplier

        # Confidence adjustment
        quantity *= signal.confidence

        # Max position cap
        max_value = self.available_capital * self.MAX_POSITION_PCT
        max_quantity = max_value / signal.price

        final_quantity = min(quantity, max_quantity)

        logger.info(
            f"Position sizing: regime={signal.regime.value}, "
            f"multiplier={regime_multiplier}, "
            f"vol_ratio={signal.volume_ratio:.1f}x, "
            f"qty={final_quantity:.6f}"
        )

        return final_quantity

    def execute_trade(self, signal: Signal) -> Optional[Dict]:
        """Execute a trade"""
        # Daily loss limit: 3%
        daily_loss_limit = self.capital * 0.03
        if self.daily_loss >= daily_loss_limit:
            logger.warning(f"Daily loss limit hit (${daily_loss_limit:.2f}) - no new trades")
            return None

        # Consecutive loss halt
        if self.consecutive_losses >= 5:
            logger.warning("5 consecutive losses - halting trading")
            return None

        quantity = self.calculate_position_size(signal)
        if quantity <= 0:
            return None

        # Round to appropriate precision
        if 'BTC' in signal.symbol:
            quantity = round(quantity, 6)
        else:
            quantity = round(quantity, 4)

        trade_record = {
            'timestamp': signal.timestamp.isoformat(),
            'symbol': signal.symbol,
            'side': signal.signal_type.value,
            'quantity': quantity,
            'price': signal.price,
            'ema_fast': signal.ema_fast,
            'ema_slow': signal.ema_slow,
            'macd_histogram': signal.macd_histogram,
            'volume_ratio': signal.volume_ratio,
            'entropy': signal.entropy,
            'regime': signal.regime.value,
            'stop_loss': signal.stop_loss,
            'confidence': signal.confidence,
            'reason': signal.reason,
            'paper': self.paper_mode,
            'status': 'pending'
        }

        is_closing = signal.symbol in self.positions

        if self.paper_mode:
            trade_record['status'] = 'filled'
            trade_record['fill_price'] = signal.price

            if is_closing:
                position = self.positions[signal.symbol]
                pnl = (signal.price - position.entry_price) * position.quantity

                trade_record['pnl'] = pnl
                self.pnl_today += pnl

                if pnl < 0:
                    self.daily_loss += abs(pnl)
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

                self.available_capital += position.quantity * position.entry_price + pnl
                del self.positions[signal.symbol]

                prefix = "[PAPER]" if self.paper_mode else "[LIVE]"
                logger.info(
                    f"{prefix} CLOSE {signal.symbol}: "
                    f"{quantity:.6f} @ ${signal.price:,.2f} | "
                    f"PnL: ${pnl:,.2f} | {signal.reason}"
                )
            else:
                self.positions[signal.symbol] = Position(
                    symbol=signal.symbol,
                    side='long',
                    entry_price=signal.price,
                    quantity=quantity,
                    entry_time=signal.timestamp,
                    stop_loss=signal.stop_loss,
                    ema_fast_at_entry=signal.ema_fast,
                    ema_slow_at_entry=signal.ema_slow,
                    regime_at_entry=signal.regime
                )
                self.available_capital -= quantity * signal.price

                prefix = "[PAPER]" if self.paper_mode else "[LIVE]"
                logger.info(
                    f"{prefix} OPEN LONG {signal.symbol}: "
                    f"{quantity:.6f} @ ${signal.price:,.2f} | "
                    f"Stop: ${signal.stop_loss:,.2f} | "
                    f"Vol: {signal.volume_ratio:.1f}x | "
                    f"Regime: {signal.regime.value} | {signal.reason}"
                )

        else:
            # Live trading via Alpaca — use AlpacaCryptoClient for proper fill polling
            try:
                from bots.alpaca_crypto_client import AlpacaCryptoClient
                alpaca_client = AlpacaCryptoClient()

                if signal.signal_type == SignalType.BUY:
                    order = alpaca_client.create_market_order(
                        product_id=signal.symbol,
                        side='BUY',
                        quote_size=str(round(quantity * signal.price, 2))
                    )
                else:
                    order = alpaca_client.create_market_order(
                        product_id=signal.symbol,
                        side='SELL',
                        base_size=str(round(quantity, 8))
                    )

                if order and order.get('success', True):
                    trade_record['order_id'] = order.get('order_id', '')
                    trade_record['status'] = 'filled'
                    # Use broker fill price if available
                    fill_price = float(order.get('filled_avg_price', 0))
                    actual_price = fill_price if fill_price > 0 else signal.price
                    if fill_price > 0:
                        trade_record['fill_price'] = fill_price

                    # Track position state for live mode
                    if is_closing:
                        position = self.positions.get(signal.symbol)
                        if position:
                            pnl = (actual_price - position.entry_price) * position.quantity
                            trade_record['pnl'] = pnl
                            self.pnl_today += pnl
                            if pnl < 0:
                                self.daily_loss += abs(pnl)
                                self.consecutive_losses += 1
                            else:
                                self.consecutive_losses = 0
                            self.available_capital += position.quantity * position.entry_price + pnl
                            del self.positions[signal.symbol]
                    else:
                        self.positions[signal.symbol] = Position(
                            symbol=signal.symbol,
                            side='long',
                            entry_price=actual_price,
                            quantity=quantity,
                            entry_time=signal.timestamp,
                            stop_loss=signal.stop_loss,
                            ema_fast_at_entry=signal.ema_fast,
                            ema_slow_at_entry=signal.ema_slow,
                            regime_at_entry=signal.regime
                        )
                        self.available_capital -= quantity * actual_price

                    logger.info(f"[LIVE] Order filled: {signal.symbol} {signal.signal_type.value} qty={quantity:.6f}")
                else:
                    trade_record['status'] = 'failed'
                    trade_record['error'] = order.get('error', 'Order failed') if order else 'No response'

            except Exception as e:
                logger.error(f"Order failed: {e}")
                trade_record['status'] = 'failed'
                trade_record['error'] = str(e)

        self.trades_today.append(trade_record)
        return trade_record

    def run_scan(self) -> List[Dict]:
        """Scan all symbols and execute trades. Called by master_orchestrator."""
        executed = []

        for symbol in self.SYMBOLS:
            logger.info(f"Analyzing {symbol}...")

            signal = self.analyze(symbol)

            if signal and signal.signal_type != SignalType.HOLD:
                trade = self.execute_trade(signal)
                if trade and trade['status'] == 'filled':
                    executed.append(trade)

        return executed

    def get_status(self) -> Dict:
        """Get bot status for dashboard"""
        return {
            'name': 'AlpacaCryptoEMAMomentum',
            'version': 'V4',
            'strategy': (
                f'EMA({self.EMA_FAST_PERIOD}/{self.EMA_SLOW_PERIOD}) '
                f'Crossover + Volume({self.VOLUME_CONFIRMATION_RATIO}x)'
            ),
            'timeframe': f'{self.TIMEFRAME_MINUTES}min',
            'capital': self.capital,
            'available_capital': self.available_capital,
            'paper_mode': self.paper_mode,
            'positions': len(self.positions),
            'trades_today': len(self.trades_today),
            'pnl_today': self.pnl_today,
            'daily_loss': self.daily_loss,
            'consecutive_losses': self.consecutive_losses,
            'data_connected': self.data_client is not None,
            'trading_connected': self.trading_client is not None,
            'regime_detector': self.regime_detector is not None,
        }


def main():
    """Test the V4 EMA momentum bot"""
    print("=" * 60)
    print("ALPACA CRYPTO EMA MOMENTUM BOT V4 - TEST RUN")
    print("=" * 60)
    print(f"Strategy: EMA(12/26) Crossover + Volume Confirmation")
    print(f"  Entry: EMA(12) crosses ABOVE EMA(26) + Vol > 1.5x avg")
    print(f"  Exit:  EMA(12) crosses BELOW EMA(26) or stop loss")
    print(f"  Timeframe: 15-minute")
    print(f"  Stop Loss: 4%")
    print(f"  Entropy Filter: Skip if > 0.8")
    print(f"  Regime Sizing: Bull=100%, Crash=15%")
    print("=" * 60)

    bot = AlpacaCryptoMomentumBot(
        capital=135.0,
        paper_mode=True
    )

    print(f"\nBot Status: {bot.get_status()}")

    for i in range(3):
        print(f"\n--- Scan {i+1} ---")
        trades = bot.run_scan()
        print(f"Executed {len(trades)} trades")

        if trades:
            for trade in trades:
                print(
                    f"  {trade['symbol']}: {trade['side'].upper()} "
                    f"{trade['quantity']:.6f} @ ${trade['price']:,.2f} "
                    f"[{trade['regime']}] {trade['reason']}"
                )

        time.sleep(1)

    print(f"\nFinal Status: {bot.get_status()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
