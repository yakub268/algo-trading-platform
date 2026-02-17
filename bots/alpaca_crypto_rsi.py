"""
ALPACA CRYPTO RSI MOMENTUM BOT (V4)
===================================
RSI Momentum strategy on BTC/USD and ETH/USD.

V4 CHANGES (Feb 2026):
- Changed from mean reversion to MOMENTUM strategy
- RSI(14) instead of RSI(2)
- Entry: RSI crosses ABOVE 50 (trend starting)
- Exit: RSI crosses BELOW 50 (trend ending)
- 4-hour timeframe (was 5-minute)
- BTC/USD and ETH/USD only
- 5% stop loss (wider for crypto volatility)
- Regime-based position sizing
- Entropy filter (skip trades when entropy > 0.8)

Research: Bitcoin trend-following CAGR 115% vs 94% buy-and-hold
Mean reversion does NOT work on crypto - momentum does.

Author: Trading Bot Arsenal
Updated: February 2026 (V4 Optimization)
"""

import os
import sys
import time
import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
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
    print("Warning: Regime detector not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AlpacaCryptoMomentum')


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Regime(Enum):
    """Market regimes for position sizing"""
    BULL = "bull"
    BULL_VOLATILE = "bull_volatile"
    SIDEWAYS = "sideways"
    CORRECTION = "correction"
    CRASH = "crash"


# V4 Regime-based position sizing multipliers
REGIME_POSITION_MULTIPLIERS = {
    Regime.BULL: 1.0,           # 100% position
    Regime.BULL_VOLATILE: 0.85,  # 85% position
    Regime.SIDEWAYS: 0.50,       # 50% position
    Regime.CORRECTION: 0.30,     # 30% position
    Regime.CRASH: 0.15,          # 15% position
}


@dataclass
class Position:
    """Track an open position"""
    symbol: str
    side: str  # 'long' only for momentum (no shorting in V4)
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    rsi_at_entry: float
    regime_at_entry: Regime


@dataclass
class Signal:
    """Trading signal"""
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    rsi: float
    entropy: float
    regime: Regime
    confidence: float
    stop_loss: Optional[float] = None
    reason: str = ""


class AlpacaCryptoRSIBot:
    """
    V4 RSI Momentum Crypto Trading Bot.

    Strategy (momentum, NOT mean reversion):
    - RSI(14) crosses ABOVE 50 = BUY (trend starting)
    - RSI(14) crosses BELOW 50 = SELL (trend ending)
    - 4-hour timeframe for crypto
    - 5% stop loss (wider for volatility)
    - Regime-based position sizing
    - Entropy filter (skip if entropy > 0.8)
    """

    # V4: Expanded crypto universe (6 pairs)
    SYMBOLS = [
        'BTC/USD',   # Core - largest, most liquid
        'ETH/USD',   # Core - second largest
        'SOL/USD',   # Core - best volatility/liquidity ratio
        'XRP/USD',   # Added - high volume, clear levels
        'DOGE/USD',  # Satellite - high volatility for signals
        'LINK/USD',  # Satellite - steady oracle leader
    ]

    # V4 Strategy parameters - MOMENTUM
    RSI_PERIOD = 14          # Standard RSI period (was 2 for mean reversion)
    RSI_ENTRY_THRESHOLD = 50  # Buy when RSI crosses ABOVE 50
    RSI_EXIT_THRESHOLD = 50   # Sell when RSI crosses BELOW 50

    # V4: 4-hour timeframe (was 5 minutes)
    TIMEFRAME_HOURS = 4

    # V4: 5% stop loss for crypto volatility
    STOP_LOSS_PCT = 0.05

    # V4: Entropy filter threshold
    ENTROPY_THRESHOLD = 0.8

    # V4: Risk per trade
    RISK_PER_TRADE = 0.02  # 2% risk per trade
    MAX_POSITION_PCT = 0.30  # 30% max per position (V4 allocation)

    def __init__(
        self,
        capital: float = 135.0,  # V4: 30% of $450
        paper_mode: bool = None
    ):
        self.capital = capital
        # Safe default: read from environment, default to PAPER if not set
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

        # Initialize regime detector
        if REGIME_DETECTOR_AVAILABLE:
            try:
                self.regime_detector = RegimeDetector()
                logger.info("Regime detector initialized")
            except Exception as e:
                logger.warning(f"Regime detector init failed: {e}")

        # Track positions and state
        self.positions: Dict[str, Position] = {}
        self.trades_today: List[Dict] = []
        self.pnl_today: float = 0.0
        self.previous_rsi: Dict[str, float] = {}  # For crossover detection
        self.consecutive_losses: int = 0
        self.daily_loss: float = 0.0

        logger.info(f"AlpacaCryptoMomentumBot V4 initialized - Capital: ${capital}")

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI(14) for momentum"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_entropy(self, prices: pd.Series, window: int = 20) -> float:
        """
        Calculate Shannon entropy of price changes.
        High entropy (>0.8) = random/noisy market = skip trade.
        Low entropy = more predictable = trade.
        """
        returns = prices.pct_change().dropna().tail(window)
        if len(returns) < window:
            return 0.5  # Default moderate entropy

        # Bin returns into categories
        bins = [-float('inf'), -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, float('inf')]
        hist, _ = np.histogram(returns, bins=bins)

        # Calculate probabilities
        probs = hist / hist.sum()
        probs = probs[probs > 0]  # Remove zeros

        # Shannon entropy (normalized to 0-1)
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(bins) - 1)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return normalized_entropy

    def detect_regime(self, df: pd.DataFrame) -> Regime:
        """
        Detect current market regime for position sizing.
        Uses simple rules if HMM not available.
        """
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

        # Fallback: Simple rule-based regime detection
        if len(df) < 50:
            return Regime.SIDEWAYS

        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1]
        trend = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)

        if volatility > 0.05:  # >5% daily volatility
            if trend < -0.10:
                return Regime.CRASH
            return Regime.BULL_VOLATILE
        elif trend > 0.05:
            return Regime.BULL
        elif trend < -0.05:
            return Regime.CORRECTION
        else:
            return Regime.SIDEWAYS

    def get_market_data(self, symbol: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """Get 4-hour bars from Alpaca. Returns None if data unavailable (never uses mock data)."""
        if not self.data_client:
            logger.error(f"No data client available for {symbol} - cannot fetch market data. Skipping.")
            return None

        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=bars * self.TIMEFRAME_HOURS * 2)

            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )

            bars_data = self.data_client.get_crypto_bars(request)

            if hasattr(bars_data, 'df'):
                df = bars_data.df
            else:
                df = pd.DataFrame(bars_data[symbol])

            if len(df) < 20:
                logger.warning(f"Insufficient data for {symbol}")
                return None

            # Resample to 4-hour bars
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            df_4h = df.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            return df_4h.tail(bars)

        except Exception as e:
            logger.error(f"Alpaca API error getting data for {symbol}: {e} - skipping signal generation (no mock data fallback)")
            return None

    def _get_mock_data(self, symbol: str, bars: int) -> pd.DataFrame:
        """
        Generate mock data for OFFLINE TESTING ONLY.

        WARNING: This method must NEVER be called from production code paths.
        Trading on fake data can cause real money loss. The get_market_data()
        method returns None when the API is unavailable, which causes the bot
        to skip signal generation entirely (the safe behavior).
        """
        logger.warning(f"_get_mock_data called for {symbol} - this should only happen in explicit test mode")
        import random

        base_price = 95000 if 'BTC' in symbol else 3200
        timestamps = pd.date_range(
            end=datetime.now(timezone.utc),
            periods=bars,
            freq='4h'
        )

        data = []
        price = base_price

        for ts in timestamps:
            # Simulate trending behavior for momentum testing
            trend = 0.001 if random.random() > 0.45 else -0.001
            change = trend + random.uniform(-0.01, 0.01)
            price = price * (1 + change)

            high = price * (1 + random.uniform(0, 0.005))
            low = price * (1 - random.uniform(0, 0.005))

            data.append({
                'timestamp': ts,
                'open': price * (1 + random.uniform(-0.002, 0.002)),
                'high': high,
                'low': low,
                'close': price,
                'volume': random.uniform(1000, 10000)
            })

        df = pd.DataFrame(data)
        df = df.set_index('timestamp')
        return df

    def analyze(self, symbol: str) -> Optional[Signal]:
        """
        Analyze a symbol and generate MOMENTUM trading signal.
        V4: Buy on RSI cross ABOVE 50, Sell on cross BELOW 50.
        """
        df = self.get_market_data(symbol)
        if df is None or len(df) < 30:
            return None

        # Calculate indicators
        df['rsi'] = self.calculate_rsi(df['close'], self.RSI_PERIOD)

        current = df.iloc[-1]
        previous = df.iloc[-2]
        current_rsi = current['rsi']
        previous_rsi = previous['rsi']
        price = current['close']

        # Calculate entropy
        entropy = self.calculate_entropy(df['close'])

        # Detect regime
        regime = self.detect_regime(df)

        # V4: Skip trade if entropy too high (market too random)
        if entropy > self.ENTROPY_THRESHOLD:
            logger.info(f"{symbol}: Entropy {entropy:.2f} > {self.ENTROPY_THRESHOLD} - skipping")
            return None

        # Check for existing position
        if symbol in self.positions:
            return self._check_exit(symbol, price, current_rsi, previous_rsi, entropy, regime)

        # V4 MOMENTUM ENTRY: RSI crosses ABOVE 50
        if previous_rsi < self.RSI_ENTRY_THRESHOLD and current_rsi >= self.RSI_ENTRY_THRESHOLD:
            # Bullish crossover - trend starting
            stop_loss = price * (1 - self.STOP_LOSS_PCT)
            confidence = min((current_rsi - 50) / 50 + 0.5, 1.0)

            return Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                price=price,
                rsi=current_rsi,
                entropy=entropy,
                regime=regime,
                confidence=confidence,
                stop_loss=stop_loss,
                reason=f"RSI crossed above 50 ({previous_rsi:.1f} -> {current_rsi:.1f})"
            )

        return None

    def _check_exit(self, symbol: str, current_price: float, current_rsi: float,
                    previous_rsi: float, entropy: float, regime: Regime) -> Optional[Signal]:
        """Check if we should exit an existing position"""
        position = self.positions[symbol]

        should_exit = False
        reason = ""

        # V4 MOMENTUM EXIT: RSI crosses BELOW 50
        if previous_rsi >= self.RSI_EXIT_THRESHOLD and current_rsi < self.RSI_EXIT_THRESHOLD:
            should_exit = True
            reason = f"RSI crossed below 50 ({previous_rsi:.1f} -> {current_rsi:.1f})"

        # Stop loss check
        elif current_price <= position.stop_loss:
            should_exit = True
            reason = f"Stop loss hit (${position.stop_loss:,.2f})"

        # Crash regime - emergency exit
        elif regime == Regime.CRASH and current_price < position.entry_price:
            should_exit = True
            reason = "Crash regime detected - emergency exit"

        if should_exit:
            return Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                price=current_price,
                rsi=current_rsi,
                entropy=entropy,
                regime=regime,
                confidence=1.0,
                reason=reason
            )

        return None

    def calculate_position_size(self, signal: Signal) -> float:
        """
        Calculate position size based on risk and regime.
        V4: Apply regime multiplier to position size.
        """
        # Base position from risk
        risk_amount = self.available_capital * self.RISK_PER_TRADE
        stop_distance = abs(signal.price - signal.stop_loss)

        if stop_distance == 0:
            return 0

        base_quantity = risk_amount / stop_distance

        # V4: Apply regime multiplier
        regime_multiplier = REGIME_POSITION_MULTIPLIERS.get(signal.regime, 0.5)
        quantity = base_quantity * regime_multiplier

        # Adjust for confidence
        quantity *= signal.confidence

        # Max position cap (30% of capital for V4)
        max_value = self.available_capital * self.MAX_POSITION_PCT
        max_quantity = max_value / signal.price

        final_quantity = min(quantity, max_quantity)

        logger.info(f"Position sizing: regime={signal.regime.value}, multiplier={regime_multiplier}, qty={final_quantity:.6f}")

        return final_quantity

    def execute_trade(self, signal: Signal) -> Optional[Dict]:
        """Execute a trade"""
        # V4: Check daily loss limit (3% = $13.50)
        daily_loss_limit = self.capital * 0.03
        if self.daily_loss >= daily_loss_limit:
            logger.warning(f"Daily loss limit hit (${daily_loss_limit:.2f}) - no new trades")
            return None

        # V4: Check consecutive losses (halt after 5)
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
            'rsi': signal.rsi,
            'entropy': signal.entropy,
            'regime': signal.regime.value,
            'stop_loss': signal.stop_loss,
            'confidence': signal.confidence,
            'reason': signal.reason,
            'paper': self.paper_mode,
            'status': 'pending'
        }

        # Check if this is closing an existing position
        is_closing = signal.symbol in self.positions

        if self.paper_mode:
            trade_record['status'] = 'filled'
            trade_record['fill_price'] = signal.price

            if is_closing:
                position = self.positions[signal.symbol]
                pnl = (signal.price - position.entry_price) * position.quantity

                trade_record['pnl'] = pnl
                self.pnl_today += pnl

                # Track for risk controls
                if pnl < 0:
                    self.daily_loss += abs(pnl)
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

                self.available_capital += position.quantity * position.entry_price + pnl
                del self.positions[signal.symbol]

                logger.info(
                    f"[PAPER] CLOSE {signal.symbol}: "
                    f"{quantity:.6f} @ ${signal.price:,.2f} | PnL: ${pnl:,.2f} | {signal.reason}"
                )
            else:
                # Opening new position
                self.positions[signal.symbol] = Position(
                    symbol=signal.symbol,
                    side='long',
                    entry_price=signal.price,
                    quantity=quantity,
                    entry_time=signal.timestamp,
                    stop_loss=signal.stop_loss,
                    rsi_at_entry=signal.rsi,
                    regime_at_entry=signal.regime
                )
                self.available_capital -= quantity * signal.price

                logger.info(
                    f"[PAPER] OPEN LONG {signal.symbol}: "
                    f"{quantity:.6f} @ ${signal.price:,.2f} | Stop: ${signal.stop_loss:,.2f} | "
                    f"Regime: {signal.regime.value} | {signal.reason}"
                )

        else:
            # Live trading
            try:
                order_side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL

                order_request = MarketOrderRequest(
                    symbol=signal.symbol.replace('/', ''),
                    qty=quantity,
                    side=order_side,
                    time_in_force=TimeInForce.GTC
                )

                order = self.trading_client.submit_order(order_request)
                trade_record['order_id'] = str(order.id)
                trade_record['status'] = str(order.status)

                logger.info(f"[LIVE] Order submitted: {order}")

            except Exception as e:
                logger.error(f"Order failed: {e}")
                trade_record['status'] = 'failed'
                trade_record['error'] = str(e)

        self.trades_today.append(trade_record)
        return trade_record

    def run_scan(self) -> List[Dict]:
        """Scan all symbols and execute trades"""
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
        """Get bot status"""
        return {
            'name': 'AlpacaCryptoMomentum',
            'version': 'V4',
            'strategy': 'RSI Momentum (entry >50, exit <50)',
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
            'regime_detector': self.regime_detector is not None
        }


def main():
    """Test the V4 momentum bot"""
    print("=" * 60)
    print("ALPACA CRYPTO RSI MOMENTUM BOT V4 - TEST RUN")
    print("=" * 60)
    print("Strategy: RSI(14) Momentum")
    print("  Entry: RSI crosses ABOVE 50")
    print("  Exit: RSI crosses BELOW 50")
    print("  Timeframe: 4-hour")
    print("  Stop Loss: 5%")
    print("  Entropy Filter: Skip if > 0.8")
    print("  Regime Sizing: Bull=100%, Crash=15%")
    print("=" * 60)

    bot = AlpacaCryptoRSIBot(
        capital=135.0,  # V4: 30% of $450
        paper_mode=True
    )

    print(f"\nBot Status: {bot.get_status()}")

    # Run multiple scans to simulate trading
    for i in range(3):
        print(f"\n--- Scan {i+1} ---")
        trades = bot.run_scan()
        print(f"Executed {len(trades)} trades")

        if trades:
            for trade in trades:
                print(f"  {trade['symbol']}: {trade['side'].upper()} "
                      f"{trade['quantity']:.6f} @ ${trade['price']:,.2f} "
                      f"[{trade['regime']}] {trade['reason']}")

        time.sleep(1)

    print(f"\nFinal Status: {bot.get_status()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
