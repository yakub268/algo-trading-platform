"""
Volatility Breakout Bot
Strategy: Bollinger Band compression → expansion breakout
- Detects BB compression (squeeze)
- Trades breakout above upper band on volume surge
- Dynamic stops and targets based on BB width
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
import traceback

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType

try:
    from bots.alpaca_crypto_client import AlpacaCryptoClient
except ImportError:
    AlpacaCryptoClient = None

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None


class VolatilityBreakoutBot(FleetBot):
    """
    Bollinger Band compression-expansion breakout trader.

    Logic:
    1. Calculate BB(20, 2.0σ)
    2. Detect compression: BB width < 3% of middle band
    3. On breakout above upper BB + volume surge → BUY
    4. Stop: middle BB, Target: 2x BB width above entry
    """

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="Volatility-Breakout",
                bot_type=BotType.CRYPTO,
                symbols=["BTC/USD", "ETH/USD", "SOL/USD"],
                schedule_seconds=300,
                max_position_usd=50.0,
                max_daily_trades=20,
                min_confidence=0.60,
                enabled=True,
                paper_mode=True,
                extra={'base_position_size': 25.0}
            )

        super().__init__(config)

        # Strategy parameters
        self.bb_period = 20
        self.bb_std = 2.0
        self.compression_threshold = 0.03  # 3% width
        self.min_compression_bars = 5
        self.volume_surge_multiplier = 1.5
        self.lookback_candles = 100

        # Alpaca client
        self.client = None
        if AlpacaCryptoClient:
            try:
                self.client = AlpacaCryptoClient()
            except Exception as e:
                self.logger.warning(f"Failed to initialize AlpacaCryptoClient: {e}")

        # Track compression state
        self.compression_state = {}  # symbol -> bars_compressed

    def scan(self) -> List[FleetSignal]:
        """Scan for BB compression breakouts."""
        if not self.client:
            self.logger.error("AlpacaCryptoClient not available")
            return []

        if np is None or pd is None:
            self.logger.error("numpy/pandas not available")
            return []

        signals = []

        for symbol in self.config.symbols:
            try:
                signal = self._scan_symbol(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
                self.logger.debug(traceback.format_exc())

        return signals

    def _scan_symbol(self, symbol: str) -> Optional[FleetSignal]:
        """Scan single symbol for breakout setup."""
        try:
            # Fetch candles
            candles = self.client.get_candles(
                symbol=symbol,
                granularity='5Min',
                limit=self.lookback_candles
            )

            if not candles or len(candles) < self.bb_period + 10:
                self.logger.debug(f"{symbol}: insufficient candles ({len(candles) if candles else 0})")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Calculate Bollinger Bands
            df['sma'] = df['close'].rolling(window=self.bb_period).mean()
            df['std'] = df['close'].rolling(window=self.bb_period).std()
            df['upper_bb'] = df['sma'] + (self.bb_std * df['std'])
            df['lower_bb'] = df['sma'] - (self.bb_std * df['std'])
            df['bb_width'] = (df['upper_bb'] - df['lower_bb']) / df['sma']

            # Volume average
            df['volume_avg'] = df['volume'].rolling(window=20).mean()

            # Drop NaN rows
            df = df.dropna()

            if len(df) < 10:
                return None

            # Current values
            current_price = df['close'].iloc[-1]
            current_bb_width = df['bb_width'].iloc[-1]
            current_upper = df['upper_bb'].iloc[-1]
            current_middle = df['sma'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume_avg'].iloc[-1]

            # Check for compression
            is_compressed = current_bb_width < self.compression_threshold

            # Track compression duration
            if symbol not in self.compression_state:
                self.compression_state[symbol] = 0

            if is_compressed:
                self.compression_state[symbol] += 1
            else:
                self.compression_state[symbol] = 0

            compression_bars = self.compression_state[symbol]

            # Look for breakout
            if compression_bars >= self.min_compression_bars:
                # Price breaking above upper BB
                price_breakout = current_price > current_upper

                # Volume surge
                volume_surge = current_volume > (avg_volume * self.volume_surge_multiplier)

                if price_breakout and volume_surge:
                    # Calculate confidence
                    compression_score = min(compression_bars / 20.0, 1.0)  # Cap at 20 bars
                    volume_score = min((current_volume / avg_volume) / 3.0, 1.0)  # Cap at 3x
                    breakout_strength = (current_price - current_upper) / current_upper
                    breakout_score = min(breakout_strength / 0.02, 1.0)  # Cap at 2%

                    confidence = 0.5 + (0.5 * np.mean([compression_score, volume_score, breakout_score]))

                    if confidence < self.config.min_confidence:
                        return None

                    # Calculate stop loss and take profit
                    stop_loss = current_middle
                    bb_width_dollars = current_upper - current_middle
                    take_profit = current_price + (2 * bb_width_dollars)

                    # Position sizing based on confidence
                    position_size = self._calculate_position_size(confidence)

                    signal = FleetSignal(
                        bot_name=self.name,
                        bot_type=self.bot_type.value,
                        symbol=symbol,
                        side='BUY',
                        entry_price=current_price,
                        target_price=take_profit,
                        stop_loss=stop_loss,
                        quantity=position_size / current_price,
                        position_size_usd=position_size,
                        confidence=confidence,
                        edge=float(breakout_strength) * 100,
                        reason=f"BB breakout after {compression_bars} bars compression, Vol={current_volume/avg_volume:.1f}x avg",
                        metadata={
                            'bb_width': float(current_bb_width),
                            'compression_bars': compression_bars,
                            'volume_ratio': float(current_volume / avg_volume),
                            'breakout_strength': float(breakout_strength),
                            'upper_bb': float(current_upper),
                            'middle_bb': float(current_middle)
                        }
                    )

                    self.logger.info(
                        f"BREAKOUT: {symbol} @ ${current_price:.2f} | "
                        f"Conf: {confidence:.2%} | Compressed: {compression_bars} bars | "
                        f"Vol: {current_volume/avg_volume:.1f}x | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}"
                    )

                    # Reset compression tracker
                    self.compression_state[symbol] = 0

                    return signal

            return None

        except Exception as e:
            self.logger.error(f"Error in _scan_symbol for {symbol}: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence."""
        base = (self.config.max_position_usd * 0.5)
        max_size = self.config.max_position_usd

        # Linear scaling between base and max
        size = base + (max_size - base) * (confidence - self.config.min_confidence) / (1.0 - self.config.min_confidence)

        return round(size, 2)


def main():
    """Test runner."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = VolatilityBreakoutBot()

    print(f"\n{'='*60}")
    print(f"Volatility Breakout Bot - Test Scan")
    print(f"{'='*60}\n")

    signals = bot.scan()

    if signals:
        print(f"Found {len(signals)} signal(s):\n")
        for sig in signals:
            print(f"Symbol: {sig.symbol}")
            print(f"Direction: {sig.direction.upper()}")
            print(f"Confidence: {sig.confidence:.2%}")
            print(f"Entry: ${sig.entry_price:.2f}")
            print(f"Stop Loss: ${sig.stop_loss:.2f}")
            print(f"Take Profit: ${sig.take_profit:.2f}")
            print(f"Position Size: ${sig.position_size:.2f}")
            print(f"Metadata: {sig.metadata}")
            print(f"{'-'*60}\n")
    else:
        print("No signals generated.\n")


if __name__ == "__main__":
    main()
