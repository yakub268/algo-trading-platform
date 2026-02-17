"""
Pairs Trader Bot
Strategy: BTC/ETH ratio z-score mean reversion
- Calculates price ratio and z-score
- Trades when ratio deviates >2σ from mean
- Exits when ratio reverts to ±0.5σ
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


class PairsTraderBot(FleetBot):
    """
    BTC/ETH ratio mean reversion trader.

    Logic:
    1. Calculate BTC/ETH price ratio
    2. Compute z-score vs 30-period MA
    3. z > 2.0: ratio high → BUY ETH (BTC expensive relative to ETH)
    4. z < -2.0: ratio low → BUY BTC (ETH expensive relative to BTC)
    5. Exit when z reverts to ±0.5σ
    """

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="Pairs-Trader",
                bot_type=BotType.CRYPTO,
                symbols=["BTC/USD", "ETH/USD"],
                schedule_seconds=300,
                max_position_usd=40.0,
                max_daily_trades=15,
                min_confidence=0.60,
                enabled=True,
                paper_mode=True,
                extra={'base_position_size': 20.0}
            )

        super().__init__(config)

        # Strategy parameters
        self.ratio_period = 30  # MA period for ratio
        self.z_entry_threshold = 2.0  # Enter when |z| > 2.0
        self.z_exit_threshold = 0.5  # Exit when |z| < 0.5
        self.lookback_candles = 100

        # Alpaca client
        self.client = None
        if AlpacaCryptoClient:
            try:
                self.client = AlpacaCryptoClient()
            except Exception as e:
                self.logger.warning(f"Failed to initialize AlpacaCryptoClient: {e}")

        # Track open positions for exit signals
        self.open_positions = {}  # symbol -> {'direction', 'entry_z', 'entry_time'}

    def scan(self) -> List[FleetSignal]:
        """Scan for ratio mean reversion opportunities."""
        if not self.client:
            self.logger.error("AlpacaCryptoClient not available")
            return []

        if np is None or pd is None:
            self.logger.error("numpy/pandas not available")
            return []

        signals = []

        try:
            # Fetch candles for both assets
            btc_candles = self.client.get_candles(
                symbol="BTC/USD",
                granularity='5m',
                limit=self.lookback_candles
            )

            eth_candles = self.client.get_candles(
                symbol="ETH/USD",
                granularity='5m',
                limit=self.lookback_candles
            )

            if not btc_candles or not eth_candles:
                self.logger.debug("Missing candle data for BTC or ETH")
                return []

            if len(btc_candles) < self.ratio_period + 10 or len(eth_candles) < self.ratio_period + 10:
                self.logger.debug(
                    f"Insufficient candles: BTC={len(btc_candles)}, ETH={len(eth_candles)}"
                )
                return []

            # Convert to DataFrames
            btc_df = pd.DataFrame(btc_candles)
            eth_df = pd.DataFrame(eth_candles)

            btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
            eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'])

            # Merge on timestamp
            df = pd.merge(
                btc_df[['timestamp', 'close']],
                eth_df[['timestamp', 'close']],
                on='timestamp',
                suffixes=('_btc', '_eth')
            )

            df = df.sort_values('timestamp').reset_index(drop=True)

            if len(df) < self.ratio_period + 10:
                self.logger.debug(f"Insufficient aligned data: {len(df)} rows")
                return []

            # Calculate ratio
            df['ratio'] = df['close_btc'] / df['close_eth']

            # Calculate rolling mean and std
            df['ratio_ma'] = df['ratio'].rolling(window=self.ratio_period).mean()
            df['ratio_std'] = df['ratio'].rolling(window=self.ratio_period).std()

            # Calculate z-score
            df['z_score'] = (df['ratio'] - df['ratio_ma']) / df['ratio_std']

            # Drop NaN
            df = df.dropna()

            if len(df) < 5:
                return []

            # Current values
            current_z = df['z_score'].iloc[-1]
            current_ratio = df['ratio'].iloc[-1]
            btc_price = df['close_btc'].iloc[-1]
            eth_price = df['close_eth'].iloc[-1]

            # Check for entry signals
            if abs(current_z) > self.z_entry_threshold:
                if current_z > 0:
                    # Ratio too high: BTC expensive → BUY ETH
                    target_symbol = "ETH/USD"
                    direction = "buy"
                    rationale = "BTC/ETH ratio high, ETH undervalued"
                else:
                    # Ratio too low: ETH expensive → BUY BTC
                    target_symbol = "BTC/USD"
                    direction = "buy"
                    rationale = "BTC/ETH ratio low, BTC undervalued"

                # Calculate confidence based on z-score magnitude
                confidence = min(0.6 + (abs(current_z) - self.z_entry_threshold) * 0.1, 0.95)

                # Position size based on confidence
                position_size = self._calculate_position_size(confidence)

                # Calculate targets
                current_price = btc_price if target_symbol == "BTC/USD" else eth_price

                # Stop loss: if z-score moves against us by 1.0 more
                # Approximate price move (simplified)
                if current_z > 0:
                    # If buying ETH, stop if ETH drops significantly
                    stop_loss = current_price * 0.97  # 3% stop
                else:
                    # If buying BTC, stop if BTC drops significantly
                    stop_loss = current_price * 0.97  # 3% stop

                # Take profit: when z reverts to 0 (approximate)
                # Expected price move from z=2 to z=0 is ~2 standard deviations in ratio
                ratio_std_dollars = df['ratio_std'].iloc[-1]
                if current_z > 0:
                    # ETH should gain as ratio normalizes
                    expected_ratio_change = -current_z * ratio_std_dollars
                    expected_eth_gain = abs(expected_ratio_change) * btc_price / (current_ratio ** 2)
                    take_profit = current_price * (1 + min(expected_eth_gain / current_price, 0.10))
                else:
                    # BTC should gain as ratio normalizes
                    expected_ratio_change = -current_z * ratio_std_dollars
                    expected_btc_gain = abs(expected_ratio_change)
                    take_profit = current_price * (1 + min(expected_btc_gain / current_price, 0.10))

                signal = FleetSignal(
                    bot_name=self.name,
                    bot_type=self.bot_type.value,
                    symbol=target_symbol,
                    side=direction.upper(),
                    entry_price=current_price,
                    target_price=take_profit,
                    stop_loss=stop_loss,
                    quantity=position_size / current_price,
                    position_size_usd=position_size,
                    confidence=confidence,
                    edge=abs(current_z) * 10,  # z-score as edge proxy
                    reason=rationale,
                    metadata={
                        'z_score': float(current_z),
                        'ratio': float(current_ratio),
                        'ratio_ma': float(df['ratio_ma'].iloc[-1]),
                        'btc_price': float(btc_price),
                        'eth_price': float(eth_price),
                        'rationale': rationale
                    }
                )

                self.logger.info(
                    f"PAIRS SIGNAL: {rationale} | Z={current_z:.2f} | "
                    f"Ratio={current_ratio:.2f} | Target: {target_symbol} @ ${current_price:.2f} | "
                    f"Conf: {confidence:.2%}"
                )

                signals.append(signal)

                # Track position
                self.open_positions[target_symbol] = {
                    'direction': direction,
                    'entry_z': current_z,
                    'entry_time': datetime.now(timezone.utc)
                }

            # Check for exit signals on open positions
            for symbol in list(self.open_positions.keys()):
                if abs(current_z) < self.z_exit_threshold:
                    # Z-score reverted, generate exit signal
                    pos = self.open_positions[symbol]
                    current_price = btc_price if symbol == "BTC/USD" else eth_price

                    exit_signal = FleetSignal(
                        bot_name=self.name,
                        bot_type=self.bot_type.value,
                        symbol=symbol,
                        side='SELL',
                        entry_price=current_price,
                        target_price=0.0,
                        stop_loss=0.0,
                        quantity=0.0,  # Exit full position
                        position_size_usd=0.0,
                        confidence=0.80,  # High confidence on mean reversion exit
                        edge=0.0,
                        reason=f"Z reverted from {pos['entry_z']:.2f} to {current_z:.2f}",
                        metadata={
                            'z_score': float(current_z),
                            'entry_z': float(pos['entry_z']),
                            'exit_reason': 'mean_reversion',
                            'exit_signal': True
                        }
                    )

                    self.logger.info(
                        f"PAIRS EXIT: {symbol} | Z reverted from {pos['entry_z']:.2f} to {current_z:.2f}"
                    )

                    signals.append(exit_signal)

                    # Remove from tracking
                    del self.open_positions[symbol]

        except Exception as e:
            self.logger.error(f"Error in scan: {e}")
            self.logger.debug(traceback.format_exc())

        return signals

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence."""
        base = (self.config.max_position_usd * 0.5)
        max_size = self.config.max_position_usd

        # Linear scaling
        size = base + (max_size - base) * (confidence - self.config.min_confidence) / (1.0 - self.config.min_confidence)

        return round(size, 2)


def main():
    """Test runner."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = PairsTraderBot()

    print(f"\n{'='*60}")
    print(f"Pairs Trader Bot - Test Scan")
    print(f"{'='*60}\n")

    signals = bot.scan()

    if signals:
        print(f"Found {len(signals)} signal(s):\n")
        for sig in signals:
            print(f"Symbol: {sig.symbol}")
            print(f"Direction: {sig.direction.upper()}")
            print(f"Confidence: {sig.confidence:.2%}")
            print(f"Entry: ${sig.entry_price:.2f}")
            print(f"Stop Loss: ${sig.stop_loss:.2f if sig.stop_loss else 'N/A'}")
            print(f"Take Profit: ${sig.take_profit:.2f if sig.take_profit else 'N/A'}")
            print(f"Position Size: ${sig.position_size:.2f}")
            print(f"Metadata: {sig.metadata}")
            print(f"{'-'*60}\n")
    else:
        print("No signals generated.\n")


if __name__ == "__main__":
    main()
