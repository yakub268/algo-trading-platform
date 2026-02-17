"""
Grid Trader Bot
Strategy: Geometric grid trading in sideways markets
- ONLY trades when HMM regime == 'sideways'
- Sets up geometric grid around current price
- Buys at levels below, sells at levels above
- Profits from oscillation within range
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
    from filters.regime_detector import RegimeDetector
except ImportError:
    RegimeDetector = None

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None


class GridTraderBot(FleetBot):
    """
    Geometric grid trader for sideways markets.

    Logic:
    1. Check HMM regime → ONLY trade in 'sideways'
    2. Set up geometric grid (0.5% spacing, 10 levels each direction)
    3. BUY at levels below current price
    4. SELL at levels above current price
    5. Reset grid if price escapes range
    """

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="Grid-Trader",
                bot_type=BotType.CRYPTO,
                symbols=["BTC/USD"],  # Most liquid
                schedule_seconds=60,  # Fast response
                max_position_usd=20.0,
                max_daily_trades=100,
                min_confidence=0.50,
                enabled=True,
                paper_mode=True,
                extra={'base_position_size': 10.0}
            )

        super().__init__(config)

        # Grid parameters
        self.grid_spacing = 0.005  # 0.5% geometric spacing
        self.grid_levels = 10  # Levels above and below
        self.position_per_level = 10.0  # Fixed $10 per level

        # Regime detector
        self.regime_detector = None
        if RegimeDetector:
            try:
                self.regime_detector = RegimeDetector()
            except Exception as e:
                self.logger.warning(f"Failed to initialize RegimeDetector: {e}")

        # Alpaca client
        self.client = None
        if AlpacaCryptoClient:
            try:
                self.client = AlpacaCryptoClient()
            except Exception as e:
                self.logger.warning(f"Failed to initialize AlpacaCryptoClient: {e}")

        # Grid state tracking
        self.active_grid = {}  # symbol -> {'center': price, 'buy_levels': [], 'sell_levels': [], 'last_update': ts}

    def scan(self) -> List[FleetSignal]:
        """Scan for grid trading opportunities."""
        if not self.client:
            self.logger.error("AlpacaCryptoClient not available")
            return []

        if not self.regime_detector:
            self.logger.error("RegimeDetector not available")
            return []

        if np is None or pd is None:
            self.logger.error("numpy/pandas not available")
            return []

        signals = []

        for symbol in self.config.symbols:
            try:
                signal_list = self._scan_symbol(symbol)
                if signal_list:
                    signals.extend(signal_list)
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
                self.logger.debug(traceback.format_exc())

        return signals

    def _scan_symbol(self, symbol: str) -> List[FleetSignal]:
        """Scan single symbol for grid opportunities."""
        try:
            # Fetch candles for regime detection
            candles = self.client.get_candles(
                symbol=symbol,
                granularity='1Hour',
                limit=150
            )

            if not candles or len(candles) < 100:
                self.logger.debug(f"{symbol}: insufficient candles ({len(candles) if candles else 0})")
                return []

            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Detect regime
            regime = self.regime_detector.detect(df)

            # CRITICAL: Only trade in sideways regime
            if regime != 'sideways':
                self.logger.debug(f"{symbol}: regime '{regime}' != 'sideways', skipping grid")
                # Clear any active grid
                if symbol in self.active_grid:
                    del self.active_grid[symbol]
                return []

            # Get current price
            current_price = df['close'].iloc[-1]

            # Check if we need to reset grid (price escaped range)
            need_reset = False
            if symbol in self.active_grid:
                grid = self.active_grid[symbol]
                grid_min = min(grid['buy_levels']) if grid['buy_levels'] else grid['center']
                grid_max = max(grid['sell_levels']) if grid['sell_levels'] else grid['center']

                # Reset if price moved >5% from center
                price_change = abs(current_price - grid['center']) / grid['center']
                if price_change > 0.05:
                    self.logger.info(f"{symbol}: price escaped grid ({price_change:.2%}), resetting")
                    need_reset = True

            # Initialize or reset grid
            if symbol not in self.active_grid or need_reset:
                grid = self._initialize_grid(symbol, current_price)
                self.active_grid[symbol] = grid
                self.logger.info(
                    f"{symbol}: Grid initialized @ ${current_price:.2f} | "
                    f"Range: ${min(grid['buy_levels']):.2f} - ${max(grid['sell_levels']):.2f}"
                )

            grid = self.active_grid[symbol]

            # Generate signals for grid levels
            signals = []

            # BUY signals: price at or below buy levels
            for level in grid['buy_levels']:
                if current_price <= level * 1.001:  # Within 0.1% tolerance
                    # Calculate confidence based on distance from grid center
                    distance_ratio = abs(level - grid['center']) / grid['center']
                    confidence = 0.5 + (0.4 * distance_ratio / (self.grid_spacing * self.grid_levels))

                    # Stop loss: below next grid level
                    idx = grid['buy_levels'].index(level)
                    if idx < len(grid['buy_levels']) - 1:
                        stop_loss = grid['buy_levels'][idx + 1] * 0.995
                    else:
                        stop_loss = level * (1 - 2 * self.grid_spacing)

                    # Take profit: next sell level above current price
                    take_profit = None
                    for sell_level in grid['sell_levels']:
                        if sell_level > current_price:
                            take_profit = sell_level
                            break

                    if not take_profit:
                        take_profit = current_price * (1 + self.grid_spacing)

                    signal = FleetSignal(
                        bot_name=self.name,
                        bot_type=self.bot_type.value,
                        symbol=symbol,
                        side='BUY',
                        entry_price=current_price,
                        target_price=take_profit,
                        stop_loss=stop_loss,
                        quantity=self.position_per_level / current_price,
                        position_size_usd=self.position_per_level,
                        confidence=confidence,
                        edge=distance_ratio * 100,
                        reason=f"Grid buy at {level:.2f} in sideways regime",
                        metadata={
                            'grid_level': float(level),
                            'grid_center': float(grid['center']),
                            'grid_type': 'buy',
                            'regime': regime
                        }
                    )

                    signals.append(signal)

            # SELL signals: price at or above sell levels
            # NOTE: For crypto, we can't short on Alpaca, so these are profit-taking signals
            # The orchestrator should handle these as exits for existing long positions
            for level in grid['sell_levels']:
                if current_price >= level * 0.999:  # Within 0.1% tolerance
                    distance_ratio = abs(level - grid['center']) / grid['center']
                    confidence = 0.5 + (0.4 * distance_ratio / (self.grid_spacing * self.grid_levels))

                    # For sell signals in spot crypto, these are take-profit targets
                    # We'll generate them but mark as 'sell' for orchestrator
                    signal = FleetSignal(
                        bot_name=self.name,
                        bot_type=self.bot_type.value,
                        symbol=symbol,
                        side='SELL',
                        entry_price=current_price,
                        target_price=0.0,
                        stop_loss=0.0,
                        quantity=self.position_per_level / current_price,
                        position_size_usd=self.position_per_level,
                        confidence=confidence,
                        edge=distance_ratio * 100,
                        reason=f"Grid sell at {level:.2f} in sideways regime",
                        metadata={
                            'grid_level': float(level),
                            'grid_center': float(grid['center']),
                            'grid_type': 'sell',
                            'regime': regime,
                            'exit_signal': True  # Flag for orchestrator
                        }
                    )

                    signals.append(signal)

            if signals:
                self.logger.info(
                    f"{symbol}: Generated {len(signals)} grid signals @ ${current_price:.2f} | Regime: {regime}"
                )

            return signals

        except Exception as e:
            self.logger.error(f"Error in _scan_symbol for {symbol}: {e}")
            self.logger.debug(traceback.format_exc())
            return []

    def _initialize_grid(self, symbol: str, center_price: float) -> Dict:
        """Initialize geometric grid around center price."""
        buy_levels = []
        sell_levels = []

        # Calculate buy levels (below center)
        for i in range(1, self.grid_levels + 1):
            level = center_price * (1 - i * self.grid_spacing)
            buy_levels.append(level)

        # Calculate sell levels (above center)
        for i in range(1, self.grid_levels + 1):
            level = center_price * (1 + i * self.grid_spacing)
            sell_levels.append(level)

        return {
            'center': center_price,
            'buy_levels': sorted(buy_levels, reverse=True),  # Descending
            'sell_levels': sorted(sell_levels),  # Ascending
            'last_update': datetime.now(timezone.utc)
        }


def main():
    """Test runner."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = GridTraderBot()

    print(f"\n{'='*60}")
    print(f"Grid Trader Bot - Test Scan")
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
