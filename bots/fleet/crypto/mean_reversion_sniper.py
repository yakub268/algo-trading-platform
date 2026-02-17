"""
Mean Reversion Sniper — Bollinger Band breakout with RSI and volume confirmation.

Strategy:
- BB(20, 2.5σ) mean reversion on ETH/USD, SOL/USD, AVAX/USD
- Entry: price < lower BB AND RSI < 25 AND volume > 1.5x avg
- Exit: middle BB (SMA 20)
- Stop loss: 3% below entry
- Position size: $25-50 based on distance below BB
- High confidence when far below BB

Schedule: 300s (5 minutes)
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
import statistics

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType

try:
    from bots.alpaca_crypto_client import AlpacaCryptoClient
except ImportError:
    AlpacaCryptoClient = None


class MeanReversionSniper(FleetBot):
    """Mean reversion bot using Bollinger Bands, RSI, and volume."""

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="Mean-Reversion-Sniper",
                bot_type=BotType.CRYPTO,
                schedule_seconds=300,  # 5 minutes
                max_position_usd=50.0,
                max_daily_trades=20,
                min_confidence=0.65,
                symbols=["ETH/USD", "SOL/USD", "AVAX/USD"],
                enabled=True,
                paper_mode=True,
                extra={
                    'bb_period': 20,
                    'bb_std_dev': 2.5,
                    'rsi_period': 14,
                    'rsi_threshold': 25,
                    'volume_multiplier': 1.5,
                    'stop_loss_pct': 3.0,
                    'min_position': 25.0,
                    'max_position': 50.0,
                }
            )
        super().__init__(config)

        self.client = AlpacaCryptoClient() if AlpacaCryptoClient else None
        self.bb_period = config.extra.get('bb_period', 20)
        self.bb_std_dev = config.extra.get('bb_std_dev', 2.5)
        self.rsi_period = config.extra.get('rsi_period', 14)
        self.rsi_threshold = config.extra.get('rsi_threshold', 25)
        self.volume_multiplier = config.extra.get('volume_multiplier', 1.5)
        self.stop_loss_pct = config.extra.get('stop_loss_pct', 3.0)
        self.min_position = config.extra.get('min_position', 25.0)
        self.max_position = config.extra.get('max_position', 50.0)

        # Track open positions for exit management
        self._open_positions: Dict[str, Dict] = {}

    def scan(self) -> List[FleetSignal]:
        """Scan for mean reversion entries."""
        if not self.client or not self.client._initialized:
            self.logger.warning("Alpaca client not initialized")
            return []

        signals = []

        for symbol in self.config.symbols:
            try:
                signal = self._analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                continue

        return signals

    def _analyze_symbol(self, symbol: str) -> Optional[FleetSignal]:
        """Analyze symbol for mean reversion setup."""
        # Get current price
        current_price = self.client.get_price(symbol)
        if not current_price or current_price <= 0:
            self.logger.warning(f"Invalid price for {symbol}: {current_price}")
            return None

        # Get candles
        candles = self._get_candles(symbol, limit=100)
        if not candles or len(candles) < self.bb_period + 1:
            self.logger.warning(f"Insufficient candles for {symbol}")
            return None

        # Calculate indicators
        bb = self._calculate_bollinger_bands(candles)
        if not bb:
            return None

        rsi = self._calculate_rsi(candles)
        if rsi is None:
            return None

        volume_check = self._check_volume(candles)

        # Entry conditions:
        # 1. Price below lower BB
        # 2. RSI < threshold
        # 3. Volume > average
        lower_bb = bb['lower']
        middle_bb = bb['middle']
        upper_bb = bb['upper']

        if current_price >= lower_bb:
            self.logger.debug(f"{symbol}: Price {current_price:.2f} not below BB {lower_bb:.2f}")
            return None

        if rsi >= self.rsi_threshold:
            self.logger.debug(f"{symbol}: RSI {rsi:.1f} not oversold")
            return None

        if not volume_check['elevated']:
            self.logger.debug(f"{symbol}: Volume not elevated")
            return None

        # Calculate distance below BB for position sizing
        bb_distance_pct = ((lower_bb - current_price) / lower_bb) * 100

        # Position size: scale from min to max based on how far below BB
        # 1% below BB = min, 5%+ below BB = max
        position_multiplier = min(bb_distance_pct / 5.0, 1.0)
        position_size = self.min_position + (self.max_position - self.min_position) * position_multiplier
        position_size = max(self.min_position, min(position_size, self.max_position))

        # Confidence: higher when further below BB and lower RSI
        base_confidence = 0.65
        bb_bonus = min(bb_distance_pct * 0.05, 0.20)  # Up to +0.20
        rsi_bonus = (self.rsi_threshold - rsi) / 100  # Up to ~0.25
        confidence = min(base_confidence + bb_bonus + rsi_bonus, 0.95)

        # Targets
        stop_loss = current_price * (1 - self.stop_loss_pct / 100)
        target_price = middle_bb  # Exit at middle BB

        reason = (
            f"Below BB by {bb_distance_pct:.1f}%, RSI={rsi:.1f}, "
            f"Vol={volume_check['ratio']:.2f}x avg"
        )

        signal = FleetSignal(
            bot_name=self.name,
            bot_type=self.bot_type.value,
            symbol=symbol,
            side="BUY",
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            quantity=position_size / current_price,
            position_size_usd=position_size,
            confidence=confidence,
            edge=bb_distance_pct,  # Distance below BB as edge proxy
            reason=reason,
            metadata={
                'bb_lower': lower_bb,
                'bb_middle': middle_bb,
                'bb_upper': upper_bb,
                'bb_distance_pct': bb_distance_pct,
                'rsi': rsi,
                'volume_ratio': volume_check['ratio'],
                'strategy': 'mean_reversion_sniper',
            }
        )

        self.logger.info(
            f"Mean Reversion Signal: {symbol} ${position_size:.2f} | "
            f"Below BB {bb_distance_pct:.1f}% | RSI={rsi:.1f} | conf={confidence:.2f}"
        )

        return signal

    def check_exits(self) -> List[FleetSignal]:
        """Check for exit conditions on open positions."""
        # TODO: Implement exit checking based on middle BB or stop loss
        # For now, let orchestrator handle exits
        return []

    def _get_candles(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Fetch candles from Alpaca."""
        try:
            candles = self.client.get_candles(symbol, granularity='5m', limit=limit)
            if not candles:
                return []
            return candles
        except Exception as e:
            self.logger.error(f"Error fetching candles for {symbol}: {e}")
            return []

    def _calculate_bollinger_bands(self, candles: List[Dict]) -> Optional[Dict]:
        """Calculate Bollinger Bands."""
        if len(candles) < self.bb_period:
            return None

        try:
            closes = [c['close'] for c in candles[-self.bb_period:]]

            # Middle band = SMA
            middle = statistics.mean(closes)

            # Standard deviation
            std_dev = statistics.stdev(closes)

            # Upper and lower bands
            upper = middle + (self.bb_std_dev * std_dev)
            lower = middle - (self.bb_std_dev * std_dev)

            return {
                'middle': middle,
                'upper': upper,
                'lower': lower,
                'std_dev': std_dev,
            }

        except Exception as e:
            self.logger.error(f"BB calculation error: {e}")
            return None

    def _calculate_rsi(self, candles: List[Dict]) -> Optional[float]:
        """Calculate RSI from candles."""
        if len(candles) < self.rsi_period + 1:
            return None

        try:
            closes = [c['close'] for c in candles[-self.rsi_period-1:]]
            deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]

            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]

            avg_gain = sum(gains) / self.rsi_period
            avg_loss = sum(losses) / self.rsi_period

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        except Exception as e:
            self.logger.error(f"RSI calculation error: {e}")
            return None

    def _check_volume(self, candles: List[Dict]) -> Dict:
        """Check if current volume is elevated."""
        if len(candles) < 20:
            return {'elevated': False, 'ratio': 0.0}

        try:
            # Get last 20 volumes
            volumes = [c['volume'] for c in candles[-20:]]
            avg_volume = statistics.mean(volumes)

            # Current volume
            current_volume = candles[-1]['volume']

            if avg_volume <= 0:
                return {'elevated': False, 'ratio': 0.0}

            volume_ratio = current_volume / avg_volume
            elevated = volume_ratio >= self.volume_multiplier

            return {
                'elevated': elevated,
                'ratio': volume_ratio,
                'current': current_volume,
                'average': avg_volume,
            }

        except Exception as e:
            self.logger.error(f"Volume check error: {e}")
            return {'elevated': False, 'ratio': 0.0}

    def on_trade_result(self, signal: FleetSignal, success: bool, fill_info: Optional[Dict] = None):
        """Track open positions for exit management."""
        super().on_trade_result(signal, success, fill_info)
        if success and signal.side == "BUY":
            self._open_positions[signal.symbol] = {
                'entry_price': signal.entry_price,
                'target_price': signal.target_price,
                'stop_loss': signal.stop_loss,
                'timestamp': signal.timestamp,
                'trade_id': signal.trade_id,
            }
