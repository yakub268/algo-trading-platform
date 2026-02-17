"""
Momentum Weekly — Long-only BTC/ETH momentum (Han et al. 2024).

Strategy:
- Academic finding: BTC weekly momentum is significant LONG-ONLY (t-stat 3.28)
- Calculate 1-week and 4-week returns
- Buy signal: 1-week return > 0 AND 4-week return > 0 (both positive)
- Position size: $20-50 scaled by momentum strength
- Hold for 1 week, but check exits more frequently
- Exit: price drops 5% from peak, or 4-week return turns negative
- LONG ONLY — never short

Schedule: 604800s (1 week) for entries, but run more frequently for exit checks
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType

try:
    from bots.alpaca_crypto_client import AlpacaCryptoClient
except ImportError:
    AlpacaCryptoClient = None


class MomentumWeekly(FleetBot):
    """Long-only weekly momentum bot for BTC/ETH."""

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="Momentum-Weekly",
                bot_type=BotType.CRYPTO,
                schedule_seconds=21600,  # 6 hours (check frequently for exits)
                max_position_usd=50.0,
                max_daily_trades=4,
                min_confidence=0.60,
                symbols=["BTC/USD", "ETH/USD"],
                enabled=True,
                paper_mode=True,
                extra={
                    'weekly_entry_interval': 604800,  # 7 days for new entries
                    'one_week_hours': 168,   # 7 days in hours
                    'four_week_hours': 672,  # 28 days in hours
                    'min_position': 20.0,
                    'max_position': 50.0,
                    'peak_drawdown_exit': 5.0,  # % from peak
                    'momentum_threshold': 0.0,  # Must be positive (>0%)
                }
            )
        super().__init__(config)

        self.client = AlpacaCryptoClient() if AlpacaCryptoClient else None
        self.weekly_entry_interval = config.extra.get('weekly_entry_interval', 604800)
        self.one_week_hours = config.extra.get('one_week_hours', 168)
        self.four_week_hours = config.extra.get('four_week_hours', 672)
        self.min_position = config.extra.get('min_position', 20.0)
        self.max_position = config.extra.get('max_position', 50.0)
        self.peak_drawdown_exit = config.extra.get('peak_drawdown_exit', 5.0)
        self.momentum_threshold = config.extra.get('momentum_threshold', 0.0)

        # Track open positions for exit management
        self._open_positions: Dict[str, Dict] = {}
        self._last_entry_time: Dict[str, datetime] = {}

    def scan(self) -> List[FleetSignal]:
        """Scan for momentum entries (weekly)."""
        if not self.client or not self.client._initialized:
            self.logger.warning("Alpaca client not initialized")
            return []

        signals = []

        for symbol in self.config.symbols:
            try:
                # Check if we should enter (weekly interval)
                if not self._should_check_entry(symbol):
                    self.logger.debug(f"{symbol}: Not time for entry check yet")
                    continue

                signal = self._analyze_symbol(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                continue

        return signals

    def check_exits(self) -> List[FleetSignal]:
        """Check exit conditions for open positions."""
        # This would be called by orchestrator to check if positions should close
        # For now, exits handled via metadata in on_trade_result
        return []

    def _should_check_entry(self, symbol: str) -> bool:
        """Check if enough time has passed since last entry."""
        if symbol not in self._last_entry_time:
            return True

        elapsed = (datetime.now(timezone.utc) - self._last_entry_time[symbol]).total_seconds()
        return elapsed >= self.weekly_entry_interval

    def _analyze_symbol(self, symbol: str) -> Optional[FleetSignal]:
        """Analyze symbol for momentum signal."""
        # Get current price
        current_price = self.client.get_price(symbol)
        if not current_price or current_price <= 0:
            self.logger.warning(f"Invalid price for {symbol}: {current_price}")
            return None

        # Get candles for momentum calculation
        # Need at least 4 weeks of hourly data
        candles = self._get_candles(symbol, limit=800)  # ~33 days
        if not candles or len(candles) < self.four_week_hours:
            self.logger.warning(f"Insufficient candles for {symbol}: {len(candles) if candles else 0}")
            return None

        # Calculate returns
        one_week_return = self._calculate_return(candles, self.one_week_hours)
        four_week_return = self._calculate_return(candles, self.four_week_hours)

        if one_week_return is None or four_week_return is None:
            return None

        # Entry condition: BOTH 1-week and 4-week returns must be positive
        if one_week_return <= self.momentum_threshold:
            self.logger.debug(
                f"{symbol}: 1-week return {one_week_return:.2f}% not positive"
            )
            return None

        if four_week_return <= self.momentum_threshold:
            self.logger.debug(
                f"{symbol}: 4-week return {four_week_return:.2f}% not positive"
            )
            return None

        # Calculate combined momentum strength
        momentum_strength = (one_week_return + four_week_return) / 2

        # Position sizing: scale from min to max based on momentum
        # 0-5% avg momentum = min, 10%+ = max
        position_multiplier = min(momentum_strength / 10.0, 1.0)
        position_size = self.min_position + (self.max_position - self.min_position) * position_multiplier
        position_size = max(self.min_position, min(position_size, self.max_position))

        # Confidence: higher momentum = higher confidence
        base_confidence = 0.60
        momentum_bonus = min(momentum_strength * 0.02, 0.30)  # Up to +0.30
        confidence = min(base_confidence + momentum_bonus, 0.90)

        reason = (
            f"Dual momentum: 1w={one_week_return:+.1f}%, "
            f"4w={four_week_return:+.1f}%, avg={momentum_strength:.1f}%"
        )

        # Calculate trailing stop (5% from peak)
        stop_loss = current_price * (1 - self.peak_drawdown_exit / 100)

        signal = FleetSignal(
            bot_name=self.name,
            bot_type=self.bot_type.value,
            symbol=symbol,
            side="BUY",
            entry_price=current_price,
            target_price=0.0,  # No fixed target
            stop_loss=stop_loss,
            quantity=position_size / current_price,
            position_size_usd=position_size,
            confidence=confidence,
            edge=momentum_strength,
            reason=reason,
            metadata={
                'one_week_return': one_week_return,
                'four_week_return': four_week_return,
                'momentum_strength': momentum_strength,
                'peak_price': current_price,
                'trailing_stop_pct': self.peak_drawdown_exit,
                'entry_timestamp': datetime.now(timezone.utc).isoformat(),
                'strategy': 'momentum_weekly',
            }
        )

        self.logger.info(
            f"Momentum Signal: {symbol} ${position_size:.2f} | "
            f"1w={one_week_return:+.1f}% 4w={four_week_return:+.1f}% | "
            f"conf={confidence:.2f}"
        )

        return signal

    def _calculate_return(self, candles: List[Dict], lookback_hours: int) -> Optional[float]:
        """Calculate return over lookback period."""
        if len(candles) < lookback_hours:
            return None

        try:
            # Current price (last candle close)
            current_price = candles[-1]['close']

            # Price N hours ago
            past_price = candles[-lookback_hours]['close']

            if past_price <= 0:
                return None

            return_pct = ((current_price - past_price) / past_price) * 100
            return return_pct

        except Exception as e:
            self.logger.error(f"Return calculation error: {e}")
            return None

    def _get_candles(self, symbol: str, limit: int = 800) -> List[Dict]:
        """Fetch hourly candles from Alpaca."""
        try:
            candles = self.client.get_candles(symbol, granularity='1h', limit=limit)
            if not candles:
                return []
            return candles
        except Exception as e:
            self.logger.error(f"Error fetching candles for {symbol}: {e}")
            return []

    def on_trade_result(self, signal: FleetSignal, success: bool, fill_info: Optional[Dict] = None):
        """Track entries and update last entry time."""
        super().on_trade_result(signal, success, fill_info)
        if success and signal.side == "BUY":
            self._last_entry_time[signal.symbol] = datetime.now(timezone.utc)
            self._open_positions[signal.symbol] = {
                'entry_price': signal.entry_price,
                'peak_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'timestamp': signal.timestamp,
                'trade_id': signal.trade_id,
                'metadata': signal.metadata,
            }
            self.logger.info(
                f"Momentum entry: {signal.symbol} @ ${signal.entry_price:.2f}, "
                f"hold until 4w momentum turns negative or -5% from peak"
            )

    def update_position_tracking(self, symbol: str, current_price: float) -> bool:
        """Update peak price for trailing stop. Returns True if should exit."""
        if symbol not in self._open_positions:
            return False

        pos = self._open_positions[symbol]

        # Update peak price
        if current_price > pos['peak_price']:
            pos['peak_price'] = current_price
            pos['stop_loss'] = current_price * (1 - self.peak_drawdown_exit / 100)
            self.logger.debug(
                f"{symbol}: New peak ${current_price:.2f}, "
                f"trailing stop now ${pos['stop_loss']:.2f}"
            )

        # Check if stop hit
        if current_price <= pos['stop_loss']:
            self.logger.info(
                f"{symbol}: Trailing stop hit at ${current_price:.2f} "
                f"(peak was ${pos['peak_price']:.2f})"
            )
            return True

        return False
