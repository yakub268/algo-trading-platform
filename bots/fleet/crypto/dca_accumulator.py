"""
DCA Accumulator — Smart dollar-cost averaging with RSI and drawdown triggers.

Strategy:
- Base DCA: $10 every 15 minutes on BTC/USD and ETH/USD
- Enhanced triggers:
  - RSI < 30: double to $20
  - Price down > 5% in 24h: triple to $30
  - RSI < 20 AND price down > 8%: max buy $50
- Skips buy if RSI > 70 (overbought)
- Tracks average entry price in metadata

Schedule: 900s (15 minutes)
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import statistics

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType

try:
    from bots.alpaca_crypto_client import AlpacaCryptoClient
except ImportError:
    AlpacaCryptoClient = None


class DCAAccumulator(FleetBot):
    """Smart DCA bot with RSI and drawdown triggers."""

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="DCA-Accumulator",
                bot_type=BotType.CRYPTO,
                schedule_seconds=900,  # 15 minutes
                max_position_usd=50.0,
                max_daily_trades=96,  # 24h / 15min = 96 buys per symbol
                min_confidence=0.5,
                symbols=["BTC/USD", "ETH/USD"],
                enabled=True,
                paper_mode=True,
                extra={
                    'base_dca_amount': 10.0,
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_extreme': 20,
                    'rsi_overbought': 70,
                    'drawdown_threshold': 5.0,  # %
                    'extreme_drawdown': 8.0,    # %
                }
            )
        super().__init__(config)

        self.client = AlpacaCryptoClient() if AlpacaCryptoClient else None
        self.base_dca = config.extra.get('base_dca_amount', 10.0)
        self.rsi_period = config.extra.get('rsi_period', 14)
        self.rsi_oversold = config.extra.get('rsi_oversold', 30)
        self.rsi_extreme = config.extra.get('rsi_extreme', 20)
        self.rsi_overbought = config.extra.get('rsi_overbought', 70)
        self.drawdown_threshold = config.extra.get('drawdown_threshold', 5.0)
        self.extreme_drawdown = config.extra.get('extreme_drawdown', 8.0)

        # Track average entry prices
        self._avg_entry_prices: Dict[str, float] = {}
        self._total_invested: Dict[str, float] = {}

    def scan(self) -> List[FleetSignal]:
        """Generate DCA buy signals with enhanced triggers."""
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
        """Analyze a single symbol for DCA opportunity."""
        # Get current price
        current_price = self.client.get_price(symbol)
        if not current_price or current_price <= 0:
            self.logger.warning(f"Invalid price for {symbol}: {current_price}")
            return None

        # Get candles for RSI and drawdown
        candles = self._get_candles(symbol, limit=200)
        if not candles or len(candles) < self.rsi_period + 1:
            self.logger.warning(f"Insufficient candles for {symbol}")
            return None

        # Calculate RSI
        rsi = self._calculate_rsi(candles, period=self.rsi_period)
        if rsi is None:
            return None

        # Skip if overbought
        if rsi > self.rsi_overbought:
            self.logger.debug(f"Skipping {symbol}: RSI={rsi:.1f} (overbought)")
            return None

        # Calculate 24h drawdown
        drawdown_pct = self._calculate_24h_drawdown(candles, current_price)

        # Determine position size based on triggers
        position_size = self.base_dca  # Default
        confidence = 0.6
        reason_parts = ["DCA buy"]

        # RSI trigger
        if rsi < self.rsi_extreme:
            position_size = max(position_size, self.base_dca * 2)
            confidence = 0.85
            reason_parts.append(f"RSI={rsi:.1f} (extreme)")
        elif rsi < self.rsi_oversold:
            position_size = max(position_size, self.base_dca * 1.5)
            confidence = 0.75
            reason_parts.append(f"RSI={rsi:.1f} (oversold)")

        # Drawdown trigger
        if drawdown_pct >= self.extreme_drawdown:
            position_size = max(position_size, self.base_dca * 3)
            confidence = max(confidence, 0.9)
            reason_parts.append(f"down {drawdown_pct:.1f}% in 24h")
        elif drawdown_pct >= self.drawdown_threshold:
            position_size = max(position_size, self.base_dca * 2)
            confidence = max(confidence, 0.8)
            reason_parts.append(f"down {drawdown_pct:.1f}% in 24h")

        # Combined extreme trigger
        if rsi < self.rsi_extreme and drawdown_pct >= self.extreme_drawdown:
            position_size = 50.0  # Max buy
            confidence = 0.95
            reason_parts = [f"EXTREME: RSI={rsi:.1f}, down {drawdown_pct:.1f}%"]

        # Cap position size
        position_size = min(position_size, self.config.max_position_usd)

        # Update average entry tracking
        avg_entry = self._update_avg_entry(symbol, current_price, position_size)

        reason = " + ".join(reason_parts)

        signal = FleetSignal(
            bot_name=self.name,
            bot_type=self.bot_type.value,
            symbol=symbol,
            side="BUY",
            entry_price=current_price,
            target_price=0.0,  # No target for DCA
            stop_loss=0.0,     # No SL for DCA
            quantity=position_size / current_price,
            position_size_usd=position_size,
            confidence=confidence,
            edge=0.0,
            reason=reason,
            metadata={
                'rsi': rsi,
                'drawdown_24h': drawdown_pct,
                'avg_entry_price': avg_entry,
                'total_invested': self._total_invested.get(symbol, 0.0),
                'strategy': 'dca_accumulator',
            }
        )

        self.logger.info(
            f"DCA Signal: {symbol} ${position_size:.2f} | "
            f"RSI={rsi:.1f} | 24h={drawdown_pct:+.1f}% | conf={confidence:.2f}"
        )

        return signal

    def _get_candles(self, symbol: str, limit: int = 200) -> List[Dict]:
        """Fetch candles from Alpaca."""
        try:
            # Use 1-hour candles
            candles = self.client.get_candles(symbol, granularity='1h', limit=limit)
            if not candles:
                return []
            return candles
        except Exception as e:
            self.logger.error(f"Error fetching candles for {symbol}: {e}")
            return []

    def _calculate_rsi(self, candles: List[Dict], period: int = 14) -> Optional[float]:
        """Calculate RSI from candles."""
        if len(candles) < period + 1:
            return None

        try:
            closes = [c['close'] for c in candles[-period-1:]]
            deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]

            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]

            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        except Exception as e:
            self.logger.error(f"RSI calculation error: {e}")
            return None

    def _calculate_24h_drawdown(self, candles: List[Dict], current_price: float) -> float:
        """Calculate drawdown from 24h high."""
        try:
            # Get last 24 hours of candles (24 hourly candles)
            recent_candles = candles[-24:] if len(candles) >= 24 else candles
            if not recent_candles:
                return 0.0

            high_24h = max(c['high'] for c in recent_candles)
            if high_24h <= 0:
                return 0.0

            drawdown = ((current_price - high_24h) / high_24h) * 100
            return abs(drawdown) if drawdown < 0 else 0.0

        except Exception as e:
            self.logger.error(f"Drawdown calculation error: {e}")
            return 0.0

    def _update_avg_entry(self, symbol: str, price: float, amount: float) -> float:
        """Update and return average entry price."""
        if symbol not in self._avg_entry_prices:
            self._avg_entry_prices[symbol] = price
            self._total_invested[symbol] = amount
            return price

        # Weighted average
        old_total = self._total_invested.get(symbol, 0.0)
        old_avg = self._avg_entry_prices.get(symbol, price)

        new_total = old_total + amount
        new_avg = ((old_avg * old_total) + (price * amount)) / new_total

        self._avg_entry_prices[symbol] = new_avg
        self._total_invested[symbol] = new_total

        return new_avg

    def on_trade_result(self, signal: FleetSignal, success: bool, fill_info: Optional[Dict] = None):
        """Update tracking on successful trades."""
        super().on_trade_result(signal, success, fill_info)
        if success:
            # Entry price updated in _update_avg_entry during scan
            self.logger.info(
                f"DCA executed: {signal.symbol} ${signal.position_size_usd:.2f} | "
                f"Avg entry now: ${self._avg_entry_prices.get(signal.symbol, 0):.2f}"
            )
