"""
Meme Coin Sniper Bot
====================

Aggressive trading bot targeting volatile meme coins on Coinbase.
Looks for momentum (price up 10%+ in 2h) or volume spikes (5x normal).

Strategy:
- Quick scalp trades with 5% take profit, 4% stop loss
- Max hold time: 1 hour (memes can reverse fast)
- Small position sizes ($30-50) due to high volatility

Target Coins (USDC pairs):
PEPE, BONK, WIF, SHIB, DOGE, FLOKI, TRUMP, PNUT, MOG, TURBO, POPCAT, BRETT

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bots.aggressive.base_aggressive_bot import BaseAggressiveBot, TradeSignal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MemeCoinSniper')


class MemeCoinSniper(BaseAggressiveBot):
    """
    Meme coin sniper - catches momentum moves in volatile meme tokens.

    Entry Signals:
    - Price up 10%+ in last 2 hours, OR
    - Volume spike 5x above average

    Exit Strategy:
    - Take profit: 5%
    - Stop loss: 4%
    - Max hold time: 1 hour
    """

    # Target meme coins - will check which have USDC pairs
    TARGET_COINS = [
        'PEPE', 'BONK', 'WIF', 'SHIB', 'DOGE', 'FLOKI',
        'TRUMP', 'PNUT', 'MOG', 'TURBO', 'POPCAT', 'BRETT'
    ]

    # Entry thresholds
    MOMENTUM_THRESHOLD = 0.10  # 10% price increase
    VOLUME_SPIKE_MULTIPLIER = 5.0  # 5x normal volume
    LOOKBACK_HOURS = 2

    # Position and risk parameters
    MIN_POSITION_SIZE = 30.0  # $30 minimum
    MAX_POSITION_SIZE = 50.0  # $50 maximum
    TAKE_PROFIT_PCT = 0.05  # 5%
    STOP_LOSS_PCT = 0.04  # 4%
    MAX_HOLD_SECONDS = 3600  # 1 hour

    def __init__(
        self,
        capital: float = 200.0,
        paper_mode: bool = None,
        position_size: float = None,
    ):
        """
        Initialize Meme Coin Sniper.

        Args:
            capital: Total capital for this bot
            paper_mode: Paper trading mode
            position_size: Override position size (default $30-50)
        """
        # Use class defaults for risk parameters
        super().__init__(
            capital=capital,
            paper_mode=paper_mode,
            position_size=position_size or self.MAX_POSITION_SIZE,
            take_profit_pct=self.TAKE_PROFIT_PCT,
            stop_loss_pct=self.STOP_LOSS_PCT,
            max_positions=4,  # Allow up to 4 concurrent meme positions
        )

        # Validated USDC trading pairs
        self.valid_pairs: List[str] = []
        self._validate_pairs()

        # Volume tracking for spike detection
        self.volume_history: Dict[str, List[float]] = {}

        logger.info(
            f"MemeCoinSniper initialized - "
            f"Targets: {len(self.valid_pairs)} coins, "
            f"Momentum threshold: {self.MOMENTUM_THRESHOLD:.0%}, "
            f"Volume spike: {self.VOLUME_SPIKE_MULTIPLIER}x"
        )

    def _validate_pairs(self):
        """Check which target coins have valid USDC trading pairs on Coinbase"""
        self.valid_pairs = []

        if not self.coinbase._initialized:
            logger.warning("Coinbase not initialized - using default pairs for paper trading")
            # Default pairs known to exist (fallback for paper mode)
            self.valid_pairs = [
                'DOGE-USDC', 'SHIB-USDC', 'PEPE-USDC', 'BONK-USDC', 'FLOKI-USDC'
            ]
            return

        for coin in self.TARGET_COINS:
            symbol = f"{coin}-USDC"
            try:
                product = self.coinbase.get_product(symbol)
                if product:
                    # Check if trading is enabled
                    status = getattr(product, 'status', None) or product.get('status', '')
                    if status == 'online' or not status:  # Some APIs don't return status
                        self.valid_pairs.append(symbol)
                        logger.debug(f"  Valid pair: {symbol}")
            except Exception as e:
                logger.debug(f"  {symbol} not available: {e}")

        logger.info(f"Found {len(self.valid_pairs)} valid meme coin pairs: {self.valid_pairs}")

    def _get_2h_price_change(self, symbol: str) -> Optional[float]:
        """
        Calculate price change over last 2 hours.

        Returns:
            Percentage change (0.10 = 10%) or None if data unavailable
        """
        # Get hourly candles for last 3 hours (need buffer)
        candles = self.get_candles(symbol, granularity="ONE_HOUR", limit=3)

        if len(candles) < 2:
            logger.debug(f"{symbol}: Insufficient candle data")
            return None

        # Price 2 hours ago (oldest candle open)
        price_2h_ago = candles[0]['open']

        # Current price (most recent candle close)
        current_price = candles[-1]['close']

        if price_2h_ago == 0:
            return None

        change = (current_price - price_2h_ago) / price_2h_ago
        logger.debug(f"{symbol}: 2h change = {change:.2%} (${price_2h_ago:.6f} -> ${current_price:.6f})")

        return change

    def _get_volume_spike(self, symbol: str) -> Optional[float]:
        """
        Calculate volume spike ratio vs average.

        Returns:
            Spike multiplier (5.0 = 5x normal) or None if data unavailable
        """
        # Get hourly candles for volume comparison
        candles = self.get_candles(symbol, granularity="ONE_HOUR", limit=24)

        if len(candles) < 6:
            logger.debug(f"{symbol}: Insufficient volume data")
            return None

        # Current hour volume
        current_volume = candles[-1]['volume']

        # Average volume (excluding current hour)
        historical_volumes = [c['volume'] for c in candles[:-1]]
        avg_volume = sum(historical_volumes) / len(historical_volumes)

        if avg_volume == 0:
            return None

        spike = current_volume / avg_volume
        logger.debug(f"{symbol}: Volume spike = {spike:.1f}x (current: {current_volume:.0f}, avg: {avg_volume:.0f})")

        return spike

    def _calculate_position_size(self, confidence: float) -> float:
        """
        Calculate position size based on confidence.

        Higher confidence = larger position (up to MAX)
        """
        # Scale between MIN and MAX based on confidence
        size_range = self.MAX_POSITION_SIZE - self.MIN_POSITION_SIZE
        size = self.MIN_POSITION_SIZE + (size_range * confidence)

        # Ensure we don't exceed capital limits
        max_from_capital = self.capital / self.max_positions
        return min(size, max_from_capital, self.MAX_POSITION_SIZE)

    def run_scan(self) -> List[TradeSignal]:
        """
        Scan meme coins for entry opportunities.

        Returns:
            List of TradeSignal objects for coins meeting criteria
        """
        signals = []
        now = datetime.now(timezone.utc)

        logger.info(f"Scanning {len(self.valid_pairs)} meme coins for opportunities...")

        # Skip coins we already have positions in
        coins_to_scan = [p for p in self.valid_pairs if p not in self.active_positions]

        if len(self.active_positions) >= self.max_positions:
            logger.info(f"Max positions ({self.max_positions}) reached - checking exits only")
            return signals

        for symbol in coins_to_scan:
            try:
                # Get current price
                current_price = self.get_price(symbol)
                if current_price is None:
                    continue

                # Check momentum signal
                price_change = self._get_2h_price_change(symbol)
                momentum_signal = price_change is not None and price_change >= self.MOMENTUM_THRESHOLD

                # Check volume spike signal
                volume_spike = self._get_volume_spike(symbol)
                volume_signal = volume_spike is not None and volume_spike >= self.VOLUME_SPIKE_MULTIPLIER

                if momentum_signal or volume_signal:
                    # Determine reason and confidence
                    reasons = []
                    confidence = 0.5  # Base confidence

                    if momentum_signal:
                        reasons.append(f"Price +{price_change:.1%} in 2h")
                        confidence += 0.2

                    if volume_signal:
                        reasons.append(f"Volume {volume_spike:.1f}x normal")
                        confidence += 0.2

                    # Both signals = higher confidence
                    if momentum_signal and volume_signal:
                        confidence += 0.1

                    confidence = min(confidence, 1.0)
                    reason = " | ".join(reasons)

                    # Calculate targets
                    target_price = current_price * (1 + self.TAKE_PROFIT_PCT)
                    stop_loss = current_price * (1 - self.STOP_LOSS_PCT)
                    position_size = self._calculate_position_size(confidence)

                    signal = TradeSignal(
                        symbol=symbol,
                        side='BUY',
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        position_size_usd=position_size,
                        confidence=confidence,
                        reason=reason,
                        timestamp=now,
                        metadata={
                            'price_change_2h': price_change,
                            'volume_spike': volume_spike,
                            'max_hold_seconds': self.MAX_HOLD_SECONDS,
                        }
                    )

                    signals.append(signal)
                    logger.info(
                        f"SIGNAL: {symbol} @ ${current_price:.6f} - {reason} "
                        f"(confidence: {confidence:.0%})"
                    )

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue

        # Sort by confidence (best opportunities first)
        signals.sort(key=lambda s: s.confidence, reverse=True)

        logger.info(f"Scan complete - {len(signals)} opportunities found")
        return signals

    def execute_signal(self, signal: TradeSignal) -> Optional[dict]:
        """Execute signal with max hold time tracking"""
        trade = super().execute_signal(signal)

        if trade and trade.get('status') == 'filled':
            # Add max hold time to position tracking
            if signal.symbol in self.active_positions:
                self.active_positions[signal.symbol]['max_hold_time'] = self.MAX_HOLD_SECONDS

        return trade

    def get_status(self) -> dict:
        """Get enhanced status for meme sniper"""
        status = super().get_status()
        status.update({
            'target_coins': len(self.valid_pairs),
            'valid_pairs': self.valid_pairs,
            'momentum_threshold': self.MOMENTUM_THRESHOLD,
            'volume_spike_multiplier': self.VOLUME_SPIKE_MULTIPLIER,
            'max_hold_time_seconds': self.MAX_HOLD_SECONDS,
        })
        return status


if __name__ == "__main__":
    print("=" * 60)
    print("MEME COIN SNIPER BOT")
    print("=" * 60)

    # Initialize bot in paper mode
    bot = MemeCoinSniper(capital=200.0, paper_mode=True)

    print(f"\nStatus: {bot.get_status()}")

    print(f"\n--- Valid Trading Pairs ---")
    for pair in bot.valid_pairs:
        print(f"  {pair}")

    print("\n--- Running Scan ---")
    signals = bot.run_scan()

    if signals:
        print(f"\nFound {len(signals)} trade signals:")
        for signal in signals:
            print(
                f"  {signal.symbol}: {signal.side} @ ${signal.entry_price:.6f}, "
                f"Target: ${signal.target_price:.6f}, Stop: ${signal.stop_loss:.6f}, "
                f"Size: ${signal.position_size_usd:.2f}, Confidence: {signal.confidence:.0%}"
            )
            print(f"    Reason: {signal.reason}")

        # Execute top signal in paper mode
        print("\n--- Executing Top Signal (Paper) ---")
        trade = bot.execute_signal(signals[0])
        print(f"Trade result: {trade}")

    else:
        print("\nNo trade signals found (market conditions not met)")
        print("This is normal - meme coins need momentum or volume spikes to trigger")

    print("\n" + "=" * 60)
