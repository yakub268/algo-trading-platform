"""
Longshot Fader Bot - Exploits Favorite-Longshot Bias on Kalshi

Based on Becker 2025 research showing 72.1M Kalshi trades exhibit systematic
mispricing: longshots (low probability events) are overpriced, favorites underpriced.

Strategy:
- Scan ALL open Kalshi markets for contracts where YES ask is 3-7 cents
- These are systematically overpriced longshots
- Buy NO on these contracts (expected +EV)
- Position size: $5-15 per contract (portfolio approach with many small bets)
- Expected: high win rate (~85-90%), but many positions needed for edge to materialize
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
    from bots.kalshi_client import KalshiClient
    KALSHI_AVAILABLE = True
except ImportError:
    KALSHI_AVAILABLE = False
    logging.warning("KalshiClient not available - LongshotFader will run in dry-run mode")


class LongshotFader(FleetBot):
    """
    Exploits favorite-longshot bias by fading (buying NO on) overpriced longshots.

    Research basis: Becker (2025) analysis of 72.1M Kalshi trades shows
    contracts priced 3-7 cents have realized probabilities ~40-60% lower than price.
    """

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="Longshot-Fader",
                bot_type=BotType.KALSHI,
                schedule_seconds=300,  # 5 minutes
                max_position_usd=15.0,
                max_daily_trades=30,
                min_confidence=0.6,
                symbols=[],
                enabled=True,
                paper_mode=True,
                extra={
                    'min_position_size_usd': 5.0,
                    'stop_loss_pct': 0.50,
                    'take_profit_pct': 0.80,
                }
            )

        super().__init__(config)

        # Strategy parameters
        self.MIN_LONGSHOT_PRICE = 0.03  # 3 cents - clear longshot
        self.MAX_LONGSHOT_PRICE = 0.07  # 7 cents - still longshot territory
        self.MIN_HOURS_TO_CLOSE = 24    # Avoid contracts closing soon
        self.MIN_VOLUME = 1             # Must have some liquidity
        self.MAX_SPREAD_WIDTH = 0.05    # 5 cents max spread

        # Price bands for fair value estimation (from Becker research)
        self.FAIR_VALUE_ADJUSTMENTS = {
            0.03: 0.015,  # 3 cent contract -> ~1.5% fair value
            0.04: 0.020,  # 4 cent -> ~2%
            0.05: 0.025,  # 5 cent -> ~2.5%
            0.06: 0.032,  # 6 cent -> ~3.2%
            0.07: 0.040,  # 7 cent -> ~4%
        }

        # Initialize Kalshi client
        self.kalshi_client = None
        if KALSHI_AVAILABLE:
            try:
                self.kalshi_client = KalshiClient()
                self.logger.info("KalshiClient initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize KalshiClient: {e}")

    def scan(self) -> List[FleetSignal]:
        """
        Scan all open Kalshi markets for overpriced longshots.

        Returns:
            List of FleetSignal objects for longshot fade opportunities
        """
        if not self.kalshi_client:
            self.logger.warning("KalshiClient not available - returning empty signals")
            return []

        signals = []

        try:
            # Get all open markets
            markets = self.kalshi_client.get_markets(status='open', limit=200)

            if not markets:
                self.logger.info("No open markets returned from Kalshi")
                return []

            self.logger.info(f"Scanning {len(markets)} open Kalshi markets for longshots")

            # Filter for longshot opportunities
            for market in markets:
                signal = self._evaluate_market(market)
                if signal:
                    signals.append(signal)

            self.logger.info(f"Found {len(signals)} longshot fade opportunities")

            # Sort by confidence (highest first)
            signals.sort(key=lambda s: s.confidence, reverse=True)

            # Limit to top opportunities (don't overwhelm position limit)
            max_signals = min(10, self.config.max_open_positions - len(self.get_open_positions()))
            signals = signals[:max_signals]

        except Exception as e:
            self.logger.error(f"Error scanning Kalshi markets: {e}", exc_info=True)

        return signals

    def _evaluate_market(self, market: Dict) -> Optional[FleetSignal]:
        """
        Evaluate a single market for longshot fade opportunity.

        Args:
            market: Market data dict from KalshiClient

        Returns:
            FleetSignal if opportunity found, None otherwise
        """
        try:
            ticker = market.get('ticker', '')
            title = market.get('title', '')
            yes_ask = market.get('yes_ask')
            no_ask = market.get('no_ask')
            volume = market.get('volume', 0)
            close_time_str = market.get('close_time', '')

            # Validate required fields
            if not all([ticker, yes_ask is not None, no_ask is not None]):
                return None

            yes_ask = float(yes_ask)
            no_ask = float(no_ask)

            # Filter 1: YES price must be in longshot range
            if yes_ask < self.MIN_LONGSHOT_PRICE or yes_ask > self.MAX_LONGSHOT_PRICE:
                return None

            # Filter 2: Must have minimum volume
            if volume < self.MIN_VOLUME:
                return None

            # Filter 3: Check spread width (liquidity check)
            spread = abs((yes_ask + no_ask) - 1.0)
            if spread > self.MAX_SPREAD_WIDTH:
                return None

            # Filter 4: Time to close must be sufficient
            if close_time_str:
                close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                hours_to_close = (close_time - datetime.now(timezone.utc)).total_seconds() / 3600
                if hours_to_close < self.MIN_HOURS_TO_CLOSE:
                    return None
            else:
                hours_to_close = 168  # Default to 1 week if not provided

            # Filter 5: Avoid obvious parlay markets (multi-leg sports bets)
            title_lower = title.lower()
            if any(word in title_lower for word in ['parlay', 'multi-leg', 'accumulator']):
                return None

            # Calculate fair value estimate from research-based price bands
            fair_value = self._estimate_fair_value(yes_ask)

            # Edge calculation: Expected value of buying NO
            # We pay no_ask, we win if YES doesn't hit (prob = 1 - fair_value)
            # EV = (1 - no_ask) * (1 - fair_value) - no_ask * fair_value
            # Simplified: EV = (1 - fair_value) - no_ask
            edge = (1 - fair_value) - no_ask

            # Minimum edge threshold (account for fees ~2%)
            if edge < 0.10:  # Need at least 10% edge
                return None

            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(yes_ask, volume, hours_to_close, spread)

            # Position sizing based on confidence and edge
            kelly_fraction = (edge * confidence) / 0.5  # Simplified Kelly with 50% max payout
            kelly_fraction = max(0.02, min(0.10, kelly_fraction))  # Clamp to 2-10% of bankroll

            position_size = self.config.max_position_usd * (confidence + 0.5)  # Scale by confidence
            position_size = max(self.config.extra.get('min_position_size_usd', 5.0),
                              min(self.config.max_position_usd, position_size))

            # Calculate quantity (contracts to buy)
            quantity = int(position_size / no_ask)
            if quantity < 1:
                return None

            # Create signal
            signal = FleetSignal(
                bot_name=self.name,
                bot_type=self.bot_type.value,
                symbol=ticker,
                side='NO',  # Fade the longshot by buying NO
                entry_price=no_ask,
                target_price=no_ask * (1 - self.config.extra.get('take_profit_pct', 0.80)),
                stop_loss=no_ask * (1 + self.config.extra.get('stop_loss_pct', 0.50)),
                quantity=quantity,
                position_size_usd=position_size,
                confidence=confidence,
                edge=edge,
                reason=self._generate_reasoning(title, yes_ask, no_ask, edge, fair_value, volume),
                metadata={
                    'yes_ask': yes_ask,
                    'no_ask': no_ask,
                    'fair_value': fair_value,
                    'volume': volume,
                    'hours_to_close': hours_to_close,
                    'spread': spread,
                    'strategy': 'longshot_fade'
                }
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error evaluating market {market.get('ticker', 'UNKNOWN')}: {e}")
            return None

    def _estimate_fair_value(self, yes_price: float) -> float:
        """
        Estimate fair value based on research-backed price bands.

        Args:
            yes_price: Current YES ask price

        Returns:
            Estimated fair probability
        """
        # Find closest price band
        closest_band = min(self.FAIR_VALUE_ADJUSTMENTS.keys(),
                          key=lambda x: abs(x - yes_price))
        return self.FAIR_VALUE_ADJUSTMENTS[closest_band]

    def _calculate_confidence(self, yes_price: float, volume: int,
                             hours_to_close: float, spread: float) -> float:
        """
        Calculate confidence score based on market characteristics.

        Args:
            yes_price: YES ask price
            volume: Trading volume
            hours_to_close: Hours until market closes
            spread: Bid-ask spread width

        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence

        # Lower YES price = higher confidence (clearer longshot)
        if yes_price <= 0.04:
            confidence += 0.20
        elif yes_price <= 0.05:
            confidence += 0.10

        # Higher volume = higher confidence (more liquid)
        if volume >= 1000:
            confidence += 0.15
        elif volume >= 100:
            confidence += 0.10
        elif volume >= 10:
            confidence += 0.05

        # More time to close = higher confidence (less event risk)
        if hours_to_close >= 168:  # 1 week+
            confidence += 0.10
        elif hours_to_close >= 72:  # 3 days+
            confidence += 0.05

        # Tighter spread = higher confidence (better liquidity)
        if spread <= 0.02:
            confidence += 0.10
        elif spread <= 0.03:
            confidence += 0.05

        return min(1.0, confidence)

    def _generate_reasoning(self, title: str, yes_price: float, no_price: float,
                           edge: float, fair_value: float, volume: int) -> str:
        """
        Generate human-readable reasoning for the trade signal.

        Args:
            title: Market title
            yes_price: YES ask price
            no_price: NO ask price
            edge: Calculated edge
            fair_value: Estimated fair probability
            volume: Trading volume

        Returns:
            Reasoning string
        """
        return (f"Longshot fade on '{title}'. "
                f"YES trading at {yes_price:.1%} but fair value ~{fair_value:.1%}. "
                f"Buying NO at {no_price:.1%} with {edge:.1%} edge. "
                f"Volume: {volume} contracts. "
                f"Strategy: Exploit favorite-longshot bias (Becker 2025).")


if __name__ == "__main__":
    # Test/debug mode
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = LongshotFader()
    print(f"Bot: {bot.config.bot_name}")
    print(f"Type: {bot.config.bot_type}")
    print(f"Max positions: {bot.config.max_open_positions}")
    print(f"Position size: ${bot.config.position_size_usd}")

    print("\nScanning for longshot opportunities...")
    signals = bot.scan()

    print(f"\nFound {len(signals)} signals:")
    for i, signal in enumerate(signals, 1):
        print(f"\n{i}. {signal.ticker}")
        print(f"   Action: {signal.action} {signal.side}")
        print(f"   Quantity: {signal.quantity} @ ${signal.entry_price:.3f}")
        print(f"   Confidence: {signal.confidence:.1%}")
        print(f"   Edge: {signal.edge_pct:.1f}%")
        print(f"   Reasoning: {signal.reasoning}")
