"""
Weather Prophet Bot - Temperature forecasting edge on Kalshi

Wraps existing WeatherEdgeFinder infrastructure with fleet bot interface.
Adds maker-order preference to reduce fees (Kalshi maker fees 4x cheaper than taker).

Strategy:
- Use WeatherEdgeFinder to identify weather markets with NWS/GFS ensemble edge
- Convert opportunities to FleetSignals with maker order preference
- Set limit orders 1 cent better than market to capture maker rebates
- Max 3 concurrent weather positions (concentration risk management)
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType

try:
    from bots.weather_edge_finder import WeatherEdgeFinder, WeatherOpportunity
    WEATHER_FINDER_AVAILABLE = True
except ImportError:
    WEATHER_FINDER_AVAILABLE = False
    logging.warning("WeatherEdgeFinder not available - WeatherProphet will run in dry-run mode")

try:
    from bots.kalshi_client import KalshiClient
    KALSHI_AVAILABLE = True
except ImportError:
    KALSHI_AVAILABLE = False
    logging.warning("KalshiClient not available - WeatherProphet will run in dry-run mode")


class WeatherProphet(FleetBot):
    """
    Temperature forecasting bot using NWS/GFS ensemble data.

    Leverages existing WeatherEdgeFinder infrastructure with fleet bot interface.
    Optimizes for maker orders to reduce fees (4x cheaper on Kalshi).
    """

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="Weather-Prophet",
                bot_type=BotType.KALSHI,
                schedule_seconds=600,  # 10 minutes
                max_position_usd=50.0,
                max_daily_trades=10,
                min_confidence=0.6,
                symbols=[],
                enabled=True,
                paper_mode=True,
                extra={
                    'min_position_size_usd': 15.0,
                    'stop_loss_pct': 0.40,
                    'take_profit_pct': 0.50,
                }
            )

        super().__init__(config)

        # Strategy parameters
        self.MIN_EDGE_THRESHOLD = 0.10  # 10% minimum edge
        self.MIN_CONFIDENCE = 0.60      # 60% minimum confidence
        self.MAKER_PRICE_IMPROVEMENT = 0.01  # 1 cent better for maker orders
        self.MAX_ENTRY_PRICE = 0.70     # Don't buy above 70 cents

        # Initialize WeatherEdgeFinder
        self.weather_finder = None
        if WEATHER_FINDER_AVAILABLE:
            try:
                self.weather_finder = WeatherEdgeFinder()
                self.logger.info("WeatherEdgeFinder initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize WeatherEdgeFinder: {e}")

        # Initialize KalshiClient for market data
        self.kalshi_client = None
        if KALSHI_AVAILABLE:
            try:
                self.kalshi_client = KalshiClient()
                self.logger.info("KalshiClient initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize KalshiClient: {e}")

    def scan(self) -> List[FleetSignal]:
        """
        Scan for weather trading opportunities using WeatherEdgeFinder.

        Returns:
            List of FleetSignal objects for weather trades
        """
        if not self.weather_finder:
            self.logger.warning("WeatherEdgeFinder not available - returning empty signals")
            return []

        signals = []

        try:
            # Get weather opportunities from existing infrastructure
            opportunities = self.weather_finder.find_opportunities()

            if not opportunities:
                self.logger.info("No weather opportunities found")
                return []

            self.logger.info(f"Found {len(opportunities)} weather opportunities from WeatherEdgeFinder")

            # Convert each opportunity to FleetSignal
            for opp in opportunities:
                signal = self._convert_opportunity(opp)
                if signal:
                    signals.append(signal)

            self.logger.info(f"Converted {len(signals)} opportunities to fleet signals")

            # Sort by edge (highest first)
            signals.sort(key=lambda s: s.edge, reverse=True)

            # Limit to avoid exceeding max positions
            open_positions = len(self.get_open_positions())
            max_new_signals = self.config.max_open_positions - open_positions
            signals = signals[:max_new_signals]

        except Exception as e:
            self.logger.error(f"Error scanning weather opportunities: {e}", exc_info=True)

        return signals

    def _convert_opportunity(self, opp) -> Optional[FleetSignal]:
        """
        Convert WeatherOpportunity to FleetSignal with maker order optimization.

        Args:
            opp: WeatherOpportunity object from WeatherEdgeFinder

        Returns:
            FleetSignal if valid, None otherwise
        """
        try:
            # Extract opportunity data
            ticker = opp.ticker
            edge = opp.edge
            confidence = opp.confidence
            side = opp.side.lower()  # 'yes' or 'no'
            market_price = opp.market_price

            # Filter 1: Edge threshold
            if edge < self.MIN_EDGE_THRESHOLD:
                self.logger.debug(f"{ticker}: Edge {edge:.1%} below threshold {self.MIN_EDGE_THRESHOLD:.1%}")
                return None

            # Filter 2: Confidence threshold
            if confidence < self.MIN_CONFIDENCE:
                self.logger.debug(f"{ticker}: Confidence {confidence:.1%} below threshold {self.MIN_CONFIDENCE:.1%}")
                return None

            # Filter 3: Entry price cap (avoid buying expensive contracts)
            if market_price > self.MAX_ENTRY_PRICE:
                self.logger.debug(f"{ticker}: Price {market_price:.1%} above max {self.MAX_ENTRY_PRICE:.1%}")
                return None

            # Optimize for maker order: set limit price 1 cent better than market
            # This gets us in the queue as maker (4x cheaper fees on Kalshi)
            entry_price = max(0.01, market_price - self.MAKER_PRICE_IMPROVEMENT)

            # Get current market data for stop/take profit calculation
            current_ask = market_price

            # Position sizing based on confidence and edge
            kelly_fraction = (edge * confidence) / 1.0  # Kelly with 100% payout
            kelly_fraction = max(0.03, min(0.15, kelly_fraction))  # Clamp to 3-15%

            position_size = self.config.max_position_usd * (1 + confidence * 0.5)  # Scale by confidence
            position_size = max(self.config.extra.get('min_position_size_usd', 5.0),
                              min(self.config.max_position_usd, position_size))

            # Calculate quantity
            quantity = int(position_size / entry_price)
            if quantity < 1:
                return None

            # Set stop loss and take profit
            stop_loss = entry_price * (1 + self.config.extra.get('stop_loss_pct', 0.50))
            take_profit = entry_price * (1 - self.config.extra.get('take_profit_pct', 0.80))

            # Generate reasoning with weather context
            reasoning = self._generate_reasoning(opp, entry_price, market_price)

            # Create signal
            signal = FleetSignal(
                bot_name=self.name,
                bot_type=self.bot_type.value,
                symbol=ticker,
                side=side.upper(),
                entry_price=entry_price,
                target_price=take_profit,
                stop_loss=stop_loss,
                quantity=quantity,
                position_size_usd=position_size,
                confidence=confidence,
                edge=edge,
                reason=reasoning,
                metadata={
                    'market_price': market_price,
                    'maker_improvement': self.MAKER_PRICE_IMPROVEMENT,
                    'order_type': 'limit',  # Maker order
                    'forecast_source': getattr(opp, 'source', 'NWS/GFS'),
                    'city': getattr(opp, 'city', 'unknown'),
                    'threshold': getattr(opp, 'threshold', 'unknown'),
                    'strategy': 'weather_forecast'
                }
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error converting opportunity: {e}", exc_info=True)
            return None

    def _generate_reasoning(self, opp, entry_price: float, market_price: float) -> str:
        """
        Generate human-readable reasoning for weather trade.

        Args:
            opp: WeatherOpportunity object
            entry_price: Calculated entry price
            market_price: Current market price

        Returns:
            Reasoning string
        """
        city = getattr(opp, 'city', 'unknown')
        threshold = getattr(opp, 'threshold', 'unknown')
        forecast_value = getattr(opp, 'forecast_value', 'unknown')
        source = getattr(opp, 'source', 'NWS/GFS')

        reasoning = (f"Weather edge: {city} temp {threshold}°F. "
                    f"{source} forecast: {forecast_value}. "
                    f"Buying {opp.side.upper()} at {entry_price:.1%} "
                    f"(market {market_price:.1%}, maker order -1¢). "
                    f"Edge: {opp.edge:.1%}, Confidence: {opp.confidence:.1%}.")

        return reasoning

    def get_open_positions(self) -> List[Dict]:
        """
        Override to filter for weather positions only.

        Returns:
            List of open weather positions
        """
        all_positions = super().get_open_positions()
        # Filter for weather tickers (KXHIGH prefix)
        weather_positions = [p for p in all_positions
                           if p.get('symbol', '').startswith('KXHIGH')]
        return weather_positions


if __name__ == "__main__":
    # Test/debug mode
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = WeatherProphet()
    print(f"Bot: {bot.config.bot_name}")
    print(f"Type: {bot.config.bot_type}")
    print(f"Max positions: {bot.config.max_open_positions}")
    print(f"Position size: ${bot.config.position_size_usd}")
    print(f"Min edge: {bot.MIN_EDGE_THRESHOLD:.1%}")
    print(f"Min confidence: {bot.MIN_CONFIDENCE:.1%}")

    print("\nScanning for weather opportunities...")
    signals = bot.scan()

    print(f"\nFound {len(signals)} signals:")
    for i, signal in enumerate(signals, 1):
        print(f"\n{i}. {signal.ticker}")
        print(f"   Action: {signal.action} {signal.side}")
        print(f"   Quantity: {signal.quantity} @ ${signal.entry_price:.3f}")
        print(f"   Market price: ${signal.metadata.get('market_price', 0):.3f}")
        print(f"   Confidence: {signal.confidence:.1%}")
        print(f"   Edge: {signal.edge_pct:.1f}%")
        print(f"   Reasoning: {signal.reasoning}")
