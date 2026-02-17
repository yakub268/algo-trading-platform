"""
Economic Event Trader Bot - FRED/CPI/GDP edge exploitation on Kalshi

Wraps existing EdgeDetector infrastructure for economic prediction markets.
Targets FRED, CPI, GDP, Jobs, and Inflation markets with nowcast/forecast edges.

Strategy:
- Scan Kalshi economic tickers (KXFED, KXCPI, KXGDP, KXJOB, KXINF)
- Use EdgeDetector with specialized economic sources (Cleveland Fed, GDPNow, etc.)
- Filter for high-confidence, high-edge opportunities
- Position size based on Kelly criterion
- Max 5 concurrent economic positions
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
    from bots.event_trading.edge_detector import EdgeDetector
    EDGE_DETECTOR_AVAILABLE = True
except ImportError:
    EDGE_DETECTOR_AVAILABLE = False
    logging.warning("EdgeDetector not available - EconEventTrader will run in dry-run mode")

try:
    from bots.kalshi_client import KalshiClient
    KALSHI_AVAILABLE = True
except ImportError:
    KALSHI_AVAILABLE = False
    logging.warning("KalshiClient not available - EconEventTrader will run in dry-run mode")


class EconEventTrader(FleetBot):
    """
    Economic event prediction bot using EdgeDetector infrastructure.

    Focuses on economic markets where nowcasts/forecasts provide edge:
    - FRED rate decisions (Cleveland Fed nowcasts)
    - CPI inflation (Cleveland Fed CPI nowcast)
    - GDP growth (Atlanta Fed GDPNow)
    - Jobs reports
    - Inflation expectations
    """

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="Econ-Event-Trader",
                bot_type=BotType.KALSHI,
                schedule_seconds=900,  # 15 minutes
                max_position_usd=80.0,
                max_daily_trades=10,
                min_confidence=0.6,
                symbols=[],
                enabled=True,
                paper_mode=True,
                extra={
                    'min_position_size_usd': 20.0,
                    'stop_loss_pct': 0.50,
                    'take_profit_pct': 0.60,
                }
            )

        super().__init__(config)

        # Strategy parameters
        self.MIN_EDGE_THRESHOLD = 0.10  # 10% minimum edge
        self.MIN_CONFIDENCE = 0.60      # 60% minimum confidence
        self.MAX_ENTRY_PRICE = 0.75     # Don't buy above 75 cents

        # Economic ticker prefixes to scan
        self.ECON_PREFIXES = ['KXFED', 'KXCPI', 'KXGDP', 'KXJOB', 'KXINF', 'KXUNR', 'KXPCE']

        # Initialize EdgeDetector
        self.edge_detector = None
        if EDGE_DETECTOR_AVAILABLE:
            try:
                self.edge_detector = EdgeDetector()
                self.logger.info("EdgeDetector initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize EdgeDetector: {e}")

        # Initialize KalshiClient
        self.kalshi_client = None
        if KALSHI_AVAILABLE:
            try:
                self.kalshi_client = KalshiClient()
                self.logger.info("KalshiClient initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize KalshiClient: {e}")

    def scan(self) -> List[FleetSignal]:
        """
        Scan economic markets for edge opportunities.

        Returns:
            List of FleetSignal objects for economic trades
        """
        if not self.edge_detector or not self.kalshi_client:
            self.logger.warning("EdgeDetector or KalshiClient not available - returning empty signals")
            return []

        signals = []

        try:
            # Get all open economic markets
            all_markets = self.kalshi_client.get_markets(status='open', limit=200)
            markets = [m for m in all_markets
                       if any(m.get('ticker', '').startswith(p) for p in self.ECON_PREFIXES)]

            if not markets:
                self.logger.info("No open economic markets found")
                return []

            self.logger.info(f"Scanning {len(markets)} economic markets for edge")

            # Evaluate each market with EdgeDetector
            for market in markets:
                signal = self._evaluate_market(market)
                if signal:
                    signals.append(signal)

            self.logger.info(f"Found {len(signals)} economic trading signals")

            # Sort by edge * confidence (expected value proxy)
            signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)

            # Limit to available position slots
            open_positions = len(self.get_open_positions())
            max_new_signals = self.config.max_open_positions - open_positions
            signals = signals[:max_new_signals]

        except Exception as e:
            self.logger.error(f"Error scanning economic markets: {e}", exc_info=True)

        return signals

    def _evaluate_market(self, market: Dict) -> Optional[FleetSignal]:
        """
        Evaluate a single economic market using EdgeDetector.

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

            if not ticker or yes_ask is None or no_ask is None:
                return None

            yes_ask = float(yes_ask)
            no_ask = float(no_ask)

            # Get edge signal from EdgeDetector
            edge_signal = self.edge_detector.analyze(market)

            if not edge_signal:
                return None

            edge = edge_signal.edge
            confidence = edge_signal.confidence
            suggested_side = edge_signal.direction.lower()  # 'yes' or 'no'

            # Filter 1: Edge threshold
            if edge < self.MIN_EDGE_THRESHOLD:
                return None

            # Filter 2: Confidence threshold
            if confidence < self.MIN_CONFIDENCE:
                return None

            # Determine entry price based on suggested side
            if suggested_side == 'yes':
                entry_price = yes_ask
            else:
                entry_price = no_ask

            # Filter 3: Entry price cap
            if entry_price > self.MAX_ENTRY_PRICE:
                self.logger.debug(f"{ticker}: Entry price {entry_price:.1%} above max {self.MAX_ENTRY_PRICE:.1%}")
                return None

            # Filter 4: Minimum liquidity (volume check)
            if volume < 5:
                self.logger.debug(f"{ticker}: Volume {volume} below minimum 5")
                return None

            # Position sizing using Kelly criterion
            # Kelly fraction = (edge * confidence) / odds
            # For binary markets: odds ≈ 1 / entry_price - 1
            odds = (1.0 / entry_price) - 1.0 if entry_price > 0 else 1.0
            kelly_fraction = (edge * confidence) / max(odds, 0.1)
            kelly_fraction = max(0.05, min(0.20, kelly_fraction))  # Clamp to 5-20%

            position_size = self.config.max_position_usd * (1 + kelly_fraction)
            position_size = max(self.config.extra.get('min_position_size_usd', 5.0),
                              min(self.config.max_position_usd, position_size))

            # Calculate quantity
            quantity = int(position_size / entry_price)
            if quantity < 1:
                return None

            # Set stop loss and take profit
            stop_loss = entry_price * (1 + self.config.extra.get('stop_loss_pct', 0.50))
            take_profit = entry_price * (1 - self.config.extra.get('take_profit_pct', 0.80))

            # Generate reasoning
            reasoning = self._generate_reasoning(edge_signal, ticker, title, entry_price)

            # Create signal
            signal = FleetSignal(
                bot_name=self.name,
                bot_type=self.bot_type.value,
                symbol=ticker,
                side=suggested_side.upper(),
                entry_price=entry_price,
                target_price=take_profit,
                stop_loss=stop_loss,
                quantity=quantity,
                position_size_usd=position_size,
                confidence=confidence,
                edge=edge,
                reason=reasoning,
                metadata={
                    'yes_ask': yes_ask,
                    'no_ask': no_ask,
                    'volume': volume,
                    'sources': getattr(edge_signal, 'sources', []),
                    'kelly_fraction': kelly_fraction,
                    'strategy': 'economic_event'
                }
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error evaluating market {market.get('ticker', 'UNKNOWN')}: {e}")
            return None

    def _generate_reasoning(self, edge_signal, ticker: str, title: str, entry_price: float) -> str:
        """
        Generate human-readable reasoning for economic trade.

        Args:
            edge_signal: EdgeSignal object from EdgeDetector
            ticker: Market ticker
            title: Market title
            entry_price: Calculated entry price

        Returns:
            Reasoning string
        """
        sources = getattr(edge_signal, 'sources', [])
        source_str = ', '.join(sources[:2]) if sources else 'EdgeDetector'

        reasoning = (f"Economic event edge on {title}. "
                    f"Sources: {source_str}. "
                    f"Buying {edge_signal.side.upper()} at {entry_price:.1%}. "
                    f"Edge: {edge_signal.edge:.1%}, Confidence: {edge_signal.confidence:.1%}.")

        return reasoning

    def get_open_positions(self) -> List[Dict]:
        """
        Override to filter for economic positions only.

        Returns:
            List of open economic positions
        """
        all_positions = super().get_open_positions()
        # Filter for economic tickers
        econ_positions = [p for p in all_positions
                         if any(p.get('ticker', '').startswith(prefix)
                               for prefix in self.ECON_PREFIXES)]
        return econ_positions


if __name__ == "__main__":
    # Test/debug mode
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = EconEventTrader()
    print(f"Bot: {bot.config.bot_name}")
    print(f"Type: {bot.config.bot_type}")
    print(f"Max positions: {bot.config.max_open_positions}")
    print(f"Position size: ${bot.config.position_size_usd}")
    print(f"Min edge: {bot.MIN_EDGE_THRESHOLD:.1%}")
    print(f"Min confidence: {bot.MIN_CONFIDENCE:.1%}")
    print(f"Scanning prefixes: {', '.join(bot.ECON_PREFIXES)}")

    print("\nScanning for economic opportunities...")
    signals = bot.scan()

    print(f"\nFound {len(signals)} signals:")
    for i, signal in enumerate(signals, 1):
        print(f"\n{i}. {signal.ticker}")
        print(f"   Action: {signal.action} {signal.side}")
        print(f"   Quantity: {signal.quantity} @ ${signal.entry_price:.3f}")
        print(f"   Confidence: {signal.confidence:.1%}")
        print(f"   Edge: {signal.edge_pct:.1f}%")
        print(f"   Reasoning: {signal.reasoning}")
