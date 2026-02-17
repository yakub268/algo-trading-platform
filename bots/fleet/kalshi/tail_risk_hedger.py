"""
Tail Risk Hedger Bot - Portfolio insurance via cheap tail event contracts

Strategy:
- Buy cheap tail event contracts as portfolio hedges
- Target extreme events: government shutdown, debt default, market crash, geopolitical crisis
- Entry threshold: < 10 cents (cheap insurance)
- Very small position sizes ($2-5 per contract) - portfolio insurance, not speculation
- Max 5 concurrent tail positions
- Run daily, hold for tail protection + exit when doubled (asymmetric payoff)

Theory: Tail events are often underpriced due to recency bias and low base rates.
Small allocation provides asymmetric protection if extreme event occurs.
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Set

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType

try:
    from bots.kalshi_client import KalshiClient
    KALSHI_AVAILABLE = True
except ImportError:
    KALSHI_AVAILABLE = False
    logging.warning("KalshiClient not available - TailRiskHedger will run in dry-run mode")


class TailRiskHedger(FleetBot):
    """
    Tail risk hedger - buys cheap insurance against extreme events.

    Portfolio protection strategy, not alpha generation. Small positions
    on low-probability, high-impact events provide asymmetric payoff structure.
    """

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="Tail-Risk-Hedger",
                bot_type=BotType.KALSHI,
                schedule_seconds=86400,  # Daily (24 hours)
                max_position_usd=5.0,
                max_daily_trades=5,
                min_confidence=0.3,
                symbols=[],
                enabled=True,
                paper_mode=True,
                extra={
                    'min_position_size_usd': 2.0,
                    'stop_loss_pct': 0.70,
                    'take_profit_pct': 1.00,
                }
            )

        super().__init__(config)

        # Strategy parameters
        self.MAX_ENTRY_PRICE = 0.10  # 10 cents max - must be cheap insurance
        self.MIN_DAYS_TO_CLOSE = 7   # At least 1 week to allow event to materialize

        # Tail event keywords (expand as new event types emerge)
        self.TAIL_KEYWORDS = {
            # Government/political crises
            'shutdown', 'government shutdown', 'default', 'debt ceiling', 'debt default',
            'impeachment', 'constitutional crisis', 'martial law',

            # Market crashes
            'crash', 'market crash', 'recession', 'depression', 'financial crisis',
            'bank failure', 'systemic risk', 'liquidity crisis',

            # Geopolitical
            'war', 'nuclear', 'invasion', 'attack', 'terrorist', 'coup',
            'embargo', 'sanctions', 'conflict escalation',

            # Economic shocks
            'hyperinflation', 'deflation', 'sovereign default', 'currency crisis',
            'oil shock', 'supply shock',

            # Natural disasters (extreme)
            'catastrophic', 'mega-earthquake', 'tsunami', 'pandemic',

            # Technology/infrastructure
            'cyberattack', 'grid failure', 'internet outage', 'infrastructure collapse',

            # Extreme policy
            'emergency', 'state of emergency', 'suspension'
        }

        # Anti-keywords (exclude these - not true tail events)
        self.ANTI_KEYWORDS = {
            'win', 'lose', 'score', 'game', 'playoff', 'championship',  # Sports
            'album', 'movie', 'award', 'grammy', 'oscar',  # Entertainment
            'release', 'launch', 'announcement'  # Normal business events
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
        Scan Kalshi markets for cheap tail event insurance.

        Returns:
            List of FleetSignal objects for tail hedge positions
        """
        if not self.kalshi_client:
            self.logger.warning("KalshiClient not available - returning empty signals")
            return []

        signals = []

        try:
            # Get all open markets (we need to scan broadly for tail events)
            markets = self.kalshi_client.get_markets(status='open', limit=200)

            if not markets:
                self.logger.info("No open markets returned from Kalshi")
                return []

            self.logger.info(f"Scanning {len(markets)} markets for tail event hedges")

            # Filter for tail event opportunities
            for market in markets:
                signal = self._evaluate_market(market)
                if signal:
                    signals.append(signal)

            self.logger.info(f"Found {len(signals)} tail hedge opportunities")

            # Sort by asymmetry score (price * potential_upside)
            signals.sort(key=lambda s: s.metadata.get('asymmetry_score', 0), reverse=True)

            # Check existing positions for exit opportunities
            self._check_exit_opportunities()

            # Limit new positions
            open_positions = len(self.get_open_positions())
            max_new_signals = self.config.max_open_positions - open_positions
            signals = signals[:max_new_signals]

        except Exception as e:
            self.logger.error(f"Error scanning tail events: {e}", exc_info=True)

        return signals

    def _evaluate_market(self, market: Dict) -> Optional[FleetSignal]:
        """
        Evaluate market for tail event hedge opportunity.

        Args:
            market: Market data dict from KalshiClient

        Returns:
            FleetSignal if tail hedge opportunity, None otherwise
        """
        try:
            ticker = market.get('ticker', '')
            title = market.get('title', '')
            yes_ask = market.get('yes_ask')
            volume = market.get('volume', 0)
            close_time_str = market.get('close_time', '')

            if not all([ticker, title, yes_ask is not None]):
                return None

            yes_ask = float(yes_ask)
            if yes_ask <= 0:
                return None
            title_lower = title.lower()

            # Filter 1: Must match tail event keywords
            has_tail_keyword = any(keyword in title_lower for keyword in self.TAIL_KEYWORDS)
            if not has_tail_keyword:
                return None

            # Filter 2: Must not match anti-keywords (exclude sports/entertainment)
            has_anti_keyword = any(keyword in title_lower for keyword in self.ANTI_KEYWORDS)
            if has_anti_keyword:
                return None

            # Filter 3: Price must be cheap (insurance threshold)
            if yes_ask > self.MAX_ENTRY_PRICE:
                return None

            # Filter 4: Time to close
            if close_time_str:
                close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
                days_to_close = (close_time - datetime.now(timezone.utc)).days
                if days_to_close < self.MIN_DAYS_TO_CLOSE:
                    return None
            else:
                days_to_close = 30  # Default assumption

            # Calculate asymmetry score (lower price = higher asymmetry)
            # Max upside: (1.00 - yes_ask) / yes_ask
            max_upside_ratio = (1.00 - yes_ask) / yes_ask if yes_ask > 0 else 0
            asymmetry_score = max_upside_ratio * (1 - yes_ask)  # Weight by cheapness

            # Confidence based on price (lower = potentially better value for tail)
            # But also consider that extremely cheap might be rationally priced
            if yes_ask <= 0.03:
                confidence = 0.30  # Very cheap but maybe for good reason
            elif yes_ask <= 0.05:
                confidence = 0.40
            elif yes_ask <= 0.07:
                confidence = 0.45
            else:
                confidence = 0.50

            # Adjust confidence based on volume (some liquidity = more confidence)
            if volume >= 100:
                confidence += 0.10
            elif volume >= 10:
                confidence += 0.05

            # Position sizing - very small (insurance allocation)
            position_size = self.config.max_position_usd

            # Calculate quantity
            quantity = int(position_size / yes_ask)
            if quantity < 1:
                return None

            # Set exits
            # Stop loss: if price decays significantly (tail event becomes less likely)
            stop_loss = yes_ask * (1 + self.config.extra.get('stop_loss_pct', 0.50))

            # Take profit: if price doubles (tail event probability increasing)
            take_profit = yes_ask * (1 - self.config.extra.get('take_profit_pct', 1.00))  # Sells at 2x

            # Edge calculation (speculative for tail events)
            # Assume market underprices by 50% due to recency bias
            estimated_fair_value = yes_ask * 1.5
            edge = (estimated_fair_value - yes_ask) / yes_ask

            # Generate reasoning
            reasoning = self._generate_reasoning(title, yes_ask, asymmetry_score, days_to_close)

            # Create signal
            signal = FleetSignal(
                bot_name=self.name,
                bot_type=self.bot_type.value,
                symbol=ticker,
                side='YES',  # Buy YES on tail events
                entry_price=yes_ask,
                target_price=take_profit,
                stop_loss=stop_loss,
                quantity=quantity,
                position_size_usd=position_size,
                confidence=confidence,
                edge=edge,
                reason=reasoning,
                metadata={
                    'yes_ask': yes_ask,
                    'volume': volume,
                    'days_to_close': days_to_close,
                    'asymmetry_score': asymmetry_score,
                    'max_upside_ratio': max_upside_ratio,
                    'strategy': 'tail_risk_hedge'
                }
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error evaluating market {market.get('ticker', 'UNKNOWN')}: {e}")
            return None

    def _generate_reasoning(self, title: str, price: float,
                           asymmetry_score: float, days_to_close: int) -> str:
        """
        Generate human-readable reasoning for tail hedge.

        Args:
            title: Market title
            price: Entry price
            asymmetry_score: Calculated asymmetry
            days_to_close: Days until market closes

        Returns:
            Reasoning string
        """
        max_upside = ((1.00 - price) / price) if price > 0 else 0

        reasoning = (f"Tail hedge: '{title}'. "
                    f"Insurance premium: {price:.1%} for {max_upside:.1f}x upside. "
                    f"Asymmetry score: {asymmetry_score:.2f}. "
                    f"Duration: {days_to_close} days. "
                    f"Portfolio protection, not speculation.")

        return reasoning

    def _check_exit_opportunities(self):
        """
        Check existing tail positions for exit opportunities.

        Tail events can appreciate quickly when probability increases.
        Exit when position has doubled (take_profit) or sell if appropriate.
        """
        try:
            open_positions = self.get_open_positions()

            for position in open_positions:
                ticker = position.get('ticker', '')
                entry_price = position.get('entry_price', 0)

                if not ticker or not entry_price:
                    continue

                # Get current market data
                try:
                    market = self.kalshi_client.get_market(ticker)
                    if not market:
                        continue

                    current_price = market.get('yes_bid', 0)  # Use bid for selling

                    # Check if we should exit
                    if current_price >= entry_price * 2:
                        self.logger.info(f"Exit opportunity: {ticker} doubled from "
                                       f"{entry_price:.3f} to {current_price:.3f}")
                        # Position manager will handle actual exit

                except Exception as e:
                    self.logger.error(f"Error checking exit for {ticker}: {e}")

        except Exception as e:
            self.logger.error(f"Error checking exit opportunities: {e}")

    def get_open_positions(self) -> List[Dict]:
        """
        Get open tail hedge positions.

        Returns:
            List of open tail hedge positions
        """
        all_positions = super().get_open_positions()
        # Filter for tail hedge strategy
        tail_positions = [p for p in all_positions
                         if p.get('metadata', {}).get('strategy') == 'tail_risk_hedge']
        return tail_positions


if __name__ == "__main__":
    # Test/debug mode
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = TailRiskHedger()
    print(f"Bot: {bot.config.bot_name}")
    print(f"Type: {bot.config.bot_type}")
    print(f"Max positions: {bot.config.max_open_positions}")
    print(f"Position size: ${bot.config.position_size_usd}")
    print(f"Max entry price: {bot.MAX_ENTRY_PRICE:.1%}")
    print(f"Tail keywords: {len(bot.TAIL_KEYWORDS)} categories")

    print("\nScanning for tail hedge opportunities...")
    signals = bot.scan()

    print(f"\nFound {len(signals)} signals:")
    for i, signal in enumerate(signals, 1):
        print(f"\n{i}. {signal.ticker}")
        print(f"   Action: {signal.action} {signal.side}")
        print(f"   Quantity: {signal.quantity} @ ${signal.entry_price:.3f}")
        print(f"   Max upside: {signal.metadata.get('max_upside_ratio', 0):.1f}x")
        print(f"   Asymmetry score: {signal.metadata.get('asymmetry_score', 0):.2f}")
        print(f"   Confidence: {signal.confidence:.1%}")
        print(f"   Reasoning: {signal.reasoning}")
