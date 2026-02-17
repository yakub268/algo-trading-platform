"""
Sports Edge Enhanced Bot - Fleet Trading System

Wraps SportsEdgeFinder and adds ELO-Odds hybrid weighting.
Combines fundamental Elo ratings with market-implied probabilities
for sharper edge detection on NBA/NFL games.

Author: Fleet Trading Bot System
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

# Project root path resolution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

try:
    from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType
except ImportError as e:
    logging.error(f"Failed to import FleetBot base: {e}")
    raise

try:
    from bots.sports_edge_finder import SportsEdgeFinder, SportsOpportunity
except ImportError as e:
    logging.error(f"Failed to import SportsEdgeFinder: {e}")
    raise

try:
    from bots.kalshi_client import KalshiClient
except ImportError as e:
    logging.error(f"Failed to import KalshiClient: {e}")
    raise


class SportsEdgeEnhancedBot(FleetBot):
    """
    Enhanced sports prediction bot using hybrid Elo-Odds methodology.

    Strategy:
    - Leverage SportsEdgeFinder's Elo ratings (60% weight)
    - Combine with market-implied odds (40% weight)
    - Focus on NBA/NFL where Elo models are most reliable
    - Edge threshold: 8%+
    - Confidence threshold: 55%+
    - Max 5 concurrent sports positions
    """

    # League preferences (more data = better Elo models)
    PREFERRED_LEAGUES = {'NBA', 'NFL', 'NCAAF', 'NCAAB'}

    # Weighting for hybrid probability calculation
    ELO_WEIGHT = 0.60
    ODDS_WEIGHT = 0.40

    # Thresholds
    MIN_EDGE = 0.08  # 8% minimum edge after hybrid calculation
    MIN_CONFIDENCE = 0.55  # 55% minimum confidence
    MAX_POSITIONS = 5  # Maximum concurrent sports positions

    # Position sizing
    MIN_POSITION_SIZE = 5  # $5 minimum
    MAX_POSITION_SIZE = 15  # $15 maximum

    def __init__(self, config: Optional[FleetBotConfig] = None):
        """
        Initialize Sports Edge Enhanced bot.

        Args:
            config: Fleet bot configuration. Uses defaults if None.
        """
        if config is None:
            config = FleetBotConfig(
                name="Sports-Edge-Enhanced",
                bot_type=BotType.PREDICTION,
                schedule_seconds=900,  # 15 minutes
                max_position_usd=15.0,
                max_daily_trades=5,
                min_confidence=self.MIN_CONFIDENCE,
                symbols=[],
                enabled=True,
                paper_mode=True,
                extra={'min_edge': self.MIN_EDGE}
            )

        super().__init__(config)

        # Initialize SportsEdgeFinder
        try:
            self.sports_finder = SportsEdgeFinder()
            self.logger.info("SportsEdgeFinder initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize SportsEdgeFinder: {e}")
            self.sports_finder = None

        # Initialize Kalshi client for market data
        try:
            self.kalshi = KalshiClient()
            self.logger.info("KalshiClient initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize KalshiClient: {e}")
            self.kalshi = None

        self.logger.info(f"SportsEdgeEnhancedBot initialized: {self.MAX_POSITIONS} max positions, {self.MIN_EDGE*100:.1f}% min edge")

    def scan(self) -> List[FleetSignal]:
        """
        Scan for sports betting opportunities using hybrid Elo-Odds methodology.

        Returns:
            List of FleetSignal objects representing trading opportunities
        """
        signals = []

        try:
            # Check if we have all required components
            if not self.sports_finder:
                self.logger.warning("SportsEdgeFinder not available, skipping scan")
                return signals

            # Get opportunities from SportsEdgeFinder
            self.logger.info("Scanning for sports opportunities via SportsEdgeFinder...")
            opportunities = self.sports_finder.find_opportunities()

            if not opportunities:
                self.logger.info("No opportunities found by SportsEdgeFinder")
                return signals

            self.logger.info(f"Found {len(opportunities)} raw opportunities from SportsEdgeFinder")

            # Process each opportunity with hybrid methodology
            for opp in opportunities:
                try:
                    signal = self._process_opportunity(opp)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    self.logger.error(f"Error processing opportunity {getattr(opp, 'ticker', 'UNKNOWN')}: {e}")
                    continue

            # Sort by edge (highest first)
            signals.sort(key=lambda s: s.edge, reverse=True)

            # Limit to top opportunities
            if len(signals) > self.MAX_POSITIONS:
                self.logger.info(f"Limiting {len(signals)} signals to top {self.MAX_POSITIONS}")
                signals = signals[:self.MAX_POSITIONS]

            self.logger.info(f"Generated {len(signals)} final trading signals")

        except Exception as e:
            self.logger.error(f"Error in sports scan: {e}", exc_info=True)

        return signals

    def _process_opportunity(self, opp: 'SportsOpportunity') -> Optional[FleetSignal]:
        """
        Process a single sports opportunity with hybrid Elo-Odds methodology.

        Args:
            opp: SportsOpportunity from SportsEdgeFinder

        Returns:
            FleetSignal if opportunity meets criteria, None otherwise
        """
        # Filter by preferred leagues
        league = getattr(opp, 'league', 'UNKNOWN')
        if league not in self.PREFERRED_LEAGUES:
            self.logger.debug(f"Skipping {opp.ticker}: league {league} not in preferred list")
            return None

        # Get Elo-based probability
        elo_probability = getattr(opp, 'our_probability', None)
        if elo_probability is None:
            self.logger.warning(f"No Elo probability for {opp.ticker}, skipping")
            return None

        # Get market price
        market_price = getattr(opp, 'market_price', None)
        if market_price is None or market_price <= 0:
            self.logger.warning(f"Invalid market price for {opp.ticker}, skipping")
            return None

        # Calculate hybrid probability
        # If we have actual odds data, use hybrid weighting
        # Otherwise fall back to pure Elo
        hybrid_probability = self._calculate_hybrid_probability(opp, elo_probability, market_price)

        # Calculate edge with hybrid probability
        edge = hybrid_probability - market_price
        edge_pct = edge * 100

        # Check edge threshold
        if edge < self.MIN_EDGE:
            self.logger.debug(f"Skipping {opp.ticker}: edge {edge_pct:.2f}% below threshold {self.MIN_EDGE*100:.1f}%")
            return None

        # Calculate confidence score
        # Higher Elo differential + higher edge = higher confidence
        confidence = self._calculate_confidence(opp, edge, elo_probability)

        # Check confidence threshold
        if confidence < self.MIN_CONFIDENCE:
            self.logger.debug(f"Skipping {opp.ticker}: confidence {confidence:.2f} below threshold {self.MIN_CONFIDENCE}")
            return None

        # Calculate position size based on edge and confidence
        position_size = self._calculate_position_size(edge, confidence)

        # Build reasoning
        reasoning = self._build_reasoning(opp, elo_probability, hybrid_probability, edge, confidence)

        # Extract metadata
        metadata = {
            'league': league,
            'home_team': getattr(opp, 'home_team', 'UNKNOWN'),
            'away_team': getattr(opp, 'away_team', 'UNKNOWN'),
            'elo_home': getattr(opp, 'elo_home', None),
            'elo_away': getattr(opp, 'elo_away', None),
            'elo_probability': elo_probability,
            'hybrid_probability': hybrid_probability,
            'market_price': market_price,
            'original_reasoning': getattr(opp, 'reasoning', '')
        }

        # Create FleetSignal
        signal = FleetSignal(
            bot_name=self.name,
            bot_type=self.bot_type.value,
            symbol=opp.ticker,
            side=getattr(opp, 'side', 'yes').upper(),  # Default to 'yes' if not specified
            entry_price=market_price,
            target_price=0.0,  # Kalshi doesn't use target
            stop_loss=0.0,  # Kalshi doesn't use SL
            quantity=position_size / market_price if market_price > 0 else 0,
            position_size_usd=position_size,
            confidence=confidence,
            edge=edge,
            reason=reasoning,
            metadata=metadata
        )

        self.logger.info(f"Generated signal: {opp.ticker} {signal.side.upper()} | Edge: {edge_pct:.2f}% | Conf: {confidence:.2f} | Size: ${position_size}")

        return signal

    def _calculate_hybrid_probability(
        self,
        opp: 'SportsOpportunity',
        elo_probability: float,
        market_price: float
    ) -> float:
        """
        Calculate hybrid probability using Elo + market-implied odds.

        Args:
            opp: SportsOpportunity object
            elo_probability: Probability from Elo rating
            market_price: Current market price (0-1)

        Returns:
            Hybrid probability weighted 60% Elo, 40% market consensus
        """
        # Check if we have both Elo ratings for a more robust calculation
        elo_home = getattr(opp, 'elo_home', None)
        elo_away = getattr(opp, 'elo_away', None)

        # If we have strong Elo data (both ratings available)
        # Use market price as reality check, not equal partner
        if elo_home and elo_away:
            # Strong Elo data: 70% Elo, 30% market
            return (0.70 * elo_probability) + (0.30 * market_price)
        else:
            # Weaker Elo data: 60% Elo, 40% market (default)
            return (self.ELO_WEIGHT * elo_probability) + (self.ODDS_WEIGHT * market_price)

    def _calculate_confidence(
        self,
        opp: 'SportsOpportunity',
        edge: float,
        elo_probability: float
    ) -> float:
        """
        Calculate confidence score for the opportunity.

        Factors:
        - Edge magnitude (bigger edge = higher confidence)
        - Elo differential (bigger gap = higher confidence)
        - Probability extremes (very high/low = higher confidence)

        Args:
            opp: SportsOpportunity object
            edge: Calculated edge
            elo_probability: Elo-based probability

        Returns:
            Confidence score 0-1
        """
        confidence = 0.50  # Base confidence

        # Edge contribution (up to +0.25)
        # 8% edge = +0.10, 20% edge = +0.25
        edge_contrib = min(0.25, (edge - self.MIN_EDGE) / 0.12 * 0.15 + 0.10)
        confidence += edge_contrib

        # Elo differential contribution (up to +0.15)
        elo_home = getattr(opp, 'elo_home', None)
        elo_away = getattr(opp, 'elo_away', None)
        if elo_home and elo_away:
            elo_diff = abs(elo_home - elo_away)
            # 100 point gap = +0.05, 300+ point gap = +0.15
            elo_contrib = min(0.15, elo_diff / 300 * 0.15)
            confidence += elo_contrib

        # Probability extreme contribution (up to +0.10)
        # Very confident predictions (>0.75 or <0.25) get bonus
        prob_extreme = abs(elo_probability - 0.5)
        if prob_extreme > 0.25:
            extreme_contrib = min(0.10, (prob_extreme - 0.25) / 0.25 * 0.10)
            confidence += extreme_contrib

        # Cap at 1.0
        return min(1.0, confidence)

    def _calculate_position_size(self, edge: float, confidence: float) -> float:
        """
        Calculate position size based on edge and confidence.

        Uses fractional Kelly criterion: f = (edge / variance) * confidence_factor
        Simplified: position scales with edge * confidence

        Args:
            edge: Calculated edge (0-1)
            confidence: Confidence score (0-1)

        Returns:
            Position size in dollars
        """
        # Kelly-inspired sizing: edge * confidence
        # Scale to range: MIN_POSITION_SIZE to MAX_POSITION_SIZE
        kelly_fraction = edge * confidence * 10  # Multiply by 10 to scale appropriately

        position = self.MIN_POSITION_SIZE + (kelly_fraction * (self.MAX_POSITION_SIZE - self.MIN_POSITION_SIZE))

        # Clamp to min/max
        position = max(self.MIN_POSITION_SIZE, min(self.MAX_POSITION_SIZE, position))

        return round(position, 2)

    def _build_reasoning(
        self,
        opp: 'SportsOpportunity',
        elo_probability: float,
        hybrid_probability: float,
        edge: float,
        confidence: float
    ) -> str:
        """
        Build human-readable reasoning for the trade.

        Args:
            opp: SportsOpportunity object
            elo_probability: Elo-based probability
            hybrid_probability: Hybrid probability
            edge: Calculated edge
            confidence: Confidence score

        Returns:
            Reasoning string
        """
        league = getattr(opp, 'league', 'UNKNOWN')
        home = getattr(opp, 'home_team', 'Home')
        away = getattr(opp, 'away_team', 'Away')
        side = getattr(opp, 'side', 'yes')
        market_price = getattr(opp, 'market_price', 0)

        elo_home = getattr(opp, 'elo_home', None)
        elo_away = getattr(opp, 'elo_away', None)

        reasoning_parts = [
            f"{league}: {home} vs {away}",
            f"Side: {side.upper()}",
            f"Elo probability: {elo_probability*100:.1f}%"
        ]

        if elo_home and elo_away:
            reasoning_parts.append(f"Elo ratings: {home} {elo_home:.0f} vs {away} {elo_away:.0f}")

        reasoning_parts.extend([
            f"Market price: {market_price*100:.1f}%",
            f"Hybrid probability: {hybrid_probability*100:.1f}%",
            f"Edge: {edge*100:.1f}%",
            f"Confidence: {confidence:.2f}"
        ])

        # Add original reasoning if available
        original = getattr(opp, 'reasoning', '')
        if original:
            reasoning_parts.append(f"Analysis: {original}")

        return " | ".join(reasoning_parts)


# Entry point for testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = SportsEdgeEnhancedBot()

    print("\n=== Sports Edge Enhanced Bot Test ===\n")
    print(f"Bot ID: {bot.config.bot_id}")
    print(f"Bot Type: {bot.config.bot_type.value}")
    print(f"Schedule: {bot.config.schedule_seconds}s")
    print(f"Max Positions: {bot.config.max_positions}")
    print(f"Min Edge: {bot.config.min_edge_pct}%")
    print(f"Confidence Threshold: {bot.config.confidence_threshold}")
    print("\nScanning for opportunities...\n")

    signals = bot.scan()

    print(f"\nFound {len(signals)} signals:\n")
    for sig in signals:
        print(f"  {sig.ticker} {sig.side.upper()}")
        print(f"    Edge: {sig.edge_pct:.2f}% | Confidence: {sig.confidence:.2f} | Size: ${sig.position_size}")
        print(f"    {sig.reasoning[:100]}...")
        print()
