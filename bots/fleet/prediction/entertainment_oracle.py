"""
Entertainment Oracle Bot - Fleet Trading System

Exploits awards precursor correlation patterns.
DGA/PGA/SAG winners predict Oscar outcomes with 60-70% accuracy.

Author: Fleet Trading Bot System
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Set
import re

# Project root path resolution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

try:
    from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType
except ImportError as e:
    logging.error(f"Failed to import FleetBot base: {e}")
    raise

try:
    from bots.kalshi_client import KalshiClient
except ImportError as e:
    logging.error(f"Failed to import KalshiClient: {e}")
    raise

try:
    import requests
except ImportError as e:
    logging.error(f"Failed to import requests: {e}")
    raise


class EntertainmentOracleBot(FleetBot):
    """
    Entertainment/Awards prediction bot using precursor correlation.

    Strategy:
    - DGA (Directors Guild) → 70% predicts Best Director Oscar
    - PGA (Producers Guild) → 65% predicts Best Picture Oscar
    - SAG Ensemble → 60% predicts Best Picture Oscar
    - Multiple precursors agreeing → higher confidence
    - Seasonal: Most action Jan-Mar (awards season)
    - Small positions ($3-8) due to high variance
    """

    # Award keywords for Kalshi market scanning
    AWARD_KEYWORDS = {
        'oscar', 'academy award', 'academy awards', 'oscars',
        'golden globe', 'golden globes',
        'emmy', 'emmys',
        'grammy', 'grammys',
        'dga', 'directors guild',
        'pga', 'producers guild',
        'sag', 'screen actors guild',
        'bafta', 'british academy'
    }

    # Precursor correlation strengths
    PRECURSOR_WEIGHTS = {
        'DGA': 0.70,  # DGA → Best Director
        'PGA': 0.65,  # PGA → Best Picture
        'SAG_ENSEMBLE': 0.60,  # SAG Ensemble → Best Picture
        'GOLDEN_GLOBE_DRAMA': 0.55,  # GG Drama → Best Picture
        'GOLDEN_GLOBE_DIRECTOR': 0.50,  # GG Director → Best Director
        'BAFTA': 0.50,  # BAFTA → Oscar correlation
    }

    # Position sizing
    MIN_POSITION_SIZE = 3  # $3 minimum (very speculative)
    MAX_POSITION_SIZE = 8  # $8 maximum

    # Edge threshold (lower than sports due to correlation strength)
    MIN_EDGE = 0.10  # 10% minimum edge
    MIN_CONFIDENCE = 0.50  # 50% minimum confidence

    # Max positions
    MAX_POSITIONS = 3  # Conservative - entertainment is high variance

    # Awards season (most activity)
    AWARDS_SEASON_START = (1, 1)  # Jan 1
    AWARDS_SEASON_END = (3, 31)    # Mar 31

    def __init__(self, config: Optional[FleetBotConfig] = None):
        """
        Initialize Entertainment Oracle bot.

        Args:
            config: Fleet bot configuration. Uses defaults if None.
        """
        if config is None:
            config = FleetBotConfig(
                name="Entertainment-Oracle",
                bot_type=BotType.PREDICTION,
                schedule_seconds=86400,  # Daily (very seasonal)
                max_position_usd=8.0,
                max_daily_trades=3,
                min_confidence=self.MIN_CONFIDENCE,
                symbols=[],
                enabled=True,
                paper_mode=True,
                extra={'min_edge': self.MIN_EDGE}
            )

        super().__init__(config)

        # Initialize Kalshi client
        try:
            self.kalshi = KalshiClient()
            self.logger.info("KalshiClient initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize KalshiClient: {e}")
            self.kalshi = None

        # Cache for precursor winners (avoid re-scraping)
        self.precursor_cache: Dict[str, Dict] = {}
        self.cache_timestamp = None

        self.logger.info(f"EntertainmentOracleBot initialized: {self.MAX_POSITIONS} max positions, {self.MIN_EDGE*100:.1f}% min edge")

    def scan(self) -> List[FleetSignal]:
        """
        Scan for entertainment/awards opportunities using precursor correlation.

        Returns:
            List of FleetSignal objects representing trading opportunities
        """
        signals = []

        try:
            # Check if Kalshi client available
            if not self.kalshi:
                self.logger.warning("KalshiClient not available, skipping scan")
                return signals

            # Check if we're in awards season (otherwise very low activity)
            if not self._is_awards_season():
                self.logger.info("Not in awards season (Jan-Mar), skipping scan")
                return signals

            # Refresh precursor winners cache if stale
            self._refresh_precursor_cache()

            # Scan Kalshi for entertainment/awards markets
            self.logger.info("Scanning Kalshi for entertainment/awards markets...")
            markets = self._get_entertainment_markets()

            if not markets:
                self.logger.info("No entertainment markets found on Kalshi")
                return signals

            self.logger.info(f"Found {len(markets)} entertainment markets")

            # Process each market for precursor correlation
            for market in markets:
                try:
                    signal = self._process_market(market)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    ticker = market.get('ticker', 'UNKNOWN')
                    self.logger.error(f"Error processing market {ticker}: {e}")
                    continue

            # Sort by edge (highest first)
            signals.sort(key=lambda s: s.edge, reverse=True)

            # Limit to max positions
            if len(signals) > self.MAX_POSITIONS:
                self.logger.info(f"Limiting {len(signals)} signals to top {self.MAX_POSITIONS}")
                signals = signals[:self.MAX_POSITIONS]

            self.logger.info(f"Generated {len(signals)} final trading signals")

        except Exception as e:
            self.logger.error(f"Error in entertainment scan: {e}", exc_info=True)

        return signals

    def _is_awards_season(self) -> bool:
        """Check if current date is in awards season (Jan-Mar)."""
        now = datetime.now(timezone.utc)
        month = now.month
        day = now.day

        start_month, start_day = self.AWARDS_SEASON_START
        end_month, end_day = self.AWARDS_SEASON_END

        # Simple month/day range check
        if start_month <= month <= end_month:
            if month == start_month and day < start_day:
                return False
            if month == end_month and day > end_day:
                return False
            return True

        return False

    def _get_entertainment_markets(self) -> List[Dict]:
        """
        Get entertainment/awards markets from Kalshi.

        Returns:
            List of market dictionaries
        """
        markets = []

        try:
            # Search Kalshi for entertainment markets
            # Try multiple keyword searches
            for keyword in ['oscar', 'golden globe', 'emmy', 'grammy', 'award']:
                try:
                    result = self.kalshi.get_markets(status='open', limit=200)

                    if result:
                        for market in result:
                            title = market.get('title', '').lower()
                            ticker = market.get('ticker', '').lower()

                            # Check if market matches entertainment keywords
                            if any(kw in title or kw in ticker for kw in self.AWARD_KEYWORDS):
                                if market not in markets:  # Avoid duplicates
                                    markets.append(market)

                except Exception as e:
                    self.logger.warning(f"Error searching Kalshi for keyword '{keyword}': {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error getting entertainment markets: {e}")

        return markets

    def _refresh_precursor_cache(self):
        """
        Refresh cache of precursor award winners.
        Cache is valid for 24 hours.
        """
        now = datetime.now(timezone.utc)

        # Check if cache is still valid
        if self.cache_timestamp:
            age = (now - self.cache_timestamp).total_seconds()
            if age < 86400:  # 24 hours
                self.logger.debug("Precursor cache still valid")
                return

        self.logger.info("Refreshing precursor winners cache...")

        # Scrape/update precursor winners
        # In production, this would scrape award announcement sites
        # For now, use a simple structure that can be manually updated

        # Example structure (would be populated by web scraping in production):
        # self.precursor_cache = {
        #     'DGA_2026': {'winner': 'Oppenheimer', 'date': '2026-02-15'},
        #     'PGA_2026': {'winner': 'Oppenheimer', 'date': '2026-02-20'},
        #     'SAG_ENSEMBLE_2026': {'winner': 'Oppenheimer', 'date': '2026-02-24'},
        # }

        # For MVP, just log that we would scrape here
        self.logger.info("Precursor cache refresh: would scrape DGA/PGA/SAG winners here")

        self.cache_timestamp = now

    def _process_market(self, market: Dict) -> Optional[FleetSignal]:
        """
        Process a single entertainment market for precursor correlation.

        Args:
            market: Market dictionary from Kalshi

        Returns:
            FleetSignal if opportunity found, None otherwise
        """
        ticker = market.get('ticker', '')
        title = market.get('title', '')

        # Extract nominee/option from title
        # Example: "Will [Movie Name] win Best Picture?"
        nominee = self._extract_nominee(title)
        if not nominee:
            self.logger.debug(f"Could not extract nominee from: {title}")
            return None

        # Check for precursor correlation
        precursors = self._find_matching_precursors(nominee, title)
        if not precursors:
            self.logger.debug(f"No precursor match for {nominee}")
            return None

        # Calculate implied probability from precursors
        precursor_probability = self._calculate_precursor_probability(precursors)

        # Get current market price
        market_price = self._get_market_price(market)
        if market_price is None or market_price <= 0:
            self.logger.warning(f"Invalid market price for {ticker}")
            return None

        # Calculate edge
        edge = precursor_probability - market_price
        edge_pct = edge * 100

        # Check edge threshold
        if edge < self.MIN_EDGE:
            self.logger.debug(f"Skipping {ticker}: edge {edge_pct:.2f}% below threshold {self.MIN_EDGE*100:.1f}%")
            return None

        # Calculate confidence based on precursor strength and agreement
        confidence = self._calculate_confidence(precursors, edge)

        # Check confidence threshold
        if confidence < self.MIN_CONFIDENCE:
            self.logger.debug(f"Skipping {ticker}: confidence {confidence:.2f} below threshold {self.MIN_CONFIDENCE}")
            return None

        # Calculate position size (conservative for entertainment)
        position_size = self._calculate_position_size(edge, confidence)

        # Build reasoning
        reasoning = self._build_reasoning(title, nominee, precursors, precursor_probability, market_price, edge, confidence)

        # Metadata
        metadata = {
            'nominee': nominee,
            'precursors': [p['type'] for p in precursors],
            'precursor_probability': precursor_probability,
            'market_price': market_price,
            'title': title
        }

        # Create FleetSignal
        signal = FleetSignal(
            bot_name=self.name,
            bot_type=self.bot_type.value,
            symbol=ticker,
            side='YES',  # Buy YES on precursor winner
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

        self.logger.info(f"Generated signal: {ticker} YES | Edge: {edge_pct:.2f}% | Conf: {confidence:.2f} | Size: ${position_size}")

        return signal

    def _extract_nominee(self, title: str) -> Optional[str]:
        """
        Extract nominee/film/artist name from market title.

        Args:
            title: Market title

        Returns:
            Nominee name or None
        """
        # Common patterns:
        # "Will [Name] win Best Picture?"
        # "[Name] to win Best Director Oscar"
        # "Oscar Best Picture: [Name]"

        # Try regex patterns
        patterns = [
            r'Will\s+([^?]+?)\s+win',
            r'([^:]+?)\s+to win',
            r':\s+([^?]+)',
            r'"([^"]+)"'  # Quoted titles
        ]

        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                nominee = match.group(1).strip()
                # Clean up common words
                nominee = re.sub(r'\s+(win|wins|winning|the)\s+', ' ', nominee, flags=re.IGNORECASE)
                return nominee.strip()

        return None

    def _find_matching_precursors(self, nominee: str, title: str) -> List[Dict]:
        """
        Find precursor awards won by this nominee.

        Args:
            nominee: Nominee name
            title: Market title (for category context)

        Returns:
            List of matching precursor dictionaries
        """
        precursors = []

        # In production, this would check self.precursor_cache
        # For MVP, use placeholder logic

        # Example: if nominee appears in cache with matching category
        # precursors.append({
        #     'type': 'DGA',
        #     'weight': 0.70,
        #     'date': '2026-02-15'
        # })

        # For now, return empty list (would be populated by real scraping)
        return precursors

    def _calculate_precursor_probability(self, precursors: List[Dict]) -> float:
        """
        Calculate implied probability from precursor correlation.

        Args:
            precursors: List of matching precursor awards

        Returns:
            Implied probability (0-1)
        """
        if not precursors:
            return 0.5  # No info = 50/50

        # If multiple precursors agree, use weighted average
        # with diminishing returns for additional signals

        weights = [p['weight'] for p in precursors]

        if len(weights) == 1:
            return weights[0]

        # Multiple precursors: weighted average with bonus for agreement
        avg_weight = sum(weights) / len(weights)
        agreement_bonus = min(0.10, (len(weights) - 1) * 0.05)  # +5% per additional precursor, max +10%

        probability = avg_weight + agreement_bonus

        return min(0.95, probability)  # Cap at 95%

    def _get_market_price(self, market: Dict) -> Optional[float]:
        """
        Get current market price for YES side.

        Args:
            market: Market dictionary

        Returns:
            Market price (0-1) or None
        """
        try:
            # Kalshi markets have yes_bid/yes_ask
            yes_bid = market.get('yes_bid', 0)
            yes_ask = market.get('yes_ask', 0)

            if yes_bid > 0 and yes_ask > 0:
                # Use midpoint
                return (yes_bid + yes_ask) / 2 / 100  # Convert cents to 0-1
            elif yes_ask > 0:
                return yes_ask / 100
            elif yes_bid > 0:
                return yes_bid / 100

            # Fallback: use last_price if available
            last_price = market.get('last_price')
            if last_price:
                return last_price / 100

        except Exception as e:
            self.logger.warning(f"Error getting market price: {e}")

        return None

    def _calculate_confidence(self, precursors: List[Dict], edge: float) -> float:
        """
        Calculate confidence score.

        Args:
            precursors: List of matching precursor awards
            edge: Calculated edge

        Returns:
            Confidence score (0-1)
        """
        # Base confidence from number of precursors
        if len(precursors) == 0:
            return 0.30
        elif len(precursors) == 1:
            confidence = 0.50
        elif len(precursors) == 2:
            confidence = 0.65
        else:
            confidence = 0.75

        # Edge contribution (larger edge = higher confidence)
        edge_contrib = min(0.15, edge * 0.50)
        confidence += edge_contrib

        # Cap at 0.90 (entertainment is inherently uncertain)
        return min(0.90, confidence)

    def _calculate_position_size(self, edge: float, confidence: float) -> float:
        """
        Calculate position size (conservative for entertainment).

        Args:
            edge: Calculated edge
            confidence: Confidence score

        Returns:
            Position size in dollars
        """
        # Very conservative sizing for entertainment
        kelly_fraction = edge * confidence * 5  # Half the multiplier of sports

        position = self.MIN_POSITION_SIZE + (kelly_fraction * (self.MAX_POSITION_SIZE - self.MIN_POSITION_SIZE))

        # Clamp to min/max
        position = max(self.MIN_POSITION_SIZE, min(self.MAX_POSITION_SIZE, position))

        return round(position, 2)

    def _build_reasoning(
        self,
        title: str,
        nominee: str,
        precursors: List[Dict],
        precursor_probability: float,
        market_price: float,
        edge: float,
        confidence: float
    ) -> str:
        """Build human-readable reasoning for the trade."""
        reasoning_parts = [
            f"Entertainment: {title}",
            f"Nominee: {nominee}"
        ]

        if precursors:
            precursor_names = [p['type'] for p in precursors]
            reasoning_parts.append(f"Precursors won: {', '.join(precursor_names)}")
            reasoning_parts.append(f"Implied probability: {precursor_probability*100:.1f}%")

        reasoning_parts.extend([
            f"Market price: {market_price*100:.1f}%",
            f"Edge: {edge*100:.1f}%",
            f"Confidence: {confidence:.2f}"
        ])

        return " | ".join(reasoning_parts)


# Entry point for testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = EntertainmentOracleBot()

    print("\n=== Entertainment Oracle Bot Test ===\n")
    print(f"Bot ID: {bot.config.bot_id}")
    print(f"Bot Type: {bot.config.bot_type.value}")
    print(f"Schedule: {bot.config.schedule_seconds}s")
    print(f"Max Positions: {bot.config.max_positions}")
    print(f"Min Edge: {bot.config.min_edge_pct}%")
    print(f"Confidence Threshold: {bot.config.confidence_threshold}")
    print(f"Awards Season: {bot._is_awards_season()}")
    print("\nScanning for opportunities...\n")

    signals = bot.scan()

    print(f"\nFound {len(signals)} signals:\n")
    for sig in signals:
        print(f"  {sig.ticker} {sig.side.upper()}")
        print(f"    Edge: {sig.edge_pct:.2f}% | Confidence: {sig.confidence:.2f} | Size: ${sig.position_size}")
        print(f"    {sig.reasoning}")
        print()
