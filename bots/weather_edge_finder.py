"""
Weather Edge Finder

Combines NWS forecasts with Kalshi weather markets to find edge opportunities.

Author: Trading Bot
Created: January 2026
"""

import os
import sys
import re
import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.market_scanner import MarketScanner
from scrapers.weather_scraper import WeatherScraper, CityForecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('WeatherEdgeFinder')


class ConfidenceLevel(Enum):
    """Confidence level based on forecast horizon and data quality."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class WeatherOpportunity:
    """A weather trading opportunity with edge"""
    ticker: str
    title: str
    city: str
    date: str
    market_type: str  # 'above', 'below', 'bracket'
    threshold: float
    nws_forecast: int  # NWS temperature forecast
    our_probability: float
    market_price: float
    edge: float
    side: str  # 'YES' or 'NO'
    reasoning: str
    confidence: float = 0.0  # Confidence level 0-1
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    adjusted_probability: float = 0.0  # Probability adjusted for confidence
    kelly_fraction: float = 0.0  # Recommended position size


@dataclass
class ExecutionResult:
    """Result of executing a weather opportunity trade."""
    success: bool
    ticker: str
    side: str
    quantity: int
    price: float
    total_cost: float
    order_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class WeatherEdgeFinder:
    """Find edge opportunities in Kalshi weather markets using NWS data"""

    # NWS forecast typical error (standard deviation in Fahrenheit)
    FORECAST_UNCERTAINTY = 3.0

    # Minimum edge to consider
    MIN_EDGE = 0.05

    def __init__(self):
        self.scanner = MarketScanner()
        self.scraper = WeatherScraper()

    @staticmethod
    def norm_cdf(x: float) -> float:
        """Standard normal cumulative distribution function"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def calc_prob_above(self, forecast: int, threshold: int) -> float:
        """
        Calculate probability that actual temp will be >= threshold.

        Args:
            forecast: NWS forecast temperature
            threshold: Temperature threshold

        Returns:
            Probability (0-1)
        """
        z = (forecast - threshold) / self.FORECAST_UNCERTAINTY
        return self.norm_cdf(z)

    def calc_prob_below(self, forecast: int, threshold: int) -> float:
        """Calculate probability that actual temp will be < threshold"""
        return 1 - self.calc_prob_above(forecast, threshold)

    def calc_prob_bracket(self, forecast: int, low: int, high: int) -> float:
        """
        Calculate probability that actual temp will be in [low, high] bracket.

        Args:
            forecast: NWS forecast temperature
            low: Lower bound of bracket
            high: Upper bound of bracket

        Returns:
            Probability (0-1)
        """
        # P(low <= X <= high) = P(X <= high) - P(X < low)
        prob_le_high = self.norm_cdf((forecast - high) / self.FORECAST_UNCERTAINTY)
        prob_lt_low = self.norm_cdf((forecast - low) / self.FORECAST_UNCERTAINTY)

        # Actually we want: P(X >= low) - P(X > high)
        prob_ge_low = self.norm_cdf((forecast - low) / self.FORECAST_UNCERTAINTY)
        prob_gt_high = self.norm_cdf((forecast - (high + 1)) / self.FORECAST_UNCERTAINTY)

        bracket_prob = prob_ge_low - prob_gt_high

        return max(0, min(1, bracket_prob))

    def calculate_probability_with_confidence(
        self,
        forecast: int,
        threshold: float,
        market_type: str,
        days_ahead: int,
        low: Optional[int] = None,
        high: Optional[int] = None
    ) -> Tuple[float, float, ConfidenceLevel]:
        """
        Calculate probability with confidence adjustment based on forecast horizon.

        NWS forecast accuracy degrades with time:
        - Day 1-2: ~90% accurate within 3°F
        - Day 3-4: ~80% accurate within 3°F
        - Day 5-7: ~70% accurate within 3°F

        Args:
            forecast: NWS forecast temperature
            threshold: Temperature threshold
            market_type: 'above', 'below', or 'bracket'
            days_ahead: Number of days until the market resolves
            low: Lower bracket bound (for bracket type)
            high: Upper bracket bound (for bracket type)

        Returns:
            Tuple of (base_probability, confidence, confidence_level)
        """
        # Adjust uncertainty based on forecast horizon
        if days_ahead <= 2:
            uncertainty = self.FORECAST_UNCERTAINTY * 1.0
            confidence = 0.90
            confidence_level = ConfidenceLevel.HIGH
        elif days_ahead <= 4:
            uncertainty = self.FORECAST_UNCERTAINTY * 1.3
            confidence = 0.75
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            uncertainty = self.FORECAST_UNCERTAINTY * 1.6
            confidence = 0.55
            confidence_level = ConfidenceLevel.LOW

        # Calculate base probability with adjusted uncertainty
        if market_type == 'above':
            z = (forecast - threshold) / uncertainty
            base_prob = self.norm_cdf(z)
        elif market_type == 'below':
            z = (forecast - threshold) / uncertainty
            base_prob = 1 - self.norm_cdf(z)
        elif market_type == 'bracket' and low is not None and high is not None:
            prob_ge_low = self.norm_cdf((forecast - low) / uncertainty)
            prob_gt_high = self.norm_cdf((forecast - (high + 1)) / uncertainty)
            base_prob = max(0, min(1, prob_ge_low - prob_gt_high))
        else:
            base_prob = 0.5
            confidence = 0.0

        return base_prob, confidence, confidence_level

    def calculate_kelly_fraction(
        self,
        probability: float,
        market_price: float,
        confidence: float,
        max_kelly: float = 0.25
    ) -> float:
        """
        Calculate Kelly criterion fraction for position sizing.

        Uses half-Kelly with confidence adjustment for conservative sizing.

        Args:
            probability: Our estimated probability
            market_price: Current market price (cost of YES contract)
            confidence: Confidence in our probability estimate (0-1)
            max_kelly: Maximum Kelly fraction (cap for risk management)

        Returns:
            Recommended fraction of bankroll to bet
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0

        # Expected edge
        edge = probability - market_price

        if edge <= 0:
            return 0.0

        # Kelly formula: f = (p * b - q) / b
        # where p = probability, b = odds (payout - 1), q = 1 - p
        payout = 1.0 / market_price  # e.g., $1 contract costs $0.30 -> payout = 3.33
        b = payout - 1  # Net profit per dollar

        kelly = (probability * b - (1 - probability)) / b

        # Apply half-Kelly for safety
        kelly = kelly / 2

        # Adjust by confidence
        kelly = kelly * confidence

        # Cap at max kelly
        kelly = min(kelly, max_kelly)

        return max(0, kelly)

    async def execute_weather_opportunity(
        self,
        opportunity: WeatherOpportunity,
        max_contracts: int = 100,
        max_cost: float = 50.0,
        dry_run: bool = False
    ) -> ExecutionResult:
        """
        Execute a trade on a weather opportunity.

        Args:
            opportunity: WeatherOpportunity to execute
            max_contracts: Maximum number of contracts to buy
            max_cost: Maximum total cost in dollars
            dry_run: If True, simulate without placing order

        Returns:
            ExecutionResult with trade details
        """
        try:
            # Get current orderbook
            orderbook = self.scanner.client.get_orderbook(opportunity.ticker)
            yes_ask, no_ask = self.scanner.parse_orderbook(orderbook)

            if opportunity.side == 'YES':
                price = yes_ask
            else:
                price = no_ask

            if price <= 0 or price >= 1:
                return ExecutionResult(
                    success=False,
                    ticker=opportunity.ticker,
                    side=opportunity.side,
                    quantity=0,
                    price=0,
                    total_cost=0,
                    error="Invalid price from orderbook"
                )

            # Calculate quantity based on max cost
            price_cents = int(price * 100)
            max_by_cost = int(max_cost / price) if price > 0 else 0
            quantity = min(max_contracts, max_by_cost)

            if quantity <= 0:
                return ExecutionResult(
                    success=False,
                    ticker=opportunity.ticker,
                    side=opportunity.side,
                    quantity=0,
                    price=price,
                    total_cost=0,
                    error="Quantity would be zero"
                )

            total_cost = quantity * price

            if dry_run:
                logger.info(f"[DRY RUN] Would buy {quantity} {opportunity.side} contracts of {opportunity.ticker} @ {price:.2%}")
                return ExecutionResult(
                    success=True,
                    ticker=opportunity.ticker,
                    side=opportunity.side,
                    quantity=quantity,
                    price=price,
                    total_cost=total_cost,
                    order_id="DRY_RUN"
                )

            # Place actual order
            response = self.scanner.client.create_order(
                ticker=opportunity.ticker,
                side=opportunity.side.lower(),
                action='buy',
                count=quantity,
                type='limit',
                yes_price=price_cents if opportunity.side == 'YES' else None,
                no_price=price_cents if opportunity.side == 'NO' else None
            )

            order_id = response.get('order', {}).get('order_id')

            logger.info(f"Placed order: {quantity} {opportunity.side} contracts of {opportunity.ticker} @ {price:.2%}")

            # Send telegram alert if available
            try:
                from utils.telegram_bot import send_opportunity_alert
                send_opportunity_alert(
                    source="WEATHER",
                    symbol=opportunity.ticker,
                    opportunity_type=opportunity.side,
                    edge=opportunity.edge * 100,
                    confidence=opportunity.confidence,
                    details=f"{opportunity.city} {opportunity.date}: {opportunity.reasoning}",
                    priority="high" if opportunity.edge > 0.10 else "medium"
                )
            except ImportError:
                pass

            return ExecutionResult(
                success=True,
                ticker=opportunity.ticker,
                side=opportunity.side,
                quantity=quantity,
                price=price,
                total_cost=total_cost,
                order_id=order_id
            )

        except Exception as e:
            logger.error(f"Failed to execute opportunity {opportunity.ticker}: {e}")
            return ExecutionResult(
                success=False,
                ticker=opportunity.ticker,
                side=opportunity.side,
                quantity=0,
                price=0,
                total_cost=0,
                error=str(e)
            )

    def execute_weather_opportunity_sync(
        self,
        opportunity: WeatherOpportunity,
        max_contracts: int = 100,
        max_cost: float = 50.0,
        dry_run: bool = False
    ) -> ExecutionResult:
        """Synchronous wrapper for execute_weather_opportunity."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.execute_weather_opportunity(opportunity, max_contracts, max_cost, dry_run)
        )

    def parse_ticker(self, ticker: str, title: str) -> Optional[Dict]:
        """
        Parse a Kalshi weather ticker to extract market details.

        Examples:
            KXHIGHNY-26JAN27-T23  -> above 23°
            KXHIGHNY-26JAN27-T16  -> below 16° (when title says "<")
            KXHIGHNY-26JAN27-B22.5 -> bracket 22-23°

        Returns:
            Dict with 'city', 'date', 'type', 'threshold', etc.
        """
        result = {}

        # Determine city
        city_patterns = [
            ('NYC', ['NY', 'NYC', 'new york']),
            ('LA', ['LA', 'los angeles']),
            ('Chicago', ['CHI', 'chicago']),
            ('Miami', ['MIA', 'miami']),
            ('Phoenix', ['PHX', 'phoenix']),
        ]

        for city_code, patterns in city_patterns:
            for pattern in patterns:
                if pattern.upper() in ticker.upper() or pattern.lower() in title.lower():
                    result['city'] = city_code
                    break
            if 'city' in result:
                break

        if 'city' not in result:
            return None

        # Parse date: format is YY{MONTH}DD (e.g., 26JAN27 = Jan 27, 2026)
        date_match = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker)
        if not date_match:
            return None

        year = '20' + date_match.group(1)
        month_abbr = date_match.group(2)
        day = date_match.group(3)

        months = {
            'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
            'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
            'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
        }
        month = months.get(month_abbr)
        if not month:
            return None

        result['date'] = f'{year}-{month}-{day}'

        # Parse market type and threshold
        threshold_match = re.search(r'-T(\d+)', ticker)
        bracket_match = re.search(r'-B(\d+\.?\d*)', ticker)

        title_lower = title.lower()

        if threshold_match:
            threshold = int(threshold_match.group(1))
            result['threshold'] = threshold

            # Determine if above or below based on title
            if '<' in title_lower or 'less than' in title_lower or 'below' in title_lower:
                result['type'] = 'below'
            else:
                result['type'] = 'above'

        elif bracket_match:
            mid = float(bracket_match.group(1))
            result['type'] = 'bracket'
            result['low'] = int(mid - 0.5)
            result['high'] = int(mid + 0.5)
            result['threshold'] = mid

        else:
            return None

        # Determine if high or low temp market
        if 'HIGH' in ticker.upper() or 'high temp' in title_lower:
            result['metric'] = 'high'
        elif 'LOW' in ticker.upper() or 'low temp' in title_lower:
            result['metric'] = 'low'
        else:
            result['metric'] = 'high'  # Default assumption

        return result

    def find_opportunities(self) -> List[WeatherOpportunity]:
        """
        Find all weather market opportunities with edge.

        Returns:
            List of WeatherOpportunity objects sorted by edge
        """
        logger.info("Finding weather market opportunities...")

        # Get Kalshi weather markets
        markets = self.scanner.find_weather_markets()
        logger.info(f"Found {len(markets)} weather markets")

        # Get NWS forecasts
        forecasts = self.scraper.get_all_forecasts()
        logger.info(f"Got forecasts for {len(forecasts)} cities")

        opportunities = []

        for market in markets:
            ticker = market.get('ticker', '')
            title = market.get('title', '')

            # Parse ticker
            parsed = self.parse_ticker(ticker, title)
            if not parsed:
                continue

            city = parsed['city']
            target_date = parsed['date']
            metric = parsed.get('metric', 'high')

            # Find matching forecast
            city_forecasts = forecasts.get(city, [])
            matching_fc = None
            for fc in city_forecasts:
                if fc.date == target_date:
                    matching_fc = fc
                    break

            if not matching_fc:
                continue

            # Get the relevant temperature
            if metric == 'high':
                nws_temp = matching_fc.high_temp
            else:
                nws_temp = matching_fc.low_temp

            if nws_temp is None:
                continue

            # Calculate days ahead for confidence adjustment
            try:
                market_date = datetime.strptime(target_date, '%Y-%m-%d')
                days_ahead = (market_date.date() - datetime.now().date()).days
            except Exception as e:
                logger.debug(f"Error parsing market date: {e}")
                days_ahead = 3  # Default assumption

            # Calculate our probability with confidence
            market_type = parsed['type']
            low_bound = parsed.get('low')
            high_bound = parsed.get('high')

            base_prob, confidence, confidence_level = self.calculate_probability_with_confidence(
                forecast=nws_temp,
                threshold=parsed['threshold'],
                market_type=market_type,
                days_ahead=days_ahead,
                low=low_bound,
                high=high_bound
            )

            # Also calculate simple probability for comparison
            if market_type == 'above':
                our_prob = self.calc_prob_above(nws_temp, parsed['threshold'])
            elif market_type == 'below':
                our_prob = self.calc_prob_below(nws_temp, parsed['threshold'])
            elif market_type == 'bracket':
                our_prob = self.calc_prob_bracket(nws_temp, parsed['low'], parsed['high'])
            else:
                continue

            # Get market price
            try:
                orderbook = self.scanner.client.get_orderbook(ticker)
                yes_ask, no_ask = self.scanner.parse_orderbook(orderbook)
            except Exception as e:
                logger.debug(f"Could not get orderbook for {ticker}: {e}")
                continue

            # Calculate edge for YES side
            yes_edge = our_prob - yes_ask
            # Calculate edge for NO side
            no_edge = (1 - our_prob) - no_ask

            # Determine best opportunity
            if yes_edge >= self.MIN_EDGE:
                edge = yes_edge
                side = 'YES'
                market_price = yes_ask
            elif no_edge >= self.MIN_EDGE:
                edge = no_edge
                side = 'NO'
                market_price = no_ask
                our_prob = 1 - our_prob
            else:
                continue

            # Build reasoning
            if market_type == 'bracket':
                reasoning = f"NWS: {nws_temp}F, bracket [{parsed['low']}-{parsed['high']}]"
            else:
                reasoning = f"NWS: {nws_temp}F vs threshold {parsed['threshold']}F"

            # Calculate Kelly fraction for position sizing
            kelly = self.calculate_kelly_fraction(
                probability=our_prob,
                market_price=market_price,
                confidence=confidence
            )

            # Use the confidence-adjusted probability from the statistical model.
            # calculate_probability_with_confidence() already widens the uncertainty
            # (standard deviation) for farther-out forecasts, which naturally pulls
            # extreme probabilities toward 50%. Adding a SECOND regression toward 50%
            # here was a double-penalty that suppressed legitimate strong signals.
            adjusted_prob = base_prob

            opportunities.append(WeatherOpportunity(
                ticker=ticker,
                title=title[:60],
                city=city,
                date=target_date,
                market_type=market_type,
                threshold=parsed.get('threshold', 0),
                nws_forecast=nws_temp,
                our_probability=our_prob,
                market_price=market_price,
                edge=edge,
                side=side,
                reasoning=reasoning,
                confidence=confidence,
                confidence_level=confidence_level,
                adjusted_probability=adjusted_prob,
                kelly_fraction=kelly
            ))

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge, reverse=True)
        logger.info(f"Found {len(opportunities)} opportunities with edge >= {self.MIN_EDGE:.0%}")

        return opportunities

    def print_opportunities(self, opportunities: List[WeatherOpportunity], limit: int = 10):
        """Print opportunities in a formatted way"""
        print()
        print("=" * 70)
        print("WEATHER TRADING OPPORTUNITIES")
        print("=" * 70)
        print()

        if not opportunities:
            print("No opportunities found with sufficient edge.")
            return

        for i, opp in enumerate(opportunities[:limit], 1):
            conf_emoji = "🟢" if opp.confidence_level == ConfidenceLevel.HIGH else ("🟡" if opp.confidence_level == ConfidenceLevel.MEDIUM else "🔴")
            print(f"{i}. {opp.ticker}")
            print(f"   {opp.title}")
            print(f"   City: {opp.city} | Date: {opp.date}")
            print(f"   NWS Forecast: {opp.nws_forecast}F | Type: {opp.market_type}")
            print(f"   Our Probability: {opp.our_probability:.0%} | Adjusted: {opp.adjusted_probability:.0%}")
            print(f"   Market Price:    {opp.market_price:.0%}")
            print(f"   EDGE: {opp.edge:.0%} -> BUY {opp.side}")
            print(f"   Confidence: {conf_emoji} {opp.confidence:.0%} ({opp.confidence_level.value})")
            print(f"   Kelly Size: {opp.kelly_fraction:.1%} of bankroll")
            print(f"   {opp.reasoning}")
            print()

        print("=" * 70)


def main():
    """Run the weather edge finder"""
    from dotenv import load_dotenv
    load_dotenv()

    finder = WeatherEdgeFinder()
    opportunities = finder.find_opportunities()
    finder.print_opportunities(opportunities)

    # Summary
    print()
    print("SUMMARY")
    print("-" * 40)
    print(f"Total opportunities: {len(opportunities)}")
    if opportunities:
        avg_edge = sum(o.edge for o in opportunities) / len(opportunities)
        print(f"Average edge: {avg_edge:.1%}")
        print(f"Best edge: {opportunities[0].edge:.1%} on {opportunities[0].ticker}")


if __name__ == "__main__":
    main()
