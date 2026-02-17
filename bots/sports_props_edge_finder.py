"""
Sports Props Edge Finder

Matches team total estimates to Kalshi sports prop markets.
Uses pace and efficiency stats to calculate expected team totals.

Author: Trading Bot
Created: January 2026
"""

import os
import sys
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient
from scrapers.sports_props_scraper import SportsPropsScaper, GameProps
from scrapers.sports_scraper import SportsScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SportsPropsEdgeFinder')


@dataclass
class SportsPropOpportunity:
    """Sports prop trading opportunity"""
    ticker: str
    title: str
    league: str
    team: str
    prop_type: str  # team_total, game_total
    threshold: float
    direction: str  # over or under
    side: str  # YES or NO
    our_probability: float
    market_price: float
    edge: float
    expected_value: float
    reasoning: str
    confidence: str


class SportsPropsEdgeFinder:
    """
    Finds edge in Kalshi sports prop markets.

    Focuses on team totals where we can model expected scoring.
    """

    MIN_EDGE = 0.05  # 5% minimum edge
    MIN_PRICE = 0.05
    MAX_PRICE = 0.95

    # Kalshi team total series
    TEAM_TOTAL_SERIES = {
        'NFL': ['KXNFLTEAMTOTAL'],
        'NBA': ['KXNBATEAMTOTAL'],
    }

    TRADEABLE_STATUSES = ['active', 'open', 'trading']

    def __init__(self):
        """Initialize the sports props edge finder"""
        self.client = KalshiClient()
        self.props_scraper = SportsPropsScaper()
        self.sports_scraper = SportsScraper()

        logger.info("SportsPropsEdgeFinder initialized")

    def find_team_total_markets(self) -> Dict[str, List[Dict]]:
        """Find all active team total markets on Kalshi"""
        markets_by_league = {'NFL': [], 'NBA': []}

        for league, series_list in self.TEAM_TOTAL_SERIES.items():
            for series_ticker in series_list:
                try:
                    data = self.client._request('GET', '/markets', params={
                        'series_ticker': series_ticker,
                        'limit': 100
                    })
                    markets = data.get('markets', [])

                    for market in markets:
                        status = market.get('status', '').lower()
                        if status in self.TRADEABLE_STATUSES:
                            market['league'] = league
                            markets_by_league[league].append(market)

                except Exception as e:
                    logger.debug(f"Error fetching {series_ticker}: {e}")

        total = sum(len(m) for m in markets_by_league.values())
        logger.info(f"Found {total} team total markets (NFL: {len(markets_by_league['NFL'])}, "
                   f"NBA: {len(markets_by_league['NBA'])})")

        return markets_by_league

    def parse_team_total_market(self, market: Dict) -> Optional[Dict]:
        """
        Parse team total market to extract team, threshold, direction.

        Examples:
        - KXNFLTEAMTOTAL-26FEB08SEANE-SEA28 -> Seattle, 28 points, over
        - KXNBATEAMTOTAL-*-BOS115 -> Boston, 115 points, over

        Args:
            market: Kalshi market dict

        Returns:
            Dict with team, threshold, direction or None
        """
        ticker = market.get('ticker', '').upper()
        title = market.get('title', '').lower()

        # Pattern: -TEAMTHRESHOLD at end of ticker
        # Examples: -SEA28, -BOS115, -DEN120
        ticker_pattern = r'-([A-Z]{2,3})(\d+)$'
        match = re.search(ticker_pattern, ticker)

        if match:
            team_abbr = match.group(1)
            threshold = float(match.group(2))

            # Most team total markets are "over" bets (will team score X+?)
            direction = 'over'

            # Check title for direction hints
            if 'under' in title or 'fewer' in title or 'less' in title:
                direction = 'under'

            return {
                'team': team_abbr,
                'threshold': threshold,
                'direction': direction
            }

        # Try parsing from title
        # "Will Seattle score 28 or more points?"
        title_pattern = r'(\w+)\s+(?:score|total)[^\d]*(\d+)'
        match = re.search(title_pattern, title)

        if match:
            team_name = match.group(1).upper()
            threshold = float(match.group(2))

            # Map common team names to abbreviations
            name_to_abbr = {
                'SEATTLE': 'SEA', 'PATRIOTS': 'NE', 'BOSTON': 'BOS',
                'DENVER': 'DEN', 'LAKERS': 'LAL', 'WARRIORS': 'GSW',
                'CELTICS': 'BOS', 'BRONCOS': 'DEN', 'CHIEFS': 'KC',
            }

            team_abbr = name_to_abbr.get(team_name, team_name[:3])

            direction = 'over' if 'or more' in title or 'over' in title else 'under'

            return {
                'team': team_abbr,
                'threshold': threshold,
                'direction': direction
            }

        return None

    def extract_game_info_from_market(self, market: Dict) -> Optional[Tuple[str, str]]:
        """
        Extract home and away team from market.

        Args:
            market: Kalshi market dict

        Returns:
            Tuple of (home_team, away_team) or None
        """
        ticker = market.get('ticker', '').upper()
        title = market.get('title', '').lower()

        # Pattern in ticker: -26FEB08SEANE- (Seattle at New England)
        game_pattern = r'-\d+[A-Z]{3}\d+-?([A-Z]{2,3})([A-Z]{2,3})-'
        match = re.search(game_pattern, ticker)

        if match:
            away = match.group(1)
            home = match.group(2)
            return (home, away)

        # Try title: "Seattle at New England"
        title_pattern = r'(\w+)\s+(?:at|vs|@)\s+(\w+)'
        match = re.search(title_pattern, title)

        if match:
            away_name = match.group(1).upper()
            home_name = match.group(2).upper()

            # Map to abbreviations
            name_to_abbr = {
                'SEATTLE': 'SEA', 'NEW ENGLAND': 'NE', 'PATRIOTS': 'NE',
                'BOSTON': 'BOS', 'DENVER': 'DEN', 'LOS ANGELES': 'LA',
            }

            away = name_to_abbr.get(away_name, away_name[:3])
            home = name_to_abbr.get(home_name, home_name[:3])

            return (home, away)

        return None

    def get_market_price(self, market: Dict) -> Optional[float]:
        """Get current market price (YES probability)"""
        try:
            ticker = market.get('ticker')
            orderbook = self.client.get_orderbook(ticker)

            if not orderbook:
                return None

            yes_orders = orderbook.get('yes', [])
            no_orders = orderbook.get('no', [])

            best_yes_bid = 0
            if yes_orders:
                prices = [order[0] for order in yes_orders if len(order) >= 2 and order[1] > 0]
                if prices:
                    best_yes_bid = max(prices)

            best_no_bid = 0
            if no_orders:
                prices = [order[0] for order in no_orders if len(order) >= 2 and order[1] > 0]
                if prices:
                    best_no_bid = max(prices)

            yes_ask = 100 - best_no_bid if best_no_bid > 0 else 100

            if best_yes_bid > 0:
                midpoint = (best_yes_bid + yes_ask) / 200
                return midpoint
            elif yes_ask < 100:
                return yes_ask / 100

            if market.get('last_price'):
                return market.get('last_price') / 100

            return None

        except Exception as e:
            logger.debug(f"Error getting price for {market.get('ticker')}: {e}")
            return None

    def find_opportunities(self) -> List[SportsPropOpportunity]:
        """
        Find sports prop trading opportunities.

        Returns:
            List of SportsPropOpportunity with positive edge
        """
        opportunities = []

        # Fetch team stats
        logger.info("Fetching team statistics...")
        nba_stats = self.props_scraper.fetch_nba_team_stats()
        nfl_stats = self.props_scraper.get_nfl_team_stats()

        logger.info(f"  NBA: {len(nba_stats)} teams")
        logger.info(f"  NFL: {len(nfl_stats)} teams")

        # Find markets
        markets_by_league = self.find_team_total_markets()

        # Process each market
        for league, markets in markets_by_league.items():
            stats = nba_stats if league == 'NBA' else nfl_stats

            for market in markets:
                opp = self._evaluate_team_total_market(market, league, stats)
                if opp:
                    opportunities.append(opp)

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge, reverse=True)
        logger.info(f"Found {len(opportunities)} sports prop opportunities with edge >= {self.MIN_EDGE:.0%}")

        return opportunities

    def _evaluate_team_total_market(
        self,
        market: Dict,
        league: str,
        stats: Dict
    ) -> Optional[SportsPropOpportunity]:
        """Evaluate a team total market"""
        # Parse market
        parsed = self.parse_team_total_market(market)
        if not parsed:
            return None

        team = parsed['team']
        threshold = parsed['threshold']
        direction = parsed['direction']

        # Get game info
        game_info = self.extract_game_info_from_market(market)
        if not game_info:
            # If we can't determine opponent, use average defense
            home_team, away_team = team, 'AVG'
            is_home = True
        else:
            home_team, away_team = game_info
            is_home = (team == home_team)

        # Get team stats
        team_stats = stats.get(team)
        if not team_stats:
            return None

        # Calculate expected total
        if league == 'NBA':
            opp_abbr = away_team if is_home else home_team
            expected, std_dev, reasoning = self.props_scraper.estimate_nba_team_total(
                team, opp_abbr, home=is_home, stats=stats
            )
        else:
            opp_abbr = away_team if is_home else home_team
            expected, std_dev, reasoning = self.props_scraper.estimate_nfl_team_total(
                team, opp_abbr, home=is_home
            )

        # Calculate probability
        prob_over, prob_under = self.props_scraper.calculate_over_under_probability(
            expected, std_dev, threshold
        )

        our_prob = prob_over if direction == 'over' else prob_under

        # Get market price
        market_price = self.get_market_price(market)
        if market_price is None or market_price < self.MIN_PRICE or market_price > self.MAX_PRICE:
            return None

        # Calculate edge
        edge = our_prob - market_price

        if abs(edge) >= self.MIN_EDGE:
            if edge > 0:
                side = 'YES'
            else:
                side = 'NO'
                edge = abs(edge)
                our_prob = 1 - our_prob
                market_price = 1 - market_price

            return SportsPropOpportunity(
                ticker=market.get('ticker', ''),
                title=market.get('title', ''),
                league=league,
                team=team,
                prop_type='team_total',
                threshold=threshold,
                direction=direction,
                side=side,
                our_probability=our_prob,
                market_price=market_price,
                edge=edge,
                expected_value=expected,
                reasoning=f"Expected: {expected:.1f}, Threshold: {threshold}. {reasoning}",
                confidence='MEDIUM'
            )

        return None

    def print_opportunities(self, opportunities: List[SportsPropOpportunity]):
        """Print opportunities in formatted output"""
        print()
        print("=" * 70)
        print("SPORTS PROP OPPORTUNITIES")
        print("=" * 70)

        if not opportunities:
            print("No opportunities found.")
            return

        for i, opp in enumerate(opportunities[:15], 1):
            print(f"\n{i}. [{opp.league}] {opp.ticker}")
            print(f"   {opp.title}")
            print(f"   {opp.team} {opp.direction} {opp.threshold}")
            print(f"   Action: BUY {opp.side} @ {opp.market_price:.0%}")
            print(f"   Our Probability: {opp.our_probability:.0%}")
            print(f"   Edge: {opp.edge:.1%} [{opp.confidence}]")
            print(f"   Expected: {opp.expected_value:.1f}")

        print()
        print("=" * 70)


def main():
    """Run the sports props edge finder"""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("SPORTS PROPS EDGE FINDER")
    print("=" * 60)

    finder = SportsPropsEdgeFinder()

    print("\n[1] Scanning for opportunities...")
    opportunities = finder.find_opportunities()

    print("\n[2] Results:")
    finder.print_opportunities(opportunities)

    print("\n[3] Summary:")
    print("-" * 40)
    print(f"Total opportunities: {len(opportunities)}")

    if opportunities:
        by_league = {}
        for opp in opportunities:
            by_league[opp.league] = by_league.get(opp.league, 0) + 1

        for league, count in sorted(by_league.items()):
            print(f"  {league}: {count}")

        avg_edge = sum(o.edge for o in opportunities) / len(opportunities)
        print(f"\nAverage edge: {avg_edge:.1%}")


if __name__ == "__main__":
    main()
