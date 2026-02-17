"""
Sports Edge Finder

Matches sports probability estimates to Kalshi sports markets.
Identifies edge opportunities based on Elo-derived win probabilities.

Author: Trading Bot
Created: January 2026
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient
from bots.market_scanner import MarketScanner
from scrapers.sports_scraper import SportsScraper, GameData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SportsEdgeFinder')


@dataclass
class SportsOpportunity:
    """Sports trading opportunity"""
    ticker: str
    title: str
    league: str
    home_team: str
    away_team: str
    side: str  # YES or NO
    our_probability: float
    market_price: float
    edge: float
    elo_home: Optional[float]
    elo_away: Optional[float]
    reasoning: str
    game_date: str
    quantity: int = 1
    confidence: float = 0.5
    stop_loss: float = 0.0
    take_profit: float = 0.0


class SportsEdgeFinder:
    """
    Finds edge in Kalshi sports markets using Elo ratings.

    Matches ESPN games with FiveThirtyEight Elo ratings to
    calculate expected win probabilities and find mispriced markets.
    """

    MIN_EDGE = 0.05  # 5% minimum edge
    MIN_PRICE = 0.05
    MAX_PRICE = 0.95
    KALSHI_BUDGET = 150.0  # Total Kalshi bankroll
    MAX_PER_TRADE = 15.0   # Max capital allocation per trade
    MAX_CONTRACTS_PER_TRADE = 3  # Cap on contracts per position

    # Kalshi sports series tickers (verified from API)
    SPORTS_SERIES = {
        'NFL': ['KXNFLGAME', 'KXNFL1QWINNER', 'KXNFLTEAMTOTAL'],
        'NBA': ['KXNBAGAME', 'KXNBATEAMTOTAL', 'KXWNBAGAME'],
        'MLB': ['KXMLBGAME', 'KXMLBSERIESGAMETOTAL'],
    }

    # Market statuses that we can trade
    TRADEABLE_STATUSES = ['active', 'open', 'trading']

    def __init__(self):
        """Initialize the sports edge finder"""
        self.client = KalshiClient()
        self.scanner = MarketScanner(client=self.client)
        self.scraper = SportsScraper()

        logger.info("SportsEdgeFinder initialized")

    @staticmethod
    def _calc_confidence(edge: float) -> float:
        """Map edge magnitude to confidence score."""
        abs_edge = abs(edge)
        if abs_edge >= 0.20:
            return 0.9
        elif abs_edge >= 0.10:
            return 0.7
        else:
            return 0.5  # 5-10% edge (below 5% is filtered by MIN_EDGE)

    def _calc_quantity(self, market_price: float) -> int:
        """Calculate contracts to buy based on capital allocation."""
        price_dollars = market_price  # market_price is already 0-1 representing dollars
        if price_dollars <= 0:
            return 1
        quantity = max(1, int(self.MAX_PER_TRADE / price_dollars))
        return min(quantity, self.MAX_CONTRACTS_PER_TRADE)

    @staticmethod
    def _calc_stop_loss(entry_price: float) -> float:
        """Calculate stop loss for a prediction market position."""
        return round(entry_price * 0.5, 4)

    @staticmethod
    def _calc_take_profit(entry_price: float) -> float:
        """Calculate take profit for a prediction market position."""
        return round(min(0.95, entry_price * 1.5), 4)

    def find_sports_markets(self) -> List[Dict]:
        """Find all active sports markets on Kalshi"""
        all_markets = []

        for league, series_list in self.SPORTS_SERIES.items():
            for series_ticker in series_list:
                try:
                    # Get markets directly by series ticker
                    data = self.client._request('GET', '/markets', params={
                        'series_ticker': series_ticker,
                        'limit': 100
                    })
                    markets = data.get('markets', [])

                    for market in markets:
                        # Only include tradeable markets
                        status = market.get('status', '').lower()
                        if status in self.TRADEABLE_STATUSES or status == 'initialized':
                            market['league'] = league
                            all_markets.append(market)

                except Exception as e:
                    logger.debug(f"Error fetching {series_ticker}: {e}")

        logger.info(f"Found {len(all_markets)} sports markets")
        return all_markets

    def parse_market_teams(self, market: Dict) -> Optional[Dict]:
        """
        Parse team information from market title/ticker.

        Args:
            market: Kalshi market dict

        Returns:
            Dict with home_team, away_team, league or None
        """
        ticker = market.get('ticker', '').upper()
        title = market.get('title', '')

        # Try to extract teams from title
        # Format: "Will X beat Y?" or "X vs Y"
        import re

        # Pattern: "Team A" vs/at/@ "Team B"
        vs_pattern = r'(.+?)\s+(?:vs\.?|at|@)\s+(.+?)(?:\s+on|\s+\?|$)'
        match = re.search(vs_pattern, title, re.IGNORECASE)

        if match:
            team1 = match.group(1).strip()
            team2 = match.group(2).strip()

            # Determine home/away (team after "at" or "@" is home)
            if ' at ' in title.lower() or ' @ ' in title:
                return {
                    'away_team': team1,
                    'home_team': team2,
                    'league': market.get('league', 'unknown')
                }
            else:
                return {
                    'home_team': team1,
                    'away_team': team2,
                    'league': market.get('league', 'unknown')
                }

        return None

    def match_market_to_game(
        self,
        market: Dict,
        games: List[GameData]
    ) -> Optional[GameData]:
        """
        Match a Kalshi market to an ESPN game.

        Args:
            market: Kalshi market dict
            games: List of GameData from scraper

        Returns:
            Matched GameData or None
        """
        parsed = self.parse_market_teams(market)
        if not parsed:
            return None

        home_team = parsed['home_team'].lower()
        away_team = parsed['away_team'].lower()

        for game in games:
            game_home = game.home_team.lower()
            game_away = game.away_team.lower()

            # Check for team name match (partial matching for flexibility)
            home_match = (
                home_team in game_home or
                game_home in home_team or
                game.home_team_abbr.lower() in home_team
            )
            away_match = (
                away_team in game_away or
                game_away in away_team or
                game.away_team_abbr.lower() in away_team
            )

            if home_match and away_match:
                return game

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

            # Get best bid for YES side
            best_yes_bid = 0
            if yes_orders:
                prices = [order[0] for order in yes_orders if len(order) >= 2 and order[1] > 0]
                if prices:
                    best_yes_bid = max(prices)

            # Get best bid for NO side to calculate YES ask
            best_no_bid = 0
            if no_orders:
                prices = [order[0] for order in no_orders if len(order) >= 2 and order[1] > 0]
                if prices:
                    best_no_bid = max(prices)

            # Market price is midpoint of YES bid and YES ask (100 - NO bid)
            yes_ask = 100 - best_no_bid if best_no_bid > 0 else 100

            if best_yes_bid > 0:
                midpoint = (best_yes_bid + yes_ask) / 200
                return midpoint
            elif yes_ask < 100:
                return yes_ask / 100

            return None

        except Exception as e:
            logger.debug(f"Error getting price for {market.get('ticker')}: {e}")
            return None

    def find_opportunities(self) -> List[SportsOpportunity]:
        """
        Find sports trading opportunities.

        Returns:
            List of SportsOpportunity with positive edge
        """
        opportunities = []

        # Get games with Elo probabilities
        all_games = self.scraper.get_all_upcoming_games()
        all_games_flat = []
        for games in all_games.values():
            all_games_flat.extend(games)

        logger.info(f"Got {len(all_games_flat)} upcoming games with Elo data")

        # Get Kalshi sports markets
        markets = self.find_sports_markets()
        logger.info(f"Scanning {len(markets)} sports markets")

        for market in markets:
            ticker = market.get('ticker', '')
            title = market.get('title', '')

            # Match to game
            game = self.match_market_to_game(market, all_games_flat)
            if not game or game.home_win_prob is None:
                continue

            # Get market price
            market_price = self.get_market_price(market)
            if market_price is None:
                continue

            if market_price < self.MIN_PRICE or market_price > self.MAX_PRICE:
                continue

            # Determine if market is for home or away team
            our_prob = game.home_win_prob  # Default: assume YES = home wins

            # Check if title suggests away team
            if game.away_team.lower() in title.lower() and 'win' in title.lower():
                our_prob = 1 - game.home_win_prob

            # Calculate edge
            edge_yes = our_prob - market_price
            edge_no = (1 - our_prob) - (1 - market_price)

            # Find best side
            if edge_yes >= self.MIN_EDGE:
                opportunities.append(SportsOpportunity(
                    ticker=ticker,
                    title=title,
                    league=game.league,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    side='YES',
                    our_probability=our_prob,
                    market_price=market_price,
                    edge=edge_yes,
                    elo_home=game.home_elo,
                    elo_away=game.away_elo,
                    reasoning=f"Elo: {game.home_team} {game.home_elo:.0f} vs {game.away_team} {game.away_elo:.0f}",
                    game_date=game.game_date,
                    quantity=self._calc_quantity(market_price),
                    confidence=self._calc_confidence(edge_yes),
                    stop_loss=self._calc_stop_loss(market_price),
                    take_profit=self._calc_take_profit(market_price),
                ))
            elif edge_no >= self.MIN_EDGE:
                no_price = 1 - market_price
                opportunities.append(SportsOpportunity(
                    ticker=ticker,
                    title=title,
                    league=game.league,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    side='NO',
                    our_probability=1 - our_prob,
                    market_price=no_price,
                    edge=edge_no,
                    elo_home=game.home_elo,
                    elo_away=game.away_elo,
                    reasoning=f"Elo: {game.home_team} {game.home_elo:.0f} vs {game.away_team} {game.away_elo:.0f}",
                    game_date=game.game_date,
                    quantity=self._calc_quantity(no_price),
                    confidence=self._calc_confidence(edge_no),
                    stop_loss=self._calc_stop_loss(no_price),
                    take_profit=self._calc_take_profit(no_price),
                ))

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge, reverse=True)
        logger.info(f"Found {len(opportunities)} sports opportunities with edge >= {self.MIN_EDGE:.0%}")

        return opportunities

    def print_opportunities(self, opportunities: List[SportsOpportunity]):
        """Print opportunities in formatted output"""
        print()
        print("=" * 70)
        print("SPORTS OPPORTUNITIES (Elo-based)")
        print("=" * 70)

        if not opportunities:
            print("No opportunities found.")
            return

        for i, opp in enumerate(opportunities[:10], 1):
            print(f"\n{i}. [{opp.league}] {opp.ticker}")
            print(f"   {opp.away_team} @ {opp.home_team}")
            print(f"   Action: BUY {opp.side} @ {opp.market_price:.0%}")
            print(f"   Our Probability: {opp.our_probability:.0%}")
            print(f"   Edge: {opp.edge:.1%}")
            print(f"   Quantity: {opp.quantity} contracts | Confidence: {opp.confidence}")
            print(f"   Stop Loss: {opp.stop_loss:.2%} | Take Profit: {opp.take_profit:.2%}")
            print(f"   {opp.reasoning}")

        print()
        print("=" * 70)


def main():
    """Run the sports edge finder"""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("SPORTS EDGE FINDER")
    print("=" * 60)

    finder = SportsEdgeFinder()

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
        print(f"Best opportunity: {opportunities[0].ticker} ({opportunities[0].edge:.1%} edge)")


if __name__ == "__main__":
    main()
