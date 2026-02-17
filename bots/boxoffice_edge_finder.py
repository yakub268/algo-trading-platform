"""
Box Office Edge Finder

Matches box office probability estimates to Kalshi movie markets.
Identifies edge opportunities based on opening weekend predictions.

Author: Trading Bot
Created: January 2026
"""

import os
import sys
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient
from bots.market_scanner import MarketScanner
from scrapers.boxoffice_scraper import BoxOfficeScraper, MovieData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BoxOfficeEdgeFinder')


@dataclass
class BoxOfficeOpportunity:
    """Box office trading opportunity"""
    ticker: str
    title: str
    movie_title: str
    threshold: float  # In millions
    side: str  # YES or NO
    our_probability: float
    market_price: float
    edge: float
    estimate_low: float
    estimate_high: float
    rt_score: Optional[int]
    reasoning: str


class BoxOfficeEdgeFinder:
    """
    Finds edge in Kalshi box office markets.

    Uses opening weekend estimates based on:
    - Budget
    - Franchise
    - Rotten Tomatoes scores
    - Historical patterns
    """

    MIN_EDGE = 0.05  # 5% minimum edge
    MIN_PRICE = 0.05
    MAX_PRICE = 0.95

    # Kalshi box office/movie/award series tickers (verified from API)
    BOX_OFFICE_SERIES = [
        'KXNETFLIXRANKMOVIEGLOBAL',  # Netflix movie rankings
        'KXNETFLIXMOVIERANKD',       # Netflix daily rankings
        'KXOSCARACTR',               # Oscar Best Actress
        'KXOSCARCOUNT',              # Oscar nomination counts
        'KXOSCARCOSTUME',            # Oscar Best Costume
        'KXGGDRAMAFILM',             # Golden Globe Drama
        'KXGGMCOMFILM',              # Golden Globe Musical/Comedy
        'KXEMMYDSERIES',             # Emmy Drama Series
        'KXEMMYCSERIES',             # Emmy Comedy Series
        'KXBAFTAFILM',               # BAFTA Best Film
        'KXRTCOMPARISON',            # Rotten Tomatoes comparisons
    ]

    # Market statuses that we can trade
    TRADEABLE_STATUSES = ['active', 'open', 'trading']

    def __init__(self):
        """Initialize the box office edge finder"""
        self.client = KalshiClient()
        self.scanner = MarketScanner(client=self.client)
        self.scraper = BoxOfficeScraper()

        logger.info("BoxOfficeEdgeFinder initialized")

    def find_box_office_markets(self) -> List[Dict]:
        """Find all active box office/movie/award markets on Kalshi"""
        all_markets = []

        for series_ticker in self.BOX_OFFICE_SERIES:
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
                    if status in self.TRADEABLE_STATUSES:
                        all_markets.append(market)

            except Exception as e:
                logger.debug(f"Error fetching {series_ticker}: {e}")

        # Deduplicate
        seen = set()
        unique_markets = []
        for market in all_markets:
            ticker = market.get('ticker')
            if ticker and ticker not in seen:
                seen.add(ticker)
                unique_markets.append(market)

        logger.info(f"Found {len(unique_markets)} box office/movie markets")
        return unique_markets

    def parse_market_movie(self, market: Dict) -> Optional[Dict]:
        """
        Parse movie information from market title/ticker.

        Args:
            market: Kalshi market dict

        Returns:
            Dict with movie_title, threshold or None
        """
        title = market.get('title', '')
        ticker = market.get('ticker', '').upper()

        # Pattern: "Will [Movie] gross over $XXM?"
        pattern1 = r"[Ww]ill\s+[\"']?(.+?)[\"']?\s+(?:gross|earn|make).*?\$(\d+)\s*[Mm]"
        match = re.search(pattern1, title)

        if match:
            return {
                'movie_title': match.group(1).strip(),
                'threshold': float(match.group(2))
            }

        # Pattern: "Opening weekend over $XXM"
        pattern2 = r"[\"']?(.+?)[\"']?\s+opening\s+weekend.*?(?:over|above|exceed).*?\$(\d+)"
        match = re.search(pattern2, title, re.IGNORECASE)

        if match:
            return {
                'movie_title': match.group(1).strip(),
                'threshold': float(match.group(2))
            }

        # Pattern from ticker: KXBOX-MOVIE-O100 (Over $100M)
        ticker_pattern = r'KXBOX.*?-O(\d+)'
        match = re.search(ticker_pattern, ticker)

        if match:
            threshold = float(match.group(1))
            # Try to extract movie name from title
            movie_name = re.sub(r'(?:gross|over|under|\$\d+[Mm]|million|\?).*', '', title).strip()
            movie_name = re.sub(r'^Will\s+', '', movie_name).strip()

            if movie_name:
                return {
                    'movie_title': movie_name,
                    'threshold': threshold
                }

        return None

    def match_market_to_movie(
        self,
        market: Dict,
        movies: List[MovieData]
    ) -> Optional[MovieData]:
        """
        Match a Kalshi market to a movie.

        Args:
            market: Kalshi market dict
            movies: List of MovieData from scraper

        Returns:
            Matched MovieData or None
        """
        parsed = self.parse_market_movie(market)
        if not parsed:
            return None

        market_movie = parsed['movie_title'].lower()

        for movie in movies:
            movie_title = movie.title.lower()

            # Check for title match (partial matching)
            if market_movie in movie_title or movie_title in market_movie:
                return movie

            # Try matching key words
            market_words = set(market_movie.split())
            movie_words = set(movie_title.split())

            overlap = len(market_words & movie_words)
            if overlap >= 2:  # At least 2 common words
                return movie

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

            # Get best bid for NO side
            best_no_bid = 0
            if no_orders:
                prices = [order[0] for order in no_orders if len(order) >= 2 and order[1] > 0]
                if prices:
                    best_no_bid = max(prices)

            # Market price is midpoint
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

    def find_opportunities(self) -> List[BoxOfficeOpportunity]:
        """
        Find box office trading opportunities.

        Returns:
            List of BoxOfficeOpportunity with positive edge
        """
        opportunities = []

        # Get movie data
        upcoming = self.scraper.fetch_upcoming_releases()
        weekend = self.scraper.fetch_bom_weekend()

        # Create MovieData objects
        movies = []
        for m in upcoming:
            movie_data = self.scraper.get_movie_data(
                m.get('title', ''),
                release_date=m.get('release_date'),
                franchise=m.get('franchise')
            )
            movies.append(movie_data)

        for m in weekend:
            movie_data = self.scraper.get_movie_data(
                m.get('title', ''),
                opening_weekend=m.get('weekend_gross'),
                total_gross=m.get('total_gross'),
                theaters=m.get('theaters')
            )
            movies.append(movie_data)

        logger.info(f"Got {len(movies)} movies for matching")

        # Get Kalshi box office markets
        markets = self.find_box_office_markets()
        logger.info(f"Scanning {len(markets)} box office markets")

        for market in markets:
            ticker = market.get('ticker', '')
            title = market.get('title', '')

            # Parse market info
            parsed = self.parse_market_movie(market)
            if not parsed:
                continue

            threshold = parsed['threshold']

            # Match to movie
            movie = self.match_market_to_movie(market, movies)
            if not movie:
                # Create a generic movie for the title
                movie = self.scraper.get_movie_data(parsed['movie_title'])

            # Get estimate
            estimate_low, estimate_high, reasoning = self.scraper.estimate_opening_weekend(
                movie.title,
                budget=movie.budget,
                franchise=movie.franchise,
                rt_score=movie.rt_score,
                theaters=movie.theaters
            )

            # Calculate probability
            our_prob = self.scraper.calculate_threshold_probability(
                estimate_low, estimate_high, threshold
            )

            # Get market price
            market_price = self.get_market_price(market)
            if market_price is None:
                continue

            if market_price < self.MIN_PRICE or market_price > self.MAX_PRICE:
                continue

            # Calculate edge
            edge_yes = our_prob - market_price
            edge_no = (1 - our_prob) - (1 - market_price)

            # Find best side
            if edge_yes >= self.MIN_EDGE:
                opportunities.append(BoxOfficeOpportunity(
                    ticker=ticker,
                    title=title,
                    movie_title=movie.title,
                    threshold=threshold,
                    side='YES',
                    our_probability=our_prob,
                    market_price=market_price,
                    edge=edge_yes,
                    estimate_low=estimate_low,
                    estimate_high=estimate_high,
                    rt_score=movie.rt_score,
                    reasoning=reasoning
                ))
            elif edge_no >= self.MIN_EDGE:
                opportunities.append(BoxOfficeOpportunity(
                    ticker=ticker,
                    title=title,
                    movie_title=movie.title,
                    threshold=threshold,
                    side='NO',
                    our_probability=1 - our_prob,
                    market_price=1 - market_price,
                    edge=edge_no,
                    estimate_low=estimate_low,
                    estimate_high=estimate_high,
                    rt_score=movie.rt_score,
                    reasoning=reasoning
                ))

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge, reverse=True)
        logger.info(f"Found {len(opportunities)} box office opportunities with edge >= {self.MIN_EDGE:.0%}")

        return opportunities

    def print_opportunities(self, opportunities: List[BoxOfficeOpportunity]):
        """Print opportunities in formatted output"""
        print()
        print("=" * 70)
        print("BOX OFFICE OPPORTUNITIES")
        print("=" * 70)

        if not opportunities:
            print("No opportunities found.")
            return

        for i, opp in enumerate(opportunities[:10], 1):
            print(f"\n{i}. {opp.ticker}")
            print(f"   Movie: {opp.movie_title}")
            print(f"   Threshold: ${opp.threshold:.0f}M opening weekend")
            print(f"   Action: BUY {opp.side} @ {opp.market_price:.0%}")
            print(f"   Our Probability: {opp.our_probability:.0%}")
            print(f"   Edge: {opp.edge:.1%}")
            print(f"   Estimate: ${opp.estimate_low:.0f}M - ${opp.estimate_high:.0f}M")
            if opp.rt_score:
                print(f"   RT Score: {opp.rt_score}%")
            print(f"   {opp.reasoning}")

        print()
        print("=" * 70)


def main():
    """Run the box office edge finder"""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("BOX OFFICE EDGE FINDER")
    print("=" * 60)

    finder = BoxOfficeEdgeFinder()

    print("\n[1] Scanning for opportunities...")
    opportunities = finder.find_opportunities()

    print("\n[2] Results:")
    finder.print_opportunities(opportunities)

    print("\n[3] Summary:")
    print("-" * 40)
    print(f"Total opportunities: {len(opportunities)}")

    if opportunities:
        avg_edge = sum(o.edge for o in opportunities) / len(opportunities)
        print(f"Average edge: {avg_edge:.1%}")
        print(f"Best opportunity: {opportunities[0].ticker} ({opportunities[0].edge:.1%} edge)")


if __name__ == "__main__":
    main()
