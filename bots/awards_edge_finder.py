"""
Awards Edge Finder

Matches award show predictions to Kalshi entertainment markets.
Uses Gold Derby predictions and precursor results.

Markets covered:
- Oscars (KXOSCAR*)
- Golden Globes (KXGG*)
- Emmy Awards (KXEMMY*)
- BAFTA (KXBAFTA*)

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
from scrapers.awards_scraper import AwardsScraper, AwardPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AwardsEdgeFinder')


@dataclass
class AwardsOpportunity:
    """Awards market trading opportunity"""
    ticker: str
    title: str
    award_show: str
    category: str
    nominee: str
    side: str  # YES or NO
    our_probability: float
    market_price: float
    edge: float
    confidence: str
    reasoning: str


class AwardsEdgeFinder:
    """
    Finds edge in Kalshi entertainment/awards markets.

    Uses Gold Derby predictions and precursor results.
    """

    MIN_EDGE = 0.08  # 8% minimum edge (higher for entertainment)
    MIN_PRICE = 0.05
    MAX_PRICE = 0.95

    # Kalshi award series tickers
    AWARD_SERIES = {
        'OSCAR': [
            'KXOSCARPIC', 'KXOSCARDIR', 'KXOSCARACTR', 'KXOSCARACTRSS',
            'KXOSCARSUPACTR', 'KXOSCARSUPACTRSS', 'KXOSCARANIM',
            'KXOSCARINTL', 'KXOSCARDOC', 'KXOSCAROGSCREEN', 'KXOSCARADSCREEN',
            'KXOSCARCOUNT', 'KXOSCARCOSTUME', 'KXOSCARFILM', 'KXOSCARSCORE',
        ],
        'GOLDEN_GLOBE': [
            'KXGGDRAMAFILM', 'KXGGMCOMFILM', 'KXGGDRAMAACTR', 'KXGGDRAMAACTRSS',
            'KXGGCOMACTR', 'KXGGCOMACTRSS', 'KXGGANIFILM', 'KXGGDIR',
        ],
        'EMMY': [
            'KXEMMYDSERIES', 'KXEMMYCSERIES', 'KXEMMYLSERIES',
            'KXEMMYDACTR', 'KXEMMYDACTRSS', 'KXEMMYCACTR', 'KXEMMYCACTRSS',
        ],
        'BAFTA': [
            'KXBAFTAFILM', 'KXBAFTADIR', 'KXBAFTAACTR', 'KXBAFTAACTRSS',
        ],
    }

    # Category name mappings (market title -> prediction category)
    CATEGORY_MAPPINGS = {
        'best picture': 'Best Picture',
        'best director': 'Best Director',
        'best actor': 'Best Actor',
        'lead actor': 'Best Actor',
        'best actress': 'Best Actress',
        'lead actress': 'Best Actress',
        'supporting actor': 'Best Supporting Actor',
        'supporting actress': 'Best Supporting Actress',
        'animated': 'Best Animated Feature',
        'international': 'Best International Feature',
        'documentary': 'Best Documentary',
        'original screenplay': 'Best Original Screenplay',
        'adapted screenplay': 'Best Adapted Screenplay',
        'drama series': 'Best Drama Series',
        'comedy series': 'Best Comedy Series',
        'limited series': 'Best Limited Series',
        'drama film': 'Best Drama Film',
        'comedy film': 'Best Comedy/Musical Film',
        'musical film': 'Best Comedy/Musical Film',
    }

    TRADEABLE_STATUSES = ['active', 'open', 'trading']

    def __init__(self):
        """Initialize the awards edge finder"""
        self.client = KalshiClient()
        self.scraper = AwardsScraper()

        # Cache predictions
        self._predictions_cache: Dict[str, List[AwardPrediction]] = {}

        logger.info("AwardsEdgeFinder initialized")

    def find_award_markets(self) -> Dict[str, List[Dict]]:
        """Find all active award markets on Kalshi"""
        markets_by_show = {show: [] for show in self.AWARD_SERIES.keys()}

        for show, series_list in self.AWARD_SERIES.items():
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
                            market['award_show'] = show
                            markets_by_show[show].append(market)

                except Exception as e:
                    logger.debug(f"Error fetching {series_ticker}: {e}")

        total = sum(len(m) for m in markets_by_show.values())
        logger.info(f"Found {total} award markets (Oscar: {len(markets_by_show['OSCAR'])}, "
                   f"GG: {len(markets_by_show['GOLDEN_GLOBE'])}, Emmy: {len(markets_by_show['EMMY'])})")

        return markets_by_show

    def parse_award_market(self, market: Dict) -> Optional[Dict]:
        """
        Parse award market to extract category and nominee.

        Examples:
        - KXOSCARACTR-BRODY -> Best Actor, Adrien Brody
        - KXGGDRAMAFILM-BRUTALIST -> Best Drama Film, The Brutalist

        Args:
            market: Kalshi market dict

        Returns:
            Dict with category, nominee or None
        """
        ticker = market.get('ticker', '').upper()
        title = market.get('title', '').lower()
        subtitle = market.get('subtitle', '').lower()

        # Try to extract nominee from ticker
        # Pattern: KXOSCAR*-NOMINEENAME
        ticker_pattern = r'-([A-Z]+)$'
        match = re.search(ticker_pattern, ticker)

        nominee_hint = None
        if match:
            nominee_hint = match.group(1)

        # Try to determine category from series ticker
        category = None
        for cat_key, cat_name in self.CATEGORY_MAPPINGS.items():
            if cat_key in title or cat_key in ticker.lower():
                category = cat_name
                break

        # Extract nominee from title
        # "Will Adrien Brody win Best Actor?"
        nominee = None

        # Pattern 1: "Will X win"
        will_win = re.search(r'will\s+(.+?)\s+win', title)
        if will_win:
            nominee = will_win.group(1).strip()
            # Clean up nominee name
            nominee = re.sub(r'\s+for\s+.*', '', nominee)
            nominee = re.sub(r'\s+at\s+.*', '', nominee)

        # Pattern 2: "X to win"
        to_win = re.search(r'(.+?)\s+to\s+win', title)
        if not nominee and to_win:
            nominee = to_win.group(1).strip()

        # Pattern 3: Subtitle often has nominee
        if not nominee and subtitle:
            # Subtitle might be just the nominee name
            nominee = subtitle.title()

        # Use ticker hint as fallback
        if not nominee and nominee_hint:
            # Map common abbreviations
            abbr_map = {
                'BRODY': 'Adrien Brody',
                'CHALAMET': 'Timothée Chalamet',
                'FIENNES': 'Ralph Fiennes',
                'MOORE': 'Demi Moore',
                'MADISON': 'Mikey Madison',
                'CULKIN': 'Kieran Culkin',
                'SALDANA': 'Zoe Saldaña',
                'GRANDE': 'Ariana Grande',
                'ANORA': 'Anora',
                'BRUTALIST': 'The Brutalist',
                'CONCLAVE': 'Conclave',
                'WICKED': 'Wicked',
                'EMILIA': 'Emilia Pérez',
                'SHOGUN': 'Shōgun',
                'HACKS': 'Hacks',
                'BEAR': 'The Bear',
            }
            nominee = abbr_map.get(nominee_hint, nominee_hint.title())

        if category and nominee:
            return {
                'category': category,
                'nominee': nominee
            }

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

    def get_predictions(self, award_show: str) -> List[AwardPrediction]:
        """Get cached or fresh predictions"""
        if award_show not in self._predictions_cache:
            self._predictions_cache[award_show] = self.scraper.get_all_predictions(award_show)
        return self._predictions_cache[award_show]

    def find_opportunities(self) -> List[AwardsOpportunity]:
        """
        Find awards trading opportunities.

        Returns:
            List of AwardsOpportunity with positive edge
        """
        opportunities = []

        # Fetch predictions for each award show
        logger.info("Fetching award predictions...")
        for show in self.AWARD_SERIES.keys():
            predictions = self.get_predictions(show)
            logger.info(f"  {show}: {len(predictions)} categories")

        # Find markets
        markets_by_show = self.find_award_markets()

        # Process each market
        for show, markets in markets_by_show.items():
            predictions = self.get_predictions(show)

            for market in markets:
                opp = self._evaluate_award_market(market, show, predictions)
                if opp:
                    opportunities.append(opp)

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge, reverse=True)
        logger.info(f"Found {len(opportunities)} awards opportunities with edge >= {self.MIN_EDGE:.0%}")

        return opportunities

    def _evaluate_award_market(
        self,
        market: Dict,
        award_show: str,
        predictions: List[AwardPrediction]
    ) -> Optional[AwardsOpportunity]:
        """Evaluate an award market against predictions"""
        # Parse market
        parsed = self.parse_award_market(market)
        if not parsed:
            return None

        category = parsed['category']
        nominee = parsed['nominee']

        # Find matching prediction
        prediction = None
        for pred in predictions:
            if self._categories_match(pred.category, category):
                prediction = pred
                break

        if not prediction:
            return None

        # Find nominee probability
        our_prob = None
        matched_name = None

        for name, prob in prediction.all_probabilities.items():
            if self._names_match(name, nominee):
                our_prob = prob
                matched_name = name
                break

        if our_prob is None:
            # Nominee not in our predictions - skip
            return None

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

            return AwardsOpportunity(
                ticker=market.get('ticker', ''),
                title=market.get('title', ''),
                award_show=award_show,
                category=category,
                nominee=matched_name or nominee,
                side=side,
                our_probability=our_prob,
                market_price=market_price,
                edge=edge,
                confidence=prediction.confidence,
                reasoning=f"Gold Derby: {prediction.reasoning}"
            )

        return None

    def _categories_match(self, cat1: str, cat2: str) -> bool:
        """Check if two category names match"""
        c1 = cat1.lower().replace('best ', '').strip()
        c2 = cat2.lower().replace('best ', '').strip()
        return c1 == c2 or c1 in c2 or c2 in c1

    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two names match (fuzzy)"""
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()

        if n1 == n2:
            return True
        if n1 in n2 or n2 in n1:
            return True

        # First word match
        w1 = n1.split()[0] if n1 else ''
        w2 = n2.split()[0] if n2 else ''
        if w1 and w2 and len(w1) > 3 and w1 == w2:
            return True

        # Last name match (for actors)
        l1 = n1.split()[-1] if n1 else ''
        l2 = n2.split()[-1] if n2 else ''
        if l1 and l2 and len(l1) > 3 and l1 == l2:
            return True

        return False

    def print_opportunities(self, opportunities: List[AwardsOpportunity]):
        """Print opportunities in formatted output"""
        print()
        print("=" * 70)
        print("AWARDS OPPORTUNITIES")
        print("=" * 70)

        if not opportunities:
            print("No opportunities found.")
            return

        for i, opp in enumerate(opportunities[:15], 1):
            print(f"\n{i}. [{opp.award_show}] {opp.ticker}")
            print(f"   {opp.title}")
            print(f"   Category: {opp.category}")
            print(f"   Nominee: {opp.nominee}")
            print(f"   Action: BUY {opp.side} @ {opp.market_price:.0%}")
            print(f"   Our Probability: {opp.our_probability:.0%}")
            print(f"   Edge: {opp.edge:.1%} [{opp.confidence}]")
            print(f"   Source: {opp.reasoning}")

        print()
        print("=" * 70)


def main():
    """Run the awards edge finder"""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("AWARDS EDGE FINDER")
    print("=" * 60)

    finder = AwardsEdgeFinder()

    print("\n[1] Scanning for opportunities...")
    opportunities = finder.find_opportunities()

    print("\n[2] Results:")
    finder.print_opportunities(opportunities)

    print("\n[3] Summary:")
    print("-" * 40)
    print(f"Total opportunities: {len(opportunities)}")

    if opportunities:
        by_show = {}
        for opp in opportunities:
            by_show[opp.award_show] = by_show.get(opp.award_show, 0) + 1

        for show, count in sorted(by_show.items()):
            print(f"  {show}: {count}")

        high_conf = [o for o in opportunities if o.confidence == 'HIGH']
        print(f"\nHigh confidence: {len(high_conf)}")

        avg_edge = sum(o.edge for o in opportunities) / len(opportunities)
        print(f"Average edge: {avg_edge:.1%}")


if __name__ == "__main__":
    main()
