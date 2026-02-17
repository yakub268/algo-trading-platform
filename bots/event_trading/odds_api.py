"""
Odds API Integration
Connects to The Odds API for cross-platform sportsbook odds comparison.
Compares implied probabilities from DraftKings, FanDuel, BetMGM, etc.
against Kalshi prices to detect arbitrage opportunities.

Free tier: 500 requests/month.

All timestamps in Mountain Time (America/Denver).
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import requests

from dotenv import load_dotenv

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

MT = ZoneInfo("America/Denver")

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# Supported bookmakers
BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet"]

logger = logging.getLogger("EventEdge.OddsAPI")


def mt_now() -> datetime:
    return datetime.now(MT)


def mt_str(dt: datetime = None) -> str:
    return (dt or mt_now()).strftime("%Y-%m-%d %H:%M:%S MT")


def _retry_request(method: str, url: str, **kwargs) -> requests.Response:
    """HTTP request with 3-retry exponential backoff."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.request(method, url, timeout=15, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == RETRY_ATTEMPTS:
                raise
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning(f"Request attempt {attempt} to {url} failed: {e}. Retry in {wait}s")
            time.sleep(wait)


@dataclass
class OddsSignal:
    """Cross-platform odds signal compatible with EdgeSignal pipeline."""
    market_id: str           # Kalshi ticker
    book: str                # e.g. "draftkings"
    book_implied_prob: float
    kalshi_price: float      # 0-1
    edge: float              # book_implied - kalshi
    sport: str
    event_name: str
    home_team: str = ""
    away_team: str = ""
    market_type: str = "h2h" # h2h, spreads, totals
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = mt_str()

    def to_dict(self) -> Dict:
        return {
            "market_id": self.market_id,
            "book": self.book,
            "book_implied_prob": round(self.book_implied_prob, 4),
            "kalshi_price": round(self.kalshi_price, 4),
            "edge": round(self.edge, 4),
            "sport": self.sport,
            "event_name": self.event_name,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "market_type": self.market_type,
            "timestamp": self.timestamp,
        }


class OddsAPI:
    """Interface to The Odds API for sportsbook odds retrieval."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or ODDS_API_KEY
        if not self.api_key:
            logger.warning("No ODDS_API_KEY set. Set env var or pass api_key.")
        self.remaining_requests: Optional[int] = None

    def get_sports(self) -> List[Dict]:
        """Get list of available sports."""
        if not self.api_key:
            return []
        try:
            resp = _retry_request("GET", f"{ODDS_API_BASE}/sports", params={
                "apiKey": self.api_key,
            })
            self.remaining_requests = int(resp.headers.get("x-requests-remaining", -1))
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch sports: {e}")
            return []

    def get_odds(self, sport: str, regions: str = "us",
                 markets: str = "h2h", bookmakers: str = None) -> List[Dict]:
        """
        Fetch odds for a sport from The Odds API.

        Args:
            sport: Sport key (e.g. "basketball_ncaab", "americanfootball_nfl")
            regions: Comma-separated regions (default "us")
            markets: Comma-separated market types ("h2h", "spreads", "totals")
            bookmakers: Comma-separated bookmaker keys (default: all US books)

        Returns:
            List of event dicts with nested bookmaker odds
        """
        if not self.api_key:
            logger.error("No API key configured")
            return []

        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers

        try:
            resp = _retry_request("GET", f"{ODDS_API_BASE}/sports/{sport}/odds", params=params)
            self.remaining_requests = int(resp.headers.get("x-requests-remaining", -1))
            logger.info(f"Odds API: {len(resp.json())} events, {self.remaining_requests} requests remaining")
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch odds for {sport}: {e}")
            return []

    @staticmethod
    def normalize_odds(american_odds: int) -> float:
        """
        Convert American odds to implied probability.

        +150 → 1 / (1 + 150/100) = 0.400
        -200 → 200 / (200 + 100) = 0.667
        """
        if american_odds > 0:
            return 100.0 / (american_odds + 100.0)
        elif american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100.0)
        else:
            return 0.5

    def extract_book_probabilities(self, events: List[Dict]) -> List[Dict]:
        """
        Extract implied probabilities per outcome per bookmaker.

        Returns list of dicts:
        {
            "event_name": str,
            "home_team": str,
            "away_team": str,
            "outcomes": {
                "Team A": {"draftkings": 0.55, "fanduel": 0.53, ...},
                "Team B": {"draftkings": 0.45, "fanduel": 0.47, ...},
            },
            "consensus": {"Team A": 0.54, "Team B": 0.46}
        }
        """
        results = []
        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            event_name = f"{away} @ {home}" if home and away else event.get("id", "unknown")

            outcomes: Dict[str, Dict[str, float]] = {}

            for bm in event.get("bookmakers", []):
                book_key = bm.get("key", "")
                for market in bm.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price", 0)
                        if name and price:
                            prob = self.normalize_odds(price)
                            if name not in outcomes:
                                outcomes[name] = {}
                            outcomes[name][book_key] = prob

            # Compute consensus (average across books)
            consensus = {}
            for name, book_probs in outcomes.items():
                if book_probs:
                    consensus[name] = sum(book_probs.values()) / len(book_probs)

            results.append({
                "event_name": event_name,
                "home_team": home,
                "away_team": away,
                "outcomes": outcomes,
                "consensus": consensus,
            })

        return results

    def find_arbs(self, kalshi_markets: List[Dict], sport: str = None,
                  events: List[Dict] = None) -> List[OddsSignal]:
        """
        Find arbitrage opportunities between sportsbook odds and Kalshi prices.

        Args:
            kalshi_markets: List of Kalshi market dicts (ticker, title, yes_price)
            sport: Sport key to fetch odds for (if events not provided)
            events: Pre-fetched events from get_odds()

        Returns:
            List of OddsSignal where sportsbook implied prob differs from Kalshi
        """
        if events is None and sport:
            events = self.get_odds(sport)
        if not events:
            return []

        probabilities = self.extract_book_probabilities(events)
        signals = []

        for kalshi_mkt in kalshi_markets:
            ticker = kalshi_mkt.get("ticker", "")
            title = (kalshi_mkt.get("title", "") or "").lower()
            yes_price = kalshi_mkt.get("yes_price") or kalshi_mkt.get("current_yes_price")
            if not ticker or yes_price is None:
                continue

            kalshi_prob = yes_price / 100.0 if yes_price > 1 else yes_price
            title_words = set(title.split())

            # Try to match Kalshi market to a sportsbook event
            for prob_data in probabilities:
                event_name = prob_data["event_name"].lower()
                home = prob_data["home_team"].lower()
                away = prob_data["away_team"].lower()

                # Fuzzy match: check if team names appear in Kalshi title
                match_score = 0
                for team in [home, away]:
                    team_words = set(team.split())
                    if team_words & title_words:
                        match_score += len(team_words & title_words)

                if match_score < 1:
                    continue

                # Compare each bookmaker's implied prob vs Kalshi
                for outcome_name, book_probs in prob_data["outcomes"].items():
                    # Check if this outcome maps to Kalshi YES
                    outcome_lower = outcome_name.lower()
                    outcome_in_title = any(w in title for w in outcome_lower.split())
                    if not outcome_in_title:
                        continue

                    for book, book_prob in book_probs.items():
                        edge = book_prob - kalshi_prob
                        if abs(edge) > 0.03:  # 3% minimum edge
                            signals.append(OddsSignal(
                                market_id=ticker,
                                book=book,
                                book_implied_prob=book_prob,
                                kalshi_price=kalshi_prob,
                                edge=edge,
                                sport=sport or "unknown",
                                event_name=prob_data["event_name"],
                                home_team=prob_data["home_team"],
                                away_team=prob_data["away_team"],
                            ))

        # Sort by absolute edge descending
        signals.sort(key=lambda s: abs(s.edge), reverse=True)
        return signals

    def find_best_odds(self, events: List[Dict]) -> List[Dict]:
        """
        Find the best available odds across all bookmakers for each outcome.
        Useful for finding soft lines and shopping for value.
        """
        results = []
        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            best: Dict[str, Dict] = {}  # outcome -> {book, price, prob}

            for bm in event.get("bookmakers", []):
                book_key = bm.get("key", "")
                for market in bm.get("markets", []):
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price", 0)
                        prob = self.normalize_odds(price)

                        if name not in best or prob > best[name]["prob"]:
                            best[name] = {
                                "book": book_key,
                                "american_odds": price,
                                "implied_prob": prob,
                            }

            if best:
                results.append({
                    "event": f"{away} @ {home}",
                    "best_odds": best,
                })

        return results
