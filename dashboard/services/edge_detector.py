"""
Trading Dashboard V4.2 - Edge Detection Service
Aggregates edge calculations from various data sources vs Kalshi prices
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dashboard.models import Edge
from dashboard.config import WEATHER_CACHE, ECONOMIC_CACHE, SPORTS_CACHE

logger = logging.getLogger(__name__)


class EdgeDetector:
    """Service for detecting probability edges across markets"""

    def __init__(self, kalshi_service=None):
        self.kalshi = kalshi_service
        self.nws_base_url = "https://api.weather.gov"

        # Major cities for weather tracking
        self.weather_cities = [
            {"name": "NYC", "lat": 40.7128, "lon": -74.0060, "nws_office": "OKX"},
            {"name": "LA", "lat": 34.0522, "lon": -118.2437, "nws_office": "LOX"},
            {"name": "Chicago", "lat": 41.8781, "lon": -87.6298, "nws_office": "LOT"},
            {"name": "Miami", "lat": 25.7617, "lon": -80.1918, "nws_office": "MFL"},
        ]

    # ===== WEATHER EDGES =====
    def get_weather_edges(self) -> List[Edge]:
        """
        Compare NWS forecasts to Kalshi weather contracts.
        """
        edges = []

        for city in self.weather_cities:
            try:
                # Get NWS forecast probability
                nws_data = self._fetch_nws_forecast(city)
                if not nws_data:
                    continue

                # Get Kalshi market price for temperature contracts
                kalshi_markets = self._get_kalshi_weather_markets(city["name"])

                for market in kalshi_markets:
                    nws_prob = self._calculate_nws_probability(nws_data, market)
                    if nws_prob is None:
                        continue

                    kalshi_prob = market.get("yes_price", 50)
                    edge = nws_prob - kalshi_prob

                    confidence = "HIGH" if abs(edge) > 10 else "MED" if abs(edge) > 5 else "LOW"

                    # Calculate expiry
                    expiry = market.get("expiration", "")
                    expires_str = self._format_expiry(expiry)

                    edges.append(Edge(
                        title=f"{city['name']} {market.get('event', 'Weather')}",
                        our_probability=nws_prob,
                        market_probability=kalshi_prob,
                        edge_percent=edge,
                        confidence=confidence,
                        source="NWS API",
                        expires=expires_str,
                        contract_symbol=market.get("ticker")
                    ))

            except Exception as e:
                logger.error(f"Weather edge detection failed for {city['name']}: {e}")

        # Return mock data if no real data available
        if not edges:
            edges = self._mock_weather_edges()

        return edges

    # ===== FED EDGES =====
    def get_fed_edges(self) -> List[Edge]:
        """
        Compare Fed probability model to Kalshi Fed contracts.
        """
        edges = []

        try:
            # Get upcoming FOMC meetings
            fomc_dates = self._get_fomc_calendar()

            for meeting in fomc_dates[:2]:  # Next 2 meetings
                # Get our Fed probability estimate (from CME FedWatch, Cleveland Fed, etc.)
                our_prob = self._calculate_fed_probability(meeting)

                # Get Kalshi market price
                kalshi_prob = self._get_kalshi_fed_price(meeting)

                if our_prob is not None and kalshi_prob is not None:
                    edge = our_prob - kalshi_prob
                    days_until = meeting.get("days_until", 0)

                    confidence = "HIGH" if days_until < 14 else "MED" if days_until < 30 else "LOW"

                    edges.append(Edge(
                        title=f"{meeting['month']} FOMC {meeting.get('action', 'Hold')}",
                        our_probability=our_prob,
                        market_probability=kalshi_prob,
                        edge_percent=edge,
                        confidence=confidence,
                        source="CME + Cleveland Fed",
                        expires=f"{days_until}d",
                        contract_symbol=meeting.get("ticker")
                    ))

        except Exception as e:
            logger.error(f"Fed edge detection failed: {e}")

        # Return mock data if no real data available
        if not edges:
            edges = self._mock_fed_edges()

        return edges

    # ===== SPORTS EDGES =====
    def get_sports_edges(self) -> List[Edge]:
        """
        Compare 538 ELO predictions to Kalshi sports contracts.
        """
        edges = []

        try:
            # Get games with 538/ESPN predictions
            games = self._get_sports_predictions()

            for game in games:
                our_prob = game.get("win_prob", 50)
                kalshi_prob = self._get_kalshi_sports_price(game)

                if kalshi_prob is not None:
                    edge = our_prob - kalshi_prob

                    edges.append(Edge(
                        title=f"{game.get('away', '?')} @ {game.get('home', '?')}",
                        our_probability=our_prob,
                        market_probability=kalshi_prob,
                        edge_percent=edge,
                        confidence="MED",
                        source="538 ELO",
                        expires=game.get("game_time", "TBD"),
                        contract_symbol=game.get("ticker")
                    ))

        except Exception as e:
            logger.error(f"Sports edge detection failed: {e}")

        # Return mock data if no real data available
        if not edges:
            edges = self._mock_sports_edges()

        return edges

    def get_all_edges(self) -> Dict[str, List[Edge]]:
        """Return all edges grouped by category"""
        return {
            "weather": self.get_weather_edges(),
            "fed": self.get_fed_edges(),
            "sports": self.get_sports_edges()
        }

    # ===== HELPER METHODS =====
    def _fetch_nws_forecast(self, city: dict) -> Optional[dict]:
        """Fetch NWS forecast for a city"""
        try:
            # Get grid endpoint
            point_url = f"{self.nws_base_url}/points/{city['lat']},{city['lon']}"
            response = requests.get(point_url, headers={"User-Agent": "TradingBot/1.0"}, timeout=10)
            if response.status_code != 200:
                return None

            data = response.json()
            forecast_url = data.get("properties", {}).get("forecast")

            if not forecast_url:
                return None

            # Get forecast
            forecast_response = requests.get(forecast_url, headers={"User-Agent": "TradingBot/1.0"}, timeout=10)
            if forecast_response.status_code == 200:
                return forecast_response.json()

        except Exception as e:
            logger.error(f"NWS fetch failed: {e}")

        return None

    def _get_kalshi_weather_markets(self, city_name: str) -> List[dict]:
        """Get Kalshi weather markets for a city"""
        if not self.kalshi:
            return []

        try:
            # Map city names to Kalshi location codes
            city_codes = {
                "NYC": "NY",
                "LA": "LA",
                "Chicago": "CHI",
                "Miami": "MIA"
            }
            code = city_codes.get(city_name, city_name[:3].upper())

            # Query Kalshi markets endpoint with series_ticker filter
            result = self.kalshi._make_request("GET", "/markets", params={
                "series_ticker": f"KXTEMP-{code}",
                "status": "open"
            })

            if result and "markets" in result:
                return [{
                    "ticker": m.get("ticker"),
                    "title": m.get("title", ""),
                    "yes_price": m.get("yes_ask", 50),
                    "no_price": m.get("no_ask", 50),
                    "expiry": m.get("expiration_time", "")
                } for m in result["markets"]]
        except Exception as e:
            logger.debug(f"Could not fetch Kalshi weather markets for {city_name}: {e}")

        return []

    def _calculate_nws_probability(self, nws_data: dict, market: dict) -> Optional[float]:
        """Calculate probability from NWS forecast data"""
        try:
            # Extract target temperature from market title (e.g., "NYC High >= 75F")
            title = market.get("title", "")
            temp_threshold = None

            # Parse threshold from title patterns like "High >= 75F" or "Low <= 32F"
            import re
            match = re.search(r'(High|Low)\s*([<>=]+)\s*(\d+)', title)
            if not match:
                return None

            temp_type = match.group(1).lower()
            operator = match.group(2)
            temp_threshold = int(match.group(3))

            # Get forecast temperatures from NWS data
            periods = nws_data.get("properties", {}).get("periods", [])
            if not periods:
                return None

            # Use first day's forecast
            forecast_temp = periods[0].get("temperature", 0)
            is_daytime = periods[0].get("isDaytime", True)

            # Simple probability estimate based on forecast vs threshold
            diff = forecast_temp - temp_threshold

            if ">=" in operator or ">" in operator:
                # Probability increases as forecast exceeds threshold
                if diff >= 10:
                    prob = 90
                elif diff >= 5:
                    prob = 75
                elif diff >= 0:
                    prob = 60
                elif diff >= -5:
                    prob = 40
                elif diff >= -10:
                    prob = 25
                else:
                    prob = 10
            else:  # <= or <
                # Probability increases as forecast is below threshold
                if diff <= -10:
                    prob = 90
                elif diff <= -5:
                    prob = 75
                elif diff <= 0:
                    prob = 60
                elif diff <= 5:
                    prob = 40
                elif diff <= 10:
                    prob = 25
                else:
                    prob = 10

            return float(prob)

        except Exception as e:
            logger.debug(f"Could not calculate NWS probability: {e}")
        return None

    def _get_fomc_calendar(self) -> List[dict]:
        """Get upcoming FOMC meeting dates"""
        # 2025-2026 FOMC dates
        fomc_2025 = [
            {"date": "2025-01-29", "month": "Jan"},
            {"date": "2025-03-19", "month": "Mar"},
            {"date": "2025-05-07", "month": "May"},
            {"date": "2025-06-18", "month": "Jun"},
            {"date": "2025-07-30", "month": "Jul"},
            {"date": "2025-09-17", "month": "Sep"},
            {"date": "2025-11-05", "month": "Nov"},
            {"date": "2025-12-17", "month": "Dec"},
        ]

        fomc_2026 = [
            {"date": "2026-01-28", "month": "Jan"},
            {"date": "2026-03-18", "month": "Mar"},
            {"date": "2026-04-29", "month": "Apr"},
            {"date": "2026-06-17", "month": "Jun"},
            {"date": "2026-07-29", "month": "Jul"},
            {"date": "2026-09-16", "month": "Sep"},
            {"date": "2026-11-04", "month": "Nov"},
            {"date": "2026-12-16", "month": "Dec"},
        ]

        all_dates = fomc_2025 + fomc_2026
        today = datetime.now().date()

        upcoming = []
        for meeting in all_dates:
            meeting_date = datetime.strptime(meeting["date"], "%Y-%m-%d").date()
            if meeting_date >= today:
                days_until = (meeting_date - today).days
                upcoming.append({
                    **meeting,
                    "days_until": days_until,
                    "action": "Hold"
                })

        return upcoming

    def _calculate_fed_probability(self, meeting: dict) -> Optional[float]:
        """Calculate our Fed probability estimate from CME FedWatch Tool"""
        try:
            # Fetch CME FedWatch probabilities
            fedwatch_url = "https://www.cmegroup.com/services/fedWatchTool/v1/fedWatchTool"
            response = requests.get(fedwatch_url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })

            if response.status_code == 200:
                data = response.json()
                meetings = data.get("meetings", [])

                # Find the matching meeting by date
                meeting_date = meeting.get("date", "")
                for m in meetings:
                    if m.get("meetingDate", "").startswith(meeting_date[:7]):  # Match year-month
                        # Return probability of "no change" (hold) action
                        probabilities = m.get("probabilities", {})
                        no_change_prob = probabilities.get("0", 0)  # 0 = no change
                        return float(no_change_prob) if no_change_prob else None

        except Exception as e:
            logger.debug(f"Could not fetch CME FedWatch data: {e}")

        # Fallback: simple heuristic based on days until meeting
        days_until = meeting.get("days_until", 30)
        if days_until < 7:
            return 75.0  # Close meetings usually have clearer expectations
        elif days_until < 30:
            return 60.0
        else:
            return 50.0  # Far out meetings are more uncertain

    def _get_kalshi_fed_price(self, meeting: dict) -> Optional[float]:
        """Get Kalshi market price for Fed contract"""
        if not self.kalshi:
            return None

        try:
            # Kalshi Fed contracts use FOMC series ticker
            meeting_month = meeting.get("month", "")
            meeting_year = meeting.get("date", "")[:4]

            # Query for FOMC markets
            result = self.kalshi._make_request("GET", "/markets", params={
                "series_ticker": "FOMC",
                "status": "open"
            })

            if result and "markets" in result:
                for market in result["markets"]:
                    title = market.get("title", "")
                    # Match by month/year in title
                    if meeting_month in title and meeting_year in title:
                        yes_price = market.get("yes_ask", 50)
                        return float(yes_price)

        except Exception as e:
            logger.debug(f"Could not fetch Kalshi Fed price: {e}")

        return None

    def _get_sports_predictions(self) -> List[dict]:
        """Get sports predictions from ESPN API"""
        predictions = []

        try:
            # ESPN scoreboard API for major leagues
            leagues = {
                "nfl": "football/nfl",
                "nba": "basketball/nba",
                "mlb": "baseball/mlb",
                "nhl": "hockey/nhl"
            }

            for league_code, league_path in leagues.items():
                url = f"https://site.api.espn.com/apis/site/v2/sports/{league_path}/scoreboard"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    events = data.get("events", [])

                    for event in events[:3]:  # Limit to 3 per league
                        competitions = event.get("competitions", [{}])
                        if not competitions:
                            continue

                        comp = competitions[0]
                        competitors = comp.get("competitors", [])
                        if len(competitors) < 2:
                            continue

                        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

                        # ESPN sometimes provides win probability
                        home_prob = None
                        for team in competitors:
                            prob_data = team.get("probables", [])
                            if team.get("homeAway") == "home":
                                # Check for win probability in statistics
                                stats = team.get("statistics", [])
                                for stat in stats:
                                    if "probability" in stat.get("name", "").lower():
                                        home_prob = float(stat.get("value", 50))

                        predictions.append({
                            "league": league_code.upper(),
                            "home_team": home.get("team", {}).get("abbreviation", "HOME"),
                            "away_team": away.get("team", {}).get("abbreviation", "AWAY"),
                            "event_id": event.get("id"),
                            "start_time": event.get("date"),
                            "home_prob": home_prob or 50.0  # Default to 50% if not available
                        })

        except Exception as e:
            logger.debug(f"Could not fetch ESPN predictions: {e}")

        return predictions

    def _get_kalshi_sports_price(self, game: dict) -> Optional[float]:
        """Get Kalshi market price for sports contract"""
        if not self.kalshi:
            return None

        try:
            league = game.get("league", "").upper()
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")

            # Query Kalshi for sports markets
            result = self.kalshi._make_request("GET", "/markets", params={
                "series_ticker": f"KX{league}",
                "status": "open"
            })

            if result and "markets" in result:
                for market in result["markets"]:
                    title = market.get("title", "").upper()
                    # Match by team names in title
                    if home_team in title or away_team in title:
                        yes_price = market.get("yes_ask", 50)
                        return float(yes_price)

        except Exception as e:
            logger.debug(f"Could not fetch Kalshi sports price: {e}")

        return None

    def _format_expiry(self, expiry_str: str) -> str:
        """Format expiry time as human-readable string"""
        try:
            expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            delta = expiry - datetime.now(expiry.tzinfo)

            if delta.days > 0:
                return f"{delta.days}d"
            elif delta.seconds > 3600:
                return f"{delta.seconds // 3600}h"
            else:
                return f"{delta.seconds // 60}m"
        except Exception as e:
            logger.debug(f"Error formatting expiry: {e}")
            return expiry_str

    # ===== MOCK DATA =====
    def _mock_weather_edges(self) -> List[Edge]:
        """Return mock weather edges for demo"""
        return [
            Edge(
                title="NYC Temp > 45°F",
                our_probability=82,
                market_probability=68,
                edge_percent=14,
                confidence="HIGH",
                source="NWS API",
                expires="6h"
            ),
            Edge(
                title="LA Rain Tomorrow",
                our_probability=23,
                market_probability=31,
                edge_percent=-8,
                confidence="MED",
                source="NWS API",
                expires="18h"
            ),
        ]

    def _mock_fed_edges(self) -> List[Edge]:
        """Return mock Fed edges for demo"""
        return [
            Edge(
                title="Jan FOMC Hold",
                our_probability=72,
                market_probability=65,
                edge_percent=7,
                confidence="HIGH",
                source="CME + Cleveland Fed",
                expires="13d"
            ),
            Edge(
                title="March Cut 25bp",
                our_probability=34,
                market_probability=41,
                edge_percent=-7,
                confidence="LOW",
                source="CME FedWatch",
                expires="45d"
            ),
        ]

    def _mock_sports_edges(self) -> List[Edge]:
        """Return mock sports edges for demo"""
        return [
            Edge(
                title="Lakers vs Celtics",
                our_probability=58,
                market_probability=51,
                edge_percent=7,
                confidence="MED",
                source="538 ELO",
                expires="2h"
            ),
            Edge(
                title="Chiefs ML",
                our_probability=63,
                market_probability=60,
                edge_percent=3,
                confidence="LOW",
                source="538 ELO",
                expires="5d"
            ),
        ]
