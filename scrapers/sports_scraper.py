"""
Sports Data Scraper

Fetches sports data for Kalshi sports contracts:
- ESPN API for NFL/NBA/MLB game schedules and odds
- FiveThirtyEight Elo ratings for win probability models
- Calculate win probabilities from historical models

Author: Trading Bot
Created: January 2026
"""

import os
import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import csv
import io
import calendar
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SportsScraper')


@dataclass
class GameData:
    """Sports game data"""
    game_id: str
    league: str  # NFL, NBA, MLB
    home_team: str
    away_team: str
    home_team_abbr: str
    away_team_abbr: str
    game_date: str
    game_time: Optional[str]
    home_score: Optional[int]
    away_score: Optional[int]
    status: str  # scheduled, in_progress, final
    venue: Optional[str]
    home_elo: Optional[float]
    away_elo: Optional[float]
    home_win_prob: Optional[float]
    source: str
    fetched_at: datetime


@dataclass
class SportsProbability:
    """Probability estimate for sports outcome"""
    game_id: str
    league: str
    ticker_pattern: str
    home_team: str
    away_team: str
    outcome: str  # home_win, away_win, over, under
    our_probability: float
    elo_based: bool
    reasoning: str
    confidence: str


# ESPN API endpoints (public, no key needed)
ESPN_API = {
    'NFL': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
    'NBA': 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',
    'MLB': 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard',
    'NHL': 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard',
}

# FiveThirtyEight Elo data URLs
FIVETHIRTYEIGHT_ELO = {
    'NFL': 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv',
    'NBA': 'https://projects.fivethirtyeight.com/nba-model/nba_elo.csv',
    'MLB': 'https://projects.fivethirtyeight.com/mlb-api/mlb_elo.csv',
}

# Team abbreviation mappings (ESPN to standard 3-letter Kalshi codes)
TEAM_ABBR_MAP = {
    # NFL
    'LAR': 'LAR', 'JAX': 'JAX', 'WSH': 'WAS',
    # NBA - Map to 3-letter codes for Kalshi
    'GS': 'GSW', 'NY': 'NYK', 'SA': 'SAS', 'NO': 'NOP',
    # MLB
    'CHW': 'CWS', 'SD': 'SDP', 'SF': 'SFG', 'TB': 'TBR', 'KC': 'KCR',
}

# Month abbreviations for Kalshi ticker format (uppercase)
MONTH_ABBR = {
    1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
    7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'
}


class SportsScraper:
    """
    Scrapes sports data from ESPN and FiveThirtyEight.

    Uses free public APIs - no API keys required.
    """

    CACHE_DURATION = timedelta(hours=2)

    def __init__(self, cache_dir: str = "data/sports_cache"):
        """Initialize the sports scraper"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Elo ratings cache
        self._elo_cache: Dict[str, Dict] = {}

        logger.info("SportsScraper initialized")

    def _get_cache_path(self, key: str) -> str:
        """Get cache file path"""
        safe_key = key.replace('/', '_').replace(':', '_')
        return os.path.join(self.cache_dir, f"{safe_key}.json")

    def _load_cache(self, key: str) -> Optional[Dict]:
        """Load from cache if valid"""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                cached_time = datetime.fromisoformat(data.get('cached_at', ''))
                if datetime.now(timezone.utc) - cached_time < self.CACHE_DURATION:
                    return data.get('data')
            except Exception as e:
                logger.debug(f"Cache load error for {key}: {e}")
        return None

    def _save_cache(self, key: str, data: any):
        """Save to cache"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'cached_at': datetime.now(timezone.utc).isoformat(),
                    'data': data
                }, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"Cache save error for {key}: {e}")

    def fetch_espn_games(self, league: str, dates: Optional[str] = None) -> List[Dict]:
        """
        Fetch games from ESPN API.

        Args:
            league: NFL, NBA, MLB, or NHL
            dates: Optional date string (YYYYMMDD)

        Returns:
            List of game dictionaries
        """
        if league not in ESPN_API:
            logger.error(f"Unknown league: {league}")
            return []

        cache_key = f"espn_{league}_{dates or 'today'}"
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        try:
            url = ESPN_API[league]
            params = {}
            if dates:
                params['dates'] = dates

            response = self.session.get(url, params=params, timeout=15)

            if response.status_code != 200:
                logger.warning(f"ESPN API returned {response.status_code} for {league}")
                return []

            data = response.json()
            games = []

            for event in data.get('events', []):
                competition = event.get('competitions', [{}])[0]
                competitors = competition.get('competitors', [])

                if len(competitors) < 2:
                    continue

                home = next((c for c in competitors if c.get('homeAway') == 'home'), competitors[0])
                away = next((c for c in competitors if c.get('homeAway') == 'away'), competitors[1])

                game = {
                    'game_id': event.get('id'),
                    'league': league,
                    'name': event.get('name'),
                    'date': event.get('date'),
                    'status': event.get('status', {}).get('type', {}).get('name', 'scheduled'),
                    'home_team': home.get('team', {}).get('displayName'),
                    'home_abbr': home.get('team', {}).get('abbreviation'),
                    'home_score': home.get('score'),
                    'away_team': away.get('team', {}).get('displayName'),
                    'away_abbr': away.get('team', {}).get('abbreviation'),
                    'away_score': away.get('score'),
                    'venue': competition.get('venue', {}).get('fullName'),
                    'odds': competition.get('odds', [{}])[0] if competition.get('odds') else {}
                }
                games.append(game)

            self._save_cache(cache_key, games)
            logger.info(f"[ESPN] Fetched {len(games)} {league} games")
            return games

        except Exception as e:
            logger.error(f"ESPN API error for {league}: {e}")
            return []

    def fetch_fivethirtyeight_elo(self, league: str) -> Dict[str, float]:
        """
        Get team Elo ratings.

        Note: FiveThirtyEight CSVs are no longer publicly available.
        Using default Elo ratings based on historical performance.

        Args:
            league: NFL, NBA, or MLB

        Returns:
            Dict mapping team abbreviation to current Elo rating
        """
        cache_key = f"elo_{league}"
        if cache_key in self._elo_cache:
            return self._elo_cache[cache_key]

        cached = self._load_cache(cache_key)
        if cached:
            self._elo_cache[cache_key] = cached
            return cached

        # Default Elo ratings based on typical team strengths
        # These would ideally be updated periodically from standings
        default_elos = {
            'NFL': {
                'KC': 1650, 'SF': 1620, 'BAL': 1600, 'DAL': 1580, 'DET': 1570,
                'PHI': 1560, 'BUF': 1550, 'MIA': 1540, 'CLE': 1530, 'CIN': 1520,
                'JAX': 1510, 'GB': 1505, 'PIT': 1500, 'LAR': 1495, 'SEA': 1490,
                'TB': 1485, 'MIN': 1480, 'ATL': 1475, 'NO': 1470, 'LAC': 1465,
                'DEN': 1460, 'IND': 1455, 'NYJ': 1450, 'TEN': 1445, 'CHI': 1440,
                'HOU': 1435, 'LV': 1430, 'WAS': 1425, 'NYG': 1420, 'CAR': 1415,
                'NE': 1410, 'ARI': 1405,
            },
            'NBA': {
                'BOS': 1700, 'DEN': 1680, 'MIL': 1650, 'PHX': 1620, 'LAC': 1600,
                'MIA': 1580, 'PHI': 1570, 'GSW': 1560, 'LAL': 1550, 'SAC': 1540,
                'NYK': 1530, 'CLE': 1520, 'MEM': 1510, 'BKN': 1500, 'ATL': 1490,
                'MIN': 1485, 'DAL': 1480, 'NOP': 1475, 'TOR': 1470, 'CHI': 1465,
                'OKC': 1460, 'IND': 1455, 'ORL': 1450, 'UTA': 1445, 'POR': 1440,
                'WAS': 1435, 'CHA': 1430, 'HOU': 1425, 'SAS': 1420, 'DET': 1415,
            },
            'MLB': {
                'ATL': 1580, 'LAD': 1570, 'HOU': 1560, 'TB': 1550, 'NYY': 1540,
                'PHI': 1530, 'SD': 1520, 'SEA': 1510, 'TOR': 1505, 'STL': 1500,
                'NYM': 1495, 'CLE': 1490, 'BAL': 1485, 'MIN': 1480, 'MIL': 1475,
                'TEX': 1470, 'SF': 1465, 'CHC': 1460, 'BOS': 1455, 'LAA': 1450,
                'ARI': 1445, 'MIA': 1440, 'DET': 1435, 'CIN': 1430, 'KC': 1425,
                'COL': 1420, 'PIT': 1415, 'CWS': 1410, 'OAK': 1405, 'WAS': 1400,
            },
            'NHL': {
                'BOS': 1600, 'CAR': 1580, 'NJ': 1560, 'COL': 1550, 'VGK': 1540,
                'DAL': 1530, 'TOR': 1520, 'NYR': 1510, 'EDM': 1505, 'MIN': 1500,
                'LA': 1495, 'SEA': 1490, 'WPG': 1485, 'TB': 1480, 'FLA': 1475,
            }
        }

        elo_ratings = default_elos.get(league, {})
        self._save_cache(cache_key, elo_ratings)
        self._elo_cache[cache_key] = elo_ratings
        logger.info(f"[ELO] Using default Elo ratings for {len(elo_ratings)} {league} teams")
        return elo_ratings

    def calculate_elo_win_probability(
        self,
        home_elo: float,
        away_elo: float,
        home_advantage: float = 65.0
    ) -> float:
        """
        Calculate win probability from Elo ratings.

        Uses the standard Elo formula with home-field advantage.

        Args:
            home_elo: Home team Elo rating
            away_elo: Away team Elo rating
            home_advantage: Home field advantage in Elo points (default 65)

        Returns:
            Home team win probability (0-1)
        """
        # Apply home field advantage
        adjusted_home_elo = home_elo + home_advantage

        # Elo expected score formula
        elo_diff = adjusted_home_elo - away_elo
        win_prob = 1 / (1 + 10 ** (-elo_diff / 400))

        return win_prob

    def normalize_team_abbr(self, abbr: str, league: str) -> str:
        """Normalize team abbreviation for matching"""
        abbr = abbr.upper()
        return TEAM_ABBR_MAP.get(abbr, abbr)

    def generate_kalshi_ticker(
        self,
        league: str,
        game_date: str,
        away_team_abbr: str,
        home_team_abbr: str,
        winner_abbr: str
    ) -> str:
        """
        Generate Kalshi-format ticker for sports games.

        Kalshi format: KXNBAGAME-{YY}{MMM}{DD}{AWAY}{HOME}-{WINNER}
        Example: KXNBAGAME-26FEB01CLEPOR-CLE (Cleveland at Portland, Cleveland wins)

        IMPORTANT: Kalshi uses Eastern Time for game dates. ESPN returns UTC times.
        A game at 10pm Eastern on Feb 1 shows as 3am UTC Feb 2 in ESPN.
        We must convert UTC to Eastern before extracting the date.

        Args:
            league: NBA, NFL, MLB, NHL
            game_date: ISO date string (e.g., '2026-02-01T19:00:00Z' or '2026-02-01')
            away_team_abbr: 3-letter away team code
            home_team_abbr: 3-letter home team code
            winner_abbr: 3-letter winner team code

        Returns:
            Kalshi ticker string (e.g., 'KXNBAGAME-26FEB01CLEPOR-CLE')
        """
        try:
            # Parse the date - handle both ISO format with time and simple date
            if 'T' in game_date:
                dt = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
                # If no timezone, assume UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = datetime.strptime(game_date[:10], '%Y-%m-%d')
                dt = dt.replace(tzinfo=timezone.utc)

            # CRITICAL: Convert UTC to Eastern Time for Kalshi ticker
            # Kalshi uses the game date in Eastern Time, not UTC
            eastern = ZoneInfo('America/New_York')
            dt_eastern = dt.astimezone(eastern)

            # Format: YY + MMM + DD (e.g., 26FEB01) using Eastern time
            year_str = str(dt_eastern.year)[-2:]  # Last 2 digits
            month_str = MONTH_ABBR.get(dt_eastern.month, 'JAN')
            day_str = f"{dt_eastern.day:02d}"

            date_part = f"{year_str}{month_str}{day_str}"

            # Ensure 3-letter team codes (uppercase)
            away = away_team_abbr.upper()[:3]
            home = home_team_abbr.upper()[:3]
            winner = winner_abbr.upper()[:3]

            # Format: KX{LEAGUE}GAME-{DATE}{AWAY}{HOME}-{WINNER}
            ticker = f"KX{league}GAME-{date_part}{away}{home}-{winner}"

            return ticker

        except Exception as e:
            logger.warning(f"Error generating Kalshi ticker: {e}")
            # Fallback to old format if parsing fails
            return f"KX{league}GAME-{away_team_abbr}{home_team_abbr}-{winner_abbr}"

    def get_games_with_elo(self, league: str) -> List[GameData]:
        """
        Get games with Elo-based win probabilities.

        Args:
            league: NFL, NBA, or MLB

        Returns:
            List of GameData with win probabilities
        """
        games = self.fetch_espn_games(league)
        elo_ratings = self.fetch_fivethirtyeight_elo(league)

        # Home advantage by league
        home_advantages = {
            'NFL': 48,
            'NBA': 100,
            'MLB': 24,
            'NHL': 33,
        }
        home_adv = home_advantages.get(league, 50)

        game_data = []
        for game in games:
            home_abbr = self.normalize_team_abbr(game.get('home_abbr', ''), league)
            away_abbr = self.normalize_team_abbr(game.get('away_abbr', ''), league)

            home_elo = elo_ratings.get(home_abbr)
            away_elo = elo_ratings.get(away_abbr)

            win_prob = None
            if home_elo and away_elo:
                win_prob = self.calculate_elo_win_probability(home_elo, away_elo, home_adv)

            game_data.append(GameData(
                game_id=game.get('game_id', ''),
                league=league,
                home_team=game.get('home_team', ''),
                away_team=game.get('away_team', ''),
                home_team_abbr=home_abbr,
                away_team_abbr=away_abbr,
                game_date=game.get('date', ''),
                game_time=None,
                home_score=int(game['home_score']) if game.get('home_score') else None,
                away_score=int(game['away_score']) if game.get('away_score') else None,
                status=game.get('status', 'scheduled'),
                venue=game.get('venue'),
                home_elo=home_elo,
                away_elo=away_elo,
                home_win_prob=win_prob,
                source='ESPN + FiveThirtyEight',
                fetched_at=datetime.now(timezone.utc)
            ))

        return game_data

    def get_all_upcoming_games(self, days_ahead: int = 2) -> Dict[str, List[GameData]]:
        """
        Get upcoming games for all leagues.

        Fetches games for today and the next few days to ensure we capture
        all games that Kalshi might have markets for. ESPN returns UTC times
        but Kalshi uses Eastern time, so we need to fetch multiple days to
        catch all games near the date boundary.

        Args:
            days_ahead: Number of days to fetch (default 2)

        Returns:
            Dict mapping league to list of GameData
        """
        all_games = {}

        # Get current date in Eastern time for accurate date range
        eastern = ZoneInfo('America/New_York')
        now_eastern = datetime.now(eastern)

        # Generate date strings for ESPN API (YYYYMMDD format)
        dates_to_fetch = []
        for i in range(days_ahead):
            date = now_eastern + timedelta(days=i)
            dates_to_fetch.append(date.strftime('%Y%m%d'))

        for league in ['NFL', 'NBA', 'MLB']:
            seen_game_ids = set()
            games = []

            # Fetch default (today's) games first
            today_games = self.get_games_with_elo(league)
            for g in today_games:
                if g.game_id not in seen_game_ids:
                    seen_game_ids.add(g.game_id)
                    games.append(g)

            # Also fetch games for specific dates to catch edge cases
            for date_str in dates_to_fetch:
                date_games = self.fetch_espn_games(league, dates=date_str)
                elo_ratings = self.fetch_fivethirtyeight_elo(league)
                home_adv = {'NFL': 48, 'NBA': 100, 'MLB': 24, 'NHL': 33}.get(league, 50)

                for game in date_games:
                    game_id = game.get('game_id', '')
                    if game_id in seen_game_ids:
                        continue
                    seen_game_ids.add(game_id)

                    home_abbr = self.normalize_team_abbr(game.get('home_abbr', ''), league)
                    away_abbr = self.normalize_team_abbr(game.get('away_abbr', ''), league)
                    home_elo = elo_ratings.get(home_abbr)
                    away_elo = elo_ratings.get(away_abbr)
                    win_prob = None
                    if home_elo and away_elo:
                        win_prob = self.calculate_elo_win_probability(home_elo, away_elo, home_adv)

                    games.append(GameData(
                        game_id=game_id,
                        league=league,
                        home_team=game.get('home_team', ''),
                        away_team=game.get('away_team', ''),
                        home_team_abbr=home_abbr,
                        away_team_abbr=away_abbr,
                        game_date=game.get('date', ''),
                        game_time=None,
                        home_score=int(game['home_score']) if game.get('home_score') else None,
                        away_score=int(game['away_score']) if game.get('away_score') else None,
                        status=game.get('status', 'scheduled'),
                        venue=game.get('venue'),
                        home_elo=home_elo,
                        away_elo=away_elo,
                        home_win_prob=win_prob,
                        source='ESPN + FiveThirtyEight',
                        fetched_at=datetime.now(timezone.utc)
                    ))

            # Filter to scheduled games only (ESPN uses STATUS_SCHEDULED, STATUS_PRE formats)
            upcoming = [g for g in games if 'scheduled' in g.status.lower() or 'pre' in g.status.lower()]
            all_games[league] = upcoming
            logger.info(f"[{league}] {len(upcoming)} upcoming games (from {len(games)} total)")

        return all_games

    def generate_probability_estimates(
        self,
        games: Dict[str, List[GameData]]
    ) -> List[SportsProbability]:
        """
        Generate probability estimates for Kalshi sports contracts.

        Kalshi ticker format: KXNBAGAME-{YY}{MMM}{DD}{AWAY}{HOME}-{WINNER}
        Example: KXNBAGAME-26FEB01CLEPOR-CLE (Cleveland at Portland, Cleveland wins)

        Args:
            games: Dict of league -> games

        Returns:
            List of SportsProbability estimates
        """
        estimates = []

        for league, league_games in games.items():
            for game in league_games:
                if game.home_win_prob is None:
                    continue

                # Home team win - generate Kalshi-format ticker
                home_ticker = self.generate_kalshi_ticker(
                    league=league,
                    game_date=game.game_date,
                    away_team_abbr=game.away_team_abbr,
                    home_team_abbr=game.home_team_abbr,
                    winner_abbr=game.home_team_abbr
                )

                estimates.append(SportsProbability(
                    game_id=game.game_id,
                    league=league,
                    ticker_pattern=home_ticker,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    outcome='home_win',
                    our_probability=game.home_win_prob,
                    elo_based=True,
                    reasoning=f"{game.home_team} Elo: {game.home_elo:.0f}, {game.away_team} Elo: {game.away_elo:.0f}",
                    confidence='MEDIUM'
                ))

                # Away team win - generate Kalshi-format ticker
                away_ticker = self.generate_kalshi_ticker(
                    league=league,
                    game_date=game.game_date,
                    away_team_abbr=game.away_team_abbr,
                    home_team_abbr=game.home_team_abbr,
                    winner_abbr=game.away_team_abbr
                )

                estimates.append(SportsProbability(
                    game_id=game.game_id,
                    league=league,
                    ticker_pattern=away_ticker,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    outcome='away_win',
                    our_probability=1 - game.home_win_prob,
                    elo_based=True,
                    reasoning=f"Inverse of home win probability",
                    confidence='MEDIUM'
                ))

        logger.info(f"Generated {len(estimates)} sports probability estimates")
        return estimates


def main():
    """Test the sports scraper"""
    print("=" * 60)
    print("SPORTS SCRAPER TEST")
    print("=" * 60)

    scraper = SportsScraper()

    print("\n[1] Fetching ESPN Games...")
    print("-" * 40)

    for league in ['NFL', 'NBA', 'MLB']:
        games = scraper.fetch_espn_games(league)
        print(f"\n{league}: {len(games)} games")
        for game in games[:2]:
            print(f"  {game.get('away_team')} @ {game.get('home_team')}")
            print(f"    Status: {game.get('status')}")

    print("\n[2] Fetching Elo Ratings...")
    print("-" * 40)

    for league in ['NFL', 'NBA', 'MLB']:
        elo = scraper.fetch_fivethirtyeight_elo(league)
        print(f"\n{league}: {len(elo)} teams with Elo ratings")
        if elo:
            top_teams = sorted(elo.items(), key=lambda x: x[1], reverse=True)[:3]
            for team, rating in top_teams:
                print(f"  {team}: {rating:.0f}")

    print("\n[3] Games with Win Probabilities...")
    print("-" * 40)

    all_games = scraper.get_all_upcoming_games()

    for league, games in all_games.items():
        print(f"\n{league}:")
        for game in games[:3]:
            if game.home_win_prob:
                print(f"  {game.away_team} @ {game.home_team}")
                print(f"    Home Win Prob: {game.home_win_prob:.0%}")

    print("\n[4] Probability Estimates...")
    print("-" * 40)

    estimates = scraper.generate_probability_estimates(all_games)
    print(f"\nGenerated {len(estimates)} estimates")

    for est in estimates[:5]:
        print(f"  [{est.league}] {est.ticker_pattern}: {est.our_probability:.0%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
