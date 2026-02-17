"""
Sports Props Scraper

Fetches team statistics for prop betting models:
- NBA: Team pace, offensive/defensive ratings from NBA Stats API
- NFL: Team scoring averages, quarter patterns
- Expected team totals based on matchup analysis

Author: Trading Bot
Created: January 2026
"""

import os
import json
import logging
import requests
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SportsPropsScaper')


@dataclass
class TeamStats:
    """Team statistics for prop betting"""
    team_id: str
    team_name: str
    team_abbr: str
    league: str
    pace: float  # Possessions per game (NBA)
    offensive_rating: float  # Points per 100 possessions
    defensive_rating: float  # Points allowed per 100 possessions
    points_per_game: float
    points_allowed_per_game: float
    last_10_ppg: Optional[float]
    home_ppg: Optional[float]
    away_ppg: Optional[float]
    source: str
    fetched_at: datetime


@dataclass
class GameProps:
    """Prop estimates for a specific game"""
    game_id: str
    league: str
    home_team: str
    away_team: str
    home_team_abbr: str
    away_team_abbr: str
    game_date: str
    expected_home_total: float
    expected_away_total: float
    expected_game_total: float
    home_total_std: float  # Standard deviation
    away_total_std: float
    reasoning: str


@dataclass
class PropProbability:
    """Probability estimate for a prop bet"""
    game_id: str
    league: str
    team: str
    ticker_pattern: str
    prop_type: str  # 'team_total', 'game_total', 'quarter'
    threshold: float
    direction: str  # 'over' or 'under'
    our_probability: float
    expected_value: float
    reasoning: str
    confidence: str


# NBA Stats API endpoints
NBA_STATS_BASE = "https://stats.nba.com/stats"
NBA_STATS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Referer': 'https://www.nba.com/',
    'Origin': 'https://www.nba.com',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
}

# NBA Team IDs (2024-25 season)
NBA_TEAMS = {
    'ATL': {'id': 1610612737, 'name': 'Atlanta Hawks'},
    'BOS': {'id': 1610612738, 'name': 'Boston Celtics'},
    'BKN': {'id': 1610612751, 'name': 'Brooklyn Nets'},
    'CHA': {'id': 1610612766, 'name': 'Charlotte Hornets'},
    'CHI': {'id': 1610612741, 'name': 'Chicago Bulls'},
    'CLE': {'id': 1610612739, 'name': 'Cleveland Cavaliers'},
    'DAL': {'id': 1610612742, 'name': 'Dallas Mavericks'},
    'DEN': {'id': 1610612743, 'name': 'Denver Nuggets'},
    'DET': {'id': 1610612765, 'name': 'Detroit Pistons'},
    'GSW': {'id': 1610612744, 'name': 'Golden State Warriors'},
    'HOU': {'id': 1610612745, 'name': 'Houston Rockets'},
    'IND': {'id': 1610612754, 'name': 'Indiana Pacers'},
    'LAC': {'id': 1610612746, 'name': 'LA Clippers'},
    'LAL': {'id': 1610612747, 'name': 'Los Angeles Lakers'},
    'MEM': {'id': 1610612763, 'name': 'Memphis Grizzlies'},
    'MIA': {'id': 1610612748, 'name': 'Miami Heat'},
    'MIL': {'id': 1610612749, 'name': 'Milwaukee Bucks'},
    'MIN': {'id': 1610612750, 'name': 'Minnesota Timberwolves'},
    'NOP': {'id': 1610612740, 'name': 'New Orleans Pelicans'},
    'NYK': {'id': 1610612752, 'name': 'New York Knicks'},
    'OKC': {'id': 1610612760, 'name': 'Oklahoma City Thunder'},
    'ORL': {'id': 1610612753, 'name': 'Orlando Magic'},
    'PHI': {'id': 1610612755, 'name': 'Philadelphia 76ers'},
    'PHX': {'id': 1610612756, 'name': 'Phoenix Suns'},
    'POR': {'id': 1610612757, 'name': 'Portland Trail Blazers'},
    'SAC': {'id': 1610612758, 'name': 'Sacramento Kings'},
    'SAS': {'id': 1610612759, 'name': 'San Antonio Spurs'},
    'TOR': {'id': 1610612761, 'name': 'Toronto Raptors'},
    'UTA': {'id': 1610612762, 'name': 'Utah Jazz'},
    'WAS': {'id': 1610612764, 'name': 'Washington Wizards'},
}

# NFL Team averages (would be updated periodically from Pro Football Reference)
NFL_TEAM_STATS = {
    'KC': {'ppg': 27.5, 'papg': 17.2, 'home_ppg': 29.1, 'away_ppg': 25.9},
    'SF': {'ppg': 28.8, 'papg': 18.5, 'home_ppg': 30.2, 'away_ppg': 27.4},
    'BAL': {'ppg': 28.2, 'papg': 16.8, 'home_ppg': 29.8, 'away_ppg': 26.6},
    'DET': {'ppg': 27.1, 'papg': 21.3, 'home_ppg': 28.5, 'away_ppg': 25.7},
    'PHI': {'ppg': 25.9, 'papg': 19.1, 'home_ppg': 27.3, 'away_ppg': 24.5},
    'DAL': {'ppg': 24.8, 'papg': 22.4, 'home_ppg': 26.2, 'away_ppg': 23.4},
    'MIA': {'ppg': 26.3, 'papg': 23.1, 'home_ppg': 27.7, 'away_ppg': 24.9},
    'BUF': {'ppg': 26.8, 'papg': 20.5, 'home_ppg': 28.2, 'away_ppg': 25.4},
    'SEA': {'ppg': 22.1, 'papg': 20.8, 'home_ppg': 23.5, 'away_ppg': 20.7},
    'NE': {'ppg': 18.4, 'papg': 21.5, 'home_ppg': 19.8, 'away_ppg': 17.0},
    # Add more teams as needed
}


class SportsPropsScaper:
    """
    Scrapes team statistics for sports prop betting models.

    NBA: Uses NBA Stats API for advanced metrics
    NFL: Uses historical scoring averages
    """

    CACHE_DURATION = timedelta(hours=6)

    def __init__(self, cache_dir: str = "data/sports_props_cache"):
        """Initialize the sports props scraper"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update(NBA_STATS_HEADERS)

        self._team_stats_cache: Dict[str, TeamStats] = {}

        logger.info("SportsPropsScaper initialized")

    def _get_cache_path(self, key: str) -> str:
        """Get cache file path"""
        safe_key = re.sub(r'[^\w\-]', '_', key)
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

    def fetch_nba_team_stats(self) -> Dict[str, TeamStats]:
        """
        Fetch NBA team statistics from NBA Stats API.

        Returns:
            Dict mapping team abbreviation to TeamStats
        """
        cache_key = "nba_team_stats"
        cached = self._load_cache(cache_key)

        if cached:
            stats = {}
            for abbr, data in cached.items():
                stats[abbr] = TeamStats(**data)
            return stats

        try:
            # Fetch team advanced stats
            url = f"{NBA_STATS_BASE}/leaguedashteamstats"
            params = {
                'Conference': '',
                'DateFrom': '',
                'DateTo': '',
                'Division': '',
                'GameScope': '',
                'GameSegment': '',
                'Height': '',
                'LastNGames': 0,
                'LeagueID': '00',
                'Location': '',
                'MeasureType': 'Advanced',
                'Month': 0,
                'OpponentTeamID': 0,
                'Outcome': '',
                'PORound': 0,
                'PaceAdjust': 'N',
                'PerMode': 'PerGame',
                'Period': 0,
                'PlayerExperience': '',
                'PlayerPosition': '',
                'PlusMinus': 'N',
                'Rank': 'N',
                'Season': '2025-26',
                'SeasonSegment': '',
                'SeasonType': 'Regular Season',
                'ShotClockRange': '',
                'StarterBench': '',
                'TeamID': 0,
                'TwoWay': 0,
                'VsConference': '',
                'VsDivision': ''
            }

            response = self.session.get(url, params=params, timeout=30)

            if response.status_code != 200:
                logger.warning(f"NBA Stats API returned {response.status_code}")
                return self._get_fallback_nba_stats()

            data = response.json()
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']

            # Find relevant column indices
            team_id_idx = headers.index('TEAM_ID')
            team_name_idx = headers.index('TEAM_NAME')
            pace_idx = headers.index('PACE') if 'PACE' in headers else None
            off_rtg_idx = headers.index('OFF_RATING') if 'OFF_RATING' in headers else None
            def_rtg_idx = headers.index('DEF_RATING') if 'DEF_RATING' in headers else None

            stats = {}
            for row in rows:
                team_id = row[team_id_idx]

                # Find team abbreviation
                team_abbr = None
                for abbr, info in NBA_TEAMS.items():
                    if info['id'] == team_id:
                        team_abbr = abbr
                        break

                if not team_abbr:
                    continue

                pace = row[pace_idx] if pace_idx else 100.0
                off_rtg = row[off_rtg_idx] if off_rtg_idx else 110.0
                def_rtg = row[def_rtg_idx] if def_rtg_idx else 110.0

                # Estimate PPG from pace and rating
                ppg = (pace * off_rtg) / 100
                papg = (pace * def_rtg) / 100

                stats[team_abbr] = TeamStats(
                    team_id=str(team_id),
                    team_name=row[team_name_idx],
                    team_abbr=team_abbr,
                    league='NBA',
                    pace=pace,
                    offensive_rating=off_rtg,
                    defensive_rating=def_rtg,
                    points_per_game=ppg,
                    points_allowed_per_game=papg,
                    last_10_ppg=None,
                    home_ppg=ppg * 1.03,  # Home court advantage
                    away_ppg=ppg * 0.97,
                    source='NBA Stats API',
                    fetched_at=datetime.now(timezone.utc)
                )

            # Save to cache
            cache_data = {abbr: {
                'team_id': s.team_id,
                'team_name': s.team_name,
                'team_abbr': s.team_abbr,
                'league': s.league,
                'pace': s.pace,
                'offensive_rating': s.offensive_rating,
                'defensive_rating': s.defensive_rating,
                'points_per_game': s.points_per_game,
                'points_allowed_per_game': s.points_allowed_per_game,
                'last_10_ppg': s.last_10_ppg,
                'home_ppg': s.home_ppg,
                'away_ppg': s.away_ppg,
                'source': s.source,
                'fetched_at': s.fetched_at.isoformat()
            } for abbr, s in stats.items()}

            self._save_cache(cache_key, cache_data)
            logger.info(f"[NBA] Fetched stats for {len(stats)} teams")

            return stats

        except Exception as e:
            logger.error(f"NBA Stats API error: {e}")
            return self._get_fallback_nba_stats()

    def _get_fallback_nba_stats(self) -> Dict[str, TeamStats]:
        """Return fallback NBA stats based on typical values"""
        stats = {}

        # Default stats for all teams
        default_pace = 99.5
        default_off_rtg = 113.0
        default_def_rtg = 113.0

        # Adjusted ratings for top/bottom teams
        team_adjustments = {
            'BOS': {'off': 5, 'def': -5, 'pace': 2},
            'DEN': {'off': 4, 'def': -2, 'pace': 0},
            'OKC': {'off': 3, 'def': -4, 'pace': 3},
            'MIL': {'off': 3, 'def': -1, 'pace': 1},
            'PHX': {'off': 2, 'def': 1, 'pace': 2},
            'IND': {'off': 2, 'def': 3, 'pace': 5},  # High pace
            'DET': {'off': -5, 'def': 5, 'pace': -2},
            'WAS': {'off': -4, 'def': 6, 'pace': 1},
            'POR': {'off': -3, 'def': 4, 'pace': 0},
        }

        for abbr, info in NBA_TEAMS.items():
            adj = team_adjustments.get(abbr, {'off': 0, 'def': 0, 'pace': 0})

            pace = default_pace + adj.get('pace', 0)
            off_rtg = default_off_rtg + adj.get('off', 0)
            def_rtg = default_def_rtg + adj.get('def', 0)

            ppg = (pace * off_rtg) / 100
            papg = (pace * def_rtg) / 100

            stats[abbr] = TeamStats(
                team_id=str(info['id']),
                team_name=info['name'],
                team_abbr=abbr,
                league='NBA',
                pace=pace,
                offensive_rating=off_rtg,
                defensive_rating=def_rtg,
                points_per_game=ppg,
                points_allowed_per_game=papg,
                last_10_ppg=ppg,
                home_ppg=ppg * 1.03,
                away_ppg=ppg * 0.97,
                source='Fallback estimates',
                fetched_at=datetime.now(timezone.utc)
            )

        return stats

    def get_nfl_team_stats(self) -> Dict[str, TeamStats]:
        """Get NFL team statistics (from stored data)"""
        stats = {}

        for abbr, data in NFL_TEAM_STATS.items():
            stats[abbr] = TeamStats(
                team_id=abbr,
                team_name=abbr,
                team_abbr=abbr,
                league='NFL',
                pace=0,  # Not applicable to NFL
                offensive_rating=0,
                defensive_rating=0,
                points_per_game=data['ppg'],
                points_allowed_per_game=data['papg'],
                last_10_ppg=data['ppg'],
                home_ppg=data.get('home_ppg', data['ppg'] * 1.05),
                away_ppg=data.get('away_ppg', data['ppg'] * 0.95),
                source='Historical averages',
                fetched_at=datetime.now(timezone.utc)
            )

        return stats

    def estimate_nba_team_total(
        self,
        team_abbr: str,
        opponent_abbr: str,
        home: bool = True,
        stats: Optional[Dict[str, TeamStats]] = None
    ) -> Tuple[float, float, str]:
        """
        Estimate NBA team total points for a game.

        Formula: Expected = (Team Pace + Opp Pace) / 2 * (Team OffRtg / Opp DefRtg) * adj

        Args:
            team_abbr: Team abbreviation
            opponent_abbr: Opponent abbreviation
            home: Whether team is playing at home
            stats: Pre-fetched team stats (optional)

        Returns:
            Tuple of (expected_total, std_deviation, reasoning)
        """
        if stats is None:
            stats = self.fetch_nba_team_stats()

        team = stats.get(team_abbr)
        opp = stats.get(opponent_abbr)

        if not team or not opp:
            return (110.0, 10.0, "Missing team data")

        # Calculate game pace (average of both teams)
        game_pace = (team.pace + opp.pace) / 2

        # Calculate expected points
        # Points = Pace * (OffRtg / 100) adjusted for opponent defense
        matchup_efficiency = (team.offensive_rating / opp.defensive_rating)
        expected = game_pace * matchup_efficiency

        # Home court advantage (~3% boost)
        if home:
            expected *= 1.03
        else:
            expected *= 0.97

        # Standard deviation (NBA scoring variance is ~12-15 points)
        std_dev = 12.0

        reasoning = (
            f"{team.team_name}: Pace={team.pace:.1f}, OffRtg={team.offensive_rating:.1f} "
            f"vs {opp.team_name} DefRtg={opp.defensive_rating:.1f}"
        )

        return (expected, std_dev, reasoning)

    def estimate_nfl_team_total(
        self,
        team_abbr: str,
        opponent_abbr: str,
        home: bool = True
    ) -> Tuple[float, float, str]:
        """
        Estimate NFL team total points for a game.

        Args:
            team_abbr: Team abbreviation
            opponent_abbr: Opponent abbreviation
            home: Whether team is playing at home

        Returns:
            Tuple of (expected_total, std_deviation, reasoning)
        """
        stats = self.get_nfl_team_stats()

        team = stats.get(team_abbr)
        opp = stats.get(opponent_abbr)

        if not team or not opp:
            return (21.0, 7.0, "Missing team data")

        # Base expected points
        if home:
            base_points = team.home_ppg if team.home_ppg else team.points_per_game * 1.05
        else:
            base_points = team.away_ppg if team.away_ppg else team.points_per_game * 0.95

        # Adjust for opponent defense
        opp_def_factor = opp.points_allowed_per_game / 21.5  # League average
        expected = base_points * (opp_def_factor ** 0.5)  # Partial adjustment

        # NFL has high variance
        std_dev = 7.0

        reasoning = f"{team_abbr}: PPG={team.points_per_game:.1f} vs {opponent_abbr} PAPG={opp.points_allowed_per_game:.1f}"

        return (expected, std_dev, reasoning)

    def calculate_over_under_probability(
        self,
        expected: float,
        std_dev: float,
        threshold: float
    ) -> Tuple[float, float]:
        """
        Calculate probability of over/under a threshold.

        Args:
            expected: Expected value
            std_dev: Standard deviation
            threshold: Threshold value

        Returns:
            Tuple of (prob_over, prob_under)
        """
        import math

        z = (threshold - expected) / std_dev

        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        prob_under = norm_cdf(z)
        prob_over = 1 - prob_under

        return (prob_over, prob_under)

    def generate_game_props(
        self,
        home_abbr: str,
        away_abbr: str,
        league: str,
        game_id: str = "",
        game_date: str = ""
    ) -> GameProps:
        """
        Generate prop estimates for a game.

        Args:
            home_abbr: Home team abbreviation
            away_abbr: Away team abbreviation
            league: NBA or NFL
            game_id: Optional game ID
            game_date: Optional game date

        Returns:
            GameProps object
        """
        if league == 'NBA':
            stats = self.fetch_nba_team_stats()
            home_exp, home_std, home_reason = self.estimate_nba_team_total(
                home_abbr, away_abbr, home=True, stats=stats
            )
            away_exp, away_std, away_reason = self.estimate_nba_team_total(
                away_abbr, home_abbr, home=False, stats=stats
            )
        else:  # NFL
            home_exp, home_std, home_reason = self.estimate_nfl_team_total(
                home_abbr, away_abbr, home=True
            )
            away_exp, away_std, away_reason = self.estimate_nfl_team_total(
                away_abbr, home_abbr, home=False
            )

        return GameProps(
            game_id=game_id,
            league=league,
            home_team=home_abbr,
            away_team=away_abbr,
            home_team_abbr=home_abbr,
            away_team_abbr=away_abbr,
            game_date=game_date or datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            expected_home_total=home_exp,
            expected_away_total=away_exp,
            expected_game_total=home_exp + away_exp,
            home_total_std=home_std,
            away_total_std=away_std,
            reasoning=f"Home: {home_reason}; Away: {away_reason}"
        )

    def generate_probability_estimates(
        self,
        games: List[GameProps],
        thresholds: Optional[Dict[str, List[float]]] = None
    ) -> List[PropProbability]:
        """
        Generate probability estimates for props.

        Args:
            games: List of GameProps
            thresholds: Optional dict of thresholds by league

        Returns:
            List of PropProbability estimates
        """
        if thresholds is None:
            thresholds = {
                'NBA': [95, 100, 105, 110, 115, 120, 125],
                'NFL': [14, 17, 21, 24, 28, 31, 35],
            }

        estimates = []

        for game in games:
            league_thresholds = thresholds.get(game.league, [])

            for threshold in league_thresholds:
                # Home team total
                prob_over, prob_under = self.calculate_over_under_probability(
                    game.expected_home_total, game.home_total_std, threshold
                )

                estimates.append(PropProbability(
                    game_id=game.game_id,
                    league=game.league,
                    team=game.home_team,
                    ticker_pattern=f'KX{game.league}TEAMTOTAL-*-{game.home_team_abbr}{threshold}',
                    prop_type='team_total',
                    threshold=threshold,
                    direction='over',
                    our_probability=prob_over,
                    expected_value=game.expected_home_total,
                    reasoning=f"{game.home_team} expected: {game.expected_home_total:.1f}",
                    confidence='MEDIUM'
                ))

                # Away team total
                prob_over, prob_under = self.calculate_over_under_probability(
                    game.expected_away_total, game.away_total_std, threshold
                )

                estimates.append(PropProbability(
                    game_id=game.game_id,
                    league=game.league,
                    team=game.away_team,
                    ticker_pattern=f'KX{game.league}TEAMTOTAL-*-{game.away_team_abbr}{threshold}',
                    prop_type='team_total',
                    threshold=threshold,
                    direction='over',
                    our_probability=prob_over,
                    expected_value=game.expected_away_total,
                    reasoning=f"{game.away_team} expected: {game.expected_away_total:.1f}",
                    confidence='MEDIUM'
                ))

        logger.info(f"Generated {len(estimates)} prop probability estimates")
        return estimates


def main():
    """Test the sports props scraper"""
    print("=" * 60)
    print("SPORTS PROPS SCRAPER TEST")
    print("=" * 60)

    scraper = SportsPropsScaper()

    print("\n[1] Fetching NBA Team Stats...")
    print("-" * 40)
    nba_stats = scraper.fetch_nba_team_stats()

    print(f"Got stats for {len(nba_stats)} NBA teams")
    for abbr in ['BOS', 'DEN', 'OKC', 'DET'][:4]:
        if abbr in nba_stats:
            team = nba_stats[abbr]
            print(f"  {abbr}: Pace={team.pace:.1f}, OffRtg={team.offensive_rating:.1f}, "
                  f"DefRtg={team.defensive_rating:.1f}, PPG={team.points_per_game:.1f}")

    print("\n[2] Estimating Team Totals...")
    print("-" * 40)

    # Example matchup: Boston vs Denver
    props = scraper.generate_game_props('BOS', 'DEN', 'NBA')
    print(f"Boston vs Denver:")
    print(f"  Boston expected: {props.expected_home_total:.1f} pts")
    print(f"  Denver expected: {props.expected_away_total:.1f} pts")
    print(f"  Game total: {props.expected_game_total:.1f} pts")

    # NFL example
    props_nfl = scraper.generate_game_props('SEA', 'NE', 'NFL')
    print(f"\nSeattle vs New England:")
    print(f"  Seattle expected: {props_nfl.expected_home_total:.1f} pts")
    print(f"  New England expected: {props_nfl.expected_away_total:.1f} pts")

    print("\n[3] Probability Estimates...")
    print("-" * 40)

    games = [props, props_nfl]
    estimates = scraper.generate_probability_estimates(games)

    for est in estimates[:10]:
        print(f"  [{est.league}] {est.team} > {est.threshold}: {est.our_probability:.0%}")
        print(f"    Expected: {est.expected_value:.1f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
