"""
Data Aggregator

Central hub that combines all scrapers and provides unified probability estimates
for the multi-market scanner.

Manages:
- Weather data (NWS API)
- Economic data (FRED + scraping)
- Economic releases (Cleveland Fed CPI, Atlanta Fed GDP)
- Crypto data (CoinGecko + Fear/Greed + Technicals)
- Earnings data (Yahoo Finance)
- Sports data (ESPN, Elo ratings)
- Sports props (NBA Stats, team totals)
- Awards data (Gold Derby, precursors)
- Climate data (NOAA temperature records)
- Box office data (BOM, RT)

Author: Trading Bot
Created: January 2026
"""

import os
import json
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DataAggregator')


# =============================================================================
# PROBABILITY CALIBRATION
# =============================================================================

def calibrate_probability(
    raw_prob: float,
    event_date: Optional[datetime] = None,
    confidence: str = 'MEDIUM',
    max_prob: float = 0.85,
    min_prob: float = 0.15
) -> float:
    """
    Calibrate raw probability estimates to be more realistic.

    Applies:
    1. Hard cap at max_prob (never 100% certain)
    2. Hard floor at min_prob (never 0% certain)
    3. Time-based uncertainty (farther out = regress toward 50%)
    4. Confidence-based adjustment
    5. Softer sigmoid curve instead of hard thresholds

    Args:
        raw_prob: Raw probability from model (0-1)
        event_date: When the event resolves (for time decay)
        confidence: 'HIGH', 'MEDIUM', or 'LOW'
        max_prob: Maximum allowed probability
        min_prob: Minimum allowed probability

    Returns:
        Calibrated probability (0-1)
    """
    # 1. Apply sigmoid softening to extreme values
    # This compresses probabilities toward 50% slightly
    def soft_sigmoid(p, strength=0.15):
        """Soften extreme probabilities toward 0.5"""
        # Map p from [0,1] to centered range, apply compression, map back
        centered = p - 0.5
        compressed = centered * (1 - strength)
        return compressed + 0.5

    prob = soft_sigmoid(raw_prob)

    # 2. Time-based uncertainty decay
    # Events further out regress toward 50%
    if event_date:
        now = datetime.now(timezone.utc)
        if event_date.tzinfo is None:
            event_date = event_date.replace(tzinfo=timezone.utc)

        days_out = (event_date - now).days

        if days_out > 0:
            # Decay factor: more days = more regression to 50%
            # At 30 days, uncertainty adds ~10%
            # At 90 days, uncertainty adds ~25%
            decay = min(0.4, days_out / 200)  # Max 40% regression
            prob = prob * (1 - decay) + 0.5 * decay

    # 3. Confidence-based adjustment
    confidence_factors = {
        'HIGH': 0.0,      # No additional uncertainty
        'MEDIUM': 0.05,   # 5% regression toward 50%
        'LOW': 0.15       # 15% regression toward 50%
    }
    conf_factor = confidence_factors.get(confidence.upper(), 0.10)
    prob = prob * (1 - conf_factor) + 0.5 * conf_factor

    # 4. Apply hard caps
    prob = max(min_prob, min(max_prob, prob))

    return prob


def calibrate_estimate(estimate: Dict) -> Dict:
    """
    Calibrate a single probability estimate dict.

    Args:
        estimate: Dict with 'our_probability' and optional 'confidence', 'date' keys

    Returns:
        Estimate dict with calibrated probability
    """
    raw_prob = estimate.get('our_probability', 0.5)
    confidence = estimate.get('confidence', 'MEDIUM')

    # Try to parse event date from ticker or estimate
    event_date = None
    ticker = estimate.get('ticker_pattern', '')

    # Try to extract date from ticker patterns like KXCPI-26FEB-T0.4
    # or KXBTC-26JAN3116-T95000
    import re
    date_match = re.search(r'(\d{2})([A-Z]{3})(\d{2})?', ticker)
    if date_match:
        try:
            year = 2000 + int(date_match.group(1))
            month_str = date_match.group(2)
            day = int(date_match.group(3)) if date_match.group(3) else 15

            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            month = month_map.get(month_str, 1)
            event_date = datetime(year, month, min(day, 28), tzinfo=timezone.utc)
        except Exception as e:
            logger.debug(f"Error parsing event date from ticker: {e}")

    # Detect weather tickers for logging purposes
    # Weather tickers start with: KXHIGH (high temp), KXLOWT (low temp), KXRAIN (precipitation)
    is_weather = ticker.upper().startswith(('KXHIGH', 'KXLOWT', 'KXRAIN'))

    # Weather probabilities come from a normal distribution model that already
    # accounts for NWS forecast uncertainty (~3F standard error). Adding extra
    # regression toward 50% AND a hard 0.70 cap on top of that creates a
    # double-penalty that suppresses legitimate strong signals.
    #
    # Fix: Weather goes through the same calibration as everything else.
    # The standard pipeline (sigmoid softening + time decay + confidence
    # adjustment + 0.85 cap) provides sufficient conservatism without
    # destroying edge on clear forecasts.
    calibrated_prob = calibrate_probability(
        raw_prob=raw_prob,
        event_date=event_date,
        confidence=confidence
    )

    # Return new dict with calibrated probability
    calibrated = estimate.copy()
    calibrated['our_probability'] = calibrated_prob
    calibrated['raw_probability'] = raw_prob  # Keep original for debugging
    if is_weather:
        calibrated['weather_calibrated'] = True  # Flag for debugging

    return calibrated


@dataclass
class AggregatedData:
    """Combined data from all sources"""
    timestamp: datetime
    weather: Dict[str, Any]
    economic: Dict[str, Any]
    economic_releases: Dict[str, Any]  # CPI/GDP nowcasts
    crypto: Dict[str, Any]
    earnings: Dict[str, Any]
    sports: Dict[str, Any]
    sports_props: Dict[str, Any]  # Team totals
    awards: Dict[str, Any]  # Gold Derby predictions
    climate: Dict[str, Any]  # NOAA temperature
    boxoffice: Dict[str, Any]
    probability_estimates: Dict[str, List[Dict]]


class DataAggregator:
    """
    Central hub for all scraped data.

    Coordinates scraping across all sources, normalizes data,
    and provides unified probability estimates for the market scanner.
    """

    CACHE_FILE = "data/aggregated_data.json"
    CACHE_DURATION = timedelta(hours=1)

    def __init__(self, fred_api_key: str = None):
        """Initialize the data aggregator"""
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')

        # Lazy load scrapers to avoid import errors
        self._weather_scraper = None
        self._economic_scraper = None
        self._economic_releases_scraper = None
        self._crypto_scraper = None
        self._earnings_scraper = None
        self._sports_scraper = None
        self._sports_props_scraper = None
        self._awards_scraper = None
        self._climate_scraper = None
        self._boxoffice_scraper = None

        self._last_fetch: Optional[datetime] = None
        self._cached_data: Optional[AggregatedData] = None

        os.makedirs('data', exist_ok=True)
        logger.info("DataAggregator initialized")

    @property
    def weather_scraper(self):
        if self._weather_scraper is None:
            from scrapers.weather_scraper import WeatherScraper
            self._weather_scraper = WeatherScraper()
        return self._weather_scraper

    @property
    def economic_scraper(self):
        if self._economic_scraper is None:
            from scrapers.economic_scraper import EconomicScraper
            self._economic_scraper = EconomicScraper(fred_api_key=self.fred_api_key)
        return self._economic_scraper

    @property
    def crypto_scraper(self):
        if self._crypto_scraper is None:
            from scrapers.crypto_scraper import CryptoScraper
            self._crypto_scraper = CryptoScraper()
        return self._crypto_scraper

    @property
    def earnings_scraper(self):
        if self._earnings_scraper is None:
            from scrapers.earnings_scraper import EarningsScraper
            self._earnings_scraper = EarningsScraper()
        return self._earnings_scraper

    @property
    def sports_scraper(self):
        if self._sports_scraper is None:
            from scrapers.sports_scraper import SportsScraper
            self._sports_scraper = SportsScraper()
        return self._sports_scraper

    @property
    def boxoffice_scraper(self):
        if self._boxoffice_scraper is None:
            from scrapers.boxoffice_scraper import BoxOfficeScraper
            self._boxoffice_scraper = BoxOfficeScraper()
        return self._boxoffice_scraper

    @property
    def economic_releases_scraper(self):
        if self._economic_releases_scraper is None:
            from scrapers.economic_releases_scraper import EconomicReleasesScraper
            self._economic_releases_scraper = EconomicReleasesScraper()
        return self._economic_releases_scraper

    @property
    def sports_props_scraper(self):
        if self._sports_props_scraper is None:
            from scrapers.sports_props_scraper import SportsPropsScaper
            self._sports_props_scraper = SportsPropsScaper()
        return self._sports_props_scraper

    @property
    def awards_scraper(self):
        if self._awards_scraper is None:
            from scrapers.awards_scraper import AwardsScraper
            self._awards_scraper = AwardsScraper()
        return self._awards_scraper

    @property
    def climate_scraper(self):
        if self._climate_scraper is None:
            from scrapers.climate_scraper import ClimateScraper
            self._climate_scraper = ClimateScraper()
        return self._climate_scraper

    def _load_cache(self) -> Optional[AggregatedData]:
        """Load cached aggregated data"""
        if not os.path.exists(self.CACHE_FILE):
            return None

        try:
            with open(self.CACHE_FILE, 'r') as f:
                data = json.load(f)

            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now(timezone.utc) - cached_time < self.CACHE_DURATION:
                return AggregatedData(
                    timestamp=cached_time,
                    weather=data.get('weather', {}),
                    economic=data.get('economic', {}),
                    economic_releases=data.get('economic_releases', {}),
                    crypto=data.get('crypto', {}),
                    earnings=data.get('earnings', {}),
                    sports=data.get('sports', {}),
                    sports_props=data.get('sports_props', {}),
                    awards=data.get('awards', {}),
                    climate=data.get('climate', {}),
                    boxoffice=data.get('boxoffice', {}),
                    probability_estimates=data.get('probability_estimates', {})
                )
        except Exception as e:
            logger.debug(f"Cache load error: {e}")

        return None

    def _save_cache(self, data: AggregatedData):
        """Save aggregated data to cache"""
        try:
            cache_data = {
                'timestamp': data.timestamp.isoformat(),
                'weather': data.weather,
                'economic': data.economic,
                'economic_releases': data.economic_releases,
                'crypto': data.crypto,
                'earnings': data.earnings,
                'sports': data.sports,
                'sports_props': data.sports_props,
                'awards': data.awards,
                'climate': data.climate,
                'boxoffice': data.boxoffice,
                'probability_estimates': data.probability_estimates
            }
            with open(self.CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Cache save error: {e}")

    def fetch_weather_data(self) -> Dict[str, Any]:
        """Fetch weather data from NWS"""
        logger.info("[WEATHER] Fetching weather forecasts...")
        result = {
            'forecasts': {},
            'probability_estimates': [],
            'status': 'ok'
        }

        try:
            forecasts = self.weather_scraper.get_all_forecasts()

            # Convert to serializable format
            for city, city_forecasts in forecasts.items():
                result['forecasts'][city] = [
                    {
                        'date': fc.date,
                        'high_temp': fc.high_temp,
                        'low_temp': fc.low_temp,
                        'precip_probability': fc.precip_probability,
                        'conditions': fc.conditions
                    }
                    for fc in city_forecasts
                ]

            # Generate probability estimates
            estimates = self.weather_scraper.generate_probability_estimates(forecasts)
            result['probability_estimates'] = [
                {
                    'ticker_pattern': e.ticker_pattern,
                    'city': e.city,
                    'date': e.date,
                    'threshold': e.threshold,
                    'our_probability': e.our_probability,
                    'reasoning': e.reasoning
                }
                for e in estimates
            ]

            logger.info(f"[WEATHER] Got {len(forecasts)} cities, {len(estimates)} estimates")

        except Exception as e:
            logger.error(f"[WEATHER] Error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def fetch_economic_data(self) -> Dict[str, Any]:
        """Fetch economic indicators from FRED"""
        logger.info("[ECONOMIC] Fetching economic indicators...")
        result = {
            'indicators': {},
            'probability_estimates': [],
            'status': 'ok'
        }

        try:
            indicators = self.economic_scraper.get_latest_indicators()

            # Convert to serializable format
            for name, ind in indicators.items():
                result['indicators'][name] = {
                    'value': ind.current_value,
                    'previous': ind.previous_value,
                    'unit': ind.unit,
                    'date': ind.release_date
                }

            # Generate probability estimates
            estimates = self.economic_scraper.generate_probability_estimates(indicators)
            result['probability_estimates'] = [
                {
                    'indicator': e.indicator,
                    'ticker_pattern': e.ticker_pattern,
                    'threshold': e.threshold,
                    'direction': e.direction,
                    'our_probability': e.our_probability,
                    'reasoning': e.reasoning,
                    'confidence': e.confidence
                }
                for e in estimates
            ]

            logger.info(f"[ECONOMIC] Got {len(indicators)} indicators, {len(estimates)} estimates")

        except Exception as e:
            logger.error(f"[ECONOMIC] Error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def fetch_crypto_data(self) -> Dict[str, Any]:
        """Fetch crypto data from CoinGecko and Fear/Greed"""
        logger.info("[CRYPTO] Fetching crypto data...")
        result = {
            'prices': {},
            'fear_greed': None,
            'probability_estimates': [],
            'status': 'ok'
        }

        try:
            # Get prices
            prices = self.crypto_scraper.fetch_crypto_prices(['BTC', 'ETH'])
            for symbol, data in prices.items():
                result['prices'][symbol] = {
                    'price': data.price_usd,
                    'change_24h': data.change_24h_pct,
                    'high_24h': data.high_24h,
                    'low_24h': data.low_24h
                }

            # Get Fear & Greed
            fg = self.crypto_scraper.fetch_fear_greed_index()
            if fg:
                result['fear_greed'] = {
                    'value': fg.value,
                    'classification': fg.classification,
                    'previous': fg.previous_value
                }

            # Generate probability estimates
            estimates = self.crypto_scraper.generate_probability_estimates(prices, fg)
            result['probability_estimates'] = [
                {
                    'symbol': e.symbol,
                    'ticker_pattern': e.ticker_pattern,
                    'threshold': e.threshold,
                    'timeframe': e.timeframe,
                    'direction': e.direction,
                    'our_probability': e.our_probability,
                    'current_price': e.current_price,
                    'reasoning': e.reasoning
                }
                for e in estimates
            ]

            logger.info(f"[CRYPTO] Got {len(prices)} prices, {len(estimates)} estimates")

        except Exception as e:
            logger.error(f"[CRYPTO] Error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def fetch_earnings_data(self) -> Dict[str, Any]:
        """Fetch earnings data from Yahoo Finance"""
        logger.info("[EARNINGS] Fetching earnings data...")
        result = {
            'upcoming': [],
            'probability_estimates': [],
            'status': 'ok'
        }

        try:
            earnings = self.earnings_scraper.get_upcoming_earnings()

            # Convert to serializable format
            for est in earnings:
                result['upcoming'].append({
                    'symbol': est.symbol,
                    'company': est.company_name,
                    'report_date': est.report_date,
                    'eps_estimate': est.eps_estimate,
                    'historical_beat_rate': est.historical_beat_rate
                })

            # Generate probability estimates
            estimates = self.earnings_scraper.generate_probability_estimates(earnings)
            result['probability_estimates'] = [
                {
                    'symbol': e.symbol,
                    'ticker_pattern': e.ticker_pattern,
                    'outcome': e.outcome,
                    'our_probability': e.our_probability,
                    'reasoning': e.reasoning,
                    'confidence': e.confidence
                }
                for e in estimates
            ]

            logger.info(f"[EARNINGS] Got {len(earnings)} earnings, {len(estimates)} estimates")

        except Exception as e:
            logger.error(f"[EARNINGS] Error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def fetch_sports_data(self) -> Dict[str, Any]:
        """Fetch sports data from ESPN and FiveThirtyEight"""
        logger.info("[SPORTS] Fetching sports data...")
        result = {
            'games': {},
            'probability_estimates': [],
            'status': 'ok'
        }

        try:
            all_games = self.sports_scraper.get_all_upcoming_games()

            # Convert to serializable format
            for league, games in all_games.items():
                result['games'][league] = [
                    {
                        'game_id': g.game_id,
                        'home_team': g.home_team,
                        'away_team': g.away_team,
                        'home_abbr': g.home_team_abbr,
                        'away_abbr': g.away_team_abbr,
                        'game_date': g.game_date,
                        'home_elo': g.home_elo,
                        'away_elo': g.away_elo,
                        'home_win_prob': g.home_win_prob
                    }
                    for g in games
                ]

            # Generate probability estimates
            estimates = self.sports_scraper.generate_probability_estimates(all_games)
            result['probability_estimates'] = [
                {
                    'game_id': e.game_id,
                    'league': e.league,
                    'ticker_pattern': e.ticker_pattern,
                    'home_team': e.home_team,
                    'away_team': e.away_team,
                    'outcome': e.outcome,
                    'our_probability': e.our_probability,
                    'reasoning': e.reasoning,
                    'confidence': e.confidence
                }
                for e in estimates
            ]

            total_games = sum(len(games) for games in all_games.values())
            logger.info(f"[SPORTS] Got {total_games} games, {len(estimates)} estimates")

        except Exception as e:
            logger.error(f"[SPORTS] Error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def fetch_boxoffice_data(self) -> Dict[str, Any]:
        """Fetch box office data from BOM and other sources"""
        logger.info("[BOXOFFICE] Fetching box office data...")
        result = {
            'weekend': [],
            'upcoming': [],
            'probability_estimates': [],
            'status': 'ok'
        }

        try:
            weekend = self.boxoffice_scraper.fetch_bom_weekend()
            upcoming = self.boxoffice_scraper.fetch_upcoming_releases()

            result['weekend'] = weekend
            result['upcoming'] = upcoming

            # Create MovieData objects for estimates
            movies = []
            for m in upcoming[:10]:  # Top 10 upcoming
                movie_data = self.boxoffice_scraper.get_movie_data(
                    m.get('title', ''),
                    release_date=m.get('release_date'),
                    franchise=m.get('franchise')
                )
                movies.append(movie_data)

            # Generate probability estimates
            estimates = self.boxoffice_scraper.generate_probability_estimates(movies)
            result['probability_estimates'] = [
                {
                    'movie_title': e.movie_title,
                    'ticker_pattern': e.ticker_pattern,
                    'threshold': e.threshold,
                    'direction': e.direction,
                    'our_probability': e.our_probability,
                    'reasoning': e.reasoning,
                    'confidence': e.confidence
                }
                for e in estimates
            ]

            logger.info(f"[BOXOFFICE] Got {len(weekend)} weekend, {len(upcoming)} upcoming, {len(estimates)} estimates")

        except Exception as e:
            logger.error(f"[BOXOFFICE] Error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def fetch_economic_releases_data(self) -> Dict[str, Any]:
        """Fetch economic releases (CPI/GDP nowcasts) from Fed sources"""
        logger.info("[ECONOMIC_RELEASES] Fetching nowcasts...")
        result = {
            'cpi': None,
            'gdp': None,
            'jobs': None,
            'probability_estimates': [],
            'status': 'ok'
        }

        try:
            # Fetch CPI nowcast
            cpi = self.economic_releases_scraper.fetch_cleveland_fed_cpi()
            if cpi:
                result['cpi'] = {
                    'headline_cpi': cpi.headline_cpi,
                    'core_cpi': cpi.core_cpi,
                    'source': cpi.source,
                    'forecast_date': cpi.forecast_date
                }

            # Fetch GDP nowcast
            gdp = self.economic_releases_scraper.fetch_atlanta_fed_gdp()
            if gdp:
                result['gdp'] = {
                    'estimate': gdp.gdp_estimate,
                    'quarter': gdp.target_quarter,
                    'source': gdp.source
                }

            # Fetch jobless claims
            jobs = self.economic_releases_scraper.fetch_jobless_claims()
            if jobs:
                result['jobs'] = {
                    'weekly_claims': jobs.weekly_claims,
                    'continuing_claims': jobs.continuing_claims,
                    'claims_trend': jobs.claims_trend
                }

            # Note: probability estimates generated by edge finder
            logger.info(f"[ECONOMIC_RELEASES] CPI: {result['cpi'] is not None}, GDP: {result['gdp'] is not None}")

        except Exception as e:
            logger.error(f"[ECONOMIC_RELEASES] Error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def fetch_sports_props_data(self) -> Dict[str, Any]:
        """Fetch sports props data (team stats for totals)"""
        logger.info("[SPORTS_PROPS] Fetching team stats...")
        result = {
            'nba_stats': {},
            'nfl_stats': {},
            'probability_estimates': [],
            'status': 'ok'
        }

        try:
            # Fetch NBA stats
            nba_stats = self.sports_props_scraper.fetch_nba_team_stats()
            for team, stats in nba_stats.items():
                result['nba_stats'][team] = {
                    'pace': stats.pace,
                    'off_rating': stats.offensive_rating,
                    'def_rating': stats.defensive_rating,
                    'avg_points': stats.points_per_game
                }

            # NFL stats (seasonal)
            nfl_stats = self.sports_props_scraper.get_nfl_team_stats()
            for team, stats in nfl_stats.items():
                result['nfl_stats'][team] = {
                    'avg_points': stats.points_per_game,
                    'avg_allowed': stats.points_allowed_per_game
                }

            logger.info(f"[SPORTS_PROPS] NBA: {len(nba_stats)}, NFL: {len(nfl_stats)} teams")

        except Exception as e:
            logger.error(f"[SPORTS_PROPS] Error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def fetch_awards_data(self) -> Dict[str, Any]:
        """Fetch awards predictions from Gold Derby"""
        logger.info("[AWARDS] Fetching predictions...")
        result = {
            'predictions': {},
            'probability_estimates': [],
            'status': 'ok'
        }

        try:
            # Fetch predictions for each show
            for show in ['OSCAR', 'GOLDEN_GLOBE', 'EMMY']:
                predictions = self.awards_scraper.get_all_predictions(show)
                result['predictions'][show] = []

                for pred in predictions:
                    result['predictions'][show].append({
                        'category': pred.category,
                        'predicted_winner': pred.predicted_winner,
                        'win_probability': pred.win_probability,
                        'confidence': pred.confidence,
                        'all_probabilities': pred.all_probabilities
                    })

            total = sum(len(preds) for preds in result['predictions'].values())
            logger.info(f"[AWARDS] Got {total} predictions across shows")

        except Exception as e:
            logger.error(f"[AWARDS] Error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def fetch_climate_data(self) -> Dict[str, Any]:
        """Fetch climate data from NOAA"""
        logger.info("[CLIMATE] Fetching temperature data...")
        result = {
            'current_month': None,
            'current_year': None,
            'record_probabilities': {},
            'probability_estimates': [],
            'status': 'ok'
        }

        try:
            summary = self.climate_scraper.get_climate_summary()

            result['current_month'] = summary.get('current_month')
            result['current_year'] = summary.get('current_year')
            result['record_probabilities'] = summary.get('probabilities', {})

            logger.info(f"[CLIMATE] Monthly anomaly: {result['current_month'].get('anomaly') if result['current_month'] else 'N/A'}")

        except Exception as e:
            logger.error(f"[CLIMATE] Error: {e}")
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def fetch_all(self, use_cache: bool = True) -> AggregatedData:
        """
        Fetch data from all sources.

        Args:
            use_cache: If True, return cached data if fresh

        Returns:
            AggregatedData with all fetched data
        """
        # Check cache
        if use_cache:
            cached = self._load_cache()
            if cached:
                logger.info("Using cached aggregated data")
                return cached

        logger.info("=" * 50)
        logger.info("FETCHING ALL DATA SOURCES")
        logger.info("=" * 50)

        # Fetch from all sources
        weather = self.fetch_weather_data()
        time.sleep(0.5)

        economic = self.fetch_economic_data()
        time.sleep(0.5)

        economic_releases = self.fetch_economic_releases_data()
        time.sleep(0.5)

        crypto = self.fetch_crypto_data()
        time.sleep(0.5)

        earnings = self.fetch_earnings_data()
        time.sleep(0.5)

        sports = self.fetch_sports_data()
        time.sleep(0.5)

        sports_props = self.fetch_sports_props_data()
        time.sleep(0.5)

        awards = self.fetch_awards_data()
        time.sleep(0.5)

        climate = self.fetch_climate_data()
        time.sleep(0.5)

        boxoffice = self.fetch_boxoffice_data()

        # Apply probability calibration to all estimates
        # Update each source dict with calibrated probabilities
        sources = [
            ('weather', weather),
            ('economic', economic),
            ('economic_releases', economic_releases),
            ('crypto', crypto),
            ('earnings', earnings),
            ('sports', sports),
            ('sports_props', sports_props),
            ('awards', awards),
            ('climate', climate),
            ('boxoffice', boxoffice)
        ]

        all_estimates = {}
        total_calibrated = 0
        for name, source in sources:
            raw_estimates = source.get('probability_estimates', [])
            calibrated = [calibrate_estimate(est) for est in raw_estimates]
            # Update the source dict with calibrated estimates
            source['probability_estimates'] = calibrated
            all_estimates[name] = calibrated
            total_calibrated += len(calibrated)

        logger.info(f"Applied probability calibration to {total_calibrated} estimates")

        # Create aggregated data
        data = AggregatedData(
            timestamp=datetime.now(timezone.utc),
            weather=weather,
            economic=economic,
            economic_releases=economic_releases,
            crypto=crypto,
            earnings=earnings,
            sports=sports,
            sports_props=sports_props,
            awards=awards,
            climate=climate,
            boxoffice=boxoffice,
            probability_estimates=all_estimates
        )

        # Save to cache
        self._save_cache(data)

        logger.info("=" * 50)
        logger.info("DATA AGGREGATION COMPLETE")
        logger.info("=" * 50)

        return data

    def get_probability_for_ticker(self, ticker: str) -> Optional[float]:
        """
        Get our probability estimate for a specific Kalshi ticker.

        Args:
            ticker: Kalshi market ticker

        Returns:
            Probability (0-1) or None if no estimate
        """
        data = self.fetch_all(use_cache=True)

        # Search through all estimates
        ticker_upper = ticker.upper()

        for category, estimates in data.probability_estimates.items():
            for est in estimates:
                pattern = est.get('ticker_pattern', '').upper()
                # Simple pattern matching
                if pattern.replace('*', '') in ticker_upper or ticker_upper in pattern:
                    return est.get('our_probability')

        return None

    def get_all_estimates_flat(self) -> List[Dict]:
        """
        Get all probability estimates as a flat list.

        Returns:
            List of all probability estimates with category added
        """
        data = self.fetch_all(use_cache=True)

        flat = []
        for category, estimates in data.probability_estimates.items():
            for est in estimates:
                est_copy = dict(est)
                est_copy['category'] = category
                flat.append(est_copy)

        return flat

    def get_summary(self) -> Dict:
        """Get a summary of all aggregated data"""
        data = self.fetch_all(use_cache=True)

        return {
            'timestamp': data.timestamp.isoformat(),
            'weather': {
                'cities': len(data.weather.get('forecasts', {})),
                'estimates': len(data.weather.get('probability_estimates', [])),
                'status': data.weather.get('status')
            },
            'economic': {
                'indicators': len(data.economic.get('indicators', {})),
                'estimates': len(data.economic.get('probability_estimates', [])),
                'status': data.economic.get('status')
            },
            'economic_releases': {
                'cpi': data.economic_releases.get('cpi') is not None,
                'gdp': data.economic_releases.get('gdp') is not None,
                'jobs': data.economic_releases.get('jobs') is not None,
                'estimates': len(data.economic_releases.get('probability_estimates', [])),
                'status': data.economic_releases.get('status')
            },
            'crypto': {
                'prices': len(data.crypto.get('prices', {})),
                'fear_greed': data.crypto.get('fear_greed', {}).get('value') if data.crypto.get('fear_greed') else None,
                'estimates': len(data.crypto.get('probability_estimates', [])),
                'status': data.crypto.get('status')
            },
            'earnings': {
                'upcoming': len(data.earnings.get('upcoming', [])),
                'estimates': len(data.earnings.get('probability_estimates', [])),
                'status': data.earnings.get('status')
            },
            'sports': {
                'games': sum(len(games) for games in data.sports.get('games', {}).values()),
                'estimates': len(data.sports.get('probability_estimates', [])),
                'status': data.sports.get('status')
            },
            'sports_props': {
                'nba_teams': len(data.sports_props.get('nba_stats', {})),
                'nfl_teams': len(data.sports_props.get('nfl_stats', {})),
                'estimates': len(data.sports_props.get('probability_estimates', [])),
                'status': data.sports_props.get('status')
            },
            'awards': {
                'predictions': sum(len(preds) for preds in data.awards.get('predictions', {}).values()),
                'estimates': len(data.awards.get('probability_estimates', [])),
                'status': data.awards.get('status')
            },
            'climate': {
                'monthly_anomaly': data.climate.get('current_month', {}).get('anomaly') if data.climate.get('current_month') else None,
                'estimates': len(data.climate.get('probability_estimates', [])),
                'status': data.climate.get('status')
            },
            'boxoffice': {
                'weekend': len(data.boxoffice.get('weekend', [])),
                'upcoming': len(data.boxoffice.get('upcoming', [])),
                'estimates': len(data.boxoffice.get('probability_estimates', [])),
                'status': data.boxoffice.get('status')
            },
            'total_estimates': sum(
                len(estimates) for estimates in data.probability_estimates.values()
            )
        }


def main():
    """Test the data aggregator"""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("DATA AGGREGATOR TEST")
    print("=" * 60)

    aggregator = DataAggregator()

    print("\n[1] Fetching All Data...")
    print("-" * 40)
    data = aggregator.fetch_all(use_cache=False)

    print("\n[2] Summary:")
    print("-" * 40)
    summary = aggregator.get_summary()

    for category, info in summary.items():
        if isinstance(info, dict):
            print(f"\n{category.upper()}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"{category}: {info}")

    print("\n[3] Sample Probability Estimates:")
    print("-" * 40)
    flat = aggregator.get_all_estimates_flat()

    for est in flat[:10]:
        print(f"  [{est['category'].upper()}] {est.get('ticker_pattern', 'N/A')}")
        print(f"    Probability: {est.get('our_probability', 0):.0%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
