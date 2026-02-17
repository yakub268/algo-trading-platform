"""
Weather Scraper using NWS API

Fetches weather forecasts from the National Weather Service API (free, no key needed).
Provides probability estimates for Kalshi weather markets.

Target cities (high-volume Kalshi markets):
- New York City (Central Park)
- Los Angeles (Downtown)
- Chicago (O'Hare)
- Miami
- Philadelphia

Author: Trading Bot
Created: January 2026
"""

import os
import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('WeatherScraper')


@dataclass
class CityForecast:
    """Weather forecast for a city"""
    city: str
    city_code: str  # Kalshi ticker prefix (NY, LA, CHI, etc.)
    date: str  # YYYY-MM-DD
    high_temp: Optional[int]  # Fahrenheit
    low_temp: Optional[int]  # Fahrenheit
    precip_probability: Optional[int]  # 0-100
    conditions: str  # Sunny, Cloudy, Rain, etc.
    detailed_forecast: str
    fetched_at: datetime


@dataclass
class WeatherProbability:
    """Probability estimate for a Kalshi weather contract"""
    ticker_pattern: str  # e.g., "KXHIGHNY-26JAN31-T40"
    city: str
    metric: str  # "high", "low", "rain"
    date: str
    threshold: int  # Temperature threshold or rain threshold
    our_probability: float  # 0-1
    nws_value: int  # The actual NWS forecast value
    reasoning: str


# City configurations with NWS grid points
# To find grid points: GET https://api.weather.gov/points/{lat},{lon}
CITY_CONFIGS = {
    'NYC': {
        'name': 'New York City',
        'kalshi_code': 'NY',
        'lat': 40.7128,
        'lon': -74.0060,
        'nws_office': 'OKX',
        'grid_x': 33,
        'grid_y': 37,
    },
    'LA': {
        'name': 'Los Angeles',
        'kalshi_code': 'LA',
        'lat': 34.0522,
        'lon': -118.2437,
        'nws_office': 'LOX',
        'grid_x': 154,
        'grid_y': 44,
    },
    'Chicago': {
        'name': 'Chicago',
        'kalshi_code': 'CHI',
        'lat': 41.8781,
        'lon': -87.6298,
        'nws_office': 'LOT',
        'grid_x': 65,
        'grid_y': 76,
    },
    'Miami': {
        'name': 'Miami',
        'kalshi_code': 'MIA',
        'lat': 25.7617,
        'lon': -80.1918,
        'nws_office': 'MFL',
        'grid_x': 109,
        'grid_y': 50,
    },
    'Philadelphia': {
        'name': 'Philadelphia',
        'kalshi_code': 'PHIL',
        'lat': 39.95,
        'lon': -75.16,
        'nws_office': 'PHI',
        'grid_x': 50,
        'grid_y': 76,
    },
}


class WeatherScraper:
    """
    Scrapes weather data from NWS API for Kalshi weather market analysis.

    The NWS API is free and requires no API key.
    Rate limit: Be reasonable, they ask for User-Agent identification.
    """

    BASE_URL = "https://api.weather.gov"
    CACHE_DURATION = timedelta(hours=1)

    def __init__(self, cache_dir: str = "data/weather_cache"):
        """Initialize the weather scraper"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingBot/1.0 (weather@example.com)',
            'Accept': 'application/geo+json',
        })

        self.forecast_cache: Dict[str, Tuple[datetime, Dict]] = {}
        logger.info("WeatherScraper initialized")

    def _get_cache_path(self, city: str) -> str:
        """Get cache file path for a city"""
        return os.path.join(self.cache_dir, f"{city.lower()}_forecast.json")

    def _load_cache(self, city: str) -> Optional[Dict]:
        """Load cached forecast if still valid"""
        cache_path = self._get_cache_path(city)
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)

            cached_time = datetime.fromisoformat(data.get('cached_at', ''))
            if datetime.now(timezone.utc) - cached_time < self.CACHE_DURATION:
                logger.debug(f"Using cached forecast for {city}")
                return data.get('forecast')
        except Exception as e:
            logger.debug(f"Cache load error: {e}")

        return None

    def _save_cache(self, city: str, forecast: Dict):
        """Save forecast to cache"""
        cache_path = self._get_cache_path(city)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'cached_at': datetime.now(timezone.utc).isoformat(),
                    'city': city,
                    'forecast': forecast
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Cache save error: {e}")

    def fetch_forecast(self, city: str) -> Optional[Dict]:
        """
        Fetch 7-day forecast from NWS API.

        Args:
            city: City key from CITY_CONFIGS

        Returns:
            Forecast data dictionary
        """
        if city not in CITY_CONFIGS:
            logger.error(f"Unknown city: {city}")
            return None

        # Check cache first
        cached = self._load_cache(city)
        if cached:
            return cached

        config = CITY_CONFIGS[city]
        office = config['nws_office']
        grid_x = config['grid_x']
        grid_y = config['grid_y']

        # Fetch from NWS
        url = f"{self.BASE_URL}/gridpoints/{office}/{grid_x},{grid_y}/forecast"

        try:
            logger.info(f"Fetching NWS forecast for {config['name']}...")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            # Save to cache
            self._save_cache(city, data)

            # Rate limiting - be nice to NWS
            time.sleep(0.5)

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"NWS API error for {city}: {e}")
            return None

    def parse_forecast(self, city: str, forecast_data: Dict) -> List[CityForecast]:
        """
        Parse NWS forecast response into CityForecast objects.

        Args:
            city: City key
            forecast_data: Raw NWS API response

        Returns:
            List of CityForecast objects for next 7 days
        """
        if not forecast_data:
            return []

        config = CITY_CONFIGS.get(city, {})
        periods = forecast_data.get('properties', {}).get('periods', [])

        forecasts = []
        daily_data: Dict[str, Dict] = {}

        for period in periods:
            name = period.get('name', '')
            temp = period.get('temperature')
            is_daytime = period.get('isDaytime', True)
            start_time = period.get('startTime', '')
            precip = period.get('probabilityOfPrecipitation', {}).get('value')
            conditions = period.get('shortForecast', '')
            detailed = period.get('detailedForecast', '')

            # Parse date from start time
            try:
                dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                date_str = dt.strftime('%Y-%m-%d')
            except Exception as e:
                logger.debug(f"Error parsing forecast time: {e}")
                continue

            if date_str not in daily_data:
                daily_data[date_str] = {
                    'high': None,
                    'low': None,
                    'precip': None,
                    'conditions': '',
                    'detailed': '',
                }

            if is_daytime:
                daily_data[date_str]['high'] = temp
                daily_data[date_str]['conditions'] = conditions
                daily_data[date_str]['detailed'] = detailed
                if precip is not None:
                    daily_data[date_str]['precip'] = precip
            else:
                daily_data[date_str]['low'] = temp
                if precip is not None and daily_data[date_str]['precip'] is None:
                    daily_data[date_str]['precip'] = precip

        # Create CityForecast objects
        for date_str, data in sorted(daily_data.items()):
            forecasts.append(CityForecast(
                city=config.get('name', city),
                city_code=config.get('kalshi_code', city[:3].upper()),
                date=date_str,
                high_temp=data['high'],
                low_temp=data['low'],
                precip_probability=data['precip'],
                conditions=data['conditions'],
                detailed_forecast=data['detailed'],
                fetched_at=datetime.now(timezone.utc)
            ))

        return forecasts

    def get_all_forecasts(self) -> Dict[str, List[CityForecast]]:
        """
        Fetch forecasts for all configured cities.

        Returns:
            Dict mapping city key to list of CityForecast objects
        """
        all_forecasts = {}

        for city in CITY_CONFIGS:
            try:
                raw = self.fetch_forecast(city)
                if raw:
                    forecasts = self.parse_forecast(city, raw)
                    all_forecasts[city] = forecasts
                    logger.info(f"  {city}: Got {len(forecasts)} daily forecasts")
            except Exception as e:
                logger.error(f"Error fetching {city}: {e}")
                all_forecasts[city] = []

        return all_forecasts

    def calculate_temperature_probability(
        self,
        forecast_temp: int,
        threshold: int,
        direction: str = 'above',
        uncertainty: int = 3
    ) -> float:
        """
        Calculate probability that actual temp will be above/below threshold.

        Uses a simple normal distribution model with NWS forecast as mean.
        NWS forecasts have typical error of ~3-4F.

        Args:
            forecast_temp: NWS forecast temperature
            threshold: Temperature threshold to compare
            direction: 'above' or 'below'
            uncertainty: Standard deviation in Fahrenheit

        Returns:
            Probability (0-1)
        """
        import math

        # Distance from threshold in standard deviations
        z = (forecast_temp - threshold) / uncertainty

        # Standard normal CDF approximation
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        if direction == 'above':
            # P(actual >= threshold) when forecast is above threshold
            return norm_cdf(z)
        else:
            # P(actual <= threshold)
            return 1 - norm_cdf(z)

    def generate_probability_estimates(
        self,
        forecasts: Dict[str, List[CityForecast]]
    ) -> List[WeatherProbability]:
        """
        Generate probability estimates for potential Kalshi contracts.

        Args:
            forecasts: Dict of city forecasts

        Returns:
            List of WeatherProbability estimates
        """
        estimates = []

        # Kalshi city code mappings (different codes for different contract types)
        # High temp uses: NY, LA→LAX, CHI, MIA, PHIL
        # Low temp uses: NYC, LAX, CHI, MIA, PHIL (note: LOWT not LOW)
        # Rain uses: NYC, LAX, CHI, MIA, PHIL
        kalshi_high_codes = {'NYC': 'NY', 'LA': 'LAX', 'Chicago': 'CHI', 'Miami': 'MIA', 'Philadelphia': 'PHIL'}
        kalshi_low_codes = {'NYC': 'NYC', 'LA': 'LAX', 'Chicago': 'CHI', 'Miami': 'MIA', 'Philadelphia': 'PHIL'}
        kalshi_rain_codes = {'NYC': 'NYC', 'LA': 'LAX', 'Chicago': 'CHI', 'Miami': 'MIA', 'Philadelphia': 'PHIL'}

        for city_key, city_forecasts in forecasts.items():
            config = CITY_CONFIGS.get(city_key, {})

            for forecast in city_forecasts:
                date_str = forecast.date
                # Format date for Kalshi ticker: YY{MON}DD (e.g., "26JAN31")
                try:
                    dt = datetime.strptime(date_str, '%Y-%m-%d')
                    date_code = dt.strftime('%y%b%d').upper()  # e.g., "26JAN31"
                except Exception as e:
                    logger.debug(f"Error parsing date for Kalshi ticker: {e}")
                    continue

                # High temperature estimates
                if forecast.high_temp is not None:
                    high = forecast.high_temp
                    high_code = kalshi_high_codes.get(city_key, city_key[:3].upper())

                    # Common Kalshi temperature thresholds
                    for threshold in [30, 40, 50, 60, 70, 80, 90, 100]:
                        # Skip irrelevant thresholds
                        if abs(high - threshold) > 20:
                            continue

                        prob_above = self.calculate_temperature_probability(
                            high, threshold, 'above'
                        )

                        # Kalshi format: KXHIGHNY-26JAN31-T29
                        estimates.append(WeatherProbability(
                            ticker_pattern=f"KXHIGH{high_code}-{date_code}-T{threshold}",
                            city=forecast.city,
                            metric='high',
                            date=date_str,
                            threshold=threshold,
                            our_probability=prob_above,
                            nws_value=high,
                            reasoning=f"NWS forecast high {high}F, prob >= {threshold}F"
                        ))

                # Low temperature estimates
                if forecast.low_temp is not None:
                    low = forecast.low_temp
                    low_code = kalshi_low_codes.get(city_key, city_key[:3].upper())

                    for threshold in [20, 30, 40, 50, 60, 70, 80]:
                        if abs(low - threshold) > 20:
                            continue

                        prob_above = self.calculate_temperature_probability(
                            low, threshold, 'above'
                        )

                        # Kalshi format: KXLOWTNYC-26JAN31-T5 (note: LOWT not LOW)
                        estimates.append(WeatherProbability(
                            ticker_pattern=f"KXLOWT{low_code}-{date_code}-T{threshold}",
                            city=forecast.city,
                            metric='low',
                            date=date_str,
                            threshold=threshold,
                            our_probability=prob_above,
                            nws_value=low,
                            reasoning=f"NWS forecast low {low}F, prob >= {threshold}F"
                        ))

                # Precipitation estimates
                if forecast.precip_probability is not None:
                    precip = forecast.precip_probability / 100  # Convert to 0-1
                    rain_code = kalshi_rain_codes.get(city_key, city_key[:3].upper())

                    # Kalshi format: KXRAINNYC-26JAN31-T0 (rain needs -T0 suffix)
                    estimates.append(WeatherProbability(
                        ticker_pattern=f"KXRAIN{rain_code}-{date_code}-T0",
                        city=forecast.city,
                        metric='rain',
                        date=date_str,
                        threshold=0,  # Any measurable precipitation
                        our_probability=precip,
                        nws_value=forecast.precip_probability,
                        reasoning=f"NWS precip probability {forecast.precip_probability}%"
                    ))

        return estimates

    def match_to_kalshi_markets(
        self,
        estimates: List[WeatherProbability],
        kalshi_markets: List[Dict]
    ) -> List[Dict]:
        """
        Match our probability estimates to actual Kalshi market tickers.

        Args:
            estimates: Our probability estimates
            kalshi_markets: List of Kalshi weather markets

        Returns:
            List of matched opportunities with edge calculations
        """
        matches = []

        # Build lookup by pattern elements
        estimate_lookup = {}
        for est in estimates:
            # Create multiple keys for flexible matching
            keys = [
                est.ticker_pattern,
                f"{est.metric.upper()}{est.city[:3].upper()}-{est.date[-5:].replace('-', '')}",
            ]
            for key in keys:
                estimate_lookup[key] = est

        for market in kalshi_markets:
            ticker = market.get('ticker', '').upper()
            title = market.get('title', '').lower()

            # Try to find matching estimate
            matched_estimate = None

            # Direct ticker match
            for pattern, est in estimate_lookup.items():
                if pattern in ticker:
                    matched_estimate = est
                    break

            # Parse title for temperature threshold
            if not matched_estimate and 'temperature' in title:
                # Extract city and threshold from title
                for city_key, config in CITY_CONFIGS.items():
                    city_name = config['name'].lower()
                    kalshi_code = config['kalshi_code']

                    if city_name in title or kalshi_code.lower() in title:
                        # Found city, now find threshold
                        import re
                        temp_match = re.search(r'(\d+)\s*(°|degrees|f)', title)
                        if temp_match:
                            threshold = int(temp_match.group(1))
                            # Find matching estimate
                            for est in estimates:
                                if (est.city.lower() == city_name and
                                    est.threshold == threshold):
                                    matched_estimate = est
                                    break

            if matched_estimate:
                matches.append({
                    'market': market,
                    'estimate': matched_estimate,
                    'ticker': ticker,
                    'our_prob': matched_estimate.our_probability,
                })

        return matches


def main():
    """Test the weather scraper"""
    print("=" * 60)
    print("WEATHER SCRAPER TEST")
    print("=" * 60)

    scraper = WeatherScraper()

    print("\n[1] Fetching forecasts for all cities...")
    forecasts = scraper.get_all_forecasts()

    print("\n[2] Forecast Summary:")
    print("-" * 40)

    for city, city_forecasts in forecasts.items():
        config = CITY_CONFIGS.get(city, {})
        print(f"\n{config.get('name', city)} ({config.get('kalshi_code', '')}):")

        for fc in city_forecasts[:3]:  # Show next 3 days
            precip_str = f", {fc.precip_probability}% precip" if fc.precip_probability else ""
            print(f"  {fc.date}: High {fc.high_temp}F, Low {fc.low_temp}F{precip_str}")
            print(f"    {fc.conditions}")

    print("\n[3] Probability Estimates for Kalshi:")
    print("-" * 40)

    estimates = scraper.generate_probability_estimates(forecasts)
    print(f"Generated {len(estimates)} probability estimates")

    # Show some examples
    print("\nSample estimates:")
    for est in estimates[:10]:
        print(f"  {est.ticker_pattern}: {est.our_probability:.1%}")
        print(f"    {est.reasoning}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
