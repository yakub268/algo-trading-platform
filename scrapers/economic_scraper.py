"""
Economic Data Scraper

Fetches economic indicator forecasts and consensus estimates:
- FRED API for historical data and current values
- Trading Economics website for consensus forecasts
- Economic calendar for upcoming releases

Target indicators:
- CPI (Consumer Price Index)
- NFP (Non-Farm Payrolls)
- GDP Growth
- Unemployment Rate
- Retail Sales
- Fed Funds Rate

Author: Trading Bot
Created: January 2026
"""

import os
import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EconomicScraper')


@dataclass
class EconomicIndicator:
    """Economic indicator data"""
    name: str
    code: str  # FRED series ID or internal code
    current_value: Optional[float]
    previous_value: Optional[float]
    consensus_forecast: Optional[float]
    forecast_range_low: Optional[float]
    forecast_range_high: Optional[float]
    release_date: Optional[str]
    unit: str  # percent, thousands, billions, etc.
    source: str
    fetched_at: datetime


@dataclass
class EconomicProbability:
    """Probability estimate for economic outcome"""
    indicator: str
    ticker_pattern: str
    threshold: float
    direction: str  # 'above', 'below', 'between'
    our_probability: float
    reasoning: str
    confidence: str  # HIGH, MEDIUM, LOW


# FRED Series IDs for key indicators
FRED_SERIES = {
    'fed_funds': 'DFF',           # Effective Federal Funds Rate
    'cpi': 'CPIAUCSL',            # Consumer Price Index
    'cpi_yoy': 'CPIAUCNS',        # CPI Year-over-Year
    'unemployment': 'UNRATE',      # Unemployment Rate
    'nfp': 'PAYEMS',              # Non-Farm Payrolls (thousands)
    'gdp': 'GDP',                 # Gross Domestic Product
    'gdp_growth': 'A191RL1Q225SBEA',  # Real GDP Growth Rate
    'retail_sales': 'RSAFS',      # Retail Sales
    'pce': 'PCEPI',               # PCE Price Index
    'core_pce': 'PCEPILFE',       # Core PCE (Fed's preferred inflation measure)
    'yield_10y': 'DGS10',         # 10-Year Treasury Yield
    'yield_2y': 'DGS2',           # 2-Year Treasury Yield
    'yield_curve': 'T10Y2Y',      # 10Y-2Y Spread
}

# Trading Economics calendar URLs
TRADING_ECONOMICS_BASE = "https://tradingeconomics.com"


class EconomicScraper:
    """
    Scrapes economic data from multiple sources.

    Primary sources:
    - FRED API (free with key)
    - Trading Economics (web scraping)
    """

    CACHE_DURATION = timedelta(hours=2)

    def __init__(self, fred_api_key: str = None, cache_dir: str = "data/economic_cache"):
        """Initialize the economic scraper"""
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        self.cache: Dict[str, Tuple[datetime, any]] = {}
        logger.info("EconomicScraper initialized")

    def _get_cache_path(self, key: str) -> str:
        """Get cache file path"""
        safe_key = re.sub(r'[^\w\-]', '_', key)
        return os.path.join(self.cache_dir, f"{safe_key}.json")

    def _load_cache(self, key: str) -> Optional[Dict]:
        """Load from cache if valid"""
        # Memory cache first
        if key in self.cache:
            cached_time, data = self.cache[key]
            if datetime.now(timezone.utc) - cached_time < self.CACHE_DURATION:
                return data

        # File cache
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                cached_time = datetime.fromisoformat(data.get('cached_at', ''))
                if datetime.now(timezone.utc) - cached_time < self.CACHE_DURATION:
                    self.cache[key] = (cached_time, data.get('data'))
                    return data.get('data')
            except Exception as e:
                logger.debug(f"Cache load error: {e}")

        return None

    def _save_cache(self, key: str, data: any):
        """Save to cache"""
        now = datetime.now(timezone.utc)
        self.cache[key] = (now, data)

        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'cached_at': now.isoformat(),
                    'key': key,
                    'data': data
                }, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Cache save error: {e}")

    def fetch_fred_series(self, series_id: str, limit: int = 10) -> Optional[List[Dict]]:
        """
        Fetch data from FRED API.

        Args:
            series_id: FRED series identifier
            limit: Number of observations to fetch

        Returns:
            List of observation dicts with 'date' and 'value'
        """
        if not self.fred_api_key:
            logger.warning("No FRED API key available")
            return None

        cache_key = f"fred_{series_id}"
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'sort_order': 'desc',
            'limit': limit
        }

        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            observations = data.get('observations', [])
            result = []
            for obs in observations:
                try:
                    value = float(obs['value']) if obs['value'] != '.' else None
                    result.append({
                        'date': obs['date'],
                        'value': value
                    })
                except (ValueError, KeyError):
                    continue

            self._save_cache(cache_key, result)
            time.sleep(0.3)  # Rate limiting
            return result

        except Exception as e:
            logger.error(f"FRED API error for {series_id}: {e}")
            return None

    def get_latest_indicators(self) -> Dict[str, EconomicIndicator]:
        """
        Get latest values for all key economic indicators.

        Returns:
            Dict mapping indicator name to EconomicIndicator
        """
        indicators = {}

        for name, series_id in FRED_SERIES.items():
            data = self.fetch_fred_series(series_id, limit=5)
            if data and len(data) >= 2:
                current = data[0]
                previous = data[1]

                # Determine unit based on series
                if 'rate' in name.lower() or 'yield' in name.lower() or name in ['cpi_yoy', 'unemployment', 'gdp_growth']:
                    unit = 'percent'
                elif name == 'nfp':
                    unit = 'thousands'
                elif name == 'gdp' or name == 'retail_sales':
                    unit = 'billions'
                else:
                    unit = 'index'

                indicators[name] = EconomicIndicator(
                    name=name,
                    code=series_id,
                    current_value=current['value'],
                    previous_value=previous['value'],
                    consensus_forecast=None,  # Would need scraping
                    forecast_range_low=None,
                    forecast_range_high=None,
                    release_date=current['date'],
                    unit=unit,
                    source='FRED',
                    fetched_at=datetime.now(timezone.utc)
                )

        logger.info(f"Fetched {len(indicators)} economic indicators from FRED")
        return indicators

    def scrape_trading_economics_calendar(self) -> List[Dict]:
        """
        Scrape upcoming economic releases from Trading Economics.

        Returns:
            List of upcoming economic events with consensus estimates
        """
        cache_key = "te_calendar"
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        url = f"{TRADING_ECONOMICS_BASE}/united-states/calendar"

        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            events = []

            # Find calendar table
            table = soup.find('table', {'id': 'calendar'})
            if not table:
                # Try alternative structure
                rows = soup.find_all('tr', {'data-event': True})
            else:
                rows = table.find_all('tr')[1:]  # Skip header

            for row in rows[:20]:  # Limit to next 20 events
                try:
                    cells = row.find_all('td')
                    if len(cells) < 5:
                        continue

                    event = {
                        'date': cells[0].get_text(strip=True) if cells else '',
                        'time': cells[1].get_text(strip=True) if len(cells) > 1 else '',
                        'event': cells[2].get_text(strip=True) if len(cells) > 2 else '',
                        'actual': cells[3].get_text(strip=True) if len(cells) > 3 else '',
                        'forecast': cells[4].get_text(strip=True) if len(cells) > 4 else '',
                        'previous': cells[5].get_text(strip=True) if len(cells) > 5 else '',
                    }

                    # Only include US events with forecasts
                    if event['forecast'] and 'US' in str(row):
                        events.append(event)

                except Exception as e:
                    continue

            self._save_cache(cache_key, events)
            logger.info(f"Scraped {len(events)} upcoming economic events")
            return events

        except Exception as e:
            logger.error(f"Trading Economics scrape error: {e}")
            return []

    def calculate_cpi_probability(
        self,
        current_cpi: float,
        threshold: float,
        direction: str = 'above'
    ) -> Tuple[float, str]:
        """
        Calculate probability for CPI-related contracts.

        Uses historical volatility to estimate probability distribution.
        CPI month-over-month typically has std dev of ~0.2-0.3%

        Args:
            current_cpi: Current CPI value
            threshold: Threshold to compare against
            direction: 'above' or 'below'

        Returns:
            Tuple of (probability, reasoning)
        """
        import math

        # CPI volatility (standard deviation of MoM change)
        cpi_std = 0.25  # ~0.25% typical monthly volatility

        # For YoY comparison, use different volatility
        if threshold > 10:  # Likely a YoY percentage
            cpi_std = 0.3  # Slightly higher for YoY

        z = (current_cpi - threshold) / cpi_std

        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        if direction == 'above':
            prob = norm_cdf(z)
            reasoning = f"Current CPI: {current_cpi:.2f}%, threshold: {threshold}%"
        else:
            prob = 1 - norm_cdf(z)
            reasoning = f"Current CPI: {current_cpi:.2f}%, threshold: {threshold}%"

        return prob, reasoning

    def calculate_unemployment_probability(
        self,
        current_rate: float,
        threshold: float,
        direction: str = 'above'
    ) -> Tuple[float, str]:
        """
        Calculate probability for unemployment rate contracts.

        Unemployment is typically sticky, with ~0.1-0.2% monthly volatility.
        """
        import math

        unemp_std = 0.15  # Typical monthly volatility

        z = (current_rate - threshold) / unemp_std

        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        if direction == 'above':
            prob = norm_cdf(z)
        else:
            prob = 1 - norm_cdf(z)

        reasoning = f"Current unemployment: {current_rate:.1f}%, threshold: {threshold}%"
        return prob, reasoning

    def calculate_nfp_probability(
        self,
        recent_avg: float,
        threshold: float,
        direction: str = 'above'
    ) -> Tuple[float, str]:
        """
        Calculate probability for Non-Farm Payrolls contracts.

        NFP has high volatility, typically 50-100K std dev.
        """
        import math

        nfp_std = 75  # ~75K typical standard deviation

        z = (recent_avg - threshold) / nfp_std

        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        if direction == 'above':
            prob = norm_cdf(z)
        else:
            prob = 1 - norm_cdf(z)

        reasoning = f"Recent NFP avg: {recent_avg:.0f}K, threshold: {threshold}K"
        return prob, reasoning

    def _get_next_release_date(self, indicator: str) -> Tuple[str, str]:
        """
        Get the next release date for an economic indicator.

        Returns:
            Tuple of (cpi_format: YYMMM, gdp_format: YYMMMDD)
        """
        from datetime import datetime, timedelta

        now = datetime.now()

        # CPI is typically released mid-month for the previous month
        # GDP is released at end of month
        if indicator == 'CPI':
            # Next month's release (for current month's data)
            next_month = now.replace(day=1) + timedelta(days=32)
            next_month = next_month.replace(day=1)
            # Format: YYMMM (e.g., 26FEB)
            date_str = next_month.strftime('%y%b').upper()
            return date_str, date_str
        elif indicator == 'GDP':
            # GDP released at end of quarter
            # Find next quarter end
            quarter_end_months = [1, 4, 7, 10]  # GDP releases for previous quarter
            for month in quarter_end_months:
                if now.month < month or (now.month == month and now.day < 28):
                    release_date = now.replace(month=month, day=30)
                    if month == 2:
                        release_date = now.replace(month=month, day=28)
                    break
            else:
                # Next year Q1
                release_date = now.replace(year=now.year + 1, month=1, day=30)
            # Format: YYMMMDD (e.g., 26JAN30)
            date_str = release_date.strftime('%y%b%d').upper()
            return date_str, date_str
        else:
            # Default to next month
            next_month = now.replace(day=1) + timedelta(days=32)
            date_str = next_month.strftime('%y%b').upper()
            return date_str, date_str

    def generate_probability_estimates(
        self,
        indicators: Dict[str, EconomicIndicator]
    ) -> List[EconomicProbability]:
        """
        Generate probability estimates for potential Kalshi economic contracts.

        Args:
            indicators: Dict of current economic indicators

        Returns:
            List of EconomicProbability estimates
        """
        estimates = []

        # Get next release dates for ticker formatting
        cpi_date, _ = self._get_next_release_date('CPI')
        gdp_date, _ = self._get_next_release_date('GDP')

        # CPI estimates
        # Kalshi format: KXCPI-{YY}{MMM}-T{threshold} (e.g., KXCPI-26FEB-T0.4)
        # Note: Kalshi CPI contracts are for MoM change, not YoY
        cpi_mom_estimate = None

        # Try to calculate MoM from CPI index values
        if 'cpi' in indicators and indicators['cpi'].current_value and indicators['cpi'].previous_value:
            current_cpi_index = indicators['cpi'].current_value
            previous_cpi_index = indicators['cpi'].previous_value
            # Calculate MoM percentage change from index values
            cpi_mom_estimate = ((current_cpi_index - previous_cpi_index) / previous_cpi_index) * 100
            mom_source = "calculated from CPI index"
        else:
            # Fall back to historical average MoM (~0.2-0.3%)
            cpi_mom_estimate = 0.25
            mom_source = "historical average estimate"

        if cpi_mom_estimate is not None:
            for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:  # MoM thresholds (percent)
                prob, reasoning = self.calculate_cpi_probability(cpi_mom_estimate, threshold, 'above')
                reasoning = f"Expected MoM CPI: {cpi_mom_estimate:.2f}% ({mom_source}), threshold: {threshold}%"
                estimates.append(EconomicProbability(
                    indicator='CPI',
                    ticker_pattern=f'KXCPI-{cpi_date}-T{threshold}',
                    threshold=threshold,
                    direction='above',
                    our_probability=prob,
                    reasoning=reasoning,
                    confidence='MEDIUM'
                ))

        # Unemployment estimates
        # Kalshi format: KXUNEMPLOYMENT-{YY}{MMM}-T{threshold}
        if 'unemployment' in indicators and indicators['unemployment'].current_value:
            unemp = indicators['unemployment'].current_value
            unemp_date, _ = self._get_next_release_date('CPI')  # Same timing as CPI

            for threshold in [3.5, 4.0, 4.5, 5.0, 5.5]:
                prob, reasoning = self.calculate_unemployment_probability(unemp, threshold, 'above')
                estimates.append(EconomicProbability(
                    indicator='Unemployment',
                    ticker_pattern=f'KXUNEMPLOYMENT-{unemp_date}-T{threshold}',
                    threshold=threshold,
                    direction='above',
                    our_probability=prob,
                    reasoning=reasoning,
                    confidence='MEDIUM'
                ))

        # NFP estimates (if we have recent data)
        # Kalshi format: KXNFP-{YY}{MMM}-T{threshold}
        if 'nfp' in indicators and indicators['nfp'].current_value:
            # Use current value as proxy for recent average
            nfp = indicators['nfp'].current_value
            nfp_date, _ = self._get_next_release_date('CPI')  # NFP released monthly

            # Calculate month-over-month change
            if indicators['nfp'].previous_value:
                nfp_change = nfp - indicators['nfp'].previous_value

                for threshold in [100, 150, 200, 250, 300]:
                    prob, reasoning = self.calculate_nfp_probability(nfp_change, threshold, 'above')
                    estimates.append(EconomicProbability(
                        indicator='NFP',
                        ticker_pattern=f'KXNFP-{nfp_date}-T{threshold}',
                        threshold=threshold,
                        direction='above',
                        our_probability=prob,
                        reasoning=reasoning,
                        confidence='LOW'  # NFP is highly volatile
                    ))

        return estimates


def main():
    """Test the economic scraper"""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("ECONOMIC SCRAPER TEST")
    print("=" * 60)

    scraper = EconomicScraper()

    print("\n[1] Fetching FRED Indicators...")
    print("-" * 40)
    indicators = scraper.get_latest_indicators()

    for name, ind in list(indicators.items())[:8]:
        prev = f" (prev: {ind.previous_value:.2f})" if ind.previous_value else ""
        print(f"  {name}: {ind.current_value:.2f} {ind.unit}{prev}")

    print("\n[2] Generating Probability Estimates...")
    print("-" * 40)
    estimates = scraper.generate_probability_estimates(indicators)

    print(f"Generated {len(estimates)} probability estimates")
    for est in estimates[:10]:
        print(f"  {est.ticker_pattern}: {est.our_probability:.0%}")
        print(f"    {est.reasoning}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
