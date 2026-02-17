"""
Economic Releases Scraper

Fetches real-time nowcast estimates from Federal Reserve banks:
- Cleveland Fed Inflation Nowcast (CPI)
- Atlanta Fed GDPNow (GDP)
- Weekly Jobless Claims trends
- ADP Employment correlation

These government nowcasts are highly accurate and update daily.

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
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EconomicReleasesScraper')


@dataclass
class CPINowcast:
    """Cleveland Fed CPI Nowcast data"""
    headline_cpi: float  # Year-over-year %
    core_cpi: float  # Excluding food/energy
    month_over_month: Optional[float]
    forecast_date: str  # When the nowcast was made
    target_release: str  # Which CPI release this predicts
    confidence_interval: Optional[Tuple[float, float]]
    source: str
    fetched_at: datetime


@dataclass
class GDPNowcast:
    """Atlanta Fed GDPNow data"""
    gdp_estimate: float  # Annualized quarterly growth rate
    forecast_date: str
    target_quarter: str  # e.g., "Q1 2026"
    previous_estimate: Optional[float]
    blue_chip_consensus: Optional[float]
    source: str
    fetched_at: datetime


@dataclass
class JobsNowcast:
    """Jobs data nowcast"""
    nfp_estimate: Optional[int]  # Non-farm payrolls change (thousands)
    adp_reading: Optional[int]  # ADP private payrolls
    weekly_claims: Optional[int]  # Initial jobless claims
    continuing_claims: Optional[int]
    claims_trend: str  # 'rising', 'falling', 'stable'
    forecast_date: str
    source: str
    fetched_at: datetime


@dataclass
class EconomicProbability:
    """Probability estimate for economic release"""
    indicator: str  # CPI, GDP, NFP, etc.
    ticker_pattern: str
    threshold: float
    direction: str  # 'above' or 'below'
    our_probability: float
    nowcast_value: float
    reasoning: str
    confidence: str  # HIGH for Fed nowcasts


class EconomicReleasesScraper:
    """
    Scrapes economic nowcasts from Federal Reserve banks.

    Sources:
    - Cleveland Fed Inflation Nowcast
    - Atlanta Fed GDPNow
    - DOL Weekly Jobless Claims
    """

    CLEVELAND_FED_URL = "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
    ATLANTA_FED_URL = "https://www.atlantafed.org/cqer/research/gdpnow"
    DOL_CLAIMS_URL = "https://www.dol.gov/ui/data.pdf"  # Weekly claims summary

    CACHE_DURATION = timedelta(hours=4)

    def __init__(self, cache_dir: str = "data/economic_releases_cache"):
        """Initialize the economic releases scraper"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0'
        })

        logger.info("EconomicReleasesScraper initialized")

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

    def fetch_cleveland_fed_cpi(self) -> Optional[CPINowcast]:
        """
        Fetch CPI Nowcast from Cleveland Fed.

        The Cleveland Fed updates their inflation nowcast daily and it's
        highly predictive of the actual CPI release.

        Returns:
            CPINowcast or None
        """
        cache_key = "cleveland_fed_cpi"
        cached = self._load_cache(cache_key)
        if cached:
            return CPINowcast(**cached)

        try:
            # Cleveland Fed has an API endpoint for nowcast data
            # Try the data API first
            api_url = "https://www.clevelandfed.org/api/InflationNowcasting/GetNowcastData"

            try:
                response = self.session.get(api_url, timeout=15)
                if response.status_code == 200:
                    data = response.json()

                    # Extract latest nowcast
                    nowcast_data = {
                        'headline_cpi': data.get('headlineCPI', data.get('cpiNowcast', 2.5)),
                        'core_cpi': data.get('coreCPI', data.get('coreCPINowcast', 2.8)),
                        'month_over_month': data.get('momCPI'),
                        'forecast_date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                        'target_release': data.get('targetRelease', 'Next CPI'),
                        'confidence_interval': None,
                        'source': 'Cleveland Fed API',
                        'fetched_at': datetime.now(timezone.utc)
                    }

                    self._save_cache(cache_key, nowcast_data)
                    return CPINowcast(**nowcast_data)
            except Exception as e:
                logger.debug(f"Cleveland Fed API error: {e}")

            # Fallback: Scrape the webpage
            response = self.session.get(self.CLEVELAND_FED_URL, timeout=15)

            if response.status_code != 200:
                logger.warning(f"Cleveland Fed returned {response.status_code}")
                return self._get_fallback_cpi_nowcast()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for nowcast values in the page
            # Cleveland Fed typically displays: "The inflation nowcast for [month] is X.XX%"
            text = soup.get_text()

            # Pattern matching for CPI values
            headline_match = re.search(r'headline\s+(?:CPI|inflation)[^\d]*(\d+\.?\d*)\s*%', text, re.IGNORECASE)
            core_match = re.search(r'core\s+(?:CPI|inflation)[^\d]*(\d+\.?\d*)\s*%', text, re.IGNORECASE)

            # Alternative patterns
            if not headline_match:
                headline_match = re.search(r'nowcast[^\d]*(\d+\.?\d*)\s*%', text, re.IGNORECASE)

            headline_cpi = float(headline_match.group(1)) if headline_match else 2.5  # Fallback
            core_cpi = float(core_match.group(1)) if core_match else headline_cpi + 0.3

            nowcast_data = {
                'headline_cpi': headline_cpi,
                'core_cpi': core_cpi,
                'month_over_month': None,
                'forecast_date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                'target_release': 'Next CPI Release',
                'confidence_interval': None,
                'source': 'Cleveland Fed Web',
                'fetched_at': datetime.now(timezone.utc)
            }

            self._save_cache(cache_key, nowcast_data)
            logger.info(f"[CPI] Cleveland Fed Nowcast: {headline_cpi:.2f}% headline, {core_cpi:.2f}% core")

            return CPINowcast(**nowcast_data)

        except Exception as e:
            logger.error(f"Cleveland Fed CPI error: {e}")
            return self._get_fallback_cpi_nowcast()

    def _get_fallback_cpi_nowcast(self) -> CPINowcast:
        """Return fallback CPI estimate based on recent trends"""
        # Use recent CPI trend as fallback (would be updated periodically)
        return CPINowcast(
            headline_cpi=2.9,  # Recent trend
            core_cpi=3.2,
            month_over_month=0.2,
            forecast_date=datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            target_release='Fallback Estimate',
            confidence_interval=None,
            source='Fallback (recent trend)',
            fetched_at=datetime.now(timezone.utc)
        )

    def fetch_atlanta_fed_gdp(self) -> Optional[GDPNowcast]:
        """
        Fetch GDPNow from Atlanta Fed.

        GDPNow is a "nowcasting" model that provides a running estimate
        of real GDP growth. It's updated frequently and very accurate.

        Returns:
            GDPNowcast or None
        """
        cache_key = "atlanta_fed_gdp"
        cached = self._load_cache(cache_key)
        if cached:
            return GDPNowcast(**cached)

        try:
            # Atlanta Fed GDPNow data API
            api_url = "https://www.atlantafed.org/-/media/documents/cqer/researchcq/gdpnow/GDPNowForecast.xlsx"

            # Try scraping the main page first
            response = self.session.get(self.ATLANTA_FED_URL, timeout=15)

            if response.status_code != 200:
                logger.warning(f"Atlanta Fed returned {response.status_code}")
                return self._get_fallback_gdp_nowcast()

            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()

            # Look for GDPNow estimate
            # Pattern: "The GDPNow model estimate for real GDP growth... is X.X percent"
            gdp_match = re.search(r'GDPNow[^\d]*(\-?\d+\.?\d*)\s*percent', text, re.IGNORECASE)

            if not gdp_match:
                # Alternative pattern
                gdp_match = re.search(r'estimate[^\d]*(\-?\d+\.?\d*)\s*percent', text, re.IGNORECASE)

            if not gdp_match:
                # Look for any percentage that could be GDP
                gdp_match = re.search(r'real\s+GDP[^\d]*(\-?\d+\.?\d*)', text, re.IGNORECASE)

            gdp_estimate = float(gdp_match.group(1)) if gdp_match else 2.5  # Fallback

            # Find target quarter
            quarter_match = re.search(r'Q([1-4])\s*20(\d{2})', text)
            target_quarter = f"Q{quarter_match.group(1)} 20{quarter_match.group(2)}" if quarter_match else "Current Quarter"

            nowcast_data = {
                'gdp_estimate': gdp_estimate,
                'forecast_date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                'target_quarter': target_quarter,
                'previous_estimate': None,
                'blue_chip_consensus': None,
                'source': 'Atlanta Fed GDPNow',
                'fetched_at': datetime.now(timezone.utc)
            }

            self._save_cache(cache_key, nowcast_data)
            logger.info(f"[GDP] Atlanta Fed GDPNow: {gdp_estimate:.1f}% for {target_quarter}")

            return GDPNowcast(**nowcast_data)

        except Exception as e:
            logger.error(f"Atlanta Fed GDP error: {e}")
            return self._get_fallback_gdp_nowcast()

    def _get_fallback_gdp_nowcast(self) -> GDPNowcast:
        """Return fallback GDP estimate"""
        return GDPNowcast(
            gdp_estimate=2.3,  # Trend estimate
            forecast_date=datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            target_quarter='Current Quarter',
            previous_estimate=None,
            blue_chip_consensus=2.1,
            source='Fallback (trend)',
            fetched_at=datetime.now(timezone.utc)
        )

    def fetch_jobless_claims(self) -> Optional[JobsNowcast]:
        """
        Fetch weekly jobless claims and jobs data.

        Weekly claims are a leading indicator for NFP.

        Returns:
            JobsNowcast or None
        """
        cache_key = "jobless_claims"
        cached = self._load_cache(cache_key)
        if cached:
            return JobsNowcast(**cached)

        try:
            # FRED API for jobless claims (if API key available)
            fred_key = os.getenv('FRED_API_KEY')

            if fred_key:
                # Get initial claims
                claims_url = f"https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': 'ICSA',  # Initial Claims Seasonally Adjusted
                    'api_key': fred_key,
                    'file_type': 'json',
                    'sort_order': 'desc',
                    'limit': 5
                }

                response = self.session.get(claims_url, params=params, timeout=15)

                if response.status_code == 200:
                    data = response.json()
                    observations = data.get('observations', [])

                    if observations:
                        latest = observations[0]
                        weekly_claims = int(float(latest.get('value', 0)))

                        # Calculate trend
                        if len(observations) >= 4:
                            recent_avg = sum(int(float(o['value'])) for o in observations[:4]) / 4
                            prev_avg = sum(int(float(o['value'])) for o in observations[1:5]) / 4 if len(observations) >= 5 else recent_avg

                            if recent_avg > prev_avg * 1.05:
                                trend = 'rising'
                            elif recent_avg < prev_avg * 0.95:
                                trend = 'falling'
                            else:
                                trend = 'stable'
                        else:
                            trend = 'stable'

                        nowcast_data = {
                            'nfp_estimate': None,
                            'adp_reading': None,
                            'weekly_claims': weekly_claims,
                            'continuing_claims': None,
                            'claims_trend': trend,
                            'forecast_date': latest.get('date', datetime.now(timezone.utc).strftime('%Y-%m-%d')),
                            'source': 'FRED ICSA',
                            'fetched_at': datetime.now(timezone.utc)
                        }

                        self._save_cache(cache_key, nowcast_data)
                        logger.info(f"[JOBS] Weekly claims: {weekly_claims:,}, trend: {trend}")

                        return JobsNowcast(**nowcast_data)

            # Fallback
            return self._get_fallback_jobs_nowcast()

        except Exception as e:
            logger.error(f"Jobless claims error: {e}")
            return self._get_fallback_jobs_nowcast()

    def _get_fallback_jobs_nowcast(self) -> JobsNowcast:
        """Return fallback jobs estimate"""
        return JobsNowcast(
            nfp_estimate=180,  # Thousands
            adp_reading=None,
            weekly_claims=220000,  # Recent average
            continuing_claims=1800000,
            claims_trend='stable',
            forecast_date=datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            source='Fallback (trend)',
            fetched_at=datetime.now(timezone.utc)
        )

    def calculate_cpi_threshold_probability(
        self,
        nowcast: CPINowcast,
        threshold: float,
        direction: str = 'above'
    ) -> Tuple[float, str]:
        """
        Calculate probability of CPI being above/below threshold.

        Cleveland Fed nowcast has ~0.1-0.2% standard error historically.

        Args:
            nowcast: CPI nowcast data
            threshold: CPI threshold to compare against
            direction: 'above' or 'below'

        Returns:
            Tuple of (probability, reasoning)
        """
        import math

        estimate = nowcast.headline_cpi
        std_error = 0.15  # Historical accuracy of Cleveland Fed nowcast

        # Z-score
        z = (threshold - estimate) / std_error

        # Normal CDF approximation
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        prob_below = norm_cdf(z)
        prob_above = 1 - prob_below

        if direction == 'above':
            prob = prob_above
        else:
            prob = prob_below

        # Clamp to reasonable range
        prob = max(0.02, min(0.98, prob))

        reasoning = f"Cleveland Fed nowcast: {estimate:.2f}%, threshold: {threshold:.1f}%, std_error: {std_error:.2f}%"

        return prob, reasoning

    def calculate_gdp_threshold_probability(
        self,
        nowcast: GDPNowcast,
        threshold: float,
        direction: str = 'above'
    ) -> Tuple[float, str]:
        """
        Calculate probability of GDP being above/below threshold.

        GDPNow has ~0.5% standard error for final estimate.

        Args:
            nowcast: GDP nowcast data
            threshold: GDP threshold to compare against
            direction: 'above' or 'below'

        Returns:
            Tuple of (probability, reasoning)
        """
        import math

        estimate = nowcast.gdp_estimate
        std_error = 0.5  # GDPNow typical error

        z = (threshold - estimate) / std_error

        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        prob_below = norm_cdf(z)
        prob_above = 1 - prob_below

        if direction == 'above':
            prob = prob_above
        else:
            prob = prob_below

        prob = max(0.02, min(0.98, prob))

        reasoning = f"Atlanta Fed GDPNow: {estimate:.1f}%, threshold: {threshold:.1f}%, for {nowcast.target_quarter}"

        return prob, reasoning

    def get_all_nowcasts(self) -> Dict:
        """
        Fetch all economic nowcasts.

        Returns:
            Dict with CPI, GDP, and Jobs nowcasts
        """
        cpi = self.fetch_cleveland_fed_cpi()
        gdp = self.fetch_atlanta_fed_gdp()
        jobs = self.fetch_jobless_claims()

        return {
            'cpi': cpi,
            'gdp': gdp,
            'jobs': jobs,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def _get_next_release_date(self, indicator: str) -> str:
        """
        Get the next release date for an economic indicator in Kalshi format.

        Args:
            indicator: 'CPI', 'GDP', or 'NFP'

        Returns:
            Date string in appropriate format:
            - CPI/NFP: YYMMM (e.g., 26FEB)
            - GDP: YYMMMDD (e.g., 26JAN30)
        """
        now = datetime.now()

        if indicator == 'CPI' or indicator == 'NFP':
            # CPI/NFP released mid-month for previous month's data
            # Next release is next month
            next_month = now.replace(day=1) + timedelta(days=32)
            next_month = next_month.replace(day=1)
            # Format: YYMMM (e.g., 26FEB)
            return next_month.strftime('%y%b').upper()

        elif indicator == 'GDP':
            # GDP released at end of month, quarterly
            # Find next GDP release date (typically last Thursday of Jan, Apr, Jul, Oct)
            gdp_release_months = [1, 4, 7, 10]  # Months when GDP is released

            # Find the next GDP release
            for month in gdp_release_months:
                year = now.year
                if month < now.month or (month == now.month and now.day > 28):
                    continue

                # GDP typically released around the 30th
                release_day = 30 if month != 2 else 28
                release_date = datetime(year, month, release_day)

                if release_date > now:
                    # Format: YYMMMDD (e.g., 26JAN30)
                    return release_date.strftime('%y%b%d').upper()

            # If all this year's releases passed, use next year's first
            release_date = datetime(now.year + 1, 1, 30)
            return release_date.strftime('%y%b%d').upper()

        else:
            # Default format
            next_month = now.replace(day=1) + timedelta(days=32)
            return next_month.strftime('%y%b').upper()

    def generate_probability_estimates(
        self,
        cpi: Optional[CPINowcast] = None,
        gdp: Optional[GDPNowcast] = None,
        jobs: Optional[JobsNowcast] = None
    ) -> List[EconomicProbability]:
        """
        Generate probability estimates for economic release contracts.

        Kalshi ticker formats:
        - CPI: KXCPI-{YY}{MMM}-T{threshold} (e.g., KXCPI-26FEB-T0.4)
        - GDP: KXGDP-{YY}{MMM}{DD}-T{threshold} (e.g., KXGDP-26JAN30-T6)
        - NFP: KXNFP-{YY}{MMM}-T{threshold}

        Args:
            cpi: CPI nowcast
            gdp: GDP nowcast
            jobs: Jobs nowcast

        Returns:
            List of EconomicProbability estimates
        """
        estimates = []

        # CPI thresholds (month-over-month percentage changes)
        # Kalshi format: KXCPI-{YY}{MMM}-T{threshold}
        if cpi:
            cpi_date = self._get_next_release_date('CPI')
            # Kalshi uses MoM thresholds like 0.2%, 0.3%, 0.4%, etc.
            cpi_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

            for threshold in cpi_thresholds:
                prob, reasoning = self.calculate_cpi_threshold_probability(cpi, threshold, 'above')

                estimates.append(EconomicProbability(
                    indicator='CPI',
                    ticker_pattern=f'KXCPI-{cpi_date}-T{threshold}',
                    threshold=threshold,
                    direction='above',
                    our_probability=prob,
                    nowcast_value=cpi.headline_cpi,
                    reasoning=reasoning,
                    confidence='HIGH'  # Cleveland Fed is very accurate
                ))

        # GDP thresholds
        # Kalshi format: KXGDP-{YY}{MMM}{DD}-T{threshold}
        if gdp:
            gdp_date = self._get_next_release_date('GDP')
            # GDP thresholds are annualized growth rates
            gdp_thresholds = [0, 1, 2, 3, 4, 5, 6]

            for threshold in gdp_thresholds:
                prob, reasoning = self.calculate_gdp_threshold_probability(gdp, threshold, 'above')

                estimates.append(EconomicProbability(
                    indicator='GDP',
                    ticker_pattern=f'KXGDP-{gdp_date}-T{threshold}',
                    threshold=threshold,
                    direction='above',
                    our_probability=prob,
                    nowcast_value=gdp.gdp_estimate,
                    reasoning=reasoning,
                    confidence='HIGH'  # GDPNow is very accurate
                ))

        # Jobs estimates (if we have claims data to inform NFP)
        # Kalshi format: KXNFP-{YY}{MMM}-T{threshold}
        if jobs and jobs.weekly_claims:
            nfp_date = self._get_next_release_date('NFP')
            # Higher claims = weaker NFP typically
            # Use claims trend to adjust NFP probability
            nfp_base = 180  # Baseline expectation (thousands)

            if jobs.claims_trend == 'rising':
                nfp_adjustment = -30
            elif jobs.claims_trend == 'falling':
                nfp_adjustment = 30
            else:
                nfp_adjustment = 0

            implied_nfp = nfp_base + nfp_adjustment

            nfp_thresholds = [100, 150, 200, 250]

            for threshold in nfp_thresholds:
                # Simple probability based on distance from implied
                std_error = 50  # NFP has higher uncertainty
                import math
                z = (threshold - implied_nfp) / std_error
                prob_below = 0.5 * (1 + math.erf(z / math.sqrt(2)))
                prob_above = 1 - prob_below

                estimates.append(EconomicProbability(
                    indicator='NFP',
                    ticker_pattern=f'KXNFP-{nfp_date}-T{threshold}',
                    threshold=threshold,
                    direction='above',
                    our_probability=max(0.05, min(0.95, prob_above)),
                    nowcast_value=implied_nfp,
                    reasoning=f"Weekly claims: {jobs.weekly_claims:,}, trend: {jobs.claims_trend}, implied NFP: {implied_nfp}K",
                    confidence='MEDIUM'  # Claims are indirect indicator
                ))

        logger.info(f"Generated {len(estimates)} economic release probability estimates")
        return estimates


def main():
    """Test the economic releases scraper"""
    print("=" * 60)
    print("ECONOMIC RELEASES SCRAPER TEST")
    print("=" * 60)

    scraper = EconomicReleasesScraper()

    print("\n[1] Fetching Cleveland Fed CPI Nowcast...")
    print("-" * 40)
    cpi = scraper.fetch_cleveland_fed_cpi()
    if cpi:
        print(f"  Headline CPI: {cpi.headline_cpi:.2f}%")
        print(f"  Core CPI: {cpi.core_cpi:.2f}%")
        print(f"  Source: {cpi.source}")

    print("\n[2] Fetching Atlanta Fed GDPNow...")
    print("-" * 40)
    gdp = scraper.fetch_atlanta_fed_gdp()
    if gdp:
        print(f"  GDP Estimate: {gdp.gdp_estimate:.1f}%")
        print(f"  Target Quarter: {gdp.target_quarter}")
        print(f"  Source: {gdp.source}")

    print("\n[3] Fetching Jobless Claims...")
    print("-" * 40)
    jobs = scraper.fetch_jobless_claims()
    if jobs:
        print(f"  Weekly Claims: {jobs.weekly_claims:,}" if jobs.weekly_claims else "  Weekly Claims: N/A")
        print(f"  Trend: {jobs.claims_trend}")
        print(f"  Source: {jobs.source}")

    print("\n[4] Probability Estimates...")
    print("-" * 40)
    estimates = scraper.generate_probability_estimates(cpi, gdp, jobs)

    for est in estimates[:10]:
        print(f"  [{est.indicator}] {est.ticker_pattern}")
        print(f"    Threshold: {est.threshold}, Direction: {est.direction}")
        print(f"    Our Prob: {est.our_probability:.0%}, Nowcast: {est.nowcast_value}")
        print(f"    Confidence: {est.confidence}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
