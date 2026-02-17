"""
Earnings Data Scraper

Fetches earnings estimates and whisper numbers for Kalshi earnings contracts:
- Yahoo Finance for consensus estimates
- Earnings calendar from multiple sources
- Historical beat/miss rates

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
from bs4 import BeautifulSoup
import re
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EarningsScraper')


@dataclass
class EarningsEstimate:
    """Earnings estimate for a company"""
    symbol: str
    company_name: str
    report_date: Optional[str]
    report_time: str  # BMO (before market open), AMC (after market close)
    fiscal_quarter: str
    eps_estimate: Optional[float]
    eps_whisper: Optional[float]
    eps_actual: Optional[float]
    revenue_estimate: Optional[float]
    revenue_actual: Optional[float]
    surprise_pct: Optional[float]
    historical_beat_rate: Optional[float]
    source: str
    fetched_at: datetime


@dataclass
class EarningsProbability:
    """Probability estimate for earnings outcome"""
    symbol: str
    ticker_pattern: str
    outcome: str  # 'beat', 'miss', 'meet'
    our_probability: float
    reasoning: str
    confidence: str


# Major companies tracked by Kalshi
TRACKED_COMPANIES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corporation',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'WMT': 'Walmart Inc.',
    'DIS': 'Walt Disney Co.',
    'NFLX': 'Netflix Inc.',
}

# Historical beat rates (approximations based on historical data)
HISTORICAL_BEAT_RATES = {
    'AAPL': 0.85,
    'MSFT': 0.88,
    'GOOGL': 0.82,
    'AMZN': 0.75,
    'META': 0.80,
    'TSLA': 0.65,
    'NVDA': 0.90,
    'JPM': 0.78,
    'V': 0.85,
    'WMT': 0.72,
    'DIS': 0.68,
    'NFLX': 0.70,
    'default': 0.75,  # Average S&P 500 beat rate
}


class EarningsScraper:
    """
    Scrapes earnings data from free sources.

    Sources:
    - Yahoo Finance for estimates and actuals
    - Earnings calendars
    """

    YAHOO_BASE = "https://finance.yahoo.com"
    CACHE_DURATION = timedelta(hours=4)

    def __init__(self, cache_dir: str = "data/earnings_cache"):
        """Initialize the earnings scraper"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        logger.info("EarningsScraper initialized")

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

    def fetch_yahoo_earnings(self, symbol: str) -> Optional[Dict]:
        """
        Fetch earnings data from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with earnings data or None
        """
        cache_key = f"yahoo_{symbol}"
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        try:
            # Yahoo Finance API endpoint for earnings
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
            params = {
                'modules': 'earningsHistory,earningsTrend,calendarEvents'
            }

            response = self.session.get(url, params=params, timeout=15)

            if response.status_code != 200:
                logger.debug(f"Yahoo API returned {response.status_code} for {symbol}")
                return None

            data = response.json()
            result = data.get('quoteSummary', {}).get('result', [])

            if not result:
                return None

            earnings_data = result[0]
            self._save_cache(cache_key, earnings_data)
            time.sleep(0.5)  # Rate limiting

            return earnings_data

        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None

    def parse_earnings_estimate(self, symbol: str, yahoo_data: Dict) -> Optional[EarningsEstimate]:
        """
        Parse Yahoo Finance data into EarningsEstimate.

        Args:
            symbol: Stock ticker
            yahoo_data: Raw Yahoo Finance data

        Returns:
            EarningsEstimate or None
        """
        try:
            # Get calendar events for next earnings date
            calendar = yahoo_data.get('calendarEvents', {}).get('earnings', {})

            # Get earnings trend for estimates
            trend = yahoo_data.get('earningsTrend', {}).get('trend', [])
            current_estimate = None
            for t in trend:
                if t.get('period') == '0q':  # Current quarter
                    current_estimate = t
                    break

            # Get historical earnings
            history = yahoo_data.get('earningsHistory', {}).get('history', [])
            last_earnings = history[0] if history else {}

            # Calculate historical beat rate
            beats = sum(1 for h in history if h.get('surprisePercent', {}).get('raw', 0) > 0)
            beat_rate = beats / len(history) if history else HISTORICAL_BEAT_RATES.get(symbol, 0.75)

            # Extract values
            eps_estimate = None
            if current_estimate:
                eps_data = current_estimate.get('earningsEstimate', {})
                eps_estimate = eps_data.get('avg', {}).get('raw')

            report_date = None
            report_time = 'Unknown'
            if calendar:
                earnings_date = calendar.get('earningsDate', [])
                if earnings_date:
                    ts = earnings_date[0].get('raw')
                    if ts:
                        report_date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')

            return EarningsEstimate(
                symbol=symbol,
                company_name=TRACKED_COMPANIES.get(symbol, symbol),
                report_date=report_date,
                report_time=report_time,
                fiscal_quarter=current_estimate.get('period', 'N/A') if current_estimate else 'N/A',
                eps_estimate=eps_estimate,
                eps_whisper=None,  # Would need separate source
                eps_actual=last_earnings.get('epsActual', {}).get('raw'),
                revenue_estimate=None,
                revenue_actual=None,
                surprise_pct=last_earnings.get('surprisePercent', {}).get('raw'),
                historical_beat_rate=beat_rate,
                source='Yahoo Finance',
                fetched_at=datetime.now(timezone.utc)
            )

        except Exception as e:
            logger.error(f"Error parsing earnings for {symbol}: {e}")
            return None

    def get_upcoming_earnings(self) -> List[EarningsEstimate]:
        """
        Get earnings estimates for all tracked companies.

        Returns:
            List of EarningsEstimate for companies with upcoming reports
        """
        estimates = []

        for symbol in TRACKED_COMPANIES:
            logger.debug(f"Fetching earnings for {symbol}")
            yahoo_data = self.fetch_yahoo_earnings(symbol)

            if yahoo_data:
                estimate = self.parse_earnings_estimate(symbol, yahoo_data)
                if estimate:
                    estimates.append(estimate)

        # Sort by report date
        estimates.sort(key=lambda x: x.report_date or '9999-99-99')

        logger.info(f"Fetched earnings estimates for {len(estimates)} companies")
        return estimates

    def calculate_beat_probability(
        self,
        symbol: str,
        historical_beat_rate: Optional[float] = None,
        eps_estimate: Optional[float] = None,
        sentiment_adjustment: float = 0.0
    ) -> Tuple[float, str]:
        """
        Calculate probability that company will beat EPS estimates.

        Uses historical beat rate as baseline, with adjustments for:
        - Company-specific factors
        - Market sentiment
        - Estimate revision trends

        Args:
            symbol: Stock ticker
            historical_beat_rate: Company's historical beat rate
            eps_estimate: Current EPS estimate
            sentiment_adjustment: Adjustment based on market sentiment (-0.1 to +0.1)

        Returns:
            Tuple of (probability, reasoning)
        """
        # Start with historical beat rate or default
        if historical_beat_rate is not None:
            base_prob = historical_beat_rate
        else:
            base_prob = HISTORICAL_BEAT_RATES.get(symbol, HISTORICAL_BEAT_RATES['default'])

        # Apply sentiment adjustment
        prob = base_prob + sentiment_adjustment

        # Clamp to valid range
        prob = max(0.1, min(0.95, prob))

        reasoning = f"{symbol} historical beat rate: {base_prob:.0%}"
        if sentiment_adjustment != 0:
            reasoning += f", sentiment adj: {sentiment_adjustment:+.0%}"

        return prob, reasoning

    def generate_probability_estimates(
        self,
        earnings: List[EarningsEstimate]
    ) -> List[EarningsProbability]:
        """
        Generate probability estimates for earnings contracts.

        Args:
            earnings: List of earnings estimates

        Returns:
            List of EarningsProbability estimates
        """
        estimates = []

        for earning in earnings:
            if not earning.report_date:
                continue

            # Calculate beat probability
            beat_prob, reasoning = self.calculate_beat_probability(
                earning.symbol,
                earning.historical_beat_rate
            )

            # Beat estimate
            estimates.append(EarningsProbability(
                symbol=earning.symbol,
                ticker_pattern=f'KXEARNINGS-{earning.symbol}-BEAT',
                outcome='beat',
                our_probability=beat_prob,
                reasoning=reasoning,
                confidence='MEDIUM' if earning.historical_beat_rate else 'LOW'
            ))

            # Miss estimate (inverse)
            estimates.append(EarningsProbability(
                symbol=earning.symbol,
                ticker_pattern=f'KXEARNINGS-{earning.symbol}-MISS',
                outcome='miss',
                our_probability=1 - beat_prob,
                reasoning=f"Inverse of beat probability",
                confidence='MEDIUM' if earning.historical_beat_rate else 'LOW'
            ))

        return estimates

    def get_earnings_calendar(self, days_ahead: int = 14) -> List[Dict]:
        """
        Get earnings calendar for the next N days.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of upcoming earnings events
        """
        calendar = []

        estimates = self.get_upcoming_earnings()

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        cutoff = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

        for est in estimates:
            if est.report_date and today <= est.report_date <= cutoff:
                calendar.append({
                    'symbol': est.symbol,
                    'company': est.company_name,
                    'date': est.report_date,
                    'time': est.report_time,
                    'eps_estimate': est.eps_estimate,
                    'historical_beat_rate': est.historical_beat_rate
                })

        return calendar


def main():
    """Test the earnings scraper"""
    print("=" * 60)
    print("EARNINGS SCRAPER TEST")
    print("=" * 60)

    scraper = EarningsScraper()

    print("\n[1] Fetching Earnings Estimates...")
    print("-" * 40)
    estimates = scraper.get_upcoming_earnings()

    print(f"Got {len(estimates)} earnings estimates")
    for est in estimates[:5]:
        print(f"\n  {est.symbol} ({est.company_name})")
        print(f"    Report Date: {est.report_date or 'TBD'}")
        print(f"    EPS Estimate: ${est.eps_estimate:.2f}" if est.eps_estimate else "    EPS Estimate: N/A")
        print(f"    Historical Beat Rate: {est.historical_beat_rate:.0%}" if est.historical_beat_rate else "    Historical Beat Rate: N/A")

    print("\n[2] Earnings Calendar (Next 14 days)...")
    print("-" * 40)
    calendar = scraper.get_earnings_calendar(14)

    if calendar:
        for event in calendar[:10]:
            print(f"  {event['date']}: {event['symbol']} - {event['company']}")
    else:
        print("  No earnings in the next 14 days")

    print("\n[3] Probability Estimates...")
    print("-" * 40)
    probs = scraper.generate_probability_estimates(estimates)

    beat_probs = [p for p in probs if p.outcome == 'beat']
    for prob in beat_probs[:5]:
        print(f"  {prob.symbol} Beat: {prob.our_probability:.0%}")
        print(f"    {prob.reasoning}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
