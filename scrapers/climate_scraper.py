"""
Climate Scraper

Scrapes NOAA climate data for temperature record markets:
- Global temperature anomalies
- Hottest/coldest month records
- Regional temperature records
- Sea surface temperatures

Data sources:
1. NOAA Climate at a Glance (global/regional temp data)
2. NASA GISS Surface Temperature Analysis
3. NOAA National Centers for Environmental Information

Author: Trading Bot
Created: January 2026
"""

import os
import re
import logging
import requests
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ClimateScraper')


@dataclass
class TemperatureAnomaly:
    """Global or regional temperature anomaly data"""
    period: str  # e.g., "January 2026", "2025", "Q4 2025"
    anomaly: float  # Temperature anomaly in °C (relative to baseline)
    baseline_period: str  # e.g., "1901-2000"
    rank: Optional[int]  # Rank (1 = warmest on record)
    region: str  # "Global", "US", etc.
    source: str
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MonthlyRecord:
    """Monthly temperature record data"""
    month: str  # "January", "February", etc.
    year: int
    is_record: bool  # True if this was a record-breaking month
    anomaly: float  # Temperature anomaly
    previous_record_year: Optional[int]
    previous_record_anomaly: Optional[float]
    percentile: float  # Where this month ranks (0-100)


@dataclass
class ClimateEstimate:
    """Estimated climate probability"""
    metric: str  # "hottest_year", "monthly_record", etc.
    threshold: Optional[float]
    probability: float
    confidence: str
    reasoning: str


class ClimateScraper:
    """
    Scrapes NOAA and NASA climate data.

    Provides temperature anomaly data and record tracking.
    """

    # NOAA Climate at a Glance API
    NOAA_BASE = "https://www.ncei.noaa.gov/cag/global/time-series"

    # NASA GISS URLs
    NASA_GISS_BASE = "https://data.giss.nasa.gov/gistemp"

    # Historical temperature anomaly data (baseline 1901-2000)
    # Recent years' anomalies for reference
    HISTORICAL_ANOMALIES = {
        2024: 1.29,  # Hottest year on record
        2023: 1.18,
        2022: 0.89,
        2021: 0.84,
        2020: 1.02,
        2019: 0.95,
        2018: 0.82,
        2017: 0.91,
        2016: 1.00,
        2015: 0.93,
    }

    # Monthly records (anomaly in °C, baseline 1901-2000)
    MONTHLY_RECORDS = {
        'January': {'year': 2024, 'anomaly': 1.27},
        'February': {'year': 2024, 'anomaly': 1.28},
        'March': {'year': 2024, 'anomaly': 1.32},
        'April': {'year': 2024, 'anomaly': 1.32},
        'May': {'year': 2024, 'anomaly': 1.18},
        'June': {'year': 2024, 'anomaly': 1.20},
        'July': {'year': 2024, 'anomaly': 1.21},
        'August': {'year': 2024, 'anomaly': 1.26},
        'September': {'year': 2024, 'anomaly': 1.28},
        'October': {'year': 2024, 'anomaly': 1.27},
        'November': {'year': 2024, 'anomaly': 1.21},
        'December': {'year': 2024, 'anomaly': 1.17},
    }

    def __init__(self):
        """Initialize the climate scraper"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) TradingBot/1.0'
        })

        # Cache for fetched data
        self._cache: Dict[str, TemperatureAnomaly] = {}
        self._cache_time: Optional[datetime] = None

        logger.info("ClimateScraper initialized")

    def fetch_global_temperature(self, year: Optional[int] = None, month: Optional[int] = None) -> Optional[TemperatureAnomaly]:
        """
        Fetch global temperature anomaly data from NOAA.

        Args:
            year: Year to fetch (default: current year)
            month: Month to fetch (1-12, default: latest available)

        Returns:
            TemperatureAnomaly or None
        """
        try:
            now = datetime.now(timezone.utc)
            year = year or now.year
            month = month or now.month

            # NOAA Climate at a Glance API format
            # /globe/land_ocean/ytd/12/1880-2024.json
            url = f"{self.NOAA_BASE}/globe/land_ocean/1/{month}/1880-{year}.json"

            response = self.session.get(url, timeout=15)

            if response.status_code == 200:
                data = response.json()

                # Parse NOAA response
                if 'data' in data and str(year) in data['data']:
                    anomaly_str = data['data'][str(year)]
                    anomaly = float(anomaly_str) if anomaly_str else None

                    if anomaly is not None:
                        month_name = datetime(year, month, 1).strftime('%B')
                        return TemperatureAnomaly(
                            period=f"{month_name} {year}",
                            anomaly=anomaly,
                            baseline_period="1901-2000",
                            rank=self._calculate_rank(anomaly, month),
                            region="Global",
                            source="NOAA NCEI"
                        )

        except Exception as e:
            logger.debug(f"Error fetching NOAA data: {e}")

        # Fall back to estimated data
        return self._get_estimated_anomaly(year, month)

    def _get_estimated_anomaly(self, year: int, month: int) -> TemperatureAnomaly:
        """
        Get estimated anomaly based on trends.

        Uses recent warming trend to estimate current anomalies.
        """
        # Recent warming trend: ~0.02°C per year
        base_anomaly = self.HISTORICAL_ANOMALIES.get(2024, 1.29)
        years_diff = year - 2024
        trend_adjustment = years_diff * 0.02

        # Seasonal variation (slightly warmer in boreal summer)
        seasonal = {
            1: 0.02, 2: 0.03, 3: 0.05, 4: 0.02, 5: -0.01, 6: -0.02,
            7: -0.02, 8: -0.01, 9: 0.02, 10: 0.03, 11: 0.02, 12: 0.01
        }

        estimated = base_anomaly + trend_adjustment + seasonal.get(month, 0)
        month_name = datetime(year, month, 1).strftime('%B')

        return TemperatureAnomaly(
            period=f"{month_name} {year}",
            anomaly=round(estimated, 2),
            baseline_period="1901-2000",
            rank=self._calculate_rank(estimated, month),
            region="Global",
            source="Estimated (trend-based)"
        )

    def _calculate_rank(self, anomaly: float, month: int) -> int:
        """Calculate approximate rank for a given anomaly"""
        month_name = datetime(2024, month, 1).strftime('%B')
        record = self.MONTHLY_RECORDS.get(month_name, {})
        record_anomaly = record.get('anomaly', 1.20)

        if anomaly >= record_anomaly:
            return 1  # Likely new record
        elif anomaly >= record_anomaly - 0.05:
            return 2
        elif anomaly >= record_anomaly - 0.10:
            return 3
        elif anomaly >= record_anomaly - 0.15:
            return 5
        else:
            return 10

    def fetch_annual_temperature(self, year: int) -> Optional[TemperatureAnomaly]:
        """
        Fetch annual global temperature anomaly.

        Args:
            year: Year to fetch

        Returns:
            TemperatureAnomaly or None
        """
        try:
            url = f"{self.NOAA_BASE}/globe/land_ocean/ann/12/1880-{year}.json"
            response = self.session.get(url, timeout=15)

            if response.status_code == 200:
                data = response.json()

                if 'data' in data and str(year) in data['data']:
                    anomaly_str = data['data'][str(year)]
                    anomaly = float(anomaly_str) if anomaly_str else None

                    if anomaly is not None:
                        return TemperatureAnomaly(
                            period=str(year),
                            anomaly=anomaly,
                            baseline_period="1901-2000",
                            rank=self._calculate_annual_rank(anomaly),
                            region="Global",
                            source="NOAA NCEI"
                        )

        except Exception as e:
            logger.debug(f"Error fetching annual data: {e}")

        # Fallback
        anomaly = self.HISTORICAL_ANOMALIES.get(year, 1.29 + (year - 2024) * 0.02)
        return TemperatureAnomaly(
            period=str(year),
            anomaly=round(anomaly, 2),
            baseline_period="1901-2000",
            rank=self._calculate_annual_rank(anomaly),
            region="Global",
            source="Estimated"
        )

    def _calculate_annual_rank(self, anomaly: float) -> int:
        """Calculate rank for annual anomaly"""
        # Sort historical anomalies
        all_anomalies = sorted(self.HISTORICAL_ANOMALIES.values(), reverse=True)

        for i, hist_anomaly in enumerate(all_anomalies, 1):
            if anomaly >= hist_anomaly:
                return i

        return len(all_anomalies) + 1

    def check_monthly_record(self, year: int, month: int) -> MonthlyRecord:
        """
        Check if a month is likely to set a record.

        Args:
            year: Year to check
            month: Month (1-12)

        Returns:
            MonthlyRecord with analysis
        """
        month_name = datetime(year, month, 1).strftime('%B')
        record = self.MONTHLY_RECORDS.get(month_name, {})

        record_year = record.get('year', 2024)
        record_anomaly = record.get('anomaly', 1.20)

        # Get current/estimated anomaly
        current = self.fetch_global_temperature(year, month)
        current_anomaly = current.anomaly if current else record_anomaly

        # Is it a record?
        is_record = current_anomaly > record_anomaly

        # Calculate percentile (simplified)
        # Higher anomaly = higher percentile
        base_percentile = 95.0  # Recent months are all in top 5%
        if current_anomaly >= record_anomaly:
            percentile = 100.0
        elif current_anomaly >= record_anomaly - 0.10:
            percentile = 99.0
        else:
            percentile = base_percentile

        return MonthlyRecord(
            month=month_name,
            year=year,
            is_record=is_record,
            anomaly=current_anomaly,
            previous_record_year=record_year,
            previous_record_anomaly=record_anomaly,
            percentile=percentile
        )

    def calculate_record_probability(
        self,
        year: int,
        month: int,
        threshold_type: str = 'record'
    ) -> ClimateEstimate:
        """
        Calculate probability of temperature threshold.

        Args:
            year: Year to estimate
            month: Month (1-12)
            threshold_type: 'record' (new monthly record) or 'top5' (top 5 warmest)

        Returns:
            ClimateEstimate with probability
        """
        month_record = self.check_monthly_record(year, month)

        # Current estimated anomaly
        current_anomaly = month_record.anomaly
        record_anomaly = month_record.previous_record_anomaly or 1.20

        # Standard deviation of temperature anomalies (~0.15°C for recent years)
        std_dev = 0.15

        if threshold_type == 'record':
            # Probability of breaking the record
            # P(X > record) where X ~ N(current, std_dev)
            z = (current_anomaly - record_anomaly) / std_dev

            # If we're already above record, high probability
            if current_anomaly > record_anomaly:
                prob = 0.85 + min(0.14, (current_anomaly - record_anomaly) / 0.20 * 0.14)
            else:
                # Calculate from normal distribution
                prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
                prob = max(0.05, min(0.95, prob))

            confidence = 'HIGH' if abs(current_anomaly - record_anomaly) > 0.10 else 'MEDIUM'

            return ClimateEstimate(
                metric='monthly_record',
                threshold=record_anomaly,
                probability=round(prob, 3),
                confidence=confidence,
                reasoning=f"Current anomaly: {current_anomaly}°C, Record: {record_anomaly}°C ({month_record.previous_record_year})"
            )

        elif threshold_type == 'top5':
            # Probability of being in top 5 warmest
            # Given recent warming, this is very likely
            top5_threshold = record_anomaly - 0.20  # Approximate

            if current_anomaly >= top5_threshold:
                prob = 0.95
            else:
                z = (current_anomaly - top5_threshold) / std_dev
                prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
                prob = max(0.05, min(0.95, prob))

            return ClimateEstimate(
                metric='top5_warmest',
                threshold=top5_threshold,
                probability=round(prob, 3),
                confidence='HIGH',
                reasoning=f"Current anomaly: {current_anomaly}°C, Top 5 threshold: ~{top5_threshold}°C"
            )

        else:
            return ClimateEstimate(
                metric=threshold_type,
                threshold=None,
                probability=0.5,
                confidence='LOW',
                reasoning="Unknown threshold type"
            )

    def calculate_annual_record_probability(self, year: int) -> ClimateEstimate:
        """
        Calculate probability of year being hottest on record.

        Args:
            year: Year to estimate

        Returns:
            ClimateEstimate with probability
        """
        # Get current annual estimate
        annual = self.fetch_annual_temperature(year)
        current_anomaly = annual.anomaly if annual else 1.30

        # 2024 record: 1.29°C
        record_anomaly = 1.29
        std_dev = 0.08  # Annual variability is lower

        z = (current_anomaly - record_anomaly) / std_dev

        if current_anomaly > record_anomaly:
            prob = 0.80 + min(0.19, (current_anomaly - record_anomaly) / 0.10 * 0.19)
        else:
            prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            prob = max(0.05, min(0.95, prob))

        confidence = 'HIGH' if current_anomaly >= record_anomaly - 0.05 else 'MEDIUM'

        return ClimateEstimate(
            metric='hottest_year',
            threshold=record_anomaly,
            probability=round(prob, 3),
            confidence=confidence,
            reasoning=f"Estimated {year} anomaly: {current_anomaly}°C, 2024 record: {record_anomaly}°C"
        )

    def get_climate_summary(self) -> Dict:
        """
        Get summary of current climate data.

        Returns:
            Dict with key climate metrics
        """
        now = datetime.now(timezone.utc)
        year = now.year
        month = now.month

        # Get current data
        monthly = self.fetch_global_temperature(year, month)
        annual = self.fetch_annual_temperature(year)
        monthly_record = self.check_monthly_record(year, month)

        # Calculate probabilities
        record_prob = self.calculate_record_probability(year, month, 'record')
        annual_prob = self.calculate_annual_record_probability(year)

        return {
            'current_month': {
                'period': monthly.period if monthly else f"{month}/{year}",
                'anomaly': monthly.anomaly if monthly else None,
                'rank': monthly.rank if monthly else None,
                'is_record': monthly_record.is_record,
            },
            'current_year': {
                'year': year,
                'anomaly': annual.anomaly if annual else None,
                'rank': annual.rank if annual else None,
            },
            'probabilities': {
                'monthly_record': record_prob.probability,
                'hottest_year': annual_prob.probability,
            },
            'timestamp': now.isoformat()
        }


def main():
    """Test the climate scraper"""
    print("=" * 60)
    print("CLIMATE SCRAPER TEST")
    print("=" * 60)

    scraper = ClimateScraper()

    # Test monthly temperature
    print("\n[1] Fetching current global temperature...")
    now = datetime.now(timezone.utc)
    temp = scraper.fetch_global_temperature(now.year, now.month)
    if temp:
        print(f"  Period: {temp.period}")
        print(f"  Anomaly: {temp.anomaly}°C")
        print(f"  Rank: {temp.rank}")
        print(f"  Source: {temp.source}")

    # Test annual temperature
    print(f"\n[2] Fetching {now.year} annual temperature...")
    annual = scraper.fetch_annual_temperature(now.year)
    if annual:
        print(f"  Year: {annual.period}")
        print(f"  Anomaly: {annual.anomaly}°C")
        print(f"  Rank: {annual.rank}")

    # Test record check
    print(f"\n[3] Checking monthly record probability...")
    record_prob = scraper.calculate_record_probability(now.year, now.month, 'record')
    print(f"  Probability of new record: {record_prob.probability:.0%}")
    print(f"  Confidence: {record_prob.confidence}")
    print(f"  Reasoning: {record_prob.reasoning}")

    # Test annual record
    print(f"\n[4] Checking annual record probability...")
    annual_prob = scraper.calculate_annual_record_probability(now.year)
    print(f"  Probability of hottest year: {annual_prob.probability:.0%}")
    print(f"  Confidence: {annual_prob.confidence}")
    print(f"  Reasoning: {annual_prob.reasoning}")

    # Summary
    print("\n[5] Climate Summary:")
    summary = scraper.get_climate_summary()
    print(f"  Monthly anomaly: {summary['current_month']['anomaly']}°C")
    print(f"  Annual anomaly: {summary['current_year']['anomaly']}°C")
    print(f"  P(monthly record): {summary['probabilities']['monthly_record']:.0%}")
    print(f"  P(hottest year): {summary['probabilities']['hottest_year']:.0%}")


if __name__ == "__main__":
    main()
