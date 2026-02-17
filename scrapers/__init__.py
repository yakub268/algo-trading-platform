"""
Scrapers package for external data sources.

Provides probability estimates for Kalshi market edge detection.
"""

from .weather_scraper import WeatherScraper
from .economic_scraper import EconomicScraper
from .crypto_scraper import CryptoScraper
from .earnings_scraper import EarningsScraper
from .sports_scraper import SportsScraper
from .boxoffice_scraper import BoxOfficeScraper
from .data_aggregator import DataAggregator

__all__ = [
    'WeatherScraper',
    'EconomicScraper',
    'CryptoScraper',
    'EarningsScraper',
    'SportsScraper',
    'BoxOfficeScraper',
    'DataAggregator'
]
