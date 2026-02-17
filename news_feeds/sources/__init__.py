"""
News Sources - Individual connectors for various news APIs
"""

from .espn_connector import ESPNConnector
from .financial_connector import FinancialNewsConnector
from .reddit_connector import RedditConnector
from .base_connector import BaseNewsConnector

__all__ = [
    'ESPNConnector',
    'FinancialNewsConnector',
    'RedditConnector',
    'BaseNewsConnector'
]