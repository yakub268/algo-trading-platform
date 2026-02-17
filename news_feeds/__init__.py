"""
News Feeds Module - Live news integration for trading bots

Provides real-time news data from multiple sources for enhanced trading decisions.
"""

from .aggregator import NewsAggregator
from .sentiment import SentimentAnalyzer
from .cache_manager import CacheManager

__all__ = ['NewsAggregator', 'SentimentAnalyzer', 'CacheManager']