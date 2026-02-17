"""
Alternative Data Integration System
==================================

A comprehensive system for integrating non-traditional data sources
to enhance trading decisions across all asset classes.

Features:
- Social sentiment analysis
- Satellite imagery analysis
- Weather pattern integration
- Economic calendar tracking
- Options flow detection
- Insider trading monitoring
- Crypto on-chain metrics
- News sentiment scoring
- Google Trends analysis
- Data quality scoring
- Real-time caching
- ML model integration

Author: Trading Bot Arsenal
Created: February 2026
"""

from .core.data_manager import AlternativeDataManager
from .core.quality_scorer import DataQualityScorer
from .core.cache_manager import CacheManager

__version__ = "1.0.0"
__all__ = [
    "AlternativeDataManager",
    "DataQualityScorer",
    "CacheManager"
]