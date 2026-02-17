"""
Core Alternative Data Infrastructure
===================================

Base classes and utilities for alternative data integration.
"""

from .base_connector import BaseDataConnector, DataSource
from .data_manager import AlternativeDataManager
from .quality_scorer import DataQualityScorer, QualityMetrics
from .cache_manager import CacheManager
from .rate_limiter import RateLimiter

__all__ = [
    "BaseDataConnector",
    "DataSource",
    "AlternativeDataManager",
    "DataQualityScorer",
    "QualityMetrics",
    "CacheManager",
    "RateLimiter"
]