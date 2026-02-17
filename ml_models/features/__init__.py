"""
Feature Engineering Package
==========================

Advanced feature engineering for financial ML models.
"""

from .feature_engineer import FeatureEngineer
from .technical_indicators import TechnicalIndicatorCalculator
from .alternative_data import AlternativeDataProcessor
from .sentiment_features import SentimentFeatureExtractor

__all__ = [
    "FeatureEngineer",
    "TechnicalIndicatorCalculator",
    "AlternativeDataProcessor",
    "SentimentFeatureExtractor"
]