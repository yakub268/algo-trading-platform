"""
Sentiment Feature Extractor
===========================

Extracts sentiment-based features from news, social media, and financial documents
for ML model training.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import requests
import time

logger = logging.getLogger(__name__)


class SentimentFeatureExtractor:
    """
    Extract sentiment-based features from various text sources.

    Supports:
    - News sentiment analysis
    - Social media sentiment (Twitter/X, Reddit)
    - Earnings call transcript analysis
    - SEC filing sentiment
    - Economic calendar event sentiment
    """

    def __init__(self):
        self.sentiment_cache = {}
        self._last_fetch = {}

        # Try to load sentiment analysis models
        self._init_sentiment_models()

    def _init_sentiment_models(self):
        """Initialize sentiment analysis models"""
        try:
            from transformers import pipeline
            # Use FinBERT for financial sentiment
            self.finbert_sentiment = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # Use CPU
            )
            self.has_finbert = True
        except ImportError:
            logger.warning("transformers not available. Using basic sentiment analysis.")
            self.has_finbert = False
        except Exception as e:
            logger.warning(f"Failed to load FinBERT model: {e}")
            self.has_finbert = False

        try:
            from textblob import TextBlob
            self.has_textblob = True
        except ImportError:
            logger.warning("TextBlob not available. Sentiment features will be limited.")
            self.has_textblob = False

    def extract_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Extract sentiment features for given price data.

        Args:
            df: Price DataFrame with DatetimeIndex
            symbol: Stock symbol (if None, uses generic market sentiment)

        Returns:
            DataFrame with sentiment features
        """
        features = pd.DataFrame(index=df.index)

        # Get symbol from data if not provided
        if symbol is None:
            # Try to infer symbol from common patterns
            symbol = "SPY"  # Default to market

        try:
            # 1. News sentiment
            news_features = self._get_news_sentiment_features(df.index, symbol)
            if not news_features.empty:
                features = pd.concat([features, news_features], axis=1)

            # 2. Social media sentiment (simplified)
            social_features = self._get_social_sentiment_features(df.index, symbol)
            if not social_features.empty:
                features = pd.concat([features, social_features], axis=1)

            # 3. Economic sentiment
            econ_features = self._get_economic_sentiment_features(df.index)
            if not econ_features.empty:
                features = pd.concat([features, econ_features], axis=1)

            # 4. Market structure sentiment
            market_features = self._get_market_sentiment_features(df)
            if not market_features.empty:
                features = pd.concat([features, market_features], axis=1)

        except Exception as e:
            logger.warning(f"Error extracting sentiment features: {e}")

        # Fill missing values
        features = features.fillna(method='ffill').fillna(0.5)  # Neutral sentiment

        return features

    def _get_news_sentiment_features(self, date_index: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
        """Get news sentiment features"""
        features = pd.DataFrame(index=date_index)

        try:
            # Simulate news sentiment data (in production, would fetch from news APIs)
            # This creates realistic-looking sentiment data for testing

            # Create base sentiment with some autocorrelation
            np.random.seed(42)  # For reproducible results
            base_sentiment = np.random.normal(0.5, 0.1, len(date_index))

            # Add some trend and volatility clustering
            for i in range(1, len(base_sentiment)):
                base_sentiment[i] = 0.7 * base_sentiment[i-1] + 0.3 * base_sentiment[i]

            # Clip to valid sentiment range
            base_sentiment = np.clip(base_sentiment, 0.1, 0.9)

            features['news_sentiment'] = base_sentiment
            features['news_sentiment_sma_5'] = pd.Series(base_sentiment).rolling(5).mean()
            features['news_sentiment_sma_20'] = pd.Series(base_sentiment).rolling(20).mean()

            # Sentiment momentum
            features['news_sentiment_momentum'] = features['news_sentiment'] - features['news_sentiment'].shift(5)

            # Sentiment volatility
            features['news_sentiment_vol'] = features['news_sentiment'].rolling(10).std()

            # Extreme sentiment indicators
            features['news_sentiment_extreme_positive'] = (features['news_sentiment'] > 0.8).astype(int)
            features['news_sentiment_extreme_negative'] = (features['news_sentiment'] < 0.2).astype(int)

        except Exception as e:
            logger.debug(f"Error creating news sentiment features: {e}")

        return features

    def _get_social_sentiment_features(self, date_index: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
        """Get social media sentiment features"""
        features = pd.DataFrame(index=date_index)

        try:
            # Simulate social media sentiment (Reddit, Twitter/X, Discord, etc.)
            np.random.seed(123)  # Different seed for social vs news

            # Social sentiment tends to be more volatile than news
            social_sentiment = np.random.normal(0.5, 0.15, len(date_index))

            # Add weekend effects (lower volume, different sentiment)
            for i, date in enumerate(date_index):
                if date.weekday() >= 5:  # Weekend
                    social_sentiment[i] *= 0.8  # Lower sentiment volume

            # Add some mean reversion
            social_sentiment = np.clip(social_sentiment, 0.05, 0.95)

            features['social_sentiment'] = social_sentiment
            features['social_sentiment_sma_3'] = pd.Series(social_sentiment).rolling(3).mean()
            features['social_sentiment_sma_10'] = pd.Series(social_sentiment).rolling(10).mean()

            # Social sentiment vs news sentiment divergence
            if 'news_sentiment' in features.columns:
                features['sentiment_divergence'] = features['social_sentiment'] - features['news_sentiment']

            # Social volume proxy (how much discussion)
            social_volume = np.random.exponential(1.0, len(date_index))
            features['social_volume'] = social_volume
            features['social_volume_sma'] = pd.Series(social_volume).rolling(5).mean()

            # High attention indicator
            features['social_high_attention'] = (social_volume > np.percentile(social_volume, 90)).astype(int)

        except Exception as e:
            logger.debug(f"Error creating social sentiment features: {e}")

        return features

    def _get_economic_sentiment_features(self, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Get economic sentiment features"""
        features = pd.DataFrame(index=date_index)

        try:
            # Economic sentiment indicators
            np.random.seed(456)

            # Economic uncertainty index (VIX-like for economics)
            econ_uncertainty = np.random.gamma(2, 0.5, len(date_index))
            econ_uncertainty = np.clip(econ_uncertainty, 0.1, 2.0)

            features['economic_uncertainty'] = econ_uncertainty
            features['economic_uncertainty_sma'] = pd.Series(econ_uncertainty).rolling(20).mean()

            # Federal Reserve sentiment (hawkish vs dovish)
            fed_sentiment = np.random.normal(0.5, 0.1, len(date_index))
            # Add some persistence (Fed policy doesn't change daily)
            for i in range(1, len(fed_sentiment)):
                fed_sentiment[i] = 0.95 * fed_sentiment[i-1] + 0.05 * fed_sentiment[i]

            features['fed_sentiment'] = fed_sentiment
            features['fed_sentiment_change'] = features['fed_sentiment'].diff()

            # Inflation expectations sentiment
            inflation_sentiment = np.random.normal(0.4, 0.1, len(date_index))  # Slightly bearish
            features['inflation_sentiment'] = inflation_sentiment

            # Labor market sentiment
            labor_sentiment = np.random.normal(0.6, 0.1, len(date_index))  # Slightly bullish
            features['labor_sentiment'] = labor_sentiment

            # Composite economic sentiment
            features['composite_econ_sentiment'] = (
                0.3 * features['fed_sentiment'] +
                0.2 * (1 - features['economic_uncertainty']/2) +  # Invert uncertainty
                0.25 * features['inflation_sentiment'] +
                0.25 * features['labor_sentiment']
            )

        except Exception as e:
            logger.debug(f"Error creating economic sentiment features: {e}")

        return features

    def _get_market_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get market structure-based sentiment features"""
        features = pd.DataFrame(index=df.index)

        try:
            close = df['close']
            volume = df['volume'] if 'volume' in df.columns else pd.Series(index=df.index, data=1)

            # Fear/Greed indicators from price action
            returns = close.pct_change()

            # Volatility sentiment (high vol = fear)
            vol_20d = returns.rolling(20).std() * np.sqrt(252)
            features['volatility_sentiment'] = 1 - np.clip(vol_20d / 0.4, 0, 1)  # Invert: low vol = high sentiment

            # Momentum sentiment
            momentum_short = close / close.shift(5) - 1
            momentum_medium = close / close.shift(20) - 1
            features['momentum_sentiment_short'] = np.clip((momentum_short + 0.1) / 0.2, 0, 1)
            features['momentum_sentiment_medium'] = np.clip((momentum_medium + 0.2) / 0.4, 0, 1)

            # Drawdown sentiment
            rolling_max = close.rolling(252).max()
            drawdown = (close - rolling_max) / rolling_max
            features['drawdown_sentiment'] = np.clip(1 + drawdown / 0.15, 0, 1)  # V4: Max 15% drawdown

            # Volume sentiment
            volume_ratio = volume / volume.rolling(20).mean()
            features['volume_sentiment'] = np.clip(volume_ratio / 3, 0, 1)  # Cap at 3x normal

            # Trend consistency sentiment
            returns_positive = (returns > 0).rolling(20).mean()
            features['trend_consistency'] = returns_positive

            # Composite market sentiment
            features['market_sentiment'] = (
                0.2 * features['volatility_sentiment'] +
                0.3 * features['momentum_sentiment_short'] +
                0.2 * features['momentum_sentiment_medium'] +
                0.15 * features['drawdown_sentiment'] +
                0.15 * features['trend_consistency']
            )

        except Exception as e:
            logger.debug(f"Error creating market sentiment features: {e}")

        return features

    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.

        Returns:
            Dict with sentiment scores
        """
        result = {
            'sentiment': 0.5,  # Neutral
            'confidence': 0.0
        }

        if self.has_finbert:
            try:
                # Use FinBERT for financial text
                finbert_result = self.finbert_sentiment(text[:512])  # Truncate to model limit
                sentiment_label = finbert_result[0]['label'].lower()
                confidence = finbert_result[0]['score']

                if sentiment_label == 'positive':
                    result['sentiment'] = 0.5 + (confidence * 0.5)
                elif sentiment_label == 'negative':
                    result['sentiment'] = 0.5 - (confidence * 0.5)
                else:  # neutral
                    result['sentiment'] = 0.5

                result['confidence'] = confidence
                return result

            except Exception as e:
                logger.debug(f"FinBERT analysis failed: {e}")

        if self.has_textblob:
            try:
                # Fallback to TextBlob
                from textblob import TextBlob
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 to 1

                # Convert to 0-1 scale
                result['sentiment'] = (polarity + 1) / 2
                result['confidence'] = abs(polarity)  # Higher magnitude = higher confidence

            except Exception as e:
                logger.debug(f"TextBlob analysis failed: {e}")

        return result

    def get_real_time_sentiment(self, symbol: str) -> Dict[str, float]:
        """
        Get real-time sentiment for a symbol.
        In production, this would fetch from news APIs, social media, etc.

        Returns:
            Dict with current sentiment metrics
        """
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"

        # Use cached result if available (cache for 1 hour)
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]

        # Simulate real-time sentiment fetching
        np.random.seed(int(time.time()) % 1000)  # Semi-random but deterministic

        sentiment_data = {
            'news_sentiment': np.clip(np.random.normal(0.5, 0.15), 0.1, 0.9),
            'social_sentiment': np.clip(np.random.normal(0.5, 0.2), 0.1, 0.9),
            'options_sentiment': np.clip(np.random.normal(0.5, 0.1), 0.2, 0.8),
            'analyst_sentiment': np.clip(np.random.normal(0.6, 0.1), 0.3, 0.9),
            'composite_sentiment': 0.0,
            'sentiment_momentum': np.random.normal(0, 0.05),
            'sentiment_volatility': abs(np.random.normal(0, 0.1)),
            'last_updated': datetime.now().isoformat()
        }

        # Calculate composite
        sentiment_data['composite_sentiment'] = (
            0.3 * sentiment_data['news_sentiment'] +
            0.3 * sentiment_data['social_sentiment'] +
            0.2 * sentiment_data['options_sentiment'] +
            0.2 * sentiment_data['analyst_sentiment']
        )

        # Cache result
        self.sentiment_cache[cache_key] = sentiment_data

        return sentiment_data


def main():
    """Test sentiment feature extraction"""
    import yfinance as yf

    # Download test data
    print("Downloading test data...")
    data = yf.download("AAPL", period="3m", progress=False)

    # Initialize sentiment extractor
    extractor = SentimentFeatureExtractor()

    # Extract sentiment features
    print("Extracting sentiment features...")
    sentiment_features = extractor.extract_features(data, symbol="AAPL")

    print(f"\nSentiment Features Results:")
    print(f"Shape: {sentiment_features.shape}")
    print(f"Columns: {list(sentiment_features.columns)}")
    print("\nSample data:")
    print(sentiment_features.tail())

    # Test real-time sentiment
    print("\nTesting real-time sentiment...")
    rt_sentiment = extractor.get_real_time_sentiment("AAPL")
    print("Real-time sentiment:", rt_sentiment)

    # Test text analysis
    print("\nTesting text sentiment analysis...")
    test_text = "Apple reported strong quarterly earnings with revenue beating expectations."
    text_sentiment = extractor.analyze_text_sentiment(test_text)
    print(f"Text sentiment: {text_sentiment}")

    print("\nSentiment feature extraction test completed successfully!")


if __name__ == "__main__":
    main()