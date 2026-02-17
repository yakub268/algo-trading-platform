"""
Feature Engineering Pipeline
===========================

Central feature engineering system that combines technical indicators,
sentiment analysis, market microstructure, and alternative data features
for ML model training and inference.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import talib
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .technical_indicators import TechnicalIndicatorCalculator
from .sentiment_features import SentimentFeatureExtractor
from .alternative_data import AlternativeDataProcessor

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    lookback_periods: List[int] = None
    technical_indicators: List[str] = None
    use_sentiment: bool = True
    use_volume_profile: bool = True
    use_market_regime: bool = True
    use_alternative_data: bool = True
    scaling_method: str = "robust"  # robust, standard, minmax
    feature_selection: bool = True
    max_features: int = 50

    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50, 100]
        if self.technical_indicators is None:
            self.technical_indicators = [
                'sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'adx',
                'stoch', 'cci', 'williams_r', 'obv', 'vwap'
            ]


@dataclass
class FeatureSet:
    """Container for engineered features"""
    features: pd.DataFrame
    feature_names: List[str]
    target: Optional[pd.Series] = None
    metadata: Optional[Dict] = None
    scaler: Optional[object] = None
    selector: Optional[object] = None


class FeatureEngineer:
    """
    Advanced feature engineering pipeline for financial ML models.

    Combines multiple data sources and feature types:
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Price-based features (returns, volatility, momentum)
    - Volume analysis (OBV, VWAP, volume profile)
    - Market microstructure (bid-ask spreads, order flow)
    - Sentiment analysis (news, social media, earnings calls)
    - Alternative data (economic indicators, sector rotation)
    - Regime detection (volatility regimes, trend states)
    """

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()

        # Initialize component processors
        self.tech_indicators = TechnicalIndicatorCalculator()
        self.sentiment_processor = SentimentFeatureExtractor()
        self.alt_data_processor = AlternativeDataProcessor()

        # Feature processing components
        self.scaler = None
        self.selector = None
        self.pca = None

        # Cache for expensive computations
        self._feature_cache = {}

        logger.info("FeatureEngineer initialized")

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        features = pd.DataFrame(index=df.index)

        # Basic price features
        features['price'] = df['close']
        features['high'] = df['high']
        features['low'] = df['low']
        features['volume'] = df['volume'] if 'volume' in df.columns else 0

        # Returns at multiple horizons
        for period in self.config.lookback_periods:
            features[f'return_{period}d'] = df['close'].pct_change(period)
            features[f'log_return_{period}d'] = np.log(df['close']).diff(period)

        # Volatility features
        for period in [5, 10, 20]:
            returns = df['close'].pct_change()
            features[f'volatility_{period}d'] = returns.rolling(period).std()
            features[f'realized_vol_{period}d'] = np.sqrt((returns**2).rolling(period).sum())

        # Price momentum and mean reversion
        features['momentum_short'] = df['close'] / df['close'].shift(5) - 1
        features['momentum_medium'] = df['close'] / df['close'].shift(20) - 1
        features['momentum_long'] = df['close'] / df['close'].shift(60) - 1

        # Price levels relative to recent history
        for period in [20, 50, 100]:
            features[f'price_percentile_{period}d'] = (
                df['close'].rolling(period).rank(pct=True)
            )

        # Gap analysis
        if 'open' in df.columns:
            features['overnight_return'] = (df['open'] / df['close'].shift(1) - 1)
            features['intraday_return'] = (df['close'] / df['open'] - 1)
            features['gap_size'] = abs(features['overnight_return'])

        return features

    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        features = pd.DataFrame(index=df.index)

        if 'volume' not in df.columns:
            logger.warning("No volume data available")
            return features

        volume = df['volume']
        price = df['close']

        # Volume statistics
        for period in [5, 10, 20]:
            features[f'volume_sma_{period}'] = volume.rolling(period).mean()
            features[f'volume_ratio_{period}'] = volume / volume.rolling(period).mean()

        # Volume momentum
        features['volume_momentum'] = volume / volume.shift(5)

        # Price-volume relationship
        features['volume_weighted_price'] = (price * volume).rolling(20).sum() / volume.rolling(20).sum()
        features['relative_volume'] = volume / volume.rolling(50).mean()

        # Volume profile features (if high/low data available)
        if all(col in df.columns for col in ['high', 'low']):
            # Simplified volume profile
            price_range = df['high'] - df['low']
            features['volume_density'] = volume / (price_range + 1e-8)
            features['volume_at_price'] = volume * (price / df['high'])

        return features

    def create_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime and state features"""
        features = pd.DataFrame(index=df.index)

        returns = df['close'].pct_change()

        # Trend strength indicators
        for period in [10, 20, 50]:
            # Trend consistency (how often returns have same sign)
            trend_consistency = returns.rolling(period).apply(
                lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
            )
            features[f'trend_consistency_{period}'] = trend_consistency

            # Trend strength (average return magnitude)
            features[f'trend_strength_{period}'] = abs(returns).rolling(period).mean()

        # Volatility regime detection
        vol_20d = returns.rolling(20).std()
        vol_percentile = vol_20d.rolling(252).rank(pct=True)  # Yearly percentile

        features['volatility_regime'] = pd.cut(
            vol_percentile,
            bins=[0, 0.25, 0.75, 1.0],
            labels=[0, 1, 2]  # Low, medium, high vol regimes
        ).astype(float)

        # Market stress indicators
        features['vol_spike'] = (vol_20d > vol_20d.rolling(60).quantile(0.95)).astype(int)
        features['extreme_move'] = (abs(returns) > returns.rolling(60).quantile(0.95)).astype(int)

        # Correlation with market (if we have SPY data)
        try:
            import yfinance as yf
            spy_data = yf.download("SPY", period="2y", progress=False)
            if len(spy_data) > 0:
                spy_returns = spy_data['Close'].pct_change()

                # Align dates
                aligned_data = pd.concat([returns, spy_returns], axis=1, join='inner')
                aligned_data.columns = ['asset', 'spy']

                # Rolling correlation with market
                features['market_beta'] = aligned_data['asset'].rolling(60).corr(aligned_data['spy'])
                features['market_alpha'] = (
                    aligned_data['asset'].rolling(20).mean() -
                    aligned_data['spy'].rolling(20).mean()
                )
        except Exception as e:
            logger.debug(f"Could not calculate market features: {e}")
            features['market_beta'] = 0.5
            features['market_alpha'] = 0.0

        return features

    def create_target_variable(self, df: pd.DataFrame,
                             target_type: str = "direction",
                             horizon: int = 5) -> pd.Series:
        """
        Create target variable for ML training.

        Args:
            df: Price data
            target_type: "direction", "return", "volatility", or "regime"
            horizon: Forward-looking periods

        Returns:
            Target variable series
        """
        price = df['close']

        if target_type == "direction":
            # Multi-class direction: -1 (down), 0 (sideways), 1 (up)
            future_return = price.shift(-horizon) / price - 1

            # Define thresholds for sideways movement
            threshold = future_return.rolling(252).std().median()  # Adaptive threshold

            target = pd.Series(index=df.index, dtype=int)
            target[future_return > threshold] = 1   # Up
            target[future_return < -threshold] = -1  # Down
            target[abs(future_return) <= threshold] = 0  # Sideways

        elif target_type == "return":
            # Continuous returns
            target = (price.shift(-horizon) / price - 1) * 100  # Percentage returns

        elif target_type == "volatility":
            # Future realized volatility
            returns = price.pct_change()
            target = returns.shift(-horizon).rolling(horizon).std() * np.sqrt(252) * 100

        elif target_type == "regime":
            # Volatility regime classification
            returns = price.pct_change()
            future_vol = returns.shift(-horizon).rolling(horizon).std()
            vol_percentile = future_vol.rolling(252).rank(pct=True)

            target = pd.cut(
                vol_percentile,
                bins=[0, 0.33, 0.67, 1.0],
                labels=[0, 1, 2]  # Low, medium, high vol
            ).astype(float)

        else:
            raise ValueError(f"Unknown target type: {target_type}")

        return target

    def engineer_features(self, df: pd.DataFrame,
                         target_type: str = "direction",
                         target_horizon: int = 5) -> FeatureSet:
        """
        Main feature engineering pipeline.

        Args:
            df: Input price data with OHLCV columns
            target_type: Type of target variable to create
            target_horizon: Forward-looking periods for target

        Returns:
            FeatureSet with engineered features and target
        """
        logger.info(f"Engineering features for {len(df)} samples")

        # Flatten MultiIndex columns (yfinance returns ('Close', 'BTC-USD') tuples)
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        # Normalize column names to lowercase (yfinance returns uppercase)
        df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]

        # Validate input data
        required_columns = ['close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input data must contain columns: {required_columns}")

        # Create feature components
        feature_dfs = []

        # 1. Price-based features
        logger.debug("Creating price features...")
        price_features = self.create_price_features(df)
        feature_dfs.append(price_features)

        # 2. Technical indicators
        logger.debug("Creating technical indicator features...")
        tech_features = self.tech_indicators.calculate_all(df)
        feature_dfs.append(tech_features)

        # 3. Volume features
        logger.debug("Creating volume features...")
        volume_features = self.create_volume_features(df)
        feature_dfs.append(volume_features)

        # 4. Market regime features
        logger.debug("Creating market regime features...")
        regime_features = self.create_market_regime_features(df)
        feature_dfs.append(regime_features)

        # 5. Sentiment features (if enabled)
        if self.config.use_sentiment:
            try:
                logger.debug("Creating sentiment features...")
                sentiment_features = self.sentiment_processor.extract_features(df)
                if not sentiment_features.empty:
                    feature_dfs.append(sentiment_features)
            except Exception as e:
                logger.warning(f"Failed to create sentiment features: {e}")

        # 6. Alternative data features (if enabled)
        if self.config.use_alternative_data:
            try:
                logger.debug("Creating alternative data features...")
                alt_features = self.alt_data_processor.extract_features(df)
                if not alt_features.empty:
                    feature_dfs.append(alt_features)
            except Exception as e:
                logger.warning(f"Failed to create alternative data features: {e}")

        # Combine all features
        combined_features = pd.concat(feature_dfs, axis=1)

        # Create target variable
        target = self.create_target_variable(df, target_type, target_horizon)

        # Clean features
        features_clean = self._clean_features(combined_features)

        # Feature selection (if enabled)
        selector = None
        if self.config.feature_selection:
            features_clean, selector = self._select_features(features_clean, target)
            self.selector = selector

        # Feature scaling
        features_scaled, scaler = self._scale_features(features_clean)
        self.scaler = scaler

        # Create metadata
        metadata = {
            'target_type': target_type,
            'target_horizon': target_horizon,
            'n_samples': len(features_scaled),
            'n_features': len(features_scaled.columns),
            'feature_engineering_date': datetime.now().isoformat(),
            'config': self.config
        }

        logger.info(f"Feature engineering complete: {len(features_scaled.columns)} features, {len(features_scaled)} samples")

        return FeatureSet(
            features=features_scaled,
            feature_names=list(features_scaled.columns),
            target=target,
            metadata=metadata,
            scaler=scaler,
            selector=selector
        )

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess features"""
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)

        # Forward fill then backward fill NaNs
        features = features.fillna(method='ffill').fillna(method='bfill')

        # Remove features with too many NaNs (>50%)
        nan_threshold = len(features) * 0.5
        features = features.dropna(axis=1, thresh=nan_threshold)

        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            logger.debug(f"Removing {len(constant_features)} constant features")
            features = features.drop(columns=constant_features)

        return features

    def _scale_features(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
        """Scale features using specified method"""
        if self.config.scaling_method == "standard":
            scaler = StandardScaler()
        elif self.config.scaling_method == "robust":
            scaler = RobustScaler()
        elif self.config.scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            logger.warning(f"Unknown scaling method: {self.config.scaling_method}. Using robust.")
            scaler = RobustScaler()

        # Fit and transform
        scaled_values = scaler.fit_transform(features)
        scaled_df = pd.DataFrame(
            scaled_values,
            index=features.index,
            columns=features.columns
        )

        return scaled_df, scaler

    def _select_features(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, object]:
        """Select most important features"""
        # Align features and target
        aligned_data = pd.concat([features, target], axis=1, join='inner')
        aligned_features = aligned_data.iloc[:, :-1]
        aligned_target = aligned_data.iloc[:, -1]

        # Remove NaN targets
        valid_idx = aligned_target.notna()
        clean_features = aligned_features[valid_idx]
        clean_target = aligned_target[valid_idx]

        if len(clean_features) == 0:
            logger.warning("No valid samples for feature selection")
            return features, None

        # Use mutual information for feature selection
        k_best = min(self.config.max_features, len(clean_features.columns))
        selector = SelectKBest(score_func=mutual_info_regression, k=k_best)

        try:
            selected_features = selector.fit_transform(clean_features, clean_target)
            selected_columns = clean_features.columns[selector.get_support()]

            selected_df = pd.DataFrame(
                selected_features,
                index=clean_features.index,
                columns=selected_columns
            )

            # Reindex to match original features index
            selected_df = selected_df.reindex(features.index, method='nearest')

            logger.info(f"Selected {len(selected_columns)} features from {len(features.columns)}")
            return selected_df, selector

        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return features, None

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scalers and selectors"""
        if self.scaler is None:
            raise ValueError("FeatureEngineer not fitted. Call engineer_features first.")

        # Flatten MultiIndex columns (yfinance returns ('Close', 'BTC-USD') tuples)
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        # Normalize column names to lowercase (yfinance returns uppercase)
        df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]

        # Engineer features (without target)
        feature_dfs = []

        # Price features
        price_features = self.create_price_features(df)
        feature_dfs.append(price_features)

        # Technical indicators
        tech_features = self.tech_indicators.calculate_all(df)
        feature_dfs.append(tech_features)

        # Volume features
        volume_features = self.create_volume_features(df)
        feature_dfs.append(volume_features)

        # Market regime features
        regime_features = self.create_market_regime_features(df)
        feature_dfs.append(regime_features)

        # Sentiment features (if available)
        if self.config.use_sentiment:
            try:
                sentiment_features = self.sentiment_processor.extract_features(df)
                if not sentiment_features.empty:
                    feature_dfs.append(sentiment_features)
            except Exception as e:
                logger.debug(f"Error extracting sentiment features: {e}")

        # Alternative data features (if available)
        if self.config.use_alternative_data:
            try:
                alt_features = self.alt_data_processor.extract_features(df)
                if not alt_features.empty:
                    feature_dfs.append(alt_features)
            except Exception as e:
                logger.debug(f"Error extracting alternative data features: {e}")

        # Combine features
        combined_features = pd.concat(feature_dfs, axis=1)

        # Clean features
        features_clean = self._clean_features(combined_features)

        # Apply feature selection (if fitted)
        if self.selector is not None:
            try:
                # Ensure we have the same columns as training
                expected_columns = self.selector.get_feature_names_out() if hasattr(self.selector, 'get_feature_names_out') else None
                if expected_columns is not None:
                    # Select only the columns that were selected during training
                    missing_cols = set(expected_columns) - set(features_clean.columns)
                    if missing_cols:
                        logger.warning(f"Missing columns for feature selection: {missing_cols}")
                        # Add missing columns with zeros
                        for col in missing_cols:
                            features_clean[col] = 0

                    features_clean = features_clean[expected_columns]

            except Exception as e:
                logger.warning(f"Could not apply feature selection to new data: {e}")

        # Apply scaling
        scaled_values = self.scaler.transform(features_clean)
        scaled_df = pd.DataFrame(
            scaled_values,
            index=features_clean.index,
            columns=features_clean.columns
        )

        return scaled_df

    def get_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> pd.Series:
        """Calculate feature importance using mutual information"""
        # Align data
        aligned_data = pd.concat([features, target], axis=1, join='inner')
        aligned_features = aligned_data.iloc[:, :-1]
        aligned_target = aligned_data.iloc[:, -1]

        # Remove NaN targets
        valid_idx = aligned_target.notna()
        clean_features = aligned_features[valid_idx]
        clean_target = aligned_target[valid_idx]

        if len(clean_features) == 0:
            return pd.Series(index=features.columns, data=0)

        try:
            # Calculate mutual information
            mi_scores = mutual_info_regression(clean_features, clean_target, random_state=42)
            importance = pd.Series(mi_scores, index=clean_features.columns)
            return importance.sort_values(ascending=False)
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            return pd.Series(index=features.columns, data=0)


def main():
    """Test feature engineering pipeline"""
    import yfinance as yf

    # Download test data
    print("Downloading test data...")
    data = yf.download("AAPL", period="1y", progress=False)

    # Initialize feature engineer
    config = FeatureConfig(
        technical_indicators=['sma', 'ema', 'rsi', 'macd', 'bollinger'],
        use_sentiment=False,  # Disable for testing
        use_alternative_data=False,  # Disable for testing
        max_features=30
    )

    engineer = FeatureEngineer(config)

    # Engineer features
    print("Engineering features...")
    feature_set = engineer.engineer_features(data, target_type="direction")

    print(f"\nFeature Engineering Results:")
    print(f"Features shape: {feature_set.features.shape}")
    print(f"Target shape: {feature_set.target.shape}")
    print(f"Feature names: {feature_set.feature_names[:10]}...")  # First 10

    # Calculate feature importance
    importance = engineer.get_feature_importance(feature_set.features, feature_set.target)
    print(f"\nTop 10 Most Important Features:")
    print(importance.head(10))

    # Test transformation of new data
    print("\nTesting new data transformation...")
    new_data = data.tail(30)  # Last 30 days
    transformed = engineer.transform_new_data(new_data)
    print(f"Transformed data shape: {transformed.shape}")

    print("\nFeature engineering test completed successfully!")


if __name__ == "__main__":
    main()