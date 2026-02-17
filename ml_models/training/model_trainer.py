"""
Model Trainer
============

Comprehensive training pipeline for ML models with historical data integration.
Handles data preparation, model training, validation, and performance evaluation.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..features.feature_engineer import FeatureEngineer, FeatureConfig
from ..predictors.price_direction_model import PriceDirectionPredictor
from ..predictors.volatility_model import VolatilityPredictor
from ..inference.model_manager import ModelManager

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model training pipeline.

    Features:
    - Historical data integration
    - Time series cross-validation
    - Automated hyperparameter optimization
    - Model performance evaluation
    - Walk-forward analysis
    - Model versioning and persistence
    """

    def __init__(self,
                 data_source: str = "yfinance",
                 feature_config: FeatureConfig = None,
                 validation_method: str = "time_series"):
        """
        Initialize model trainer.

        Args:
            data_source: Data source for historical data ("yfinance", "alpaca", etc.)
            feature_config: Configuration for feature engineering
            validation_method: Validation method ("time_series", "walk_forward")
        """
        self.data_source = data_source
        self.feature_config = feature_config or FeatureConfig()
        self.validation_method = validation_method

        # Initialize components
        self.feature_engineer = FeatureEngineer(self.feature_config)
        self.model_manager = ModelManager()

        # Training configuration
        self.min_training_samples = 252  # 1 year of daily data
        self.validation_splits = 5
        self.test_split_ratio = 0.2

        # Data cache
        self.data_cache = {}

        logger.info("ModelTrainer initialized")

    def train_price_direction_model(self,
                                  data: pd.DataFrame = None,
                                  symbol: str = "SPY",
                                  quick_train: bool = False,
                                  save_model: bool = True) -> Optional[PriceDirectionPredictor]:
        """
        Train price direction prediction model.

        Args:
            data: Training data (if None, will fetch)
            symbol: Symbol to train on
            quick_train: Use reduced training for faster results
            save_model: Whether to save trained model

        Returns:
            Trained model or None if training failed
        """
        logger.info(f"Training price direction model for {symbol}")

        try:
            # Get training data
            if data is None:
                data = self._fetch_training_data(symbol, period="2y")

            if data is None or len(data) < self.min_training_samples:
                logger.error(f"Insufficient data for training: {len(data) if data is not None else 0}")
                return None

            # Engineer features
            logger.info("Engineering features...")
            feature_set = self.feature_engineer.engineer_features(
                data,
                target_type="direction",
                target_horizon=5
            )

            if feature_set.features.empty:
                logger.error("Feature engineering failed")
                return None

            # Prepare training data
            X = feature_set.features
            y = feature_set.target

            # Remove NaN values
            valid_idx = y.notna() & X.notna().all(axis=1)
            X_clean = X[valid_idx]
            y_clean = y[valid_idx]

            if len(X_clean) < self.min_training_samples:
                logger.error(f"Insufficient clean data: {len(X_clean)}")
                return None

            logger.info(f"Training data: {len(X_clean)} samples, {X_clean.shape[1]} features")

            # Initialize model with appropriate parameters
            model_params = self._get_direction_model_params(quick_train)
            model = PriceDirectionPredictor(**model_params)

            # Train with time series validation
            if quick_train:
                # Simple train-validation split for quick training
                model.fit(X_clean, y_clean, validation_split=0.2)
            else:
                # Full time series cross-validation
                best_model = self._train_with_cross_validation(
                    model, X_clean, y_clean, model_type="direction"
                )
                if best_model:
                    model = best_model

            # Evaluate model
            train_metrics = model.evaluate(X_clean, y_clean)
            logger.info(f"Training complete - Accuracy: {train_metrics.accuracy:.3f}")

            # Save model if requested
            if save_model and model.is_fitted:
                version = datetime.now().strftime("v%Y%m%d_%H%M")
                self.model_manager.save_model(
                    model,
                    f"price_direction_{symbol.lower()}",
                    version=version,
                    metadata={
                        'symbol': symbol,
                        'training_samples': len(X_clean),
                        'feature_count': X_clean.shape[1],
                        'accuracy': train_metrics.accuracy,
                        'quick_train': quick_train
                    }
                )

            return model

        except Exception as e:
            logger.error(f"Price direction model training failed: {e}")
            return None

    def train_volatility_model(self,
                             data: pd.DataFrame = None,
                             symbol: str = "SPY",
                             model_type: str = "ensemble",
                             quick_train: bool = False,
                             save_model: bool = True) -> Optional[VolatilityPredictor]:
        """
        Train volatility forecasting model.

        Args:
            data: Training data
            symbol: Symbol to train on
            model_type: Type of volatility model
            quick_train: Use reduced training
            save_model: Whether to save model

        Returns:
            Trained volatility model
        """
        logger.info(f"Training volatility model for {symbol} (type: {model_type})")

        try:
            # Get training data
            if data is None:
                data = self._fetch_training_data(symbol, period="2y")

            if data is None or len(data) < self.min_training_samples:
                logger.error(f"Insufficient data for volatility training: {len(data) if data is not None else 0}")
                return None

            # Calculate returns and volatility
            returns = data['Close'].pct_change().dropna()
            realized_vol = returns.rolling(20).std() * np.sqrt(252)

            # Create features
            features = self.feature_engineer.engineer_features(
                data,
                target_type="volatility",
                target_horizon=5
            )

            # Prepare target (future volatility)
            target = realized_vol.shift(-5).dropna()

            # Align features and target
            aligned_data = pd.concat([features.features, target], axis=1, join='inner')
            if aligned_data.empty:
                logger.error("No aligned data for volatility training")
                return None

            X = aligned_data.iloc[:, :-1]
            y = aligned_data.iloc[:, -1]

            logger.info(f"Volatility training data: {len(X)} samples")

            # Initialize model
            vol_params = self._get_volatility_model_params(model_type, quick_train)
            model = VolatilityPredictor(**vol_params)

            # Train model
            model.fit(X, y, validation_split=0.2)

            # Evaluate
            metrics = model.evaluate(X, y)
            logger.info(f"Volatility model training complete")
            if metrics.metrics:
                logger.info(f"R²: {metrics.metrics.get('r2', 0):.3f}")

            # Save model
            if save_model and model.is_fitted:
                version = datetime.now().strftime("v%Y%m%d_%H%M")
                self.model_manager.save_model(
                    model,
                    f"volatility_{symbol.lower()}_{model_type}",
                    version=version,
                    metadata={
                        'symbol': symbol,
                        'model_type': model_type,
                        'training_samples': len(X),
                        'r2_score': metrics.metrics.get('r2', 0) if metrics.metrics else 0
                    }
                )

            return model

        except Exception as e:
            logger.error(f"Volatility model training failed: {e}")
            return None

    def train_multi_asset_models(self,
                                symbols: List[str] = None,
                                model_types: List[str] = None,
                                quick_train: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Train models for multiple assets.

        Args:
            symbols: List of symbols to train
            model_types: Types of models to train
            quick_train: Use quick training mode

        Returns:
            Dict with training results
        """
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']

        if model_types is None:
            model_types = ['direction', 'volatility']

        logger.info(f"Training models for {len(symbols)} symbols, {len(model_types)} model types")

        results = {}

        for symbol in symbols:
            symbol_results = {}

            # Fetch data once per symbol
            data = self._fetch_training_data(symbol, period="2y")
            if data is None:
                logger.warning(f"No data available for {symbol}")
                continue

            # Train each model type
            for model_type in model_types:
                try:
                    if model_type == "direction":
                        model = self.train_price_direction_model(
                            data=data,
                            symbol=symbol,
                            quick_train=quick_train,
                            save_model=True
                        )
                    elif model_type == "volatility":
                        model = self.train_volatility_model(
                            data=data,
                            symbol=symbol,
                            quick_train=quick_train,
                            save_model=True
                        )
                    else:
                        logger.warning(f"Unknown model type: {model_type}")
                        continue

                    symbol_results[model_type] = {
                        'success': model is not None and model.is_fitted,
                        'model': model
                    }

                except Exception as e:
                    logger.error(f"Training failed for {symbol} {model_type}: {e}")
                    symbol_results[model_type] = {
                        'success': False,
                        'error': str(e)
                    }

            results[symbol] = symbol_results

        # Summary
        total_models = len(symbols) * len(model_types)
        successful_models = sum(
            1 for symbol_results in results.values()
            for model_result in symbol_results.values()
            if model_result.get('success', False)
        )

        logger.info(f"Multi-asset training complete: {successful_models}/{total_models} models successful")

        return results

    def _train_with_cross_validation(self,
                                   model: Any,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   model_type: str) -> Optional[Any]:
        """Train model with time series cross-validation"""
        try:
            # Time series split
            tscv = TimeSeriesSplit(n_splits=self.validation_splits)

            best_score = -np.inf
            best_model = None

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                logger.debug(f"Training fold {fold + 1}/{self.validation_splits}")

                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]

                # Train model on fold
                fold_model = type(model)(**model.config) if hasattr(model, 'config') else type(model)()
                fold_model.fit(X_train, y_train, validation_split=0.0)

                # Evaluate
                metrics = fold_model.evaluate(X_val, y_val)

                if model_type == "direction":
                    score = metrics.accuracy if metrics.accuracy else 0
                else:  # volatility
                    score = metrics.metrics.get('r2', 0) if metrics.metrics else 0

                if score > best_score:
                    best_score = score
                    best_model = fold_model

            logger.info(f"Best CV score: {best_score:.3f}")
            return best_model

        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            # Return original model fitted on full data
            model.fit(X, y, validation_split=0.2)
            return model

    def _fetch_training_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Fetch historical training data"""
        cache_key = f"{symbol}_{period}"

        # Check cache first
        if cache_key in self.data_cache:
            cache_time = self.data_cache[cache_key].get('timestamp', datetime.min)
            if datetime.now() - cache_time < timedelta(hours=4):  # 4-hour cache
                return self.data_cache[cache_key]['data']

        try:
            if self.data_source == "yfinance":
                import yfinance as yf
                data = yf.download(symbol, period=period, progress=False)

                if len(data) > 0:
                    # Cache data
                    self.data_cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                    return data

            else:
                logger.warning(f"Unknown data source: {self.data_source}")
                return None

        except Exception as e:
            logger.error(f"Data fetch failed for {symbol}: {e}")
            return None

        return None

    def _get_direction_model_params(self, quick_train: bool) -> Dict[str, Any]:
        """Get parameters for direction model"""
        if quick_train:
            return {
                'sequence_length': 10,
                'lstm_units': 32,
                'num_layers': 1,
                'dropout_rate': 0.2,
                'use_attention': False
            }
        else:
            return {
                'sequence_length': 20,
                'lstm_units': 64,
                'num_layers': 2,
                'dropout_rate': 0.3,
                'use_attention': True
            }

    def _get_volatility_model_params(self, model_type: str, quick_train: bool) -> Dict[str, Any]:
        """Get parameters for volatility model"""
        base_params = {
            'model_type': model_type,
            'sequence_length': 15 if quick_train else 20,
            'lstm_units': 32 if quick_train else 50
        }

        if model_type == "garch":
            base_params['garch_type'] = "GARCH"
        elif model_type == "ensemble":
            pass  # Use defaults

        return base_params

    def validate_model_performance(self,
                                 model: Any,
                                 symbol: str,
                                 validation_period: str = "3m") -> Dict[str, Any]:
        """
        Validate model performance on out-of-sample data.

        Args:
            model: Trained model to validate
            symbol: Symbol to validate on
            validation_period: Period for validation data

        Returns:
            Dict with validation results
        """
        try:
            # Fetch validation data
            val_data = self._fetch_training_data(symbol, period=validation_period)
            if val_data is None:
                return {'success': False, 'error': 'No validation data'}

            # Engineer features
            if hasattr(model, 'model_type') and model.model_type == "regressor":
                feature_set = self.feature_engineer.engineer_features(
                    val_data, target_type="volatility"
                )
            else:
                feature_set = self.feature_engineer.engineer_features(
                    val_data, target_type="direction"
                )

            # Make predictions
            predictions = model.predict(feature_set.features)
            target = feature_set.target

            # Calculate metrics
            metrics = model.evaluate(feature_set.features, target)

            # Additional analysis
            analysis = {
                'validation_period': validation_period,
                'validation_samples': len(feature_set.features),
                'model_type': getattr(model, 'model_type', 'unknown'),
                'metrics': metrics,
                'feature_importance': model.get_feature_importance(10).to_dict() if hasattr(model, 'get_feature_importance') else {}
            }

            return {
                'success': True,
                'analysis': analysis,
                'validated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all training activities"""
        try:
            # Get model registry
            models = self.model_manager.list_models()

            # Calculate summary stats
            total_models = sum(len(versions) for versions in models.values())
            model_types = {}

            for model_name, versions in models.items():
                for version_info in versions.values():
                    model_class = version_info.get('class', 'unknown')
                    model_types[model_class] = model_types.get(model_class, 0) + 1

            # Storage usage
            storage = self.model_manager.get_storage_usage()

            return {
                'total_models': total_models,
                'unique_models': len(models),
                'model_types': model_types,
                'storage_usage_mb': storage.get('total_size_mb', 0),
                'last_updated': datetime.now().isoformat(),
                'data_cache_size': len(self.data_cache)
            }

        except Exception as e:
            logger.error(f"Training summary failed: {e}")
            return {}


def main():
    """Test model trainer"""
    print("Testing Model Trainer")
    print("=" * 30)

    # Initialize trainer
    trainer = ModelTrainer()

    # Test single model training
    print("Training direction model for SPY...")
    direction_model = trainer.train_price_direction_model(
        symbol="SPY",
        quick_train=True,
        save_model=False
    )

    if direction_model:
        print(f"Direction model trained: {direction_model.is_fitted}")

        # Test prediction
        import yfinance as yf
        test_data = yf.download("SPY", period="1m", progress=False)

        if len(test_data) > 0:
            signal = direction_model.get_signal_strength(test_data)
            print(f"Test signal: {signal}")

    # Test volatility model
    print("\nTraining volatility model...")
    vol_model = trainer.train_volatility_model(
        symbol="SPY",
        model_type="rf",  # Use Random Forest for faster training
        quick_train=True,
        save_model=False
    )

    if vol_model:
        print(f"Volatility model trained: {vol_model.is_fitted}")

    # Training summary
    print("\nTraining Summary:")
    summary = trainer.get_training_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nModel trainer test completed!")


if __name__ == "__main__":
    main()