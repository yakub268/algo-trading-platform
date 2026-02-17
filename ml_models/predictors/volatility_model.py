"""
Volatility Forecasting Model
===========================

Advanced volatility forecasting using GARCH, ML, and ensemble methods.
Combines traditional econometric models with modern ML techniques.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for base class import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_predictor import BasePredictor, PredictionResult, ModelMetrics

# Try to import ARCH for GARCH models
try:
    from arch import arch_model
    from arch.univariate import GARCH, EGARCH, GJR_GARCH
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

# Standard ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


class VolatilityPredictor(BasePredictor):
    """
    Advanced volatility forecasting model.

    Combines multiple approaches:
    1. GARCH family models (traditional econometrics)
    2. LSTM neural networks (deep learning)
    3. Random Forest (ensemble machine learning)
    4. Ensemble combination of all methods

    Predicts:
    - Realized volatility (historical)
    - Implied volatility features
    - Volatility regime changes
    - Confidence intervals
    """

    def __init__(self,
                 model_type: str = "ensemble",
                 garch_type: str = "GARCH",
                 lstm_units: int = 50,
                 sequence_length: int = 20,
                 volatility_window: int = 20):
        """
        Initialize volatility predictor.

        Args:
            model_type: "garch", "lstm", "rf", or "ensemble"
            garch_type: "GARCH", "EGARCH", or "GJRGARCH"
            lstm_units: Number of LSTM units
            sequence_length: Sequence length for LSTM
            volatility_window: Window for realized volatility calculation
        """
        super().__init__(model_name="VolatilityForecaster", model_type="regressor")

        self.model_type = model_type
        self.garch_type = garch_type
        self.lstm_units = lstm_units
        self.sequence_length = sequence_length
        self.volatility_window = volatility_window

        # Model components
        self.garch_model = None
        self.lstm_model = None
        self.rf_model = None
        self.ensemble_weights = None

        # Data preprocessing
        self.scaler = StandardScaler()
        self.returns_series = None
        self.feature_scaler_fitted = False

        # Volatility calculation parameters
        self.annualization_factor = 252  # Trading days per year

        # Store configuration
        self.config = {
            'model_type': model_type,
            'garch_type': garch_type,
            'lstm_units': lstm_units,
            'sequence_length': sequence_length,
            'volatility_window': volatility_window,
            'arch_available': ARCH_AVAILABLE,
            'tf_available': TF_AVAILABLE
        }

        logger.info(f"VolatilityPredictor initialized (type: {model_type})")
        if not ARCH_AVAILABLE:
            logger.warning("ARCH package not available. GARCH models will be disabled.")

    def _build_model(self, **kwargs) -> Any:
        """Build volatility forecasting model(s)"""
        models = {}

        if self.model_type in ["garch", "ensemble"]:
            models['garch'] = self._build_garch_model()

        if self.model_type in ["lstm", "ensemble"]:
            models['lstm'] = self._build_lstm_model()

        if self.model_type in ["rf", "ensemble"]:
            models['rf'] = self._build_random_forest_model()

        # Return single model or ensemble
        if self.model_type == "ensemble":
            return models
        else:
            return models.get(self.model_type)

    def _build_garch_model(self) -> Optional[Any]:
        """Build GARCH model"""
        if not ARCH_AVAILABLE:
            logger.warning("Cannot build GARCH model - ARCH package not available")
            return None

        # GARCH model will be created during fitting with returns data
        # This is just a placeholder
        return {'type': 'garch', 'model': None}

    def _build_lstm_model(self) -> Optional[Any]:
        """Build LSTM model for volatility forecasting"""
        if not TF_AVAILABLE:
            logger.warning("Cannot build LSTM model - TensorFlow not available")
            return None

        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            LSTM(self.lstm_units, return_sequences=True, dropout=0.2),
            LSTM(self.lstm_units // 2, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Volatility is positive, but we'll use post-processing
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def _build_random_forest_model(self) -> Any:
        """Build Random Forest model"""
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

    def _prepare_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[Any, Any]:
        """Prepare data for volatility forecasting"""
        # Calculate returns if price data is provided
        if 'close' in X.columns or 'price' in X.columns:
            price_col = 'close' if 'close' in X.columns else 'price'
            returns = X[price_col].pct_change().dropna()
            self.returns_series = returns
        else:
            # Assume returns are already in the data
            returns = X.iloc[:, 0]  # Use first column as returns
            self.returns_series = returns

        # Calculate realized volatility features
        vol_features = self._calculate_volatility_features(X)

        # Scale features for LSTM and RF
        if not self.feature_scaler_fitted:
            X_scaled = self.scaler.fit_transform(vol_features)
            self.feature_scaler_fitted = True
        else:
            X_scaled = self.scaler.transform(vol_features)

        # Prepare target (future realized volatility)
        y_processed = None
        if y is not None:
            y_processed = y.values if isinstance(y, pd.Series) else y

        # For LSTM, create sequences
        if self.model_type in ["lstm", "ensemble"]:
            X_lstm = self._create_volatility_sequences(X_scaled)
            if y is not None:
                y_lstm = y_processed[self.sequence_length-1:]
            else:
                y_lstm = None
        else:
            X_lstm = X_scaled
            y_lstm = y_processed

        return X_lstm, y_lstm

    def _calculate_volatility_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-related features"""
        features = pd.DataFrame(index=X.index)

        # Get returns
        if hasattr(self, 'returns_series') and self.returns_series is not None:
            returns = self.returns_series
        elif 'close' in X.columns or 'price' in X.columns:
            price_col = 'close' if 'close' in X.columns else 'price'
            returns = X[price_col].pct_change()
        else:
            returns = X.iloc[:, 0].pct_change()

        # Realized volatility at different horizons
        for window in [5, 10, 20, 60]:
            rv = returns.rolling(window).std() * np.sqrt(self.annualization_factor)
            features[f'realized_vol_{window}d'] = rv

        # GARCH-like volatility measures
        features['squared_returns'] = returns ** 2
        features['abs_returns'] = returns.abs()

        # Volatility of volatility
        vol_20d = features['realized_vol_20d']
        features['vol_of_vol'] = vol_20d.rolling(20).std()

        # Volatility momentum
        features['vol_momentum'] = vol_20d / vol_20d.shift(5) - 1

        # Volatility mean reversion
        vol_mean = vol_20d.rolling(252).mean()
        features['vol_mean_reversion'] = vol_20d / vol_mean - 1

        # Volatility percentile
        features['vol_percentile'] = vol_20d.rolling(252).rank(pct=True)

        # Range-based volatility (Garman-Klass if OHLC available)
        if all(col in X.columns for col in ['high', 'low', 'open', 'close']):
            # Garman-Klass volatility estimator
            ln_hl = np.log(X['high'] / X['low'])
            ln_co = np.log(X['close'] / X['open'])
            gk_vol = (0.5 * ln_hl**2 - (2*np.log(2) - 1) * ln_co**2)
            features['garman_klass_vol'] = np.sqrt(gk_vol * self.annualization_factor)

        # Volatility clustering measures
        features['vol_persistence'] = vol_20d.rolling(10).corr(vol_20d.shift(1))

        # Regime indicators
        high_vol_threshold = vol_20d.rolling(252).quantile(0.8)
        low_vol_threshold = vol_20d.rolling(252).quantile(0.2)
        features['high_vol_regime'] = (vol_20d > high_vol_threshold).astype(int)
        features['low_vol_regime'] = (vol_20d < low_vol_threshold).astype(int)

        # Clean features
        features = features.fillna(method='ffill').fillna(0)

        return features

    def _create_volatility_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM volatility prediction"""
        sequences = []
        for i in range(self.sequence_length - 1, len(data)):
            sequences.append(data[i - self.sequence_length + 1:i + 1])
        return np.array(sequences)

    def _fit_model(self, X: Any, y: Any, **kwargs) -> None:
        """Train volatility forecasting model(s)"""
        if self.model_type == "ensemble":
            self._fit_ensemble_models(X, y, **kwargs)
        elif self.model_type == "garch":
            self._fit_garch_model(X, y, **kwargs)
        elif self.model_type == "lstm":
            self._fit_lstm_model(X, y, **kwargs)
        elif self.model_type == "rf":
            self._fit_rf_model(X, y, **kwargs)

    def _fit_garch_model(self, X: Any, y: Any, **kwargs) -> None:
        """Fit GARCH model"""
        if not ARCH_AVAILABLE or self.returns_series is None:
            logger.warning("Cannot fit GARCH model")
            return

        try:
            # Create GARCH model
            returns_clean = self.returns_series.dropna() * 100  # Scale to percentage

            if self.garch_type == "GARCH":
                garch_model = arch_model(returns_clean, vol='Garch', p=1, q=1)
            elif self.garch_type == "EGARCH":
                garch_model = arch_model(returns_clean, vol='EGarch', p=1, q=1)
            elif self.garch_type == "GJRGARCH":
                garch_model = arch_model(returns_clean, vol='Garch', p=1, o=1, q=1)
            else:
                garch_model = arch_model(returns_clean, vol='Garch', p=1, q=1)

            # Fit model
            self.garch_model = garch_model.fit(disp='off', show_warning=False)
            logger.info(f"GARCH model fitted successfully")

        except Exception as e:
            logger.warning(f"GARCH model fitting failed: {e}")
            self.garch_model = None

    def _fit_lstm_model(self, X: Any, y: Any, **kwargs) -> None:
        """Fit LSTM model"""
        if not TF_AVAILABLE or not isinstance(self.model, dict):
            return

        lstm_model = self.model.get('lstm') if isinstance(self.model, dict) else self.model
        if lstm_model is None:
            return

        try:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                )
            ]

            history = lstm_model.fit(
                X, y,
                batch_size=32,
                epochs=100,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )

            if isinstance(self.model, dict):
                self.model['lstm'] = lstm_model
            else:
                self.lstm_model = lstm_model

            logger.info("LSTM volatility model fitted successfully")

        except Exception as e:
            logger.warning(f"LSTM model fitting failed: {e}")

    def _fit_rf_model(self, X: Any, y: Any, **kwargs) -> None:
        """Fit Random Forest model"""
        try:
            rf_model = self.model.get('rf') if isinstance(self.model, dict) else self.model
            rf_model.fit(X, y)

            if isinstance(self.model, dict):
                self.model['rf'] = rf_model
            else:
                self.rf_model = rf_model

            logger.info("Random Forest volatility model fitted successfully")

        except Exception as e:
            logger.warning(f"Random Forest model fitting failed: {e}")

    def _fit_ensemble_models(self, X: Any, y: Any, **kwargs) -> None:
        """Fit all ensemble models and calculate weights"""
        models_fitted = []

        # Fit GARCH
        if ARCH_AVAILABLE:
            self._fit_garch_model(X, y, **kwargs)
            if self.garch_model is not None:
                models_fitted.append('garch')

        # Fit LSTM
        if TF_AVAILABLE and self.model.get('lstm') is not None:
            self._fit_lstm_model(X, y, **kwargs)
            models_fitted.append('lstm')

        # Fit Random Forest
        self._fit_rf_model(X, y, **kwargs)
        models_fitted.append('rf')

        # Calculate ensemble weights based on validation performance
        self._calculate_ensemble_weights(X, y, models_fitted)

        logger.info(f"Ensemble fitted with models: {models_fitted}")

    def _calculate_ensemble_weights(self, X: Any, y: Any, models_fitted: List[str]) -> None:
        """Calculate ensemble weights based on out-of-sample performance"""
        if len(models_fitted) == 0:
            return

        # Split data for weight calculation
        split_point = int(len(y) * 0.8)
        X_val = X[split_point:]
        y_val = y[split_point:]

        if len(y_val) < 10:  # Not enough validation data
            # Use equal weights
            self.ensemble_weights = {model: 1.0 / len(models_fitted) for model in models_fitted}
            return

        # Calculate validation performance for each model
        model_errors = {}

        for model_name in models_fitted:
            try:
                predictions = self._predict_single_model(X_val, model_name)
                if predictions is not None and len(predictions) == len(y_val):
                    mse = mean_squared_error(y_val, predictions)
                    model_errors[model_name] = mse
            except Exception as e:
                logger.debug(f"Error calculating {model_name} validation performance: {e}")

        if not model_errors:
            # Fallback to equal weights
            self.ensemble_weights = {model: 1.0 / len(models_fitted) for model in models_fitted}
            return

        # Calculate inverse error weights (better models get higher weights)
        total_inv_error = sum(1.0 / error for error in model_errors.values())
        self.ensemble_weights = {
            model: (1.0 / error) / total_inv_error
            for model, error in model_errors.items()
        }

        logger.info(f"Ensemble weights: {self.ensemble_weights}")

    def _predict_model(self, X: Any) -> np.ndarray:
        """Make volatility predictions"""
        if self.model_type == "ensemble":
            return self._predict_ensemble(X)
        elif self.model_type == "garch":
            return self._predict_garch(X)
        elif self.model_type == "lstm":
            return self._predict_lstm(X)
        elif self.model_type == "rf":
            return self._predict_rf(X)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _predict_single_model(self, X: Any, model_name: str) -> Optional[np.ndarray]:
        """Predict with a single model"""
        if model_name == "garch":
            return self._predict_garch(X)
        elif model_name == "lstm":
            return self._predict_lstm(X)
        elif model_name == "rf":
            return self._predict_rf(X)
        return None

    def _predict_garch(self, X: Any) -> Optional[np.ndarray]:
        """Predict with GARCH model"""
        if self.garch_model is None:
            return None

        try:
            # GARCH forecasts conditional volatility
            forecast = self.garch_model.forecast(horizon=1)
            vol_forecast = np.sqrt(forecast.variance.values[-1, :]) / 100  # Convert back from percentage

            # Extend forecast to match X length (simplified)
            predictions = np.full(len(X), vol_forecast[0])
            return predictions

        except Exception as e:
            logger.debug(f"GARCH prediction failed: {e}")
            return None

    def _predict_lstm(self, X: Any) -> Optional[np.ndarray]:
        """Predict with LSTM model"""
        lstm_model = self.model.get('lstm') if isinstance(self.model, dict) else self.lstm_model
        if lstm_model is None:
            return None

        try:
            predictions = lstm_model.predict(X, verbose=0)
            return predictions.flatten()
        except Exception as e:
            logger.debug(f"LSTM prediction failed: {e}")
            return None

    def _predict_rf(self, X: Any) -> Optional[np.ndarray]:
        """Predict with Random Forest model"""
        rf_model = self.model.get('rf') if isinstance(self.model, dict) else self.rf_model
        if rf_model is None:
            return None

        try:
            predictions = rf_model.predict(X)
            return predictions
        except Exception as e:
            logger.debug(f"Random Forest prediction failed: {e}")
            return None

    def _predict_ensemble(self, X: Any) -> np.ndarray:
        """Predict with ensemble of models"""
        if not self.ensemble_weights:
            # Fallback: use Random Forest only
            rf_pred = self._predict_rf(X)
            if rf_pred is not None:
                return rf_pred
            else:
                return np.full(len(X), 0.2)  # Default volatility

        predictions = []
        weights = []

        for model_name, weight in self.ensemble_weights.items():
            pred = self._predict_single_model(X, model_name)
            if pred is not None:
                predictions.append(pred)
                weights.append(weight)

        if not predictions:
            return np.full(len(X), 0.2)  # Default volatility

        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights) / np.sum(weights)  # Normalize weights

        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred

    def predict_volatility(self, X: pd.DataFrame, horizon: int = 1) -> Dict[str, Any]:
        """
        Predict volatility with comprehensive analysis.

        Args:
            X: Feature matrix
            horizon: Forecast horizon in days

        Returns:
            Dict with volatility predictions and analysis
        """
        prediction_result = self.predict(X, return_probabilities=False)

        predictions = prediction_result.predictions

        # Ensure predictions are positive (volatility cannot be negative)
        predictions = np.maximum(predictions, 0.001)

        # Calculate prediction intervals (simplified)
        prediction_std = np.std(predictions) if len(predictions) > 1 else predictions[0] * 0.1
        confidence_level = 0.95
        z_score = 1.96  # 95% confidence interval

        lower_bound = predictions - z_score * prediction_std
        upper_bound = predictions + z_score * prediction_std

        # Ensure bounds are positive
        lower_bound = np.maximum(lower_bound, 0.001)

        result = {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'prediction_interval': confidence_level,
            'prediction_time': prediction_result.prediction_time,
            'model_version': prediction_result.model_version,
            'horizon': horizon
        }

        # Add summary statistics
        if len(predictions) > 0:
            result['summary'] = {
                'mean_volatility': float(np.mean(predictions)),
                'volatility_trend': float(np.mean(np.diff(predictions))) if len(predictions) > 1 else 0.0,
                'volatility_level': self._classify_volatility_level(np.mean(predictions)),
                'forecast_uncertainty': float(prediction_std)
            }

        return result

    def _classify_volatility_level(self, volatility: float) -> str:
        """Classify volatility level"""
        if volatility < 0.15:
            return "LOW"
        elif volatility < 0.25:
            return "MEDIUM"
        elif volatility < 0.35:
            return "HIGH"
        else:
            return "EXTREME"

    def get_volatility_regime(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify current volatility regime.

        Returns:
            Dict with regime analysis
        """
        # Calculate recent volatility
        vol_features = self._calculate_volatility_features(X)
        current_vol = float(vol_features['realized_vol_20d'].iloc[-1])

        # Historical volatility percentile
        vol_percentile = float(vol_features['vol_percentile'].iloc[-1])

        # Volatility clustering
        vol_persistence = float(vol_features['vol_persistence'].iloc[-1])

        # Classify regime
        if vol_percentile > 0.8:
            regime = "HIGH_VOLATILITY"
        elif vol_percentile < 0.2:
            regime = "LOW_VOLATILITY"
        else:
            regime = "NORMAL_VOLATILITY"

        # Check for volatility breakout
        vol_momentum = float(vol_features['vol_momentum'].iloc[-1])
        if vol_momentum > 0.5:
            breakout_direction = "INCREASING"
        elif vol_momentum < -0.3:
            breakout_direction = "DECREASING"
        else:
            breakout_direction = "STABLE"

        return {
            'current_regime': regime,
            'current_volatility': float(current_vol),
            'volatility_percentile': float(vol_percentile),
            'volatility_momentum': float(vol_momentum),
            'breakout_direction': breakout_direction,
            'persistence': float(vol_persistence),
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Test volatility forecasting model"""
    import yfinance as yf

    print("Testing Volatility Forecasting Model")
    print("=" * 50)

    # Download test data
    print("Downloading test data...")
    data = yf.download("AAPL", period="2y", progress=False)

    # Calculate returns
    returns = data['Close'].pct_change().dropna()

    # Calculate realized volatility (target)
    window = 20
    realized_vol = returns.rolling(window).std() * np.sqrt(252)

    # Create features
    features = pd.DataFrame(index=data.index)
    features['close'] = data['Close']

    # Remove NaN values
    aligned_data = pd.concat([features, realized_vol.shift(-5)], axis=1).dropna()
    features_clean = aligned_data.iloc[:, :-1]
    target_clean = aligned_data.iloc[:, -1]

    print(f"Data shape: {features_clean.shape}")
    print(f"Target range: {target_clean.min():.4f} - {target_clean.max():.4f}")

    # Test different model types
    model_types = ["rf", "ensemble"] if not TF_AVAILABLE else ["rf", "lstm", "ensemble"]

    for model_type in model_types:
        print(f"\nTesting {model_type.upper()} model...")

        # Initialize predictor
        predictor = VolatilityPredictor(
            model_type=model_type,
            sequence_length=10,
            lstm_units=32,
            volatility_window=20
        )

        # Train model
        split_point = int(len(features_clean) * 0.8)
        train_X = features_clean.iloc[:split_point]
        train_y = target_clean.iloc[:split_point]
        test_X = features_clean.iloc[split_point:]
        test_y = target_clean.iloc[split_point:]

        predictor.fit(train_X, train_y, validation_split=0.2)

        # Test predictions
        vol_prediction = predictor.predict_volatility(test_X)

        print(f"Predictions made: {len(vol_prediction['predictions'])}")
        print(f"Mean predicted volatility: {vol_prediction['summary']['mean_volatility']:.4f}")
        print(f"Volatility level: {vol_prediction['summary']['volatility_level']}")

        # Evaluate performance
        metrics = predictor.evaluate(test_X, test_y)
        if metrics.metrics:
            print(f"RMSE: {metrics.metrics['rmse']:.4f}")
            print(f"R²: {metrics.metrics['r2']:.4f}")

        # Get volatility regime
        regime = predictor.get_volatility_regime(test_X)
        print(f"Current regime: {regime['current_regime']}")
        print(f"Volatility trend: {regime['breakout_direction']}")

    print("\nVolatility forecasting test completed!")


if __name__ == "__main__":
    main()