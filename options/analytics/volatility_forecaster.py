"""
Volatility Forecasting Models

Advanced volatility forecasting using multiple methodologies:
- GARCH models (GARCH, EGARCH, GJR-GARCH)
- Realized volatility models
- Machine learning approaches (LSTM, Random Forest)
- Ensemble volatility forecasting
- Volatility regime switching models
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Suppress GARCH warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logging.warning("arch package not available - GARCH models disabled")

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available - LSTM models disabled")

logger = logging.getLogger(__name__)


class VolatilityModel(Enum):
    """Available volatility forecasting models"""
    GARCH = "garch"
    EGARCH = "egarch"
    GJR_GARCH = "gjr_garch"
    REALIZED_VOL = "realized_vol"
    LSTM = "lstm"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"
    HISTORICAL_AVERAGE = "historical_average"


@dataclass
class VolatilityForecast:
    """Container for volatility forecast results"""
    model_type: str
    forecast_horizon: int
    forecast_values: List[float]
    confidence_intervals: Optional[List[Tuple[float, float]]] = None
    forecast_dates: Optional[List[datetime]] = None
    model_accuracy: Optional[Dict[str, float]] = None
    forecast_timestamp: datetime = None

    def __post_init__(self):
        if self.forecast_timestamp is None:
            self.forecast_timestamp = datetime.now()


class VolatilityForecaster:
    """
    Advanced volatility forecasting system

    Supports multiple models and ensemble forecasting for robust
    volatility predictions.
    """

    def __init__(self,
                 lookback_window: int = 252,
                 forecast_horizon: int = 22,
                 models: Optional[List[VolatilityModel]] = None):
        """
        Initialize volatility forecaster

        Args:
            lookback_window: Historical data window (trading days)
            forecast_horizon: Forecast horizon (trading days)
            models: List of models to use (default: all available)
        """
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon

        # Set default models based on available packages
        if models is None:
            models = [VolatilityModel.HISTORICAL_AVERAGE, VolatilityModel.REALIZED_VOL]
            if ARCH_AVAILABLE:
                models.extend([VolatilityModel.GARCH, VolatilityModel.EGARCH])
            if TF_AVAILABLE:
                models.append(VolatilityModel.LSTM)
            models.extend([VolatilityModel.RANDOM_FOREST, VolatilityModel.ENSEMBLE])

        self.models = models
        self.trained_models = {}
        self.feature_cache = {}

        logger.info(f"Initialized VolatilityForecaster with models: {[m.value for m in self.models]}")

    def calculate_realized_volatility(self,
                                    price_data: pd.Series,
                                    window: int = 22,
                                    annualized: bool = True) -> pd.Series:
        """
        Calculate realized volatility using close-to-close method

        Args:
            price_data: Price series (typically close prices)
            window: Rolling window for volatility calculation
            annualized: Whether to annualize the volatility

        Returns:
            Realized volatility series
        """
        try:
            # Calculate returns
            returns = np.log(price_data / price_data.shift(1)).dropna()

            # Calculate rolling volatility
            realized_vol = returns.rolling(window=window).std()

            if annualized:
                realized_vol *= np.sqrt(252)  # Annualize assuming 252 trading days

            return realized_vol.dropna()

        except Exception as e:
            logger.error(f"Error calculating realized volatility: {e}")
            return pd.Series()

    def calculate_parkinson_volatility(self,
                                     high_prices: pd.Series,
                                     low_prices: pd.Series,
                                     window: int = 22,
                                     annualized: bool = True) -> pd.Series:
        """
        Calculate Parkinson volatility estimator (uses high/low prices)

        Args:
            high_prices: High price series
            low_prices: Low price series
            window: Rolling window
            annualized: Whether to annualize

        Returns:
            Parkinson volatility series
        """
        try:
            # Parkinson volatility: sqrt(1/(4*ln(2)) * ln(H/L)^2)
            hl_ratio = np.log(high_prices / low_prices) ** 2
            parkinson_vol = np.sqrt(hl_ratio / (4 * np.log(2)))

            # Apply rolling window
            if window > 1:
                parkinson_vol = parkinson_vol.rolling(window=window).mean()

            if annualized:
                parkinson_vol *= np.sqrt(252)

            return parkinson_vol.dropna()

        except Exception as e:
            logger.error(f"Error calculating Parkinson volatility: {e}")
            return pd.Series()

    def calculate_garman_klass_volatility(self,
                                        open_prices: pd.Series,
                                        high_prices: pd.Series,
                                        low_prices: pd.Series,
                                        close_prices: pd.Series,
                                        window: int = 22,
                                        annualized: bool = True) -> pd.Series:
        """
        Calculate Garman-Klass volatility estimator

        Args:
            open_prices: Open price series
            high_prices: High price series
            low_prices: Low price series
            close_prices: Close price series
            window: Rolling window
            annualized: Whether to annualize

        Returns:
            Garman-Klass volatility series
        """
        try:
            # GK volatility components
            hl_term = 0.5 * (np.log(high_prices / low_prices)) ** 2
            co_term = (2 * np.log(2) - 1) * (np.log(close_prices / open_prices)) ** 2

            gk_vol = np.sqrt(hl_term - co_term)

            # Apply rolling window
            if window > 1:
                gk_vol = gk_vol.rolling(window=window).mean()

            if annualized:
                gk_vol *= np.sqrt(252)

            return gk_vol.dropna()

        except Exception as e:
            logger.error(f"Error calculating Garman-Klass volatility: {e}")
            return pd.Series()

    def prepare_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature set for ML models

        Args:
            price_data: DataFrame with OHLCV data

        Returns:
            Feature DataFrame
        """
        try:
            features = pd.DataFrame(index=price_data.index)

            # Basic price features
            features['returns'] = np.log(price_data['close'] / price_data['close'].shift(1))
            features['log_volume'] = np.log(price_data['volume'] + 1)
            features['high_low_ratio'] = price_data['high'] / price_data['low']
            features['close_open_ratio'] = price_data['close'] / price_data['open']

            # Volatility features
            features['realized_vol_5'] = self.calculate_realized_volatility(price_data['close'], window=5)
            features['realized_vol_22'] = self.calculate_realized_volatility(price_data['close'], window=22)
            features['parkinson_vol'] = self.calculate_parkinson_volatility(price_data['high'], price_data['low'])

            # Technical indicators
            features['rsi'] = self._calculate_rsi(price_data['close'])
            features['bb_position'] = self._calculate_bb_position(price_data['close'])
            features['momentum_5'] = price_data['close'] / price_data['close'].shift(5) - 1
            features['momentum_22'] = price_data['close'] / price_data['close'].shift(22) - 1

            # Lagged features
            for lag in [1, 2, 3, 5]:
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
                features[f'realized_vol_lag_{lag}'] = features['realized_vol_22'].shift(lag)

            # Rolling statistics
            features['returns_mean_22'] = features['returns'].rolling(22).mean()
            features['returns_std_22'] = features['returns'].rolling(22).std()
            features['returns_skew_22'] = features['returns'].rolling(22).skew()
            features['returns_kurt_22'] = features['returns'].rolling(22).kurt()

            # Market microstructure
            features['price_impact'] = np.abs(features['returns']) / features['log_volume']
            features['volatility_volume_ratio'] = features['realized_vol_22'] / features['log_volume']

            return features.dropna()

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bb_position(self, prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
        """Calculate Bollinger Band position"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_bb = sma + (std * num_std)
        lower_bb = sma - (std * num_std)
        bb_position = (prices - lower_bb) / (upper_bb - lower_bb)
        return bb_position

    def fit_garch_model(self, returns: pd.Series, model_type: str = 'GARCH') -> Optional[object]:
        """
        Fit GARCH model to return series

        Args:
            returns: Return series
            model_type: Type of GARCH model ('GARCH', 'EGARCH', 'GJR-GARCH')

        Returns:
            Fitted GARCH model
        """
        if not ARCH_AVAILABLE:
            logger.warning("ARCH package not available for GARCH modeling")
            return None

        try:
            # Remove any infinite or NaN values
            clean_returns = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()

            if len(clean_returns) < 50:
                logger.warning("Insufficient data for GARCH model")
                return None

            # Scale returns to percentage for numerical stability
            scaled_returns = clean_returns * 100

            # Configure model based on type
            if model_type.upper() == 'GARCH':
                model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
            elif model_type.upper() == 'EGARCH':
                model = arch_model(scaled_returns, vol='EGARCH', p=1, o=1, q=1, dist='normal')
            elif model_type.upper() == 'GJR-GARCH':
                model = arch_model(scaled_returns, vol='GARCH', p=1, q=1, o=1, dist='normal')
            else:
                model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')

            # Fit model
            fitted_model = model.fit(disp='off', show_warning=False)

            logger.info(f"Successfully fitted {model_type} model")
            return fitted_model

        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            return None

    def forecast_garch_volatility(self, fitted_model: object, horizon: int) -> Optional[List[float]]:
        """
        Generate volatility forecast from fitted GARCH model

        Args:
            fitted_model: Fitted GARCH model
            horizon: Forecast horizon

        Returns:
            List of volatility forecasts
        """
        try:
            forecast = fitted_model.forecast(horizon=horizon)

            # Extract volatility forecasts and convert back to decimal scale
            vol_forecast = np.sqrt(forecast.variance.values[-1, :]) / 100  # Convert back from percentage

            # Annualize
            vol_forecast_annual = vol_forecast * np.sqrt(252)

            return vol_forecast_annual.tolist()

        except Exception as e:
            logger.error(f"Error forecasting GARCH volatility: {e}")
            return None

    def fit_lstm_model(self, features: pd.DataFrame, target: pd.Series, lookback: int = 22) -> Optional[object]:
        """
        Fit LSTM model for volatility forecasting

        Args:
            features: Feature DataFrame
            target: Target volatility series
            lookback: Lookback window for sequences

        Returns:
            Fitted LSTM model
        """
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available for LSTM modeling")
            return None

        try:
            # Prepare sequences
            X, y = self._create_sequences(features.values, target.values, lookback)

            if len(X) < 50:
                logger.warning("Insufficient data for LSTM model")
                return None

            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Build LSTM model
            model = keras.Sequential([
                keras.layers.LSTM(50, return_sequences=True, input_shape=(lookback, X.shape[2])),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(50, return_sequences=False),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(25),
                keras.layers.Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Train model
            early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

            model.fit(X_train, y_train,
                     batch_size=32,
                     epochs=100,
                     validation_data=(X_test, y_test),
                     callbacks=[early_stopping],
                     verbose=0)

            logger.info("Successfully fitted LSTM model")
            return model

        except Exception as e:
            logger.error(f"Error fitting LSTM model: {e}")
            return None

    def _create_sequences(self, features: np.ndarray, target: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(lookback, len(features)):
            X.append(features[i-lookback:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def fit_random_forest_model(self, features: pd.DataFrame, target: pd.Series) -> Optional[RandomForestRegressor]:
        """
        Fit Random Forest model for volatility forecasting

        Args:
            features: Feature DataFrame
            target: Target volatility series

        Returns:
            Fitted Random Forest model
        """
        try:
            # Align features and target
            aligned_data = pd.concat([features, target], axis=1).dropna()
            if len(aligned_data) < 50:
                logger.warning("Insufficient data for Random Forest model")
                return None

            X = aligned_data.iloc[:, :-1].values
            y = aligned_data.iloc[:, -1].values

            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            rf_model.fit(X, y)

            logger.info("Successfully fitted Random Forest model")
            return rf_model

        except Exception as e:
            logger.error(f"Error fitting Random Forest model: {e}")
            return None

    def generate_forecast(self,
                         price_data: pd.DataFrame,
                         model_type: VolatilityModel = VolatilityModel.ENSEMBLE) -> Optional[VolatilityForecast]:
        """
        Generate volatility forecast using specified model

        Args:
            price_data: DataFrame with OHLCV data
            model_type: Model type to use for forecasting

        Returns:
            VolatilityForecast object
        """
        try:
            # Prepare data
            returns = np.log(price_data['close'] / price_data['close'].shift(1)).dropna()
            realized_vol = self.calculate_realized_volatility(price_data['close'], window=22)

            if model_type == VolatilityModel.HISTORICAL_AVERAGE:
                return self._forecast_historical_average(realized_vol)

            elif model_type == VolatilityModel.REALIZED_VOL:
                return self._forecast_realized_vol(realized_vol)

            elif model_type == VolatilityModel.GARCH:
                return self._forecast_garch(returns, 'GARCH')

            elif model_type == VolatilityModel.EGARCH:
                return self._forecast_garch(returns, 'EGARCH')

            elif model_type == VolatilityModel.GJR_GARCH:
                return self._forecast_garch(returns, 'GJR-GARCH')

            elif model_type == VolatilityModel.LSTM:
                return self._forecast_lstm(price_data, realized_vol)

            elif model_type == VolatilityModel.RANDOM_FOREST:
                return self._forecast_random_forest(price_data, realized_vol)

            elif model_type == VolatilityModel.ENSEMBLE:
                return self._forecast_ensemble(price_data)

            else:
                logger.error(f"Unknown model type: {model_type}")
                return None

        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return None

    def _forecast_historical_average(self, realized_vol: pd.Series) -> VolatilityForecast:
        """Simple historical average forecast"""
        recent_vol = realized_vol.tail(self.lookback_window)
        avg_vol = recent_vol.mean()

        forecast = [avg_vol] * self.forecast_horizon

        return VolatilityForecast(
            model_type="historical_average",
            forecast_horizon=self.forecast_horizon,
            forecast_values=forecast
        )

    def _forecast_realized_vol(self, realized_vol: pd.Series) -> VolatilityForecast:
        """Realized volatility trend extrapolation"""
        recent_vol = realized_vol.tail(self.lookback_window)

        # Simple exponential smoothing
        alpha = 0.2
        forecast = []
        last_vol = recent_vol.iloc[-1]

        for _ in range(self.forecast_horizon):
            forecast.append(last_vol)
            last_vol = alpha * last_vol + (1 - alpha) * recent_vol.mean()

        return VolatilityForecast(
            model_type="realized_vol",
            forecast_horizon=self.forecast_horizon,
            forecast_values=forecast
        )

    def _forecast_garch(self, returns: pd.Series, garch_type: str) -> Optional[VolatilityForecast]:
        """GARCH model forecast"""
        fitted_model = self.fit_garch_model(returns.tail(self.lookback_window), garch_type)

        if fitted_model is None:
            return None

        forecast_values = self.forecast_garch_volatility(fitted_model, self.forecast_horizon)

        if forecast_values is None:
            return None

        return VolatilityForecast(
            model_type=garch_type.lower(),
            forecast_horizon=self.forecast_horizon,
            forecast_values=forecast_values
        )

    def _forecast_lstm(self, price_data: pd.DataFrame, target_vol: pd.Series) -> Optional[VolatilityForecast]:
        """LSTM model forecast"""
        features = self.prepare_features(price_data)

        if features.empty:
            return None

        model = self.fit_lstm_model(features, target_vol)

        if model is None:
            return None

        # Generate forecast (simplified)
        last_sequence = features.tail(22).values.reshape(1, 22, -1)
        forecast_values = []

        for _ in range(self.forecast_horizon):
            pred = model.predict(last_sequence, verbose=0)[0, 0]
            forecast_values.append(pred)

            # Update sequence (simplified rolling prediction)
            # In practice, you'd want to update features properly

        return VolatilityForecast(
            model_type="lstm",
            forecast_horizon=self.forecast_horizon,
            forecast_values=forecast_values
        )

    def _forecast_random_forest(self, price_data: pd.DataFrame, target_vol: pd.Series) -> Optional[VolatilityForecast]:
        """Random Forest model forecast"""
        features = self.prepare_features(price_data)

        if features.empty:
            return None

        model = self.fit_random_forest_model(features, target_vol)

        if model is None:
            return None

        # Generate forecast using last features
        last_features = features.tail(1).values
        forecast_values = []

        for _ in range(self.forecast_horizon):
            pred = model.predict(last_features)[0]
            forecast_values.append(pred)

        return VolatilityForecast(
            model_type="random_forest",
            forecast_horizon=self.forecast_horizon,
            forecast_values=forecast_values
        )

    def _forecast_ensemble(self, price_data: pd.DataFrame) -> Optional[VolatilityForecast]:
        """Ensemble forecast combining multiple models"""
        individual_forecasts = []
        model_weights = {}

        # Generate individual forecasts
        for model_type in self.models:
            if model_type == VolatilityModel.ENSEMBLE:
                continue  # Skip ensemble in ensemble

            forecast = self.generate_forecast(price_data, model_type)
            if forecast is not None:
                individual_forecasts.append(forecast)
                model_weights[model_type.value] = 1.0  # Equal weights for now

        if not individual_forecasts:
            return None

        # Combine forecasts
        ensemble_forecast = np.zeros(self.forecast_horizon)
        total_weight = sum(model_weights.values())

        for forecast in individual_forecasts:
            weight = model_weights[forecast.model_type] / total_weight
            ensemble_forecast += np.array(forecast.forecast_values) * weight

        return VolatilityForecast(
            model_type="ensemble",
            forecast_horizon=self.forecast_horizon,
            forecast_values=ensemble_forecast.tolist()
        )

    def evaluate_forecast_accuracy(self,
                                  forecasts: List[VolatilityForecast],
                                  actual_volatility: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate forecast accuracy against realized volatility

        Args:
            forecasts: List of volatility forecasts
            actual_volatility: Actual realized volatility

        Returns:
            Dictionary of accuracy metrics by model
        """
        results = {}

        for forecast in forecasts:
            if len(forecast.forecast_values) > len(actual_volatility):
                forecast_values = forecast.forecast_values[:len(actual_volatility)]
            else:
                forecast_values = forecast.forecast_values

            if len(forecast_values) == 0:
                continue

            actual_values = actual_volatility.iloc[:len(forecast_values)].values

            # Calculate metrics
            mse = mean_squared_error(actual_values, forecast_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_values, forecast_values)
            mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100

            # Direction accuracy
            actual_direction = np.diff(actual_values) > 0
            forecast_direction = np.diff(forecast_values) > 0
            direction_accuracy = np.mean(actual_direction == forecast_direction) * 100

            results[forecast.model_type] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'direction_accuracy': direction_accuracy
            }

        return results