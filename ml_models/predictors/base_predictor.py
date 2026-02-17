"""
Base Predictor Class
===================

Abstract base class for all ML prediction models.
Provides common functionality and interface.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import joblib
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results"""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    features_used: Optional[List[str]] = None
    model_version: Optional[str] = None
    prediction_time: Optional[datetime] = None
    metadata: Optional[Dict] = None


@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    metrics: Optional[Dict] = None


class BasePredictor(ABC):
    """
    Abstract base class for ML prediction models.

    Provides:
    - Model lifecycle management (train, predict, evaluate)
    - Model persistence and versioning
    - Feature validation and preprocessing
    - Performance tracking and monitoring
    - Common utilities for all predictors
    """

    def __init__(self, model_name: str, model_type: str = "classifier"):
        self.model_name = model_name
        self.model_type = model_type  # "classifier", "regressor"
        self.model = None
        self.is_fitted = False

        # Model metadata
        self.model_version = "1.0.0"
        self.created_at = datetime.now()
        self.last_trained = None
        self.last_updated = None

        # Feature information
        self.feature_names = []
        self.n_features = 0
        self.feature_importance = None

        # Performance tracking
        self.training_metrics = None
        self.validation_metrics = None
        self.recent_predictions = []

        # Model configuration
        self.config = {}

        logger.info(f"Initialized {model_name} predictor")

    @abstractmethod
    def _build_model(self, **kwargs) -> Any:
        """Build the ML model architecture. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _prepare_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[Any, Any]:
        """Prepare data for training/prediction. Must be implemented by subclasses."""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2, **kwargs) -> 'BasePredictor':
        """
        Train the model on provided data.

        Args:
            X: Feature matrix
            y: Target variable
            validation_split: Fraction of data for validation
            **kwargs: Additional training parameters

        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.model_name} on {len(X)} samples, {X.shape[1]} features")

        # Store feature information
        self.feature_names = list(X.columns)
        self.n_features = X.shape[1]

        # Validate data
        X_clean, y_clean = self._validate_training_data(X, y)

        # Split data for validation
        if validation_split > 0:
            split_idx = int(len(X_clean) * (1 - validation_split))
            X_train, X_val = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
            y_train, y_val = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
        else:
            X_train, X_val = X_clean, None
            y_train, y_val = y_clean, None

        # Prepare data for model
        X_train_processed, y_train_processed = self._prepare_data(X_train, y_train)

        # Build model if not already built
        if self.model is None:
            self.model = self._build_model(**kwargs)

        # Train model
        start_time = datetime.now()
        self._fit_model(X_train_processed, y_train_processed, **kwargs)
        training_time = (datetime.now() - start_time).total_seconds()

        # Update model state
        self.is_fitted = True
        self.last_trained = datetime.now()
        self.last_updated = datetime.now()

        # Evaluate on training data
        train_predictions = self.predict(X_train)
        self.training_metrics = self._calculate_metrics(y_train, train_predictions.predictions)

        # Evaluate on validation data if available
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            self.validation_metrics = self._calculate_metrics(y_val, val_predictions.predictions)

            logger.info(f"Training completed in {training_time:.1f}s")
            logger.info(f"Training accuracy: {self.training_metrics.accuracy:.3f}")
            logger.info(f"Validation accuracy: {self.validation_metrics.accuracy:.3f}")
        else:
            logger.info(f"Training completed in {training_time:.1f}s")
            logger.info(f"Training accuracy: {self.training_metrics.accuracy:.3f}")

        # Calculate feature importance if possible
        self._calculate_feature_importance()

        return self

    @abstractmethod
    def _fit_model(self, X: Any, y: Any, **kwargs) -> None:
        """Fit the actual model. Must be implemented by subclasses."""
        pass

    def predict(self, X: pd.DataFrame, return_probabilities: bool = True) -> PredictionResult:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix
            return_probabilities: Whether to return prediction probabilities

        Returns:
            PredictionResult with predictions and metadata
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Validate features match training data
        self._validate_prediction_features(X)

        # Prepare data
        X_processed, _ = self._prepare_data(X)

        # Make predictions
        predictions = self._predict_model(X_processed)

        # Get probabilities if supported and requested
        probabilities = None
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self._predict_probabilities(X_processed)
            except Exception as e:
                logger.debug(f"Could not compute probabilities: {e}")

        # Calculate confidence scores
        confidence = self._calculate_prediction_confidence(predictions, probabilities)

        # Create result
        result = PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            confidence=confidence,
            features_used=self.feature_names,
            model_version=self.model_version,
            prediction_time=datetime.now(),
            metadata={
                'model_name': self.model_name,
                'n_samples': len(X),
                'n_features': self.n_features
            }
        )

        # Store recent predictions for monitoring
        self.recent_predictions.append({
            'timestamp': datetime.now(),
            'n_samples': len(X),
            'predictions': predictions[:10] if len(predictions) > 10 else predictions  # Sample
        })

        # Keep only recent predictions (last 100)
        if len(self.recent_predictions) > 100:
            self.recent_predictions = self.recent_predictions[-100:]

        return result

    @abstractmethod
    def _predict_model(self, X: Any) -> np.ndarray:
        """Make predictions with the fitted model. Must be implemented by subclasses."""
        pass

    def _predict_probabilities(self, X: Any) -> np.ndarray:
        """Get prediction probabilities if supported by model."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """
        Evaluate model performance on test data.

        Args:
            X: Feature matrix
            y: True target values

        Returns:
            ModelMetrics with performance scores
        """
        predictions = self.predict(X)
        metrics = self._calculate_metrics(y, predictions.predictions)

        logger.info(f"Model evaluation - Accuracy: {metrics.accuracy:.3f}")
        if metrics.f1_score:
            logger.info(f"F1 Score: {metrics.f1_score:.3f}")
        if metrics.roc_auc:
            logger.info(f"ROC AUC: {metrics.roc_auc:.3f}")

        return metrics

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> ModelMetrics:
        """Calculate performance metrics"""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

            metrics = ModelMetrics()

            if self.model_type == "classifier":
                # Classification metrics
                metrics.accuracy = accuracy_score(y_true, y_pred)

                # Handle multi-class case
                if len(np.unique(y_true)) > 2:
                    avg_method = 'weighted'
                else:
                    avg_method = 'binary'

                try:
                    metrics.precision = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
                    metrics.recall = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
                    metrics.f1_score = f1_score(y_true, y_pred, average=avg_method, zero_division=0)
                except Exception as e:
                    logger.debug(f"Error computing classification metrics: {e}")

                # ROC AUC for binary classification
                if len(np.unique(y_true)) == 2:
                    try:
                        metrics.roc_auc = roc_auc_score(y_true, y_pred)
                    except Exception as e:
                        logger.debug(f"Error computing ROC AUC: {e}")

            elif self.model_type == "regressor":
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                metrics.metrics = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'mae': mae,
                    'r2': r2
                }

            # Trading-specific metrics
            if self.model_type == "classifier":
                try:
                    # Calculate Sharpe ratio if we can infer returns
                    returns = self._calculate_strategy_returns(y_true, y_pred)
                    if returns is not None:
                        metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
                        metrics.max_drawdown = self._calculate_max_drawdown(returns)
                        metrics.win_rate = float((returns > 0).mean())

                        positive_returns = float(returns[returns > 0].sum())
                        negative_returns = float(abs(returns[returns < 0].sum()))
                        metrics.profit_factor = positive_returns / negative_returns if negative_returns > 0 else np.inf
                except Exception as e:
                    logger.debug(f"Error computing trading metrics: {e}")

            return metrics

        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return ModelMetrics()

    def _calculate_strategy_returns(self, y_true: pd.Series, y_pred: np.ndarray) -> Optional[pd.Series]:
        """Calculate strategy returns from predictions and true values"""
        try:
            # This is a simplified approach - assumes binary classification
            # and that y_true represents direction (1 for up, -1 for down, 0 for sideways)
            if len(np.unique(y_true)) <= 3:  # Direction prediction
                # Convert predictions to signals
                signals = np.where(y_pred == 1, 1, np.where(y_pred == -1, -1, 0))

                # Calculate returns (simplified - assumes perfect entry/exit)
                returns = signals * np.where(y_true == 1, 0.01, np.where(y_true == -1, -0.01, 0))
                return pd.Series(returns)
        except Exception as e:
            logger.debug(f"Error calculating strategy returns: {e}")
        return None

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = float(returns.mean()) * 252 - risk_free_rate  # Annualized
        volatility = float(returns.std()) * np.sqrt(252)  # Annualized
        return excess_returns / volatility if volatility > 0 else 0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_prediction_confidence(self, predictions: np.ndarray,
                                       probabilities: Optional[np.ndarray]) -> np.ndarray:
        """Calculate confidence scores for predictions"""
        if probabilities is not None:
            # For classification: use max probability as confidence
            confidence = np.max(probabilities, axis=1)
        else:
            # Default confidence (no information)
            confidence = np.full(len(predictions), 0.5)

        return confidence

    def _calculate_feature_importance(self) -> None:
        """Calculate feature importance if model supports it"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.Series(
                    self.model.feature_importances_,
                    index=self.feature_names
                ).sort_values(ascending=False)
            elif hasattr(self.model, 'coef_'):
                # For linear models, use absolute coefficients
                coef = self.model.coef_
                if coef.ndim > 1:
                    coef = coef[0]  # Take first class for binary classification
                self.feature_importance = pd.Series(
                    np.abs(coef),
                    index=self.feature_names
                ).sort_values(ascending=False)
        except Exception as e:
            logger.debug(f"Could not calculate feature importance: {e}")

    def _validate_training_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate and clean training data"""
        # Remove NaN targets
        valid_idx = y.notna()
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        if len(X_clean) == 0:
            raise ValueError("No valid training samples after removing NaN targets")

        # Remove constant features
        constant_features = X_clean.columns[X_clean.nunique() <= 1]
        if len(constant_features) > 0:
            logger.warning(f"Removing {len(constant_features)} constant features")
            X_clean = X_clean.drop(columns=constant_features)

        logger.info(f"Training data: {len(X_clean)} samples, {X_clean.shape[1]} features")
        return X_clean, y_clean

    def _validate_prediction_features(self, X: pd.DataFrame) -> None:
        """Validate that prediction features match training features"""
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing features for prediction: {missing_features}")

            extra_features = set(X.columns) - set(self.feature_names)
            if extra_features:
                logger.debug(f"Extra features in prediction data: {extra_features}")

    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        """Get top N most important features"""
        if self.feature_importance is not None:
            return self.feature_importance.head(top_n)
        return pd.Series()

    def save_model(self, filepath: str) -> None:
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'model_type': self.model_type,
                'model_version': self.model_version,
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names,
                'n_features': self.n_features,
                'feature_importance': self.feature_importance,
                'created_at': self.created_at,
                'last_trained': self.last_trained,
                'last_updated': self.last_updated,
                'training_metrics': self.training_metrics,
                'validation_metrics': self.validation_metrics,
                'config': self.config
            }

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, filepath: str) -> 'BasePredictor':
        """Load model from disk"""
        try:
            model_data = joblib.load(filepath)

            # Restore model state
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.model_type = model_data['model_type']
            self.model_version = model_data['model_version']
            self.is_fitted = model_data['is_fitted']
            self.feature_names = model_data['feature_names']
            self.n_features = model_data['n_features']
            self.feature_importance = model_data.get('feature_importance')
            self.created_at = model_data['created_at']
            self.last_trained = model_data.get('last_trained')
            self.last_updated = model_data.get('last_updated')
            self.training_metrics = model_data.get('training_metrics')
            self.validation_metrics = model_data.get('validation_metrics')
            self.config = model_data.get('config', {})

            logger.info(f"Model loaded from {filepath}")
            return self

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'model_version': self.model_version,
            'is_fitted': self.is_fitted,
            'n_features': self.n_features,
            'feature_names': self.feature_names[:10] if self.feature_names else [],  # First 10
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'training_metrics': self._metrics_to_dict(self.training_metrics),
            'validation_metrics': self._metrics_to_dict(self.validation_metrics),
            'top_features': self.get_feature_importance(5).to_dict() if self.feature_importance is not None else {},
            'recent_predictions_count': len(self.recent_predictions)
        }

    def _metrics_to_dict(self, metrics: Optional[ModelMetrics]) -> Optional[Dict]:
        """Convert ModelMetrics to dictionary"""
        if metrics is None:
            return None

        return {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'roc_auc': metrics.roc_auc,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'additional_metrics': metrics.metrics
        }