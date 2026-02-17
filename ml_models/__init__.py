"""
ML Models Package
================

Advanced machine learning price prediction system for the trading bot.
Provides LSTM/Transformer models for price direction prediction and volatility forecasting.

Key Components:
- predictors/: Core ML prediction models (LSTM, Transformer, etc.)
- features/: Feature engineering and data preprocessing
- training/: Model training pipelines and hyperparameter optimization
- inference/: Real-time prediction engine
- validation/: Backtesting and model validation framework

Author: Trading Bot Arsenal
Created: February 2026
"""

from .inference.prediction_engine import PredictionEngine
from .predictors.price_direction_model import PriceDirectionPredictor
from .predictors.volatility_model import VolatilityPredictor
from .features.feature_engineer import FeatureEngineer
from .training.model_trainer import ModelTrainer

__version__ = "1.0.0"
__author__ = "Trading Bot Arsenal"

# Export main classes
__all__ = [
    "PredictionEngine",
    "PriceDirectionPredictor",
    "VolatilityPredictor",
    "FeatureEngineer",
    "ModelTrainer"
]