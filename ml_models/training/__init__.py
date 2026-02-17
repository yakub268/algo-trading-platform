"""
Training Package
===============

Model training pipeline and utilities.
"""

import logging

logger = logging.getLogger('MLTraining')

from .model_trainer import ModelTrainer

try:
    from .hyperparameter_optimizer import HyperparameterOptimizer
except ImportError:
    HyperparameterOptimizer = None
    logger.warning("hyperparameter_optimizer module not available - HyperparameterOptimizer disabled")

__all__ = [
    "ModelTrainer",
    "HyperparameterOptimizer"
]