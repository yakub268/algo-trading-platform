"""
Real-Time Prediction Engine
===========================

High-performance inference engine for real-time trading predictions.
Handles model loading, caching, and parallel prediction execution.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import time
import threading
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..features.feature_engineer import FeatureEngineer
from ..predictors.price_direction_model import PriceDirectionPredictor
from ..predictors.volatility_model import VolatilityPredictor
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    """Container for prediction request"""
    symbol: str
    data: pd.DataFrame
    prediction_types: List[str]  # ['direction', 'volatility', 'regime']
    horizon: int = 5
    features_only: bool = False
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.request_id is None:
            self.request_id = f"{self.symbol}_{int(time.time())}"


@dataclass
class PredictionResponse:
    """Container for prediction response"""
    request_id: str
    symbol: str
    predictions: Dict[str, Any]
    features: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    timestamp: Optional[datetime] = None
    success: bool = True
    error: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PredictionEngine:
    """
    High-performance real-time prediction engine.

    Features:
    - Multi-model ensemble predictions
    - Parallel processing for multiple assets
    - Feature caching and optimization
    - Real-time model updates
    - Performance monitoring
    - Fallback mechanisms
    """

    def __init__(self,
                 model_path: str = None,
                 cache_size: int = 1000,
                 max_workers: int = 4,
                 enable_caching: bool = True):
        """
        Initialize prediction engine.

        Args:
            model_path: Directory containing trained models
            cache_size: Maximum number of cached feature sets
            max_workers: Maximum number of worker threads
            enable_caching: Enable feature and prediction caching
        """
        self.model_path = model_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.cache_size = cache_size
        self.max_workers = max_workers
        self.enable_caching = enable_caching

        # Core components
        self.feature_engineer = FeatureEngineer()
        self.model_manager = ModelManager(self.model_path)

        # Models
        self.models = {}
        self.model_versions = {}

        # Caching
        self.feature_cache = {}
        self.prediction_cache = {}
        self.cache_timestamps = {}

        # Performance monitoring
        self.performance_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'avg_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Thread safety
        self.cache_lock = threading.Lock()
        self.stats_lock = threading.Lock()

        # Initialize models
        self._initialize_models()

        logger.info(f"PredictionEngine initialized (workers: {max_workers}, caching: {enable_caching})")

    def _initialize_models(self):
        """Initialize default models"""
        try:
            # Initialize price direction model
            self.models['direction'] = PriceDirectionPredictor(
                sequence_length=15,
                lstm_units=64,
                num_layers=2
            )

            # Initialize volatility model
            self.models['volatility'] = VolatilityPredictor(
                model_type="ensemble",
                sequence_length=20
            )

            logger.info("Default models initialized")

            # Try to load pre-trained models
            self._load_models()

        except Exception as e:
            logger.warning(f"Model initialization error: {e}")

    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            loaded_models = self.model_manager.load_all_models()
            for name, model in loaded_models.items():
                if model is not None:
                    self.models[name] = model
                    logger.info(f"Loaded pre-trained model: {name}")
        except Exception as e:
            logger.debug(f"Could not load pre-trained models: {e}")

    def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make predictions for a single asset.

        Args:
            request: Prediction request

        Returns:
            Prediction response
        """
        start_time = time.time()

        try:
            # Check cache first
            if self.enable_caching:
                cached_response = self._get_cached_prediction(request)
                if cached_response:
                    self._update_stats('cache_hit')
                    return cached_response

            self._update_stats('cache_miss')

            # Engineer features
            features = self._get_features(request.symbol, request.data)

            # Return features only if requested
            if request.features_only:
                return PredictionResponse(
                    request_id=request.request_id,
                    symbol=request.symbol,
                    predictions={},
                    features=self._serialize_features(features),
                    processing_time=time.time() - start_time,
                    success=True
                )

            # Make predictions
            predictions = {}
            for pred_type in request.prediction_types:
                try:
                    pred_result = self._predict_type(pred_type, features, request)
                    predictions[pred_type] = pred_result
                except Exception as e:
                    logger.warning(f"Prediction failed for {pred_type}: {e}")
                    predictions[pred_type] = {'error': str(e)}

            # Create response
            response = PredictionResponse(
                request_id=request.request_id,
                symbol=request.symbol,
                predictions=predictions,
                features=self._serialize_features(features) if len(predictions) == 0 else None,
                processing_time=time.time() - start_time,
                metadata={
                    'model_versions': self.model_versions,
                    'feature_count': len(features.feature_names) if hasattr(features, 'feature_names') else 0,
                    'data_samples': len(request.data)
                },
                success=True
            )

            # Cache response
            if self.enable_caching:
                self._cache_prediction(request, response)

            self._update_stats('success', time.time() - start_time)
            return response

        except Exception as e:
            logger.error(f"Prediction failed for {request.symbol}: {e}")
            self._update_stats('failure')
            return PredictionResponse(
                request_id=request.request_id,
                symbol=request.symbol,
                predictions={},
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )

    def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """
        Make predictions for multiple assets in parallel.

        Args:
            requests: List of prediction requests

        Returns:
            List of prediction responses
        """
        if len(requests) == 0:
            return []

        if len(requests) == 1:
            return [self.predict_single(requests[0])]

        # Submit all requests to thread pool
        future_to_request = {
            self.executor.submit(self.predict_single, request): request
            for request in requests
        }

        responses = []
        for future in as_completed(future_to_request, timeout=30):
            try:
                response = future.result()
                responses.append(response)
            except Exception as e:
                request = future_to_request[future]
                logger.error(f"Batch prediction failed for {request.symbol}: {e}")
                responses.append(PredictionResponse(
                    request_id=request.request_id,
                    symbol=request.symbol,
                    predictions={},
                    success=False,
                    error=str(e)
                ))

        # Sort responses by request order
        request_ids = [req.request_id for req in requests]
        responses.sort(key=lambda r: request_ids.index(r.request_id) if r.request_id in request_ids else 999)

        return responses

    def _get_features(self, symbol: str, data: pd.DataFrame) -> Any:
        """Get engineered features for prediction"""
        # Check feature cache
        cache_key = f"{symbol}_{hash(str(data.index[-1]))}"

        if self.enable_caching:
            with self.cache_lock:
                if cache_key in self.feature_cache:
                    cache_time = self.cache_timestamps.get(cache_key, datetime.min)
                    if datetime.now() - cache_time < timedelta(minutes=5):  # 5-minute cache
                        return self.feature_cache[cache_key]

        # Engineer features
        feature_set = self.feature_engineer.engineer_features(
            data,
            target_type="direction",  # Default target type
            target_horizon=5
        )

        # Cache features
        if self.enable_caching:
            with self.cache_lock:
                self.feature_cache[cache_key] = feature_set
                self.cache_timestamps[cache_key] = datetime.now()

                # Limit cache size
                if len(self.feature_cache) > self.cache_size:
                    oldest_key = min(self.cache_timestamps.keys(),
                                   key=lambda k: self.cache_timestamps[k])
                    del self.feature_cache[oldest_key]
                    del self.cache_timestamps[oldest_key]

        return feature_set

    def _predict_type(self, pred_type: str, features: Any, request: PredictionRequest) -> Dict[str, Any]:
        """Make prediction for specific type"""
        if pred_type not in self.models:
            raise ValueError(f"No model available for prediction type: {pred_type}")

        model = self.models[pred_type]

        if not model.is_fitted:
            raise ValueError(f"Model {pred_type} is not trained")

        if pred_type == "direction":
            # Price direction prediction
            result = model.predict_direction(
                features.features,
                return_confidence=True
            )

            # Add signal strength analysis
            signal = model.get_signal_strength(features.features)
            result['signal_analysis'] = signal

            return result

        elif pred_type == "volatility":
            # Volatility prediction
            result = model.predict_volatility(
                features.features,
                horizon=request.horizon
            )

            # Add volatility regime analysis
            regime = model.get_volatility_regime(features.features)
            result['regime_analysis'] = regime

            return result

        else:
            # Generic prediction
            pred_result = model.predict(features.features)
            return {
                'predictions': pred_result.predictions.tolist(),
                'confidence': pred_result.confidence.tolist() if pred_result.confidence is not None else None,
                'metadata': pred_result.metadata
            }

    def _serialize_features(self, features: Any) -> Dict[str, float]:
        """Serialize features for response"""
        if hasattr(features, 'features'):
            # Return last row (latest features) as dict
            latest_features = features.features.iloc[-1].to_dict()
            return {k: float(v) if not pd.isna(v) else 0.0 for k, v in latest_features.items()}
        return {}

    def _get_cached_prediction(self, request: PredictionRequest) -> Optional[PredictionResponse]:
        """Get cached prediction if available"""
        cache_key = f"{request.symbol}_{request.prediction_types}_{hash(str(request.data.index[-1]))}"

        with self.cache_lock:
            if cache_key in self.prediction_cache:
                cache_time = self.cache_timestamps.get(cache_key, datetime.min)
                if datetime.now() - cache_time < timedelta(minutes=2):  # 2-minute cache for predictions
                    cached = self.prediction_cache[cache_key]
                    # Update request ID and timestamp
                    cached.request_id = request.request_id
                    cached.timestamp = datetime.now()
                    return cached

        return None

    def _cache_prediction(self, request: PredictionRequest, response: PredictionResponse):
        """Cache prediction response"""
        cache_key = f"{request.symbol}_{request.prediction_types}_{hash(str(request.data.index[-1]))}"

        with self.cache_lock:
            self.prediction_cache[cache_key] = response
            self.cache_timestamps[cache_key] = datetime.now()

            # Limit cache size
            if len(self.prediction_cache) > self.cache_size // 2:
                oldest_key = min(
                    [k for k in self.cache_timestamps.keys() if k.startswith(request.symbol)],
                    key=lambda k: self.cache_timestamps[k],
                    default=None
                )
                if oldest_key:
                    del self.prediction_cache[oldest_key]
                    del self.cache_timestamps[oldest_key]

    def _update_stats(self, stat_type: str, processing_time: float = 0.0):
        """Update performance statistics"""
        with self.stats_lock:
            self.performance_stats['total_predictions'] += 1

            if stat_type == 'success':
                self.performance_stats['successful_predictions'] += 1
            elif stat_type == 'failure':
                self.performance_stats['failed_predictions'] += 1
            elif stat_type == 'cache_hit':
                self.performance_stats['cache_hits'] += 1
            elif stat_type == 'cache_miss':
                self.performance_stats['cache_misses'] += 1

            if processing_time > 0:
                # Update running average
                current_avg = self.performance_stats['avg_processing_time']
                total = self.performance_stats['successful_predictions']
                self.performance_stats['avg_processing_time'] = (
                    (current_avg * (total - 1) + processing_time) / total
                )

    def update_model(self, model_name: str, model: Any):
        """Update a model in the engine"""
        self.models[model_name] = model
        self.model_versions[model_name] = getattr(model, 'model_version', '1.0.0')

        # Clear relevant caches
        if self.enable_caching:
            with self.cache_lock:
                # Clear prediction cache for this model type
                keys_to_remove = [k for k in self.prediction_cache.keys() if model_name in k]
                for key in keys_to_remove:
                    if key in self.prediction_cache:
                        del self.prediction_cache[key]
                    if key in self.cache_timestamps:
                        del self.cache_timestamps[key]

        logger.info(f"Model {model_name} updated")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        with self.stats_lock:
            stats = self.performance_stats.copy()

        # Add cache statistics
        if self.enable_caching:
            with self.cache_lock:
                stats['cache_size'] = len(self.feature_cache)
                stats['prediction_cache_size'] = len(self.prediction_cache)

        # Calculate derived metrics
        total = stats['total_predictions']
        if total > 0:
            stats['success_rate'] = stats['successful_predictions'] / total
            stats['failure_rate'] = stats['failed_predictions'] / total

            if stats['cache_hits'] + stats['cache_misses'] > 0:
                stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])

        return stats

    def clear_cache(self):
        """Clear all caches"""
        if self.enable_caching:
            with self.cache_lock:
                self.feature_cache.clear()
                self.prediction_cache.clear()
                self.cache_timestamps.clear()

        logger.info("All caches cleared")

    def warmup(self, symbols: List[str], sample_data: Dict[str, pd.DataFrame]):
        """Warm up the engine by pre-computing features"""
        logger.info(f"Warming up engine for {len(symbols)} symbols...")

        warmup_requests = [
            PredictionRequest(
                symbol=symbol,
                data=data,
                prediction_types=['direction'],
                features_only=True
            )
            for symbol, data in sample_data.items()
            if symbol in symbols
        ]

        # Process warmup requests
        self.predict_batch(warmup_requests)

        logger.info("Engine warmup completed")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the prediction engine"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_loaded': len(self.models),
            'models_fitted': sum(1 for m in self.models.values() if hasattr(m, 'is_fitted') and m.is_fitted),
            'cache_enabled': self.enable_caching,
            'thread_pool_active': not self.executor._shutdown
        }

        # Check model health
        model_health = {}
        for name, model in self.models.items():
            try:
                model_health[name] = {
                    'fitted': hasattr(model, 'is_fitted') and model.is_fitted,
                    'version': getattr(model, 'model_version', 'unknown'),
                    'last_trained': getattr(model, 'last_trained', None)
                }
            except Exception as e:
                model_health[name] = {'error': str(e)}

        health['models'] = model_health

        return health

    def shutdown(self):
        """Shutdown the prediction engine"""
        logger.info("Shutting down prediction engine...")
        self.executor.shutdown(wait=True)
        self.clear_cache()
        logger.info("Prediction engine shutdown complete")


def main():
    """Test prediction engine"""
    import yfinance as yf

    print("Testing Prediction Engine")
    print("=" * 40)

    # Initialize engine
    engine = PredictionEngine(max_workers=2, enable_caching=True)

    # Download test data (crypto only — stocks disabled due to PDT rule)
    print("Downloading test data...")
    symbols = ['BTC-USD', 'ETH-USD']
    test_data = {}

    for symbol in symbols:
        data = yf.download(symbol, period="6m", progress=False)
        test_data[symbol] = data

    # Warmup engine
    engine.warmup(symbols, test_data)

    # Test single prediction
    print("\nTesting single prediction...")
    request = PredictionRequest(
        symbol='BTC-USD',
        data=test_data['BTC-USD'],
        prediction_types=['direction', 'volatility']
    )

    response = engine.predict_single(request)
    print(f"Success: {response.success}")
    print(f"Processing time: {response.processing_time:.3f}s")
    print(f"Predictions: {list(response.predictions.keys())}")

    # Test batch prediction
    print("\nTesting batch prediction...")
    batch_requests = [
        PredictionRequest(
            symbol=symbol,
            data=test_data[symbol],
            prediction_types=['direction']
        )
        for symbol in symbols
    ]

    batch_responses = engine.predict_batch(batch_requests)
    print(f"Batch size: {len(batch_responses)}")
    print(f"All successful: {all(r.success for r in batch_responses)}")

    # Performance stats
    print("\nPerformance Statistics:")
    stats = engine.get_performance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Health check
    print("\nHealth Check:")
    health = engine.health_check()
    print(f"Status: {health['status']}")
    print(f"Models loaded: {health['models_loaded']}")

    # Shutdown
    engine.shutdown()
    print("\nPrediction engine test completed!")


if __name__ == "__main__":
    main()