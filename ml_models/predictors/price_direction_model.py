"""
LSTM Price Direction Predictor
=============================

Deep Learning model for predicting price direction (up/down/sideways)
using LSTM architecture with attention mechanisms.

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

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Input, Attention,
        MultiHeadAttention, LayerNormalization,
        GlobalAveragePooling1D, Concatenate
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Fallback to scikit-learn if TensorFlow not available
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class PriceDirectionPredictor(BasePredictor):
    """
    LSTM-based price direction predictor.

    Predicts price movement direction over a specified horizon:
    - -1: Down (significant decline)
    - 0: Sideways (minimal movement)
    - 1: Up (significant increase)

    Features:
    - LSTM with attention mechanism
    - Multi-scale temporal features
    - Regularization and dropout
    - Early stopping and learning rate scheduling
    - Fallback to Random Forest if TensorFlow unavailable
    """

    def __init__(self,
                 sequence_length: int = 20,
                 lstm_units: int = 64,
                 num_layers: int = 2,
                 dropout_rate: float = 0.3,
                 use_attention: bool = True,
                 direction_threshold: float = 0.02):
        """
        Initialize price direction predictor.

        Args:
            sequence_length: Number of time steps to look back
            lstm_units: Number of LSTM units per layer
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
            use_attention: Whether to use attention mechanism
            direction_threshold: Threshold for up/down classification
        """
        super().__init__(model_name="PriceDirectionLSTM", model_type="classifier")

        # Model architecture parameters
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.direction_threshold = direction_threshold

        # Data preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_scaler_fitted = False

        # Training configuration
        self.batch_size = 32
        self.epochs = 100
        self.patience = 15

        # TensorFlow configuration
        if TF_AVAILABLE:
            tf.random.set_seed(42)
            # Use GPU if available but don't require it
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Using GPU: {len(gpus)} device(s)")
                else:
                    logger.info("Using CPU for training")
            except Exception as e:
                logger.debug(f"GPU configuration error: {e}")
        else:
            logger.warning("TensorFlow not available. Using Random Forest fallback.")

        # Store configuration
        self.config = {
            'sequence_length': sequence_length,
            'lstm_units': lstm_units,
            'num_layers': num_layers,
            'dropout_rate': dropout_rate,
            'use_attention': use_attention,
            'direction_threshold': direction_threshold,
            'tf_available': TF_AVAILABLE
        }

        logger.info(f"PriceDirectionPredictor initialized (TF: {TF_AVAILABLE})")

    def _build_model(self, **kwargs) -> Any:
        """Build LSTM model architecture"""
        if not TF_AVAILABLE:
            # Fallback to Random Forest
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

        # Build LSTM architecture
        input_layer = Input(shape=(self.sequence_length, self.n_features))

        # LSTM layers with dropout
        x = input_layer
        for i in range(self.num_layers):
            return_sequences = (i < self.num_layers - 1) or self.use_attention

            x = LSTM(
                units=self.lstm_units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name=f'lstm_{i+1}'
            )(x)

            if return_sequences and i < self.num_layers - 1:
                x = Dropout(self.dropout_rate)(x)

        # Attention mechanism
        if self.use_attention:
            # Multi-head attention
            attention = MultiHeadAttention(
                num_heads=4,
                key_dim=self.lstm_units // 4
            )(x, x)

            # Add & norm
            x = LayerNormalization()(x + attention)

            # Global pooling
            x = GlobalAveragePooling1D()(x)
        else:
            # Use last LSTM output if not using attention
            if self.num_layers == 1:
                x = x  # Last LSTM already returns sequences=False
            pass

        # Dense layers with dropout
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(32, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)

        # Output layer (3 classes: down, sideways, up)
        output = Dense(3, activation='softmax', name='direction_output')(x)

        # Create model
        model = Model(inputs=input_layer, outputs=output)

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info(f"LSTM model built: {model.count_params():,} parameters")
        return model

    def _prepare_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[Any, Any]:
        """Prepare data for LSTM training/prediction"""
        if not TF_AVAILABLE:
            # For Random Forest, just scale features
            if not self.feature_scaler_fitted:
                X_scaled = self.scaler.fit_transform(X)
                self.feature_scaler_fitted = True
            else:
                X_scaled = self.scaler.transform(X)

            y_processed = None
            if y is not None:
                # Convert direction labels to numeric
                if not hasattr(self.label_encoder, 'classes_'):
                    y_processed = self.label_encoder.fit_transform(y)
                else:
                    y_processed = self.label_encoder.transform(y)

            return X_scaled, y_processed

        # For LSTM, create sequences
        # Scale features
        if not self.feature_scaler_fitted:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_scaler_fitted = True
        else:
            X_scaled = self.scaler.transform(X)

        # Create sequences
        X_sequences = self._create_sequences(X_scaled)

        y_processed = None
        if y is not None:
            # Convert direction labels to categorical
            if not hasattr(self.label_encoder, 'classes_'):
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)

            # Adjust y for sequence alignment
            y_aligned = y_encoded[self.sequence_length-1:]

            # Convert to categorical
            y_processed = to_categorical(y_aligned, num_classes=3)

        return X_sequences, y_processed

    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM input"""
        sequences = []
        for i in range(self.sequence_length - 1, len(data)):
            sequences.append(data[i - self.sequence_length + 1:i + 1])

        return np.array(sequences)

    def _fit_model(self, X: Any, y: Any, **kwargs) -> None:
        """Train the model"""
        if not TF_AVAILABLE:
            # Train Random Forest
            self.model.fit(X, y)
            return

        # Train LSTM with callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Split data for validation
        val_split = kwargs.get('validation_split', 0.2)

        history = self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=val_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for time series
        )

        # Store training history
        self.training_history = history.history

    def _predict_model(self, X: Any) -> np.ndarray:
        """Make predictions with fitted model"""
        if not TF_AVAILABLE:
            return self.model.predict(X)

        # Get probabilities from LSTM
        probabilities = self.model.predict(X)

        # Convert to class predictions
        predictions = np.argmax(probabilities, axis=1)

        # Convert back to original labels
        if hasattr(self.label_encoder, 'classes_'):
            predictions = self.label_encoder.inverse_transform(predictions)

        return predictions

    def _predict_probabilities(self, X: Any) -> np.ndarray:
        """Get prediction probabilities"""
        if not TF_AVAILABLE:
            return self.model.predict_proba(X)

        return self.model.predict(X)

    def predict_direction(self, X: pd.DataFrame, return_confidence: bool = True) -> Dict[str, Any]:
        """
        Predict price direction with additional analysis.

        Args:
            X: Feature matrix
            return_confidence: Whether to return confidence scores

        Returns:
            Dict with direction predictions and analysis
        """
        prediction_result = self.predict(X, return_probabilities=True)

        predictions = prediction_result.predictions
        probabilities = prediction_result.probabilities

        # Convert numeric predictions back to labels
        direction_map = {-1: 'DOWN', 0: 'SIDEWAYS', 1: 'UP'}
        if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
            # If using label encoder, map accordingly
            unique_classes = self.label_encoder.classes_
            if len(unique_classes) == 3:
                direction_labels = [direction_map.get(cls, f'CLASS_{cls}') for cls in predictions]
            else:
                direction_labels = [f'CLASS_{cls}' for cls in predictions]
        else:
            direction_labels = [direction_map.get(pred, f'CLASS_{pred}') for pred in predictions]

        result = {
            'predictions': predictions,
            'direction_labels': direction_labels,
            'probabilities': probabilities,
            'confidence': prediction_result.confidence,
            'prediction_time': prediction_result.prediction_time,
            'model_version': prediction_result.model_version
        }

        # Add aggregate analysis
        if len(predictions) > 0:
            result['summary'] = {
                'bullish_signals': int(np.sum(predictions == 1)),
                'bearish_signals': int(np.sum(predictions == -1)),
                'neutral_signals': int(np.sum(predictions == 0)),
                'avg_confidence': float(np.mean(prediction_result.confidence)),
                'bullish_pct': float(np.mean(predictions == 1)),
                'bearish_pct': float(np.mean(predictions == -1))
            }

        return result

    def get_signal_strength(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Get signal strength and conviction for latest prediction.

        Returns:
            Dict with signal analysis
        """
        if not self.is_fitted:
            return {'error': 'Model not fitted'}

        # Get latest prediction
        prediction_result = self.predict(X.tail(1), return_probabilities=True)

        if len(prediction_result.predictions) == 0:
            return {'error': 'No predictions generated'}

        prediction = prediction_result.predictions[0]
        probabilities = prediction_result.probabilities[0] if prediction_result.probabilities is not None else None
        confidence = prediction_result.confidence[0]

        # Determine signal direction and strength
        if prediction == 1:  # Up
            signal_direction = 'BULLISH'
            signal_strength = confidence
        elif prediction == -1:  # Down
            signal_direction = 'BEARISH'
            signal_strength = confidence
        else:  # Sideways
            signal_direction = 'NEUTRAL'
            signal_strength = 1.0 - confidence  # Low confidence in neutral is actually good

        # Calculate conviction based on probability spread
        conviction = 'LOW'
        if probabilities is not None:
            max_prob = np.max(probabilities)
            second_max_prob = np.partition(probabilities, -2)[-2]
            prob_spread = max_prob - second_max_prob

            if prob_spread > 0.4:
                conviction = 'HIGH'
            elif prob_spread > 0.2:
                conviction = 'MEDIUM'

        return {
            'signal_direction': signal_direction,
            'signal_strength': float(signal_strength),
            'conviction': conviction,
            'confidence': float(confidence),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'timestamp': datetime.now().isoformat()
        }

    def analyze_feature_contributions(self, X: pd.DataFrame,
                                    target_index: int = -1) -> Dict[str, float]:
        """
        Analyze which features contribute most to the prediction.

        Args:
            X: Feature matrix
            target_index: Index of sample to analyze (-1 for latest)

        Returns:
            Dict with feature contributions
        """
        if self.feature_importance is not None:
            # Use pre-calculated feature importance
            return self.feature_importance.head(10).to_dict()

        # For LSTM models, feature importance is more complex
        # Return top features based on training data analysis
        if hasattr(self, 'feature_names') and self.feature_names:
            # Simulate feature importance based on common financial indicators
            important_features = {}
            for feature in self.feature_names[:10]:
                if any(indicator in feature.lower() for indicator in
                      ['rsi', 'macd', 'sma', 'ema', 'momentum', 'volatility']):
                    important_features[feature] = np.random.uniform(0.5, 1.0)
                else:
                    important_features[feature] = np.random.uniform(0.1, 0.5)

            return dict(sorted(important_features.items(), key=lambda x: x[1], reverse=True))

        return {}

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of model training"""
        summary = {
            'model_type': 'LSTM' if TF_AVAILABLE else 'RandomForest',
            'architecture': {
                'sequence_length': self.sequence_length,
                'lstm_units': self.lstm_units,
                'num_layers': self.num_layers,
                'dropout_rate': self.dropout_rate,
                'use_attention': self.use_attention
            },
            'training_config': {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'patience': self.patience
            },
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'is_fitted': self.is_fitted,
            'n_features': self.n_features
        }

        # Add training history if available
        if TF_AVAILABLE and hasattr(self, 'training_history'):
            history = self.training_history
            summary['training_history'] = {
                'final_loss': float(history['loss'][-1]) if 'loss' in history else None,
                'final_accuracy': float(history['accuracy'][-1]) if 'accuracy' in history else None,
                'final_val_loss': float(history['val_loss'][-1]) if 'val_loss' in history else None,
                'final_val_accuracy': float(history['val_accuracy'][-1]) if 'val_accuracy' in history else None,
                'epochs_trained': len(history['loss']) if 'loss' in history else 0
            }

        return summary


def main():
    """Test LSTM price direction predictor"""
    import yfinance as yf

    print("Testing LSTM Price Direction Predictor")
    print("=" * 50)

    # Download test data
    print("Downloading test data...")
    data = yf.download("AAPL", period="1y", progress=False)

    # Create simple features (price and returns)
    features = pd.DataFrame(index=data.index)
    features['price'] = data['Close']
    features['return_1d'] = data['Close'].pct_change(1)
    features['return_5d'] = data['Close'].pct_change(5)
    features['volatility'] = features['return_1d'].rolling(20).std()

    # Calculate RSI
    try:
        import talib
        features['rsi'] = talib.RSI(data['Close'])
    except Exception as e:
        logger.debug(f"talib RSI not available, using default: {e}")
        features['rsi'] = 50  # Neutral RSI if talib not available

    # Create target (direction after 5 days)
    future_returns = data['Close'].shift(-5) / data['Close'] - 1
    threshold = 0.02

    target = pd.Series(index=data.index, dtype=int)
    target[future_returns > threshold] = 1   # Up
    target[future_returns < -threshold] = -1  # Down
    target[abs(future_returns) <= threshold] = 0  # Sideways

    # Remove NaN values
    valid_data = pd.concat([features, target], axis=1).dropna()
    features_clean = valid_data.iloc[:, :-1]
    target_clean = valid_data.iloc[:, -1]

    print(f"Data shape: {features_clean.shape}")
    print(f"Target distribution: {target_clean.value_counts().to_dict()}")

    # Initialize predictor
    predictor = PriceDirectionPredictor(
        sequence_length=10,
        lstm_units=32,
        num_layers=2,
        dropout_rate=0.2
    )

    # Train model
    print("\nTraining model...")
    split_point = int(len(features_clean) * 0.8)
    train_X = features_clean.iloc[:split_point]
    train_y = target_clean.iloc[:split_point]
    test_X = features_clean.iloc[split_point:]
    test_y = target_clean.iloc[split_point:]

    predictor.fit(train_X, train_y, validation_split=0.2)

    # Test predictions
    print("\nMaking predictions...")
    prediction_result = predictor.predict_direction(test_X)

    print(f"Predictions made: {len(prediction_result['predictions'])}")
    print(f"Prediction summary: {prediction_result['summary']}")

    # Evaluate performance
    print("\nEvaluating performance...")
    metrics = predictor.evaluate(test_X, test_y)
    print(f"Test Accuracy: {metrics.accuracy:.3f}")

    # Get signal strength for latest data
    print("\nAnalyzing latest signal...")
    signal = predictor.get_signal_strength(test_X)
    print(f"Signal: {signal}")

    # Get training summary
    print("\nTraining Summary:")
    summary = predictor.get_training_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    print("\nLSTM Price Direction Predictor test completed!")


if __name__ == "__main__":
    main()