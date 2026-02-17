"""
ML Ensemble Bot
Strategy: XGBoost on engineered features (BB, EMA, ROC, RSI, Volume, ATR)
- If model exists: use XGBoost predictions
- If no model: fall back to rule-based majority vote
- Includes train() method for model building
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
import traceback
import pickle

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType

try:
    from bots.alpaca_crypto_client import AlpacaCryptoClient
except ImportError:
    AlpacaCryptoClient = None

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class MLEnsembleBot(FleetBot):
    """
    XGBoost ensemble with feature engineering.

    Features:
    - BB position (price relative to bands)
    - EMA 9, 21, 50 crossover signals
    - ROC (Rate of Change) 5, 10, 20 periods
    - RSI 14
    - Volume ratio (current/average)
    - ATR 14 for volatility

    Logic:
    1. If XGBoost model exists → use predictions
    2. If no model → rule-based majority vote
    """

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="ML-Ensemble",
                bot_type=BotType.CRYPTO,
                symbols=["BTC/USD", "ETH/USD", "SOL/USD"],
                schedule_seconds=600,
                max_position_usd=45.0,
                max_daily_trades=15,
                min_confidence=0.60,
                enabled=True,
                paper_mode=True,
                extra={'base_position_size': 25.0}
            )

        super().__init__(config)

        # Model paths
        self.model_dir = os.path.join(_PROJECT_ROOT, 'ml_models', 'trained')
        os.makedirs(self.model_dir, exist_ok=True)

        # Loaded models (symbol -> model)
        self.models = {}

        # Alpaca client
        self.client = None
        if AlpacaCryptoClient:
            try:
                self.client = AlpacaCryptoClient()
            except Exception as e:
                self.logger.warning(f"Failed to initialize AlpacaCryptoClient: {e}")

        # Load existing models
        self._load_models()

    def _load_models(self):
        """Load pre-trained XGBoost models."""
        if not xgb:
            self.logger.warning("XGBoost not available, will use rule-based fallback")
            return

        for symbol in self.config.symbols:
            model_file = self._get_model_path(symbol)
            if os.path.exists(model_file):
                try:
                    with open(model_file, 'rb') as f:
                        self.models[symbol] = pickle.load(f)
                    self.logger.info(f"Loaded XGBoost model for {symbol}")
                except Exception as e:
                    self.logger.error(f"Failed to load model for {symbol}: {e}")
            else:
                self.logger.info(f"No model found for {symbol}, will use rule-based")

    def _get_model_path(self, symbol: str) -> str:
        """Get model file path for symbol."""
        clean_symbol = symbol.replace('/', '_')
        return os.path.join(self.model_dir, f'fleet_xgb_{clean_symbol}.pkl')

    def scan(self) -> List[FleetSignal]:
        """Scan for ML-based signals."""
        if not self.client:
            self.logger.error("AlpacaCryptoClient not available")
            return []

        if np is None or pd is None:
            self.logger.error("numpy/pandas not available")
            return []

        signals = []

        for symbol in self.config.symbols:
            try:
                signal = self._scan_symbol(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
                self.logger.debug(traceback.format_exc())

        return signals

    def _scan_symbol(self, symbol: str) -> Optional[FleetSignal]:
        """Scan single symbol for ML signal."""
        try:
            # Fetch candles
            candles = self.client.get_candles(
                symbol=symbol,
                granularity='5Min',
                limit=200
            )

            if not candles or len(candles) < 100:
                self.logger.debug(f"{symbol}: insufficient candles ({len(candles) if candles else 0})")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Engineer features
            df = self._engineer_features(df)

            # Drop NaN rows
            df = df.dropna()

            if len(df) < 10:
                return None

            # Get current features
            feature_cols = [
                'bb_position', 'ema9', 'ema21', 'ema50', 'ema9_above_21',
                'ema21_above_50', 'roc5', 'roc10', 'roc20', 'rsi',
                'volume_ratio', 'atr'
            ]

            current_features = df[feature_cols].iloc[-1].values.reshape(1, -1)
            current_price = df['close'].iloc[-1]

            # Predict using model or fallback
            if symbol in self.models:
                prediction, confidence = self._predict_with_model(symbol, current_features)
            else:
                prediction, confidence = self._predict_rule_based(df.iloc[-1])

            # Generate signal if prediction is bullish
            if prediction == 'buy' and confidence >= self.config.min_confidence:
                position_size = self._calculate_position_size(confidence)

                # Dynamic stops based on ATR
                atr = df['atr'].iloc[-1]
                stop_loss = current_price - (2.0 * atr)
                take_profit = current_price + (3.0 * atr)

                signal = FleetSignal(
                    bot_name=self.name,
                    bot_type=self.bot_type.value,
                    symbol=symbol,
                    side='BUY',
                    entry_price=current_price,
                    target_price=take_profit,
                    stop_loss=stop_loss,
                    quantity=position_size / current_price,
                    position_size_usd=position_size,
                    confidence=confidence,
                    edge=(take_profit - current_price) / current_price * 100,
                    reason=f"ML {'model' if symbol in self.models else 'rule-based'} prediction",
                    metadata={
                        'model_used': symbol in self.models,
                        'bb_position': float(df['bb_position'].iloc[-1]),
                        'rsi': float(df['rsi'].iloc[-1]),
                        'ema9_above_21': bool(df['ema9_above_21'].iloc[-1]),
                        'ema21_above_50': bool(df['ema21_above_50'].iloc[-1]),
                        'volume_ratio': float(df['volume_ratio'].iloc[-1]),
                        'atr': float(atr)
                    }
                )

                self.logger.info(
                    f"ML SIGNAL: {symbol} @ ${current_price:.2f} | "
                    f"Conf: {confidence:.2%} | Model: {symbol in self.models} | "
                    f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}"
                )

                return signal

            return None

        except Exception as e:
            self.logger.error(f"Error in _scan_symbol for {symbol}: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer technical features."""
        # Bollinger Bands
        df['bb_sma'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_sma'] + (2.0 * df['bb_std'])
        df['bb_lower'] = df['bb_sma'] - (2.0 * df['bb_std'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # EMAs
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema9_above_21'] = (df['ema9'] > df['ema21']).astype(int)
        df['ema21_above_50'] = (df['ema21'] > df['ema50']).astype(int)

        # Rate of Change
        df['roc5'] = df['close'].pct_change(periods=5) * 100
        df['roc10'] = df['close'].pct_change(periods=10) * 100
        df['roc20'] = df['close'].pct_change(periods=20) * 100

        # RSI (14-period)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Volume ratio
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_avg']

        # ATR (Average True Range)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()

        return df

    def _predict_with_model(self, symbol: str, features: np.ndarray) -> tuple:
        """Predict using XGBoost model."""
        try:
            model = self.models[symbol]
            dmatrix = xgb.DMatrix(features)
            prob = model.predict(dmatrix)[0]

            # Assume binary classification: prob > 0.5 = buy
            if prob > 0.5:
                return 'buy', float(prob)
            else:
                return 'hold', float(1.0 - prob)

        except Exception as e:
            self.logger.error(f"Error predicting with model for {symbol}: {e}")
            return 'hold', 0.0

    def _predict_rule_based(self, row: pd.Series) -> tuple:
        """Fall back to rule-based prediction."""
        signals = []

        # BB position: in lower half = bullish (oversold)
        if row['bb_position'] < 0.3:
            signals.append(1)
        elif row['bb_position'] > 0.7:
            signals.append(-1)
        else:
            signals.append(0)

        # EMA crossovers
        if row['ema9_above_21'] and row['ema21_above_50']:
            signals.append(1)  # Bullish alignment
        elif not row['ema9_above_21'] and not row['ema21_above_50']:
            signals.append(-1)  # Bearish alignment
        else:
            signals.append(0)

        # ROC momentum
        roc_avg = (row['roc5'] + row['roc10'] + row['roc20']) / 3
        if roc_avg > 1.0:
            signals.append(1)
        elif roc_avg < -1.0:
            signals.append(-1)
        else:
            signals.append(0)

        # RSI
        if row['rsi'] < 30:
            signals.append(1)  # Oversold
        elif row['rsi'] > 70:
            signals.append(-1)  # Overbought
        else:
            signals.append(0)

        # Volume surge
        if row['volume_ratio'] > 1.5:
            signals.append(1)  # Volume confirmation
        else:
            signals.append(0)

        # Majority vote
        bullish_count = sum(1 for s in signals if s == 1)
        bearish_count = sum(1 for s in signals if s == -1)
        total_signals = len(signals)

        if bullish_count > bearish_count:
            confidence = 0.5 + (bullish_count / total_signals) * 0.3
            return 'buy', confidence
        else:
            confidence = 0.5 + (bearish_count / total_signals) * 0.3
            return 'hold', confidence

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence."""
        base = (self.config.max_position_usd * 0.5)
        max_size = self.config.max_position_usd

        size = base + (max_size - base) * (confidence - self.config.min_confidence) / (1.0 - self.config.min_confidence)

        return round(size, 2)

    def train(self, symbol: str, train_data: pd.DataFrame, labels: pd.Series):
        """
        Train XGBoost model for a symbol.

        Args:
            symbol: Trading symbol
            train_data: DataFrame with engineered features
            labels: Binary labels (1 = buy, 0 = hold/sell)
        """
        if not xgb:
            self.logger.error("XGBoost not available")
            return

        try:
            feature_cols = [
                'bb_position', 'ema9', 'ema21', 'ema50', 'ema9_above_21',
                'ema21_above_50', 'roc5', 'roc10', 'roc20', 'rsi',
                'volume_ratio', 'atr'
            ]

            X = train_data[feature_cols].values
            y = labels.values

            # Create DMatrix
            dtrain = xgb.DMatrix(X, label=y)

            # Parameters
            params = {
                'objective': 'binary:logistic',
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'logloss',
                'seed': 42
            }

            # Train
            model = xgb.train(params, dtrain, num_boost_round=100)

            # Save model
            model_path = self._get_model_path(symbol)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Load into memory
            self.models[symbol] = model

            self.logger.info(f"Trained and saved XGBoost model for {symbol} to {model_path}")

        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {e}")
            self.logger.debug(traceback.format_exc())


def main():
    """Test runner."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = MLEnsembleBot()

    print(f"\n{'='*60}")
    print(f"ML Ensemble Bot - Test Scan")
    print(f"{'='*60}\n")

    print(f"Models loaded: {list(bot.models.keys())}\n")

    signals = bot.scan()

    if signals:
        print(f"Found {len(signals)} signal(s):\n")
        for sig in signals:
            print(f"Symbol: {sig.symbol}")
            print(f"Direction: {sig.direction.upper()}")
            print(f"Confidence: {sig.confidence:.2%}")
            print(f"Entry: ${sig.entry_price:.2f}")
            print(f"Stop Loss: ${sig.stop_loss:.2f}")
            print(f"Take Profit: ${sig.take_profit:.2f}")
            print(f"Position Size: ${sig.position_size:.2f}")
            print(f"Metadata: {sig.metadata}")
            print(f"{'-'*60}\n")
    else:
        print("No signals generated.\n")


if __name__ == "__main__":
    main()
