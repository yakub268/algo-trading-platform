"""
ML Prediction Bot
================

Advanced ML-powered trading bot that integrates with the master orchestrator.
Uses LSTM models for price direction and volatility forecasting.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import time
import json

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ML infrastructure
try:
    from ml_models.inference.prediction_engine import PredictionEngine, PredictionRequest
    from ml_models.features.feature_engineer import FeatureEngineer, FeatureConfig
    from ml_models.predictors.price_direction_model import PriceDirectionPredictor
    from ml_models.predictors.volatility_model import VolatilityPredictor
    from ml_models.training.model_trainer import ModelTrainer
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    logging.warning(f"ML components not available: {e}")

# Standard trading bot imports
try:
    from utils.data_paths import get_db_path
except ImportError:
    def get_db_path(name): return f"data/{name}"

logger = logging.getLogger(__name__)

# Retry backoff constants
MAX_CONSECUTIVE_FAILURES = 3
FAILURE_COOLDOWN_HOURS = 4


class MLPredictionBot:
    """
    Advanced ML-powered trading bot.

    Features:
    - Real-time price direction prediction using LSTM
    - Volatility forecasting for risk management
    - Multi-asset analysis capabilities
    - Automatic model training and retraining
    - Performance monitoring and optimization
    - Risk-adjusted position sizing
    """

    def __init__(self, paper_mode: bool = True):
        # Respect ALPACA_PAPER_MODE env var — paper-only API keys fail with paper=False
        alpaca_paper_env = os.environ.get('ALPACA_PAPER_MODE', '').lower()
        if alpaca_paper_env == 'true':
            paper_mode = True
        self.paper_mode = paper_mode
        self.bot_name = "ML-Prediction-Bot"

        # ML Infrastructure
        self.ml_available = ML_AVAILABLE
        if self.ml_available:
            self.prediction_engine = PredictionEngine(
                max_workers=2,
                enable_caching=True
            )
            self.feature_engineer = FeatureEngineer()
            self.model_trainer = None  # Initialized when needed
        else:
            logger.warning("ML infrastructure not available. Bot will use simplified predictions.")

        # Initialize Alpaca client for order execution
        self.trading_client = None
        try:
            from alpaca.trading.client import TradingClient
            api_key = os.environ.get('ALPACA_API_KEY', '')
            secret_key = os.environ.get('ALPACA_SECRET_KEY', '')
            if api_key and secret_key:
                self.trading_client = TradingClient(api_key, secret_key, paper=paper_mode)
                # Validate API key on init — catch auth failures early instead of on every order
                try:
                    acct = self.trading_client.get_account()
                    self._buying_power = float(acct.buying_power)
                    logger.info(f"Alpaca client OK for ML bot (account status: {acct.status}, buying_power: ${self._buying_power:,.2f})")
                except Exception as auth_err:
                    logger.error(f"ALPACA AUTH FAILED for ML bot: {auth_err} — orders will fail until key is fixed")
                    self._buying_power = 0.0
                    self.trading_client = None
            else:
                logger.warning("Alpaca credentials not set - execute_trade will use paper simulation")
        except ImportError:
            logger.warning("Alpaca SDK not available - execute_trade will use paper simulation")

        # Trading configuration — crypto pairs ONLY (stocks disabled due to PDT rule)
        # IMPORTANT: This bot must NEVER trade stocks. Only crypto symbols allowed.
        self.default_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        self.ALLOWED_CRYPTO_SUFFIXES = ('-USD', '-USDC', '/USD', '/USDC')
        self.min_confidence_threshold = 0.65
        self._xgboost_models = {}  # Cache for loaded XGBoost models
        self._load_xgboost_models()
        self.max_position_pct = 0.10  # 10% of buying power per trade
        self.min_position_usd = 5.0   # Minimum $5 per trade (avoid dust)
        self.max_position_usd = 75.0  # Maximum $75 per trade
        if not hasattr(self, '_buying_power'):
            self._buying_power = 0.0
        self.prediction_horizon = 5  # days
        self.volatility_lookback = 20

        # Data and state
        self.market_data = {}
        self.current_predictions = {}
        self.model_performance = {}
        self.last_training = None
        self.positions = {}  # Track open positions for exit logic
        self.MAX_OPEN_POSITIONS = 3  # Hard cap: one per symbol max

        # Retry backoff tracking: symbol -> list of failure timestamps
        self._consecutive_failures: Dict[str, List[datetime]] = {}
        self._last_failure_time: Dict[str, datetime] = {}

        # Database for tracking
        self.db_path = get_db_path("ml_predictions.db")
        self._init_database()

        logger.info(f"MLPredictionBot initialized (ML: {self.ml_available}, paper: {paper_mode})")

    def _init_database(self):
        """Initialize database for tracking predictions and performance"""
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    prediction_type TEXT,
                    prediction_value REAL,
                    confidence REAL,
                    actual_outcome REAL,
                    prediction_horizon INTEGER,
                    model_version TEXT,
                    features_json TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    signal_type TEXT,
                    signal_strength REAL,
                    direction TEXT,
                    volatility_forecast REAL,
                    risk_score REAL,
                    recommended_position REAL,
                    executed BOOLEAN DEFAULT FALSE
                )
            ''')

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def _is_symbol_in_cooldown(self, symbol: str) -> bool:
        """Check if a symbol should be skipped due to consecutive failures."""
        if symbol not in self._consecutive_failures:
            return False

        cutoff = datetime.now() - timedelta(hours=FAILURE_COOLDOWN_HOURS)
        # Prune old failures outside the cooldown window
        self._consecutive_failures[symbol] = [
            t for t in self._consecutive_failures[symbol] if t > cutoff
        ]

        if len(self._consecutive_failures[symbol]) >= MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                f"Skipping {symbol}: {len(self._consecutive_failures[symbol])} consecutive "
                f"failures in the last {FAILURE_COOLDOWN_HOURS} hours (cooldown active)"
            )
            return True
        return False

    def _record_failure(self, symbol: str):
        """Record a failure for a symbol."""
        now = datetime.now()
        self._consecutive_failures.setdefault(symbol, []).append(now)
        self._last_failure_time[symbol] = now

    def _clear_failures(self, symbol: str):
        """Clear failure history for a symbol after a successful scan."""
        self._consecutive_failures.pop(symbol, None)
        self._last_failure_time.pop(symbol, None)

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if a symbol is a valid crypto pair. Rejects stock tickers."""
        return any(symbol.upper().endswith(suffix) for suffix in self.ALLOWED_CRYPTO_SUFFIXES)

    def _count_open_positions(self) -> int:
        """Count open ML-Prediction-Bot positions in the orchestrator DB."""
        try:
            import sqlite3
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'data', 'live', 'trading_master.db')
            conn = sqlite3.connect(db_path)
            cur = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE bot_name=? AND status='open'",
                (self.bot_name,)
            )
            count = cur.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.warning(f"Could not check open positions: {e}")
            return 0

    def _get_open_symbols(self) -> set:
        """Get symbols with open ML-Prediction-Bot positions."""
        try:
            import sqlite3
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'data', 'live', 'trading_master.db')
            conn = sqlite3.connect(db_path)
            cur = conn.execute(
                "SELECT symbol FROM trades WHERE bot_name=? AND status='open'",
                (self.bot_name,)
            )
            symbols = {row[0] for row in cur.fetchall()}
            conn.close()
            return symbols
        except Exception as e:
            logger.warning(f"Could not check open symbols: {e}")
            return set()

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol. Used by orchestrator exit checker."""
        try:
            import yfinance as yf
            yf_symbol = symbol.replace('/USD', '-USD').replace('-USDC', '-USD')
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period='1d')
            if len(hist) > 0:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"get_price({symbol}) failed: {e}")
        return None

    def run_scan(self) -> List[Dict[str, Any]]:
        """
        Main scanning method called by master orchestrator.

        Returns:
            List of trading signals with ML predictions
        """
        logger.info("Starting ML prediction scan...")

        # Guard: ensure default_symbols only contains crypto
        self.default_symbols = [s for s in self.default_symbols if self._is_crypto_symbol(s)]
        if not self.default_symbols:
            logger.error("No valid crypto symbols configured — aborting scan")
            return []

        # Overnight filter: 00:00-06:00 MT has 17-31% WR (-$14/49 trades)
        # Skip new entries during these hours to avoid guaranteed losses
        current_hour = datetime.now().hour
        if current_hour < 6:
            logger.info(f"Overnight filter: skipping scan (hour={current_hour}, resume at 06:00)")
            return []

        try:
            # Check open position count from orchestrator DB
            open_position_count = self._count_open_positions()
            if open_position_count >= self.MAX_OPEN_POSITIONS:
                logger.info(f"Position cap reached: {open_position_count}/{self.MAX_OPEN_POSITIONS} open — skipping scan")
                return []

            # Get market data (filter out symbols in cooldown)
            market_data = self._fetch_market_data()
            if not market_data:
                logger.warning("No market data available")
                return []

            # Filter out symbols that are in failure cooldown AND symbols with open positions
            open_symbols = self._get_open_symbols()
            filtered_data = {}
            for symbol, data in market_data.items():
                if self._is_symbol_in_cooldown(symbol):
                    continue
                if symbol in open_symbols:
                    logger.info(f"Skipping {symbol} — already has open position")
                    continue
                filtered_data[symbol] = data

            if not filtered_data:
                logger.warning("All symbols are in cooldown or have open positions")
                return []

            # Generate ML predictions: LSTM → XGBoost → RSI fallback
            signals = []
            if self.ml_available:
                signals = self._generate_ml_signals(filtered_data)
            if not signals and self._xgboost_models:
                signals = self._generate_xgboost_signals(filtered_data)
            if not signals:
                signals = self._generate_fallback_signals(filtered_data)

            # Track successes and failures
            symbols_with_signals = {s['symbol'] for s in signals}
            for symbol in filtered_data:
                if symbol in symbols_with_signals:
                    self._clear_failures(symbol)
                else:
                    self._record_failure(symbol)

            # Store predictions in database
            self._store_predictions(signals)

            logger.info(f"Generated {len(signals)} ML signals")
            return signals

        except Exception as e:
            logger.error(f"ML prediction scan failed: {e}")
            # Record failure for all symbols that were attempted
            for symbol in self.default_symbols:
                self._record_failure(symbol)
            return []

    def _fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch market data for analysis"""
        market_data = {}

        try:
            import yfinance as yf

            for symbol in self.default_symbols:
                try:
                    # Fetch recent data (3 months for sufficient history)
                    data = yf.download(symbol, period="3mo", progress=False)

                    # Flatten MultiIndex columns from yfinance (e.g. ('Close', 'BTC-USD') → 'Close')
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

                    if len(data) > 50:  # Ensure we have enough data
                        market_data[symbol] = data
                        logger.debug(f"Fetched data for {symbol}: {len(data)} rows")
                    else:
                        logger.warning(f"Insufficient data for {symbol}")

                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")

            self.market_data = market_data
            return market_data

        except ImportError:
            logger.error("yfinance not available for data fetching")
            return {}
        except Exception as e:
            logger.error(f"Market data fetch failed: {e}")
            return {}

    def _generate_ml_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate trading signals using ML models"""
        signals = []

        try:
            # Create prediction requests
            prediction_requests = []
            for symbol, data in market_data.items():
                request = PredictionRequest(
                    symbol=symbol,
                    data=data,
                    prediction_types=['direction', 'volatility'],
                    horizon=self.prediction_horizon
                )
                prediction_requests.append(request)

            # Get predictions from engine
            prediction_responses = self.prediction_engine.predict_batch(prediction_requests)

            # Process predictions into trading signals
            for response in prediction_responses:
                if not response.success:
                    logger.warning(f"Prediction failed for {response.symbol}: {response.error}")
                    continue

                try:
                    signal = self._process_prediction_response(response, market_data[response.symbol])
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.warning(f"Failed to process prediction for {response.symbol}: {e}")

            return signals

        except Exception as e:
            logger.error(f"ML signal generation failed: {e}")
            return []

    def _process_prediction_response(self, response, market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Process ML prediction response into trading signal"""
        try:
            predictions = response.predictions

            # Extract direction prediction
            direction_pred = predictions.get('direction', {})
            if not direction_pred:
                return None

            # Get signal components
            signal_analysis = direction_pred.get('signal_analysis', {})
            signal_direction = signal_analysis.get('signal_direction', 'NEUTRAL')
            signal_strength = signal_analysis.get('signal_strength', 0.5)
            confidence = signal_analysis.get('confidence', 0.5)

            # Skip low confidence signals
            if confidence < self.min_confidence_threshold:
                logger.debug(f"Skipping {response.symbol} - low confidence: {confidence:.2f}")
                return None

            # Extract volatility prediction
            volatility_pred = predictions.get('volatility', {})
            vol_forecast = 0.2  # Default volatility
            if volatility_pred:
                summary = volatility_pred.get('summary', {})
                vol_forecast = summary.get('mean_volatility', 0.2)

            # Calculate risk-adjusted position size
            position_size = self._calculate_position_size(
                signal_strength, confidence, vol_forecast
            )

            # BTC penalty: 31% WR vs 45-46% for ETH/SOL — halve position size
            if 'BTC' in response.symbol:
                position_size *= 0.5

            # Determine action
            action = self._determine_action(signal_direction, signal_strength, confidence)

            # Get current price (ensure it's a scalar, not a Series)
            current_price = float(market_data['Close'].iloc[-1])

            # Create trading signal
            entry_price = current_price
            signal = {
                'symbol': response.symbol,
                'action': action,
                'signal_type': 'ML_PREDICTION',
                'direction': signal_direction,
                'strength': signal_strength,
                'confidence': confidence,
                'volatility_forecast': vol_forecast,
                'price': current_price,
                'stop_loss': entry_price * 0.985,      # 1.5% stop loss (cut losers fast)
                'take_profit': entry_price * 1.02,     # 2% take profit (take modest wins)
                'position_size_usd': position_size,    # USD amount (orchestrator converts to qty)
                'horizon_days': self.prediction_horizon,
                'model_version': response.metadata.get('model_versions', {}),
                'timestamp': datetime.now(),
                'reasoning': f"ML {signal_direction} signal with {confidence:.1%} confidence",

                # Additional ML insights
                'ml_insights': {
                    'direction_probabilities': direction_pred.get('probabilities'),
                    'volatility_level': volatility_pred.get('summary', {}).get('volatility_level', 'UNKNOWN'),
                    'regime_analysis': volatility_pred.get('regime_analysis', {}),
                    'feature_count': response.metadata.get('feature_count', 0)
                }
            }

            return signal

        except Exception as e:
            logger.error(f"Failed to process prediction response: {e}")
            return None

    def _determine_action(self, direction: str, strength: float, confidence: float) -> str:
        """Determine trading action based on ML prediction"""
        # Require high confidence and strength for action
        min_action_threshold = 0.7

        if direction == 'BULLISH' and strength >= min_action_threshold and confidence >= self.min_confidence_threshold:
            return 'buy'
        elif direction == 'BEARISH' and strength >= min_action_threshold and confidence >= self.min_confidence_threshold:
            return 'sell'
        else:
            return 'hold'

    def _calculate_position_size(self, strength: float, confidence: float, volatility: float) -> float:
        """Calculate risk-adjusted position size in USD.

        Returns a dollar amount (not a percentage or coin quantity).
        The orchestrator converts USD -> coin quantity via position_size_usd / price.
        """
        # Refresh buying power periodically (every signal cycle, already cached by Alpaca SDK)
        capital = self._buying_power if self._buying_power > 0 else 500.0  # fallback $500

        # Base: percentage of capital scaled by signal quality
        base_pct = self.max_position_pct * strength * confidence

        # Adjust for volatility (reduce size for high volatility)
        volatility_adjustment = max(0.3, 1.0 - (volatility - 0.15) * 2)
        adjusted_pct = base_pct * volatility_adjustment

        # Convert percentage to USD
        position_usd = adjusted_pct * capital

        # Clamp to min/max USD bounds
        position_usd = max(self.min_position_usd, min(position_usd, self.max_position_usd))

        return position_usd

    def _generate_fallback_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate simple signals when ML is not available"""
        signals = []

        for symbol, data in market_data.items():
            try:
                # Handle MultiIndex columns from yfinance (newer versions)
                if isinstance(data.columns, pd.MultiIndex):
                    # Flatten: ('Close', 'SPY') -> 'Close'
                    data = data.copy()
                    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

                # Get close prices - handle both Series and DataFrame cases
                close_col = data['Close']
                if isinstance(close_col, pd.DataFrame):
                    close_prices = close_col.iloc[:, 0]  # Take first column if DataFrame
                else:
                    close_prices = close_col

                # Ensure we have a proper Series
                if not isinstance(close_prices, pd.Series):
                    logger.warning(f"Unexpected close_prices type for {symbol}: {type(close_prices)}")
                    continue

                # Calculate simple RSI
                delta = close_prices.diff()
                gain = delta.clip(lower=0).rolling(window=14).mean()
                loss = (-delta).clip(lower=0).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                # Safely extract scalar values
                rsi_last = rsi.iloc[-1]
                price_last = close_prices.iloc[-1]

                # Handle case where iloc returns Series/array instead of scalar
                if isinstance(rsi_last, (pd.Series, pd.DataFrame)):
                    rsi_last = rsi_last.iloc[0]
                if isinstance(price_last, (pd.Series, pd.DataFrame)):
                    price_last = price_last.iloc[0]

                # Use .item() if numpy type to guarantee Python scalar
                if hasattr(rsi_last, 'item'):
                    rsi_last = rsi_last.item()
                if hasattr(price_last, 'item'):
                    price_last = price_last.item()

                # Convert to Python float, skip if NaN
                try:
                    current_rsi = float(rsi_last)
                    current_price = float(price_last)
                except (TypeError, ValueError):
                    continue

                # Check for NaN (use math.isnan for guaranteed scalar check)
                if math.isnan(current_rsi) or math.isnan(current_price):
                    continue

                # Generate signal based on RSI
                if current_rsi < 30:  # Oversold
                    action = 'buy'
                    direction = 'BULLISH'
                    strength = (30 - current_rsi) / 30
                elif current_rsi > 70:  # Overbought
                    action = 'sell'
                    direction = 'BEARISH'
                    strength = (current_rsi - 70) / 30
                else:
                    continue  # No signal

                entry_price = current_price
                signal = {
                    'symbol': symbol,
                    'action': action,
                    'signal_type': 'SIMPLE_RSI',
                    'direction': direction,
                    'strength': strength,
                    'confidence': 0.6,  # Fixed confidence for simple signals
                    'volatility_forecast': 0.2,
                    'price': current_price,
                    'stop_loss': entry_price * 0.95,       # 5% stop loss
                    'take_profit': entry_price * 1.08,     # 8% take profit
                    'position_size_usd': self._calculate_position_size(strength, 0.6, 0.2),
                    'timestamp': datetime.now(),
                    'reasoning': f"RSI {direction} signal (RSI: {current_rsi:.1f})",
                    'rsi': current_rsi
                }

                signals.append(signal)

            except Exception as e:
                logger.warning(f"Fallback signal generation failed for {symbol}: {e}")

        return signals

    def _load_xgboost_models(self):
        """Load pre-trained XGBoost models from disk."""
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("xgboost not installed — XGBoost prediction path disabled")
            return

        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'ml_models', 'models')
        for symbol in self.default_symbols:
            safe_name = symbol.replace('/', '_')
            model_path = os.path.join(models_dir, f'xgboost_{safe_name}.json')
            meta_path = os.path.join(models_dir, f'xgboost_{safe_name}_meta.json')
            if os.path.exists(model_path):
                try:
                    model = xgb.XGBClassifier()
                    model.load_model(model_path)
                    meta = {}
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                    self._xgboost_models[symbol] = {'model': model, 'meta': meta}
                    acc = meta.get('metrics', {}).get('accuracy', '?')
                    logger.info(f"Loaded XGBoost model for {symbol} (accuracy: {acc})")
                except Exception as e:
                    logger.warning(f"Failed to load XGBoost model for {symbol}: {e}")

    def _generate_xgboost_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate signals using trained XGBoost models (faster + actually fitted)."""
        if not self._xgboost_models:
            return []

        # Reuse the feature engineering from train_live.py
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                            'ml_models', 'training'))
            from train_live import engineer_features
        except ImportError:
            logger.warning("Cannot import train_live feature engineering")
            return []

        label_map_inv = {0: -1, 1: 0, 2: 1}  # Reverse: 0→bearish, 1→neutral, 2→bullish
        signals = []

        for symbol, data in market_data.items():
            if symbol not in self._xgboost_models:
                continue

            try:
                model_info = self._xgboost_models[symbol]
                model = model_info['model']

                # Flatten MultiIndex columns if needed
                df = data.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

                features = engineer_features(df, symbol)
                if features.empty or len(features) < 5:
                    continue

                # Get latest features row
                latest = features.iloc[[-1]]

                # Predict probabilities
                proba = model.predict_proba(latest)[0]  # [bearish, neutral, bullish]
                pred_class = int(model.predict(latest)[0])
                direction_val = label_map_inv.get(pred_class, 0)

                # Map to signal
                if direction_val == 1:
                    direction = 'BULLISH'
                    strength = float(proba[2])  # Bullish probability
                elif direction_val == -1:
                    direction = 'BEARISH'
                    strength = float(proba[0])  # Bearish probability
                else:
                    continue  # Skip neutral predictions

                confidence = float(max(proba))

                # Determine action
                action = self._determine_action(direction, strength, confidence)

                close_col = df['Close']
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                current_price = float(close_col.iloc[-1])

                signal = {
                    'symbol': symbol,
                    'action': action,
                    'signal_type': 'XGBOOST',
                    'direction': direction,
                    'strength': strength,
                    'confidence': confidence,
                    'volatility_forecast': 0.2,
                    'price': current_price,
                    'stop_loss': current_price * 0.985,
                    'take_profit': current_price * 1.02,
                    'position_size_usd': self._calculate_position_size(strength, confidence, 0.2),
                    'horizon_days': self.prediction_horizon,
                    'timestamp': datetime.now(),
                    'reasoning': f"XGBoost {direction} (prob={strength:.1%}, conf={confidence:.1%})",
                    'ml_insights': {
                        'probabilities': {'bearish': float(proba[0]),
                                          'neutral': float(proba[1]),
                                          'bullish': float(proba[2])},
                        'model_accuracy': model_info['meta'].get('metrics', {}).get('accuracy', 0),
                    }
                }
                signals.append(signal)
                logger.info(f"XGBoost signal: {symbol} {direction} strength={strength:.2f} conf={confidence:.2f} action={action}")

            except Exception as e:
                logger.warning(f"XGBoost prediction failed for {symbol}: {e}")

        return signals

    def _store_predictions(self, signals: List[Dict[str, Any]]):
        """Store predictions in database for performance tracking"""
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for signal in signals:
                cursor.execute('''
                    INSERT INTO ml_signals
                    (timestamp, symbol, signal_type, signal_strength, direction,
                     volatility_forecast, recommended_position, executed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal['timestamp'].isoformat(),
                    signal['symbol'],
                    signal['signal_type'],
                    signal.get('strength', 0),
                    signal.get('direction', 'NEUTRAL'),
                    signal.get('volatility_forecast', 0),
                    signal.get('quantity', 0),
                    False  # Not executed yet
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store predictions: {e}")

    def get_model_status(self) -> Dict[str, Any]:
        """Get current status of ML models"""
        if not self.ml_available:
            return {'status': 'ML_NOT_AVAILABLE'}

        try:
            # Get prediction engine health
            health = self.prediction_engine.health_check()

            # Get performance stats
            perf_stats = self.prediction_engine.get_performance_stats()

            return {
                'status': 'OPERATIONAL',
                'ml_available': True,
                'engine_health': health,
                'performance_stats': perf_stats,
                'last_scan': datetime.now().isoformat(),
                'models_loaded': health.get('models_loaded', 0),
                'models_fitted': health.get('models_fitted', 0)
            }

        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'ml_available': False
            }

    def retrain_models(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Retrain ML models with latest data"""
        if not self.ml_available:
            return {'success': False, 'error': 'ML not available'}

        try:
            # Check if retraining is needed
            if not force_retrain and self.last_training:
                days_since_training = (datetime.now() - self.last_training).days
                if days_since_training < 7:  # Retrain weekly
                    return {'success': False, 'reason': 'Recent training exists'}

            logger.info("Starting model retraining...")

            # Initialize trainer if needed
            if self.model_trainer is None:
                from ml_models.training.model_trainer import ModelTrainer
                self.model_trainer = ModelTrainer()

            # Fetch training data
            training_data = {}
            for symbol in self.default_symbols[:3]:  # Train on subset for speed
                if symbol in self.market_data:
                    training_data[symbol] = self.market_data[symbol]

            if not training_data:
                return {'success': False, 'error': 'No training data available'}

            # Train models
            results = {}
            for symbol, data in training_data.items():
                try:
                    # Train direction model
                    direction_model = self.model_trainer.train_price_direction_model(
                        data, symbol=symbol, quick_train=True
                    )

                    # Update engine with new model
                    if direction_model and direction_model.is_fitted:
                        self.prediction_engine.update_model('direction', direction_model)
                        results[f'{symbol}_direction'] = 'success'

                except Exception as e:
                    logger.warning(f"Training failed for {symbol}: {e}")
                    results[f'{symbol}_direction'] = f'failed: {e}'

            self.last_training = datetime.now()

            return {
                'success': True,
                'training_results': results,
                'trained_at': self.last_training.isoformat(),
                'models_trained': len([r for r in results.values() if r == 'success'])
            }

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from historical predictions"""
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get recent predictions (last 30 days)
            thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()

            cursor.execute('''
                SELECT COUNT(*) as total_signals,
                       AVG(signal_strength) as avg_strength,
                       AVG(CASE WHEN executed = 1 THEN 1.0 ELSE 0.0 END) as execution_rate
                FROM ml_signals
                WHERE timestamp > ?
            ''', (thirty_days_ago,))

            stats = cursor.fetchone()

            cursor.execute('''
                SELECT signal_type, COUNT(*) as count
                FROM ml_signals
                WHERE timestamp > ?
                GROUP BY signal_type
            ''', (thirty_days_ago,))

            signal_types = dict(cursor.fetchall())

            conn.close()

            return {
                'total_signals_30d': stats[0] if stats else 0,
                'avg_signal_strength': stats[1] if stats else 0,
                'execution_rate': stats[2] if stats else 0,
                'signal_types': signal_types,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}

    def execute_trade(self, signal) -> Optional[Dict]:
        """
        Execute a trade via Alpaca.

        Handles both paper simulation and live order submission.
        Called by the master orchestrator after signal generation.
        """
        # Normalize signal to dict
        if isinstance(signal, dict):
            sig = signal
        elif hasattr(signal, '__dict__'):
            sig = vars(signal)
        else:
            logger.error(f"Unsupported signal type: {type(signal)}")
            return None

        symbol = sig.get('symbol', '')
        action = sig.get('action', 'hold')
        price = sig.get('price', 0)

        # Convert position_size_usd to coin quantity if needed
        if 'quantity' in sig:
            quantity = sig['quantity']
        elif 'position_size_usd' in sig and price > 0:
            quantity = sig['position_size_usd'] / price
        else:
            quantity = 0

        if action == 'hold' or quantity <= 0:
            return None

        # CRITICAL: Reject non-crypto symbols (defense against stock trading)
        if not self._is_crypto_symbol(symbol):
            logger.error(f"BLOCKED: Attempted to trade non-crypto symbol '{symbol}'. Only crypto pairs allowed.")
            return None

        trade_record = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now().isoformat(),
            'paper': self.paper_mode,
        }

        if self.paper_mode:
            # Paper mode: simulate fill immediately
            trade_record['status'] = 'filled'
            trade_record['fill_price'] = price
            trade_record['success'] = True

            if action == 'buy':
                self.positions[symbol] = {'entry_price': price, 'quantity': quantity}
                logger.info(f"[PAPER] ML BUY {symbol}: {quantity} @ ${price:.2f}")
            elif action == 'sell' and symbol in self.positions:
                pos = self.positions.pop(symbol)
                pnl = (price - pos['entry_price']) * pos['quantity']
                trade_record['pnl'] = pnl
                logger.info(f"[PAPER] ML SELL {symbol}: {quantity} @ ${price:.2f} | PnL: ${pnl:.2f}")

            return trade_record

        # Live mode: submit via Alpaca
        if self.trading_client is None:
            logger.error("No Alpaca client available for live execution")
            trade_record['status'] = 'failed'
            trade_record['success'] = False
            trade_record['error'] = 'No trading client'
            return trade_record

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_side = OrderSide.BUY if action == 'buy' else OrderSide.SELL

            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )

            # Use order_fill_helper for confirmed fill prices (avoids zero-P&L bug)
            try:
                from utils.order_fill_helper import submit_and_wait_for_fill
                fill_result = submit_and_wait_for_fill(
                    self.trading_client, order_request, timeout=30
                )
                trade_record['status'] = 'filled'
                trade_record['success'] = True
                trade_record['order_id'] = fill_result['order_id']
                trade_record['filled_avg_price'] = str(fill_result['fill_price'])
                trade_record['fill_price'] = fill_result['fill_price']
                trade_record['filled_qty'] = str(fill_result['fill_qty'])
                logger.info(
                    f"[LIVE] ML order FILLED: {action} {symbol} "
                    f"{fill_result['fill_qty']} @ ${fill_result['fill_price']:.6f}"
                )
            except ImportError:
                # Fallback: submit without waiting for fill
                order = self.trading_client.submit_order(order_request)
                trade_record['status'] = str(order.status)
                trade_record['success'] = True
                trade_record['order_id'] = str(order.id)
                if order.filled_avg_price:
                    trade_record['filled_avg_price'] = str(order.filled_avg_price)
                logger.info(f"[LIVE] ML order submitted: {action} {symbol} x{quantity}")
            except (ValueError, RuntimeError, TimeoutError) as e:
                trade_record['status'] = 'failed'
                trade_record['success'] = False
                trade_record['error'] = str(e)
                logger.error(f"[LIVE] ML order fill failed: {action} {symbol}: {e}")
            return trade_record

        except Exception as e:
            logger.error(f"ML order execution failed: {e}")
            trade_record['status'] = 'failed'
            trade_record['success'] = False
            trade_record['error'] = str(e)
            return trade_record

    def cleanup_resources(self):
        """Cleanup ML resources"""
        if self.ml_available and hasattr(self, 'prediction_engine'):
            try:
                self.prediction_engine.shutdown()
                logger.info("ML resources cleaned up")
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")


def main():
    """Test ML prediction bot"""
    print("Testing ML Prediction Bot")
    print("=" * 40)

    # Initialize bot
    bot = MLPredictionBot(paper_mode=True)

    # Check status
    print("Model Status:")
    status = bot.get_model_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Run scan
    print("\nRunning ML scan...")
    signals = bot.run_scan()

    print(f"\nGenerated {len(signals)} signals:")
    for signal in signals[:3]:  # Show first 3 signals
        print(f"  {signal['symbol']}: {signal['action']} ({signal['confidence']:.1%} confidence)")
        print(f"    Direction: {signal['direction']}, Strength: {signal['strength']:.2f}")
        print(f"    Reasoning: {signal['reasoning']}")

    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = bot.get_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Cleanup
    bot.cleanup_resources()

    print("\nML prediction bot test completed!")


if __name__ == "__main__":
    main()