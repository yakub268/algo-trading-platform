"""
Ensemble Prediction Combiner
==============================
Combines LLM predictions, technical indicators, news sentiment,
MTF confluence, macro data, cross-asset signals, crypto derivatives,
on-chain metrics, ML models, and bot consensus into unified prediction scores.
Confidence-weighted position sizing with auto-calibrating weights.

Author: Trading Bot Arsenal
Created: February 2026 | V6+ AI Upgrade
Updated: February 2026 | V7 Crystal Ball Upgrade (8 sources)
"""

import os
import sys
import json
import time
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('EnsembleCombiner')


@dataclass
class EnsemblePrediction:
    """Unified prediction from all signal sources"""
    symbol: str
    direction: str           # bullish/bearish/neutral
    confidence: int          # 0-100 unified score
    components: Dict         # individual source scores
    position_size_pct: float # recommended position size as % of allocation
    reasoning: str
    timestamp: str


# V7: Default weights for 8 sources (sum to 1.0) - auto-calibrating after 50+ predictions
SOURCE_WEIGHTS = {
    'analyst': 0.15,           # Market analyst LLM prediction
    'mtf_confluence': 0.15,    # Multi-timeframe agreement
    'technical': 0.10,         # Raw technical indicators
    'news_sentiment': 0.05,    # News sentiment
    'macro': 0.10,             # Fear/Greed + FRED + FedWatch
    'cross_asset': 0.05,       # DXY, ETH/BTC, BTC.D
    'crypto_derivatives': 0.10, # Funding rates + liquidations + OI
    'ml_model': 0.20,          # ML quantitative predictions
    'bot_consensus': 0.10,     # Inter-strategy agreement
}

# Position sizing tiers based on ensemble confidence
SIZING_TIERS = [
    (85, 1.0),   # 85-100 confidence -> full position
    (70, 0.75),  # 70-84 -> 75% position
    (55, 0.50),  # 55-69 -> 50% position
    (40, 0.25),  # 40-54 -> 25% position
    (0, 0.0),    # below 40 -> no position
]

# Path for persisted weights
WEIGHTS_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "ensemble_weights.db")


class EnsembleCombiner:
    """
    Combines multiple prediction sources into a unified signal.

    V7 Sources (8 total):
    1. Market Analyst (LLM) - directional prediction + confidence
    2. MTF Confluence - multi-timeframe agreement score
    3. Technical Indicators - RSI, EMA, MACD from raw data
    4. News Sentiment - aggregated news sentiment score
    5. Macro - Fear/Greed + FRED + FedWatch (from data_hub)
    6. Cross-Asset - DXY, ETH/BTC, BTC.D (from data_hub)
    7. Crypto Derivatives - Funding rates + liquidations (from data_hub)
    8. ML Model - Quantitative predictions (from PredictionEngine)
    9. Bot Consensus - Inter-strategy agreement (from MetaSignal)

    Output:
    - Unified direction (bullish/bearish/neutral)
    - Weighted confidence score (0-100)
    - Recommended position size
    """

    def __init__(self, market_analyst=None, mtf_engine=None, data_hub=None,
                 prediction_engine=None, meta_signal=None,
                 weights: Dict[str, float] = None):
        self.market_analyst = market_analyst
        self.mtf_engine = mtf_engine
        self.data_hub = data_hub
        self.prediction_engine = prediction_engine
        self.meta_signal = meta_signal
        self.weights = weights or self._load_persisted_weights() or SOURCE_WEIGHTS.copy()
        self._lock = threading.Lock()

        # Accuracy tracking for dynamic weight adjustment
        self._source_accuracy = {source: {'correct': 0, 'total': 0}
                                 for source in SOURCE_WEIGHTS}

        # Calibration counter
        self._accuracy_check_count = 0

        # Cache latest ensemble predictions
        self._predictions: Dict[str, EnsemblePrediction] = {}
        self._yf_cache: Dict[str, tuple] = {}  # symbol -> (data, timestamp)

        logger.info(f"EnsembleCombiner initialized with {len(self.weights)} sources: "
                    f"{list(self.weights.keys())}")

    def predict(self, symbol: str, technical_data: Dict = None,
                news_sentiment: float = 0.0) -> Optional[EnsemblePrediction]:
        """
        Generate ensemble prediction for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
            technical_data: Dict with RSI, EMA, MACD, etc.
            news_sentiment: -1.0 to 1.0 sentiment score

        Returns:
            EnsemblePrediction or None
        """
        components = {}
        weighted_bullish = 0.0
        weighted_bearish = 0.0
        total_weight = 0.0

        # Helper to accumulate a source signal
        def _add_source(name: str, score: Optional[Dict]):
            nonlocal weighted_bullish, weighted_bearish, total_weight
            if score is None:
                return
            components[name] = score
            w = self.weights.get(name, 0)
            if w <= 0:
                return
            if score['direction'] == 'bullish':
                weighted_bullish += w * (score['confidence'] / 100)
            elif score['direction'] == 'bearish':
                weighted_bearish += w * (score['confidence'] / 100)
            total_weight += w

        # 1. Market Analyst prediction
        _add_source('analyst', self._get_analyst_score(symbol))

        # 2. MTF Confluence
        _add_source('mtf_confluence', self._get_mtf_score(symbol))

        # 3. Technical indicators
        if technical_data:
            _add_source('technical', self._score_technicals(technical_data))

        # 4. News sentiment
        if news_sentiment != 0.0:
            _add_source('news_sentiment', self._score_news(news_sentiment))

        # 5. Macro signal (from data_hub)
        if self.data_hub:
            try:
                hub_data = self.data_hub.get_data()
                _add_source('macro', hub_data.get('macro_signal'))
                _add_source('cross_asset', hub_data.get('cross_asset_signal'))
                _add_source('crypto_derivatives', hub_data.get('crypto_derivatives_signal'))
            except Exception:
                pass

        # 8. ML Model predictions
        _add_source('ml_model', self._get_ml_score(symbol))

        # 9. Bot Consensus
        _add_source('bot_consensus', self._get_bot_consensus_score(symbol))

        if total_weight == 0:
            return None

        # Require minimum 2 active sources for a prediction
        active_count = len(components)
        if active_count < 2:
            return None

        # Apply coverage penalty instead of full normalization
        # With few sources, keep confidence proportionally weak
        coverage_penalty = min(1.0, total_weight)
        weighted_bullish *= (1.0 / total_weight) * coverage_penalty
        weighted_bearish *= (1.0 / total_weight) * coverage_penalty

        # Determine direction and confidence
        if weighted_bullish > weighted_bearish:
            direction = 'bullish'
            raw_confidence = weighted_bullish * 100
            opposition_penalty = weighted_bearish * 50
            confidence = max(0, min(100, int(raw_confidence - opposition_penalty)))
        elif weighted_bearish > weighted_bullish:
            direction = 'bearish'
            raw_confidence = weighted_bearish * 100
            opposition_penalty = weighted_bullish * 50
            confidence = max(0, min(100, int(raw_confidence - opposition_penalty)))
        else:
            direction = 'neutral'
            confidence = 0

        # Source agreement bonus: more sources agree = higher conviction
        agreeing_sources = sum(1 for c in components.values()
                              if c.get('direction') == direction)
        if agreeing_sources >= 6:
            confidence = min(100, confidence + 20)
        elif agreeing_sources >= 5:
            confidence = min(100, confidence + 15)
        elif agreeing_sources >= 4:
            confidence = min(100, confidence + 10)
        elif agreeing_sources >= 3:
            confidence = min(100, confidence + 5)

        # Calculate position size
        position_size_pct = self._confidence_to_size(confidence)

        # Build reasoning
        reasoning_parts = []
        for source, score in components.items():
            reasoning_parts.append(
                f"{source}: {score['direction']} ({score['confidence']}%)"
            )
        reasoning = ' | '.join(reasoning_parts)

        prediction = EnsemblePrediction(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            components=components,
            position_size_pct=position_size_pct,
            reasoning=reasoning,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        # Cache
        with self._lock:
            self._predictions[symbol] = prediction

        return prediction

    def predict_all(self, symbols: List[str] = None,
                    technical_data: Dict[str, Dict] = None,
                    news_sentiments: Dict[str, float] = None) -> Dict[str, EnsemblePrediction]:
        """Run ensemble predictions for multiple symbols"""
        if symbols is None:
            symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'DOGE/USD',
                       'XRP/USD', 'AVAX/USD', 'LINK/USD', 'ADA/USD']

        results = {}
        for symbol in symbols:
            tech = (technical_data or {}).get(symbol, {})
            sentiment = (news_sentiments or {}).get(symbol, 0.0)

            pred = self.predict(symbol, tech, sentiment)
            if pred:
                results[symbol] = pred

        # Log summary
        bullish = [s for s, p in results.items() if p.direction == 'bullish' and p.confidence >= 50]
        bearish = [s for s, p in results.items() if p.direction == 'bearish' and p.confidence >= 50]
        active_sources = len([s for s in self.weights if self.weights[s] > 0])
        logger.info(f"Ensemble ({active_sources} sources): {len(bullish)} bullish, "
                    f"{len(bearish)} bearish signals (50%+ conf)")

        for symbol, pred in results.items():
            if pred.confidence >= 50:
                logger.info(f"  {symbol}: {pred.direction.upper()} ({pred.confidence}%) "
                           f"size={pred.position_size_pct:.0%} [{pred.reasoning}]")

        return results

    def get_prediction(self, symbol: str) -> Optional[Dict]:
        """Get cached ensemble prediction for symbol"""
        with self._lock:
            pred = self._predictions.get(symbol)
            if pred:
                return asdict(pred)
        return None

    def get_all_predictions(self) -> Dict[str, Dict]:
        """Get all cached ensemble predictions"""
        with self._lock:
            return {s: asdict(p) for s, p in self._predictions.items()}

    # ---- Source Scorers ----

    def _get_analyst_score(self, symbol: str) -> Optional[Dict]:
        """Extract score from market analyst prediction"""
        if not self.market_analyst:
            return None
        try:
            pred = self.market_analyst.get_prediction_for_symbol(symbol)
            if pred:
                return {
                    'direction': pred['direction'],
                    'confidence': pred['confidence'],
                    'source': 'analyst'
                }
        except Exception:
            pass
        return None

    def _get_mtf_score(self, symbol: str) -> Optional[Dict]:
        """Extract score from MTF confluence"""
        if not self.mtf_engine:
            return None
        try:
            conf = self.mtf_engine.get_confluence(symbol)
            if conf:
                return {
                    'direction': conf['direction'],
                    'confidence': int(conf.get('weighted_score', 0)),
                    'strength': conf.get('strength', 'weak'),
                    'timeframes': conf.get('agreeing_timeframes', []),
                    'source': 'mtf_confluence'
                }
        except Exception:
            pass
        return None

    def _get_ml_score(self, symbol: str) -> Optional[Dict]:
        """Extract score from trained XGBoost model."""
        try:
            import os
            import xgboost as xgb

            ticker = symbol.replace('/', '-')
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'ml_models', 'models', f'xgboost_{ticker}.json'
            )
            if not os.path.exists(model_path):
                return None

            model = xgb.XGBClassifier()
            model.load_model(model_path)

            from ml_models.training.train_live import engineer_features, fetch_training_data

            # Cache yfinance data for 30 minutes
            cache_entry = self._yf_cache.get(ticker)
            if cache_entry and (time.time() - cache_entry[1]) < 1800:
                df = cache_entry[0]
            else:
                df = fetch_training_data(ticker, period='3mo')
                self._yf_cache[ticker] = (df, time.time())

            if len(df) < 30:
                return None
            features = engineer_features(df, ticker)

            latest = features.iloc[[-1]]
            proba = model.predict_proba(latest)[0]  # [bearish, neutral, bullish]
            pred_class = int(model.predict(latest)[0])

            class_map = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
            direction = class_map.get(pred_class, 'neutral')
            conf = int(max(proba) * 100)

            return {
                'direction': direction,
                'confidence': min(90, conf),
                'source': 'ml_model'
            }
        except Exception as e:
            logger.debug(f"ML score failed for {symbol}: {e}")
        return None

    def _get_bot_consensus_score(self, symbol: str) -> Optional[Dict]:
        """Extract score from MetaSignal bot consensus"""
        if not self.meta_signal:
            return None
        try:
            consensus = self.meta_signal.get_consensus(symbol)
            if consensus:
                return {
                    'direction': consensus['direction'],
                    'confidence': consensus['confidence'],
                    'agreement': consensus.get('agreement_ratio', 0),
                    'source': 'bot_consensus'
                }
        except Exception:
            pass
        return None

    def _score_technicals(self, data: Dict) -> Dict:
        """Convert raw technical data into directional score"""
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0

        # RSI
        rsi = data.get('rsi', 50)
        if rsi < 30:
            bullish_signals += 2
        elif rsi < 40:
            bullish_signals += 1
        elif rsi > 70:
            bearish_signals += 2
        elif rsi > 60:
            bearish_signals += 1
        total_signals += 2

        # EMA crossover
        ema_signal = data.get('ema_signal', '')
        if ema_signal == 'bullish':
            bullish_signals += 1.5
        elif ema_signal == 'bearish':
            bearish_signals += 1.5
        total_signals += 1.5

        # MACD
        macd = data.get('macd', data.get('macd_pct', 0))
        if macd > 0:
            bullish_signals += 1
        elif macd < 0:
            bearish_signals += 1
        total_signals += 1

        # Volume
        vol_ratio = data.get('volume_ratio', 1.0)
        if vol_ratio > 1.5:
            if bullish_signals > bearish_signals:
                bullish_signals += 0.5
            else:
                bearish_signals += 0.5
            total_signals += 0.5

        # Bollinger position
        bb_pos = data.get('bb_position', 0.5)
        if bb_pos < 0.15:
            bullish_signals += 1
        elif bb_pos > 0.85:
            bearish_signals += 1
        total_signals += 1

        if total_signals == 0:
            return {'direction': 'neutral', 'confidence': 0, 'source': 'technical'}

        if bullish_signals > bearish_signals:
            direction = 'bullish'
            confidence = int((bullish_signals / total_signals) * 100)
        elif bearish_signals > bullish_signals:
            direction = 'bearish'
            confidence = int((bearish_signals / total_signals) * 100)
        else:
            direction = 'neutral'
            confidence = 25

        return {
            'direction': direction,
            'confidence': min(90, confidence),
            'source': 'technical'
        }

    def _score_news(self, sentiment: float) -> Dict:
        """Convert news sentiment (-1 to 1) into directional score"""
        if sentiment > 0.15:
            direction = 'bullish'
            confidence = int(min(80, sentiment * 100))
        elif sentiment < -0.15:
            direction = 'bearish'
            confidence = int(min(80, abs(sentiment) * 100))
        else:
            direction = 'neutral'
            confidence = 15

        return {
            'direction': direction,
            'confidence': confidence,
            'source': 'news_sentiment'
        }

    def _confidence_to_size(self, confidence: int) -> float:
        """Map ensemble confidence to position size percentage"""
        for threshold, size in SIZING_TIERS:
            if confidence >= threshold:
                return size
        return 0.0

    # ---- Accuracy Tracking & Calibration ----

    def update_accuracy(self, symbol: str, source: str, was_correct: bool):
        """Update source accuracy tracking for weight calibration"""
        if source in self._source_accuracy:
            stats = self._source_accuracy[source]
            stats['total'] += 1
            if was_correct:
                stats['correct'] += 1

            self._accuracy_check_count += 1

            # Auto-calibrate every 10 accuracy checks
            if self._accuracy_check_count % 10 == 0:
                self.calibrate_weights()

    def calibrate_weights(self):
        """Adjust source weights based on tracked accuracy. Persists to SQLite."""
        with self._lock:
            accuracies = {}
            for source, stats in self._source_accuracy.items():
                if stats['total'] >= 10:  # Minimum sample size
                    accuracies[source] = stats['correct'] / stats['total']

            if len(accuracies) < 2:
                return  # Not enough data to calibrate

            # Normalize accuracies to weights (sum to 1.0)
            total_accuracy = sum(accuracies.values())
            if total_accuracy > 0:
                for source, accuracy in accuracies.items():
                    # Blend: 70% data-driven + 30% original weight
                    data_weight = accuracy / total_accuracy
                    original = SOURCE_WEIGHTS.get(source, 0.1)
                    self.weights[source] = 0.7 * data_weight + 0.3 * original

                # Normalize all weights (including uncalibrated)
                total = sum(self.weights.values())
                for source in self.weights:
                    self.weights[source] = round(self.weights[source] / total, 3)

                logger.info(f"Ensemble weights calibrated: {self.weights}")
                self._persist_weights()

    def _persist_weights(self):
        """Save weights to SQLite so they survive restarts."""
        try:
            os.makedirs(os.path.dirname(WEIGHTS_DB_PATH), exist_ok=True)
            conn = sqlite3.connect(WEIGHTS_DB_PATH)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ensemble_weights (
                    id INTEGER PRIMARY KEY,
                    weights_json TEXT,
                    accuracy_json TEXT,
                    timestamp TEXT
                )
            ''')
            conn.execute(
                'INSERT INTO ensemble_weights (weights_json, accuracy_json, timestamp) VALUES (?, ?, ?)',
                (json.dumps(self.weights),
                 json.dumps({k: v for k, v in self._source_accuracy.items()}),
                 datetime.now(timezone.utc).isoformat())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to persist weights: {e}")

    def _load_persisted_weights(self) -> Optional[Dict]:
        """Load most recent calibrated weights from SQLite."""
        try:
            if os.path.exists(WEIGHTS_DB_PATH):
                conn = sqlite3.connect(WEIGHTS_DB_PATH)
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT weights_json, accuracy_json FROM ensemble_weights '
                    'ORDER BY id DESC LIMIT 1'
                )
                row = cursor.fetchone()
                conn.close()
                if row:
                    weights = json.loads(row[0])
                    accuracy = json.loads(row[1]) if row[1] else {}
                    # Restore accuracy tracking
                    if accuracy:
                        for source, stats in accuracy.items():
                            if hasattr(self, '_source_accuracy') and source in self._source_accuracy:
                                self._source_accuracy[source] = stats
                    logger.info(f"Loaded persisted ensemble weights: {weights}")
                    return weights
        except Exception as e:
            logger.debug(f"Failed to load persisted weights: {e}")
        return None

    def get_stats(self) -> Dict:
        """Get ensemble statistics"""
        accuracies = {}
        for source, stats in self._source_accuracy.items():
            total = stats['total']
            correct = stats['correct']
            accuracies[source] = {
                'total': total,
                'correct': correct,
                'accuracy': round(correct / max(total, 1) * 100, 1)
            }

        return {
            'weights': self.weights,
            'source_accuracy': accuracies,
            'cached_predictions': len(self._predictions),
            'calibration_count': self._accuracy_check_count,
            'active_sources': len([s for s in self.weights if self.weights[s] > 0]),
        }
