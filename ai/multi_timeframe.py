"""
Multi-Timeframe Prediction Engine
==================================
Generates predictions across 1h, 4h, and 24h timeframes.
Detects confluence (all timeframes agree) for high-conviction signals.
Tracks accuracy per timeframe for self-calibration.

Author: Trading Bot Arsenal
Created: February 2026 | V6 AI Upgrade
"""

import os
import sys
import json
import time
import logging
import sqlite3
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.llm_client import LLMClient, LLMResponse

logger = logging.getLogger('MultiTimeframe')


@dataclass
class TimeframePrediction:
    """Prediction for a single timeframe"""
    symbol: str
    timeframe: str  # '1h', '4h', '24h'
    direction: str  # bullish/bearish/neutral
    confidence: int  # 0-100
    key_signal: str
    price_at_prediction: float
    target_pct: float  # expected move %
    stop_pct: float    # stop loss %
    timestamp: str


@dataclass
class ConfluenceSignal:
    """High-conviction signal when multiple timeframes agree"""
    symbol: str
    direction: str  # bullish/bearish
    strength: str   # 'strong', 'moderate', 'weak'
    agreeing_timeframes: List[str]
    avg_confidence: float
    max_confidence: float
    weighted_score: float  # 0-100, timeframe-weighted
    predictions: List[TimeframePrediction]
    timestamp: str


# Timeframe weights (longer = more weight for trend, shorter = more weight for timing)
TIMEFRAME_WEIGHTS = {
    '1h': 0.2,   # Short-term momentum
    '4h': 0.45,  # Medium swing (most actionable)
    '24h': 0.35  # Trend direction
}

TIMEFRAME_CONFIGS = {
    '1h': {
        'yf_period': '2d',
        'yf_interval': '15m',
        'min_bars': 16,
        'rsi_period': 14,
        'ema_fast': 8,
        'ema_slow': 21,
        'description': 'next 1 hour'
    },
    '4h': {
        'yf_period': '5d',
        'yf_interval': '1h',
        'min_bars': 24,
        'rsi_period': 14,
        'ema_fast': 12,
        'ema_slow': 26,
        'description': 'next 4 hours'
    },
    '24h': {
        'yf_period': '30d',
        'yf_interval': '4h',
        'min_bars': 30,
        'rsi_period': 14,
        'ema_fast': 12,
        'ema_slow': 50,
        'description': 'next 24 hours'
    }
}

MTF_SYSTEM_PROMPT = """You are a quantitative analyst generating a directional prediction for a specific timeframe.

RULES:
1. Base prediction ONLY on the data provided. Never fabricate.
2. Different timeframes require different analysis:
   - 1h: Focus on momentum, order flow, recent candle patterns
   - 4h: Focus on trend structure, support/resistance, volume
   - 24h: Focus on macro trend, fundamentals, regime shifts
3. Be contrarian at extremes (RSI > 80 or < 20)
4. Cross-asset correlations matter (BTC leads alts)

OUTPUT (JSON only):
{
    "direction": "bullish" | "bearish" | "neutral",
    "confidence": 0-100,
    "key_signal": "The dominant signal for this timeframe",
    "target_pct": 0.5,
    "stop_pct": -1.0,
    "reasoning": "Under 30 words"
}

CONFIDENCE CALIBRATION:
- 75-100: Multiple strong signals aligned (trend+momentum+volume)
- 50-74: Primary signal clear, some conflicting data
- 25-49: Mixed signals, low conviction
- 0-24: No actionable signal"""


class MultiTimeframeEngine:
    """
    Generates predictions across multiple timeframes and detects confluence.

    Confluence detection:
    - STRONG: All 3 timeframes agree on direction with avg confidence > 60
    - MODERATE: 2/3 timeframes agree with avg confidence > 50
    - WEAK: Only 1 timeframe has signal, or conflicting signals
    """

    CRYPTO_SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'DOGE/USD', 'XRP/USD',
                      'AVAX/USD', 'LINK/USD', 'ADA/USD']

    def __init__(self, llm_client: Optional[LLMClient] = None, db_path: str = None):
        self.llm = llm_client or LLMClient()

        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "cache", "multi_timeframe.db")

        self.db_path = db_path
        self._init_db()

        # Latest predictions per symbol per timeframe
        self._predictions: Dict[str, Dict[str, TimeframePrediction]] = {}
        # Latest confluence signals
        self._confluence: Dict[str, ConfluenceSignal] = {}
        self._lock = threading.Lock()

        # Accuracy tracking per timeframe
        self._accuracy = {'1h': {'correct': 0, 'total': 0},
                         '4h': {'correct': 0, 'total': 0},
                         '24h': {'correct': 0, 'total': 0}}

        logger.info("MultiTimeframeEngine initialized")

    def _init_db(self):
        """Initialize prediction tracking database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS mtf_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                timeframe TEXT,
                direction TEXT,
                confidence INTEGER,
                key_signal TEXT,
                price_at_prediction REAL,
                target_pct REAL,
                stop_pct REAL,
                price_after REAL,
                actual_direction TEXT,
                correct INTEGER,
                checked INTEGER DEFAULT 0
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS confluence_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                strength TEXT,
                agreeing_timeframes TEXT,
                avg_confidence REAL,
                weighted_score REAL
            )
        ''')
        conn.commit()
        conn.close()

    def _fetch_timeframe_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Fetch OHLCV data for a specific timeframe and compute technicals"""
        config = TIMEFRAME_CONFIGS[timeframe]

        try:
            import yfinance as yf
            ticker = symbol.replace('/', '-')
            stock = yf.Ticker(ticker)
            hist = stock.history(period=config['yf_period'], interval=config['yf_interval'])

            if len(hist) < config['min_bars']:
                return None

            closes = hist['Close'].values
            highs = hist['High'].values
            lows = hist['Low'].values
            volumes = hist['Volume'].values
            current_price = float(closes[-1])

            # RSI
            rsi_period = config['rsi_period']
            deltas = np.diff(closes[-(rsi_period + 1):])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            rs = avg_gain / max(avg_loss, 1e-10)
            rsi = 100 - (100 / (1 + rs))

            # EMAs
            ema_fast = self._ema(closes, config['ema_fast'])
            ema_slow = self._ema(closes, config['ema_slow'])
            ema_signal = "bullish" if ema_fast > ema_slow else "bearish"

            # MACD
            macd = ema_fast - ema_slow
            macd_pct = (macd / current_price) * 100

            # Recent candle patterns
            last_3 = closes[-3:]
            recent_trend = "up" if last_3[-1] > last_3[0] else "down"
            recent_change = ((last_3[-1] - last_3[0]) / last_3[0]) * 100

            # Volume analysis
            avg_vol = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)
            vol_ratio = float(volumes[-1] / max(avg_vol, 1))

            # Bollinger position
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
            std_20 = np.std(closes[-20:]) if len(closes) >= 20 else np.std(closes)
            bb_upper = sma_20 + 2 * std_20
            bb_lower = sma_20 - 2 * std_20
            bb_range = bb_upper - bb_lower
            bb_pos = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5

            # ATR for target/stop calibration
            if len(hist) >= 15:
                tr_highs = highs[-14:]
                tr_lows = lows[-14:]
                tr_prev = closes[-15:-1]
                tr = np.maximum(tr_highs - tr_lows,
                               np.maximum(np.abs(tr_highs - tr_prev),
                                         np.abs(tr_lows - tr_prev)))
                atr = float(np.mean(tr))
                atr_pct = (atr / current_price) * 100
            else:
                atr = 0
                atr_pct = 0

            # Support/resistance
            support = float(np.percentile(lows[-20:], 10))
            resistance = float(np.percentile(highs[-20:], 90))

            # Higher timeframe trend (for context)
            if len(closes) >= 50:
                ht_trend = "up" if closes[-1] > np.mean(closes[-50:]) else "down"
            else:
                ht_trend = "unknown"

            return {
                'price': round(current_price, 2),
                'rsi': round(rsi, 1),
                'ema_fast': round(ema_fast, 2),
                'ema_slow': round(ema_slow, 2),
                'ema_signal': ema_signal,
                'macd_pct': round(macd_pct, 4),
                'recent_trend': recent_trend,
                'recent_change_pct': round(recent_change, 2),
                'volume_ratio': round(vol_ratio, 2),
                'bb_position': round(bb_pos, 2),
                'atr_pct': round(atr_pct, 3),
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'higher_tf_trend': ht_trend,
                'num_bars': len(closes)
            }

        except Exception as e:
            logger.debug(f"Failed to fetch {timeframe} data for {symbol}: {e}")
            return None

    def _ema(self, data, period: int) -> float:
        """Calculate EMA"""
        if len(data) < period:
            return float(np.mean(data))
        multiplier = 2 / (period + 1)
        ema = float(data[-period])
        for price in data[-period + 1:]:
            ema = (float(price) - ema) * multiplier + ema
        return ema

    def predict_symbol_timeframe(self, symbol: str, timeframe: str,
                                  technical_data: Dict = None) -> Optional[TimeframePrediction]:
        """Generate prediction for a specific symbol and timeframe"""
        if technical_data is None:
            technical_data = self._fetch_timeframe_data(symbol, timeframe)

        if not technical_data:
            return None

        config = TIMEFRAME_CONFIGS[timeframe]

        prompt = f"""PREDICTION: {symbol} | Timeframe: {timeframe} ({config['description']})

TECHNICAL DATA ({timeframe} bars):
  Price: ${technical_data['price']}
  RSI({config['rsi_period']}): {technical_data['rsi']}
  EMA({config['ema_fast']}): ${technical_data['ema_fast']}
  EMA({config['ema_slow']}): ${technical_data['ema_slow']}
  EMA Signal: {technical_data['ema_signal']}
  MACD%: {technical_data['macd_pct']}%
  Recent Trend: {technical_data['recent_trend']} ({technical_data['recent_change_pct']}%)
  Volume vs Avg: {technical_data['volume_ratio']}x
  Bollinger Position: {technical_data['bb_position']} (0=lower band, 1=upper band)
  ATR%: {technical_data['atr_pct']}%
  Support: ${technical_data['support']} | Resistance: ${technical_data['resistance']}
  Longer-term trend: {technical_data['higher_tf_trend']}
  Bars analyzed: {technical_data['num_bars']}

Predict direction for {config['description']}. JSON only:"""

        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            response = loop.run_until_complete(
                self.llm.query(
                    prompt=prompt,
                    system_prompt=MTF_SYSTEM_PROMPT,
                    max_tokens=250,
                    temperature=0.15,
                    use_cache=False
                )
            )

            # Parse response
            content = response.content.strip()
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start < 0 or json_end <= json_start:
                return None

            data = json.loads(content[json_start:json_end])

            prediction = TimeframePrediction(
                symbol=symbol,
                timeframe=timeframe,
                direction=data.get('direction', 'neutral'),
                confidence=min(100, max(0, int(data.get('confidence', 50)))),
                key_signal=data.get('key_signal', 'unknown')[:100],
                price_at_prediction=technical_data['price'],
                target_pct=float(data.get('target_pct', 0)),
                stop_pct=float(data.get('stop_pct', 0)),
                timestamp=datetime.now(timezone.utc).isoformat()
            )

            return prediction

        except Exception as e:
            logger.debug(f"MTF prediction failed for {symbol} {timeframe}: {e}")
            return None

    def _analyze_symbol(self, symbol: str) -> Tuple[str, Dict[str, 'TimeframePrediction']]:
        """Analyze a single symbol across all timeframes (runs in thread pool)"""
        symbol_preds = {}
        for tf in ['1h', '4h', '24h']:
            try:
                pred = self.predict_symbol_timeframe(symbol, tf)
                if pred:
                    symbol_preds[tf] = pred
                    logger.debug(f"  {symbol} {tf}: {pred.direction} ({pred.confidence}%)")
            except Exception as e:
                logger.debug(f"  {symbol} {tf}: failed - {e}")
        return symbol, symbol_preds

    def run_multi_timeframe_analysis(self, symbols: List[str] = None) -> Dict[str, ConfluenceSignal]:
        """
        Run predictions across all timeframes for all symbols.
        Uses ThreadPoolExecutor for symbol-level parallelism (8 symbols in parallel).
        Returns confluence signals for each symbol.
        """
        if symbols is None:
            symbols = self.CRYPTO_SYMBOLS

        confluence_signals = {}
        all_predictions = {}

        start_time = time.time()
        logger.info(f"MTF Analysis: scanning {len(symbols)} symbols across 3 timeframes (parallel)...")

        # Parallel symbol analysis - 8 symbols concurrently
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="mtf") as executor:
            futures = {executor.submit(self._analyze_symbol, symbol): symbol
                      for symbol in symbols}

            for future in as_completed(futures, timeout=120):
                try:
                    symbol, symbol_preds = future.result()
                    if not symbol_preds:
                        continue

                    all_predictions[symbol] = symbol_preds

                    # Detect confluence
                    confluence = self._detect_confluence(symbol, symbol_preds)
                    if confluence:
                        confluence_signals[symbol] = confluence
                        self._store_confluence(confluence)

                    # Store predictions
                    for pred in symbol_preds.values():
                        self._store_prediction(pred)
                except Exception as e:
                    sym = futures[future]
                    logger.debug(f"  {sym}: parallel analysis failed - {e}")

        elapsed = time.time() - start_time

        # Update shared state
        with self._lock:
            self._predictions = all_predictions
            self._confluence = confluence_signals

        # Check old prediction accuracy
        self._check_accuracy()

        # Log summary
        strong = [s for s, c in confluence_signals.items() if c.strength == 'strong']
        moderate = [s for s, c in confluence_signals.items() if c.strength == 'moderate']
        logger.info(f"MTF Analysis complete in {elapsed:.1f}s: {len(strong)} strong, {len(moderate)} moderate confluence signals")

        for symbol, conf in confluence_signals.items():
            if conf.strength in ('strong', 'moderate'):
                logger.info(f"  {symbol}: {conf.direction.upper()} ({conf.strength}) "
                           f"score={conf.weighted_score:.0f} [{','.join(conf.agreeing_timeframes)}]")

        return confluence_signals

    def _detect_confluence(self, symbol: str,
                           predictions: Dict[str, TimeframePrediction]) -> Optional[ConfluenceSignal]:
        """Detect when multiple timeframes agree on direction"""
        if not predictions:
            return None

        # Count directional votes
        bullish_tfs = []
        bearish_tfs = []
        bullish_conf = []
        bearish_conf = []

        for tf, pred in predictions.items():
            if pred.direction == 'bullish' and pred.confidence >= 30:
                bullish_tfs.append(tf)
                bullish_conf.append(pred.confidence)
            elif pred.direction == 'bearish' and pred.confidence >= 30:
                bearish_tfs.append(tf)
                bearish_conf.append(pred.confidence)

        # Determine dominant direction
        if len(bullish_tfs) >= len(bearish_tfs) and bullish_tfs:
            direction = 'bullish'
            agreeing = bullish_tfs
            confidences = bullish_conf
        elif bearish_tfs:
            direction = 'bearish'
            agreeing = bearish_tfs
            confidences = bearish_conf
        else:
            return None

        # Calculate weighted score
        weighted_score = 0
        for tf in agreeing:
            pred = predictions[tf]
            weighted_score += pred.confidence * TIMEFRAME_WEIGHTS.get(tf, 0.33)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        max_conf = max(confidences) if confidences else 0

        # Determine strength
        if len(agreeing) >= 3 and avg_conf >= 55:
            strength = 'strong'
        elif len(agreeing) >= 2 and avg_conf >= 45:
            strength = 'moderate'
        else:
            strength = 'weak'

        all_preds = list(predictions.values())

        return ConfluenceSignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            agreeing_timeframes=agreeing,
            avg_confidence=round(avg_conf, 1),
            max_confidence=max_conf,
            weighted_score=round(weighted_score, 1),
            predictions=all_preds,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def get_confluence(self, symbol: str) -> Optional[Dict]:
        """Get latest confluence signal for a symbol (no LLM call)"""
        with self._lock:
            conf = self._confluence.get(symbol)
            if conf:
                return {
                    'direction': conf.direction,
                    'strength': conf.strength,
                    'agreeing_timeframes': conf.agreeing_timeframes,
                    'avg_confidence': conf.avg_confidence,
                    'weighted_score': conf.weighted_score,
                    'timestamp': conf.timestamp
                }
        return None

    def get_all_confluence(self) -> Dict[str, Dict]:
        """Get all current confluence signals"""
        with self._lock:
            result = {}
            for symbol, conf in self._confluence.items():
                result[symbol] = {
                    'direction': conf.direction,
                    'strength': conf.strength,
                    'agreeing_timeframes': conf.agreeing_timeframes,
                    'avg_confidence': conf.avg_confidence,
                    'weighted_score': conf.weighted_score
                }
            return result

    def get_prediction(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get latest prediction for a symbol+timeframe"""
        with self._lock:
            symbol_preds = self._predictions.get(symbol, {})
            pred = symbol_preds.get(timeframe)
            if pred:
                return asdict(pred)
        return None

    def _store_prediction(self, pred: TimeframePrediction):
        """Store prediction in DB"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO mtf_predictions
                (timestamp, symbol, timeframe, direction, confidence, key_signal,
                 price_at_prediction, target_pct, stop_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (pred.timestamp, pred.symbol, pred.timeframe, pred.direction,
                  pred.confidence, pred.key_signal, pred.price_at_prediction,
                  pred.target_pct, pred.stop_pct))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to store prediction: {e}")

    def _store_confluence(self, conf: ConfluenceSignal):
        """Store confluence signal in DB"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO confluence_signals
                (timestamp, symbol, direction, strength, agreeing_timeframes,
                 avg_confidence, weighted_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (conf.timestamp, conf.symbol, conf.direction, conf.strength,
                  json.dumps(conf.agreeing_timeframes), conf.avg_confidence,
                  conf.weighted_score))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to store confluence: {e}")

    def _check_accuracy(self):
        """Check accuracy of predictions that have aged enough"""
        check_windows = {'1h': 1, '4h': 4, '24h': 24}

        try:
            import yfinance as yf
            conn = sqlite3.connect(self.db_path)

            for tf, hours in check_windows.items():
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, symbol, direction, price_at_prediction
                    FROM mtf_predictions
                    WHERE checked = 0 AND timeframe = ? AND timestamp < ?
                ''', (tf, cutoff))

                for row in cursor.fetchall():
                    pred_id, symbol, direction, price_at = row
                    try:
                        ticker = symbol.replace('/', '-')
                        stock = yf.Ticker(ticker)
                        hist = stock.history(period="1d", interval="1h")
                        if len(hist) > 0:
                            current_price = float(hist['Close'].iloc[-1])
                            change_pct = ((current_price - price_at) / price_at) * 100

                            # V7: Raised threshold to 1.0% for crypto
                            if change_pct > 1.0:
                                actual = 'bullish'
                            elif change_pct < -1.0:
                                actual = 'bearish'
                            else:
                                actual = 'neutral'

                            correct = (direction == actual) or \
                                     (direction == 'neutral' and abs(change_pct) < 1.5)

                            conn.execute('''
                                UPDATE mtf_predictions
                                SET price_after = ?, actual_direction = ?,
                                    correct = ?, checked = 1
                                WHERE id = ?
                            ''', (current_price, actual, 1 if correct else 0, pred_id))

                            stats = self._accuracy[tf]
                            stats['total'] += 1
                            if correct:
                                stats['correct'] += 1

                    except Exception:
                        pass

            conn.commit()
            conn.close()

        except Exception as e:
            logger.debug(f"Accuracy check failed: {e}")

    def get_accuracy_stats(self) -> Dict:
        """Get accuracy by timeframe"""
        stats = {}
        for tf, data in self._accuracy.items():
            total = data['total']
            correct = data['correct']
            stats[tf] = {
                'total': total,
                'correct': correct,
                'accuracy': round(correct / max(total, 1) * 100, 1)
            }

        # Also check DB for historical data
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timeframe,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct
                FROM mtf_predictions WHERE checked = 1
                GROUP BY timeframe
            ''')
            for row in cursor.fetchall():
                tf, total, correct = row
                if tf in stats:
                    stats[tf]['db_total'] = total
                    stats[tf]['db_correct'] = correct
                    stats[tf]['db_accuracy'] = round(correct / max(total, 1) * 100, 1)
            conn.close()
        except Exception:
            pass

        return stats
