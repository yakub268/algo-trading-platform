"""
AI Market Analyst - The Brain
==============================
Periodic market analysis engine that synthesizes technical data, news sentiment,
and market conditions into actionable predictions using LLM.

Runs every 30 minutes via orchestrator schedule.
Generates predictions, tracks accuracy, and feeds context to all bots.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import json
import time
import logging
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.llm_client import LLMClient, LLMResponse
from ai.prompts.analyst_prompt import (
    ANALYST_SYSTEM_PROMPT, PREDICTION_SYSTEM_PROMPT,
    TRADE_REASONING_PROMPT,
    build_analysis_prompt, build_prediction_prompt, build_trade_reasoning_prompt
)

logger = logging.getLogger('AIMarketAnalyst')


@dataclass
class Prediction:
    """Single asset prediction"""
    symbol: str
    direction: str  # bullish/bearish/neutral
    confidence: int  # 0-100
    timeframe: str
    key_factors: List[str]
    price_at_prediction: float
    timestamp: str
    actual_direction: Optional[str] = None
    correct: Optional[bool] = None


@dataclass
class MarketAnalysis:
    """Full market analysis result"""
    timestamp: str
    market_regime: str
    risk_level: str
    predictions: List[Prediction]
    portfolio_recommendations: Dict
    key_risks: List[str]
    conviction_trade: Dict
    summary: str
    llm_latency_ms: float
    raw_response: str


class AIMarketAnalyst:
    """
    AI-powered market analysis engine.

    Features:
    - Multi-source data aggregation (prices, technicals, news)
    - LLM-powered analysis and directional predictions
    - Prediction accuracy tracking with SQLite
    - Dynamic bot parameter recommendations
    - Portfolio-level risk assessment
    - Trade reasoning generation for Telegram
    """

    CRYPTO_SYMBOLS = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'DOGE/USD', 'XRP/USD',
                      'AVAX/USD', 'LINK/USD', 'ADA/USD']

    def __init__(self, llm_client: Optional[LLMClient] = None, db_path: str = None,
                 data_hub=None):
        self.llm = llm_client or LLMClient()
        self.data_hub = data_hub  # V7: Central data aggregator

        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "cache", "market_analyst.db")

        self.db_path = db_path
        self._init_db()

        # Latest analysis (shared state for other components)
        self.latest_analysis: Optional[MarketAnalysis] = None
        self.latest_predictions: Dict[str, Prediction] = {}  # symbol -> prediction
        self._lock = threading.Lock()

        # News aggregator (lazy init)
        self._news_agg = None

        # Stats
        self.total_analyses = 0
        self.total_predictions = 0
        self.correct_predictions = 0

        logger.info("AIMarketAnalyst initialized")

    def _init_db(self):
        """Initialize prediction tracking database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                confidence INTEGER,
                price_at_prediction REAL,
                price_after REAL,
                actual_direction TEXT,
                correct INTEGER,
                timeframe TEXT,
                key_factors TEXT,
                checked INTEGER DEFAULT 0
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                market_regime TEXT,
                risk_level TEXT,
                summary TEXT,
                predictions_json TEXT,
                recommendations_json TEXT,
                latency_ms REAL
            )
        ''')
        conn.commit()
        conn.close()

    def _get_news_aggregator(self):
        """Lazy-init news aggregator"""
        if self._news_agg is None:
            try:
                from news_feeds.aggregator import NewsAggregator
                self._news_agg = NewsAggregator(cache_ttl=600)
                logger.info("News aggregator initialized for market analyst")
            except Exception as e:
                logger.warning(f"News aggregator unavailable: {e}")
        return self._news_agg

    def _fetch_crypto_data(self) -> Dict[str, Dict]:
        """Fetch real-time crypto prices and compute technicals"""
        data = {}
        try:
            import yfinance as yf
            import numpy as np

            for symbol in self.CRYPTO_SYMBOLS:
                ticker = symbol.replace('/', '-')
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="5d", interval="1h")

                    if len(hist) < 20:
                        continue

                    closes = hist['Close'].values
                    volumes = hist['Volume'].values
                    current_price = float(closes[-1])

                    # RSI(14)
                    deltas = np.diff(closes[-15:])
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    avg_gain = np.mean(gains) if len(gains) > 0 else 0
                    avg_loss = np.mean(losses) if len(losses) > 0 else 0
                    rs = avg_gain / max(avg_loss, 1e-10)
                    rsi = 100 - (100 / (1 + rs))

                    # EMAs
                    ema_12 = self._ema(closes, 12)
                    ema_26 = self._ema(closes, 26)
                    ema_signal = "bullish" if ema_12 > ema_26 else "bearish"

                    # MACD
                    macd = ema_12 - ema_26

                    # 24h change
                    if len(closes) >= 24:
                        change_24h = ((current_price - closes[-24]) / closes[-24]) * 100
                    else:
                        change_24h = ((current_price - closes[0]) / closes[0]) * 100

                    # Volume ratio
                    avg_vol = np.mean(volumes[-24:]) if len(volumes) >= 24 else np.mean(volumes)
                    vol_ratio = float(volumes[-1] / max(avg_vol, 1)) if avg_vol > 0 else 1.0

                    # Bollinger Bands position
                    sma_20 = np.mean(closes[-20:])
                    std_20 = np.std(closes[-20:])
                    upper_bb = sma_20 + 2 * std_20
                    lower_bb = sma_20 - 2 * std_20
                    bb_range = upper_bb - lower_bb
                    bb_position = (current_price - lower_bb) / bb_range if bb_range > 0 else 0.5

                    # Support/Resistance (simple)
                    recent_lows = hist['Low'].values[-48:]
                    recent_highs = hist['High'].values[-48:]
                    support = float(np.percentile(recent_lows, 10))
                    resistance = float(np.percentile(recent_highs, 90))

                    # ATR
                    if len(hist) >= 15:
                        highs = hist['High'].values[-15:]
                        lows = hist['Low'].values[-15:]
                        prev_closes = closes[-16:-1]
                        tr = np.maximum(highs - lows,
                                       np.maximum(np.abs(highs - prev_closes),
                                                 np.abs(lows - prev_closes)))
                        atr = float(np.mean(tr))
                    else:
                        atr = 0

                    data[symbol] = {
                        'price': round(current_price, 2),
                        'change_24h': round(change_24h, 2),
                        'rsi': round(rsi, 1),
                        'ema_12': round(ema_12, 2),
                        'ema_26': round(ema_26, 2),
                        'ema_signal': ema_signal,
                        'macd': round(macd, 2),
                        'volume_ratio': round(vol_ratio, 2),
                        'bb_position': round(bb_position, 2),
                        'support': round(support, 2),
                        'resistance': round(resistance, 2),
                        'atr': round(atr, 2),
                    }

                except Exception as e:
                    logger.debug(f"Failed to fetch {symbol}: {e}")

        except ImportError:
            logger.warning("yfinance not available for market data")

        return data

    def _ema(self, data, period: int) -> float:
        """Calculate EMA"""
        import numpy as np
        if len(data) < period:
            return float(np.mean(data))
        multiplier = 2 / (period + 1)
        ema = float(data[-period])
        for price in data[-period + 1:]:
            ema = (float(price) - ema) * multiplier + ema
        return ema

    def _fetch_news_sentiment(self) -> str:
        """Fetch recent news and generate sentiment summary"""
        agg = self._get_news_aggregator()
        if not agg:
            return "No news data available."

        try:
            # Fetch financial + crypto news
            articles = agg.fetch_financial_news(
                symbols=['bitcoin', 'ethereum', 'crypto', 'fed', 'inflation'],
                include_crypto=True,
                limit=15
            )

            if not articles:
                return "No recent news articles found."

            summary_parts = []
            for article in articles[:10]:
                sentiment_label = "positive" if article.sentiment > 0.2 else "negative" if article.sentiment < -0.2 else "neutral"
                summary_parts.append(
                    f"- [{sentiment_label}] {article.title} (sentiment: {article.sentiment:.2f})"
                )

            # Overall sentiment
            sentiments = [a.sentiment for a in articles if a.sentiment != 0]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            overall = "bullish" if avg_sentiment > 0.15 else "bearish" if avg_sentiment < -0.15 else "neutral"

            return f"Overall News Sentiment: {overall} ({avg_sentiment:.2f})\n" + "\n".join(summary_parts)

        except Exception as e:
            logger.warning(f"News fetch failed: {e}")
            return "News feed temporarily unavailable."

    def _fetch_market_context(self) -> Dict:
        """Fetch VIX, SPY trend, and other market context"""
        context = {
            'vix': 20.0,
            'spy_trend': 'unknown',
            'day_of_week': datetime.now().strftime('%A'),
            'time': datetime.now().strftime('%H:%M'),
        }

        try:
            import yfinance as yf

            # VIX
            try:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="2d")
                if len(vix_hist) > 0:
                    context['vix'] = round(float(vix_hist['Close'].iloc[-1]), 1)
            except Exception:
                pass

            # SPY trend
            try:
                spy = yf.Ticker("SPY")
                spy_hist = spy.history(period="5d")
                if len(spy_hist) >= 2:
                    spy_change = (spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[-2]) / spy_hist['Close'].iloc[-2] * 100
                    if spy_change > 0.3:
                        context['spy_trend'] = 'up'
                    elif spy_change < -0.3:
                        context['spy_trend'] = 'down'
                    else:
                        context['spy_trend'] = 'sideways'
            except Exception:
                pass

        except ImportError:
            pass

        return context

    def run_analysis(self, portfolio_state: Dict = None) -> Optional[MarketAnalysis]:
        """
        Run comprehensive market analysis. Main entry point.
        Called by orchestrator every 30 minutes.
        """
        self.total_analyses += 1
        logger.info("=" * 60)
        logger.info("AI MARKET ANALYST - Starting analysis...")

        # 1. Gather data
        crypto_data = self._fetch_crypto_data()
        if not crypto_data:
            logger.warning("No crypto data available, skipping analysis")
            return None

        news_summary = self._fetch_news_sentiment()
        market_ctx = self._fetch_market_context()

        # Merge portfolio state with market context
        full_portfolio = {**(portfolio_state or {}), **market_ctx}

        # Check previous prediction accuracy
        recent_preds = self._get_recent_predictions(hours=24)
        self._check_prediction_accuracy(crypto_data)

        # V7: Get data hub context for enriched prompts
        hub_context = None
        if self.data_hub:
            try:
                hub_context = self.data_hub.get_prompt_context()
                hub_data = self.data_hub.get_data()
                # Inject BTC dominance into portfolio state for prompt
                if hub_data.get('btc_dominance'):
                    full_portfolio['btc_dominance'] = hub_data['btc_dominance']
            except Exception as e:
                logger.debug(f"Data hub context failed: {e}")

        # 2. Build prompt
        prompt = build_analysis_prompt(crypto_data, news_summary, full_portfolio, recent_preds,
                                       data_hub_context=hub_context)

        # 3. Query LLM
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
                    system_prompt=ANALYST_SYSTEM_PROMPT,
                    max_tokens=1000,
                    temperature=0.2,
                    use_cache=False  # Always fresh analysis
                )
            )

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return None

        # 4. Parse response
        analysis = self._parse_analysis(response, crypto_data)
        if not analysis:
            logger.error("Failed to parse analysis response")
            return None

        # 5. Store
        with self._lock:
            self.latest_analysis = analysis
            for pred in analysis.predictions:
                self.latest_predictions[pred.symbol] = pred

        self._store_analysis(analysis)
        self._store_predictions(analysis.predictions, crypto_data)

        # 6. Log results
        logger.info(f"Market Regime: {analysis.market_regime} | Risk: {analysis.risk_level}")
        logger.info(f"Summary: {analysis.summary}")
        for pred in analysis.predictions:
            logger.info(f"  {pred.symbol}: {pred.direction.upper()} ({pred.confidence}% conf) - {', '.join(pred.key_factors[:2])}")

        if analysis.conviction_trade.get('direction', 'none') != 'none':
            ct = analysis.conviction_trade
            logger.info(f"CONVICTION TRADE: {ct.get('direction', '').upper()} {ct.get('symbol', '')} - {ct.get('reasoning', '')}")

        logger.info(f"Analysis complete ({response.latency_ms:.0f}ms, provider: {response.provider.value})")
        logger.info("=" * 60)

        return analysis

    def predict_symbol(self, symbol: str, technical_data: Dict = None) -> Optional[Dict]:
        """Generate prediction for a specific symbol. Used by bots for signal confluence."""
        if technical_data is None:
            # Fetch fresh data
            all_data = self._fetch_crypto_data()
            technical_data = all_data.get(symbol, {})

        if not technical_data:
            return None

        # Get news sentiment
        news_sentiment = 0.0
        try:
            agg = self._get_news_aggregator()
            if agg:
                sentiment_data = agg.fetch_market_sentiment(
                    [symbol.split('/')[0].lower()], limit=10
                )
                news_sentiment = sentiment_data.get('overall_sentiment', 0.0)
        except Exception:
            pass

        # Get market regime from latest analysis
        market_regime = 'unknown'
        if self.latest_analysis:
            market_regime = self.latest_analysis.market_regime

        prompt = build_prediction_prompt(symbol, technical_data, news_sentiment, market_regime)

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
                    system_prompt=PREDICTION_SYSTEM_PROMPT,
                    max_tokens=300,
                    temperature=0.15
                )
            )

            # Parse
            content = response.content.strip()
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(content[json_start:json_end])
                result['latency_ms'] = response.latency_ms
                result['provider'] = response.provider.value
                return result

        except Exception as e:
            logger.debug(f"Symbol prediction failed for {symbol}: {e}")

        return None

    def generate_trade_reasoning(self, signal: Dict) -> str:
        """Generate AI reasoning for a trade signal. Used for Telegram alerts."""
        market_context = {}
        if self.latest_analysis:
            market_context['market_regime'] = self.latest_analysis.market_regime
            market_context['risk_level'] = self.latest_analysis.risk_level

            # Find prediction for this symbol
            symbol = signal.get('symbol', '')
            pred = self.latest_predictions.get(symbol)
            if pred:
                market_context['prediction_direction'] = pred.direction
                market_context['prediction_confidence'] = pred.confidence

        prompt = build_trade_reasoning_prompt(signal, market_context)

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
                    system_prompt=TRADE_REASONING_PROMPT,
                    max_tokens=100,
                    temperature=0.2
                )
            )

            reasoning = response.content.strip()
            # Remove any JSON wrapping
            if reasoning.startswith('{'):
                try:
                    data = json.loads(reasoning)
                    reasoning = data.get('reasoning', data.get('explanation', reasoning))
                except json.JSONDecodeError:
                    pass

            return reasoning[:200]  # Cap at 200 chars

        except Exception as e:
            logger.debug(f"Trade reasoning generation failed: {e}")
            return f"AI analysis unavailable: {str(e)[:50]}"

    def get_prediction_for_symbol(self, symbol: str) -> Optional[Dict]:
        """Get latest prediction for a symbol (from cache, no LLM call)."""
        with self._lock:
            pred = self.latest_predictions.get(symbol)
            if pred:
                return {
                    'direction': pred.direction,
                    'confidence': pred.confidence,
                    'key_factors': pred.key_factors,
                    'timestamp': pred.timestamp
                }
        return None

    def get_market_regime(self) -> str:
        """Get current market regime assessment."""
        with self._lock:
            if self.latest_analysis:
                return self.latest_analysis.market_regime
        return 'unknown'

    def get_risk_level(self) -> str:
        """Get current risk level."""
        with self._lock:
            if self.latest_analysis:
                return self.latest_analysis.risk_level
        return 'medium'

    def _parse_analysis(self, response: LLMResponse, crypto_data: Dict) -> Optional[MarketAnalysis]:
        """Parse LLM response into MarketAnalysis"""
        content = response.content.strip()

        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start < 0 or json_end <= json_start:
                raise ValueError("No JSON found in response")

            data = json.loads(content[json_start:json_end])

            # Parse predictions
            predictions = []
            for pred_data in data.get('predictions', []):
                symbol = pred_data.get('symbol', '')
                price = crypto_data.get(symbol, {}).get('price', 0)

                predictions.append(Prediction(
                    symbol=symbol,
                    direction=pred_data.get('direction', 'neutral'),
                    confidence=min(100, max(0, int(pred_data.get('confidence', 50)))),
                    timeframe=pred_data.get('timeframe', '4h'),
                    key_factors=pred_data.get('key_factors', [])[:4],
                    price_at_prediction=price,
                    timestamp=datetime.now(timezone.utc).isoformat()
                ))

            self.total_predictions += len(predictions)

            return MarketAnalysis(
                timestamp=datetime.now(timezone.utc).isoformat(),
                market_regime=data.get('market_regime', 'unknown'),
                risk_level=data.get('risk_level', 'medium'),
                predictions=predictions,
                portfolio_recommendations=data.get('portfolio_recommendations', {}),
                key_risks=data.get('key_risks', [])[:5],
                conviction_trade=data.get('conviction_trade', {}),
                summary=data.get('summary', 'Analysis complete.')[:300],
                llm_latency_ms=response.latency_ms,
                raw_response=content[:2000]
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse analysis JSON: {e}")
            # Fallback: create minimal analysis from text
            return MarketAnalysis(
                timestamp=datetime.now(timezone.utc).isoformat(),
                market_regime='unknown',
                risk_level='medium',
                predictions=[],
                portfolio_recommendations={},
                key_risks=[],
                conviction_trade={},
                summary=content[:200],
                llm_latency_ms=response.latency_ms,
                raw_response=content[:2000]
            )

    def _store_analysis(self, analysis: MarketAnalysis):
        """Store analysis in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO analyses (timestamp, market_regime, risk_level, summary,
                    predictions_json, recommendations_json, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.timestamp,
                analysis.market_regime,
                analysis.risk_level,
                analysis.summary,
                json.dumps([asdict(p) for p in analysis.predictions]),
                json.dumps(analysis.portfolio_recommendations),
                analysis.llm_latency_ms
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to store analysis: {e}")

    def _store_predictions(self, predictions: List[Prediction], crypto_data: Dict):
        """Store predictions for accuracy tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            for pred in predictions:
                conn.execute('''
                    INSERT INTO predictions (timestamp, symbol, direction, confidence,
                        price_at_prediction, timeframe, key_factors)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pred.timestamp,
                    pred.symbol,
                    pred.direction,
                    pred.confidence,
                    pred.price_at_prediction,
                    pred.timeframe,
                    json.dumps(pred.key_factors)
                ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to store predictions: {e}")

    def _check_prediction_accuracy(self, current_data: Dict):
        """Check accuracy of predictions made 4+ hours ago"""
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()

            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, symbol, direction, price_at_prediction
                FROM predictions
                WHERE checked = 0 AND timestamp < ?
            ''', (cutoff,))

            rows = cursor.fetchall()
            for row in rows:
                pred_id, symbol, direction, price_at = row
                current_price = current_data.get(symbol, {}).get('price')

                if current_price and price_at > 0:
                    actual_change = (current_price - price_at) / price_at * 100

                    # V7: Raised threshold to 1.0% for crypto (was 0.3%)
                    if actual_change > 1.0:
                        actual_dir = 'bullish'
                    elif actual_change < -1.0:
                        actual_dir = 'bearish'
                    else:
                        actual_dir = 'neutral'

                    correct = (direction == actual_dir) or (direction == 'neutral' and abs(actual_change) < 1.5)

                    conn.execute('''
                        UPDATE predictions
                        SET price_after = ?, actual_direction = ?, correct = ?, checked = 1
                        WHERE id = ?
                    ''', (current_price, actual_dir, 1 if correct else 0, pred_id))

                    if correct:
                        self.correct_predictions += 1

                    logger.debug(f"Prediction check: {symbol} predicted {direction}, actual {actual_dir} "
                                f"({actual_change:+.1f}%) - {'CORRECT' if correct else 'WRONG'}")

            conn.commit()
            conn.close()

        except Exception as e:
            logger.debug(f"Prediction accuracy check failed: {e}")

    def _get_recent_predictions(self, hours: int = 24) -> List[Dict]:
        """Get recent predictions with accuracy data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

            cursor = conn.cursor()
            cursor.execute('''
                SELECT symbol, direction, confidence, actual_direction, correct
                FROM predictions
                WHERE timestamp > ? AND checked = 1
                ORDER BY timestamp DESC
                LIMIT 20
            ''', (cutoff,))

            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    'symbol': r[0],
                    'direction': r[1],
                    'confidence': r[2],
                    'actual': r[3],
                    'correct': bool(r[4]) if r[4] is not None else None
                }
                for r in rows
            ]

        except Exception as e:
            logger.debug(f"Failed to get recent predictions: {e}")
            return []

    def get_accuracy_stats(self) -> Dict:
        """Get prediction accuracy statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM predictions WHERE checked = 1')
            total_checked = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM predictions WHERE checked = 1 AND correct = 1')
            total_correct = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM predictions WHERE checked = 0')
            pending = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM analyses')
            total_analyses = cursor.fetchone()[0]

            # Accuracy by symbol
            cursor.execute('''
                SELECT symbol,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct
                FROM predictions WHERE checked = 1
                GROUP BY symbol
            ''')
            by_symbol = {r[0]: {'total': r[1], 'correct': r[2], 'accuracy': round(r[2]/max(r[1],1)*100, 1)}
                        for r in cursor.fetchall()}

            conn.close()

            accuracy = round(total_correct / max(total_checked, 1) * 100, 1)

            return {
                'total_analyses': total_analyses,
                'total_predictions': total_checked + pending,
                'checked_predictions': total_checked,
                'correct_predictions': total_correct,
                'pending_predictions': pending,
                'accuracy_pct': accuracy,
                'by_symbol': by_symbol,
                'latest_regime': self.latest_analysis.market_regime if self.latest_analysis else 'unknown',
                'latest_risk': self.latest_analysis.risk_level if self.latest_analysis else 'unknown'
            }

        except Exception as e:
            logger.debug(f"Failed to get accuracy stats: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s %(name)s %(levelname)s: %(message)s')

    analyst = AIMarketAnalyst()

    portfolio = {
        'total_capital': 450,
        'current_pnl': 0,
        'open_positions': 0,
        'trades_today': 0,
        'win_rate': 0,
        'active_bots': 9
    }

    analysis = analyst.run_analysis(portfolio)
    if analysis:
        print(f"\nMarket Regime: {analysis.market_regime}")
        print(f"Risk Level: {analysis.risk_level}")
        print(f"Summary: {analysis.summary}")
        print(f"\nPredictions:")
        for p in analysis.predictions:
            print(f"  {p.symbol}: {p.direction} ({p.confidence}%)")
        print(f"\nAccuracy: {analyst.get_accuracy_stats()}")
