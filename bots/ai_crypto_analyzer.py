"""
AI Crypto Analyzer Bot
======================
Uses AI to analyze crypto assets before buy/sell decisions.

Features:
- Real-time news sentiment analysis
- Technical indicator analysis
- On-chain metrics (when available)
- AI-generated buy/sell/hold recommendations
- Confidence scoring

Integrates with DeepSeek API for AI analysis.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from decimal import Decimal

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AICryptoAnalyzer')


@dataclass
class CryptoAnalysis:
    """AI analysis result for a cryptocurrency"""
    symbol: str
    current_price: float
    price_change_24h: float
    price_change_7d: float
    recommendation: str  # BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
    confidence: float  # 0-1
    reasoning: str
    key_factors: List[str]
    entry_price: Optional[float]  # Suggested entry if BUY
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_level: str  # LOW, MEDIUM, HIGH
    news_sentiment: str  # BULLISH, BEARISH, NEUTRAL
    technical_signal: str  # BULLISH, BEARISH, NEUTRAL
    timestamp: datetime


@dataclass
class NewsItem:
    """News article about a crypto"""
    title: str
    summary: str
    source: str
    sentiment: str
    published: datetime


class AICryptoAnalyzer:
    """
    AI-powered cryptocurrency analyzer.

    Analyzes cryptos using:
    1. Price action and technicals
    2. News sentiment
    3. Market conditions
    4. AI reasoning
    """

    # Cryptos to analyze
    WATCHLIST = {
        'XRP': {'coingecko_id': 'ripple', 'name': 'XRP'},
        'HBAR': {'coingecko_id': 'hedera-hashgraph', 'name': 'Hedera'},
        'XLM': {'coingecko_id': 'stellar', 'name': 'Stellar'},
        'BTC': {'coingecko_id': 'bitcoin', 'name': 'Bitcoin'},
        'ETH': {'coingecko_id': 'ethereum', 'name': 'Ethereum'},
        'SOL': {'coingecko_id': 'solana', 'name': 'Solana'},
        'ADA': {'coingecko_id': 'cardano', 'name': 'Cardano'},
        'DOGE': {'coingecko_id': 'dogecoin', 'name': 'Dogecoin'},
    }

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.analyses: Dict[str, CryptoAnalysis] = {}

        # Get DeepSeek API key
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            logger.warning("DEEPSEEK_API_KEY not found - AI analysis will use fallback")

        # Cache for API calls
        self._price_cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(minutes=5)

        logger.info(f"AICryptoAnalyzer initialized (paper_mode={paper_mode})")

    def _get_prices(self) -> Dict[str, Dict]:
        """Fetch current prices from CoinGecko"""
        # Check cache
        if self._cache_time and datetime.now() - self._cache_time < self._cache_duration:
            return self._price_cache

        try:
            ids = ','.join([info['coingecko_id'] for info in self.WATCHLIST.values()])
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': ids,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_7d_change': 'true',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Map back to symbols
            prices = {}
            for symbol, info in self.WATCHLIST.items():
                cg_id = info['coingecko_id']
                if cg_id in data:
                    prices[symbol] = {
                        'price': data[cg_id].get('usd', 0),
                        'change_24h': data[cg_id].get('usd_24h_change', 0),
                        'change_7d': data[cg_id].get('usd_7d_change', 0),
                        'market_cap': data[cg_id].get('usd_market_cap', 0),
                        'volume_24h': data[cg_id].get('usd_24h_vol', 0)
                    }

            self._price_cache = prices
            self._cache_time = datetime.now()
            return prices

        except Exception as e:
            logger.error(f"Failed to fetch prices: {e}")
            return self._price_cache or {}

    def _get_news(self, symbol: str) -> List[NewsItem]:
        """Fetch recent news for a crypto"""
        news_items = []

        try:
            # Use CryptoPanic API (free tier)
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                'auth_token': os.getenv('CRYPTOPANIC_API_KEY', 'free'),
                'currencies': symbol,
                'filter': 'hot',
                'public': 'true'
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('results', [])[:5]:
                    sentiment = 'NEUTRAL'
                    if item.get('votes', {}).get('positive', 0) > item.get('votes', {}).get('negative', 0):
                        sentiment = 'BULLISH'
                    elif item.get('votes', {}).get('negative', 0) > item.get('votes', {}).get('positive', 0):
                        sentiment = 'BEARISH'

                    news_items.append(NewsItem(
                        title=item.get('title', ''),
                        summary=item.get('title', ''),  # API doesn't provide summary
                        source=item.get('source', {}).get('title', 'Unknown'),
                        sentiment=sentiment,
                        published=datetime.now(timezone.utc)
                    ))
        except Exception as e:
            logger.debug(f"News fetch failed for {symbol}: {e}")

        return news_items

    def _get_technical_signal(self, symbol: str, price_data: Dict) -> str:
        """Calculate technical signal based on price action"""
        change_24h = price_data.get('change_24h', 0)
        change_7d = price_data.get('change_7d', 0)

        # Simple momentum-based signal
        if change_24h > 5 and change_7d > 10:
            return 'BULLISH'
        elif change_24h < -5 and change_7d < -10:
            return 'BEARISH'
        elif change_24h > 2 or change_7d > 5:
            return 'SLIGHTLY_BULLISH'
        elif change_24h < -2 or change_7d < -5:
            return 'SLIGHTLY_BEARISH'
        else:
            return 'NEUTRAL'

    def _ai_analyze(self, symbol: str, price_data: Dict, news: List[NewsItem], technical: str) -> Dict:
        """Use AI to generate analysis"""

        if not self.api_key:
            # Fallback analysis without AI
            return self._fallback_analysis(symbol, price_data, technical)

        try:
            # Build prompt
            news_summary = "\n".join([f"- {n.title} ({n.sentiment})" for n in news[:5]]) or "No recent news"

            prompt = f"""Analyze {symbol} cryptocurrency for a trading decision.

CURRENT DATA:
- Price: ${price_data.get('price', 0):.4f}
- 24h Change: {price_data.get('change_24h', 0):+.2f}%
- 7d Change: {price_data.get('change_7d', 0):+.2f}%
- Market Cap: ${price_data.get('market_cap', 0):,.0f}
- 24h Volume: ${price_data.get('volume_24h', 0):,.0f}
- Technical Signal: {technical}

RECENT NEWS:
{news_summary}

Provide analysis in this exact JSON format:
{{
    "recommendation": "BUY" or "SELL" or "HOLD" or "STRONG_BUY" or "STRONG_SELL",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief 1-2 sentence explanation",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_level": "LOW" or "MEDIUM" or "HIGH",
    "entry_price": null or suggested entry price,
    "stop_loss": null or suggested stop loss,
    "take_profit": null or suggested take profit,
    "news_sentiment": "BULLISH" or "BEARISH" or "NEUTRAL"
}}

Be conservative. Only recommend BUY if there's clear evidence. Consider the user may already be holding at a loss."""

            # Call DeepSeek API
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are a crypto analyst. Provide objective analysis. Be conservative with buy recommendations. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']

                # Parse JSON from response
                # Handle potential markdown code blocks
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0]

                analysis = json.loads(content.strip())
                return analysis
            else:
                logger.warning(f"DeepSeek API error: {response.status_code}")
                return self._fallback_analysis(symbol, price_data, technical)

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._fallback_analysis(symbol, price_data, technical)

    def _fallback_analysis(self, symbol: str, price_data: Dict, technical: str) -> Dict:
        """Fallback analysis when AI is unavailable"""
        change_24h = price_data.get('change_24h', 0)
        change_7d = price_data.get('change_7d', 0)
        price = price_data.get('price', 0)

        # Simple rule-based analysis
        if change_24h > 10 and change_7d > 15:
            recommendation = 'HOLD'  # Already pumped, risky to buy
            confidence = 0.6
            reasoning = "Strong recent gains suggest waiting for pullback"
            risk = 'HIGH'
        elif change_24h < -10 and change_7d < -20:
            recommendation = 'HOLD'  # Falling knife
            confidence = 0.5
            reasoning = "Sharp decline - wait for stabilization before buying"
            risk = 'HIGH'
        elif change_24h > 3 and change_7d > 0:
            recommendation = 'BUY'
            confidence = 0.55
            reasoning = "Positive momentum with room to grow"
            risk = 'MEDIUM'
            entry = price * 0.98  # 2% below current
        elif change_24h < -3 and change_7d < -5:
            recommendation = 'HOLD'
            confidence = 0.5
            reasoning = "Negative momentum - wait for trend reversal"
            risk = 'MEDIUM'
        else:
            recommendation = 'HOLD'
            confidence = 0.5
            reasoning = "No clear signal - sideways action"
            risk = 'LOW'

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reasoning': reasoning,
            'key_factors': [f"24h: {change_24h:+.1f}%", f"7d: {change_7d:+.1f}%", technical],
            'risk_level': risk,
            'entry_price': price * 0.98 if recommendation == 'BUY' else None,
            'stop_loss': price * 0.90 if recommendation == 'BUY' else None,
            'take_profit': price * 1.15 if recommendation == 'BUY' else None,
            'news_sentiment': 'NEUTRAL'
        }

    def analyze(self, symbol: str) -> Optional[CryptoAnalysis]:
        """Analyze a single cryptocurrency"""
        if symbol not in self.WATCHLIST:
            logger.warning(f"Symbol {symbol} not in watchlist")
            return None

        logger.info(f"Analyzing {symbol}...")

        # Get data
        prices = self._get_prices()
        if symbol not in prices:
            logger.error(f"No price data for {symbol}")
            return None

        price_data = prices[symbol]
        news = self._get_news(symbol)
        technical = self._get_technical_signal(symbol, price_data)

        # Get AI analysis
        ai_result = self._ai_analyze(symbol, price_data, news, technical)

        # Build analysis object
        analysis = CryptoAnalysis(
            symbol=symbol,
            current_price=price_data.get('price', 0),
            price_change_24h=price_data.get('change_24h', 0),
            price_change_7d=price_data.get('change_7d', 0),
            recommendation=ai_result.get('recommendation', 'HOLD'),
            confidence=ai_result.get('confidence', 0.5),
            reasoning=ai_result.get('reasoning', 'No analysis available'),
            key_factors=ai_result.get('key_factors', []),
            entry_price=ai_result.get('entry_price'),
            stop_loss=ai_result.get('stop_loss'),
            take_profit=ai_result.get('take_profit'),
            risk_level=ai_result.get('risk_level', 'MEDIUM'),
            news_sentiment=ai_result.get('news_sentiment', 'NEUTRAL'),
            technical_signal=technical,
            timestamp=datetime.now(timezone.utc)
        )

        self.analyses[symbol] = analysis
        return analysis

    def analyze_all(self) -> Dict[str, CryptoAnalysis]:
        """Analyze all cryptos in watchlist"""
        results = {}
        for symbol in self.WATCHLIST:
            analysis = self.analyze(symbol)
            if analysis:
                results[symbol] = analysis
        return results

    def run_scan(self) -> List[Dict]:
        """
        Main method called by orchestrator.
        Returns list of signals.
        """
        logger.info("Starting AI crypto analysis scan...")

        signals = []
        analyses = self.analyze_all()

        for symbol, analysis in analyses.items():
            # Log all analyses
            logger.info(
                f"{symbol}: {analysis.recommendation} "
                f"(conf={analysis.confidence:.0%}, {analysis.reasoning})"
            )

            # Only return actionable signals
            if analysis.recommendation in ['BUY', 'STRONG_BUY'] and analysis.confidence >= 0.6:
                signals.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'price': analysis.current_price,
                    'confidence': analysis.confidence,
                    'reasoning': analysis.reasoning,
                    'entry_price': analysis.entry_price,
                    'stop_loss': analysis.stop_loss,
                    'take_profit': analysis.take_profit,
                    'risk_level': analysis.risk_level,
                    'ai_analyzed': True
                })
            elif analysis.recommendation in ['SELL', 'STRONG_SELL'] and analysis.confidence >= 0.6:
                signals.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'price': analysis.current_price,
                    'confidence': analysis.confidence,
                    'reasoning': analysis.reasoning,
                    'risk_level': analysis.risk_level,
                    'ai_analyzed': True
                })

        logger.info(f"AI crypto scan complete: {len(analyses)} analyzed, {len(signals)} signals")
        return signals

    def get_recommendation(self, symbol: str) -> str:
        """Get a formatted recommendation for a symbol"""
        analysis = self.analyze(symbol)
        if not analysis:
            return f"Unable to analyze {symbol}"

        rec_emoji = {
            'STRONG_BUY': '🟢🟢',
            'BUY': '🟢',
            'HOLD': '🟡',
            'SELL': '🔴',
            'STRONG_SELL': '🔴🔴'
        }

        emoji = rec_emoji.get(analysis.recommendation, '⚪')

        output = f"""
{'='*50}
{emoji} {symbol} ANALYSIS {emoji}
{'='*50}
Price: ${analysis.current_price:.4f}
24h:   {analysis.price_change_24h:+.2f}%
7d:    {analysis.price_change_7d:+.2f}%

RECOMMENDATION: {analysis.recommendation}
Confidence: {analysis.confidence:.0%}
Risk Level: {analysis.risk_level}

Reasoning: {analysis.reasoning}

Key Factors:
{chr(10).join(['  - ' + f for f in analysis.key_factors])}

Technical: {analysis.technical_signal}
Sentiment: {analysis.news_sentiment}
"""

        if analysis.entry_price:
            output += f"""
Suggested Entry: ${analysis.entry_price:.4f}
Stop Loss: ${analysis.stop_loss:.4f}
Take Profit: ${analysis.take_profit:.4f}
"""

        return output


# Quick test
if __name__ == "__main__":
    analyzer = AICryptoAnalyzer(paper_mode=True)

    # Analyze specific cryptos
    for symbol in ['XRP', 'HBAR', 'XLM']:
        print(analyzer.get_recommendation(symbol))
