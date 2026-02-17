"""
AI Prediction Market Analyzer

Integrates local LLM (Ollama) for real-time news analysis and prediction market opportunities.
Connects to your existing Kalshi infrastructure with AI-powered edge detection.

Features:
- Real-time news sentiment analysis using local LLM
- Event outcome prediction for Kalshi markets
- Cross-market arbitrage detection with AI correlation analysis
- Automated confidence scoring for trades
- Integration with existing orchestrator and risk management

Author: AI Trading Enhancement
Created: February 2026
"""

import os
import sys
import json
import asyncio
import aiohttp
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ai_prediction_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class NewsEvent:
    """Structure for news events"""
    title: str
    content: str
    source: str
    timestamp: datetime
    confidence: float
    sentiment: float  # -1 to 1
    predicted_outcome: str
    kalshi_markets: List[str] = None

@dataclass
class AIOpportunity:
    """AI-detected trading opportunity"""
    market_ticker: str
    market_title: str
    ai_prediction: str  # YES/NO
    ai_confidence: float  # 0-1
    current_price: float
    suggested_price: float
    edge_percentage: float
    reasoning: str
    news_sources: List[str]
    risk_level: str  # LOW/MEDIUM/HIGH

class OllamaClient:
    """Client for local Ollama LLM integration"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def generate(self, prompt: str) -> str:
        """Generate text using local LLM"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }

            async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "")
                else:
                    logger.error(f"Ollama API error: {response.status}")
                    return ""
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return ""

    async def analyze_news_for_prediction(self, news_text: str, market_description: str) -> Dict[str, Any]:
        """Analyze news text for prediction market implications"""
        prompt = f"""
        You are an expert prediction market analyst. Analyze this news and determine its impact on the prediction market.

        NEWS: {news_text[:1000]}

        MARKET: {market_description}

        Provide analysis in this exact JSON format:
        {{
            "prediction": "YES" or "NO",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation",
            "sentiment": -1.0 to 1.0,
            "key_factors": ["factor1", "factor2"],
            "risk_level": "LOW", "MEDIUM", or "HIGH"
        }}

        Focus on factual analysis. Consider both direct and indirect implications.
        """

        response = await self.generate(prompt)

        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end] if start != -1 and end > start else "{}"
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response: {response[:200]}")
            return {
                "prediction": "NO",
                "confidence": 0.0,
                "reasoning": "Failed to analyze",
                "sentiment": 0.0,
                "key_factors": [],
                "risk_level": "HIGH"
            }

class AIPredictionAnalyzer:
    """Main AI Prediction Market Analyzer"""

    def __init__(self, kalshi_client: KalshiClient = None, ollama_model: str = "llama2", paper_mode: bool = True):
        # Create Kalshi client if not provided
        if kalshi_client is None:
            try:
                self.kalshi = KalshiClient()
                logger.info("Created internal Kalshi client")
            except Exception as e:
                logger.warning(f"Could not create Kalshi client: {e}")
                self.kalshi = None
        else:
            self.kalshi = kalshi_client

        self.ollama_model = ollama_model
        self.paper_mode = paper_mode
        self.opportunities = []
        self.news_cache = {}

        # Configuration
        self.min_confidence = 0.7  # Minimum AI confidence for trades
        self.min_edge = 5.0        # Minimum edge percentage
        self.max_positions = 10    # Max concurrent AI positions

        logger.info(f"AIPredictionAnalyzer initialized (paper_mode={paper_mode})")

    async def get_news_sources(self) -> List[Dict[str, str]]:
        """Get news from multiple sources"""
        # Placeholder - integrate with your preferred news APIs
        news_sources = [
            {
                "title": "Fed Minutes Show Dovish Sentiment on Inflation",
                "content": "Federal Reserve meeting minutes released today show growing consensus among officials that inflation pressures are moderating...",
                "source": "Reuters",
                "timestamp": datetime.now(timezone.utc)
            }
        ]
        return news_sources

    async def find_relevant_markets(self, news_event: NewsEvent) -> List[Dict[str, Any]]:
        """Find Kalshi markets relevant to a news event"""
        try:
            # Get active markets from Kalshi
            markets = await self.kalshi.get_markets()

            relevant_markets = []
            keywords = self._extract_keywords(news_event.content)

            for market in markets:
                market_text = f"{market['title']} {market.get('description', '')}".lower()

                # Check for keyword matches
                relevance_score = sum(1 for keyword in keywords if keyword.lower() in market_text)

                if relevance_score > 0:
                    relevant_markets.append({
                        "ticker": market['ticker'],
                        "title": market['title'],
                        "relevance_score": relevance_score
                    })

            # Sort by relevance
            return sorted(relevant_markets, key=lambda x: x['relevance_score'], reverse=True)[:5]

        except Exception as e:
            logger.error(f"Error finding relevant markets: {e}")
            return []

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from news text"""
        # Simple keyword extraction - could be enhanced with NLP
        important_terms = [
            "election", "inflation", "unemployment", "GDP", "CPI", "Fed", "Federal Reserve",
            "interest rates", "recession", "earnings", "Congress", "Supreme Court",
            "oil", "gas", "climate", "weather", "sports", "Olympics", "World Cup"
        ]

        text_lower = text.lower()
        found_keywords = [term for term in important_terms if term.lower() in text_lower]
        return found_keywords

    async def analyze_opportunity(self, market: Dict[str, Any], news_event: NewsEvent) -> Optional[AIOpportunity]:
        """Analyze a specific market opportunity using AI"""
        try:
            async with OllamaClient(model=self.ollama_model) as ollama:
                # Get AI analysis
                analysis = await ollama.analyze_news_for_prediction(
                    news_event.content,
                    market['title']
                )

                # Get current market price
                market_data = await self.kalshi.get_market(market['ticker'])
                if not market_data:
                    return None

                yes_price = market_data.get('yes_price', 0.5)
                ai_confidence = analysis.get('confidence', 0.0)

                # Calculate edge
                if analysis['prediction'] == "YES":
                    suggested_price = ai_confidence
                    edge = (suggested_price - yes_price) / yes_price * 100
                else:
                    suggested_price = 1 - ai_confidence
                    edge = (suggested_price - (1 - yes_price)) / (1 - yes_price) * 100

                # Only consider opportunities with sufficient edge and confidence
                if ai_confidence >= self.min_confidence and abs(edge) >= self.min_edge:
                    return AIOpportunity(
                        market_ticker=market['ticker'],
                        market_title=market['title'],
                        ai_prediction=analysis['prediction'],
                        ai_confidence=ai_confidence,
                        current_price=yes_price,
                        suggested_price=suggested_price,
                        edge_percentage=edge,
                        reasoning=analysis.get('reasoning', 'AI analysis'),
                        news_sources=[news_event.source],
                        risk_level=analysis.get('risk_level', 'MEDIUM')
                    )

        except Exception as e:
            logger.error(f"Error analyzing opportunity: {e}")

        return None

    async def scan_opportunities(self) -> List[AIOpportunity]:
        """Main method to scan for AI-detected opportunities"""
        logger.info("Starting AI opportunity scan...")

        opportunities = []

        try:
            # Get latest news
            news_events = await self.get_news_sources()

            for news_item in news_events:
                news_event = NewsEvent(
                    title=news_item['title'],
                    content=news_item['content'],
                    source=news_item['source'],
                    timestamp=news_item['timestamp'],
                    confidence=1.0,
                    sentiment=0.0,
                    predicted_outcome=""
                )

                # Find relevant markets
                relevant_markets = await self.find_relevant_markets(news_event)

                # Analyze each relevant market
                for market in relevant_markets:
                    opportunity = await self.analyze_opportunity(market, news_event)
                    if opportunity:
                        opportunities.append(opportunity)
                        logger.info(f"Found AI opportunity: {opportunity.market_ticker} - {opportunity.edge_percentage:.2f}% edge")

        except Exception as e:
            logger.error(f"Error in opportunity scan: {e}")

        self.opportunities = opportunities
        return opportunities

    async def get_cross_market_arbitrage(self) -> List[Dict[str, Any]]:
        """Find arbitrage opportunities between Kalshi and other prediction markets"""
        # Placeholder for cross-market arbitrage detection
        # This would integrate with Polymarket, PredictIt, etc.
        logger.info("Scanning for cross-market arbitrage...")
        return []

    def generate_trading_signals(self) -> List[Dict[str, Any]]:
        """Generate trading signals from AI opportunities"""
        signals = []

        for opp in self.opportunities:
            if opp.risk_level == "HIGH":
                continue

            # Calculate position size based on Kelly criterion
            kelly_fraction = self._calculate_kelly_fraction(opp)

            signal = {
                "action": "BUY" if opp.ai_prediction == "YES" else "SELL",
                "ticker": opp.market_ticker,
                "confidence": opp.ai_confidence,
                "edge": opp.edge_percentage,
                "kelly_fraction": kelly_fraction,
                "reasoning": opp.reasoning,
                "risk_level": opp.risk_level
            }
            signals.append(signal)

        return signals

    def run_scan(self) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for scan_opportunities.
        Called by the orchestrator.
        """
        if self.kalshi is None:
            logger.warning("Kalshi client not available - skipping AI prediction scan")
            return []

        try:
            # Run async scan in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                opportunities = loop.run_until_complete(self.scan_opportunities())
            finally:
                loop.close()

            # Generate and return trading signals
            signals = self.generate_trading_signals()
            logger.info(f"AI Prediction scan complete: {len(opportunities)} opportunities, {len(signals)} signals")
            return signals

        except Exception as e:
            logger.error(f"Error in run_scan: {e}")
            return []

    def _calculate_kelly_fraction(self, opportunity: AIOpportunity) -> float:
        """Calculate Kelly criterion position size"""
        # Simplified Kelly calculation
        p = opportunity.ai_confidence  # Probability of winning
        b = abs(opportunity.edge_percentage) / 100  # Odds

        if p <= 0.5:  # Don't bet if probability <= 50%
            return 0.0

        kelly = (p * (b + 1) - 1) / b

        # Cap at 10% of bankroll for safety
        return min(kelly, 0.10)

async def main():
    """Main execution function"""
    # Initialize clients
    kalshi = KalshiClient()

    # Initialize AI analyzer
    analyzer = AIPredictionAnalyzer(kalshi, ollama_model="llama2")

    try:
        # Test Ollama connection
        async with OllamaClient() as ollama:
            test_response = await ollama.generate("Say 'AI Trading Bot Online!'")
            logger.info(f"Ollama test: {test_response[:50]}")

        # Scan for opportunities
        opportunities = await analyzer.scan_opportunities()

        # Generate trading signals
        signals = analyzer.generate_trading_signals()

        logger.info(f"Found {len(opportunities)} AI opportunities")
        logger.info(f"Generated {len(signals)} trading signals")

        # Print results
        for signal in signals:
            print(f"SIGNAL: {signal['action']} {signal['ticker']} - {signal['confidence']:.2f} confidence, {signal['edge']:.2f}% edge")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())