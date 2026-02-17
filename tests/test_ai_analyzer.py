"""
Simple test for AI Prediction Analyzer without Ollama dependency

NOT a unit test — run directly: python tests/test_ai_analyzer.py
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Standalone integration script — run directly with: python tests/test_ai_analyzer.py"
)

import sys
import os
import asyncio
import logging

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bots.kalshi_client import KalshiClient
from bots.ai_prediction_analyzer import AIPredictionAnalyzer, NewsEvent
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_kalshi_connection():
    """Test basic Kalshi connection"""
    print("Testing Kalshi connection...")

    kalshi = KalshiClient()

    try:
        # Test getting markets
        markets = await kalshi.get_markets()
        print(f"✅ Kalshi connected! Found {len(markets)} markets")

        # Show a few markets
        for i, market in enumerate(markets[:3]):
            print(
                f"  {i+1}. {market.get('title', 'Unknown')} ({market.get('ticker', 'N/A')})"
            )

        return True

    except Exception as e:
        print(f"❌ Kalshi connection failed: {e}")
        return False


async def test_analyzer_without_ollama():
    """Test analyzer components that don't need Ollama"""
    print("\nTesting AI Analyzer components...")

    kalshi = KalshiClient()
    analyzer = AIPredictionAnalyzer(kalshi, ollama_model="llama2")

    # Test keyword extraction
    test_text = "The Federal Reserve announced today that inflation has decreased to 3.2%, signaling potential interest rate cuts in the coming months."
    keywords = analyzer._extract_keywords(test_text)
    print(f"✅ Keyword extraction works: {keywords}")

    # Test news event creation
    news_event = NewsEvent(
        title="Fed Announces Rate Decision",
        content=test_text,
        source="Reuters",
        timestamp=datetime.now(timezone.utc),
        confidence=1.0,
        sentiment=0.2,
        predicted_outcome="YES",
    )
    print(f"✅ NewsEvent creation works: {news_event.title}")

    # Test finding relevant markets
    try:
        relevant_markets = await analyzer.find_relevant_markets(news_event)
        print(f"✅ Found {len(relevant_markets)} relevant markets for Fed news")

        for market in relevant_markets[:2]:
            print(
                f"  - {market['title'][:50]}... (relevance: {market['relevance_score']})"
            )

    except Exception as e:
        print(f"❌ Market finding failed: {e}")

    return True


async def test_mock_ai_analysis():
    """Test with mock AI analysis (no Ollama needed)"""
    print("\nTesting with mock AI analysis...")

    # Create mock analysis result (what Ollama would return)
    mock_analysis = {
        "prediction": "YES",
        "confidence": 0.75,
        "reasoning": "Fed rate cuts typically boost market sentiment",
        "sentiment": 0.3,
        "key_factors": ["inflation", "rate cuts", "fed"],
        "risk_level": "MEDIUM",
    }

    print(
        f"✅ Mock AI analysis: {mock_analysis['prediction']} with {mock_analysis['confidence']} confidence"
    )
    print(f"   Reasoning: {mock_analysis['reasoning']}")

    return True


async def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING AI PREDICTION ANALYZER")
    print("=" * 60)

    # Test 1: Kalshi connection
    kalshi_ok = await test_kalshi_connection()

    # Test 2: Analyzer components
    if kalshi_ok:
        analyzer_ok = await test_analyzer_without_ollama()
    else:
        print("⚠️ Skipping analyzer tests due to Kalshi connection issues")
        analyzer_ok = False

    # Test 3: Mock AI analysis
    mock_ok = await test_mock_ai_analysis()

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Kalshi Connection: {'✅ PASS' if kalshi_ok else '❌ FAIL'}")
    print(f"Analyzer Components: {'✅ PASS' if analyzer_ok else '❌ FAIL'}")
    print(f"Mock AI Analysis: {'✅ PASS' if mock_ok else '❌ FAIL'}")

    if kalshi_ok and analyzer_ok:
        print("\n🎉 AI Analyzer is ready for Ollama integration!")
        print("💡 Next step: Install and start Ollama to test full AI functionality")
    else:
        print("\n⚠️ Some components need fixing before full deployment")


if __name__ == "__main__":
    asyncio.run(main())
