"""
Test connecting to real Kalshi API and finding markets
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")

from bots.kalshi_client import KalshiClient


def test_kalshi_connection():
    """Test actual connection to Kalshi API"""
    print("Testing real Kalshi API connection...")

    try:
        kalshi = KalshiClient()
        print(f"Kalshi client created")

        # Try to get markets - this will test our authentication
        markets = kalshi.get_markets(limit=5)  # Just get 5 markets to start

        print(f"Successfully retrieved {len(markets)} markets")

        # Show what we got
        for i, market in enumerate(markets, 1):
            title = market.get("title", "No title")
            ticker = market.get("ticker", "No ticker")
            status = market.get("status", "unknown")
            print(f"  {i}. {ticker}: {title[:50]}... (status: {status})")

        return True, markets

    except Exception as e:
        print(f"ERROR: {e}")
        return False, []


def find_news_relevant_markets(markets, news_keywords):
    """Find markets that match our news keywords"""
    print(f"\nLooking for markets matching keywords: {news_keywords}")

    relevant_markets = []

    for market in markets:
        market_text = (
            f"{market.get('title', '')} {market.get('description', '')}".lower()
        )

        # Check for keyword matches
        matches = [kw for kw in news_keywords if kw.lower() in market_text]

        if matches:
            relevant_markets.append({"market": market, "matched_keywords": matches})

    return relevant_markets


def main():
    """Test the complete flow"""

    # Step 1: Test Kalshi connection
    success, markets = test_kalshi_connection()

    if not success:
        print("Cannot proceed - Kalshi connection failed")
        return

    # Step 2: Test news analysis
    sample_news = (
        "Federal Reserve officials signal interest rate cuts due to declining inflation"
    )

    # Extract keywords (our simple function)
    def extract_keywords(text):
        terms = [
            "election",
            "inflation",
            "unemployment",
            "fed",
            "federal reserve",
            "interest rates",
        ]
        return [term for term in terms if term.lower() in text.lower()]

    keywords = extract_keywords(sample_news)
    print(f"\nNews: {sample_news}")
    print(f"Extracted keywords: {keywords}")

    # Step 3: Find relevant markets
    if keywords:
        relevant = find_news_relevant_markets(markets, keywords)

        print(f"\nFound {len(relevant)} relevant markets:")
        for i, item in enumerate(relevant, 1):
            market = item["market"]
            matched = item["matched_keywords"]
            print(
                f"  {i}. {market.get('ticker', 'N/A')}: {market.get('title', 'N/A')[:60]}..."
            )
            print(f"     Matched keywords: {matched}")

    if success and keywords:
        print(
            f"\nSUCCESS: Found connection between news and {len(relevant)} Kalshi markets!"
        )
        return True
    else:
        print(f"\nPartial success - need to improve keyword matching")
        return False


if __name__ == "__main__":
    main()
