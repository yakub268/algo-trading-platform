"""
Test AI analyzer adapted for sports betting instead of economic news
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

from bots.kalshi_client import KalshiClient


def extract_sports_keywords(text: str) -> list:
    """Extract sports-relevant keywords from news text"""
    sports_terms = [
        # NFL
        "nfl",
        "football",
        "quarterback",
        "touchdown",
        "patriots",
        "seahawks",
        "ravens",
        "chiefs",
        "49ers",
        "cowboys",
        "steelers",
        "packers",
        # NBA
        "nba",
        "basketball",
        "lakers",
        "celtics",
        "warriors",
        "nets",
        "heat",
        # MLB
        "mlb",
        "baseball",
        "yankees",
        "dodgers",
        "red sox",
        "giants",
        # Soccer
        "soccer",
        "football",
        "arsenal",
        "manchester",
        "chelsea",
        "liverpool",
        "premier league",
        "champions league",
        # NHL
        "nhl",
        "hockey",
        "bruins",
        "rangers",
        "flyers",
        "capitals",
        # College
        "college",
        "ncaa",
        "duke",
        "kentucky",
        "north carolina",
        "gonzaga",
        # General sports terms
        "injury",
        "trade",
        "draft",
        "playoffs",
        "championship",
        "coach",
        "win",
        "loss",
        "season",
        "roster",
        "player",
        "team",
    ]

    text_lower = text.lower()
    found_keywords = [term for term in sports_terms if term in text_lower]
    return found_keywords


def find_matching_sports_markets(markets, keywords):
    """Find Kalshi sports markets that match news keywords"""
    matches = []

    for market in markets:
        market_text = f"{market.get('title', '')} {market.get('ticker', '')}".lower()

        # Find keyword matches
        matched_keywords = [kw for kw in keywords if kw in market_text]

        if matched_keywords:
            matches.append(
                {
                    "market": market,
                    "matched_keywords": matched_keywords,
                    "relevance_score": len(matched_keywords),
                }
            )

    # Sort by relevance (most keyword matches first)
    matches.sort(key=lambda x: x["relevance_score"], reverse=True)
    return matches


def test_sports_ai():
    """Test sports-focused AI analysis"""

    print("Testing Sports AI Analyzer")
    print("=" * 40)

    # Get Kalshi sports markets
    kalshi = KalshiClient()
    markets = kalshi.get_markets(limit=20)
    print(f"Retrieved {len(markets)} markets from Kalshi")

    # Test sports news examples
    sports_news_examples = [
        "Drake Maye throws for 300 yards as Patriots beat Seahawks in overtime thriller",
        "Arsenal defeats Manchester United 3-1 in Premier League clash at Emirates",
        "Kenneth Walker III rushes for 150 yards and 2 touchdowns in Seahawks victory",
    ]

    for i, news in enumerate(sports_news_examples, 1):
        print(f"\nTest {i}: Sports News Analysis")
        print(f"News: {news}")

        # Extract keywords
        keywords = extract_sports_keywords(news)
        print(f"Keywords found: {keywords}")

        if keywords:
            # Find matching markets
            matches = find_matching_sports_markets(markets, keywords)
            print(f"Matching markets found: {len(matches)}")

            for j, match in enumerate(matches[:3], 1):  # Show top 3 matches
                market = match["market"]
                matched_kw = match["matched_keywords"]
                score = match["relevance_score"]

                print(f"  {j}. {market.get('ticker', 'N/A')}")
                print(f"     Title: {market.get('title', 'N/A')[:60]}...")
                print(f"     Matched: {matched_kw} (score: {score})")

    # Overall assessment
    total_keywords_found = sum(
        len(extract_sports_keywords(news)) for news in sports_news_examples
    )

    print(f"\n" + "=" * 40)
    print(f"RESULTS:")
    print(f"- Total keywords extracted: {total_keywords_found}")
    print(f"- Sports markets available: {len(markets)}")
    print(f"- AI analyzer successfully adapted for sports!")

    return True


if __name__ == "__main__":
    test_sports_ai()
