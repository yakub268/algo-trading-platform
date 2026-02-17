"""
Test News Feed Integration - Comprehensive testing of live news feeds

This script demonstrates how the new news feed system integrates with
the existing Sports AI bot and other trading components.
"""

import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, ".")

# Import news feed components
from news_feeds import NewsAggregator
from news_feeds.sources.espn_connector import ESPNConnector
from news_feeds.sources.financial_connector import FinancialNewsConnector
from news_feeds.sources.reddit_connector import RedditConnector
from news_feeds.sentiment import SentimentAnalyzer

# Import updated Sports AI bot
try:
    from bots.sports_ai_bot import SportsAIBot
except ImportError:
    SportsAIBot = None


def test_individual_connectors():
    """Test each news connector individually"""

    print("=" * 60)
    print("TESTING INDIVIDUAL NEWS CONNECTORS")
    print("=" * 60)

    # Test ESPN Connector
    print("\\n1. Testing ESPN Sports Connector...")
    try:
        espn = ESPNConnector()
        espn_articles = espn.fetch_news(category="nfl", limit=3)

        print(f"   [OK] ESPN: Retrieved {len(espn_articles)} articles")
        for i, article in enumerate(espn_articles[:2], 1):
            print(f"   {i}. {article.title[:60]}...")

    except Exception as e:
        print(f"   [ERROR] ESPN Error: {e}")

    # Test Financial News Connector
    print("\\n2. Testing Financial News Connector...")
    try:
        financial = FinancialNewsConnector()
        financial_articles = financial.fetch_news(category="market_news", limit=3)

        print(f"   [OK] Financial: Retrieved {len(financial_articles)} articles")
        for i, article in enumerate(financial_articles[:2], 1):
            print(f"   {i}. {article.title[:60]}...")
            print(f"      Sentiment: {article.sentiment:.2f}")

    except Exception as e:
        print(f"   [ERROR] Financial Error: {e}")

    # Test Reddit Connector
    print("\\n3. Testing Reddit Connector...")
    try:
        reddit = RedditConnector()
        reddit_articles = reddit.fetch_news(category="investing", limit=3)

        print(f"   [OK] Reddit: Retrieved {len(reddit_articles)} articles")
        for i, article in enumerate(reddit_articles[:2], 1):
            print(f"   {i}. {article.title[:60]}...")
            print(f"      Sentiment: {article.sentiment:.2f}")
            print(f"      Relevance: {article.relevance_score:.2f}")

    except Exception as e:
        print(f"   [ERROR] Reddit Error: {e}")


def test_sentiment_analysis():
    """Test sentiment analysis system"""

    print("\\n" + "=" * 60)
    print("TESTING SENTIMENT ANALYSIS")
    print("=" * 60)

    analyzer = SentimentAnalyzer()

    test_headlines = [
        "Drake Maye throws 3 touchdowns in Patriots dominant victory",
        "Kenneth Walker injured during practice, questionable for Sunday",
        "Stock market surges on positive economic data",
        "Bitcoin crashes 15% amid regulatory concerns",
        "Fed raises interest rates dramatically, markets fall",
    ]

    for i, headline in enumerate(test_headlines, 1):
        result = analyzer.analyze_sentiment(headline)

        print(f'\\n{i}. "{headline}"')
        print(f"   Sentiment: {result.score:.3f} (confidence: {result.confidence:.2f})")
        print(f"   Category: {result.category}")
        print(f"   Keywords: {', '.join(result.keywords_found)}")


def test_news_aggregator():
    """Test the unified NewsAggregator"""

    print("\\n" + "=" * 60)
    print("TESTING NEWS AGGREGATOR")
    print("=" * 60)

    try:
        aggregator = NewsAggregator()

        # Test system status
        status = aggregator.get_system_status()
        print("\\nSystem Status:")
        for connector, data in status["connectors"].items():
            status_icon = "[OK]" if data["connected"] else "[FAIL]"
            print(f"  {status_icon} {connector}: {data['error_count']} errors")

        # Test sports news
        print("\\nFetching sports news for specific players...")
        sports_news = aggregator.fetch_sports_news(
            ["drake maye", "kenneth walker"], limit=5
        )

        print(f"Retrieved {len(sports_news)} sports articles:")
        for i, article in enumerate(sports_news[:3], 1):
            print(f"  {i}. {article.title[:70]}...")
            print(f"     Source: {article.source}, Sentiment: {article.sentiment:.2f}")

        # Test market sentiment
        print("\\nTesting market sentiment analysis...")
        sentiment_data = aggregator.fetch_market_sentiment(["AAPL", "TSLA"], limit=10)

        print("Market Sentiment Analysis:")
        print(f"  Overall Sentiment: {sentiment_data['overall_sentiment']:.2f}")
        print(f"  Confidence: {sentiment_data['confidence']:.2f}")
        print(f"  Articles Analyzed: {sentiment_data['article_count']}")

        # Test Fed news
        print("\\nTesting Fed news retrieval...")
        fed_news = aggregator.fetch_fed_news(limit=3)

        print(f"Retrieved {len(fed_news)} Fed articles:")
        for i, article in enumerate(fed_news[:2], 1):
            print(f"  {i}. {article.title[:70]}...")
            print(f"     Sentiment: {article.sentiment:.2f}")

        aggregator.shutdown()

    except Exception as e:
        print(f"NewsAggregator Error: {e}")


def test_sports_ai_integration():
    """Test the updated Sports AI bot with live news"""

    print("\\n" + "=" * 60)
    print("TESTING SPORTS AI BOT WITH LIVE NEWS")
    print("=" * 60)

    try:
        print("Initializing Sports AI Bot with live news integration...")
        bot = SportsAIBot()

        if hasattr(bot, "use_live_news") and bot.use_live_news:
            print("[OK] Live news integration active")
        else:
            print("[WARN] Using fallback news data")

        # Run the bot strategy
        print("\\nRunning Sports AI strategy...")
        results = bot.run_strategy()

        print("\\nResults:")
        print(f"  Status: {results['status']}")
        print(f"  Message: {results['message']}")
        print(f"  Total Opportunities: {results['total_opportunities']}")
        print(f"  High-Value Opportunities: {results['high_value_opportunities']}")

        if results["trades"]:
            print("\\nHigh-Value Trading Opportunities:")
            for i, trade in enumerate(results["trades"][:3], 1):
                print(f"  {i}. {trade['player']} - {trade['prop']}")
                print(f"     Direction: {trade['direction']}")
                print(f"     Confidence: {trade['confidence']}")
                print(f"     Expected Value: {trade['expected_value']}")
                print(f"     Edge: {trade['edge']}")

        # Cleanup
        bot.close_positions()

    except Exception as e:
        print(f"Sports AI Bot Error: {e}")
        import traceback

        traceback.print_exc()


def test_cache_system():
    """Test the caching system"""

    print("\\n" + "=" * 60)
    print("TESTING CACHE SYSTEM")
    print("=" * 60)

    try:
        from news_feeds.cache_manager import CacheManager

        cache = CacheManager()

        # Test caching
        test_articles = [
            {
                "title": "Test Article",
                "content": "Test content",
                "source": "Test",
                "published_at": datetime.now().isoformat(),
            }
        ]

        # Cache articles
        cache.cache_articles(test_articles, "ESPN", "sports", ["test"])

        # Retrieve from cache
        cached = cache.get_cached_articles("ESPN", "sports", ["test"])

        print(f"Cache test: {'[OK] Passed' if cached else '[FAIL] Failed'}")

        # Get cache stats
        stats = cache.get_cache_stats()
        print("Cache Stats:")
        print(f"  Article cache entries: {stats['article_cache_entries']}")
        print(f"  Memory cache entries: {stats['memory_cache_entries']}")
        print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")

    except Exception as e:
        print(f"Cache System Error: {e}")


def main():
    """Run comprehensive news feed integration tests"""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("TRADING BOT NEWS FEED INTEGRATION TEST")
    print("=" * 60)
    print(f"Test started: {datetime.now()}")

    # Run all tests
    test_individual_connectors()
    test_sentiment_analysis()
    test_cache_system()
    test_news_aggregator()
    test_sports_ai_integration()

    print("\\n" + "=" * 60)
    print("NEWS FEED INTEGRATION TEST COMPLETE")
    print("=" * 60)

    print("\\nNEXT STEPS:")
    print("1. Install new dependencies: pip install -r requirements.txt")
    print("2. Configure news feed settings in .env file")
    print("3. Run live trading with: python master_orchestrator.py")
    print("4. Monitor logs for news feed activity")


if __name__ == "__main__":
    main()
