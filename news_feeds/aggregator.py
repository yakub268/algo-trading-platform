"""
News Aggregator - Unified interface for all news sources

Combines multiple news sources and provides a single interface for all trading bots.
Handles caching, rate limiting, sentiment analysis, and error recovery.
"""

import os
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .sources.espn_connector import ESPNConnector
from .sources.financial_connector import FinancialNewsConnector
from .sources.reddit_connector import RedditConnector
from .sources.base_connector import NewsArticle
from .sentiment import SentimentAnalyzer
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)

class NewsAggregator:
    """Unified news aggregation system for trading bots"""

    def __init__(self, cache_ttl: int = 300, max_workers: int = 3):
        self.cache_ttl = cache_ttl
        self.max_workers = max_workers

        # Initialize components
        self.cache_manager = CacheManager(default_ttl=cache_ttl)
        self.sentiment_analyzer = SentimentAnalyzer()

        # Health monitoring (must be before _init_connectors)
        self.last_health_check = time.time()
        self.connector_status = {}

        # Initialize connectors
        self._init_connectors()

        # Background tasks
        self._cleanup_thread = None
        self._start_background_tasks()

    def _init_connectors(self):
        """Initialize news source connectors"""

        self.connectors = {
            'espn': ESPNConnector(),
            'financial': FinancialNewsConnector(),
            'reddit': RedditConnector()
        }

        # Test connections on startup
        logger.info("Initializing news connectors...")
        for name, connector in self.connectors.items():
            try:
                status = connector.test_connection()
                self.connector_status[name] = {
                    'connected': status,
                    'last_check': time.time(),
                    'error_count': 0
                }
                logger.info(f"{name.title()} connector: {'Connected' if status else 'Failed'}")
            except Exception as e:
                logger.error(f"Failed to initialize {name} connector: {e}")
                self.connector_status[name] = {
                    'connected': False,
                    'last_check': time.time(),
                    'error_count': 1
                }

    def _start_background_tasks(self):
        """Start background maintenance tasks"""

        def cleanup_task():
            while True:
                try:
                    time.sleep(600)  # Run every 10 minutes
                    self.cache_manager.cleanup_expired_cache()
                    self._health_check()
                except Exception as e:
                    logger.error(f"Background cleanup error: {e}")

        self._cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        self._cleanup_thread.start()

    def fetch_news(self, sources: List[str] = None, categories: List[str] = None,
                   keywords: List[str] = None, limit: int = 50,
                   time_range_hours: int = 24, use_cache: bool = True) -> List[NewsArticle]:
        """
        Fetch news from multiple sources with unified interface

        Args:
            sources: List of source names ('espn', 'financial', 'reddit')
            categories: List of categories to fetch
            keywords: Keywords to filter by
            limit: Maximum number of articles
            time_range_hours: Only include articles from last N hours
            use_cache: Whether to use cached results

        Returns:
            List of NewsArticle objects sorted by relevance and recency
        """

        # Default to all sources if none specified
        if sources is None:
            sources = list(self.connectors.keys())

        # Default categories based on sources
        if categories is None:
            categories = self._get_default_categories(sources)

        all_articles = []

        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_source = {}

            for source in sources:
                if source not in self.connectors:
                    logger.warning(f"Unknown source: {source}")
                    continue

                # Submit fetch tasks for each category
                for category in categories:
                    if self._is_source_available(source):
                        future = executor.submit(
                            self._fetch_from_source,
                            source, category, keywords, limit // len(sources) // len(categories) + 5, use_cache
                        )
                        future_to_source[future] = (source, category)

            # Collect results as they complete
            for future in as_completed(future_to_source, timeout=30):
                source, category = future_to_source[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                    logger.debug(f"Fetched {len(articles)} articles from {source}/{category}")
                except Exception as e:
                    logger.error(f"Failed to fetch from {source}/{category}: {e}")
                    self._record_connector_error(source)

        # Filter by time range
        if time_range_hours:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_range_hours)
            all_articles = [a for a in all_articles if a.published_at >= cutoff_time]

        # Remove duplicates
        all_articles = self._remove_duplicates(all_articles)

        # Apply sentiment analysis
        all_articles = self._apply_sentiment_analysis(all_articles)

        # Sort by relevance and recency
        all_articles = self._sort_articles(all_articles)

        return all_articles[:limit]

    def _fetch_from_source(self, source: str, category: str, keywords: List[str],
                          limit: int, use_cache: bool) -> List[NewsArticle]:
        """Fetch news from a specific source with caching"""

        # Check cache first
        if use_cache:
            cached_articles = self.cache_manager.get_cached_articles(source, category, keywords)
            if cached_articles:
                for a in cached_articles:
                    if isinstance(a.get('published_at'), str):
                        from datetime import datetime as _dt
                        try:
                            a['published_at'] = _dt.fromisoformat(a['published_at'])
                        except (ValueError, TypeError):
                            a['published_at'] = _dt.now()
                return [NewsArticle(**article) for article in cached_articles]

        # Fetch fresh data
        try:
            connector = self.connectors[source]
            articles = connector.fetch_news(category=category, limit=limit, keywords=keywords)

            # Cache the results
            if use_cache and articles:
                article_dicts = [article.to_dict() for article in articles]
                self.cache_manager.cache_articles(article_dicts, source, category, keywords)

            return articles

        except Exception as e:
            logger.error(f"Error fetching from {source}: {e}")
            self._record_connector_error(source)
            return []

    def _get_default_categories(self, sources: List[str]) -> List[str]:
        """Get default categories based on sources"""

        categories = []

        for source in sources:
            if source == 'espn':
                categories.extend(['nfl', 'general'])
            elif source == 'financial':
                categories.extend(['market_news', 'crypto', 'economics'])
            elif source == 'reddit':
                categories.extend(['investing', 'stocks', 'cryptocurrency'])

        return list(set(categories))  # Remove duplicates

    def _is_source_available(self, source: str) -> bool:
        """Check if a source is currently available"""

        if source not in self.connector_status:
            return True  # Assume available if not checked yet

        status = self.connector_status[source]
        return status['connected'] and status['error_count'] < 5

    def _record_connector_error(self, source: str):
        """Record an error for a connector"""

        if source in self.connector_status:
            self.connector_status[source]['error_count'] += 1
            self.connector_status[source]['last_error'] = time.time()

    def _remove_duplicates(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity"""

        unique_articles = []
        seen_titles = set()

        for article in articles:
            # Create a normalized title for comparison
            normalized_title = article.title.lower().strip()

            # Simple duplicate detection
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_articles.append(article)

        return unique_articles

    def _apply_sentiment_analysis(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Apply sentiment analysis to articles"""

        for article in articles:
            try:
                sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                    f"{article.title} {article.content}",
                    category=article.category
                )

                article.sentiment = sentiment_result.score
                if hasattr(article, 'sentiment_confidence'):
                    article.sentiment_confidence = sentiment_result.confidence

            except Exception as e:
                logger.warning(f"Failed to analyze sentiment for article {article.id}: {e}")
                article.sentiment = 0.0

        return articles

    def _sort_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Sort articles by relevance and recency"""

        def sort_key(article):
            # Combine relevance score, recency, and sentiment confidence
            recency_score = self._calculate_recency_score(article.published_at)
            relevance = getattr(article, 'relevance_score', 1.0)

            # Boost score for high-confidence sentiment
            sentiment_boost = abs(article.sentiment) * 0.2 if abs(article.sentiment) > 0.5 else 0

            return relevance + recency_score + sentiment_boost

        return sorted(articles, key=sort_key, reverse=True)

    def _calculate_recency_score(self, published_at: datetime) -> float:
        """Calculate recency score (newer = higher score)"""

        now = datetime.now(timezone.utc)
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)

        age_hours = (now - published_at).total_seconds() / 3600

        # Exponential decay: articles lose relevance over time
        if age_hours <= 1:
            return 10.0  # Very recent
        elif age_hours <= 6:
            return 5.0   # Recent
        elif age_hours <= 24:
            return 2.0   # Today
        else:
            return max(0.1, 2.0 / (age_hours / 24))  # Older

    # Specialized fetch methods for different use cases

    def fetch_sports_news(self, players: List[str] = None, limit: int = 20) -> List[NewsArticle]:
        """Fetch sports news, optionally filtered by player names"""

        sources = ['espn']
        categories = ['nfl', 'nba', 'general']

        return self.fetch_news(
            sources=sources,
            categories=categories,
            keywords=players,
            limit=limit,
            time_range_hours=24
        )

    def fetch_financial_news(self, symbols: List[str] = None, include_crypto: bool = True,
                            limit: int = 30) -> List[NewsArticle]:
        """Fetch financial and economic news"""

        sources = ['financial', 'reddit']
        categories = ['market_news', 'economics', 'investing', 'stocks']

        if include_crypto:
            categories.extend(['crypto', 'cryptocurrency'])

        return self.fetch_news(
            sources=sources,
            categories=categories,
            keywords=symbols,
            limit=limit,
            time_range_hours=12
        )

    def fetch_fed_news(self, limit: int = 15) -> List[NewsArticle]:
        """Fetch Federal Reserve and monetary policy news"""

        fed_keywords = ['federal reserve', 'fed', 'fomc', 'jerome powell', 'interest rate', 'monetary policy']

        return self.fetch_news(
            sources=['financial'],
            categories=['economics'],
            keywords=fed_keywords,
            limit=limit,
            time_range_hours=24
        )

    def fetch_market_sentiment(self, symbols: List[str], limit: int = 50) -> Dict[str, Any]:
        """Fetch market sentiment analysis for specific symbols"""

        articles = self.fetch_news(
            sources=['reddit', 'financial'],
            categories=['investing', 'stocks', 'market_news'],
            keywords=symbols,
            limit=limit,
            time_range_hours=24
        )

        if not articles:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'sentiment_distribution': {},
                'trending_symbols': []
            }

        sentiments = [a.sentiment for a in articles if a.sentiment != 0.0]
        overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Calculate sentiment distribution
        positive_count = len([s for s in sentiments if s > 0.1])
        negative_count = len([s for s in sentiments if s < -0.1])
        neutral_count = len(sentiments) - positive_count - negative_count

        return {
            'overall_sentiment': overall_sentiment,
            'confidence': min(len(articles) / 10, 1.0),  # More articles = higher confidence
            'article_count': len(articles),
            'sentiment_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            },
            'recent_articles': [
                {
                    'title': a.title,
                    'source': a.source,
                    'sentiment': a.sentiment,
                    'published': a.published_at.isoformat()
                }
                for a in articles[:10]
            ]
        }

    def _health_check(self):
        """Perform health check on all connectors"""

        current_time = time.time()

        for name, connector in self.connectors.items():
            try:
                status = connector.test_connection()
                self.connector_status[name].update({
                    'connected': status,
                    'last_check': current_time,
                    'error_count': max(0, self.connector_status[name]['error_count'] - 1)  # Decay errors
                })
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                self.connector_status[name]['error_count'] += 1

        self.last_health_check = current_time

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""

        cache_stats = self.cache_manager.get_cache_stats()

        return {
            'connectors': {
                name: {
                    'connected': status['connected'],
                    'error_count': status['error_count'],
                    'last_check_ago': time.time() - status['last_check']
                }
                for name, status in self.connector_status.items()
            },
            'cache': cache_stats,
            'last_health_check_ago': time.time() - self.last_health_check,
            'uptime_hours': (time.time() - self.last_health_check) / 3600
        }

    def shutdown(self):
        """Gracefully shutdown the news aggregator"""

        logger.info("Shutting down NewsAggregator...")

        # Save cache
        self.cache_manager.save_cache()

        # Stop background tasks
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            # Note: Thread will stop on next iteration due to daemon=True
            pass

        logger.info("NewsAggregator shutdown complete")

def main():
    """Test NewsAggregator"""

    logging.basicConfig(level=logging.INFO)

    print("Testing News Aggregator")
    print("=" * 50)

    aggregator = NewsAggregator()

    # Test system status
    status = aggregator.get_system_status()
    print(f"System Status:")
    for connector, data in status['connectors'].items():
        print(f"  {connector}: {'✓' if data['connected'] else '✗'} (errors: {data['error_count']})")

    # Test sports news
    print(f"\\nFetching sports news for Drake Maye...")
    sports_articles = aggregator.fetch_sports_news(['drake maye', 'kenneth walker'], limit=5)

    for i, article in enumerate(sports_articles, 1):
        print(f"\\n{i}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   Sentiment: {article.sentiment:.2f}")
        print(f"   Relevance: {getattr(article, 'relevance_score', 0):.2f}")
        print(f"   Published: {article.published_at}")

    # Test market sentiment
    print(f"\\n\\nTesting market sentiment for AAPL...")
    sentiment_data = aggregator.fetch_market_sentiment(['AAPL', 'apple'], limit=10)

    print(f"Overall Sentiment: {sentiment_data['overall_sentiment']:.2f}")
    print(f"Confidence: {sentiment_data['confidence']:.2f}")
    print(f"Articles Analyzed: {sentiment_data['article_count']}")
    print(f"Distribution: {sentiment_data['sentiment_distribution']}")

    # Test Fed news
    print(f"\\n\\nTesting Fed news...")
    fed_articles = aggregator.fetch_fed_news(limit=3)

    for i, article in enumerate(fed_articles, 1):
        print(f"\\n{i}. {article.title}")
        print(f"   Sentiment: {article.sentiment:.2f}")

    aggregator.shutdown()

if __name__ == "__main__":
    main()