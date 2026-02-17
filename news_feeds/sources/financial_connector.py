"""
Financial News Connector - Economic and market news from multiple sources

Aggregates financial news from Yahoo Finance, MarketWatch, and other free sources.
Focuses on economic indicators, market news, and crypto updates.
"""

import json
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

from .base_connector import BaseNewsConnector, NewsArticle
import logging

logger = logging.getLogger(__name__)

class FinancialNewsConnector(BaseNewsConnector):
    """Financial and economic news connector"""

    def __init__(self, api_key: str = "", rate_limit: int = 120, cache_ttl: int = 300):
        super().__init__(api_key, rate_limit, cache_ttl)

        # Free financial news sources (updated Feb 2026)
        self.news_sources = {
            # MarketWatch RSS feeds (working)
            'market_news': 'https://feeds.marketwatch.com/marketwatch/topstories',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines',
            # CNBC RSS feeds
            'cnbc_markets': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069',
            'cnbc_economy': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258',
            'crypto': 'https://news.google.com/rss/search?q=crypto+bitcoin+ethereum&hl=en-US&gl=US&ceid=US:en',

            # Investing.com RSS (working)
            'economics': 'https://www.investing.com/rss/news.rss',

            # Google News Finance RSS (working fallback)
            'google_finance': 'https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en',
            'google_crypto': 'https://news.google.com/rss/search?q=cryptocurrency+bitcoin&hl=en-US&gl=US&ceid=US:en',
            'google_fed': 'https://news.google.com/rss/search?q=federal+reserve+interest+rates&hl=en-US&gl=US&ceid=US:en'
        }

        # Alternative API endpoints (free tiers)
        self.api_endpoints = {
            # Alpha Vantage (free tier: 5 calls/minute, 500 calls/day)
            'alpha_vantage_news': 'https://www.alphavantage.co/query',

            # NewsAPI (free tier: 1000 calls/day)
            'newsapi': 'https://newsapi.org/v2/everything',

            # FRED Economic Data (free)
            'fred_releases': 'https://api.stlouisfed.org/fred/releases'
        }

        # Economic indicators to track
        self.economic_keywords = [
            'federal reserve', 'fed meeting', 'interest rate', 'inflation',
            'gdp', 'employment', 'unemployment', 'jobs report', 'cpi',
            'ppi', 'fomc', 'jerome powell', 'monetary policy'
        ]

        # Market-related keywords
        self.market_keywords = [
            'stock market', 'nasdaq', 'dow jones', 's&p 500', 'volatility',
            'earnings', 'merger', 'acquisition', 'ipo', 'market crash',
            'bull market', 'bear market', 'correction'
        ]

        # Crypto keywords
        self.crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'cryptocurrency', 'crypto',
            'blockchain', 'defi', 'nft', 'altcoin', 'coinbase', 'binance'
        ]

    @property
    def source_name(self) -> str:
        return "Financial_News"

    @property
    def categories(self) -> List[str]:
        return ['market_news', 'crypto', 'economics', 'reuters_business',
                'marketwatch', 'general', 'fomc', 'earnings']

    def fetch_news(self, category: str = "market_news", limit: int = 20,
                   keywords: List[str] = None) -> List[NewsArticle]:
        """Fetch financial news articles"""

        articles = []

        # Try multiple sources for better coverage
        if category == 'general' or category == 'market_news':
            sources_to_try = ['market_news', 'marketwatch', 'reuters_business']
        elif category == 'crypto':
            sources_to_try = ['crypto', 'marketwatch_crypto']
        elif category == 'economics':
            sources_to_try = ['economics', 'reuters_markets']
        elif category == 'fomc':
            # Specific FOMC/Fed news
            fed_keywords = ['federal reserve', 'fed meeting', 'fomc', 'jerome powell']
            return self.fetch_fed_news(limit, fed_keywords)
        else:
            sources_to_try = [category] if category in self.news_sources else ['market_news']

        for source in sources_to_try:
            try:
                source_articles = self._fetch_from_rss(source, limit // len(sources_to_try) + 5)
                articles.extend(source_articles)

                if len(articles) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Failed to fetch from {source}: {e}")
                continue

        # Filter by keywords if provided
        if keywords:
            articles = self._filter_by_keywords(articles, keywords)

        # Sort by publication date (newest first)
        articles.sort(key=lambda x: x.published_at, reverse=True)

        return articles[:limit]

    def _fetch_from_rss(self, source: str, limit: int) -> List[NewsArticle]:
        """Fetch articles from RSS feed"""

        articles = []

        if source not in self.news_sources:
            return articles

        try:
            response = self._make_request(self.news_sources[source])
            if not response:
                return articles

            # Parse RSS XML
            root = ET.fromstring(response.content)

            # Handle different RSS formats
            items = root.findall('.//item')
            if not items:  # Try alternative format
                items = root.findall('.//entry')

            for item in items[:limit]:
                try:
                    article = self._parse_rss_item(item, source)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse RSS item: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to fetch RSS from {source}: {e}")

        return articles

    def _parse_rss_item(self, item, source: str) -> Optional[NewsArticle]:
        """Parse RSS item into NewsArticle"""

        try:
            # Try different XML structures
            title = ""
            content = ""
            url = ""
            pub_date = datetime.now(timezone.utc)

            # Standard RSS format
            if item.find('title') is not None:
                title = item.find('title').text or ""

            if item.find('description') is not None:
                content = item.find('description').text or ""

            if item.find('link') is not None:
                url = item.find('link').text or ""

            # Alternative Atom format
            if not title and item.find('{http://www.w3.org/2005/Atom}title') is not None:
                title = item.find('{http://www.w3.org/2005/Atom}title').text or ""

            if not content and item.find('{http://www.w3.org/2005/Atom}content') is not None:
                content = item.find('{http://www.w3.org/2005/Atom}content').text or ""

            # Parse publication date
            pub_date_elem = item.find('pubDate')
            if pub_date_elem is None:
                pub_date_elem = item.find('published')
            if pub_date_elem is None:
                pub_date_elem = item.find('{http://www.w3.org/2005/Atom}published')
            if pub_date_elem is not None:
                try:
                    date_str = pub_date_elem.text
                    # Try multiple date formats
                    for fmt in ['%a, %d %b %Y %H:%M:%S %Z', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%SZ']:
                        try:
                            pub_date = datetime.strptime(date_str, fmt)
                            if pub_date.tzinfo is None:
                                pub_date = pub_date.replace(tzinfo=timezone.utc)
                            break
                        except ValueError:
                            continue
                except Exception:
                    pub_date = datetime.now(timezone.utc)

            # Clean content (remove HTML tags)
            content = re.sub(r'<[^>]+>', '', content)

            # Determine category based on content and source
            category = self._categorize_article(title, content, source)

            # Calculate basic sentiment
            sentiment = self._calculate_basic_sentiment(title, content)

            return NewsArticle(
                title=title,
                content=content,
                source=f"Financial_News_{source}",
                published_at=pub_date,
                url=url,
                category=category,
                sentiment=sentiment,
                tags=[category, source.replace('_', ' ')]
            )

        except Exception as e:
            logger.error(f"Failed to parse financial RSS item: {e}")
            return None

    def _categorize_article(self, title: str, content: str, source: str) -> str:
        """Categorize article based on content"""

        text = f"{title} {content}".lower()

        # Check for economic indicators
        if any(keyword in text for keyword in self.economic_keywords):
            return "economics"

        # Check for crypto
        if any(keyword in text for keyword in self.crypto_keywords):
            return "crypto"

        # Check for market news
        if any(keyword in text for keyword in self.market_keywords):
            return "markets"

        # Default based on source
        if 'crypto' in source:
            return "crypto"
        elif 'economics' in source or 'economy' in source:
            return "economics"
        else:
            return "markets"

    def _calculate_basic_sentiment(self, title: str, content: str) -> float:
        """Calculate basic sentiment score (-1 to 1)"""

        text = f"{title} {content}".lower()

        # Positive words
        positive_words = [
            'gains', 'up', 'rise', 'surge', 'rally', 'bullish', 'growth',
            'increase', 'positive', 'strong', 'boost', 'optimistic', 'profit'
        ]

        # Negative words
        negative_words = [
            'falls', 'drop', 'decline', 'crash', 'bearish', 'loss', 'down',
            'decrease', 'negative', 'weak', 'concern', 'pessimistic', 'fear'
        ]

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        total_words = len(text.split())
        if total_words == 0:
            return 0.0

        # Calculate sentiment score
        sentiment = (positive_count - negative_count) / (total_words / 100)
        return max(-1.0, min(1.0, sentiment))

    def _filter_by_keywords(self, articles: List[NewsArticle], keywords: List[str]) -> List[NewsArticle]:
        """Filter articles by keywords and calculate relevance"""

        filtered = []
        keywords_lower = [kw.lower() for kw in keywords]

        for article in articles:
            text = f"{article.title} {article.content}".lower()

            if any(keyword in text for keyword in keywords_lower):
                # Calculate relevance score
                relevance = 0.0
                for keyword in keywords_lower:
                    title_matches = article.title.lower().count(keyword)
                    content_matches = article.content.lower().count(keyword)
                    # Title matches are worth more
                    relevance += (title_matches * 3) + content_matches

                article.relevance_score = relevance
                filtered.append(article)

        # Sort by relevance
        filtered.sort(key=lambda x: x.relevance_score, reverse=True)
        return filtered

    def fetch_fed_news(self, limit: int = 10, fed_keywords: List[str] = None) -> List[NewsArticle]:
        """Fetch Federal Reserve and FOMC-specific news"""

        if not fed_keywords:
            fed_keywords = ['federal reserve', 'fed', 'fomc', 'jerome powell', 'interest rate']

        all_articles = []

        # Fetch from multiple sources
        for source in ['market_news', 'reuters_business', 'economics']:
            try:
                articles = self._fetch_from_rss(source, limit * 2)
                fed_articles = self._filter_by_keywords(articles, fed_keywords)
                all_articles.extend(fed_articles)
            except Exception as e:
                logger.warning(f"Failed to fetch Fed news from {source}: {e}")

        # Remove duplicates and sort
        seen_titles = set()
        unique_articles = []

        for article in all_articles:
            title_key = article.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)

        unique_articles.sort(key=lambda x: (x.relevance_score, x.published_at), reverse=True)

        return unique_articles[:limit]

    def fetch_earnings_news(self, symbols: List[str] = None, limit: int = 10) -> List[NewsArticle]:
        """Fetch earnings-related news"""

        earnings_keywords = ['earnings', 'quarterly results', 'eps', 'revenue', 'guidance']

        if symbols:
            # Add stock symbols to keywords
            earnings_keywords.extend([f"${symbol}" for symbol in symbols])
            earnings_keywords.extend(symbols)

        articles = self.fetch_news(category='market_news', limit=limit*3, keywords=earnings_keywords)

        return articles[:limit]

def main():
    """Test financial news connector"""

    logging.basicConfig(level=logging.INFO)

    print("Testing Financial News Connector")
    print("=" * 50)

    connector = FinancialNewsConnector()

    # Test connection
    print(f"Connection test: {connector.test_connection()}")

    # Test market news
    print(f"\nFetching market news...")
    articles = connector.fetch_news(category='market_news', limit=5)

    for i, article in enumerate(articles, 1):
        print(f"\n{i}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   Category: {article.category}")
        print(f"   Sentiment: {article.sentiment:.2f}")
        print(f"   Published: {article.published_at}")
        print(f"   Content: {article.content[:100]}...")

    # Test Fed news
    print(f"\n\nTesting Fed-specific news...")
    fed_articles = connector.fetch_fed_news(limit=3)

    for i, article in enumerate(fed_articles, 1):
        print(f"\n{i}. {article.title}")
        print(f"   Relevance: {article.relevance_score:.2f}")
        print(f"   Sentiment: {article.sentiment:.2f}")

    # Test crypto news
    print(f"\n\nTesting crypto news...")
    crypto_articles = connector.fetch_news(category='crypto', limit=3)

    for i, article in enumerate(crypto_articles, 1):
        print(f"\n{i}. {article.title}")
        print(f"   Category: {article.category}")
        print(f"   Tags: {', '.join(article.tags)}")

if __name__ == "__main__":
    main()