"""
ESPN News Connector - Real-time sports news from ESPN API

Uses ESPN's public API endpoints for sports news and updates.
Note: ESPN's official API is limited, this uses their public RSS feeds and web scraping.
"""

import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

from .base_connector import BaseNewsConnector, NewsArticle
import logging

logger = logging.getLogger(__name__)

class ESPNConnector(BaseNewsConnector):
    """ESPN sports news connector"""

    def __init__(self, api_key: str = "", rate_limit: int = 60, cache_ttl: int = 600):
        super().__init__(api_key, rate_limit, cache_ttl)
        self.base_url = "https://www.espn.com"

        # ESPN RSS feeds for different sports
        self.rss_feeds = {
            'nfl': 'https://www.espn.com/espn/rss/nfl/news',
            'nba': 'https://www.espn.com/espn/rss/nba/news',
            'mlb': 'https://www.espn.com/espn/rss/mlb/news',
            'soccer': 'https://www.espn.com/espn/rss/soccer/news',
            'general': 'https://www.espn.com/espn/rss/news',
            'fantasy': 'https://www.espn.com/espn/rss/fantasy/news'
        }

        # ESPN unofficial API endpoints
        self.api_endpoints = {
            'nfl_scores': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
            'nba_scores': 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',
            'nfl_news': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/news',
            'nba_news': 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news'
        }

    @property
    def source_name(self) -> str:
        return "ESPN"

    @property
    def categories(self) -> List[str]:
        return list(self.rss_feeds.keys())

    def fetch_news(self, category: str = "general", limit: int = 20,
                   keywords: List[str] = None) -> List[NewsArticle]:
        """Fetch ESPN news articles"""

        articles = []

        # Try API endpoint first (more structured data)
        if category in ['nfl', 'nba']:
            api_articles = self._fetch_from_api(category, limit)
            articles.extend(api_articles)

        # Fall back to RSS feed if API didn't provide enough articles
        if len(articles) < limit:
            rss_articles = self._fetch_from_rss(category, limit - len(articles))
            articles.extend(rss_articles)

        # Filter by keywords if provided
        if keywords:
            articles = self._filter_by_keywords(articles, keywords)

        return articles[:limit]

    def _fetch_from_api(self, category: str, limit: int) -> List[NewsArticle]:
        """Fetch from ESPN's unofficial API endpoints"""

        articles = []
        api_key = f"{category}_news"

        if api_key not in self.api_endpoints:
            return articles

        try:
            response = self._make_request(self.api_endpoints[api_key])
            if not response:
                return articles

            data = response.json()

            # Parse ESPN API response
            for item in data.get('articles', [])[:limit]:
                try:
                    article = self._parse_api_article(item, category)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse API article: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to fetch from ESPN API: {e}")

        return articles

    def _fetch_from_rss(self, category: str, limit: int) -> List[NewsArticle]:
        """Fetch from ESPN RSS feeds"""

        articles = []

        if category not in self.rss_feeds:
            category = 'general'

        try:
            response = self._make_request(self.rss_feeds[category])
            if not response:
                return articles

            # Parse RSS XML
            root = ET.fromstring(response.content)

            for item in root.findall('.//item')[:limit]:
                try:
                    article = self._parse_rss_article(item, category)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse RSS article: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to fetch ESPN RSS for {category}: {e}")

        return articles

    def _parse_api_article(self, item: Dict, category: str) -> Optional[NewsArticle]:
        """Parse article from ESPN API response"""

        try:
            title = item.get('headline', '')
            content = item.get('description', '') or item.get('story', '')

            # Parse publication date
            pub_date_str = item.get('published')
            if pub_date_str:
                pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
            else:
                pub_date = datetime.now(timezone.utc)

            # Extract URL
            url = ""
            links = item.get('links', {})
            if 'web' in links:
                url = links['web'].get('href', '')

            # Extract tags from categories
            tags = []
            categories = item.get('categories', [])
            for cat in categories:
                if isinstance(cat, dict) and 'description' in cat:
                    tags.append(cat['description'].lower())

            return NewsArticle(
                title=title,
                content=content,
                source=self.source_name,
                published_at=pub_date,
                url=url,
                category=category,
                tags=tags
            )

        except Exception as e:
            logger.error(f"Failed to parse ESPN API article: {e}")
            return None

    def _parse_rss_article(self, item, category: str) -> Optional[NewsArticle]:
        """Parse article from RSS feed"""

        try:
            title = item.find('title').text if item.find('title') is not None else ""
            content = item.find('description').text if item.find('description') is not None else ""
            url = item.find('link').text if item.find('link') is not None else ""

            # Parse publication date
            pub_date_elem = item.find('pubDate')
            if pub_date_elem is not None:
                try:
                    # RSS date format: "Mon, 01 Feb 2026 10:30:00 GMT"
                    pub_date = datetime.strptime(pub_date_elem.text, '%a, %d %b %Y %H:%M:%S %Z')
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
                except Exception as e:
                    logger.debug(f"Error parsing pub date: {e}")
                    pub_date = datetime.now(timezone.utc)
            else:
                pub_date = datetime.now(timezone.utc)

            # Clean content (remove HTML tags if present)
            content = re.sub(r'<[^>]+>', '', content)

            return NewsArticle(
                title=title,
                content=content,
                source=self.source_name,
                published_at=pub_date,
                url=url,
                category=category,
                tags=[category]
            )

        except Exception as e:
            logger.error(f"Failed to parse ESPN RSS article: {e}")
            return None

    def _filter_by_keywords(self, articles: List[NewsArticle], keywords: List[str]) -> List[NewsArticle]:
        """Filter articles by keywords"""

        filtered = []
        keywords_lower = [kw.lower() for kw in keywords]

        for article in articles:
            text = f"{article.title} {article.content}".lower()

            # Check if any keyword appears in the article
            if any(keyword in text for keyword in keywords_lower):
                article.relevance_score = self._calculate_relevance_score(text, keywords_lower)
                filtered.append(article)

        # Sort by relevance score (highest first)
        filtered.sort(key=lambda x: x.relevance_score, reverse=True)

        return filtered

    def _calculate_relevance_score(self, text: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword frequency"""

        total_score = 0.0
        text_words = text.split()

        for keyword in keywords:
            # Count exact matches
            exact_matches = text.lower().count(keyword.lower())

            # Count word matches
            word_matches = sum(1 for word in text_words if keyword.lower() in word.lower())

            # Score: exact matches worth 2 points, word matches worth 1 point
            keyword_score = (exact_matches * 2) + word_matches
            total_score += keyword_score

        # Normalize by text length (prefer shorter, more focused articles)
        if len(text_words) > 0:
            total_score = total_score / (len(text_words) / 100)  # Per 100 words

        return total_score

    def fetch_player_news(self, player_names: List[str], limit: int = 10) -> List[NewsArticle]:
        """Fetch news specifically mentioning certain players"""

        all_articles = []

        # Fetch from multiple sports categories
        for category in ['nfl', 'nba', 'general']:
            try:
                articles = self.fetch_news(category=category, limit=limit*2, keywords=player_names)
                all_articles.extend(articles)
            except Exception as e:
                logger.warning(f"Failed to fetch {category} news: {e}")

        # Remove duplicates and sort by relevance
        seen_ids = set()
        unique_articles = []

        for article in all_articles:
            if article.id not in seen_ids:
                seen_ids.add(article.id)
                unique_articles.append(article)

        # Sort by relevance score and published date
        unique_articles.sort(key=lambda x: (x.relevance_score, x.published_at), reverse=True)

        return unique_articles[:limit]

def main():
    """Test ESPN connector"""

    logging.basicConfig(level=logging.INFO)

    print("Testing ESPN News Connector")
    print("=" * 50)

    connector = ESPNConnector()

    # Test connection
    print(f"Connection test: {connector.test_connection()}")

    # Test general news
    print(f"\nFetching general sports news...")
    articles = connector.fetch_news(category='nfl', limit=5)

    for i, article in enumerate(articles, 1):
        print(f"\n{i}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   Category: {article.category}")
        print(f"   Published: {article.published_at}")
        print(f"   Content: {article.content[:100]}...")
        if article.url:
            print(f"   URL: {article.url}")

    # Test player-specific news
    print(f"\n\nTesting player-specific news...")
    player_articles = connector.fetch_player_news(['drake maye', 'kenneth walker'], limit=3)

    for i, article in enumerate(player_articles, 1):
        print(f"\n{i}. {article.title}")
        print(f"   Relevance: {article.relevance_score:.2f}")
        print(f"   Tags: {', '.join(article.tags)}")

if __name__ == "__main__":
    main()