"""
Reddit News Connector - Market sentiment and discussions from Reddit

Uses Reddit's public API and RSS feeds to gather market sentiment from
relevant subreddits like r/investing, r/stocks, r/cryptocurrency, etc.
"""

import json
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, quote

from .base_connector import BaseNewsConnector, NewsArticle
import logging

logger = logging.getLogger(__name__)

class RedditConnector(BaseNewsConnector):
    """Reddit sentiment and discussions connector"""

    def __init__(self, api_key: str = "", rate_limit: int = 30, cache_ttl: int = 600):
        super().__init__(api_key, rate_limit, cache_ttl)

        # Reddit doesn't require API key for public data
        self.base_url = "https://www.reddit.com"

        # Relevant subreddits for trading/market sentiment
        self.subreddits = {
            'investing': 'r/investing',
            'stocks': 'r/stocks',
            'wallstreetbets': 'r/wallstreetbets',
            'cryptocurrency': 'r/cryptocurrency',
            'bitcoinmarkets': 'r/bitcoinmarkets',
            'economics': 'r/economics',
            'securityanalysis': 'r/securityanalysis',
            'value_investing': 'r/valueinvesting',
            'options': 'r/options',
            'forex': 'r/forex',
            'financialindependence': 'r/financialindependence',
            'nfl': 'r/nfl',
            'nba': 'r/nba',
            'sportsbook': 'r/sportsbook',
            'sportsbetting': 'r/sportsbetting'
        }

        # Reddit JSON API endpoints (no auth needed for public data)
        self.json_endpoints = {
            'hot': '.json',
            'new': '/new.json',
            'top': '/top.json',
            'rising': '/rising.json'
        }

        # Sentiment keywords
        self.bullish_keywords = [
            'bullish', 'buy', 'long', 'calls', 'moon', 'pump', 'gains',
            'rocket', 'diamond hands', 'hodl', 'to the moon', 'bull run',
            'buying opportunity', 'undervalued', 'breakout'
        ]

        self.bearish_keywords = [
            'bearish', 'sell', 'short', 'puts', 'dump', 'crash', 'loss',
            'falling knife', 'bag holder', 'bubble', 'overvalued', 'dump',
            'bear market', 'correction', 'selloff'
        ]

    @property
    def source_name(self) -> str:
        return "Reddit"

    @property
    def categories(self) -> List[str]:
        return list(self.subreddits.keys())

    def fetch_news(self, category: str = "investing", limit: int = 20,
                   keywords: List[str] = None) -> List[NewsArticle]:
        """Fetch Reddit posts as news articles"""

        articles = []

        if category not in self.subreddits:
            category = 'investing'

        subreddit = self.subreddits[category]

        # Fetch from different sorting methods
        for sort_method in ['hot', 'new', 'top']:
            try:
                posts = self._fetch_reddit_posts(subreddit, sort_method, limit // 3 + 5)
                articles.extend(posts)

                if len(articles) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Failed to fetch {sort_method} posts from {subreddit}: {e}")
                continue

        # Filter by keywords if provided
        if keywords:
            articles = self._filter_by_keywords(articles, keywords)

        # Remove duplicates
        seen_ids = set()
        unique_articles = []

        for article in articles:
            if article.id not in seen_ids:
                seen_ids.add(article.id)
                unique_articles.append(article)

        # Sort by score and recency
        unique_articles.sort(key=lambda x: (x.relevance_score, x.published_at), reverse=True)

        return unique_articles[:limit]

    def _fetch_reddit_posts(self, subreddit: str, sort_method: str, limit: int) -> List[NewsArticle]:
        """Fetch posts from a specific subreddit"""

        articles = []

        try:
            # Build URL for Reddit JSON API
            url = f"{self.base_url}/{subreddit}"

            if sort_method in self.json_endpoints:
                url += self.json_endpoints[sort_method]

            url += f"?limit={min(limit, 25)}"  # Reddit limit is 25 per request

            # Make request
            response = self._make_request(url)
            if not response:
                return articles

            data = response.json()

            # Parse Reddit JSON response
            if 'data' in data and 'children' in data['data']:
                for post_data in data['data']['children']:
                    try:
                        if post_data['kind'] == 't3':  # Link/text post
                            article = self._parse_reddit_post(post_data['data'], subreddit)
                            if article:
                                articles.append(article)
                    except Exception as e:
                        logger.warning(f"Failed to parse Reddit post: {e}")
                        continue

        except Exception as e:
            logger.error(f"Failed to fetch Reddit posts from {subreddit}: {e}")

        return articles

    def _parse_reddit_post(self, post_data: Dict, subreddit: str) -> Optional[NewsArticle]:
        """Parse Reddit post into NewsArticle format"""

        try:
            title = post_data.get('title', '')
            selftext = post_data.get('selftext', '')

            # Use selftext as content, or title if no selftext
            content = selftext if selftext else title

            # Clean markdown formatting
            content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Remove bold
            content = re.sub(r'\*(.*?)\*', r'\1', content)      # Remove italic
            content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)  # Remove links

            # Create URL to the Reddit post
            permalink = post_data.get('permalink', '')
            url = f"https://www.reddit.com{permalink}" if permalink else ""

            # Parse creation time
            created_utc = post_data.get('created_utc', 0)
            pub_date = datetime.fromtimestamp(created_utc, tz=timezone.utc) if created_utc else datetime.now(timezone.utc)

            # Extract metrics for relevance scoring
            score = post_data.get('score', 0)
            num_comments = post_data.get('num_comments', 0)
            upvote_ratio = post_data.get('upvote_ratio', 0.5)

            # Calculate sentiment
            sentiment = self._calculate_reddit_sentiment(title, content)

            # Calculate relevance score based on Reddit metrics
            relevance = self._calculate_reddit_relevance(score, num_comments, upvote_ratio)

            # Extract flair as a tag
            flair = post_data.get('link_flair_text', '')
            tags = [subreddit.replace('r/', '')]
            if flair:
                tags.append(flair.lower())

            # Determine category
            category = self._categorize_reddit_post(title, content, subreddit)

            return NewsArticle(
                title=title,
                content=content,
                source=f"Reddit_{subreddit}",
                published_at=pub_date,
                url=url,
                category=category,
                sentiment=sentiment,
                relevance_score=relevance,
                tags=tags
            )

        except Exception as e:
            logger.error(f"Failed to parse Reddit post: {e}")
            return None

    def _calculate_reddit_sentiment(self, title: str, content: str) -> float:
        """Calculate sentiment based on Reddit post content"""

        text = f"{title} {content}".lower()

        # Count sentiment keywords
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text)

        # Calculate sentiment score (-1 to 1)
        total_sentiment_words = bullish_count + bearish_count

        if total_sentiment_words == 0:
            return 0.0

        sentiment = (bullish_count - bearish_count) / total_sentiment_words

        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, sentiment))

    def _calculate_reddit_relevance(self, score: int, num_comments: int, upvote_ratio: float) -> float:
        """Calculate relevance based on Reddit engagement metrics"""

        # Base score from upvotes
        base_score = max(0, score)

        # Bonus for high engagement (comments)
        comment_bonus = min(num_comments * 0.1, 10)

        # Bonus for high upvote ratio
        ratio_bonus = upvote_ratio * 5

        # Combine scores
        total_relevance = base_score + comment_bonus + ratio_bonus

        # Log scale to prevent extremely high scores
        import math
        return math.log(total_relevance + 1)

    def _categorize_reddit_post(self, title: str, content: str, subreddit: str) -> str:
        """Categorize Reddit post based on subreddit and content"""

        text = f"{title} {content}".lower()

        # Sports subreddits
        if subreddit in ['r/nfl', 'r/nba', 'r/sportsbook', 'r/sportsbetting']:
            return 'sports'

        # Crypto subreddits
        elif subreddit in ['r/cryptocurrency', 'r/bitcoinmarkets']:
            return 'crypto'

        # Financial subreddits - look for specific topics
        elif 'earnings' in text or 'eps' in text:
            return 'earnings'

        elif any(word in text for word in ['fed', 'federal reserve', 'interest rate', 'inflation']):
            return 'economics'

        elif any(word in text for word in ['options', 'calls', 'puts', 'derivatives']):
            return 'options'

        else:
            return 'general'

    def _filter_by_keywords(self, articles: List[NewsArticle], keywords: List[str]) -> List[NewsArticle]:
        """Filter and score articles by keywords"""

        filtered = []
        keywords_lower = [kw.lower() for kw in keywords]

        for article in articles:
            text = f"{article.title} {article.content}".lower()

            # Check if any keyword appears
            keyword_matches = 0
            for keyword in keywords_lower:
                if keyword in text:
                    keyword_matches += text.count(keyword)

            if keyword_matches > 0:
                # Boost relevance score based on keyword matches
                article.relevance_score += keyword_matches * 2
                filtered.append(article)

        return filtered

    def fetch_sentiment_summary(self, symbols: List[str], limit: int = 50) -> Dict[str, Any]:
        """Fetch Reddit sentiment summary for specific stock symbols"""

        all_articles = []

        # Search relevant subreddits
        for category in ['stocks', 'investing', 'wallstreetbets', 'securityanalysis']:
            try:
                articles = self.fetch_news(category=category, limit=limit//4, keywords=symbols)
                all_articles.extend(articles)
            except Exception as e:
                logger.warning(f"Failed to fetch sentiment from {category}: {e}")

        if not all_articles:
            return {'sentiment': 0.0, 'confidence': 0.0, 'mentions': 0}

        # Calculate overall sentiment
        total_sentiment = sum(article.sentiment for article in all_articles)
        avg_sentiment = total_sentiment / len(all_articles)

        # Calculate confidence based on number of mentions and engagement
        total_relevance = sum(article.relevance_score for article in all_articles)
        confidence = min(total_relevance / 100, 1.0)  # Cap at 1.0

        return {
            'sentiment': avg_sentiment,
            'confidence': confidence,
            'mentions': len(all_articles),
            'top_posts': [
                {
                    'title': article.title,
                    'sentiment': article.sentiment,
                    'score': article.relevance_score,
                    'url': article.url
                }
                for article in sorted(all_articles, key=lambda x: x.relevance_score, reverse=True)[:5]
            ]
        }

    def fetch_trending_topics(self, category: str = "wallstreetbets", limit: int = 10) -> List[str]:
        """Extract trending ticker symbols/topics from Reddit"""

        articles = self.fetch_news(category=category, limit=limit*2)

        # Extract potential ticker symbols (3-4 letter combinations in caps)
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        ticker_counts = {}

        for article in articles:
            text = f"{article.title} {article.content}"
            matches = re.findall(ticker_pattern, text)

            for match in matches:
                # Filter out common non-ticker words
                if match not in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'OUT', 'NEW']:
                    ticker_counts[match] = ticker_counts.get(match, 0) + 1

        # Sort by frequency
        trending = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)

        return [ticker for ticker, count in trending[:limit] if count >= 2]

def main():
    """Test Reddit connector"""

    logging.basicConfig(level=logging.INFO)

    print("Testing Reddit News Connector")
    print("=" * 50)

    connector = RedditConnector()

    # Test connection
    print(f"Connection test: {connector.test_connection()}")

    # Test general investing news
    print(f"\nFetching investing discussions...")
    articles = connector.fetch_news(category='investing', limit=5)

    for i, article in enumerate(articles, 1):
        print(f"\n{i}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   Sentiment: {article.sentiment:.2f}")
        print(f"   Relevance: {article.relevance_score:.2f}")
        print(f"   Published: {article.published_at}")
        print(f"   Content: {article.content[:100]}...")

    # Test sentiment analysis
    print(f"\n\nTesting sentiment for AAPL...")
    sentiment_data = connector.fetch_sentiment_summary(['AAPL', 'apple'], limit=20)

    print(f"Sentiment: {sentiment_data['sentiment']:.2f}")
    print(f"Confidence: {sentiment_data['confidence']:.2f}")
    print(f"Mentions: {sentiment_data['mentions']}")

    # Test trending topics
    print(f"\n\nTrending topics on WallStreetBets...")
    trending = connector.fetch_trending_topics('wallstreetbets', limit=10)
    print(f"Trending tickers: {', '.join(trending)}")

if __name__ == "__main__":
    main()