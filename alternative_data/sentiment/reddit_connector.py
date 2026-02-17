"""
Reddit Connector
================

Reddit sentiment analysis focusing on financial subreddits
and trading communities.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json

try:
    import praw  # Python Reddit API Wrapper
    import asyncpraw  # Async version
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from ..core.base_connector import BaseDataConnector, DataPoint, DataSource, ConnectorConfig
from .sentiment_analyzer import SentimentAnalyzer, SentimentScore


class RedditConnector(BaseDataConnector):
    """
    Reddit sentiment analysis connector

    Features:
    - Subreddit monitoring
    - Hot/rising post tracking
    - Comment sentiment analysis
    - Karma-weighted scoring
    - WSB-style meme detection
    """

    def __init__(self, config: ConnectorConfig):
        super().__init__(DataSource.REDDIT, config)

        self.sentiment_analyzer = SentimentAnalyzer()

        # Reddit API credentials
        self.client_id = config.api_key
        self.client_secret = config.custom_params.get('client_secret')
        self.user_agent = config.custom_params.get('user_agent', 'TradingBot/1.0')
        self.username = config.custom_params.get('username')
        self.password = config.custom_params.get('password')

        # Reddit client
        self.reddit = None

        # Target subreddits
        self.subreddits = config.custom_params.get('subreddits', [
            'wallstreetbets', 'investing', 'stocks', 'SecurityAnalysis',
            'ValueInvesting', 'CryptoCurrency', 'Bitcoin', 'ethereum',
            'pennystocks', 'options', 'StockMarket', 'financialindependence'
        ])

        # Quality filters
        self.min_upvotes = config.custom_params.get('min_upvotes', 10)
        self.min_comments = config.custom_params.get('min_comments', 3)
        self.min_karma = config.custom_params.get('min_karma', 100)

        self._initialize_reddit_api()

    def _initialize_reddit_api(self) -> None:
        """Initialize Reddit API client"""
        if not REDDIT_AVAILABLE:
            self.logger.error("PRAW not available. Install with: pip install praw asyncpraw")
            return

        try:
            self.reddit = asyncpraw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                username=self.username,
                password=self.password,
                check_for_async=False
            )

            self.logger.info("Reddit API initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Reddit API: {e}")
            self.reddit = None

    async def fetch_data(
        self,
        symbols: List[str],
        lookback_hours: int = 24,
        **kwargs
    ) -> List[DataPoint]:
        """
        Fetch Reddit data for symbols

        Args:
            symbols: List of symbols to search for
            lookback_hours: Hours of historical data to fetch
            **kwargs: Additional parameters

        Returns:
            List of DataPoint objects with Reddit sentiment data
        """
        if not self.reddit:
            self.logger.error("Reddit API not available")
            return []

        all_data_points = []

        # Search each subreddit for symbol mentions
        for subreddit_name in self.subreddits:
            try:
                subreddit = await self.reddit.subreddit(subreddit_name)

                # Get posts for symbols
                for symbol in symbols:
                    posts = await self._search_subreddit_posts(subreddit, symbol, lookback_hours)

                    for post_data in posts:
                        data_point = await self._process_reddit_post(post_data, symbol, subreddit_name)
                        if data_point:
                            all_data_points.append(data_point)

                    # Rate limiting
                    await asyncio.sleep(0.5)

            except Exception as e:
                self.logger.error(f"Failed to fetch from r/{subreddit_name}: {e}")

        self.logger.info(f"Fetched {len(all_data_points)} Reddit data points")
        return all_data_points

    async def get_real_time_data(self, symbols: List[str]) -> List[DataPoint]:
        """Get real-time Reddit data (last 30 minutes)"""
        return await self.fetch_data(symbols, lookback_hours=0.5)

    async def _search_subreddit_posts(
        self,
        subreddit,
        symbol: str,
        lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Search for posts mentioning a symbol in a subreddit"""
        posts = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)

        try:
            # Search recent posts
            search_queries = self._create_reddit_search_queries(symbol)

            for query in search_queries:
                try:
                    async for submission in subreddit.search(
                        query,
                        sort='new',
                        time_filter='day' if lookback_hours >= 12 else 'hour',
                        limit=50
                    ):
                        # Check if post is within time window
                        post_time = datetime.fromtimestamp(submission.created_utc)
                        if post_time < cutoff_time:
                            continue

                        # Check quality filters
                        if not self._is_quality_post(submission):
                            continue

                        # Get post data
                        post_data = await self._extract_post_data(submission)
                        if post_data:
                            posts.append(post_data)

                except Exception as e:
                    self.logger.warning(f"Search failed for query '{query}': {e}")

            # Also check hot posts for symbol mentions
            async for submission in subreddit.hot(limit=100):
                post_time = datetime.fromtimestamp(submission.created_utc)
                if post_time < cutoff_time:
                    continue

                # Check if symbol mentioned in title or text
                title_text = f"{submission.title} {getattr(submission, 'selftext', '')}".lower()
                if symbol.lower() in title_text or f'${symbol.lower()}' in title_text:
                    if self._is_quality_post(submission):
                        post_data = await self._extract_post_data(submission)
                        if post_data:
                            posts.append(post_data)

        except Exception as e:
            self.logger.error(f"Failed to search subreddit: {e}")

        return posts

    def _create_reddit_search_queries(self, symbol: str) -> List[str]:
        """Create Reddit search queries for a symbol"""
        queries = []

        # Direct symbol searches
        queries.append(f"${symbol}")
        queries.append(f"{symbol} stock")
        queries.append(f"{symbol} calls")
        queries.append(f"{symbol} puts")

        # Crypto-specific
        if len(symbol) <= 4:
            queries.append(f"{symbol} crypto")
            queries.append(f"{symbol} coin")

        return queries

    def _is_quality_post(self, submission) -> bool:
        """Check if post meets quality criteria"""
        try:
            # Minimum engagement
            if submission.score < self.min_upvotes:
                return False

            if submission.num_comments < self.min_comments:
                return False

            # Check author karma (if available)
            if hasattr(submission.author, 'comment_karma'):
                if submission.author.comment_karma < self.min_karma:
                    return False

            # Filter deleted/removed posts
            if submission.removed_by_category or getattr(submission, 'selftext', '') == '[deleted]':
                return False

            # Minimum content length
            title_content = f"{submission.title} {getattr(submission, 'selftext', '')}"
            if len(title_content.strip()) < 20:
                return False

            return True

        except Exception as e:
            self.logger.debug(f"Error checking post quality: {e}")
            return False

    async def _extract_post_data(self, submission) -> Optional[Dict[str, Any]]:
        """Extract relevant data from Reddit submission"""
        try:
            # Get author info safely
            author_name = None
            author_karma = 0
            if submission.author and not submission.author.name == '[deleted]':
                author_name = submission.author.name
                try:
                    author_karma = getattr(submission.author, 'comment_karma', 0)
                except Exception as e:
                    logger.debug(f"Error fetching author karma: {e}")
                    author_karma = 0

            # Get flair
            link_flair = getattr(submission, 'link_flair_text', '')

            post_data = {
                'id': submission.id,
                'title': submission.title,
                'selftext': getattr(submission, 'selftext', ''),
                'url': submission.url,
                'score': submission.score,
                'upvote_ratio': getattr(submission, 'upvote_ratio', 0.5),
                'num_comments': submission.num_comments,
                'created_utc': submission.created_utc,
                'subreddit': submission.subreddit.display_name,
                'author': author_name,
                'author_karma': author_karma,
                'flair': link_flair,
                'is_self': submission.is_self,
                'permalink': submission.permalink
            }

            # Get top comments if this is a discussion post
            if submission.is_self and submission.num_comments > 0:
                comments = await self._get_top_comments(submission, limit=5)
                post_data['top_comments'] = comments

            return post_data

        except Exception as e:
            self.logger.error(f"Failed to extract post data: {e}")
            return None

    async def _get_top_comments(self, submission, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top comments from a submission"""
        comments = []

        try:
            await submission.comments.replace_more(limit=0)

            for comment in submission.comments.list()[:limit]:
                if hasattr(comment, 'body') and comment.body != '[deleted]':
                    comment_data = {
                        'id': comment.id,
                        'body': comment.body,
                        'score': getattr(comment, 'score', 0),
                        'author': getattr(comment.author, 'name', '[deleted]') if comment.author else '[deleted]',
                        'created_utc': comment.created_utc
                    }
                    comments.append(comment_data)

        except Exception as e:
            self.logger.debug(f"Failed to get comments: {e}")

        return comments

    async def _process_reddit_post(
        self,
        post_data: Dict[str, Any],
        symbol: str,
        subreddit_name: str
    ) -> Optional[DataPoint]:
        """Process a Reddit post into a DataPoint"""
        try:
            # Combine title and text for sentiment analysis
            full_text = f"{post_data['title']} {post_data.get('selftext', '')}"

            # Add top comments to text
            if post_data.get('top_comments'):
                comment_text = " ".join([c['body'] for c in post_data['top_comments']])
                full_text += f" COMMENTS: {comment_text}"

            # Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze_sentiment(full_text, symbol)

            # Calculate Reddit-specific scores
            wsb_score = self._calculate_wsb_score(full_text, post_data)
            community_score = self._calculate_community_engagement_score(post_data)
            credibility_score = self._calculate_reddit_credibility_score(post_data)

            # Create raw data
            raw_data = {
                'post_id': post_data['id'],
                'title': post_data['title'],
                'selftext': post_data.get('selftext', ''),
                'subreddit': subreddit_name,
                'author': post_data.get('author'),
                'score': post_data['score'],
                'upvote_ratio': post_data.get('upvote_ratio', 0.5),
                'num_comments': post_data['num_comments'],
                'created_utc': post_data['created_utc'],
                'flair': post_data.get('flair', ''),
                'url': post_data.get('url', ''),
                'permalink': post_data.get('permalink', ''),
                'top_comments': post_data.get('top_comments', []),
                'search_symbol': symbol
            }

            # Create processed data
            processed_data = {
                'sentiment': sentiment.to_dict(),
                'wsb_score': wsb_score,
                'community_engagement': community_score,
                'credibility_score': credibility_score,
                'relevance_score': self._calculate_reddit_relevance_score(full_text, symbol),
                'viral_potential': self._calculate_reddit_viral_potential(post_data),
                'subreddit_category': self._categorize_subreddit(subreddit_name)
            }

            # Determine asset class
            asset_class = self._determine_asset_class(symbol, full_text, subreddit_name)

            # Create data point
            data_point = DataPoint(
                source=DataSource.REDDIT,
                timestamp=datetime.fromtimestamp(post_data['created_utc']),
                symbol=symbol,
                asset_class=asset_class,
                raw_data=raw_data,
                processed_data=processed_data,
                confidence=sentiment.confidence * (community_score / 100.0)
            )

            return data_point

        except Exception as e:
            self.logger.error(f"Failed to process Reddit post: {e}")
            return None

    def _calculate_wsb_score(self, text: str, post_data: Dict[str, Any]) -> float:
        """Calculate WSB-style meme/hype score"""
        text_lower = text.lower()

        # WSB-specific terms
        wsb_terms = {
            'diamond hands': 15, 'paper hands': -10, 'hodl': 10, 'yolo': 12,
            'to the moon': 15, 'moon': 8, '🚀': 10, 'rocket': 8,
            'ape': 8, 'retard': 5, 'autist': 5, 'tendies': 10,
            'stonks': 8, 'brrrr': 8, 'wsb': 5, 'degenerates': 5
        }

        # Calculate WSB term score
        wsb_score = 0
        for term, points in wsb_terms.items():
            wsb_score += text_lower.count(term) * points

        # Emoji bonus
        rocket_count = text.count('🚀') + text.count('🌙') + text.count('💎')
        wsb_score += rocket_count * 3

        # All caps bonus (shouting)
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:
            wsb_score += 10

        # Upvote multiplier for WSB subreddit
        if post_data.get('subreddit', '').lower() == 'wallstreetbets':
            wsb_score *= 1.5

        return min(100, max(0, wsb_score))

    def _calculate_community_engagement_score(self, post_data: Dict[str, Any]) -> float:
        """Calculate community engagement score"""
        score = post_data.get('score', 0)
        comments = post_data.get('num_comments', 0)
        upvote_ratio = post_data.get('upvote_ratio', 0.5)

        # Normalize scores
        score_component = min(50, score / 20)  # Max 50 points for score
        comment_component = min(30, comments * 2)  # Max 30 points for comments
        ratio_component = upvote_ratio * 20  # Max 20 points for ratio

        return score_component + comment_component + ratio_component

    def _calculate_reddit_credibility_score(self, post_data: Dict[str, Any]) -> float:
        """Calculate post credibility score"""
        # Author karma
        author_karma = post_data.get('author_karma', 0)
        karma_score = min(30, author_karma / 1000)

        # Flair credibility
        flair = post_data.get('flair', '').lower()
        flair_bonus = 0
        if any(term in flair for term in ['dd', 'due diligence', 'analysis', 'discussion']):
            flair_bonus = 20
        elif any(term in flair for term in ['meme', 'shitpost', 'loss', 'gain']):
            flair_bonus = -10

        # Content quality (post length, formatting)
        content_length = len(post_data.get('selftext', ''))
        length_score = min(25, content_length / 100)

        # Subreddit credibility
        subreddit = post_data.get('subreddit', '').lower()
        subreddit_multiplier = {
            'investing': 1.2,
            'securityanalysis': 1.3,
            'valueinvesting': 1.3,
            'wallstreetbets': 0.8,
            'pennystocks': 0.7,
            'cryptocurrency': 0.9
        }.get(subreddit, 1.0)

        base_score = karma_score + length_score + flair_bonus
        final_score = base_score * subreddit_multiplier

        return max(0, min(100, final_score))

    def _calculate_reddit_relevance_score(self, text: str, symbol: str) -> float:
        """Calculate relevance to trading symbol"""
        text_lower = text.lower()
        symbol_lower = symbol.lower()

        # Symbol mention count
        direct_mentions = text_lower.count(f'${symbol_lower}') * 20
        indirect_mentions = text_lower.count(symbol_lower) * 10

        # Trading context
        trading_terms = [
            'buy', 'sell', 'hold', 'calls', 'puts', 'options', 'strike',
            'expiry', 'earnings', 'dd', 'due diligence', 'analysis'
        ]
        context_score = sum(10 for term in trading_terms if term in text_lower)

        # Price/target mentions
        price_mentions = len(re.findall(r'\$\d+', text)) * 5

        total_score = direct_mentions + indirect_mentions + context_score + price_mentions
        return min(100, total_score)

    def _calculate_reddit_viral_potential(self, post_data: Dict[str, Any]) -> float:
        """Calculate viral potential"""
        # Comment to upvote ratio
        comments = post_data.get('num_comments', 0)
        score = max(1, post_data.get('score', 1))
        comment_ratio = (comments / score) * 100

        # Upvote ratio (controversial posts get more engagement)
        upvote_ratio = post_data.get('upvote_ratio', 0.5)
        controversy_bonus = 20 if 0.4 <= upvote_ratio <= 0.6 else 0

        # Time factor (newer posts have higher viral potential)
        post_age_hours = (datetime.utcnow().timestamp() - post_data['created_utc']) / 3600
        time_factor = max(0, 100 - post_age_hours * 2)  # Decreases with age

        viral_score = comment_ratio + controversy_bonus + (time_factor * 0.3)
        return min(100, viral_score)

    def _categorize_subreddit(self, subreddit_name: str) -> str:
        """Categorize subreddit type"""
        subreddit_lower = subreddit_name.lower()

        categories = {
            'meme': ['wallstreetbets', 'wsb'],
            'analysis': ['investing', 'securityanalysis', 'valueinvesting'],
            'crypto': ['cryptocurrency', 'bitcoin', 'ethereum', 'crypto'],
            'speculation': ['pennystocks', 'robinhoodpennystocks'],
            'options': ['options', 'thetagang'],
            'general': ['stocks', 'stockmarket', 'personalfinance']
        }

        for category, subreddits in categories.items():
            if any(sub in subreddit_lower for sub in subreddits):
                return category

        return 'other'

    def _determine_asset_class(self, symbol: str, text: str, subreddit: str) -> str:
        """Determine asset class from context"""
        text_lower = text.lower()
        subreddit_lower = subreddit.lower()

        # Crypto subreddits
        if any(crypto in subreddit_lower for crypto in ['crypto', 'bitcoin', 'ethereum']):
            return 'crypto'

        # Crypto keywords in text
        crypto_terms = ['coin', 'token', 'crypto', 'blockchain', 'defi']
        if any(term in text_lower for term in crypto_terms):
            return 'crypto'

        # Short symbols likely crypto
        if len(symbol) <= 4:
            return 'crypto'

        # Options context
        if any(term in text_lower for term in ['calls', 'puts', 'strike', 'expiry']):
            return 'options'

        # Default to stocks
        return 'stocks'

    def validate_data(self, data_point: DataPoint) -> Tuple[bool, float]:
        """Validate Reddit data point quality"""
        try:
            raw_data = data_point.raw_data
            processed_data = data_point.processed_data

            # Basic validation
            if not raw_data.get('title') or not raw_data.get('post_id'):
                return False, 0.0

            # Quality components
            community_score = processed_data.get('community_engagement', 0) / 100.0
            credibility_score = processed_data.get('credibility_score', 0) / 100.0
            relevance_score = processed_data.get('relevance_score', 0) / 100.0
            sentiment_confidence = processed_data.get('sentiment', {}).get('confidence', 0)

            # Weighted quality score
            quality_score = (
                community_score * 0.25 +
                credibility_score * 0.3 +
                relevance_score * 0.25 +
                sentiment_confidence * 0.2
            )

            # Minimum thresholds
            is_valid = (
                quality_score >= 0.3 and
                relevance_score >= 0.1 and
                raw_data.get('score', 0) >= self.min_upvotes
            )

            return is_valid, quality_score

        except Exception as e:
            self.logger.error(f"Failed to validate Reddit data: {e}")
            return False, 0.0