"""
Twitter Connector
=================

Real-time Twitter sentiment analysis for trading signals.
Monitors mentions, hashtags, and influencer posts.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from ..core.base_connector import BaseDataConnector, DataPoint, DataSource, ConnectorConfig
from .sentiment_analyzer import SentimentAnalyzer, SentimentScore


class TwitterConnector(BaseDataConnector):
    """
    Twitter sentiment analysis connector

    Features:
    - Real-time tweet monitoring
    - Influencer tracking
    - Hashtag analysis
    - Volume spike detection
    - Sentiment scoring with FinBERT
    """

    def __init__(self, config: ConnectorConfig):
        super().__init__(DataSource.TWITTER, config)

        self.sentiment_analyzer = SentimentAnalyzer()

        # Twitter API credentials
        self.bearer_token = config.custom_params.get('bearer_token')
        self.api_key = config.api_key
        self.api_secret = config.custom_params.get('api_secret')
        self.access_token = config.custom_params.get('access_token')
        self.access_token_secret = config.custom_params.get('access_token_secret')

        # Twitter API client
        self.api = None
        self.client = None

        # Influencer accounts to monitor
        self.influencers = config.custom_params.get('influencers', [
            'elonmusk', 'michael_saylor', 'cz_binance', 'naval',
            'APompliano', 'VitalikButerin', 'cryptocurrency', 'coindesk'
        ])

        # Minimum followers for tweet consideration
        self.min_followers = config.custom_params.get('min_followers', 1000)
        self.min_retweets = config.custom_params.get('min_retweets', 5)
        self.min_likes = config.custom_params.get('min_likes', 10)

        self._initialize_twitter_api()

    def _initialize_twitter_api(self) -> None:
        """Initialize Twitter API clients"""
        if not TWEEPY_AVAILABLE:
            self.logger.error("Tweepy not available. Install with: pip install tweepy")
            return

        try:
            # Initialize Twitter API v2 client
            if self.bearer_token:
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    consumer_key=self.api_key,
                    consumer_secret=self.api_secret,
                    access_token=self.access_token,
                    access_token_secret=self.access_token_secret,
                    wait_on_rate_limit=True
                )

                # Test API connection
                me = self.client.get_me()
                if me.data:
                    self.logger.info(f"Twitter API initialized for user: {me.data.username}")
                else:
                    self.logger.warning("Twitter API authentication may have failed")

        except Exception as e:
            self.logger.error(f"Failed to initialize Twitter API: {e}")
            self.client = None

    async def fetch_data(
        self,
        symbols: List[str],
        lookback_hours: int = 24,
        **kwargs
    ) -> List[DataPoint]:
        """
        Fetch Twitter data for symbols

        Args:
            symbols: List of symbols to search for
            lookback_hours: Hours of historical data to fetch
            **kwargs: Additional parameters

        Returns:
            List of DataPoint objects with Twitter sentiment data
        """
        if not self.client:
            self.logger.error("Twitter API not available")
            return []

        all_data_points = []

        # Create search queries for each symbol
        for symbol in symbols:
            try:
                # Fetch tweets for symbol
                tweets = await self._fetch_symbol_tweets(symbol, lookback_hours)

                # Process tweets into data points
                for tweet_data in tweets:
                    data_point = await self._process_tweet(tweet_data, symbol)
                    if data_point:
                        all_data_points.append(data_point)

            except Exception as e:
                self.logger.error(f"Failed to fetch Twitter data for {symbol}: {e}")

        self.logger.info(f"Fetched {len(all_data_points)} Twitter data points")
        return all_data_points

    async def get_real_time_data(self, symbols: List[str]) -> List[DataPoint]:
        """
        Get real-time Twitter data

        Args:
            symbols: Symbols to monitor

        Returns:
            List of current DataPoint objects
        """
        # For real-time, fetch recent tweets (last 15 minutes)
        return await self.fetch_data(symbols, lookback_hours=0.25)

    async def _fetch_symbol_tweets(self, symbol: str, lookback_hours: int) -> List[Dict[str, Any]]:
        """Fetch tweets mentioning a symbol"""
        if not self.client:
            return []

        tweets = []

        try:
            # Create search queries
            queries = self._create_search_queries(symbol)

            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=lookback_hours) if lookback_hours > 0 else end_time - timedelta(minutes=15)

            for query in queries:
                try:
                    # Search tweets
                    response = self.client.search_recent_tweets(
                        query=query,
                        start_time=start_time,
                        end_time=end_time,
                        max_results=100,
                        tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations', 'lang'],
                        user_fields=['public_metrics', 'verified']
                    )

                    if response.data:
                        # Process tweets
                        for tweet in response.data:
                            tweet_dict = {
                                'id': tweet.id,
                                'text': tweet.text,
                                'created_at': tweet.created_at,
                                'author_id': tweet.author_id,
                                'public_metrics': tweet.public_metrics,
                                'lang': getattr(tweet, 'lang', 'en'),
                                'context_annotations': getattr(tweet, 'context_annotations', [])
                            }

                            # Add user data if available
                            if response.includes and 'users' in response.includes:
                                for user in response.includes['users']:
                                    if user.id == tweet.author_id:
                                        tweet_dict['user'] = {
                                            'username': user.username,
                                            'followers_count': user.public_metrics['followers_count'],
                                            'verified': getattr(user, 'verified', False)
                                        }
                                        break

                            tweets.append(tweet_dict)

                    # Rate limiting pause
                    await asyncio.sleep(1)

                except Exception as e:
                    self.logger.warning(f"Failed to search tweets for query '{query}': {e}")

        except Exception as e:
            self.logger.error(f"Failed to fetch tweets for {symbol}: {e}")

        return tweets

    def _create_search_queries(self, symbol: str) -> List[str]:
        """Create optimized Twitter search queries for a symbol"""
        queries = []

        # Basic symbol searches
        queries.append(f"${symbol} lang:en -is:retweet")
        queries.append(f"{symbol} (buy OR sell OR bull OR bear) lang:en -is:retweet")

        # Crypto-specific searches
        if len(symbol) <= 4:  # Likely crypto
            queries.append(f"{symbol} (coin OR token OR crypto) lang:en -is:retweet")
            queries.append(f"#{symbol.lower()} lang:en -is:retweet")

        # Stock-specific searches
        else:
            queries.append(f"{symbol} (stock OR earnings OR price) lang:en -is:retweet")

        # Influencer mentions
        influencer_mentions = " OR from:".join(self.influencers)
        queries.append(f"(${symbol} OR {symbol}) (from:{influencer_mentions}) lang:en")

        return queries

    async def _process_tweet(self, tweet_data: Dict[str, Any], symbol: str) -> Optional[DataPoint]:
        """Process a tweet into a DataPoint"""
        try:
            # Filter low-quality tweets
            if not self._is_quality_tweet(tweet_data):
                return None

            # Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze_sentiment(
                tweet_data['text'], symbol
            )

            # Calculate influence score
            influence_score = self._calculate_influence_score(tweet_data)

            # Create raw data
            raw_data = {
                'tweet_id': str(tweet_data['id']),
                'text': tweet_data['text'],
                'author_id': tweet_data.get('author_id'),
                'user': tweet_data.get('user', {}),
                'created_at': tweet_data['created_at'].isoformat() if isinstance(tweet_data['created_at'], datetime) else tweet_data['created_at'],
                'metrics': tweet_data.get('public_metrics', {}),
                'lang': tweet_data.get('lang', 'en'),
                'influence_score': influence_score,
                'search_symbol': symbol
            }

            # Create processed data
            processed_data = {
                'sentiment': sentiment.to_dict(),
                'influence_score': influence_score,
                'engagement_rate': self._calculate_engagement_rate(tweet_data),
                'viral_potential': self._calculate_viral_potential(tweet_data),
                'relevance_score': self._calculate_relevance_score(tweet_data, symbol),
                'credibility_score': self._calculate_credibility_score(tweet_data)
            }

            # Determine asset class
            asset_class = self._determine_asset_class(symbol, tweet_data['text'])

            # Create data point
            data_point = DataPoint(
                source=DataSource.TWITTER,
                timestamp=tweet_data['created_at'] if isinstance(tweet_data['created_at'], datetime) else datetime.fromisoformat(tweet_data['created_at']),
                symbol=symbol,
                asset_class=asset_class,
                raw_data=raw_data,
                processed_data=processed_data,
                confidence=sentiment.confidence * (influence_score / 100.0)
            )

            return data_point

        except Exception as e:
            self.logger.error(f"Failed to process tweet: {e}")
            return None

    def _is_quality_tweet(self, tweet_data: Dict[str, Any]) -> bool:
        """Filter out low-quality tweets"""
        # Check language
        if tweet_data.get('lang', 'en') != 'en':
            return False

        # Check minimum engagement
        metrics = tweet_data.get('public_metrics', {})
        if metrics.get('retweet_count', 0) < self.min_retweets and metrics.get('like_count', 0) < self.min_likes:
            # Allow if from influencer
            user_data = tweet_data.get('user', {})
            if user_data.get('followers_count', 0) < self.min_followers:
                return False

        # Filter spam indicators
        text = tweet_data.get('text', '').lower()
        spam_indicators = ['follow me', 'dm me', 'check bio', 'link in bio', 'free money', 'guaranteed']
        if any(spam in text for spam in spam_indicators):
            return False

        # Filter very short tweets
        if len(tweet_data.get('text', '')) < 20:
            return False

        return True

    def _calculate_influence_score(self, tweet_data: Dict[str, Any]) -> float:
        """Calculate influence score for a tweet"""
        user_data = tweet_data.get('user', {})
        metrics = tweet_data.get('public_metrics', {})

        # Base score from follower count
        followers = user_data.get('followers_count', 0)
        follower_score = min(50, followers / 10000)  # Max 50 points for followers

        # Verification bonus
        verified_score = 20 if user_data.get('verified', False) else 0

        # Engagement score
        likes = metrics.get('like_count', 0)
        retweets = metrics.get('retweet_count', 0)
        replies = metrics.get('reply_count', 0)

        engagement_score = min(30, (likes + retweets * 3 + replies * 2) / 10)

        total_score = follower_score + verified_score + engagement_score

        return min(100, total_score)

    def _calculate_engagement_rate(self, tweet_data: Dict[str, Any]) -> float:
        """Calculate engagement rate"""
        user_data = tweet_data.get('user', {})
        metrics = tweet_data.get('public_metrics', {})

        followers = user_data.get('followers_count', 1)
        total_engagement = sum([
            metrics.get('like_count', 0),
            metrics.get('retweet_count', 0),
            metrics.get('reply_count', 0)
        ])

        return min(1.0, total_engagement / followers) if followers > 0 else 0.0

    def _calculate_viral_potential(self, tweet_data: Dict[str, Any]) -> float:
        """Calculate viral potential score"""
        metrics = tweet_data.get('public_metrics', {})

        # Retweet velocity (retweets per like)
        likes = metrics.get('like_count', 1)
        retweets = metrics.get('retweet_count', 0)
        retweet_ratio = retweets / likes if likes > 0 else 0

        # Reply engagement
        replies = metrics.get('reply_count', 0)
        reply_ratio = replies / likes if likes > 0 else 0

        # Combined score
        viral_score = (retweet_ratio * 0.6 + reply_ratio * 0.4) * 100

        return min(100, viral_score)

    def _calculate_relevance_score(self, tweet_data: Dict[str, Any], symbol: str) -> float:
        """Calculate relevance to trading symbol"""
        text = tweet_data.get('text', '').lower()

        # Direct symbol mentions
        symbol_mentions = text.count(f'${symbol.lower()}') + text.count(symbol.lower())
        mention_score = min(50, symbol_mentions * 25)

        # Context relevance
        trading_terms = [
            'buy', 'sell', 'hold', 'moon', 'dump', 'pump', 'bull', 'bear',
            'breakout', 'support', 'resistance', 'target', 'stop loss'
        ]

        context_score = min(30, sum(5 for term in trading_terms if term in text))

        # Financial context
        financial_terms = ['price', 'market', 'trading', 'investment', 'profit', 'loss']
        financial_score = min(20, sum(4 for term in financial_terms if term in text))

        return min(100, mention_score + context_score + financial_score)

    def _calculate_credibility_score(self, tweet_data: Dict[str, Any]) -> float:
        """Calculate source credibility score"""
        user_data = tweet_data.get('user', {})

        # Verification status
        verified_score = 30 if user_data.get('verified', False) else 0

        # Follower count credibility
        followers = user_data.get('followers_count', 0)
        if followers > 100000:
            follower_cred = 40
        elif followers > 10000:
            follower_cred = 30
        elif followers > 1000:
            follower_cred = 20
        else:
            follower_cred = 10

        # Account age (would need additional API call to get)
        # For now, assume moderate credibility
        age_score = 20

        # Spam indicators
        text = tweet_data.get('text', '').lower()
        spam_penalty = 0
        spam_words = ['follow me', 'dm me', 'free', 'guaranteed', '100%']
        for word in spam_words:
            if word in text:
                spam_penalty += 10

        total_score = verified_score + follower_cred + age_score - spam_penalty

        return max(0, min(100, total_score))

    def _determine_asset_class(self, symbol: str, text: str) -> str:
        """Determine asset class from symbol and context"""
        text_lower = text.lower()

        # Crypto indicators
        crypto_terms = ['coin', 'token', 'crypto', 'blockchain', 'defi', 'nft']
        if any(term in text_lower for term in crypto_terms) or len(symbol) <= 4:
            return 'crypto'

        # Forex indicators
        forex_pairs = ['EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        if symbol in forex_pairs or any(pair in symbol for pair in forex_pairs):
            return 'forex'

        # Commodity indicators
        commodity_terms = ['gold', 'silver', 'oil', 'gas', 'wheat', 'corn']
        if any(term in text_lower for term in commodity_terms):
            return 'commodities'

        # Default to stocks
        return 'stocks'

    def validate_data(self, data_point: DataPoint) -> Tuple[bool, float]:
        """
        Validate Twitter data point quality

        Args:
            data_point: DataPoint to validate

        Returns:
            (is_valid, quality_score)
        """
        try:
            raw_data = data_point.raw_data
            processed_data = data_point.processed_data

            # Basic validation
            if not raw_data.get('text') or not raw_data.get('tweet_id'):
                return False, 0.0

            # Quality factors
            influence_score = processed_data.get('influence_score', 0) / 100.0
            credibility_score = processed_data.get('credibility_score', 0) / 100.0
            relevance_score = processed_data.get('relevance_score', 0) / 100.0
            sentiment_confidence = processed_data.get('sentiment', {}).get('confidence', 0)

            # Weighted quality score
            quality_score = (
                influence_score * 0.3 +
                credibility_score * 0.25 +
                relevance_score * 0.25 +
                sentiment_confidence * 0.2
            )

            # Minimum thresholds
            is_valid = (
                quality_score >= 0.3 and
                relevance_score >= 0.2 and
                credibility_score >= 0.1
            )

            return is_valid, quality_score

        except Exception as e:
            self.logger.error(f"Failed to validate Twitter data: {e}")
            return False, 0.0