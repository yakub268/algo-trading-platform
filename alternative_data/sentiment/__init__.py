"""
Social Sentiment Analysis
========================

Connectors for social media sentiment analysis across
Twitter, Reddit, Discord, and other platforms.
"""

from .twitter_connector import TwitterConnector
from .reddit_connector import RedditConnector
from .discord_connector import DiscordConnector
from .sentiment_analyzer import SentimentAnalyzer
from .social_aggregator import SocialSentimentAggregator

__all__ = [
    "TwitterConnector",
    "RedditConnector",
    "DiscordConnector",
    "SentimentAnalyzer",
    "SocialSentimentAggregator"
]