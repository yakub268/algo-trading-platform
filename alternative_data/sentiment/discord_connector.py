"""
Discord Connector
=================

Discord sentiment analysis for crypto and trading communities.
Monitors channels and servers for trading signals.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json

try:
    import discord
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

from ..core.base_connector import BaseDataConnector, DataPoint, DataSource, ConnectorConfig
from .sentiment_analyzer import SentimentAnalyzer, SentimentScore


class DiscordConnector(BaseDataConnector):
    """
    Discord sentiment analysis connector

    Features:
    - Server/channel monitoring
    - Real-time message analysis
    - Influencer tracking
    - Crypto community sentiment
    - Alpha leak detection
    """

    def __init__(self, config: ConnectorConfig):
        super().__init__(DataSource.DISCORD, config)

        self.sentiment_analyzer = SentimentAnalyzer()

        # Discord bot token
        self.bot_token = config.api_key

        # Discord client
        self.client = None
        self.is_running = False

        # Target servers and channels
        self.target_servers = config.custom_params.get('servers', [])
        self.target_channels = config.custom_params.get('channels', [])

        # Message storage
        self.recent_messages = []
        self.max_messages = config.custom_params.get('max_messages', 1000)

        # Quality filters
        self.min_reactions = config.custom_params.get('min_reactions', 2)
        self.min_message_length = config.custom_params.get('min_message_length', 10)

        # VIP users (known traders/influencers)
        self.vip_users = config.custom_params.get('vip_users', [])

        self._initialize_discord_client()

    def _initialize_discord_client(self) -> None:
        """Initialize Discord client"""
        if not DISCORD_AVAILABLE:
            self.logger.error("discord.py not available. Install with: pip install discord.py")
            return

        if not self.bot_token:
            self.logger.error("Discord bot token not provided")
            return

        try:
            intents = discord.Intents.default()
            intents.message_content = True
            intents.guild_messages = True

            self.client = discord.Client(intents=intents)

            # Set up event handlers
            @self.client.event
            async def on_ready():
                self.logger.info(f'Discord client logged in as {self.client.user}')
                self.is_running = True

            @self.client.event
            async def on_message(message):
                await self._process_message(message)

            self.logger.info("Discord client initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize Discord client: {e}")
            self.client = None

    async def fetch_data(
        self,
        symbols: List[str],
        lookback_hours: int = 24,
        **kwargs
    ) -> List[DataPoint]:
        """
        Fetch Discord data for symbols

        Args:
            symbols: List of symbols to search for
            lookback_hours: Hours of historical data to fetch
            **kwargs: Additional parameters

        Returns:
            List of DataPoint objects with Discord sentiment data
        """
        if not self.client or not self.is_running:
            # Start client if not running
            if self.client and not self.is_running:
                await self._start_client()

            if not self.is_running:
                self.logger.error("Discord client not available")
                return []

        all_data_points = []

        # Search through recent messages for symbol mentions
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)

        for message_data in self.recent_messages:
            message_time = datetime.fromtimestamp(message_data['timestamp'])
            if message_time < cutoff_time:
                continue

            # Check if message mentions any symbols
            for symbol in symbols:
                if self._message_mentions_symbol(message_data, symbol):
                    data_point = await self._process_discord_message(message_data, symbol)
                    if data_point:
                        all_data_points.append(data_point)

        # Also fetch historical messages from target channels if available
        if self.target_channels:
            historical_data = await self._fetch_historical_messages(symbols, lookback_hours)
            all_data_points.extend(historical_data)

        self.logger.info(f"Fetched {len(all_data_points)} Discord data points")
        return all_data_points

    async def get_real_time_data(self, symbols: List[str]) -> List[DataPoint]:
        """Get real-time Discord data (last 15 minutes)"""
        return await self.fetch_data(symbols, lookback_hours=0.25)

    async def _start_client(self) -> None:
        """Start Discord client"""
        try:
            # Start client in background
            asyncio.create_task(self.client.start(self.bot_token))

            # Wait for client to be ready
            timeout = 30
            while not self.is_running and timeout > 0:
                await asyncio.sleep(1)
                timeout -= 1

            if not self.is_running:
                self.logger.error("Discord client failed to start")

        except Exception as e:
            self.logger.error(f"Failed to start Discord client: {e}")

    async def _process_message(self, message: discord.Message) -> None:
        """Process incoming Discord message"""
        try:
            # Skip bot messages
            if message.author.bot:
                return

            # Skip messages in non-target channels if specified
            if self.target_channels and message.channel.id not in self.target_channels:
                return

            # Skip messages in non-target servers if specified
            if self.target_servers and message.guild and message.guild.id not in self.target_servers:
                return

            # Extract message data
            message_data = {
                'id': message.id,
                'content': message.content,
                'author_id': message.author.id,
                'author_name': message.author.display_name,
                'channel_id': message.channel.id,
                'channel_name': message.channel.name,
                'guild_id': message.guild.id if message.guild else None,
                'guild_name': message.guild.name if message.guild else None,
                'timestamp': message.created_at.timestamp(),
                'reactions': [{'emoji': str(r.emoji), 'count': r.count} for r in message.reactions],
                'attachments': [att.url for att in message.attachments],
                'embeds': len(message.embeds),
                'reply_to': message.reference.message_id if message.reference else None
            }

            # Add to recent messages
            self.recent_messages.append(message_data)

            # Keep only recent messages
            if len(self.recent_messages) > self.max_messages:
                self.recent_messages = self.recent_messages[-self.max_messages:]

        except Exception as e:
            self.logger.error(f"Failed to process Discord message: {e}")

    async def _fetch_historical_messages(
        self,
        symbols: List[str],
        lookback_hours: int
    ) -> List[DataPoint]:
        """Fetch historical messages from target channels"""
        data_points = []

        if not self.client or not self.is_running:
            return data_points

        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)

        try:
            for channel_id in self.target_channels:
                try:
                    channel = self.client.get_channel(channel_id)
                    if not channel:
                        continue

                    # Fetch recent messages
                    async for message in channel.history(
                        limit=200,
                        after=cutoff_time
                    ):
                        if message.author.bot:
                            continue

                        # Check for symbol mentions
                        for symbol in symbols:
                            if self._message_mentions_symbol_text(message.content, symbol):
                                message_data = {
                                    'id': message.id,
                                    'content': message.content,
                                    'author_id': message.author.id,
                                    'author_name': message.author.display_name,
                                    'channel_id': message.channel.id,
                                    'channel_name': message.channel.name,
                                    'guild_id': message.guild.id if message.guild else None,
                                    'guild_name': message.guild.name if message.guild else None,
                                    'timestamp': message.created_at.timestamp(),
                                    'reactions': [{'emoji': str(r.emoji), 'count': r.count} for r in message.reactions],
                                    'attachments': [att.url for att in message.attachments],
                                    'embeds': len(message.embeds)
                                }

                                data_point = await self._process_discord_message(message_data, symbol)
                                if data_point:
                                    data_points.append(data_point)

                    # Rate limiting
                    await asyncio.sleep(1)

                except Exception as e:
                    self.logger.warning(f"Failed to fetch from channel {channel_id}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to fetch historical Discord messages: {e}")

        return data_points

    def _message_mentions_symbol(self, message_data: Dict[str, Any], symbol: str) -> bool:
        """Check if message mentions a symbol"""
        return self._message_mentions_symbol_text(message_data.get('content', ''), symbol)

    def _message_mentions_symbol_text(self, content: str, symbol: str) -> bool:
        """Check if message text mentions a symbol"""
        content_lower = content.lower()
        symbol_lower = symbol.lower()

        # Direct mentions
        if f'${symbol_lower}' in content_lower or f' {symbol_lower} ' in content_lower:
            return True

        # Crypto-style mentions
        if len(symbol) <= 4 and any(
            f'{symbol_lower}{suffix}' in content_lower
            for suffix in ['coin', 'token', '/usdt', '/btc', '/usd']
        ):
            return True

        return False

    async def _process_discord_message(
        self,
        message_data: Dict[str, Any],
        symbol: str
    ) -> Optional[DataPoint]:
        """Process Discord message into DataPoint"""
        try:
            # Filter low-quality messages
            if not self._is_quality_message(message_data):
                return None

            content = message_data.get('content', '')

            # Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze_sentiment(content, symbol)

            # Calculate Discord-specific scores
            alpha_score = self._calculate_alpha_score(content, message_data)
            community_score = self._calculate_discord_community_score(message_data)
            influencer_score = self._calculate_influencer_score(message_data)

            # Create raw data
            raw_data = {
                'message_id': str(message_data['id']),
                'content': content,
                'author_id': str(message_data['author_id']),
                'author_name': message_data.get('author_name', ''),
                'channel_id': str(message_data['channel_id']),
                'channel_name': message_data.get('channel_name', ''),
                'guild_id': str(message_data.get('guild_id', '')),
                'guild_name': message_data.get('guild_name', ''),
                'timestamp': message_data['timestamp'],
                'reactions': message_data.get('reactions', []),
                'attachments': message_data.get('attachments', []),
                'embeds': message_data.get('embeds', 0),
                'reply_to': message_data.get('reply_to'),
                'search_symbol': symbol
            }

            # Create processed data
            processed_data = {
                'sentiment': sentiment.to_dict(),
                'alpha_score': alpha_score,
                'community_engagement': community_score,
                'influencer_score': influencer_score,
                'relevance_score': self._calculate_discord_relevance_score(content, symbol),
                'urgency_score': self._calculate_urgency_score(content),
                'signal_strength': self._calculate_signal_strength(message_data, sentiment)
            }

            # Determine asset class
            asset_class = self._determine_asset_class(symbol, content, message_data)

            # Create data point
            data_point = DataPoint(
                source=DataSource.DISCORD,
                timestamp=datetime.fromtimestamp(message_data['timestamp']),
                symbol=symbol,
                asset_class=asset_class,
                raw_data=raw_data,
                processed_data=processed_data,
                confidence=sentiment.confidence * (influencer_score / 100.0 if influencer_score > 0 else 0.5)
            )

            return data_point

        except Exception as e:
            self.logger.error(f"Failed to process Discord message: {e}")
            return None

    def _is_quality_message(self, message_data: Dict[str, Any]) -> bool:
        """Check if message meets quality criteria"""
        content = message_data.get('content', '')

        # Minimum length
        if len(content) < self.min_message_length:
            return False

        # Minimum reactions
        total_reactions = sum(r['count'] for r in message_data.get('reactions', []))
        if total_reactions < self.min_reactions:
            # Exception for VIP users
            if message_data.get('author_id') not in self.vip_users:
                return False

        # Filter spam
        if content.count('http') > 2:  # Too many links
            return False

        # Filter very repetitive messages
        words = content.split()
        if len(set(words)) < len(words) * 0.5:  # Too repetitive
            return False

        return True

    def _calculate_alpha_score(self, content: str, message_data: Dict[str, Any]) -> float:
        """Calculate alpha/insider information score"""
        content_lower = content.lower()

        # Alpha keywords
        alpha_terms = {
            'insider': 20, 'leak': 15, 'rumor': 10, 'heard': 8,
            'source': 12, 'confirms': 15, 'partnership': 18,
            'acquisition': 20, 'merger': 20, 'announcement': 15,
            'news': 8, 'breaking': 12, 'exclusive': 18
        }

        alpha_score = 0
        for term, points in alpha_terms.items():
            alpha_score += content_lower.count(term) * points

        # Urgency indicators
        urgency_terms = ['now', 'asap', 'quick', 'fast', 'soon', 'today']
        urgency_bonus = sum(3 for term in urgency_terms if term in content_lower)
        alpha_score += urgency_bonus

        # Channel context (some channels more likely to have alpha)
        channel_name = message_data.get('channel_name', '').lower()
        if any(term in channel_name for term in ['alpha', 'insider', 'vip', 'premium']):
            alpha_score *= 1.5

        return min(100, alpha_score)

    def _calculate_discord_community_score(self, message_data: Dict[str, Any]) -> float:
        """Calculate community engagement score"""
        # Reaction count and types
        reactions = message_data.get('reactions', [])
        total_reactions = sum(r['count'] for r in reactions)
        reaction_diversity = len(reactions)

        reaction_score = min(40, total_reactions * 2) + min(10, reaction_diversity * 2)

        # Message features
        has_attachments = len(message_data.get('attachments', [])) > 0
        has_embeds = message_data.get('embeds', 0) > 0
        is_reply = message_data.get('reply_to') is not None

        feature_score = (has_attachments * 10) + (has_embeds * 5) + (is_reply * 5)

        # Server size context (if available)
        # This would require additional API calls to get member counts
        server_bonus = 10  # Default bonus

        return min(100, reaction_score + feature_score + server_bonus)

    def _calculate_influencer_score(self, message_data: Dict[str, Any]) -> float:
        """Calculate influencer score"""
        author_id = str(message_data.get('author_id', ''))

        # VIP user bonus
        if author_id in self.vip_users:
            return 100

        # Role-based scoring (would need additional API calls)
        # For now, use reaction patterns as proxy
        reactions = message_data.get('reactions', [])
        total_reactions = sum(r['count'] for r in reactions)

        if total_reactions > 20:
            return 80
        elif total_reactions > 10:
            return 60
        elif total_reactions > 5:
            return 40
        else:
            return 20

    def _calculate_discord_relevance_score(self, content: str, symbol: str) -> float:
        """Calculate relevance to trading symbol"""
        content_lower = content.lower()
        symbol_lower = symbol.lower()

        # Direct symbol mentions
        direct_mentions = content_lower.count(f'${symbol_lower}') * 25
        indirect_mentions = content_lower.count(symbol_lower) * 15

        # Trading context
        trading_terms = [
            'buy', 'sell', 'long', 'short', 'pump', 'dump', 'moon',
            'target', 'entry', 'exit', 'profit', 'loss'
        ]
        context_score = sum(8 for term in trading_terms if term in content_lower)

        # Price mentions
        price_mentions = len(re.findall(r'\$\d+', content)) * 10

        total_score = direct_mentions + indirect_mentions + context_score + price_mentions
        return min(100, total_score)

    def _calculate_urgency_score(self, content: str) -> float:
        """Calculate message urgency score"""
        content_lower = content.lower()

        urgency_indicators = {
            'now': 20, 'asap': 25, 'urgent': 25, 'breaking': 30,
            'quick': 15, 'fast': 15, 'immediately': 25,
            'alert': 20, 'warning': 15
        }

        urgency_score = 0
        for indicator, points in urgency_indicators.items():
            urgency_score += content_lower.count(indicator) * points

        # All caps indicates urgency
        caps_ratio = sum(1 for c in content if c.isupper()) / len(content) if content else 0
        if caps_ratio > 0.5:
            urgency_score += 20

        # Multiple exclamation marks
        exclamation_count = content.count('!')
        urgency_score += min(15, exclamation_count * 3)

        return min(100, urgency_score)

    def _calculate_signal_strength(
        self,
        message_data: Dict[str, Any],
        sentiment: SentimentScore
    ) -> float:
        """Calculate overall signal strength"""
        # Combine multiple factors
        community_score = self._calculate_discord_community_score(message_data)
        influencer_score = self._calculate_influencer_score(message_data)
        alpha_score = self._calculate_alpha_score(message_data.get('content', ''), message_data)

        # Weight factors
        signal_strength = (
            community_score * 0.25 +
            influencer_score * 0.35 +
            alpha_score * 0.25 +
            abs(sentiment.compound) * 100 * 0.15
        )

        return min(100, signal_strength)

    def _determine_asset_class(self, symbol: str, content: str, message_data: Dict[str, Any]) -> str:
        """Determine asset class from context"""
        content_lower = content.lower()

        # Server/channel context
        guild_name = message_data.get('guild_name', '').lower()
        channel_name = message_data.get('channel_name', '').lower()

        # Crypto indicators
        crypto_indicators = ['crypto', 'coin', 'token', 'defi', 'nft', 'blockchain']
        if (any(indicator in content_lower for indicator in crypto_indicators) or
            any(indicator in guild_name for indicator in crypto_indicators) or
            any(indicator in channel_name for indicator in crypto_indicators)):
            return 'crypto'

        # Short symbols likely crypto
        if len(symbol) <= 4:
            return 'crypto'

        # Stock indicators
        stock_indicators = ['stock', 'shares', 'earnings', 'nasdaq', 'nyse']
        if any(indicator in content_lower for indicator in stock_indicators):
            return 'stocks'

        # Default based on context
        if 'trading' in guild_name or 'trading' in channel_name:
            return 'crypto'  # Discord trading communities are often crypto-focused

        return 'crypto'  # Default for Discord

    def validate_data(self, data_point: DataPoint) -> Tuple[bool, float]:
        """Validate Discord data point quality"""
        try:
            raw_data = data_point.raw_data
            processed_data = data_point.processed_data

            # Basic validation
            if not raw_data.get('content') or not raw_data.get('message_id'):
                return False, 0.0

            # Quality components
            signal_strength = processed_data.get('signal_strength', 0) / 100.0
            influencer_score = processed_data.get('influencer_score', 0) / 100.0
            relevance_score = processed_data.get('relevance_score', 0) / 100.0
            sentiment_confidence = processed_data.get('sentiment', {}).get('confidence', 0)

            # Weighted quality score
            quality_score = (
                signal_strength * 0.3 +
                influencer_score * 0.25 +
                relevance_score * 0.25 +
                sentiment_confidence * 0.2
            )

            # Minimum thresholds
            is_valid = (
                quality_score >= 0.3 and
                relevance_score >= 0.1 and
                len(raw_data.get('content', '')) >= self.min_message_length
            )

            return is_valid, quality_score

        except Exception as e:
            self.logger.error(f"Failed to validate Discord data: {e}")
            return False, 0.0

    async def shutdown(self) -> None:
        """Shutdown Discord client"""
        if self.client and self.is_running:
            await self.client.close()
            self.is_running = False
            self.logger.info("Discord client shut down")