"""
Crypto Sentiment Analysis Bot

Uses FinBERT and social media signals to detect sentiment extremes.
Combines with RSI for confirmation.

Data Sources:
- Reddit (r/wallstreetbets, r/cryptocurrency, r/stocks)
- StockTwits API
- News headlines

Strategy Logic:
- Calculate rolling sentiment score per ticker
- Signal when sentiment z-score > 2 (extreme bullish) or < -2 (extreme bearish)
- Combine with RSI for confirmation (avoid buying overbought on bullish sentiment)

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import json
import logging
import sqlite3
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SentimentBot')

# Constants
CACHE_DIR = Path(__file__).parent.parent / "data" / "sentiment_cache"
DB_PATH = Path(__file__).parent.parent / "data" / "event_trades.db"


@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    symbol: str
    timestamp: datetime
    score: float  # -1 (bearish) to +1 (bullish)
    magnitude: float  # Confidence/strength
    source: str
    text_sample: str
    post_count: int


@dataclass
class SentimentSignal:
    """Trading signal based on sentiment"""
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    sentiment_score: float
    sentiment_zscore: float
    rsi: Optional[float]
    confidence: float
    reasoning: str
    timestamp: datetime


class SentimentAnalyzer:
    """
    Multi-source sentiment analyzer using FinBERT and rule-based methods.
    """
    
    # Sentiment keywords (rule-based fallback)
    BULLISH_KEYWORDS = [
        'moon', 'rocket', 'bullish', 'buy', 'long', 'calls', 'pump',
        'breakout', 'green', 'gains', 'profit', 'ATH', 'undervalued',
        'accumulate', 'hodl', 'diamond hands', '🚀', '📈', '💎'
    ]
    
    BEARISH_KEYWORDS = [
        'crash', 'dump', 'bearish', 'sell', 'short', 'puts', 'red',
        'loss', 'overvalued', 'bubble', 'correction', 'dead', 'rekt',
        'paper hands', 'rug pull', '📉', '🔻', '💀'
    ]
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.finbert_available = False
        
        # Try to load FinBERT
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.finbert_available = True
            logger.info("FinBERT loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT not available, using rule-based: {e}")
    
    def analyze_text(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text.
        
        Returns:
            Tuple of (score, magnitude) where score is -1 to +1
        """
        if self.finbert_available:
            return self._finbert_analyze(text)
        else:
            return self._rule_based_analyze(text)
    
    def _finbert_analyze(self, text: str) -> Tuple[float, float]:
        """Analyze using FinBERT model"""
        try:
            import torch
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.detach().numpy()[0]
            
            # FinBERT classes: negative, neutral, positive
            negative, neutral, positive = probs
            
            # Convert to -1 to +1 scale
            score = positive - negative
            magnitude = 1 - neutral  # Higher magnitude = more confident
            
            return float(score), float(magnitude)
        except Exception as e:
            logger.warning(f"FinBERT failed: {e}")
            return self._rule_based_analyze(text)
    
    def _rule_based_analyze(self, text: str) -> Tuple[float, float]:
        """Simple rule-based sentiment analysis"""
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw.lower() in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw.lower() in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, 0.0
        
        score = (bullish_count - bearish_count) / total
        magnitude = min(total / 5, 1.0)  # Cap at 1.0
        
        return score, magnitude


class RedditScraper:
    """Scrape sentiment from Reddit"""
    
    SUBREDDITS = ['wallstreetbets', 'stocks', 'investing', 'cryptocurrency', 'CryptoMarkets']
    
    def __init__(self):
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = os.getenv('REDDIT_USER_AGENT', 'TradingBot/1.0')
        self.access_token = None
        self.token_expiry = None
    
    def _get_access_token(self) -> Optional[str]:
        """Get Reddit OAuth token"""
        if not self.client_id or not self.client_secret:
            return None
        
        if self.access_token and self.token_expiry and datetime.now() < self.token_expiry:
            return self.access_token
        
        try:
            auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
            data = {'grant_type': 'client_credentials'}
            headers = {'User-Agent': self.user_agent}
            
            response = requests.post(
                'https://www.reddit.com/api/v1/access_token',
                auth=auth, data=data, headers=headers, timeout=10
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                self.token_expiry = datetime.now() + timedelta(seconds=token_data['expires_in'] - 60)
                return self.access_token
        except Exception as e:
            logger.warning(f"Reddit auth failed: {e}")
        
        return None
    
    def get_posts(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get Reddit posts mentioning a symbol"""
        posts = []
        
        # Try authenticated API first
        token = self._get_access_token()
        
        for subreddit in self.SUBREDDITS:
            try:
                if token:
                    headers = {
                        'Authorization': f'Bearer {token}',
                        'User-Agent': self.user_agent
                    }
                    url = f'https://oauth.reddit.com/r/{subreddit}/search'
                    params = {'q': symbol, 'limit': limit // len(self.SUBREDDITS), 'sort': 'new', 't': 'day'}
                else:
                    # Fallback to public JSON API
                    headers = {'User-Agent': self.user_agent}
                    url = f'https://www.reddit.com/r/{subreddit}/search.json'
                    params = {'q': symbol, 'limit': limit // len(self.SUBREDDITS), 'sort': 'new', 't': 'day'}
                
                response = requests.get(url, headers=headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    for post in data.get('data', {}).get('children', []):
                        post_data = post.get('data', {})
                        posts.append({
                            'title': post_data.get('title', ''),
                            'selftext': post_data.get('selftext', ''),
                            'score': post_data.get('score', 0),
                            'created_utc': post_data.get('created_utc', 0),
                            'subreddit': subreddit,
                            'num_comments': post_data.get('num_comments', 0)
                        })
            except Exception as e:
                logger.warning(f"Reddit scrape failed for r/{subreddit}: {e}")
        
        return posts


class StockTwitsScraper:
    """Scrape sentiment from StockTwits"""
    
    BASE_URL = "https://api.stocktwits.com/api/2"
    
    def get_messages(self, symbol: str, limit: int = 30) -> List[Dict]:
        """Get StockTwits messages for a symbol"""
        try:
            url = f"{self.BASE_URL}/streams/symbol/{symbol}.json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                messages = []
                
                for msg in data.get('messages', [])[:limit]:
                    sentiment = msg.get('entities', {}).get('sentiment', {})
                    messages.append({
                        'body': msg.get('body', ''),
                        'created_at': msg.get('created_at', ''),
                        'sentiment': sentiment.get('basic') if sentiment else None,
                        'likes': msg.get('likes', {}).get('total', 0)
                    })
                
                return messages
        except Exception as e:
            logger.warning(f"StockTwits scrape failed: {e}")
        
        return []


class SentimentBot:
    """
    Sentiment-Based Trading Bot
    
    Aggregates sentiment from multiple sources and generates trading signals
    when sentiment reaches extreme levels.
    
    Entry Criteria:
    1. Sentiment z-score > 2 (bullish) or < -2 (bearish)
    2. RSI confirmation (not overbought for buys, not oversold for sells)
    3. Minimum post volume threshold
    
    Risk Management:
    - Small position sizes (2% per trade)
    - Sentiment mean-reverts quickly - short holding periods
    """
    
    ZSCORE_THRESHOLD = 2.0
    MIN_POST_COUNT = 10
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    POSITION_SIZE_PCT = 0.02
    LOOKBACK_DAYS = 30
    
    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.analyzer = SentimentAnalyzer()
        self.reddit = RedditScraper()
        self.stocktwits = StockTwitsScraper()
        
        # Historical sentiment for z-score calculation
        self.sentiment_history: Dict[str, deque] = {}
        
        # Create cache directory
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"SentimentBot initialized (paper={paper_mode})")
    
    def _init_database(self):
        """Initialize sentiment database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT,
                score REAL,
                magnitude REAL,
                zscore REAL,
                source TEXT,
                post_count INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT,
                signal_type TEXT,
                sentiment_score REAL,
                sentiment_zscore REAL,
                rsi REAL,
                confidence REAL,
                reasoning TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_aggregated_sentiment(self, symbol: str) -> SentimentScore:
        """
        Get aggregated sentiment from all sources.
        
        Args:
            symbol: Stock/crypto ticker
            
        Returns:
            SentimentScore with aggregated data
        """
        all_texts = []
        all_scores = []
        
        # Reddit
        reddit_posts = self.reddit.get_posts(symbol)
        for post in reddit_posts:
            text = f"{post['title']} {post['selftext']}"
            if text.strip():
                all_texts.append(text)
                score, magnitude = self.analyzer.analyze_text(text)
                # Weight by engagement
                weight = 1 + np.log1p(post['score'] + post['num_comments'])
                all_scores.append((score, magnitude, weight))
        
        # StockTwits
        st_messages = self.stocktwits.get_messages(symbol)
        for msg in st_messages:
            text = msg['body']
            if text:
                all_texts.append(text)
                
                # Use StockTwits sentiment if available
                if msg['sentiment'] == 'Bullish':
                    score = 0.8
                elif msg['sentiment'] == 'Bearish':
                    score = -0.8
                else:
                    score, _ = self.analyzer.analyze_text(text)
                
                weight = 1 + np.log1p(msg['likes'])
                all_scores.append((score, 0.7, weight))
        
        # Calculate weighted average
        if all_scores:
            total_weight = sum(s[2] for s in all_scores)
            weighted_score = sum(s[0] * s[2] for s in all_scores) / total_weight
            avg_magnitude = sum(s[1] * s[2] for s in all_scores) / total_weight
        else:
            weighted_score = 0.0
            avg_magnitude = 0.0
        
        return SentimentScore(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            score=weighted_score,
            magnitude=avg_magnitude,
            source="aggregated",
            text_sample=all_texts[0][:200] if all_texts else "",
            post_count=len(all_texts)
        )
    
    def calculate_zscore(self, symbol: str, current_score: float) -> float:
        """Calculate z-score of current sentiment vs historical"""
        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = deque(maxlen=self.LOOKBACK_DAYS)
        
        history = self.sentiment_history[symbol]
        
        if len(history) < 5:
            # Not enough history
            return 0.0
        
        mean = np.mean(history)
        std = np.std(history)
        
        if std == 0:
            return 0.0
        
        return (current_score - mean) / std
    
    def get_rsi(self, symbol: str) -> Optional[float]:
        """Get current RSI for symbol"""
        try:
            import yfinance as yf
            
            data = yf.download(symbol, period="1mo", interval="1d", progress=False)
            if len(data) < 14:
                return None
            
            close = data['Close']
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            
            avg_gain = gain.ewm(com=13, min_periods=14).mean()
            avg_loss = loss.ewm(com=13, min_periods=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
        except Exception as e:
            logger.warning(f"Failed to get RSI for {symbol}: {e}")
            return None
    
    def generate_signal(self, symbol: str) -> SentimentSignal:
        """
        Generate trading signal based on sentiment.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            SentimentSignal with recommendation
        """
        # Get current sentiment
        sentiment = self.get_aggregated_sentiment(symbol)
        
        # Update history
        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = deque(maxlen=self.LOOKBACK_DAYS)
        self.sentiment_history[symbol].append(sentiment.score)
        
        # Calculate z-score
        zscore = self.calculate_zscore(symbol, sentiment.score)
        
        # Get RSI for confirmation
        rsi = self.get_rsi(symbol)
        
        # Default to hold
        signal_type = "hold"
        confidence = 0.5
        reasoning = f"Sentiment score: {sentiment.score:.2f}, z-score: {zscore:.2f}"
        
        # Check for signals
        if sentiment.post_count < self.MIN_POST_COUNT:
            reasoning = f"Insufficient data: only {sentiment.post_count} posts"
        
        elif zscore > self.ZSCORE_THRESHOLD:
            # Extreme bullish sentiment
            if rsi and rsi < self.RSI_OVERBOUGHT:
                signal_type = "buy"
                confidence = min(0.5 + abs(zscore - self.ZSCORE_THRESHOLD) * 0.1, 0.85)
                reasoning = f"Extreme bullish sentiment (z={zscore:.1f}), RSI={rsi:.0f} not overbought"
            else:
                reasoning = f"Bullish sentiment but RSI={rsi:.0f} overbought - wait for pullback"
        
        elif zscore < -self.ZSCORE_THRESHOLD:
            # Extreme bearish sentiment - contrarian buy opportunity
            if rsi and rsi < self.RSI_OVERSOLD:
                signal_type = "buy"  # Contrarian
                confidence = min(0.5 + abs(zscore + self.ZSCORE_THRESHOLD) * 0.1, 0.80)
                reasoning = f"Extreme fear (z={zscore:.1f}), RSI={rsi:.0f} oversold - contrarian buy"
            else:
                reasoning = f"Bearish sentiment (z={zscore:.1f}), waiting for RSI oversold"
        
        signal = SentimentSignal(
            symbol=symbol,
            signal_type=signal_type,
            sentiment_score=sentiment.score,
            sentiment_zscore=zscore,
            rsi=rsi,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Save to database
        self._save_signal(signal)
        
        return signal
    
    def _save_signal(self, signal: SentimentSignal):
        """Save signal to database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sentiment_signals 
            (symbol, timestamp, signal_type, sentiment_score, sentiment_zscore, rsi, confidence, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.symbol,
            signal.timestamp.isoformat(),
            signal.signal_type,
            signal.sentiment_score,
            signal.sentiment_zscore,
            signal.rsi,
            signal.confidence,
            signal.reasoning
        ))
        
        conn.commit()
        conn.close()
    
    def scan(self) -> List[SentimentSignal]:
        """Scan default watchlist for sentiment signals (orchestrator entry point)."""
        default_symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'BTC-USD', 'ETH-USD']
        return self.scan_watchlist(default_symbols)

    def scan_watchlist(self, symbols: List[str]) -> List[SentimentSignal]:
        """
        Scan multiple symbols for sentiment signals.
        
        Args:
            symbols: List of tickers to scan
            
        Returns:
            List of signals with recommendations
        """
        signals = []
        
        for symbol in symbols:
            try:
                signal = self.generate_signal(symbol)
                signals.append(signal)
                
                if signal.signal_type != "hold":
                    logger.info(f"{symbol}: {signal.signal_type.upper()} - {signal.reasoning}")
                    
            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals


def send_telegram_alert(message: str):
    """Send Telegram alert"""
    try:
        from utils.telegram_bot import TelegramBot
        bot = TelegramBot()
        bot.send_message(message)
    except Exception as e:
        logger.warning(f"Telegram failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("SENTIMENT ANALYSIS BOT")
    print("=" * 60)
    
    bot = SentimentBot(paper_mode=True)
    
    # Test watchlist
    watchlist = ['SPY', 'AAPL', 'TSLA', 'BTC-USD', 'ETH-USD']
    
    print(f"\nScanning {len(watchlist)} symbols...")
    signals = bot.scan_watchlist(watchlist)
    
    print(f"\nSignals:")
    for signal in signals:
        print(f"  {signal.symbol}: {signal.signal_type.upper()}")
        print(f"    Sentiment: {signal.sentiment_score:.2f} (z={signal.sentiment_zscore:.1f})")
        print(f"    RSI: {signal.rsi:.0f}" if signal.rsi else "    RSI: N/A")
        print(f"    Confidence: {signal.confidence:.0%}")
        print(f"    {signal.reasoning}")
        print()
