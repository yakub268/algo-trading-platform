"""
Sentiment Analyzer
==================

Advanced sentiment analysis using FinBERT and other NLP models
specifically trained for financial text analysis.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


@dataclass
class SentimentScore:
    """Comprehensive sentiment analysis result"""
    text: str
    positive: float = 0.0
    negative: float = 0.0
    neutral: float = 0.0
    compound: float = 0.0  # Overall sentiment score
    confidence: float = 0.0
    financial_sentiment: Optional[str] = None
    keywords: List[str] = None
    emotion: Optional[str] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

    @property
    def dominant_sentiment(self) -> str:
        """Get dominant sentiment label"""
        scores = {
            'positive': self.positive,
            'negative': self.negative,
            'neutral': self.neutral
        }
        return max(scores.items(), key=lambda x: x[1])[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'positive': self.positive,
            'negative': self.negative,
            'neutral': self.neutral,
            'compound': self.compound,
            'confidence': self.confidence,
            'dominant': self.dominant_sentiment,
            'financial_sentiment': self.financial_sentiment,
            'keywords': self.keywords,
            'emotion': self.emotion
        }


class SentimentAnalyzer:
    """
    Advanced sentiment analyzer for financial text

    Features:
    - FinBERT for financial sentiment
    - TextBlob as fallback
    - Keyword extraction
    - Emotion detection
    - Market-specific sentiment scoring
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.logger = logging.getLogger("altdata.sentiment")
        self.model_name = model_name
        self.finbert_pipeline = None
        self.tokenizer = None
        self.model = None

        # Financial keywords for enhancement
        self.positive_keywords = {
            'bullish', 'moon', 'pump', 'rally', 'surge', 'breakout', 'support',
            'accumulate', 'buy', 'long', 'calls', 'momentum', 'strength',
            'green', 'profit', 'gain', 'up', 'rise', 'bull', 'uptrend'
        }

        self.negative_keywords = {
            'bearish', 'dump', 'crash', 'fall', 'drop', 'resistance', 'sell',
            'short', 'puts', 'weakness', 'red', 'loss', 'lose', 'down',
            'decline', 'bear', 'downtrend', 'correction', 'panic'
        }

        self.neutral_keywords = {
            'sideways', 'consolidation', 'range', 'flat', 'stable', 'hold',
            'wait', 'watch', 'neutral', 'unchanged'
        }

        # Crypto-specific terms
        self.crypto_terms = {
            'hodl', 'diamond hands', 'paper hands', 'ape', 'fud', 'fomo',
            'shitcoin', 'altcoin', 'defi', 'nft', 'whale', 'bag', 'rekt'
        }

        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize sentiment analysis models"""
        if TRANSFORMERS_AVAILABLE:
            try:
                self.finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.logger.info(f"Initialized FinBERT model: {self.model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load FinBERT: {e}")
                self.finbert_pipeline = None

        if not TRANSFORMERS_AVAILABLE or self.finbert_pipeline is None:
            if TEXTBLOB_AVAILABLE:
                self.logger.info("Using TextBlob as fallback sentiment analyzer")
            else:
                self.logger.error("No sentiment analysis libraries available")

    def analyze_sentiment(self, text: str, symbol: Optional[str] = None) -> SentimentScore:
        """
        Analyze sentiment of text

        Args:
            text: Text to analyze
            symbol: Optional symbol for context-specific analysis

        Returns:
            SentimentScore with comprehensive sentiment analysis
        """
        if not text or not text.strip():
            return SentimentScore(text=text)

        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)

        result = SentimentScore(text=text)

        # Try FinBERT first
        if self.finbert_pipeline:
            result = self._analyze_with_finbert(cleaned_text, result)
        elif TEXTBLOB_AVAILABLE:
            result = self._analyze_with_textblob(cleaned_text, result)

        # Extract keywords and enhance sentiment
        result.keywords = self._extract_keywords(text)
        result = self._enhance_with_keywords(result)

        # Add symbol-specific context
        if symbol:
            result = self._add_symbol_context(result, symbol, text)

        # Detect emotion
        result.emotion = self._detect_emotion(text)

        return result

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'[@#](\w+)', r'\\1', text)

        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text).strip()

        # Handle emojis and special characters
        # Keep basic punctuation but remove excessive symbols
        text = re.sub(r'[🚀📈📉💎🦍🌙]+', '', text)  # Common trading emojis

        return text[:512]  # Limit length for model input

    def _analyze_with_finbert(self, text: str, result: SentimentScore) -> SentimentScore:
        """Analyze sentiment using FinBERT"""
        try:
            finbert_result = self.finbert_pipeline(text)[0]

            label_mapping = {
                'LABEL_0': 'negative',  # or 'bearish'
                'LABEL_1': 'neutral',   # or 'neutral'
                'LABEL_2': 'positive',  # or 'bullish'
                'negative': 'negative',
                'neutral': 'neutral',
                'positive': 'positive'
            }

            confidence = finbert_result['score']
            label = label_mapping.get(finbert_result['label'].lower(), 'neutral')

            # Set scores based on FinBERT output
            if label == 'positive':
                result.positive = confidence
                result.negative = (1 - confidence) * 0.5
                result.neutral = (1 - confidence) * 0.5
            elif label == 'negative':
                result.negative = confidence
                result.positive = (1 - confidence) * 0.5
                result.neutral = (1 - confidence) * 0.5
            else:
                result.neutral = confidence
                result.positive = (1 - confidence) * 0.5
                result.negative = (1 - confidence) * 0.5

            result.confidence = confidence
            result.financial_sentiment = label

            # Calculate compound score
            result.compound = result.positive - result.negative

        except Exception as e:
            self.logger.error(f"FinBERT analysis failed: {e}")
            # Fall back to TextBlob
            if TEXTBLOB_AVAILABLE:
                result = self._analyze_with_textblob(text, result)

        return result

    def _analyze_with_textblob(self, text: str, result: SentimentScore) -> SentimentScore:
        """Analyze sentiment using TextBlob as fallback"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1

            # Convert polarity to positive/negative/neutral
            if polarity > 0.1:
                result.positive = (polarity + 1) / 2
                result.negative = (1 - result.positive) * 0.5
                result.neutral = 1 - result.positive - result.negative
            elif polarity < -0.1:
                result.negative = (1 - polarity) / 2
                result.positive = (1 - result.negative) * 0.5
                result.neutral = 1 - result.positive - result.negative
            else:
                result.neutral = 0.7
                result.positive = 0.15
                result.negative = 0.15

            result.compound = polarity
            result.confidence = 1 - subjectivity  # More objective = higher confidence

        except Exception as e:
            self.logger.error(f"TextBlob analysis failed: {e}")
            # Return neutral sentiment as last resort
            result.positive = result.negative = result.neutral = 0.33
            result.compound = 0.0
            result.confidence = 0.1

        return result

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        text_lower = text.lower()
        found_keywords = []

        # Find financial keywords
        all_keywords = self.positive_keywords | self.negative_keywords | self.neutral_keywords | self.crypto_terms

        for keyword in all_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        # Extract ticker symbols
        ticker_pattern = r'\\$([A-Z]{1,5})\\b'
        tickers = re.findall(ticker_pattern, text)
        found_keywords.extend(tickers)

        # Extract crypto symbols
        crypto_pattern = r'\\b([A-Z]{3,4})\\b(?=.*(?:coin|token|crypto))'
        cryptos = re.findall(crypto_pattern, text, re.IGNORECASE)
        found_keywords.extend(cryptos)

        return list(set(found_keywords))  # Remove duplicates

    def _enhance_with_keywords(self, result: SentimentScore) -> SentimentScore:
        """Enhance sentiment analysis with keyword-based adjustments"""
        positive_count = sum(1 for kw in result.keywords if kw.lower() in self.positive_keywords)
        negative_count = sum(1 for kw in result.keywords if kw.lower() in self.negative_keywords)

        if positive_count > negative_count:
            # Boost positive sentiment
            boost = min(0.1, (positive_count - negative_count) * 0.05)
            result.positive = min(1.0, result.positive + boost)
            result.negative = max(0.0, result.negative - boost * 0.5)
        elif negative_count > positive_count:
            # Boost negative sentiment
            boost = min(0.1, (negative_count - positive_count) * 0.05)
            result.negative = min(1.0, result.negative + boost)
            result.positive = max(0.0, result.positive - boost * 0.5)

        # Recalculate compound
        result.compound = result.positive - result.negative

        return result

    def _add_symbol_context(self, result: SentimentScore, symbol: str, text: str) -> SentimentScore:
        """Add symbol-specific context to sentiment analysis"""
        symbol_mentions = text.lower().count(symbol.lower())

        if symbol_mentions > 1:
            # Multiple mentions suggest higher relevance
            result.confidence = min(1.0, result.confidence * 1.2)

        # Check for symbol-specific sentiment patterns
        symbol_pattern = f"\\${symbol}|{symbol}\\s+(?:to the moon|moon|pump|dump)"
        if re.search(symbol_pattern, text, re.IGNORECASE):
            # Direct symbol mention with action words
            result.confidence = min(1.0, result.confidence * 1.1)

        return result

    def _detect_emotion(self, text: str) -> Optional[str]:
        """Detect emotion from text"""
        emotions = {
            'fear': ['scared', 'afraid', 'panic', 'worried', 'anxious', 'terrified'],
            'greed': ['moon', 'lambo', 'rich', 'wealthy', 'profit', 'gains'],
            'excitement': ['excited', 'pumped', 'hyped', 'amazing', 'incredible'],
            'anger': ['angry', 'mad', 'furious', 'pissed', 'hate'],
            'sadness': ['sad', 'depressed', 'disappointed', 'crushed'],
            'hope': ['hope', 'optimistic', 'confident', 'believe']
        }

        text_lower = text.lower()
        emotion_scores = {}

        for emotion, words in emotions.items():
            score = sum(1 for word in words if word in text_lower)
            if score > 0:
                emotion_scores[emotion] = score

        if emotion_scores:
            return max(emotion_scores.items(), key=lambda x: x[1])[0]

        return None

    def analyze_batch(self, texts: List[str], symbols: Optional[List[str]] = None) -> List[SentimentScore]:
        """Analyze sentiment for multiple texts"""
        if symbols is None:
            symbols = [None] * len(texts)

        results = []
        for text, symbol in zip(texts, symbols):
            result = self.analyze_sentiment(text, symbol)
            results.append(result)

        return results

    def get_aggregate_sentiment(self, scores: List[SentimentScore]) -> SentimentScore:
        """Calculate aggregate sentiment from multiple scores"""
        if not scores:
            return SentimentScore(text="")

        # Weight by confidence
        total_weight = sum(s.confidence for s in scores)
        if total_weight == 0:
            total_weight = len(scores)

        weighted_positive = sum(s.positive * s.confidence for s in scores) / total_weight
        weighted_negative = sum(s.negative * s.confidence for s in scores) / total_weight
        weighted_neutral = sum(s.neutral * s.confidence for s in scores) / total_weight
        weighted_compound = sum(s.compound * s.confidence for s in scores) / total_weight

        avg_confidence = sum(s.confidence for s in scores) / len(scores)

        # Aggregate keywords
        all_keywords = []
        for s in scores:
            all_keywords.extend(s.keywords)
        unique_keywords = list(set(all_keywords))

        return SentimentScore(
            text=f"Aggregate of {len(scores)} texts",
            positive=weighted_positive,
            negative=weighted_negative,
            neutral=weighted_neutral,
            compound=weighted_compound,
            confidence=avg_confidence,
            keywords=unique_keywords
        )