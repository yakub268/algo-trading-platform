"""
Sentiment Analysis - Advanced sentiment analysis for news articles

Uses multiple methods to determine sentiment: keyword-based, pattern matching,
and basic NLP techniques for trading-relevant content.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    score: float          # -1.0 (very negative) to 1.0 (very positive)
    confidence: float     # 0.0 to 1.0
    category: str         # emotion category
    keywords_found: List[str]
    explanation: str

class SentimentAnalyzer:
    """Advanced sentiment analysis for trading news"""

    def __init__(self):
        # Trading-specific sentiment lexicon
        self.sentiment_lexicon = {
            # Very positive (2.0)
            'surge': 2.0, 'soar': 2.0, 'rocket': 2.0, 'moon': 2.0, 'breakout': 2.0,
            'explosion': 2.0, 'rally': 1.8, 'bullish': 1.8, 'boom': 1.8,

            # Positive (1.0 to 1.5)
            'gain': 1.5, 'rise': 1.3, 'increase': 1.2, 'up': 1.0, 'growth': 1.4,
            'profit': 1.5, 'buy': 1.2, 'long': 1.1, 'bull': 1.3, 'optimistic': 1.4,
            'strong': 1.2, 'outperform': 1.6, 'beat': 1.4, 'exceed': 1.3,

            # Neutral to slightly positive (0.5 to 1.0)
            'stable': 0.7, 'maintain': 0.6, 'hold': 0.5, 'steady': 0.6,

            # Negative (-1.0 to -1.5)
            'fall': -1.3, 'drop': -1.4, 'decline': -1.2, 'down': -1.0,
            'loss': -1.5, 'sell': -1.1, 'short': -1.2, 'bear': -1.3,
            'weak': -1.2, 'concern': -1.1, 'miss': -1.4, 'underperform': -1.6,

            # Very negative (-2.0)
            'crash': -2.0, 'plunge': -2.0, 'tank': -1.9, 'collapse': -2.0,
            'dump': -1.8, 'bearish': -1.8, 'panic': -2.0, 'disaster': -2.0
        }

        # Sports-specific sentiment
        self.sports_sentiment = {
            # Positive performance indicators
            'touchdown': 1.5, 'td': 1.5, 'rushing': 1.2, 'passing': 1.1,
            'yards': 1.0, 'completed': 1.2, 'victory': 1.8, 'win': 1.5,
            'dominant': 1.7, 'excellent': 1.6, 'outstanding': 1.8,

            # Negative performance indicators
            'injured': -1.8, 'injury': -1.8, 'questionable': -1.3, 'out': -1.5,
            'benched': -1.6, 'fumble': -1.4, 'interception': -1.5, 'sack': -1.2,
            'loss': -1.5, 'defeat': -1.4, 'poor': -1.3, 'struggle': -1.4
        }

        # Economic/Federal Reserve sentiment
        self.economic_sentiment = {
            # Positive economic indicators
            'growth': 1.4, 'expansion': 1.5, 'recovery': 1.6, 'improvement': 1.3,
            'strong economy': 1.7, 'job creation': 1.5, 'low unemployment': 1.4,

            # Negative economic indicators
            'recession': -2.0, 'inflation': -1.5, 'rate hike': -1.3, 'hawkish': -1.4,
            'tightening': -1.2, 'slowdown': -1.5, 'contraction': -1.7,
            'unemployment': -1.6, 'dovish': 0.8  # Actually positive for markets
        }

        # Combine all lexicons
        self.combined_lexicon = {
            **self.sentiment_lexicon,
            **self.sports_sentiment,
            **self.economic_sentiment
        }

        # Sentiment modifiers
        self.intensifiers = {
            'very': 1.3, 'extremely': 1.5, 'highly': 1.2, 'significantly': 1.3,
            'substantially': 1.4, 'dramatically': 1.5, 'massively': 1.6
        }

        self.diminishers = {
            'slightly': 0.7, 'somewhat': 0.8, 'moderately': 0.9, 'marginally': 0.6
        }

        self.negators = ['not', 'no', 'never', 'none', 'nothing', 'neither', 'without']

        # Emotional categories
        self.emotion_patterns = {
            'excitement': ['rocket', 'moon', 'explosion', 'surge', 'soar'],
            'optimism': ['bullish', 'growth', 'rally', 'strong', 'positive'],
            'fear': ['crash', 'panic', 'collapse', 'plunge', 'disaster'],
            'pessimism': ['bearish', 'decline', 'weak', 'concern', 'negative'],
            'uncertainty': ['volatile', 'unclear', 'uncertain', 'mixed', 'confused']
        }

    def analyze_sentiment(self, text: str, category: str = "general") -> SentimentResult:
        """Perform comprehensive sentiment analysis"""

        if not text or not text.strip():
            return SentimentResult(0.0, 0.0, "neutral", [], "Empty text")

        text_lower = text.lower()

        # Tokenize text
        words = re.findall(r'\b\w+\b', text_lower)

        # Calculate sentiment scores
        sentiment_score = 0.0
        keywords_found = []
        word_count = 0

        for i, word in enumerate(words):
            if word in self.combined_lexicon:
                base_score = self.combined_lexicon[word]

                # Apply modifiers
                modified_score = self._apply_modifiers(words, i, base_score)

                sentiment_score += modified_score
                keywords_found.append(word)
                word_count += 1

        # Normalize score
        if word_count > 0:
            sentiment_score = sentiment_score / word_count

        # Apply category-specific adjustments
        sentiment_score = self._apply_category_adjustments(sentiment_score, text_lower, category)

        # Ensure score is within bounds
        sentiment_score = max(-1.0, min(1.0, sentiment_score))

        # Calculate confidence
        confidence = self._calculate_confidence(word_count, len(words), keywords_found)

        # Determine emotion category
        emotion_category = self._determine_emotion_category(keywords_found)

        # Generate explanation
        explanation = self._generate_explanation(sentiment_score, keywords_found, emotion_category)

        return SentimentResult(
            score=sentiment_score,
            confidence=confidence,
            category=emotion_category,
            keywords_found=keywords_found,
            explanation=explanation
        )

    def _apply_modifiers(self, words: List[str], word_index: int, base_score: float) -> float:
        """Apply intensifiers, diminishers, and negators"""

        modified_score = base_score

        # Check for modifiers in the surrounding context
        context_start = max(0, word_index - 2)
        context_end = min(len(words), word_index + 1)
        context = words[context_start:context_end]

        # Apply intensifiers
        for modifier, multiplier in self.intensifiers.items():
            if modifier in context:
                modified_score *= multiplier
                break

        # Apply diminishers
        for modifier, multiplier in self.diminishers.items():
            if modifier in context:
                modified_score *= multiplier
                break

        # Apply negation
        for negator in self.negators:
            if negator in context:
                modified_score *= -0.8  # Flip and reduce intensity
                break

        return modified_score

    def _apply_category_adjustments(self, score: float, text: str, category: str) -> float:
        """Apply category-specific adjustments"""

        if category == "sports":
            # Sports news tends to be more dramatic, scale down slightly
            score *= 0.9

            # Player injury news is especially negative
            if any(word in text for word in ['injured', 'injury', 'out', 'questionable']):
                score -= 0.3

        elif category == "economics":
            # Economic news has more market impact, scale up
            score *= 1.1

            # Fed-related news is critical
            if any(word in text for word in ['fed', 'federal reserve', 'jerome powell', 'fomc']):
                score *= 1.2

        elif category == "crypto":
            # Crypto sentiment tends to be more volatile
            score *= 1.15

            # Regulatory news is especially impactful
            if any(word in text for word in ['regulation', 'ban', 'legal', 'sec']):
                if score < 0:
                    score *= 1.3  # Amplify negative regulatory sentiment

        return score

    def _calculate_confidence(self, sentiment_words: int, total_words: int, keywords: List[str]) -> float:
        """Calculate confidence in sentiment analysis"""

        if total_words == 0:
            return 0.0

        # Base confidence from sentiment word density
        density = sentiment_words / total_words
        base_confidence = min(density * 5, 1.0)  # Cap at 1.0

        # Boost confidence for strong sentiment words
        strong_words = [word for word in keywords if abs(self.combined_lexicon.get(word, 0)) >= 1.5]
        strong_word_boost = min(len(strong_words) * 0.1, 0.3)

        # Boost confidence for multiple sentiment indicators
        if sentiment_words >= 3:
            multiple_indicators_boost = 0.15
        else:
            multiple_indicators_boost = 0.0

        total_confidence = base_confidence + strong_word_boost + multiple_indicators_boost
        return min(total_confidence, 1.0)

    def _determine_emotion_category(self, keywords: List[str]) -> str:
        """Determine primary emotion category"""

        category_scores = {}

        for emotion, emotion_words in self.emotion_patterns.items():
            score = sum(1 for keyword in keywords if keyword in emotion_words)
            if score > 0:
                category_scores[emotion] = score

        if not category_scores:
            return "neutral"

        # Return the emotion with highest score
        return max(category_scores.items(), key=lambda x: x[1])[0]

    def _generate_explanation(self, score: float, keywords: List[str], category: str) -> str:
        """Generate human-readable explanation of sentiment"""

        if abs(score) < 0.1:
            return f"Neutral sentiment (category: {category})"

        intensity = "strong" if abs(score) > 0.6 else "moderate" if abs(score) > 0.3 else "weak"
        direction = "positive" if score > 0 else "negative"

        explanation = f"{intensity.title()} {direction} sentiment"

        if keywords:
            key_words = ', '.join(keywords[:3])  # Show top 3 keywords
            explanation += f" based on keywords: {key_words}"

        explanation += f" (category: {category})"

        return explanation

    def batch_analyze(self, texts: List[str], category: str = "general") -> List[SentimentResult]:
        """Analyze sentiment for multiple texts"""

        return [self.analyze_sentiment(text, category) for text in texts]

    def get_sentiment_summary(self, results: List[SentimentResult]) -> Dict[str, any]:
        """Generate summary statistics from multiple sentiment results"""

        if not results:
            return {
                'average_sentiment': 0.0,
                'average_confidence': 0.0,
                'dominant_category': 'neutral',
                'distribution': {},
                'total_analyzed': 0
            }

        scores = [r.score for r in results]
        confidences = [r.confidence for r in results]
        categories = [r.category for r in results]

        # Calculate distribution
        from collections import Counter
        category_distribution = Counter(categories)

        return {
            'average_sentiment': sum(scores) / len(scores),
            'average_confidence': sum(confidences) / len(confidences),
            'dominant_category': category_distribution.most_common(1)[0][0],
            'distribution': dict(category_distribution),
            'total_analyzed': len(results),
            'positive_ratio': len([s for s in scores if s > 0.1]) / len(scores),
            'negative_ratio': len([s for s in scores if s < -0.1]) / len(scores),
            'neutral_ratio': len([s for s in scores if -0.1 <= s <= 0.1]) / len(scores)
        }

def main():
    """Test sentiment analyzer"""

    analyzer = SentimentAnalyzer()

    test_texts = [
        "Drake Maye throws 3 touchdowns in dominant Patriots victory",
        "Kenneth Walker injured during practice, questionable for Sunday",
        "Stock market crashes as Fed raises interest rates dramatically",
        "Bitcoin surges to new highs on institutional adoption",
        "Company beats earnings expectations with strong growth"
    ]

    print("Testing Sentiment Analyzer")
    print("=" * 50)

    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze_sentiment(text)

        print(f"\\n{i}. \"{text}\"")
        print(f"   Sentiment: {result.score:.3f}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Category: {result.category}")
        print(f"   Keywords: {', '.join(result.keywords_found)}")
        print(f"   Explanation: {result.explanation}")

    # Test batch analysis
    print(f"\\n\\nBatch Analysis Summary:")
    results = analyzer.batch_analyze(test_texts)
    summary = analyzer.get_sentiment_summary(results)

    for key, value in summary.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()