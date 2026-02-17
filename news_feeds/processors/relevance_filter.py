"""
Relevance Filter - Filter and rank news articles by relevance to trading
"""

from typing import List, Dict, Any
from datetime import datetime, timezone

class RelevanceFilter:
    """Filter and rank articles by trading relevance"""

    def __init__(self):
        # Trading-relevant keywords with weights
        self.trading_keywords = {
            # Market terms
            'earnings': 3.0, 'revenue': 2.5, 'profit': 2.5, 'loss': 2.0,
            'stock': 2.0, 'shares': 2.0, 'market': 1.5, 'trading': 3.0,

            # Economic indicators
            'gdp': 2.5, 'inflation': 3.0, 'unemployment': 2.5, 'fed': 3.0,
            'interest rate': 3.0, 'federal reserve': 3.0,

            # Sports betting terms
            'touchdown': 2.0, 'yards': 1.5, 'injury': 2.5, 'injured': 2.5,
            'questionable': 2.0, 'out': 2.0, 'performance': 1.8,

            # Crypto terms
            'bitcoin': 2.5, 'crypto': 2.5, 'blockchain': 2.0, 'ethereum': 2.5
        }

    def calculate_relevance(self, title: str, content: str, category: str) -> float:
        """Calculate relevance score for an article"""
        text = f"{title} {content}".lower()

        score = 0.0
        for keyword, weight in self.trading_keywords.items():
            if keyword in text:
                score += weight

        # Category bonuses
        if category in ['economics', 'earnings', 'crypto']:
            score *= 1.2
        elif category in ['sports']:
            score *= 1.1

        return score

    def filter_relevant(self, articles: List[Dict], min_score: float = 1.0) -> List[Dict]:
        """Filter articles by minimum relevance score"""
        relevant = []

        for article in articles:
            score = self.calculate_relevance(
                article.get('title', ''),
                article.get('content', ''),
                article.get('category', '')
            )

            if score >= min_score:
                article['relevance_score'] = score
                relevant.append(article)

        # Sort by relevance score
        return sorted(relevant, key=lambda x: x['relevance_score'], reverse=True)