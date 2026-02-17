"""
Text Processor - Clean and normalize text content for analysis
"""

import re
import html
from typing import List, Dict, Optional

class TextProcessor:
    """Text cleaning and normalization utilities"""

    def __init__(self):
        # Common HTML entities
        self.html_entities = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&apos;': "'",
            '&nbsp;': ' '
        }

        # Words to remove (stopwords relevant to news)
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can'
        }

    def clean_html(self, text: str) -> str:
        """Remove HTML tags and decode entities"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Decode HTML entities
        text = html.unescape(text)

        return text.strip()

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_keywords(self, text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
        """Extract important keywords from text"""
        # Clean text
        clean_text = self.clean_html(text).lower()

        # Extract words (alphanumeric only)
        words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', clean_text)

        # Filter stopwords
        keywords = [word for word in words if word not in self.stopwords]

        # Count frequency
        from collections import Counter
        word_counts = Counter(keywords)

        # Return most frequent keywords
        return [word for word, count in word_counts.most_common(max_keywords)]