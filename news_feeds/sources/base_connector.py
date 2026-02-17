"""
Base News Connector - Abstract base class for all news source connectors
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import requests
import time
import logging

logger = logging.getLogger(__name__)

class NewsArticle:
    """Standard news article format"""

    def __init__(self, title: str, content: str, source: str,
                 published_at: datetime, url: str = "",
                 category: str = "", sentiment: float = 0.0,
                 relevance_score: float = 0.0, tags: List[str] = None,
                 id: str = None, **kwargs):
        self.title = title
        self.content = content
        self.source = source
        self.published_at = published_at
        self.url = url
        self.category = category
        self.sentiment = sentiment
        self.relevance_score = relevance_score
        self.tags = tags or []
        self.id = id if id else self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID based on title and source"""
        import hashlib
        content = f"{self.title}{self.source}{self.published_at.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'source': self.source,
            'published_at': self.published_at.isoformat(),
            'url': self.url,
            'category': self.category,
            'sentiment': self.sentiment,
            'relevance_score': self.relevance_score,
            'tags': self.tags
        }

class BaseNewsConnector(ABC):
    """Abstract base class for news connectors"""

    def __init__(self, api_key: str = "", rate_limit: int = 100, cache_ttl: int = 300):
        self.api_key = api_key
        self.rate_limit = rate_limit  # requests per hour
        self.cache_ttl = cache_ttl    # seconds
        self.last_request_times = []
        self.session = requests.Session()

        # Set user agent
        self.session.headers.update({
            'User-Agent': 'TradingBot/1.0 (Educational Purpose)'
        })

    def _check_rate_limit(self) -> bool:
        """Check if we can make another request"""
        now = time.time()

        # Remove old requests (older than 1 hour)
        self.last_request_times = [
            req_time for req_time in self.last_request_times
            if now - req_time < 3600
        ]

        # Check if we've exceeded rate limit
        if len(self.last_request_times) >= self.rate_limit:
            return False

        return True

    def _make_request(self, url: str, params: Dict = None, headers: Dict = None) -> Optional[requests.Response]:
        """Make rate-limited HTTP request"""

        if not self._check_rate_limit():
            logger.warning(f"{self.__class__.__name__}: Rate limit exceeded")
            return None

        try:
            # Record request time
            self.last_request_times.append(time.time())

            # Add any custom headers
            request_headers = {}
            if headers:
                request_headers.update(headers)

            response = self.session.get(
                url,
                params=params or {},
                headers=request_headers,
                timeout=10
            )

            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                logger.warning(f"Rate limited by {self.source_name}")
                return None
            else:
                logger.error(f"HTTP {response.status_code} from {url}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the news source"""
        pass

    @property
    @abstractmethod
    def categories(self) -> List[str]:
        """Available categories for this source"""
        pass

    @abstractmethod
    def fetch_news(self, category: str = "general", limit: int = 20,
                   keywords: List[str] = None) -> List[NewsArticle]:
        """Fetch news articles from the source"""
        pass

    def test_connection(self) -> bool:
        """Test if the connector can successfully connect to the source"""
        try:
            articles = self.fetch_news(limit=1)
            return len(articles) >= 0  # Even 0 articles means successful connection
        except Exception as e:
            logger.error(f"Connection test failed for {self.source_name}: {e}")
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the connector"""
        return {
            'source': self.source_name,
            'connected': self.test_connection(),
            'rate_limit_remaining': max(0, self.rate_limit - len(self.last_request_times)),
            'last_request': self.last_request_times[-1] if self.last_request_times else None
        }