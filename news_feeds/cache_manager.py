"""
Cache Manager - Efficient caching system for news articles

Prevents duplicate API calls and provides fast access to recent news data.
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import logging
import hashlib

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching for news articles and API responses"""

    def __init__(self, cache_dir: str = "news_feeds/cache", default_ttl: int = 300):
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl  # seconds

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # In-memory cache for frequently accessed data
        self.memory_cache = {}
        self.cache_timestamps = {}

        # Cache files
        self.article_cache_file = os.path.join(cache_dir, "articles.json")
        self.api_cache_file = os.path.join(cache_dir, "api_responses.json")

        # Load existing cache data
        self._load_cache_files()

    def _load_cache_files(self):
        """Load existing cache data from files"""

        try:
            if os.path.exists(self.article_cache_file):
                with open(self.article_cache_file, 'r', encoding='utf-8') as f:
                    self.article_cache = json.load(f)
            else:
                self.article_cache = {}

            if os.path.exists(self.api_cache_file):
                with open(self.api_cache_file, 'r', encoding='utf-8') as f:
                    self.api_cache = json.load(f)
            else:
                self.api_cache = {}

        except Exception as e:
            logger.error(f"Failed to load cache files: {e}")
            self.article_cache = {}
            self.api_cache = {}

    def _save_cache_files(self):
        """Save cache data to files"""

        try:
            with open(self.article_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.article_cache, f, indent=2, default=str)

            with open(self.api_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.api_cache, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save cache files: {e}")

    def _generate_cache_key(self, source: str, category: str, keywords: List[str] = None) -> str:
        """Generate a unique cache key"""

        key_components = [source, category]

        if keywords:
            key_components.extend(sorted(keywords))

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: float, ttl: int = None) -> bool:
        """Check if cached data is still valid"""

        if ttl is None:
            ttl = self.default_ttl

        return (time.time() - timestamp) < ttl

    def get_cached_articles(self, source: str, category: str, keywords: List[str] = None,
                          ttl: int = None) -> Optional[List[Dict]]:
        """Get cached articles if available and valid"""

        cache_key = self._generate_cache_key(source, category, keywords)

        # Check memory cache first
        if cache_key in self.memory_cache:
            cache_data = self.memory_cache[cache_key]
            if self._is_cache_valid(cache_data['timestamp'], ttl):
                logger.debug(f"Memory cache hit for {cache_key}")
                return cache_data['articles']

        # Check file cache
        if cache_key in self.article_cache:
            cache_data = self.article_cache[cache_key]
            if self._is_cache_valid(cache_data['timestamp'], ttl):
                # Load into memory cache for faster future access
                self.memory_cache[cache_key] = cache_data
                logger.debug(f"File cache hit for {cache_key}")
                return cache_data['articles']

        logger.debug(f"Cache miss for {cache_key}")
        return None

    def cache_articles(self, articles: List[Dict], source: str, category: str,
                      keywords: List[str] = None, ttl: int = None):
        """Cache articles for future use"""

        cache_key = self._generate_cache_key(source, category, keywords)
        current_time = time.time()

        cache_data = {
            'articles': articles,
            'timestamp': current_time,
            'source': source,
            'category': category,
            'keywords': keywords or [],
            'ttl': ttl or self.default_ttl
        }

        # Store in memory cache
        self.memory_cache[cache_key] = cache_data

        # Store in file cache
        self.article_cache[cache_key] = cache_data

        # Save to file periodically (not every time for performance)
        if len(self.memory_cache) % 10 == 0:
            self._save_cache_files()

        logger.debug(f"Cached {len(articles)} articles for {cache_key}")

    def get_cached_api_response(self, url: str, params: Dict = None, ttl: int = None) -> Optional[Dict]:
        """Get cached API response"""

        # Create cache key from URL and parameters
        cache_key = hashlib.md5(f"{url}|{json.dumps(params or {}, sort_keys=True)}".encode()).hexdigest()

        if cache_key in self.api_cache:
            cache_data = self.api_cache[cache_key]
            if self._is_cache_valid(cache_data['timestamp'], ttl):
                logger.debug(f"API cache hit for {url}")
                return cache_data['response']

        return None

    def cache_api_response(self, url: str, response_data: Dict, params: Dict = None, ttl: int = None):
        """Cache API response"""

        cache_key = hashlib.md5(f"{url}|{json.dumps(params or {}, sort_keys=True)}".encode()).hexdigest()

        cache_data = {
            'response': response_data,
            'timestamp': time.time(),
            'url': url,
            'params': params or {},
            'ttl': ttl or self.default_ttl
        }

        self.api_cache[cache_key] = cache_data

        # Save periodically
        if len(self.api_cache) % 5 == 0:
            self._save_cache_files()

        logger.debug(f"Cached API response for {url}")

    def cleanup_expired_cache(self):
        """Remove expired cache entries"""

        current_time = time.time()
        expired_keys = []

        # Check article cache
        for key, data in self.article_cache.items():
            ttl = data.get('ttl', self.default_ttl)
            if not self._is_cache_valid(data['timestamp'], ttl):
                expired_keys.append(key)

        for key in expired_keys:
            del self.article_cache[key]
            if key in self.memory_cache:
                del self.memory_cache[key]

        # Check API cache
        expired_keys = []
        for key, data in self.api_cache.items():
            ttl = data.get('ttl', self.default_ttl)
            if not self._is_cache_valid(data['timestamp'], ttl):
                expired_keys.append(key)

        for key in expired_keys:
            del self.api_cache[key]

        if expired_keys:
            self._save_cache_files()
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""

        total_articles = sum(len(data['articles']) for data in self.article_cache.values())
        cache_size_mb = self._calculate_cache_size()

        return {
            'article_cache_entries': len(self.article_cache),
            'api_cache_entries': len(self.api_cache),
            'memory_cache_entries': len(self.memory_cache),
            'total_cached_articles': total_articles,
            'cache_size_mb': cache_size_mb,
            'cache_directory': self.cache_dir
        }

    def _calculate_cache_size(self) -> float:
        """Calculate total cache size in MB"""

        total_size = 0

        try:
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)

            return total_size / (1024 * 1024)  # Convert to MB

        except Exception:
            return 0.0

    def clear_cache(self):
        """Clear all cached data"""

        self.memory_cache.clear()
        self.article_cache.clear()
        self.api_cache.clear()

        # Remove cache files
        try:
            if os.path.exists(self.article_cache_file):
                os.remove(self.article_cache_file)

            if os.path.exists(self.api_cache_file):
                os.remove(self.api_cache_file)

        except Exception as e:
            logger.error(f"Failed to remove cache files: {e}")

        logger.info("Cache cleared")

    def save_cache(self):
        """Force save cache to files"""

        self._save_cache_files()
        logger.info("Cache saved to files")

def main():
    """Test cache manager"""

    logging.basicConfig(level=logging.DEBUG)

    print("Testing Cache Manager")
    print("=" * 30)

    cache = CacheManager()

    # Test article caching
    test_articles = [
        {
            'title': 'Test Article 1',
            'content': 'Test content 1',
            'source': 'Test',
            'published_at': datetime.now().isoformat()
        },
        {
            'title': 'Test Article 2',
            'content': 'Test content 2',
            'source': 'Test',
            'published_at': datetime.now().isoformat()
        }
    ]

    # Cache articles
    cache.cache_articles(test_articles, 'ESPN', 'sports', ['player1', 'player2'])

    # Retrieve from cache
    cached_articles = cache.get_cached_articles('ESPN', 'sports', ['player1', 'player2'])
    print(f"Retrieved {len(cached_articles) if cached_articles else 0} articles from cache")

    # Test API response caching
    test_response = {'status': 'success', 'data': [1, 2, 3]}
    cache.cache_api_response('https://api.example.com/test', test_response)

    cached_response = cache.get_cached_api_response('https://api.example.com/test')
    print(f"Retrieved API response from cache: {cached_response is not None}")

    # Get stats
    stats = cache.get_cache_stats()
    print(f"\\nCache Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Cleanup
    cache.cleanup_expired_cache()

if __name__ == "__main__":
    main()