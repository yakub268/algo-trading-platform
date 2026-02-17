"""
Cache Manager
=============

Intelligent caching system for alternative data with TTL,
compression, and cost-optimization features.
"""

import json
import time
import gzip
import pickle
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import threading
from dataclasses import asdict

from .base_connector import DataPoint, DataSource


class CacheEntry:
    """Single cache entry with metadata"""

    def __init__(
        self,
        data: Any,
        ttl_seconds: int,
        compressed: bool = False,
        cost: float = 0.0
    ):
        self.data = data
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds
        self.compressed = compressed
        self.cost = cost
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def access(self) -> Any:
        self.access_count += 1
        self.last_accessed = time.time()
        return self.data


class CacheManager:
    """
    Intelligent cache manager for alternative data

    Features:
    - TTL-based expiration
    - LRU eviction policy
    - Compression for large data
    - Cost tracking and optimization
    - Thread-safe operations
    - Persistence to disk
    """

    def __init__(
        self,
        max_size_mb: int = 1024,  # 1GB cache
        default_ttl_minutes: int = 5,
        compression_threshold_kb: int = 100,
        cache_dir: Optional[str] = None
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_minutes * 60
        self.compression_threshold_bytes = compression_threshold_kb * 1024

        self.cache: Dict[str, CacheEntry] = {}
        self.current_size_bytes = 0
        self.lock = threading.RLock()

        # Cache directory for persistence
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/altdata")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_cost_saved = 0.0

        self.logger = logging.getLogger("altdata.cache")
        self._load_persistent_cache()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                self.misses += 1
                return None

            # Update access statistics
            data = entry.access()
            self.hits += 1

            # Track cost savings
            self.total_cost_saved += entry.cost

            self.logger.debug(f"Cache hit for key: {key[:50]}...")
            return data

    def put(
        self,
        key: str,
        data: Any,
        ttl_minutes: Optional[int] = None,
        cost: float = 0.0
    ) -> None:
        """Put item in cache"""
        ttl_seconds = (ttl_minutes or self.default_ttl_seconds // 60) * 60

        with self.lock:
            # Serialize and optionally compress data
            serialized_data = self._serialize_data(data)
            compressed = len(serialized_data) > self.compression_threshold_bytes

            if compressed:
                serialized_data = gzip.compress(serialized_data)

            # Calculate data size
            data_size = len(serialized_data)

            # Make room if necessary
            self._make_room(data_size)

            # Store entry
            entry = CacheEntry(
                data=serialized_data,
                ttl_seconds=ttl_seconds,
                compressed=compressed,
                cost=cost
            )

            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)

            self.cache[key] = entry
            self.current_size_bytes += data_size

            self.logger.debug(
                f"Cached data with key: {key[:50]}... "
                f"(size: {data_size} bytes, compressed: {compressed})"
            )

    def get_datapoints(
        self,
        source: DataSource,
        symbols: List[str],
        lookback_hours: int = 24
    ) -> List[DataPoint]:
        """Get cached data points for specific query"""
        cache_key = self._generate_datapoint_key(source, symbols, lookback_hours)
        cached_data = self.get(cache_key)

        if cached_data is None:
            return []

        # Deserialize DataPoint objects
        try:
            return [self._deserialize_datapoint(dp_data) for dp_data in cached_data]
        except Exception as e:
            self.logger.warning(f"Failed to deserialize cached data points: {e}")
            self.invalidate(cache_key)
            return []

    def put_datapoints(
        self,
        source: DataSource,
        symbols: List[str],
        data_points: List[DataPoint],
        lookback_hours: int = 24,
        ttl_minutes: Optional[int] = None,
        cost: float = 0.0
    ) -> None:
        """Cache data points for specific query"""
        if not data_points:
            return

        cache_key = self._generate_datapoint_key(source, symbols, lookback_hours)

        # Serialize DataPoint objects
        serialized_points = [self._serialize_datapoint(dp) for dp in data_points]

        self.put(
            key=cache_key,
            data=serialized_points,
            ttl_minutes=ttl_minutes,
            cost=cost
        )

    def _generate_datapoint_key(
        self,
        source: DataSource,
        symbols: List[str],
        lookback_hours: int
    ) -> str:
        """Generate cache key for data points query"""
        # Create deterministic key
        symbols_str = ",".join(sorted(symbols))
        key_data = f"{source.value}:{symbols_str}:{lookback_hours}"

        # Hash for consistent key length
        return hashlib.md5(key_data.encode()).hexdigest()

    def _serialize_datapoint(self, data_point: DataPoint) -> Dict[str, Any]:
        """Serialize DataPoint to dictionary"""
        return data_point.to_dict()

    def _deserialize_datapoint(self, data: Dict[str, Any]) -> DataPoint:
        """Deserialize dictionary to DataPoint"""
        # Convert timestamp string back to datetime
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['source'] = DataSource(data['source'])

        return DataPoint(**data)

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data to bytes"""
        try:
            if isinstance(data, (list, dict)):
                return json.dumps(data, default=str).encode('utf-8')
            else:
                return pickle.dumps(data)
        except Exception as e:
            self.logger.warning(f"Failed to serialize data: {e}")
            return pickle.dumps(data)

    def _deserialize_data(self, data: bytes, compressed: bool) -> Any:
        """Deserialize data from bytes"""
        if compressed:
            data = gzip.decompress(data)

        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)

    def _make_room(self, needed_bytes: int) -> None:
        """Make room in cache using LRU eviction"""
        if self.current_size_bytes + needed_bytes <= self.max_size_bytes:
            return

        # Remove expired entries first
        self._cleanup_expired()

        # If still need space, use LRU eviction
        while self.current_size_bytes + needed_bytes > self.max_size_bytes:
            if not self.cache:
                break

            # Find LRU entry
            lru_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].last_accessed
            )

            self._remove_entry(lru_key)
            self.evictions += 1

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache"""
        if key in self.cache:
            entry = self.cache[key]
            data_size = len(entry.data)
            self.current_size_bytes -= data_size
            del self.cache[key]

    def _cleanup_expired(self) -> None:
        """Remove all expired entries"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]

        for key in expired_keys:
            self._remove_entry(key)

    def invalidate(self, key: str) -> None:
        """Manually invalidate cache entry"""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                self.logger.debug(f"Invalidated cache key: {key[:50]}...")

    def invalidate_source(self, source: DataSource) -> None:
        """Invalidate all entries for a data source"""
        with self.lock:
            keys_to_remove = [
                key for key in self.cache.keys()
                if key.startswith(source.value)
            ]

            for key in keys_to_remove:
                self._remove_entry(key)

            self.logger.info(f"Invalidated {len(keys_to_remove)} entries for {source.value}")

    def clear(self) -> None:
        """Clear entire cache"""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
            self.logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                'size_mb': round(self.current_size_bytes / (1024 * 1024), 2),
                'max_size_mb': round(self.max_size_bytes / (1024 * 1024), 2),
                'utilization': round(self.current_size_bytes / self.max_size_bytes, 3),
                'entries': len(self.cache),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': round(hit_rate, 3),
                'evictions': self.evictions,
                'cost_saved': round(self.total_cost_saved, 2)
            }

    def _load_persistent_cache(self) -> None:
        """Load cache from disk on startup"""
        cache_file = self.cache_dir / "cache.pkl"

        if not cache_file.exists():
            return

        try:
            with open(cache_file, 'rb') as f:
                persistent_data = pickle.load(f)

            # Restore non-expired entries
            current_time = time.time()
            for key, entry in persistent_data.items():
                if current_time <= entry.expires_at:
                    self.cache[key] = entry
                    self.current_size_bytes += len(entry.data)

            self.logger.info(f"Loaded {len(self.cache)} cached entries from disk")

        except Exception as e:
            self.logger.warning(f"Failed to load persistent cache: {e}")

    def save_persistent_cache(self) -> None:
        """Save cache to disk"""
        cache_file = self.cache_dir / "cache.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)

            self.logger.info(f"Saved {len(self.cache)} cache entries to disk")

        except Exception as e:
            self.logger.error(f"Failed to save persistent cache: {e}")

    def __del__(self):
        """Save cache on destruction"""
        try:
            self.save_persistent_cache()
        except Exception:
            pass  # Ignore errors during cleanup