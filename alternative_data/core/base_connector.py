"""
Base Data Connector
==================

Abstract base class for all alternative data connectors.
Provides standardized interface and common functionality.
"""

import time
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from .rate_limiter import RateLimiter
from .quality_scorer import QualityMetrics


class DataSource(Enum):
    """Enumeration of supported data sources"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    DISCORD = "discord"
    SATELLITE = "satellite"
    WEATHER = "weather"
    FRED = "fred"
    OPTIONS_FLOW = "options_flow"
    SEC_FILINGS = "sec_filings"
    BLOCKCHAIN = "blockchain"
    NEWS = "news"
    GOOGLE_TRENDS = "google_trends"
    INSIDER_TRADING = "insider_trading"
    DARK_POOL = "dark_pool"


@dataclass
class DataPoint:
    """Single data point with metadata"""
    source: DataSource
    timestamp: datetime
    symbol: Optional[str]
    asset_class: str  # stocks, crypto, forex, commodities
    raw_data: Dict[str, Any]
    processed_data: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    quality_score: float = 0.0
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission"""
        return {
            'source': self.source.value,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'asset_class': self.asset_class,
            'raw_data': self.raw_data,
            'processed_data': self.processed_data,
            'confidence': self.confidence,
            'quality_score': self.quality_score,
            'latency_ms': self.latency_ms
        }


@dataclass
class ConnectorConfig:
    """Configuration for data connector"""
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    cache_ttl_minutes: int = 5
    quality_threshold: float = 0.5
    enabled: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)


class BaseDataConnector(ABC):
    """
    Abstract base class for all alternative data connectors

    Provides:
    - Rate limiting
    - Error handling and retries
    - Data quality validation
    - Caching interface
    - Standardized logging
    """

    def __init__(self, source: DataSource, config: ConnectorConfig):
        self.source = source
        self.config = config
        self.logger = logging.getLogger(f"altdata.{source.value}")
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)
        self.is_healthy = True
        self.last_error: Optional[str] = None
        self.total_requests = 0
        self.successful_requests = 0

        # Initialize connector
        self._initialize()

    def _initialize(self) -> None:
        """Initialize connector-specific setup"""
        self.logger.info(f"Initializing {self.source.value} connector")

    @abstractmethod
    async def fetch_data(
        self,
        symbols: List[str],
        lookback_hours: int = 24,
        **kwargs
    ) -> List[DataPoint]:
        """
        Fetch data for given symbols

        Args:
            symbols: List of symbols to fetch data for
            lookback_hours: Hours of historical data to fetch
            **kwargs: Connector-specific parameters

        Returns:
            List of DataPoint objects
        """
        pass

    @abstractmethod
    async def get_real_time_data(self, symbols: List[str]) -> List[DataPoint]:
        """
        Get real-time data for symbols

        Args:
            symbols: List of symbols to monitor

        Returns:
            List of current DataPoint objects
        """
        pass

    @abstractmethod
    def validate_data(self, data_point: DataPoint) -> Tuple[bool, float]:
        """
        Validate data quality for a data point

        Args:
            data_point: DataPoint to validate

        Returns:
            (is_valid, quality_score)
        """
        pass

    async def fetch_with_retry(
        self,
        fetch_func: callable,
        *args,
        **kwargs
    ) -> List[DataPoint]:
        """
        Execute fetch with rate limiting, retries, and error handling
        """
        if not self.config.enabled:
            return []

        await self.rate_limiter.acquire()

        for attempt in range(self.config.retry_attempts):
            try:
                start_time = time.time()
                self.total_requests += 1

                # Execute the fetch
                data_points = await asyncio.wait_for(
                    fetch_func(*args, **kwargs),
                    timeout=self.config.timeout_seconds
                )

                # Calculate latency
                latency_ms = int((time.time() - start_time) * 1000)

                # Update latency for all data points
                for dp in data_points:
                    dp.latency_ms = latency_ms

                # Validate and score data quality
                validated_points = []
                for dp in data_points:
                    is_valid, quality_score = self.validate_data(dp)
                    dp.quality_score = quality_score

                    if is_valid and quality_score >= self.config.quality_threshold:
                        validated_points.append(dp)

                self.successful_requests += 1
                self.is_healthy = True
                self.last_error = None

                self.logger.debug(
                    f"Fetched {len(validated_points)} valid data points "
                    f"from {self.source.value} (latency: {latency_ms}ms)"
                )

                return validated_points

            except asyncio.TimeoutError:
                error_msg = f"Timeout after {self.config.timeout_seconds}s"
                self.logger.warning(f"Attempt {attempt + 1}: {error_msg}")
                self.last_error = error_msg

            except Exception as e:
                error_msg = f"Error fetching data: {str(e)}"
                self.logger.warning(f"Attempt {attempt + 1}: {error_msg}")
                self.last_error = error_msg

                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

        self.is_healthy = False
        self.logger.error(f"All retry attempts failed for {self.source.value}")
        return []

    def get_health_status(self) -> Dict[str, Any]:
        """Get connector health metrics"""
        success_rate = (
            self.successful_requests / self.total_requests
            if self.total_requests > 0 else 0.0
        )

        return {
            'source': self.source.value,
            'healthy': self.is_healthy,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'success_rate': success_rate,
            'last_error': self.last_error,
            'rate_limit_remaining': self.rate_limiter.remaining_requests(),
            'config': {
                'enabled': self.config.enabled,
                'rate_limit': self.config.rate_limit_per_minute,
                'quality_threshold': self.config.quality_threshold
            }
        }

    def reset_health(self) -> None:
        """Reset health status and counters"""
        self.is_healthy = True
        self.last_error = None
        self.total_requests = 0
        self.successful_requests = 0
        self.logger.info(f"Health status reset for {self.source.value}")