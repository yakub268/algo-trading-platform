"""
Alternative Data Manager
========================

Central orchestration system for all alternative data sources.
Coordinates data fetching, quality scoring, caching, and ML integration.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .base_connector import BaseDataConnector, DataPoint, DataSource
from .quality_scorer import DataQualityScorer, QualityMetrics
from .cache_manager import CacheManager
from .rate_limiter import RateLimiter


@dataclass
class DataRequest:
    """Request for alternative data"""
    sources: List[DataSource]
    symbols: List[str]
    asset_classes: List[str]
    lookback_hours: int = 24
    min_quality_score: float = 0.5
    max_latency_ms: int = 10000
    use_cache: bool = True
    priority: int = 1  # 1=high, 5=low

    def __post_init__(self):
        self.request_id = f"{int(time.time())}_{hash(tuple(self.symbols))}"


@dataclass
class DataResponse:
    """Response with alternative data"""
    request_id: str
    data_points: List[DataPoint] = field(default_factory=list)
    quality_metrics: Dict[DataSource, QualityMetrics] = field(default_factory=dict)
    cache_hit_rate: float = 0.0
    total_latency_ms: int = 0
    successful_sources: Set[DataSource] = field(default_factory=set)
    failed_sources: Set[DataSource] = field(default_factory=set)
    cost_estimate: float = 0.0


class AlternativeDataManager:
    """
    Central manager for all alternative data sources

    Features:
    - Parallel data fetching
    - Quality scoring and filtering
    - Intelligent caching
    - Cost optimization
    - Health monitoring
    - ML model integration
    """

    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        quality_scorer: Optional[DataQualityScorer] = None,
        max_concurrent_requests: int = 10
    ):
        self.connectors: Dict[DataSource, BaseDataConnector] = {}
        self.cache_manager = cache_manager or CacheManager()
        self.quality_scorer = quality_scorer or DataQualityScorer()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)

        # Health monitoring
        self.request_count = 0
        self.successful_requests = 0
        self.total_latency_ms = 0

        # Cost tracking
        self.daily_cost = 0.0
        self.cost_by_source: Dict[DataSource, float] = {}

        self.logger = logging.getLogger("altdata.manager")

    def register_connector(
        self,
        source: DataSource,
        connector: BaseDataConnector
    ) -> None:
        """Register a data connector"""
        self.connectors[source] = connector
        self.cost_by_source[source] = 0.0
        self.logger.info(f"Registered connector for {source.value}")

    def unregister_connector(self, source: DataSource) -> None:
        """Unregister a data connector"""
        if source in self.connectors:
            del self.connectors[source]
            self.logger.info(f"Unregistered connector for {source.value}")

    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """
        Fetch alternative data from multiple sources

        Args:
            request: DataRequest with sources and parameters

        Returns:
            DataResponse with collected data and metrics
        """
        start_time = time.time()
        self.request_count += 1

        response = DataResponse(request_id=request.request_id)

        try:
            # Check cache first if enabled
            cached_data = []
            if request.use_cache:
                cached_data = await self._fetch_from_cache(request)
                response.data_points.extend(cached_data)

            # Determine which sources need fresh data
            sources_to_fetch = self._determine_sources_to_fetch(
                request, len(cached_data) > 0
            )

            # Fetch data from sources in parallel
            if sources_to_fetch:
                fresh_data = await self._fetch_from_sources(
                    sources_to_fetch, request
                )
                response.data_points.extend(fresh_data)

            # Quality scoring and filtering
            response.data_points = self._filter_by_quality(
                response.data_points, request.min_quality_score
            )

            # Calculate metrics
            response = self._calculate_response_metrics(response, request, start_time)

            # Cache fresh data
            if request.use_cache and response.data_points:
                await self._cache_response_data(request, response.data_points)

            self.successful_requests += 1
            self.logger.info(
                f"Completed data request {request.request_id}: "
                f"{len(response.data_points)} points from "
                f"{len(response.successful_sources)} sources"
            )

        except Exception as e:
            self.logger.error(f"Failed to fetch data: {e}")
            response.failed_sources.update(request.sources)

        return response

    async def _fetch_from_cache(self, request: DataRequest) -> List[DataPoint]:
        """Fetch data from cache"""
        cached_data = []

        for source in request.sources:
            if source not in self.connectors:
                continue

            cache_data = self.cache_manager.get_datapoints(
                source=source,
                symbols=request.symbols,
                lookback_hours=request.lookback_hours
            )

            if cache_data:
                # Filter by quality and recency
                valid_cache_data = [
                    dp for dp in cache_data
                    if dp.quality_score >= request.min_quality_score
                    and (datetime.utcnow() - dp.timestamp).total_seconds() / 3600 <= request.lookback_hours
                ]
                cached_data.extend(valid_cache_data)

        return cached_data

    def _determine_sources_to_fetch(
        self,
        request: DataRequest,
        has_cached_data: bool
    ) -> List[DataSource]:
        """Determine which sources need fresh data"""
        sources_to_fetch = []

        for source in request.sources:
            if source not in self.connectors:
                self.logger.warning(f"No connector registered for {source.value}")
                continue

            connector = self.connectors[source]

            # Skip if connector is unhealthy
            if not connector.is_healthy:
                self.logger.warning(f"Skipping unhealthy connector: {source.value}")
                continue

            # Skip if we have sufficient cached data for low priority requests
            if has_cached_data and request.priority > 3:
                continue

            sources_to_fetch.append(source)

        return sources_to_fetch

    async def _fetch_from_sources(
        self,
        sources: List[DataSource],
        request: DataRequest
    ) -> List[DataPoint]:
        """Fetch data from multiple sources in parallel"""
        tasks = []

        for source in sources:
            connector = self.connectors[source]
            task = asyncio.create_task(
                self._fetch_from_single_source(connector, request)
            )
            tasks.append((source, task))

        all_data = []
        for source, task in tasks:
            try:
                data_points = await task
                all_data.extend(data_points)
                self.logger.debug(f"Fetched {len(data_points)} points from {source.value}")
            except Exception as e:
                self.logger.error(f"Failed to fetch from {source.value}: {e}")

        return all_data

    async def _fetch_from_single_source(
        self,
        connector: BaseDataConnector,
        request: DataRequest
    ) -> List[DataPoint]:
        """Fetch data from a single source"""
        try:
            data_points = await connector.fetch_with_retry(
                connector.fetch_data,
                symbols=request.symbols,
                lookback_hours=request.lookback_hours
            )

            # Filter by asset classes if specified
            if request.asset_classes:
                data_points = [
                    dp for dp in data_points
                    if dp.asset_class in request.asset_classes
                ]

            return data_points

        except Exception as e:
            self.logger.error(f"Error fetching from {connector.source.value}: {e}")
            return []

    def _filter_by_quality(
        self,
        data_points: List[DataPoint],
        min_quality_score: float
    ) -> List[DataPoint]:
        """Filter data points by quality score"""
        filtered_points = []

        for dp in data_points:
            # Score quality if not already done
            if dp.quality_score == 0.0:
                metrics = self.quality_scorer.score_data_point(dp)
                dp.quality_score = metrics.overall_score

            # Apply quality filter
            if dp.quality_score >= min_quality_score:
                filtered_points.append(dp)

        self.logger.debug(
            f"Quality filter: {len(filtered_points)}/{len(data_points)} points passed"
        )

        return filtered_points

    def _calculate_response_metrics(
        self,
        response: DataResponse,
        request: DataRequest,
        start_time: float
    ) -> DataResponse:
        """Calculate response metrics"""
        # Total latency
        response.total_latency_ms = int((time.time() - start_time) * 1000)
        self.total_latency_ms += response.total_latency_ms

        # Successful/failed sources
        sources_with_data = set(dp.source for dp in response.data_points)
        response.successful_sources = sources_with_data
        response.failed_sources = set(request.sources) - sources_with_data

        # Cache hit rate calculation
        total_possible_sources = len([
            s for s in request.sources if s in self.connectors
        ])
        cached_sources = len(sources_with_data) if request.use_cache else 0
        response.cache_hit_rate = cached_sources / total_possible_sources if total_possible_sources > 0 else 0

        # Quality metrics by source
        for source in DataSource:
            source_points = [dp for dp in response.data_points if dp.source == source]
            if source_points:
                avg_quality = sum(dp.quality_score for dp in source_points) / len(source_points)
                # Create aggregate quality metrics (simplified)
                metrics = QualityMetrics()
                metrics.overall_score = avg_quality
                response.quality_metrics[source] = metrics

        # Cost estimation (this would integrate with billing APIs)
        response.cost_estimate = self._estimate_cost(response.data_points)

        return response

    def _estimate_cost(self, data_points: List[DataPoint]) -> float:
        """Estimate API cost for data points"""
        # Cost per API call by source (example rates)
        cost_per_call = {
            DataSource.TWITTER: 0.001,
            DataSource.REDDIT: 0.0005,
            DataSource.SATELLITE: 0.10,
            DataSource.WEATHER: 0.005,
            DataSource.FRED: 0.0,  # Free
            DataSource.OPTIONS_FLOW: 0.02,
            DataSource.SEC_FILINGS: 0.001,
            DataSource.BLOCKCHAIN: 0.01,
            DataSource.NEWS: 0.003,
            DataSource.GOOGLE_TRENDS: 0.001
        }

        # Count API calls by source
        source_calls = {}
        for dp in data_points:
            source_calls[dp.source] = source_calls.get(dp.source, 0) + 1

        # Calculate total cost
        total_cost = 0.0
        for source, calls in source_calls.items():
            cost = cost_per_call.get(source, 0.001) * calls
            total_cost += cost
            self.cost_by_source[source] = self.cost_by_source.get(source, 0) + cost

        self.daily_cost += total_cost
        return total_cost

    async def _cache_response_data(
        self,
        request: DataRequest,
        data_points: List[DataPoint]
    ) -> None:
        """Cache response data by source"""
        # Group data points by source
        points_by_source = {}
        for dp in data_points:
            if dp.source not in points_by_source:
                points_by_source[dp.source] = []
            points_by_source[dp.source].append(dp)

        # Cache each source separately
        for source, points in points_by_source.items():
            self.cache_manager.put_datapoints(
                source=source,
                symbols=request.symbols,
                data_points=points,
                lookback_hours=request.lookback_hours,
                ttl_minutes=5,  # 5 minute default TTL
                cost=self._estimate_cost(points)
            )

    async def get_real_time_data(
        self,
        sources: List[DataSource],
        symbols: List[str]
    ) -> List[DataPoint]:
        """Get real-time data from specified sources"""
        tasks = []

        for source in sources:
            if source not in self.connectors:
                continue

            connector = self.connectors[source]
            if not connector.is_healthy:
                continue

            task = asyncio.create_task(
                connector.get_real_time_data(symbols)
            )
            tasks.append(task)

        all_data = []
        for task in tasks:
            try:
                data_points = await task
                all_data.extend(data_points)
            except Exception as e:
                self.logger.error(f"Failed to get real-time data: {e}")

        return all_data

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        connector_health = {}
        total_connectors = len(self.connectors)
        healthy_connectors = 0

        for source, connector in self.connectors.items():
            health = connector.get_health_status()
            connector_health[source.value] = health
            if health['healthy']:
                healthy_connectors += 1

        avg_latency = (
            self.total_latency_ms / self.request_count
            if self.request_count > 0 else 0
        )

        success_rate = (
            self.successful_requests / self.request_count
            if self.request_count > 0 else 0
        )

        return {
            'system_health': {
                'healthy_connectors': healthy_connectors,
                'total_connectors': total_connectors,
                'system_health_percentage': (healthy_connectors / total_connectors * 100) if total_connectors > 0 else 0,
                'total_requests': self.request_count,
                'successful_requests': self.successful_requests,
                'success_rate': success_rate,
                'average_latency_ms': avg_latency,
                'daily_cost': self.daily_cost
            },
            'cache_stats': self.cache_manager.get_stats(),
            'cost_by_source': self.cost_by_source,
            'connectors': connector_health
        }

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at midnight)"""
        self.daily_cost = 0.0
        self.cost_by_source = {source: 0.0 for source in self.cost_by_source}
        self.logger.info("Reset daily statistics")

    def shutdown(self) -> None:
        """Graceful shutdown"""
        self.cache_manager.save_persistent_cache()
        self.executor.shutdown(wait=True)
        self.logger.info("Alternative Data Manager shutdown complete")