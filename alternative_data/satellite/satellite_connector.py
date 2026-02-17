"""
Satellite Data Connector
========================

Satellite imagery analysis for trading signals using multiple providers
including NASA, ESA, and commercial satellite data.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, date
import numpy as np

try:
    import requests
    import aiohttp
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

try:
    from PIL import Image
    import io
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False

from ..core.base_connector import BaseDataConnector, DataPoint, DataSource, ConnectorConfig


class SatelliteConnector(BaseDataConnector):
    """
    Satellite imagery analysis connector

    Features:
    - NASA Earth Observing System Data (free)
    - ESA Sentinel satellite data
    - USDA crop monitoring
    - Economic activity indicators
    - Infrastructure monitoring
    - Weather pattern analysis
    """

    def __init__(self, config: ConnectorConfig):
        super().__init__(DataSource.SATELLITE, config)

        # API endpoints
        self.nasa_api_key = config.api_key  # NASA API key (free)
        self.esa_api_key = config.custom_params.get('esa_api_key')
        self.usda_api_key = config.custom_params.get('usda_api_key')

        # API endpoints
        self.apis = {
            'nasa_earth': 'https://api.nasa.gov/planetary/earth',
            'nasa_modis': 'https://modis.ornl.gov/rst/api/v1',
            'usda_nass': 'https://quickstats.nass.usda.gov/api',
            'esa_sentinel': 'https://scihub.copernicus.eu/dhus/api/stub/products',
            'worldbank': 'https://api.worldbank.org/v2'
        }

        # Commodities to regions mapping
        self.commodity_regions = config.custom_params.get('commodity_regions', {
            'CORN': [
                {'name': 'Iowa', 'lat': 41.8781, 'lon': -93.0977, 'radius': 200},
                {'name': 'Illinois', 'lat': 40.6331, 'lon': -89.3985, 'radius': 200},
                {'name': 'Nebraska', 'lat': 41.4925, 'lon': -99.9018, 'radius': 150}
            ],
            'WHEAT': [
                {'name': 'Kansas', 'lat': 38.5266, 'lon': -96.7265, 'radius': 200},
                {'name': 'North Dakota', 'lat': 47.5514, 'lon': -101.0020, 'radius': 150}
            ],
            'SOY': [
                {'name': 'Iowa', 'lat': 41.8781, 'lon': -93.0977, 'radius': 200},
                {'name': 'Illinois', 'lat': 40.6331, 'lon': -89.3985, 'radius': 200}
            ]
        })

        # Economic activity monitoring locations
        self.economic_regions = config.custom_params.get('economic_regions', {
            'CHINA_MANUFACTURING': [
                {'name': 'Shanghai', 'lat': 31.2304, 'lon': 121.4737, 'type': 'port'},
                {'name': 'Shenzhen', 'lat': 22.3193, 'lon': 114.1694, 'type': 'manufacturing'}
            ],
            'US_ENERGY': [
                {'name': 'Permian Basin', 'lat': 31.8457, 'lon': -102.3676, 'type': 'oil'},
                {'name': 'Bakken', 'lat': 47.7511, 'lon': -101.7777, 'type': 'oil'}
            ]
        })

        self.session = None

    async def fetch_data(
        self,
        symbols: List[str],
        lookback_hours: int = 24,
        **kwargs
    ) -> List[DataPoint]:
        """
        Fetch satellite data for symbols

        Args:
            symbols: List of commodity/economic symbols
            lookback_hours: Hours of historical data (limited by satellite revisit times)
            **kwargs: Additional parameters

        Returns:
            List of DataPoint objects with satellite analysis
        """
        if not HTTP_AVAILABLE:
            self.logger.error("HTTP libraries not available")
            return []

        all_data_points = []

        # Create HTTP session
        async with aiohttp.ClientSession() as session:
            self.session = session

            for symbol in symbols:
                try:
                    # Determine symbol type and get relevant data
                    if self._is_commodity_symbol(symbol):
                        data_points = await self._fetch_commodity_data(symbol, lookback_hours)
                    elif self._is_economic_symbol(symbol):
                        data_points = await self._fetch_economic_data(symbol, lookback_hours)
                    else:
                        # Try both approaches
                        commodity_data = await self._fetch_commodity_data(symbol, lookback_hours)
                        economic_data = await self._fetch_economic_data(symbol, lookback_hours)
                        data_points = commodity_data + economic_data

                    all_data_points.extend(data_points)

                except Exception as e:
                    self.logger.error(f"Failed to fetch satellite data for {symbol}: {e}")

                # Rate limiting
                await asyncio.sleep(2)

        self.logger.info(f"Fetched {len(all_data_points)} satellite data points")
        return all_data_points

    async def get_real_time_data(self, symbols: List[str]) -> List[DataPoint]:
        """
        Get most recent satellite data (within last week due to revisit constraints)
        """
        return await self.fetch_data(symbols, lookback_hours=168)  # 1 week

    def _is_commodity_symbol(self, symbol: str) -> bool:
        """Check if symbol is a commodity"""
        commodities = ['CORN', 'WHEAT', 'SOY', 'SUGAR', 'COTTON', 'RICE', 'COFFEE']
        return symbol.upper() in commodities or any(comm in symbol.upper() for comm in commodities)

    def _is_economic_symbol(self, symbol: str) -> bool:
        """Check if symbol relates to economic activity"""
        economic_indicators = ['OIL', 'CRUDE', 'MANUFACTURING', 'SHIPPING', 'COPPER', 'STEEL']
        return any(indicator in symbol.upper() for indicator in economic_indicators)

    async def _fetch_commodity_data(self, symbol: str, lookback_hours: int) -> List[DataPoint]:
        """Fetch commodity-related satellite data"""
        data_points = []

        # Get regions for this commodity
        regions = self.commodity_regions.get(symbol.upper(), [])
        if not regions and self._is_commodity_symbol(symbol):
            # Default to major agricultural regions
            regions = self.commodity_regions.get('CORN', [])

        for region in regions:
            try:
                # Fetch vegetation indices (NDVI) from NASA MODIS
                ndvi_data = await self._fetch_ndvi_data(
                    region['lat'],
                    region['lon'],
                    region.get('radius', 100),
                    lookback_hours
                )

                # Fetch weather data
                weather_data = await self._fetch_satellite_weather_data(
                    region['lat'],
                    region['lon'],
                    lookback_hours
                )

                # Process into data points
                if ndvi_data:
                    data_point = await self._process_commodity_data(
                        symbol, region, ndvi_data, weather_data
                    )
                    if data_point:
                        data_points.append(data_point)

            except Exception as e:
                self.logger.warning(f"Failed to fetch data for region {region['name']}: {e}")

        return data_points

    async def _fetch_economic_data(self, symbol: str, lookback_hours: int) -> List[DataPoint]:
        """Fetch economic activity satellite data"""
        data_points = []

        # Economic activity indicators
        economic_types = {
            'OIL': 'US_ENERGY',
            'CRUDE': 'US_ENERGY',
            'MANUFACTURING': 'CHINA_MANUFACTURING'
        }

        region_key = None
        for key, value in economic_types.items():
            if key in symbol.upper():
                region_key = value
                break

        if not region_key:
            region_key = 'CHINA_MANUFACTURING'  # Default

        regions = self.economic_regions.get(region_key, [])

        for region in regions:
            try:
                # Fetch nighttime lights data (economic activity proxy)
                lights_data = await self._fetch_nighttime_lights_data(
                    region['lat'],
                    region['lon'],
                    lookback_hours
                )

                # Fetch infrastructure changes
                infrastructure_data = await self._fetch_infrastructure_data(
                    region['lat'],
                    region['lon'],
                    lookback_hours
                )

                # Process into data points
                if lights_data or infrastructure_data:
                    data_point = await self._process_economic_data(
                        symbol, region, lights_data, infrastructure_data
                    )
                    if data_point:
                        data_points.append(data_point)

            except Exception as e:
                self.logger.warning(f"Failed to fetch economic data for {region['name']}: {e}")

        return data_points

    async def _fetch_ndvi_data(
        self,
        lat: float,
        lon: float,
        radius: int,
        lookback_hours: int
    ) -> Optional[Dict[str, Any]]:
        """Fetch NDVI (vegetation index) data from NASA MODIS"""
        try:
            # NASA MODIS API for vegetation indices
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(hours=lookback_hours)

            # MODIS data is typically available every 16 days
            # Adjust date range accordingly
            if lookback_hours < 24 * 16:  # Less than 16 days
                start_date = end_date - timedelta(days=30)

            params = {
                'product': 'MOD13Q1',  # MODIS Vegetation Indices
                'latitude': lat,
                'longitude': lon,
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d'),
                'kmAboveBelow': radius // 2,  # Convert radius to km above/below
                'kmLeftRight': radius // 2
            }

            if self.nasa_api_key:
                params['api_key'] = self.nasa_api_key

            # Use NASA ORNL MODIS API
            url = f"{self.apis['nasa_modis']}/subset"

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    self.logger.warning(f"NDVI API returned status {response.status}")

        except Exception as e:
            self.logger.error(f"Failed to fetch NDVI data: {e}")

        return None

    async def _fetch_satellite_weather_data(
        self,
        lat: float,
        lon: float,
        lookback_hours: int
    ) -> Optional[Dict[str, Any]]:
        """Fetch weather data from satellite sources"""
        try:
            # NASA Earth API for satellite imagery
            end_date = datetime.utcnow()

            params = {
                'lat': lat,
                'lon': lon,
                'date': end_date.strftime('%Y-%m-%d'),
                'dim': 0.10  # Image dimension
            }

            if self.nasa_api_key:
                params['api_key'] = self.nasa_api_key

            url = f"{self.apis['nasa_earth']}/imagery"

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    # Get image metadata
                    metadata_url = f"{self.apis['nasa_earth']}/assets"
                    async with self.session.get(metadata_url, params=params) as meta_response:
                        if meta_response.status == 200:
                            metadata = await meta_response.json()
                            return {
                                'date': end_date.isoformat(),
                                'location': {'lat': lat, 'lon': lon},
                                'metadata': metadata,
                                'image_available': True
                            }

        except Exception as e:
            self.logger.error(f"Failed to fetch satellite weather data: {e}")

        return None

    async def _fetch_nighttime_lights_data(
        self,
        lat: float,
        lon: float,
        lookback_hours: int
    ) -> Optional[Dict[str, Any]]:
        """Fetch nighttime lights data (economic activity indicator)"""
        try:
            # Note: This would typically use VIIRS Day/Night Band data
            # For demo purposes, we'll simulate economic activity indicators

            # In a real implementation, you would:
            # 1. Access NASA's VIIRS Day/Night Band data
            # 2. Calculate light intensity changes over time
            # 3. Correlate with economic activity

            economic_activity_proxy = {
                'location': {'lat': lat, 'lon': lon},
                'light_intensity_change': np.random.normal(0, 0.1),  # Simulated
                'activity_score': np.random.uniform(0.4, 0.9),
                'data_source': 'VIIRS_DNB',
                'timestamp': datetime.utcnow().isoformat()
            }

            return economic_activity_proxy

        except Exception as e:
            self.logger.error(f"Failed to fetch nighttime lights data: {e}")

        return None

    async def _fetch_infrastructure_data(
        self,
        lat: float,
        lon: float,
        lookback_hours: int
    ) -> Optional[Dict[str, Any]]:
        """Fetch infrastructure development data"""
        try:
            # This would typically analyze high-resolution satellite imagery
            # to detect construction, port activity, etc.

            # Simulated infrastructure activity indicators
            infrastructure_activity = {
                'location': {'lat': lat, 'lon': lon},
                'construction_activity': np.random.uniform(0, 1),
                'port_activity': np.random.uniform(0, 1) if 'port' in str(lat) else 0,
                'manufacturing_activity': np.random.uniform(0, 1),
                'change_detection_score': np.random.normal(0, 0.2),
                'timestamp': datetime.utcnow().isoformat()
            }

            return infrastructure_activity

        except Exception as e:
            self.logger.error(f"Failed to fetch infrastructure data: {e}")

        return None

    async def _process_commodity_data(
        self,
        symbol: str,
        region: Dict[str, Any],
        ndvi_data: Dict[str, Any],
        weather_data: Optional[Dict[str, Any]]
    ) -> Optional[DataPoint]:
        """Process commodity satellite data into DataPoint"""
        try:
            # Calculate crop health indicators
            crop_health_score = self._calculate_crop_health_score(ndvi_data)
            weather_impact_score = self._calculate_weather_impact_score(weather_data)

            # Create raw data
            raw_data = {
                'symbol': symbol,
                'region': region,
                'ndvi_data': ndvi_data,
                'weather_data': weather_data,
                'data_sources': ['MODIS', 'NASA_EARTH'],
                'analysis_type': 'commodity_agriculture'
            }

            # Create processed data
            processed_data = {
                'crop_health_score': crop_health_score,
                'weather_impact_score': weather_impact_score,
                'vegetation_index': self._extract_vegetation_index(ndvi_data),
                'seasonal_deviation': self._calculate_seasonal_deviation(ndvi_data, symbol),
                'yield_prediction_impact': self._calculate_yield_impact(
                    crop_health_score, weather_impact_score
                ),
                'trading_signal_strength': self._calculate_commodity_signal_strength(
                    crop_health_score, weather_impact_score
                )
            }

            # Create data point
            data_point = DataPoint(
                source=DataSource.SATELLITE,
                timestamp=datetime.utcnow(),
                symbol=symbol,
                asset_class='commodities',
                raw_data=raw_data,
                processed_data=processed_data,
                confidence=self._calculate_commodity_confidence(ndvi_data, weather_data)
            )

            return data_point

        except Exception as e:
            self.logger.error(f"Failed to process commodity data: {e}")
            return None

    async def _process_economic_data(
        self,
        symbol: str,
        region: Dict[str, Any],
        lights_data: Optional[Dict[str, Any]],
        infrastructure_data: Optional[Dict[str, Any]]
    ) -> Optional[DataPoint]:
        """Process economic activity satellite data into DataPoint"""
        try:
            # Calculate economic activity indicators
            activity_score = self._calculate_economic_activity_score(
                lights_data, infrastructure_data
            )

            growth_momentum = self._calculate_growth_momentum(
                lights_data, infrastructure_data
            )

            # Create raw data
            raw_data = {
                'symbol': symbol,
                'region': region,
                'lights_data': lights_data,
                'infrastructure_data': infrastructure_data,
                'data_sources': ['VIIRS', 'HIGH_RES_IMAGERY'],
                'analysis_type': 'economic_activity'
            }

            # Create processed data
            processed_data = {
                'economic_activity_score': activity_score,
                'growth_momentum': growth_momentum,
                'infrastructure_development': self._extract_infrastructure_score(infrastructure_data),
                'light_intensity_change': self._extract_light_change(lights_data),
                'manufacturing_indicator': self._calculate_manufacturing_indicator(region, infrastructure_data),
                'trading_signal_strength': self._calculate_economic_signal_strength(
                    activity_score, growth_momentum
                )
            }

            # Determine asset class based on symbol
            asset_class = 'commodities'
            if any(term in symbol.upper() for term in ['OIL', 'CRUDE']):
                asset_class = 'energy'
            elif 'MANUFACTURING' in symbol.upper():
                asset_class = 'industrial'

            # Create data point
            data_point = DataPoint(
                source=DataSource.SATELLITE,
                timestamp=datetime.utcnow(),
                symbol=symbol,
                asset_class=asset_class,
                raw_data=raw_data,
                processed_data=processed_data,
                confidence=self._calculate_economic_confidence(lights_data, infrastructure_data)
            )

            return data_point

        except Exception as e:
            self.logger.error(f"Failed to process economic data: {e}")
            return None

    def _calculate_crop_health_score(self, ndvi_data: Dict[str, Any]) -> float:
        """Calculate crop health score from NDVI data"""
        if not ndvi_data:
            return 50.0  # Neutral score

        # Extract NDVI values (would be more complex in real implementation)
        # NDVI ranges from -1 to 1, with higher values indicating healthier vegetation

        # Simulate NDVI analysis
        avg_ndvi = np.random.uniform(0.3, 0.8)  # Typical agricultural NDVI range

        # Convert to 0-100 score
        # NDVI > 0.6 is excellent, 0.4-0.6 is good, < 0.4 is poor
        if avg_ndvi > 0.6:
            health_score = 70 + (avg_ndvi - 0.6) * 150  # 70-100 range
        elif avg_ndvi > 0.4:
            health_score = 40 + (avg_ndvi - 0.4) * 150  # 40-70 range
        else:
            health_score = avg_ndvi * 100  # 0-40 range

        return min(100, max(0, health_score))

    def _calculate_weather_impact_score(self, weather_data: Optional[Dict[str, Any]]) -> float:
        """Calculate weather impact on crops"""
        if not weather_data:
            return 50.0  # Neutral impact

        # In real implementation, this would analyze:
        # - Cloud cover (drought indicators)
        # - Precipitation estimates
        # - Temperature anomalies
        # - Storm activity

        # Simulate weather impact analysis
        weather_impact = np.random.uniform(0.2, 0.9)
        return weather_impact * 100

    def _extract_vegetation_index(self, ndvi_data: Dict[str, Any]) -> float:
        """Extract normalized vegetation index"""
        if not ndvi_data:
            return 0.5

        # In real implementation, extract actual NDVI values
        return np.random.uniform(0.3, 0.8)

    def _calculate_seasonal_deviation(self, ndvi_data: Dict[str, Any], symbol: str) -> float:
        """Calculate deviation from seasonal norms"""
        # This would compare current NDVI with historical seasonal averages
        return np.random.normal(0, 0.15)  # Simulated deviation

    def _calculate_yield_impact(self, crop_health: float, weather_impact: float) -> float:
        """Calculate predicted yield impact"""
        # Combine crop health and weather factors
        combined_score = (crop_health * 0.6 + weather_impact * 0.4)

        # Convert to yield impact (-20% to +20%)
        yield_impact = (combined_score - 50) / 50 * 0.2
        return max(-0.2, min(0.2, yield_impact))

    def _calculate_commodity_signal_strength(self, crop_health: float, weather_impact: float) -> float:
        """Calculate trading signal strength for commodities"""
        # Strong signals come from:
        # 1. Extreme crop health scores (very good or very bad)
        # 2. Significant weather impacts
        # 3. Large deviations from normal

        health_signal = abs(crop_health - 50) / 50  # 0-1 scale
        weather_signal = abs(weather_impact - 50) / 50  # 0-1 scale

        signal_strength = (health_signal + weather_signal) * 50
        return min(100, signal_strength)

    def _calculate_economic_activity_score(
        self,
        lights_data: Optional[Dict[str, Any]],
        infrastructure_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate economic activity score"""
        score = 50  # Base score

        if lights_data:
            activity_score = lights_data.get('activity_score', 0.5)
            score += (activity_score - 0.5) * 100

        if infrastructure_data:
            infra_activity = infrastructure_data.get('manufacturing_activity', 0.5)
            score += (infra_activity - 0.5) * 50

        return min(100, max(0, score))

    def _calculate_growth_momentum(
        self,
        lights_data: Optional[Dict[str, Any]],
        infrastructure_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate economic growth momentum"""
        momentum = 0

        if lights_data:
            light_change = lights_data.get('light_intensity_change', 0)
            momentum += light_change * 100

        if infrastructure_data:
            change_score = infrastructure_data.get('change_detection_score', 0)
            momentum += change_score * 50

        return max(-100, min(100, momentum))

    def _extract_infrastructure_score(self, infrastructure_data: Optional[Dict[str, Any]]) -> float:
        """Extract infrastructure development score"""
        if not infrastructure_data:
            return 50

        construction = infrastructure_data.get('construction_activity', 0.5)
        return construction * 100

    def _extract_light_change(self, lights_data: Optional[Dict[str, Any]]) -> float:
        """Extract light intensity change"""
        if not lights_data:
            return 0

        return lights_data.get('light_intensity_change', 0)

    def _calculate_manufacturing_indicator(
        self,
        region: Dict[str, Any],
        infrastructure_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate manufacturing activity indicator"""
        base_score = 50

        if infrastructure_data:
            manufacturing = infrastructure_data.get('manufacturing_activity', 0.5)
            base_score = manufacturing * 100

        # Adjust based on region type
        if region.get('type') == 'manufacturing':
            base_score *= 1.2

        return min(100, base_score)

    def _calculate_economic_signal_strength(self, activity_score: float, momentum: float) -> float:
        """Calculate economic trading signal strength"""
        # Strong signals from extreme activity levels or momentum changes
        activity_signal = abs(activity_score - 50) / 50
        momentum_signal = abs(momentum) / 100

        signal_strength = (activity_signal + momentum_signal) * 50
        return min(100, signal_strength)

    def _calculate_commodity_confidence(
        self,
        ndvi_data: Dict[str, Any],
        weather_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for commodity analysis"""
        confidence = 0.5  # Base confidence

        if ndvi_data:
            confidence += 0.3  # NDVI data available

        if weather_data:
            confidence += 0.2  # Weather data available

        return min(1.0, confidence)

    def _calculate_economic_confidence(
        self,
        lights_data: Optional[Dict[str, Any]],
        infrastructure_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for economic analysis"""
        confidence = 0.3  # Base confidence (lower than commodity due to complexity)

        if lights_data:
            confidence += 0.3

        if infrastructure_data:
            confidence += 0.4

        return min(1.0, confidence)

    def validate_data(self, data_point: DataPoint) -> Tuple[bool, float]:
        """Validate satellite data point quality"""
        try:
            raw_data = data_point.raw_data
            processed_data = data_point.processed_data

            # Basic validation
            if not raw_data.get('symbol') or not raw_data.get('region'):
                return False, 0.0

            # Quality factors
            data_sources_count = len(raw_data.get('data_sources', []))
            signal_strength = processed_data.get('trading_signal_strength', 0) / 100.0

            # Calculate quality score
            data_availability = min(1.0, data_sources_count / 2)  # Expect 1-2 sources

            quality_score = (
                data_availability * 0.4 +
                signal_strength * 0.4 +
                data_point.confidence * 0.2
            )

            # Minimum thresholds
            is_valid = (
                quality_score >= 0.3 and
                data_sources_count >= 1
            )

            return is_valid, quality_score

        except Exception as e:
            self.logger.error(f"Failed to validate satellite data: {e}")
            return False, 0.0