"""
Multi-Source Edge Detection Engine
Aggregates signals from FRED, NWS, GFS ensemble, Cleveland Fed, GDPNow,
and orderbook analysis to compute probability edges against Kalshi market prices.
"""

import os
import re
import time
import math
import sqlite3
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from bots.kalshi_client import KalshiClient

MT = ZoneInfo("America/Denver")
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "live", "event_trading.db")

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2

BLOCKED_PREFIXES = ()  # Weather unblocked — GFS ensemble provides real signal

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

FRED_SERIES = {
    "UNRATE": "unemployment_rate",
    "CPIAUCSL": "cpi_all_items",
    "GDP": "gdp",
    "FEDFUNDS": "fed_funds_rate",
    "T10Y2Y": "yield_curve_spread",
    "PAYEMS": "nonfarm_payrolls",
    "UMCSENT": "consumer_sentiment",
    "RSXFS": "retail_sales",
    "HOUST": "housing_starts",
    "PCE": "pce_inflation",
}

NWS_BASE = "https://api.weather.gov"
NWS_FORECAST_ERROR = 3.0

TICKER_CITY_MAP = {
    "NY": {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    "CHI": {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
    "LA": {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    "MIA": {"name": "Miami", "lat": 25.7617, "lon": -80.1918},
    "DEN": {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
    "SF": {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    "DAL": {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    "DC": {"name": "Washington DC", "lat": 38.9072, "lon": -77.0369},
    "HOU": {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
    "PHX": {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740},
    "SEA": {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
    "BOS": {"name": "Boston", "lat": 42.3601, "lon": -71.0589},
    "ATL": {"name": "Atlanta", "lat": 33.7490, "lon": -84.3880},
    "PHI": {"name": "Philadelphia", "lat": 39.9526, "lon": -75.1652},
    "MSP": {"name": "Minneapolis", "lat": 44.9778, "lon": -93.2650},
    "DET": {"name": "Detroit", "lat": 42.3314, "lon": -83.0458},
}

DOMAIN_WEIGHTS = {
    "weather": {"gfs_ensemble": 0.80, "nws": 0.15, "whale": 0.05},
    "economics": {"fred": 0.30, "cpi_nowcast": 0.30, "gdpnow": 0.25, "whale": 0.05, "sentiment": 0.10},
    "sports": {"odds": 0.55, "sentiment": 0.35, "whale": 0.10},
    "generic": {"fred": 0.40, "weather": 0.25, "whale": 0.10, "sentiment": 0.15, "odds": 0.10},
}

logger = logging.getLogger("EventEdge.Detector")


def mt_now() -> datetime:
    return datetime.now(MT)


def mt_str(dt: datetime = None) -> str:
    return (dt or mt_now()).strftime("%Y-%m-%d %H:%M:%S MT")


def _retry_request(method: str, url: str, **kwargs) -> requests.Response:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.request(method, url, timeout=15, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == RETRY_ATTEMPTS:
                raise
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning(f"Request attempt {attempt} to {url} failed: {e}. Retry in {wait}s")
            time.sleep(wait)


@dataclass
class EdgeSignal:
    market_id: str
    ticker: str
    edge: float
    confidence: float
    direction: str
    expected_value: float
    ensemble_probability: float
    market_probability: float
    fred_signal: Optional[float] = None
    weather_signal: Optional[float] = None
    whale_signal: Optional[float] = None
    gfs_ensemble_signal: Optional[float] = None
    cpi_nowcast_signal: Optional[float] = None
    gdpnow_signal: Optional[float] = None
    nws_signal: Optional[float] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = mt_str()

    def to_dict(self) -> Dict:
        return {
            "market_id": self.market_id, "ticker": self.ticker,
            "edge": round(self.edge, 4), "confidence": round(self.confidence, 4),
            "direction": self.direction, "expected_value": round(self.expected_value, 4),
            "ensemble_probability": round(self.ensemble_probability, 4),
            "market_probability": round(self.market_probability, 4),
            "fred_signal": round(self.fred_signal, 4) if self.fred_signal is not None else None,
            "weather_signal": round(self.weather_signal, 4) if self.weather_signal is not None else None,
            "whale_signal": round(self.whale_signal, 4) if self.whale_signal is not None else None,
            "gfs_ensemble_signal": round(self.gfs_ensemble_signal, 4) if self.gfs_ensemble_signal is not None else None,
            "cpi_nowcast_signal": round(self.cpi_nowcast_signal, 4) if self.cpi_nowcast_signal is not None else None,
            "gdpnow_signal": round(self.gdpnow_signal, 4) if self.gdpnow_signal is not None else None,
            "nws_signal": round(self.nws_signal, 4) if self.nws_signal is not None else None,
            "timestamp": self.timestamp,
        }


def init_db(db_path: str = None):
    db = db_path or DB_PATH
    conn = sqlite3.connect(db, timeout=30)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS edge_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT, edge REAL, confidence REAL,
                ensemble_probability REAL, market_probability REAL,
                direction TEXT, fred_signal REAL, weather_signal REAL,
                whale_signal REAL, expected_value REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS source_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL, market_category TEXT NOT NULL,
                brier_score REAL, hit_rate REAL, n_predictions INTEGER,
                calibration_error REAL, last_updated TIMESTAMP,
                UNIQUE(source, market_category)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT, price REAL, yes_price REAL,
                no_price REAL, volume INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.close()


class FREDSource:
    TICKER_SERIES = {
        "KXGDP":    ("A191RL1Q225SBEA", "level"),
        "KXCPI":    ("CPIAUCSL", "change"),
        "KXPCE":    ("PCE", "change"),
        "KXFED":    ("FEDFUNDS", "level"),
        "KXUNRATE": ("UNRATE", "level"),
        "KXJOBS":   ("PAYEMS", "change"),
        "KXPAYEM":  ("PAYEMS", "change"),
    }

    ERROR_BAND = {
        "A191RL1Q225SBEA": 1.5, "CPIAUCSL": 0.15, "PCE": 0.15,
        "FEDFUNDS": 0.25, "UNRATE": 0.3, "PAYEMS": 80.0,
    }

    _cache = {}
    _CACHE_TTL = 3600

    def __init__(self):
        self.api_key = FRED_API_KEY

    @staticmethod
    def _parse_econ_ticker(ticker: str) -> Tuple[Optional[str], Optional[float]]:
        if not ticker:
            return None, None
        parts = ticker.upper().split("-")
        prefix = parts[0]
        threshold = None
        i = 1
        while i < len(parts):
            p = parts[i]
            m = re.match(r'^[TBH](\d+(?:\.\d+)?)$', p)
            if m:
                threshold = float(m.group(1))
                break
            if re.match(r'^[TBH]$', p) and i + 1 < len(parts):
                try:
                    threshold = -float(parts[i + 1])
                    break
                except ValueError:
                    pass
            i += 1
        return prefix, threshold

    def get_signal(self, market_title: str, market_category: str,
                   ticker: str = "") -> Optional[float]:
        if market_category not in ("economics",):
            return None
        prefix, threshold = self._parse_econ_ticker(ticker)
        if prefix is None or threshold is None:
            return None
        if prefix == "KXFEDDECISION":
            return None

        series_info = None
        for pfx, info in self.TICKER_SERIES.items():
            if prefix.startswith(pfx):
                series_info = info
                break
        if series_info is None:
            return None

        series_id, mode = series_info
        values = self._fetch_series(series_id)
        if values is None or len(values) < 4:
            return None

        if mode == "change":
            changes = []
            for i in range(len(values) - 1):
                if values[i + 1] != 0:
                    changes.append((values[i] - values[i + 1]) / values[i + 1] * 100)
            if len(changes) < 3:
                return None
            metric_values = changes
        else:
            metric_values = values

        recent_mean = statistics.mean(metric_values[:6])
        recent_std = statistics.stdev(metric_values[:6]) if len(metric_values[:6]) > 2 else 0.5
        error_band = self.ERROR_BAND.get(series_id, 1.0)
        forecast_std = max(recent_std, error_band)

        z = (recent_mean - threshold) / forecast_std
        prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        prob = max(0.05, min(0.95, prob))
        logger.debug(f"FRED {series_id}: mean={recent_mean:.2f} threshold={threshold} prob={prob:.3f}")
        return prob

    def _fetch_series(self, series_id: str) -> Optional[List[float]]:
        now = time.time()
        cached = self._cache.get(series_id)
        if cached and (now - cached[0]) < self._CACHE_TTL:
            return cached[1]
        try:
            params = {"series_id": series_id, "sort_order": "desc", "limit": 24, "file_type": "json"}
            if self.api_key:
                params["api_key"] = self.api_key
            resp = _retry_request("GET", FRED_BASE, params=params)
            data = resp.json()
            values = []
            for obs in data.get("observations", []):
                v = obs.get("value", ".")
                if v != ".":
                    try:
                        values.append(float(v))
                    except ValueError:
                        pass
            if len(values) >= 3:
                self._cache[series_id] = (now, values)
                return values
            return None
        except Exception as e:
            logger.debug(f"FRED {series_id} fetch failed: {e}")
            if cached:
                return cached[1]
            return None


class NWSSource:
    def get_signal(self, market_title: str, market_category: str,
                   ticker: str = "") -> Optional[float]:
        if market_category != "weather":
            return None
        city, threshold, is_high = self._parse_weather_ticker(ticker)
        if city is None:
            title_lower = market_title.lower()
            for code, info in TICKER_CITY_MAP.items():
                if info["name"].lower() in title_lower or code.lower() in title_lower:
                    city = info
                    break
            if city is None:
                return None
            threshold = self._extract_threshold_from_title(market_title)
            is_high = True
        return self._fetch_forecast_probability(city, threshold, is_high, market_title)

    def _fetch_forecast_probability(self, city: Dict, threshold: Optional[float],
                                     is_high: bool, market_title: str) -> Optional[float]:
        try:
            headers = {"User-Agent": "(KalshiTrader, contact@example.com)"}
            point_url = f"{NWS_BASE}/points/{city['lat']},{city['lon']}"
            resp = _retry_request("GET", point_url, headers=headers)
            point_data = resp.json()
            forecast_url = point_data.get("properties", {}).get("forecast")
            if not forecast_url:
                return None
            resp = _retry_request("GET", forecast_url, headers=headers)
            forecast = resp.json()
            periods = forecast.get("properties", {}).get("periods", [])
            if not periods:
                return None

            if is_high:
                temps = [p["temperature"] for p in periods[:4]
                         if p.get("isDaytime", True) and p.get("temperature") is not None]
            else:
                temps = [p["temperature"] for p in periods[:4]
                         if not p.get("isDaytime", True) and p.get("temperature") is not None]
            if not temps:
                temps = [p["temperature"] for p in periods[:4] if p.get("temperature") is not None]
            if not temps:
                return None

            forecast_temp = temps[0]
            if threshold is not None:
                z = (forecast_temp - threshold) / NWS_FORECAST_ERROR
                prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
                title_lower = market_title.lower()
                if any(kw in title_lower for kw in ("below", "under", "lower", "colder")):
                    prob = 1.0 - prob
            elif any(kw in market_title.lower() for kw in ("rain", "snow", "precip")):
                precip_probs = []
                for p in periods[:4]:
                    pp = p.get("probabilityOfPrecipitation", {})
                    if isinstance(pp, dict) and pp.get("value") is not None:
                        precip_probs.append(pp["value"] / 100.0)
                prob = statistics.mean(precip_probs) if precip_probs else None
                if prob is None:
                    return None
            else:
                return None
            prob = max(0.05, min(0.95, prob))
            return prob
        except Exception as e:
            logger.debug(f"NWS forecast failed for {city['name']}: {e}")
            return None

    @staticmethod
    def _parse_weather_ticker(ticker: str) -> Tuple[Optional[Dict], Optional[float], bool]:
        if not ticker:
            return None, None, True
        parts = ticker.split("-")
        if not parts:
            return None, None, True
        prefix = parts[0].upper()
        is_high = True
        city_code = None
        if prefix.startswith("KXHIGH"):
            city_code = prefix[6:]
        elif prefix.startswith("KXLOW"):
            city_code = prefix[5:]
            is_high = False
        elif prefix.startswith("KXRAIN") or prefix.startswith("KXSNOW"):
            city_code = prefix[6:]
        city = TICKER_CITY_MAP.get(city_code) if city_code else None
        threshold = None
        for part in parts[1:]:
            m = re.match(r'^[A-Za-z](\d+(?:\.\d+)?)$', part)
            if m:
                try:
                    threshold = float(m.group(1))
                except ValueError:
                    pass
                break
        return city, threshold, is_high

    @staticmethod
    def _extract_threshold_from_title(title: str) -> Optional[float]:
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:[°Ff]|degrees?)', title)
        if match:
            return float(match.group(1))
        return None


class GFSEnsembleSource:
    ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
    _cache = {}
    _CACHE_TTL = 1800

    @staticmethod
    def _parse_target_date(ticker: str) -> Optional[str]:
        parts = ticker.upper().split("-")
        if len(parts) < 2:
            return None
        date_part = parts[1]
        m = re.match(r'^(\d{2})([A-Z]{3})(\d{2})$', date_part)
        if not m:
            return None
        year = 2000 + int(m.group(1))
        month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                     "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
        month = month_map.get(m.group(2))
        if not month:
            return None
        day = int(m.group(3))
        try:
            return f"{year}-{month:02d}-{day:02d}"
        except Exception:
            return None

    def get_signal(self, market_title: str, market_category: str,
                   ticker: str = "") -> Optional[float]:
        if market_category != "weather":
            return None
        city, threshold, is_high = NWSSource._parse_weather_ticker(ticker)
        if city is None or threshold is None:
            return None
        target_date = self._parse_target_date(ticker)
        if not target_date:
            return None
        city_code = None
        for code, info in TICKER_CITY_MAP.items():
            if info == city:
                city_code = code
                break
        if not city_code:
            return None

        cache_key = (city_code, target_date, is_high)
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached and (now - cached[0]) < self._CACHE_TTL:
            member_extremes = cached[1]
        else:
            member_extremes = self._fetch_ensemble(city, target_date, is_high)
            if member_extremes is not None:
                self._cache[cache_key] = (now, member_extremes)
        if not member_extremes:
            return None

        if is_high:
            exceeding = sum(1 for t in member_extremes if t >= threshold)
        else:
            exceeding = sum(1 for t in member_extremes if t <= threshold)
        prob = exceeding / len(member_extremes)

        title_lower = market_title.lower()
        if is_high and any(kw in title_lower for kw in ("below", "under", "lower", "colder")):
            prob = 1.0 - prob
        prob = max(0.05, min(0.95, prob))

        spread = max(member_extremes) - min(member_extremes)
        logger.info(
            f"GFS Ensemble {city['name']}: {len(member_extremes)} members, "
            f"threshold={threshold}, prob={prob:.3f}, spread={spread:.1f}F"
        )
        return prob

    def _fetch_ensemble(self, city: Dict, target_date: str,
                        is_high: bool) -> Optional[List[float]]:
        try:
            params = {
                "latitude": city["lat"], "longitude": city["lon"],
                "hourly": "temperature_2m", "temperature_unit": "fahrenheit",
                "models": "gfs_seamless",
                "start_date": target_date, "end_date": target_date,
            }
            resp = _retry_request("GET", self.ENSEMBLE_URL, params=params)
            data = resp.json()
            hourly = data.get("hourly", {})
            time_list = hourly.get("time", [])
            if not time_list:
                return None

            member_series = {}
            for key, values in hourly.items():
                if key.startswith("temperature_2m_member"):
                    member_series[key] = values
            if not member_series:
                single = hourly.get("temperature_2m")
                if single and isinstance(single, list):
                    member_series["temperature_2m"] = single
            if not member_series:
                return None

            member_extremes = []
            for member_key, temps in member_series.items():
                if not temps or all(t is None for t in temps):
                    continue
                if is_high:
                    daytime = [t for i, t in enumerate(temps) if t is not None and 10 <= i <= 18]
                    if daytime:
                        member_extremes.append(max(daytime))
                    elif any(t is not None for t in temps):
                        member_extremes.append(max(t for t in temps if t is not None))
                else:
                    nighttime = [t for i, t in enumerate(temps) if t is not None and (i <= 8 or i >= 22)]
                    if nighttime:
                        member_extremes.append(min(nighttime))
                    elif any(t is not None for t in temps):
                        member_extremes.append(min(t for t in temps if t is not None))
            if len(member_extremes) < 5:
                return None
            return member_extremes
        except Exception as e:
            logger.debug(f"GFS ensemble fetch failed for {city['name']}: {e}")
            return None


class ClevelandFedSource:
    NOWCAST_URL = "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
    _cache = None
    _CACHE_TTL = 14400

    def get_signal(self, market_title: str, market_category: str,
                   ticker: str = "") -> Optional[float]:
        if market_category != "economics":
            return None
        if not ticker.upper().startswith("KXCPI"):
            return None
        prefix, threshold = FREDSource._parse_econ_ticker(ticker)
        if threshold is None:
            return None
        nowcast = self._fetch_nowcast()
        if nowcast is None:
            return None
        error_band = 0.10
        z = (nowcast - threshold) / error_band
        prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        prob = max(0.05, min(0.95, prob))
        logger.info(f"CPI Nowcast: {nowcast:.2f}% vs threshold {threshold}%, prob={prob:.3f}")
        return prob

    def _fetch_nowcast(self) -> Optional[float]:
        now = time.time()
        if self._cache and (now - self._cache[0]) < self._CACHE_TTL:
            return self._cache[1]
        value = self._fetch_from_page()
        if value is not None:
            ClevelandFedSource._cache = (now, value)
            return value
        if self._cache:
            return self._cache[1]
        return None

    def _fetch_from_page(self) -> Optional[float]:
        try:
            resp = _retry_request("GET", self.NOWCAST_URL,
                                  headers={"User-Agent": "KalshiTrader/1.0"})
            text = resp.text
            patterns = [
                r'"(?:cpi|CPI|headline)"\s*:\s*"?(\-?\d+\.\d+)"?',
                r'(?:MoM|Month.over.Month|Monthly).*?(\-?\d+\.\d{2})',
                r'CPI.*?(\-?\d+\.\d{2})\s*(?:percent|%)',
                r'(\-?\d+\.\d{2})\s*(?:percent|%)\s*.*?CPI',
                r'nowcast.*?(\-?\d+\.\d{2})',
                r'(\-?\d+\.\d{2}).*?nowcast',
            ]
            for pattern in patterns:
                m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if m:
                    value = float(m.group(1))
                    if -2.0 <= value <= 2.0:
                        return value
            return None
        except Exception as e:
            logger.debug(f"Cleveland Fed page fetch failed: {e}")
            return None


class GDPNowSource:
    GDPNOW_SERIES = "GDPNOW"
    _cache = None
    _CACHE_TTL = 3600

    def __init__(self):
        self.api_key = FRED_API_KEY

    def get_signal(self, market_title: str, market_category: str,
                   ticker: str = "") -> Optional[float]:
        if market_category != "economics":
            return None
        if not ticker.upper().startswith("KXGDP"):
            return None
        prefix, threshold = FREDSource._parse_econ_ticker(ticker)
        if threshold is None:
            return None
        gdpnow = self._fetch_gdpnow()
        if gdpnow is None:
            return None
        error_band = 0.8
        z = (gdpnow - threshold) / error_band
        prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        prob = max(0.05, min(0.95, prob))
        logger.info(f"GDPNow: {gdpnow:.2f}% vs threshold {threshold}%, prob={prob:.3f}")
        return prob

    def _fetch_gdpnow(self) -> Optional[float]:
        now = time.time()
        if self._cache and (now - self._cache[0]) < self._CACHE_TTL:
            return self._cache[1]
        try:
            params = {"series_id": self.GDPNOW_SERIES, "sort_order": "desc",
                      "limit": 1, "file_type": "json"}
            if self.api_key:
                params["api_key"] = self.api_key
            resp = _retry_request("GET", FRED_BASE, params=params)
            data = resp.json()
            observations = data.get("observations", [])
            if observations:
                v = observations[0].get("value", ".")
                if v != ".":
                    value = float(v)
                    GDPNowSource._cache = (now, value)
                    return value
            return None
        except Exception as e:
            logger.debug(f"GDPNow fetch failed: {e}")
            if self._cache:
                return self._cache[1]
            return None


class WhaleTracker:
    DUNE_POLYMARKET_URL = "https://api.dune.com/api/v1/query/3294028/results"

    def get_signal(self, market_title: str, market_category: str, market_id: str) -> Optional[float]:
        return self._analyze_orderbook_skew(market_id)

    def _analyze_orderbook_skew(self, market_id: str) -> Optional[float]:
        try:
            client = KalshiClient()
            if not client._initialized:
                return None
            ob = client.get_orderbook(market_id)
            yes_levels = ob.get("yes", [])
            no_levels = ob.get("no", [])
            if not yes_levels and not no_levels:
                return None
            yes_depth = sum(level[1] for level in yes_levels) if yes_levels else 0
            no_depth = sum(level[1] for level in no_levels) if no_levels else 0
            total = yes_depth + no_depth
            if total == 0:
                return None
            skew_prob = yes_depth / total
            dampen_factor = min(0.5, 1000 / max(total, 1))
            dampened = dampen_factor * 0.5 + (1 - dampen_factor) * skew_prob
            return dampened
        except Exception as e:
            logger.debug(f"Orderbook skew analysis failed for {market_id}: {e}")
            return None


class EdgeDetector:
    """Multi-source edge detection engine."""

    def __init__(self, client: KalshiClient = None, db_path: str = None):
        self.client = client or KalshiClient()
        self.db_path = db_path or DB_PATH
        self.fred = FREDSource()
        self.nws = NWSSource()
        self.whale = WhaleTracker()
        self.gfs_ensemble = GFSEnsembleSource()
        self.cpi_nowcast = ClevelandFedSource()
        self.gdpnow = GDPNowSource()

        self.odds_api = None
        self.sentiment = None
        try:
            from bots.event_trading.odds_api import OddsAPI
            self.odds_api = OddsAPI()
            if not self.odds_api.api_key:
                self.odds_api = None
        except (ImportError, Exception):
            pass

        try:
            from bots.event_trading.sentiment_scraper import SentimentScraper
            self.sentiment = SentimentScraper()
        except (ImportError, Exception):
            pass

        self.polymarket = None

        init_db(self.db_path)

    @staticmethod
    def _categorize_market(ticker: str, category: str, title: str) -> str:
        t = (ticker or "").upper()
        if any(t.startswith(p) for p in ("KXHIGH", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP")):
            return "weather"
        if any(t.startswith(p) for p in ("KXFED", "KXGDP", "KXCPI", "KXPCE", "KXJOBS", "KXUNRATE", "KXPAYEM")):
            return "economics"
        if category in ("weather",):
            return "weather"
        if category in ("economics", "economy"):
            return "economics"
        if category in ("sports",):
            return "sports"
        return "generic"

    def _get_domain_signals(self, domain: str, market_id: str, title: str,
                            category: str, market_prob: float) -> Tuple[List[float], List[float], Dict]:
        signal_dict = {}

        if domain == "weather":
            gfs = self.gfs_ensemble.get_signal(title, "weather", ticker=market_id)
            ws = self.nws.get_signal(title, "weather", ticker=market_id)
            wh = self.whale.get_signal(title, "weather", market_id)
            if gfs is not None: signal_dict["gfs_ensemble"] = gfs
            if ws is not None: signal_dict["nws"] = ws
            if wh is not None: signal_dict["whale"] = wh

        elif domain == "economics":
            fs = self.fred.get_signal(title, "economics", ticker=market_id)
            cpi = self.cpi_nowcast.get_signal(title, "economics", ticker=market_id)
            gdp = self.gdpnow.get_signal(title, "economics", ticker=market_id)
            wh = self.whale.get_signal(title, category, market_id)
            ss = self._get_sentiment_signal(title, "economics") if self.sentiment else None
            if fs is not None: signal_dict["fred"] = fs
            if cpi is not None: signal_dict["cpi_nowcast"] = cpi
            if gdp is not None: signal_dict["gdpnow"] = gdp
            if wh is not None: signal_dict["whale"] = wh
            if ss is not None: signal_dict["sentiment"] = ss

        elif domain == "sports":
            os_sig = self._get_odds_signal(market_id, title, "sports", market_prob) if self.odds_api else None
            ss = self._get_sentiment_signal(title, "sports") if self.sentiment else None
            wh = self.whale.get_signal(title, category, market_id)
            if os_sig is not None: signal_dict["odds"] = os_sig
            if ss is not None: signal_dict["sentiment"] = ss
            if wh is not None: signal_dict["whale"] = wh

        else:
            fs = self.fred.get_signal(title, category, ticker=market_id)
            ws = self.nws.get_signal(title, category, ticker=market_id)
            wh = self.whale.get_signal(title, category, market_id)
            os_sig = self._get_odds_signal(market_id, title, category, market_prob) if self.odds_api else None
            ss = self._get_sentiment_signal(title, category) if self.sentiment else None
            if fs is not None: signal_dict["fred"] = fs
            if ws is not None: signal_dict["weather"] = ws
            if wh is not None: signal_dict["whale"] = wh
            if os_sig is not None: signal_dict["odds"] = os_sig
            if ss is not None: signal_dict["sentiment"] = ss

        weights_dict = self._get_weights(domain, signal_dict)
        signals = []
        weights = []
        for source, sig_val in signal_dict.items():
            if source in weights_dict:
                signals.append(sig_val)
                weights.append(weights_dict[source])

        return signals, weights, signal_dict

    def _get_weights(self, domain: str, available_signals: Dict) -> Dict[str, float]:
        try:
            from bots.event_trading.source_calibration import get_adaptive_weights
            adaptive = get_adaptive_weights(domain)
            if adaptive:
                filtered = {s: w for s, w in adaptive.items() if s in available_signals}
                if len(filtered) >= 2:
                    total = sum(filtered.values())
                    return {s: w / total for s, w in filtered.items()}
        except (ImportError, Exception):
            pass

        domain_defaults = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["generic"])
        filtered = {s: w for s, w in domain_defaults.items() if s in available_signals}
        if filtered:
            total = sum(filtered.values())
            return {s: w / total for s, w in filtered.items()}

        n = len(available_signals)
        return {s: 1.0 / n for s in available_signals} if n > 0 else {}

    def _record_price_snapshot(self, market_id: str, yes_price: float,
                                no_price: float = None, volume: int = None):
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            with conn:
                conn.execute(
                    """INSERT INTO price_snapshots (market_id, price, yes_price, no_price, volume, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (market_id, yes_price, yes_price, no_price, volume, mt_str()),
                )
            conn.close()
        except Exception as e:
            logger.debug(f"Price snapshot failed for {market_id}: {e}")

    def analyze(self, market: Dict) -> Optional[EdgeSignal]:
        market_id = market.get("market_id") or market.get("ticker", "")
        title = market.get("title", "")
        category = market.get("category", "other")
        yes_price = market.get("yes_price") or market.get("current_yes_price")

        if not market_id or yes_price is None:
            return None
        if any(market_id.startswith(prefix) for prefix in BLOCKED_PREFIXES):
            return None

        market_prob = yes_price / 100.0 if yes_price > 1 else yes_price
        if market_prob < 0.05 or market_prob > 0.95:
            return None

        self._record_price_snapshot(market_id, yes_price,
                                    market.get("no_price"), market.get("volume_24h"))

        domain = self._categorize_market(market_id, category, title)
        signals, weights, signal_dict = self._get_domain_signals(
            domain, market_id, title, category, market_prob)

        if not signals:
            return None

        total_weight = sum(weights)
        ensemble_prob = sum(s * w for s, w in zip(signals, weights)) / total_weight

        if len(signals) == 1:
            confidence = 0.40
        elif len(signals) == 2:
            stddev = statistics.stdev(signals)
            agreement = max(0.0, min(1.0, 1.0 - (stddev * 2.0)))
            confidence = agreement * 0.78
        else:
            stddev = statistics.stdev(signals)
            agreement = max(0.0, min(1.0, 1.0 - (stddev * 2.0)))
            confidence = agreement * min(1.0, 0.30 * len(signals))

        edge = ensemble_prob - market_prob
        direction = "YES" if edge > 0 else "NO"

        if direction == "YES":
            ev_per_contract = edge * (100 - yes_price) / 100.0
        else:
            ev_per_contract = abs(edge) * yes_price / 100.0

        signal = EdgeSignal(
            market_id=market_id, ticker=market_id,
            edge=edge, confidence=confidence, direction=direction,
            expected_value=ev_per_contract,
            ensemble_probability=ensemble_prob, market_probability=market_prob,
            fred_signal=signal_dict.get("fred"),
            weather_signal=signal_dict.get("weather") or signal_dict.get("nws"),
            whale_signal=signal_dict.get("whale"),
            gfs_ensemble_signal=signal_dict.get("gfs_ensemble"),
            cpi_nowcast_signal=signal_dict.get("cpi_nowcast"),
            gdpnow_signal=signal_dict.get("gdpnow"),
            nws_signal=signal_dict.get("nws"),
        )
        self._log_signal(signal)
        return signal

    def analyze_batch(self, markets: List[Dict]) -> List[EdgeSignal]:
        signals = []
        for m in markets:
            try:
                sig = self.analyze(m)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                logger.error(f"Analysis failed for {m.get('market_id', '?')}: {e}")
        signals.sort(key=lambda s: abs(s.edge), reverse=True)
        return signals

    def _get_odds_signal(self, market_id: str, title: str, category: str,
                         market_prob: float) -> Optional[float]:
        if not self.odds_api or category not in ("sports", "entertainment"):
            return None
        if not hasattr(self, '_odds_cache'):
            self._odds_cache = {}
            self._odds_cache_time = None
        now = time.time()
        if self._odds_cache_time is None or now - self._odds_cache_time > 300:
            try:
                events = self.odds_api.get_odds("basketball_ncaab")
                if events:
                    probs = self.odds_api.extract_book_probabilities(events)
                    self._odds_cache = {p["event_name"]: p for p in probs}
                    self._odds_cache_time = now
            except Exception:
                return None
        title_words = set(title.lower().split())
        for event_name, data in self._odds_cache.items():
            event_words = set(event_name.lower().split())
            if len(title_words & event_words) >= 2 and data.get("consensus"):
                consensus_values = list(data["consensus"].values())
                if consensus_values:
                    return max(0.05, min(0.95, consensus_values[0]))
        return None

    def _get_sentiment_signal(self, title: str, category: str) -> Optional[float]:
        if not self.sentiment:
            return None
        try:
            query = self.sentiment.build_query(title, category)
            signals = self.sentiment.scan_news(query, days_back=1)
            if not signals:
                return None
            agg = self.sentiment.aggregate_sentiment(signals)
            return max(0.05, min(0.95, 0.5 + agg * 0.25))
        except Exception as e:
            logger.debug(f"Sentiment signal failed: {e}")
            return None

    def _log_signal(self, signal: EdgeSignal):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            with conn:
                conn.execute(
                    """INSERT INTO edge_signals
                       (market_id, edge, confidence, ensemble_probability,
                        market_probability, direction, fred_signal, weather_signal,
                        whale_signal, expected_value, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (signal.market_id, signal.edge, signal.confidence,
                     signal.ensemble_probability, signal.market_probability,
                     signal.direction, signal.fred_signal, signal.weather_signal,
                     signal.whale_signal, signal.expected_value, signal.timestamp),
                )
        finally:
            conn.close()
