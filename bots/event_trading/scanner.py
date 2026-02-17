"""
Autonomous Kalshi Market Discovery Engine
Scans ALL open Kalshi markets, filters by volume/category, ranks by liquidity + price movement.
"""

import os
import time
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from bots.kalshi_client import KalshiClient

MT = ZoneInfo("America/Denver")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DB_PATH = os.path.join(DATA_DIR, "live", "event_trading.db")

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2

INCLUDE_CATEGORIES = {
    "economics", "economy", "fed", "fomc", "gdp", "cpi", "inflation", "unemployment",
    "jobs", "nonfarm", "interest rate", "treasury", "recession",
    "weather", "temperature", "snowfall", "rainfall", "hurricane",
    "election", "politics", "congress", "senate",
}

EXCLUDE_PATTERNS = {
    "super bowl lx", "kxsb-26", "superbowl", "sb lx", "nfl", "super bowl",
}

MIN_VOLUME_24H = 5000

logger = logging.getLogger("EventEdge.Scanner")


def mt_now() -> datetime:
    return datetime.now(MT)


def mt_str(dt: datetime = None) -> str:
    return (dt or mt_now()).strftime("%Y-%m-%d %H:%M:%S MT")


@dataclass
class DiscoveredMarket:
    market_id: str
    ticker: str
    title: str
    subtitle: str
    category: str
    volume_24h: float
    open_interest: int
    yes_price: Optional[float]
    no_price: Optional[float]
    price_change_24h: float
    liquidity_score: float
    discovery_timestamp: str

    @property
    def current_yes_price(self) -> float:
        return self.yes_price or 0.0

    def to_dict(self) -> Dict:
        return {
            "market_id": self.market_id,
            "ticker": self.ticker,
            "title": self.title,
            "subtitle": self.subtitle,
            "category": self.category,
            "volume_24h": self.volume_24h,
            "open_interest": self.open_interest,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "price_change_24h": self.price_change_24h,
            "liquidity_score": self.liquidity_score,
            "discovery_timestamp": self.discovery_timestamp,
        }


def _retry_api_call(func, *args, **kwargs):
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == RETRY_ATTEMPTS:
                logger.error(f"API call failed after {RETRY_ATTEMPTS} attempts: {e}")
                raise
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning(f"API call attempt {attempt} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)


def init_db(db_path: str = None):
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS discovered_markets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT,
            ticker TEXT,
            volume_24h REAL,
            current_yes_price REAL,
            no_price REAL,
            open_interest INTEGER DEFAULT 0,
            category TEXT DEFAULT '',
            title TEXT DEFAULT '',
            subtitle TEXT DEFAULT '',
            price_change_24h REAL DEFAULT 0,
            liquidity_score REAL DEFAULT 0,
            discovery_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            settlement_result TEXT,
            settlement_time TIMESTAMP,
            UNIQUE(market_id, discovery_timestamp)
        )
    """)
    conn.commit()
    conn.close()


def _categorize_market(title: str, subtitle: str, ticker: str) -> str:
    blob = f"{title} {subtitle} {ticker}".lower()
    if any(kw in blob for kw in ("fed", "fomc", "rate", "treasury", "gdp", "cpi",
                                   "inflation", "unemployment", "jobs", "nonfarm",
                                   "recession", "economy", "economic")):
        return "economics"
    if any(kw in blob for kw in ("temperature", "weather", "snowfall", "rainfall",
                                   "hurricane", "tornado", "heat", "cold", "freeze",
                                   "high temp", "low temp", "kxhigh", "kxlow", "kxsnow",
                                   "kxrain", "kxwx")):
        return "weather"
    if any(kw in blob for kw in ("nfl", "nba", "mlb", "nhl", "soccer", "ufc",
                                   "boxing", "sports", "game", "match", "championship",
                                   "playoff", "finals", "series")):
        return "sports"
    if any(kw in blob for kw in ("bitcoin", "btc", "ethereum", "eth", "crypto",
                                   "solana", "sol")):
        return "crypto"
    if any(kw in blob for kw in ("election", "president", "congress", "senate",
                                   "governor", "democrat", "republican", "vote")):
        return "politics"
    if any(kw in blob for kw in ("oscar", "grammy", "emmy", "box office", "movie",
                                   "award", "entertainment")):
        return "entertainment"
    if any(kw in blob for kw in ("climate", "energy", "oil", "gas", "carbon")):
        return "energy"
    return "other"


def _is_excluded(title: str, subtitle: str, ticker: str) -> bool:
    blob = f"{title} {subtitle} {ticker}".lower()
    return any(pat in blob for pat in EXCLUDE_PATTERNS)


class AutonomousScanner:
    """Discovers all high-volume Kalshi markets across all categories."""

    def __init__(self, client: KalshiClient = None, db_path: str = None):
        self.client = client or KalshiClient()
        self.db_path = db_path or DB_PATH
        init_db(self.db_path)

    TARGET_SERIES = [
        "KXFED", "KXCPI", "KXGDP", "KXPCE", "KXJOBLESS",
        "KXRETAIL", "KXHOUSING", "KXISI",
        "KXHIGHNY", "KXHIGHLA", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHDEN",
        "KXSNOW", "KXRAIN",
        "KXTRUMP", "KXAPPROVAL", "KXGOV",
    ]

    def discover(self, top_n: int = 20) -> List[DiscoveredMarket]:
        logger.info(f"Market discovery scan at {mt_str()}")
        all_markets = {}

        for series in self.TARGET_SERIES:
            try:
                markets = _retry_api_call(self.client.get_markets, series_ticker=series, limit=200)
                added = 0
                for m in markets:
                    ticker = m.get("ticker", "")
                    if ticker and ticker not in all_markets:
                        all_markets[ticker] = m
                        added += 1
                if added > 0:
                    logger.debug(f"  {series}: {len(markets)} markets, {added} new")
            except Exception as e:
                logger.debug(f"  {series}: {e}")

        discoveries = []
        for ticker, m in all_markets.items():
            title = m.get("title", "")
            subtitle = m.get("subtitle", "")
            status = m.get("status", "")

            if status not in ("open", "active", None, ""):
                continue
            if _is_excluded(title, subtitle, ticker):
                continue

            category = _categorize_market(title, subtitle, ticker)
            volume = m.get("volume", 0) or 0
            volume_24h = m.get("volume_24h", volume) or volume
            if volume_24h < MIN_VOLUME_24H:
                continue

            open_interest = m.get("open_interest", 0) or 0
            yes_price = m.get("yes_bid") or m.get("last_price")
            no_price = m.get("no_bid")
            prev_price = m.get("previous_price") or m.get("previous_yes_bid")
            price_change = abs(yes_price - prev_price) if prev_price and yes_price else 0.0
            liquidity_score = (volume_24h * 0.6) + (open_interest * 0.3) + (price_change * 10000 * 0.1)

            dm = DiscoveredMarket(
                market_id=ticker, ticker=ticker, title=title, subtitle=subtitle,
                category=category, volume_24h=volume_24h, open_interest=open_interest,
                yes_price=yes_price, no_price=no_price, price_change_24h=price_change,
                liquidity_score=liquidity_score, discovery_timestamp=mt_str(),
            )
            discoveries.append(dm)

        discoveries.sort(key=lambda d: d.liquidity_score, reverse=True)
        top = discoveries[:top_n]

        logger.info(f"Discovery: {len(discoveries)} markets above ${MIN_VOLUME_24H} vol, returning top {len(top)}")
        self._log_discoveries(top)
        return top

    def _log_discoveries(self, markets: List[DiscoveredMarket]):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                for dm in markets:
                    conn.execute(
                        """INSERT OR REPLACE INTO discovered_markets
                           (market_id, ticker, volume_24h, current_yes_price, no_price,
                            open_interest, category, title, subtitle, price_change_24h,
                            liquidity_score, discovery_timestamp)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (dm.market_id, dm.ticker, dm.volume_24h, dm.yes_price, dm.no_price,
                         dm.open_interest, dm.category, dm.title, dm.subtitle,
                         dm.price_change_24h, dm.liquidity_score, dm.discovery_timestamp),
                    )
        finally:
            conn.close()
