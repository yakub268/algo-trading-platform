"""
Price Intelligence Engine
Analyzes price_snapshots data for smart entry timing.

Features:
- Drift detection: is price converging or diverging from our estimate?
- Mean reversion: detect spikes and wait for reversal
- VWAP analysis: are we buying above or below average?
- Smart entry timing: avoid buying during spikes
"""

import os
import sqlite3
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from zoneinfo import ZoneInfo

MT = ZoneInfo("America/Denver")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DB_PATH = os.path.join(DATA_DIR, "live", "event_trading.db")

SPIKE_THRESHOLD = 10   # cents — price move > this in < 1 hour = spike
MIN_SNAPSHOTS = 3      # minimum snapshots for analysis

logger = logging.getLogger("EventEdge.PriceIntelligence")


@dataclass
class PriceSignal:
    """Price intelligence signal for a market."""
    ticker: str
    drift: float            # -1 to +1: negative = price falling, positive = rising
    is_spike: bool          # True if recent spike detected
    spike_direction: str    # "up", "down", or "none"
    vwap: float             # Volume-weighted average price (cents)
    current_price: float    # Latest price (cents)
    vwap_deviation: float   # Current price vs VWAP
    volatility: float       # Price std dev over window
    entry_quality: str      # "good", "neutral", "bad"
    snapshots_used: int

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "drift": round(self.drift, 4),
            "is_spike": self.is_spike,
            "spike_direction": self.spike_direction,
            "vwap": round(self.vwap, 2),
            "current_price": round(self.current_price, 2),
            "vwap_deviation": round(self.vwap_deviation, 2),
            "volatility": round(self.volatility, 2),
            "entry_quality": self.entry_quality,
            "snapshots_used": self.snapshots_used,
        }


def get_price_history(ticker: str, hours: float = 4.0) -> List[Dict]:
    """Get price snapshots for a ticker over the last N hours."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    cutoff = (datetime.now(MT) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    rows = conn.execute(
        """SELECT * FROM price_snapshots
           WHERE market_id=? AND timestamp >= ?
           ORDER BY timestamp ASC""",
        (ticker, cutoff)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def analyze_price(ticker: str) -> Optional[PriceSignal]:
    """
    Analyze price trajectory for a market.
    Returns PriceSignal or None if insufficient data.
    """
    snapshots = get_price_history(ticker, hours=4.0)
    if len(snapshots) < MIN_SNAPSHOTS:
        return None

    prices = []
    volumes = []
    for s in snapshots:
        p = s.get("yes_price") or s.get("price", 0)
        v = s.get("volume", 1) or 1
        if p and p > 0:
            prices.append(p)
            volumes.append(max(v, 1))

    if len(prices) < MIN_SNAPSHOTS:
        return None

    current_price = prices[-1]

    # Drift: linear regression slope normalized
    n = len(prices)
    x_mean = (n - 1) / 2
    y_mean = statistics.mean(prices)
    numerator = sum((i - x_mean) * (p - y_mean) for i, p in enumerate(prices))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    slope = numerator / denominator if denominator > 0 else 0
    price_range = max(prices) - min(prices) if max(prices) != min(prices) else 1
    drift = max(-1.0, min(1.0, slope / max(price_range, 1)))

    # Spike detection
    is_spike = False
    spike_direction = "none"
    if len(prices) >= 2:
        recent_move = prices[-1] - prices[-2]
        recent_move_3 = prices[-1] - prices[-3] if len(prices) >= 3 else recent_move
        max_move = max(abs(recent_move), abs(recent_move_3))
        if max_move >= SPIKE_THRESHOLD:
            is_spike = True
            spike_direction = "up" if (recent_move > 0 or recent_move_3 > 0) else "down"

    # VWAP
    total_pv = sum(p * v for p, v in zip(prices, volumes))
    total_v = sum(volumes)
    vwap = total_pv / total_v if total_v > 0 else current_price
    vwap_deviation = current_price - vwap

    # Volatility
    volatility = statistics.stdev(prices) if len(prices) > 1 else 0.0

    # Entry quality assessment
    if is_spike and spike_direction == "up":
        entry_quality = "bad"
    elif is_spike and spike_direction == "down":
        entry_quality = "good"
    elif vwap_deviation < -2:
        entry_quality = "good"
    elif vwap_deviation > 5:
        entry_quality = "bad"
    else:
        entry_quality = "neutral"

    return PriceSignal(
        ticker=ticker, drift=drift, is_spike=is_spike,
        spike_direction=spike_direction, vwap=vwap,
        current_price=current_price, vwap_deviation=vwap_deviation,
        volatility=volatility, entry_quality=entry_quality,
        snapshots_used=len(snapshots),
    )


def should_enter(ticker: str, direction: str) -> Tuple[bool, str]:
    """
    Quick check: should we enter this market now based on price intelligence?
    Returns (should_enter: bool, reason: str)
    """
    signal = analyze_price(ticker)
    if signal is None:
        return True, "insufficient_price_data"

    if signal.entry_quality == "bad":
        if direction == "YES" and signal.spike_direction == "up":
            return False, f"spike_up: price spiked +{abs(signal.vwap_deviation):.0f}c, wait for reversion"
        elif direction == "NO" and signal.spike_direction == "down":
            return False, f"spike_down: price dropped, wait for reversion"
        elif signal.vwap_deviation > 5 and direction == "YES":
            return False, f"above_vwap: price {signal.vwap_deviation:+.0f}c above VWAP"
        return False, "bad_entry_quality"

    return True, signal.entry_quality
