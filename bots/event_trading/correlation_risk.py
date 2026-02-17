"""
Correlation Risk Manager
Tracks all open positions and computes portfolio-level correlation risk.
Blocks new trades that would exceed correlation limits.

Rules:
- Same event/series: ~0.9 correlation
- Same category: ~0.5 correlation
- Opposing positions: -1.0 correlation
- Different categories: 0.0 correlation

All timestamps in Mountain Time (America/Denver).
"""

import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DB_PATH = os.path.join(DATA_DIR, "live", "event_trading.db")

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

MT = ZoneInfo("America/Denver")
logger = logging.getLogger("EventEdge.CorrelationRisk")


def mt_now() -> datetime:
    return datetime.now(MT)


def mt_str(dt: datetime = None) -> str:
    return (dt or mt_now()).strftime("%Y-%m-%d %H:%M:%S MT")


@dataclass
class PositionRisk:
    """A position's risk profile for correlation analysis."""
    ticker: str
    side: str                # "yes" or "no"
    exposure: float          # dollars at risk
    category: str = ""       # e.g. "economics", "sports", "crypto"
    event_group: str = ""    # group correlated positions (e.g. "KXNCAA", "KXSB")

    @property
    def series_prefix(self) -> str:
        """Extract series prefix from ticker (e.g. 'KXNCAA' from 'KXNCAA-26-DUKE')."""
        parts = self.ticker.split("-")
        return parts[0] if parts else self.ticker

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "side": self.side,
            "exposure": round(self.exposure, 2),
            "category": self.category,
            "event_group": self.event_group or self.series_prefix,
        }


class CorrelationRiskManager:
    """
    Portfolio-level correlation risk manager.
    Tracks positions and enforces correlation-based exposure limits.
    """

    def __init__(self, max_correlated_exposure: float = 200.0,
                 max_single_category: float = 300.0, db_path: str = None):
        """
        Args:
            max_correlated_exposure: Max dollars at risk in highly correlated positions
            max_single_category: Max dollars at risk in a single category
        """
        self.max_correlated_exposure = max_correlated_exposure
        self.max_single_category = max_single_category
        self.db_path = db_path or DB_PATH
        self.positions: List[PositionRisk] = []

    def add_position(self, pos: PositionRisk):
        """Add a position to the portfolio."""
        self.positions.append(pos)
        logger.info(f"Added position: {pos.ticker} {pos.side} ${pos.exposure:.2f}")

    def remove_position(self, ticker: str):
        """Remove a position by ticker."""
        self.positions = [p for p in self.positions if p.ticker != ticker]

    def load_from_db(self):
        """Load open positions from autonomous_trades table."""
        self.positions.clear()
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM autonomous_trades WHERE status='OPEN'"
            ).fetchall()
            conn.close()

            for r in rows:
                r = dict(r)
                exposure = (r.get("entry_price", 0) * r.get("quantity", 0)) / 100.0
                ticker = r.get("ticker", "")
                side = r.get("side", "yes")

                # Infer category from ticker patterns
                category = self._infer_category(ticker)
                event_group = ticker.split("-")[0] if "-" in ticker else ticker

                self.positions.append(PositionRisk(
                    ticker=ticker,
                    side=side,
                    exposure=exposure,
                    category=category,
                    event_group=event_group,
                ))

            logger.info(f"Loaded {len(self.positions)} positions from DB")
        except Exception as e:
            logger.error(f"Failed to load positions from DB: {e}")

    @staticmethod
    def _infer_category(ticker: str) -> str:
        """Infer market category from ticker prefix."""
        prefix = ticker.split("-")[0].upper() if "-" in ticker else ticker.upper()
        category_map = {
            "KXNCAA": "sports", "KXCBB": "sports", "KXMM": "sports",
            "KXSB": "sports", "KXNFL": "sports", "KXNBA": "sports",
            "KXMLB": "sports", "KXNHL": "sports",
            "KXFED": "economics", "KXCPI": "economics", "KXGDP": "economics",
            "KXJOBS": "economics", "KXRATE": "economics",
            "KXBTC": "crypto", "KXETH": "crypto",
            "KXPRES": "politics", "KXELEC": "politics",
            "KXTEMP": "weather", "KXRAIN": "weather",
        }
        return category_map.get(prefix, "other")

    def get_correlation(self, ticker_a: str, ticker_b: str) -> float:
        """
        Estimate correlation between two positions.

        Returns:
            Float from -1.0 to 1.0
        """
        if ticker_a == ticker_b:
            return 1.0

        prefix_a = ticker_a.split("-")[0] if "-" in ticker_a else ticker_a
        prefix_b = ticker_b.split("-")[0] if "-" in ticker_b else ticker_b

        # Same series = high correlation
        if prefix_a == prefix_b:
            return 0.9

        cat_a = self._infer_category(ticker_a)
        cat_b = self._infer_category(ticker_b)

        # Same category = moderate correlation
        if cat_a == cat_b:
            return 0.5

        # Cross-category = low/no correlation
        return 0.0

    def _check_opposing(self, pos_a: PositionRisk, pos_b: PositionRisk) -> float:
        """Check if two positions are opposing (YES vs NO on same/related market)."""
        prefix_a = pos_a.series_prefix
        prefix_b = pos_b.series_prefix

        if prefix_a == prefix_b and pos_a.side != pos_b.side:
            return -1.0  # Opposing positions in same series
        return self.get_correlation(pos_a.ticker, pos_b.ticker)

    def can_add(self, new_pos: PositionRisk) -> Tuple[bool, str]:
        """
        Check if adding a new position would exceed correlation limits.

        Returns:
            (allowed: bool, reason: str)
        """
        if not self.positions:
            return True, "No existing positions — trade allowed"

        # Check 1: Category exposure
        category_exposure = sum(
            p.exposure for p in self.positions if p.category == new_pos.category
        )
        if category_exposure + new_pos.exposure > self.max_single_category:
            return False, (
                f"Category '{new_pos.category}' exposure would be "
                f"${category_exposure + new_pos.exposure:.2f} "
                f"(limit: ${self.max_single_category:.2f})"
            )

        # Check 2: Correlated exposure (sum of exposure * correlation for all positions)
        correlated_exposure = 0.0
        for p in self.positions:
            corr = self._check_opposing(p, new_pos)
            if corr > 0:
                correlated_exposure += p.exposure * corr

        if correlated_exposure + new_pos.exposure > self.max_correlated_exposure:
            return False, (
                f"Correlated exposure would be ${correlated_exposure + new_pos.exposure:.2f} "
                f"(limit: ${self.max_correlated_exposure:.2f})"
            )

        # Check 3: Opposing position warning (allow but warn)
        for p in self.positions:
            if self._check_opposing(p, new_pos) < 0:
                return True, (
                    f"WARNING: Opposing position detected — "
                    f"{p.ticker} {p.side} vs {new_pos.ticker} {new_pos.side}. "
                    f"This creates a hedge but locks in a spread cost."
                )

        return True, "Trade within risk limits"

    def max_correlated_loss(self) -> float:
        """
        Compute maximum possible loss if all correlated positions move against us.
        Worst case: all highly correlated positions lose simultaneously.
        """
        if not self.positions:
            return 0.0

        # Group by event_group
        groups: Dict[str, float] = {}
        for p in self.positions:
            group = p.event_group or p.series_prefix
            groups[group] = groups.get(group, 0) + p.exposure

        # Max correlated loss = largest group exposure
        return max(groups.values()) if groups else 0.0

    def diversification_score(self) -> float:
        """
        Compute diversification score (0-1).
        1.0 = perfectly diversified (all uncorrelated)
        0.0 = completely concentrated (all same series)
        """
        if len(self.positions) <= 1:
            return 0.0

        total_exposure = sum(p.exposure for p in self.positions)
        if total_exposure == 0:
            return 0.0

        # Count unique categories and event groups
        categories = set(p.category for p in self.positions)
        groups = set(p.event_group or p.series_prefix for p in self.positions)

        # Herfindahl index by group (lower = more diversified)
        group_exposures: Dict[str, float] = {}
        for p in self.positions:
            g = p.event_group or p.series_prefix
            group_exposures[g] = group_exposures.get(g, 0) + p.exposure

        hhi = sum((e / total_exposure) ** 2 for e in group_exposures.values())

        # Convert HHI to diversification score (1/N is minimum HHI for N groups)
        n = len(group_exposures)
        min_hhi = 1.0 / n if n > 0 else 1.0
        if hhi <= min_hhi:
            return 1.0

        # Score: 1.0 when HHI = min, 0.0 when HHI = 1.0
        return max(0.0, 1.0 - (hhi - min_hhi) / (1.0 - min_hhi))

    def portfolio_risk_summary(self) -> Dict:
        """Get complete portfolio risk summary."""
        total_exposure = sum(p.exposure for p in self.positions)

        # Group by category
        by_category: Dict[str, float] = {}
        for p in self.positions:
            by_category[p.category] = by_category.get(p.category, 0) + p.exposure

        # Group by event group
        by_group: Dict[str, float] = {}
        for p in self.positions:
            g = p.event_group or p.series_prefix
            by_group[g] = by_group.get(g, 0) + p.exposure

        return {
            "total_positions": len(self.positions),
            "total_exposure": round(total_exposure, 2),
            "max_correlated_loss": round(self.max_correlated_loss(), 2),
            "diversification_score": round(self.diversification_score(), 4),
            "by_category": {k: round(v, 2) for k, v in sorted(by_category.items())},
            "by_group": {k: round(v, 2) for k, v in sorted(by_group.items(), key=lambda x: -x[1])},
            "positions": [p.to_dict() for p in self.positions],
            "limits": {
                "max_correlated_exposure": self.max_correlated_exposure,
                "max_single_category": self.max_single_category,
            },
            "timestamp": mt_str(),
        }

    def print_summary(self):
        """Pretty-print portfolio risk summary."""
        summary = self.portfolio_risk_summary()

        print("\n" + "=" * 70)
        print(f"  PORTFOLIO CORRELATION RISK — {mt_str()}")
        print("=" * 70)

        print(f"\n  Positions: {summary['total_positions']}")
        print(f"  Total Exposure: ${summary['total_exposure']:.2f}")
        print(f"  Max Correlated Loss: ${summary['max_correlated_loss']:.2f}")
        print(f"  Diversification Score: {summary['diversification_score']:.2f}")

        if summary["by_category"]:
            print(f"\n  By Category:")
            for cat, exp in summary["by_category"].items():
                pct = exp / summary["total_exposure"] * 100 if summary["total_exposure"] else 0
                bar = "#" * int(pct / 5)
                print(f"    {cat:15s}: ${exp:>8.2f} ({pct:4.1f}%) {bar}")

        if summary["by_group"]:
            print(f"\n  By Event Group:")
            for grp, exp in summary["by_group"].items():
                print(f"    {grp:15s}: ${exp:>8.2f}")

        if summary["positions"]:
            print(f"\n  Positions:")
            for p in summary["positions"]:
                print(f"    {p['ticker']:25s} {p['side']:3s} ${p['exposure']:>8.2f} [{p['category']}]")

        print(f"\n  Limits: max_correlated=${summary['limits']['max_correlated_exposure']:.0f}"
              f" max_category=${summary['limits']['max_single_category']:.0f}")
        print("=" * 70 + "\n")
