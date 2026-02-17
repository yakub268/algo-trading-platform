"""
Consensus Engine for Kalshi Prediction Markets
When 2+ independent sources agree, fire a HIGH CONVICTION signal.
Returns ConsensusSignal list — does NOT execute trades directly.

The EventEdgeBot adapter handles execution via the orchestrator.
"""

import os
import time
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from bots.kalshi_client import KalshiClient
from bots.event_trading.edge_detector import EdgeDetector
from bots.event_trading.sentiment_scraper import SentimentScraper

try:
    from bots.event_trading.odds_api import OddsAPI
    _HAS_ODDS = True
except ImportError:
    _HAS_ODDS = False

MT = ZoneInfo("America/Denver")

MIN_CONSENSUS_SOURCES = 2
CONSENSUS_KELLY_FRACTION = 0.5   # Half-Kelly for consensus (2x normal)
CONSENSUS_MAX_POSITION = 75.0    # $75 max per consensus trade

logger = logging.getLogger("EventEdge.Consensus")


def mt_now() -> datetime:
    return datetime.now(MT)


@dataclass
class ConsensusReport:
    """Report of consensus across multiple signal sources."""
    market_id: str
    ticker: str
    sources_checked: int
    sources_with_data: int
    sources_agreeing_yes: int
    sources_agreeing_no: int
    consensus_direction: Optional[str]  # "YES", "NO", or None
    consensus_strength: float
    consensus_edge: float
    is_high_conviction: bool
    individual_signals: Dict[str, float]
    market_price: float
    timestamp: datetime


@dataclass
class ConsensusSignal:
    """Trading signal based on consensus across sources."""
    market_id: str
    ticker: str
    direction: str  # "YES" or "NO"
    consensus_edge: float
    consensus_strength: float
    sources_agreeing: int
    total_sources: int
    is_high_conviction: bool
    individual_signals: Dict[str, float]
    recommended_kelly_fraction: float
    market_price: float
    timestamp: datetime


class ConsensusEngine:
    """Aggregates signals from multiple sources to find high-conviction trades."""

    def __init__(self, client: KalshiClient = None, dry_run: bool = False):
        self.client = client or KalshiClient()
        self.dry_run = dry_run

        self.edge_detector = EdgeDetector(self.client)
        self.sentiment_scraper = SentimentScraper()
        self.odds_api = OddsAPI() if _HAS_ODDS else None

        self._odds_cache: Dict = {}
        self._odds_cache_time: Optional[float] = None

        logger.info(f"ConsensusEngine initialized (dry_run={dry_run})")

    @staticmethod
    def _categorize_market(ticker: str, category: str, title: str) -> str:
        """Categorize market by ticker prefix, then DB category, then title."""
        t = (ticker or "").upper()
        if any(t.startswith(p) for p in ("KXHIGH", "KXLOW", "KXRAIN", "KXSNOW", "KXTEMP")):
            return "weather"
        if any(t.startswith(p) for p in ("KXFED", "KXGDP", "KXCPI", "KXPCE", "KXJOBS", "KXUNRATE")):
            return "economics"
        if category in ("weather",):
            return "weather"
        if category in ("economics", "economy"):
            return "economics"
        if category in ("sports",):
            return "sports"
        return "generic"

    def gather_all_signals(self, market: Dict) -> ConsensusReport:
        """Query all available sources for a single market."""
        market_id = market.get('market_id') or market.get('ticker', '')
        ticker = market.get('ticker', '') or market_id
        title = market.get('title', ticker)

        logger.debug(f"Gathering signals for {ticker}...")

        individual_signals = {}
        sources_checked = 0
        sources_with_data = 0

        # 1. EdgeDetector — ensemble probability
        sources_checked += 1
        try:
            edge_signal = self.edge_detector.analyze(market)
            if edge_signal:
                individual_signals['edge_detector'] = edge_signal.ensemble_probability
                sources_with_data += 1
        except Exception as e:
            logger.error(f"  EdgeDetector failed: {e}")

        # 2. OddsAPI — sportsbook implied probability (sports markets only)
        if self.odds_api:
            sources_checked += 1
            try:
                category = self._categorize_market(ticker, market.get('category', ''), title)
                if category == "sports" and self.odds_api.api_key:
                    now = time.time()
                    if self._odds_cache_time is None or now - self._odds_cache_time > 300:
                        try:
                            events = self.odds_api.get_odds("basketball_ncaab")
                            if events:
                                probs = self.odds_api.extract_book_probabilities(events)
                                self._odds_cache = {p["event_name"]: p for p in probs}
                                self._odds_cache_time = now
                        except Exception:
                            pass
                    title_words = set(title.lower().split())
                    for event_name, data in self._odds_cache.items():
                        event_words = set(event_name.lower().split())
                        if len(title_words & event_words) >= 2 and data.get("consensus"):
                            consensus_values = list(data["consensus"].values())
                            if consensus_values:
                                odds_prob = max(0.05, min(0.95, consensus_values[0]))
                                individual_signals['odds_api'] = odds_prob
                                sources_with_data += 1
                            break
            except Exception as e:
                logger.error(f"  OddsAPI failed: {e}")

        # 3. SentimentScraper — sentiment direction
        sources_checked += 1
        try:
            category = self._categorize_market(ticker, market.get('category', ''), title)
            query = self.sentiment_scraper.build_query(title, category)
            signals = self.sentiment_scraper.scan_news(query, days_back=1)
            if signals:
                agg = self.sentiment_scraper.aggregate_sentiment(signals)
                sentiment_prob = max(0.05, min(0.95, 0.5 + agg * 0.25))
                individual_signals['sentiment'] = sentiment_prob
                sources_with_data += 1
        except Exception as e:
            logger.error(f"  Sentiment failed: {e}")

        # Analyze consensus
        if sources_with_data == 0:
            return ConsensusReport(
                market_id=market_id, ticker=ticker,
                sources_checked=sources_checked, sources_with_data=0,
                sources_agreeing_yes=0, sources_agreeing_no=0,
                consensus_direction=None, consensus_strength=0.0,
                consensus_edge=0.0, is_high_conviction=False,
                individual_signals={}, market_price=0.5,
                timestamp=mt_now(),
            )

        sources_agreeing_yes = sum(1 for p in individual_signals.values() if p > 0.5)
        sources_agreeing_no = sum(1 for p in individual_signals.values() if p < 0.5)

        consensus_direction = None
        if sources_agreeing_yes >= MIN_CONSENSUS_SOURCES:
            consensus_direction = "YES"
        elif sources_agreeing_no >= MIN_CONSENSUS_SOURCES:
            consensus_direction = "NO"

        max_agreement = max(sources_agreeing_yes, sources_agreeing_no)
        consensus_strength = max_agreement / sources_with_data if sources_with_data > 0 else 0.0

        consensus_edge = 0.0
        yes_price = market.get("yes_price") or market.get("current_yes_price") or 50
        market_prob = yes_price / 100.0 if yes_price > 1 else yes_price
        if consensus_direction:
            agreeing_probs = [
                p for p in individual_signals.values()
                if (consensus_direction == "YES" and p > 0.5) or
                   (consensus_direction == "NO" and p < 0.5)
            ]
            if agreeing_probs:
                avg_prob = sum(agreeing_probs) / len(agreeing_probs)
                consensus_edge = abs(avg_prob - market_prob)

        is_high_conviction = max_agreement >= MIN_CONSENSUS_SOURCES

        return ConsensusReport(
            market_id=market_id, ticker=ticker,
            sources_checked=sources_checked, sources_with_data=sources_with_data,
            sources_agreeing_yes=sources_agreeing_yes,
            sources_agreeing_no=sources_agreeing_no,
            consensus_direction=consensus_direction,
            consensus_strength=consensus_strength,
            consensus_edge=consensus_edge,
            is_high_conviction=is_high_conviction,
            individual_signals=individual_signals,
            market_price=market_prob,
            timestamp=mt_now(),
        )

    def scan_for_consensus(self, markets: List[Dict]) -> List[ConsensusSignal]:
        """
        Batch scan markets and return HIGH CONVICTION signals.
        Does NOT execute trades — returns signals for the bot adapter.
        """
        logger.info(f"Scanning {len(markets)} markets for consensus...")

        consensus_signals = []

        for market in markets:
            ticker = market.get('ticker', '')
            try:
                report = self.gather_all_signals(market)

                agreeing_count = max(report.sources_agreeing_yes, report.sources_agreeing_no)
                if agreeing_count == 2:
                    min_confidence = 0.85
                elif agreeing_count >= 3:
                    min_confidence = 0.75
                else:
                    min_confidence = 1.0

                if (report.is_high_conviction
                        and report.consensus_edge >= 0.05
                        and report.consensus_strength >= min_confidence):
                    signal = ConsensusSignal(
                        market_id=report.market_id,
                        ticker=report.ticker,
                        direction=report.consensus_direction,
                        consensus_edge=report.consensus_edge,
                        consensus_strength=report.consensus_strength,
                        sources_agreeing=max(report.sources_agreeing_yes, report.sources_agreeing_no),
                        total_sources=report.sources_with_data,
                        is_high_conviction=True,
                        individual_signals=report.individual_signals,
                        recommended_kelly_fraction=CONSENSUS_KELLY_FRACTION,
                        market_price=report.market_price,
                        timestamp=report.timestamp,
                    )
                    consensus_signals.append(signal)
                    logger.info(f"HIGH CONVICTION: {ticker} {signal.direction} edge={signal.consensus_edge:.1%}")

            except Exception as e:
                logger.error(f"Failed to analyze {ticker}: {e}")

        logger.info(f"Found {len(consensus_signals)} high conviction signals")
        return consensus_signals
