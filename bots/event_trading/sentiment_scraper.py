"""
Sentiment Scraper
Monitors public data sources for real-time signals:
- Reddit (public API): game threads, injury reports
- NewsAPI (free tier): breaking sports news
- RSS feeds: ESPN, injury reports

Returns sentiment signals compatible with EdgeDetector pipeline.

All timestamps in Mountain Time (America/Denver).
"""

import os
import re
import time
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import requests

from dotenv import load_dotenv

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

MT = ZoneInfo("America/Denver")

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# High-impact keywords by domain
IMPACT_KEYWORDS = {
    "negative": ["OUT", "questionable", "injury", "injured", "inactive", "ruled out",
                  "doubtful", "torn", "fracture", "concussion", "suspension", "suspended",
                  "DNP", "did not practice", "limited", "sidelined", "miss", "misses",
                  "hawkish", "miss", "revised down", "contraction", "decline",
                  "cold front", "record low", "freeze warning", "storm", "blizzard"],
    "positive": ["cleared", "active", "playing", "upgraded", "full practice",
                  "return", "returning", "healthy", "available", "starting",
                  "dovish", "beat expectations", "revised up", "expansion", "growth",
                  "heat wave", "record high", "warming", "above normal"],
}

# Default RSS feeds for sports injury/news monitoring
DEFAULT_RSS_FEEDS = [
    "https://www.espn.com/espn/rss/news",
    "https://www.cbssports.com/rss/headlines",
]

logger = logging.getLogger("EventEdge.SentimentScraper")


def _find_xml(parent, *tags):
    """Find first matching XML child element. Avoids Element.__bool__ gotcha."""
    for tag in tags:
        el = parent.find(tag)
        if el is not None:
            return el
    return None


def _findall_xml(parent, *xpaths):
    """Find all matching XML elements using first non-empty xpath result."""
    for xpath in xpaths:
        results = parent.findall(xpath)
        if results:
            return results
    return []


def mt_now() -> datetime:
    return datetime.now(MT)


def mt_str(dt: datetime = None) -> str:
    return (dt or mt_now()).strftime("%Y-%m-%d %H:%M:%S MT")


def _retry_request(method: str, url: str, **kwargs) -> requests.Response:
    """HTTP request with 3-retry exponential backoff."""
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
class SentimentSignal:
    """Sentiment signal compatible with EdgeDetector pipeline."""
    source: str              # "reddit", "newsapi", "rss"
    headline: str
    impact: str              # "positive", "negative", "neutral"
    confidence: float        # 0-1
    relevant_teams: List[str] = field(default_factory=list)
    url: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = mt_str()

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "headline": self.headline,
            "impact": self.impact,
            "confidence": round(self.confidence, 4),
            "relevant_teams": self.relevant_teams,
            "url": self.url,
            "timestamp": self.timestamp,
        }


class SentimentScraper:
    """Multi-source sentiment scraper for trading signals."""

    # Class-level cache shared across instances — {query: (timestamp, signals)}
    _news_cache: Dict[str, tuple] = {}
    _news_cache_ttl = 1800  # 30 minutes
    _daily_request_count = 0
    _daily_request_date = None
    _daily_request_limit = 40  # Conservative: 50 per 12h window, keep headroom
    _rate_limit_backoff_until = 0.0  # Unix timestamp — skip all requests until this time
    _consecutive_429s = 0  # Escalating backoff: 15min → 1h → 6h

    def __init__(self, newsapi_key: str = None):
        self.newsapi_key = newsapi_key or NEWSAPI_KEY
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "EventTradingBot/1.0 (sentiment scraper)"
        })

    @staticmethod
    def build_query(market_title: str, category: str = "other") -> str:
        """
        Build a smart NewsAPI query from market metadata instead of raw title.
        Raw titles return 0 results; domain-specific queries return relevant news.
        """
        title_lower = market_title.lower()

        if category == "weather":
            # Extract city name from title
            cities = ["new york", "chicago", "los angeles", "miami", "denver",
                      "san francisco", "dallas", "houston", "phoenix", "seattle",
                      "boston", "atlanta", "philadelphia", "minneapolis", "detroit",
                      "washington"]
            city = next((c for c in cities if c in title_lower), "")
            if "rain" in title_lower or "snow" in title_lower or "precip" in title_lower:
                return f"{city} weather precipitation forecast".strip()
            return f"{city} weather forecast temperature".strip()

        if category in ("economics", "economy"):
            if any(kw in title_lower for kw in ("fed", "fomc", "rate", "interest")):
                return "federal reserve rate decision FOMC"
            if any(kw in title_lower for kw in ("gdp", "growth", "recession")):
                return "GDP growth forecast economic"
            if any(kw in title_lower for kw in ("cpi", "inflation", "pce")):
                return "CPI inflation report consumer prices"
            if any(kw in title_lower for kw in ("jobs", "unemployment", "payroll", "nonfarm")):
                return "jobs report unemployment nonfarm payrolls"
            if any(kw in title_lower for kw in ("retail", "consumer", "spending")):
                return "retail sales consumer spending"
            return "economic indicators forecast"

        if category == "sports":
            # Try to extract team names (capitalized words that aren't common)
            skip = {"will", "the", "win", "over", "under", "above", "below", "vs",
                    "total", "points", "game", "match", "score", "odds"}
            words = [w for w in market_title.split() if w[0:1].isupper() and w.lower() not in skip]
            teams = " ".join(words[:3]) if words else "sports"
            return f"{teams} game preview injury report"

        # Generic: extract meaningful words
        skip_words = {"will", "the", "be", "above", "below", "over", "under",
                      "or", "and", "by", "on", "in", "at", "to", "of", "a", "an"}
        words = [w for w in market_title.split() if w.lower() not in skip_words][:5]
        return " ".join(words) if words else market_title[:40]

    def _classify_impact(self, text: str) -> tuple:
        """
        Classify text impact as positive/negative/neutral.
        Returns (impact, confidence).
        """
        text_lower = text.lower()
        neg_hits = sum(1 for kw in IMPACT_KEYWORDS["negative"] if kw.lower() in text_lower)
        pos_hits = sum(1 for kw in IMPACT_KEYWORDS["positive"] if kw.lower() in text_lower)

        if neg_hits > pos_hits:
            confidence = min(1.0, 0.4 + neg_hits * 0.15)
            return "negative", confidence
        elif pos_hits > neg_hits:
            confidence = min(1.0, 0.4 + pos_hits * 0.15)
            return "positive", confidence
        else:
            return "neutral", 0.3

    def _extract_teams(self, text: str, known_teams: List[str] = None) -> List[str]:
        """Extract team names mentioned in text."""
        found = []
        if known_teams:
            text_lower = text.lower()
            for team in known_teams:
                if team.lower() in text_lower:
                    found.append(team)
        return found

    def scan_reddit(self, subreddits: List[str], limit: int = 25,
                    teams: List[str] = None) -> List[SentimentSignal]:
        """
        Scan Reddit subreddits for high-impact posts.
        Uses public JSON API (no auth needed, rate-limited).
        """
        signals = []

        for sub in subreddits:
            try:
                url = f"https://www.reddit.com/r/{sub}/new.json"
                resp = self.session.get(url, params={"limit": limit}, timeout=10)
                if resp.status_code == 429:
                    logger.warning(f"Reddit rate limit on r/{sub} — skipping")
                    continue
                resp.raise_for_status()
                data = resp.json()

                for post in data.get("data", {}).get("children", []):
                    pd = post.get("data", {})
                    title = pd.get("title", "")
                    selftext = pd.get("selftext", "")[:500]
                    full_text = f"{title} {selftext}"

                    impact, confidence = self._classify_impact(full_text)
                    if impact == "neutral" and confidence < 0.5:
                        continue  # Skip low-signal posts

                    found_teams = self._extract_teams(full_text, teams)

                    signals.append(SentimentSignal(
                        source="reddit",
                        headline=title[:200],
                        impact=impact,
                        confidence=confidence,
                        relevant_teams=found_teams,
                        url=f"https://reddit.com{pd.get('permalink', '')}",
                    ))

            except Exception as e:
                logger.error(f"Reddit scan failed for r/{sub}: {e}")

        logger.info(f"Reddit: {len(signals)} signals from {len(subreddits)} subreddits")
        return signals

    def scan_news(self, query: str, days_back: int = 1,
                  teams: List[str] = None) -> List[SentimentSignal]:
        """
        Search news via NewsAPI for breaking sports stories.
        Free tier: 100 requests/day, 1-month history.
        Includes response cache (30min TTL), daily rate limiter, and 429 backoff.
        """
        if not self.newsapi_key:
            logger.warning("No NEWSAPI_KEY configured — skipping news scan")
            return []

        # --- Rate limit backoff: fall back to RSS if NewsAPI is cooling down ---
        now_ts = time.time()
        if now_ts < SentimentScraper._rate_limit_backoff_until:
            remaining = int(SentimentScraper._rate_limit_backoff_until - now_ts)
            logger.debug(f"NewsAPI backoff active ({remaining}s) — using RSS fallback")
            return self._rss_fallback(query, teams)

        # --- Daily request budget ---
        today = mt_now().strftime("%Y-%m-%d")
        if SentimentScraper._daily_request_date != today:
            SentimentScraper._daily_request_count = 0
            SentimentScraper._daily_request_date = today
        if SentimentScraper._daily_request_count >= SentimentScraper._daily_request_limit:
            logger.debug(f"NewsAPI daily limit reached ({SentimentScraper._daily_request_count}/{SentimentScraper._daily_request_limit}) — using RSS fallback")
            return self._rss_fallback(query, teams)

        # --- Cache check (30min TTL) ---
        cache_key = query.strip().lower()
        if cache_key in SentimentScraper._news_cache:
            cached_ts, cached_signals = SentimentScraper._news_cache[cache_key]
            if now_ts - cached_ts < SentimentScraper._news_cache_ttl:
                logger.debug(f"NewsAPI cache hit for '{query}' ({len(cached_signals)} signals)")
                return cached_signals

        signals = []
        from_date = (mt_now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        try:
            SentimentScraper._daily_request_count += 1
            resp = requests.get("https://newsapi.org/v2/everything", params={
                "q": query,
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 20,
                "apiKey": self.newsapi_key,
            }, timeout=15)

            if resp.status_code == 429:
                # Escalating backoff: 15min → 1h → 6h
                SentimentScraper._consecutive_429s += 1
                if SentimentScraper._consecutive_429s >= 3:
                    backoff_secs = 21600  # 6 hours — API key likely exhausted for this 12h window
                elif SentimentScraper._consecutive_429s >= 2:
                    backoff_secs = 3600   # 1 hour
                else:
                    backoff_secs = 900    # 15 minutes
                SentimentScraper._rate_limit_backoff_until = now_ts + backoff_secs
                logger.warning(f"NewsAPI 429 (#{SentimentScraper._consecutive_429s}) — backing off {backoff_secs//60} minutes")
                return self._rss_fallback(query, teams)

            resp.raise_for_status()
            data = resp.json()

            for article in data.get("articles", []):
                title = article.get("title", "")
                desc = article.get("description", "") or ""
                full_text = f"{title} {desc}"

                impact, confidence = self._classify_impact(full_text)
                found_teams = self._extract_teams(full_text, teams)

                signals.append(SentimentSignal(
                    source="newsapi",
                    headline=title[:200],
                    impact=impact,
                    confidence=confidence,
                    relevant_teams=found_teams,
                    url=article.get("url", ""),
                ))

            # Cache the result and reset 429 counter
            SentimentScraper._news_cache[cache_key] = (now_ts, signals)
            SentimentScraper._consecutive_429s = 0

        except Exception as e:
            if "429" in str(e):
                SentimentScraper._consecutive_429s += 1
                backoff_secs = min(21600, 900 * (2 ** (SentimentScraper._consecutive_429s - 1)))
                SentimentScraper._rate_limit_backoff_until = now_ts + backoff_secs
                logger.warning(f"NewsAPI 429 (#{SentimentScraper._consecutive_429s}) — backing off {backoff_secs//60} minutes")
                return self._rss_fallback(query, teams)
            else:
                logger.error(f"NewsAPI scan failed for '{query}': {e}")

        logger.info(f"NewsAPI: {len(signals)} signals for '{query}' (requests today: {SentimentScraper._daily_request_count}/{SentimentScraper._daily_request_limit})")
        return signals

    def _rss_fallback(self, query: str, teams: List[str] = None) -> List[SentimentSignal]:
        """
        RSS fallback when NewsAPI is rate-limited.
        Scans ESPN/CBS RSS feeds, classifies all items (not just non-neutral),
        and filters by query keywords.
        Cached for 30 min like NewsAPI results.
        """
        cache_key = f"rss:{query.strip().lower()}"
        now_ts = time.time()
        if cache_key in SentimentScraper._news_cache:
            cached_ts, cached_signals = SentimentScraper._news_cache[cache_key]
            if now_ts - cached_ts < SentimentScraper._news_cache_ttl:
                return cached_signals

        # Fetch and classify ALL RSS items (not just non-neutral)
        all_signals = self._scan_rss_unfiltered(teams=teams)
        if not all_signals:
            SentimentScraper._news_cache[cache_key] = (now_ts, [])
            return []

        # Filter RSS results by query keywords (at least 1 word match)
        query_words = {w for w in query.lower().split() if len(w) > 2}
        filtered = [s for s in all_signals
                    if query_words & set(s.headline.lower().split())]

        SentimentScraper._news_cache[cache_key] = (now_ts, filtered)
        if filtered:
            logger.info(f"RSS fallback: {len(filtered)} signals for '{query}'")
        return filtered

    def _scan_rss_unfiltered(self, feeds: List[str] = None,
                              teams: List[str] = None) -> List[SentimentSignal]:
        """Scan RSS feeds and return ALL items classified (including neutral)."""
        feeds = feeds or DEFAULT_RSS_FEEDS
        signals = []

        for feed_url in feeds:
            try:
                resp = self.session.get(feed_url, timeout=10)
                resp.raise_for_status()

                root = ET.fromstring(resp.content)
                items = _findall_xml(root, ".//item",
                                     ".//{http://www.w3.org/2005/Atom}entry")

                for item in items[:20]:
                    title_el = _find_xml(item, "title",
                                         "{http://www.w3.org/2005/Atom}title")
                    link_el = _find_xml(item, "link",
                                        "{http://www.w3.org/2005/Atom}link")
                    desc_el = _find_xml(item, "description",
                                        "{http://www.w3.org/2005/Atom}summary")

                    title = title_el.text if title_el is not None and title_el.text else ""
                    link = ""
                    if link_el is not None:
                        link = link_el.text or link_el.get("href", "")
                    desc = desc_el.text if desc_el is not None and desc_el.text else ""
                    desc = re.sub(r'<[^>]+>', '', desc)[:300]
                    full_text = f"{title} {desc}"

                    impact, confidence = self._classify_impact(full_text)
                    found_teams = self._extract_teams(full_text, teams)

                    signals.append(SentimentSignal(
                        source="rss",
                        headline=title[:200],
                        impact=impact,
                        confidence=confidence,
                        relevant_teams=found_teams,
                        url=link,
                    ))

            except Exception as e:
                logger.error(f"RSS fallback scan failed for {feed_url}: {e}")

        return signals

    def scan_rss(self, feeds: List[str] = None,
                 teams: List[str] = None) -> List[SentimentSignal]:
        """
        Scan RSS feeds for sports news.
        Parses standard RSS/Atom XML.
        """
        feeds = feeds or DEFAULT_RSS_FEEDS
        signals = []

        for feed_url in feeds:
            try:
                resp = self.session.get(feed_url, timeout=10)
                resp.raise_for_status()

                root = ET.fromstring(resp.content)

                # Handle both RSS and Atom formats
                items = _findall_xml(root, ".//item",
                                     ".//{http://www.w3.org/2005/Atom}entry")

                for item in items[:15]:
                    title_el = _find_xml(item, "title",
                                         "{http://www.w3.org/2005/Atom}title")
                    link_el = _find_xml(item, "link",
                                        "{http://www.w3.org/2005/Atom}link")
                    desc_el = _find_xml(item, "description",
                                        "{http://www.w3.org/2005/Atom}summary")

                    title = title_el.text if title_el is not None and title_el.text else ""
                    link = ""
                    if link_el is not None:
                        link = link_el.text or link_el.get("href", "")
                    desc = desc_el.text if desc_el is not None and desc_el.text else ""

                    # Strip HTML tags from description
                    desc = re.sub(r'<[^>]+>', '', desc)[:300]
                    full_text = f"{title} {desc}"

                    impact, confidence = self._classify_impact(full_text)
                    found_teams = self._extract_teams(full_text, teams)

                    # Only include if we found relevant teams or high-impact keywords
                    if impact != "neutral" or found_teams:
                        signals.append(SentimentSignal(
                            source="rss",
                            headline=title[:200],
                            impact=impact,
                            confidence=confidence,
                            relevant_teams=found_teams,
                            url=link,
                        ))

            except Exception as e:
                logger.error(f"RSS scan failed for {feed_url}: {e}")

        logger.info(f"RSS: {len(signals)} signals from {len(feeds)} feeds")
        return signals

    def get_injury_alerts(self, teams: List[str],
                          subreddits: List[str] = None) -> List[SentimentSignal]:
        """
        Specifically look for injury/status alerts for given teams.
        Combines all sources, filters to injury-related signals only.
        """
        all_signals = []

        # Reddit
        subs = subreddits or ["nba", "nfl", "CollegeBasketball", "sportsbook"]
        reddit_signals = self.scan_reddit(subs, limit=50, teams=teams)
        all_signals.extend(reddit_signals)

        # News
        for team in teams:
            news_signals = self.scan_news(f"{team} injury OR inactive OR OUT", teams=teams)
            all_signals.extend(news_signals)

        # RSS
        rss_signals = self.scan_rss(teams=teams)
        all_signals.extend(rss_signals)

        # Filter to high-impact injury signals only
        injury_keywords = {"out", "injury", "injured", "questionable", "inactive",
                           "doubtful", "concussion", "torn", "fracture", "dnp",
                           "cleared", "active", "return"}
        injury_signals = []
        for sig in all_signals:
            headline_lower = sig.headline.lower()
            if any(kw in headline_lower for kw in injury_keywords) and sig.relevant_teams:
                injury_signals.append(sig)

        logger.info(f"Injury alerts: {len(injury_signals)} for teams {teams}")
        return injury_signals

    def aggregate_sentiment(self, signals: List[SentimentSignal]) -> float:
        """
        Aggregate sentiment signals into a single score.
        Returns float from -1 (very negative) to +1 (very positive).
        """
        if not signals:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0

        for sig in signals:
            if sig.impact == "positive":
                score = sig.confidence
            elif sig.impact == "negative":
                score = -sig.confidence
            else:
                score = 0.0

            # Weight by source reliability
            source_weight = {"newsapi": 1.0, "rss": 0.8, "reddit": 0.5}.get(sig.source, 0.5)
            w = sig.confidence * source_weight

            weighted_sum += score * w
            weight_total += w

        if weight_total == 0:
            return 0.0

        return max(-1.0, min(1.0, weighted_sum / weight_total))
