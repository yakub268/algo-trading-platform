"""
Settlement Collector
Queries Kalshi for settlement results on discovered markets.
Backfills historical data and runs as a recurring background job.
"""

import os
import time
import sqlite3
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from bots.kalshi_client import KalshiClient
from dotenv import load_dotenv

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DB_PATH = os.path.join(DATA_DIR, "live", "event_trading.db")

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

MT = ZoneInfo("America/Denver")

POLL_INTERVAL = 1800  # 30 minutes
API_DELAY = 0.15      # Seconds between API calls

logger = logging.getLogger("EventEdge.SettlementCollector")


def ensure_schema():
    """Add settlement columns to discovered_markets if missing."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(discovered_markets)").fetchall()}
    with conn:
        if "settlement_result" not in columns:
            conn.execute("ALTER TABLE discovered_markets ADD COLUMN settlement_result TEXT")
            logger.info("Added settlement_result column")
        if "settlement_time" not in columns:
            conn.execute("ALTER TABLE discovered_markets ADD COLUMN settlement_time TIMESTAMP")
            logger.info("Added settlement_time column")
    conn.close()


def collect_settlements(client: KalshiClient, limit: int = 0) -> dict:
    """
    Query unsettled markets and check for settlement results.
    Returns stats dict with counts.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    query = """
        SELECT DISTINCT ticker FROM discovered_markets
        WHERE settlement_result IS NULL AND ticker IS NOT NULL AND ticker != ''
    """
    if limit > 0:
        query += " LIMIT ?"
        rows = conn.execute(query, (limit,)).fetchall()
    else:
        rows = conn.execute(query).fetchall()
    tickers = [r["ticker"] for r in rows]
    conn.close()

    if not tickers:
        logger.info("No unsettled markets to check")
        return {"checked": 0, "settled": 0, "errors": 0}

    logger.info(f"Checking {len(tickers)} unsettled markets...")
    stats = {"checked": 0, "settled": 0, "errors": 0}

    for i, ticker in enumerate(tickers):
        try:
            market = client.get_market(ticker)
            status = (market.get("status") or "").lower()

            if status in ("settled", "closed", "finalized"):
                result = (market.get("result") or "").lower()
                if result in ("yes", "no"):
                    _update_settlement(ticker, result)
                    stats["settled"] += 1
                else:
                    _update_settlement(ticker, "voided")
                    stats["settled"] += 1

            stats["checked"] += 1
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{len(tickers)} checked, {stats['settled']} settled")
            time.sleep(API_DELAY)

        except Exception as e:
            stats["errors"] += 1
            if "404" in str(e) or "not found" in str(e).lower():
                _update_settlement(ticker, "voided")
                stats["settled"] += 1
            else:
                logger.debug(f"Error checking {ticker}: {e}")
            time.sleep(API_DELAY)

    logger.info(f"Settlement collection: {stats['checked']} checked, {stats['settled']} settled, {stats['errors']} errors")
    return stats


def _update_settlement(ticker: str, result: str):
    """Update all rows for a ticker with settlement result."""
    now = datetime.now(MT).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH, timeout=30)
    with conn:
        conn.execute(
            "UPDATE discovered_markets SET settlement_result=?, settlement_time=? WHERE ticker=?",
            (result, now, ticker),
        )
    conn.close()
    logger.debug(f"Settlement: {ticker} -> {result}")


def get_stats() -> dict:
    """Get settlement collection statistics."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    total = conn.execute("SELECT COUNT(DISTINCT ticker) FROM discovered_markets").fetchone()[0]
    settled = conn.execute(
        "SELECT COUNT(DISTINCT ticker) FROM discovered_markets WHERE settlement_result IS NOT NULL"
    ).fetchone()[0]
    yes_count = conn.execute(
        "SELECT COUNT(DISTINCT ticker) FROM discovered_markets WHERE settlement_result='yes'"
    ).fetchone()[0]
    no_count = conn.execute(
        "SELECT COUNT(DISTINCT ticker) FROM discovered_markets WHERE settlement_result='no'"
    ).fetchone()[0]
    voided = conn.execute(
        "SELECT COUNT(DISTINCT ticker) FROM discovered_markets WHERE settlement_result='voided'"
    ).fetchone()[0]
    conn.close()
    return {
        "total_tickers": total, "settled": settled, "unsettled": total - settled,
        "yes": yes_count, "no": no_count, "voided": voided,
        "settlement_rate": settled / total if total > 0 else 0,
    }
