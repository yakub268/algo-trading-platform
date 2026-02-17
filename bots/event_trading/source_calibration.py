"""
Source Calibration Engine
Computes per-source accuracy metrics (Brier score, calibration curve)
using settlement data from discovered_markets.

Replaces hardcoded source weights with empirical weights based on
actual predictive accuracy.
"""

import os
import sqlite3
import logging
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from zoneinfo import ZoneInfo

MT = ZoneInfo("America/Denver")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DB_PATH = os.path.join(DATA_DIR, "live", "event_trading.db")

logger = logging.getLogger("EventEdge.SourceCalibration")


def ensure_schema():
    """Create source_accuracy table if needed."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS source_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                market_category TEXT NOT NULL,
                brier_score REAL,
                hit_rate REAL,
                n_predictions INTEGER,
                calibration_error REAL,
                last_updated TIMESTAMP,
                UNIQUE(source, market_category)
            )
        """)
    conn.close()


@dataclass
class SourceMetrics:
    source: str
    category: str
    brier_score: float
    hit_rate: float
    n_predictions: int
    calibration_error: float


def compute_source_accuracy() -> List[SourceMetrics]:
    """
    For each settled market with edge signals, compute per-source accuracy.
    Joins edge_signals with discovered_markets settlement data.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            e.market_id,
            e.fred_signal,
            e.weather_signal,
            e.whale_signal,
            e.ensemble_probability,
            e.direction,
            d.settlement_result,
            d.category
        FROM edge_signals e
        JOIN discovered_markets d ON e.market_id = d.ticker
        WHERE d.settlement_result IN ('yes', 'no')
        GROUP BY e.market_id
        HAVING e.id = MAX(e.id)
    """).fetchall()
    conn.close()

    if not rows:
        logger.info("No settled markets with edge signals found")
        return []

    logger.info(f"Computing accuracy on {len(rows)} settled markets with signals")

    # Collect per-source predictions: source -> category -> [(pred_prob, actual)]
    source_preds: Dict[str, Dict[str, List[Tuple[float, int]]]] = {}

    for row in rows:
        actual = 1 if row["settlement_result"] == "yes" else 0
        category = row["category"] or "other"

        for source, val in [("fred", row["fred_signal"]),
                            ("weather", row["weather_signal"]),
                            ("whale", row["whale_signal"]),
                            ("ensemble", row["ensemble_probability"])]:
            if val is not None:
                source_preds.setdefault(source, {}).setdefault(category, []).append((val, actual))
                source_preds[source].setdefault("all", []).append((val, actual))

    # Compute metrics per source per category
    results = []
    for source, categories in source_preds.items():
        for category, preds in categories.items():
            if len(preds) < 5:
                continue

            brier_sum = 0.0
            correct = 0
            cal_errors = []

            for pred_prob, actual in preds:
                brier_sum += (pred_prob - actual) ** 2
                if (pred_prob > 0.5) == (actual == 1):
                    correct += 1
                cal_errors.append(abs(pred_prob - actual))

            n = len(preds)
            results.append(SourceMetrics(
                source=source, category=category,
                brier_score=brier_sum / n, hit_rate=correct / n,
                n_predictions=n, calibration_error=statistics.mean(cal_errors),
            ))

    return results


def save_accuracy(metrics: List[SourceMetrics]):
    """Save source accuracy metrics to DB."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    now = datetime.now(MT).strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        for m in metrics:
            conn.execute("""
                INSERT OR REPLACE INTO source_accuracy
                (source, market_category, brier_score, hit_rate, n_predictions, calibration_error, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (m.source, m.category, m.brier_score, m.hit_rate, m.n_predictions, m.calibration_error, now))
    conn.close()
    logger.info(f"Saved {len(metrics)} source accuracy records")


def load_source_accuracy(category: str = "all") -> Dict[str, Dict]:
    """
    Load source accuracy from DB. Returns dict of source -> {brier_score, n}.
    Used by EdgeDetector for adaptive weights.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT source, brier_score, n_predictions FROM source_accuracy WHERE market_category=?",
        (category,)
    ).fetchall()
    conn.close()
    return {r["source"]: {"brier_score": r["brier_score"], "n": r["n_predictions"]} for r in rows}


def get_adaptive_weights(category: str, min_samples: int = 20) -> Optional[Dict[str, float]]:
    """
    Compute adaptive weights from source accuracy data.
    Returns None if insufficient data (caller should use fallback weights).
    Blends default weights with calibrated weights between 20-100 samples.
    """
    accuracies = load_source_accuracy(category)
    if not accuracies:
        accuracies = load_source_accuracy("all")
    if not accuracies:
        return None

    # Filter to sources with enough data (exclude ensemble — it's not a source)
    valid = {src: data for src, data in accuracies.items()
             if data["n"] >= min_samples and src != "ensemble"}
    if len(valid) < 2:
        return None

    # Weight = (1 - brier_score) / sum(1 - brier_scores)
    raw_weights = {src: max(0.01, 1.0 - data["brier_score"]) for src, data in valid.items()}
    total = sum(raw_weights.values())
    calibrated_weights = {src: w / total for src, w in raw_weights.items()}

    # Blend with default weights based on sample count (20-100 range)
    # Default weights for blending (weather/fred/whale equal)
    default_weights = {"weather": 0.33, "fred": 0.33, "whale": 0.34}

    # Find minimum sample count across all sources
    min_n = min(data["n"] for data in valid.values())

    if min_n >= 100:
        # Full calibration
        return calibrated_weights
    else:
        # Blend: 0.0 at 20 samples, 1.0 at 100+ samples
        blend_factor = min(1.0, (min_n - 20) / 80)
        blended_weights = {}
        for src in calibrated_weights:
            data_weight = calibrated_weights[src]
            default_weight = default_weights.get(src, 1.0 / len(calibrated_weights))
            blended_weights[src] = blend_factor * data_weight + (1 - blend_factor) * default_weight

        # Renormalize
        total_blended = sum(blended_weights.values())
        return {src: w / total_blended for src, w in blended_weights.items()}
