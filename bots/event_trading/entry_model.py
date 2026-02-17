"""
ML Entry Model
Gradient-boosted model trained on settlement data to predict P(YES settlement).

Features: market_price, edge_signal, confidence, category, volume,
          orderbook_skew, source_count, source_agreement
"""

import os
import pickle
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
MODEL_PATH = os.path.join(DATA_DIR, "live", "entry_model.pkl")

logger = logging.getLogger("EventEdge.EntryModel")

CATEGORY_MAP = {
    "weather": 0, "economics": 1, "crypto": 2, "politics": 3,
    "sports": 4, "entertainment": 5, "energy": 6, "other": 7,
}

FEATURE_NAMES = [
    "market_price", "abs_edge", "confidence", "ensemble_prob",
    "fred_signal", "weather_signal", "whale_signal",
    "volume_24h", "yes_price", "open_interest", "liquidity_score",
    "category_code", "source_count", "source_agreement",
]


@dataclass
class Prediction:
    """Model prediction result."""
    ticker: str
    ml_probability: float
    market_price: float
    edge: float
    confidence: float
    should_trade: bool

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "ml_probability": round(self.ml_probability, 4),
            "market_price": round(self.market_price, 4),
            "edge": round(self.edge, 4),
            "confidence": round(self.confidence, 4),
            "should_trade": self.should_trade,
        }


def build_training_data() -> Tuple[List[List[float]], List[int], List[str]]:
    """
    Build features and labels from settled markets with edge signals.
    Returns (features, labels, tickers).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT
            e.market_id, e.edge, e.confidence,
            e.ensemble_probability, e.market_probability,
            e.fred_signal, e.weather_signal, e.whale_signal,
            d.category, d.volume_24h, d.current_yes_price,
            d.open_interest, d.liquidity_score, d.settlement_result
        FROM edge_signals e
        JOIN discovered_markets d ON e.market_id = d.ticker
        WHERE d.settlement_result IN ('yes', 'no')
        GROUP BY e.market_id
        HAVING e.id = MAX(e.id)
    """).fetchall()
    conn.close()

    features, labels, tickers = [], [], []

    for row in rows:
        market_price = row["market_probability"] or 0.5
        fred = row["fred_signal"] if row["fred_signal"] is not None else 0.5
        weather = row["weather_signal"] if row["weather_signal"] is not None else 0.5
        # Skip weather markets where NWS returned no useful signal (0.5 = uninformative default)
        # Non-weather markets (FED, GDP, etc.) naturally have weather=0.5 — don't skip those
        is_weather_market = any(row["market_id"].startswith(p) for p in ("KXHIGH", "KXLOW"))
        if is_weather_market and weather == 0.5 and row["weather_signal"] is not None:
            continue
        whale = row["whale_signal"] if row["whale_signal"] is not None else 0.5
        volume = row["volume_24h"] or 0
        yes_price = (row["current_yes_price"] or 50) / 100.0
        open_interest = row["open_interest"] or 0
        liquidity = row["liquidity_score"] or 0
        category_code = CATEGORY_MAP.get(row["category"] or "other", 7)

        source_count = sum(1 for s in [row["fred_signal"], row["weather_signal"], row["whale_signal"]]
                          if s is not None)
        available = [s for s in [fred, weather, whale] if s != 0.5]
        source_agreement = 1.0 - (statistics.stdev(available) * 2 if len(available) > 1 else 0.5)
        source_agreement = max(0, min(1, source_agreement))

        features.append([
            market_price, abs(row["edge"] or 0), row["confidence"] or 0.5,
            row["ensemble_probability"] or 0.5,
            fred, weather, whale, volume, yes_price,
            open_interest, liquidity, category_code,
            source_count, source_agreement,
        ])
        labels.append(1 if row["settlement_result"] == "yes" else 0)
        tickers.append(row["market_id"])

    logger.info(f"Built {len(features)} training samples ({sum(labels)} YES, {len(labels)-sum(labels)} NO)")
    return features, labels, tickers


def train_model() -> Optional[object]:
    """Train gradient boosted model on settlement data."""
    features, labels, tickers = build_training_data()

    if len(features) < 50:
        print(f"\nInsufficient data: {len(features)} settled markets with signals (need 50+).")
        print("Run settlement_collector.py to backfill.\n")
        return None

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import brier_score_loss, log_loss
    except ImportError:
        print("sklearn not installed. Install: pip install scikit-learn")
        return None

    # Try LightGBM, fallback to sklearn
    model_type = "sklearn_gbc"
    try:
        import lightgbm as lgb
        base_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8, verbose=-1,
        )
        model_type = "lightgbm"
    except ImportError:
        base_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            min_samples_leaf=10, subsample=0.8,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    base_model.fit(X_train, y_train)

    # Isotonic calibration for well-calibrated probabilities
    calibrated = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
    calibrated.fit(X_train, y_train)

    y_pred_proba = calibrated.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    accuracy = sum(1 for p, a in zip(y_pred_proba, y_test) if (p > 0.5) == (a == 1)) / len(y_test)

    print(f"\n  Model Performance ({model_type}, test set):")
    print(f"  Brier Score: {brier:.4f} ({'good' if brier < 0.2 else 'fair' if brier < 0.25 else 'poor'})")
    print(f"  Log Loss:    {logloss:.4f}")
    print(f"  Accuracy:    {accuracy:.1%}")
    print(f"  Train/Test:  {len(X_train)}/{len(X_test)}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": calibrated, "type": f"{model_type}_calibrated",
            "features": FEATURE_NAMES, "brier": brier, "n_train": len(X_train),
        }, f)
    logger.info(f"Model saved to {MODEL_PATH} (brier={brier:.4f})")
    return calibrated


def load_model():
    """Load trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Loaded {data['type']} model (brier={data.get('brier', '?')})")
        return data["model"]
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def predict(model, market_data: Dict) -> Optional[Prediction]:
    """
    Predict P(YES settlement) for a market.
    market_data should have edge signal + market fields.
    """
    if model is None:
        return None

    market_price = market_data.get("market_probability", 0.5)
    fred = market_data.get("fred_signal") if market_data.get("fred_signal") is not None else 0.5
    weather = market_data.get("weather_signal") if market_data.get("weather_signal") is not None else 0.5
    whale = market_data.get("whale_signal") if market_data.get("whale_signal") is not None else 0.5
    volume = market_data.get("volume_24h", 0)
    yp = market_data.get("yes_price", 50)
    yes_price = yp / 100.0 if yp > 1 else yp
    open_interest = market_data.get("open_interest", 0)
    liquidity = market_data.get("liquidity_score", 0)
    category = CATEGORY_MAP.get(market_data.get("category", "other"), 7)

    source_count = sum(1 for s in [market_data.get("fred_signal"), market_data.get("weather_signal"),
                                    market_data.get("whale_signal")] if s is not None)
    available = [s for s in [fred, weather, whale] if s != 0.5]
    source_agreement = 1.0 - (statistics.stdev(available) * 2 if len(available) > 1 else 0.5)
    source_agreement = max(0, min(1, source_agreement))

    feature_vec = [[
        market_price, abs(market_data.get("edge", 0)),
        market_data.get("confidence", 0.5), market_data.get("ensemble_probability", 0.5),
        fred, weather, whale, volume, yes_price, open_interest,
        liquidity, category, source_count, source_agreement,
    ]]

    try:
        ml_prob = model.predict_proba(feature_vec)[0][1]
        ml_edge = ml_prob - market_price
        ml_confidence = abs(ml_prob - 0.5) * 2  # Further from 0.5 = more confident
        should_trade = abs(ml_edge) > 0.05 and ml_confidence > 0.3

        return Prediction(
            ticker=market_data.get("ticker", market_data.get("market_id", "")),
            ml_probability=ml_prob, market_price=market_price,
            edge=ml_edge, confidence=max(0.1, ml_confidence),
            should_trade=should_trade,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None


def walk_forward_evaluate():
    """Walk-forward validation: train on first 80%, test on last 20% chronologically."""
    features, labels, tickers = build_training_data()
    if len(features) < 100:
        print(f"\nInsufficient data for walk-forward: {len(features)} samples (need 100+)\n")
        return

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import brier_score_loss
    except ImportError:
        print("sklearn not installed. Install: pip install scikit-learn")
        return

    split_idx = int(len(features) * 0.8)
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_pred)
    accuracy = sum(1 for p, a in zip(y_pred, y_test) if (p > 0.5) == (a == 1)) / len(y_test)

    # Simulate P&L on test set
    pnl = 0.0
    trades = 0
    for pred, actual, feat in zip(y_pred, y_test, X_test):
        market_price = feat[0]
        ml_edge = pred - market_price
        if abs(ml_edge) > 0.05:
            trades += 1
            if ml_edge > 0:
                pnl += ((1.0 - market_price) * 10 / market_price) if actual == 1 else -10
            else:
                pnl += (market_price * 10 / (1.0 - market_price)) if actual == 0 else -10

    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD EVALUATION")
    print(f"{'='*60}")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Brier Score:   {brier:.4f}")
    print(f"  Accuracy:      {accuracy:.1%}")
    print(f"  Simulated P&L: ${pnl:+.2f} ({trades} trades)")
    if trades > 0:
        print(f"  Avg P&L/trade: ${pnl/trades:+.2f}")
    print(f"{'='*60}\n")
