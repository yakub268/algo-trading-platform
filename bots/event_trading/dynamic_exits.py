"""
Dynamic Exit Sizing
Game-state-aware exit strategy replacing fixed 25% tranches.
Considers score, quarter, clock, win probability estimate, position size.

Rules:
- High confidence win (>80% implied): hold larger portion
- Uncertain game (45-55% implied): exit faster in smaller tranches
- Q4 close game: rapid exit (volatility too high)
- Blowout: hold for settlement (payout is certain)

All timestamps in Mountain Time (America/Denver).
"""

import os
import math
import logging
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
from zoneinfo import ZoneInfo

# ML imports (optional)
try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import brier_score_loss, log_loss
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = None
    LogisticRegression = None
    GradientBoostingClassifier = None
    CalibratedClassifierCV = None
    joblib = None

from dotenv import load_dotenv

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

MT = ZoneInfo("America/Denver")

logger = logging.getLogger("EventEdge.DynamicExits")


def mt_now() -> datetime:
    return datetime.now(MT)


def mt_str(dt: datetime = None) -> str:
    return (dt or mt_now()).strftime("%Y-%m-%d %H:%M:%S MT")


@dataclass
class ExitRecommendation:
    """Exit recommendation from the dynamic exit manager."""
    pct_to_sell: float      # 0-1 (fraction of position to sell)
    target_price: int       # cents
    urgency: str            # "immediate", "limit", "hold"
    reason: str
    win_probability: float = 0.0
    hold_value: float = 0.0  # expected value of holding vs selling

    def to_dict(self) -> Dict:
        return {
            "pct_to_sell": round(self.pct_to_sell, 4),
            "target_price": self.target_price,
            "urgency": self.urgency,
            "reason": self.reason,
            "win_probability": round(self.win_probability, 4),
            "hold_value": round(self.hold_value, 4),
        }


class WinProbabilityModel:
    """
    ML-based win probability estimator.
    Uses LogisticRegression or GradientBoostingClassifier with feature engineering.

    Features:
    - score_diff: Point differential
    - time_remaining_pct: Fraction of game remaining
    - quarter: Quarter number (1-4, 5=OT)
    - possession: Whether our team has possession (0/1)
    - timeouts_remaining_diff: Timeout differential
    - is_home: Home field advantage (0/1)
    - momentum: Scoring run (points in last 5 min)
    - score_diff_squared: Nonlinear effect
    - log_time_remaining: Log-transformed time
    """

    def __init__(self, model_type: str = "logistic"):
        """
        Args:
            model_type: "logistic" or "gradient_boost"
        """
        if not ML_AVAILABLE:
            raise ImportError("sklearn not available — install with: pip install scikit-learn")

        self.model_type = model_type
        self.model = None
        self.calibrated_model = None
        self.feature_names = [
            "score_diff",
            "time_remaining_pct",
            "quarter",
            "possession",
            "timeouts_remaining_diff",
            "is_home",
            "momentum",
            "score_diff_squared",
            "log_time_remaining",
        ]

    def _extract_features(self, game_state: Dict) -> np.ndarray:
        """Extract feature vector from game state."""
        score_diff = game_state.get("score_diff", 0)
        time_remaining_pct = game_state.get("time_remaining_pct", 0.5)
        quarter = game_state.get("quarter", 1)
        possession = float(game_state.get("possession", 0))
        timeouts_diff = game_state.get("timeouts_remaining_diff", 0)
        is_home = float(game_state.get("is_home", False))
        momentum = game_state.get("momentum", 0)

        # Nonlinear features
        score_diff_sq = score_diff ** 2
        log_time = math.log(max(time_remaining_pct, 0.001))

        return np.array([
            score_diff,
            time_remaining_pct,
            quarter,
            possession,
            timeouts_diff,
            is_home,
            momentum,
            score_diff_sq,
            log_time,
        ])

    def generate_training_data(self, n_samples: int = 10000) -> tuple:
        """
        Generate synthetic training data based on known win probability curves.

        Known relationships:
        - At halftime, team leading by 10 wins ~75%
        - With 2 min left, team leading by 5+ wins ~95%
        - Home field advantage ~3-4%
        - Comeback probability decreases exponentially with time

        Returns:
            (X, y) where X is feature matrix, y is binary outcomes
        """
        np.random.seed(42)

        X = []
        y = []

        for _ in range(n_samples):
            # Random game state
            quarter = np.random.choice([1, 2, 3, 4, 5], p=[0.25, 0.25, 0.25, 0.23, 0.02])
            time_remaining_pct = np.random.uniform(0, 1) if quarter <= 4 else np.random.uniform(0, 0.05)
            score_diff = np.random.normal(0, 10)
            is_home = np.random.choice([0, 1])
            possession = np.random.choice([0, 1])
            timeouts_diff = np.random.randint(-3, 4)
            momentum = np.random.normal(0, 5)

            # Calculate "true" win probability using known curves
            # Base win prob from score differential + time
            time_factor = 0.15 + (1 - time_remaining_pct) * 0.85  # 0.15 to 1.0
            wp = 1.0 / (1.0 + math.exp(-score_diff * time_factor * 0.12))

            # Home field adjustment (+3-4%)
            if is_home:
                wp += 0.035

            # Momentum adjustment
            wp += momentum * 0.005

            # Possession adjustment (small, ~1-2%)
            if possession:
                wp += 0.015

            # Timeout adjustment
            wp += timeouts_diff * 0.01

            # Clamp to [0.01, 0.99]
            wp = max(0.01, min(0.99, wp))

            # Generate binary outcome from probability
            outcome = 1 if np.random.random() < wp else 0

            # Build feature vector
            game_state = {
                "score_diff": score_diff,
                "time_remaining_pct": time_remaining_pct,
                "quarter": quarter,
                "possession": possession,
                "timeouts_remaining_diff": timeouts_diff,
                "is_home": bool(is_home),
                "momentum": momentum,
            }

            X.append(self._extract_features(game_state))
            y.append(outcome)

        return np.array(X), np.array(y)

    def train(self, game_data: list = None, n_synthetic: int = 10000):
        """
        Train the model on historical or synthetic data.

        Args:
            game_data: List of dicts with game_state and "outcome" (0/1)
            n_synthetic: Number of synthetic samples to generate if game_data is None
        """
        if game_data:
            X = np.array([self._extract_features(d) for d in game_data])
            y = np.array([d["outcome"] for d in game_data])
        else:
            logger.info(f"Generating {n_synthetic} synthetic training samples...")
            X, y = self.generate_training_data(n_synthetic)

        logger.info(f"Training {self.model_type} model on {len(y)} samples...")

        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train base model
        if self.model_type == "gradient_boost":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
            )
        else:
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
            )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, y_pred)
        logloss = log_loss(y_test, y_pred)

        logger.info(f"  Test Brier Score: {brier:.4f}")
        logger.info(f"  Test Log Loss: {logloss:.4f}")

        return {"brier_score": brier, "log_loss": logloss}

    def calibrate(self, X=None, y=None):
        """
        Apply Platt scaling for calibrated probabilities.

        Args:
            X, y: Calibration data (if None, uses synthetic data)
        """
        if self.model is None:
            raise ValueError("Must train base model before calibration")

        if X is None or y is None:
            logger.info("Generating calibration data...")
            X, y = self.generate_training_data(n_samples=5000)

        logger.info("Calibrating model with Platt scaling...")
        self.calibrated_model = CalibratedClassifierCV(self.model, method="sigmoid", cv=3)
        self.calibrated_model.fit(X, y)
        logger.info("  Calibration complete")

    def predict(self, game_state: Dict) -> float:
        """
        Predict win probability for a game state.

        Args:
            game_state: Dict with game features

        Returns:
            Win probability (0-1)
        """
        if self.calibrated_model is None and self.model is None:
            raise ValueError("Model not trained")

        X = self._extract_features(game_state).reshape(1, -1)

        if self.calibrated_model:
            wp = self.calibrated_model.predict_proba(X)[0, 1]
        else:
            wp = self.model.predict_proba(X)[0, 1]

        return max(0.01, min(0.99, wp))

    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        joblib.dump({
            "model_type": self.model_type,
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "feature_names": self.feature_names,
        }, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "WinProbabilityModel":
        """Load model from disk."""
        if not ML_AVAILABLE:
            raise ImportError("sklearn not available")

        data = joblib.load(path)
        instance = cls(model_type=data["model_type"])
        instance.model = data["model"]
        instance.calibrated_model = data.get("calibrated_model")
        instance.feature_names = data["feature_names"]
        logger.info(f"Model loaded from {path}")
        return instance


class DynamicExitManager:
    """
    Game-state-aware exit sizing.
    Replaces fixed tranche exits with dynamic recommendations.
    """

    def __init__(self, position_size: int, entry_price: int, side: str):
        """
        Args:
            position_size: Number of contracts held
            entry_price: Entry price in cents
            side: "yes" or "no"
        """
        self.position_size = position_size
        self.entry_price = entry_price
        self.side = side
        self.remaining = position_size
        self.total_sold = 0
        self.realized_pnl = 0.0

        # Try to load ML model
        self.ml_model = None
        if ML_AVAILABLE:
            model_path = os.path.join(DATA_DIR, "live", "win_prob_model.pkl")
            if os.path.exists(model_path):
                try:
                    self.ml_model = WinProbabilityModel.load(model_path)
                    logger.info("Loaded ML win probability model")
                except Exception as e:
                    logger.warning(f"Failed to load ML model: {e}")
        else:
            logger.info("sklearn not available — using sigmoid fallback")

    def estimate_win_probability(self, game_state: Dict) -> float:
        """
        Estimate win probability from game state.

        Tries ML model first, falls back to sigmoid if unavailable.

        Args:
            game_state: dict with keys:
                - home_score, away_score (or team-specific)
                - quarter (1-4, 5=OT)
                - clock (str "MM:SS")
                - our_side_score, their_side_score (relative to our position)
        """
        our_score = game_state.get("our_side_score", game_state.get("home_score", 0))
        their_score = game_state.get("their_side_score", game_state.get("away_score", 0))
        quarter = game_state.get("quarter", 1)
        clock = game_state.get("clock", "15:00")

        score_diff = our_score - their_score

        # Convert clock to seconds remaining in game
        try:
            parts = clock.split(":")
            quarter_secs = int(parts[0]) * 60 + int(parts[1])
        except (ValueError, IndexError):
            quarter_secs = 900  # 15 min default

        # Total seconds remaining in regulation
        if quarter <= 4:
            total_secs = quarter_secs + (4 - quarter) * 900
        else:
            total_secs = quarter_secs  # OT

        max_game_secs = 3600  # 60 minutes
        time_remaining_pct = total_secs / max_game_secs if max_game_secs > 0 else 0

        # Try ML model first
        if self.ml_model:
            try:
                ml_state = {
                    "score_diff": score_diff,
                    "time_remaining_pct": time_remaining_pct,
                    "quarter": quarter,
                    "possession": game_state.get("possession", 0),
                    "timeouts_remaining_diff": game_state.get("timeouts_remaining_diff", 0),
                    "is_home": game_state.get("home_field", False),
                    "momentum": game_state.get("momentum", 0),
                }
                return self.ml_model.predict(ml_state)
            except Exception as e:
                logger.warning(f"ML model prediction failed: {e}, falling back to sigmoid")

        # Fallback: sigmoid model
        # Scaling factor: higher as time runs out
        if time_remaining_pct > 0.5:
            time_scale = 0.15  # Early game: score diff less impactful
        elif time_remaining_pct > 0.25:
            time_scale = 0.25  # Mid game
        elif time_remaining_pct > 0.1:
            time_scale = 0.40  # Late game
        elif time_remaining_pct > 0.02:
            time_scale = 0.60  # Final minutes
        else:
            time_scale = 1.00  # Last 30 seconds

        # Base advantage (home field ≈ 3% wp)
        base = 0.03 if game_state.get("home_field", False) else 0.0

        # Win probability via sigmoid
        z = score_diff * time_scale + base
        wp = 1.0 / (1.0 + math.exp(-z))

        return max(0.01, min(0.99, wp))

    def recommend_exit(self, game_state: Dict, current_price: int) -> ExitRecommendation:
        """
        Recommend exit action based on game state and current price.

        Args:
            game_state: dict with score, quarter, clock
            current_price: current market price in cents (for our side)

        Returns:
            ExitRecommendation
        """
        wp = self.estimate_win_probability(game_state)
        quarter = game_state.get("quarter", 1)
        clock = game_state.get("clock", "15:00")

        # Calculate unrealized P&L
        price_diff = current_price - self.entry_price
        unrealized_pnl_pct = price_diff / self.entry_price if self.entry_price > 0 else 0

        # Expected value of holding: wp * (100 - entry) - (1-wp) * entry
        if self.side == "yes":
            ev_hold = wp * (100 - self.entry_price) - (1 - wp) * self.entry_price
        else:
            ev_hold = (1 - wp) * (100 - self.entry_price) - wp * self.entry_price
        ev_hold_per_contract = ev_hold / 100.0

        # Expected value of selling now
        ev_sell = price_diff  # cents per contract

        # Decision logic
        # Case 1: Blowout — hold for settlement
        if wp > 0.95:
            return ExitRecommendation(
                pct_to_sell=0.0,
                target_price=current_price,
                urgency="hold",
                reason=f"Blowout: {wp:.0%} win probability — hold for settlement",
                win_probability=wp,
                hold_value=ev_hold_per_contract,
            )

        # Case 2: High confidence win — hold majority
        if wp > 0.80:
            # Sell 10-20% to lock in some profit
            pct = 0.15
            return ExitRecommendation(
                pct_to_sell=pct,
                target_price=max(current_price, self.entry_price + 5),
                urgency="limit",
                reason=f"High confidence ({wp:.0%}) — sell {pct:.0%} to lock profit, hold rest",
                win_probability=wp,
                hold_value=ev_hold_per_contract,
            )

        # Case 3: Profitable but uncertain — take profit in tranches
        if 0.55 < wp <= 0.80 and current_price > self.entry_price:
            pct = 0.25 + (0.80 - wp) * 0.5  # 25-37% as uncertainty rises
            return ExitRecommendation(
                pct_to_sell=min(pct, 0.50),
                target_price=current_price,
                urgency="limit",
                reason=f"Profitable but uncertain ({wp:.0%}) — take {min(pct, 0.50):.0%} profit",
                win_probability=wp,
                hold_value=ev_hold_per_contract,
            )

        # Case 4: Coin flip — exit quickly
        if 0.45 <= wp <= 0.55:
            pct = 0.50  # Exit half immediately
            return ExitRecommendation(
                pct_to_sell=pct,
                target_price=current_price,
                urgency="immediate",
                reason=f"Coin flip ({wp:.0%}) — exit 50% to reduce risk",
                win_probability=wp,
                hold_value=ev_hold_per_contract,
            )

        # Case 5: Q4 close game — rapid exit
        if quarter == 4 and abs(wp - 0.5) < 0.15:
            pct = 0.75
            return ExitRecommendation(
                pct_to_sell=pct,
                target_price=current_price - 2,  # Aggressive: accept 2c below market
                urgency="immediate",
                reason=f"Q4 close game ({wp:.0%}) — exit 75% to avoid volatility",
                win_probability=wp,
                hold_value=ev_hold_per_contract,
            )

        # Case 6: Losing position — cut losses
        if wp < 0.30 and current_price < self.entry_price:
            pct = 0.80
            return ExitRecommendation(
                pct_to_sell=pct,
                target_price=current_price,
                urgency="immediate",
                reason=f"Losing ({wp:.0%}) — cut 80% of losses",
                win_probability=wp,
                hold_value=ev_hold_per_contract,
            )

        # Case 7: Our side losing badly — exit everything
        if wp < 0.15:
            return ExitRecommendation(
                pct_to_sell=1.0,
                target_price=max(1, current_price - 3),
                urgency="immediate",
                reason=f"Critical ({wp:.0%}) — full exit",
                win_probability=wp,
                hold_value=ev_hold_per_contract,
            )

        # Default: small tranche exit
        return ExitRecommendation(
            pct_to_sell=0.10,
            target_price=current_price,
            urgency="limit",
            reason=f"Default tranche ({wp:.0%}) — sell 10%",
            win_probability=wp,
            hold_value=ev_hold_per_contract,
        )

    def apply_exit(self, rec: ExitRecommendation) -> int:
        """
        Apply an exit recommendation, returning contracts to sell.
        Updates internal state.
        """
        contracts_to_sell = max(1, int(self.remaining * rec.pct_to_sell))
        contracts_to_sell = min(contracts_to_sell, self.remaining)

        if rec.urgency == "hold" or rec.pct_to_sell == 0:
            return 0

        self.remaining -= contracts_to_sell
        self.total_sold += contracts_to_sell

        pnl = (rec.target_price - self.entry_price) * contracts_to_sell / 100.0
        self.realized_pnl += pnl

        logger.info(
            f"EXIT: {contracts_to_sell} contracts @ {rec.target_price}c "
            f"(pnl=${pnl:+.2f}) | Remaining: {self.remaining}/{self.position_size}"
        )

        return contracts_to_sell
