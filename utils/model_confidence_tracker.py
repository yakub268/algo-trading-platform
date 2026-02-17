"""
Model Confidence Tracker
========================

Tracks Brier scores and prediction accuracy for each Kalshi signal source:
- CME FedWatch (Fed decisions)
- NWS Forecasts (Weather)
- Elo Ratings (Sports)
- FRED Data (Economic releases)
- CoinGecko (Crypto prices)

Only trades signals from models with Brier score < 0.15.
Provides model-specific confidence scoring.

Brier Score Formula:
  BS = (1/N) * Σ(predicted_prob - actual_outcome)²

Where:
  - predicted_prob = Our entry price (implied probability)
  - actual_outcome = Settlement result (1.00 or 0.00)
  - N = Number of predictions

Interpretation:
  - 0.00 = Perfect predictions
  - 0.25 = Random guessing
  - < 0.15 = Good model
  - > 0.25 = Worse than random

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import sqlite3
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('ModelConfidenceTracker')


@dataclass
class ModelScore:
    """Confidence score for a prediction model"""
    model_name: str
    signal_source: str  # 'fed', 'weather', 'sports', 'economic', 'crypto'
    total_predictions: int
    settled_predictions: int
    brier_score: float
    win_rate: float
    avg_edge_claimed: float
    avg_edge_realized: float
    confidence_level: str  # 'HIGH', 'MEDIUM', 'LOW', 'BLOCKED'
    last_updated: str


class ModelConfidenceTracker:
    """Tracks prediction model performance and confidence"""

    # Confidence thresholds
    BRIER_EXCELLENT = 0.10
    BRIER_GOOD = 0.15
    BRIER_ACCEPTABLE = 0.20
    MIN_PREDICTIONS = 10  # Minimum before scoring

    def __init__(
        self,
        db_path: str = None,
        scores_path: str = None
    ):
        """
        Initialize model confidence tracker.

        Args:
            db_path: Path to event_trades.db
            scores_path: Path to save model scores JSON
        """
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'event_trades.db'
        )

        self.scores_path = scores_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'model_confidence_scores.json'
        )

        logger.info(f"ModelConfidenceTracker initialized")

    def update_scores(self) -> Dict[str, ModelScore]:
        """
        Update Brier scores for all prediction models.

        Returns:
            Dict of model_name -> ModelScore
        """
        try:
            # Get all settled trades grouped by source
            trades_by_source = self._get_trades_by_source()

            scores = {}

            for source, trades in trades_by_source.items():
                score = self._calculate_model_score(source, trades)
                if score:
                    scores[source] = score

            # Save scores
            self._save_scores(scores)

            logger.info(f"Updated scores for {len(scores)} models")
            return scores

        except Exception as e:
            logger.error(f"Error updating scores: {e}")
            return {}

    def should_trade_signal(self, source: str, edge: float) -> Tuple[bool, str]:
        """
        Check if a signal should be traded based on model confidence.

        Args:
            source: Signal source ('fed', 'weather', 'sports', 'economic', 'crypto')
            edge: Claimed edge percentage

        Returns:
            (should_trade, reason)
        """
        try:
            scores = self._load_scores()

            if source not in scores:
                # No history for this model yet
                if edge >= 0.10:  # Require 10% edge for unproven models
                    return True, "No history, high edge required"
                else:
                    return False, f"No history, edge {edge:.1%} < 10% required"

            score = scores[source]

            # Check if model is blocked
            if score['confidence_level'] == 'BLOCKED':
                return False, f"Model blocked (Brier {score['brier_score']:.3f} > 0.20)"

            # Check Brier score
            if score['brier_score'] > self.BRIER_GOOD:
                return False, f"Brier score {score['brier_score']:.3f} > {self.BRIER_GOOD}"

            # Check edge requirement based on confidence
            min_edge = self._get_min_edge_for_confidence(score['confidence_level'])

            if edge < min_edge:
                return False, f"Edge {edge:.1%} < {min_edge:.1%} required for {score['confidence_level']} confidence"

            return True, f"Model approved ({score['confidence_level']}, Brier {score['brier_score']:.3f})"

        except Exception as e:
            logger.error(f"Error checking signal: {e}")
            return False, "Error checking model confidence"

    def _get_trades_by_source(self) -> Dict[str, List[Dict]]:
        """Get all settled trades grouped by signal source"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT ticker, side, cost, settled_price, reasoning, timestamp
                FROM trades
                WHERE settled_at IS NOT NULL
                AND status = 'paper'
                ORDER BY timestamp DESC
            """)

            trades_by_source = defaultdict(list)

            for row in cursor.fetchall():
                trade = dict(row)

                # Determine source from ticker
                ticker = trade['ticker']
                if 'KXFED' in ticker:
                    source = 'fed'
                elif 'WEATHER' in ticker or 'TEMP' in ticker:
                    source = 'weather'
                elif 'NFL' in ticker or 'NBA' in ticker or 'MLB' in ticker:
                    source = 'sports'
                elif 'CPI' in ticker or 'GDP' in ticker or 'PAYEMS' in ticker:
                    source = 'economic'
                elif 'BTC' in ticker or 'ETH' in ticker:
                    source = 'crypto'
                else:
                    source = 'other'

                trades_by_source[source].append(trade)

            conn.close()
            return dict(trades_by_source)

        except Exception as e:
            logger.error(f"Error getting trades by source: {e}")
            return {}

    def _calculate_model_score(self, source: str, trades: List[Dict]) -> Optional[ModelScore]:
        """Calculate Brier score and confidence for a model"""
        try:
            if not trades:
                return None

            total_predictions = len(trades)

            # Calculate Brier score
            total_squared_error = 0.0
            winners = 0
            total_edge_claimed = 0.0
            total_edge_realized = 0.0

            for trade in trades:
                # Predicted probability (entry price)
                predicted_prob = trade['cost']

                # Actual outcome (settlement price)
                actual_outcome = trade['settled_price']

                # Squared error
                squared_error = (predicted_prob - actual_outcome) ** 2
                total_squared_error += squared_error

                # Win tracking
                if actual_outcome > 0.5:  # Won
                    winners += 1

                # Edge tracking
                # Claimed edge (from reasoning if available)
                reasoning = trade.get('reasoning', '')
                if 'edge=' in reasoning:
                    try:
                        edge_str = reasoning.split('edge=')[1].split('%')[0]
                        edge = float(edge_str) / 100
                        total_edge_claimed += edge
                    except Exception:
                        pass

                # Realized edge
                realized_edge = actual_outcome - predicted_prob
                total_edge_realized += realized_edge

            brier_score = total_squared_error / total_predictions
            win_rate = winners / total_predictions
            avg_edge_claimed = total_edge_claimed / total_predictions if total_predictions > 0 else 0.0
            avg_edge_realized = total_edge_realized / total_predictions if total_predictions > 0 else 0.0

            # Determine confidence level
            if total_predictions < self.MIN_PREDICTIONS:
                confidence = 'LOW'
            elif brier_score < self.BRIER_EXCELLENT:
                confidence = 'HIGH'
            elif brier_score < self.BRIER_GOOD:
                confidence = 'MEDIUM'
            elif brier_score < self.BRIER_ACCEPTABLE:
                confidence = 'LOW'
            else:
                confidence = 'BLOCKED'

            return ModelScore(
                model_name=source.title(),
                signal_source=source,
                total_predictions=total_predictions,
                settled_predictions=total_predictions,
                brier_score=round(brier_score, 4),
                win_rate=round(win_rate, 4),
                avg_edge_claimed=round(avg_edge_claimed, 4),
                avg_edge_realized=round(avg_edge_realized, 4),
                confidence_level=confidence,
                last_updated=datetime.now(timezone.utc).isoformat()
            )

        except Exception as e:
            logger.error(f"Error calculating model score for {source}: {e}")
            return None

    def _get_min_edge_for_confidence(self, confidence: str) -> float:
        """Get minimum edge requirement based on confidence level"""
        if confidence == 'HIGH':
            return 0.05  # 5% edge for high confidence models
        elif confidence == 'MEDIUM':
            return 0.08  # 8% edge for medium confidence
        elif confidence == 'LOW':
            return 0.12  # 12% edge for low confidence
        else:  # BLOCKED
            return 1.00  # Effectively block all trades

    def _load_scores(self) -> Dict[str, Dict]:
        """Load model scores from file"""
        try:
            if not os.path.exists(self.scores_path):
                return {}

            with open(self.scores_path, 'r') as f:
                data = json.load(f)

            return data.get('scores', {})

        except Exception as e:
            logger.error(f"Error loading scores: {e}")
            return {}

    def _save_scores(self, scores: Dict[str, ModelScore]):
        """Save model scores to file"""
        try:
            os.makedirs(os.path.dirname(self.scores_path), exist_ok=True)

            data = {
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'scores': {
                    source: asdict(score)
                    for source, score in scores.items()
                }
            }

            with open(self.scores_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(scores)} model scores to {self.scores_path}")

        except Exception as e:
            logger.error(f"Error saving scores: {e}")

    def get_report(self) -> str:
        """Generate human-readable report of model confidence"""
        try:
            scores = self._load_scores()

            if not scores:
                return "No model scores available"

            report = "="*80 + "\n"
            report += "MODEL CONFIDENCE REPORT\n"
            report += "="*80 + "\n\n"

            # Sort by Brier score
            sorted_models = sorted(
                scores.items(),
                key=lambda x: x[1]['brier_score']
            )

            for source, score in sorted_models:
                icon = {
                    'HIGH': '🟢',
                    'MEDIUM': '🟡',
                    'LOW': '🟠',
                    'BLOCKED': '🔴'
                }.get(score['confidence_level'], '⚪')

                report += f"{icon} **{score['model_name']}** ({score['confidence_level']})\n"
                report += f"   Brier Score: {score['brier_score']:.3f}\n"
                report += f"   Win Rate: {score['win_rate']:.1%}\n"
                report += f"   Predictions: {score['settled_predictions']}\n"
                report += f"   Avg Edge Claimed: {score['avg_edge_claimed']:.1%}\n"
                report += f"   Avg Edge Realized: {score['avg_edge_realized']:.1%}\n"
                report += f"   Min Edge Required: {self._get_min_edge_for_confidence(score['confidence_level']):.1%}\n"
                report += "\n"

            report += "="*80 + "\n"

            return report

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return "Error generating report"


def main():
    """CLI entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    tracker = ModelConfidenceTracker()

    print("\nUpdating model confidence scores...\n")
    scores = tracker.update_scores()

    print(tracker.get_report())

    # Test signal approval
    print("="*80)
    print("SIGNAL APPROVAL TESTS")
    print("="*80 + "\n")

    test_cases = [
        ('fed', 0.05),
        ('fed', 0.10),
        ('weather', 0.08),
        ('sports', 0.12),
        ('unknown_model', 0.15)
    ]

    for source, edge in test_cases:
        should_trade, reason = tracker.should_trade_signal(source, edge)
        status = "✅ APPROVED" if should_trade else "❌ REJECTED"
        print(f"{status} - {source.upper()} with {edge:.1%} edge")
        print(f"         Reason: {reason}\n")


if __name__ == '__main__':
    main()
