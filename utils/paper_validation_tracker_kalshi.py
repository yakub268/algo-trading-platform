"""
Kalshi Paper Validation Tracker
================================

Tracks paper trading performance for 30-day validation period.
Monitors:
- Win rate
- Total P&L
- Profit factor
- Number of settled trades

Go-live criteria:
- Win rate >= 55%
- Profit factor >= 1.5
- At least 20 settled trades
- Brier score < 0.15 (prediction accuracy)

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import sqlite3
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('KalshiPaperValidation')


@dataclass
class ValidationMetrics:
    """Validation metrics for go-live decision"""
    start_date: str
    end_date: str
    total_trades: int
    settled_trades: int
    unsettled_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    max_win: float
    max_loss: float
    profit_factor: float
    brier_score: float
    ready_for_live: bool
    missing_criteria: List[str]


class KalshiPaperValidationTracker:
    """Tracks paper trading validation progress"""

    # Go-live criteria
    MIN_WIN_RATE = 0.55  # 55%
    MIN_PROFIT_FACTOR = 1.5
    MIN_SETTLED_TRADES = 20
    MAX_BRIER_SCORE = 0.15
    VALIDATION_DAYS = 30

    def __init__(self, db_path: str = None, config_path: str = None):
        """
        Initialize validation tracker.

        Args:
            db_path: Path to event_trades.db
            config_path: Path to save validation config/progress
        """
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'event_trades.db'
        )

        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'kalshi_validation_config.json'
        )

        logger.info(f"KalshiPaperValidationTracker initialized")

    def start_validation(self) -> bool:
        """
        Start a new 30-day validation period.

        Returns:
            True if validation started, False if already in progress
        """
        try:
            # Check if validation already in progress
            config = self._load_config()

            if config and config.get('in_progress'):
                logger.warning("Validation already in progress")
                return False

            # Create new validation config
            start_date = datetime.now(timezone.utc)
            end_date = start_date + timedelta(days=self.VALIDATION_DAYS)

            config = {
                'in_progress': True,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'criteria': {
                    'min_win_rate': self.MIN_WIN_RATE,
                    'min_profit_factor': self.MIN_PROFIT_FACTOR,
                    'min_settled_trades': self.MIN_SETTLED_TRADES,
                    'max_brier_score': self.MAX_BRIER_SCORE
                }
            }

            self._save_config(config)
            logger.info(f"Started 30-day validation period: {start_date.date()} to {end_date.date()}")

            return True

        except Exception as e:
            logger.error(f"Failed to start validation: {e}")
            return False

    def get_metrics(self) -> Optional[ValidationMetrics]:
        """
        Get current validation metrics.

        Returns:
            ValidationMetrics object or None if no validation in progress
        """
        try:
            config = self._load_config()

            if not config or not config.get('in_progress'):
                logger.warning("No validation in progress")
                return None

            start_date = datetime.fromisoformat(config['start_date'])
            end_date = datetime.fromisoformat(config['end_date'])

            # Get trade stats
            stats = self._get_trade_stats(start_date, end_date)

            # Calculate Brier score
            brier_score = self._calculate_brier_score(start_date, end_date)

            # Check if ready for live
            ready, missing = self._check_go_live_criteria(stats, brier_score)

            return ValidationMetrics(
                start_date=config['start_date'],
                end_date=config['end_date'],
                total_trades=stats['total_trades'],
                settled_trades=stats['settled_trades'],
                unsettled_trades=stats['unsettled_trades'],
                win_rate=stats['win_rate'],
                total_pnl=stats['total_pnl'],
                avg_pnl_per_trade=stats['avg_pnl'],
                max_win=stats['max_win'],
                max_loss=stats['max_loss'],
                profit_factor=stats['profit_factor'],
                brier_score=brier_score,
                ready_for_live=ready,
                missing_criteria=missing
            )

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return None

    def _get_trade_stats(self, start_date: datetime, end_date: datetime) -> Dict:
        """Get trade statistics for validation period"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all trades in validation period
            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN settled_at IS NOT NULL THEN 1 ELSE 0 END) as settled_trades,
                    SUM(CASE WHEN settled_at IS NULL THEN 1 ELSE 0 END) as unsettled_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losers,
                    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as total_wins,
                    SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as total_losses,
                    SUM(COALESCE(pnl, 0)) as total_pnl,
                    AVG(COALESCE(pnl, 0)) as avg_pnl,
                    MAX(COALESCE(pnl, 0)) as max_win,
                    MIN(COALESCE(pnl, 0)) as max_loss
                FROM trades
                WHERE timestamp >= ? AND timestamp <= ?
                AND status = 'paper'
            """, (start_date.isoformat(), end_date.isoformat()))

            row = cursor.fetchone()
            conn.close()

            if not row or row[0] == 0:
                return {
                    'total_trades': 0,
                    'settled_trades': 0,
                    'unsettled_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_pnl': 0.0,
                    'max_win': 0.0,
                    'max_loss': 0.0,
                    'profit_factor': 0.0
                }

            (total, settled, unsettled, winners, losers,
             total_wins, total_losses, total_pnl, avg_pnl, max_win, max_loss) = row

            # Calculate metrics
            win_rate = (winners or 0) / settled if settled > 0 else 0.0
            profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')

            return {
                'total_trades': total,
                'settled_trades': settled or 0,
                'unsettled_trades': unsettled or 0,
                'win_rate': win_rate,
                'total_pnl': total_pnl or 0.0,
                'avg_pnl': avg_pnl or 0.0,
                'max_win': max_win or 0.0,
                'max_loss': max_loss or 0.0,
                'profit_factor': profit_factor
            }

        except Exception as e:
            logger.error(f"Error getting trade stats: {e}")
            return {}

    def _calculate_brier_score(self, start_date: datetime, end_date: datetime) -> float:
        """
        Calculate Brier score for prediction accuracy.

        Brier Score = (1/N) * Σ(predicted_prob - actual_outcome)²
        Lower is better. < 0.15 is good.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get settled trades with probability data
            cursor.execute("""
                SELECT side, cost, settled_price
                FROM trades
                WHERE timestamp >= ? AND timestamp <= ?
                AND settled_at IS NOT NULL
                AND status = 'paper'
            """, (start_date.isoformat(), end_date.isoformat()))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return 1.0  # Worst possible score

            # Calculate Brier score
            total_squared_error = 0.0
            count = 0

            for side, cost, settled_price in rows:
                # Our predicted probability (implied by entry price)
                predicted_prob = cost  # Entry price IS our probability estimate

                # Actual outcome (1 if won, 0 if lost)
                actual_outcome = settled_price  # 1.00 or 0.00

                # Squared error
                squared_error = (predicted_prob - actual_outcome) ** 2
                total_squared_error += squared_error
                count += 1

            brier_score = total_squared_error / count if count > 0 else 1.0

            return round(brier_score, 4)

        except Exception as e:
            logger.error(f"Error calculating Brier score: {e}")
            return 1.0

    def _check_go_live_criteria(self, stats: Dict, brier_score: float) -> tuple[bool, List[str]]:
        """
        Check if all go-live criteria are met.

        Returns:
            (ready, missing_criteria)
        """
        missing = []

        # Check win rate
        if stats.get('win_rate', 0) < self.MIN_WIN_RATE:
            missing.append(f"Win rate {stats.get('win_rate', 0):.1%} < {self.MIN_WIN_RATE:.1%}")

        # Check profit factor
        pf = stats.get('profit_factor', 0)
        if pf < self.MIN_PROFIT_FACTOR:
            missing.append(f"Profit factor {pf:.2f} < {self.MIN_PROFIT_FACTOR}")

        # Check settled trades
        settled = stats.get('settled_trades', 0)
        if settled < self.MIN_SETTLED_TRADES:
            missing.append(f"Settled trades {settled} < {self.MIN_SETTLED_TRADES}")

        # Check Brier score
        if brier_score > self.MAX_BRIER_SCORE:
            missing.append(f"Brier score {brier_score:.3f} > {self.MAX_BRIER_SCORE}")

        ready = len(missing) == 0

        return ready, missing

    def _load_config(self) -> Optional[Dict]:
        """Load validation config from file"""
        try:
            if not os.path.exists(self.config_path):
                return None

            with open(self.config_path, 'r') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return None

    def _save_config(self, config: Dict):
        """Save validation config to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Saved validation config to {self.config_path}")

        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def end_validation(self, force: bool = False):
        """
        End validation period.

        Args:
            force: Force end even if criteria not met
        """
        try:
            config = self._load_config()

            if not config or not config.get('in_progress'):
                logger.warning("No validation in progress")
                return

            metrics = self.get_metrics()

            if not force and not metrics.ready_for_live:
                logger.warning(f"Criteria not met: {', '.join(metrics.missing_criteria)}")
                return

            config['in_progress'] = False
            config['completed_at'] = datetime.now(timezone.utc).isoformat()
            config['final_metrics'] = asdict(metrics) if metrics else None

            self._save_config(config)
            logger.info("Validation period ended")

        except Exception as e:
            logger.error(f"Error ending validation: {e}")


def main():
    """CLI entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    import argparse
    parser = argparse.ArgumentParser(description='Kalshi Paper Validation Tracker')
    parser.add_argument('--start', action='store_true', help='Start validation period')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--end', action='store_true', help='End validation period')
    args = parser.parse_args()

    tracker = KalshiPaperValidationTracker()

    if args.start:
        tracker.start_validation()

    if args.status or not any([args.start, args.end]):
        metrics = tracker.get_metrics()

        if not metrics:
            print("\n❌ No validation in progress")
            print("Run with --start to begin validation period")
        else:
            print("\n" + "="*80)
            print("KALSHI PAPER VALIDATION STATUS")
            print("="*80 + "\n")
            print(f"Period: {metrics.start_date[:10]} to {metrics.end_date[:10]}")
            print(f"\n**Trades:**")
            print(f"  Total: {metrics.total_trades}")
            print(f"  Settled: {metrics.settled_trades}")
            print(f"  Unsettled: {metrics.unsettled_trades}")
            print(f"\n**Performance:**")
            print(f"  Win Rate: {metrics.win_rate:.1%} (need {tracker.MIN_WIN_RATE:.1%})")
            print(f"  Total P&L: ${metrics.total_pnl:+.2f}")
            print(f"  Avg P&L: ${metrics.avg_pnl_per_trade:+.2f}")
            print(f"  Profit Factor: {metrics.profit_factor:.2f} (need {tracker.MIN_PROFIT_FACTOR})")
            print(f"  Brier Score: {metrics.brier_score:.3f} (need <{tracker.MAX_BRIER_SCORE})")
            print(f"\n**Best/Worst:**")
            print(f"  Max Win: ${metrics.max_win:+.2f}")
            print(f"  Max Loss: ${metrics.max_loss:+.2f}")

            if metrics.ready_for_live:
                print(f"\n✅ **READY FOR LIVE TRADING**")
            else:
                print(f"\n❌ **NOT READY - Missing criteria:**")
                for criterion in metrics.missing_criteria:
                    print(f"  • {criterion}")

            print("\n" + "="*80 + "\n")

    if args.end:
        tracker.end_validation()


if __name__ == '__main__':
    main()
