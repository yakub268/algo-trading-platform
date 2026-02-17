"""
Kalshi Position Reconciliation
==============================

Reconciles positions between Kalshi API and local database every hour.
Alerts via Telegram if discrepancies are found.

This prevents:
- Ghost positions (DB thinks we have position, Kalshi doesn't)
- Missing positions (Kalshi has position, DB doesn't)
- Price/quantity mismatches

Schedule: Every hour

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient

logger = logging.getLogger('KalshiPositionReconciler')


@dataclass
class PositionDiscrepancy:
    """Position mismatch between DB and Kalshi"""
    ticker: str
    discrepancy_type: str  # 'ghost', 'missing', 'quantity_mismatch', 'price_mismatch'
    db_quantity: int
    api_quantity: int
    db_price: float
    api_price: float
    severity: str  # 'critical', 'warning', 'info'


class KalshiPositionReconciler:
    """Reconciles Kalshi positions with database"""

    def __init__(
        self,
        db_path: str = None,
        api_key_id: str = None,
        private_key_path: str = None,
        telegram_alerts: bool = True
    ):
        """
        Initialize position reconciler.

        Args:
            db_path: Path to event_trades.db
            api_key_id: Kalshi API key (or uses env var)
            private_key_path: Path to PEM private key (or uses env var)
            telegram_alerts: Send Telegram alerts for discrepancies
        """
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'event_trades.db'
        )

        self.client = KalshiClient(api_key_id, private_key_path)
        self.telegram_alerts = telegram_alerts

        # Initialize Telegram if enabled
        if self.telegram_alerts:
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils'))
                from telegram_alerts import send_telegram_alert
                self.send_alert = send_telegram_alert
            except ImportError:
                logger.warning("Telegram alerts not available")
                self.telegram_alerts = False

        logger.info(f"KalshiPositionReconciler initialized")

    def reconcile(self) -> List[PositionDiscrepancy]:
        """
        Reconcile positions between database and Kalshi API.

        Returns:
            List of discrepancies found
        """
        discrepancies = []

        try:
            # Get positions from both sources
            db_positions = self._get_db_positions()
            api_positions = self._get_api_positions()

            logger.info(f"Reconciling {len(db_positions)} DB positions vs {len(api_positions)} API positions")

            # Create lookup maps
            db_map = {p['ticker']: p for p in db_positions}
            api_map = {p['ticker']: p for p in api_positions}

            # Check for ghost positions (in DB but not on Kalshi)
            for ticker in db_map:
                if ticker not in api_map:
                    discrepancies.append(PositionDiscrepancy(
                        ticker=ticker,
                        discrepancy_type='ghost',
                        db_quantity=db_map[ticker]['quantity'],
                        api_quantity=0,
                        db_price=db_map[ticker]['price'],
                        api_price=0.0,
                        severity='critical'
                    ))
                    logger.warning(f"Ghost position: {ticker} in DB but not on Kalshi")

            # Check for missing positions (on Kalshi but not in DB)
            for ticker in api_map:
                if ticker not in db_map:
                    discrepancies.append(PositionDiscrepancy(
                        ticker=ticker,
                        discrepancy_type='missing',
                        db_quantity=0,
                        api_quantity=api_map[ticker]['quantity'],
                        db_price=0.0,
                        api_price=api_map[ticker]['price'],
                        severity='critical'
                    ))
                    logger.warning(f"Missing position: {ticker} on Kalshi but not in DB")

            # Check for quantity/price mismatches
            for ticker in set(db_map.keys()) & set(api_map.keys()):
                db_pos = db_map[ticker]
                api_pos = api_map[ticker]

                # Quantity mismatch
                if db_pos['quantity'] != api_pos['quantity']:
                    discrepancies.append(PositionDiscrepancy(
                        ticker=ticker,
                        discrepancy_type='quantity_mismatch',
                        db_quantity=db_pos['quantity'],
                        api_quantity=api_pos['quantity'],
                        db_price=db_pos['price'],
                        api_price=api_pos['price'],
                        severity='warning'
                    ))
                    logger.warning(
                        f"Quantity mismatch {ticker}: DB={db_pos['quantity']}, "
                        f"API={api_pos['quantity']}"
                    )

                # Price mismatch (tolerance: 1 cent)
                if abs(db_pos['price'] - api_pos['price']) > 0.01:
                    discrepancies.append(PositionDiscrepancy(
                        ticker=ticker,
                        discrepancy_type='price_mismatch',
                        db_quantity=db_pos['quantity'],
                        api_quantity=api_pos['quantity'],
                        db_price=db_pos['price'],
                        api_price=api_pos['price'],
                        severity='info'
                    ))
                    logger.info(
                        f"Price mismatch {ticker}: DB=${db_pos['price']:.2f}, "
                        f"API=${api_pos['price']:.2f}"
                    )

            # Send alert if critical discrepancies found
            if discrepancies and self.telegram_alerts:
                critical = [d for d in discrepancies if d.severity == 'critical']
                if critical:
                    self._send_discrepancy_alert(discrepancies)

            return discrepancies

        except Exception as e:
            logger.error(f"Reconciliation error: {e}", exc_info=True)
            return discrepancies

    def _get_db_positions(self) -> List[Dict]:
        """Get open positions from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all unsettled trades
            cursor.execute("""
                SELECT
                    ticker,
                    side,
                    SUM(quantity) as quantity,
                    AVG(cost) as price
                FROM trades
                WHERE settled_at IS NULL
                AND status = 'paper'
                GROUP BY ticker, side
            """)

            positions = []
            for row in cursor.fetchall():
                positions.append({
                    'ticker': row['ticker'],
                    'side': row['side'],
                    'quantity': row['quantity'],
                    'price': row['price']
                })

            conn.close()
            return positions

        except Exception as e:
            logger.error(f"Database error: {e}")
            return []

    def _get_api_positions(self) -> List[Dict]:
        """Get positions from Kalshi API"""
        try:
            # Get portfolio from Kalshi
            portfolio = self.client.get_portfolio()

            if not portfolio:
                logger.warning("No portfolio data from Kalshi API")
                return []

            positions = []
            market_positions = portfolio.get('market_positions', [])

            for pos in market_positions:
                # Skip closed positions
                if pos.get('position', 0) == 0:
                    continue

                positions.append({
                    'ticker': pos.get('ticker'),
                    'side': 'yes' if pos.get('position', 0) > 0 else 'no',
                    'quantity': abs(pos.get('position', 0)),
                    'price': pos.get('total_cost', 0) / abs(pos.get('position', 1))
                })

            return positions

        except Exception as e:
            logger.error(f"Kalshi API error: {e}")
            return []

    def _send_discrepancy_alert(self, discrepancies: List[PositionDiscrepancy]):
        """Send Telegram alert for position discrepancies"""
        try:
            critical = [d for d in discrepancies if d.severity == 'critical']
            warnings = [d for d in discrepancies if d.severity == 'warning']

            message = f"⚠️ **Kalshi Position Reconciliation Alert**\n\n"
            message += f"**Total Discrepancies:** {len(discrepancies)}\n"
            message += f"**Critical:** {len(critical)} | **Warnings:** {len(warnings)}\n\n"

            # Critical issues
            if critical:
                message += "**Critical Issues:**\n"
                for d in critical[:5]:  # Top 5
                    if d.discrepancy_type == 'ghost':
                        message += f"  🔴 Ghost: {d.ticker} (DB:{d.db_quantity}, API:0)\n"
                    elif d.discrepancy_type == 'missing':
                        message += f"  🔴 Missing: {d.ticker} (DB:0, API:{d.api_quantity})\n"
                message += "\n"

            # Warnings
            if warnings:
                message += "**Warnings:**\n"
                for d in warnings[:3]:  # Top 3
                    message += f"  🟡 {d.ticker}: DB qty={d.db_quantity}, API qty={d.api_quantity}\n"

            message += "\n**Action:** Check positions immediately"

            self.send_alert(message)
            logger.info(f"Sent discrepancy alert: {len(discrepancies)} issues")

        except Exception as e:
            logger.error(f"Failed to send discrepancy alert: {e}")

    def auto_fix_ghosts(self, dry_run: bool = True) -> int:
        """
        Auto-fix ghost positions by marking them as closed in DB.

        Args:
            dry_run: If True, only log what would be fixed

        Returns:
            Number of positions fixed
        """
        fixed = 0

        try:
            discrepancies = self.reconcile()
            ghosts = [d for d in discrepancies if d.discrepancy_type == 'ghost']

            if not ghosts:
                logger.info("No ghost positions to fix")
                return 0

            if dry_run:
                logger.info(f"DRY RUN: Would fix {len(ghosts)} ghost positions")
                for ghost in ghosts:
                    logger.info(f"  Would close: {ghost.ticker} (qty={ghost.db_quantity})")
                return len(ghosts)

            # Actually fix ghosts
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for ghost in ghosts:
                # Mark as closed with zero P&L
                cursor.execute("""
                    UPDATE trades
                    SET settled_price = cost,
                        settled_at = ?,
                        pnl = 0.0
                    WHERE ticker = ?
                    AND settled_at IS NULL
                """, (datetime.now(timezone.utc).isoformat(), ghost.ticker))

                fixed += 1
                logger.info(f"Fixed ghost position: {ghost.ticker}")

            conn.commit()
            conn.close()

            logger.info(f"Fixed {fixed} ghost positions")
            return fixed

        except Exception as e:
            logger.error(f"Error fixing ghosts: {e}")
            return fixed


def main():
    """CLI entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    reconciler = KalshiPositionReconciler()

    print("\n" + "="*80)
    print("KALSHI POSITION RECONCILIATION")
    print("="*80 + "\n")

    discrepancies = reconciler.reconcile()

    if not discrepancies:
        print("✅ No discrepancies found - positions match!")
    else:
        print(f"⚠️  Found {len(discrepancies)} discrepancies:\n")

        critical = [d for d in discrepancies if d.severity == 'critical']
        warnings = [d for d in discrepancies if d.severity == 'warning']
        info = [d for d in discrepancies if d.severity == 'info']

        if critical:
            print(f"CRITICAL ({len(critical)}):")
            for d in critical:
                print(f"  {d.discrepancy_type.upper()}: {d.ticker}")
                print(f"    DB: {d.db_quantity} @ ${d.db_price:.2f}")
                print(f"    API: {d.api_quantity} @ ${d.api_price:.2f}")

        if warnings:
            print(f"\nWARNINGS ({len(warnings)}):")
            for d in warnings:
                print(f"  {d.ticker}: DB={d.db_quantity}, API={d.api_quantity}")

        if info:
            print(f"\nINFO ({len(info)}):")
            for d in info:
                print(f"  {d.ticker}: Price diff ${abs(d.db_price - d.api_price):.2f}")

        # Offer to auto-fix ghosts
        if critical:
            print("\n" + "="*80)
            print("Run with --fix to auto-fix ghost positions (dry run mode)")
            print("="*80)

    print()


if __name__ == '__main__':
    main()
