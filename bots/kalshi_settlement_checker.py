"""
Kalshi Settlement Checker
=========================

Checks Kalshi positions for settlement and updates P&L in database.

This runs daily to:
1. Query Kalshi API for all closed/settled markets
2. Update event_trades.db with settlement prices and P&L
3. Send Telegram alerts for settled positions

Schedule: Daily at 10:00 AM ET (after most Fed announcements)

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient

logger = logging.getLogger('KalshiSettlementChecker')


@dataclass
class SettledTrade:
    """Settled trade information"""
    trade_id: str
    ticker: str
    side: str
    quantity: int
    entry_price: float
    settlement_price: float
    pnl: float
    settled_at: str


class KalshiSettlementChecker:
    """Checks for settled Kalshi positions and updates database"""

    def __init__(
        self,
        db_path: str = None,
        api_key_id: str = None,
        private_key_path: str = None,
        telegram_alerts: bool = True
    ):
        """
        Initialize settlement checker.

        Args:
            db_path: Path to event_trades.db
            api_key_id: Kalshi API key (or uses env var)
            private_key_path: Path to PEM private key (or uses env var)
            telegram_alerts: Send Telegram notifications for settlements
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

        logger.info(f"KalshiSettlementChecker initialized (db={self.db_path})")

    def check_settlements(self) -> List[SettledTrade]:
        """
        Check all unsettled trades for settlements.

        Returns:
            List of newly settled trades
        """
        settled_trades = []

        try:
            # Get all unsettled trades from database
            unsettled = self._get_unsettled_trades()
            logger.info(f"Checking {len(unsettled)} unsettled trades")

            for trade in unsettled:
                # Check if market has settled
                settlement = self._check_market_settlement(trade['ticker'])

                if settlement:
                    # Calculate P&L and update database
                    pnl = self._calculate_pnl(
                        side=trade['side'],
                        quantity=trade['quantity'],
                        entry_price=trade['cost'],
                        settlement_price=settlement['result_price']
                    )

                    # Update database
                    self._update_settlement(
                        trade_id=trade['trade_id'],
                        settlement_price=settlement['result_price'],
                        settled_at=settlement['close_time'],
                        pnl=pnl
                    )

                    # Track for reporting
                    settled_trade = SettledTrade(
                        trade_id=trade['trade_id'],
                        ticker=trade['ticker'],
                        side=trade['side'],
                        quantity=trade['quantity'],
                        entry_price=trade['cost'],
                        settlement_price=settlement['result_price'],
                        pnl=pnl,
                        settled_at=settlement['close_time']
                    )
                    settled_trades.append(settled_trade)

                    logger.info(f"Settled {trade['ticker']}: P&L=${pnl:.2f}")

            # Send summary alert
            if settled_trades and self.telegram_alerts:
                self._send_settlement_alert(settled_trades)

            return settled_trades

        except Exception as e:
            logger.error(f"Error checking settlements: {e}", exc_info=True)
            return settled_trades

    def _get_unsettled_trades(self) -> List[Dict]:
        """Get all trades without settlement data"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT trade_id, ticker, side, quantity, cost
                FROM trades
                WHERE settled_at IS NULL
                AND status = 'paper'
                ORDER BY timestamp ASC
            """)

            trades = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return trades

        except Exception as e:
            logger.error(f"Database error: {e}")
            return []

    def _check_market_settlement(self, ticker: str) -> Optional[Dict]:
        """
        Check if a market has settled via Kalshi API.

        Args:
            ticker: Market ticker (e.g., KXFEDDECISION-26JUN-H0)

        Returns:
            Dict with settlement info if settled, None otherwise
        """
        try:
            # Get market info from Kalshi
            market = self.client.get_market(ticker)

            if not market:
                logger.debug(f"Market {ticker} not found")
                return None

            # Check if market is closed and settled
            if market.get('status') != 'closed':
                return None

            # Get settlement price (YES settled at 100, NO at 0)
            result = market.get('result')
            if result not in ['yes', 'no']:
                logger.debug(f"Market {ticker} closed but not settled (result={result})")
                return None

            # Result price: 1.00 if YES won, 0.00 if NO won
            result_price = 1.00 if result == 'yes' else 0.00

            return {
                'result_price': result_price,
                'result': result,
                'close_time': market.get('close_time') or market.get('close_date'),
                'settlement_value': market.get('settlement_value')
            }

        except Exception as e:
            logger.debug(f"Error checking market {ticker}: {e}")
            return None

    def _calculate_pnl(
        self,
        side: str,
        quantity: int,
        entry_price: float,
        settlement_price: float
    ) -> float:
        """
        Calculate P&L for a settled trade.

        Args:
            side: 'yes' or 'no'
            quantity: Number of contracts
            entry_price: Entry price per contract (cents)
            settlement_price: Settlement price (1.00 or 0.00)

        Returns:
            P&L in USD
        """
        # Entry cost (already paid)
        entry_cost = quantity * entry_price

        # Settlement payout
        payout = quantity * settlement_price

        # P&L = Payout - Cost
        pnl = payout - entry_cost

        return round(pnl, 2)

    def _update_settlement(
        self,
        trade_id: str,
        settlement_price: float,
        settled_at: str,
        pnl: float
    ):
        """Update database with settlement information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE trades
                SET settled_price = ?,
                    settled_at = ?,
                    pnl = ?
                WHERE trade_id = ?
            """, (settlement_price, settled_at, pnl, trade_id))

            conn.commit()
            conn.close()

            logger.info(f"Updated settlement for {trade_id}: price={settlement_price}, pnl=${pnl:.2f}")

        except Exception as e:
            logger.error(f"Failed to update settlement for {trade_id}: {e}")

    def _send_settlement_alert(self, settled_trades: List[SettledTrade]):
        """Send Telegram alert for settled trades"""
        try:
            total_pnl = sum(t.pnl for t in settled_trades)
            winners = [t for t in settled_trades if t.pnl > 0]
            losers = [t for t in settled_trades if t.pnl < 0]

            message = f"🎯 **Kalshi Settlements** ({len(settled_trades)} trades)\n\n"

            message += f"**Total P&L:** ${total_pnl:+.2f}\n"
            message += f"**Winners:** {len(winners)} | **Losers:** {len(losers)}\n\n"

            # Top 3 winners
            if winners:
                message += "**Top Winners:**\n"
                for trade in sorted(winners, key=lambda t: t.pnl, reverse=True)[:3]:
                    message += f"  • {trade.ticker}: +${trade.pnl:.2f}\n"
                message += "\n"

            # Top 3 losers
            if losers:
                message += "**Top Losers:**\n"
                for trade in sorted(losers, key=lambda t: t.pnl)[:3]:
                    message += f"  • {trade.ticker}: ${trade.pnl:.2f}\n"

            self.send_alert(message)
            logger.info(f"Sent settlement alert: {len(settled_trades)} trades, ${total_pnl:+.2f}")

        except Exception as e:
            logger.error(f"Failed to send settlement alert: {e}")

    def get_settlement_stats(self, days: int = 30) -> Dict:
        """
        Get settlement statistics for the last N days.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with win rate, total P&L, avg trade P&L
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losers,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as max_win,
                    MIN(pnl) as max_loss
                FROM trades
                WHERE settled_at IS NOT NULL
                AND settled_at >= ?
            """, (since,))

            row = cursor.fetchone()
            conn.close()

            if not row or row[0] == 0:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_pnl': 0.0,
                    'max_win': 0.0,
                    'max_loss': 0.0
                }

            total, winners, losers, total_pnl, avg_pnl, max_win, max_loss = row

            return {
                'total_trades': total,
                'winners': winners or 0,
                'losers': losers or 0,
                'win_rate': (winners or 0) / total if total > 0 else 0.0,
                'total_pnl': total_pnl or 0.0,
                'avg_pnl': avg_pnl or 0.0,
                'max_win': max_win or 0.0,
                'max_loss': max_loss or 0.0
            }

        except Exception as e:
            logger.error(f"Error getting settlement stats: {e}")
            return {}


def main():
    """CLI entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    checker = KalshiSettlementChecker()

    # Check settlements
    print("\n" + "="*80)
    print("CHECKING KALSHI SETTLEMENTS")
    print("="*80 + "\n")

    settled = checker.check_settlements()

    print(f"Settled {len(settled)} trades:")
    for trade in settled:
        print(f"  {trade.ticker}: ${trade.pnl:+.2f} ({trade.side.upper()})")

    # Get stats
    print("\n" + "="*80)
    print("30-DAY SETTLEMENT STATS")
    print("="*80 + "\n")

    stats = checker.get_settlement_stats(days=30)
    if stats.get('total_trades', 0) > 0:
        print(f"Total Trades:  {stats['total_trades']}")
        print(f"Win Rate:      {stats['win_rate']:.1%}")
        print(f"Total P&L:     ${stats['total_pnl']:+.2f}")
        print(f"Avg P&L:       ${stats['avg_pnl']:+.2f}")
        print(f"Max Win:       ${stats['max_win']:+.2f}")
        print(f"Max Loss:      ${stats['max_loss']:+.2f}")
    else:
        print("No settled trades in last 30 days")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
