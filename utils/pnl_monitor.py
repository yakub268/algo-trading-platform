"""
P&L MONITOR - Zero-P&L Trade Detector
======================================
The canary in the coal mine for the zero-P&L bug.

Scans the database for suspicious zero-P&L trades that indicate
the fill price capture bug is still present.

Run standalone: python utils/pnl_monitor.py
Or import: from utils.pnl_monitor import check_zero_pnl
"""

import os
import sys
import sqlite3
import logging
from typing import List, Tuple
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('PnLMonitor')


def check_zero_pnl(db_path='data/live/trading_master.db') -> bool:
    """
    Check for zero-P&L trades in the database.

    Args:
        db_path: Path to trading database

    Returns:
        True if no suspicious trades found, False otherwise
    """
    # Ensure path is absolute
    if not os.path.isabs(db_path):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(script_dir, db_path)

    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return False

    try:
        conn = sqlite3.connect(db_path)

        # Find all closed trades with suspicious P&L
        suspicious = conn.execute("""
            SELECT trade_id, symbol, entry_price, exit_price, pnl,
                   entry_time, exit_time, bot_name
            FROM trades
            WHERE status = 'closed'
            AND (
                pnl = 0
                OR pnl IS NULL
                OR exit_price IS NULL
                OR exit_price = 0
                OR abs(exit_price - entry_price) < 0.0000001
            )
            ORDER BY exit_time DESC
            LIMIT 100
        """).fetchall()

        conn.close()

        if suspicious:
            logger.warning(f"🚨 FOUND {len(suspicious)} ZERO-P&L TRADES:")
            for row in suspicious:
                trade_id = row[0]
                symbol = row[1]
                entry_price = row[2] if row[2] else 0
                exit_price = row[3] if row[3] else 0
                pnl = row[4] if row[4] else 0
                entry_time = row[5]
                exit_time = row[6]
                bot_name = row[7] if row[7] else 'unknown'

                logger.warning(
                    f"  {trade_id}: {symbol} ({bot_name}) | "
                    f"entry=${entry_price:.6f} exit=${exit_price:.6f} "
                    f"pnl=${pnl:.6f} | {entry_time} → {exit_time}"
                )

            return False
        else:
            logger.info("✅ No zero-P&L trades found")
            return True

    except Exception as e:
        logger.error(f"Error checking zero-P&L trades: {e}")
        return False


def check_recent_trades(db_path='data/live/trading_master.db', hours=24) -> dict:
    """
    Check recent trades for any issues.

    Args:
        db_path: Path to trading database
        hours: How many hours back to check

    Returns:
        dict with statistics
    """
    # Ensure path is absolute
    if not os.path.isabs(db_path):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(script_dir, db_path)

    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return {}

    try:
        conn = sqlite3.connect(db_path)

        # Get recent closed trades
        recent = conn.execute(f"""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN pnl = 0 OR pnl IS NULL THEN 1 ELSE 0 END) as zero_pnl,
                   SUM(CASE WHEN exit_price IS NULL OR exit_price = 0 THEN 1 ELSE 0 END) as zero_exit,
                   SUM(CASE WHEN abs(exit_price - entry_price) < 0.0000001 THEN 1 ELSE 0 END) as identical_prices,
                   AVG(pnl) as avg_pnl,
                   SUM(pnl) as total_pnl
            FROM trades
            WHERE status = 'closed'
            AND exit_time >= datetime('now', '-{hours} hours')
        """).fetchone()

        conn.close()

        stats = {
            'total_closed': recent[0] if recent[0] else 0,
            'zero_pnl_count': recent[1] if recent[1] else 0,
            'zero_exit_count': recent[2] if recent[2] else 0,
            'identical_prices_count': recent[3] if recent[3] else 0,
            'avg_pnl': recent[4] if recent[4] else 0,
            'total_pnl': recent[5] if recent[5] else 0,
        }

        logger.info(f"Recent {hours}h trade stats:")
        logger.info(f"  Total closed: {stats['total_closed']}")
        logger.info(f"  Zero P&L: {stats['zero_pnl_count']}")
        logger.info(f"  Zero exit price: {stats['zero_exit_count']}")
        logger.info(f"  Identical prices: {stats['identical_prices_count']}")
        logger.info(f"  Avg P&L: ${stats['avg_pnl']:.2f}")
        logger.info(f"  Total P&L: ${stats['total_pnl']:.2f}")

        return stats

    except Exception as e:
        logger.error(f"Error checking recent trades: {e}")
        return {}


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )

    print("\n" + "="*60)
    print("P&L MONITOR - Zero-P&L Trade Detector")
    print("="*60 + "\n")

    # Check both live and paper databases
    for mode in ['live', 'paper']:
        db_path = f'data/{mode}/trading_master.db'
        print(f"\nChecking {mode.upper()} database: {db_path}")
        print("-" * 60)

        # Check for zero-P&L trades
        is_clean = check_zero_pnl(db_path)

        if not is_clean:
            print(f"\n⚠️  {mode.upper()} database has zero-P&L trades!")
        else:
            print(f"\n✅ {mode.upper()} database is clean")

        # Check recent trades
        print(f"\nRecent 24h statistics for {mode.upper()}:")
        print("-" * 60)
        check_recent_trades(db_path, hours=24)

    print("\n" + "="*60)
    print("Monitor complete")
    print("="*60 + "\n")
