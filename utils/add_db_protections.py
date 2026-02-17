"""
ADD DATABASE PROTECTIONS
========================
Enable WAL mode and add validation checks to trading_master.db.

Part of PHASE5 zero-P&L bug fix.
"""

import os
import sys
import sqlite3
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('DBProtections')


def add_database_protections(db_path='data/live/trading_master.db'):
    """
    Add database-level protections to prevent zero-P&L bugs.

    1. Enable WAL mode for concurrent read/write
    2. Set busy timeout
    3. Display current schema
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
        cursor = conn.cursor()

        # Enable WAL mode for concurrent read/write
        logger.info("Enabling WAL mode...")
        cursor.execute("PRAGMA journal_mode=WAL;")
        result = cursor.fetchone()
        logger.info(f"  Journal mode: {result[0]}")

        # Set busy timeout (5 seconds)
        logger.info("Setting busy timeout...")
        cursor.execute("PRAGMA busy_timeout=5000;")
        logger.info("  Busy timeout: 5000ms")

        # Check current schema
        logger.info("\nCurrent trades table schema:")
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='trades';")
        schema = cursor.fetchone()
        if schema:
            print(schema[0])
        else:
            logger.warning("  trades table not found!")

        # Note: SQLite doesn't support adding CHECK constraints to existing tables
        # via ALTER TABLE. These will be enforced at the application level.
        logger.info("\nNOTE: Validation constraints enforced at application level:")
        logger.info("  - entry_price must be > 0 for open/closed trades")
        logger.info("  - exit_price must be > 0 for closed trades")
        logger.info("  - pnl should not be exactly 0.0 for closed trades")
        logger.info("  These are validated in order_fill_helper.py")

        conn.commit()
        conn.close()

        logger.info("\n✅ Database protections applied successfully")
        return True

    except Exception as e:
        logger.error(f"Error adding database protections: {e}")
        return False


if __name__ == '__main__':
    print("\n" + "="*60)
    print("ADD DATABASE PROTECTIONS")
    print("="*60 + "\n")

    # Apply to both live and paper databases
    for mode in ['live', 'paper']:
        db_path = f'data/{mode}/trading_master.db'
        print(f"\nProcessing {mode.upper()} database: {db_path}")
        print("-" * 60)

        success = add_database_protections(db_path)

        if success:
            print(f"✅ {mode.upper()} database updated")
        else:
            print(f"❌ Failed to update {mode.upper()} database")

    print("\n" + "="*60)
    print("Complete")
    print("="*60 + "\n")
