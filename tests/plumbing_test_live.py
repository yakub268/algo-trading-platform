"""
END-TO-END P&L PLUMBING TEST
==============================

Tests the REAL code path for opening and closing trades to verify:
- Trade logging to database works correctly
- Exit price is captured properly
- P&L is calculated and stored correctly
- close_trade function writes all fields

This uses the SAME functions as the orchestrator - no separate DB logic.

Usage:
    python tests/plumbing_test_live.py

Warning: Executes REAL trades on Alpaca Paper Trading API ($10 BTC trade)
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Import orchestrator components - use REAL code paths
from master_orchestrator import TradingDB, PaperTrade
from bots.alpaca_crypto_client import AlpacaCryptoClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PlumbingTest')


class PlumbingTest:
    """End-to-end trade lifecycle test using real orchestrator functions"""

    def __init__(self):
        # Use LIVE database to verify real plumbing
        self.db = TradingDB(db_name='trading_master.db')
        self.client = AlpacaCryptoClient()  # Reads PAPER_MODE from env
        self.test_symbol = 'BTC/USD'
        self.test_bot_name = 'PLUMBING_TEST'
        self.trade_id = None

    def run(self):
        """Execute full test cycle"""
        try:
            logger.info("="*60)
            logger.info("STARTING P&L PLUMBING TEST")
            logger.info("="*60)

            # Step 1: Get current price
            logger.info(f"\n[STEP 1] Getting current {self.test_symbol} price...")
            entry_price = self.client.get_price(self.test_symbol)
            if not entry_price or entry_price <= 0:
                logger.error(f"Failed to get price for {self.test_symbol}")
                return False
            logger.info(f"✓ Current price: ${entry_price:,.2f}")

            # Step 2: Place market buy for $10 (Alpaca minimum)
            logger.info(f"\n[STEP 2] Placing market BUY for $10 worth of {self.test_symbol}...")
            quantity = 10.0 / entry_price  # $10 worth
            logger.info(f"Calculated quantity: {quantity:.8f} BTC")

            buy_result = self.client.create_market_order(
                product_id=self.test_symbol,
                side='buy',
                quote_size='10'  # $10 notional (Alpaca minimum)
            )

            if not buy_result or not buy_result.get('success'):
                logger.error(f"Buy order failed: {buy_result}")
                return False
            logger.info(f"✓ Buy order placed: {buy_result.get('order_id')}")

            # Step 3: Get confirmed fill price from order result
            logger.info(f"\n[STEP 3] Extracting confirmed fill from order response...")

            # CRITICAL: With the zero-P&L fix, create_market_order now polls for fills
            # and returns CONFIRMED fill prices (not estimates)
            filled_qty = float(buy_result.get('filled_qty', 0))
            actual_entry_price = float(buy_result.get('filled_avg_price', 0))

            if filled_qty <= 0 or actual_entry_price <= 0:
                logger.error(f"Invalid fill data: qty={filled_qty}, price={actual_entry_price}")
                logger.error("This indicates the zero-P&L bug is still present!")
                return False

            logger.info(f"✓ Order CONFIRMED filled - qty: {filled_qty:.8f}, price: ${actual_entry_price:,.2f}")
            logger.info(f"  (This is the ACTUAL fill price from Alpaca, not an estimate)")

            # Step 4: Log trade to DB using ORCHESTRATOR functions
            logger.info(f"\n[STEP 4] Logging trade to database using orchestrator functions...")
            self.trade_id = f"{self.test_bot_name}_{self.test_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            trade = PaperTrade(
                trade_id=self.trade_id,
                bot_name=self.test_bot_name,
                market='crypto',
                symbol=self.test_symbol,
                side='buy',
                entry_price=actual_entry_price,
                exit_price=None,
                quantity=filled_qty,
                entry_time=datetime.now(),
                exit_time=None,
                pnl=0,
                status='open'
            )

            self.db.log_trade(trade)
            logger.info(f"✓ Trade logged to DB with trade_id: {self.trade_id}")

            # Verify it was logged correctly
            open_trades = self.db.get_open_trades()
            our_trade = [t for t in open_trades if t.get('trade_id') == self.trade_id]
            if not our_trade:
                logger.error("Trade not found in DB after logging!")
                return False
            logger.info(f"✓ Trade verified in DB: status={our_trade[0].get('status')}, pnl={our_trade[0].get('pnl')}")

            # Step 5: Wait 60 seconds
            logger.info(f"\n[STEP 5] Waiting 60 seconds for price movement...")
            for i in range(12):
                time.sleep(5)
                logger.info(f"  {(i+1)*5}s elapsed...")

            # Step 6: Get current price for exit
            logger.info(f"\n[STEP 6] Getting current {self.test_symbol} price for exit...")
            exit_price = self.client.get_price(self.test_symbol)
            if not exit_price or exit_price <= 0:
                logger.error(f"Failed to get exit price for {self.test_symbol}")
                return False
            logger.info(f"✓ Current exit price: ${exit_price:,.2f}")

            # Step 7: Place market sell
            logger.info(f"\n[STEP 7] Placing market SELL for {filled_qty:.8f} {self.test_symbol}...")
            sell_result = self.client.create_market_order(
                product_id=self.test_symbol,
                side='sell',
                base_size=str(filled_qty)  # Quantity to sell
            )

            if not sell_result or not sell_result.get('success'):
                logger.error(f"Sell order failed: {sell_result}")
                return False
            logger.info(f"✓ Sell order placed: {sell_result.get('order_id')}")

            # Step 8: Get confirmed fill price from order result
            logger.info(f"\n[STEP 8] Extracting confirmed fill from sell order response...")

            # CRITICAL: With the zero-P&L fix, create_market_order now polls for fills
            # and returns CONFIRMED fill prices (not estimates)
            actual_exit_price = float(sell_result.get('filled_avg_price', 0))

            if actual_exit_price <= 0:
                logger.error(f"Invalid exit fill price: {actual_exit_price}")
                logger.error("This indicates the zero-P&L bug is still present!")
                return False

            logger.info(f"✓ Sell order CONFIRMED filled at ${actual_exit_price:,.2f}")
            logger.info(f"  (This is the ACTUAL fill price from Alpaca, not an estimate)")

            # Step 9: Calculate P&L
            logger.info(f"\n[STEP 9] Calculating P&L...")
            pnl = filled_qty * (actual_exit_price - actual_entry_price)
            pnl_pct = ((actual_exit_price - actual_entry_price) / actual_entry_price) * 100

            logger.info(f"Entry: ${actual_entry_price:,.2f}")
            logger.info(f"Exit:  ${actual_exit_price:,.2f}")
            logger.info(f"Qty:   {filled_qty:.8f} BTC")
            logger.info(f"P&L:   ${pnl:+.4f} ({pnl_pct:+.4f}%)")

            # Step 10: Close trade in DB using ORCHESTRATOR function
            logger.info(f"\n[STEP 10] Closing trade in database using orchestrator functions...")
            self.db.close_trade(
                bot_name=self.test_bot_name,
                symbol=self.test_symbol,
                exit_price=actual_exit_price,
                pnl=pnl,
                status='closed',
                gross_pnl=pnl,
                net_pnl=pnl * 0.995,  # Assume 0.5% fees
                estimated_fees=abs(pnl) * 0.005
            )
            logger.info(f"✓ Trade closed in DB")

            # Step 11: Query DB and verify
            logger.info(f"\n[STEP 11] Querying database to verify trade record...")
            with self.db._lock:
                import sqlite3
                with sqlite3.connect(self.db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT trade_id, bot_name, symbol, side, entry_price, exit_price,
                               quantity, pnl, pnl_pct, status, entry_time, exit_time
                        FROM trades WHERE trade_id = ?
                    ''', (self.trade_id,))
                    row = cursor.fetchone()

            if not row:
                logger.error("Trade not found in DB after closing!")
                return False

            logger.info(f"\n{'='*60}")
            logger.info("TRADE RECORD FROM DATABASE:")
            logger.info(f"{'='*60}")
            logger.info(f"Trade ID:     {row[0]}")
            logger.info(f"Bot:          {row[1]}")
            logger.info(f"Symbol:       {row[2]}")
            logger.info(f"Side:         {row[3]}")
            logger.info(f"Entry Price:  ${row[4]:,.2f}")
            logger.info(f"Exit Price:   ${row[5]:,.2f}")
            logger.info(f"Quantity:     {row[6]:.8f}")
            logger.info(f"P&L:          ${row[7]:+.4f}")
            logger.info(f"P&L %:        {row[8]:+.4f}%")
            logger.info(f"Status:       {row[9]}")
            logger.info(f"Entry Time:   {row[10]}")
            logger.info(f"Exit Time:    {row[11]}")
            logger.info(f"{'='*60}")

            # Step 12: VERIFY
            logger.info(f"\n[STEP 12] VERIFICATION...")
            checks = []

            # Check 1: Status is 'closed'
            if row[9] == 'closed':
                logger.info("✓ PASS: status = 'closed'")
                checks.append(True)
            else:
                logger.error(f"✗ FAIL: status = '{row[9]}' (expected 'closed')")
                checks.append(False)

            # Check 2: exit_price is not NULL and > 0
            if row[5] and row[5] > 0:
                logger.info(f"✓ PASS: exit_price = ${row[5]:,.2f} (not NULL)")
                checks.append(True)
            else:
                logger.error(f"✗ FAIL: exit_price = {row[5]} (expected > 0)")
                checks.append(False)

            # Check 3: entry_price is not NULL and > 0
            if row[4] and row[4] > 0:
                logger.info(f"✓ PASS: entry_price = ${row[4]:,.2f} (not NULL)")
                checks.append(True)
            else:
                logger.error(f"✗ FAIL: entry_price = {row[4]} (expected > 0)")
                checks.append(False)

            # Check 4: pnl is not 0 (should have some value, even if small)
            if row[7] is not None and row[7] != 0:
                logger.info(f"✓ PASS: pnl = ${row[7]:+.4f} (not 0)")
                checks.append(True)
            else:
                logger.error(f"✗ FAIL: pnl = {row[7]} (expected != 0)")
                checks.append(False)

            # Check 5: pnl_pct is not 0
            if row[8] is not None and row[8] != 0:
                logger.info(f"✓ PASS: pnl_pct = {row[8]:+.4f}% (not 0)")
                checks.append(True)
            else:
                logger.error(f"✗ FAIL: pnl_pct = {row[8]} (expected != 0)")
                checks.append(False)

            # Final verdict
            logger.info(f"\n{'='*60}")
            if all(checks):
                logger.info("🎉 TEST RESULT: PASS")
                logger.info("All P&L plumbing checks passed successfully!")
            else:
                logger.error("❌ TEST RESULT: FAIL")
                logger.error(f"Failed {len([c for c in checks if not c])}/{len(checks)} checks")
            logger.info(f"{'='*60}\n")

            return all(checks)

        except Exception as e:
            logger.error(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


if __name__ == '__main__':
    test = PlumbingTest()
    success = test.run()
    sys.exit(0 if success else 1)
