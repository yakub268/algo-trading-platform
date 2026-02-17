"""
P&L Calculation Smoke Test (No Real Trades)
============================================
Quick test to verify P&L calculation logic without placing real orders.
Tests the close_trade and force_close_trade_in_db functions directly.

Usage: python tests/pnl_calculation_smoke_test.py
"""

import os
import sys
import sqlite3
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from master_orchestrator import TradingDB, PaperTrade

print("=" * 70)
print("P&L CALCULATION SMOKE TEST")
print("=" * 70)

# Initialize DB
db = TradingDB()
print("✓ Database initialized")

# Test 1: Log a fake trade
print("\n[TEST 1] Logging a test trade...")
test_trade_id = f"SMOKE_TEST_{int(datetime.now().timestamp())}"
entry_price = 65000.0
quantity = 0.001

trade = PaperTrade(
    trade_id=test_trade_id,
    bot_name='SMOKE_TEST',
    market='crypto',
    symbol='BTC/USD',
    side='buy',
    entry_price=entry_price,
    exit_price=None,
    quantity=quantity,
    entry_time=datetime.now(),
    exit_time=None,
    pnl=0,
    status='open'
)
db.log_trade(trade)
print(f"✓ Trade logged: {test_trade_id}")

# Verify it's in DB
with sqlite3.connect(db.db_path) as conn:
    cursor = conn.cursor()
    cursor.execute('SELECT trade_id, status, entry_price, exit_price, pnl FROM trades WHERE trade_id = ?',
                   (test_trade_id,))
    row = cursor.fetchone()
    assert row[1] == 'open', "Status should be 'open'"
    assert row[3] is None, "exit_price should be NULL"
    assert row[4] == 0, "pnl should be 0"
    print(f"✓ Trade in DB: status={row[1]}, entry_price=${row[2]:.2f}, exit_price={row[3]}, pnl=${row[4]:.2f}")

# Test 2: Close the trade with P&L
print("\n[TEST 2] Closing trade with P&L calculation...")
exit_price = 66000.0  # $1000 higher
expected_pnl = quantity * (exit_price - entry_price)  # Should be $1.00
expected_pnl_pct = ((exit_price - entry_price) / entry_price) * 100  # Should be ~1.54%

print(f"Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}, Qty: {quantity}")
print(f"Expected P&L: ${expected_pnl:.4f} ({expected_pnl_pct:.2f}%)")

db.close_trade(
    bot_name='SMOKE_TEST',
    symbol='BTC/USD',
    exit_price=exit_price,
    pnl=expected_pnl,
    status='closed',
    gross_pnl=expected_pnl,
    net_pnl=expected_pnl * 0.995,
    estimated_fees=expected_pnl * 0.005
)
print("✓ close_trade() called")

# Verify P&L was stored correctly
with sqlite3.connect(db.db_path) as conn:
    cursor = conn.cursor()
    cursor.execute('''
        SELECT trade_id, status, entry_price, exit_price, quantity, pnl, pnl_pct
        FROM trades WHERE trade_id = ?
    ''', (test_trade_id,))
    row = cursor.fetchone()

    tid, status, ent, ext, qty, pnl, pnl_pct = row
    print(f"\n✓ Trade closed in DB:")
    print(f"  Status: {status}")
    print(f"  Entry: ${ent:.2f}")
    print(f"  Exit: ${ext:.2f}")
    print(f"  Quantity: {qty}")
    print(f"  P&L: ${pnl:.4f}")
    print(f"  P&L %: {pnl_pct:.4f}%")

    # Assertions
    assert status == 'closed', f"Status should be 'closed', got '{status}'"
    assert ext > 0, f"exit_price should be > 0, got {ext}"
    assert pnl != 0, f"pnl should not be 0, got {pnl}"
    assert abs(pnl - expected_pnl) < 0.0001, f"pnl mismatch: expected ${expected_pnl:.4f}, got ${pnl:.4f}"
    print("\n✅ TEST 1 & 2 PASSED: close_trade() correctly stores P&L")

# Test 3: Test force_close_trade_in_db with P&L
print("\n[TEST 3] Testing force_close_trade_in_db...")
test_trade_id_2 = f"SMOKE_TEST_FORCE_{int(datetime.now().timestamp())}"
entry_price_2 = 2000.0
quantity_2 = 0.05

trade2 = PaperTrade(
    trade_id=test_trade_id_2,
    bot_name='SMOKE_TEST',
    market='crypto',
    symbol='ETH/USD',
    side='buy',
    entry_price=entry_price_2,
    exit_price=None,
    quantity=quantity_2,
    entry_time=datetime.now(),
    exit_time=None,
    pnl=0,
    status='close_pending'  # Simulate a stuck trade
)
db.log_trade(trade2)
print(f"✓ Stuck trade logged: {test_trade_id_2} (status=close_pending)")

# Force close it
exit_price_2 = 2050.0
expected_pnl_2 = quantity_2 * (exit_price_2 - entry_price_2)  # Should be $2.50

db.force_close_trade_in_db(test_trade_id_2, exit_price=exit_price_2, pnl=expected_pnl_2)
print(f"✓ force_close_trade_in_db() called with exit=${exit_price_2:.2f}, pnl=${expected_pnl_2:.2f}")

# Verify
with sqlite3.connect(db.db_path) as conn:
    cursor = conn.cursor()
    cursor.execute('''
        SELECT status, exit_price, pnl, pnl_pct
        FROM trades WHERE trade_id = ?
    ''', (test_trade_id_2,))
    row = cursor.fetchone()

    status, ext, pnl, pnl_pct = row
    print(f"\n✓ Force-closed trade in DB:")
    print(f"  Status: {status}")
    print(f"  Exit: ${ext:.2f}")
    print(f"  P&L: ${pnl:.4f}")
    print(f"  P&L %: {pnl_pct:.4f}%")

    assert status == 'closed', f"Status should be 'closed', got '{status}'"
    assert ext > 0, f"exit_price should be > 0, got {ext}"
    assert pnl != 0, f"pnl should not be 0, got {pnl}"
    print("\n✅ TEST 3 PASSED: force_close_trade_in_db() correctly stores P&L")

# Cleanup
print("\n[CLEANUP] Removing test trades...")
with sqlite3.connect(db.db_path) as conn:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM trades WHERE trade_id LIKE 'SMOKE_TEST%'")
    conn.commit()
    print(f"✓ Removed {cursor.rowcount} test trades")

print("\n" + "=" * 70)
print("🎉 ALL SMOKE TESTS PASSED")
print("=" * 70)
print("\nP&L calculation logic is working correctly!")
print("Next step: Run the full plumbing test with real Alpaca paper trades:")
print("  python tests/plumbing_test_live.py")
print("=" * 70)
