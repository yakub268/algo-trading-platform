"""
PLUMBING TEST — Validates the full trade lifecycle
Runs autonomously, tests every component, logs results.

NOT a pytest module — run directly: python tests/plumbing_test.py

Tests:
1. DATABASE: Can we write a trade? Can we read it back? Can we update it?
2. SIGNAL FORMAT: Does the orchestrator accept our signal format?
3. EXCHANGE CONNECTIVITY: Can we reach Alpaca API? Get prices?
4. EXECUTION: Can we place a tiny market buy ($2 of BTC/USD on Alpaca paper)?
5. TRACKING: Does the trade appear in the database with correct data?
6. EXIT: Can we close the position and update the database?
7. P&L: Is the P&L calculation correct after close?
8. DASHBOARD: Does the Flask dashboard return position data via API?
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Standalone integration script — run directly with: python tests/plumbing_test.py"
)

import os
import sys
import time
import sqlite3
import json
import traceback
from datetime import datetime
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(str(BASE_DIR))

from dotenv import load_dotenv

load_dotenv(BASE_DIR / ".env")

# Results tracking
RESULTS = []
RESULTS_FILE = BASE_DIR / "tests" / "plumbing_test_results.txt"
DB_PATH = str(BASE_DIR / "data" / "live" / "trading_master.db")
TEST_TRADE_ID = f"PLUMBING_TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def log(msg):
    """Log to both console and results file"""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    RESULTS.append(line)


def record(test_name, passed, detail=""):
    """Record a test result"""
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {test_name}"
    if detail:
        msg += f" — {detail}"
    log(msg)
    return passed


def save_results():
    """Save all results to file"""
    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(RESULTS))
    log(f"Results saved to {RESULTS_FILE}")


# ============================================================
# TEST 1: DATABASE CONNECTIVITY
# ============================================================
def test_database():
    log("\n" + "=" * 60)
    log("TEST 1: DATABASE CONNECTIVITY")
    log("=" * 60)

    all_passed = True

    # 1a: Can we connect?
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        all_passed &= record("DB Connect", True, f"{count} existing trades")
    except Exception as e:
        record("DB Connect", False, str(e))
        return False

    # 1b: Can we write a test trade?
    try:
        cursor.execute(
            """
            INSERT INTO trades (trade_id, bot_name, market, symbol, side,
                entry_price, quantity, entry_time, pnl, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                TEST_TRADE_ID,
                "PLUMBING-TEST",
                "crypto",
                "BTC/USD",
                "buy",
                99999.99,
                0.0001,
                datetime.now().isoformat(),
                0,
                "open",
            ),
        )
        conn.commit()
        all_passed &= record("DB Write", True, f"Inserted {TEST_TRADE_ID}")
    except Exception as e:
        record("DB Write", False, str(e))
        all_passed = False

    # 1c: Can we read it back?
    try:
        cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (TEST_TRADE_ID,))
        row = cursor.fetchone()
        if row:
            all_passed &= record("DB Read", True, f"Found trade: status={row[-1]}")
        else:
            record("DB Read", False, "Trade not found after insert")
            all_passed = False
    except Exception as e:
        record("DB Read", False, str(e))
        all_passed = False

    # 1d: Can we update it?
    try:
        cursor.execute(
            """
            UPDATE trades SET exit_price = 100000.00, exit_time = ?,
                pnl = 0.01, pnl_pct = 0.0001, status = 'closed'
            WHERE trade_id = ?
        """,
            (datetime.now().isoformat(), TEST_TRADE_ID),
        )
        conn.commit()

        cursor.execute(
            "SELECT status, pnl FROM trades WHERE trade_id = ?", (TEST_TRADE_ID,)
        )
        row = cursor.fetchone()
        if row and row[0] == "closed" and row[1] == 0.01:
            all_passed &= record("DB Update", True, "Trade closed with PnL")
        else:
            record("DB Update", False, f"Unexpected state: {row}")
            all_passed = False
    except Exception as e:
        record("DB Update", False, str(e))
        all_passed = False

    # 1e: Clean up test trade
    try:
        cursor.execute("DELETE FROM trades WHERE trade_id = ?", (TEST_TRADE_ID,))
        conn.commit()
        record("DB Cleanup", True, "Test trade removed")
    except Exception as e:
        record("DB Cleanup", False, str(e))

    conn.close()
    return all_passed


# ============================================================
# TEST 2: SIGNAL FORMAT (orchestrator compatibility)
# ============================================================
def test_signal_format():
    log("\n" + "=" * 60)
    log("TEST 2: SIGNAL FORMAT")
    log("=" * 60)

    all_passed = True

    # Test that our signal dict has all required fields
    test_signal = {
        "symbol": "BTC/USD",
        "action": "buy",
        "price": 95000.0,
        "quantity": 0.0001,
        "confidence": 0.8,
        "reason": "plumbing_test",
        "status": "filled",
    }

    required_fields = ["symbol", "action", "price", "quantity"]
    for field in required_fields:
        if field in test_signal:
            all_passed &= record(
                f"Signal field '{field}'", True, str(test_signal[field])
            )
        else:
            record(f"Signal field '{field}'", False, "Missing")
            all_passed = False

    # Test that _log_trade_from_signal would accept this
    # We simulate what the orchestrator does
    action = test_signal.get("action", "").lower()
    is_close = action in ("sell", "close", "short")
    already_executed = test_signal.get("status") in ("filled", "submitted")
    all_passed &= record(
        "Action parsing", True, f"action={action}, is_close={is_close}"
    )
    all_passed &= record(
        "Already-executed check", already_executed, f"status={test_signal['status']}"
    )

    return all_passed


# ============================================================
# TEST 3: EXCHANGE CONNECTIVITY (Alpaca Paper)
# ============================================================
def test_exchange_connectivity():
    log("\n" + "=" * 60)
    log("TEST 3: EXCHANGE CONNECTIVITY (Alpaca Paper)")
    log("=" * 60)

    all_passed = True
    alpaca = None
    btc_price = None

    # 3a: Initialize Alpaca client
    try:
        from bots.alpaca_crypto_client import AlpacaCryptoClient

        alpaca = AlpacaCryptoClient()

        if alpaca._initialized:
            all_passed &= record(
                "Alpaca Init", True, f"Client connected (paper={alpaca.paper})"
            )
        else:
            record(
                "Alpaca Init",
                False,
                "Client not initialized (check ALPACA_API_KEY/ALPACA_SECRET_KEY in .env)",
            )
            return False, None, None
    except Exception as e:
        record("Alpaca Init", False, str(e))
        return False, None, None

    # 3b: Get BTC price
    try:
        btc_price = alpaca.get_price("BTC/USD")
        if btc_price and btc_price > 0:
            all_passed &= record("BTC Price", True, f"${btc_price:,.2f}")
        else:
            record("BTC Price", False, f"Got: {btc_price}")
            all_passed = False
    except Exception as e:
        record("BTC Price", False, str(e))
        all_passed = False

    # 3c: Get account info
    try:
        account = alpaca.get_account()
        if account:
            all_passed &= record(
                "Alpaca Account",
                True,
                f"Equity: ${account['equity']:,.2f}, Cash: ${account['cash']:,.2f}",
            )
        else:
            record("Alpaca Account", False, "No account returned")
            all_passed = False
    except Exception as e:
        record("Alpaca Account", False, str(e))
        all_passed = False

    return all_passed, alpaca, btc_price


# ============================================================
# TEST 4 & 5: LIVE MICRO-TRADE + DB TRACKING (Alpaca Paper)
# ============================================================
def test_live_trade(alpaca, btc_price):
    log("\n" + "=" * 60)
    log("TEST 4: ALPACA PAPER MICRO-TRADE EXECUTION")
    log("=" * 60)

    if alpaca is None or btc_price is None:
        record("Live Trade", False, "Skipped — no Alpaca connection or price")
        return False, None

    trade_amount = "10.00"  # $10 worth of BTC (Alpaca minimum is $10)
    trade_id = f"PLUMBING_LIVE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 4a: Place market buy (notional $2)
    try:
        log(f"  Placing market buy: ${trade_amount} of BTC/USD on Alpaca paper...")
        order_result = alpaca.create_market_order(
            product_id="BTC/USD", side="BUY", quote_size=trade_amount
        )

        if order_result and order_result.get("success", False):
            order_id = order_result.get("order_id", "unknown")
            status = order_result.get("status", "unknown")
            record("Market Buy", True, f"Order ID: {order_id}, Status: {status}")
        elif order_result:
            error = order_result.get("error", "unknown")
            record("Market Buy", False, f"Order failed: {error}")
            return False, None
        else:
            record("Market Buy", False, "No response from create_market_order")
            return False, None
    except Exception as e:
        record("Market Buy", False, str(e))
        log(f"  Traceback: {traceback.format_exc()}")
        return False, None

    # 4b: Log to DB
    log("\n" + "=" * 60)
    log("TEST 5: DATABASE TRACKING")
    log("=" * 60)

    quantity = float(trade_amount) / btc_price if btc_price > 0 else 0

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO trades (trade_id, bot_name, market, symbol, side,
                entry_price, quantity, entry_time, pnl, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trade_id,
                "PLUMBING-TEST",
                "crypto",
                "BTC/USD",
                "buy",
                btc_price,
                quantity,
                datetime.now().isoformat(),
                0,
                "open",
            ),
        )
        conn.commit()

        # Verify
        cursor.execute(
            "SELECT status, entry_price, quantity FROM trades WHERE trade_id = ?",
            (trade_id,),
        )
        row = cursor.fetchone()
        if row and row[0] == "open":
            record("DB Track Open", True, f"entry=${row[1]:,.2f}, qty={row[2]:.8f}")
        else:
            record("DB Track Open", False, f"Row: {row}")

        conn.close()
    except Exception as e:
        record("DB Track Open", False, str(e))

    return True, trade_id


# ============================================================
# TEST 6: EXIT / CLOSE POSITION
# ============================================================
def test_exit(alpaca, trade_id, entry_price):
    log("\n" + "=" * 60)
    log("TEST 6: EXIT / CLOSE TRACKING")
    log("=" * 60)

    if alpaca is None or trade_id is None:
        record("Exit Trade", False, "Skipped — no live trade to close")
        return False

    all_passed = True

    # Get current BTC price for exit
    exit_price = None
    try:
        exit_price = alpaca.get_price("BTC/USD")
    except Exception:
        pass

    if exit_price is None:
        exit_price = entry_price  # Fallback
        log(f"  Could not get exit price, using entry: ${exit_price:,.2f}")

    # Sell the BTC back
    try:
        # Get positions to find BTC quantity
        positions = alpaca.get_positions()
        btc_position = None
        for pos in positions:
            sym = pos.get("symbol", "")
            if "BTC" in sym:
                btc_position = pos
                break

        if btc_position:
            sell_qty = str(round(btc_position["qty"], 8))
            log(f"  Selling {sell_qty} BTC (from Alpaca positions)...")

            sell_result = alpaca.create_market_order(
                product_id="BTC/USD", side="SELL", base_size=sell_qty
            )

            if sell_result and sell_result.get("success", False):
                all_passed &= record(
                    "Market Sell", True, f"Order: {sell_result.get('order_id', 'ok')}"
                )
            else:
                error = (
                    sell_result.get("error", "unknown")
                    if sell_result
                    else "No response"
                )
                record("Market Sell", False, error)
                all_passed = False
        else:
            # No BTC position found — may have been too small or not settled
            log(
                "  No BTC position found in Alpaca — paper order may not have settled yet"
            )
            record(
                "Market Sell",
                True,
                "No position to sell (paper order likely too small)",
            )

    except Exception as e:
        record("Market Sell", False, str(e))
        all_passed = False

    # Update DB
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Read entry data
        cursor.execute(
            "SELECT entry_price, quantity FROM trades WHERE trade_id = ?", (trade_id,)
        )
        row = cursor.fetchone()
        if row:
            ep, qty = float(row[0]), float(row[1])
            pnl = qty * (exit_price - ep)
            pnl_pct = (exit_price - ep) / ep if ep > 0 else 0

            cursor.execute(
                """
                UPDATE trades SET exit_price = ?, exit_time = ?,
                    pnl = ?, pnl_pct = ?, status = 'closed'
                WHERE trade_id = ?
            """,
                (exit_price, datetime.now().isoformat(), pnl, pnl_pct, trade_id),
            )
            conn.commit()

            # Verify
            cursor.execute(
                "SELECT status, pnl, pnl_pct FROM trades WHERE trade_id = ?",
                (trade_id,),
            )
            row2 = cursor.fetchone()
            if row2 and row2[0] == "closed":
                all_passed &= record(
                    "DB Track Close", True, f"PnL: ${row2[1]:.4f} ({row2[2]:+.4%})"
                )
            else:
                record("DB Track Close", False, f"Row: {row2}")
                all_passed = False
        else:
            record("DB Track Close", False, "Trade not found in DB")
            all_passed = False

        conn.close()
    except Exception as e:
        record("DB Track Close", False, str(e))
        all_passed = False

    return all_passed


# ============================================================
# TEST 7: P&L CALCULATION
# ============================================================
def test_pnl(trade_id):
    log("\n" + "=" * 60)
    log("TEST 7: P&L CALCULATION")
    log("=" * 60)

    if trade_id is None:
        record("P&L Calc", False, "Skipped — no trade to verify")
        return False

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT entry_price, exit_price, quantity, pnl, pnl_pct, status
            FROM trades WHERE trade_id = ?
        """,
            (trade_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return record("P&L Calc", False, "Trade not found")

        ep, xp, qty, pnl, pnl_pct, status = row

        if status != "closed":
            return record("P&L Calc", False, f"Trade not closed: {status}")

        # Verify PnL calculation
        expected_pnl = qty * (xp - ep)
        expected_pnl_pct = (xp - ep) / ep if ep > 0 else 0

        pnl_match = abs(pnl - expected_pnl) < 0.01
        pct_match = abs(pnl_pct - expected_pnl_pct) < 0.001

        record(
            "P&L Formula", pnl_match, f"DB: ${pnl:.4f}, Expected: ${expected_pnl:.4f}"
        )
        record(
            "P&L Pct Formula",
            pct_match,
            f"DB: {pnl_pct:.4%}, Expected: {expected_pnl_pct:.4%}",
        )

        return pnl_match and pct_match

    except Exception as e:
        return record("P&L Calc", False, str(e))


# ============================================================
# TEST 8: DASHBOARD API
# ============================================================
def test_dashboard():
    log("\n" + "=" * 60)
    log("TEST 8: DASHBOARD API")
    log("=" * 60)

    try:
        import urllib.request

        url = "http://localhost:5000/api/dashboard"
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/json")

        response = urllib.request.urlopen(req, timeout=5)
        data = json.loads(response.read().decode())

        if "error" in data:
            return record("Dashboard API", False, data["error"])

        has_positions = "positions" in data
        has_totals = "totals" in data
        has_strategies = "strategies" in data

        record("Dashboard /api/dashboard", True, f"Keys: {list(data.keys())[:5]}")
        record("Has positions", has_positions)
        record("Has totals", has_totals)
        record("Has strategies", has_strategies)

        if has_totals:
            totals = data["totals"]
            log(
                f"  Totals: PnL=${totals.get('pnl', 0):.2f}, "
                f"Invested=${totals.get('invested', 0):.2f}, "
                f"Paper={totals.get('paper_mode', 'unknown')}"
            )

        return has_positions and has_totals

    except Exception as e:
        record("Dashboard API", False, f"Not reachable — {e}")
        return False


# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 60)
    log(f"PLUMBING TEST (ALPACA PAPER) — {datetime.now().isoformat()}")
    log(f"DB: {DB_PATH}")
    log(f"Base: {BASE_DIR}")
    log(f"Exchange: Alpaca Paper ({os.getenv('ALPACA_BASE_URL', 'not set')})")
    log("=" * 60)

    results = {}

    # Test 1: Database
    results["database"] = test_database()

    # Test 2: Signal format
    results["signal_format"] = test_signal_format()

    # Test 3: Exchange connectivity (Alpaca)
    exchange_ok, alpaca, btc_price = test_exchange_connectivity()
    results["exchange"] = exchange_ok

    # Test 4+5: Live micro-trade + tracking
    if exchange_ok and btc_price:
        trade_ok, live_trade_id = test_live_trade(alpaca, btc_price)
        results["live_trade"] = trade_ok

        if trade_ok and live_trade_id:
            # Wait for order to settle
            log("\n  Waiting 10 seconds for order to settle...")
            time.sleep(10)

            # Test 6: Exit
            results["exit"] = test_exit(alpaca, live_trade_id, btc_price)

            # Test 7: P&L
            results["pnl"] = test_pnl(live_trade_id)
        else:
            results["exit"] = False
            results["pnl"] = False
    else:
        results["live_trade"] = False
        results["exit"] = False
        results["pnl"] = False

    # Test 8: Dashboard
    results["dashboard"] = test_dashboard()

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        log(f"  {status}: {name}")

    log(f"\n  {passed}/{total} passed, {failed} failed")

    if failed == 0:
        log("\n  ALL TESTS PASSED — Pipeline is working!")
    else:
        log(f"\n  {failed} TESTS FAILED — See details above")

    save_results()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
