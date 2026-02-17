"""
POSITION RECONCILIATION SCRIPT
Compares exchange balances (Coinbase) with trading_master.db open positions.
Flags discrepancies: positions on exchange but not in DB, and vice versa.
"""

import os
import sys
import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / '.env')

DB_PATH = str(BASE_DIR / 'data' / 'live' / 'trading_master.db')
REPORT_PATH = BASE_DIR / 'tools' / 'reconciliation_report.txt'

lines = []


def log(msg):
    print(msg)
    lines.append(msg)


def get_coinbase_balances():
    """Get all non-zero Coinbase balances"""
    try:
        from bots.coinbase_arb_bot import CoinbaseClient
        client = CoinbaseClient()
        if not client._initialized:
            log("  WARNING: Coinbase client not initialized")
            return {}

        accounts = client.get_accounts()
        balances = {}
        for acct in accounts:
            currency = getattr(acct, 'currency', None) or acct.get('currency', '')
            balance = getattr(acct, 'available_balance', None)
            if balance:
                val = float(getattr(balance, 'value', None) or balance.get('value', '0'))
            else:
                val = 0

            # Also check hold
            hold = getattr(acct, 'hold', None)
            if hold:
                hold_val = float(getattr(hold, 'value', None) or hold.get('value', '0'))
            else:
                hold_val = 0

            total = val + hold_val
            if total > 0 and currency not in ('USD',):  # Skip fiat USD
                balances[currency] = {
                    'available': val,
                    'hold': hold_val,
                    'total': total
                }

        # Get prices for valuation
        product_ids = [f"{c}-USDC" for c in balances.keys() if c != 'USDC']
        prices = {}
        if product_ids:
            try:
                pricebooks = client.get_best_bid_ask(product_ids[:20])
                for book in pricebooks:
                    pid = getattr(book, 'product_id', None) or book.get('product_id', '')
                    bids = book.bids if hasattr(book, 'bids') else book.get('bids', [])
                    if bids:
                        prices[pid] = float(bids[0].price if hasattr(bids[0], 'price') else bids[0].get('price', 0))
            except Exception:
                pass

        for currency, bal in balances.items():
            pid = f"{currency}-USDC"
            price = prices.get(pid, 1.0 if currency == 'USDC' else 0)
            bal['price'] = price
            bal['usd_value'] = bal['total'] * price

        return balances
    except Exception as e:
        log(f"  ERROR getting Coinbase balances: {e}")
        return {}


def get_db_open_positions():
    """Get all open positions from trading_master.db"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT trade_id, bot_name, symbol, side, entry_price, quantity, entry_time
            FROM trades WHERE status = 'open'
        """)
        columns = ['trade_id', 'bot_name', 'symbol', 'side', 'entry_price', 'quantity', 'entry_time']
        rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        log(f"  ERROR reading DB: {e}")
        return []


def reconcile():
    log("=" * 70)
    log(f"POSITION RECONCILIATION — {datetime.now().isoformat()}")
    log("=" * 70)

    # Get exchange positions
    log("\n--- COINBASE BALANCES ---")
    balances = get_coinbase_balances()

    if not balances:
        log("  No balances found (or Coinbase not connected)")
    else:
        total_usd = 0
        for currency, bal in sorted(balances.items()):
            usd = bal['usd_value']
            total_usd += usd
            if usd >= 0.01:  # Only show non-dust
                log(f"  {currency:8s}: {bal['total']:>14.8f}  (${usd:>10.2f})  price=${bal['price']:,.2f}")
        log(f"  {'TOTAL':8s}: {'':>14s}  (${total_usd:>10.2f})")

    # Get DB positions
    log("\n--- DATABASE OPEN POSITIONS ---")
    db_positions = get_db_open_positions()

    if not db_positions:
        log("  No open positions in database")
    else:
        for pos in db_positions:
            log(f"  {pos['trade_id'][:40]:40s} {pos['symbol']:14s} {pos['side']:5s} "
                f"entry=${pos['entry_price']:.4f} qty={pos['quantity']:.8f} bot={pos['bot_name']}")

    # Reconcile
    log("\n--- DISCREPANCIES ---")
    discrepancies = []

    # Build set of DB symbols (normalize: BTC-USDC -> BTC)
    db_symbols = {}
    for pos in db_positions:
        sym = pos['symbol'].replace('-USDC', '').replace('-USD', '')
        if sym not in db_symbols:
            db_symbols[sym] = []
        db_symbols[sym].append(pos)

    # Check for exchange positions not in DB
    for currency, bal in balances.items():
        if currency == 'USDC':
            continue  # Skip cash
        if bal['usd_value'] < 0.50:
            continue  # Skip dust

        if currency not in db_symbols:
            msg = f"  ON EXCHANGE BUT NOT IN DB: {currency} = {bal['total']:.8f} (${bal['usd_value']:.2f})"
            log(msg)
            discrepancies.append(msg)

    # Check for DB positions not on exchange
    for sym, positions in db_symbols.items():
        if sym not in balances or balances[sym]['total'] < 1e-10:
            for pos in positions:
                msg = f"  IN DB BUT NOT ON EXCHANGE: {pos['symbol']} qty={pos['quantity']:.8f} (bot={pos['bot_name']})"
                log(msg)
                discrepancies.append(msg)

    # Check quantity mismatches
    for sym, positions in db_symbols.items():
        if sym in balances:
            db_qty = sum(p['quantity'] for p in positions)
            exchange_qty = balances[sym]['total']
            pct_diff = abs(db_qty - exchange_qty) / max(db_qty, exchange_qty, 1e-10)
            if pct_diff > 0.05:  # >5% mismatch
                msg = (f"  QUANTITY MISMATCH: {sym} — "
                       f"DB: {db_qty:.8f}, Exchange: {exchange_qty:.8f}, Diff: {pct_diff:.1%}")
                log(msg)
                discrepancies.append(msg)

    if not discrepancies:
        log("  NONE — Database and exchange are in sync")

    # Summary
    log(f"\n--- SUMMARY ---")
    log(f"  Exchange assets: {len([b for b in balances.values() if b['usd_value'] >= 0.50])}")
    log(f"  DB open positions: {len(db_positions)}")
    log(f"  Discrepancies: {len(discrepancies)}")

    # Save report
    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(lines))
    log(f"\nReport saved to {REPORT_PATH}")


if __name__ == '__main__':
    reconcile()
