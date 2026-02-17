"""
Validation script for the 13-bug fix across 6 files.
Run: python tests/validate_fixes.py
"""
import os
import sys
import sqlite3
import inspect
import traceback
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  -- {detail}")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
#  1. MOCK DATA GUARDS (BUG 1)
# ============================================================
section("1. Mock Data Guards (BUG 1)")

from bots.aggressive.rsi_extremes import RSIExtremesBot
from bots.aggressive.multi_momentum import MultiMomentumBot

rsi_src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'bots', 'aggressive', 'rsi_extremes.py')

# RSI: live mode + uninit Alpaca -> empty
bot_rsi = RSIExtremesBot(paper_mode=False)
bot_rsi.alpaca._initialized = False
candles = bot_rsi._get_candles('BTC/USD')
test("RSI live+uninit returns empty", candles == [], f"got {len(candles)} candles")

# RSI: paper mode mock generator still works (tested directly since base class
# path short-circuits to [] when uninit, but the mock fallback path guards correctly)
bot_rsi_paper = RSIExtremesBot(paper_mode=True)
mock_candles = bot_rsi_paper._generate_mock_candles('BTC/USD')
test("RSI paper mock generator still works", len(mock_candles) > 0, f"got {len(mock_candles)}")

# Verify: the guard in _get_candles fallback path allows paper, blocks live
# Read source to confirm the conditional
with open(rsi_src_path, 'r') as f:
    rsi_src_check = f.read()
test("RSI _get_candles has paper_mode guard",
     'if not self.paper_mode:' in rsi_src_check and 'refusing to use mock data' in rsi_src_check)

# Multi-Mom: live + uninit -> empty prices
bot_mm = MultiMomentumBot(paper_mode=False)
bot_mm.alpaca._initialized = False
prices = bot_mm.get_prices(['BTC/USD', 'ETH/USD'])
test("MM live+uninit get_prices returns empty", prices == {}, f"got {len(prices)} prices")

# Multi-Mom: live + uninit -> None candle
candle = bot_mm._get_24h_candle('BTC/USD')
test("MM live+uninit _get_24h_candle returns None", candle is None, f"got {candle}")

# Multi-Mom: paper + uninit -> mock prices still work
bot_mm_paper = MultiMomentumBot(paper_mode=True)
bot_mm_paper.alpaca._initialized = False
prices_paper = bot_mm_paper.get_prices(['BTC/USD'])
test("MM paper+uninit get_prices returns mock data", len(prices_paper) > 0, f"got {len(prices_paper)}")

candle_paper = bot_mm_paper._get_24h_candle('BTC/USD')
test("MM paper+uninit _get_24h_candle returns mock data", candle_paper is not None)


# ============================================================
#  2. POSITION SIZE GUARDS (BUG 2)
# ============================================================
section("2. Position Size Guards (BUG 2)")

from bots.aggressive.base_aggressive_bot import BaseAggressiveBot, TradeSignal

class TestBot(BaseAggressiveBot):
    def run_scan(self): return []

# Test: signal size > capital -> rejected
tbot = TestBot(capital=200, paper_mode=True)
oversized_signal = TradeSignal(
    symbol='BTC/USD', side='BUY', entry_price=95000, target_price=100000,
    stop_loss=90000, position_size_usd=500,  # $500 > $200 capital
    confidence=0.8, reason='test', timestamp=datetime.now(timezone.utc)
)
result = tbot.execute_signal(oversized_signal)
test("BaseBot rejects signal > capital", result is None, f"got {result}")

# Test: signal size within capital -> accepted
ok_signal = TradeSignal(
    symbol='BTC/USD', side='BUY', entry_price=95000, target_price=100000,
    stop_loss=90000, position_size_usd=50,  # $50 < $200 capital
    confidence=0.8, reason='test', timestamp=datetime.now(timezone.utc)
)
result = tbot.execute_signal(ok_signal)
test("BaseBot accepts signal within capital", result is not None and result.get('status') == 'filled')

# Test: orchestrator notional guard (check source for the pattern)
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'master_orchestrator.py'), 'r', encoding='utf-8') as f:
    orch_src = f.read()

test("Orchestrator has NOTIONAL GUARD", 'NOTIONAL GUARD' in orch_src)
test("Orchestrator checks qty * price > 50% capital",
     'notional > max_notional' in orch_src or 'self.starting_capital * 0.50' in orch_src)


# ============================================================
#  3. RSI WILDER'S SMOOTHING (BUG 5)
# ============================================================
section("3. RSI Wilder's Smoothing (BUG 5)")

bot = RSIExtremesBot(paper_mode=True)

# Test: all gains -> RSI 100
candles_up = [{'close': 100 + i} for i in range(30)]
test("All gains -> RSI=100", bot._calculate_rsi(candles_up) == 100.0)

# Test: all losses -> RSI 0
candles_down = [{'close': 100 - i * 0.5} for i in range(30)]
test("All losses -> RSI=0", bot._calculate_rsi(candles_down) == 0.0)

# Test: equal gains/losses -> RSI ~50
candles_flat = []
p = 100
for i in range(60):
    candles_flat.append({'close': p})
    p += 1 if i % 2 == 0 else -1
rsi_flat = bot._calculate_rsi(candles_flat)
test("Equal up/down -> RSI near 50", 45 <= rsi_flat <= 55, f"got {rsi_flat:.1f}")

# Test: Wilder's uses ALL history (not just last 14)
# Build sequence: 30 bars up, then 30 bars down. Wilder's RSI should be
# lower than Cutler's RSI (which only uses last 14 down changes)
candles_trend = [{'close': 100 + i * 2} for i in range(30)]
candles_trend += [{'close': candles_trend[-1]['close'] - i * 1.5} for i in range(1, 31)]
rsi_wilder = bot._calculate_rsi(candles_trend)
# Cutler's would be: last 14 changes are all negative -> RSI=0
# Wilder's remembers the earlier gains -> RSI should be > 0
test("Wilder's remembers history (RSI > 0 after reversal)", rsi_wilder > 0, f"got {rsi_wilder:.1f}")
test("Wilder's not stuck at 0 like Cutler's would be", rsi_wilder > 5, f"got {rsi_wilder:.1f}")

# Test: compare against pandas ewm-based reference
try:
    import pandas as pd
    closes = pd.Series([c['close'] for c in candles_trend])
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain_ref = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss_ref = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs_ref = avg_gain_ref / avg_loss_ref
    rsi_ref = 100 - (100 / (1 + rs_ref))
    rsi_ref_val = rsi_ref.iloc[-1]
    diff = abs(rsi_wilder - rsi_ref_val)
    test(f"Matches pandas reference (diff={diff:.2f})", diff < 2.0,
         f"ours={rsi_wilder:.2f}, pandas={rsi_ref_val:.2f}, diff={diff:.2f}")
except ImportError:
    test("Pandas reference (skipped - pandas not available)", True)

# Test: insufficient data returns None
test("Insufficient data returns None", bot._calculate_rsi([{'close': 100}] * 10) is None)


# ============================================================
#  4. RSI EXIT POSITION REFERENCE (BUG 6)
# ============================================================
section("4. RSI Exit Position Reference (BUG 6)")

# Read the source and verify the fix is in place
with open(rsi_src_path, 'r') as f:
    rsi_src = f.read()

# The fix: position is captured BEFORE _execute_exit() is called
# Old code: trade = self._execute_exit(signal) ... position = self.rsi_positions.get(...)
# New code: position = self.rsi_positions.get(...) ... trade = self._execute_exit(signal)
lines = rsi_src.split('\n')
capture_line = None
execute_line = None
for i, line in enumerate(lines):
    if 'Capture position BEFORE' in line:
        capture_line = i
    if '_execute_exit(signal)' in line and 'def' not in line:
        if capture_line and execute_line is None:
            execute_line = i

test("Position captured BEFORE _execute_exit",
     capture_line is not None and execute_line is not None and capture_line < execute_line,
     f"capture@{capture_line}, execute@{execute_line}")

# Functional test: create a position, generate exit, verify quantity preserved
from bots.aggressive.rsi_extremes import RSIPosition, RSISignal
bot_exit = RSIExtremesBot(paper_mode=True)
bot_exit.rsi_positions['SOL/USD'] = RSIPosition(
    product_id='SOL/USD', symbol='SOL', entry_price=150.0,
    quantity=0.4, entry_time=datetime.now(timezone.utc) - timedelta(hours=3),
    stop_loss=142.5, take_profit=165.0, rsi_at_entry=52.0
)

# Simulate an exit signal
exit_signal = RSISignal(
    symbol='SOL', product_id='SOL/USD', current_price=142.0,
    rsi_value=35.0, signal_type='SELL', entry_price=150.0,
    stop_loss=142.5, take_profit=165.0, confidence=1.0,
    reason='Stop loss', timestamp=datetime.now(timezone.utc)
)

# Run the exit through run_scan's logic manually
position = bot_exit.rsi_positions.get('SOL/USD')
quantity_before = position.quantity if position else None
trade = bot_exit._execute_exit(exit_signal)
test("Exit trade uses correct quantity",
     trade is not None and trade.get('quantity') == 0.4,
     f"expected 0.4, got {trade.get('quantity') if trade else 'None'}")
test("Position removed after exit", 'SOL/USD' not in bot_exit.rsi_positions)


# ============================================================
#  5. MULTI-MOMENTUM STOP-LOSS (BUG 3)
# ============================================================
section("5. Multi-Momentum Stop-Loss (BUG 3)")

from bots.aggressive.multi_momentum import MultiMomentumBot, MomentumHolding

# Verify STOP_LOSS_PCT exists
test("STOP_LOSS_PCT defined", hasattr(MultiMomentumBot, 'STOP_LOSS_PCT'))
test("STOP_LOSS_PCT = 0.05", MultiMomentumBot.STOP_LOSS_PCT == 0.05)

# Verify MIN_VOLUME_USD raised
test("MIN_VOLUME_USD = 100000", MultiMomentumBot.MIN_VOLUME_USD == 100000)

# Functional test: holdings with >5% loss should trigger stop-loss exit
bot_mm_sl = MultiMomentumBot(paper_mode=True)
bot_mm_sl.holdings['TEST/USD'] = MomentumHolding(
    product_id='TEST/USD', symbol='TEST',
    quantity=1.0, entry_price=100.0,
    entry_time=datetime.now(timezone.utc) - timedelta(hours=2),
    entry_change_pct=0.05,
    current_price=100.0, current_change_pct=0.0
)

# Simulate price drop to 94 (6% loss > 5% SL threshold)
# Override get_prices to return the dropped price
original_get_prices = bot_mm_sl.get_prices
def mock_get_prices(product_ids):
    return {'TEST/USD': {'bid': 94.0, 'ask': 94.1, 'mid': 94.05}}
bot_mm_sl.get_prices = mock_get_prices

# Override _get_24h_candle to avoid API calls
original_candle = bot_mm_sl._get_24h_candle
bot_mm_sl._get_24h_candle = lambda pid: {'open': 95.0, 'close': 94.05, 'high': 96, 'low': 93, 'volume': 1000000}

stop_exits = bot_mm_sl._update_holdings_prices()
test("Stop-loss triggered for -6% position",
     len(stop_exits) > 0,
     f"got {len(stop_exits)} exits, holdings left: {len(bot_mm_sl.holdings)}")
test("Position removed after SL exit", 'TEST/USD' not in bot_mm_sl.holdings)

# Restore
bot_mm_sl.get_prices = original_get_prices
bot_mm_sl._get_24h_candle = original_candle


# ============================================================
#  6. MULTI-MOMENTUM UTC TIMESTAMPS (BUG 12)
# ============================================================
section("6. Multi-Momentum UTC Timestamps (BUG 12)")

mm_src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'bots', 'aggressive', 'multi_momentum.py')
with open(mm_src_path, 'r') as f:
    mm_src = f.read()

# Count datetime.now() vs datetime.now(timezone.utc) in _log_trade_to_central_db
in_db_method = False
local_nows = 0
utc_nows = 0
for line in mm_src.split('\n'):
    if '_log_trade_to_central_db' in line and 'def ' in line:
        in_db_method = True
    elif in_db_method and line.strip().startswith('def '):
        break
    if in_db_method:
        if 'datetime.now()' in line and 'timezone.utc' not in line:
            local_nows += 1
        if 'datetime.now(timezone.utc)' in line:
            utc_nows += 1

test("No local datetime.now() in _log_trade_to_central_db", local_nows == 0, f"found {local_nows}")
test("Uses datetime.now(timezone.utc) in DB writes", utc_nows >= 2, f"found {utc_nows}")


# ============================================================
#  7. DUAL DB WRITES REMOVED (BUG 7)
# ============================================================
section("7. Dual DB Writes Removed (BUG 7)")

# Check that _log_trade_to_central_db is NOT called from _execute_buy or _execute_sell
# Check that the actual function CALL is removed (not just mentioned in comments)
buy_method = inspect.getsource(MultiMomentumBot._execute_buy)
sell_method = inspect.getsource(MultiMomentumBot._execute_sell)
# Filter out comment lines — only check executable lines
buy_calls = [l for l in buy_method.split('\n')
             if '_log_trade_to_central_db' in l and not l.strip().startswith('#')]
sell_calls = [l for l in sell_method.split('\n')
              if '_log_trade_to_central_db' in l and not l.strip().startswith('#')]
test("_execute_buy does NOT call _log_trade_to_central_db", len(buy_calls) == 0,
     f"found calls: {buy_calls}")
test("_execute_sell does NOT call _log_trade_to_central_db", len(sell_calls) == 0,
     f"found calls: {sell_calls}")

# Verify the method still exists (not deleted, just unused by execution flow)
test("_log_trade_to_central_db method still exists",
     hasattr(MultiMomentumBot, '_log_trade_to_central_db'))


# ============================================================
#  8. ACM MACD 3-BAR DECLINE (BUG 9)
# ============================================================
section("8. ACM MACD 3-Bar Decline (BUG 9)")

from bots.alpaca_crypto_momentum import AlpacaCryptoMomentumBot, Position, Regime

acm = AlpacaCryptoMomentumBot(paper_mode=True)

test("histogram_decline_count dict exists", hasattr(acm, 'histogram_decline_count'))
test("MACD_DECLINE_BARS_REQUIRED = 3", acm.MACD_DECLINE_BARS_REQUIRED == 3)

# Simulate: position in 2%+ profit, single declining bar -> should NOT exit
acm.positions['BTC/USD'] = Position(
    symbol='BTC/USD', side='long', entry_price=95000, quantity=0.001,
    entry_time=datetime.now(timezone.utc), stop_loss=91200,
    ema_fast_at_entry=95100, ema_slow_at_entry=94900,
    regime_at_entry=Regime.BULL
)

# One declining bar
result1 = acm._check_exit(
    'BTC/USD', price=97000,  # 2.1% profit
    current_ema_fast=97100, current_ema_slow=96900,
    prev_ema_fast=97200, prev_ema_slow=96800,
    histogram_slope=-0.001, volume_ratio=1.0, entropy=0.5, regime=Regime.BULL
)
test("1 declining bar -> no exit", result1 is None)
test("Decline counter = 1", acm.histogram_decline_count.get('BTC/USD', 0) == 1)

# Second declining bar
result2 = acm._check_exit(
    'BTC/USD', price=97000,
    current_ema_fast=97050, current_ema_slow=96950,
    prev_ema_fast=97100, prev_ema_slow=96900,
    histogram_slope=-0.002, volume_ratio=1.0, entropy=0.5, regime=Regime.BULL
)
test("2 declining bars -> no exit", result2 is None)
test("Decline counter = 2", acm.histogram_decline_count.get('BTC/USD', 0) == 2)

# Third declining bar -> should exit
result3 = acm._check_exit(
    'BTC/USD', price=97000,
    current_ema_fast=97000, current_ema_slow=97000,
    prev_ema_fast=97050, prev_ema_slow=96950,
    histogram_slope=-0.003, volume_ratio=1.0, entropy=0.5, regime=Regime.BULL
)
test("3 declining bars -> EXIT", result3 is not None and result3.signal_type.value == 'sell',
     f"got {result3}")
test("Decline counter reset to 0", acm.histogram_decline_count.get('BTC/USD', 0) == 0)

# Verify: positive slope resets counter
acm.histogram_decline_count['ETH/USD'] = 2
acm.positions['ETH/USD'] = Position(
    symbol='ETH/USD', side='long', entry_price=3000, quantity=0.01,
    entry_time=datetime.now(timezone.utc), stop_loss=2880,
    ema_fast_at_entry=3010, ema_slow_at_entry=2990,
    regime_at_entry=Regime.BULL
)
result_reset = acm._check_exit(
    'ETH/USD', price=3100,  # in profit
    current_ema_fast=3110, current_ema_slow=3090,
    prev_ema_fast=3100, prev_ema_slow=3080,
    histogram_slope=0.005, volume_ratio=1.0, entropy=0.5, regime=Regime.BULL
)
test("Positive slope resets decline counter",
     acm.histogram_decline_count.get('ETH/USD', 0) == 0)


# ============================================================
#  9. ACM LIVE EXECUTION PATH (BUG 8)
# ============================================================
section("9. ACM Live Execution Path (BUG 8)")

acm_src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'bots', 'alpaca_crypto_momentum.py')
with open(acm_src_path, 'r') as f:
    acm_src = f.read()

# Old code used: self.trading_client.submit_order(order_request) with MarketOrderRequest
# New code uses: alpaca_client.create_market_order()
test("Live path uses AlpacaCryptoClient",
     'alpaca_client.create_market_order' in acm_src or 'AlpacaCryptoClient' in acm_src)
test("Live path no longer uses submit_order directly",
     'self.trading_client.submit_order(order_request)' not in acm_src)
# Find the live execution block and check for position tracking
live_block_start = acm_src.find('Live trading via Alpaca')
if live_block_start > 0:
    live_block = acm_src[live_block_start:live_block_start+3000]
    has_pos_tracking = 'self.positions[signal.symbol] = Position(' in live_block
    has_pnl_tracking = 'self.pnl_today += pnl' in live_block
    test("Live path tracks positions on fill", has_pos_tracking,
         "Position(...) assignment missing in live block")
    test("Live path tracks P&L on close", has_pnl_tracking,
         "pnl_today update missing in live block")
else:
    test("Live execution block found", False, "Could not find 'Live trading via Alpaca'")


# ============================================================
# 10. ORCHESTRATOR EXIT THRESHOLDS (BUG 4)
# ============================================================
section("10. Orchestrator Exit Thresholds (BUG 4)")

test("TP threshold = 15%", 'tp_threshold = 0.15' in orch_src)
test("SL threshold = -10%", 'sl_threshold = -0.10' in orch_src)
test("Old 8% TP removed", 'tp_threshold = 0.08' not in orch_src)
test("Old -7% SL removed", 'sl_threshold = -0.07' not in orch_src)
test("Old -3% momentum SL removed", "sl_threshold = -0.03" not in orch_src)
test("Safety net comment present", 'safety net' in orch_src.lower() or 'SAFETY NET' in orch_src)


# ============================================================
# 11. DB CLEANUP VERIFICATION
# ============================================================
section("11. DB Cleanup Verification")

db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'live', 'trading_master.db')
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Phantom trades archived
    c.execute("SELECT COUNT(*) FROM trades WHERE trade_id IN ('test_tp','test_sl','test_time') AND status = 'archived_phantom'")
    phantom_archived = c.fetchone()[0]
    test("3 phantom trades archived", phantom_archived == 3, f"found {phantom_archived}")

    # No phantom trades still open/closed
    c.execute("SELECT COUNT(*) FROM trades WHERE trade_id IN ('test_tp','test_sl','test_time') AND status IN ('open','closed')")
    phantom_active = c.fetchone()[0]
    test("No phantom trades still active", phantom_active == 0, f"found {phantom_active}")

    # ML-Prediction-Bot open positions archived
    c.execute("SELECT COUNT(*) FROM trades WHERE bot_name = 'ML-Prediction-Bot' AND status = 'open'")
    ml_open = c.fetchone()[0]
    test("ML-Prediction-Bot 0 open positions", ml_open == 0, f"found {ml_open}")

    c.execute("SELECT COUNT(*) FROM trades WHERE bot_name = 'ML-Prediction-Bot' AND status = 'archived_leak'")
    ml_archived = c.fetchone()[0]
    test("ML-Prediction-Bot positions archived", ml_archived >= 90, f"found {ml_archived}")

    conn.close()
else:
    test("DB file exists", False, f"not found at {db_path}")


# ============================================================
# 12. FULL PAPER MODE INTEGRATION
# ============================================================
section("12. Full Paper Mode Integration")

# Each bot should instantiate and run_scan without crashing
bots_to_test = [
    ("RSIExtremesBot", lambda: RSIExtremesBot(paper_mode=True)),
    ("MultiMomentumBot", lambda: MultiMomentumBot(paper_mode=True)),
    ("AlpacaCryptoMomentumBot", lambda: AlpacaCryptoMomentumBot(paper_mode=True)),
]

for name, factory in bots_to_test:
    try:
        b = factory()
        signals = b.run_scan()
        test(f"{name} run_scan() succeeds", True)
        test(f"{name} returns list", isinstance(signals, list), f"got {type(signals)}")
    except Exception as e:
        test(f"{name} run_scan()", False, f"EXCEPTION: {e}")
        traceback.print_exc()


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
total = PASS + FAIL
print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print(f"  ALL TESTS PASSED")
else:
    print(f"  {FAIL} FAILURES — review above")
print(f"{'='*60}")
