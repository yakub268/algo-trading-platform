"""
EventEdgeBot — Kalshi prediction market edge detection bot.

Wraps the autonomous event trading system as a bot inside the orchestrator.

Orchestrator interface:
- run_scan() -> List[Dict]     — called every 60s by scheduler
- place_order(ticker, side, quantity, price) — called by orchestrator execution
- check_exits()                — called every 60s by separate scheduled task
"""

import os
import re
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from bots.kalshi_client import KalshiClient

from bots.event_trading.scanner import AutonomousScanner
from bots.event_trading.edge_detector import EdgeDetector, EdgeSignal
from bots.event_trading.correlation_risk import CorrelationRiskManager, PositionRisk
from bots.event_trading.consensus_engine import ConsensusEngine, ConsensusSignal

try:
    from bots.event_trading.entry_model import load_model as load_entry_model, predict as ml_predict
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

try:
    from bots.event_trading.price_intelligence import should_enter as price_should_enter
    _HAS_PRICE_INTEL = True
except ImportError:
    _HAS_PRICE_INTEL = False

MT = ZoneInfo("America/Denver")
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "live", "event_trading.db")

# Trading parameters (from autonomous_trader.py)
EDGE_THRESHOLD = 0.10
CONFIDENCE_THRESHOLD = 0.75
MAX_POSITION_PCT = 0.15
KELLY_FRACTION = 0.25
CONSENSUS_KELLY_FRACTION = 0.50
MIN_BALANCE = 10.0
MAX_CONTRACTS = 100
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2

# Safety
CONSECUTIVE_LOSS_LIMIT = 3
LOSS_REDUCTION_FACTOR = 0.5

# Market blocks
BLOCKED_PREFIXES = ()  # Weather unblocked

logger = logging.getLogger("EventEdge.Bot")


def mt_now() -> datetime:
    return datetime.now(MT)


def mt_str(dt: datetime = None) -> str:
    return (dt or mt_now()).strftime("%Y-%m-%d %H:%M:%S MT")


def _init_db(db_path: str):
    """Create autonomous_trades table if needed."""
    conn = sqlite3.connect(db_path, timeout=30)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS autonomous_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT,
                ticker TEXT,
                side TEXT,
                entry_price REAL,
                quantity INTEGER,
                expected_value REAL,
                signal_edge REAL,
                signal_confidence REAL,
                entry_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_price REAL,
                exit_timestamp TIMESTAMP,
                realized_pnl REAL,
                status TEXT DEFAULT 'OPEN',
                order_id TEXT DEFAULT ''
            )
        """)
    conn.close()


def kelly_criterion(edge: float, probability: float, bankroll: float,
                    fraction: float = KELLY_FRACTION,
                    max_pct: float = MAX_POSITION_PCT) -> int:
    """Fractional Kelly Criterion position sizing for binary markets."""
    if edge <= 0 or probability <= 0 or probability >= 1 or bankroll <= 0:
        return 0

    market_price = probability - edge
    if market_price <= 0 or market_price >= 1:
        return 0

    b = (1.0 / market_price) - 1.0
    q = 1.0 - probability
    kelly_pct = max(0, (b * probability - q) / b)
    position_pct = min(kelly_pct * fraction, max_pct)
    position_dollars = bankroll * position_pct
    price_per_contract = market_price
    contracts = int(position_dollars / price_per_contract) if price_per_contract > 0 else 0
    return max(0, contracts)


class EventEdgeBot:
    """Kalshi prediction market edge detection bot for the orchestrator."""

    def __init__(self, paper_mode: bool = True):
        self.client = KalshiClient()
        self.paper_mode = paper_mode
        self.infrastructure = None  # Injected by orchestrator

        # Sub-components
        self.scanner = AutonomousScanner(self.client, db_path=DB_PATH)
        self.detector = EdgeDetector(self.client, db_path=DB_PATH)
        self.risk_manager = CorrelationRiskManager(
            db_path=DB_PATH, max_correlated_exposure=200, max_single_category=300
        )
        self.consensus = ConsensusEngine(client=self.client)

        # ML entry model
        self.entry_model = None
        if _HAS_ML:
            self.entry_model = load_entry_model()
            if self.entry_model:
                logger.info("ML Entry Model loaded")

        # State
        self.consecutive_losses = 0
        self.position_size_multiplier = 1.0
        self.db_path = DB_PATH

        _init_db(self.db_path)
        self._restore_loss_streak()

        logger.info(f"EventEdgeBot initialized (paper={paper_mode})")

    def _restore_loss_streak(self):
        """Restore consecutive loss count from recent trades."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT realized_pnl FROM autonomous_trades WHERE status='CLOSED' ORDER BY id DESC LIMIT 10"
        ).fetchall()
        conn.close()

        streak = 0
        for r in rows:
            pnl = r["realized_pnl"] or 0
            if pnl < 0:
                streak += 1
            else:
                break

        self.consecutive_losses = streak
        if streak >= CONSECUTIVE_LOSS_LIMIT:
            self.position_size_multiplier = LOSS_REDUCTION_FACTOR
            logger.warning(f"Restored {streak} consecutive losses — size at {self.position_size_multiplier*100:.0f}%")

    def _get_balance(self) -> float:
        """Get Kalshi balance with retry."""
        for attempt in range(3):
            try:
                bal = self.client.get_balance()
                return bal.get("balance", 0) / 100.0
            except Exception as e:
                logger.error(f"Balance fetch attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
        return -1.0

    def _get_open_positions(self) -> List[Dict]:
        """Get open positions from event_trading DB."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM autonomous_trades WHERE status='OPEN'").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Orchestrator Interface ─────────────────────────────────────────

    def run_scan(self) -> List[Dict]:
        """Single scan cycle returning orchestrator-format signals."""
        signals = []
        balance = self._get_balance()

        if balance < 0:
            logger.warning("API unreachable — skipping cycle")
            return []

        if balance < MIN_BALANCE:
            logger.info(f"Balance ${balance:.2f} < ${MIN_BALANCE} — skip new trades")
            return []

        # Phase 1: Market discovery
        logger.info("Phase 1: Market discovery...")
        markets = self.scanner.discover(top_n=20)
        if not markets:
            return []

        # Phase 2: Edge detection
        logger.info("Phase 2: Edge detection...")
        market_dicts = [m.to_dict() for m in markets]
        edge_signals = self.detector.analyze_batch(market_dicts)

        # Phase 3: Filter + convert to orchestrator format
        logger.info(f"Phase 3: Evaluating {len(edge_signals)} signals...")
        traded_this_cycle = set()

        for sig in edge_signals:
            if not self._passes_all_gates(sig, traded_this_cycle, balance):
                continue

            orch_signal = self._to_orch_signal(sig, balance)
            if orch_signal:
                signals.append(orch_signal)
                traded_this_cycle.add(sig.market_id)
                traded_this_cycle.add(self._extract_underlying(sig.market_id))

        # Phase 4: Consensus (only if no edge signals passed)
        if not signals and self.consensus:
            try:
                logger.info("Phase 4: Consensus scan...")
                consensus_signals = self.consensus.scan_for_consensus(market_dicts[:10])
                for cs in consensus_signals:
                    orch_signal = self._consensus_to_signal(cs, balance)
                    if orch_signal:
                        signals.append(orch_signal)
            except Exception as e:
                logger.error(f"Consensus scan error: {e}")

        logger.info(f"Scan complete: {len(signals)} actionable signals")
        return signals

    def place_order(self, ticker: str, side: str, quantity: int, price: int) -> Optional[Dict]:
        """Execute via KalshiClient. Called by orchestrator."""
        if self.paper_mode:
            logger.info(f"[PAPER] Order: {side} {quantity}x {ticker} @ {price}c")
            return {"paper": True, "ticker": ticker, "side": side, "quantity": quantity, "price": price}

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                order = self.client.create_order(
                    ticker=ticker, side=side, action="buy",
                    count=quantity, price=price, order_type="limit",
                )
                order_id = order.get("order_id", "unknown")

                # Record trade in event DB
                self._record_trade(ticker, side, quantity, price, order_id)

                logger.info(f"ORDER: {side} {quantity}x {ticker} @ {price}c | ID: {order_id}")
                return order

            except Exception as e:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.error(f"Order attempt {attempt} failed: {e}")
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(wait)

        logger.error(f"All {RETRY_ATTEMPTS} order attempts failed for {ticker}")
        return None

    def check_exits(self):
        """Tiered exit management. Called by orchestrator every 60s."""
        positions = self._get_open_positions()

        for pos in positions:
            market_id = pos["market_id"]
            entry_price = pos["entry_price"]
            quantity = pos["quantity"]
            side = pos["side"]

            try:
                # Check if market is settled
                market_data = self.client.get_market(market_id)
                status = market_data.get("status", "")

                if status in ("settled", "closed", "finalized"):
                    result = market_data.get("result", "")
                    if result == side:
                        exit_price = 100
                    elif result:
                        exit_price = 0
                    else:
                        exit_price = entry_price
                    pnl = (exit_price - entry_price) * quantity / 100.0
                    self._close_position(pos["id"], exit_price, pnl)
                    continue

                # Minimum hold time: 5 minutes
                entry_time_str = pos.get("entry_timestamp", "")
                if entry_time_str:
                    try:
                        entry_dt = datetime.fromisoformat(
                            entry_time_str.replace(" MT", "").replace(" MST", "").replace(" MDT", "")
                        )
                        minutes_held = (datetime.now(MT) - entry_dt.replace(tzinfo=MT)).total_seconds() / 60
                        if minutes_held < 5:
                            continue
                    except Exception:
                        pass

                # Check current price for tiered exits
                ob = self.client.get_orderbook(market_id)
                bids = ob.get(side, [])
                if bids:
                    current_price = bids[0][0]
                    gain_pct = (current_price - entry_price) / max(entry_price, 1) * 100
                    gain_cents = current_price - entry_price

                    # Tier 1: +100% (doubled) — always sell
                    if gain_pct >= 100:
                        self._sell_position(pos, current_price, "TAKE_PROFIT_2X")
                    # Tier 2: +50% if entry >= 10c
                    elif gain_pct >= 50 and entry_price >= 10:
                        self._sell_position(pos, current_price, "TAKE_PROFIT_50")
                    # Tier 3: +30% or +5c
                    elif gain_pct >= 30 or gain_cents >= 5:
                        self._sell_position(pos, current_price, "TAKE_PROFIT")
                    # Stop loss: -50% if entry >= 15c
                    elif entry_price >= 15 and current_price <= entry_price * 0.5:
                        self._sell_position(pos, current_price, "STOP_LOSS")

            except Exception as e:
                logger.error(f"Exit check failed for {market_id}: {e}")

    # ── Signal Conversion ──────────────────────────────────────────────

    def _to_orch_signal(self, edge_signal: EdgeSignal, bankroll: float) -> Optional[Dict]:
        """Convert EdgeSignal to orchestrator signal dict."""
        side = "yes" if edge_signal.direction == "YES" else "no"

        if side == "yes":
            price_cents = int(edge_signal.market_probability * 100)
            kelly_prob = edge_signal.ensemble_probability
        else:
            price_cents = 100 - int(edge_signal.market_probability * 100)
            kelly_prob = 1.0 - edge_signal.ensemble_probability

        price_cents = max(1, min(99, price_cents))

        contracts = kelly_criterion(
            edge=edge_signal.edge, probability=kelly_prob,
            bankroll=bankroll, fraction=KELLY_FRACTION, max_pct=MAX_POSITION_PCT,
        )
        contracts = int(contracts * self.position_size_multiplier)

        # Dynamic cap
        dynamic_max = max(10, int(bankroll * 0.30 / max(0.01, price_cents / 100)))
        contracts = min(contracts, MAX_CONTRACTS, dynamic_max)

        if contracts < 1:
            return None

        return {
            'ticker': edge_signal.market_id,
            'action': 'buy',
            'side': side,
            'quantity': contracts,
            'price_cents': price_cents,
            'type': 'event_edge',
            'reasoning': (
                f"Edge: {edge_signal.edge*100:+.1f}%, "
                f"Conf: {edge_signal.confidence*100:.0f}%, "
                f"Dir: {edge_signal.direction}"
            ),
            'scan_only': False,
        }

    def _consensus_to_signal(self, cs: ConsensusSignal, bankroll: float) -> Optional[Dict]:
        """Convert ConsensusSignal to orchestrator signal dict with 2x sizing."""
        side = "yes" if cs.direction == "YES" else "no"
        price_cents = int(cs.market_price * 100) if side == "yes" else 100 - int(cs.market_price * 100)
        price_cents = max(1, min(99, price_cents))

        ensemble_prob = 0.5 + (cs.consensus_edge if cs.direction == "YES" else -cs.consensus_edge)
        kelly_prob = ensemble_prob if side == "yes" else 1.0 - ensemble_prob

        contracts = kelly_criterion(
            edge=cs.consensus_edge, probability=kelly_prob,
            bankroll=bankroll, fraction=CONSENSUS_KELLY_FRACTION, max_pct=MAX_POSITION_PCT,
        )
        contracts = int(contracts * self.position_size_multiplier)
        dynamic_max = max(10, int(bankroll * 0.30 / max(0.01, price_cents / 100)))
        contracts = min(contracts, MAX_CONTRACTS, dynamic_max)

        if contracts < 1:
            return None

        return {
            'ticker': cs.market_id,
            'action': 'buy',
            'side': side,
            'quantity': contracts,
            'price_cents': price_cents,
            'type': 'event_consensus',
            'reasoning': (
                f"CONSENSUS: {cs.sources_agreeing}/{cs.total_sources} sources, "
                f"Edge: {cs.consensus_edge*100:+.1f}%, "
                f"Dir: {cs.direction}"
            ),
            'scan_only': False,
        }

    # ── Gate Logic ─────────────────────────────────────────────────────

    def _passes_all_gates(self, signal: EdgeSignal, traded: set, balance: float) -> bool:
        """All entry gates from autonomous_trader Phase 3."""

        # 1. Blocked prefix check
        if any(signal.market_id.startswith(prefix) for prefix in BLOCKED_PREFIXES):
            logger.info(f"BLOCKED: {signal.market_id} — prefix block")
            return False

        # 2. Cycle dedup (market_id + underlying)
        if signal.market_id in traded:
            return False
        underlying = self._extract_underlying(signal.market_id)
        if underlying in traded:
            return False

        # 3. Edge > 10% AND confidence > 75%
        if abs(signal.edge) <= EDGE_THRESHOLD or signal.confidence <= CONFIDENCE_THRESHOLD:
            return False

        # 4. Entry price cap: max 30c
        if signal.direction == "YES":
            entry_cost_cents = signal.market_probability * 100
        else:
            entry_cost_cents = (1 - signal.market_probability) * 100
        if entry_cost_cents > 30:
            logger.info(f"PRICE CAP: {signal.ticker} costs {entry_cost_cents:.0f}c — max 30c")
            return False

        # 5. ML model gate
        if self.entry_model and _HAS_ML:
            try:
                ml_pred = ml_predict(self.entry_model, signal.to_dict())
                if ml_pred and not ml_pred.should_trade:
                    logger.info(f"ML BLOCK: {signal.ticker} ml_edge={ml_pred.edge*100:+.1f}%")
                    return False
            except Exception as e:
                logger.debug(f"ML prediction error: {e}")

        # 6. Price intelligence gate
        if _HAS_PRICE_INTEL:
            try:
                ok, reason = price_should_enter(signal.ticker, signal.direction)
                if not ok:
                    logger.info(f"PRICE BLOCK: {signal.ticker} — {reason}")
                    return False
            except Exception as e:
                logger.debug(f"Price intelligence error: {e}")

        # 7. Already has position (30-min cooldown)
        if self._already_has_position(signal.market_id):
            return False

        # 8. Conflicting position (same underlying)
        if self._has_conflicting_position(signal.market_id):
            return False

        # 9. Correlation risk check
        try:
            self.risk_manager.load_from_db()
            side_check = "yes" if signal.direction == "YES" else "no"
            est_cost = abs(signal.market_probability * 100) * 0.10
            new_pos = PositionRisk(ticker=signal.market_id, side=side_check, exposure=est_cost)
            allowed, reason = self.risk_manager.can_add(new_pos)
            if not allowed:
                logger.info(f"CORRELATION BLOCK: {signal.market_id} — {reason}")
                return False
        except Exception as e:
            logger.debug(f"Correlation check skipped: {e}")

        return True

    # ── Position Management ────────────────────────────────────────────

    def _already_has_position(self, market_id: str) -> bool:
        """Check open position or recent cooldown."""
        conn = sqlite3.connect(self.db_path, timeout=30)

        # Open position check
        row = conn.execute(
            "SELECT COUNT(*) FROM autonomous_trades WHERE market_id=? AND status='OPEN'",
            (market_id,)
        ).fetchone()
        if row[0] > 0:
            conn.close()
            return True

        # 30-min re-entry cooldown
        last_entry = conn.execute(
            "SELECT entry_timestamp FROM autonomous_trades WHERE market_id=? ORDER BY id DESC LIMIT 1",
            (market_id,)
        ).fetchone()
        if last_entry and last_entry[0]:
            try:
                ts_str = last_entry[0].replace(" MT", "").replace(" MST", "").replace(" MDT", "")
                entry_dt = datetime.fromisoformat(ts_str).replace(tzinfo=MT)
                minutes_since = (mt_now() - entry_dt).total_seconds() / 60
                if minutes_since < 30:
                    logger.info(f"RE-ENTRY BLOCK: {market_id} last entered {minutes_since:.0f}min ago")
                    conn.close()
                    return True
            except (ValueError, TypeError):
                pass

        # Escalating cooldown based on trade count
        trade_count = conn.execute(
            "SELECT COUNT(*) FROM autonomous_trades WHERE market_id=? AND status='CLOSED'",
            (market_id,)
        ).fetchone()[0]

        if trade_count >= 3:
            cooldown_hours = 24
        elif trade_count >= 2:
            cooldown_hours = 6
        elif trade_count >= 1:
            cooldown_hours = 2
        else:
            cooldown_hours = 0

        if cooldown_hours > 0:
            row2 = conn.execute(
                "SELECT exit_timestamp FROM autonomous_trades WHERE market_id=? AND status='CLOSED' "
                "ORDER BY id DESC LIMIT 1",
                (market_id,)
            ).fetchone()
            if row2 and row2[0]:
                try:
                    ts_str = row2[0].replace(" MT", "").replace(" MST", "").replace(" MDT", "")
                    exit_dt = datetime.fromisoformat(ts_str).replace(tzinfo=MT)
                    hours_since = (mt_now() - exit_dt).total_seconds() / 3600
                    if hours_since < cooldown_hours:
                        logger.info(f"COOLDOWN: {market_id} {hours_since:.1f}h ago (need {cooldown_hours}h)")
                        conn.close()
                        return True
                except (ValueError, TypeError):
                    pass

        # Ticker-prefix dedup for losers
        ticker_prefix = "-".join(market_id.split("-")[:2]) if "-" in market_id else market_id
        recent_losses = conn.execute(
            "SELECT COUNT(*) FROM autonomous_trades "
            "WHERE market_id LIKE ? AND status='CLOSED' AND realized_pnl < 0",
            (f"{ticker_prefix}%",)
        ).fetchone()[0]

        if recent_losses >= 2:
            last_loss = conn.execute(
                "SELECT exit_timestamp FROM autonomous_trades "
                "WHERE market_id LIKE ? AND status='CLOSED' AND realized_pnl < 0 "
                "ORDER BY id DESC LIMIT 1",
                (f"{ticker_prefix}%",)
            ).fetchone()
            if last_loss and last_loss[0]:
                try:
                    ts_str = last_loss[0].replace(" MT", "").replace(" MST", "").replace(" MDT", "")
                    loss_dt = datetime.fromisoformat(ts_str).replace(tzinfo=MT)
                    hours_since = (mt_now() - loss_dt).total_seconds() / 3600
                    if hours_since < 4:
                        logger.info(f"DEDUP BLOCK: {recent_losses} losses on {ticker_prefix}*")
                        conn.close()
                        return True
                except (ValueError, TypeError):
                    pass

        conn.close()
        return False

    @staticmethod
    def _extract_underlying(ticker: str) -> str:
        """Extract underlying event from ticker (remove threshold suffix)."""
        m = re.match(r'^(.+?)-[BT]-?\d', ticker)
        if m:
            return m.group(1)
        parts = ticker.split("-")
        return "-".join(parts[:2]) if len(parts) >= 2 else ticker

    def _has_conflicting_position(self, ticker: str) -> bool:
        """Block if open position on different threshold of same underlying."""
        underlying = self._extract_underlying(ticker)
        if underlying == ticker:
            return False

        conn = sqlite3.connect(self.db_path, timeout=30)
        rows = conn.execute(
            "SELECT market_id, side FROM autonomous_trades WHERE status='OPEN' "
            "AND market_id LIKE ? AND market_id != ?",
            (f"{underlying}-%", ticker)
        ).fetchall()
        conn.close()

        if rows:
            existing = [(r[0], r[1]) for r in rows]
            logger.info(
                f"CONFLICT BLOCK: {ticker} — open on {underlying}: "
                + ", ".join(f"{m} {s}" for m, s in existing)
            )
            return True
        return False

    def _record_trade(self, ticker: str, side: str, quantity: int, price: int, order_id: str):
        """Record trade in event_trading DB."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute(
                """INSERT INTO autonomous_trades
                   (market_id, ticker, side, entry_price, quantity,
                    expected_value, signal_edge, signal_confidence,
                    entry_timestamp, status, order_id)
                   VALUES (?, ?, ?, ?, ?, 0, 0, 0, ?, 'OPEN', ?)""",
                (ticker, ticker, side, price, quantity, mt_str(), order_id),
            )
        conn.close()

    def _sell_position(self, pos: Dict, exit_price: int, reason: str):
        """Sell an open position."""
        market_id = pos["market_id"]
        side = pos["side"]
        quantity = pos["quantity"]

        if self.paper_mode:
            pnl = (exit_price - pos["entry_price"]) * quantity / 100.0
            self._close_position(pos["id"], exit_price, pnl)
            logger.info(f"[PAPER] EXIT ({reason}): {market_id} @ {exit_price}c P&L ${pnl:+.2f}")
            return

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                self.client.create_order(
                    ticker=market_id, side=side, action="sell",
                    count=quantity, price=exit_price, order_type="limit",
                )
                pnl = (exit_price - pos["entry_price"]) * quantity / 100.0
                self._close_position(pos["id"], exit_price, pnl)
                logger.info(f"EXIT ({reason}): {market_id} @ {exit_price}c P&L ${pnl:+.2f}")
                return
            except Exception as e:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.error(f"Sell attempt {attempt} failed for {market_id}: {e}")
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(wait)

        logger.error(f"All sell attempts failed for {market_id}")

    def _close_position(self, trade_id: int, exit_price: float, pnl: float):
        """Update DB to close a position."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        with conn:
            conn.execute(
                "UPDATE autonomous_trades SET exit_price=?, exit_timestamp=?, realized_pnl=?, status='CLOSED' WHERE id=?",
                (exit_price, mt_str(), pnl, trade_id),
            )
        conn.close()

        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
                self.position_size_multiplier = LOSS_REDUCTION_FACTOR
                logger.warning(f"SAFETY: {self.consecutive_losses} consecutive losses — size reduced")
        else:
            self.consecutive_losses = 0
            self.position_size_multiplier = 1.0
