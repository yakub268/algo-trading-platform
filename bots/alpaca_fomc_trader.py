"""
Alpaca FOMC Drift Strategy Trading Bot

Direct Alpaca API integration for pre-FOMC drift trading.
Uses alpaca-py library for reliable execution.

Strategy: FOMC Pre-Announcement Drift
- Enter SPY position ~24 hours before FOMC announcement
- Exit position ~30 minutes before announcement
- Historical tendency: SPY drifts higher before FOMC

Author: Jacob
Created: January 2026
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from decimal import Decimal

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import order fill helper to fix zero-P&L bug
try:
    from utils.order_fill_helper import submit_and_wait_for_fill, validate_fill_price_before_db
except ImportError:
    logging.warning("order_fill_helper not available")
    submit_and_wait_for_fill = None
    validate_fill_price_before_db = None

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        TakeProfitRequest,
        StopLossRequest,
        GetOrdersRequest
    )
    from alpaca.trading.enums import (
        OrderSide,
        OrderType,
        TimeInForce,
        OrderClass,
        QueryOrderStatus
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    raise ImportError(
        "alpaca-py not installed. Run: pip install alpaca-py"
    )

from config.alpaca_config import ALPACA_CONFIG, FOMC_SCHEDULE_2026

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/alpaca_fomc.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AlpacaFOMC')


class AlpacaFOMCTrader:
    """
    Alpaca FOMC Drift Strategy Trader.

    Uses direct alpaca-py integration for reliable execution.
    Implements bracket orders with take profit and stop loss.
    """

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        paper: bool = False,  # LIVE default
        paper_mode: bool = None,  # Alias accepted by master_orchestrator
        db_path: str = None
    ):
        """
        Initialize the Alpaca FOMC trader.

        Args:
            api_key: Alpaca API key (or uses env var)
            secret_key: Alpaca secret key (or uses env var)
            paper: Use paper trading (default True)
            paper_mode: Alias for paper (used by master_orchestrator)
            db_path: Path to SQLite database for trade logging
        """
        # Use mode-specific path
        try:
            from utils.data_paths import get_db_path
            db_path = db_path or get_db_path("event_trades.db")
        except ImportError:
            db_path = db_path or "data/event_trades.db"

        self.api_key = api_key or ALPACA_CONFIG["api_key"]
        self.secret_key = secret_key or ALPACA_CONFIG["secret_key"]
        # Accept paper_mode (from orchestrator) as alias for paper
        if paper_mode is not None:
            paper = paper_mode
        self.paper = paper if paper is not None else ALPACA_CONFIG["paper_mode"]

        # Initialize trading client
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper
        )

        # Initialize data client
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

        # Strategy config
        self.config = ALPACA_CONFIG["fomc_drift"]
        self.symbol = self.config["symbol"]

        # Database setup
        self.db_path = db_path
        self._init_database()

        # State tracking
        self.daily_pnl = 0.0
        self.active_order_id: Optional[str] = None

        logger.info(f"AlpacaFOMCTrader initialized. Paper mode: {self.paper}")

    def _init_database(self):
        """Initialize SQLite database for trade logging."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fomc_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL,
                exit_price REAL,
                take_profit REAL,
                stop_loss REAL,
                pnl REAL,
                status TEXT NOT NULL,
                order_id TEXT,
                notes TEXT
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def _log_trade(
        self,
        event_type: str,
        side: str,
        quantity: float,
        entry_price: float = None,
        exit_price: float = None,
        take_profit: float = None,
        stop_loss: float = None,
        pnl: float = None,
        status: str = "pending",
        order_id: str = None,
        notes: str = None
    ):
        """Log trade to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO fomc_trades
            (timestamp, event_type, symbol, side, quantity, entry_price,
             exit_price, take_profit, stop_loss, pnl, status, order_id, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            event_type,
            self.symbol,
            side,
            quantity,
            entry_price,
            exit_price,
            take_profit,
            stop_loss,
            pnl,
            status,
            order_id,
            notes
        ))

        conn.commit()
        conn.close()

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.

        Returns:
            Dictionary with cash, buying_power, portfolio_value
        """
        try:
            account = self.trading_client.get_account()

            info = {
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "currency": account.currency,
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked,
            }

            logger.info(f"Account: Cash=${info['cash']:.2f}, "
                       f"Buying Power=${info['buying_power']:.2f}, "
                       f"Portfolio=${info['portfolio_value']:.2f}")

            return info

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    def get_current_price(self, symbol: str = None) -> Optional[float]:
        """
        Get latest quote for a symbol.

        Args:
            symbol: Stock symbol (defaults to SPY)

        Returns:
            Latest price or None
        """
        symbol = symbol or self.symbol

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data_client.get_stock_latest_quote(request)

            if symbol in quote:
                # Use midpoint of bid/ask for current price
                bid = float(quote[symbol].bid_price)
                ask = float(quote[symbol].ask_price)
                price = (bid + ask) / 2

                logger.info(f"{symbol} quote: Bid=${bid:.2f}, Ask=${ask:.2f}, Mid=${price:.2f}")
                return price

            return None

        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    def get_position(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """
        Get current position for a symbol.

        Args:
            symbol: Stock symbol (defaults to SPY)

        Returns:
            Position dictionary or None if no position
        """
        symbol = symbol or self.symbol

        try:
            positions = self.trading_client.get_all_positions()

            for pos in positions:
                if pos.symbol == symbol:
                    position = {
                        "symbol": pos.symbol,
                        "qty": float(pos.qty),
                        "side": "long" if float(pos.qty) > 0 else "short",
                        "avg_entry_price": float(pos.avg_entry_price),
                        "market_value": float(pos.market_value),
                        "cost_basis": float(pos.cost_basis),
                        "unrealized_pl": float(pos.unrealized_pl),
                        "unrealized_plpc": float(pos.unrealized_plpc),
                        "current_price": float(pos.current_price),
                    }

                    logger.info(f"Position: {position['qty']} {symbol} @ ${position['avg_entry_price']:.2f}, "
                               f"P&L: ${position['unrealized_pl']:.2f} ({position['unrealized_plpc']*100:.2f}%)")

                    return position

            logger.info(f"No position in {symbol}")
            return None

        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None

    def execute_fomc_entry(self, position_size_usd: float = None) -> Optional[str]:
        """
        Execute FOMC drift entry - buy SPY with bracket order.

        Args:
            position_size_usd: Position size in dollars (default from config)

        Returns:
            Order ID or None on failure
        """
        position_size_usd = position_size_usd or self.config["position_size_usd"]

        # Check for existing position
        existing = self.get_position()
        if existing:
            logger.warning(f"Already have position in {self.symbol}, skipping entry")
            return None

        # Get current price
        current_price = self.get_current_price()
        if not current_price:
            logger.error("Cannot get current price, aborting entry")
            return None

        # Calculate quantity - use fractional shares if enabled and needed
        use_fractional = self.config.get("use_fractional", False)

        if use_fractional:
            # Use fractional shares (note: bracket orders may not support fractional)
            qty = round(position_size_usd / current_price, 4)
            if qty < 0.01:
                logger.error(f"Position size too small: ${position_size_usd} / ${current_price:.2f} = {qty} shares")
                return None
            logger.info(f"Using fractional shares: {qty}")
        else:
            qty = int(position_size_usd / current_price)
            if qty < 1:
                logger.error(f"Position size too small: ${position_size_usd} / ${current_price:.2f} = {qty} shares")
                logger.info("Consider enabling 'use_fractional: True' in config for expensive stocks")
                return None

        # Calculate bracket order prices
        take_profit_price = round(current_price * (1 + self.config["take_profit_pct"]), 2)
        stop_loss_price = round(current_price * (1 - self.config["stop_loss_pct"]), 2)

        logger.info(f"FOMC Entry: BUY {qty} {self.symbol} @ ~${current_price:.2f}")
        logger.info(f"Bracket: TP=${take_profit_price:.2f} (+{self.config['take_profit_pct']*100}%), "
                   f"SL=${stop_loss_price:.2f} (-{self.config['stop_loss_pct']*100}%)")

        try:
            # Create bracket order (market order with take profit and stop loss)
            order_request = MarketOrderRequest(
                symbol=self.symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit_price),
                stop_loss=StopLossRequest(stop_price=stop_loss_price)
            )

            # CRITICAL FIX: Poll for confirmed fill price
            if submit_and_wait_for_fill:
                try:
                    fill_result = submit_and_wait_for_fill(
                        self.trading_client,
                        order_request,
                        timeout=30
                    )
                    confirmed_entry_price = fill_result['fill_price']
                    confirmed_qty = fill_result['fill_qty']

                    self.active_order_id = fill_result['order_id']
                    logger.info(
                        f"FOMC entry FILLED: {confirmed_qty} @ ${confirmed_entry_price:.2f}"
                    )

                    # Log to database with CONFIRMED fill price
                    self._log_trade(
                        event_type="FOMC_ENTRY",
                        side="buy",
                        quantity=confirmed_qty,
                        entry_price=confirmed_entry_price,
                        take_profit=take_profit_price,
                        stop_loss=stop_loss_price,
                        status="filled",
                        order_id=fill_result['order_id'],
                        notes=f"FOMC drift entry - bracket order"
                    )

                    return fill_result['order_id']

                except Exception as fill_err:
                    logger.error(f"FOMC entry fill failed: {fill_err}")
                    self._log_trade(
                        event_type="FOMC_ENTRY",
                        side="buy",
                        quantity=qty,
                        entry_price=current_price,
                        status="failed",
                        notes=f"Fill failed: {str(fill_err)}"
                    )
                    return None
            else:
                # Fallback (has zero-P&L bug)
                logger.warning("order_fill_helper not available, using old method")
                order = self.trading_client.submit_order(order_request)

                self.active_order_id = str(order.id)
                logger.info(f"Order submitted: ID={order.id}, Status={order.status}")

                # Log to database
                self._log_trade(
                    event_type="FOMC_ENTRY",
                    side="buy",
                    quantity=qty,
                    entry_price=current_price,
                    take_profit=take_profit_price,
                    stop_loss=stop_loss_price,
                    status=str(order.status),
                    order_id=str(order.id),
                    notes=f"FOMC drift entry - bracket order"
                )

                return str(order.id)

        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            self._log_trade(
                event_type="FOMC_ENTRY",
                side="buy",
                quantity=qty,
                entry_price=current_price,
                status="failed",
                notes=f"Order failed: {str(e)}"
            )
            return None

    def execute_fomc_exit(self) -> Optional[str]:
        """
        Execute FOMC exit - close SPY position before announcement.

        Returns:
            Order ID or None on failure
        """
        # Get current position
        position = self.get_position()
        if not position:
            logger.info(f"No position to exit in {self.symbol}")
            return None

        qty = abs(position["qty"])
        current_price = position["current_price"]
        entry_price = position["avg_entry_price"]
        unrealized_pl = position["unrealized_pl"]

        logger.info(f"FOMC Exit: SELL {qty} {self.symbol} @ ~${current_price:.2f}")
        logger.info(f"Unrealized P&L: ${unrealized_pl:.2f}")

        try:
            # First, cancel any open orders for this symbol
            self._cancel_open_orders()

            # Market sell to close
            order_request = MarketOrderRequest(
                symbol=self.symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )

            # CRITICAL FIX: Poll for confirmed fill price
            if submit_and_wait_for_fill:
                try:
                    fill_result = submit_and_wait_for_fill(
                        self.trading_client,
                        order_request,
                        timeout=30
                    )
                    confirmed_exit_price = fill_result['fill_price']
                    confirmed_qty = fill_result['fill_qty']

                    # Recalculate P&L with confirmed fill price
                    confirmed_pnl = (confirmed_exit_price - entry_price) * confirmed_qty

                    logger.info(
                        f"FOMC exit FILLED: {confirmed_qty} @ ${confirmed_exit_price:.2f} | "
                        f"P&L: ${confirmed_pnl:.2f}"
                    )

                    # Log to database with CONFIRMED fill price and P&L
                    self._log_trade(
                        event_type="FOMC_EXIT",
                        side="sell",
                        quantity=confirmed_qty,
                        entry_price=entry_price,
                        exit_price=confirmed_exit_price,
                        pnl=confirmed_pnl,
                        status="filled",
                        order_id=fill_result['order_id'],
                        notes=f"FOMC drift exit - pre-announcement close"
                    )

                    self.active_order_id = None
                    return fill_result['order_id']

                except Exception as fill_err:
                    logger.error(f"FOMC exit fill failed: {fill_err}")
                    self._log_trade(
                        event_type="FOMC_EXIT",
                        side="sell",
                        quantity=qty,
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl=unrealized_pl,
                        status="failed",
                        notes=f"Fill failed: {str(fill_err)}"
                    )
                    return None
            else:
                # Fallback (has zero-P&L bug)
                logger.warning("order_fill_helper not available, using old method")
                order = self.trading_client.submit_order(order_request)

                logger.info(f"Exit order submitted: ID={order.id}, Status={order.status}")

                # Log to database
                self._log_trade(
                    event_type="FOMC_EXIT",
                    side="sell",
                    quantity=qty,
                    entry_price=entry_price,
                    exit_price=current_price,
                    pnl=unrealized_pl,
                    status=str(order.status),
                    order_id=str(order.id),
                    notes=f"FOMC drift exit - pre-announcement close"
                )

                self.active_order_id = None
                return str(order.id)

        except Exception as e:
            logger.error(f"Failed to submit exit order: {e}")
            return None

    def _cancel_open_orders(self):
        """Cancel all open orders for the symbol."""
        try:
            request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbols=[self.symbol]
            )
            orders = self.trading_client.get_orders(request)

            for order in orders:
                try:
                    self.trading_client.cancel_order_by_id(order.id)
                    logger.info(f"Cancelled order: {order.id}")
                except Exception as e:
                    logger.warning(f"Failed to cancel order {order.id}: {e}")

        except Exception as e:
            logger.error(f"Failed to get/cancel open orders: {e}")

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an order.

        Args:
            order_id: Order ID to check

        Returns:
            Order status dictionary or None
        """
        try:
            order = self.trading_client.get_order_by_id(order_id)

            status = {
                "id": order.id,
                "symbol": order.symbol,
                "side": str(order.side),
                "qty": float(order.qty) if order.qty else 0,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "type": str(order.type),
                "status": str(order.status),
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "created_at": str(order.created_at),
                "updated_at": str(order.updated_at),
            }

            logger.info(f"Order {order_id}: {status['status']}, "
                       f"Filled {status['filled_qty']}/{status['qty']}")

            return status

        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return None

    def get_next_fomc_event(self) -> Optional[Dict[str, Any]]:
        """
        Get the next upcoming FOMC event.

        Returns:
            Dictionary with event details or None
        """
        now = datetime.utcnow()

        for event in FOMC_SCHEDULE_2026:
            event_dt = datetime.strptime(
                f"{event['date']} {event['time']}",
                "%Y-%m-%d %H:%M"
            )
            # Convert ET to UTC (add 5 hours)
            event_dt_utc = event_dt + timedelta(hours=5)

            if event_dt_utc > now:
                time_to_event = event_dt_utc - now

                return {
                    "date": event["date"],
                    "time": event["time"],
                    "type": event["type"],
                    "datetime_utc": event_dt_utc,
                    "hours_until": time_to_event.total_seconds() / 3600,
                    "entry_time": event_dt_utc - timedelta(hours=self.config["entry_hours_before"]),
                    "exit_time": event_dt_utc - timedelta(minutes=self.config["exit_minutes_before"]),
                }

        return None

    def should_enter(self) -> bool:
        """
        Check if we should enter FOMC trade now.

        Returns:
            True if within entry window
        """
        next_event = self.get_next_fomc_event()
        if not next_event:
            return False

        now = datetime.utcnow()
        entry_time = next_event["entry_time"]
        exit_time = next_event["exit_time"]

        # Enter if we're past entry time but before exit time
        if entry_time <= now < exit_time:
            hours_until = next_event["hours_until"]
            logger.info(f"Within FOMC entry window. Event in {hours_until:.1f} hours")
            return True

        return False

    def should_exit(self) -> bool:
        """
        Check if we should exit FOMC trade now.

        Returns:
            True if past exit time
        """
        next_event = self.get_next_fomc_event()
        if not next_event:
            return False

        now = datetime.utcnow()
        exit_time = next_event["exit_time"]

        # Exit if we're past exit time
        if now >= exit_time:
            logger.info("Past FOMC exit time - closing position")
            return True

        return False

    def run_strategy(self) -> Dict[str, Any]:
        """
        Run a single check cycle - called by master orchestrator.

        Alias for run_check() to match orchestrator's method_priority list.

        Returns:
            Status dictionary with action taken
        """
        return self.run_check()

    def run_check(self) -> Dict[str, Any]:
        """
        Run a single check cycle - called by scheduler.

        Returns:
            Status dictionary with action taken
        """
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "none",
            "position": None,
            "next_event": None,
        }

        try:
            # Get current state
            result["position"] = self.get_position()
            result["next_event"] = self.get_next_fomc_event()

            # Check if we should exit
            if result["position"] and self.should_exit():
                order_id = self.execute_fomc_exit()
                if order_id:
                    result["action"] = "exit"
                    result["order_id"] = order_id
                return result

            # Check if we should enter
            if not result["position"] and self.should_enter():
                order_id = self.execute_fomc_entry()
                if order_id:
                    result["action"] = "entry"
                    result["order_id"] = order_id
                return result

            # No action needed
            if result["next_event"]:
                hours = result["next_event"]["hours_until"]
                logger.info(f"No action needed. Next FOMC in {hours:.1f} hours")

        except Exception as e:
            logger.error(f"Error in run_check: {e}")
            result["error"] = str(e)

        return result


def main():
    """Entry point for testing the Alpaca FOMC trader."""
    trader = AlpacaFOMCTrader()

    # Get account info
    print("\n=== Account Info ===")
    account = trader.get_account_info()

    # Get current price
    print("\n=== SPY Quote ===")
    price = trader.get_current_price("SPY")

    # Get position
    print("\n=== Current Position ===")
    position = trader.get_position()

    # Get next FOMC event
    print("\n=== Next FOMC Event ===")
    event = trader.get_next_fomc_event()
    if event:
        print(f"Date: {event['date']} {event['time']} ET")
        print(f"Hours until: {event['hours_until']:.1f}")
        print(f"Entry window opens: {event['entry_time']}")
        print(f"Exit before: {event['exit_time']}")

    # Run check cycle
    print("\n=== Run Check ===")
    result = trader.run_check()
    print(f"Action: {result['action']}")


if __name__ == "__main__":
    main()
