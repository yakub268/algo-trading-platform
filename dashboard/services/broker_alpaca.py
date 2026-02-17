"""
Trading Dashboard V4.2 - Alpaca Broker Service
Connects to existing Alpaca integration for stocks and crypto
"""
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, CryptoLatestQuoteRequest

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dashboard.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, ALPACA_PAPER_MODE, TRADES_DB
from dashboard.models import Position, BrokerHealth, BrokerStatus, Side

logger = logging.getLogger(__name__)


class AlpacaService:
    """Service for Alpaca broker integration"""

    def __init__(self):
        self.api_key = ALPACA_API_KEY
        self.secret_key = ALPACA_SECRET_KEY
        self.paper = ALPACA_PAPER_MODE

        # Initialize trading client
        self.client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper
        )

        # Initialize data clients
        self.stock_data = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.crypto_data = CryptoHistoricalDataClient(self.api_key, self.secret_key)

        self.last_ping = None
        self.latency_ms = 0

    def get_health(self) -> BrokerHealth:
        """Check API health and measure latency"""
        try:
            start = time.time()
            account = self.client.get_account()
            self.latency_ms = int((time.time() - start) * 1000)
            self.last_ping = datetime.now()

            # Determine status based on latency
            if self.latency_ms < 200:
                status = BrokerStatus.HEALTHY
            elif self.latency_ms < 500:
                status = BrokerStatus.SLOW
            else:
                status = BrokerStatus.SLOW

            return BrokerHealth(
                name="Alpaca",
                status=status,
                latency_ms=self.latency_ms,
                buying_power=float(account.buying_power),
                last_ping=self.last_ping
            )
        except Exception as e:
            logger.error(f"Alpaca health check failed: {e}")
            return BrokerHealth(
                name="Alpaca",
                status=BrokerStatus.DISCONNECTED,
                latency_ms=0,
                buying_power=0,
                last_ping=self.last_ping
            )

    def get_positions(self) -> List[Position]:
        """Fetch all open positions from Alpaca (stocks and crypto)"""
        positions = []
        try:
            alpaca_positions = self.client.get_all_positions()

            for pos in alpaca_positions:
                symbol = pos.symbol
                entry_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price) if pos.current_price else entry_price
                quantity = float(pos.qty)
                market_value = float(pos.market_value)
                unrealized_pl = float(pos.unrealized_pl)

                # Determine side
                side = Side.LONG if quantity > 0 else Side.SHORT

                # Calculate age (from when position was opened)
                # Alpaca doesn't provide open time directly, estimate from entry
                age = self._estimate_position_age(symbol)

                # Determine strategy from database or default
                strategy = self._get_position_strategy(symbol)
                signal = self._get_position_signal(symbol)

                positions.append(Position(
                    symbol=symbol,
                    broker="Alpaca",
                    strategy=strategy,
                    side=side,
                    entry_price=entry_price,
                    current_price=current_price,
                    quantity=abs(quantity),
                    pnl=unrealized_pl,
                    age=age,
                    signal=signal
                ))

        except Exception as e:
            logger.error(f"Failed to fetch Alpaca positions: {e}")

        return positions

    def get_buying_power(self) -> float:
        """Get available buying power"""
        try:
            account = self.client.get_account()
            return float(account.buying_power)
        except Exception as e:
            logger.error(f"Failed to get buying power: {e}")
            return 0.0

    def close_position(self, symbol: str) -> bool:
        """Close a specific position"""
        try:
            self.client.close_position(symbol)
            logger.info(f"Closed position: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Close all positions"""
        try:
            self.client.close_all_positions(cancel_orders=True)
            logger.info("Closed all Alpaca positions")
            return True
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False

    def reduce_position(self, symbol: str, percent: float = 0.5) -> bool:
        """Reduce position by percentage"""
        try:
            positions = self.client.get_all_positions()
            pos = next((p for p in positions if p.symbol == symbol), None)

            if not pos:
                logger.warning(f"Position not found: {symbol}")
                return False

            current_qty = float(pos.qty)
            reduce_qty = abs(current_qty) * percent

            # Submit market order to reduce
            from alpaca.trading.requests import MarketOrderRequest

            side = OrderSide.SELL if current_qty > 0 else OrderSide.BUY
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=reduce_qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )

            self.client.submit_order(order_request)
            logger.info(f"Reduced {symbol} by {percent*100}%")
            return True

        except Exception as e:
            logger.error(f"Failed to reduce position {symbol}: {e}")
            return False

    def set_stop_loss(self, symbol: str, stop_price: float) -> bool:
        """Set a stop loss order for an existing position"""
        try:
            from alpaca.trading.requests import StopOrderRequest

            positions = self.client.get_all_positions()
            pos = next((p for p in positions if p.symbol == symbol), None)

            if not pos:
                logger.warning(f"Position not found: {symbol}")
                return False

            current_qty = float(pos.qty)

            # Determine side: sell to close long, buy to close short
            side = OrderSide.SELL if current_qty > 0 else OrderSide.BUY

            order_request = StopOrderRequest(
                symbol=symbol,
                qty=abs(current_qty),
                side=side,
                stop_price=stop_price,
                time_in_force=TimeInForce.GTC
            )

            self.client.submit_order(order_request)
            logger.info(f"Stop loss set for {symbol} at ${stop_price}")
            return True

        except Exception as e:
            logger.error(f"Failed to set stop loss for {symbol}: {e}")
            return False

    def get_account_info(self) -> dict:
        """Get account summary info"""
        try:
            account = self.client.get_account()
            return {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    def _estimate_position_age(self, symbol: str) -> str:
        """Estimate position age from trade database"""
        try:
            conn = sqlite3.connect(TRADES_DB)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entry_time FROM trades
                WHERE symbol = ? AND status = 'open' AND platform = 'Alpaca'
                ORDER BY entry_time DESC LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            conn.close()

            if row and row[0]:
                entry_time = datetime.fromisoformat(row[0].replace('Z', '+00:00'))
                delta = datetime.now() - entry_time.replace(tzinfo=None)
                hours = int(delta.total_seconds() / 3600)
                if hours < 1:
                    return f"{int(delta.total_seconds() / 60)}m"
                elif hours < 24:
                    return f"{hours}h"
                else:
                    return f"{hours // 24}d"
        except Exception as e:
            logger.debug(f"Could not get position age for {symbol}: {e}")
        return "N/A"

    def _get_position_strategy(self, symbol: str) -> str:
        """Get strategy that opened this position from database"""
        try:
            conn = sqlite3.connect(TRADES_DB)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT strategy FROM trades
                WHERE symbol = ? AND status = 'open' AND platform = 'Alpaca'
                ORDER BY entry_time DESC LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            conn.close()

            if row and row[0]:
                return row[0]
        except Exception as e:
            logger.debug(f"Could not get strategy for {symbol}: {e}")

        # Fallback heuristics
        if symbol.endswith("USD") or "/" in symbol:
            return "Crypto RSI"
        return "Momentum"

    def _get_position_signal(self, symbol: str) -> str:
        """Get the signal that triggered this position"""
        try:
            conn = sqlite3.connect(TRADES_DB)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT side, notes FROM trades
                WHERE symbol = ? AND status = 'open' AND platform = 'Alpaca'
                ORDER BY entry_time DESC LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            conn.close()

            if row:
                if row[1]:  # notes field often contains signal info
                    return row[1][:30]  # Truncate if too long
                elif row[0]:
                    return f"{row[0].upper()} Signal"
        except Exception as e:
            logger.debug(f"Could not get signal for {symbol}: {e}")
        return "Manual/Unknown"
