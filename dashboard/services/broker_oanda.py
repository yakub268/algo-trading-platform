"""
Trading Dashboard V4.2 - OANDA Broker Service
Connects to existing OANDA integration for forex trading
"""
import time
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dashboard.config import OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_BASE_URL, TRADES_DB
from dashboard.models import Position, BrokerHealth, BrokerStatus, Side

logger = logging.getLogger(__name__)

# Suppress noisy oandapyV20 library errors (401s logged at ERROR level internally)
logging.getLogger('oandapyV20.oandapyV20').setLevel(logging.CRITICAL)

try:
    from oandapyV20 import API
    from oandapyV20.endpoints import accounts, positions as oanda_positions, orders, trades
    OANDA_AVAILABLE = True
except ImportError:
    logger.warning("oandapyV20 not installed, OANDA service will be limited")
    OANDA_AVAILABLE = False


class OandaService:
    """Service for OANDA forex broker integration"""

    def __init__(self):
        self.api_key = OANDA_API_KEY
        self.account_id = OANDA_ACCOUNT_ID
        self.base_url = OANDA_BASE_URL
        self.last_ping = None
        self.latency_ms = 0

        if OANDA_AVAILABLE and self.api_key:
            env = "practice" if "practice" in self.base_url else "live"
            self.api = API(access_token=self.api_key, environment=env)
        else:
            self.api = None

    def get_health(self) -> BrokerHealth:
        """Check API health and measure latency"""
        if not self.api:
            return BrokerHealth(
                name="OANDA",
                status=BrokerStatus.DISCONNECTED,
                latency_ms=0,
                buying_power=0,
                last_ping=None
            )

        try:
            start = time.time()
            endpoint = accounts.AccountSummary(self.account_id)
            response = self.api.request(endpoint)
            self.latency_ms = int((time.time() - start) * 1000)
            self.last_ping = datetime.now()

            account = response.get("account", {})
            margin_available = float(account.get("marginAvailable", 0))

            if self.latency_ms < 200:
                status = BrokerStatus.HEALTHY
            elif self.latency_ms < 500:
                status = BrokerStatus.SLOW
            else:
                status = BrokerStatus.SLOW

            return BrokerHealth(
                name="OANDA",
                status=status,
                latency_ms=self.latency_ms,
                buying_power=margin_available,
                last_ping=self.last_ping
            )

        except Exception as e:
            if '401' in str(e) or 'Unauthorized' in str(e) or 'authorization' in str(e).lower():
                logger.warning("OANDA API key expired or invalid. Regenerate at https://www.oanda.com/account/api-keys")
            else:
                logger.error(f"OANDA health check failed: {e}")
            return BrokerHealth(
                name="OANDA",
                status=BrokerStatus.DISCONNECTED,
                latency_ms=0,
                buying_power=0,
                last_ping=self.last_ping
            )

    def get_positions(self) -> List[Position]:
        """Fetch all open forex positions from OANDA"""
        positions_list = []

        if not self.api:
            return positions_list

        try:
            endpoint = oanda_positions.OpenPositions(self.account_id)
            response = self.api.request(endpoint)

            for pos in response.get("positions", []):
                instrument = pos.get("instrument", "")
                long_units = float(pos.get("long", {}).get("units", 0))
                short_units = float(pos.get("short", {}).get("units", 0))

                # Process long position
                if long_units > 0:
                    long_data = pos.get("long", {})
                    entry_price = float(long_data.get("averagePrice", 0))
                    unrealized_pl = float(long_data.get("unrealizedPL", 0))

                    # Get current price
                    current_price = self._get_current_price(instrument)

                    positions_list.append(Position(
                        symbol=instrument,
                        broker="OANDA",
                        strategy=self._get_strategy(instrument),
                        side=Side.LONG,
                        entry_price=entry_price,
                        current_price=current_price or entry_price,
                        quantity=long_units,
                        pnl=unrealized_pl,
                        age=self._estimate_position_age(instrument),
                        signal=self._get_signal(instrument)
                    ))

                # Process short position
                if short_units < 0:
                    short_data = pos.get("short", {})
                    entry_price = float(short_data.get("averagePrice", 0))
                    unrealized_pl = float(short_data.get("unrealizedPL", 0))

                    current_price = self._get_current_price(instrument)

                    positions_list.append(Position(
                        symbol=instrument,
                        broker="OANDA",
                        strategy=self._get_strategy(instrument),
                        side=Side.SHORT,
                        entry_price=entry_price,
                        current_price=current_price or entry_price,
                        quantity=abs(short_units),
                        pnl=unrealized_pl,
                        age=self._estimate_position_age(instrument),
                        signal=self._get_signal(instrument)
                    ))

        except Exception as e:
            logger.error(f"Failed to fetch OANDA positions: {e}")

        return positions_list

    def get_margin_available(self) -> float:
        """Get available margin"""
        if not self.api:
            return 0.0

        try:
            endpoint = accounts.AccountSummary(self.account_id)
            response = self.api.request(endpoint)
            return float(response.get("account", {}).get("marginAvailable", 0))
        except Exception as e:
            logger.error(f"Failed to get OANDA margin: {e}")
            return 0.0

    def close_position(self, instrument: str) -> bool:
        """Close a specific forex position"""
        if not self.api:
            return False

        try:
            # Close long position
            data = {"longUnits": "ALL"}
            endpoint = oanda_positions.PositionClose(self.account_id, instrument, data=data)
            try:
                self.api.request(endpoint)
                logger.info(f"Closed OANDA long position: {instrument}")
            except Exception as e:
                logger.debug(f"No long position to close for {instrument}: {e}")

            # Close short position
            data = {"shortUnits": "ALL"}
            endpoint = oanda_positions.PositionClose(self.account_id, instrument, data=data)
            try:
                self.api.request(endpoint)
                logger.info(f"Closed OANDA short position: {instrument}")
            except Exception as e:
                logger.debug(f"No short position to close for {instrument}: {e}")

            return True

        except Exception as e:
            logger.error(f"Failed to close OANDA position {instrument}: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Close all OANDA positions"""
        if not self.api:
            return False

        try:
            positions = self.get_positions()
            for pos in positions:
                self.close_position(pos.symbol)
            return True
        except Exception as e:
            logger.error(f"Failed to close all OANDA positions: {e}")
            return False

    def get_account_info(self) -> dict:
        """Get account summary"""
        if not self.api:
            return {}

        try:
            endpoint = accounts.AccountSummary(self.account_id)
            response = self.api.request(endpoint)
            account = response.get("account", {})

            return {
                "balance": float(account.get("balance", 0)),
                "unrealized_pl": float(account.get("unrealizedPL", 0)),
                "nav": float(account.get("NAV", 0)),
                "margin_used": float(account.get("marginUsed", 0)),
                "margin_available": float(account.get("marginAvailable", 0)),
                "position_value": float(account.get("positionValue", 0)),
                "open_trade_count": int(account.get("openTradeCount", 0))
            }
        except Exception as e:
            logger.error(f"Failed to get OANDA account info: {e}")
            return {}

    def _get_current_price(self, instrument: str) -> Optional[float]:
        """Get current price for an instrument"""
        try:
            from oandapyV20.endpoints import pricing
            params = {"instruments": instrument}
            endpoint = pricing.PricingInfo(self.account_id, params=params)
            response = self.api.request(endpoint)

            prices = response.get("prices", [])
            if prices:
                bid = float(prices[0].get("bids", [{}])[0].get("price", 0))
                ask = float(prices[0].get("asks", [{}])[0].get("price", 0))
                return (bid + ask) / 2
        except Exception as e:
            logger.error(f"Failed to get price for {instrument}: {e}")
        return None

    def _get_strategy(self, instrument: str) -> str:
        """Determine strategy for this instrument"""
        try:
            conn = sqlite3.connect(TRADES_DB)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT strategy FROM trades
                WHERE symbol = ? AND status = 'open' AND platform = 'OANDA'
                ORDER BY entry_time DESC LIMIT 1
            """, (instrument,))
            row = cursor.fetchone()
            conn.close()

            if row and row[0]:
                return row[0]
        except Exception as e:
            logger.debug(f"Could not get strategy for {instrument}: {e}")
        return "Forex"

    def _get_signal(self, instrument: str) -> str:
        """Get signal that opened this position"""
        try:
            conn = sqlite3.connect(TRADES_DB)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT side, notes FROM trades
                WHERE symbol = ? AND status = 'open' AND platform = 'OANDA'
                ORDER BY entry_time DESC LIMIT 1
            """, (instrument,))
            row = cursor.fetchone()
            conn.close()

            if row:
                if row[1]:
                    return row[1][:30]
                elif row[0]:
                    return f"{row[0].upper()} Signal"
        except Exception as e:
            logger.debug(f"Could not get signal for {instrument}: {e}")
        return "MACD Cross"

    def _estimate_position_age(self, instrument: str) -> str:
        """Estimate position age from trades database"""
        try:
            conn = sqlite3.connect(TRADES_DB)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entry_time FROM trades
                WHERE symbol = ? AND status = 'open' AND platform = 'OANDA'
                ORDER BY entry_time DESC LIMIT 1
            """, (instrument,))
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
            logger.debug(f"Could not get position age for {instrument}: {e}")
        return "N/A"
