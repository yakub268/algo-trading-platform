"""
Trading Dashboard V4.2 - Kalshi Broker Service
Connects to existing Kalshi integration for prediction markets
"""
import time
import sqlite3
import logging
import requests
import hashlib
import base64
from datetime import datetime
from typing import List, Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dashboard.config import KALSHI_API_KEY, KALSHI_PRIVATE_KEY_PATH, KALSHI_API_BASE, TRADES_DB
from dashboard.models import Position, BrokerHealth, BrokerStatus, Side

logger = logging.getLogger(__name__)


class KalshiService:
    """Service for Kalshi prediction market integration"""

    def __init__(self):
        self.api_key = KALSHI_API_KEY
        self.private_key_path = KALSHI_PRIVATE_KEY_PATH
        self.base_url = KALSHI_API_BASE
        self.session = requests.Session()
        self.private_key = None
        self.last_ping = None
        self.latency_ms = 0

        self._load_private_key()

    def _load_private_key(self):
        """Load private key for API authentication"""
        try:
            with open(self.private_key_path, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )
        except Exception as e:
            logger.error(f"Failed to load Kalshi private key: {e}")

    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        """Sign a request with the private key using RSA-PSS (Kalshi v2 API format)"""
        if not self.private_key:
            return ""

        # Kalshi v2 API: sign the raw message string directly, not a hash
        message = f"{timestamp}{method}{path}"

        signature = self.private_key.sign(
            message.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH  # Use DIGEST_LENGTH, not MAX_LENGTH
            ),
            hashes.SHA256()
        )

        return base64.b64encode(signature).decode('utf-8')

    def _make_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make authenticated request to Kalshi API"""
        path = f"/trade-api/v2{endpoint}"
        # Handle URL construction - base_url may or may not include /trade-api/v2
        base = self.base_url.rstrip('/')
        if base.endswith('/trade-api/v2'):
            base = base[:-len('/trade-api/v2')]
        url = f"{base}{path}"
        timestamp = str(int(datetime.now().timestamp() * 1000))

        signature = self._sign_request(method.upper(), path, timestamp)

        headers = {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }

        try:
            if method.upper() == "GET":
                response = self.session.get(url, headers=headers, params=data, timeout=10)
            elif method.upper() == "POST":
                response = self.session.post(url, headers=headers, json=data, timeout=10)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, headers=headers, json=data, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Kalshi API request failed: {e}")
            return {}

    def get_health(self) -> BrokerHealth:
        """Check API health and measure latency"""
        try:
            start = time.time()
            result = self._make_request("GET", "/portfolio/balance")
            self.latency_ms = int((time.time() - start) * 1000)
            self.last_ping = datetime.now()

            if result:
                balance = result.get("balance", 0) / 100  # Convert cents to dollars

                if self.latency_ms < 300:
                    status = BrokerStatus.HEALTHY
                elif self.latency_ms < 600:
                    status = BrokerStatus.SLOW
                else:
                    status = BrokerStatus.SLOW

                return BrokerHealth(
                    name="Kalshi",
                    status=status,
                    latency_ms=self.latency_ms,
                    buying_power=balance,
                    last_ping=self.last_ping
                )
        except Exception as e:
            logger.error(f"Kalshi health check failed: {e}")

        return BrokerHealth(
            name="Kalshi",
            status=BrokerStatus.DISCONNECTED,
            latency_ms=0,
            buying_power=0,
            last_ping=self.last_ping
        )

    def get_positions(self) -> List[Position]:
        """Fetch all open Kalshi positions"""
        positions = []
        try:
            result = self._make_request("GET", "/portfolio/positions")
            if not result:
                return positions

            market_positions = result.get("market_positions", [])

            for pos in market_positions:
                ticker = pos.get("ticker", "")
                position_qty = pos.get("position", 0)  # Positive = YES, Negative = NO
                market_exposure = pos.get("market_exposure", 0) / 100  # cents to dollars

                if position_qty == 0:
                    continue

                # Get market details for current price
                market_info = self._get_market_info(ticker)
                current_price = market_info.get("yes_ask", 50) / 100 if position_qty > 0 else market_info.get("no_ask", 50) / 100

                # Estimate entry price from exposure
                entry_price = abs(market_exposure / position_qty) if position_qty != 0 else 0.50

                # Calculate unrealized P&L
                if position_qty > 0:  # YES position
                    pnl = (current_price - entry_price) * abs(position_qty)
                else:  # NO position
                    no_price = 1 - current_price
                    pnl = (no_price - (1 - entry_price)) * abs(position_qty)

                # Determine strategy based on market category
                strategy = self._categorize_market(ticker)

                positions.append(Position(
                    symbol=ticker,
                    broker="Kalshi",
                    strategy=strategy,
                    side=Side.LONG if position_qty > 0 else Side.SHORT,
                    entry_price=entry_price,
                    current_price=current_price,
                    quantity=abs(position_qty),
                    pnl=pnl,
                    age=self._estimate_position_age(ticker),
                    signal="Prob Edge"
                ))

        except Exception as e:
            logger.error(f"Failed to fetch Kalshi positions: {e}")

        return positions

    def get_balance(self) -> float:
        """Get available balance in dollars"""
        try:
            result = self._make_request("GET", "/portfolio/balance")
            if result:
                return result.get("balance", 0) / 100  # cents to dollars
        except Exception as e:
            logger.error(f"Failed to get Kalshi balance: {e}")
        return 0.0

    def close_position(self, ticker: str) -> bool:
        """Close a Kalshi position by selling contracts"""
        try:
            # Get current position
            result = self._make_request("GET", "/portfolio/positions")
            if not result:
                return False

            positions = result.get("market_positions", [])
            pos = next((p for p in positions if p.get("ticker") == ticker), None)

            if not pos or pos.get("position", 0) == 0:
                logger.warning(f"No position found for {ticker}")
                return False

            position_qty = pos.get("position", 0)

            # Submit closing order
            order_data = {
                "ticker": ticker,
                "action": "sell" if position_qty > 0 else "buy",
                "side": "yes" if position_qty > 0 else "no",
                "count": abs(position_qty),
                "type": "market"
            }

            result = self._make_request("POST", "/portfolio/orders", order_data)
            if result.get("order"):
                logger.info(f"Closed Kalshi position: {ticker}")
                return True

        except Exception as e:
            logger.error(f"Failed to close Kalshi position {ticker}: {e}")

        return False

    def _get_market_info(self, ticker: str) -> dict:
        """Get current market info for a ticker"""
        try:
            result = self._make_request("GET", f"/markets/{ticker}")
            return result.get("market", {})
        except Exception as e:
            logger.error(f"Failed to get market info for {ticker}: {e}")
            return {}

    def _categorize_market(self, ticker: str) -> str:
        """Determine strategy based on market ticker pattern"""
        ticker_upper = ticker.upper()
        if "KXTEMP" in ticker_upper or "WEATHER" in ticker_upper:
            return "Weather"
        elif "FED" in ticker_upper or "FOMC" in ticker_upper or "RATE" in ticker_upper:
            return "Fed Bot"
        elif any(sport in ticker_upper for sport in ["NBA", "NFL", "MLB", "NHL"]):
            return "Sports"
        elif "CRYPTO" in ticker_upper or "BTC" in ticker_upper or "ETH" in ticker_upper:
            return "Crypto"
        else:
            return "Prediction"

    def _estimate_position_age(self, ticker: str) -> str:
        """Estimate position age from trades database"""
        try:
            conn = sqlite3.connect(TRADES_DB)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entry_time FROM trades
                WHERE symbol = ? AND status = 'open' AND platform = 'Kalshi'
                ORDER BY entry_time DESC LIMIT 1
            """, (ticker,))
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
            logger.debug(f"Could not get position age for {ticker}: {e}")
        return "N/A"
