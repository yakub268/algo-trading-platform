"""
Kalshi API Client
Wrapper for Kalshi's prediction market trading API

Uses RSA-PSS signature authentication (API Key + Private Key).
"""

import os
import time
import base64
import logging
import threading
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class KalshiClient:
    """Kalshi API client for prediction market trading using RSA authentication"""

    # Shared rate limiter across ALL KalshiClient instances (same API key)
    _rate_lock = threading.Lock()
    _last_request_time_shared = 0.0
    MIN_REQUEST_INTERVAL = 0.15  # 150ms between requests (~6.6 req/s, safe for Kalshi)

    def __init__(
        self,
        api_key_id: str = None,
        private_key_path: str = None,
        api_base: str = None  # Auto-detect from environment
    ):
        """
        Initialize Kalshi client with RSA key authentication.

        Args:
            api_key_id: Kalshi API key ID
            private_key_path: Path to PEM private key file
            api_base: API base URL
        """
        # Check both KALSHI_API_KEY and KALSHI_API_KEY_ID for compatibility
        # (kalshi_bot.py uses KALSHI_API_KEY, some configs use KALSHI_API_KEY_ID)
        self.api_key_id = (
            api_key_id or
            os.getenv("KALSHI_API_KEY") or
            os.getenv("KALSHI_API_KEY_ID")
        )
        self.private_key_path = os.path.expanduser(
            private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH", "~/.trading_keys/kalshi_private_key.pem")
        )

        # Kalshi consolidated to single API (elections URL handles all markets including crypto)
        # trading-api.kalshi.com redirects to api.elections.kalshi.com as of Jan 2026
        if api_base:
            self.api_base = api_base.rstrip('/')
        else:
            self.api_base = os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2").rstrip('/')

        self.session = requests.Session()
        self.private_key = None
        self._initialized = False
        self._last_request_time = 0  # Track last request for rate limiting

        # Try to initialize auth, but don't fail - defer to request time
        if not self.api_key_id:
            logger.warning("Kalshi API key not found. Set KALSHI_API_KEY or KALSHI_API_KEY_ID.")
        else:
            try:
                self.private_key = self._load_private_key()
                self._initialized = True
                logger.debug(f"KalshiClient initialized with API key: {self.api_key_id[:8]}...")
            except FileNotFoundError as e:
                logger.warning(f"{e}")

    def _load_private_key(self):
        """Load RSA private key from file."""
        # Check multiple possible paths
        possible_paths = [
            self.private_key_path,
            f"./{self.private_key_path}",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), self.private_key_path),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                        backend=default_backend()
                    )
                    # Security: Only log key path in debug mode
                    if os.getenv('DEBUG', 'false').lower() == 'true':
                        logger.debug(f"Kalshi: Loaded private key from {path}")
                    return private_key

        raise FileNotFoundError(f"Private key not found. Tried: {possible_paths}")

    def _sign_request(self, method: str, path: str, timestamp: int) -> str:
        """Create RSA-PSS signature for Kalshi API request."""
        message = f"{timestamp}{method}{path}"
        signature = self.private_key.sign(
            message.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

    def _request(self, method: str, endpoint: str, _retry: int = 0, **kwargs) -> Dict:
        """Make authenticated request to Kalshi API with shared rate limiting and 429 retry."""
        # Validate auth is configured before making request
        if not self._initialized:
            raise RuntimeError(
                "KalshiClient not properly initialized. "
                "Check KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH environment variables."
            )

        # Shared rate limiting across all KalshiClient instances
        with KalshiClient._rate_lock:
            elapsed = time.time() - KalshiClient._last_request_time_shared
            if elapsed < self.MIN_REQUEST_INTERVAL:
                time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
            KalshiClient._last_request_time_shared = time.time()

        timestamp = int(time.time() * 1000)
        # Path for signing must include /trade-api/v2 prefix
        path = f"/trade-api/v2{endpoint}"
        signature = self._sign_request(method.upper(), path, timestamp)

        headers = kwargs.pop("headers", {})
        headers.update({
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp),
            "Content-Type": "application/json",
        })

        # URL uses api_base (which now includes /trade-api/v2) + endpoint
        url = f"{self.api_base}{endpoint}"
        kwargs.setdefault('timeout', 30)
        response = self.session.request(method, url, headers=headers, **kwargs)

        # Auto-retry on 429 (rate limit) with exponential backoff
        if response.status_code == 429 and _retry < 3:
            backoff = (2 ** _retry) * 0.5  # 0.5s, 1s, 2s
            logger.warning(f"Kalshi 429 rate limit on {endpoint} — backing off {backoff:.1f}s (retry {_retry + 1}/3)")
            time.sleep(backoff)
            return self._request(method, endpoint, _retry=_retry + 1, **kwargs)

        # Error handling
        if not response.ok:
            try:
                error_body = response.json()
                error_msg = error_body.get('error', {}).get('message', response.text)
                error_code = error_body.get('error', {}).get('code', 'unknown')
            except Exception:
                error_msg = response.text
                error_code = 'parse_failed'

            logger.error(f"Kalshi API Error {response.status_code} [{error_code}]: {error_msg}")
            logger.error(f"  URL: {url}")
            logger.error(f"  Method: {method}")
            if 'json' in kwargs:
                logger.error(f"  Payload: {kwargs['json']}")

            response.raise_for_status()

        return response.json()

    def get_balance(self) -> Dict:
        """Get account balance."""
        return self._request("GET", "/portfolio/balance")

    def get_markets(self, series_ticker: Optional[str] = None, status: str = "open", limit: int = 100) -> List[Dict]:
        """Get available markets."""
        # Kalshi API max limit is ~200
        limit = min(limit, 200)
        params = {'status': status, 'limit': limit}
        if series_ticker:
            params['series_ticker'] = series_ticker

        try:
            data = self._request("GET", "/markets", params=params)
            return data.get('markets', [])
        except requests.exceptions.HTTPError as e:
            if '400' in str(e) and series_ticker:
                # series_ticker may be invalid, retry without it
                logger.warning(f"Kalshi /markets failed with series_ticker={series_ticker}, retrying without filter")
                params.pop('series_ticker', None)
                data = self._request("GET", "/markets", params=params)
                return data.get('markets', [])
            raise

    def get_market(self, ticker: str) -> Dict:
        """Get a specific market by ticker."""
        data = self._request("GET", f"/markets/{ticker}")
        return data.get('market', data)

    def get_series(self, series_ticker: str) -> Optional[Dict]:
        """Get a specific series by ticker. Returns None if not found."""
        try:
            data = self._request("GET", f"/series/{series_ticker}")
            return data.get('series', data)
        except requests.exceptions.HTTPError as e:
            if '400' in str(e) or '404' in str(e):
                logger.debug(f"Series {series_ticker} not found on Kalshi")
                return None
            raise

    def get_events(self, series_ticker: Optional[str] = None, status: str = "open", limit: int = 100) -> List[Dict]:
        """Get events, optionally filtered by series."""
        params = {'status': status, 'limit': limit}
        if series_ticker:
            params['series_ticker'] = series_ticker

        data = self._request("GET", "/events", params=params)
        return data.get('events', [])

    def get_orderbook(self, ticker: str) -> Dict:
        """Get current orderbook for a market."""
        data = self._request("GET", f"/markets/{ticker}/orderbook")
        # Orderbook response has 'yes' and 'no' at top level in v2 API
        if 'orderbook' in data:
            return data['orderbook']
        return data  # Return full response if orderbook key not present

    def create_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        price: int,
        order_type: str = "limit"
    ) -> Dict:
        """
        Create a new order.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            action: "buy" or "sell" (kept for API compatibility but not sent to Kalshi)
            count: Number of contracts
            price: Price in cents (1-99)
            order_type: "limit" or "market"

        Returns:
            Order details dictionary
        """
        # Validate parameters
        if not ticker or not isinstance(ticker, str):
            raise ValueError(f"Invalid ticker: {ticker}")
        if side not in ("yes", "no"):
            raise ValueError(f"Invalid side: {side}. Must be 'yes' or 'no'")
        if action not in ("buy", "sell"):
            raise ValueError(f"Invalid action: {action}. Must be 'buy' or 'sell'")
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"Invalid count: {count}. Must be positive integer")
        if not isinstance(price, int) or price < 1 or price > 99:
            raise ValueError(f"Invalid price: {price}. Must be integer 1-99 cents")

        # Kalshi API v2 payload
        # Required: ticker, side, count, type, action, and yes_price/no_price
        payload = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": order_type,
        }

        if side == "yes":
            payload["yes_price"] = price
        else:
            payload["no_price"] = price

        data = self._request("POST", "/portfolio/orders", json=payload)
        order = data.get('order', data)
        logger.info(f"Kalshi: Order created - {action.upper()} {count} {side.upper()} @ {price}c")
        return order

    def get_positions(self, ticker: Optional[str] = None) -> List[Dict]:
        """Get your current positions."""
        params = {}
        if ticker:
            params['ticker'] = ticker

        data = self._request("GET", "/portfolio/positions", params=params)
        return data.get('market_positions', data.get('positions', []))

    def get_orders(self, ticker: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
        """Get your orders."""
        params = {}
        if ticker:
            params['ticker'] = ticker
        if status:
            params['status'] = status

        data = self._request("GET", "/portfolio/orders", params=params)
        return data.get('orders', [])

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order."""
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

    def get_fills(self, ticker: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get your trade fills."""
        params = {'limit': limit}
        if ticker:
            params['ticker'] = ticker

        data = self._request("GET", "/portfolio/fills", params=params)
        return data.get('fills', [])
