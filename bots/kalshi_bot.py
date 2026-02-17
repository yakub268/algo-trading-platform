"""
Kalshi Prediction Markets Trading Bot

Trades prediction markets on Kalshi exchange with focus on:
- Weather events
- Economic data releases
- High-probability events near expiry

Author: Jacob
Created: January 2026
"""

import os
import sys
import time
import logging
import base64
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.kalshi_config import KALSHI_CONFIG, RATE_LIMITS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kalshi.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('KalshiBot')


class KalshiBot:
    """
    Kalshi prediction markets trading bot.

    Features:
    - Authentication and session management
    - Market discovery and filtering
    - Order placement with risk management
    - Position tracking and P&L monitoring
    """

    def __init__(self, config: Dict = None):
        """Initialize the Kalshi bot with configuration."""
        self.config = config or KALSHI_CONFIG
        self.base_url = self.config["base_url"]
        self.session = requests.Session()
        self.api_key_id = self.config["api_key_id"]
        self.private_key = self._load_private_key()
        self.paper_mode = self.config["paper_mode"]
        self.daily_pnl = 0.0
        self.positions: List[Dict] = []

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0 / RATE_LIMITS["requests_per_second"]

    def _load_private_key(self):
        """Load RSA private key from file."""
        key_path = self.config["private_key_path"]
        # Check multiple possible locations
        possible_paths = [
            key_path,
            f"/app/{key_path}",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), key_path),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                with open(path, "rb") as key_file:
                    private_key = serialization.load_pem_private_key(
                        key_file.read(),
                        password=None,
                        backend=default_backend()
                    )
                    # Security: Only log key path in debug mode
                    if os.getenv('DEBUG', 'false').lower() == 'true':
                        logger.debug(f"Loaded private key from {path}")
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

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make rate-limited request to Kalshi API with RSA signature."""
        self._rate_limit()

        timestamp = int(time.time() * 1000)  # milliseconds
        path = f"/trade-api/v2{endpoint}"
        signature = self._sign_request(method.upper(), path, timestamp)

        headers = kwargs.pop("headers", {})
        headers.update({
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp),
            "Content-Type": "application/json",
        })

        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, headers=headers, **kwargs)

        response.raise_for_status()
        return response.json()

    def verify_connection(self) -> bool:
        """
        Verify API connection by fetching account balance.

        Returns:
            bool: True if connection successful
        """
        try:
            balance = self.get_balance()
            if balance is not None:
                logger.info(f"Successfully connected to Kalshi. Balance: ${balance:.2f}")
                return True
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection verification failed: {e}")
            return False

    def get_markets(self, status: str = "open", limit: int = 100) -> List[Dict]:
        """
        Fetch available markets from Kalshi.

        Args:
            status: Market status filter ("open", "closed", etc.)
            limit: Maximum number of markets to return

        Returns:
            List of market dictionaries
        """
        try:
            data = self._request(
                "GET",
                "/markets",
                params={"status": status, "limit": limit}
            )
            return data.get("markets", [])

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    def get_market(self, ticker: str) -> Optional[Dict]:
        """
        Get details for a specific market.

        Args:
            ticker: Market ticker symbol

        Returns:
            Market details dictionary or None
        """
        try:
            data = self._request("GET", f"/markets/{ticker}")
            return data.get("market")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch market {ticker}: {e}")
            return None

    def get_orderbook(self, ticker: str) -> Optional[Dict]:
        """
        Get orderbook for a specific market.

        Args:
            ticker: Market ticker symbol

        Returns:
            Orderbook data or None
        """
        try:
            data = self._request("GET", f"/markets/{ticker}/orderbook")
            return data.get("orderbook")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch orderbook for {ticker}: {e}")
            return None

    def place_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        price: int,
        order_type: str = "limit"
    ) -> Optional[Dict]:
        """
        Place an order on Kalshi.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            quantity: Number of contracts
            price: Price in cents (1-99)
            order_type: "limit" or "market"

        Returns:
            Order confirmation or None
        """
        # Check daily loss limit
        if self.daily_pnl <= -self.config["daily_loss_limit"]:
            logger.warning("Daily loss limit reached, not placing order")
            return None

        # Paper mode simulation
        if self.paper_mode:
            logger.info(f"[PAPER] Order: {side} {quantity}x {ticker} @ {price}c")
            return {
                "paper": True,
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
                "price": price,
                "timestamp": datetime.now().isoformat()
            }

        try:
            order_data = {
                "ticker": ticker,
                "side": side,
                "count": quantity,
                "type": order_type,
            }

            if side == "yes":
                order_data["yes_price"] = price
            else:
                order_data["no_price"] = price

            data = self._request(
                "POST",
                "/portfolio/orders",
                json=order_data
            )

            logger.info(f"Order placed: {side} {quantity}x {ticker} @ {price}c")
            return data.get("order")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def get_positions(self) -> List[Dict]:
        """
        Get current positions.

        Returns:
            List of position dictionaries
        """
        try:
            data = self._request("GET", "/portfolio/positions")
            self.positions = data.get("market_positions", [])
            return self.positions

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def get_balance(self) -> Optional[float]:
        """
        Get current account balance.

        Returns:
            Balance in dollars or None
        """
        try:
            data = self._request("GET", "/portfolio/balance")
            return data.get("balance", 0) / 100  # Convert cents to dollars

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch balance: {e}")
            return None

    def filter_markets(self, markets: List[Dict]) -> List[Dict]:
        """
        Filter markets based on strategy criteria.

        Filters:
        - Target series_ticker patterns (Kalshi uses series_ticker, not category)
        - Time to expiry constraints
        - Minimum liquidity

        Args:
            markets: List of market dictionaries

        Returns:
            Filtered list of markets
        """
        filtered = []
        now = datetime.now(timezone.utc)

        for market in markets:
            # Check event_ticker, ticker, and title for pattern matching
            event_ticker = market.get("event_ticker", "").upper()
            ticker = market.get("ticker", "").upper()
            title = market.get("title", "").upper()
            searchable = f"{event_ticker} {ticker} {title}"

            # Match against target patterns from config
            target_patterns = self.config.get("target_series_patterns", [])
            if target_patterns and not any(pat.upper() in searchable for pat in target_patterns):
                continue

            # Check time to expiry (prefer expected_expiration_time as it's when market settles)
            expiry_str = market.get("expected_expiration_time") or market.get("close_time")
            if expiry_str:
                try:
                    expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                    hours_to_expiry = (expiry - now).total_seconds() / 3600

                    if hours_to_expiry < self.config["min_time_to_expiry_hours"]:
                        continue
                    if hours_to_expiry > self.config["max_time_to_expiry_days"] * 24:
                        continue

                except (ValueError, TypeError):
                    continue

            # Check minimum volume/liquidity (lowered for new markets)
            volume = market.get("volume", 0) or market.get("volume_24h", 0)
            # Skip volume check for now - many good markets have low initial volume
            # if volume < 10:  # Reduced from 100 to 10
            #     continue

            filtered.append(market)

        return filtered

    def evaluate_opportunity(self, market: Dict) -> Optional[Dict]:
        """
        Evaluate a market for trading opportunity.

        Looks for mispriced markets where our estimate
        differs from implied odds by minimum edge.

        Args:
            market: Market dictionary

        Returns:
            Trade signal dictionary or None
        """
        ticker = market.get("ticker")
        series_ticker = market.get("series_ticker", "").upper()
        title = market.get("title", "")
        yes_bid = market.get("yes_bid", 0)
        yes_ask = market.get("yes_ask", 100)
        no_bid = market.get("no_bid", 0)
        no_ask = market.get("no_ask", 100)

        min_edge = self.config["min_probability_edge"]

        # Estimate true probability based on market type
        our_probability = self._estimate_probability(market)

        if our_probability is None:
            return None

        # Calculate market implied probability
        market_prob = yes_ask / 100  # Market thinks YES is worth this

        # Calculate edge
        edge = our_probability - market_prob

        signal = None

        # Buy YES if we think probability is higher than market
        if edge >= min_edge and yes_ask < 90:  # Raised from 85 to find more opportunities
            signal = {
                "ticker": ticker,
                "side": "yes",
                "action": "buy",
                "price": yes_ask,
                "our_probability": our_probability,
                "market_probability": market_prob,
                "edge": edge,
                "reasoning": f"Edge: {edge*100:.1f}% on YES"
            }

        # Buy NO if we think probability is lower than market
        elif edge <= -min_edge and no_ask < 90:  # Raised from 85 to find more opportunities
            signal = {
                "ticker": ticker,
                "side": "no",
                "action": "buy",
                "price": no_ask,
                "our_probability": 1 - our_probability,
                "market_probability": 1 - market_prob,
                "edge": abs(edge),
                "reasoning": f"Edge: {abs(edge)*100:.1f}% on NO"
            }

        # Calculate quantity based on max position size and price
        if signal is not None:
            max_position_usd = self.config.get('max_position_size', 15)
            price_per_contract = signal['price'] / 100  # Convert cents to dollars
            quantity = int(max_position_usd / price_per_contract)
            signal['quantity'] = max(1, quantity)  # At least 1 contract

        return signal

    def _estimate_probability(self, market: Dict) -> Optional[float]:
        """
        Estimate true probability for a market using external data.

        Args:
            market: Market dictionary

        Returns:
            Probability estimate (0-1) or None if unable to estimate
        """
        series_ticker = market.get("series_ticker", "").upper()
        title = market.get("title", "").lower()

        try:
            # Weather markets - use NWS data
            if "KXTEMP" in series_ticker or "weather" in title:
                return self._estimate_weather_probability(market)

            # Fed/economic markets - use CME FedWatch
            if "FOMC" in series_ticker or "fed" in title or "rate" in title:
                return self._estimate_fed_probability(market)

            # Economic data markets
            if any(x in series_ticker for x in ["KXCPI", "KXGDP", "KXJOBS", "KXUNEMP"]):
                return self._estimate_economic_probability(market)

            # Crypto price markets
            if "KXBTC" in series_ticker or "KXETH" in series_ticker:
                return self._estimate_crypto_probability(market)

        except Exception as e:
            logger.debug(f"Error estimating probability for {market.get('ticker')}: {e}")

        return None

    def _estimate_weather_probability(self, market: Dict) -> Optional[float]:
        """Estimate weather probability from NWS data."""
        try:
            import requests
            import re

            title = market.get("title", "")
            # Parse temperature threshold from title (e.g., "NYC High >= 75F")
            match = re.search(r'(High|Low)\s*([<>=]+)\s*(\d+)', title)
            if not match:
                return None

            # Get city from title
            cities = {"NYC": (40.7128, -74.0060), "LA": (34.0522, -118.2437),
                      "Chicago": (41.8781, -87.6298), "Miami": (25.7617, -80.1918)}

            city = None
            for c in cities:
                if c.lower() in title.lower():
                    city = c
                    break

            if not city:
                return None

            lat, lon = cities[city]
            temp_threshold = int(match.group(3))
            operator = match.group(2)

            # Fetch NWS forecast
            nws_url = f"https://api.weather.gov/points/{lat},{lon}"
            resp = requests.get(nws_url, timeout=10)
            if resp.status_code == 200:
                forecast_url = resp.json()["properties"]["forecast"]
                forecast_resp = requests.get(forecast_url, timeout=10)
                if forecast_resp.status_code == 200:
                    periods = forecast_resp.json()["properties"]["periods"]
                    if periods:
                        forecast_temp = periods[0]["temperature"]
                        diff = forecast_temp - temp_threshold

                        # Simple probability model
                        if ">=" in operator or ">" in operator:
                            if diff >= 10: return 0.9
                            elif diff >= 5: return 0.75
                            elif diff >= 0: return 0.6
                            elif diff >= -5: return 0.4
                            else: return 0.2
                        else:
                            if diff <= -10: return 0.9
                            elif diff <= -5: return 0.75
                            elif diff <= 0: return 0.6
                            elif diff <= 5: return 0.4
                            else: return 0.2

        except Exception as e:
            logger.debug(f"Weather probability estimation error: {e}")
        return None

    def _estimate_fed_probability(self, market: Dict) -> Optional[float]:
        """Estimate Fed decision probability from CME FedWatch."""
        try:
            import requests
            url = "https://www.cmegroup.com/services/fedWatchTool/v1/fedWatchTool"
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                data = resp.json()
                meetings = data.get("meetings", [])
                if meetings:
                    # Return probability of no change for nearest meeting
                    probs = meetings[0].get("probabilities", {})
                    return probs.get("0", 50) / 100  # 0 = no change
        except Exception as e:
            logger.debug(f"Fed probability estimation error: {e}")
        return None

    def _estimate_economic_probability(self, market: Dict) -> Optional[float]:
        """Estimate economic data probability."""
        # This would integrate with nowcasting models (Atlanta Fed GDPNow, Cleveland Fed CPI)
        # For now, return None to skip these markets
        return None

    def _estimate_crypto_probability(self, market: Dict) -> Optional[float]:
        """Estimate crypto price level probability using technical analysis."""
        try:
            import requests
            import re

            title = market.get("title", "")

            # Parse price threshold
            match = re.search(r'(BTC|Bitcoin|ETH|Ethereum).*?([<>=]+)\s*\$?([\d,]+)', title, re.IGNORECASE)
            if not match:
                return None

            crypto = match.group(1).upper()
            operator = match.group(2)
            threshold = float(match.group(3).replace(",", ""))

            # Get current price from CoinGecko
            coin_id = "bitcoin" if crypto in ["BTC", "BITCOIN"] else "ethereum"
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
            resp = requests.get(url, timeout=10)

            if resp.status_code == 200:
                price = resp.json()[coin_id]["usd"]
                pct_diff = (price - threshold) / threshold

                # Simple probability model based on distance from threshold
                if ">=" in operator or ">" in operator:
                    if pct_diff >= 0.1: return 0.85
                    elif pct_diff >= 0.05: return 0.7
                    elif pct_diff >= 0: return 0.55
                    elif pct_diff >= -0.05: return 0.4
                    else: return 0.2
                else:
                    if pct_diff <= -0.1: return 0.85
                    elif pct_diff <= -0.05: return 0.7
                    elif pct_diff <= 0: return 0.55
                    elif pct_diff <= 0.05: return 0.4
                    else: return 0.2

        except Exception as e:
            logger.debug(f"Crypto probability estimation error: {e}")
        return None

    def run_strategy(self, markets: List[Dict] = None) -> List[Dict]:
        """
        Run strategy on available markets.

        Args:
            markets: List of market dictionaries (optional, will fetch if not provided)

        Returns:
            List of trade signals
        """
        signals = []

        # If no markets provided, fetch them
        if markets is None:
            try:
                markets = self.get_markets()
                logger.info(f"Fetched {len(markets)} markets from Kalshi")
            except Exception as e:
                logger.error(f"Failed to fetch markets: {e}")
                return []

        # Filter to relevant markets
        filtered = self.filter_markets(markets)
        logger.info(f"Filtered to {len(filtered)} relevant markets")

        # Evaluate each market
        for market in filtered:
            signal = self.evaluate_opportunity(market)
            if signal:
                signals.append(signal)

        return signals

    MAX_CYCLE_SECONDS = 90  # Return early before orchestrator's 120s kill

    def run(self):
        """
        Run a single scan cycle.

        The orchestrator calls this method on schedule, so we run
        ONE cycle and return (no internal loop).
        """
        cycle_start = time.time()
        logger.info("Running Kalshi bot scan cycle...")
        logger.info(f"Paper mode: {self.paper_mode}")

        # Verify connection
        if not self.verify_connection():
            logger.error("Failed to connect to Kalshi")
            return

        try:
            # Get markets
            markets = self.get_markets()
            logger.info(f"Fetched {len(markets)} open markets")

            elapsed = time.time() - cycle_start
            if elapsed > self.MAX_CYCLE_SECONDS:
                logger.warning(f"Kalshi cycle timeout after get_markets ({elapsed:.0f}s)")
                return

            # Run strategy
            signals = self.run_strategy(markets)

            elapsed = time.time() - cycle_start
            if elapsed > self.MAX_CYCLE_SECONDS:
                logger.warning(f"Kalshi cycle timeout after run_strategy ({elapsed:.0f}s)")
                return

            # Execute signals
            for signal in signals:
                self.place_order(
                    ticker=signal["ticker"],
                    side=signal["side"],
                    quantity=signal["quantity"],
                    price=signal["price"]
                )

            # Log positions
            elapsed = time.time() - cycle_start
            if elapsed < self.MAX_CYCLE_SECONDS:
                positions = self.get_positions()
                logger.info(f"Current positions: {len(positions)}")

            total = time.time() - cycle_start
            logger.info(f"Scan cycle complete ({total:.1f}s)")

        except Exception as e:
            logger.error(f"Error in scan cycle: {e}")


def main():
    """Entry point for the Kalshi bot (single scan cycle)."""
    bot = KalshiBot()
    bot.run()


def main_loop():
    """Entry point for standalone Docker operation (continuous loop)."""
    import time
    import os

    interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
    bot = KalshiBot()
    logger.info(f"Starting Kalshi bot in continuous mode (interval: {interval}s)")

    while True:
        try:
            bot.run()
        except Exception as e:
            logger.error(f"Scan cycle failed: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        main()  # Single cycle for orchestrator
    else:
        main_loop()  # Continuous loop for Docker
