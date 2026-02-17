"""
Trading Dashboard - Coinbase Broker Service
Connects to Coinbase Advanced Trade for crypto positions and balances
"""
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dashboard.models import Position, BrokerHealth, BrokerStatus, Side

logger = logging.getLogger(__name__)


class CoinbaseService:
    """Service for Coinbase broker integration"""

    # Currencies that don't have USD/USDC pairs on Coinbase
    UNSUPPORTED_CURRENCIES = {'NU', 'CGLD', 'REP', 'NMR', 'OXT', 'CVC', 'DNT', 'LOOM'}

    def __init__(self):
        self.api_key = os.getenv('COINBASE_API_KEY', '')
        self.private_key_path = os.path.expanduser(os.getenv('COINBASE_PRIVATE_KEY_PATH', '~/.trading_keys/coinbase_private_key.pem'))
        self.client = None
        self._initialized = False
        self.last_ping = None
        self.latency_ms = 0

        if self.api_key:
            self._init_client()

    def _init_client(self):
        """Initialize Coinbase client using official SDK"""
        try:
            from coinbase.rest import RESTClient

            # Load private key
            private_key = None
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            paths = [
                self.private_key_path,
                os.path.join(base_dir, self.private_key_path),
            ]

            for path in paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        private_key = f.read()
                    break

            if not private_key:
                logger.warning(f"Coinbase private key not found")
                return

            self.client = RESTClient(api_key=self.api_key, api_secret=private_key)
            self._initialized = True
            logger.info("Coinbase client initialized")

        except ImportError:
            logger.error("coinbase-advanced-py not installed")
        except Exception as e:
            logger.error(f"Coinbase client init error: {e}")

    def get_health(self) -> BrokerHealth:
        """Check API health and measure latency"""
        try:
            if not self._initialized:
                return BrokerHealth(
                    name="Coinbase",
                    status=BrokerStatus.DISCONNECTED,
                    latency_ms=0,
                    buying_power=0,
                    last_ping=None
                )

            start = time.time()
            accounts = self.client.get_accounts()
            self.latency_ms = int((time.time() - start) * 1000)
            self.last_ping = datetime.now()

            # Calculate total USD value
            total_usd = self._calculate_total_usd(accounts)

            # Determine status based on latency
            if self.latency_ms < 300:
                status = BrokerStatus.HEALTHY
            elif self.latency_ms < 600:
                status = BrokerStatus.SLOW
            else:
                status = BrokerStatus.SLOW

            return BrokerHealth(
                name="Coinbase",
                status=status,
                latency_ms=self.latency_ms,
                buying_power=total_usd,
                last_ping=self.last_ping
            )
        except Exception as e:
            logger.error(f"Coinbase health check failed: {e}")
            return BrokerHealth(
                name="Coinbase",
                status=BrokerStatus.DISCONNECTED,
                latency_ms=0,
                buying_power=0,
                last_ping=self.last_ping
            )

    def get_positions(self) -> List[Position]:
        """Fetch all non-zero balances from Coinbase as positions"""
        positions = []
        if not self._initialized:
            return positions

        try:
            response = self.client.get_accounts()
            accounts = response.accounts if hasattr(response, 'accounts') else []

            for acc in accounts:
                name = acc.name if hasattr(acc, 'name') else acc.get('name', '')
                bal = acc.available_balance if hasattr(acc, 'available_balance') else acc.get('available_balance', {})
                value = float(bal.value if hasattr(bal, 'value') else bal.get('value', 0))
                currency = bal.currency if hasattr(bal, 'currency') else bal.get('currency', '')

                # Skip zero or tiny balances
                if value < 0.01:
                    continue

                # Skip stablecoins for "positions" view (they're cash)
                if currency in ['USD', 'USDC', 'USDT', 'DAI']:
                    continue

                # Get current price
                current_price = self._get_price(currency)
                usd_value = value * current_price if current_price else 0

                # Skip if USD value is negligible
                if usd_value < 1.0:
                    continue

                positions.append(Position(
                    symbol=f"{currency}-USD",
                    broker="Coinbase",
                    strategy="Crypto Hold",
                    side=Side.LONG,
                    entry_price=0,  # We don't track entry price
                    current_price=current_price,
                    quantity=value,
                    pnl=0,  # Would need trade history
                    age="N/A",
                    signal="Hold"
                ))

        except Exception as e:
            logger.error(f"Failed to fetch Coinbase positions: {e}")

        return positions

    def get_balances(self) -> List[Dict[str, Any]]:
        """Get all account balances (for detailed view)"""
        balances = []
        if not self._initialized:
            return balances

        try:
            response = self.client.get_accounts()
            accounts = response.accounts if hasattr(response, 'accounts') else []

            for acc in accounts:
                name = acc.name if hasattr(acc, 'name') else acc.get('name', '')
                bal = acc.available_balance if hasattr(acc, 'available_balance') else acc.get('available_balance', {})
                value = float(bal.value if hasattr(bal, 'value') else bal.get('value', 0))
                currency = bal.currency if hasattr(bal, 'currency') else bal.get('currency', '')

                if value < 0.001:
                    continue

                current_price = self._get_price(currency) if currency not in ['USD', 'USDC', 'USDT'] else 1.0
                usd_value = value * current_price if current_price else value

                balances.append({
                    'currency': currency,
                    'balance': value,
                    'usd_value': usd_value,
                    'price': current_price
                })

            # Sort by USD value descending
            balances.sort(key=lambda x: x['usd_value'], reverse=True)

        except Exception as e:
            logger.error(f"Failed to fetch Coinbase balances: {e}")

        return balances

    def get_account_info(self) -> Dict[str, Any]:
        """Get account summary info"""
        if not self._initialized:
            return {'connected': False, 'total_usd': 0, 'positions_count': 0}

        try:
            response = self.client.get_accounts()
            accounts = response.accounts if hasattr(response, 'accounts') else []

            total_usd = self._calculate_total_usd_from_accounts(accounts)
            non_zero = sum(1 for acc in accounts if self._get_balance_value(acc) > 0.01)

            return {
                'connected': True,
                'total_usd': total_usd,
                'positions_count': non_zero,
                'accounts_count': len(accounts)
            }
        except Exception as e:
            logger.error(f"Failed to get Coinbase account info: {e}")
            return {'connected': False, 'total_usd': 0, 'positions_count': 0}

    def _get_price(self, currency: str) -> float:
        """Get current price for a currency"""
        if currency in ['USD', 'USDC', 'USDT', 'DAI']:
            return 1.0

        # Skip currencies without valid trading pairs
        if currency in self.UNSUPPORTED_CURRENCIES:
            return 0

        try:
            product_id = f"{currency}-USD"
            response = self.client.get_product(product_id=product_id)
            if response:
                price = response.price if hasattr(response, 'price') else response.get('price', 0)
                return float(price)
        except Exception as e:
            logger.debug(f"Error fetching {currency}-USD price: {e}")

        # Try USDC pair
        try:
            product_id = f"{currency}-USDC"
            response = self.client.get_product(product_id=product_id)
            if response:
                price = response.price if hasattr(response, 'price') else response.get('price', 0)
                return float(price)
        except Exception as e:
            # Add to unsupported list for future runs
            logger.debug(f"Error fetching {currency}-USDC price: {e}")
            self.UNSUPPORTED_CURRENCIES.add(currency)

        return 0

    def _calculate_total_usd(self, response) -> float:
        """Calculate total USD value of all holdings"""
        accounts = response.accounts if hasattr(response, 'accounts') else []
        return self._calculate_total_usd_from_accounts(accounts)

    def _calculate_total_usd_from_accounts(self, accounts) -> float:
        """Calculate total USD value from accounts list"""
        total = 0
        for acc in accounts:
            bal = acc.available_balance if hasattr(acc, 'available_balance') else acc.get('available_balance', {})
            value = float(bal.value if hasattr(bal, 'value') else bal.get('value', 0))
            currency = bal.currency if hasattr(bal, 'currency') else bal.get('currency', '')

            if value < 0.001:
                continue

            price = self._get_price(currency)
            total += value * price

        return total

    def _get_balance_value(self, acc) -> float:
        """Extract balance value from account object"""
        bal = acc.available_balance if hasattr(acc, 'available_balance') else acc.get('available_balance', {})
        return float(bal.value if hasattr(bal, 'value') else bal.get('value', 0))


# Singleton instance
_coinbase_service = None

def get_coinbase_service() -> CoinbaseService:
    """Get or create Coinbase service singleton"""
    global _coinbase_service
    if _coinbase_service is None:
        _coinbase_service = CoinbaseService()
    return _coinbase_service
