"""
Freqtrade API Client

Connects to Freqtrade REST API for crypto trading data.
Default endpoint: localhost:8080/api/v1

Author: Trading Bot
Created: January 2026
"""

import logging
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FreqtradeClient')


class FreqtradeClient:
    """
    Client for Freqtrade REST API.

    Freqtrade must be running with --api-server flag.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        username: str = "freqtrade",
        password: str = "freqtrade"
    ):
        """
        Initialize Freqtrade client.

        Args:
            host: Freqtrade host
            port: Freqtrade API port
            username: API username
            password: API password
        """
        self.base_url = f"http://{host}:{port}/api/v1"
        self.session = requests.Session()
        self.session.auth = (username, password)
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        self._connected = None

        logger.info(f"FreqtradeClient initialized: {self.base_url}")

    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.request(method, url, timeout=10, **kwargs)

            if response.status_code == 200:
                self._connected = True
                return response.json()
            elif response.status_code == 401:
                logger.warning("Freqtrade authentication failed")
                self._connected = False
                return None
            else:
                logger.warning(f"Freqtrade API error: {response.status_code}")
                return None

        except requests.exceptions.ConnectionError:
            logger.debug("Freqtrade not running or unreachable")
            self._connected = False
            return None
        except Exception as e:
            logger.error(f"Freqtrade request error: {e}")
            self._connected = False
            return None

    def is_connected(self) -> bool:
        """Check if Freqtrade is connected"""
        if self._connected is None:
            self.ping()
        return self._connected or False

    def ping(self) -> bool:
        """Ping Freqtrade API"""
        result = self._request('GET', '/ping')
        return result is not None

    def get_status(self) -> List[Dict]:
        """
        Get open trades status.

        Returns:
            List of open trade dicts
        """
        result = self._request('GET', '/status')
        return result if result else []

    def get_profit(self) -> Optional[Dict]:
        """
        Get profit summary.

        Returns:
            Dict with profit stats:
            - profit_closed_coin: Total closed profit in base currency
            - profit_closed_percent: Total closed profit %
            - profit_all_coin: All profit including open
            - trade_count: Number of trades
            - winning_trades: Number of winning trades
            - losing_trades: Number of losing trades
        """
        return self._request('GET', '/profit')

    def get_trades(self, limit: int = 50) -> Dict:
        """
        Get trade history.

        Args:
            limit: Maximum trades to return

        Returns:
            Dict with 'trades' list and pagination info
        """
        result = self._request('GET', '/trades', params={'limit': limit})
        return result if result else {'trades': [], 'trades_count': 0}

    def get_balance(self) -> Optional[Dict]:
        """
        Get account balance.

        Returns:
            Dict with currency balances
        """
        return self._request('GET', '/balance')

    def get_performance(self) -> List[Dict]:
        """
        Get per-pair performance stats.

        Returns:
            List of dicts with pair performance:
            - pair: Trading pair
            - profit: Total profit
            - count: Number of trades
        """
        result = self._request('GET', '/performance')
        return result if result else []

    def get_daily(self, days: int = 7) -> Optional[Dict]:
        """
        Get daily profit/loss.

        Args:
            days: Number of days to return

        Returns:
            Dict with daily data
        """
        return self._request('GET', '/daily', params={'timescale': days})

    def get_stats(self) -> Optional[Dict]:
        """
        Get overall trading stats.

        Returns:
            Dict with various stats
        """
        return self._request('GET', '/stats')

    def get_show_config(self) -> Optional[Dict]:
        """Get current Freqtrade configuration"""
        return self._request('GET', '/show_config')

    def get_summary(self) -> Dict:
        """
        Get summary of all Freqtrade data.

        Returns:
            Dict with all key metrics
        """
        summary = {
            'connected': self.is_connected(),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'open_trades': [],
            'profit': None,
            'balance': None,
            'performance': [],
            'recent_trades': [],
            'stats': None
        }

        if not summary['connected']:
            return summary

        # Get all data
        summary['open_trades'] = self.get_status()
        summary['profit'] = self.get_profit()
        summary['balance'] = self.get_balance()
        summary['performance'] = self.get_performance()

        trades_data = self.get_trades(limit=20)
        summary['recent_trades'] = trades_data.get('trades', [])

        summary['stats'] = self.get_stats()

        return summary


def main():
    """Test Freqtrade client"""
    print("=" * 60)
    print("FREQTRADE CLIENT TEST")
    print("=" * 60)

    client = FreqtradeClient()

    print(f"\n[1] Connection: {'OK' if client.is_connected() else 'OFFLINE'}")

    if client.is_connected():
        print("\n[2] Open Trades:")
        status = client.get_status()
        if status:
            for trade in status:
                print(f"  {trade.get('pair')}: {trade.get('current_profit', 0):.2%}")
        else:
            print("  No open trades")

        print("\n[3] Profit Summary:")
        profit = client.get_profit()
        if profit:
            print(f"  Closed P&L: {profit.get('profit_closed_coin', 0):.4f}")
            print(f"  Win Rate: {profit.get('winning_trades', 0)}/{profit.get('trade_count', 0)}")

        print("\n[4] Performance by Pair:")
        perf = client.get_performance()
        for p in perf[:5]:
            print(f"  {p.get('pair')}: {p.get('profit', 0):.4f} ({p.get('count')} trades)")
    else:
        print("\nFreqtrade is not running. Start with: freqtrade trade --api-server")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
