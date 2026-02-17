"""
Alpaca API Client

Connects to Alpaca for stocks trading data.
Used for FOMC trading strategy monitoring.

Author: Trading Bot
Created: January 2026
"""

import os
import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AlpacaClient')


class AlpacaClient:
    """
    Client for Alpaca Trading API.

    Uses paper trading endpoint by default.
    """

    PAPER_BASE = "https://paper-api.alpaca.markets"
    LIVE_BASE = "https://api.alpaca.markets"
    DATA_BASE = "https://data.alpaca.markets"

    # =========================================================================
    # FOMC Meeting Dates (rate decision announcement days)
    # =========================================================================
    # HOW TO UPDATE ANNUALLY:
    #   1. Go to https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    #   2. Find the scheduled meeting dates for the new year
    #   3. Add the SECOND day of each 2-day meeting (that's announcement day)
    #   4. Keep previous years for historical reference if desired
    #   5. Dates MUST be sorted chronologically (oldest first)
    #
    # The system will log a warning when the last date is within 60 days,
    # giving you time to add the next year's dates before they run out.
    # =========================================================================
    FOMC_DATES = [
        # --- 2026 ---
        "2026-01-29",  # Jan 28-29
        "2026-03-19",  # Mar 18-19
        "2026-05-07",  # May 6-7
        "2026-06-18",  # Jun 17-18
        "2026-07-30",  # Jul 29-30
        "2026-09-17",  # Sep 16-17
        "2026-11-05",  # Nov 4-5
        "2026-12-17",  # Dec 16-17
        # --- 2027 ---
        # TODO: Add 2027 dates when published at:
        # https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
        # (Usually published in June of the prior year)
    ]

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        paper: bool = True
    ):
        """
        Initialize Alpaca client.

        Args:
            api_key: Alpaca API key (or ALPACA_API_KEY env var)
            api_secret: Alpaca API secret (or ALPACA_SECRET_KEY env var)
            paper: Use paper trading (default True)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY', '')
        self.api_secret = api_secret or os.getenv('ALPACA_SECRET_KEY', os.getenv('ALPACA_API_SECRET', ''))
        self.paper = paper

        self.base_url = self.PAPER_BASE if paper else self.LIVE_BASE

        self.session = self._new_session()

        self._connected = None
        self._last_success = 0  # epoch timestamp of last successful request
        self._reconnect_interval = 300  # re-check connection every 5 min when disconnected

        mode = "paper" if paper else "live"
        logger.info(f"AlpacaClient initialized ({mode})")

    def _new_session(self) -> requests.Session:
        """Create a fresh requests session with auth headers."""
        session = requests.Session()
        session.headers.update({
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        })
        return session

    def _refresh_session(self):
        """Close stale session and create a new one."""
        try:
            self.session.close()
        except Exception:
            pass
        self.session = self._new_session()
        logger.info("Alpaca session refreshed")

    def _request(self, method: str, endpoint: str, base: str = None, **kwargs) -> Optional[Dict]:
        """Make API request with auto-reconnect on connection failures."""
        base = base or self.base_url
        url = f"{base}{endpoint}"

        for attempt in range(2):
            try:
                response = self.session.request(method, url, timeout=10, **kwargs)

                if response.status_code == 200:
                    self._connected = True
                    self._last_success = time.monotonic()
                    return response.json()
                elif response.status_code == 401:
                    logger.warning("Alpaca authentication failed - check API keys")
                    self._connected = False
                    return None
                elif response.status_code == 403:
                    logger.warning("Alpaca access forbidden - subscription issue")
                    self._connected = False
                    return None
                else:
                    logger.warning(f"Alpaca API error: {response.status_code} - {response.text}")
                    return None

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt == 0:
                    logger.info(f"Alpaca connection failed ({type(e).__name__}), refreshing session and retrying...")
                    self._refresh_session()
                    continue
                logger.debug("Alpaca API unreachable after session refresh")
                self._connected = False
                return None
            except Exception as e:
                logger.error(f"Alpaca request error: {e}")
                self._connected = False
                return None

        return None

    def is_connected(self) -> bool:
        """Check if Alpaca is connected, with periodic re-probe when disconnected."""
        now = time.monotonic()
        needs_probe = (
            self._connected is None
            or (not self._connected and now - self._last_success > self._reconnect_interval)
        )
        if needs_probe:
            self.get_account()
        return self._connected or False

    def get_account(self) -> Optional[Dict]:
        """
        Get account information.

        Returns:
            Dict with account data:
            - equity: Total equity
            - cash: Available cash
            - buying_power: Available buying power
            - portfolio_value: Total portfolio value
        """
        return self._request('GET', '/v2/account')

    def get_positions(self) -> List[Dict]:
        """
        Get current positions.

        Returns:
            List of position dicts
        """
        result = self._request('GET', '/v2/positions')
        return result if result else []

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for specific symbol"""
        return self._request('GET', f'/v2/positions/{symbol}')

    def get_orders(self, status: str = 'all', limit: int = 50) -> List[Dict]:
        """
        Get orders.

        Args:
            status: Order status filter (open, closed, all)
            limit: Maximum orders to return

        Returns:
            List of order dicts
        """
        result = self._request('GET', '/v2/orders', params={
            'status': status,
            'limit': limit
        })
        return result if result else []

    def get_activities(self, activity_type: str = None, limit: int = 50) -> List[Dict]:
        """
        Get account activities (trades, dividends, etc).

        Args:
            activity_type: Filter by type (FILL, DIV, etc)
            limit: Maximum activities to return

        Returns:
            List of activity dicts
        """
        params = {'page_size': limit}
        if activity_type:
            params['activity_types'] = activity_type

        result = self._request('GET', '/v2/account/activities', params=params)
        return result if result else []

    def get_clock(self) -> Optional[Dict]:
        """
        Get market clock.

        Returns:
            Dict with:
            - is_open: Whether market is open
            - next_open: Next market open time
            - next_close: Next market close time
        """
        return self._request('GET', '/v2/clock')

    def get_calendar(self, start: str = None, end: str = None) -> List[Dict]:
        """Get market calendar"""
        params = {}
        if start:
            params['start'] = start
        if end:
            params['end'] = end

        result = self._request('GET', '/v2/calendar', params=params)
        return result if result else []

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get latest quote for symbol"""
        return self._request(
            'GET',
            f'/v2/stocks/{symbol}/quotes/latest',
            base=self.DATA_BASE
        )

    def get_fomc_status(self) -> Dict:
        """
        Get FOMC-related status info.

        Returns:
            Dict with:
            - next_meeting: Next FOMC meeting date
            - days_until: Days until next meeting
            - is_fomc_week: Whether we're in FOMC week
            - strategy_active: Whether FOMC strategy should be active
        """
        now = datetime.now(timezone.utc)
        today = now.strftime('%Y-%m-%d')

        # Warn if FOMC dates are about to run out
        if self.FOMC_DATES:
            last_date = datetime.strptime(self.FOMC_DATES[-1], '%Y-%m-%d').replace(tzinfo=timezone.utc)
            days_until_last = (last_date - now).days
            if days_until_last < 0:
                logger.warning(
                    "FOMC dates have EXPIRED! Last date was %s. "
                    "Update FOMC_DATES in alpaca_client.py with next year's schedule from: "
                    "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                    self.FOMC_DATES[-1]
                )
            elif days_until_last < 60:
                logger.warning(
                    "FOMC dates expiring soon! Only %d days until last scheduled date (%s). "
                    "Add next year's dates to FOMC_DATES in alpaca_client.py. "
                    "Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                    days_until_last, self.FOMC_DATES[-1]
                )

        # Find next FOMC date
        next_meeting = None
        for date_str in self.FOMC_DATES:
            if date_str >= today:
                next_meeting = date_str
                break

        if not next_meeting:
            # No more meetings scheduled
            return {
                'next_meeting': None,
                'days_until': None,
                'is_fomc_week': False,
                'strategy_active': False
            }

        meeting_date = datetime.strptime(next_meeting, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        days_until = (meeting_date - now).days

        # FOMC week is the week of the meeting (Mon-Fri)
        meeting_weekday = meeting_date.weekday()  # 0=Monday
        week_start = meeting_date - timedelta(days=meeting_weekday)
        week_end = week_start + timedelta(days=4)

        is_fomc_week = week_start <= now <= week_end + timedelta(days=1)

        # Strategy active 2 days before through meeting day
        strategy_window_start = meeting_date - timedelta(days=2)
        strategy_active = strategy_window_start <= now <= meeting_date + timedelta(days=1)

        return {
            'next_meeting': next_meeting,
            'days_until': days_until,
            'is_fomc_week': is_fomc_week,
            'strategy_active': strategy_active
        }

    def get_spy_position(self) -> Optional[Dict]:
        """Get SPY position specifically for FOMC strategy"""
        return self.get_position('SPY')

    def get_summary(self) -> Dict:
        """
        Get summary of all Alpaca data.

        Returns:
            Dict with all key metrics
        """
        summary = {
            'connected': False,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'paper_mode': self.paper,
            'account': None,
            'positions': [],
            'recent_orders': [],
            'market_status': None,
            'fomc': self.get_fomc_status()
        }

        # Check if API keys are configured
        if not self.api_key or not self.api_secret:
            summary['error'] = 'API keys not configured'
            return summary

        # Get account info
        account = self.get_account()
        if account:
            summary['connected'] = True
            summary['account'] = {
                'equity': float(account.get('equity', 0)),
                'cash': float(account.get('cash', 0)),
                'buying_power': float(account.get('buying_power', 0)),
                'portfolio_value': float(account.get('portfolio_value', 0)),
                'day_trade_count': account.get('daytrade_count', 0)
            }

        # Get positions
        positions = self.get_positions()
        for pos in positions:
            summary['positions'].append({
                'symbol': pos.get('symbol'),
                'qty': float(pos.get('qty', 0)),
                'market_value': float(pos.get('market_value', 0)),
                'cost_basis': float(pos.get('cost_basis', 0)),
                'unrealized_pl': float(pos.get('unrealized_pl', 0)),
                'unrealized_plpc': float(pos.get('unrealized_plpc', 0)),
                'current_price': float(pos.get('current_price', 0))
            })

        # Get recent orders
        orders = self.get_orders(status='closed', limit=10)
        for order in orders:
            summary['recent_orders'].append({
                'symbol': order.get('symbol'),
                'side': order.get('side'),
                'qty': order.get('filled_qty'),
                'filled_avg_price': order.get('filled_avg_price'),
                'status': order.get('status'),
                'filled_at': order.get('filled_at')
            })

        # Get market status
        clock = self.get_clock()
        if clock:
            summary['market_status'] = {
                'is_open': clock.get('is_open'),
                'next_open': clock.get('next_open'),
                'next_close': clock.get('next_close')
            }

        return summary


def main():
    """Test Alpaca client"""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 60)
    print("ALPACA CLIENT TEST")
    print("=" * 60)

    client = AlpacaClient(paper=True)

    print(f"\n[1] Connection: {'OK' if client.is_connected() else 'OFFLINE'}")

    if client.is_connected():
        print("\n[2] Account:")
        account = client.get_account()
        if account:
            print(f"  Equity: ${float(account.get('equity', 0)):,.2f}")
            print(f"  Cash: ${float(account.get('cash', 0)):,.2f}")
            print(f"  Buying Power: ${float(account.get('buying_power', 0)):,.2f}")

        print("\n[3] Positions:")
        positions = client.get_positions()
        if positions:
            for pos in positions:
                symbol = pos.get('symbol')
                pnl = float(pos.get('unrealized_pl', 0))
                print(f"  {symbol}: {pos.get('qty')} shares, P&L: ${pnl:+.2f}")
        else:
            print("  No positions")

        print("\n[4] Market Status:")
        clock = client.get_clock()
        if clock:
            status = "OPEN" if clock.get('is_open') else "CLOSED"
            print(f"  Market: {status}")
    else:
        print("\nAlpaca not connected. Check ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.")

    print("\n[5] FOMC Status:")
    fomc = client.get_fomc_status()
    print(f"  Next Meeting: {fomc.get('next_meeting')}")
    print(f"  Days Until: {fomc.get('days_until')}")
    print(f"  FOMC Week: {fomc.get('is_fomc_week')}")
    print(f"  Strategy Active: {fomc.get('strategy_active')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
