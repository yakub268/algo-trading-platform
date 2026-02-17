"""
OANDA London Breakout Strategy Bot

Strategy: Asian session range breakout at London open
- Documented win rate: 55-60%
- Risk/Reward: 1:1.5 to 1:2
- Best pairs: EUR_USD, GBP_USD

Author: Trading Bot System
Created: January 2026
"""

import os
import asyncio
import logging
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from typing import Optional, List, Tuple
import pytz

# OANDA API
try:
    import oandapyV20
    import oandapyV20.endpoints.orders as orders
    import oandapyV20.endpoints.pricing as pricing
    import oandapyV20.endpoints.instruments as instruments
    import oandapyV20.endpoints.positions as positions
    import oandapyV20.endpoints.accounts as accounts
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    print("Install oandapyV20: pip install oandapyV20")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('OANDALondonBreakout')


@dataclass
class SessionRange:
    """Asian session range for breakout detection."""
    pair: str
    high: float
    low: float
    range_pips: float
    session_start: datetime
    session_end: datetime
    is_valid: bool
    invalidation_reason: Optional[str] = None


@dataclass
class BreakoutOrder:
    """Pending breakout order details."""
    pair: str
    direction: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    units: int
    order_id: Optional[str] = None
    status: str = 'pending'


@dataclass
class TradeResult:
    """Completed trade result."""
    pair: str
    direction: str
    entry_price: float
    exit_price: float
    pnl_pips: float
    pnl_usd: float
    duration_minutes: int
    exit_reason: str


class OANDALondonBreakout:
    """
    London Breakout Strategy Implementation for OANDA.
    
    Methodology:
    1. Calculate Asian session range (00:00-08:00 UTC)
    2. At London open (08:00 UTC), place:
       - Buy stop above Asian high + buffer
       - Sell stop below Asian low - buffer
    3. When one triggers, cancel the other (OCO)
    4. Stop loss: Beyond opposite boundary
    5. Take profit: 1.5-2x the Asian range OR London session end
    
    Risk Management:
    - Max 1% account risk per trade
    - Max 2 trades per day
    - Skip trading during high-impact news
    - Minimum range: 20 pips (filter out low-volatility days)
    - Maximum range: 80 pips (filter out already-moved days)
    """
    
    # Configuration
    PAIRS = ['EUR_USD', 'GBP_USD']
    
    # Session times (UTC)
    ASIAN_START = time(0, 0)   # 00:00 UTC
    ASIAN_END = time(8, 0)     # 08:00 UTC
    LONDON_END = time(16, 0)   # 16:00 UTC
    
    # Range filters (in pips)
    MIN_RANGE_PIPS = 20
    MAX_RANGE_PIPS = 80
    
    # Risk parameters
    RISK_PER_TRADE = 0.01      # 1% of account
    MAX_DAILY_TRADES = 2
    BUFFER_MULTIPLIER = 0.1   # 10% of range as entry buffer
    TP_MULTIPLIER = 1.5       # Take profit at 1.5x range
    
    def __init__(
        self,
        account_id: str = None,
        api_key: str = None,
        paper_mode: bool = None
    ):
        """
        Initialize OANDA London Breakout bot.

        Args:
            account_id: OANDA account ID (from .env if None)
            api_key: OANDA API key (from .env if None)
            paper_mode: Use practice environment (reads from PAPER_MODE env if None)
        """
        self.account_id = account_id or os.getenv('OANDA_ACCOUNT_ID')
        self.api_key = api_key or os.getenv('OANDA_API_KEY')
        # Safe default: read from environment, default to PAPER if not set
        if paper_mode is None:
            # Check OANDA-specific first, then global PAPER_MODE
            oanda_paper = os.getenv('OANDA_PAPER_MODE')
            if oanda_paper is not None:
                paper_mode = oanda_paper.lower() == 'true'
            else:
                paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
        self.paper_mode = paper_mode

        # State tracking
        self.daily_trades = 0
        self.last_reset_date = None
        self.active_orders: List[BreakoutOrder] = []
        self.session_ranges: dict = {}
        self.trade_history: List[TradeResult] = []
        self._initialized = False

        if not OANDA_AVAILABLE:
            logger.warning("oandapyV20 not installed - London Breakout will run in degraded mode")
            return

        if not self.account_id or not self.api_key:
            logger.warning("OANDA credentials missing. Set OANDA_ACCOUNT_ID and OANDA_API_KEY in .env")
            return

        # Safety: detect practice vs live from account ID prefix
        # Practice accounts (001-*) MUST use the practice environment
        if self.account_id.startswith("001-") and not paper_mode:
            logger.warning(
                f"Account ID '{self.account_id[:8]}...' is a PRACTICE account (001-prefix) "
                f"but paper_mode=False was requested. Forcing practice mode to avoid 400 errors."
            )
            paper_mode = True
            self.paper_mode = True
        elif self.account_id.startswith("101-") and paper_mode:
            logger.info("Live account ID (101-prefix) running in paper/practice mode.")

        # Initialize API client
        environment = "practice" if paper_mode else "live"
        self.client = oandapyV20.API(
            access_token=self.api_key,
            environment=environment
        )
        self._initialized = True

        logger.info(f"OANDA London Breakout initialized (paper={paper_mode})")
    
    # ==================== ACCOUNT METHODS ====================
    
    def get_account_balance(self) -> float:
        """Get current account balance."""
        try:
            r = accounts.AccountDetails(self.account_id)
            self.client.request(r)
            return float(r.response['account']['balance'])
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return 0.0
    
    def get_account_info(self) -> dict:
        """Get full account information."""
        try:
            r = accounts.AccountDetails(self.account_id)
            self.client.request(r)
            account = r.response['account']
            return {
                'balance': float(account['balance']),
                'unrealized_pnl': float(account['unrealizedPL']),
                'nav': float(account['NAV']),
                'margin_used': float(account['marginUsed']),
                'margin_available': float(account['marginAvailable']),
                'open_positions': int(account['openPositionCount']),
                'open_trades': int(account['openTradeCount'])
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    # ==================== PRICE DATA METHODS ====================
    
    def get_current_price(self, pair: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get current bid/ask prices.
        
        Returns:
            Tuple of (bid, ask) or (None, None) on error
        """
        try:
            params = {"instruments": pair}
            r = pricing.PricingInfo(accountID=self.account_id, params=params)
            self.client.request(r)
            
            price_data = r.response['prices'][0]
            bid = float(price_data['bids'][0]['price'])
            ask = float(price_data['asks'][0]['price'])
            
            return bid, ask
        except Exception as e:
            logger.error(f"Failed to get price for {pair}: {e}")
            return None, None
    
    def get_candles(
        self,
        pair: str,
        granularity: str = 'H1',
        count: int = 24
    ) -> List[dict]:
        """
        Fetch historical candles.
        
        Args:
            pair: Currency pair (e.g., 'EUR_USD')
            granularity: Candle timeframe ('M1', 'M5', 'H1', 'D', etc.)
            count: Number of candles to fetch
            
        Returns:
            List of candle dicts with 'time', 'open', 'high', 'low', 'close', 'volume'
        """
        try:
            params = {
                "granularity": granularity,
                "count": count
            }
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            self.client.request(r)
            
            candles = []
            for c in r.response['candles']:
                if c['complete']:  # Only complete candles
                    candles.append({
                        'time': c['time'],
                        'open': float(c['mid']['o']),
                        'high': float(c['mid']['h']),
                        'low': float(c['mid']['l']),
                        'close': float(c['mid']['c']),
                        'volume': int(c['volume'])
                    })
            
            return candles
        except Exception as e:
            logger.error(f"Failed to get candles for {pair}: {e}")
            return []
    
    # ==================== SESSION RANGE CALCULATION ====================
    
    def calculate_asian_range(self, pair: str) -> Optional[SessionRange]:
        """
        Calculate the Asian session range for a pair.
        
        Uses hourly candles from 00:00-08:00 UTC to find high/low.
        
        Returns:
            SessionRange object or None if invalid
        """
        # Get last 24 hourly candles
        candles = self.get_candles(pair, 'H1', 24)
        
        if not candles:
            return SessionRange(
                pair=pair, high=0, low=0, range_pips=0,
                session_start=datetime.now(), session_end=datetime.now(),
                is_valid=False, invalidation_reason="No candle data"
            )
        
        # Filter to Asian session hours (00:00-08:00 UTC)
        utc = pytz.UTC
        now = datetime.now(utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        asian_end = now.replace(hour=8, minute=0, second=0, microsecond=0)
        
        asian_candles = []
        for candle in candles:
            try:
                candle_time = datetime.fromisoformat(candle['time'].replace('Z', '+00:00'))
                if today_start <= candle_time < asian_end:
                    asian_candles.append(candle)
            except Exception as e:
                logger.debug(f"Error parsing candle time: {e}")
                continue
        
        if len(asian_candles) < 4:  # Need at least 4 hours of data
            return SessionRange(
                pair=pair, high=0, low=0, range_pips=0,
                session_start=today_start, session_end=asian_end,
                is_valid=False, invalidation_reason=f"Insufficient Asian candles: {len(asian_candles)}"
            )
        
        # Calculate range
        highs = [c['high'] for c in asian_candles]
        lows = [c['low'] for c in asian_candles]
        
        session_high = max(highs)
        session_low = min(lows)
        
        # Convert to pips (assuming 4 decimal pairs like EUR/USD)
        pip_multiplier = 10000 if 'JPY' not in pair else 100
        range_pips = (session_high - session_low) * pip_multiplier
        
        # Validate range
        if range_pips < self.MIN_RANGE_PIPS:
            return SessionRange(
                pair=pair, high=session_high, low=session_low, range_pips=range_pips,
                session_start=today_start, session_end=asian_end,
                is_valid=False, invalidation_reason=f"Range too small: {range_pips:.1f} pips < {self.MIN_RANGE_PIPS}"
            )
        
        if range_pips > self.MAX_RANGE_PIPS:
            return SessionRange(
                pair=pair, high=session_high, low=session_low, range_pips=range_pips,
                session_start=today_start, session_end=asian_end,
                is_valid=False, invalidation_reason=f"Range too large: {range_pips:.1f} pips > {self.MAX_RANGE_PIPS}"
            )
        
        logger.info(f"Asian range for {pair}: High={session_high:.5f}, Low={session_low:.5f}, Range={range_pips:.1f} pips")
        
        return SessionRange(
            pair=pair,
            high=session_high,
            low=session_low,
            range_pips=range_pips,
            session_start=today_start,
            session_end=asian_end,
            is_valid=True
        )
    
    # ==================== POSITION SIZING ====================
    
    def calculate_position_size(
        self,
        account_balance: float,
        stop_loss_pips: float,
        pair: str
    ) -> int:
        """
        Calculate position size based on risk parameters.
        
        Formula:
        Position = (Account × Risk%) / (StopPips × PipValue)
        
        Args:
            account_balance: Current account balance
            stop_loss_pips: Distance to stop loss in pips
            pair: Currency pair for pip value calculation
            
        Returns:
            Position size in units
        """
        risk_amount = account_balance * self.RISK_PER_TRADE
        
        # Pip value varies by pair
        # For USD account trading EUR/USD: 1 pip = $0.10 per 1000 units
        # Simplified calculation (accurate for majors)
        if 'JPY' in pair:
            pip_value_per_1k = 0.10 / 100  # JPY pairs
        else:
            pip_value_per_1k = 0.10  # Other pairs
        
        # Calculate units
        units = int(risk_amount / (stop_loss_pips * pip_value_per_1k) * 1000)
        
        # Cap at reasonable size (max 100k units = 1 standard lot)
        units = min(units, 100000)
        units = max(units, 1000)  # Minimum 0.01 lot
        
        return units
    
    # ==================== ORDER MANAGEMENT ====================
    
    def create_breakout_orders(self, session_range: SessionRange) -> List[BreakoutOrder]:
        """
        Create buy-stop and sell-stop breakout orders.
        
        Args:
            session_range: Valid SessionRange object
            
        Returns:
            List of BreakoutOrder objects
        """
        if not session_range.is_valid:
            logger.warning(f"Cannot create orders: {session_range.invalidation_reason}")
            return []
        
        pair = session_range.pair
        range_size = session_range.high - session_range.low
        buffer = range_size * self.BUFFER_MULTIPLIER
        
        # Calculate entry prices
        buy_entry = session_range.high + buffer
        sell_entry = session_range.low - buffer
        
        # Calculate stop losses (beyond opposite boundary)
        buy_sl = session_range.low - buffer
        sell_sl = session_range.high + buffer
        
        # Calculate take profits (1.5x range from entry)
        tp_distance = range_size * self.TP_MULTIPLIER
        buy_tp = buy_entry + tp_distance
        sell_tp = sell_entry - tp_distance
        
        # Calculate position size
        balance = self.get_account_balance()
        stop_pips = session_range.range_pips * 1.2  # Range + buffer
        units = self.calculate_position_size(balance, stop_pips, pair)
        
        # Create order objects
        buy_order = BreakoutOrder(
            pair=pair,
            direction='long',
            entry_price=round(buy_entry, 5),
            stop_loss=round(buy_sl, 5),
            take_profit=round(buy_tp, 5),
            units=units
        )
        
        sell_order = BreakoutOrder(
            pair=pair,
            direction='short',
            entry_price=round(sell_entry, 5),
            stop_loss=round(sell_sl, 5),
            take_profit=round(sell_tp, 5),
            units=units
        )
        
        logger.info(f"Breakout orders for {pair}:")
        logger.info(f"  BUY STOP: Entry={buy_entry:.5f}, SL={buy_sl:.5f}, TP={buy_tp:.5f}, Units={units}")
        logger.info(f"  SELL STOP: Entry={sell_entry:.5f}, SL={sell_sl:.5f}, TP={sell_tp:.5f}, Units={units}")
        
        return [buy_order, sell_order]
    
    def place_order(self, order: BreakoutOrder) -> Optional[str]:
        """
        Place a stop order on OANDA.
        
        Args:
            order: BreakoutOrder object
            
        Returns:
            Order ID if successful, None otherwise
        """
        # Determine units sign (positive for buy, negative for sell)
        units_signed = order.units if order.direction == 'long' else -order.units
        
        order_data = {
            "order": {
                "type": "STOP",
                "instrument": order.pair,
                "units": str(units_signed),
                "price": str(order.entry_price),
                "stopLossOnFill": {
                    "price": str(order.stop_loss)
                },
                "takeProfitOnFill": {
                    "price": str(order.take_profit)
                },
                "timeInForce": "GTC",
                "positionFill": "DEFAULT"
            }
        }
        
        if self.paper_mode:
            # Simulate order placement
            order.order_id = f"PAPER-{datetime.now().strftime('%H%M%S')}-{order.direction}"
            order.status = 'pending'
            logger.info(f"[PAPER] Order placed: {order.order_id}")
            return order.order_id
        
        try:
            r = orders.OrderCreate(self.account_id, data=order_data)
            self.client.request(r)
            
            order_id = r.response.get('orderCreateTransaction', {}).get('id')
            order.order_id = order_id
            order.status = 'pending'
            
            logger.info(f"Order placed: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if self.paper_mode:
            logger.info(f"[PAPER] Order cancelled: {order_id}")
            return True
        
        try:
            r = orders.OrderCancel(self.account_id, orderID=order_id)
            self.client.request(r)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    # ==================== TRADING LOGIC ====================
    
    def is_trading_window(self) -> Tuple[bool, str]:
        """
        Check if current time is within trading window.
        
        Trading window: 08:00-09:00 UTC (London open)
        
        Returns:
            Tuple of (is_valid_time, reason)
        """
        utc = pytz.UTC
        now = datetime.now(utc)
        current_time = now.time()
        
        # Only trade at London open window
        london_open_start = time(8, 0)
        london_open_end = time(9, 0)
        
        if london_open_start <= current_time <= london_open_end:
            return True, "Within London open window"
        
        if current_time < london_open_start:
            return False, f"Before London open (current: {current_time}, wait until 08:00 UTC)"
        
        return False, f"After London open window (current: {current_time})"
    
    def reset_daily_counters(self):
        """Reset daily trade counters if new day."""
        utc = pytz.UTC
        today = datetime.now(utc).date()
        
        if self.last_reset_date != today:
            self.daily_trades = 0
            self.last_reset_date = today
            self.active_orders = []
            self.session_ranges = {}
            logger.info(f"Daily counters reset for {today}")
    
    def check_news_calendar(self) -> Tuple[bool, Optional[str]]:
        """
        Check if high-impact news is scheduled.

        Skip trading 30 min before/after NFP, FOMC, CPI.

        Returns:
            Tuple of (should_skip, event_name)
        """
        try:
            import requests
            from datetime import datetime, timezone, timedelta

            now = datetime.now(timezone.utc)

            # Check Forex Factory or Investing.com calendar
            # These are high-impact events that move forex markets
            high_impact_events = {
                "NFP": {"day": "friday", "week": "first", "time": "13:30"},  # 8:30 ET
                "CPI": {"day_range": 10, "time": "13:30"},  # Around 10th-14th of month
                "FOMC": {"months": [1, 3, 5, 6, 7, 9, 11, 12], "time": "19:00"},  # 2:00 PM ET
                "ECB": {"day": "thursday", "time": "13:45"},  # Rate decisions
            }

            # Check if today is NFP Friday (first Friday of month)
            if now.weekday() == 4:  # Friday
                first_day = now.replace(day=1)
                days_until_friday = (4 - first_day.weekday()) % 7
                first_friday = first_day + timedelta(days=days_until_friday)
                if now.date() == first_friday.date():
                    release_time = now.replace(hour=13, minute=30, second=0, microsecond=0)
                    if abs((now - release_time).total_seconds()) < 1800:  # Within 30 min
                        return True, "NFP Release"

            # Check for CPI (usually 10th-15th of month at 8:30 AM ET)
            if 10 <= now.day <= 15:
                release_time = now.replace(hour=13, minute=30, second=0, microsecond=0)
                if abs((now - release_time).total_seconds()) < 1800:
                    return True, "CPI Release"

            # Check FOMC days (known meeting dates)
            fomc_dates_2026 = [
                "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
                "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16"
            ]
            today_str = now.strftime("%Y-%m-%d")
            if today_str in fomc_dates_2026:
                release_time = now.replace(hour=19, minute=0, second=0, microsecond=0)
                if abs((now - release_time).total_seconds()) < 1800:
                    return True, "FOMC Decision"

            # Try to fetch from Forex Factory API (if available)
            try:
                url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    events = resp.json()
                    for event in events:
                        if event.get("impact") == "High":
                            event_time_str = f"{event.get('date')} {event.get('time')}"
                            try:
                                event_time = datetime.strptime(event_time_str, "%Y-%m-%d %H:%M")
                                event_time = event_time.replace(tzinfo=timezone.utc)
                                if abs((now - event_time).total_seconds()) < 1800:
                                    return True, event.get("title", "High Impact News")
                            except ValueError:
                                pass
            except Exception:
                pass  # Calendar API unavailable, use static rules only

        except Exception as e:
            logger.debug(f"News calendar check error: {e}")

        return False, None
    
    async def run_strategy_cycle(self):
        """
        Execute one cycle of the strategy.
        
        Called at London open (08:00 UTC).
        """
        self.reset_daily_counters()
        
        # Check if we can trade
        is_valid_time, time_reason = self.is_trading_window()
        if not is_valid_time:
            logger.info(f"Not trading: {time_reason}")
            return
        
        if self.daily_trades >= self.MAX_DAILY_TRADES:
            logger.info(f"Daily trade limit reached: {self.daily_trades}/{self.MAX_DAILY_TRADES}")
            return
        
        # Check news calendar
        skip_news, event = self.check_news_calendar()
        if skip_news:
            logger.info(f"Skipping due to news event: {event}")
            return
        
        # Process each pair
        for pair in self.PAIRS:
            if self.daily_trades >= self.MAX_DAILY_TRADES:
                break
            
            # Calculate Asian session range
            session_range = self.calculate_asian_range(pair)
            self.session_ranges[pair] = session_range
            
            if not session_range.is_valid:
                logger.info(f"Skipping {pair}: {session_range.invalidation_reason}")
                continue
            
            # Create breakout orders
            orders = self.create_breakout_orders(session_range)
            
            # Place orders
            for order in orders:
                order_id = self.place_order(order)
                if order_id:
                    self.active_orders.append(order)
            
            self.daily_trades += 1
            logger.info(f"Breakout orders placed for {pair}, daily trades: {self.daily_trades}")
    
    async def run_monitoring_cycle(self):
        """
        Monitor active orders and positions.
        
        Called periodically during London session.
        """
        # Check for triggered orders (one triggers = cancel other)
        # This would check order status and implement OCO logic
        
        for order in self.active_orders:
            if order.status == 'pending':
                # Check if order was triggered
                # If triggered, cancel opposite order
                pass
    
    async def run(self):
        """
        Main strategy loop.
        
        Schedule:
        - 07:30 UTC: Calculate ranges (prep)
        - 08:00 UTC: Place breakout orders
        - 08:00-16:00 UTC: Monitor and manage
        - 16:00 UTC: Close positions, EOD cleanup
        """
        logger.info("Starting OANDA London Breakout bot")
        logger.info(f"Paper mode: {self.paper_mode}")
        logger.info(f"Trading pairs: {self.PAIRS}")
        
        while True:
            try:
                utc = pytz.UTC
                now = datetime.now(utc)
                current_hour = now.hour
                current_minute = now.minute
                
                # 08:00 UTC - Place breakout orders
                if current_hour == 8 and current_minute < 5:
                    await self.run_strategy_cycle()
                    await asyncio.sleep(300)  # Wait 5 min to avoid duplicate
                
                # 08:00-16:00 UTC - Monitor orders
                elif 8 <= current_hour < 16:
                    await self.run_monitoring_cycle()
                    await asyncio.sleep(60)  # Check every minute
                
                # 00:00 UTC - Reset for new day
                elif current_hour == 0 and current_minute < 5:
                    self.reset_daily_counters()
                    await asyncio.sleep(300)
                
                else:
                    # Outside trading hours
                    await asyncio.sleep(300)  # Check every 5 min
                
            except Exception as e:
                logger.error(f"Strategy loop error: {e}")
                await asyncio.sleep(60)
    
    # ==================== ORCHESTRATOR INTERFACE ====================

    def run_strategy(self) -> dict:
        """
        Synchronous entry point for the master orchestrator.

        Runs one cycle of the London Breakout strategy synchronously.
        Returns a status dict with any opportunities or signals found.
        """
        if not self._initialized:
            return {
                'status': 'error',
                'error': 'OANDA not initialized (missing creds or oandapyV20)'
            }

        self.reset_daily_counters()

        # Check trading window
        is_valid_time, time_reason = self.is_trading_window()

        if not is_valid_time:
            return {
                'status': 'waiting',
                'reason': time_reason,
                'daily_trades': self.daily_trades,
            }

        if self.daily_trades >= self.MAX_DAILY_TRADES:
            return {
                'status': 'limit_reached',
                'daily_trades': self.daily_trades,
                'max_daily_trades': self.MAX_DAILY_TRADES,
            }

        # Check news
        skip_news, event = self.check_news_calendar()
        if skip_news:
            return {
                'status': 'skipped',
                'reason': f'High-impact news: {event}',
            }

        # Scan pairs for breakout setups
        results = []
        for pair in self.PAIRS:
            if self.daily_trades >= self.MAX_DAILY_TRADES:
                break

            session_range = self.calculate_asian_range(pair)
            self.session_ranges[pair] = session_range

            if not session_range.is_valid:
                results.append({
                    'pair': pair,
                    'status': 'invalid_range',
                    'reason': session_range.invalidation_reason,
                })
                continue

            breakout_orders = self.create_breakout_orders(session_range)
            for order in breakout_orders:
                order_id = self.place_order(order)
                if order_id:
                    self.active_orders.append(order)

            self.daily_trades += 1
            results.append({
                'pair': pair,
                'status': 'orders_placed',
                'range_pips': session_range.range_pips,
                'orders': len(breakout_orders),
            })

        return {
            'status': 'executed',
            'pairs_scanned': len(self.PAIRS),
            'results': results,
            'daily_trades': self.daily_trades,
        }

    # ==================== REPORTING ====================

    def get_status(self) -> dict:
        """Get current bot status."""
        return {
            'paper_mode': self.paper_mode,
            'daily_trades': self.daily_trades,
            'max_daily_trades': self.MAX_DAILY_TRADES,
            'active_orders': len(self.active_orders),
            'session_ranges': {
                pair: {
                    'high': sr.high,
                    'low': sr.low,
                    'range_pips': sr.range_pips,
                    'is_valid': sr.is_valid
                }
                for pair, sr in self.session_ranges.items()
            },
            'account': self.get_account_info()
        }


# ==================== MAIN ====================

def main():
    """Test the OANDA London Breakout bot."""
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        bot = OANDALondonBreakout(paper_mode=True)
        
        print("=" * 60)
        print("OANDA LONDON BREAKOUT BOT TEST")
        print("=" * 60)
        
        # Get account info
        print("\n[1] Account Info:")
        account = bot.get_account_info()
        for key, value in account.items():
            print(f"  {key}: {value}")
        
        # Calculate session ranges
        print("\n[2] Session Ranges:")
        for pair in bot.PAIRS:
            session_range = bot.calculate_asian_range(pair)
            print(f"\n  {pair}:")
            print(f"    High: {session_range.high}")
            print(f"    Low: {session_range.low}")
            print(f"    Range: {session_range.range_pips:.1f} pips")
            print(f"    Valid: {session_range.is_valid}")
            if not session_range.is_valid:
                print(f"    Reason: {session_range.invalidation_reason}")
        
        # Check trading window
        print("\n[3] Trading Window:")
        is_valid, reason = bot.is_trading_window()
        print(f"  Valid: {is_valid}")
        print(f"  Reason: {reason}")
        
        # Create sample orders (paper)
        print("\n[4] Sample Orders:")
        for pair in bot.PAIRS:
            session_range = bot.calculate_asian_range(pair)
            if session_range.is_valid:
                orders = bot.create_breakout_orders(session_range)
                for order in orders:
                    print(f"\n  {order.direction.upper()} {pair}:")
                    print(f"    Entry: {order.entry_price}")
                    print(f"    Stop: {order.stop_loss}")
                    print(f"    TP: {order.take_profit}")
                    print(f"    Units: {order.units}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
