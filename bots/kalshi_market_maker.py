"""
Kalshi Market Maker Bot

Provides liquidity to Kalshi markets and earns the bid-ask spread.
Kalshi has ZERO maker fees - makers pay nothing, only takers pay fees.
This makes market making very profitable on Kalshi.

Strategy:
1. Quote both sides (bid and ask) on selected markets
2. Earn the spread when orders fill on both sides
3. Manage inventory to stay delta-neutral
4. Adjust quotes based on volatility and position

Key Concepts:
- Spread: Difference between bid and ask (your profit)
- Inventory: Net position (want to stay near zero)
- Skew: Adjust prices to reduce inventory risk
- Maker Rebate: Kalshi pays makers 0% fees (huge advantage!)

V2 Changes (Feb 2026):
- Wired to KalshiInfrastructure (persistent positions, real fills, risk manager)
- Removed simulate_fills() — uses real fill tracking from API
- Added place_order() for orchestrator execution path
- Order lifecycle: cancel stale quotes >60s
- Tightened risk: MAX_POSITION 50→25, QUOTE_SIZE 5→3
- Spread floor by market type (weather wider, economics tighter)

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import statistics

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KalshiMarketMaker')


@dataclass
class Quote:
    """Represents a two-sided quote"""
    ticker: str
    bid_price: int  # cents
    bid_size: int
    ask_price: int  # cents
    ask_size: int
    spread: int  # cents
    mid_price: int  # cents
    timestamp: datetime


@dataclass
class RiskLimits:
    """Risk control parameters"""
    max_position: int  # Max contracts in any direction
    max_daily_loss: float  # Max daily loss in dollars
    max_order_size: int  # Max contracts per order
    min_spread: int  # Minimum spread in cents


# Spread floors by market type prefix
SPREAD_FLOORS = {
    'KXHIGH': 5,   # Weather: volatile near expiry, wider spread
    'KXLOW': 5,    # Weather
    'KXTEMP': 5,   # Weather
    'KXBTC': 4,    # Crypto: moderate volatility
    'KXETH': 4,    # Crypto
    'KXFED': 3,    # Economics: less volatile
    'KXCPI': 3,    # Economics
    'KXGDP': 3,    # Economics
    'KXUNEMP': 3,  # Economics
}

def _get_spread_floor(ticker: str) -> int:
    """Get minimum spread for a market based on its type."""
    ticker_upper = ticker.upper()
    for prefix, floor in SPREAD_FLOORS.items():
        if ticker_upper.startswith(prefix):
            return floor
    return 4  # Default


class KalshiMarketMaker:
    """
    Automated market maker for Kalshi.

    Core Logic:
    1. Calculate fair value from orderbook mid-price
    2. Set bid/ask around fair value with target spread
    3. Skew quotes based on inventory (lower bid if long, raise ask if short)
    4. Manage risk by limiting inventory and daily losses

    Kalshi Advantage:
    - 0% maker fees! Only takers pay fees
    - This means your entire spread is profit
    """

    # Default parameters (tightened for $200-500 capital)
    BASE_SPREAD = 4  # 4 cents (4%)
    MAX_POSITION = 25  # Max 25 contracts (down from 50)
    SKEW_FACTOR = 1  # 1 cent skew per 10 contracts
    QUOTE_SIZE = 3  # 3 contracts per quote (down from 5)
    REFRESH_INTERVAL = 10  # Refresh quotes every 10 seconds
    STALE_ORDER_SECONDS = 60  # Cancel unfilled orders after 60s

    def __init__(self, paper_mode: bool = True, infrastructure=None):
        """
        Initialize market maker bot.

        Args:
            paper_mode: If True, simulate trading
            infrastructure: KalshiInfrastructure instance (injected by orchestrator)
        """
        self.paper_mode = paper_mode
        self.infrastructure = infrastructure

        # Initialize Kalshi client
        if infrastructure and hasattr(infrastructure, 'client'):
            self.client = infrastructure.client
            self._connected = True
        else:
            try:
                self.client = KalshiClient()
                self._connected = True
                logger.info("Kalshi client connected")
            except Exception as e:
                logger.warning(f"Kalshi client not available: {e}")
                self.client = None
                self._connected = False

        # Risk limits
        self.risk_limits = RiskLimits(
            max_position=self.MAX_POSITION,
            max_daily_loss=20.0,  # $20 max daily loss (tightened from $50)
            max_order_size=10,
            min_spread=2  # 2 cents minimum
        )

        # Markets we're making
        self._markets: Dict[str, Dict] = {}

        # Current quotes (ticker -> Quote)
        self._quotes: Dict[str, Quote] = {}

        # Active orders (order_id -> {ticker, side, action, price, size, placed_at})
        self._orders: Dict[str, Dict] = {}

        # In-memory inventory fallback (used when no infrastructure)
        self._inventory: Dict[str, Dict] = {}

        # Daily P&L tracking (in-memory, synced from infrastructure if available)
        self._daily_pnl = 0.0
        self._daily_start = datetime.now(timezone.utc).date()
        self._fills_today = 0

        # Price history for volatility (ticker -> deque of mid prices)
        self._price_history: Dict[str, deque] = {}

        # Running state
        self._running = False

        # Database for quote logging
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'kalshi_market_maker.db'
        )
        self._init_database()

        logger.info(f"Kalshi Market Maker initialized (paper_mode={paper_mode}, infrastructure={'yes' if infrastructure else 'no'})")

    def _init_database(self):
        """Initialize SQLite database for quote logging."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ticker TEXT,
                bid_price INTEGER,
                ask_price INTEGER,
                spread INTEGER,
                mid_price INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE,
                spread_pnl REAL,
                inventory_pnl REAL,
                total_pnl REAL,
                num_fills INTEGER
            )
        ''')

        conn.commit()
        conn.close()

    # ============== Market Management ==============

    def add_market(self, ticker: str) -> bool:
        """Add a market to make."""
        if self.client and self._connected:
            try:
                market = self.client.get_market(ticker)
            except Exception as e:
                logger.error(f"Could not fetch market {ticker}: {e}")
                market = None
        else:
            market = self._get_mock_market(ticker)

        if not market:
            logger.error(f"Could not find market: {ticker}")
            return False

        status = market.get('status', 'open')
        if status not in ('open', 'active'):
            logger.warning(f"Market is not open: {ticker} (status: {status})")
            return False

        self._markets[ticker] = market
        self._inventory[ticker] = {'yes': 0, 'no': 0, 'net': 0}
        self._price_history[ticker] = deque(maxlen=100)

        title = market.get('title', market.get('subtitle', ticker))
        logger.info(f"Added market: {title[:50]}...")
        return True

    def remove_market(self, ticker: str):
        """Remove a market and cancel all quotes."""
        if ticker in self._markets:
            self._cancel_market_quotes(ticker)
            del self._markets[ticker]
            self._inventory.pop(ticker, None)
            self._quotes.pop(ticker, None)
            logger.info(f"Removed market: {ticker}")

    def _get_mock_market(self, ticker: str) -> Optional[Dict]:
        """Get mock market data for testing."""
        mock_markets = {
            'KXBTC-26JAN31-B100K': {
                'ticker': 'KXBTC-26JAN31-B100K',
                'title': 'Will Bitcoin exceed $100,000 on Jan 31?',
                'status': 'open',
                'yes_bid': 42, 'yes_ask': 45, 'no_bid': 55, 'no_ask': 58
            },
            'KXFED-26MAR-CUT': {
                'ticker': 'KXFED-26MAR-CUT',
                'title': 'Will the Fed cut rates in March 2026?',
                'status': 'open',
                'yes_bid': 38, 'yes_ask': 42, 'no_bid': 58, 'no_ask': 62
            },
            'KXSP500-26-B5500': {
                'ticker': 'KXSP500-26-B5500',
                'title': 'Will S&P 500 close above 5500 in 2026?',
                'status': 'open',
                'yes_bid': 65, 'yes_ask': 68, 'no_bid': 32, 'no_ask': 35
            }
        }
        return mock_markets.get(ticker)

    # ============== Quote Management ==============

    def calculate_quote(self, ticker: str) -> Optional[Quote]:
        """
        Calculate optimal quote for a market.

        Takes into account:
        - Current orderbook mid-price
        - Our inventory position (from infrastructure if available)
        - Market volatility
        - Spread floor by market type
        - Risk limits
        """
        market = self._markets.get(ticker)
        if not market:
            return None

        # Get current orderbook
        if self.client and self._connected:
            try:
                orderbook = self.client.get_orderbook(ticker)
                yes_bids = orderbook.get('yes', [])
                no_bids = orderbook.get('no', [])

                best_bid = max(b.get('price', 0) for b in yes_bids) if yes_bids else market.get('yes_bid', 50)
                best_ask = (100 - max(b.get('price', 0) for b in no_bids)) if no_bids else market.get('yes_ask', 50)
            except Exception as e:
                logger.debug(f"Error fetching orderbook for {ticker}: {e}")
                best_bid = market.get('yes_bid', 48)
                best_ask = market.get('yes_ask', 52)
        else:
            best_bid = market.get('yes_bid', 48)
            best_ask = market.get('yes_ask', 52)

        mid_price = (best_bid + best_ask) // 2

        # Update price history
        self._price_history[ticker].append(mid_price)

        # Calculate volatility adjustment
        vol_adjustment = self._calculate_volatility_adjustment(ticker)

        # Get inventory from infrastructure if available, else in-memory
        net_position = self._get_net_position(ticker)

        # Skew in cents: positive position = lower our bid, raise our ask
        skew = (net_position // 10) * self.SKEW_FACTOR

        # Calculate spread with market-type floor
        spread_floor = _get_spread_floor(ticker)
        spread = max(self.BASE_SPREAD, spread_floor) + vol_adjustment
        spread = max(spread, self.risk_limits.min_spread)
        spread = min(spread, 15)  # Cap at 15 cents

        half_spread = spread // 2

        # Calculate bid and ask with skew
        bid_price = mid_price - half_spread - skew
        ask_price = mid_price + half_spread - skew

        # Clamp to valid range (1-99 cents)
        bid_price = max(1, min(98, bid_price))
        ask_price = max(2, min(99, ask_price))

        # Ensure bid < ask
        if bid_price >= ask_price:
            bid_price = ask_price - 1

        # Determine quote size based on inventory
        if abs(net_position) > self.risk_limits.max_position * 0.8:
            quote_size = max(1, self.QUOTE_SIZE // 2)
        else:
            quote_size = self.QUOTE_SIZE

        return Quote(
            ticker=ticker,
            bid_price=bid_price,
            bid_size=quote_size,
            ask_price=ask_price,
            ask_size=quote_size,
            spread=ask_price - bid_price,
            mid_price=mid_price,
            timestamp=datetime.now(timezone.utc)
        )

    def _get_net_position(self, ticker: str) -> int:
        """Get net position from infrastructure or in-memory fallback."""
        if self.infrastructure:
            pos_list = self.infrastructure.position_manager.get_position(ticker)
            if pos_list and isinstance(pos_list, list):
                yes_qty = sum(p['quantity'] for p in pos_list if p['side'] == 'yes')
                no_qty = sum(p['quantity'] for p in pos_list if p['side'] == 'no')
                return yes_qty - no_qty
            elif pos_list and isinstance(pos_list, dict):
                return pos_list.get('quantity', 0) if pos_list.get('side') == 'yes' else -pos_list.get('quantity', 0)

        inv = self._inventory.get(ticker, {})
        return inv.get('net', 0)

    def _calculate_volatility_adjustment(self, ticker: str) -> int:
        """Calculate spread adjustment based on price volatility."""
        history = self._price_history.get(ticker)

        if not history or len(history) < 5:
            return 0

        try:
            stdev = statistics.stdev(list(history)[-20:])
            return min(int(stdev), 5)
        except Exception:
            return 0

    def update_quotes(self, ticker: str):
        """Update quotes for a market."""
        # Check risk limits — use infrastructure risk manager if available
        if self.infrastructure:
            allowed, reason = self.infrastructure.risk_manager._check_daily_loss()
            if not allowed:
                logger.warning(f"Risk manager blocked quoting: {reason}")
                return
        elif self._daily_pnl <= -self.risk_limits.max_daily_loss:
            logger.warning("Daily loss limit reached - not quoting")
            return

        # Cancel existing quotes for this market
        self._cancel_market_quotes(ticker)

        # Calculate new quote
        quote = self.calculate_quote(ticker)
        if not quote:
            return

        if self.paper_mode:
            self._paper_place_quotes(quote)
        else:
            self._place_quotes(quote)

        self._quotes[ticker] = quote
        self._log_quote(quote)

        logger.debug(f"Updated {ticker}: Bid {quote.bid_price}c x {quote.bid_size} | Ask {quote.ask_price}c x {quote.ask_size} | Spread {quote.spread}c")

    def _place_quotes(self, quote: Quote):
        """Place actual quotes on Kalshi."""
        if not self.client or not self._connected:
            return

        now = time.time()

        # Place bid (buy YES at bid price)
        try:
            bid_order = self.client.create_order(
                ticker=quote.ticker,
                side='yes',
                action='buy',
                count=quote.bid_size,
                price=quote.bid_price,
                order_type='limit'
            )
            if bid_order:
                order_id = bid_order.get('order_id', '')
                self._orders[order_id] = {
                    'ticker': quote.ticker,
                    'side': 'yes',
                    'action': 'buy',
                    'price': quote.bid_price,
                    'size': quote.bid_size,
                    'placed_at': now
                }
        except Exception as e:
            logger.error(f"Failed to place bid: {e}")

        # Place ask (sell YES at ask price)
        try:
            ask_order = self.client.create_order(
                ticker=quote.ticker,
                side='yes',
                action='sell',
                count=quote.ask_size,
                price=quote.ask_price,
                order_type='limit'
            )
            if ask_order:
                order_id = ask_order.get('order_id', '')
                self._orders[order_id] = {
                    'ticker': quote.ticker,
                    'side': 'yes',
                    'action': 'sell',
                    'price': quote.ask_price,
                    'size': quote.ask_size,
                    'placed_at': now
                }
        except Exception as e:
            logger.error(f"Failed to place ask: {e}")

    def _paper_place_quotes(self, quote: Quote):
        """Simulate quote placement in paper mode."""
        now = time.time()
        bid_id = f"paper_bid_{int(now)}_{quote.ticker}"
        ask_id = f"paper_ask_{int(now)}_{quote.ticker}"

        self._orders[bid_id] = {
            'ticker': quote.ticker, 'side': 'yes', 'action': 'buy',
            'price': quote.bid_price, 'size': quote.bid_size, 'placed_at': now
        }
        self._orders[ask_id] = {
            'ticker': quote.ticker, 'side': 'yes', 'action': 'sell',
            'price': quote.ask_price, 'size': quote.ask_size, 'placed_at': now
        }

    def _cancel_market_quotes(self, ticker: str):
        """Cancel all quotes for a market."""
        orders_to_cancel = [
            oid for oid, order in self._orders.items()
            if order['ticker'] == ticker
        ]

        for order_id in orders_to_cancel:
            if not self.paper_mode and self.client and not order_id.startswith('paper_'):
                try:
                    self.client.cancel_order(order_id)
                except Exception as e:
                    logger.debug(f"Failed to cancel order {order_id}: {e}")
            del self._orders[order_id]

    def cancel_all_orders(self):
        """Cancel all open orders — called on shutdown."""
        for ticker in list(self._markets.keys()):
            self._cancel_market_quotes(ticker)
        logger.info("All market maker orders cancelled")

    def _cancel_stale_orders(self):
        """Cancel orders older than STALE_ORDER_SECONDS."""
        now = time.time()
        stale = [
            oid for oid, order in self._orders.items()
            if now - order.get('placed_at', now) > self.STALE_ORDER_SECONDS
        ]

        for order_id in stale:
            if not self.paper_mode and self.client and not order_id.startswith('paper_'):
                try:
                    self.client.cancel_order(order_id)
                except Exception:
                    pass
            del self._orders[order_id]

        if stale:
            logger.debug(f"Cancelled {len(stale)} stale orders")

    def _log_quote(self, quote: Quote):
        """Log quote to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                'INSERT INTO quotes (timestamp, ticker, bid_price, ask_price, spread, mid_price) VALUES (?,?,?,?,?,?)',
                (quote.timestamp.isoformat(), quote.ticker, quote.bid_price, quote.ask_price, quote.spread, quote.mid_price)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log quote: {e}")

    # ============== Execution Interface ==============

    def place_order(self, ticker: str, side: str, quantity: int, price: int) -> Optional[Dict]:
        """
        Place a single order — called by the orchestrator's execution path.

        Args:
            ticker: Market ticker
            side: 'yes' or 'no'
            quantity: Number of contracts
            price: Price in cents (1-99)

        Returns:
            Order result dict or None
        """
        # Risk check via infrastructure
        if self.infrastructure:
            allowed, reason = self.infrastructure.risk_manager.check_trade_allowed(ticker, side, quantity)
            if not allowed:
                logger.warning(f"Risk blocked order: {reason}")
                return None

        if self.paper_mode:
            logger.info(f"[PAPER] place_order: {side.upper()} {quantity} @ {price}c on {ticker}")
            return {'paper': True, 'ticker': ticker, 'side': side, 'count': quantity, 'price': price}

        if not self.client or not self._connected:
            logger.error("Cannot place order: client not connected")
            return None

        try:
            result = self.client.create_order(
                ticker=ticker,
                side=side,
                action='buy',
                count=quantity,
                price=price,
                order_type='limit'
            )
            logger.info(f"Order placed: {side.upper()} {quantity} @ {price}c on {ticker}")
            return result
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    # ============== Bot Control ==============

    def run(self, cycles: int = 1):
        """Run the market maker for a number of cycles."""
        if not self._markets:
            logger.warning("No markets added - nothing to make")
            return

        self._running = True

        for cycle in range(cycles):
            if not self._running:
                break

            # Cancel stale orders first
            self._cancel_stale_orders()

            for ticker in list(self._markets.keys()):
                try:
                    self.update_quotes(ticker)
                except Exception as e:
                    logger.error(f"Error updating {ticker}: {e}")

            # Poll fills if infrastructure is available (replaces simulate_fills)
            if self.infrastructure and not self.paper_mode:
                new_fills = self.infrastructure.poll_fills('Kalshi-Market-Maker')
                self._fills_today += len(new_fills)

            if cycle < cycles - 1:
                time.sleep(self.REFRESH_INTERVAL)

        self._running = False

    def stop(self):
        """Stop the market maker and cancel all quotes."""
        self._running = False
        self.cancel_all_orders()
        logger.info("Market maker stopped and all quotes cancelled")

    # ============== Reporting ==============

    def get_status(self) -> Dict:
        """Get current status of the market maker."""
        status = {
            'running': self._running,
            'markets': len(self._markets),
            'active_orders': len(self._orders),
            'fills_today': self._fills_today,
            'daily_pnl': self._daily_pnl,
            'paper_mode': self.paper_mode,
            'connected': self._connected
        }

        if self.infrastructure:
            infra_status = self.infrastructure.get_status()
            status['infra_open_positions'] = infra_status.get('open_positions', 0)
            status['infra_daily_pnl'] = infra_status.get('daily_pnl_dollars', 0)

        return status

    def get_performance(self) -> Dict:
        """Get performance metrics."""
        return {
            'fills_today': self._fills_today,
            'daily_pnl': self._daily_pnl,
            'markets_making': len(self._markets),
            'paper_mode': self.paper_mode
        }

    def get_inventory_summary(self) -> List[Dict]:
        """Get inventory across all markets."""
        result = []
        for ticker in self._markets:
            net = self._get_net_position(ticker)
            result.append({
                'ticker': ticker,
                'net_position': net,
            })
        return result

    # Popular series with regular trading activity
    ACTIVE_SERIES = [
        'KXFED', 'KXCPI', 'KXGDP', 'KXUNEMP', 'KXJOBLESS',
        'KXINX', 'KXBTC', 'KXETH', 'KXHIGHNY', 'KXLOWNY',
        'KXGAS', 'KXCRUDEOIL', 'KXGOLD',
    ]

    def _get_active_markets(self) -> List[str]:
        """Get list of active markets suitable for market making."""
        if not self.client or not self._connected:
            return []

        try:
            markets = []
            for series in self.ACTIVE_SERIES:
                try:
                    series_markets = self.client.get_markets(series_ticker=series, status='open', limit=20)
                    markets.extend(series_markets)
                except Exception:
                    continue

            if not markets:
                markets = self.client.get_markets(status='open', limit=200)

            scored = []
            for market in markets:
                ticker = market.get('ticker', '')
                if not ticker:
                    continue

                volume_24h = market.get('volume_24h', 0) or 0
                open_interest = market.get('open_interest', 0) or 0
                last_price = market.get('last_price', 0) or 0
                yes_bid = market.get('yes_bid', 0) or 0
                yes_ask = market.get('yes_ask', 0) or 0

                if volume_24h == 0 and open_interest == 0 and last_price == 0:
                    continue
                if last_price > 0 and (last_price <= 5 or last_price >= 95):
                    continue

                score = 0
                if volume_24h > 0:
                    score += min(volume_24h / 50, 10)
                volume = market.get('volume', 0) or 0
                if volume > 0:
                    score += min(volume / 200, 5)
                if open_interest > 0:
                    score += min(open_interest / 50, 5)
                if yes_bid > 0 and yes_ask > 0:
                    score += 5
                    spread = yes_ask - yes_bid
                    if spread >= 2:
                        score += min(spread, 5)
                if last_price > 0:
                    price_centrality = 1.0 - abs(last_price - 50) / 50
                    score += price_centrality * 3

                if score > 0:
                    scored.append((ticker, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            result = [t for t, _ in scored[:8]]
            if result:
                logger.info(f"Found {len(scored)} suitable markets, selected top {len(result)}: {result[:3]}...")
            return result

        except Exception as e:
            logger.warning(f"Failed to get markets: {e}")
            return []

    def run_scan(self) -> List[Dict]:
        """
        Main method called by orchestrator.
        Adds markets dynamically and runs market making cycles.
        """
        logger.info("Starting market maker scan...")

        if not self._connected:
            logger.warning("Kalshi client not connected - skipping market maker")
            return []

        signals = []

        try:
            # Auto-add markets if none configured
            if not self._markets:
                active_markets = self._get_active_markets()
                for ticker in active_markets:
                    self.add_market(ticker)

            if not self._markets:
                logger.warning("No suitable markets found for market making")
                return []

            # Run 1 cycle
            self.run(cycles=1)

            # Return status as signals
            for ticker in self._markets:
                quote = self._quotes.get(ticker)
                if quote:
                    signals.append({
                        'ticker': ticker,
                        'action': 'market_making',
                        'bid': quote.bid_price,
                        'ask': quote.ask_price,
                        'spread': quote.spread,
                        'position': self._get_net_position(ticker),
                        'type': 'market_maker'
                    })

            logger.info(f"Market maker cycle complete: {len(self._markets)} markets, {self._fills_today} fills today")

        except Exception as e:
            logger.error(f"Market maker scan failed: {e}")

        return signals


def main():
    """Test market maker bot."""
    print("=" * 70)
    print("KALSHI MARKET MAKER BOT")
    print("=" * 70)

    mm = KalshiMarketMaker(paper_mode=True)

    test_markets = ['KXBTC-26JAN31-B100K', 'KXFED-26MAR-CUT', 'KXSP500-26-B5500']
    for ticker in test_markets:
        mm.add_market(ticker)

    for ticker in mm._markets.keys():
        quote = mm.calculate_quote(ticker)
        if quote:
            market = mm._markets[ticker]
            print(f"\n  {market.get('title', ticker)[:50]}...")
            print(f"  Mid: {quote.mid_price}c | Bid: {quote.bid_price}c x {quote.bid_size} | Ask: {quote.ask_price}c x {quote.ask_size} | Spread: {quote.spread}c")

    mm.run(cycles=3)

    perf = mm.get_performance()
    for key, value in perf.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nMarket maker test complete!")


if __name__ == '__main__':
    main()
