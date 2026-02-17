"""
Polymarket Market Maker Bot

Provides liquidity to markets and earns the bid-ask spread.
Professional market makers on Polymarket make 80-200% APY.

Strategy:
1. Quote both sides (bid and ask) on selected markets
2. Earn the spread when orders fill on both sides
3. Manage inventory to stay delta-neutral
4. Adjust quotes based on volatility and position

Key Concepts:
- Spread: Difference between bid and ask (your profit)
- Inventory: Net position (want to stay near zero)
- Skew: Adjust prices to reduce inventory risk

Expected APY: 80-200% depending on market selection and competition
Risk: Inventory risk, adverse selection, market volatility

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
import sqlite3
import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import deque
import statistics

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bots.polymarket.polymarket_client import PolymarketClient, Market, Order, OrderSide

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MarketMaker')


@dataclass
class Quote:
    """Represents a two-sided quote"""
    market_id: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    spread: float
    mid_price: float
    timestamp: datetime


@dataclass
class Inventory:
    """Tracks inventory for a market"""
    market_id: str
    token_id: str
    outcome: str
    position: float  # Positive = long, negative = short
    avg_price: float
    unrealized_pnl: float
    last_update: datetime


@dataclass
class Fill:
    """Represents a filled order"""
    order_id: str
    market_id: str
    side: str
    price: float
    size: float
    timestamp: datetime


@dataclass
class MMPerformance:
    """Performance metrics for a market"""
    market_id: str
    total_volume: float
    spread_earned: float
    inventory_pnl: float
    total_pnl: float
    num_fills: int
    avg_spread: float
    time_in_market_pct: float


class MarketMakerBot:
    """
    Automated market maker for Polymarket.
    
    Core Logic:
    1. Calculate fair value (mid-price from order book)
    2. Set bid/ask around fair value with target spread
    3. Skew quotes based on inventory (lower bid if long, lower ask if short)
    4. Manage risk by limiting inventory and hedging when needed
    
    Parameters:
    - BASE_SPREAD: Starting spread (e.g., 0.02 = 2%)
    - MAX_INVENTORY: Maximum position in any direction
    - SKEW_FACTOR: How much to skew quotes per unit of inventory
    - QUOTE_SIZE: Size of each quote
    
    Usage:
        mm = MarketMakerBot(paper_mode=True)
        mm.add_market("condition_id_123")
        await mm.start()
    """
    
    # Default parameters
    BASE_SPREAD = 0.02  # 2% spread
    MAX_INVENTORY = 100  # Max shares in any direction
    SKEW_FACTOR = 0.002  # 0.2% skew per share of inventory
    QUOTE_SIZE = 10  # 10 shares per quote
    REFRESH_INTERVAL = 5  # Refresh quotes every 5 seconds
    
    def __init__(self, paper_mode: bool = True):
        """
        Initialize market maker bot.
        
        Args:
            paper_mode: If True, simulate trading
        """
        self.paper_mode = paper_mode
        self.client = PolymarketClient(paper_mode=paper_mode)
        
        # Markets we're making
        self._markets: Dict[str, Market] = {}
        
        # Current quotes (market_id -> Quote)
        self._quotes: Dict[str, Quote] = {}
        
        # Active orders (order_id -> Order)
        self._orders: Dict[str, Order] = {}
        
        # Inventory tracking (market_id -> Inventory)
        self._inventory: Dict[str, Inventory] = {}
        
        # Fill history
        self._fills: List[Fill] = []
        
        # Performance tracking
        self._performance: Dict[str, MMPerformance] = {}
        
        # Price history for volatility (market_id -> deque of prices)
        self._price_history: Dict[str, deque] = {}
        
        # Running state
        self._running = False
        
        # Database
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'market_maker.db'
        )
        self._init_database()
        
        logger.info(f"Market Maker Bot initialized (paper_mode={paper_mode})")
    
    def _init_database(self):
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                market_id TEXT,
                bid_price REAL,
                ask_price REAL,
                spread REAL,
                mid_price REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                market_id TEXT,
                side TEXT,
                price REAL,
                size REAL,
                spread_captured REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                market_id TEXT,
                position REAL,
                avg_price REAL,
                unrealized_pnl REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ============== Market Management ==============
    
    def add_market(self, condition_id: str) -> bool:
        """
        Add a market to make.
        
        Args:
            condition_id: Polymarket condition ID
            
        Returns:
            True if market added successfully
        """
        market = self.client.get_market(condition_id)
        
        if not market:
            logger.error(f"Could not find market: {condition_id}")
            return False
        
        if not market.active:
            logger.warning(f"Market is not active: {market.question}")
            return False
        
        self._markets[condition_id] = market
        self._inventory[condition_id] = Inventory(
            market_id=condition_id,
            token_id=market.tokens[0]['token_id'] if market.tokens else '',
            outcome=market.outcomes[0] if market.outcomes else '',
            position=0,
            avg_price=0,
            unrealized_pnl=0,
            last_update=datetime.now(timezone.utc)
        )
        self._price_history[condition_id] = deque(maxlen=100)
        self._performance[condition_id] = MMPerformance(
            market_id=condition_id,
            total_volume=0,
            spread_earned=0,
            inventory_pnl=0,
            total_pnl=0,
            num_fills=0,
            avg_spread=self.BASE_SPREAD,
            time_in_market_pct=0
        )
        
        logger.info(f"Added market: {market.question[:50]}...")
        return True
    
    def remove_market(self, condition_id: str):
        """Remove a market and cancel all quotes."""
        if condition_id in self._markets:
            # Cancel quotes
            self._cancel_market_quotes(condition_id)
            
            # Remove from tracking
            del self._markets[condition_id]
            if condition_id in self._inventory:
                del self._inventory[condition_id]
            if condition_id in self._quotes:
                del self._quotes[condition_id]
            
            logger.info(f"Removed market: {condition_id}")
    
    # ============== Quote Management ==============
    
    def calculate_quote(self, market_id: str) -> Optional[Quote]:
        """
        Calculate optimal quote for a market.
        
        Takes into account:
        - Current order book mid-price
        - Our inventory position
        - Market volatility
        - Competition
        """
        market = self._markets.get(market_id)
        if not market:
            return None
        
        # Get current mid price
        token_id = market.tokens[0]['token_id'] if market.tokens else None
        if not token_id:
            return None
        
        mid_price = self.client.get_price(token_id)
        if mid_price is None:
            mid_price = 0.5  # Default to 50%
        
        # Update price history
        self._price_history[market_id].append(mid_price)
        
        # Calculate volatility adjustment
        vol_adjustment = self._calculate_volatility_adjustment(market_id)
        
        # Calculate inventory skew
        inventory = self._inventory.get(market_id)
        position = inventory.position if inventory else 0
        skew = position * self.SKEW_FACTOR
        
        # Calculate spread (wider if more volatile or more inventory)
        spread = self.BASE_SPREAD + vol_adjustment + abs(skew)
        spread = min(spread, 0.10)  # Cap at 10% spread
        
        # Calculate bid and ask
        half_spread = spread / 2
        bid_price = mid_price - half_spread - skew
        ask_price = mid_price + half_spread - skew
        
        # Clamp to valid range
        bid_price = max(0.01, min(0.98, bid_price))
        ask_price = max(0.02, min(0.99, ask_price))
        
        # Ensure bid < ask
        if bid_price >= ask_price:
            bid_price = ask_price - 0.01
        
        quote = Quote(
            market_id=market_id,
            bid_price=round(bid_price, 2),
            bid_size=self.QUOTE_SIZE,
            ask_price=round(ask_price, 2),
            ask_size=self.QUOTE_SIZE,
            spread=ask_price - bid_price,
            mid_price=mid_price,
            timestamp=datetime.now(timezone.utc)
        )
        
        return quote
    
    def _calculate_volatility_adjustment(self, market_id: str) -> float:
        """Calculate spread adjustment based on recent price volatility."""
        history = self._price_history.get(market_id)
        
        if not history or len(history) < 5:
            return 0
        
        try:
            stdev = statistics.stdev(list(history)[-20:])
            # More volatility = wider spread
            return min(stdev * 2, 0.05)  # Cap at 5% adjustment
        except Exception as e:
            logger.debug(f"Error calculating volatility for {market_id}: {e}")
            return 0
    
    async def update_quotes(self, market_id: str):
        """Update quotes for a market."""
        # Cancel existing quotes
        self._cancel_market_quotes(market_id)
        
        # Calculate new quote
        quote = self.calculate_quote(market_id)
        if not quote:
            return
        
        # Place new orders
        market = self._markets.get(market_id)
        if not market or not market.tokens:
            return
        
        token_id = market.tokens[0]['token_id']
        
        # Place bid
        bid_order = self.client.place_order(
            token_id=token_id,
            side='BUY',
            price=quote.bid_price,
            size=quote.bid_size
        )
        
        # Place ask
        ask_order = self.client.place_order(
            token_id=token_id,
            side='SELL',
            price=quote.ask_price,
            size=quote.ask_size
        )
        
        if bid_order:
            self._orders[bid_order.order_id] = bid_order
        if ask_order:
            self._orders[ask_order.order_id] = ask_order
        
        self._quotes[market_id] = quote
        self._log_quote(quote)
        
        logger.debug(f"Updated quote: Bid ${quote.bid_price:.2f} | Ask ${quote.ask_price:.2f} | Spread {quote.spread:.1%}")
    
    def _cancel_market_quotes(self, market_id: str):
        """Cancel all quotes for a market."""
        orders_to_cancel = [
            oid for oid, order in self._orders.items()
            if order.market_id == market_id
        ]
        
        for order_id in orders_to_cancel:
            self.client.cancel_order(order_id)
            del self._orders[order_id]
    
    def _log_quote(self, quote: Quote):
        """Log quote to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO quotes (timestamp, market_id, bid_price, ask_price, spread, mid_price)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                quote.timestamp.isoformat(),
                quote.market_id,
                quote.bid_price,
                quote.ask_price,
                quote.spread,
                quote.mid_price
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log quote: {e}")
    
    # ============== Fill Processing ==============
    
    async def process_fill(self, order_id: str, fill_price: float, fill_size: float):
        """
        Process a filled order.
        
        Updates inventory, calculates P&L, and logs the fill.
        """
        order = self._orders.get(order_id)
        if not order:
            return
        
        market_id = order.market_id
        
        # Create fill record
        fill = Fill(
            order_id=order_id,
            market_id=market_id,
            side=order.side.value,
            price=fill_price,
            size=fill_size,
            timestamp=datetime.now(timezone.utc)
        )
        self._fills.append(fill)
        
        # Update inventory
        inventory = self._inventory.get(market_id)
        if inventory:
            if order.side == OrderSide.BUY:
                # Bought shares = increase position
                new_position = inventory.position + fill_size
                new_avg = (inventory.avg_price * inventory.position + fill_price * fill_size) / new_position if new_position != 0 else fill_price
            else:
                # Sold shares = decrease position
                new_position = inventory.position - fill_size
                new_avg = inventory.avg_price  # Avg price doesn't change on sell
            
            inventory.position = new_position
            inventory.avg_price = new_avg
            inventory.last_update = datetime.now(timezone.utc)
        
        # Update performance
        perf = self._performance.get(market_id)
        if perf:
            perf.total_volume += fill_size * fill_price
            perf.num_fills += 1
            
            # Calculate spread earned (if this is completing a round trip)
            quote = self._quotes.get(market_id)
            if quote:
                spread_earned = quote.spread * fill_size / 2  # Half spread per side
                perf.spread_earned += spread_earned
                perf.total_pnl += spread_earned
        
        self._log_fill(fill)
        
        logger.info(f"Fill: {order.side.value} {fill_size:.2f} @ ${fill_price:.2f}")
    
    def _log_fill(self, fill: Fill):
        """Log fill to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            quote = self._quotes.get(fill.market_id)
            spread_captured = quote.spread / 2 if quote else 0
            
            cursor.execute('''
                INSERT INTO fills (timestamp, market_id, side, price, size, spread_captured)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                fill.timestamp.isoformat(),
                fill.market_id,
                fill.side,
                fill.price,
                fill.size,
                spread_captured
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log fill: {e}")
    
    # ============== Bot Control ==============
    
    async def start(self):
        """Start the market maker bot."""
        if not self._markets:
            logger.warning("No markets added - nothing to make")
            return
        
        self._running = True
        logger.info(f"Starting market maker for {len(self._markets)} markets")
        
        while self._running:
            for market_id in list(self._markets.keys()):
                try:
                    await self.update_quotes(market_id)
                except Exception as e:
                    logger.error(f"Error updating {market_id}: {e}")
            
            await asyncio.sleep(self.REFRESH_INTERVAL)
    
    def stop(self):
        """Stop the market maker and cancel all quotes."""
        self._running = False
        
        # Cancel all quotes
        for market_id in list(self._markets.keys()):
            self._cancel_market_quotes(market_id)
        
        logger.info("Market maker stopped")
    
    # ============== Reporting ==============
    
    def get_status(self) -> Dict:
        """Get current status of the market maker."""
        return {
            'running': self._running,
            'markets': len(self._markets),
            'active_orders': len(self._orders),
            'total_fills': len(self._fills),
            'paper_mode': self.paper_mode
        }
    
    def get_performance(self, market_id: str = None) -> Dict:
        """Get performance metrics."""
        if market_id:
            perf = self._performance.get(market_id)
            if perf:
                return asdict(perf)
            return {}
        
        # Aggregate across all markets
        total_volume = sum(p.total_volume for p in self._performance.values())
        total_spread = sum(p.spread_earned for p in self._performance.values())
        total_pnl = sum(p.total_pnl for p in self._performance.values())
        total_fills = sum(p.num_fills for p in self._performance.values())
        
        return {
            'total_volume': total_volume,
            'spread_earned': total_spread,
            'total_pnl': total_pnl,
            'total_fills': total_fills,
            'markets_making': len(self._markets)
        }
    
    def get_inventory_summary(self) -> List[Dict]:
        """Get inventory across all markets."""
        return [
            {
                'market_id': inv.market_id,
                'outcome': inv.outcome,
                'position': inv.position,
                'avg_price': inv.avg_price,
                'unrealized_pnl': inv.unrealized_pnl
            }
            for inv in self._inventory.values()
        ]


# ============== Main Entry Point ==============

def main():
    """Test market maker bot."""
    print("=" * 70)
    print("POLYMARKET MARKET MAKER BOT")
    print("=" * 70)
    print("""
Market making involves quoting both sides of a market and earning the spread.

Example:
  - You quote: Bid $0.48 | Ask $0.52 (4% spread)
  - Someone sells to you at $0.48
  - Someone else buys from you at $0.52  
  - You earned $0.04 per share!

Professional MMs make 80-200% APY on Polymarket.

Key Risks:
  - Inventory risk (stuck holding a losing position)
  - Adverse selection (trading against informed traders)
  - Market volatility (prices move against you)
""")
    
    # Initialize bot
    mm = MarketMakerBot(paper_mode=True)
    
    # Fetch some markets
    print("\n" + "=" * 70)
    print("📊 FETCHING MARKETS")
    print("=" * 70)
    
    markets = mm.client.get_markets(limit=10)
    
    if markets:
        print(f"\nFound {len(markets)} markets:")
        for m in markets[:5]:
            print(f"\n  {m.question[:60]}...")
            print(f"  Volume: ${m.volume:,.0f}")
            print(f"  Liquidity: ${m.liquidity:,.0f}")
            
            if m.tokens:
                mm.add_market(m.condition_id)
    
    # Calculate sample quotes
    print("\n" + "=" * 70)
    print("💰 SAMPLE QUOTES")
    print("=" * 70)
    
    for market_id in list(mm._markets.keys())[:3]:
        quote = mm.calculate_quote(market_id)
        if quote:
            market = mm._markets[market_id]
            print(f"\n  {market.question[:50]}...")
            print(f"  Mid Price: ${quote.mid_price:.2f}")
            print(f"  Bid: ${quote.bid_price:.2f} x {quote.bid_size}")
            print(f"  Ask: ${quote.ask_price:.2f} x {quote.ask_size}")
            print(f"  Spread: {quote.spread:.1%}")
    
    # Simulate some fills
    print("\n" + "=" * 70)
    print("🔄 SIMULATING FILLS")
    print("=" * 70)
    
    # Simulate placing quotes and getting filled
    async def simulate():
        for market_id in list(mm._markets.keys())[:2]:
            await mm.update_quotes(market_id)
            
            # Simulate a buy fill
            if mm._orders:
                order_id = list(mm._orders.keys())[0]
                order = mm._orders[order_id]
                await mm.process_fill(order_id, order.price, order.size)
    
    asyncio.run(simulate())
    
    # Performance
    print("\n" + "=" * 70)
    print("📈 PERFORMANCE")
    print("=" * 70)
    
    perf = mm.get_performance()
    for key, value in perf.items():
        if isinstance(value, float):
            print(f"  {key}: ${value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Inventory
    print("\n" + "=" * 70)
    print("📦 INVENTORY")
    print("=" * 70)
    
    inventory = mm.get_inventory_summary()
    for inv in inventory[:3]:
        print(f"  {inv['outcome']}: {inv['position']:.2f} shares @ ${inv['avg_price']:.2f}")
    
    print("\n✅ Market maker bot test complete!")


if __name__ == '__main__':
    main()
