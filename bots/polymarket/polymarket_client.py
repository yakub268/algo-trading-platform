"""
Polymarket Client - Core Integration Module

Handles authentication, API calls, and order management for Polymarket.
Uses the CLOB (Central Limit Order Book) API.

API Endpoints:
- CLOB API: https://clob.polymarket.com
- Gamma API: https://gamma-api.polymarket.com  
- WebSocket: wss://ws-subscriptions-clob.polymarket.com

Requirements:
- pip install py-clob-client web3 python-dotenv requests websockets

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import json
import time
import logging
import requests
import asyncio
import websockets
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from decimal import Decimal
import hashlib
import hmac

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
    CLOB_CLIENT_AVAILABLE = True
except ImportError:
    CLOB_CLIENT_AVAILABLE = False
    print("Warning: py-clob-client not installed. Run: pip install py-clob-client")

try:
    from web3 import Web3
    from eth_account import Account
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    print("Warning: web3 not installed. Run: pip install web3")

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PolymarketClient')


# ============== Data Classes ==============

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    LIVE = "LIVE"
    MATCHED = "MATCHED"
    CANCELLED = "CANCELLED"


@dataclass
class Market:
    """Represents a Polymarket market"""
    condition_id: str
    question: str
    description: str
    end_date: datetime
    outcomes: List[str]
    outcome_prices: Dict[str, float]  # outcome -> price
    volume: float
    liquidity: float
    active: bool
    category: str
    tokens: List[Dict]  # Token IDs for each outcome
    neg_risk: bool = False  # True for multi-outcome markets


@dataclass
class Position:
    """Represents an open position"""
    market_id: str
    outcome: str
    token_id: str
    size: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    side: str


@dataclass
class Order:
    """Represents an order"""
    order_id: str
    market_id: str
    outcome: str
    side: OrderSide
    price: float
    size: float
    filled: float
    status: OrderStatus
    created_at: datetime


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity"""
    market_id: str
    market_question: str
    opportunity_type: str  # 'sum_to_one', 'cross_platform', 'timing'
    expected_profit_pct: float
    confidence: str
    details: Dict
    timestamp: datetime


# ============== Main Client ==============

class PolymarketClient:
    """
    Main Polymarket API client.
    
    Supports:
    - Market discovery and monitoring
    - Order placement and management
    - Position tracking
    - WebSocket real-time data
    
    Usage:
        client = PolymarketClient()
        markets = client.get_markets()
        client.place_order(market_id, "YES", "BUY", price=0.55, size=10)
    """
    
    # API Endpoints
    CLOB_API = "https://clob.polymarket.com"
    GAMMA_API = "https://gamma-api.polymarket.com"
    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    
    # Polygon Chain ID (Polymarket runs on Polygon)
    CHAIN_ID = 137
    
    def __init__(self, private_key: Optional[str] = None, paper_mode: bool = True):
        """
        Initialize Polymarket client.
        
        Args:
            private_key: Ethereum private key for signing (from env if not provided)
            paper_mode: If True, simulates trades without execution
        """
        self.paper_mode = paper_mode
        self.private_key = private_key or os.getenv('POLYMARKET_PRIVATE_KEY')
        
        # Initialize Web3 if available
        self.web3 = None
        self.account = None
        self.wallet_address = None
        
        if WEB3_AVAILABLE and self.private_key:
            try:
                self.web3 = Web3()
                self.account = Account.from_key(self.private_key)
                self.wallet_address = self.account.address
                logger.info(f"Wallet connected: {self.wallet_address[:10]}...")
            except Exception as e:
                logger.error(f"Failed to initialize wallet: {e}")
        
        # Initialize CLOB client if available
        self.clob_client = None
        if CLOB_CLIENT_AVAILABLE and self.private_key and not paper_mode:
            try:
                self.clob_client = ClobClient(
                    host=self.CLOB_API,
                    chain_id=self.CHAIN_ID,
                    key=self.private_key
                )
                logger.info("CLOB client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize CLOB client: {e}")
        
        # Session for REST calls
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Cache
        self._markets_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 60  # seconds
        
        # Paper trading state
        self._paper_positions: List[Position] = []
        self._paper_orders: List[Order] = []
        self._paper_balance = 1000.0  # Starting paper balance
        
        logger.info(f"Polymarket client initialized (paper_mode={paper_mode})")
    
    # ============== Market Data ==============
    
    def get_markets(self, 
                    active_only: bool = True,
                    category: Optional[str] = None,
                    limit: int = 100) -> List[Market]:
        """
        Fetch available markets from Gamma API.
        
        Args:
            active_only: Only return active markets
            category: Filter by category (e.g., 'politics', 'crypto', 'sports')
            limit: Maximum markets to return
            
        Returns:
            List of Market objects
        """
        try:
            params = {
                'limit': limit,
                'active': str(active_only).lower()
            }
            if category:
                params['tag'] = category
            
            response = self.session.get(
                f"{self.GAMMA_API}/markets",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            markets = []
            
            for m in data:
                try:
                    # Parse outcomes and prices
                    outcomes = []
                    outcome_prices = {}
                    tokens = []
                    
                    if 'tokens' in m:
                        for token in m['tokens']:
                            outcome = token.get('outcome', 'Unknown')
                            price = float(token.get('price', 0.5))
                            outcomes.append(outcome)
                            outcome_prices[outcome] = price
                            tokens.append({
                                'token_id': token.get('token_id'),
                                'outcome': outcome,
                                'price': price
                            })
                    
                    market = Market(
                        condition_id=m.get('condition_id', ''),
                        question=m.get('question', ''),
                        description=m.get('description', ''),
                        end_date=datetime.fromisoformat(m['end_date_iso'].replace('Z', '+00:00')) if m.get('end_date_iso') else datetime.now(timezone.utc),
                        outcomes=outcomes,
                        outcome_prices=outcome_prices,
                        volume=float(m.get('volume', 0)),
                        liquidity=float(m.get('liquidity', 0)),
                        active=m.get('active', False),
                        category=m.get('category', ''),
                        tokens=tokens,
                        neg_risk=len(outcomes) > 2  # Multi-outcome = NegRisk
                    )
                    markets.append(market)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse market: {e}")
                    continue
            
            # Update cache
            self._markets_cache = {m.condition_id: m for m in markets}
            self._cache_timestamp = datetime.now()
            
            logger.info(f"Fetched {len(markets)} markets")
            return markets
            
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []
    
    def get_market(self, condition_id: str) -> Optional[Market]:
        """Get a specific market by condition ID."""
        # Check cache first
        if (self._cache_timestamp and 
            (datetime.now() - self._cache_timestamp).seconds < self._cache_ttl and
            condition_id in self._markets_cache):
            return self._markets_cache[condition_id]
        
        try:
            response = self.session.get(
                f"{self.GAMMA_API}/markets/{condition_id}",
                timeout=10
            )
            response.raise_for_status()
            m = response.json()
            
            outcomes = []
            outcome_prices = {}
            tokens = []
            
            if 'tokens' in m:
                for token in m['tokens']:
                    outcome = token.get('outcome', 'Unknown')
                    price = float(token.get('price', 0.5))
                    outcomes.append(outcome)
                    outcome_prices[outcome] = price
                    tokens.append({
                        'token_id': token.get('token_id'),
                        'outcome': outcome,
                        'price': price
                    })
            
            return Market(
                condition_id=m.get('condition_id', ''),
                question=m.get('question', ''),
                description=m.get('description', ''),
                end_date=datetime.fromisoformat(m['end_date_iso'].replace('Z', '+00:00')) if m.get('end_date_iso') else datetime.now(timezone.utc),
                outcomes=outcomes,
                outcome_prices=outcome_prices,
                volume=float(m.get('volume', 0)),
                liquidity=float(m.get('liquidity', 0)),
                active=m.get('active', False),
                category=m.get('category', ''),
                tokens=tokens,
                neg_risk=len(outcomes) > 2
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch market {condition_id}: {e}")
            return None
    
    def get_orderbook(self, token_id: str) -> Dict:
        """
        Get order book for a specific token.
        
        Returns:
            Dict with 'bids' and 'asks' lists
        """
        try:
            response = self.session.get(
                f"{self.CLOB_API}/book",
                params={'token_id': token_id},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch orderbook: {e}")
            return {'bids': [], 'asks': []}
    
    def get_price(self, token_id: str) -> Optional[float]:
        """Get current mid price for a token."""
        book = self.get_orderbook(token_id)
        
        bids = book.get('bids', [])
        asks = book.get('asks', [])
        
        if bids and asks:
            best_bid = float(bids[0]['price'])
            best_ask = float(asks[0]['price'])
            return (best_bid + best_ask) / 2
        elif bids:
            return float(bids[0]['price'])
        elif asks:
            return float(asks[0]['price'])
        return None
    
    # ============== Order Management ==============
    
    def place_order(self,
                    token_id: str,
                    side: str,  # 'BUY' or 'SELL'
                    price: float,
                    size: float,
                    order_type: str = 'GTC') -> Optional[Order]:
        """
        Place an order on Polymarket.
        
        Args:
            token_id: Token ID for the outcome
            side: 'BUY' or 'SELL'
            price: Price (0.01 to 0.99)
            size: Number of shares
            order_type: 'GTC' (Good Till Cancel) or 'FOK' (Fill or Kill)
            
        Returns:
            Order object if successful
        """
        if self.paper_mode:
            return self._paper_place_order(token_id, side, price, size)
        
        if not self.clob_client:
            logger.error("CLOB client not initialized - cannot place real orders")
            return None
        
        try:
            order_args = OrderArgs(
                price=price,
                size=size,
                side=side,
                token_id=token_id
            )
            
            response = self.clob_client.create_and_post_order(order_args)
            
            order = Order(
                order_id=response.get('orderID', ''),
                market_id=token_id,
                outcome='',
                side=OrderSide[side],
                price=price,
                size=size,
                filled=0,
                status=OrderStatus.LIVE,
                created_at=datetime.now(timezone.utc)
            )
            
            logger.info(f"Order placed: {side} {size} @ ${price}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    def _paper_place_order(self, token_id: str, side: str, price: float, size: float) -> Order:
        """Simulate order placement in paper mode."""
        order_id = f"paper_{int(time.time() * 1000)}"
        
        order = Order(
            order_id=order_id,
            market_id=token_id,
            outcome='',
            side=OrderSide[side],
            price=price,
            size=size,
            filled=size,  # Assume immediate fill in paper mode
            status=OrderStatus.MATCHED,
            created_at=datetime.now(timezone.utc)
        )
        
        self._paper_orders.append(order)
        
        # Update paper balance
        cost = price * size
        if side == 'BUY':
            self._paper_balance -= cost
        else:
            self._paper_balance += cost
        
        logger.info(f"[PAPER] Order placed: {side} {size} @ ${price:.2f} (Balance: ${self._paper_balance:.2f})")
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.paper_mode:
            for order in self._paper_orders:
                if order.order_id == order_id:
                    order.status = OrderStatus.CANCELLED
                    logger.info(f"[PAPER] Order cancelled: {order_id}")
                    return True
            return False
        
        if not self.clob_client:
            return False
        
        try:
            self.clob_client.cancel(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        if self.paper_mode:
            return [o for o in self._paper_orders if o.status == OrderStatus.LIVE]
        
        if not self.clob_client:
            return []
        
        try:
            response = self.clob_client.get_orders()
            # Parse response into Order objects
            orders = []
            for o in response:
                orders.append(Order(
                    order_id=o.get('id', ''),
                    market_id=o.get('asset_id', ''),
                    outcome='',
                    side=OrderSide[o.get('side', 'BUY')],
                    price=float(o.get('price', 0)),
                    size=float(o.get('original_size', 0)),
                    filled=float(o.get('size_matched', 0)),
                    status=OrderStatus[o.get('status', 'LIVE')],
                    created_at=datetime.now(timezone.utc)
                ))
            return orders
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    # ============== Position Management ==============
    
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        if self.paper_mode:
            return self._paper_positions
        
        # For real trading, would query the API
        # This requires more complex position tracking
        return []
    
    def get_balance(self) -> float:
        """Get available USDC balance."""
        if self.paper_mode:
            return self._paper_balance
        
        # Would query actual balance from wallet
        return 0.0
    
    # ============== WebSocket Real-Time Data ==============
    
    async def subscribe_market(self, token_ids: List[str], callback):
        """
        Subscribe to real-time market updates via WebSocket.
        
        Args:
            token_ids: List of token IDs to subscribe to
            callback: Async function to call with updates
        """
        try:
            async with websockets.connect(self.WS_URL) as ws:
                # Subscribe to markets
                subscribe_msg = {
                    "type": "subscribe",
                    "markets": token_ids
                }
                await ws.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to {len(token_ids)} markets")
                
                # Process messages
                last_ping = time.time()
                while True:
                    try:
                        # Check for heartbeat (connection can die after ~20 min)
                        if time.time() - last_ping > 60:
                            await ws.ping()
                            last_ping = time.time()
                        
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        await callback(data)
                        
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        await ws.ping()
                        last_ping = time.time()
                        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    # ============== Utility Methods ==============
    
    def search_markets(self, query: str, limit: int = 20) -> List[Market]:
        """Search markets by keyword."""
        markets = self.get_markets(limit=500)
        query_lower = query.lower()
        
        matching = [
            m for m in markets
            if query_lower in m.question.lower() or query_lower in m.description.lower()
        ]
        
        return matching[:limit]
    
    def get_sports_markets(self) -> List[Market]:
        """Get all sports-related markets."""
        return self.get_markets(category='sports')
    
    def get_crypto_markets(self) -> List[Market]:
        """Get all crypto-related markets."""
        return self.get_markets(category='crypto')
    
    def get_politics_markets(self) -> List[Market]:
        """Get all politics-related markets."""
        return self.get_markets(category='politics')
    
    def get_neg_risk_markets(self) -> List[Market]:
        """Get all multi-outcome (NegRisk) markets - prime for sum-to-one arbitrage."""
        markets = self.get_markets(limit=500)
        return [m for m in markets if m.neg_risk]


# ============== Main Entry Point ==============

def main():
    """Test the Polymarket client."""
    print("=" * 60)
    print("POLYMARKET CLIENT TEST")
    print("=" * 60)
    
    # Initialize in paper mode
    client = PolymarketClient(paper_mode=True)
    
    # Fetch markets
    print("\nFetching markets...")
    markets = client.get_markets(limit=10)
    
    if markets:
        print(f"\nFound {len(markets)} markets:")
        for m in markets[:5]:
            print(f"\n  Question: {m.question[:60]}...")
            print(f"  Outcomes: {m.outcomes}")
            print(f"  Prices: {m.outcome_prices}")
            print(f"  Volume: ${m.volume:,.0f}")
            print(f"  NegRisk: {m.neg_risk}")
    
    # Search for sports markets
    print("\n" + "=" * 60)
    print("SPORTS MARKETS")
    print("=" * 60)
    
    sports = client.get_sports_markets()
    print(f"\nFound {len(sports)} sports markets")
    
    # Search for NegRisk markets (arbitrage opportunities)
    print("\n" + "=" * 60)
    print("NEGRISK MARKETS (ARBITRAGE CANDIDATES)")
    print("=" * 60)
    
    neg_risk = client.get_neg_risk_markets()
    print(f"\nFound {len(neg_risk)} multi-outcome markets")
    
    for m in neg_risk[:3]:
        total_prob = sum(m.outcome_prices.values())
        print(f"\n  {m.question[:50]}...")
        print(f"  Outcomes: {len(m.outcomes)}")
        print(f"  Total probability: {total_prob:.2%}")
        if total_prob > 1.0:
            print(f"  ⚠️ ARBITRAGE: Overpriced by {(total_prob - 1) * 100:.2f}%")
        elif total_prob < 1.0:
            print(f"  ⚠️ ARBITRAGE: Underpriced by {(1 - total_prob) * 100:.2f}%")
    
    # Paper trade test
    print("\n" + "=" * 60)
    print("PAPER TRADING TEST")
    print("=" * 60)
    
    print(f"\nStarting balance: ${client.get_balance():.2f}")
    
    # Simulate a trade
    if markets:
        token = markets[0].tokens[0] if markets[0].tokens else None
        if token:
            order = client.place_order(
                token_id=token['token_id'],
                side='BUY',
                price=0.55,
                size=10
            )
            print(f"\nOrder placed: {order}")
            print(f"New balance: ${client.get_balance():.2f}")
    
    print("\n✅ Polymarket client test complete!")


if __name__ == '__main__':
    main()
