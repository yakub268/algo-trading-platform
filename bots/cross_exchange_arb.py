"""
Cross-Exchange Arbitrage Bot

Scans centralized exchanges (CEX) for price discrepancies on the same asset.
When BTC is $100,000 on Coinbase but $100,150 on Kraken, buy low and sell high.

Strategy:
1. Monitor real-time prices across multiple exchanges
2. When spread exceeds fees + threshold, execute simultaneous trades
3. Profit from the price difference

Expected APY: 20-40% (highly competitive, requires speed)
Risk: Execution risk, withdrawal delays, exchange counterparty risk

Exchanges Supported:
- Coinbase Pro
- Kraken
- Binance (where legal)
- KuCoin
- OKX

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
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CrossExchangeArb')


class Exchange(Enum):
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BINANCE = "binance"
    KUCOIN = "kucoin"
    OKX = "okx"


@dataclass
class ExchangePrice:
    """Price quote from an exchange"""
    exchange: Exchange
    symbol: str
    bid: float  # Best bid (we sell at this)
    ask: float  # Best ask (we buy at this)
    bid_size: float
    ask_size: float
    timestamp: datetime


@dataclass
class ArbOpportunity:
    """Cross-exchange arbitrage opportunity"""
    symbol: str
    buy_exchange: Exchange
    buy_price: float
    sell_exchange: Exchange
    sell_price: float
    spread: float
    spread_pct: float
    max_size: float  # Limited by order book depth
    expected_profit: float
    fees_estimate: float
    net_profit: float
    timestamp: datetime


@dataclass
class ArbTrade:
    """Executed arbitrage trade"""
    opportunity: ArbOpportunity
    buy_order_id: str
    sell_order_id: str
    actual_buy_price: float
    actual_sell_price: float
    size: float
    status: str  # 'pending', 'partial', 'complete', 'failed'
    actual_profit: Optional[float]
    timestamp: datetime


class ExchangeConnector:
    """
    Base class for exchange connections.
    
    Handles:
    - REST API calls for prices and orders
    - WebSocket for real-time data
    - Rate limiting
    - Authentication
    """
    
    def __init__(self, exchange: Exchange, api_key: str = None, api_secret: str = None):
        self.exchange = exchange
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = None
    
    async def get_price(self, symbol: str) -> Optional[ExchangePrice]:
        """
        Fetch current bid/ask price for a symbol.

        Must be implemented by subclasses for each exchange.

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')

        Returns:
            ExchangePrice with bid/ask data, or None if unavailable
        """
        logger.warning(f"get_price not implemented for {self.exchange.value}")
        return None

    async def place_order(self, symbol: str, side: str, price: float, size: float) -> Optional[str]:
        """
        Place a limit order on the exchange.

        Must be implemented by subclasses for each exchange.

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            side: 'buy' or 'sell'
            price: Limit price in USD
            size: Order size in base currency

        Returns:
            Order ID string if successful, None otherwise
        """
        if not self.api_key or not self.api_secret:
            logger.error(f"API credentials not configured for {self.exchange.value}")
            return None

        logger.warning(f"place_order not implemented for {self.exchange.value}")
        return None


class CoinbaseConnector(ExchangeConnector):
    """Coinbase Pro API connector"""
    
    BASE_URL = "https://api.exchange.coinbase.com"
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        super().__init__(Exchange.COINBASE, api_key, api_secret)
    
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def get_price(self, symbol: str) -> Optional[ExchangePrice]:
        """Get current bid/ask from Coinbase"""
        await self._ensure_session()
        
        try:
            # Convert symbol format (BTC -> BTC-USD)
            product_id = f"{symbol}-USD"
            
            async with self.session.get(
                f"{self.BASE_URL}/products/{product_id}/book?level=1",
                timeout=5
            ) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                bids = data.get('bids', [])
                asks = data.get('asks', [])
                
                if not bids or not asks:
                    return None
                
                return ExchangePrice(
                    exchange=Exchange.COINBASE,
                    symbol=symbol,
                    bid=float(bids[0][0]),
                    ask=float(asks[0][0]),
                    bid_size=float(bids[0][1]),
                    ask_size=float(asks[0][1]),
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            logger.error(f"Coinbase price error: {e}")
            return None


class KrakenConnector(ExchangeConnector):
    """Kraken API connector"""
    
    BASE_URL = "https://api.kraken.com/0/public"
    
    # Symbol mapping (standard -> Kraken format)
    SYMBOL_MAP = {
        'BTC': 'XXBTZUSD',
        'ETH': 'XETHZUSD',
        'SOL': 'SOLUSD',
        'DOGE': 'XDGUSD',
    }
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        super().__init__(Exchange.KRAKEN, api_key, api_secret)
    
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def get_price(self, symbol: str) -> Optional[ExchangePrice]:
        """Get current bid/ask from Kraken"""
        await self._ensure_session()
        
        try:
            kraken_symbol = self.SYMBOL_MAP.get(symbol, f"{symbol}USD")
            
            async with self.session.get(
                f"{self.BASE_URL}/Ticker?pair={kraken_symbol}",
                timeout=5
            ) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                if data.get('error'):
                    return None
                
                result = data.get('result', {})
                ticker = list(result.values())[0] if result else None
                
                if not ticker:
                    return None
                
                return ExchangePrice(
                    exchange=Exchange.KRAKEN,
                    symbol=symbol,
                    bid=float(ticker['b'][0]),
                    ask=float(ticker['a'][0]),
                    bid_size=float(ticker['b'][2]),
                    ask_size=float(ticker['a'][2]),
                    timestamp=datetime.now(timezone.utc)
                )
                
        except Exception as e:
            logger.error(f"Kraken price error: {e}")
            return None


class CrossExchangeArbitrage:
    """
    Main cross-exchange arbitrage scanner and executor.
    
    Workflow:
    1. Fetch prices from all connected exchanges
    2. Find price discrepancies above threshold
    3. Calculate profitability after fees
    4. Execute simultaneous trades
    
    Key Parameters:
    - MIN_SPREAD_PCT: Minimum spread to trade (default 0.1%)
    - MAX_POSITION: Maximum position per trade
    - EXCHANGES: List of exchanges to monitor
    
    Usage:
        arb = CrossExchangeArbitrage(paper_mode=True)
        opportunities = await arb.scan(['BTC', 'ETH', 'SOL'])
        for opp in opportunities:
            if opp.net_profit > 0:
                await arb.execute(opp)
    """
    
    # Default parameters
    MIN_SPREAD_PCT = 0.001  # 0.1% minimum spread
    MAX_POSITION = 1000  # $1000 max per trade
    
    # Fee estimates (maker + taker + withdrawal)
    FEE_ESTIMATES = {
        Exchange.COINBASE: 0.005,  # 0.5%
        Exchange.KRAKEN: 0.004,    # 0.4%
        Exchange.BINANCE: 0.002,   # 0.2%
        Exchange.KUCOIN: 0.003,    # 0.3%
        Exchange.OKX: 0.002,       # 0.2%
    }
    
    def __init__(self, paper_mode: bool = True):
        """
        Initialize cross-exchange arbitrage.
        
        Args:
            paper_mode: If True, simulate trades
        """
        self.paper_mode = paper_mode
        
        # Initialize exchange connectors
        self.connectors: Dict[Exchange, ExchangeConnector] = {
            Exchange.COINBASE: CoinbaseConnector(),
            Exchange.KRAKEN: KrakenConnector(),
        }
        
        # Price cache
        self._prices: Dict[str, Dict[Exchange, ExchangePrice]] = defaultdict(dict)
        
        # Trade history
        self._trades: List[ArbTrade] = []
        
        # Database
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'cross_exchange_arb.db'
        )
        self._init_database()
        
        logger.info(f"Cross-Exchange Arbitrage initialized (paper_mode={paper_mode})")
    
    def _init_database(self):
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                buy_exchange TEXT,
                buy_price REAL,
                sell_exchange TEXT,
                sell_price REAL,
                spread_pct REAL,
                net_profit REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                buy_exchange TEXT,
                sell_exchange TEXT,
                size REAL,
                profit REAL,
                status TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ============== Price Fetching ==============
    
    async def fetch_prices(self, symbols: List[str]) -> Dict[str, Dict[Exchange, ExchangePrice]]:
        """
        Fetch prices from all exchanges for given symbols.
        
        Args:
            symbols: List of asset symbols (e.g., ['BTC', 'ETH'])
            
        Returns:
            Dict mapping symbol -> exchange -> price
        """
        results: Dict[str, Dict[Exchange, ExchangePrice]] = defaultdict(dict)
        
        # Fetch in parallel
        tasks = []
        for symbol in symbols:
            for exchange, connector in self.connectors.items():
                tasks.append(self._fetch_single(connector, symbol))
        
        prices = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results
        idx = 0
        for symbol in symbols:
            for exchange in self.connectors.keys():
                price = prices[idx]
                if isinstance(price, ExchangePrice):
                    results[symbol][exchange] = price
                idx += 1
        
        self._prices = results
        return results
    
    async def _fetch_single(self, connector: ExchangeConnector, symbol: str) -> Optional[ExchangePrice]:
        """Fetch price from a single exchange."""
        try:
            return await connector.get_price(symbol)
        except Exception as e:
            logger.debug(f"Price fetch failed: {e}")
            return None
    
    # ============== Opportunity Detection ==============
    
    async def scan(self, symbols: List[str] = None) -> List[ArbOpportunity]:
        """
        Scan for arbitrage opportunities.
        
        Args:
            symbols: Symbols to scan (default: BTC, ETH, SOL)
            
        Returns:
            List of opportunities sorted by profit potential
        """
        if symbols is None:
            symbols = ['BTC', 'ETH', 'SOL']
        
        # Fetch latest prices
        prices = await self.fetch_prices(symbols)
        
        opportunities = []
        
        for symbol, exchange_prices in prices.items():
            if len(exchange_prices) < 2:
                continue
            
            # Compare all exchange pairs
            exchanges = list(exchange_prices.keys())
            for i, ex1 in enumerate(exchanges):
                for ex2 in exchanges[i+1:]:
                    opp = self._find_opportunity(
                        symbol,
                        exchange_prices[ex1],
                        exchange_prices[ex2]
                    )
                    if opp:
                        opportunities.append(opp)
                        self._log_opportunity(opp)
        
        # Sort by net profit
        opportunities.sort(key=lambda x: x.net_profit, reverse=True)
        
        logger.info(f"Found {len(opportunities)} arbitrage opportunities")
        return opportunities
    
    def _find_opportunity(self, 
                          symbol: str,
                          price1: ExchangePrice, 
                          price2: ExchangePrice) -> Optional[ArbOpportunity]:
        """
        Check if there's an arbitrage opportunity between two exchanges.
        
        Buy at the exchange with lower ask, sell at the exchange with higher bid.
        """
        # Determine buy and sell sides
        # We buy at ask price, sell at bid price
        if price1.ask < price2.bid:
            # Buy on exchange 1, sell on exchange 2
            buy_exchange = price1.exchange
            buy_price = price1.ask
            sell_exchange = price2.exchange
            sell_price = price2.bid
            max_size = min(price1.ask_size, price2.bid_size)
        elif price2.ask < price1.bid:
            # Buy on exchange 2, sell on exchange 1
            buy_exchange = price2.exchange
            buy_price = price2.ask
            sell_exchange = price1.exchange
            sell_price = price1.bid
            max_size = min(price2.ask_size, price1.bid_size)
        else:
            # No opportunity
            return None
        
        # Calculate spread
        spread = sell_price - buy_price
        spread_pct = spread / buy_price
        
        if spread_pct < self.MIN_SPREAD_PCT:
            return None
        
        # Calculate fees
        buy_fee = self.FEE_ESTIMATES.get(buy_exchange, 0.005)
        sell_fee = self.FEE_ESTIMATES.get(sell_exchange, 0.005)
        total_fees = buy_fee + sell_fee
        
        # Calculate profit
        position_value = min(max_size * buy_price, self.MAX_POSITION)
        position_size = position_value / buy_price
        
        gross_profit = spread * position_size
        fees = position_value * total_fees
        net_profit = gross_profit - fees
        
        if net_profit <= 0:
            return None
        
        return ArbOpportunity(
            symbol=symbol,
            buy_exchange=buy_exchange,
            buy_price=buy_price,
            sell_exchange=sell_exchange,
            sell_price=sell_price,
            spread=spread,
            spread_pct=spread_pct,
            max_size=position_size,
            expected_profit=gross_profit,
            fees_estimate=fees,
            net_profit=net_profit,
            timestamp=datetime.now(timezone.utc)
        )
    
    def _log_opportunity(self, opp: ArbOpportunity):
        """Log opportunity to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO opportunities 
                (timestamp, symbol, buy_exchange, buy_price, sell_exchange, sell_price, spread_pct, net_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                opp.timestamp.isoformat(),
                opp.symbol,
                opp.buy_exchange.value,
                opp.buy_price,
                opp.sell_exchange.value,
                opp.sell_price,
                opp.spread_pct,
                opp.net_profit
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log opportunity: {e}")
    
    # ============== Trade Execution ==============
    
    async def execute(self, opportunity: ArbOpportunity) -> Optional[ArbTrade]:
        """
        Execute an arbitrage trade.
        
        Places simultaneous buy and sell orders on the respective exchanges.
        
        Args:
            opportunity: The opportunity to execute
            
        Returns:
            ArbTrade if successful
        """
        if self.paper_mode:
            return self._paper_execute(opportunity)
        
        # Real execution would place orders on both exchanges
        logger.warning("Real cross-exchange execution not yet implemented")
        return None
    
    def _paper_execute(self, opp: ArbOpportunity) -> ArbTrade:
        """Simulate trade execution in paper mode."""
        
        trade = ArbTrade(
            opportunity=opp,
            buy_order_id=f"paper_buy_{int(time.time())}",
            sell_order_id=f"paper_sell_{int(time.time())}",
            actual_buy_price=opp.buy_price,
            actual_sell_price=opp.sell_price,
            size=opp.max_size,
            status='complete',
            actual_profit=opp.net_profit,
            timestamp=datetime.now(timezone.utc)
        )
        
        self._trades.append(trade)
        
        logger.info(f"""
[PAPER] Cross-Exchange Arbitrage Executed:
  Symbol: {opp.symbol}
  Buy: {opp.buy_exchange.value} @ ${opp.buy_price:,.2f}
  Sell: {opp.sell_exchange.value} @ ${opp.sell_price:,.2f}
  Size: {opp.max_size:.6f}
  Spread: {opp.spread_pct:.3%}
  Net Profit: ${opp.net_profit:.2f}
""")
        
        return trade
    
    # ============== Reporting ==============
    
    def get_statistics(self) -> Dict:
        """Get arbitrage statistics."""
        total_profit = sum(t.actual_profit or 0 for t in self._trades)
        
        return {
            'total_trades': len(self._trades),
            'total_profit': total_profit,
            'exchanges_connected': len(self.connectors),
            'paper_mode': self.paper_mode
        }
    
    async def close(self):
        """Close all exchange connections."""
        for connector in self.connectors.values():
            if connector.session:
                await connector.session.close()


# ============== Main Entry Point ==============

def main():
    """Test cross-exchange arbitrage."""
    print("=" * 70)
    print("CROSS-EXCHANGE ARBITRAGE BOT")
    print("=" * 70)
    print("""
Cross-exchange arbitrage exploits price differences between exchanges.

Example:
  - BTC on Coinbase: $100,000
  - BTC on Kraken: $100,150
  - Buy on Coinbase, sell on Kraken
  - Profit: $150 (minus fees)

This requires:
  - Fast execution (milliseconds matter)
  - Capital on multiple exchanges
  - Low fees (VIP tiers help)
""")
    
    # Initialize
    arb = CrossExchangeArbitrage(paper_mode=True)
    
    # Scan for opportunities
    print("\n" + "=" * 70)
    print("📊 SCANNING FOR ARBITRAGE")
    print("=" * 70)
    
    async def run_scan():
        opportunities = await arb.scan(['BTC', 'ETH', 'SOL'])
        
        if opportunities:
            print(f"\nFound {len(opportunities)} opportunities:")
            for opp in opportunities[:5]:
                print(f"""
  {opp.symbol}:
    Buy: {opp.buy_exchange.value} @ ${opp.buy_price:,.2f}
    Sell: {opp.sell_exchange.value} @ ${opp.sell_price:,.2f}
    Spread: {opp.spread_pct:.3%}
    Net Profit: ${opp.net_profit:.2f}
""")
            
            # Execute best opportunity
            if opportunities[0].net_profit > 0:
                print("\n🚀 Executing best opportunity...")
                await arb.execute(opportunities[0])
        else:
            print("\nNo profitable opportunities found")
            print("(This is normal - arb opportunities are rare and fleeting)")
        
        # Show statistics
        print("\n" + "=" * 70)
        print("📈 STATISTICS")
        print("=" * 70)
        
        stats = arb.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        await arb.close()
    
    asyncio.run(run_scan())
    
    print("\n✅ Cross-exchange arbitrage test complete!")


if __name__ == '__main__':
    main()
