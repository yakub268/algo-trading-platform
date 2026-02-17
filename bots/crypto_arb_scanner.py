"""
Cross-Exchange Crypto Arbitrage Scanner

Monitors price discrepancies across major cryptocurrency exchanges.
Identifies arbitrage opportunities when spreads exceed trading costs.

Supported Exchanges:
- Binance
- Coinbase Pro
- Kraken
- Bybit
- OKX

Strategy:
- Monitor bid/ask across exchanges
- Calculate spread after fees
- Alert when net profit > 0.3%
- Track historical opportunities for pattern analysis

Note: This is an alert system - execution requires manual review
due to latency and withdrawal time constraints.

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import json
import logging
import sqlite3
import asyncio
import aiohttp
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CryptoArbScanner')

DB_PATH = Path(__file__).parent.parent / "data" / "event_trades.db"


@dataclass
class ExchangePrice:
    """Price data from an exchange"""
    exchange: str
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: datetime


@dataclass
class ArbOpportunity:
    """Arbitrage opportunity between exchanges"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float  # Best ask (where we buy)
    sell_price: float  # Best bid (where we sell)
    gross_spread_pct: float
    estimated_fees_pct: float
    net_profit_pct: float
    available_size: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }


class ExchangeFetcher:
    """
    Fetch prices from multiple exchanges.
    Uses public APIs (no authentication required for price data).
    """
    
    # Exchange fee estimates (maker/taker averages)
    EXCHANGE_FEES = {
        'binance': 0.001,      # 0.1%
        'coinbase': 0.006,     # 0.6% (retail)
        'kraken': 0.0026,      # 0.26%
        'bybit': 0.001,        # 0.1%
        'okx': 0.001,          # 0.1%
    }
    
    # Standard trading pairs
    PAIRS = {
        'BTC/USDT': {
            'binance': 'BTCUSDT',
            'bybit': 'BTCUSDT',
            'okx': 'BTC-USDT',
            'kraken': 'XBTUSDT',
        },
        'ETH/USDT': {
            'binance': 'ETHUSDT',
            'bybit': 'ETHUSDT',
            'okx': 'ETH-USDT',
            'kraken': 'ETHUSDT',
        },
        'SOL/USDT': {
            'binance': 'SOLUSDT',
            'bybit': 'SOLUSDT',
            'okx': 'SOL-USDT',
        },
        'XRP/USDT': {
            'binance': 'XRPUSDT',
            'bybit': 'XRPUSDT',
            'okx': 'XRP-USDT',
            'kraken': 'XRPUSDT',
        },
        'HBAR/USDT': {
            'binance': 'HBARUSDT',
            'bybit': 'HBARUSDT',
            'okx': 'HBAR-USDT',
        },
        'XLM/USDT': {
            'binance': 'XLMUSDT',
            'bybit': 'XLMUSDT',
            'okx': 'XLM-USDT',
            'kraken': 'XLMUSDT',
        },
        'ADA/USDT': {
            'binance': 'ADAUSDT',
            'bybit': 'ADAUSDT',
            'okx': 'ADA-USDT',
            'kraken': 'ADAUSDT',
        },
        'DOGE/USDT': {
            'binance': 'DOGEUSDT',
            'bybit': 'DOGEUSDT',
            'okx': 'DOGE-USDT',
            'kraken': 'DOGEUSDT',
        },
        'AVAX/USDT': {
            'binance': 'AVAXUSDT',
            'bybit': 'AVAXUSDT',
            'okx': 'AVAX-USDT',
            'kraken': 'AVAXUSDT',
        },
        'DOT/USDT': {
            'binance': 'DOTUSDT',
            'bybit': 'DOTUSDT',
            'okx': 'DOT-USDT',
            'kraken': 'DOTUSDT',
        },
        'LINK/USDT': {
            'binance': 'LINKUSDT',
            'bybit': 'LINKUSDT',
            'okx': 'LINK-USDT',
            'kraken': 'LINKUSDT',
        },
        'LTC/USDT': {
            'binance': 'LTCUSDT',
            'bybit': 'LTCUSDT',
            'okx': 'LTC-USDT',
            'kraken': 'LTCUSDT',
        },
        'MATIC/USDT': {
            'binance': 'MATICUSDT',
            'bybit': 'MATICUSDT',
            'okx': 'MATIC-USDT',
            'kraken': 'MATICUSDT',
        },
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'CryptoArbScanner/1.0'})
    
    def fetch_binance(self, symbol: str) -> Optional[ExchangePrice]:
        """Fetch from Binance"""
        try:
            pair = self.PAIRS.get(symbol, {}).get('binance')
            if not pair:
                return None
            
            url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={pair}"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return ExchangePrice(
                    exchange='binance',
                    symbol=symbol,
                    bid=float(data['bidPrice']),
                    ask=float(data['askPrice']),
                    bid_size=float(data['bidQty']),
                    ask_size=float(data['askQty']),
                    timestamp=datetime.now(timezone.utc)
                )
        except Exception as e:
            logger.debug(f"Binance error for {symbol}: {e}")
        return None
    
    def fetch_coinbase(self, symbol: str) -> Optional[ExchangePrice]:
        """Fetch from Coinbase Pro"""
        try:
            # Coinbase uses USD, not USDT
            base = symbol.split('/')[0]
            pair = f"{base}-USD"
            
            url = f"https://api.exchange.coinbase.com/products/{pair}/ticker"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                bid = float(data.get('bid', 0))
                ask = float(data.get('ask', 0))
                
                if bid > 0 and ask > 0:
                    return ExchangePrice(
                        exchange='coinbase',
                        symbol=symbol,
                        bid=bid,
                        ask=ask,
                        bid_size=float(data.get('size', 0)),
                        ask_size=float(data.get('size', 0)),
                        timestamp=datetime.now(timezone.utc)
                    )
        except Exception as e:
            logger.debug(f"Coinbase error for {symbol}: {e}")
        return None
    
    def fetch_kraken(self, symbol: str) -> Optional[ExchangePrice]:
        """Fetch from Kraken"""
        try:
            pair = self.PAIRS.get(symbol, {}).get('kraken')
            if not pair:
                return None
            
            url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if not data.get('error'):
                    result = list(data['result'].values())[0]
                    return ExchangePrice(
                        exchange='kraken',
                        symbol=symbol,
                        bid=float(result['b'][0]),
                        ask=float(result['a'][0]),
                        bid_size=float(result['b'][2]),
                        ask_size=float(result['a'][2]),
                        timestamp=datetime.now(timezone.utc)
                    )
        except Exception as e:
            logger.debug(f"Kraken error for {symbol}: {e}")
        return None
    
    def fetch_bybit(self, symbol: str) -> Optional[ExchangePrice]:
        """Fetch from Bybit"""
        try:
            pair = self.PAIRS.get(symbol, {}).get('bybit')
            if not pair:
                return None
            
            url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={pair}"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('result', {}).get('list'):
                    ticker = data['result']['list'][0]
                    bid = float(ticker.get('bid1Price', 0))
                    ask = float(ticker.get('ask1Price', 0))
                    
                    if bid > 0 and ask > 0:
                        return ExchangePrice(
                            exchange='bybit',
                            symbol=symbol,
                            bid=bid,
                            ask=ask,
                            bid_size=float(ticker.get('bid1Size', 0)),
                            ask_size=float(ticker.get('ask1Size', 0)),
                            timestamp=datetime.now(timezone.utc)
                        )
        except Exception as e:
            logger.debug(f"Bybit error for {symbol}: {e}")
        return None
    
    def fetch_okx(self, symbol: str) -> Optional[ExchangePrice]:
        """Fetch from OKX"""
        try:
            pair = self.PAIRS.get(symbol, {}).get('okx')
            if not pair:
                return None
            
            url = f"https://www.okx.com/api/v5/market/ticker?instId={pair}"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    ticker = data['data'][0]
                    bid = float(ticker.get('bidPx', 0))
                    ask = float(ticker.get('askPx', 0))
                    
                    if bid > 0 and ask > 0:
                        return ExchangePrice(
                            exchange='okx',
                            symbol=symbol,
                            bid=bid,
                            ask=ask,
                            bid_size=float(ticker.get('bidSz', 0)),
                            ask_size=float(ticker.get('askSz', 0)),
                            timestamp=datetime.now(timezone.utc)
                        )
        except Exception as e:
            logger.debug(f"OKX error for {symbol}: {e}")
        return None
    
    def fetch_all(self, symbol: str) -> Dict[str, ExchangePrice]:
        """
        Fetch prices from all exchanges in parallel.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            
        Returns:
            Dict of exchange name to ExchangePrice
        """
        prices = {}
        
        fetchers = [
            ('binance', self.fetch_binance),
            ('coinbase', self.fetch_coinbase),
            ('kraken', self.fetch_kraken),
            ('bybit', self.fetch_bybit),
            ('okx', self.fetch_okx),
        ]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fn, symbol): name for name, fn in fetchers}
            
            for future in as_completed(futures, timeout=10):
                name = futures[future]
                try:
                    result = future.result()
                    if result:
                        prices[name] = result
                except Exception as e:
                    logger.debug(f"Failed to fetch from {name}: {e}")
        
        return prices


class CryptoArbScanner:
    """
    Cross-Exchange Cryptocurrency Arbitrage Scanner
    
    Monitors prices across multiple exchanges and identifies
    arbitrage opportunities when spreads exceed fees.
    
    Profitability Calculation:
    - Gross spread = (sell_bid - buy_ask) / buy_ask
    - Net profit = gross_spread - buy_fee - sell_fee - withdrawal_fee
    
    Alert Threshold: Net profit > 0.3% (after all fees)
    """
    
    MIN_PROFIT_THRESHOLD = 0.003  # 0.3% minimum net profit
    WITHDRAWAL_FEE_ESTIMATE = 0.001  # 0.1% withdrawal fee estimate
    
    def __init__(self):
        self.fetcher = ExchangeFetcher()
        self._init_database()
        
        logger.info("CryptoArbScanner initialized")
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS arb_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                buy_exchange TEXT,
                sell_exchange TEXT,
                buy_price REAL,
                sell_price REAL,
                gross_spread_pct REAL,
                net_profit_pct REAL,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def find_arbitrage(self, symbol: str) -> List[ArbOpportunity]:
        """
        Find arbitrage opportunities for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            
        Returns:
            List of profitable arbitrage opportunities
        """
        prices = self.fetcher.fetch_all(symbol)
        
        if len(prices) < 2:
            return []
        
        opportunities = []
        exchanges = list(prices.keys())
        
        # Compare all exchange pairs
        for i, buy_exchange in enumerate(exchanges):
            for sell_exchange in exchanges[i+1:]:
                # Check both directions
                for ex_buy, ex_sell in [(buy_exchange, sell_exchange), (sell_exchange, buy_exchange)]:
                    buy_price = prices[ex_buy]
                    sell_price = prices[ex_sell]
                    
                    # Buy at ask, sell at bid
                    if buy_price.ask > 0 and sell_price.bid > 0:
                        gross_spread = (sell_price.bid - buy_price.ask) / buy_price.ask
                        
                        # Calculate fees
                        buy_fee = self.fetcher.EXCHANGE_FEES.get(ex_buy, 0.002)
                        sell_fee = self.fetcher.EXCHANGE_FEES.get(ex_sell, 0.002)
                        total_fees = buy_fee + sell_fee + self.WITHDRAWAL_FEE_ESTIMATE
                        
                        net_profit = gross_spread - total_fees
                        
                        # Only include profitable opportunities
                        if net_profit > self.MIN_PROFIT_THRESHOLD:
                            available_size = min(buy_price.ask_size, sell_price.bid_size)
                            
                            opp = ArbOpportunity(
                                symbol=symbol,
                                buy_exchange=ex_buy,
                                sell_exchange=ex_sell,
                                buy_price=buy_price.ask,
                                sell_price=sell_price.bid,
                                gross_spread_pct=gross_spread,
                                estimated_fees_pct=total_fees,
                                net_profit_pct=net_profit,
                                available_size=available_size,
                                timestamp=datetime.now(timezone.utc)
                            )
                            opportunities.append(opp)
                            
                            # Log to database
                            self._save_opportunity(opp)
        
        return sorted(opportunities, key=lambda x: x.net_profit_pct, reverse=True)
    
    def _save_opportunity(self, opp: ArbOpportunity):
        """Save opportunity to database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO arb_opportunities 
            (symbol, buy_exchange, sell_exchange, buy_price, sell_price, 
             gross_spread_pct, net_profit_pct, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            opp.symbol, opp.buy_exchange, opp.sell_exchange,
            opp.buy_price, opp.sell_price, opp.gross_spread_pct,
            opp.net_profit_pct, opp.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def scan_all_pairs(self) -> List[ArbOpportunity]:
        """
        Scan all configured pairs for arbitrage.
        
        Returns:
            List of all opportunities found
        """
        all_opportunities = []
        
        for symbol in self.fetcher.PAIRS.keys():
            try:
                opps = self.find_arbitrage(symbol)
                all_opportunities.extend(opps)
                
                if opps:
                    logger.info(f"{symbol}: Found {len(opps)} arbitrage opportunities")
                    for opp in opps[:3]:
                        logger.info(f"  Buy {opp.buy_exchange} @ ${opp.buy_price:,.2f} -> "
                                   f"Sell {opp.sell_exchange} @ ${opp.sell_price:,.2f} "
                                   f"= {opp.net_profit_pct:.2%} profit")
            except Exception as e:
                logger.warning(f"Error scanning {symbol}: {e}")
        
        return sorted(all_opportunities, key=lambda x: x.net_profit_pct, reverse=True)
    
    def get_price_matrix(self, symbol: str) -> Dict:
        """
        Get price comparison matrix for a symbol.
        
        Returns:
            Dict with exchange prices for easy comparison
        """
        prices = self.fetcher.fetch_all(symbol)
        
        matrix = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'exchanges': {}
        }
        
        for exchange, price in prices.items():
            matrix['exchanges'][exchange] = {
                'bid': price.bid,
                'ask': price.ask,
                'spread': (price.ask - price.bid) / price.bid if price.bid > 0 else 0,
                'bid_size': price.bid_size,
                'ask_size': price.ask_size,
            }
        
        return matrix
    
    def get_historical_stats(self, hours: int = 24) -> Dict:
        """Get historical opportunity statistics"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        
        cursor.execute('''
            SELECT symbol, COUNT(*) as count, AVG(net_profit_pct) as avg_profit,
                   MAX(net_profit_pct) as max_profit
            FROM arb_opportunities 
            WHERE timestamp > ?
            GROUP BY symbol
        ''', (cutoff,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return {
            row[0]: {
                'count': row[1],
                'avg_profit': row[2],
                'max_profit': row[3]
            }
            for row in rows
        }


def send_telegram_alert(message: str):
    """Send Telegram alert"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if bot_token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            requests.post(url, data={'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}, timeout=10)
        except Exception as e:
            logger.warning(f"Telegram failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("CRYPTO ARBITRAGE SCANNER")
    print("=" * 60)
    
    scanner = CryptoArbScanner()
    
    # Show price matrix for BTC
    print("\n📊 BTC/USDT Price Matrix:")
    matrix = scanner.get_price_matrix('BTC/USDT')
    for exchange, data in matrix.get('exchanges', {}).items():
        print(f"  {exchange:12} Bid: ${data['bid']:>10,.2f}  Ask: ${data['ask']:>10,.2f}  "
              f"Spread: {data['spread']:.3%}")
    
    # Scan for opportunities
    print("\n🔍 Scanning for arbitrage opportunities...")
    opportunities = scanner.scan_all_pairs()
    
    if opportunities:
        print(f"\n✅ Found {len(opportunities)} opportunities:")
        for opp in opportunities[:5]:
            print(f"\n  {opp.symbol}:")
            print(f"    Buy: {opp.buy_exchange:10} @ ${opp.buy_price:,.2f}")
            print(f"    Sell: {opp.sell_exchange:10} @ ${opp.sell_price:,.2f}")
            print(f"    Gross Spread: {opp.gross_spread_pct:.3%}")
            print(f"    Est. Fees: {opp.estimated_fees_pct:.3%}")
            print(f"    Net Profit: {opp.net_profit_pct:.3%}")
            print(f"    Size: {opp.available_size:.4f}")
    else:
        print("\n❌ No profitable arbitrage opportunities found.")
        print("   (Threshold: 0.3% net profit after fees)")
