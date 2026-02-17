"""
COINBASE CROSS-EXCHANGE ARBITRAGE BOT
=====================================

Monitors price differences between Coinbase and other US exchanges.
Executes trades when spread exceeds fees.

Uses official Coinbase Advanced Trade SDK for authentication.

Supported Exchanges:
- Coinbase Advanced Trade (primary - can execute)
- Kraken (price monitoring)
- Gemini (price monitoring)

Strategy:
- Monitor BTC, ETH, SOL, etc. across exchanges
- When Coinbase price differs by > fees, execute
- Delta-neutral: buy low, sell high simultaneously

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import time
import json
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CoinbaseArbBot')


@dataclass
class ExchangePrice:
    """Price from an exchange"""
    exchange: str
    symbol: str
    bid: float
    ask: float
    timestamp: datetime


@dataclass
class ArbOpportunity:
    """Arbitrage opportunity"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_pct: float
    net_profit_pct: float
    size_usd: float
    expected_profit: float


class CoinbaseClient:
    """Coinbase Advanced Trade API Client using official SDK"""

    def __init__(self):
        self.api_key = os.getenv('COINBASE_API_KEY', '')
        self.private_key_path = os.path.expanduser(os.getenv('COINBASE_PRIVATE_KEY_PATH', '~/.trading_keys/coinbase_private_key.pem'))
        self.client = None
        self._initialized = False

        if self.api_key:
            try:
                self._init_client()
            except Exception as e:
                logger.warning(f"Coinbase init failed: {e}")

    def _init_client(self):
        """Initialize Coinbase client using official SDK"""
        try:
            from coinbase.rest import RESTClient

            # Load private key
            private_key = None
            paths = [
                self.private_key_path,
                os.path.join(os.path.dirname(os.path.dirname(__file__)), self.private_key_path),
            ]

            for path in paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        private_key = f.read()
                    break

            if not private_key:
                logger.warning(f"Coinbase private key not found at {paths}")
                return

            # Initialize with Cloud API key format
            self.client = RESTClient(api_key=self.api_key, api_secret=private_key)
            self._initialized = True
            logger.info("Coinbase client initialized with official SDK")

        except ImportError:
            logger.error("coinbase-advanced-py not installed. Run: pip install coinbase-advanced-py")
        except Exception as e:
            logger.error(f"Coinbase client init error: {e}")

    def get_accounts(self) -> List[dict]:
        """Get all accounts/balances"""
        if not self._initialized:
            return []
        try:
            response = self.client.get_accounts()
            return response.accounts if hasattr(response, 'accounts') else []
        except Exception as e:
            logger.error(f"Get accounts error: {e}")
            return []

    def get_best_bid_ask(self, product_ids: List[str]) -> List[dict]:
        """Get best bid/ask for products"""
        if not self._initialized:
            return []
        try:
            response = self.client.get_best_bid_ask(product_ids=product_ids)
            return response.pricebooks if hasattr(response, 'pricebooks') else []
        except Exception as e:
            logger.error(f"Get best bid/ask error: {e}")
            return []

    def get_product(self, product_id: str) -> Optional[dict]:
        """Get single product details"""
        if not self._initialized:
            return None
        try:
            return self.client.get_product(product_id=product_id)
        except Exception as e:
            logger.error(f"Get product error: {e}")
            return None

    def create_market_order(self, product_id: str, side: str, quote_size: str = None, base_size: str = None) -> Optional[dict]:
        """Create a market order. BUY uses quote_size (dollars), SELL uses base_size (crypto qty)."""
        if not self._initialized:
            return None
        try:
            import uuid
            client_order_id = str(uuid.uuid4())

            if side.upper() == 'BUY':
                response = self.client.market_order_buy(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    quote_size=quote_size or base_size
                )
            else:
                # SELL requires base_size (quantity of crypto to sell)
                if base_size:
                    response = self.client.market_order_sell(
                        client_order_id=client_order_id,
                        product_id=product_id,
                        base_size=base_size
                    )
                elif quote_size:
                    # Convert quote_size to base_size using current price
                    try:
                        pricebooks = self.get_best_bid_ask([product_id])
                        if pricebooks:
                            book = pricebooks[0]
                            bids = book.bids if hasattr(book, 'bids') else book.get('bids', [])
                            if bids:
                                bid = float(bids[0].price if hasattr(bids[0], 'price') else bids[0].get('price', 0))
                                if bid > 0:
                                    base_size = str(round(float(quote_size) / bid, 8))
                    except Exception:
                        pass

                    if base_size:
                        response = self.client.market_order_sell(
                            client_order_id=client_order_id,
                            product_id=product_id,
                            base_size=base_size
                        )
                    else:
                        logger.error(f"Cannot sell: need base_size but couldn't convert from quote_size")
                        return None
                else:
                    logger.error("Sell order requires base_size or quote_size")
                    return None
            return response
        except Exception as e:
            logger.error(f"Create order error: {e}")
            return None


class CoinbaseArbBot:
    """
    Cross-exchange arbitrage using Coinbase.

    Monitors prices across Coinbase, Kraken, and Gemini.
    Executes when spread exceeds trading fees.
    """

    # Trading pairs to monitor
    PAIRS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD']

    # Fee estimates (conservative)
    FEES = {
        'coinbase': 0.006,  # 0.6% taker
        'kraken': 0.0026,   # 0.26%
        'gemini': 0.004,    # 0.4%
    }

    # Minimum profit threshold
    MIN_PROFIT_PCT = 0.005  # 0.5% net profit required

    def __init__(self, capital: float = 200.0, paper_mode: bool = None):
        self.capital = capital
        if paper_mode is None:
            paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
        self.paper_mode = paper_mode

        self.coinbase = CoinbaseClient()
        self.opportunities: List[ArbOpportunity] = []
        self.trades_today: List[dict] = []

        logger.info(f"CoinbaseArbBot initialized - Capital: ${capital}, Paper: {paper_mode}")

    def get_coinbase_prices(self) -> Dict[str, ExchangePrice]:
        """Get prices from Coinbase"""
        prices = {}

        try:
            pricebooks = self.coinbase.get_best_bid_ask(self.PAIRS)

            for book in pricebooks:
                product_id = book.product_id if hasattr(book, 'product_id') else book.get('product_id', '')
                bids = book.bids if hasattr(book, 'bids') else book.get('bids', [])
                asks = book.asks if hasattr(book, 'asks') else book.get('asks', [])

                if bids and asks:
                    bid_price = float(bids[0].price if hasattr(bids[0], 'price') else bids[0].get('price', 0))
                    ask_price = float(asks[0].price if hasattr(asks[0], 'price') else asks[0].get('price', 0))

                    prices[product_id] = ExchangePrice(
                        exchange='coinbase',
                        symbol=product_id,
                        bid=bid_price,
                        ask=ask_price,
                        timestamp=datetime.now(timezone.utc)
                    )
        except Exception as e:
            logger.error(f"Coinbase price fetch error: {e}")

        return prices

    def get_kraken_prices(self) -> Dict[str, ExchangePrice]:
        """Get prices from Kraken (public API)"""
        prices = {}

        # Kraken symbol mapping
        kraken_pairs = {
            'BTC-USD': 'XXBTZUSD',
            'ETH-USD': 'XETHZUSD',
            'SOL-USD': 'SOLUSD',
            'DOGE-USD': 'XDGUSD',
        }

        try:
            for our_symbol, kraken_symbol in kraken_pairs.items():
                url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_symbol}"
                response = requests.get(url, timeout=5)
                data = response.json()

                if data.get('result'):
                    ticker = list(data['result'].values())[0]
                    prices[our_symbol] = ExchangePrice(
                        exchange='kraken',
                        symbol=our_symbol,
                        bid=float(ticker['b'][0]),
                        ask=float(ticker['a'][0]),
                        timestamp=datetime.now(timezone.utc)
                    )
        except Exception as e:
            logger.error(f"Kraken price fetch error: {e}")

        return prices

    def get_gemini_prices(self) -> Dict[str, ExchangePrice]:
        """Get prices from Gemini (public API)"""
        prices = {}

        # Gemini symbol mapping
        gemini_pairs = {
            'BTC-USD': 'btcusd',
            'ETH-USD': 'ethusd',
            'SOL-USD': 'solusd',
            'DOGE-USD': 'dogeusd',
        }

        try:
            for our_symbol, gemini_symbol in gemini_pairs.items():
                url = f"https://api.gemini.com/v1/pubticker/{gemini_symbol}"
                response = requests.get(url, timeout=5)
                data = response.json()

                if 'bid' in data and 'ask' in data:
                    prices[our_symbol] = ExchangePrice(
                        exchange='gemini',
                        symbol=our_symbol,
                        bid=float(data['bid']),
                        ask=float(data['ask']),
                        timestamp=datetime.now(timezone.utc)
                    )
        except Exception as e:
            logger.error(f"Gemini price fetch error: {e}")

        return prices

    def find_opportunities(self) -> List[ArbOpportunity]:
        """Find arbitrage opportunities across exchanges"""
        opportunities = []

        # Fetch all prices
        coinbase_prices = self.get_coinbase_prices()
        kraken_prices = self.get_kraken_prices()
        gemini_prices = self.get_gemini_prices()

        all_prices = {
            'coinbase': coinbase_prices,
            'kraken': kraken_prices,
            'gemini': gemini_prices,
        }

        # Log what we got
        for exchange, prices in all_prices.items():
            if prices:
                logger.info(f"  {exchange}: {len(prices)} pairs")
                for sym, p in prices.items():
                    logger.debug(f"    {sym}: bid={p.bid:.2f}, ask={p.ask:.2f}")

        # Check each pair
        for symbol in self.PAIRS:
            exchange_prices = []

            for exchange, prices in all_prices.items():
                if symbol in prices:
                    exchange_prices.append(prices[symbol])

            if len(exchange_prices) < 2:
                continue

            # Find best buy (lowest ask) and best sell (highest bid)
            best_buy = min(exchange_prices, key=lambda x: x.ask)
            best_sell = max(exchange_prices, key=lambda x: x.bid)

            if best_buy.exchange == best_sell.exchange:
                continue

            # Calculate spread
            spread = best_sell.bid - best_buy.ask
            spread_pct = spread / best_buy.ask

            # Calculate fees
            buy_fee = self.FEES.get(best_buy.exchange, 0.005)
            sell_fee = self.FEES.get(best_sell.exchange, 0.005)
            total_fees = buy_fee + sell_fee

            # Net profit
            net_profit_pct = spread_pct - total_fees

            if net_profit_pct > self.MIN_PROFIT_PCT:
                # Calculate trade size (use 25% of capital per trade)
                size_usd = min(self.capital * 0.25, 100)
                expected_profit = size_usd * net_profit_pct

                opportunities.append(ArbOpportunity(
                    symbol=symbol,
                    buy_exchange=best_buy.exchange,
                    sell_exchange=best_sell.exchange,
                    buy_price=best_buy.ask,
                    sell_price=best_sell.bid,
                    spread_pct=spread_pct,
                    net_profit_pct=net_profit_pct,
                    size_usd=size_usd,
                    expected_profit=expected_profit
                ))
            else:
                # Log near-misses for debugging
                if spread_pct > 0:
                    logger.debug(f"{symbol}: spread={spread_pct:.3%}, fees={total_fees:.3%}, net={net_profit_pct:.3%}")

        self.opportunities = sorted(opportunities, key=lambda x: x.net_profit_pct, reverse=True)
        return self.opportunities

    def execute_arb(self, opp: ArbOpportunity) -> Optional[dict]:
        """Execute an arbitrage trade"""
        trade = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': opp.symbol,
            'buy_exchange': opp.buy_exchange,
            'sell_exchange': opp.sell_exchange,
            'buy_price': opp.buy_price,
            'sell_price': opp.sell_price,
            'size_usd': opp.size_usd,
            'expected_profit': opp.expected_profit,
            'paper': self.paper_mode,
            'status': 'pending'
        }

        if self.paper_mode:
            trade['status'] = 'filled'
            logger.info(
                f"[PAPER] ARB: Buy {opp.symbol} on {opp.buy_exchange} @ ${opp.buy_price:.2f}, "
                f"Sell on {opp.sell_exchange} @ ${opp.sell_price:.2f}, "
                f"Profit: ${opp.expected_profit:.2f} ({opp.net_profit_pct:.2%})"
            )
        else:
            # Only execute on Coinbase (we have API)
            if opp.buy_exchange == 'coinbase':
                order = self.coinbase.create_market_order(
                    product_id=opp.symbol,
                    side='BUY',
                    quote_size=str(opp.size_usd)
                )
                if order:
                    trade['buy_order_id'] = getattr(order, 'order_id', None)
                    trade['status'] = 'partial'
                    logger.info(f"[LIVE] Bought {opp.symbol} on Coinbase")

            elif opp.sell_exchange == 'coinbase':
                logger.info(f"[ALERT] Sell opportunity on Coinbase - manual execution needed")
                trade['status'] = 'alert'

        self.trades_today.append(trade)
        return trade

    def run_scan(self) -> List[dict]:
        """Run a scan and execute opportunities"""
        logger.info("Scanning for cross-exchange arbitrage...")

        opportunities = self.find_opportunities()
        executed = []

        for opp in opportunities[:2]:  # Max 2 trades per scan
            trade = self.execute_arb(opp)
            if trade:
                executed.append(trade)

        return executed

    def get_status(self) -> dict:
        """Get bot status"""
        return {
            'name': 'CoinbaseArbBot',
            'capital': self.capital,
            'paper_mode': self.paper_mode,
            'coinbase_connected': self.coinbase._initialized,
            'trades_today': len(self.trades_today),
            'opportunities': len(self.opportunities),
            'last_scan': datetime.now(timezone.utc).isoformat()
        }


if __name__ == "__main__":
    print("=" * 60)
    print("COINBASE CROSS-EXCHANGE ARBITRAGE BOT")
    print("=" * 60)

    bot = CoinbaseArbBot(capital=200.0, paper_mode=True)

    print(f"\nStatus: {bot.get_status()}")

    # Show account info if connected
    if bot.coinbase._initialized:
        print("\n--- Coinbase Accounts ---")
        accounts = bot.coinbase.get_accounts()
        for acc in accounts[:5]:
            name = acc.name if hasattr(acc, 'name') else acc.get('name', 'Unknown')
            balance = acc.available_balance if hasattr(acc, 'available_balance') else acc.get('available_balance', {})
            value = balance.value if hasattr(balance, 'value') else balance.get('value', '0')
            currency = balance.currency if hasattr(balance, 'currency') else balance.get('currency', '')
            if float(value) > 0:
                print(f"  {name}: {value} {currency}")

    print("\n--- Scanning for opportunities ---")
    trades = bot.run_scan()

    if bot.opportunities:
        print(f"\nFound {len(bot.opportunities)} opportunities:")
        for opp in bot.opportunities[:5]:
            print(f"  {opp.symbol}: Buy {opp.buy_exchange} @ ${opp.buy_price:.2f} -> "
                  f"Sell {opp.sell_exchange} @ ${opp.sell_price:.2f} = {opp.net_profit_pct:.2%} profit")
    else:
        print("\nNo arbitrage opportunities found (spreads within normal range)")
        print("This is typical - true arbs are rare and close within seconds")

    print("\n" + "=" * 60)
