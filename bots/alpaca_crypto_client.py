"""
ALPACA CRYPTO CLIENT
====================
Shared Alpaca crypto trading client for all aggressive bots.
Uses alpaca-py SDK with paper trading by default.

Symbol format: "BTC/USD", "ETH/USD", "SOL/USD" etc.
"""

import os
import sys
import logging
from typing import Optional, List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger('AlpacaCryptoClient')

# Import order fill helper to fix zero-P&L bug
try:
    from utils.order_fill_helper import submit_and_wait_for_fill
except ImportError:
    logger.warning("order_fill_helper not available, using fallback")
    submit_and_wait_for_fill = None

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.error("alpaca-py not installed. Run: pip install alpaca-py")


class AlpacaCryptoClient:
    """Alpaca crypto trading client — drop-in replacement for CoinbaseClient"""

    # Default USDC pairs mapped to Alpaca format
    COINBASE_TO_ALPACA = {
        '-USDC': '/USD',
        '-USD': '/USD',
    }

    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY', '')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY', '')
        self.paper = os.getenv('ALPACA_PAPER_MODE', os.getenv('PAPER_MODE', 'true')).lower() == 'true'

        self.trading_client = None
        self.data_client = None
        self._initialized = False

        if self.api_key and self.api_secret and ALPACA_AVAILABLE:
            try:
                self._init_clients()
            except Exception as e:
                logger.warning(f"Alpaca init failed: {e}")

    def _init_clients(self):
        """Initialize Alpaca trading and data clients"""
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=self.paper
        )
        self.data_client = CryptoHistoricalDataClient()
        self._initialized = True
        mode = "PAPER" if self.paper else "LIVE"
        logger.info(f"Alpaca crypto client initialized ({mode})")

    @staticmethod
    def to_alpaca_symbol(symbol: str) -> str:
        """Convert Coinbase format (BTC-USDC) to Alpaca format (BTC/USD)"""
        s = symbol.replace('-USDC', '/USD').replace('-USD', '/USD')
        if '/' not in s:
            s = s + '/USD'
        return s

    @staticmethod
    def to_trading_symbol(symbol: str) -> str:
        """Convert to Alpaca order format (BTCUSD — no slash)"""
        return AlpacaCryptoClient.to_alpaca_symbol(symbol).replace('/', '')

    @staticmethod
    def from_alpaca_symbol(symbol: str) -> str:
        """Convert Alpaca format (BTC/USD) back to Coinbase-style (BTC-USD)"""
        return symbol.replace('/USD', '-USD').replace('/', '-')

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a crypto symbol"""
        if not self._initialized:
            return None
        try:
            alpaca_sym = self.to_alpaca_symbol(symbol)
            request = CryptoLatestQuoteRequest(symbol_or_symbols=alpaca_sym)
            quotes = self.data_client.get_crypto_latest_quote(request)
            if quotes and alpaca_sym in quotes:
                q = quotes[alpaca_sym]
                bid = float(q.bid_price) if q.bid_price else 0
                ask = float(q.ask_price) if q.ask_price else 0
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2
                return bid or ask or None
            return None
        except Exception as e:
            logger.error(f"Get price error for {symbol}: {e}")
            return None

    def get_best_bid_ask(self, product_ids: List[str]) -> List[Dict]:
        """Get bid/ask for multiple symbols — compatible with CoinbaseClient interface"""
        if not self._initialized:
            return []
        results = []
        for pid in product_ids:
            try:
                alpaca_sym = self.to_alpaca_symbol(pid)
                request = CryptoLatestQuoteRequest(symbol_or_symbols=alpaca_sym)
                quotes = self.data_client.get_crypto_latest_quote(request)
                if quotes and alpaca_sym in quotes:
                    q = quotes[alpaca_sym]
                    results.append({
                        'product_id': pid,
                        'bid': float(q.bid_price) if q.bid_price else 0,
                        'ask': float(q.ask_price) if q.ask_price else 0,
                        'mid': ((float(q.bid_price) + float(q.ask_price)) / 2
                                if q.bid_price and q.ask_price else 0),
                    })
            except Exception as e:
                logger.debug(f"Quote error for {pid}: {e}")
        return results

    def get_candles(self, symbol: str, granularity: str = '1h', limit: int = 100) -> List[Dict]:
        """Get historical candle/bar data for a crypto symbol.

        Args:
            symbol: e.g. 'BTC/USD' or 'BTC-USD'
            granularity: '1m','5m','15m','1h','4h','day','1d','hour'
            limit: approximate number of bars (translates to lookback period)

        Returns:
            List of dicts with keys: timestamp, open, high, low, close, volume
        """
        if not self._initialized or not self.data_client:
            return []
        try:
            from datetime import datetime, timezone, timedelta

            alpaca_sym = self.to_alpaca_symbol(symbol)

            from alpaca.data.timeframe import TimeFrameUnit

            # Map granularity string to TimeFrame + timedelta multiplier
            granularity_map = {
                '1m': (TimeFrame.Minute, 1),
                '5m': (TimeFrame(5, TimeFrameUnit.Minute), 5),
                '5min': (TimeFrame(5, TimeFrameUnit.Minute), 5),
                '15m': (TimeFrame(15, TimeFrameUnit.Minute), 15),
                '1h': (TimeFrame.Hour, 60),
                'hour': (TimeFrame.Hour, 60),
                '4h': (TimeFrame(4, TimeFrameUnit.Hour), 240),
                'day': (TimeFrame.Day, 1440),
                '1d': (TimeFrame.Day, 1440),
            }

            tf, minutes_per_bar = granularity_map.get(granularity.lower(), (TimeFrame.Hour, 60))

            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=minutes_per_bar * limit * 2)

            request = CryptoBarsRequest(
                symbol_or_symbols=alpaca_sym,
                timeframe=tf,
                start=start,
                end=end
            )

            bars_data = self.data_client.get_crypto_bars(request)

            if hasattr(bars_data, 'df'):
                df = bars_data.df.reset_index()
            else:
                import pandas as pd
                df = pd.DataFrame(bars_data.get(alpaca_sym, []))

            if df.empty:
                return []

            # Convert to list of dicts
            records = []
            for _, row in df.tail(limit).iterrows():
                records.append({
                    'timestamp': str(row.get('timestamp', '')),
                    'open': float(row.get('open', 0)),
                    'high': float(row.get('high', 0)),
                    'low': float(row.get('low', 0)),
                    'close': float(row.get('close', 0)),
                    'volume': float(row.get('volume', 0)),
                })

            return records

        except Exception as e:
            logger.error(f"get_candles error for {symbol}: {e}")
            return []

    def create_market_order(self, product_id: str, side: str,
                            quote_size: str = None, base_size: str = None) -> Optional[Dict]:
        """Place a market order — compatible with CoinbaseClient interface.
        BUY: uses notional (dollar amount) if quote_size given, else qty.
        SELL: uses qty (base_size).
        """
        if not self._initialized:
            return None

        alpaca_sym = self.to_trading_symbol(product_id)
        order_side = OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL

        try:
            if side.upper() == 'BUY' and quote_size:
                # Notional (dollar) order
                order_request = MarketOrderRequest(
                    symbol=alpaca_sym,
                    notional=float(quote_size),
                    side=order_side,
                    time_in_force=TimeInForce.GTC
                )
            elif base_size:
                order_request = MarketOrderRequest(
                    symbol=alpaca_sym,
                    qty=float(base_size),
                    side=order_side,
                    time_in_force=TimeInForce.GTC
                )
            elif quote_size and side.upper() == 'SELL':
                # Need to convert dollars to qty
                price = self.get_price(product_id)
                if price and price > 0:
                    qty = float(quote_size) / price
                    order_request = MarketOrderRequest(
                        symbol=alpaca_sym,
                        qty=round(qty, 8),
                        side=order_side,
                        time_in_force=TimeInForce.GTC
                    )
                else:
                    logger.error(f"Cannot sell: no price for {product_id}")
                    return None
            else:
                logger.error("Order requires quote_size or base_size")
                return None

            # CRITICAL FIX: Poll for fill instead of reading immediate response
            # submit_order() returns filled_avg_price=None before fill completes
            if submit_and_wait_for_fill:
                try:
                    fill_result = submit_and_wait_for_fill(
                        self.trading_client,
                        order_request,
                        timeout=30
                    )

                    result = {
                        'success': True,
                        'order_id': fill_result['order_id'],
                        'status': fill_result['status'],
                        'symbol': product_id,
                        'side': side.upper(),
                        'filled_qty': str(fill_result['fill_qty']),
                        'filled_avg_price': str(fill_result['fill_price']),
                    }
                    logger.info(
                        f"Order FILLED: {side} {product_id} | "
                        f"{fill_result['fill_qty']} @ ${fill_result['fill_price']:.6f}"
                    )
                    return result

                except (ValueError, RuntimeError, TimeoutError) as e:
                    logger.error(f"Order fill error for {product_id}: {e}")
                    return {'success': False, 'error': str(e)}
            else:
                # Fallback to old behavior (will have zero-P&L bug)
                logger.warning(
                    "order_fill_helper not available, using immediate response "
                    "(may result in zero P&L bug)"
                )
                order = self.trading_client.submit_order(order_request)
                result = {
                    'success': True,
                    'order_id': str(order.id),
                    'status': str(order.status),
                    'symbol': product_id,
                    'side': side.upper(),
                    'filled_qty': str(order.filled_qty) if order.filled_qty else '0',
                    'filled_avg_price': str(order.filled_avg_price) if order.filled_avg_price else '0',
                }
                logger.info(f"Order submitted: {side} {product_id} | ID: {order.id} | Status: {order.status}")
                return result

        except Exception as e:
            logger.error(f"Order error for {product_id}: {e}")
            return {'success': False, 'error': str(e)}

    def get_account(self) -> Optional[Dict]:
        """Get account info"""
        if not self._initialized:
            return None
        try:
            acct = self.trading_client.get_account()
            return {
                'equity': float(acct.equity),
                'cash': float(acct.cash),
                'buying_power': float(acct.buying_power),
                'portfolio_value': float(acct.portfolio_value),
                'currency': 'USD',
            }
        except Exception as e:
            logger.error(f"Get account error: {e}")
            return None

    def get_accounts(self) -> List[Dict]:
        """Get accounts — compatible with CoinbaseClient interface"""
        acct = self.get_account()
        if acct:
            return [{'currency': 'USD', 'available_balance': {'value': str(acct['cash'])}}]
        return []

    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        if not self._initialized:
            return []
        try:
            positions = self.trading_client.get_all_positions()
            result = []
            for pos in positions:
                result.append({
                    'symbol': str(pos.symbol),
                    'qty': float(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'side': str(pos.side),
                })
            return result
        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return []

    def close_position(self, symbol: str) -> bool:
        """Close an open position on Alpaca by symbol.

        Args:
            symbol: Any format (BTC/USD, BTC-USD, BTC-USDC, BTCUSD)

        Returns:
            True if closed successfully (or no position existed), False on error
        """
        if not self._initialized:
            logger.error("Cannot close position: Alpaca client not initialized")
            return False

        trading_symbol = self.to_trading_symbol(symbol)
        try:
            self.trading_client.close_position(trading_symbol)
            logger.info(f"Position closed on exchange: {trading_symbol}")

            # Dust cleanup: wait briefly and check for residual
            import time as _time
            _time.sleep(0.5)
            positions = self.get_positions()
            for pos in positions:
                if pos['symbol'].replace('/', '') == trading_symbol:
                    if abs(pos['market_value']) < 0.50:
                        logger.warning(f"Dust remaining for {trading_symbol}: ${pos['market_value']:.4f} — retrying close")
                        try:
                            self.trading_client.close_position(trading_symbol)
                            logger.info(f"Dust cleanup succeeded for {trading_symbol}")
                        except Exception as dust_err:
                            logger.warning(f"Dust cleanup failed for {trading_symbol}: {dust_err}")
                    break

            return True
        except Exception as e:
            err_str = str(e)
            # "position does not exist" is not a failure — position already closed
            if '40410000' in err_str or 'position does not exist' in err_str.lower():
                logger.info(f"No position to close for {trading_symbol} (already closed)")
                return True
            logger.error(f"Failed to close position {trading_symbol}: {e}")
            return False

    def get_usdc_balance(self) -> float:
        """Get available cash (USD) — compatible with BaseAggressiveBot"""
        acct = self.get_account()
        return acct['cash'] if acct else 0.0
