"""
Base Aggressive Bot
===================

Base class for high-risk, aggressive trading strategies.
All aggressive bots inherit from this and must implement run_scan().

Features:
- Coinbase integration via CoinbaseClient
- Common risk management parameters
- Logging configuration
- Position management helpers

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bots.alpaca_crypto_client import AlpacaCryptoClient


@dataclass
class TradeSignal:
    """Trade signal from aggressive bot scan"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    target_price: float
    stop_loss: float
    position_size_usd: float
    confidence: float  # 0-1
    reason: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAggressiveBot(ABC):
    """
    Base class for aggressive trading bots.

    All aggressive bots must:
    1. Inherit from this class
    2. Implement run_scan() returning List[TradeSignal]
    3. Use Coinbase for execution
    """

    # Default risk parameters (can be overridden)
    DEFAULT_POSITION_SIZE = 50.0  # USD
    DEFAULT_TAKE_PROFIT_PCT = 0.05  # 5%
    DEFAULT_STOP_LOSS_PCT = 0.04  # 4%
    DEFAULT_MAX_POSITIONS = 3

    def __init__(
        self,
        capital: float = 200.0,
        paper_mode: bool = None,
        position_size: float = None,
        take_profit_pct: float = None,
        stop_loss_pct: float = None,
        max_positions: int = None,
    ):
        """
        Initialize aggressive bot.

        Args:
            capital: Total capital allocated to this bot
            paper_mode: True for paper trading (default from env)
            position_size: USD size per position
            take_profit_pct: Take profit percentage (0.05 = 5%)
            stop_loss_pct: Stop loss percentage (0.04 = 4%)
            max_positions: Maximum concurrent positions
        """
        self.capital = capital

        if paper_mode is None:
            paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
        self.paper_mode = paper_mode

        # Risk parameters with defaults
        self.position_size = position_size or self.DEFAULT_POSITION_SIZE
        self.take_profit_pct = take_profit_pct or self.DEFAULT_TAKE_PROFIT_PCT
        self.stop_loss_pct = stop_loss_pct or self.DEFAULT_STOP_LOSS_PCT
        self.max_positions = max_positions or self.DEFAULT_MAX_POSITIONS

        # Initialize Alpaca crypto client (drop-in replacement for Coinbase)
        self.alpaca = AlpacaCryptoClient()
        # Keep self.coinbase as alias for backward compatibility
        self.coinbase = self.alpaca

        # Track positions and trades
        self.active_positions: Dict[str, dict] = {}
        self.trades_history: List[dict] = []
        self.signals_history: List[TradeSignal] = []

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(
            f"{self.__class__.__name__} initialized - "
            f"Capital: ${capital}, Position: ${self.position_size}, "
            f"TP: {self.take_profit_pct:.1%}, SL: {self.stop_loss_pct:.1%}, "
            f"Paper: {paper_mode}"
        )

    @abstractmethod
    def run_scan(self) -> List[TradeSignal]:
        """
        Scan for trading opportunities.

        Returns:
            List of TradeSignal objects for actionable opportunities
        """
        pass

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol via Alpaca"""
        try:
            return self.alpaca.get_price(symbol)
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def get_candles(
        self,
        symbol: str,
        granularity: str = "ONE_HOUR",
        limit: int = 24
    ) -> List[dict]:
        """
        Get historical candles for a symbol via Alpaca Crypto Data API.

        Args:
            symbol: Product ID (e.g., 'BTC/USD' or 'BTC-USDC')
            granularity: Candle size (ONE_HOUR, ONE_DAY, etc.)
            limit: Number of candles to fetch

        Returns:
            List of candle dicts with: time, open, high, low, close, volume
        """
        if not self.alpaca._initialized:
            return []

        try:
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from datetime import datetime, timedelta, timezone

            alpaca_sym = self.alpaca.to_alpaca_symbol(symbol)

            # Map Coinbase-style granularity to Alpaca TimeFrame
            tf_map = {
                "ONE_MINUTE": TimeFrame.Minute,
                "FIVE_MINUTE": TimeFrame(5, 'Min'),
                "FIFTEEN_MINUTE": TimeFrame(15, 'Min'),
                "ONE_HOUR": TimeFrame.Hour,
                "SIX_HOUR": TimeFrame(6, 'Hour'),
                "ONE_DAY": TimeFrame.Day,
            }
            timeframe = tf_map.get(granularity, TimeFrame.Hour)

            # Granularity to seconds for time range calculation
            granularity_seconds = {
                "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900,
                "ONE_HOUR": 3600, "SIX_HOUR": 21600, "ONE_DAY": 86400,
            }
            seconds = granularity_seconds.get(granularity, 3600)

            end = datetime.now(timezone.utc)
            start = end - timedelta(seconds=seconds * limit * 2)

            data_client = self.alpaca.data_client or CryptoHistoricalDataClient()
            request = CryptoBarsRequest(
                symbol_or_symbols=alpaca_sym,
                timeframe=timeframe,
                start=start,
                end=end
            )
            bars_data = data_client.get_crypto_bars(request)

            candles = []
            if hasattr(bars_data, 'df'):
                df = bars_data.df.reset_index()
                for _, row in df.iterrows():
                    candles.append({
                        'time': int(row['timestamp'].timestamp()) if hasattr(row['timestamp'], 'timestamp') else 0,
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']),
                    })

            candles.sort(key=lambda x: x['time'])
            return candles[-limit:]

        except Exception as e:
            self.logger.error(f"Error getting candles for {symbol}: {e}")
            return []

    def calculate_position_size(self, price: float) -> float:
        """Calculate position size in USD, respecting capital limits"""
        max_from_capital = self.capital / self.max_positions
        return min(self.position_size, max_from_capital)

    def execute_signal(self, signal: TradeSignal) -> Optional[dict]:
        """
        Execute a trade signal.

        Args:
            signal: TradeSignal to execute

        Returns:
            Trade dict if executed, None otherwise
        """
        # GUARD: Reject trades where notional value exceeds capital
        if signal.position_size_usd > self.capital:
            self.logger.error(
                f"POSITION SIZE GUARD: {signal.symbol} size ${signal.position_size_usd:.2f} "
                f"exceeds capital ${self.capital:.2f} — trade rejected"
            )
            return None

        trade = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': signal.symbol,
            'side': signal.side,
            'entry_price': signal.entry_price,
            'target_price': signal.target_price,
            'stop_loss': signal.stop_loss,
            'position_size_usd': signal.position_size_usd,
            'reason': signal.reason,
            'paper': self.paper_mode,
            'status': 'pending'
        }

        if self.paper_mode:
            trade['status'] = 'filled'
            trade['filled_avg_price'] = signal.entry_price
            self.logger.info(
                f"[PAPER] {signal.side} {signal.symbol} @ ${signal.entry_price:.6f}, "
                f"Target: ${signal.target_price:.6f}, Stop: ${signal.stop_loss:.6f}, "
                f"Size: ${signal.position_size_usd:.2f}"
            )

            # Track position
            self.active_positions[signal.symbol] = {
                'entry_price': signal.entry_price,
                'target_price': signal.target_price,
                'stop_loss': signal.stop_loss,
                'size_usd': signal.position_size_usd,
                'entry_time': datetime.now(timezone.utc),
            }
        else:
            # Live execution via Alpaca
            if signal.side == 'BUY':
                order = self.alpaca.create_market_order(
                    product_id=signal.symbol,
                    side='BUY',
                    quote_size=str(signal.position_size_usd)
                )
                if order and order.get('success', True):
                    trade['order_id'] = order.get('order_id')
                    trade['status'] = 'filled'

                    # CRITICAL: Use broker's actual fill price, not signal estimate
                    broker_fill_price = float(order.get('filled_avg_price', 0))
                    if broker_fill_price > 0:
                        actual_entry = broker_fill_price
                        if abs(actual_entry - signal.entry_price) / signal.entry_price > 0.02:
                            self.logger.warning(
                                f"[PRICE DRIFT] {signal.symbol}: signal estimated ${signal.entry_price:.6f}, "
                                f"broker filled @ ${actual_entry:.6f} "
                                f"(diff: {abs(actual_entry - signal.entry_price) / signal.entry_price:.2%})"
                            )
                    else:
                        actual_entry = signal.entry_price
                        self.logger.warning(
                            f"[FILL PRICE MISSING] {signal.symbol}: broker returned no fill price, "
                            f"using signal estimate ${signal.entry_price:.6f}"
                        )

                    trade['entry_price'] = actual_entry
                    trade['filled_avg_price'] = actual_entry
                    self.logger.info(
                        f"[LIVE] Bought {signal.symbol} @ ${actual_entry:.6f} (broker fill)"
                    )

                    # Recompute TP/SL from actual fill price
                    actual_target = actual_entry * (1 + self.take_profit_pct)
                    actual_stop = actual_entry * (1 - self.stop_loss_pct)

                    self.active_positions[signal.symbol] = {
                        'entry_price': actual_entry,
                        'target_price': actual_target,
                        'stop_loss': actual_stop,
                        'size_usd': signal.position_size_usd,
                        'entry_time': datetime.now(timezone.utc),
                        'order_id': trade['order_id'],
                    }
                else:
                    trade['status'] = 'failed'

        self.trades_history.append(trade)
        self.signals_history.append(signal)
        return trade

    def execute_trade(self, signal: dict) -> dict:
        """
        Execute a trade from orchestrator signal dict format.
        This bridges the orchestrator's signal format with the bot's execution.

        Args:
            signal: Dict with keys: action, symbol, price, quantity, etc.

        Returns:
            Dict with execution result (status, success, etc.)
        """
        action = signal.get('action', signal.get('side', '')).upper()
        symbol = signal.get('symbol', '')

        # CRITICAL: For exits, ALWAYS use live market price, never fall back to entry_price
        if action in ('SELL', 'CLOSE'):
            # Try to get price from signal, then fetch live price if missing
            price = signal.get('price') or signal.get('exit_price')
            if not price:
                live_price = self.get_price(symbol)
                if live_price:
                    price = float(live_price)
                    self.logger.warning(f"Exit signal missing price for {symbol}, using live price ${price:.6f}")
                else:
                    self.logger.error(f"Cannot get live price for {symbol} exit - trade aborted")
                    return {'success': False, 'error': 'No exit price available'}
            else:
                price = float(price)
        else:
            # For buys, entry_price fallback is acceptable
            price = float(signal.get('price', signal.get('entry_price', 0)))

        size_usd = float(signal.get('quantity', signal.get('position_size_usd', self.position_size)))

        if action == 'BUY':
            # Create TradeSignal and use existing execute_signal
            trade_signal = TradeSignal(
                symbol=symbol,
                side='BUY',
                entry_price=price,
                target_price=price * (1 + self.take_profit_pct),
                stop_loss=price * (1 - self.stop_loss_pct),
                position_size_usd=size_usd,
                confidence=float(signal.get('confidence', 0.5)),
                reason=signal.get('reason', 'orchestrator_signal'),
                timestamp=datetime.now(timezone.utc),
                metadata=signal.get('metadata', {})
            )
            result = self.execute_signal(trade_signal)
            if result:
                result['success'] = result.get('status') == 'filled'
                return result
            return {'success': False, 'error': 'execute_signal returned None'}

        elif action in ('SELL', 'CLOSE'):
            if self.paper_mode:
                self.logger.info(f"[PAPER SELL] {symbol} @ ${price:.6f}, Size: ${size_usd:.2f}")
                if symbol in self.active_positions:
                    del self.active_positions[symbol]
                return {'status': 'filled', 'success': True, 'paper': True}
            else:
                # Live: sell via Alpaca
                try:
                    base_size = size_usd / price if price > 0 else 0
                    order = self.alpaca.create_market_order(
                        product_id=symbol,
                        side='SELL',
                        base_size=str(round(base_size, 8))
                    )
                    if order and order.get('success', True):
                        if symbol in self.active_positions:
                            del self.active_positions[symbol]
                        return {
                            'status': 'filled',
                            'success': True,
                            'order_id': order.get('order_id')
                        }
                    return {'success': False, 'error': order.get('error', 'Order failed') if order else 'Order returned None'}
                except Exception as e:
                    self.logger.error(f"Sell execution failed for {symbol}: {e}")
                    return {'success': False, 'error': str(e)}

        return {'success': False, 'error': f'Unknown action: {action}'}

    def check_exits(self) -> List[dict]:
        """
        Check active positions for exit conditions.

        Returns:
            List of closed position dicts with 'exit_price' set to LIVE market price
        """
        closed = []

        for symbol, position in list(self.active_positions.items()):
            # CRITICAL: Always fetch fresh live price for exit decisions
            current_price = self.get_price(symbol)
            if current_price is None:
                self.logger.warning(f"Cannot get live price for {symbol}, skipping exit check")
                continue

            # Validate price is reasonable (not zero, not negative)
            if current_price <= 0:
                self.logger.error(f"Invalid price ${current_price} for {symbol}, skipping exit")
                continue

            entry_price = position['entry_price']
            pnl_pct = (current_price - entry_price) / entry_price

            exit_reason = None

            # Check take profit
            if current_price >= position['target_price']:
                exit_reason = 'take_profit'

            # Check stop loss
            elif current_price <= position['stop_loss']:
                exit_reason = 'stop_loss'

            # Check max hold time if set
            if 'max_hold_time' in position:
                hold_time = (datetime.now(timezone.utc) - position['entry_time']).total_seconds()
                if hold_time > position['max_hold_time']:
                    exit_reason = 'max_hold_time'

            if exit_reason:
                result = {
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'exit_price': current_price,  # LIVE market price, NOT entry_price
                    'pnl_pct': pnl_pct,
                    'pnl_usd': position['size_usd'] * pnl_pct,
                    'exit_reason': exit_reason,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                }

                self.logger.info(
                    f"[EXIT] {symbol} - {exit_reason}: "
                    f"Entry ${entry_price:.6f} -> Exit ${current_price:.6f} (LIVE), "
                    f"PnL: {pnl_pct:.2%} (${result['pnl_usd']:.2f})"
                )

                del self.active_positions[symbol]
                closed.append(result)

        return closed

    def get_status(self) -> dict:
        """Get bot status"""
        return {
            'name': self.__class__.__name__,
            'capital': self.capital,
            'paper_mode': self.paper_mode,
            'alpaca_connected': self.alpaca._initialized,
            'active_positions': len(self.active_positions),
            'total_trades': len(self.trades_history),
            'total_signals': len(self.signals_history),
            'position_size': self.position_size,
            'take_profit_pct': self.take_profit_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'last_scan': datetime.now(timezone.utc).isoformat()
        }
