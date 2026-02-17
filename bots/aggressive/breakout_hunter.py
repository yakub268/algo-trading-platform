"""
BREAKOUT HUNTER BOT
===================

Detects volume spikes and price breakouts on Coinbase USDC pairs.

Strategy:
- Scan USDC pairs for volume spike (current volume > 3x 24h average)
- Confirm price breakout (price breaks above 24h high)
- BUY when both conditions are met
- Stop-loss: 5% below entry
- Take profit: 15% above entry OR trail after 10% gain

Position Size: $75 per trade
Risk Level: HIGH

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal

# Add parent directories for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

# Import base class (will be created by another agent)
try:
    from bots.aggressive.base_aggressive_bot import BaseAggressiveBot
    HAS_BASE_CLASS = True
except ImportError:
    HAS_BASE_CLASS = False
    BaseAggressiveBot = object  # Fallback for standalone operation

# Import CoinbaseClient for market data
try:
    from bots.coinbase_arb_bot import CoinbaseClient
    HAS_COINBASE = True
except ImportError:
    HAS_COINBASE = False
    CoinbaseClient = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BreakoutSignal:
    """A detected breakout signal"""
    symbol: str
    current_price: float
    breakout_price: float  # 24h high that was broken
    current_volume: float
    avg_volume_24h: float
    volume_ratio: float  # current / average
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: datetime
    reason: str


@dataclass
class ActivePosition:
    """Track an active position"""
    symbol: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float]
    highest_price: float  # For trailing stop
    entry_time: datetime


class BreakoutHunter(BaseAggressiveBot if HAS_BASE_CLASS else object):
    """
    Breakout Hunter Bot

    Scans Coinbase USDC pairs for breakout opportunities:
    1. Volume spike: Current volume > 3x 24h average
    2. Price breakout: Price above 24h high

    Risk Management:
    - Stop-loss: 5% below entry
    - Take profit: 15% above entry
    - Trailing stop: Activates after 10% gain, trails at 3%
    """

    # USDC pairs to scan on Coinbase
    USDC_PAIRS = [
        'BTC-USDC', 'ETH-USDC', 'SOL-USDC', 'DOGE-USDC',
        'AVAX-USDC', 'LINK-USDC', 'DOT-USDC', 'MATIC-USDC',
        'ADA-USDC', 'XRP-USDC', 'ATOM-USDC', 'UNI-USDC',
        'LTC-USDC', 'BCH-USDC', 'NEAR-USDC', 'APT-USDC',
        'ARB-USDC', 'OP-USDC', 'AAVE-USDC', 'MKR-USDC',
    ]

    # Strategy parameters
    VOLUME_SPIKE_THRESHOLD = 3.0   # Volume must be 3x average
    POSITION_SIZE_USD = 75.0       # $75 per trade
    STOP_LOSS_PCT = 0.05           # 5% stop loss
    TAKE_PROFIT_PCT = 0.15         # 15% take profit
    TRAIL_ACTIVATION_PCT = 0.10    # Trailing stop activates after 10% gain
    TRAIL_DISTANCE_PCT = 0.03      # Trail 3% behind highest price

    # Risk limits
    MAX_POSITIONS = 3              # Max concurrent positions
    MAX_DAILY_TRADES = 10          # Max trades per day
    COOLDOWN_MINUTES = 15          # Cooldown between trades on same pair

    def __init__(self, capital: float = 225.0, paper_mode: bool = None):
        """
        Initialize Breakout Hunter.

        Args:
            capital: Trading capital (default $225 for aggressive allocation)
            paper_mode: Paper trading mode (default from env)
        """
        # Initialize base class if available
        if HAS_BASE_CLASS:
            super().__init__(
                capital=capital,
                paper_mode=paper_mode
            )
            self.name = 'BreakoutHunter'
        else:
            self.name = 'BreakoutHunter'
            self.capital = capital
            if paper_mode is None:
                paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
            self.paper_mode = paper_mode

        # Initialize Coinbase client
        self.coinbase = CoinbaseClient() if HAS_COINBASE else None

        # Track positions and state
        self.positions: Dict[str, ActivePosition] = {}
        self.trades_today: List[dict] = []
        self.last_trade_time: Dict[str, datetime] = {}  # Per-symbol cooldown
        self.pnl_today: float = 0.0

        # Cache for 24h data
        self._cache: Dict[str, dict] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=5)

        logger.info(f"BreakoutHunter initialized - Capital: ${capital}, Paper: {self.paper_mode}")
        logger.info(f"  Scanning {len(self.USDC_PAIRS)} USDC pairs")
        logger.info(f"  Volume threshold: {self.VOLUME_SPIKE_THRESHOLD}x, Position: ${self.POSITION_SIZE_USD}")

    def _get_24h_stats(self, symbol: str) -> Optional[dict]:
        """
        Get 24h statistics for a symbol (volume, high, low).

        Returns:
            Dict with 'volume_24h', 'high_24h', 'low_24h', 'current_price'
        """
        if not self.coinbase or not self.coinbase._initialized:
            return self._get_mock_stats(symbol)

        try:
            # Get product details for 24h stats
            product = self.coinbase.get_product(symbol)

            if not product:
                return None

            # Extract 24h stats from product
            stats = {
                'symbol': symbol,
                'volume_24h': float(getattr(product, 'volume_24h', 0) or 0),
                'high_24h': float(getattr(product, 'high_24h', 0) or 0),
                'low_24h': float(getattr(product, 'low_24h', 0) or 0),
                'current_price': float(getattr(product, 'price', 0) or 0),
                'price_change_24h': float(getattr(product, 'price_percentage_change_24h', 0) or 0),
            }

            # Also get current bid/ask
            pricebooks = self.coinbase.get_best_bid_ask([symbol])
            if pricebooks:
                book = pricebooks[0]
                bids = book.bids if hasattr(book, 'bids') else book.get('bids', [])
                asks = book.asks if hasattr(book, 'asks') else book.get('asks', [])

                if bids:
                    stats['bid'] = float(bids[0].price if hasattr(bids[0], 'price') else bids[0].get('price', 0))
                if asks:
                    stats['ask'] = float(asks[0].price if hasattr(asks[0], 'price') else asks[0].get('price', 0))
                    stats['current_price'] = stats['ask']  # Use ask for buying

            return stats

        except Exception as e:
            logger.debug(f"Error getting stats for {symbol}: {e}")
            return None

    def _get_mock_stats(self, symbol: str) -> dict:
        """Generate mock stats for testing when Coinbase unavailable"""
        import random

        base_prices = {
            'BTC-USDC': 95000, 'ETH-USDC': 3200, 'SOL-USDC': 180,
            'DOGE-USDC': 0.35, 'AVAX-USDC': 45, 'LINK-USDC': 22,
            'DOT-USDC': 8, 'MATIC-USDC': 0.90, 'ADA-USDC': 0.95,
            'XRP-USDC': 2.50, 'ATOM-USDC': 12, 'UNI-USDC': 15,
        }

        base = base_prices.get(symbol, 100)
        price = base * (1 + random.uniform(-0.05, 0.08))
        high_24h = price * (1 + random.uniform(0.01, 0.05))
        low_24h = price * (1 - random.uniform(0.02, 0.06))

        # Simulate volume spike occasionally
        avg_volume = random.uniform(10000, 100000)
        if random.random() < 0.15:  # 15% chance of volume spike
            current_volume = avg_volume * random.uniform(3.5, 6.0)
        else:
            current_volume = avg_volume * random.uniform(0.8, 1.5)

        return {
            'symbol': symbol,
            'volume_24h': avg_volume,
            'current_volume': current_volume,
            'high_24h': high_24h,
            'low_24h': low_24h,
            'current_price': price,
            'bid': price * 0.999,
            'ask': price,
            'price_change_24h': random.uniform(-5, 10),
        }

    def _estimate_current_volume(self, symbol: str, volume_24h: float) -> float:
        """
        Estimate current volume rate compared to 24h average.

        In production, this would use recent trade data or websocket feeds.
        For now, we simulate or use available data.
        """
        # In production: fetch recent trades and calculate volume rate
        # Here we use a simplified approach

        if not self.coinbase or not self.coinbase._initialized:
            # Mock: return volume from mock stats
            mock = self._get_mock_stats(symbol)
            return mock.get('current_volume', volume_24h)

        # For real implementation, you'd want to:
        # 1. Get recent trades (last hour)
        # 2. Extrapolate to 24h rate
        # 3. Compare to actual 24h volume

        # Simplified: assume current rate is similar to 24h average
        # with some variance based on time of day
        hour = datetime.now(timezone.utc).hour

        # Higher volume during US trading hours (14-22 UTC)
        if 14 <= hour <= 22:
            multiplier = 1.3
        elif 6 <= hour <= 14:  # Asian/European overlap
            multiplier = 1.1
        else:
            multiplier = 0.8

        return volume_24h * multiplier

    def _check_breakout(self, stats: dict) -> Optional[BreakoutSignal]:
        """
        Check if a symbol meets breakout criteria.

        Criteria:
        1. Volume spike: current volume rate > 3x 24h average
        2. Price breakout: current price > 24h high

        Returns:
            BreakoutSignal if breakout detected, None otherwise
        """
        symbol = stats['symbol']
        current_price = stats.get('current_price', 0)
        high_24h = stats.get('high_24h', 0)
        volume_24h = stats.get('volume_24h', 0)

        if current_price <= 0 or high_24h <= 0 or volume_24h <= 0:
            return None

        # Get current volume estimate
        current_volume = stats.get('current_volume') or self._estimate_current_volume(symbol, volume_24h)

        # Calculate volume ratio
        volume_ratio = current_volume / volume_24h if volume_24h > 0 else 0

        # Check volume spike
        has_volume_spike = volume_ratio >= self.VOLUME_SPIKE_THRESHOLD

        # Check price breakout (above 24h high)
        has_price_breakout = current_price > high_24h

        if not has_volume_spike or not has_price_breakout:
            return None

        # Both conditions met - generate signal
        entry_price = stats.get('ask', current_price)
        stop_loss = entry_price * (1 - self.STOP_LOSS_PCT)
        take_profit = entry_price * (1 + self.TAKE_PROFIT_PCT)

        # Calculate confidence based on how strong the signals are
        volume_confidence = min(1.0, (volume_ratio - self.VOLUME_SPIKE_THRESHOLD) / 2)
        breakout_strength = (current_price - high_24h) / high_24h
        breakout_confidence = min(1.0, breakout_strength * 20)  # 5% breakout = 100% confidence

        confidence = (volume_confidence + breakout_confidence) / 2

        signal = BreakoutSignal(
            symbol=symbol,
            current_price=current_price,
            breakout_price=high_24h,
            current_volume=current_volume,
            avg_volume_24h=volume_24h,
            volume_ratio=volume_ratio,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            reason=f"Volume {volume_ratio:.1f}x avg, broke above 24h high ${high_24h:.2f}"
        )

        logger.info(f"BREAKOUT DETECTED: {symbol}")
        logger.info(f"  Price: ${current_price:.4f} > 24h high ${high_24h:.4f}")
        logger.info(f"  Volume: {volume_ratio:.1f}x average")
        logger.info(f"  Confidence: {confidence:.1%}")

        return signal

    def _can_trade(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if we can trade a symbol.

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check max positions
        if len(self.positions) >= self.MAX_POSITIONS:
            return False, f"Max positions reached ({self.MAX_POSITIONS})"

        # Check if already in position
        if symbol in self.positions:
            return False, f"Already in position for {symbol}"

        # Check daily trade limit
        if len(self.trades_today) >= self.MAX_DAILY_TRADES:
            return False, f"Max daily trades reached ({self.MAX_DAILY_TRADES})"

        # Check cooldown
        last_trade = self.last_trade_time.get(symbol)
        if last_trade:
            time_since = datetime.now(timezone.utc) - last_trade
            if time_since < timedelta(minutes=self.COOLDOWN_MINUTES):
                remaining = self.COOLDOWN_MINUTES - time_since.total_seconds() / 60
                return False, f"Cooldown active ({remaining:.1f}min remaining)"

        return True, "OK"

    def _execute_entry(self, signal: BreakoutSignal) -> Optional[dict]:
        """
        Execute an entry trade.

        Returns:
            Trade record dict or None if failed
        """
        can_trade, reason = self._can_trade(signal.symbol)
        if not can_trade:
            logger.info(f"Cannot trade {signal.symbol}: {reason}")
            return None

        # Calculate quantity
        quantity = self.POSITION_SIZE_USD / signal.entry_price

        trade = {
            'timestamp': signal.timestamp.isoformat(),
            'symbol': signal.symbol,
            'side': 'BUY',
            'quantity': quantity,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'volume_ratio': signal.volume_ratio,
            'confidence': signal.confidence,
            'reason': signal.reason,
            'paper': self.paper_mode,
            'status': 'pending'
        }

        if self.paper_mode:
            # Paper trade - simulate fill
            trade['status'] = 'filled'
            trade['fill_price'] = signal.entry_price

            # Track position
            self.positions[signal.symbol] = ActivePosition(
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                quantity=quantity,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                trailing_stop=None,
                highest_price=signal.entry_price,
                entry_time=signal.timestamp
            )

            logger.info(
                f"[PAPER] BUY {signal.symbol}: {quantity:.6f} @ ${signal.entry_price:.4f} "
                f"| Stop: ${signal.stop_loss:.4f} | TP: ${signal.take_profit:.4f}"
            )

        else:
            # Live trade via Coinbase
            if self.coinbase and self.coinbase._initialized:
                try:
                    order = self.coinbase.create_market_order(
                        product_id=signal.symbol,
                        side='BUY',
                        quote_size=str(self.POSITION_SIZE_USD)
                    )

                    if order:
                        trade['order_id'] = getattr(order, 'order_id', None)
                        trade['status'] = 'submitted'

                        # Track position (will need to verify fill)
                        self.positions[signal.symbol] = ActivePosition(
                            symbol=signal.symbol,
                            entry_price=signal.entry_price,
                            quantity=quantity,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            trailing_stop=None,
                            highest_price=signal.entry_price,
                            entry_time=signal.timestamp
                        )

                        logger.info(f"[LIVE] BUY order submitted: {signal.symbol}")
                    else:
                        trade['status'] = 'failed'
                        trade['error'] = 'Order returned None'

                except Exception as e:
                    trade['status'] = 'failed'
                    trade['error'] = str(e)
                    logger.error(f"Order failed: {e}")
            else:
                trade['status'] = 'failed'
                trade['error'] = 'Coinbase not available'

        # Record trade
        self.trades_today.append(trade)
        self.last_trade_time[signal.symbol] = datetime.now(timezone.utc)

        return trade

    def _check_exits(self) -> List[dict]:
        """
        Check all positions for exit conditions.

        Exit triggers:
        1. Stop loss hit
        2. Take profit hit
        3. Trailing stop hit (if activated)

        Returns:
            List of exit trades executed
        """
        exits = []

        for symbol, position in list(self.positions.items()):
            stats = self._get_24h_stats(symbol)
            if not stats:
                continue

            current_price = stats.get('current_price', 0)
            if current_price <= 0:
                continue

            should_exit = False
            exit_reason = ""
            exit_price = current_price

            # Update highest price for trailing stop
            if current_price > position.highest_price:
                position.highest_price = current_price

                # Check if we should activate trailing stop
                gain_pct = (current_price - position.entry_price) / position.entry_price
                if gain_pct >= self.TRAIL_ACTIVATION_PCT and position.trailing_stop is None:
                    position.trailing_stop = current_price * (1 - self.TRAIL_DISTANCE_PCT)
                    logger.info(f"{symbol}: Trailing stop activated at ${position.trailing_stop:.4f}")
                elif position.trailing_stop:
                    # Update trailing stop
                    new_trail = current_price * (1 - self.TRAIL_DISTANCE_PCT)
                    if new_trail > position.trailing_stop:
                        position.trailing_stop = new_trail

            # Check exit conditions
            if current_price <= position.stop_loss:
                should_exit = True
                exit_reason = f"Stop loss hit (${position.stop_loss:.4f})"
                exit_price = position.stop_loss

            elif current_price >= position.take_profit:
                should_exit = True
                exit_reason = f"Take profit hit (${position.take_profit:.4f})"
                exit_price = position.take_profit

            elif position.trailing_stop and current_price <= position.trailing_stop:
                should_exit = True
                exit_reason = f"Trailing stop hit (${position.trailing_stop:.4f})"
                exit_price = position.trailing_stop

            if should_exit:
                exit_trade = self._execute_exit(position, exit_price, exit_reason)
                if exit_trade:
                    exits.append(exit_trade)

        return exits

    def _execute_exit(self, position: ActivePosition, exit_price: float, reason: str) -> Optional[dict]:
        """Execute an exit trade."""

        pnl = (exit_price - position.entry_price) * position.quantity
        pnl_pct = (exit_price - position.entry_price) / position.entry_price

        trade = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': position.symbol,
            'side': 'SELL',
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'hold_time_minutes': (datetime.now(timezone.utc) - position.entry_time).total_seconds() / 60,
            'paper': self.paper_mode,
            'status': 'pending'
        }

        if self.paper_mode:
            trade['status'] = 'filled'

            logger.info(
                f"[PAPER] SELL {position.symbol}: {position.quantity:.6f} @ ${exit_price:.4f} "
                f"| PnL: ${pnl:.2f} ({pnl_pct:.1%}) | {reason}"
            )

        else:
            # Live trade via Coinbase
            if self.coinbase and self.coinbase._initialized:
                try:
                    order = self.coinbase.create_market_order(
                        product_id=position.symbol,
                        side='SELL',
                        quote_size=str(position.quantity * exit_price)
                    )

                    if order:
                        trade['order_id'] = getattr(order, 'order_id', None)
                        trade['status'] = 'submitted'
                        logger.info(f"[LIVE] SELL order submitted: {position.symbol}")
                    else:
                        trade['status'] = 'failed'

                except Exception as e:
                    trade['status'] = 'failed'
                    trade['error'] = str(e)
                    logger.error(f"Exit order failed: {e}")
            else:
                trade['status'] = 'failed'

        # Update state
        if trade['status'] in ('filled', 'submitted'):
            self.pnl_today += pnl
            del self.positions[position.symbol]

        self.trades_today.append(trade)
        return trade

    def run_scan(self) -> List[dict]:
        """
        Scan for breakouts and manage positions.

        Returns:
            List of trade signals/executions
        """
        logger.info("BreakoutHunter scanning...")
        results = []

        # First, check existing positions for exits
        exits = self._check_exits()
        results.extend(exits)

        # Then scan for new breakouts
        for symbol in self.USDC_PAIRS:
            try:
                # Skip if we can't trade
                can_trade, _ = self._can_trade(symbol)
                if not can_trade:
                    continue

                # Get stats
                stats = self._get_24h_stats(symbol)
                if not stats:
                    continue

                # Check for breakout
                signal = self._check_breakout(stats)
                if signal:
                    trade = self._execute_entry(signal)
                    if trade:
                        results.append(trade)

            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
                continue

        logger.info(f"Scan complete: {len(results)} actions, {len(self.positions)} open positions")
        return results

    def get_status(self) -> dict:
        """Get bot status."""
        return {
            'name': self.name,
            'capital': self.capital,
            'paper_mode': self.paper_mode,
            'coinbase_connected': self.coinbase._initialized if self.coinbase else False,
            'positions': len(self.positions),
            'trades_today': len(self.trades_today),
            'pnl_today': self.pnl_today,
            'position_details': [
                {
                    'symbol': p.symbol,
                    'entry': p.entry_price,
                    'current_stop': p.trailing_stop or p.stop_loss,
                    'highest': p.highest_price
                }
                for p in self.positions.values()
            ],
            'last_scan': datetime.now(timezone.utc).isoformat()
        }


if __name__ == "__main__":
    print("=" * 60)
    print("BREAKOUT HUNTER BOT - TEST RUN")
    print("=" * 60)
    print("Strategy: Volume Spike + Price Breakout on Coinbase USDC pairs")
    print(f"  Volume threshold: 3x 24h average")
    print(f"  Stop loss: 5%")
    print(f"  Take profit: 15%")
    print(f"  Trailing stop: Activates at 10% gain, trails 3%")
    print(f"  Position size: $75")
    print("=" * 60)

    bot = BreakoutHunter(capital=225.0, paper_mode=True)

    print(f"\nStatus: {bot.get_status()}")

    # Run a few scans
    for i in range(3):
        print(f"\n--- Scan {i+1} ---")
        trades = bot.run_scan()

        if trades:
            print(f"Actions: {len(trades)}")
            for t in trades:
                side = t.get('side', 'UNKNOWN')
                symbol = t.get('symbol', '')
                if side == 'BUY':
                    print(f"  BUY {symbol} @ ${t.get('entry_price', 0):.4f} "
                          f"| Volume {t.get('volume_ratio', 0):.1f}x")
                else:
                    print(f"  SELL {symbol} @ ${t.get('exit_price', 0):.4f} "
                          f"| PnL: ${t.get('pnl', 0):.2f}")
        else:
            print("  No actions")

    print(f"\nFinal Status: {bot.get_status()}")
    print("=" * 60)
