"""
CRYPTO DCA ACCUMULATION BOT
===========================

Executes DCA accumulation strategy for XRP, HBAR, XLM on Coinbase.

Strategy:
- Monitor prices for entry zone opportunities
- Execute partial buys when price enters DCA zones
- Track average cost basis across multiple DCA batches
- Monitor stop losses for existing positions

Allocations:
- XRP: $0 (user already has position, monitor only with $1.50 stop)
- HBAR: $300 allocation
- XLM: $200 allocation

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Import CoinbaseClient from arb bot
from bots.coinbase_arb_bot import CoinbaseClient

# Import strategy
from strategies.crypto_dca_accumulation import CryptoDCAStrategy, TOKEN_CONFIGS

# Import Telegram alerts
try:
    from utils.telegram_alerts import send_strategy_alert as send_telegram_alert
except ImportError:
    def send_telegram_alert(signal: Dict) -> bool:
        """Fallback if telegram not configured"""
        logging.info(f"[ALERT] {signal}")
        return True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CryptoDCABot')


@dataclass
class DCAPosition:
    """Track a DCA position for a token"""
    symbol: str
    product_id: str
    quantity: float = 0.0
    total_cost: float = 0.0
    batches: List[Dict] = field(default_factory=list)

    @property
    def average_cost(self) -> float:
        """Calculate weighted average cost"""
        if self.quantity == 0:
            return 0.0
        return self.total_cost / self.quantity

    def add_batch(self, price: float, qty: float, amount_usd: float):
        """Record a DCA buy"""
        self.quantity += qty
        self.total_cost += amount_usd
        self.batches.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'price': price,
            'quantity': qty,
            'amount_usd': amount_usd
        })

    def get_pnl(self, current_price: float) -> tuple:
        """Get P&L for position"""
        if self.quantity == 0:
            return 0.0, 0.0
        current_value = self.quantity * current_price
        pnl = current_value - self.total_cost
        pnl_pct = pnl / self.total_cost if self.total_cost > 0 else 0
        return pnl, pnl_pct


class CryptoDCABot:
    """
    Crypto DCA Accumulation Bot for Coinbase.

    Monitors XRP, HBAR, XLM and executes DCA buys when price
    enters defined entry zones.
    """

    # Token configurations
    TOKENS = ['XRP', 'HBAR', 'XLM']
    PRODUCT_IDS = {
        'XRP': 'XRP-USDC',
        'HBAR': 'HBAR-USDC',
        'XLM': 'XLM-USDC'
    }
    # Note: Using USDC pairs since user has USDC balance, not USD

    # Allocation amounts
    ALLOCATIONS = {
        'XRP': 0.0,      # User already has position - monitor only
        'HBAR': 300.0,   # $300 to accumulate HBAR
        'XLM': 200.0     # $200 to accumulate XLM
    }

    # Stop losses
    STOP_LOSSES = {
        'XRP': 1.50,     # User is down $3k, stop at $1.50
        'HBAR': 0.085,
        'XLM': 0.17
    }

    def __init__(self, capital: float = 500.0, paper_mode: bool = None):
        """
        Initialize DCA Bot.

        Args:
            capital: Total capital for DCA ($500 default = $300 HBAR + $200 XLM)
            paper_mode: Paper trading mode (reads from env if None)
        """
        self.capital = capital

        # Safe default: paper mode unless explicitly set
        if paper_mode is None:
            paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
        self.paper_mode = paper_mode

        # Initialize Coinbase client
        self.coinbase = CoinbaseClient()

        # Initialize strategies for each token
        self.strategies: Dict[str, CryptoDCAStrategy] = {}
        for token in self.TOKENS:
            self.strategies[token] = CryptoDCAStrategy(
                symbol=token,
                paper_mode=paper_mode
            )

        # Track positions
        self.positions: Dict[str, DCAPosition] = {}
        for token in self.TOKENS:
            self.positions[token] = DCAPosition(
                symbol=token,
                product_id=self.PRODUCT_IDS[token]
            )

        # Track spending per token
        self.spent: Dict[str, float] = {token: 0.0 for token in self.TOKENS}

        # Trade history
        self.trades_today: List[Dict] = []

        # Price cache
        self.last_prices: Dict[str, float] = {}

        # Auto-load existing XRP position (user has ~1251 XRP @ $2.89)
        # This ensures position is tracked even when running via orchestrator
        self._load_existing_xrp_position()

        logger.info(
            f"CryptoDCABot initialized - Capital: ${capital}, Paper: {paper_mode}, "
            f"Coinbase: {'Connected' if self.coinbase._initialized else 'Not Connected'}"
        )

    def _load_existing_xrp_position(self):
        """Load the user's existing XRP position for monitoring."""
        # Real position: $3,615 invested at $2.89 avg = ~1251 XRP
        xrp_qty = 3615.0 / 2.89  # ~1250.87 XRP
        xrp_avg_cost = 2.89
        xrp_total_cost = 3615.0

        self.positions['XRP'].quantity = xrp_qty
        self.positions['XRP'].total_cost = xrp_total_cost
        self.positions['XRP'].batches.append({
            'timestamp': '2025-01-01T00:00:00+00:00',  # Historical
            'price': xrp_avg_cost,
            'quantity': xrp_qty,
            'amount_usd': xrp_total_cost,
            'note': 'Existing position loaded automatically'
        })
        logger.info(f"Loaded existing XRP position: {xrp_qty:.4f} @ ${xrp_avg_cost}")

    def get_prices(self) -> Dict[str, float]:
        """
        Fetch current prices for XRP-USD, HBAR-USD, XLM-USD from Coinbase.

        Returns:
            Dict of symbol -> price
        """
        prices = {}
        product_ids = list(self.PRODUCT_IDS.values())

        try:
            if self.coinbase._initialized:
                pricebooks = self.coinbase.get_best_bid_ask(product_ids)

                for book in pricebooks:
                    product_id = book.product_id if hasattr(book, 'product_id') else book.get('product_id', '')
                    asks = book.asks if hasattr(book, 'asks') else book.get('asks', [])

                    if asks:
                        ask_price = float(asks[0].price if hasattr(asks[0], 'price') else asks[0].get('price', 0))

                        # Map product_id back to symbol
                        for symbol, pid in self.PRODUCT_IDS.items():
                            if pid == product_id:
                                prices[symbol] = ask_price
                                break

                logger.debug(f"Fetched prices: {prices}")
            else:
                # Fallback to mock prices for testing
                prices = self._get_mock_prices()

        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            prices = self._get_mock_prices()

        self.last_prices = prices
        return prices

    def _get_mock_prices(self) -> Dict[str, float]:
        """Generate mock prices for paper trading"""
        import random
        return {
            'XRP': 2.10 + random.uniform(-0.20, 0.20),
            'HBAR': 0.11 + random.uniform(-0.02, 0.02),
            'XLM': 0.22 + random.uniform(-0.03, 0.03)
        }

    def analyze_all(self) -> Dict[str, Dict]:
        """
        Run strategy analysis on all three tokens.

        Returns:
            Dict of symbol -> analysis result
        """
        prices = self.get_prices()
        results = {}

        for symbol in self.TOKENS:
            price = prices.get(symbol)
            if price is None:
                continue

            strategy = self.strategies[symbol]
            config = TOKEN_CONFIGS[symbol]

            # Determine zone
            zone = self._get_price_zone(symbol, price)

            # Calculate remaining allocation
            remaining = self.ALLOCATIONS[symbol] - self.spent[symbol]

            # Check stop loss
            stop_triggered = price < self.STOP_LOSSES[symbol]

            # Get position P&L
            position = self.positions[symbol]
            pnl, pnl_pct = position.get_pnl(price)

            results[symbol] = {
                'symbol': symbol,
                'price': price,
                'zone': zone,
                'allocation': self.ALLOCATIONS[symbol],
                'spent': self.spent[symbol],
                'remaining': remaining,
                'stop_loss': self.STOP_LOSSES[symbol],
                'stop_triggered': stop_triggered,
                'position_qty': position.quantity,
                'position_avg_cost': position.average_cost,
                'position_pnl': pnl,
                'position_pnl_pct': pnl_pct,
                'batches': len(position.batches),
                'should_buy': zone in ['aggressive', 'conservative'] and remaining > 10 and not stop_triggered,
                'should_alert_stop': stop_triggered and position.quantity > 0
            }

            logger.info(
                f"{symbol}: ${price:.4f} ({zone}) | "
                f"Remaining: ${remaining:.2f} | "
                f"Position: {position.quantity:.4f} @ ${position.average_cost:.4f}"
            )

        return results

    def _get_price_zone(self, symbol: str, price: float) -> str:
        """Determine which zone the price is in"""
        config = TOKEN_CONFIGS[symbol]

        if price < config.stop_loss:
            return 'below_stop'
        elif config.aggressive_entry_low <= price <= config.aggressive_entry_high:
            return 'aggressive'
        elif config.conservative_entry_low and config.conservative_entry_high:
            if config.conservative_entry_low <= price <= config.conservative_entry_high:
                return 'conservative'
        elif price > config.aggressive_entry_high:
            return 'above_zone'

        return 'neutral'

    def execute_dca_buy(self, symbol: str, amount_usd: float) -> Optional[Dict]:
        """
        Execute a DCA buy for a token.

        Args:
            symbol: Token symbol (XRP, HBAR, XLM)
            amount_usd: USD amount to spend

        Returns:
            Trade result dict or None
        """
        if symbol not in self.TOKENS:
            logger.error(f"Unknown symbol: {symbol}")
            return None

        # Check remaining allocation
        remaining = self.ALLOCATIONS[symbol] - self.spent[symbol]
        if amount_usd > remaining:
            amount_usd = remaining

        if amount_usd < 1.0:  # Minimum $1 trade
            logger.info(f"Insufficient remaining allocation for {symbol}")
            return None

        # Get current price
        prices = self.get_prices()
        price = prices.get(symbol)
        if price is None:
            logger.error(f"Could not get price for {symbol}")
            return None

        # Check stop loss
        if price < self.STOP_LOSSES[symbol]:
            logger.warning(f"{symbol} below stop loss - not buying")
            return None

        # Calculate quantity
        quantity = amount_usd / price

        trade = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'product_id': self.PRODUCT_IDS[symbol],
            'side': 'BUY',
            'price': price,
            'quantity': quantity,
            'amount_usd': amount_usd,
            'paper': self.paper_mode,
            'status': 'pending'
        }

        if self.paper_mode:
            # Paper trade - simulate fill
            trade['status'] = 'filled'
            trade['fill_price'] = price

            # Update position
            self.positions[symbol].add_batch(price, quantity, amount_usd)
            self.spent[symbol] += amount_usd

            logger.info(
                f"[PAPER] DCA BUY {symbol}: {quantity:.6f} @ ${price:.4f} "
                f"(${amount_usd:.2f}) | Avg: ${self.positions[symbol].average_cost:.4f}"
            )

            # Send alert
            send_telegram_alert({
                'strategy': 'crypto_dca',
                'symbol': symbol,
                'signal': 'buy',
                'price': price,
                'confidence': 0.8,
                'reasoning': f"DCA buy: {quantity:.6f} {symbol} @ ${price:.4f}"
            })

        else:
            # Live trade via Coinbase
            try:
                order = self.coinbase.create_market_order(
                    product_id=self.PRODUCT_IDS[symbol],
                    side='BUY',
                    quote_size=str(round(amount_usd, 2))
                )

                if order:
                    trade['order_id'] = getattr(order, 'order_id', str(order))
                    trade['status'] = 'submitted'

                    # Update position (assume fill at current price)
                    self.positions[symbol].add_batch(price, quantity, amount_usd)
                    self.spent[symbol] += amount_usd

                    logger.info(f"[LIVE] DCA BUY {symbol}: Order submitted - {order}")

                    # Send alert
                    send_telegram_alert({
                        'strategy': 'crypto_dca',
                        'symbol': symbol,
                        'signal': 'buy',
                        'price': price,
                        'confidence': 0.9,
                        'reasoning': f"LIVE DCA buy: {quantity:.6f} {symbol} @ ${price:.4f}"
                    })
                else:
                    trade['status'] = 'failed'
                    logger.error(f"Order failed for {symbol}")

            except Exception as e:
                trade['status'] = 'error'
                trade['error'] = str(e)
                logger.error(f"Order error for {symbol}: {e}")

        self.trades_today.append(trade)
        return trade

    def check_stop_losses(self) -> List[Dict]:
        """
        Monitor stop losses and alert if triggered.

        Returns:
            List of stop loss alerts
        """
        prices = self.get_prices()
        alerts = []

        for symbol in self.TOKENS:
            price = prices.get(symbol)
            if price is None:
                continue

            stop_price = self.STOP_LOSSES[symbol]
            position = self.positions[symbol]

            if price < stop_price:
                pnl, pnl_pct = position.get_pnl(price)

                alert = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'alert_type': 'STOP_LOSS',
                    'current_price': price,
                    'stop_price': stop_price,
                    'position_qty': position.quantity,
                    'position_avg_cost': position.average_cost,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                }
                alerts.append(alert)

                logger.warning(
                    f"STOP LOSS ALERT: {symbol} @ ${price:.4f} < ${stop_price:.4f} | "
                    f"Position: {position.quantity:.4f} | P&L: ${pnl:.2f} ({pnl_pct:.1%})"
                )

                # Send Telegram alert
                send_telegram_alert({
                    'strategy': 'crypto_dca',
                    'symbol': symbol,
                    'signal': 'sell',
                    'price': price,
                    'confidence': 1.0,
                    'stop_loss': stop_price,
                    'reasoning': f"STOP LOSS: {symbol} @ ${price:.4f} below ${stop_price:.4f}. "
                                f"Position: {position.quantity:.4f} ({pnl_pct:+.1%})"
                })

        return alerts

    def run_scan(self) -> Dict:
        """
        Main loop method - analyze all tokens and execute DCA buys if appropriate.

        Returns:
            Scan results
        """
        logger.info("=" * 50)
        logger.info("Running DCA scan...")

        # Check stop losses first
        stop_alerts = self.check_stop_losses()

        # Analyze all tokens
        analysis = self.analyze_all()

        # Execute DCA buys where appropriate
        executed_trades = []
        for symbol, result in analysis.items():
            if result['should_buy']:
                # Calculate buy amount based on zone
                zone = result['zone']
                remaining = result['remaining']

                # DCA sizing: buy more at lower prices
                if zone == 'aggressive':
                    # Buy 30-50% of remaining based on position in zone
                    buy_pct = 0.40
                elif zone == 'conservative':
                    # Buy 20% at conservative zone
                    buy_pct = 0.20
                else:
                    buy_pct = 0.0

                buy_amount = min(remaining * buy_pct, remaining)
                buy_amount = max(buy_amount, 10.0)  # Minimum $10
                buy_amount = min(buy_amount, remaining)  # Don't exceed remaining

                if buy_amount >= 10.0:
                    trade = self.execute_dca_buy(symbol, buy_amount)
                    if trade:
                        executed_trades.append(trade)

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'stop_alerts': stop_alerts,
            'analysis': analysis,
            'executed_trades': executed_trades,
            'paper_mode': self.paper_mode
        }

    def get_status(self) -> Dict:
        """
        Return current bot state.

        Returns:
            Status dict with all positions and allocations
        """
        prices = self.last_prices or self.get_prices()

        positions_status = {}
        for symbol in self.TOKENS:
            position = self.positions[symbol]
            price = prices.get(symbol, 0)
            pnl, pnl_pct = position.get_pnl(price)

            positions_status[symbol] = {
                'price': price,
                'quantity': position.quantity,
                'average_cost': position.average_cost,
                'total_cost': position.total_cost,
                'current_value': position.quantity * price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'batches': len(position.batches),
                'allocation': self.ALLOCATIONS[symbol],
                'spent': self.spent[symbol],
                'remaining': self.ALLOCATIONS[symbol] - self.spent[symbol],
                'stop_loss': self.STOP_LOSSES[symbol]
            }

        total_spent = sum(self.spent.values())
        total_value = sum(p.quantity * prices.get(p.symbol, 0) for p in self.positions.values())
        total_pnl = total_value - total_spent

        return {
            'name': 'CryptoDCABot',
            'capital': self.capital,
            'paper_mode': self.paper_mode,
            'coinbase_connected': self.coinbase._initialized,
            'positions': positions_status,
            'total_spent': total_spent,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'trades_today': len(self.trades_today),
            'last_scan': datetime.now(timezone.utc).isoformat()
        }

    def set_existing_position(self, symbol: str, quantity: float,
                               avg_cost: float, total_cost: float = None):
        """
        Set an existing position (e.g., user's current XRP holdings).

        Args:
            symbol: Token symbol
            quantity: Amount held
            avg_cost: Average cost per token
            total_cost: Total USD invested (calculated if not provided)
        """
        if symbol not in self.positions:
            logger.error(f"Unknown symbol: {symbol}")
            return

        if total_cost is None:
            total_cost = quantity * avg_cost

        self.positions[symbol] = DCAPosition(
            symbol=symbol,
            product_id=self.PRODUCT_IDS[symbol],
            quantity=quantity,
            total_cost=total_cost
        )

        logger.info(
            f"Set existing position: {symbol} {quantity:.4f} @ ${avg_cost:.4f} "
            f"(${total_cost:.2f})"
        )


def main():
    """Test the DCA bot"""
    print("=" * 60)
    print("CRYPTO DCA ACCUMULATION BOT")
    print("=" * 60)
    print("\nTargets: XRP (monitor), HBAR ($300), XLM ($200)")
    print("Total Capital: $500")
    print("=" * 60)

    bot = CryptoDCABot(capital=500.0, paper_mode=True)

    # Simulate existing XRP position (user is down $3k)
    # Real position: $3,615 invested at $2.89 avg, down ~$2,062 (-57%)
    xrp_qty = 3615 / 2.89  # ~1,251 XRP
    bot.set_existing_position('XRP', quantity=xrp_qty, avg_cost=2.89)

    print(f"\nInitial Status:")
    status = bot.get_status()
    for symbol in bot.TOKENS:
        pos = status['positions'][symbol]
        print(f"  {symbol}: ${pos['price']:.4f} | Qty: {pos['quantity']:.4f} | "
              f"Allocation: ${pos['allocation']:.0f} | Stop: ${pos['stop_loss']:.4f}")

    print("\n--- Running Scan ---")
    result = bot.run_scan()

    print(f"\nScan Results:")
    print(f"  Stop Alerts: {len(result['stop_alerts'])}")
    print(f"  Trades Executed: {len(result['executed_trades'])}")

    for trade in result['executed_trades']:
        print(f"    {trade['symbol']}: {trade['quantity']:.6f} @ ${trade['price']:.4f} "
              f"(${trade['amount_usd']:.2f})")

    print("\n--- Final Status ---")
    status = bot.get_status()
    print(f"  Total Spent: ${status['total_spent']:.2f}")
    print(f"  Total Value: ${status['total_value']:.2f}")
    print(f"  Total P&L: ${status['total_pnl']:.2f}")

    for symbol in bot.TOKENS:
        pos = status['positions'][symbol]
        if pos['quantity'] > 0:
            print(f"  {symbol}: {pos['quantity']:.4f} @ ${pos['average_cost']:.4f} "
                  f"| P&L: ${pos['pnl']:.2f} ({pos['pnl_pct']:.1%})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
