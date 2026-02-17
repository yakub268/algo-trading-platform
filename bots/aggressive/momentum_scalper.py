"""
MOMENTUM SCALPER BOT
====================

Aggressive momentum-based scalping strategy for USDC pairs.

Strategy:
- Scans all USDC pairs for coins up 5%+ in last 1 hour
- Buys top 3 momentum coins
- Sets 3% trailing stop-loss
- Takes profit at 8% gain or exits after 4 hours
- Position size: $50-100 per trade from USDC balance

Risk Management:
- 3% trailing stop-loss
- 8% take profit
- 4 hour max hold time
- Max 3 concurrent positions
- $50-100 per position

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from bots.aggressive.base_aggressive_bot import BaseAggressiveBot, TradeSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MomentumScalper')


class MomentumScalper(BaseAggressiveBot):
    """
    Momentum Scalper - Quick scalps on trending coins.

    Looks for coins with strong short-term momentum (5%+ in 1 hour)
    and rides the trend with tight trailing stops.
    """

    # Strategy parameters
    MIN_MOMENTUM_PCT = 0.025  # 2.5% minimum 1h gain to qualify
    MAX_HOLD_SECONDS = 4 * 3600  # 4 hours max hold

    # USDC pairs to scan (common liquid pairs)
    USDC_PAIRS = [
        'BTC-USDC', 'ETH-USDC', 'SOL-USDC', 'DOGE-USDC',
        'XRP-USDC', 'AVAX-USDC', 'LINK-USDC', 'DOT-USDC',
        'SHIB-USDC', 'MATIC-USDC', 'UNI-USDC', 'ATOM-USDC',
        'LTC-USDC', 'ADA-USDC', 'NEAR-USDC', 'APT-USDC',
        'ARB-USDC', 'OP-USDC', 'SUI-USDC', 'HBAR-USDC',
        'PEPE-USDC', 'WIF-USDC', 'BONK-USDC', 'FLOKI-USDC'
    ]

    def __init__(
        self,
        capital: float = 300.0,
        paper_mode: bool = None,
        position_size: float = 75.0,  # $50-100 range, default $75
        take_profit_pct: float = 0.08,  # 8% TP
        stop_loss_pct: float = 0.03,    # 3% SL (trailing)
        max_positions: int = 3,
    ):
        """
        Initialize Momentum Scalper.

        Args:
            capital: Total capital for this strategy ($300 default for 3x$100 positions)
            paper_mode: Paper trading mode
            position_size: Position size per trade ($50-100)
            take_profit_pct: Take profit percentage (8% default)
            stop_loss_pct: Stop loss percentage (3% trailing)
            max_positions: Maximum concurrent positions (3 default)
        """
        super().__init__(
            capital=capital,
            paper_mode=paper_mode,
            position_size=position_size,
            take_profit_pct=take_profit_pct,
            stop_loss_pct=stop_loss_pct,
            max_positions=max_positions
        )

        # Price history for 1h momentum calculation
        self.price_history: Dict[str, List[Dict]] = {}
        self.history_max_age_hours = 2

        # Track highest price for trailing stops
        self.trailing_highs: Dict[str, float] = {}

        self.logger.info(
            f"MomentumScalper ready - "
            f"Min momentum: {self.MIN_MOMENTUM_PCT:.0%}, "
            f"Position: ${position_size}, "
            f"TP: {take_profit_pct:.0%}, Trailing SL: {stop_loss_pct:.0%}"
        )

    def get_usdc_pairs(self) -> List[str]:
        """Get list of USDC pairs to scan"""
        return self.USDC_PAIRS

    def get_prices(self, pairs: List[str]) -> Dict[str, Dict]:
        """
        Get current prices for multiple pairs via Alpaca.

        Returns dict of {product_id: {bid, ask, mid, spread}}
        """
        prices = {}

        if not self.alpaca._initialized:
            return prices

        try:
            results = self.alpaca.get_best_bid_ask(pairs)
            for item in results:
                pid = item.get('product_id', '')
                bid = item.get('bid', 0)
                ask = item.get('ask', 0)
                mid = item.get('mid', (bid + ask) / 2 if bid and ask else 0)
                spread = (ask - bid) / mid if mid > 0 else 0
                prices[pid] = {
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'spread': spread
                }
        except Exception as e:
            self.logger.error(f"Error fetching prices: {e}")

        return prices

    def _store_price_history(self, prices: Dict[str, Dict]):
        """Store current prices for 1h momentum calculation"""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self.history_max_age_hours)

        for product_id, price_data in prices.items():
            if product_id not in self.price_history:
                self.price_history[product_id] = []

            self.price_history[product_id].append({
                'timestamp': now,
                'price': price_data['mid']
            })

            # Prune old entries
            self.price_history[product_id] = [
                p for p in self.price_history[product_id]
                if p['timestamp'] > cutoff
            ]

    def _get_price_1h_ago(self, product_id: str) -> Optional[float]:
        """Get price from ~1 hour ago for momentum calculation"""
        if product_id not in self.price_history:
            return None

        history = self.price_history[product_id]
        if not history:
            return None

        target_time = datetime.now(timezone.utc) - timedelta(hours=1)

        # Find closest price to 1 hour ago
        closest = None
        closest_diff = timedelta(hours=24)

        for entry in history:
            diff = abs(entry['timestamp'] - target_time)
            if diff < closest_diff:
                closest_diff = diff
                closest = entry

        # Only use if within 15 minutes of target
        if closest and closest_diff < timedelta(minutes=15):
            return closest['price']

        return None

    def _get_1h_change_from_candles(self, product_id: str) -> Optional[float]:
        """
        Get 1h price change using candle data.
        More reliable than price history on first scan.
        """
        candles = self.get_candles(product_id, granularity="ONE_HOUR", limit=2)
        if len(candles) >= 2:
            prev_close = candles[-2]['close']
            curr_close = candles[-1]['close']
            if prev_close > 0:
                return (curr_close - prev_close) / prev_close
        return None

    def _calculate_momentum_score(
        self,
        change_1h_pct: float,
        spread_pct: float
    ) -> float:
        """
        Calculate composite momentum score.
        Higher score = better candidate.
        """
        base_score = change_1h_pct * 100
        spread_penalty = spread_pct * 50
        return base_score - spread_penalty

    def run_scan(self) -> List[TradeSignal]:
        """
        Main scan method - finds momentum opportunities and generates signals.

        1. Update trailing stops and check exits
        2. Scan for new momentum candidates
        3. Generate signals for top candidates

        Returns:
            List of TradeSignal for the orchestrator to execute
        """
        self.logger.info("=" * 50)
        self.logger.info("MomentumScalper - Running scan...")

        signals = []

        # 1. Check existing positions for exits (trailing stop / time limit)
        exits = self._check_trailing_stops()
        for exit_info in exits:
            self.logger.info(f"EXIT: {exit_info}")

        # 2. Check if we can open new positions
        open_count = len(self.active_positions)
        slots_available = self.max_positions - open_count

        self.logger.info(f"Open positions: {open_count}/{self.max_positions}")

        if slots_available <= 0:
            self.logger.info("Max positions reached - skipping new entries")
            return signals

        # 3. Get prices for all USDC pairs
        pairs = self.get_usdc_pairs()
        prices = self.get_prices(pairs)
        self._store_price_history(prices)

        # 4. Find momentum candidates
        candidates = []

        for product_id, price_data in prices.items():
            # Skip if already have position
            if product_id in self.active_positions:
                continue

            # Skip high spread pairs
            if price_data['spread'] > 0.005:  # 0.5% max spread
                continue

            current_price = price_data['mid']

            # Get 1h change - prefer candles, fall back to price history
            change_1h = self._get_1h_change_from_candles(product_id)
            if change_1h is None:
                price_1h_ago = self._get_price_1h_ago(product_id)
                if price_1h_ago and price_1h_ago > 0:
                    change_1h = (current_price - price_1h_ago) / price_1h_ago
                else:
                    continue  # No historical data yet

            # Check momentum threshold
            if change_1h < self.MIN_MOMENTUM_PCT:
                continue

            # Calculate score
            score = self._calculate_momentum_score(change_1h, price_data['spread'])

            candidates.append({
                'product_id': product_id,
                'symbol': product_id.replace('-USDC', ''),
                'price': current_price,
                'change_1h': change_1h,
                'spread': price_data['spread'],
                'score': score
            })

            self.logger.info(
                f"MOMENTUM: {product_id} +{change_1h:.1%} | "
                f"Price: ${current_price:.6f} | Score: {score:.1f}"
            )

        # 5. Sort by score and generate signals for top candidates
        candidates.sort(key=lambda x: x['score'], reverse=True)

        for candidate in candidates[:slots_available]:
            entry_price = candidate['price']
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)

            # Calculate position size (respect $50-100 range)
            position_usd = min(
                max(50.0, self.position_size),
                100.0,
                self.capital / self.max_positions
            )

            signal = TradeSignal(
                symbol=candidate['product_id'],
                side='BUY',
                entry_price=entry_price,
                target_price=take_profit,
                stop_loss=stop_loss,
                position_size_usd=position_usd,
                confidence=min(0.9, 0.5 + candidate['score'] / 20),
                reason=f"1h momentum +{candidate['change_1h']:.1%}, score {candidate['score']:.1f}",
                timestamp=datetime.now(timezone.utc),
                metadata={
                    'change_1h': candidate['change_1h'],
                    'score': candidate['score'],
                    'spread': candidate['spread'],
                    'trailing_stop_pct': self.stop_loss_pct,
                    'max_hold_seconds': self.MAX_HOLD_SECONDS
                }
            )

            signals.append(signal)
            self.logger.info(
                f"SIGNAL: BUY {candidate['symbol']} @ ${entry_price:.6f}, "
                f"TP: ${take_profit:.6f} (+{self.take_profit_pct:.0%}), "
                f"SL: ${stop_loss:.6f} (-{self.stop_loss_pct:.0%})"
            )

            # Initialize trailing high
            self.trailing_highs[candidate['product_id']] = entry_price

        self.logger.info(f"Scan complete - {len(signals)} signals generated")
        return signals

    def _check_trailing_stops(self) -> List[dict]:
        """
        Check and update trailing stops for active positions.
        Also check time-based exits.
        """
        exits = []

        if not self.active_positions:
            return exits

        # Get current prices
        pairs = list(self.active_positions.keys())
        prices = self.get_prices(pairs)
        now = datetime.now(timezone.utc)

        for symbol, position in list(self.active_positions.items()):
            price_data = prices.get(symbol)
            if not price_data:
                continue

            current_price = price_data['mid']
            entry_price = position['entry_price']

            # Update trailing high
            if symbol not in self.trailing_highs:
                self.trailing_highs[symbol] = entry_price

            if current_price > self.trailing_highs[symbol]:
                self.trailing_highs[symbol] = current_price
                # Update stop loss based on new high
                new_stop = current_price * (1 - self.stop_loss_pct)
                if new_stop > position['stop_loss']:
                    position['stop_loss'] = new_stop
                    self.logger.info(
                        f"TRAILING STOP updated: {symbol} "
                        f"new high ${current_price:.6f}, stop ${new_stop:.6f}"
                    )

            exit_reason = None

            # Check take profit
            if current_price >= position['target_price']:
                exit_reason = 'take_profit'

            # Check trailing stop
            elif current_price <= position['stop_loss']:
                exit_reason = 'stop_loss'

            # Check max hold time
            if 'entry_time' in position:
                hold_time = (now - position['entry_time']).total_seconds()
                if hold_time >= self.MAX_HOLD_SECONDS:
                    exit_reason = 'max_hold_time'

            if exit_reason:
                pnl_pct = (current_price - entry_price) / entry_price
                pnl_usd = position['size_usd'] * pnl_pct

                exits.append({
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'reason': exit_reason
                })

                self.logger.info(
                    f"[EXIT] {symbol} ({exit_reason}): "
                    f"${entry_price:.6f} -> ${current_price:.6f}, "
                    f"PnL: {pnl_pct:+.2%} (${pnl_usd:+.2f})"
                )

                del self.active_positions[symbol]
                if symbol in self.trailing_highs:
                    del self.trailing_highs[symbol]

        return exits

    def get_status(self) -> dict:
        """Get detailed bot status"""
        base_status = super().get_status()

        # Add momentum-specific info
        base_status['strategy'] = 'MomentumScalper'
        base_status['min_momentum_pct'] = self.MIN_MOMENTUM_PCT
        base_status['max_hold_hours'] = self.MAX_HOLD_SECONDS / 3600
        base_status['trailing_stops'] = self.stop_loss_pct
        base_status['price_history_pairs'] = len(self.price_history)
        base_status['trailing_highs'] = {
            k: v for k, v in self.trailing_highs.items()
        }

        return base_status


def main():
    """Test the Momentum Scalper"""
    logger.info("=" * 60)
    logger.info("MOMENTUM SCALPER - Aggressive Crypto Trading")
    logger.info("=" * 60)

    bot = MomentumScalper(
        capital=300.0,
        paper_mode=True,
        position_size=75.0,
        take_profit_pct=0.08,
        stop_loss_pct=0.03,
        max_positions=3
    )

    logger.info(f"\nBot Status:")
    status = bot.get_status()
    for key, value in status.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n--- Running Scans ---")

    # Run a few scans
    for i in range(3):
        logger.info(f"\n[Scan {i+1}/3]")
        signals = bot.run_scan()

        if signals:
            logger.info(f"\nSignals generated: {len(signals)}")
            for signal in signals:
                logger.info(f"  {signal.side} {signal.symbol} @ ${signal.entry_price:.6f}")
                logger.info(f"    Reason: {signal.reason}")
                logger.info(f"    TP: ${signal.target_price:.6f}, SL: ${signal.stop_loss:.6f}")

                # Execute signal for testing
                bot.execute_signal(signal)

        if i < 2:
            logger.info("\nWaiting 2 seconds...")
            time.sleep(2)

    logger.info("\n--- Final Status ---")
    status = bot.get_status()
    logger.info(f"  Active Positions: {status['active_positions']}")
    logger.info(f"  Total Trades: {status['total_trades']}")
    logger.info(f"  Total Signals: {status['total_signals']}")

    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()
