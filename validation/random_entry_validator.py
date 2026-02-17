"""
Random Entry Validator - Test If Exits Are The Real Edge
=========================================================

HYPOTHESIS: If random entries + our exit rules produce >40% win rate,
then entries are NOT our edge - exits/sizing are.

This validates whether we should:
- STOP optimizing entries
- FOCUS on exit optimization and position sizing

Methodology:
1. Generate random entry points on historical data
2. Apply our 3-tier scaled exit system
3. Apply ATR-based stops
4. Measure: win rate, profit factor, expectancy
5. Compare to 40% threshold

If Result > 40%: Entries don't matter, focus on exits
If Result < 40%: Entries do matter, continue optimizing

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import json

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RandomEntryValidator')


@dataclass
class RandomTrade:
    """A trade with random entry"""
    entry_idx: int
    entry_price: float
    entry_date: datetime
    stop_loss: float
    atr: float
    direction: str = "long"  # or "short"

    # Exit fields (filled after simulation)
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0
    is_winner: bool = False

    # Tier tracking
    tier1_exit: Optional[float] = None
    tier2_exit: Optional[float] = None
    tier3_exit: Optional[float] = None
    tier1_pnl: float = 0.0
    tier2_pnl: float = 0.0
    tier3_pnl: float = 0.0


class ExitRule(Enum):
    """Available exit rule configurations"""
    SCALED_3TIER = "scaled_3tier"      # Our 3-tier system
    FIXED_RR = "fixed_rr"               # Fixed risk/reward
    TRAILING_ONLY = "trailing_only"     # Only trailing stop
    ATR_BASED = "atr_based"             # ATR-based exits


class RandomEntryValidator:
    """
    Validates if exits are our real edge by testing random entries.

    Usage:
        validator = RandomEntryValidator()
        results = validator.run_validation(
            df=price_data,
            num_trades=1000,
            exit_rule=ExitRule.SCALED_3TIER
        )

        if results['win_rate'] > 0.40:
            print("EXITS ARE THE EDGE - Stop optimizing entries!")
        else:
            print("ENTRIES MATTER - Continue entry optimization")
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        min_bars_between_trades: int = 5,
        max_holding_period: int = 20,
        tier1_r: float = 1.0,
        tier2_r: float = 2.0,
        tier3_trailing_atr: float = 1.5
    ):
        """
        Initialize validator.

        Args:
            atr_period: Period for ATR calculation
            atr_stop_multiplier: Multiplier for stop loss (2x ATR default)
            min_bars_between_trades: Minimum bars between random entries
            max_holding_period: Maximum bars to hold before force exit
            tier1_r: R-multiple for tier 1 exit
            tier2_r: R-multiple for tier 2 exit
            tier3_trailing_atr: ATR multiplier for tier 3 trailing stop
        """
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.min_bars_between_trades = min_bars_between_trades
        self.max_holding_period = max_holding_period
        self.tier1_r = tier1_r
        self.tier2_r = tier2_r
        self.tier3_trailing_atr = tier3_trailing_atr

        self.trades: List[RandomTrade] = []
        self.results: Dict = {}

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()

        return atr

    def generate_random_entries(
        self,
        df: pd.DataFrame,
        num_trades: int,
        seed: Optional[int] = None
    ) -> List[int]:
        """
        Generate random entry indices.

        Args:
            df: Price DataFrame
            num_trades: Number of random entries to generate
            seed: Random seed for reproducibility

        Returns:
            List of entry indices
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Valid range (skip first ATR period, leave room for exits)
        start_idx = self.atr_period + 10
        end_idx = len(df) - self.max_holding_period - 10

        if end_idx <= start_idx:
            raise ValueError("DataFrame too short for validation")

        # Generate random indices with minimum spacing
        valid_indices = list(range(start_idx, end_idx))
        entries = []

        while len(entries) < num_trades and valid_indices:
            idx = random.choice(valid_indices)
            entries.append(idx)

            # Remove nearby indices to ensure spacing
            valid_indices = [
                i for i in valid_indices
                if abs(i - idx) >= self.min_bars_between_trades
            ]

        return sorted(entries)

    def simulate_3tier_exit(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop_loss: float,
        atr: float
    ) -> RandomTrade:
        """
        Simulate 3-tier scaled exit for a trade.

        Args:
            df: Price DataFrame
            entry_idx: Entry bar index
            entry_price: Entry price
            stop_loss: Initial stop loss
            atr: ATR at entry

        Returns:
            RandomTrade with exit details
        """
        close = df['Close'] if 'Close' in df.columns else df['close']
        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        dates = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)

        risk_per_share = entry_price - stop_loss

        # Tier targets
        tier1_target = entry_price + (risk_per_share * self.tier1_r)
        tier2_target = entry_price + (risk_per_share * self.tier2_r)

        # Track each tier
        tier1_exited = False
        tier2_exited = False
        tier3_exited = False

        tier1_exit_price = None
        tier2_exit_price = None
        tier3_exit_price = None

        # Trailing stop for tier 3
        highest_price = entry_price
        trailing_stop = stop_loss

        exit_idx = None
        exit_price = None
        exit_reason = None

        # Simulate bar by bar
        for i in range(entry_idx + 1, min(entry_idx + self.max_holding_period + 1, len(df))):
            bar_high = high.values[i] if hasattr(high, 'values') else high.iloc[i]
            bar_low = low.values[i] if hasattr(low, 'values') else low.iloc[i]
            bar_close = close.values[i] if hasattr(close, 'values') else close.iloc[i]

            # Ensure scalar values
            if hasattr(bar_high, 'item'):
                bar_high = bar_high.item()
            if hasattr(bar_low, 'item'):
                bar_low = bar_low.item()
            if hasattr(bar_close, 'item'):
                bar_close = bar_close.item()

            # Update trailing stop
            if bar_high > highest_price:
                highest_price = bar_high
                new_trailing = highest_price - (atr * self.tier3_trailing_atr)
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing

            # Check tier 1 exit (target hit)
            if not tier1_exited and bar_high >= tier1_target:
                tier1_exit_price = tier1_target
                tier1_exited = True

            # Check tier 2 exit (target hit)
            if not tier2_exited and bar_high >= tier2_target:
                tier2_exit_price = tier2_target
                tier2_exited = True

            # Check stop hit (affects all remaining tiers)
            if bar_low <= stop_loss:
                # All non-exited tiers stop out
                if not tier1_exited:
                    tier1_exit_price = stop_loss
                    tier1_exited = True
                if not tier2_exited:
                    tier2_exit_price = stop_loss
                    tier2_exited = True
                if not tier3_exited:
                    tier3_exit_price = stop_loss
                    tier3_exited = True
                exit_idx = i
                exit_reason = "stop"
                break

            # Check trailing stop for tier 3
            if tier1_exited and tier2_exited and not tier3_exited:
                if bar_low <= trailing_stop:
                    tier3_exit_price = trailing_stop
                    tier3_exited = True
                    exit_idx = i
                    exit_reason = "trailing_stop"
                    break

            # All tiers exited?
            if tier1_exited and tier2_exited and tier3_exited:
                exit_idx = i
                exit_reason = "all_targets"
                break

        # Force exit at max holding if not exited
        if not (tier1_exited and tier2_exited and tier3_exited):
            force_exit_idx = min(entry_idx + self.max_holding_period, len(df) - 1)
            force_exit_price = close.iloc[force_exit_idx]

            if not tier1_exited:
                tier1_exit_price = force_exit_price
            if not tier2_exited:
                tier2_exit_price = force_exit_price
            if not tier3_exited:
                tier3_exit_price = force_exit_price

            exit_idx = force_exit_idx
            exit_reason = "max_holding"

        # Calculate P&L for each tier (33%, 33%, 34% allocation)
        tier1_pnl = (tier1_exit_price - entry_price) / entry_price * 0.33 if tier1_exit_price else 0
        tier2_pnl = (tier2_exit_price - entry_price) / entry_price * 0.33 if tier2_exit_price else 0
        tier3_pnl = (tier3_exit_price - entry_price) / entry_price * 0.34 if tier3_exit_price else 0

        total_pnl_pct = tier1_pnl + tier2_pnl + tier3_pnl

        # Calculate weighted average exit for R-multiple
        weighted_exit = (
            (tier1_exit_price or entry_price) * 0.33 +
            (tier2_exit_price or entry_price) * 0.33 +
            (tier3_exit_price or entry_price) * 0.34
        )
        r_multiple = (weighted_exit - entry_price) / risk_per_share if risk_per_share > 0 else 0

        trade = RandomTrade(
            entry_idx=entry_idx,
            entry_price=entry_price,
            entry_date=dates[entry_idx] if hasattr(dates, '__getitem__') else datetime.now(),
            stop_loss=stop_loss,
            atr=atr,
            exit_idx=exit_idx,
            exit_price=weighted_exit,
            exit_date=dates[exit_idx] if exit_idx and hasattr(dates, '__getitem__') else None,
            exit_reason=exit_reason,
            pnl=total_pnl_pct * 10000,  # Assume $10k position
            pnl_pct=total_pnl_pct,
            r_multiple=r_multiple,
            is_winner=total_pnl_pct > 0,
            tier1_exit=tier1_exit_price,
            tier2_exit=tier2_exit_price,
            tier3_exit=tier3_exit_price,
            tier1_pnl=tier1_pnl * 10000,
            tier2_pnl=tier2_pnl * 10000,
            tier3_pnl=tier3_pnl * 10000
        )

        return trade

    def run_validation(
        self,
        df: pd.DataFrame,
        num_trades: int = 1000,
        exit_rule: ExitRule = ExitRule.SCALED_3TIER,
        seed: Optional[int] = 42,
        verbose: bool = True
    ) -> Dict:
        """
        Run the random entry validation.

        Args:
            df: Price DataFrame with OHLC data
            num_trades: Number of random trades to simulate
            exit_rule: Exit rule to test
            seed: Random seed for reproducibility
            verbose: Print progress updates

        Returns:
            Dictionary with validation results
        """
        if verbose:
            print("=" * 60)
            print("RANDOM ENTRY VALIDATOR")
            print("Testing if exits are our real edge")
            print("=" * 60)
            print(f"\nConfiguration:")
            print(f"  Trades to simulate: {num_trades}")
            print(f"  Exit rule: {exit_rule.value}")
            print(f"  ATR period: {self.atr_period}")
            print(f"  Stop multiplier: {self.atr_stop_multiplier}x ATR")
            print(f"  Max holding: {self.max_holding_period} bars")
            print(f"  Random seed: {seed}")

        # Calculate ATR
        atr = self.calculate_atr(df)
        close = df['Close'] if 'Close' in df.columns else df['close']

        # Generate random entries
        entry_indices = self.generate_random_entries(df, num_trades, seed)
        actual_trades = len(entry_indices)

        if verbose:
            print(f"\n  Generated {actual_trades} random entries")

        # Simulate each trade
        self.trades = []

        for i, entry_idx in enumerate(entry_indices):
            entry_price = close.iloc[entry_idx]
            entry_atr = atr.iloc[entry_idx]

            if pd.isna(entry_atr) or entry_atr <= 0:
                continue

            stop_loss = entry_price - (entry_atr * self.atr_stop_multiplier)

            if exit_rule == ExitRule.SCALED_3TIER:
                trade = self.simulate_3tier_exit(
                    df, entry_idx, entry_price, stop_loss, entry_atr
                )
            else:
                # Add other exit rules here
                trade = self.simulate_3tier_exit(
                    df, entry_idx, entry_price, stop_loss, entry_atr
                )

            self.trades.append(trade)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{actual_trades} trades...")

        # Calculate results
        self.results = self._calculate_results()

        if verbose:
            self._print_results()

        return self.results

    def _calculate_results(self) -> Dict:
        """Calculate validation results."""
        if not self.trades:
            return {'error': 'No trades simulated'}

        wins = [t for t in self.trades if t.is_winner]
        losses = [t for t in self.trades if not t.is_winner]

        total_pnl = sum(t.pnl for t in self.trades)
        avg_winner = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loser = np.mean([t.pnl for t in losses]) if losses else 0

        win_rate = len(wins) / len(self.trades) if self.trades else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy per trade
        expectancy = total_pnl / len(self.trades) if self.trades else 0

        # R-multiple stats
        avg_r = np.mean([t.r_multiple for t in self.trades])
        avg_winner_r = np.mean([t.r_multiple for t in wins]) if wins else 0
        avg_loser_r = np.mean([t.r_multiple for t in losses]) if losses else 0

        # Exit reason breakdown
        exit_reasons = {}
        for t in self.trades:
            reason = t.exit_reason or 'unknown'
            if reason not in exit_reasons:
                exit_reasons[reason] = 0
            exit_reasons[reason] += 1

        # Tier performance
        tier1_total = sum(t.tier1_pnl for t in self.trades)
        tier2_total = sum(t.tier2_pnl for t in self.trades)
        tier3_total = sum(t.tier3_pnl for t in self.trades)

        # The key question: Is win rate > 40%?
        exits_are_edge = win_rate > 0.40

        return {
            'total_trades': len(self.trades),
            'winners': len(wins),
            'losers': len(losses),
            'win_rate': win_rate,
            'win_rate_pct': f"{win_rate:.1%}",

            'total_pnl': total_pnl,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'profit_factor': profit_factor,
            'expectancy': expectancy,

            'avg_r_multiple': avg_r,
            'avg_winner_r': avg_winner_r,
            'avg_loser_r': avg_loser_r,

            'tier1_pnl': tier1_total,
            'tier2_pnl': tier2_total,
            'tier3_pnl': tier3_total,

            'exit_reasons': exit_reasons,

            'exits_are_edge': exits_are_edge,
            'recommendation': (
                "STOP optimizing entries - FOCUS on exits/sizing"
                if exits_are_edge else
                "CONTINUE entry optimization - entries still matter"
            ),

            'threshold': 0.40,
            'above_threshold': exits_are_edge
        }

    def _print_results(self):
        """Print formatted results."""
        r = self.results

        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)

        print(f"\n📊 TRADE STATISTICS:")
        print(f"  Total trades: {r['total_trades']}")
        print(f"  Winners: {r['winners']}")
        print(f"  Losers: {r['losers']}")
        print(f"  Win Rate: {r['win_rate_pct']}")

        print(f"\n💰 P&L METRICS:")
        print(f"  Total P&L: ${r['total_pnl']:,.2f}")
        print(f"  Avg Winner: ${r['avg_winner']:,.2f}")
        print(f"  Avg Loser: ${r['avg_loser']:,.2f}")
        print(f"  Profit Factor: {r['profit_factor']:.2f}")
        print(f"  Expectancy: ${r['expectancy']:,.2f}/trade")

        print(f"\n📈 R-MULTIPLE ANALYSIS:")
        print(f"  Avg R: {r['avg_r_multiple']:.2f}R")
        print(f"  Avg Winner R: {r['avg_winner_r']:.2f}R")
        print(f"  Avg Loser R: {r['avg_loser_r']:.2f}R")

        print(f"\n🎯 3-TIER BREAKDOWN:")
        print(f"  Tier 1 (1R target): ${r['tier1_pnl']:,.2f}")
        print(f"  Tier 2 (2R target): ${r['tier2_pnl']:,.2f}")
        print(f"  Tier 3 (trailing): ${r['tier3_pnl']:,.2f}")

        print(f"\n📤 EXIT REASONS:")
        for reason, count in r['exit_reasons'].items():
            pct = count / r['total_trades'] * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")

        print("\n" + "=" * 60)
        print("🔑 KEY FINDING")
        print("=" * 60)

        if r['exits_are_edge']:
            print(f"\n  ✅ WIN RATE ({r['win_rate_pct']}) > 40% THRESHOLD")
            print("\n  🎯 EXITS ARE OUR REAL EDGE!")
            print("\n  RECOMMENDATION:")
            print("  → STOP optimizing entries")
            print("  → FOCUS on exit rules and position sizing")
            print("  → Random entries + good exits = profitable")
        else:
            print(f"\n  ❌ WIN RATE ({r['win_rate_pct']}) < 40% THRESHOLD")
            print("\n  📊 ENTRIES STILL MATTER")
            print("\n  RECOMMENDATION:")
            print("  → Continue entry optimization")
            print("  → Entry quality affects profitability")

        print("\n" + "=" * 60)

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        # Convert trades to serializable format
        trades_data = []
        for t in self.trades:
            trades_data.append({
                'entry_idx': t.entry_idx,
                'entry_price': t.entry_price,
                'stop_loss': t.stop_loss,
                'exit_price': t.exit_price,
                'exit_reason': t.exit_reason,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'r_multiple': t.r_multiple,
                'is_winner': t.is_winner
            })

        output = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results,
            'trades': trades_data
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {filepath}")


def fetch_test_data(symbol: str = "SPY", period: str = "2y") -> pd.DataFrame:
    """Fetch test data using yfinance."""
    try:
        import yfinance as yf
        df = yf.download(symbol, period=period, progress=False)

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        print(f"Fetched {len(df)} bars of {symbol} data")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" RANDOM ENTRY VALIDATOR - Testing If Exits Are Our Edge")
    print("=" * 70)

    # Fetch test data
    print("\n📥 Fetching historical data...")
    df = fetch_test_data("SPY", "2y")

    if df is None or len(df) < 100:
        print("ERROR: Could not fetch sufficient data")
        sys.exit(1)

    # Create validator
    validator = RandomEntryValidator(
        atr_period=14,
        atr_stop_multiplier=2.0,
        min_bars_between_trades=5,
        max_holding_period=20,
        tier1_r=1.0,
        tier2_r=2.0,
        tier3_trailing_atr=1.5
    )

    # Run validation with different trade counts
    for num_trades in [100, 500, 1000]:
        print(f"\n\n{'='*70}")
        print(f" VALIDATION RUN: {num_trades} Random Trades")
        print('='*70)

        results = validator.run_validation(
            df=df,
            num_trades=num_trades,
            exit_rule=ExitRule.SCALED_3TIER,
            seed=42
        )

    # Save final results
    os.makedirs('validation', exist_ok=True)
    validator.save_results('validation/random_entry_results.json')

    # Summary
    print("\n" + "=" * 70)
    print(" FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Win Rate with Random Entries: {results['win_rate_pct']}")
    print(f"  Threshold: 40%")
    print(f"\n  CONCLUSION: {results['recommendation']}")
    print("\n" + "=" * 70)
