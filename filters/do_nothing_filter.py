"""
Do Nothing Filter
==================

"More than one investor has lost their entire savings in 'safe' big companies."
"The least active traders: 18.5% return; most active: 11.4%"

This filter implements the research finding that sometimes the best trade is NO trade.
It checks market conditions and returns a "do nothing" signal when:
- Market entropy is high (too random to trade)
- Volatility is extremely high (choppy conditions)
- Session is during circadian mismatch hours
- Recent drawdown suggests regime change

Research Base:
- Voya Corporate Leaders Trust (1935) - "Do Nothing" outperformed most active managers
- Activity != Progress - trading frequency inversely correlated with returns
- Shannon entropy > 0.8 means market is too random

Usage:
    filter = DoNothingFilter()

    # Before running any bot
    if filter.should_do_nothing(df, current_time):
        logger.info("DoNothing filter active - skipping all trades")
        return None

    # Or get detailed analysis
    result = filter.analyze(df, current_time)
    if result['do_nothing']:
        logger.info(f"Skipping: {result['reason']}")

Author: Trading Bot Arsenal
Created: January 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DoNothingFilter')


class DoNothingReason(Enum):
    """Reasons for the do-nothing signal"""
    NONE = "none"  # Trading allowed
    HIGH_ENTROPY = "high_entropy"
    EXTREME_VOLATILITY = "extreme_volatility"
    CIRCADIAN_MISMATCH = "circadian_mismatch"
    RECENT_DRAWDOWN = "recent_drawdown"
    LOW_VOLUME = "low_volume"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


@dataclass
class DoNothingResult:
    """Result of do-nothing analysis"""
    do_nothing: bool
    reason: DoNothingReason
    entropy: float
    volatility_percentile: float
    is_tired_hours: bool
    recent_drawdown: float
    reasoning: str


class DoNothingFilter:
    """
    Do Nothing Filter - Sometimes the Best Trade is No Trade

    Conditions that trigger "do nothing":
    1. Market entropy > threshold (too random)
    2. Volatility in top 10% (choppy/dangerous)
    3. Trading during circadian mismatch hours (2-6 AM local)
    4. Recent drawdown > threshold (potential regime change)
    5. Volume below threshold (illiquid)

    Parameters:
    - entropy_threshold: Shannon entropy threshold (default 0.8)
    - volatility_percentile: Top N% volatility triggers filter (default 0.90)
    - tired_hours: Hours when traders are impaired (default 2-6 AM)
    - drawdown_threshold: Recent drawdown that triggers caution (default 5%)
    - volume_threshold: Min volume ratio to trade (default 0.5)
    """

    def __init__(
        self,
        entropy_threshold: float = 0.8,
        volatility_percentile: float = 0.90,
        tired_hours: Tuple[int, int] = (2, 6),
        drawdown_threshold: float = 0.05,
        volume_threshold: float = 0.5,
        lookback_days: int = 20
    ):
        self.entropy_threshold = entropy_threshold
        self.volatility_percentile = volatility_percentile
        self.tired_hours = tired_hours
        self.drawdown_threshold = drawdown_threshold
        self.volume_threshold = volume_threshold
        self.lookback_days = lookback_days

        logger.info(
            f"DoNothingFilter initialized: "
            f"entropy_thresh={entropy_threshold}, vol_pctl={volatility_percentile}, "
            f"tired_hours={tired_hours}, drawdown_thresh={drawdown_threshold:.1%}"
        )

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns to handle yfinance MultiIndex."""
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
        else:
            df.columns = df.columns.str.lower()
        return df

    def calculate_entropy(self, returns: pd.Series, bins: int = 20) -> float:
        """
        Calculate Shannon entropy of return distribution.

        High entropy = high randomness = no edge = don't trade

        Args:
            returns: Series of returns
            bins: Number of histogram bins

        Returns:
            Normalized entropy (0-1)
        """
        if len(returns) < 10:
            return 0.5  # Unknown

        # Remove NaN values
        returns = returns.dropna()
        if len(returns) < 10:
            return 0.5

        # Create histogram (counts, not density)
        hist, _ = np.histogram(returns, bins=bins)

        # Convert to probabilities
        total = hist.sum()
        if total == 0:
            return 0.5

        probs = hist / total
        probs = probs[probs > 0]  # Remove zeros

        # Calculate Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))

        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log2(bins)
        if max_entropy == 0:
            return 0.5

        normalized_entropy = entropy / max_entropy

        return min(max(normalized_entropy, 0.0), 1.0)

    def calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        """
        Calculate where current volatility ranks in recent history.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Percentile (0-1) of current volatility
        """
        df = self.normalize_columns(df)

        if len(df) < self.lookback_days:
            return 0.5  # Unknown

        # Calculate rolling volatility (standard deviation of returns)
        returns = df['close'].pct_change()
        rolling_vol = returns.rolling(5).std()

        # Get current volatility
        current_vol = rolling_vol.iloc[-1]
        if pd.isna(current_vol):
            return 0.5

        # Calculate percentile rank
        percentile = (rolling_vol < current_vol).sum() / len(rolling_vol.dropna())

        return percentile

    def calculate_recent_drawdown(self, df: pd.DataFrame) -> float:
        """
        Calculate recent drawdown from peak.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Drawdown as positive percentage (e.g., 0.05 = 5% drawdown)
        """
        df = self.normalize_columns(df)

        if len(df) < self.lookback_days:
            return 0.0

        # Use recent window
        recent = df['close'].tail(self.lookback_days)

        # Calculate drawdown
        peak = recent.max()
        current = recent.iloc[-1]
        drawdown = (peak - current) / peak

        return max(drawdown, 0.0)

    def calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """
        Calculate current volume vs average.

        Args:
            df: DataFrame with 'volume' column

        Returns:
            Volume ratio (current / average)
        """
        df = self.normalize_columns(df)

        if 'volume' not in df.columns or len(df) < self.lookback_days:
            return 1.0  # Assume normal

        # Calculate average volume
        avg_volume = df['volume'].tail(self.lookback_days).mean()
        current_volume = df['volume'].iloc[-1]

        if avg_volume == 0:
            return 1.0

        return current_volume / avg_volume

    def is_tired_hours(self, current_time: datetime = None) -> bool:
        """
        Check if current time is during circadian mismatch hours.

        Research: Traders at suboptimal times make worse decisions.
        Peak alertness: 2-3 hours and 9-10 hours after waking.
        Worst: 2-6 AM local time.

        Args:
            current_time: Current time (default: now)

        Returns:
            True if trading during tired hours
        """
        if current_time is None:
            current_time = datetime.now()

        hour = current_time.hour

        start_hour, end_hour = self.tired_hours
        return start_hour <= hour < end_hour

    def is_weekend(self, current_time: datetime = None) -> bool:
        """Check if it's weekend (markets closed)."""
        if current_time is None:
            current_time = datetime.now()

        return current_time.weekday() >= 5  # Saturday = 5, Sunday = 6

    def analyze(self, df: pd.DataFrame = None, current_time: datetime = None) -> DoNothingResult:
        """
        Analyze conditions and determine if we should "do nothing".

        Args:
            df: DataFrame with OHLCV data (optional for time-based checks)
            current_time: Current time (default: now)

        Returns:
            DoNothingResult with analysis
        """
        if current_time is None:
            current_time = datetime.now()

        # Default values
        entropy = 0.5
        volatility_pctl = 0.5
        is_tired = False
        drawdown = 0.0
        volume_ratio = 1.0
        do_nothing = False
        reason = DoNothingReason.NONE
        reasoning = "Trading allowed"

        # Check weekend first
        if self.is_weekend(current_time):
            return DoNothingResult(
                do_nothing=True,
                reason=DoNothingReason.WEEKEND,
                entropy=0.0,
                volatility_percentile=0.0,
                is_tired_hours=False,
                recent_drawdown=0.0,
                reasoning="Weekend - markets closed"
            )

        # Check tired hours
        is_tired = self.is_tired_hours(current_time)
        if is_tired:
            do_nothing = True
            reason = DoNothingReason.CIRCADIAN_MISMATCH
            reasoning = f"Circadian mismatch: {current_time.hour}:00 is during tired hours ({self.tired_hours[0]}-{self.tired_hours[1]})"

        # If we have data, check market conditions
        if df is not None and len(df) > 0:
            df = self.normalize_columns(df)

            # Calculate entropy
            returns = df['close'].pct_change().dropna()
            if len(returns) >= 10:
                entropy = self.calculate_entropy(returns)
                if entropy > self.entropy_threshold and not do_nothing:
                    do_nothing = True
                    reason = DoNothingReason.HIGH_ENTROPY
                    reasoning = f"High entropy ({entropy:.2f} > {self.entropy_threshold}) - market too random"

            # Calculate volatility percentile
            volatility_pctl = self.calculate_volatility_percentile(df)
            if volatility_pctl > self.volatility_percentile and not do_nothing:
                do_nothing = True
                reason = DoNothingReason.EXTREME_VOLATILITY
                reasoning = f"Extreme volatility (top {(1-volatility_pctl)*100:.0f}%) - choppy conditions"

            # Calculate drawdown
            drawdown = self.calculate_recent_drawdown(df)
            if drawdown > self.drawdown_threshold and not do_nothing:
                do_nothing = True
                reason = DoNothingReason.RECENT_DRAWDOWN
                reasoning = f"Recent drawdown ({drawdown:.1%}) suggests regime change - proceed with caution"

            # Calculate volume ratio
            volume_ratio = self.calculate_volume_ratio(df)
            if volume_ratio < self.volume_threshold and not do_nothing:
                do_nothing = True
                reason = DoNothingReason.LOW_VOLUME
                reasoning = f"Low volume ({volume_ratio:.1f}x avg) - illiquid conditions"

        return DoNothingResult(
            do_nothing=do_nothing,
            reason=reason,
            entropy=entropy,
            volatility_percentile=volatility_pctl,
            is_tired_hours=is_tired,
            recent_drawdown=drawdown,
            reasoning=reasoning
        )

    def should_do_nothing(self, df: pd.DataFrame = None, current_time: datetime = None) -> bool:
        """
        Quick check if we should skip trading.

        Args:
            df: DataFrame with OHLCV data (optional)
            current_time: Current time (default: now)

        Returns:
            True if we should not trade
        """
        result = self.analyze(df, current_time)
        if result.do_nothing:
            logger.info(f"DoNothingFilter active: {result.reasoning}")
        return result.do_nothing


# Example usage
if __name__ == "__main__":
    import yfinance as yf

    print("=" * 60)
    print("DO NOTHING FILTER TEST")
    print("=" * 60)

    # Download SPY data
    try:
        spy = yf.download("SPY", period="3mo", interval="1d", progress=False)

        if len(spy) > 0:
            print(f"\nLoaded {len(spy)} days of SPY data")

            # Test the filter
            filter = DoNothingFilter()

            # Get analysis
            result = filter.analyze(spy)

            print(f"\nDoNothing Analysis:")
            print(f"  Do Nothing: {result.do_nothing}")
            print(f"  Reason: {result.reason.value}")
            print(f"  Entropy: {result.entropy:.2f} (threshold: {filter.entropy_threshold})")
            print(f"  Volatility Percentile: {result.volatility_percentile:.0%}")
            print(f"  Tired Hours: {result.is_tired_hours}")
            print(f"  Recent Drawdown: {result.recent_drawdown:.1%}")
            print(f"  Reasoning: {result.reasoning}")

            # Quick check
            print(f"\nQuick check: should_do_nothing() = {filter.should_do_nothing(spy)}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
