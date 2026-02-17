"""
RSI Hidden Divergence Detector
==============================

"Each hidden divergence strategy has shown excellent performance over the last decade,
with about 70% profitable trades."

Hidden Divergence signals CONTINUATION, not reversal:
- Bullish Hidden: Price makes higher low + RSI makes lower low = BUY (trend continues up)
- Bearish Hidden: Price makes lower high + RSI makes higher high = SELL (trend continues down)

Most traders focus on regular divergence (reversal signal), which is why hidden divergence
provides an edge - it's overlooked.

Regular Divergence (for reference - signals reversal):
- Bullish Regular: Price makes lower low + RSI makes higher low = potential reversal up
- Bearish Regular: Price makes higher high + RSI makes lower high = potential reversal down

Usage:
    detector = RSIDivergenceDetector()

    # Get divergence signals
    result = detector.detect_divergence(df)
    if result.hidden_bullish:
        # Buy signal - trend continuation
        place_long_order()

Author: Trading Bot Arsenal
Created: January 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RSIDivergence')


class DivergenceType(Enum):
    """Types of RSI divergence"""
    NONE = "none"
    HIDDEN_BULLISH = "hidden_bullish"
    HIDDEN_BEARISH = "hidden_bearish"
    REGULAR_BULLISH = "regular_bullish"
    REGULAR_BEARISH = "regular_bearish"


@dataclass
class DivergenceResult:
    """Result of divergence detection"""
    divergence_type: DivergenceType
    hidden_bullish: bool
    hidden_bearish: bool
    regular_bullish: bool
    regular_bearish: bool
    rsi_current: float
    rsi_swing: float
    price_current: float
    price_swing: float
    strength: float  # 0-1
    signal: str  # 'buy', 'sell', 'none'
    reasoning: str


class RSIDivergenceDetector:
    """
    RSI Hidden Divergence Detector

    Detects hidden divergence patterns that signal trend continuation.
    Hidden divergence is overlooked by most traders, providing an edge.

    Parameters:
    - rsi_period: RSI calculation period (default 14)
    - swing_lookback: Bars to look back for swing points (default 10)
    - min_rsi_diff: Minimum RSI difference for valid divergence (default 5)
    - min_price_diff: Minimum price % difference for valid divergence (default 0.5%)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        swing_lookback: int = 10,
        min_rsi_diff: float = 5.0,
        min_price_diff: float = 0.005
    ):
        self.rsi_period = rsi_period
        self.swing_lookback = swing_lookback
        self.min_rsi_diff = min_rsi_diff
        self.min_price_diff = min_price_diff

        logger.info(
            f"RSIDivergenceDetector initialized: "
            f"rsi_period={rsi_period}, swing_lookback={swing_lookback}, "
            f"min_rsi_diff={min_rsi_diff}, min_price_diff={min_price_diff:.2%}"
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

    def calculate_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def find_swing_lows(self, series: pd.Series, lookback: int = None) -> List[Tuple[int, float]]:
        """
        Find swing low points in a series.

        Returns list of (index, value) tuples.
        """
        lookback = lookback or self.swing_lookback
        swings = []

        for i in range(lookback, len(series) - 1):
            # Check if this is a local minimum
            window = series.iloc[i-lookback:i+lookback+1]
            if len(window) > 0 and series.iloc[i] == window.min():
                swings.append((i, series.iloc[i]))

        return swings

    def find_swing_highs(self, series: pd.Series, lookback: int = None) -> List[Tuple[int, float]]:
        """
        Find swing high points in a series.

        Returns list of (index, value) tuples.
        """
        lookback = lookback or self.swing_lookback
        swings = []

        for i in range(lookback, len(series) - 1):
            # Check if this is a local maximum
            window = series.iloc[i-lookback:i+lookback+1]
            if len(window) > 0 and series.iloc[i] == window.max():
                swings.append((i, series.iloc[i]))

        return swings

    def detect_divergence(self, df: pd.DataFrame) -> DivergenceResult:
        """
        Detect RSI divergence patterns.

        Hidden Divergence (continuation signals - 70% win rate):
        - Hidden Bullish: Price higher low + RSI lower low = BUY
        - Hidden Bearish: Price lower high + RSI higher high = SELL

        Regular Divergence (reversal signals):
        - Regular Bullish: Price lower low + RSI higher low = potential reversal up
        - Regular Bearish: Price higher high + RSI lower high = potential reversal down

        Args:
            df: DataFrame with 'close' column (or will be normalized)

        Returns:
            DivergenceResult with detected divergence patterns
        """
        df = self.normalize_columns(df)

        if len(df) < self.rsi_period + self.swing_lookback * 2:
            return DivergenceResult(
                divergence_type=DivergenceType.NONE,
                hidden_bullish=False,
                hidden_bearish=False,
                regular_bullish=False,
                regular_bearish=False,
                rsi_current=50.0,
                rsi_swing=50.0,
                price_current=0.0,
                price_swing=0.0,
                strength=0.0,
                signal='none',
                reasoning="Insufficient data for divergence detection"
            )

        # Calculate RSI
        close = df['close']
        rsi = self.calculate_rsi(close)

        # Get current values
        current_price = close.iloc[-1]
        current_rsi = rsi.iloc[-1]

        # Find swing points
        price_swing_lows = self.find_swing_lows(close)
        price_swing_highs = self.find_swing_highs(close)
        rsi_swing_lows = self.find_swing_lows(rsi)
        rsi_swing_highs = self.find_swing_highs(rsi)

        # Initialize result
        hidden_bullish = False
        hidden_bearish = False
        regular_bullish = False
        regular_bearish = False
        divergence_type = DivergenceType.NONE
        strength = 0.0
        reasoning = "No divergence detected"
        signal = 'none'
        swing_rsi = current_rsi
        swing_price = current_price

        # Check for Hidden Bullish Divergence
        # Price: Higher low, RSI: Lower low
        if len(price_swing_lows) >= 2 and len(rsi_swing_lows) >= 2:
            prev_price_low = price_swing_lows[-2][1]
            curr_price_low = price_swing_lows[-1][1]
            prev_rsi_low = rsi_swing_lows[-2][1]
            curr_rsi_low = rsi_swing_lows[-1][1]

            # Price higher low AND RSI lower low
            price_higher = (curr_price_low - prev_price_low) / prev_price_low > self.min_price_diff
            rsi_lower = prev_rsi_low - curr_rsi_low > self.min_rsi_diff

            if price_higher and rsi_lower:
                hidden_bullish = True
                divergence_type = DivergenceType.HIDDEN_BULLISH
                signal = 'buy'
                swing_rsi = curr_rsi_low
                swing_price = curr_price_low
                strength = min((prev_rsi_low - curr_rsi_low) / 20, 1.0)
                reasoning = (
                    f"Hidden Bullish: Price higher low (${curr_price_low:.4f} > ${prev_price_low:.4f}), "
                    f"RSI lower low ({curr_rsi_low:.1f} < {prev_rsi_low:.1f}) - trend continuation UP"
                )

        # Check for Hidden Bearish Divergence
        # Price: Lower high, RSI: Higher high
        if not hidden_bullish and len(price_swing_highs) >= 2 and len(rsi_swing_highs) >= 2:
            prev_price_high = price_swing_highs[-2][1]
            curr_price_high = price_swing_highs[-1][1]
            prev_rsi_high = rsi_swing_highs[-2][1]
            curr_rsi_high = rsi_swing_highs[-1][1]

            # Price lower high AND RSI higher high
            price_lower = (prev_price_high - curr_price_high) / prev_price_high > self.min_price_diff
            rsi_higher = curr_rsi_high - prev_rsi_high > self.min_rsi_diff

            if price_lower and rsi_higher:
                hidden_bearish = True
                divergence_type = DivergenceType.HIDDEN_BEARISH
                signal = 'sell'
                swing_rsi = curr_rsi_high
                swing_price = curr_price_high
                strength = min((curr_rsi_high - prev_rsi_high) / 20, 1.0)
                reasoning = (
                    f"Hidden Bearish: Price lower high (${curr_price_high:.4f} < ${prev_price_high:.4f}), "
                    f"RSI higher high ({curr_rsi_high:.1f} > {prev_rsi_high:.1f}) - trend continuation DOWN"
                )

        # Check for Regular Bullish Divergence (reversal)
        # Price: Lower low, RSI: Higher low
        if not hidden_bullish and not hidden_bearish and len(price_swing_lows) >= 2 and len(rsi_swing_lows) >= 2:
            prev_price_low = price_swing_lows[-2][1]
            curr_price_low = price_swing_lows[-1][1]
            prev_rsi_low = rsi_swing_lows[-2][1]
            curr_rsi_low = rsi_swing_lows[-1][1]

            # Price lower low AND RSI higher low
            price_lower = (prev_price_low - curr_price_low) / prev_price_low > self.min_price_diff
            rsi_higher = curr_rsi_low - prev_rsi_low > self.min_rsi_diff

            if price_lower and rsi_higher:
                regular_bullish = True
                divergence_type = DivergenceType.REGULAR_BULLISH
                signal = 'buy'  # Potential reversal
                swing_rsi = curr_rsi_low
                swing_price = curr_price_low
                strength = min((curr_rsi_low - prev_rsi_low) / 20, 1.0)
                reasoning = (
                    f"Regular Bullish: Price lower low, RSI higher low - potential reversal UP"
                )

        # Check for Regular Bearish Divergence (reversal)
        # Price: Higher high, RSI: Lower high
        if not hidden_bullish and not hidden_bearish and not regular_bullish:
            if len(price_swing_highs) >= 2 and len(rsi_swing_highs) >= 2:
                prev_price_high = price_swing_highs[-2][1]
                curr_price_high = price_swing_highs[-1][1]
                prev_rsi_high = rsi_swing_highs[-2][1]
                curr_rsi_high = rsi_swing_highs[-1][1]

                # Price higher high AND RSI lower high
                price_higher = (curr_price_high - prev_price_high) / prev_price_high > self.min_price_diff
                rsi_lower = prev_rsi_high - curr_rsi_high > self.min_rsi_diff

                if price_higher and rsi_lower:
                    regular_bearish = True
                    divergence_type = DivergenceType.REGULAR_BEARISH
                    signal = 'sell'  # Potential reversal
                    swing_rsi = curr_rsi_high
                    swing_price = curr_price_high
                    strength = min((prev_rsi_high - curr_rsi_high) / 20, 1.0)
                    reasoning = (
                        f"Regular Bearish: Price higher high, RSI lower high - potential reversal DOWN"
                    )

        return DivergenceResult(
            divergence_type=divergence_type,
            hidden_bullish=hidden_bullish,
            hidden_bearish=hidden_bearish,
            regular_bullish=regular_bullish,
            regular_bearish=regular_bearish,
            rsi_current=current_rsi,
            rsi_swing=swing_rsi,
            price_current=current_price,
            price_swing=swing_price,
            strength=strength,
            signal=signal,
            reasoning=reasoning
        )

    def get_signal(self, df: pd.DataFrame) -> Dict:
        """
        Get trading signal based on RSI divergence.

        Prioritizes hidden divergence (70% win rate) over regular divergence.

        Returns dict with:
        - signal: 'buy', 'sell', or 'none'
        - type: divergence type
        - strength: 0-1
        - reasoning: explanation
        """
        result = self.detect_divergence(df)

        return {
            'signal': result.signal,
            'type': result.divergence_type.value,
            'hidden_bullish': result.hidden_bullish,
            'hidden_bearish': result.hidden_bearish,
            'regular_bullish': result.regular_bullish,
            'regular_bearish': result.regular_bearish,
            'strength': result.strength,
            'rsi': result.rsi_current,
            'reasoning': result.reasoning
        }

    def add_divergence_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add divergence detection columns to DataFrame.

        Adds columns:
        - rsi: RSI value
        - divergence_type: hidden_bullish, hidden_bearish, regular_bullish, regular_bearish, none
        - divergence_signal: buy, sell, none
        - divergence_strength: 0-1

        Useful for backtesting.
        """
        df = self.normalize_columns(df)
        df = df.copy()

        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df['close'])

        # Initialize columns
        df['divergence_type'] = 'none'
        df['divergence_signal'] = 'none'
        df['divergence_strength'] = 0.0

        # Calculate for each bar (after minimum lookback)
        min_bars = self.rsi_period + self.swing_lookback * 2
        for i in range(min_bars, len(df)):
            subset = df.iloc[:i+1]
            result = self.detect_divergence(subset)

            df.iloc[i, df.columns.get_loc('divergence_type')] = result.divergence_type.value
            df.iloc[i, df.columns.get_loc('divergence_signal')] = result.signal
            df.iloc[i, df.columns.get_loc('divergence_strength')] = result.strength

        return df


# Example usage
if __name__ == "__main__":
    import yfinance as yf

    print("=" * 60)
    print("RSI HIDDEN DIVERGENCE DETECTOR TEST")
    print("=" * 60)

    try:
        # Download EUR/USD proxy (FXE ETF) or any stock
        data = yf.download("SPY", period="6mo", interval="1d", progress=False)

        if len(data) > 0:
            print(f"\nLoaded {len(data)} bars")

            # Test detector
            detector = RSIDivergenceDetector()

            # Get current divergence
            result = detector.detect_divergence(data)

            print(f"\nDivergence Analysis:")
            print(f"  Type: {result.divergence_type.value}")
            print(f"  Hidden Bullish: {result.hidden_bullish}")
            print(f"  Hidden Bearish: {result.hidden_bearish}")
            print(f"  Regular Bullish: {result.regular_bullish}")
            print(f"  Regular Bearish: {result.regular_bearish}")
            print(f"  Signal: {result.signal}")
            print(f"  Strength: {result.strength:.2f}")
            print(f"  RSI Current: {result.rsi_current:.1f}")
            print(f"  Reasoning: {result.reasoning}")

            # Get signal dict
            signal = detector.get_signal(data)
            print(f"\nSignal Dict:")
            print(f"  {signal}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
