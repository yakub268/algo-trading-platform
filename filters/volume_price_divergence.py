"""
Volume-Price Divergence Filter
==============================

Filters trade signals based on volume-price divergence analysis.

Key Concept:
- Volume should confirm price moves
- If price makes new highs but volume is declining = bearish divergence (weak breakout)
- If price makes new lows but volume is declining = bullish divergence (selling exhaustion)

Research Base:
- "Price-volume divergence is one of the most reliable early warning signals"
- Volume precedes price in most cases
- Divergence often signals trend exhaustion

Usage:
    filter = VolumePriceDivergenceFilter()

    # Check if a BUY signal should be filtered out
    if filter.should_filter_buy(df):
        return None  # Skip this signal

    # Or get detailed divergence info
    divergence = filter.detect_divergence(df)
    if divergence['type'] == 'bearish':
        # Reduce position size or skip

Author: Trading Bot Arsenal
Created: January 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('VolumePriceDivergence')


class DivergenceType(Enum):
    """Types of volume-price divergence"""
    NONE = "none"
    BULLISH = "bullish"    # Price down, volume down = selling exhaustion
    BEARISH = "bearish"    # Price up, volume down = weak rally
    CONFIRMATION = "confirmation"  # Price and volume aligned


@dataclass
class DivergenceResult:
    """Result of divergence analysis"""
    divergence_type: DivergenceType
    strength: float  # 0-1, higher = stronger divergence
    price_trend: str  # "up", "down", "flat"
    volume_trend: str  # "up", "down", "flat"
    should_filter_buy: bool
    should_filter_sell: bool
    reasoning: str
    lookback_period: int


class VolumePriceDivergenceFilter:
    """
    Volume-Price Divergence Filter

    Detects when price moves are not confirmed by volume, which often
    signals weak moves that are likely to fail.

    Filter Logic:
    - For BUY signals: Filter out if bearish divergence (price up, volume down)
    - For SELL signals: Filter out if bullish divergence (price down, volume down)

    Parameters:
    - lookback: Number of bars to analyze (default 10)
    - price_threshold: Min % change to consider a trend (default 2%)
    - volume_threshold: Min % change in volume to consider significant (default 15%)
    - strength_threshold: Min divergence strength to trigger filter (default 0.5)
    """

    def __init__(
        self,
        lookback: int = 10,
        price_threshold: float = 0.02,
        volume_threshold: float = 0.15,
        strength_threshold: float = 0.5
    ):
        self.lookback = lookback
        self.price_threshold = price_threshold
        self.volume_threshold = volume_threshold
        self.strength_threshold = strength_threshold

        logger.info(
            f"VolumePriceDivergenceFilter initialized: "
            f"lookback={lookback}, price_thresh={price_threshold:.1%}, "
            f"vol_thresh={volume_threshold:.1%}, strength_thresh={strength_threshold}"
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

    def _calculate_trend(self, series: pd.Series) -> Tuple[str, float]:
        """
        Calculate trend direction and strength using linear regression.

        Returns:
            Tuple of (direction: "up"/"down"/"flat", slope_normalized: float)
        """
        if len(series) < 3:
            return "flat", 0.0

        x = np.arange(len(series))
        y = series.values

        # Handle any NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            return "flat", 0.0

        x = x[mask]
        y = y[mask]

        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]

        # Normalize slope by mean value
        mean_val = np.mean(y)
        if mean_val != 0:
            normalized_slope = slope / mean_val * len(series)
        else:
            normalized_slope = 0

        # Determine direction
        if normalized_slope > self.price_threshold / 2:
            direction = "up"
        elif normalized_slope < -self.price_threshold / 2:
            direction = "down"
        else:
            direction = "flat"

        return direction, abs(normalized_slope)

    def _calculate_divergence_strength(
        self,
        price_change: float,
        volume_change: float,
        price_direction: str,
        volume_direction: str
    ) -> float:
        """
        Calculate divergence strength (0-1).

        Strength is higher when:
        - Price and volume move in opposite directions
        - The magnitude of each move is larger
        """
        # Only calculate strength if there's actual divergence
        if price_direction == volume_direction or price_direction == "flat" or volume_direction == "flat":
            return 0.0

        # Combine the magnitudes
        magnitude = (abs(price_change) + abs(volume_change)) / 2

        # Normalize to 0-1 range (cap at 1.0)
        strength = min(magnitude / 0.3, 1.0)

        return strength

    def detect_divergence(self, df: pd.DataFrame) -> DivergenceResult:
        """
        Detect volume-price divergence in the data.

        Args:
            df: DataFrame with 'close' and 'volume' columns (or will be normalized)

        Returns:
            DivergenceResult with divergence type and analysis
        """
        # Normalize columns
        df = self.normalize_columns(df)

        # Ensure we have enough data
        if len(df) < self.lookback:
            return DivergenceResult(
                divergence_type=DivergenceType.NONE,
                strength=0.0,
                price_trend="unknown",
                volume_trend="unknown",
                should_filter_buy=False,
                should_filter_sell=False,
                reasoning="Insufficient data for divergence analysis",
                lookback_period=self.lookback
            )

        # Get recent data
        recent = df.tail(self.lookback)

        # Calculate price trend
        price_direction, price_magnitude = self._calculate_trend(recent['close'])

        # Calculate volume trend
        volume_direction, volume_magnitude = self._calculate_trend(recent['volume'])

        # Calculate actual changes for logging
        price_change = (recent['close'].iloc[-1] / recent['close'].iloc[0]) - 1
        volume_change = (recent['volume'].iloc[-1] / recent['volume'].iloc[0]) - 1 if recent['volume'].iloc[0] > 0 else 0

        # Determine divergence type
        divergence_type = DivergenceType.NONE
        should_filter_buy = False
        should_filter_sell = False

        # Bearish divergence: Price up, volume down
        if price_direction == "up" and volume_direction == "down":
            divergence_type = DivergenceType.BEARISH
            should_filter_buy = True  # Don't buy into weak rally
            reasoning = f"Bearish divergence: Price trending UP ({price_change:+.1%}) but volume DOWN ({volume_change:+.1%}). Rally lacks conviction."

        # Bullish divergence: Price down, volume down
        elif price_direction == "down" and volume_direction == "down":
            divergence_type = DivergenceType.BULLISH
            should_filter_sell = True  # Don't sell into exhausted selling
            reasoning = f"Bullish divergence: Price DOWN ({price_change:+.1%}) with declining volume ({volume_change:+.1%}). Selling pressure exhausted."

        # Confirmation: Trends aligned
        elif price_direction == volume_direction and price_direction != "flat":
            divergence_type = DivergenceType.CONFIRMATION
            reasoning = f"Volume confirms price: Both {price_direction} (Price: {price_change:+.1%}, Volume: {volume_change:+.1%})"

        else:
            reasoning = f"No significant divergence detected. Price: {price_direction}, Volume: {volume_direction}"

        # Calculate strength
        strength = self._calculate_divergence_strength(
            price_change, volume_change, price_direction, volume_direction
        )

        # Only filter if strength exceeds threshold
        if strength < self.strength_threshold:
            should_filter_buy = False
            should_filter_sell = False
            if divergence_type in [DivergenceType.BEARISH, DivergenceType.BULLISH]:
                reasoning += f" (Strength {strength:.2f} below threshold {self.strength_threshold})"

        return DivergenceResult(
            divergence_type=divergence_type,
            strength=strength,
            price_trend=price_direction,
            volume_trend=volume_direction,
            should_filter_buy=should_filter_buy,
            should_filter_sell=should_filter_sell,
            reasoning=reasoning,
            lookback_period=self.lookback
        )

    def should_filter_buy(self, df: pd.DataFrame) -> bool:
        """
        Quick check if a BUY signal should be filtered.

        Returns True if bearish divergence detected (weak rally).
        """
        result = self.detect_divergence(df)
        if result.should_filter_buy:
            logger.info(f"Filtering BUY: {result.reasoning}")
        return result.should_filter_buy

    def should_filter_sell(self, df: pd.DataFrame) -> bool:
        """
        Quick check if a SELL signal should be filtered.

        Returns True if bullish divergence detected (exhausted selling).
        """
        result = self.detect_divergence(df)
        if result.should_filter_sell:
            logger.info(f"Filtering SELL: {result.reasoning}")
        return result.should_filter_sell

    def get_signal_adjustment(self, df: pd.DataFrame) -> Dict:
        """
        Get signal adjustment recommendations based on divergence.

        Returns dict with:
        - filter_buy: bool
        - filter_sell: bool
        - confidence_adjustment: float (-0.3 to 0.1)
        - position_size_multiplier: float (0.5 to 1.0)
        - reasoning: str
        """
        result = self.detect_divergence(df)

        # Default adjustments
        confidence_adj = 0.0
        size_mult = 1.0

        if result.divergence_type == DivergenceType.BEARISH:
            confidence_adj = -0.2 * result.strength
            size_mult = max(0.5, 1 - result.strength * 0.5)

        elif result.divergence_type == DivergenceType.BULLISH:
            confidence_adj = -0.1 * result.strength  # Slight reduction
            size_mult = max(0.7, 1 - result.strength * 0.3)

        elif result.divergence_type == DivergenceType.CONFIRMATION:
            confidence_adj = 0.1 * result.strength  # Boost for confirmation
            size_mult = 1.0

        return {
            'filter_buy': result.should_filter_buy,
            'filter_sell': result.should_filter_sell,
            'divergence_type': result.divergence_type.value,
            'strength': result.strength,
            'confidence_adjustment': confidence_adj,
            'position_size_multiplier': size_mult,
            'reasoning': result.reasoning
        }

    def add_divergence_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add divergence analysis columns to a DataFrame.

        Adds columns:
        - volume_price_divergence: "bullish", "bearish", "confirmation", "none"
        - divergence_strength: 0-1
        - filter_buy: bool
        - filter_sell: bool

        Useful for backtesting and visualization.
        """
        df = self.normalize_columns(df)
        df = df.copy()

        # Initialize columns
        df['volume_price_divergence'] = 'none'
        df['divergence_strength'] = 0.0
        df['filter_buy'] = False
        df['filter_sell'] = False

        # Calculate for each bar (after lookback period)
        for i in range(self.lookback, len(df)):
            subset = df.iloc[i-self.lookback:i+1]
            result = self.detect_divergence(subset)

            df.iloc[i, df.columns.get_loc('volume_price_divergence')] = result.divergence_type.value
            df.iloc[i, df.columns.get_loc('divergence_strength')] = result.strength
            df.iloc[i, df.columns.get_loc('filter_buy')] = result.should_filter_buy
            df.iloc[i, df.columns.get_loc('filter_sell')] = result.should_filter_sell

        return df


# Example usage
if __name__ == "__main__":
    import yfinance as yf

    print("=" * 60)
    print("VOLUME-PRICE DIVERGENCE FILTER TEST")
    print("=" * 60)

    # Download SPY data
    try:
        spy = yf.download("SPY", period="3mo", interval="1d", progress=False)

        if len(spy) > 0:
            print(f"\nLoaded {len(spy)} days of SPY data")

            # Test the filter
            filter = VolumePriceDivergenceFilter(lookback=10)

            # Get current divergence
            result = filter.detect_divergence(spy)

            print(f"\nCurrent Divergence Analysis:")
            print(f"  Type: {result.divergence_type.value}")
            print(f"  Strength: {result.strength:.2f}")
            print(f"  Price Trend: {result.price_trend}")
            print(f"  Volume Trend: {result.volume_trend}")
            print(f"  Filter Buy: {result.should_filter_buy}")
            print(f"  Filter Sell: {result.should_filter_sell}")
            print(f"  Reasoning: {result.reasoning}")

            # Get adjustment recommendations
            adj = filter.get_signal_adjustment(spy)
            print(f"\nSignal Adjustments:")
            print(f"  Confidence Adjustment: {adj['confidence_adjustment']:+.2f}")
            print(f"  Position Size Multiplier: {adj['position_size_multiplier']:.2f}")

            # Add columns and show recent data
            df_with_divergence = filter.add_divergence_columns(spy)
            recent = df_with_divergence.tail(5)[['close', 'volume', 'volume_price_divergence', 'divergence_strength']]
            print(f"\nRecent Divergence Data:")
            print(recent.to_string())

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
