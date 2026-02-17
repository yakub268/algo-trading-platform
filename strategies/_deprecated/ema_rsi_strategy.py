"""
EMA/RSI Trend Following Strategy for Alpaca via Freqtrade

Simple trend-following strategy for the validation phase.
DO NOT OPTIMIZE until paper trading proves baseline works.

Entry: EMA crossover + RSI confirmation
Exit: Reverse EMA crossover or trailing stop

Author: Jacob
Created: January 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    import talib.abstract as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available, using pandas-based indicators")


@dataclass
class EMARSISignal:
    """Trading signal from EMA/RSI strategy"""
    signal: str  # 'buy', 'sell', 'hold'
    symbol: str
    price: float
    ema12: float
    ema26: float
    rsi: float
    confidence: float
    timestamp: datetime


class EMARSIStrategy:
    """
    EMA Crossover with RSI Filter Strategy

    Uses 12/26 EMA crossover for trend direction with RSI
    confirmation to filter out weak signals.
    """

    def __init__(self):
        self.name = "EMARSIStrategy"

        # Strategy parameters
        self.ema_short_period = 12
        self.ema_long_period = 26
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30

        # Risk management
        self.stop_loss_pct = 0.05      # 5% stop loss
        self.take_profit_pct = 0.15    # 15% take profit
        self.trailing_stop_pct = 0.02  # 2% trailing stop

        # Required candles for calculation
        self.min_periods = max(self.ema_long_period, self.rsi_period) + 5

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        if TALIB_AVAILABLE:
            return pd.Series(ta.EMA(data.values, timeperiod=period), index=data.index)
        else:
            return data.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        if TALIB_AVAILABLE:
            return pd.Series(ta.RSI(data.values, timeperiod=period), index=data.index)
        else:
            # Pandas-based RSI calculation with division-by-zero protection
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            # Protect against division by zero (all gains, no losses)
            rs = gain / loss.replace(0, np.finfo(float).eps)
            return 100 - (100 / (1 + rs))

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        df = df.copy()

        # Calculate EMAs
        df['ema_fast'] = self.calculate_ema(df['close'], self.ema_short_period)
        df['ema_slow'] = self.calculate_ema(df['close'], self.ema_long_period)

        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)

        # EMA difference for trend strength
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        df['ema_cross_up'] = (
            (df['ema_fast'] > df['ema_slow']) &
            (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        )
        df['ema_cross_down'] = (
            (df['ema_fast'] < df['ema_slow']) &
            (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
        )

        return df

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[EMARSISignal]:
        """
        Generate trading signal for a given symbol and dataframe.

        Args:
            symbol: Trading symbol
            df: OHLCV dataframe

        Returns:
            EMARSISignal or None if no signal
        """
        if len(df) < self.min_periods:
            return None

        # Add indicators
        df_with_indicators = self.populate_indicators(df)

        # Get latest row
        latest = df_with_indicators.iloc[-1]
        prev = df_with_indicators.iloc[-2]

        current_price = latest['close']
        ema12 = latest['ema_fast']
        ema26 = latest['ema_slow']
        rsi = latest['rsi']

        # Check for buy signal
        if self._is_buy_signal(latest, prev):
            # Calculate confidence based on signal strength
            confidence = self._calculate_confidence(latest, 'buy')

            return EMARSISignal(
                signal='buy',
                symbol=symbol,
                price=current_price,
                ema12=ema12,
                ema26=ema26,
                rsi=rsi,
                confidence=confidence,
                timestamp=datetime.now(timezone.utc)
            )

        # Check for sell signal
        elif self._is_sell_signal(latest, prev):
            confidence = self._calculate_confidence(latest, 'sell')

            return EMARSISignal(
                signal='sell',
                symbol=symbol,
                price=current_price,
                ema12=ema12,
                ema26=ema26,
                rsi=rsi,
                confidence=confidence,
                timestamp=datetime.now(timezone.utc)
            )

        return None

    def _is_buy_signal(self, latest: pd.Series, prev: pd.Series) -> bool:
        """Check if current conditions indicate a buy signal"""
        return (
            # EMA bullish crossover
            latest['ema_cross_up'] and
            # RSI confirmation (momentum but not overbought)
            50 < latest['rsi'] < self.rsi_overbought and
            # Volume check (if available)
            latest.get('volume', 1) > 0
        )

    def _is_sell_signal(self, latest: pd.Series, prev: pd.Series) -> bool:
        """Check if current conditions indicate a sell signal"""
        return (
            # EMA bearish crossover
            latest['ema_cross_down'] or
            # RSI overbought exit (take profit when momentum exhausted)
            latest['rsi'] > self.rsi_overbought
            # NOTE: Removed RSI oversold exit - was incorrectly exiting longs at bottoms
        )

    def _calculate_confidence(self, latest: pd.Series, signal_type: str) -> float:
        """Calculate confidence score for the signal"""
        confidence = 0.5  # Base confidence

        if signal_type == 'buy':
            # Stronger crossover = higher confidence
            ema_diff_pct = abs(latest['ema_diff'] / latest['close']) * 100
            confidence += min(0.3, ema_diff_pct * 0.1)

            # RSI in sweet spot (55-65) = higher confidence
            if 55 <= latest['rsi'] <= 65:
                confidence += 0.2

        elif signal_type == 'sell':
            # Strong bearish crossover or extreme RSI
            if latest['ema_cross_down']:
                confidence += 0.2
            if latest['rsi'] > 80 or latest['rsi'] < 20:
                confidence += 0.3

        return min(1.0, confidence)

    def calculate_position_size(self, account_value: float, risk_pct: float = 0.02) -> float:
        """
        Calculate position size based on account value and risk.

        Args:
            account_value: Current account value
            risk_pct: Percentage of account to risk (default 2%)

        Returns:
            Position size in dollars
        """
        risk_amount = account_value * risk_pct
        position_size = risk_amount / self.stop_loss_pct
        return position_size
