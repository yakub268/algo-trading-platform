# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
"""
EMA RSI Strategy for Freqtrade
Simple trend-following with EMA crossover and RSI confirmation.
"""

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class EMARSIStrategy(IStrategy):
    """
    EMA/RSI Trend Following Strategy
    - Buy: EMA12 crosses above EMA26 AND RSI < 70
    - Sell: EMA12 crosses below EMA26 OR RSI > 80
    """

    INTERFACE_VERSION = 3

    # Minimal ROI - let trades run
    minimal_roi = {
        "0": 0.10,   # 10% profit target
        "60": 0.05,  # 5% after 1 hour
        "120": 0.02  # 2% after 2 hours
    }

    # Stoploss
    stoploss = -0.05  # 5% stoploss

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Timeframe
    timeframe = '15m'

    # Run on new candles only
    process_only_new_candles = True

    # Startup candles needed
    startup_candle_count = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add indicators to dataframe."""

        # EMA
        dataframe['ema12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema26'] = ta.EMA(dataframe, timeperiod=26)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define entry conditions."""

        dataframe.loc[
            (
                # EMA crossover (12 crosses above 26)
                (dataframe['ema12'] > dataframe['ema26']) &
                (dataframe['ema12'].shift(1) <= dataframe['ema26'].shift(1)) &
                # RSI not overbought
                (dataframe['rsi'] < 70) &
                # Volume confirmation
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit conditions."""

        dataframe.loc[
            (
                # EMA crossover down OR RSI overbought
                ((dataframe['ema12'] < dataframe['ema26']) &
                 (dataframe['ema12'].shift(1) >= dataframe['ema26'].shift(1))) |
                (dataframe['rsi'] > 80)
            ),
            'exit_long'
        ] = 1

        return dataframe
