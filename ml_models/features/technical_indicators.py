"""
Technical Indicators Calculator
==============================

Comprehensive technical indicator calculations for ML feature engineering.
Uses TA-Lib for optimal performance and reliability.

Author: Trading Bot Arsenal
Created: February 2026
"""

import numpy as np
import pandas as pd
import talib
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class TechnicalIndicatorCalculator:
    """
    Comprehensive technical indicator calculator using TA-Lib.

    Provides efficient calculation of all major technical indicators
    for use in ML feature engineering pipelines.
    """

    def __init__(self):
        self.available_indicators = [
            'sma', 'ema', 'wma', 'rsi', 'macd', 'bollinger', 'stoch',
            'atr', 'adx', 'cci', 'williams_r', 'obv', 'vwap', 'mfi',
            'sar', 'aroon', 'trix', 'dmi', 'ultosc', 'roc', 'mom',
            'bop', 'ad', 'chaikin_osc', 'cmf'
        ]

    def calculate_all(self, df: pd.DataFrame,
                     indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate all or specified technical indicators.

        Args:
            df: OHLCV DataFrame
            indicators: List of indicators to calculate (None = all)

        Returns:
            DataFrame with technical indicators
        """
        if indicators is None:
            indicators = self.available_indicators

        features = pd.DataFrame(index=df.index)

        for indicator in indicators:
            try:
                indicator_data = self.calculate_indicator(df, indicator)
                if indicator_data is not None:
                    features = pd.concat([features, indicator_data], axis=1)
            except Exception as e:
                logger.warning(f"Failed to calculate {indicator}: {e}")

        return features

    def calculate_indicator(self, df: pd.DataFrame,
                          indicator: str) -> Optional[pd.DataFrame]:
        """Calculate a specific technical indicator"""

        # Ensure we have required data
        high = df['high'].values if 'high' in df.columns else df['close'].values
        low = df['low'].values if 'low' in df.columns else df['close'].values
        close = df['close'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones_like(close)
        open_price = df['open'].values if 'open' in df.columns else close

        result = pd.DataFrame(index=df.index)

        if indicator == 'sma':
            for period in [5, 10, 20, 50, 100, 200]:
                try:
                    sma = talib.SMA(close, timeperiod=period)
                    result[f'sma_{period}'] = sma
                    result[f'sma_{period}_ratio'] = close / sma
                except Exception as e:
                    logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'ema':
            for period in [5, 10, 20, 50, 100, 200]:
                try:
                    ema = talib.EMA(close, timeperiod=period)
                    result[f'ema_{period}'] = ema
                    result[f'ema_{period}_ratio'] = close / ema
                except Exception as e:
                    logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'wma':
            for period in [10, 20, 50]:
                try:
                    wma = talib.WMA(close, timeperiod=period)
                    result[f'wma_{period}'] = wma
                    result[f'wma_{period}_ratio'] = close / wma
                except Exception as e:
                    logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'rsi':
            for period in [14, 21]:
                try:
                    rsi = talib.RSI(close, timeperiod=period)
                    result[f'rsi_{period}'] = rsi
                    result[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
                    result[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
                except Exception as e:
                    logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'macd':
            try:
                macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                result['macd'] = macd
                result['macd_signal'] = macdsignal
                result['macd_histogram'] = macdhist
                result['macd_cross_up'] = ((macd > macdsignal) & (macd.shift(1) <= macdsignal.shift(1))).astype(int)
                result['macd_cross_down'] = ((macd < macdsignal) & (macd.shift(1) >= macdsignal.shift(1))).astype(int)
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'bollinger':
            for period in [20, 50]:
                try:
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=period, nbdevup=2, nbdevdn=2)
                    result[f'bb_upper_{period}'] = bb_upper
                    result[f'bb_lower_{period}'] = bb_lower
                    result[f'bb_middle_{period}'] = bb_middle
                    result[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
                    result[f'bb_position_{period}'] = (close - bb_lower) / (bb_upper - bb_lower)
                    result[f'bb_squeeze_{period}'] = (result[f'bb_width_{period}'] < result[f'bb_width_{period}'].rolling(20).quantile(0.2)).astype(int)
                except Exception as e:
                    logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'stoch':
            try:
                slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
                result['stoch_k'] = slowk
                result['stoch_d'] = slowd
                result['stoch_oversold'] = (slowk < 20).astype(int)
                result['stoch_overbought'] = (slowk > 80).astype(int)
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'atr':
            for period in [14, 21]:
                try:
                    atr = talib.ATR(high, low, close, timeperiod=period)
                    result[f'atr_{period}'] = atr
                    result[f'atr_{period}_pct'] = atr / close * 100
                except Exception as e:
                    logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'adx':
            try:
                adx = talib.ADX(high, low, close, timeperiod=14)
                plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
                minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
                result['adx'] = adx
                result['plus_di'] = plus_di
                result['minus_di'] = minus_di
                result['adx_strong_trend'] = (adx > 25).astype(int)
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'cci':
            try:
                cci = talib.CCI(high, low, close, timeperiod=20)
                result['cci'] = cci
                result['cci_oversold'] = (cci < -100).astype(int)
                result['cci_overbought'] = (cci > 100).astype(int)
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'williams_r':
            try:
                willr = talib.WILLR(high, low, close, timeperiod=14)
                result['williams_r'] = willr
                result['williams_r_oversold'] = (willr < -80).astype(int)
                result['williams_r_overbought'] = (willr > -20).astype(int)
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'obv':
            try:
                obv = talib.OBV(close, volume)
                result['obv'] = obv
                result['obv_sma'] = talib.SMA(obv, timeperiod=20)
                result['obv_signal'] = (obv > result['obv_sma']).astype(int)
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'vwap':
            try:
                # Volume Weighted Average Price
                typical_price = (high + low + close) / 3
                vwap = (typical_price * volume).cumsum() / volume.cumsum()
                result['vwap'] = vwap
                result['vwap_ratio'] = close / vwap
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'mfi':
            try:
                mfi = talib.MFI(high, low, close, volume, timeperiod=14)
                result['mfi'] = mfi
                result['mfi_oversold'] = (mfi < 20).astype(int)
                result['mfi_overbought'] = (mfi > 80).astype(int)
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'sar':
            try:
                sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
                result['sar'] = sar
                result['sar_bullish'] = (close > sar).astype(int)
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'aroon':
            try:
                aroondown, aroonup = talib.AROON(high, low, timeperiod=25)
                result['aroon_up'] = aroonup
                result['aroon_down'] = aroondown
                result['aroon_oscillator'] = aroonup - aroondown
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'trix':
            try:
                trix = talib.TRIX(close, timeperiod=14)
                result['trix'] = trix
                result['trix_signal'] = talib.SMA(trix, timeperiod=9)
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'dmi':
            try:
                plus_dm = talib.PLUS_DM(high, low, timeperiod=14)
                minus_dm = talib.MINUS_DM(high, low, timeperiod=14)
                result['plus_dm'] = plus_dm
                result['minus_dm'] = minus_dm
                result['dx'] = talib.DX(high, low, close, timeperiod=14)
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'ultosc':
            try:
                ultosc = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
                result['ultimate_oscillator'] = ultosc
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'roc':
            for period in [10, 20]:
                try:
                    roc = talib.ROC(close, timeperiod=period)
                    result[f'roc_{period}'] = roc
                except Exception as e:
                    logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'mom':
            for period in [10, 20]:
                try:
                    mom = talib.MOM(close, timeperiod=period)
                    result[f'momentum_{period}'] = mom
                except Exception as e:
                    logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'bop':
            try:
                bop = talib.BOP(open_price, high, low, close)
                result['balance_of_power'] = bop
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'ad':
            try:
                ad = talib.AD(high, low, close, volume)
                result['accumulation_distribution'] = ad
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'chaikin_osc':
            try:
                chaikin = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
                result['chaikin_oscillator'] = chaikin
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        elif indicator == 'cmf':
            try:
                # Chaikin Money Flow
                mfv = ((close - low) - (high - close)) / (high - low) * volume
                mfv = mfv.fillna(0)
                cmf = mfv.rolling(20).sum() / volume.rolling(20).sum()
                result['chaikin_money_flow'] = cmf
            except Exception as e:
                logger.debug(f"Error computing {indicator}: {e}")

        # Additional derived features
        if len(result.columns) > 0:
            # Add momentum and change features for main indicators
            for col in result.columns:
                if not col.endswith('_momentum') and not col.endswith('_change'):
                    try:
                        result[f'{col}_momentum'] = result[col] - result[col].shift(5)
                        result[f'{col}_change'] = result[col].pct_change(5)
                    except Exception as e:
                        logger.debug(f"Error computing derived feature for {col}: {e}")

        return result if len(result.columns) > 0 else None

    def calculate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom and composite indicators"""
        result = pd.DataFrame(index=df.index)

        high = df['high'].values if 'high' in df.columns else df['close'].values
        low = df['low'].values if 'low' in df.columns else df['close'].values
        close = df['close'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones_like(close)

        try:
            # Price momentum oscillator
            result['pmo'] = (close / np.roll(close, 10) - 1) * 100

            # Volatility adjusted momentum
            returns = np.diff(np.log(close))
            vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
            result['vol_adj_momentum'] = result['pmo'] / (vol * 100)

            # Trend strength indicator
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            result['trend_strength'] = (sma_20 / sma_50 - 1) * 100

            # Market structure breaks
            swing_highs = pd.Series(high).rolling(10).max()
            swing_lows = pd.Series(low).rolling(10).min()
            result['structure_break_up'] = (close > swing_highs.shift(1)).astype(int)
            result['structure_break_down'] = (close < swing_lows.shift(1)).astype(int)

            # Composite momentum score
            rsi_14 = talib.RSI(close, timeperiod=14)
            macd, _, _ = talib.MACD(close)
            stoch_k, _ = talib.STOCH(high, low, close)

            # Normalize indicators to 0-100 scale
            rsi_norm = rsi_14
            macd_norm = np.where(macd > 0, 50 + (macd / np.nanmax(macd)) * 50, 50 + (macd / abs(np.nanmin(macd))) * 50)
            stoch_norm = stoch_k

            result['composite_momentum'] = (rsi_norm + macd_norm + stoch_norm) / 3

        except Exception as e:
            logger.warning(f"Failed to calculate custom indicators: {e}")

        return result

    def get_indicator_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available indicators"""
        return {
            'sma': 'Simple Moving Average - trend following',
            'ema': 'Exponential Moving Average - responsive trend following',
            'wma': 'Weighted Moving Average - recent price emphasis',
            'rsi': 'Relative Strength Index - momentum oscillator',
            'macd': 'MACD - trend and momentum',
            'bollinger': 'Bollinger Bands - volatility and mean reversion',
            'stoch': 'Stochastic Oscillator - momentum',
            'atr': 'Average True Range - volatility measure',
            'adx': 'Average Directional Index - trend strength',
            'cci': 'Commodity Channel Index - momentum',
            'williams_r': 'Williams %R - momentum oscillator',
            'obv': 'On Balance Volume - volume momentum',
            'vwap': 'Volume Weighted Average Price - fair value',
            'mfi': 'Money Flow Index - volume-price momentum',
            'sar': 'Parabolic SAR - trend reversal points',
            'aroon': 'Aroon - trend identification',
            'trix': 'TRIX - momentum oscillator',
            'dmi': 'Directional Movement Index - trend direction',
            'ultosc': 'Ultimate Oscillator - momentum',
            'roc': 'Rate of Change - price momentum',
            'mom': 'Momentum - price change',
            'bop': 'Balance of Power - buying/selling pressure',
            'ad': 'Accumulation/Distribution - volume flow',
            'chaikin_osc': 'Chaikin Oscillator - volume momentum',
            'cmf': 'Chaikin Money Flow - money flow'
        }


def main():
    """Test technical indicators calculator"""
    import yfinance as yf

    # Download test data
    print("Downloading test data...")
    data = yf.download("AAPL", period="1y", progress=False)

    # Initialize calculator
    calculator = TechnicalIndicatorCalculator()

    # Calculate a few indicators for testing
    print("Calculating technical indicators...")
    indicators = calculator.calculate_all(data, indicators=['sma', 'ema', 'rsi', 'macd'])

    print(f"\nTechnical Indicators Results:")
    print(f"Shape: {indicators.shape}")
    print(f"Columns: {list(indicators.columns[:10])}...")  # First 10 columns
    print("\nSample data:")
    print(indicators.tail())

    # Test custom indicators
    print("\nCalculating custom indicators...")
    custom = calculator.calculate_custom_indicators(data)
    print(f"Custom indicators shape: {custom.shape}")
    print(f"Custom columns: {list(custom.columns)}")

    print("\nTechnical indicators test completed successfully!")


if __name__ == "__main__":
    main()