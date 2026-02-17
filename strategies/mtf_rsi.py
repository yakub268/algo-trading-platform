"""
Multi-Timeframe RSI Strategy

Uses RSI confirmation across multiple timeframes to filter false signals.
Entry requires alignment across daily, 4H, and 1H timeframes.

Strategy Logic:
- Daily RSI(14) determines trend bias
- 4H RSI(14) provides intermediate confirmation  
- 1H RSI(14) triggers entry
- All must align for valid signal

Why this works:
- Reduces noise from single-timeframe signals
- Higher probability entries with trend alignment
- Better risk/reward from timing optimization

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import volume-price divergence filter
try:
    from filters.volume_price_divergence import VolumePriceDivergenceFilter
    DIVERGENCE_FILTER_AVAILABLE = True
except ImportError:
    DIVERGENCE_FILTER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MTF_RSI')


class TrendBias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TimeframeAnalysis:
    """RSI analysis for a single timeframe"""
    timeframe: str
    rsi: float
    trend: TrendBias
    oversold: bool
    overbought: bool
    rising: bool  # RSI direction


@dataclass
class MTFSignal:
    """Multi-timeframe trading signal"""
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    
    # RSI values by timeframe
    daily_rsi: float
    h4_rsi: float
    h1_rsi: float
    
    # Alignment
    trend_alignment: bool
    daily_trend: TrendBias
    
    confidence: float
    reasoning: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class MultiTimeframeRSI:
    """
    Multi-Timeframe RSI Strategy
    
    Entry Conditions (Long):
    1. Daily RSI(14) > 50 (bullish bias)
    2. 4H RSI(14) > 40 and rising
    3. 1H RSI(14) < 30 (oversold pullback)
    
    Entry Conditions (Short):
    1. Daily RSI(14) < 50 (bearish bias)
    2. 4H RSI(14) < 60 and falling
    3. 1H RSI(14) > 70 (overbought bounce)
    
    Exit:
    - RSI cross back through 50 on entry timeframe
    - Stop loss: 2% or 1.5x ATR
    - Take profit: 2:1 risk/reward
    """
    
    # RSI parameters
    RSI_PERIOD = 14
    
    # Thresholds
    DAILY_BULLISH = 50
    DAILY_BEARISH = 50
    H4_BULLISH_MIN = 40
    H4_BEARISH_MAX = 60
    H1_OVERSOLD = 30
    H1_OVERBOUGHT = 70
    
    # Risk management
    STOP_LOSS_PCT = 0.02
    ATR_STOP_MULTIPLIER = 1.5
    RISK_REWARD_RATIO = 2.0
    
    def __init__(self, symbol: str = "SPY", paper_mode: bool = None, use_divergence_filter: bool = True):
        self.symbol = symbol
        # Safe default: read from environment, default to PAPER if not set
        if paper_mode is None:
            paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
        self.paper_mode = paper_mode
        self.current_position = None
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None

        # Initialize divergence filter
        self.use_divergence_filter = use_divergence_filter and DIVERGENCE_FILTER_AVAILABLE
        if self.use_divergence_filter:
            self.divergence_filter = VolumePriceDivergenceFilter(lookback=10)
            logger.info(f"MultiTimeframeRSI initialized for {symbol} with divergence filter")
        else:
            self.divergence_filter = None
            logger.info(f"MultiTimeframeRSI initialized for {symbol}")
    
    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        # Protect against division by zero (all gains, no losses = RSI 100)
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))

        return rsi
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns to handle yfinance MultiIndex."""
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
        else:
            df.columns = df.columns.str.lower()
        return df

    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to specified timeframe.

        Args:
            df: DataFrame with OHLCV data (assumes 1-min or similar granular data)
            timeframe: Target timeframe ('1H', '4H', '1D')

        Returns:
            Resampled DataFrame
        """
        df = self.normalize_columns(df)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df.index = pd.to_datetime(df['date'])
            elif 'datetime' in df.columns:
                df.index = pd.to_datetime(df['datetime'])
        
        # Map timeframe to pandas offset
        tf_map = {
            '1H': '1H',
            '4H': '4H', 
            '1D': '1D',
            'D': '1D',
            'daily': '1D'
        }
        
        offset = tf_map.get(timeframe, timeframe)
        
        resampled = df.resample(offset).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> TimeframeAnalysis:
        """
        Analyze RSI for a single timeframe.

        Args:
            df: OHLCV DataFrame for the timeframe
            timeframe: Timeframe label

        Returns:
            TimeframeAnalysis object
        """
        df = self.normalize_columns(df)
        
        # Calculate RSI
        rsi = self.calculate_rsi(df['close'], self.RSI_PERIOD)
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi
        
        # Determine trend
        if current_rsi > 60:
            trend = TrendBias.BULLISH
        elif current_rsi < 40:
            trend = TrendBias.BEARISH
        else:
            trend = TrendBias.NEUTRAL
        
        return TimeframeAnalysis(
            timeframe=timeframe,
            rsi=current_rsi,
            trend=trend,
            oversold=current_rsi < self.H1_OVERSOLD,
            overbought=current_rsi > self.H1_OVERBOUGHT,
            rising=current_rsi > prev_rsi
        )
    
    def generate_signal(
        self, 
        daily_df: pd.DataFrame,
        h4_df: pd.DataFrame = None,
        h1_df: pd.DataFrame = None
    ) -> MTFSignal:
        """
        Generate multi-timeframe RSI signal.
        
        For daily-only data, we simulate lower timeframes from recent bars.
        
        Args:
            daily_df: Daily OHLCV data
            h4_df: 4-hour OHLCV data (optional)
            h1_df: 1-hour OHLCV data (optional)
            
        Returns:
            MTFSignal object
        """
        daily_df = self.normalize_columns(daily_df)
        
        # Analyze daily timeframe
        daily_analysis = self.analyze_timeframe(daily_df, 'daily')
        
        # If we don't have intraday data, simulate from daily
        # WARNING: This is an approximation only - daily RSI != intraday RSI
        # For accurate MTF signals, provide actual 4H and 1H data
        if h4_df is None:
            # Use recent 5 days as proxy for 4H (approximation only)
            h4_df = daily_df.tail(5)
            logger.warning("Using daily data as 4H proxy - results may differ from true MTF analysis")
        if h1_df is None:
            # Use recent 3 days as proxy for 1H (approximation only)
            h1_df = daily_df.tail(3)
            logger.warning("Using daily data as 1H proxy - results may differ from true MTF analysis")
        
        h4_analysis = self.analyze_timeframe(h4_df, '4H')
        h1_analysis = self.analyze_timeframe(h1_df, '1H')
        
        current_price = daily_df['close'].iloc[-1]
        atr = self.calculate_atr(
            daily_df['high'], 
            daily_df['low'], 
            daily_df['close']
        ).iloc[-1]
        
        signal_type = SignalType.HOLD
        trend_alignment = False
        confidence = 0.5
        reasoning = "No signal"
        stop_loss = None
        take_profit = None
        
        # Check for LONG setup
        if (daily_analysis.rsi > self.DAILY_BULLISH and
            h4_analysis.rsi > self.H4_BULLISH_MIN and
            h4_analysis.rising and
            h1_analysis.oversold):

            # Check divergence filter
            divergence_filtered = False
            if self.use_divergence_filter and self.divergence_filter:
                adj = self.divergence_filter.get_signal_adjustment(daily_df)
                if adj['filter_buy']:
                    divergence_filtered = True
                    reasoning = f"Entry filtered: {adj['reasoning']}"
                    logger.info(f"MTF_RSI BUY signal filtered by divergence")

            if not divergence_filtered:
                signal_type = SignalType.BUY
                trend_alignment = True

                # Calculate confidence based on alignment strength
                confidence = 0.6
                if daily_analysis.rsi > 55:
                    confidence += 0.1
                if h4_analysis.rsi > 50:
                    confidence += 0.1
                if h1_analysis.rsi < 25:
                    confidence += 0.1
                confidence = min(confidence, 0.95)

                # Apply divergence adjustment
                if self.use_divergence_filter and self.divergence_filter:
                    adj = self.divergence_filter.get_signal_adjustment(daily_df)
                    confidence = max(0.1, min(0.95, confidence + adj['confidence_adjustment']))

                # Calculate stops
                stop_loss = current_price - max(
                    current_price * self.STOP_LOSS_PCT,
                    atr * self.ATR_STOP_MULTIPLIER
                )
                risk = current_price - stop_loss
                take_profit = current_price + (risk * self.RISK_REWARD_RATIO)

                reasoning = (
                    f"MTF Long: Daily RSI={daily_analysis.rsi:.1f} (bullish), "
                    f"4H RSI={h4_analysis.rsi:.1f} (rising), "
                    f"1H RSI={h1_analysis.rsi:.1f} (oversold pullback)"
                )
        
        # Check for SHORT setup
        elif (daily_analysis.rsi < self.DAILY_BEARISH and
              h4_analysis.rsi < self.H4_BEARISH_MAX and
              not h4_analysis.rising and
              h1_analysis.overbought):
            
            signal_type = SignalType.SELL
            trend_alignment = True
            
            confidence = 0.6
            if daily_analysis.rsi < 45:
                confidence += 0.1
            if h4_analysis.rsi < 50:
                confidence += 0.1
            if h1_analysis.rsi > 75:
                confidence += 0.1
            confidence = min(confidence, 0.95)
            
            stop_loss = current_price + max(
                current_price * self.STOP_LOSS_PCT,
                atr * self.ATR_STOP_MULTIPLIER
            )
            risk = stop_loss - current_price
            take_profit = current_price - (risk * self.RISK_REWARD_RATIO)
            
            reasoning = (
                f"MTF Short: Daily RSI={daily_analysis.rsi:.1f} (bearish), "
                f"4H RSI={h4_analysis.rsi:.1f} (falling), "
                f"1H RSI={h1_analysis.rsi:.1f} (overbought bounce)"
            )
        
        else:
            # Explain why no signal
            reasons = []
            if daily_analysis.rsi > 45 and daily_analysis.rsi < 55:
                reasons.append(f"Daily RSI neutral ({daily_analysis.rsi:.1f})")
            if not h4_analysis.rising and daily_analysis.rsi > 50:
                reasons.append("4H not confirming bullish")
            if h4_analysis.rising and daily_analysis.rsi < 50:
                reasons.append("4H not confirming bearish")
            if not h1_analysis.oversold and not h1_analysis.overbought:
                reasons.append(f"1H RSI not at extreme ({h1_analysis.rsi:.1f})")
            
            reasoning = "No alignment: " + "; ".join(reasons) if reasons else "Waiting for setup"
        
        return MTFSignal(
            signal_type=signal_type,
            symbol=self.symbol,
            timestamp=datetime.now(timezone.utc),
            price=current_price,
            daily_rsi=daily_analysis.rsi,
            h4_rsi=h4_analysis.rsi,
            h1_rsi=h1_analysis.rsi,
            trend_alignment=trend_alignment,
            daily_trend=daily_analysis.trend,
            confidence=confidence,
            reasoning=reasoning,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def should_exit(self, entry_signal: MTFSignal, current_h1_rsi: float) -> Tuple[bool, str]:
        """Check exit conditions"""
        if entry_signal.signal_type == SignalType.BUY:
            # Exit long when 1H RSI crosses above 50 from below
            if current_h1_rsi > 50:
                return True, f"1H RSI exit: {current_h1_rsi:.1f} > 50"
        
        elif entry_signal.signal_type == SignalType.SELL:
            # Exit short when 1H RSI crosses below 50 from above
            if current_h1_rsi < 50:
                return True, f"1H RSI exit: {current_h1_rsi:.1f} < 50"
        
        return False, ""
    
    def enter_position(self, signal: MTFSignal) -> Dict:
        """Enter a position based on signal"""
        self.current_position = signal.signal_type.value
        self.entry_price = signal.price
        self.stop_loss_price = signal.stop_loss
        self.take_profit_price = signal.take_profit
        
        logger.info(
            f"[MTF ENTRY] {signal.signal_type.value.upper()} {self.symbol} @ ${signal.price:.2f}, "
            f"SL: ${signal.stop_loss:.2f}, TP: ${signal.take_profit:.2f}"
        )
        
        return {
            "action": signal.signal_type.value,
            "symbol": self.symbol,
            "price": signal.price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "confidence": signal.confidence
        }
    
    def exit_position(self, price: float, reason: str) -> Dict:
        """Exit current position"""
        if not self.entry_price:
            return {"action": "none"}
        
        if self.current_position == "buy":
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - price) / self.entry_price
        
        logger.info(f"[MTF EXIT] {self.symbol} @ ${price:.2f}, P&L: {pnl_pct:+.2%}, Reason: {reason}")
        
        result = {
            "action": "exit",
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "exit_price": price,
            "pnl_pct": pnl_pct,
            "reason": reason
        }
        
        self.current_position = None
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        
        return result
    
    def get_status(self) -> Dict:
        """Get current strategy status"""
        return {
            "symbol": self.symbol,
            "position": self.current_position,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss_price,
            "take_profit": self.take_profit_price,
            "parameters": {
                "rsi_period": self.RSI_PERIOD,
                "daily_threshold": self.DAILY_BULLISH,
                "h1_oversold": self.H1_OVERSOLD,
                "h1_overbought": self.H1_OVERBOUGHT,
                "risk_reward": self.RISK_REWARD_RATIO
            }
        }


def backtest_mtf_rsi(df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
    """
    Backtest Multi-Timeframe RSI strategy.

    Uses daily data and simulates lower timeframes.
    """
    strategy = MultiTimeframeRSI()
    df = strategy.normalize_columns(df)
    
    # Calculate indicators
    df['rsi'] = strategy.calculate_rsi(df['close'], strategy.RSI_PERIOD)
    df['atr'] = strategy.calculate_atr(df['high'], df['low'], df['close'])
    
    capital = initial_capital
    position = None
    entry_price = 0
    entry_type = None
    stop_loss = 0
    take_profit = 0
    trades = []
    
    # Need enough data for RSI
    start_idx = strategy.RSI_PERIOD + 5
    
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        
        # Get recent data for MTF simulation
        recent_daily = df.iloc[max(0, i-20):i+1]
        recent_h4 = df.iloc[max(0, i-5):i+1]  # Proxy
        recent_h1 = df.iloc[max(0, i-3):i+1]  # Proxy
        
        if position is None:
            # Generate signal
            signal = strategy.generate_signal(recent_daily, recent_h4, recent_h1)
            
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                position = signal.signal_type.value
                entry_price = row['close']
                entry_type = signal.signal_type
                entry_date = row.name
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit
        
        else:
            # Check exits
            exit_triggered = False
            exit_reason = ""
            exit_price = row['close']
            
            if position == "buy":
                if row['low'] <= stop_loss:
                    exit_triggered = True
                    exit_reason = "stop_loss"
                    exit_price = stop_loss
                elif row['high'] >= take_profit:
                    exit_triggered = True
                    exit_reason = "take_profit"
                    exit_price = take_profit
                elif row['rsi'] > 50 and df.iloc[i-1]['rsi'] < 50:
                    exit_triggered = True
                    exit_reason = "rsi_exit"
            
            else:  # short
                if row['high'] >= stop_loss:
                    exit_triggered = True
                    exit_reason = "stop_loss"
                    exit_price = stop_loss
                elif row['low'] <= take_profit:
                    exit_triggered = True
                    exit_reason = "take_profit"
                    exit_price = take_profit
                elif row['rsi'] < 50 and df.iloc[i-1]['rsi'] > 50:
                    exit_triggered = True
                    exit_reason = "rsi_exit"
            
            if exit_triggered:
                if position == "buy":
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price
                
                pnl = capital * 0.25 * pnl_pct  # 25% position size
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': row.name,
                    'type': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl': pnl,
                    'reason': exit_reason
                })
                
                capital += pnl
                position = None
    
    # Calculate metrics
    if trades:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        return {
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades),
            'total_return': (capital - initial_capital) / initial_capital,
            'final_capital': capital,
            'avg_win': np.mean([t['pnl_pct'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([t['pnl_pct'] for t in losses]) if losses else 0,
            'profit_factor': abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 0,
            'trades': trades
        }
    
    return {'total_trades': 0, 'trades': []}


if __name__ == "__main__":
    import yfinance as yf
    
    print("=" * 60)
    print("MULTI-TIMEFRAME RSI STRATEGY")
    print("=" * 60)
    
    try:
        # Download test data
        spy = yf.download("SPY", period="2y", progress=False)
        
        if len(spy) > 0:
            print(f"Loaded {len(spy)} days of SPY data")
            
            # Run backtest
            results = backtest_mtf_rsi(spy)
            
            print(f"\nBacktest Results:")
            print(f"  Total Trades: {results['total_trades']}")
            print(f"  Win Rate: {results.get('win_rate', 0):.1%}")
            print(f"  Total Return: {results.get('total_return', 0):.1%}")
            print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
            
            # Current signal
            strategy = MultiTimeframeRSI(symbol="SPY")
            signal = strategy.generate_signal(spy)
            
            print(f"\nCurrent Signal: {signal.signal_type.value}")
            print(f"  Daily RSI: {signal.daily_rsi:.1f}")
            print(f"  4H RSI: {signal.h4_rsi:.1f}")
            print(f"  1H RSI: {signal.h1_rsi:.1f}")
            print(f"  Trend Alignment: {signal.trend_alignment}")
            print(f"  Daily Trend: {signal.daily_trend.value}")
            print(f"  Confidence: {signal.confidence:.0%}")
            print(f"  Reasoning: {signal.reasoning}")
            
            if signal.stop_loss:
                print(f"  Stop Loss: ${signal.stop_loss:.2f}")
                print(f"  Take Profit: ${signal.take_profit:.2f}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
