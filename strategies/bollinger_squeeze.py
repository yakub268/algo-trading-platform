"""
Bollinger Band Squeeze Strategy

Volatility breakout strategy that identifies low-volatility consolidations
followed by explosive moves.

Strategy Logic:
- Entry: BB width at 6-month low (squeeze) + breakout above upper band + volume surge
- Exit: Price closes below middle band OR Supertrend turns bearish
- Stop: 3% or 2x ATR

Why this works:
- Low volatility periods precede high volatility moves
- Volume confirmation filters false breakouts
- Supertrend provides trend direction

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
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
logger = logging.getLogger('BollingerSqueeze')


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width_percentile: float
    volume_ratio: float
    supertrend: float
    supertrend_direction: str
    confidence: float
    reasoning: str
    stop_loss: Optional[float] = None


class BollingerSqueezeStrategy:
    """
    Bollinger Band Squeeze + Supertrend Strategy
    
    Entry Conditions:
    1. BB width at 6-month low (bottom 20%)
    2. Price breaks above upper band
    3. Volume > 1.5x 20-day average
    4. Supertrend is bullish
    
    Exit Conditions:
    1. Price closes below middle band (SMA20)
    2. Supertrend turns bearish
    3. Stop loss hit
    """
    
    # Bollinger Band parameters
    BB_PERIOD = 20
    BB_STD = 2.0
    
    # Squeeze detection
    SQUEEZE_PERCENTILE = 0.20  # Width in bottom 20%
    WIDTH_LOOKBACK = 126  # ~6 months
    
    # Volume confirmation
    VOLUME_MULTIPLIER = 1.5
    VOLUME_PERIOD = 20
    
    # Supertrend parameters
    ST_PERIOD = 10
    ST_MULTIPLIER = 3.0
    
    # Risk management
    STOP_LOSS_PCT = 0.03
    ATR_STOP_MULTIPLIER = 2.0
    
    def __init__(self, symbol: str = "SPY", paper_mode: bool = None, use_divergence_filter: bool = True):
        self.symbol = symbol
        # Safe default: read from environment, default to PAPER if not set
        if paper_mode is None:
            paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
        self.paper_mode = paper_mode
        self.current_position = None
        self.entry_price = None
        self.stop_loss_price = None

        # Initialize divergence filter
        self.use_divergence_filter = use_divergence_filter and DIVERGENCE_FILTER_AVAILABLE
        if self.use_divergence_filter:
            self.divergence_filter = VolumePriceDivergenceFilter(lookback=10)
            logger.info(f"BollingerSqueezeStrategy initialized for {symbol} with divergence filter")
        else:
            self.divergence_filter = None
            logger.info(f"BollingerSqueezeStrategy initialized for {symbol}")
    
    @staticmethod
    def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def calculate_bb_width(upper: pd.Series, lower: pd.Series, middle: pd.Series) -> pd.Series:
        """Calculate Bollinger Band width (normalized)"""
        return (upper - lower) / middle
    
    @staticmethod
    def calculate_width_percentile(width: pd.Series, lookback: int = 126) -> pd.Series:
        """Calculate rolling percentile of BB width"""
        return width.rolling(window=lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def calculate_supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                            period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Supertrend indicator.

        Returns:
            Tuple of (supertrend_line, direction)
            Direction: 1 = bullish, -1 = bearish
        """
        atr = BollingerSqueezeStrategy.calculate_atr(high, low, close, period)

        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=float)

        # Initialize first values
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1  # Start bearish (conservative)

        for i in range(1, len(close)):
            # Carry forward previous direction by default
            prev_direction = direction.iloc[i-1]
            prev_supertrend = supertrend.iloc[i-1]

            # Adjust bands to prevent whipsaws (bands can only move in trend direction)
            curr_lower = lower_band.iloc[i]
            curr_upper = upper_band.iloc[i]

            if prev_direction == 1:  # Was bullish
                # Lower band can only rise, not fall
                curr_lower = max(curr_lower, supertrend.iloc[i-1]) if not pd.isna(supertrend.iloc[i-1]) else curr_lower
            else:  # Was bearish
                # Upper band can only fall, not rise
                curr_upper = min(curr_upper, supertrend.iloc[i-1]) if not pd.isna(supertrend.iloc[i-1]) else curr_upper

            # Determine direction change based on close vs supertrend
            if prev_direction == 1:  # Was bullish
                if close.iloc[i] < prev_supertrend:
                    # Flip to bearish
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = curr_upper
                else:
                    # Stay bullish
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = curr_lower
            else:  # Was bearish
                if close.iloc[i] > prev_supertrend:
                    # Flip to bullish
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = curr_lower
                else:
                    # Stay bearish
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = curr_upper

        return supertrend, direction
    
    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns to handle yfinance MultiIndex."""
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
        else:
            df.columns = df.columns.str.lower()
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = self.normalize_columns(df)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(
            df['close'], self.BB_PERIOD, self.BB_STD
        )
        
        # BB Width and percentile
        df['bb_width'] = self.calculate_bb_width(df['bb_upper'], df['bb_lower'], df['bb_middle'])
        df['bb_width_pctl'] = self.calculate_width_percentile(df['bb_width'], self.WIDTH_LOOKBACK)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=self.VOLUME_PERIOD).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ATR for stop loss
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        
        # Supertrend
        df['supertrend'], df['st_direction'] = self.calculate_supertrend(
            df['high'], df['low'], df['close'], self.ST_PERIOD, self.ST_MULTIPLIER
        )
        
        # Conditions
        df['squeeze'] = df['bb_width_pctl'] < self.SQUEEZE_PERCENTILE
        df['breakout'] = df['close'] > df['bb_upper']
        df['volume_surge'] = df['volume_ratio'] > self.VOLUME_MULTIPLIER
        df['st_bullish'] = df['st_direction'] == 1
        
        # Entry signal: Squeeze + Breakout + Volume + Supertrend bullish
        df['buy_signal'] = df['squeeze'].shift(1) & df['breakout'] & df['volume_surge'] & df['st_bullish']
        
        # Exit signals
        df['below_middle'] = df['close'] < df['bb_middle']
        df['st_bearish'] = df['st_direction'] == -1
        df['sell_signal'] = df['below_middle'] | df['st_bearish']
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Generate trading signal"""
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal_type = SignalType.HOLD
        confidence = 0.5
        reasoning = "No signal"
        stop_loss = None
        
        if self.current_position is None:
            # Check entry conditions
            was_squeeze = prev['bb_width_pctl'] < self.SQUEEZE_PERCENTILE
            is_breakout = latest['close'] > latest['bb_upper']
            has_volume = latest['volume_ratio'] > self.VOLUME_MULTIPLIER
            st_bullish = latest['st_direction'] == 1
            
            if was_squeeze and is_breakout and has_volume and st_bullish:
                # Check divergence filter
                divergence_filtered = False
                if self.use_divergence_filter and self.divergence_filter:
                    adj = self.divergence_filter.get_signal_adjustment(df)
                    if adj['filter_buy']:
                        divergence_filtered = True
                        reasoning = f"Entry filtered: {adj['reasoning']}"
                        logger.info(f"BollingerSqueeze BUY signal filtered by divergence")

                if not divergence_filtered:
                    signal_type = SignalType.BUY

                    # Calculate confidence
                    confidence = 0.6
                    if latest['volume_ratio'] > 2.0:
                        confidence += 0.1
                    if prev['bb_width_pctl'] < 0.10:  # Very tight squeeze
                        confidence += 0.1
                    confidence = min(confidence, 0.95)

                    # Apply divergence adjustment
                    if self.use_divergence_filter and self.divergence_filter:
                        adj = self.divergence_filter.get_signal_adjustment(df)
                        confidence = max(0.1, min(0.95, confidence + adj['confidence_adjustment']))

                    reasoning = (
                        f"Squeeze breakout: Width pctl={prev['bb_width_pctl']:.0%}, "
                        f"Close ${latest['close']:.2f} > BB_Upper ${latest['bb_upper']:.2f}, "
                        f"Volume {latest['volume_ratio']:.1f}x avg, Supertrend bullish"
                    )

                    # ATR-based stop loss
                    atr_stop = latest['close'] - (self.ATR_STOP_MULTIPLIER * latest['atr'])
                    pct_stop = latest['close'] * (1 - self.STOP_LOSS_PCT)
                    stop_loss = max(atr_stop, pct_stop)  # Use tighter stop
        
        else:
            # Check exit conditions
            should_exit, exit_reason = self.should_exit(latest)
            if should_exit:
                signal_type = SignalType.SELL
                confidence = 0.85
                reasoning = exit_reason
        
        return Signal(
            signal_type=signal_type,
            symbol=self.symbol,
            timestamp=datetime.now(timezone.utc),
            price=latest['close'],
            bb_upper=latest['bb_upper'],
            bb_middle=latest['bb_middle'],
            bb_lower=latest['bb_lower'],
            bb_width_percentile=latest['bb_width_pctl'],
            volume_ratio=latest['volume_ratio'],
            supertrend=latest['supertrend'],
            supertrend_direction='bullish' if latest['st_direction'] == 1 else 'bearish',
            confidence=confidence,
            reasoning=reasoning,
            stop_loss=stop_loss
        )
    
    def should_exit(self, row) -> Tuple[bool, str]:
        """Check exit conditions"""
        # Stop loss
        if self.stop_loss_price and row['close'] <= self.stop_loss_price:
            return True, f"Stop loss: ${row['close']:.2f} <= ${self.stop_loss_price:.2f}"
        
        # Below middle band
        if row['close'] < row['bb_middle']:
            return True, f"Price ${row['close']:.2f} < Middle Band ${row['bb_middle']:.2f}"
        
        # Supertrend bearish
        if row['st_direction'] == -1:
            return True, "Supertrend turned bearish"
        
        return False, ""
    
    def enter_position(self, price: float, atr: float = None) -> Dict:
        self.current_position = "long"
        self.entry_price = price
        
        if atr:
            self.stop_loss_price = price - (self.ATR_STOP_MULTIPLIER * atr)
        else:
            self.stop_loss_price = price * (1 - self.STOP_LOSS_PCT)
        
        logger.info(f"[ENTRY] {self.symbol} @ ${price:.2f}, Stop: ${self.stop_loss_price:.2f}")
        return {"action": "buy", "symbol": self.symbol, "price": price, "stop_loss": self.stop_loss_price}
    
    def exit_position(self, price: float, reason: str) -> Dict:
        if not self.entry_price:
            return {"action": "none"}
        
        pnl_pct = (price - self.entry_price) / self.entry_price
        logger.info(f"[EXIT] {self.symbol} @ ${price:.2f}, P&L: {pnl_pct:+.2%}")
        
        result = {
            "action": "sell",
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "exit_price": price,
            "pnl_pct": pnl_pct,
            "reason": reason
        }
        
        self.current_position = None
        self.entry_price = None
        self.stop_loss_price = None
        
        return result
    
    def get_status(self) -> Dict:
        return {
            "symbol": self.symbol,
            "position": self.current_position,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss_price,
            "parameters": {
                "bb_period": self.BB_PERIOD,
                "bb_std": self.BB_STD,
                "squeeze_percentile": self.SQUEEZE_PERCENTILE,
                "volume_multiplier": self.VOLUME_MULTIPLIER,
                "supertrend_period": self.ST_PERIOD,
                "supertrend_multiplier": self.ST_MULTIPLIER
            }
        }


def backtest_bollinger_squeeze(df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
    """Backtest Bollinger Squeeze strategy"""
    strategy = BollingerSqueezeStrategy()
    df = strategy.calculate_indicators(df.copy())
    
    capital = initial_capital
    position = None
    entry_price = 0
    stop_loss = 0
    trades = []
    
    start_idx = max(strategy.BB_PERIOD, strategy.WIDTH_LOOKBACK) + 1
    
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        if position is None:
            # Check entry
            was_squeeze = prev['bb_width_pctl'] < strategy.SQUEEZE_PERCENTILE
            is_breakout = row['close'] > row['bb_upper']
            has_volume = row['volume_ratio'] > strategy.VOLUME_MULTIPLIER
            st_bullish = row['st_direction'] == 1
            
            if was_squeeze and is_breakout and has_volume and st_bullish:
                position = "long"
                entry_price = row['close']
                entry_date = row.name
                stop_loss = entry_price - (strategy.ATR_STOP_MULTIPLIER * row['atr'])
        
        else:
            # Check exit
            stop_hit = row['close'] < stop_loss
            below_middle = row['close'] < row['bb_middle']
            st_bearish = row['st_direction'] == -1
            
            if stop_hit or below_middle or st_bearish:
                exit_price = row['close']
                pnl_pct = (exit_price - entry_price) / entry_price
                pnl = capital * 0.25 * pnl_pct
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': row.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl': pnl,
                    'reason': 'stop' if stop_hit else 'middle' if below_middle else 'supertrend'
                })
                
                capital += pnl
                position = None
    
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
            'trades': trades
        }
    
    return {'total_trades': 0, 'trades': []}


if __name__ == "__main__":
    import yfinance as yf
    
    print("=" * 60)
    print("BOLLINGER SQUEEZE STRATEGY")
    print("=" * 60)
    
    try:
        spy = yf.download("SPY", period="3y", progress=False)
        
        if len(spy) > 0:
            print(f"Loaded {len(spy)} days")
            
            results = backtest_bollinger_squeeze(spy)
            
            print(f"\nBacktest Results:")
            print(f"  Trades: {results['total_trades']}")
            print(f"  Win Rate: {results.get('win_rate', 0):.1%}")
            print(f"  Return: {results.get('total_return', 0):.1%}")
            
            strategy = BollingerSqueezeStrategy()
            signal = strategy.generate_signal(spy)
            
            print(f"\nCurrent Signal: {signal.signal_type.value}")
            print(f"  BB Width Percentile: {signal.bb_width_percentile:.0%}")
            print(f"  Volume Ratio: {signal.volume_ratio:.1f}x")
            print(f"  Supertrend: {signal.supertrend_direction}")
            print(f"  Reasoning: {signal.reasoning}")
            
    except Exception as e:
        print(f"Error: {e}")
