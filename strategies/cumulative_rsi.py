"""
Cumulative RSI Strategy

Enhanced RSI-2 variant that requires sustained oversold conditions.
Documented 30.3% annual return with Sharpe 1.18.

Strategy Logic:
- Entry: 2-day cumulative RSI(2) < 10 AND price > SMA(200)
- Exit: 2-day cumulative RSI(2) > 65 OR stop loss

Why this works:
- Requires multiple days of selling pressure (more reliable signal)
- Filters out single-day dips that quickly reverse
- Better entry timing than standard RSI-2

Source: Quantitativo Research

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
logger = logging.getLogger('CumulativeRSI')


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
    cumulative_rsi: float
    rsi2: float
    sma200: float
    confidence: float
    reasoning: str
    stop_loss: Optional[float] = None


class CumulativeRSIStrategy:
    """
    Cumulative RSI Mean Reversion Strategy
    
    Entry: Sum of RSI(2) over last N days < threshold
    Exit: Sum of RSI(2) over last N days > exit_threshold
    
    This filters for sustained oversold conditions,
    avoiding false signals from single-day dips.
    """
    
    # Parameters
    RSI_PERIOD = 2
    CUMULATIVE_DAYS = 2
    ENTRY_THRESHOLD = 10  # Cumulative RSI < 10
    EXIT_THRESHOLD = 65   # Cumulative RSI > 65
    TREND_SMA = 200
    STOP_LOSS_PCT = 0.03
    MAX_RISK_PER_TRADE = 0.02
    
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
            logger.info(f"CumulativeRSIStrategy initialized for {symbol} with divergence filter")
        else:
            self.divergence_filter = None
            logger.info(f"CumulativeRSIStrategy initialized for {symbol}")
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 2) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        # Protect against division by zero (all gains, no losses = RSI 100)
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_cumulative_rsi(rsi: pd.Series, days: int = 2) -> pd.Series:
        """Calculate cumulative RSI over N days"""
        return rsi.rolling(window=days).sum()
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        return prices.rolling(window=period).mean()
    
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
        df = self.normalize_columns(df)
        
        # RSI
        df['rsi2'] = self.calculate_rsi(df['close'], self.RSI_PERIOD)
        
        # Cumulative RSI
        df['cumulative_rsi'] = self.calculate_cumulative_rsi(df['rsi2'], self.CUMULATIVE_DAYS)
        
        # Trend filter
        df['sma200'] = self.calculate_sma(df['close'], self.TREND_SMA)
        
        # Conditions
        df['in_uptrend'] = df['close'] > df['sma200']
        df['oversold'] = df['cumulative_rsi'] < self.ENTRY_THRESHOLD
        df['exit_condition'] = df['cumulative_rsi'] > self.EXIT_THRESHOLD
        
        # Signals
        df['buy_signal'] = df['in_uptrend'] & df['oversold']
        df['sell_signal'] = df['exit_condition']
        
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        
        signal_type = SignalType.HOLD
        confidence = 0.5
        reasoning = "No signal"
        stop_loss = None
        
        if self.current_position is None:
            if latest['buy_signal'] and not pd.isna(latest['cumulative_rsi']):
                # Check divergence filter
                divergence_filtered = False
                if self.use_divergence_filter and self.divergence_filter:
                    adj = self.divergence_filter.get_signal_adjustment(df)
                    if adj['filter_buy']:
                        divergence_filtered = True
                        reasoning = f"Entry filtered: {adj['reasoning']}"
                        logger.info(f"CumulativeRSI BUY signal filtered by divergence")

                if not divergence_filtered:
                    signal_type = SignalType.BUY
                    # Clamp confidence between 0.1 and 0.95 to handle edge cases
                    raw_confidence = 0.5 + (self.ENTRY_THRESHOLD - latest['cumulative_rsi']) / 20
                    confidence = max(0.1, min(0.95, raw_confidence))

                    # Apply divergence adjustment
                    if self.use_divergence_filter and self.divergence_filter:
                        adj = self.divergence_filter.get_signal_adjustment(df)
                        confidence = max(0.1, min(0.95, confidence + adj['confidence_adjustment']))

                    reasoning = (
                        f"Cumulative RSI={latest['cumulative_rsi']:.1f} < {self.ENTRY_THRESHOLD}, "
                        f"Price ${latest['close']:.2f} > SMA200 ${latest['sma200']:.2f}"
                    )
                    stop_loss = latest['close'] * (1 - self.STOP_LOSS_PCT)
        else:
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
            cumulative_rsi=latest['cumulative_rsi'],
            rsi2=latest['rsi2'],
            sma200=latest['sma200'],
            confidence=confidence,
            reasoning=reasoning,
            stop_loss=stop_loss
        )
    
    def should_exit(self, row) -> Tuple[bool, str]:
        # Stop loss
        if self.stop_loss_price and row['close'] <= self.stop_loss_price:
            return True, f"Stop loss: ${row['close']:.2f} <= ${self.stop_loss_price:.2f}"
        
        # Cumulative RSI exit
        if row['cumulative_rsi'] > self.EXIT_THRESHOLD:
            return True, f"Cumulative RSI={row['cumulative_rsi']:.1f} > {self.EXIT_THRESHOLD}"
        
        return False, ""
    
    def enter_position(self, price: float) -> Dict:
        self.current_position = "long"
        self.entry_price = price
        self.stop_loss_price = price * (1 - self.STOP_LOSS_PCT)
        
        logger.info(f"[ENTRY] {self.symbol} @ ${price:.2f}")
        return {"action": "buy", "symbol": self.symbol, "price": price}
    
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
            "parameters": {
                "cumulative_days": self.CUMULATIVE_DAYS,
                "entry_threshold": self.ENTRY_THRESHOLD,
                "exit_threshold": self.EXIT_THRESHOLD
            }
        }


def backtest_cumulative_rsi(df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
    """Backtest cumulative RSI strategy"""
    strategy = CumulativeRSIStrategy()
    df = strategy.calculate_indicators(df.copy())
    
    capital = initial_capital
    position = None
    entry_price = 0
    trades = []
    
    for i in range(strategy.TREND_SMA + strategy.CUMULATIVE_DAYS, len(df)):
        row = df.iloc[i]
        
        if position is None:
            if row['buy_signal'] and not pd.isna(row['cumulative_rsi']):
                position = "long"
                entry_price = row['close']
                entry_date = row.name
                stop_loss = entry_price * (1 - strategy.STOP_LOSS_PCT)
        else:
            stop_hit = row['close'] < stop_loss
            if row['sell_signal'] or stop_hit:
                exit_price = row['close']
                pnl_pct = (exit_price - entry_price) / entry_price
                pnl = capital * 0.33 * pnl_pct
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': row.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl': pnl
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
    print("CUMULATIVE RSI STRATEGY")
    print("=" * 60)
    
    try:
        spy = yf.download("SPY", period="3y", progress=False)
        
        if len(spy) > 0:
            print(f"Loaded {len(spy)} days")
            
            results = backtest_cumulative_rsi(spy)
            
            print(f"\nBacktest Results:")
            print(f"  Trades: {results['total_trades']}")
            print(f"  Win Rate: {results.get('win_rate', 0):.1%}")
            print(f"  Return: {results.get('total_return', 0):.1%}")
            print(f"  Avg Win: {results.get('avg_win', 0):.2%}")
            print(f"  Avg Loss: {results.get('avg_loss', 0):.2%}")
            
            # Current signal
            strategy = CumulativeRSIStrategy()
            signal = strategy.generate_signal(spy)
            
            print(f"\nCurrent Signal: {signal.signal_type.value}")
            print(f"  Cumulative RSI: {signal.cumulative_rsi:.1f}")
            print(f"  RSI(2): {signal.rsi2:.1f}")
            print(f"  Reasoning: {signal.reasoning}")
            
    except Exception as e:
        print(f"Error: {e}")
