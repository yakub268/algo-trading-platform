"""
RSI-2 Mean Reversion Strategy

Classic Larry Connors RSI-2 mean reversion strategy with documented 91% win rate.

Strategy Logic:
- Entry: RSI(2) < 10 AND price > 200 SMA (oversold in uptrend)
- Exit: RSI(2) > 90 OR close > 5-day SMA
- Stop Loss: 3% trailing stop

This is a counter-trend strategy that buys short-term pullbacks
in long-term uptrends.

Author: Trading Bot
Created: January 2026
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import volume-price divergence filter
try:
    from filters.volume_price_divergence import VolumePriceDivergenceFilter
    DIVERGENCE_FILTER_AVAILABLE = True
except ImportError:
    DIVERGENCE_FILTER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RSI2Strategy')


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    """Trading signal with all relevant data"""
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    rsi2: float
    sma200: float
    sma5: float
    confidence: float  # 0-1 confidence score
    reasoning: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Indicators:
    """Technical indicators container"""
    rsi2: float
    sma5: float
    sma200: float
    close: float
    high: float
    low: float
    volume: float
    atr: float
    timestamp: datetime


class RSI2MeanReversion:
    """
    RSI-2 Mean Reversion Strategy

    Entry Conditions:
    1. RSI(2) < 10 (extremely oversold)
    2. Price > 200 SMA (long-term uptrend)

    Exit Conditions:
    1. RSI(2) > 90 (overbought - take profit)
    2. Close > 5-day SMA (price recovered)
    3. Stop loss hit (3% trailing)

    Risk Management:
    - 3% initial stop loss
    - Trailing stop once profitable
    - Max 2% account risk per trade
    """

    # ENHANCED Strategy parameters (based on 34-year S&P 500 research)
    RSI_PERIOD = 2
    RSI_OVERSOLD = 10          # Widened from 5 to 10 for more signals
    RSI_OVERSOLD_STREAK = 10   # Threshold for consecutive bar counting
    RSI_OVERBOUGHT = 90
    SMA_TREND = 200
    SMA_EXIT = 5
    ATR_PERIOD = 14
    ATR_STOP_MULTIPLIER = 2.0  # ATR-based stops instead of fixed %
    CONSECUTIVE_BARS = 2       # Reduced from 3 to 2 for faster entries
    VOLUME_THRESHOLD = 1.2     # Volume > 1.2x average for confirmation
    STOP_LOSS_PCT = 0.03       # Fallback 3% stop loss
    MAX_RISK_PER_TRADE = 0.02  # 2% account risk

    def __init__(
        self,
        symbol: str = "SPY",
        paper_mode: bool = None,
        use_divergence_filter: bool = True
    ):
        """
        Initialize RSI-2 strategy.

        Args:
            symbol: Trading symbol (default SPY)
            paper_mode: Paper trading mode (reads from PAPER_MODE env if None)
            use_divergence_filter: Enable volume-price divergence filter
        """
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
            logger.info(f"RSI2MeanReversion initialized for {symbol} with divergence filter (paper={paper_mode})")
        else:
            self.divergence_filter = None
            logger.info(f"RSI2MeanReversion initialized for {symbol} (paper={paper_mode})")

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 2) -> pd.Series:
        """
        Calculate RSI indicator.

        Args:
            prices: Price series
            period: RSI period (default 2)

        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        # Protect against division by zero (all gains, no losses = RSI 100)
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame columns to handle yfinance MultiIndex.

        yfinance now returns MultiIndex columns like ('Close', 'SPY').
        This converts them to simple lowercase column names.
        """
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            # Extract the first level (e.g., 'Close' from ('Close', 'SPY'))
            df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
        else:
            df.columns = df.columns.str.lower()
        return df

    def count_consecutive_oversold(self, rsi_series: pd.Series, threshold: float = 10) -> pd.Series:
        """
        Count consecutive bars where RSI is below threshold.
        
        Args:
            rsi_series: RSI values
            threshold: RSI threshold (default 10)
            
        Returns:
            Series with consecutive count at each bar
        """
        below_threshold = (rsi_series < threshold).astype(int)
        
        # Count consecutive True values
        groups = (below_threshold != below_threshold.shift()).cumsum()
        consecutive = below_threshold.groupby(groups).cumsum()
        
        return consecutive
    
    def calculate_atr_stop(self, entry_price: float, atr: float, side: str = 'long') -> float:
        """
        Calculate stop loss based on ATR.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            side: Position side ('long' or 'short')
            
        Returns:
            Stop loss price
        """
        if side == 'long':
            return entry_price - (atr * self.ATR_STOP_MULTIPLIER)
        return entry_price + (atr * self.ATR_STOP_MULTIPLIER)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all required indicators with ENHANCED signals.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with added indicator columns
        """
        # Normalize column names (handles yfinance MultiIndex)
        df = self.normalize_columns(df)

        # Calculate indicators
        df['rsi2'] = self.calculate_rsi(df['close'], self.RSI_PERIOD)
        df['sma5'] = self.calculate_sma(df['close'], self.SMA_EXIT)
        df['sma200'] = self.calculate_sma(df['close'], self.SMA_TREND)
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], self.ATR_PERIOD)
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # ENHANCED: Count consecutive oversold bars
        df['consecutive_oversold'] = self.count_consecutive_oversold(df['rsi2'], self.RSI_OVERSOLD_STREAK)

        # Signal conditions
        df['in_uptrend'] = df['close'] > df['sma200']
        df['extreme_oversold'] = df['rsi2'] < self.RSI_OVERSOLD  # RSI < 5 (extreme)
        df['streak_oversold'] = df['consecutive_oversold'] >= self.CONSECUTIVE_BARS  # 3+ bars
        df['volume_confirmed'] = df['volume_ratio'] >= self.VOLUME_THRESHOLD
        df['overbought'] = df['rsi2'] > self.RSI_OVERBOUGHT
        df['above_sma5'] = df['close'] > df['sma5']

        # ENHANCED Entry signal: ALL conditions must be met
        # 1. RSI(2) < 5 (extreme oversold)
        # 2. 3+ consecutive bars with RSI < 10
        # 3. Price > 200 SMA (uptrend)
        # 4. Volume > 1.2x average (optional but increases confidence)
        df['buy_signal'] = (
            df['in_uptrend'] & 
            df['extreme_oversold'] & 
            df['streak_oversold']
        )
        
        # Also track signals with volume confirmation (higher confidence)
        df['buy_signal_confirmed'] = df['buy_signal'] & df['volume_confirmed']

        # Exit signals: overbought or above 5-day SMA
        df['sell_signal'] = df['overbought'] | df['above_sma5']

        return df

    def get_latest_indicators(self, df: pd.DataFrame) -> Indicators:
        """
        Get the latest indicator values.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Indicators dataclass with latest values
        """
        latest = df.iloc[-1]

        return Indicators(
            rsi2=latest['rsi2'],
            sma5=latest['sma5'],
            sma200=latest['sma200'],
            close=latest['close'],
            high=latest['high'],
            low=latest['low'],
            volume=latest.get('volume', 0),
            atr=latest['atr'],
            timestamp=latest.name if isinstance(latest.name, datetime) else datetime.now(timezone.utc)
        )

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Generate trading signal from price data using ENHANCED logic.

        ENHANCED Entry Conditions (ALL required):
        1. RSI(2) < 5 (extreme oversold)
        2. 3+ consecutive bars with RSI(2) < 10
        3. Price > 200 SMA (uptrend filter)
        4. Volume > 1.2x 20-day average (optional, boosts confidence)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Signal with entry/exit decision
        """
        # Calculate indicators
        df = self.calculate_indicators(df)
        indicators = self.get_latest_indicators(df)
        
        # Get consecutive oversold count and volume ratio from latest row
        latest = df.iloc[-1]
        consecutive_oversold = int(latest.get('consecutive_oversold', 0))
        volume_ratio = float(latest.get('volume_ratio', 1.0))
        volume_confirmed = volume_ratio >= self.VOLUME_THRESHOLD

        # Default to hold
        signal_type = SignalType.HOLD
        confidence = 0.5
        reasoning = "No signal conditions met"
        stop_loss = None
        take_profit = None

        # Check for ENHANCED entry signal
        if self.current_position is None:
            # Check all conditions
            extreme_oversold = indicators.rsi2 < self.RSI_OVERSOLD  # RSI < 5
            streak_met = consecutive_oversold >= self.CONSECUTIVE_BARS  # 3+ bars
            in_uptrend = indicators.close > indicators.sma200
            
            if extreme_oversold and streak_met and in_uptrend:
                # Check divergence filter before generating BUY signal
                divergence_filtered = False
                divergence_info = ""
                if self.use_divergence_filter and self.divergence_filter:
                    adj = self.divergence_filter.get_signal_adjustment(df)
                    if adj['filter_buy']:
                        divergence_filtered = True
                        divergence_info = f" [FILTERED: {adj['reasoning']}]"
                        logger.info(f"RSI2 BUY signal filtered by divergence: {adj['reasoning']}")

                if not divergence_filtered:
                    signal_type = SignalType.BUY
                    confidence = self._calculate_entry_confidence(indicators, consecutive_oversold, volume_ratio)

                    # Apply divergence adjustment to confidence
                    if self.use_divergence_filter and self.divergence_filter:
                        adj = self.divergence_filter.get_signal_adjustment(df)
                        confidence = max(0.1, min(0.95, confidence + adj['confidence_adjustment']))

                    # Use ATR-based stop loss instead of fixed percentage
                    stop_loss = self.calculate_atr_stop(indicators.close, indicators.atr, 'long')

                    # Build reasoning with all factors
                    vol_str = f", Volume {volume_ratio:.1f}x avg" if volume_confirmed else ""
                    reasoning = (
                        f"ENHANCED SIGNAL: RSI(2)={indicators.rsi2:.1f} < {self.RSI_OVERSOLD} (extreme), "
                        f"{consecutive_oversold} consecutive bars < 10, "
                        f"Price ${indicators.close:.2f} > SMA200 ${indicators.sma200:.2f}{vol_str}, "
                        f"ATR Stop: ${stop_loss:.2f} ({self.ATR_STOP_MULTIPLIER}x ATR)"
                    )
                    take_profit = None  # Use indicator-based exit
                else:
                    reasoning = f"Entry conditions met but filtered{divergence_info}"
            
            # Log near-misses for debugging
            elif extreme_oversold or streak_met:
                missing = []
                if not extreme_oversold:
                    missing.append(f"RSI={indicators.rsi2:.1f} (need <{self.RSI_OVERSOLD})")
                if not streak_met:
                    missing.append(f"Streak={consecutive_oversold} (need >={self.CONSECUTIVE_BARS})")
                if not in_uptrend:
                    missing.append("Below SMA200")
                logger.debug(f"Near-miss signal for {self.symbol}: missing {', '.join(missing)}")

        # Check for exit signal
        else:
            should_exit, exit_reason = self.should_exit(indicators)
            if should_exit:
                signal_type = SignalType.SELL
                confidence = 0.9  # High confidence on exits
                reasoning = exit_reason

        return Signal(
            signal_type=signal_type,
            symbol=self.symbol,
            timestamp=indicators.timestamp,
            price=indicators.close,
            rsi2=indicators.rsi2,
            sma200=indicators.sma200,
            sma5=indicators.sma5,
            confidence=confidence,
            reasoning=reasoning,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def _calculate_entry_confidence(self, indicators: Indicators, consecutive_bars: int = 0, volume_ratio: float = 1.0) -> float:
        """
        Calculate confidence score for ENHANCED entry signal.

        Factors:
        - How oversold (lower RSI = higher confidence)
        - Consecutive oversold bars (more = higher confidence)
        - Volume confirmation (higher = more confident)
        - Trend strength (distance above SMA200)

        Returns:
            Confidence score 0-1
        """
        confidence = 0.5

        # RSI factor: lower = better (RSI of 2 = max boost)
        # RSI < 5 required for entry, but RSI < 3 is even better
        if indicators.rsi2 < 3:
            confidence += 0.20
        elif indicators.rsi2 < 5:
            confidence += 0.15
        elif indicators.rsi2 < 7:
            confidence += 0.10
        
        # Consecutive bars factor: more = better
        # 3 bars required, but 4-5 is even better
        if consecutive_bars >= 5:
            confidence += 0.15
        elif consecutive_bars >= 4:
            confidence += 0.10
        elif consecutive_bars >= 3:
            confidence += 0.05
        
        # Volume factor: higher = better
        if volume_ratio >= 2.0:
            confidence += 0.15  # Very high volume
        elif volume_ratio >= 1.5:
            confidence += 0.10  # High volume
        elif volume_ratio >= 1.2:
            confidence += 0.05  # Above average

        # Trend factor: further above SMA200 = better (but not too far)
        trend_pct = (indicators.close - indicators.sma200) / indicators.sma200
        if 0 < trend_pct < 0.10:  # 0-10% above SMA200 (healthy uptrend)
            confidence += 0.10
        elif 0.10 <= trend_pct < 0.20:  # 10-20% above (strong uptrend)
            confidence += 0.05
        # >20% above might be overextended, no bonus

        # Cap at 0.95
        return min(confidence, 0.95)

    def should_exit(self, indicators: Indicators) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Exit conditions:
        1. RSI(2) > 90 (overbought)
        2. Close > 5-day SMA
        3. Stop loss hit

        Args:
            indicators: Current indicator values

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        # Check stop loss
        if self.entry_price and self.stop_loss_price:
            if indicators.close <= self.stop_loss_price:
                return True, f"Stop loss hit: ${indicators.close:.2f} <= ${self.stop_loss_price:.2f}"

        # Check RSI overbought
        if indicators.rsi2 > self.RSI_OVERBOUGHT:
            return True, f"RSI(2)={indicators.rsi2:.1f} > {self.RSI_OVERBOUGHT} (overbought)"

        # Check price above 5-day SMA
        if indicators.close > indicators.sma5:
            return True, f"Price ${indicators.close:.2f} > SMA5 ${indicators.sma5:.2f} (price recovered)"

        return False, ""

    def enter_position(self, price: float, atr: float = None) -> Dict:
        """
        Record position entry.

        Args:
            price: Entry price
            atr: Current ATR value for dynamic stop calculation

        Returns:
            Position details
        """
        self.current_position = "long"
        self.entry_price = price

        # Use ATR-based stop if ATR provided, otherwise fall back to fixed percentage
        if atr and atr > 0:
            self.stop_loss_price = self.calculate_atr_stop(price, atr, 'long')
        else:
            self.stop_loss_price = price * (1 - self.STOP_LOSS_PCT)

        logger.info(f"[ENTRY] {self.symbol} @ ${price:.2f}, Stop: ${self.stop_loss_price:.2f}")

        return {
            "action": "buy",
            "symbol": self.symbol,
            "price": price,
            "stop_loss": self.stop_loss_price,
            "paper_mode": self.paper_mode
        }

    def exit_position(self, price: float, reason: str) -> Dict:
        """
        Record position exit.

        Args:
            price: Exit price
            reason: Exit reason

        Returns:
            Position details with P&L
        """
        if not self.entry_price:
            return {"action": "none", "reason": "No position to exit"}

        pnl_pct = (price - self.entry_price) / self.entry_price
        pnl_pct_str = f"{pnl_pct:+.2%}"

        logger.info(f"[EXIT] {self.symbol} @ ${price:.2f}, P&L: {pnl_pct_str}, Reason: {reason}")

        result = {
            "action": "sell",
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "exit_price": price,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "paper_mode": self.paper_mode
        }

        # Reset position state
        self.current_position = None
        self.entry_price = None
        self.stop_loss_price = None

        return result

    def update_trailing_stop(self, current_price: float) -> None:
        """
        Update trailing stop if price has moved favorably.

        Args:
            current_price: Current market price
        """
        if not self.entry_price or not self.stop_loss_price:
            return

        # Only trail if in profit
        if current_price > self.entry_price:
            new_stop = current_price * (1 - self.STOP_LOSS_PCT)

            if new_stop > self.stop_loss_price:
                old_stop = self.stop_loss_price
                self.stop_loss_price = new_stop
                logger.debug(f"Trailing stop updated: ${old_stop:.2f} -> ${new_stop:.2f}")

    def calculate_position_size(
        self,
        account_balance: float,
        current_price: float
    ) -> float:
        """
        Calculate position size based on risk management.

        Uses 2% account risk with 3% stop loss.
        Position = (Account * Risk%) / Stop Loss%

        Args:
            account_balance: Total account value
            current_price: Current price

        Returns:
            Dollar amount to invest
        """
        risk_amount = account_balance * self.MAX_RISK_PER_TRADE
        position_size = risk_amount / self.STOP_LOSS_PCT

        # Cap at 50% of account
        max_position = account_balance * 0.5

        return min(position_size, max_position)

    def get_status(self) -> Dict:
        """Get current strategy status"""
        return {
            "symbol": self.symbol,
            "position": self.current_position,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss_price,
            "paper_mode": self.paper_mode,
            "parameters": {
                "rsi_period": self.RSI_PERIOD,
                "rsi_oversold": self.RSI_OVERSOLD,
                "rsi_overbought": self.RSI_OVERBOUGHT,
                "sma_trend": self.SMA_TREND,
                "sma_exit": self.SMA_EXIT,
                "stop_loss_pct": self.STOP_LOSS_PCT
            }
        }


def backtest_rsi2(
    df: pd.DataFrame,
    initial_capital: float = 10000.0
) -> Dict:
    """
    Simple backtest of RSI-2 strategy.

    Args:
        df: DataFrame with OHLCV data
        initial_capital: Starting capital

    Returns:
        Backtest results
    """
    strategy = RSI2MeanReversion(paper_mode=True)
    df = strategy.calculate_indicators(df.copy())

    capital = initial_capital
    position = None
    entry_price = 0
    trades = []

    for i in range(200, len(df)):  # Start after SMA200 has values
        row = df.iloc[i]

        if position is None:
            # Check for entry
            if row['buy_signal'] and not pd.isna(row['rsi2']):
                position = "long"
                entry_price = row['close']
                entry_date = row.name

        else:
            # Check for exit
            if row['sell_signal'] or row['close'] < entry_price * (1 - strategy.STOP_LOSS_PCT):
                exit_price = row['close']
                pnl_pct = (exit_price - entry_price) / entry_price
                pnl = capital * 0.33 * pnl_pct  # 33% position size

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

    # Calculate statistics
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
            'largest_win': max([t['pnl_pct'] for t in trades]),
            'largest_loss': min([t['pnl_pct'] for t in trades]),
            'trades': trades
        }

    return {'total_trades': 0, 'trades': []}


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    import yfinance as yf

    print("=" * 60)
    print("RSI-2 MEAN REVERSION STRATEGY TEST")
    print("=" * 60)

    # Download SPY data
    try:
        spy = yf.download("SPY", period="2y", interval="1d", progress=False)

        if len(spy) > 0:
            print(f"\nLoaded {len(spy)} days of SPY data")

            # Run backtest
            results = backtest_rsi2(spy)

            print(f"\nBacktest Results:")
            print(f"  Total Trades: {results['total_trades']}")
            print(f"  Win Rate: {results.get('win_rate', 0):.1%}")
            print(f"  Total Return: {results.get('total_return', 0):.1%}")
            print(f"  Avg Win: {results.get('avg_win', 0):.2%}")
            print(f"  Avg Loss: {results.get('avg_loss', 0):.2%}")

            # Test current signal
            strategy = RSI2MeanReversion(symbol="SPY")
            signal = strategy.generate_signal(spy)

            print(f"\nCurrent Signal:")
            print(f"  Type: {signal.signal_type.value}")
            print(f"  Price: ${signal.price:.2f}")
            print(f"  RSI(2): {signal.rsi2:.1f}")
            print(f"  SMA200: ${signal.sma200:.2f}")
            print(f"  SMA5: ${signal.sma5:.2f}")
            print(f"  Confidence: {signal.confidence:.0%}")
            print(f"  Reasoning: {signal.reasoning}")

    except Exception as e:
        print(f"Error: {e}")
        print("Install yfinance: pip install yfinance")

    print("\n" + "=" * 60)
