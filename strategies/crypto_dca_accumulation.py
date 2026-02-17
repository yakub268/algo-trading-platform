"""
Crypto DCA Accumulation Strategy

Dollar Cost Averaging strategy for accumulating XRP, HBAR, and XLM at key support levels.

Strategy Logic:
- Entry: Price enters defined support/entry zones
- DCA: Increase buy size as price drops into stronger support
- Exit: Price hits resistance levels or stop loss triggered
- Stop Loss: Below critical support levels

This is an accumulation strategy that buys more at lower prices
within defined support zones.

Author: Trading Bot
Created: February 2026
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CryptoDCAStrategy')


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
    rsi: float
    entry_zone: str
    confidence: float  # 0-1 confidence score
    reasoning: str
    dca_multiplier: float = 1.0  # How much to scale position
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Indicators:
    """Technical indicators container"""
    rsi: float
    sma20: float
    sma50: float
    bollinger_upper: float
    bollinger_lower: float
    close: float
    high: float
    low: float
    volume: float
    atr: float
    timestamp: datetime


@dataclass
class SupportResistance:
    """Support and resistance levels for a token"""
    critical_support: float
    strong_support: List[float]
    resistance_levels: List[float]
    aggressive_entry_low: float
    aggressive_entry_high: float
    conservative_entry_low: Optional[float]
    conservative_entry_high: Optional[float]
    stop_loss: float


@dataclass
class Position:
    """Track position with average cost basis"""
    symbol: str
    total_quantity: float = 0.0
    total_cost: float = 0.0
    average_cost: float = 0.0
    entry_count: int = 0
    last_entry_price: float = 0.0
    last_entry_time: Optional[datetime] = None

    def add_position(self, quantity: float, price: float, timestamp: datetime = None) -> None:
        """Add to position and recalculate average cost"""
        cost = quantity * price
        self.total_quantity += quantity
        self.total_cost += cost
        self.average_cost = self.total_cost / self.total_quantity if self.total_quantity > 0 else 0
        self.entry_count += 1
        self.last_entry_price = price
        self.last_entry_time = timestamp or datetime.now(timezone.utc)

    def close_position(self) -> Dict:
        """Close position and return stats"""
        stats = {
            'total_quantity': self.total_quantity,
            'total_cost': self.total_cost,
            'average_cost': self.average_cost,
            'entry_count': self.entry_count
        }
        self.total_quantity = 0.0
        self.total_cost = 0.0
        self.average_cost = 0.0
        self.entry_count = 0
        self.last_entry_price = 0.0
        self.last_entry_time = None
        return stats


# Token configurations with support/resistance levels
TOKEN_CONFIGS: Dict[str, SupportResistance] = {
    'XRP': SupportResistance(
        critical_support=1.50,
        strong_support=[1.69, 1.70, 1.71],
        resistance_levels=[1.93, 2.00, 2.22],
        aggressive_entry_low=1.71,
        aggressive_entry_high=1.90,
        conservative_entry_low=2.02,
        conservative_entry_high=2.10,
        stop_loss=1.50
    ),
    'HBAR': SupportResistance(
        critical_support=0.085,  # Stop loss level
        strong_support=[0.09, 0.10, 0.11],  # EMA support
        resistance_levels=[0.13, 0.15, 0.19],
        aggressive_entry_low=0.085,  # Buy from stop loss up
        aggressive_entry_high=0.12,
        conservative_entry_low=None,
        conservative_entry_high=None,
        stop_loss=0.085
    ),
    'XLM': SupportResistance(
        critical_support=0.17,  # Stop loss level
        strong_support=[0.18, 0.19, 0.20],  # Bollinger support at 0.20
        resistance_levels=[0.24, 0.26, 0.27],
        aggressive_entry_low=0.17,  # Buy from stop loss up
        aggressive_entry_high=0.21,
        conservative_entry_low=0.22,
        conservative_entry_high=0.23,
        stop_loss=0.17
    )
}


class CryptoDCAStrategy:
    """
    Crypto DCA Accumulation Strategy

    Accumulates XRP, HBAR, and XLM at key support levels using
    dollar cost averaging with increased position sizes at lower prices.

    Entry Conditions:
    1. Price enters defined entry zone (aggressive or conservative)
    2. RSI confirms oversold or neutral conditions
    3. Price above critical support

    Exit Conditions:
    1. Price hits resistance levels (take profit)
    2. Stop loss triggered (below critical support)
    3. RSI extremely overbought

    DCA Logic:
    - Base position at top of entry zone
    - 1.5x position at middle of zone
    - 2x position near critical support
    - 2.5x position at critical support (max conviction)
    """

    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    RSI_EXTREME_OVERBOUGHT = 85
    SMA_SHORT = 20
    SMA_LONG = 50
    ATR_PERIOD = 14
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2.0

    # DCA multipliers based on price level within entry zone
    DCA_LEVELS = {
        'zone_top': 1.0,        # Standard position at top of zone
        'zone_middle': 1.5,     # 1.5x at middle
        'zone_bottom': 2.0,     # 2x near bottom
        'critical_support': 2.5  # 2.5x at critical support
    }

    def __init__(
        self,
        symbol: str = "XRP",
        paper_mode: bool = None,
        base_position_pct: float = 0.05  # 5% of capital per base buy
    ):
        """
        Initialize Crypto DCA strategy.

        Args:
            symbol: Trading symbol (XRP, HBAR, or XLM)
            paper_mode: Paper trading mode (reads from PAPER_MODE env if None)
            base_position_pct: Base position size as % of capital
        """
        self.symbol = symbol.upper()

        # Validate symbol
        if self.symbol not in TOKEN_CONFIGS:
            raise ValueError(f"Unsupported symbol: {symbol}. Supported: {list(TOKEN_CONFIGS.keys())}")

        # Get token configuration
        self.config = TOKEN_CONFIGS[self.symbol]

        # Safe default: read from environment, default to PAPER if not set
        if paper_mode is None:
            paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
        self.paper_mode = paper_mode

        self.base_position_pct = base_position_pct

        # Position tracking
        self.position = Position(symbol=self.symbol)
        self.stop_loss_price = self.config.stop_loss

        logger.info(
            f"CryptoDCAStrategy initialized for {self.symbol} "
            f"(paper={paper_mode}, stop_loss=${self.stop_loss_price})"
        )

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI indicator.

        Args:
            prices: Price series
            period: RSI period (default 14)

        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        # Protect against division by zero
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Returns:
            Tuple of (middle, upper, lower) bands
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return middle, upper, lower

    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
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
        Normalize DataFrame columns to handle various data sources.

        Handles yfinance MultiIndex and other column name formats.
        """
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
        else:
            df.columns = df.columns.str.lower()
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all required indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        df = self.normalize_columns(df)

        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.RSI_PERIOD)

        # Moving Averages
        df['sma20'] = self.calculate_sma(df['close'], self.SMA_SHORT)
        df['sma50'] = self.calculate_sma(df['close'], self.SMA_LONG)

        # Bollinger Bands
        df['bb_middle'], df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(
            df['close'], self.BOLLINGER_PERIOD, self.BOLLINGER_STD
        )

        # ATR
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], self.ATR_PERIOD)

        # Support/Resistance proximity
        df['at_critical_support'] = df['close'] <= self.config.critical_support * 1.02
        df['near_strong_support'] = df['close'].apply(
            lambda x: any(abs(x - s) / s < 0.02 for s in self.config.strong_support)
        )
        df['at_resistance'] = df['close'].apply(
            lambda x: any(abs(x - r) / r < 0.02 for r in self.config.resistance_levels)
        )

        # Entry zone detection
        df['in_aggressive_zone'] = (
            (df['close'] >= self.config.aggressive_entry_low) &
            (df['close'] <= self.config.aggressive_entry_high)
        )

        if self.config.conservative_entry_low and self.config.conservative_entry_high:
            df['in_conservative_zone'] = (
                (df['close'] >= self.config.conservative_entry_low) &
                (df['close'] <= self.config.conservative_entry_high)
            )
        else:
            df['in_conservative_zone'] = False

        # Stop loss trigger
        df['stop_loss_triggered'] = df['close'] < self.config.stop_loss

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
            rsi=latest['rsi'],
            sma20=latest['sma20'],
            sma50=latest['sma50'],
            bollinger_upper=latest['bb_upper'],
            bollinger_lower=latest['bb_lower'],
            close=latest['close'],
            high=latest['high'],
            low=latest['low'],
            volume=latest.get('volume', 0),
            atr=latest['atr'],
            timestamp=latest.name if isinstance(latest.name, datetime) else datetime.now(timezone.utc)
        )

    def detect_support_resistance_level(self, price: float) -> str:
        """
        Detect which support/resistance level price is near.

        Args:
            price: Current price

        Returns:
            Level description string
        """
        config = self.config
        tolerance = 0.02  # 2% tolerance

        # Check critical support
        if abs(price - config.critical_support) / config.critical_support < tolerance:
            return f"critical_support (${config.critical_support})"

        # Check strong support levels
        for level in config.strong_support:
            if abs(price - level) / level < tolerance:
                return f"strong_support (${level})"

        # Check resistance levels
        for level in config.resistance_levels:
            if abs(price - level) / level < tolerance:
                return f"resistance (${level})"

        # Check entry zones
        if config.aggressive_entry_low <= price <= config.aggressive_entry_high:
            return f"aggressive_entry_zone (${config.aggressive_entry_low}-${config.aggressive_entry_high})"

        if config.conservative_entry_low and config.conservative_entry_high:
            if config.conservative_entry_low <= price <= config.conservative_entry_high:
                return f"conservative_entry_zone (${config.conservative_entry_low}-${config.conservative_entry_high})"

        return "neutral"

    def calculate_dca_amount(
        self,
        base_amount: float,
        current_price: float
    ) -> Tuple[float, float, str]:
        """
        Calculate DCA buy amount based on price level within entry zone.

        Lower prices within the zone = larger position sizes.

        Args:
            base_amount: Base position size in dollars
            current_price: Current market price

        Returns:
            Tuple of (adjusted_amount, multiplier, zone_level)
        """
        config = self.config

        # Check if at critical support - maximum conviction
        if current_price <= config.critical_support * 1.02:
            multiplier = self.DCA_LEVELS['critical_support']
            return base_amount * multiplier, multiplier, 'critical_support'

        # Calculate position within aggressive entry zone
        if config.aggressive_entry_low <= current_price <= config.aggressive_entry_high:
            zone_range = config.aggressive_entry_high - config.aggressive_entry_low
            price_in_zone = current_price - config.aggressive_entry_low
            zone_position = price_in_zone / zone_range  # 0 = bottom, 1 = top

            if zone_position >= 0.66:
                multiplier = self.DCA_LEVELS['zone_top']
                level = 'zone_top'
            elif zone_position >= 0.33:
                multiplier = self.DCA_LEVELS['zone_middle']
                level = 'zone_middle'
            else:
                multiplier = self.DCA_LEVELS['zone_bottom']
                level = 'zone_bottom'

            return base_amount * multiplier, multiplier, level

        # Conservative zone gets base multiplier
        if config.conservative_entry_low and config.conservative_entry_high:
            if config.conservative_entry_low <= current_price <= config.conservative_entry_high:
                multiplier = self.DCA_LEVELS['zone_top']
                return base_amount * multiplier, multiplier, 'conservative_zone'

        # Outside entry zones - no position
        return 0, 0, 'outside_zone'

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Generate trading signal from price data.

        Entry Conditions:
        1. Price in entry zone (aggressive or conservative)
        2. Price above stop loss
        3. RSI not extremely overbought

        Exit Conditions:
        1. Price at resistance
        2. RSI extremely overbought
        3. Stop loss triggered

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Signal with entry/exit decision
        """
        # Calculate indicators
        df = self.calculate_indicators(df)
        indicators = self.get_latest_indicators(df)
        latest = df.iloc[-1]

        # Default to hold
        signal_type = SignalType.HOLD
        confidence = 0.5
        reasoning = "No signal conditions met"
        entry_zone = self.detect_support_resistance_level(indicators.close)
        dca_multiplier = 1.0
        stop_loss = self.config.stop_loss
        take_profit = None

        # Check for stop loss first (highest priority)
        if latest['stop_loss_triggered']:
            signal_type = SignalType.SELL
            confidence = 1.0
            reasoning = f"STOP LOSS TRIGGERED: Price ${indicators.close:.4f} < stop ${self.config.stop_loss}"

            return Signal(
                signal_type=signal_type,
                symbol=self.symbol,
                timestamp=indicators.timestamp,
                price=indicators.close,
                rsi=indicators.rsi,
                entry_zone=entry_zone,
                confidence=confidence,
                reasoning=reasoning,
                dca_multiplier=0,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

        # Check for exit at resistance
        if self.position.total_quantity > 0:
            if latest['at_resistance'] or indicators.rsi > self.RSI_EXTREME_OVERBOUGHT:
                signal_type = SignalType.SELL
                confidence = 0.85
                pnl_pct = ((indicators.close - self.position.average_cost) /
                          self.position.average_cost) if self.position.average_cost > 0 else 0
                reasoning = (
                    f"EXIT SIGNAL: Price ${indicators.close:.4f} at resistance, "
                    f"RSI={indicators.rsi:.1f}, P&L={pnl_pct:+.2%}"
                )
                take_profit = indicators.close

                return Signal(
                    signal_type=signal_type,
                    symbol=self.symbol,
                    timestamp=indicators.timestamp,
                    price=indicators.close,
                    rsi=indicators.rsi,
                    entry_zone=entry_zone,
                    confidence=confidence,
                    reasoning=reasoning,
                    dca_multiplier=0,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

        # Check for entry signals
        in_aggressive = latest['in_aggressive_zone']
        in_conservative = latest['in_conservative_zone']
        rsi_ok = indicators.rsi < self.RSI_EXTREME_OVERBOUGHT

        if (in_aggressive or in_conservative) and rsi_ok:
            # Calculate DCA amount
            _, dca_multiplier, zone_level = self.calculate_dca_amount(100, indicators.close)

            if dca_multiplier > 0:
                signal_type = SignalType.BUY
                confidence = self._calculate_entry_confidence(indicators, latest, zone_level)

                zone_type = "aggressive" if in_aggressive else "conservative"
                reasoning = (
                    f"DCA BUY: {self.symbol} @ ${indicators.close:.4f} in {zone_type} zone, "
                    f"RSI={indicators.rsi:.1f}, DCA multiplier={dca_multiplier}x ({zone_level})"
                )

                # Set take profit at first resistance
                take_profit = self.config.resistance_levels[0] if self.config.resistance_levels else None

        return Signal(
            signal_type=signal_type,
            symbol=self.symbol,
            timestamp=indicators.timestamp,
            price=indicators.close,
            rsi=indicators.rsi,
            entry_zone=entry_zone,
            confidence=confidence,
            reasoning=reasoning,
            dca_multiplier=dca_multiplier,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def _calculate_entry_confidence(
        self,
        indicators: Indicators,
        latest: pd.Series,
        zone_level: str
    ) -> float:
        """
        Calculate confidence score for entry signal.

        Factors:
        - Zone level (lower = higher confidence)
        - RSI (more oversold = higher confidence)
        - Near support (higher confidence)
        - Volume (higher than average = more confident)

        Returns:
            Confidence score 0-1
        """
        confidence = 0.5

        # Zone level factor
        zone_confidence = {
            'critical_support': 0.25,
            'zone_bottom': 0.20,
            'zone_middle': 0.15,
            'zone_top': 0.10,
            'conservative_zone': 0.05
        }
        confidence += zone_confidence.get(zone_level, 0)

        # RSI factor - more oversold = better
        if indicators.rsi < 25:
            confidence += 0.15
        elif indicators.rsi < 35:
            confidence += 0.10
        elif indicators.rsi < 45:
            confidence += 0.05

        # Near support factor
        if latest.get('near_strong_support', False):
            confidence += 0.10
        if latest.get('at_critical_support', False):
            confidence += 0.15

        # Price below lower Bollinger Band
        if indicators.close < indicators.bollinger_lower:
            confidence += 0.10

        # Cap at 0.95
        return min(confidence, 0.95)

    def enter_position(
        self,
        price: float,
        base_amount: float,
        timestamp: datetime = None
    ) -> Dict:
        """
        Enter or add to position with DCA logic.

        Args:
            price: Entry price
            base_amount: Base position size in dollars
            timestamp: Entry timestamp

        Returns:
            Position entry details
        """
        # Calculate DCA-adjusted amount
        adjusted_amount, multiplier, zone_level = self.calculate_dca_amount(base_amount, price)

        if adjusted_amount <= 0:
            return {"action": "none", "reason": "Price outside entry zones"}

        quantity = adjusted_amount / price

        # Track position
        self.position.add_position(quantity, price, timestamp)

        logger.info(
            f"[DCA ENTRY] {self.symbol} @ ${price:.4f}, "
            f"Amount: ${adjusted_amount:.2f} ({multiplier}x), "
            f"Qty: {quantity:.4f}, Avg Cost: ${self.position.average_cost:.4f}, "
            f"Total Qty: {self.position.total_quantity:.4f}"
        )

        return {
            "action": "buy",
            "symbol": self.symbol,
            "price": price,
            "quantity": quantity,
            "amount": adjusted_amount,
            "dca_multiplier": multiplier,
            "zone_level": zone_level,
            "average_cost": self.position.average_cost,
            "total_quantity": self.position.total_quantity,
            "entry_count": self.position.entry_count,
            "stop_loss": self.config.stop_loss,
            "paper_mode": self.paper_mode
        }

    def exit_position(self, price: float, reason: str) -> Dict:
        """
        Exit entire position.

        Args:
            price: Exit price
            reason: Exit reason

        Returns:
            Position exit details with P&L
        """
        if self.position.total_quantity <= 0:
            return {"action": "none", "reason": "No position to exit"}

        exit_value = self.position.total_quantity * price
        pnl = exit_value - self.position.total_cost
        pnl_pct = (price - self.position.average_cost) / self.position.average_cost

        logger.info(
            f"[EXIT] {self.symbol} @ ${price:.4f}, "
            f"Qty: {self.position.total_quantity:.4f}, "
            f"P&L: ${pnl:.2f} ({pnl_pct:+.2%}), "
            f"Reason: {reason}"
        )

        position_stats = self.position.close_position()

        return {
            "action": "sell",
            "symbol": self.symbol,
            "exit_price": price,
            "exit_value": exit_value,
            "quantity": position_stats['total_quantity'],
            "average_cost": position_stats['average_cost'],
            "total_cost": position_stats['total_cost'],
            "entry_count": position_stats['entry_count'],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "paper_mode": self.paper_mode
        }

    def check_stop_loss(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if stop loss is triggered.

        Args:
            current_price: Current market price

        Returns:
            Tuple of (triggered: bool, reason: str)
        """
        if current_price < self.config.stop_loss:
            return True, f"Price ${current_price:.4f} below stop loss ${self.config.stop_loss}"
        return False, ""

    def get_position_status(self) -> Dict:
        """Get current position status"""
        return {
            "symbol": self.symbol,
            "has_position": self.position.total_quantity > 0,
            "total_quantity": self.position.total_quantity,
            "total_cost": self.position.total_cost,
            "average_cost": self.position.average_cost,
            "entry_count": self.position.entry_count,
            "last_entry_price": self.position.last_entry_price,
            "stop_loss": self.config.stop_loss
        }

    def get_status(self) -> Dict:
        """Get current strategy status"""
        return {
            "symbol": self.symbol,
            "paper_mode": self.paper_mode,
            "position": self.get_position_status(),
            "config": {
                "critical_support": self.config.critical_support,
                "strong_support": self.config.strong_support,
                "resistance_levels": self.config.resistance_levels,
                "aggressive_entry_zone": (self.config.aggressive_entry_low, self.config.aggressive_entry_high),
                "conservative_entry_zone": (self.config.conservative_entry_low, self.config.conservative_entry_high),
                "stop_loss": self.config.stop_loss
            },
            "parameters": {
                "rsi_period": self.RSI_PERIOD,
                "rsi_oversold": self.RSI_OVERSOLD,
                "rsi_overbought": self.RSI_OVERBOUGHT,
                "base_position_pct": self.base_position_pct,
                "dca_levels": self.DCA_LEVELS
            }
        }


def backtest_dca_strategy(
    df: pd.DataFrame,
    symbol: str = "XRP",
    initial_capital: float = 10000.0,
    base_position_pct: float = 0.05
) -> Dict:
    """
    Simple backtest of DCA strategy.

    Args:
        df: DataFrame with OHLCV data
        symbol: Token symbol
        initial_capital: Starting capital
        base_position_pct: Base position size as % of capital

    Returns:
        Backtest results
    """
    strategy = CryptoDCAStrategy(symbol=symbol, paper_mode=True, base_position_pct=base_position_pct)
    df = strategy.calculate_indicators(df.copy())

    capital = initial_capital
    trades = []

    for i in range(50, len(df)):  # Start after indicators have values
        row = df.iloc[i]
        current_price = row['close']
        timestamp = row.name if isinstance(row.name, datetime) else datetime.now(timezone.utc)

        # Check stop loss
        stop_triggered, stop_reason = strategy.check_stop_loss(current_price)
        if stop_triggered and strategy.position.total_quantity > 0:
            exit_result = strategy.exit_position(current_price, stop_reason)
            trades.append({
                'type': 'exit',
                'date': timestamp,
                'price': current_price,
                'reason': stop_reason,
                'pnl_pct': exit_result['pnl_pct'],
                'pnl': exit_result['pnl']
            })
            capital += exit_result['pnl']
            continue

        # Check entry
        if row['in_aggressive_zone'] and not row['stop_loss_triggered']:
            base_amount = capital * base_position_pct
            _, multiplier, zone_level = strategy.calculate_dca_amount(base_amount, current_price)

            if multiplier > 0:
                adjusted_amount = base_amount * multiplier
                if adjusted_amount <= capital * 0.25:  # Max 25% per entry
                    entry_result = strategy.enter_position(current_price, base_amount, timestamp)
                    trades.append({
                        'type': 'entry',
                        'date': timestamp,
                        'price': current_price,
                        'amount': adjusted_amount,
                        'multiplier': multiplier,
                        'zone_level': zone_level
                    })

        # Check exit at resistance
        if row['at_resistance'] and strategy.position.total_quantity > 0:
            exit_result = strategy.exit_position(current_price, "Resistance reached")
            trades.append({
                'type': 'exit',
                'date': timestamp,
                'price': current_price,
                'reason': 'resistance',
                'pnl_pct': exit_result['pnl_pct'],
                'pnl': exit_result['pnl']
            })
            capital += exit_result['pnl']

    # Calculate statistics
    entries = [t for t in trades if t['type'] == 'entry']
    exits = [t for t in trades if t['type'] == 'exit']
    wins = [t for t in exits if t.get('pnl', 0) > 0]
    losses = [t for t in exits if t.get('pnl', 0) <= 0]

    return {
        'symbol': symbol,
        'total_entries': len(entries),
        'total_exits': len(exits),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(exits) if exits else 0,
        'total_return': (capital - initial_capital) / initial_capital,
        'final_capital': capital,
        'avg_win': np.mean([t['pnl_pct'] for t in wins]) if wins else 0,
        'avg_loss': np.mean([t['pnl_pct'] for t in losses]) if losses else 0,
        'trades': trades
    }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("CRYPTO DCA ACCUMULATION STRATEGY TEST")
    print("=" * 60)

    # Test each supported token
    for symbol in ['XRP', 'HBAR', 'XLM']:
        print(f"\n--- {symbol} Configuration ---")
        strategy = CryptoDCAStrategy(symbol=symbol)
        status = strategy.get_status()

        print(f"  Critical Support: ${status['config']['critical_support']}")
        print(f"  Strong Support: {status['config']['strong_support']}")
        print(f"  Resistance: {status['config']['resistance_levels']}")
        print(f"  Aggressive Zone: ${status['config']['aggressive_entry_zone'][0]}-${status['config']['aggressive_entry_zone'][1]}")
        print(f"  Stop Loss: ${status['config']['stop_loss']}")

        # Test DCA calculation at different price levels
        config = TOKEN_CONFIGS[symbol]
        test_prices = [
            config.critical_support,
            config.aggressive_entry_low,
            (config.aggressive_entry_low + config.aggressive_entry_high) / 2,
            config.aggressive_entry_high
        ]

        print(f"\n  DCA Multipliers at Price Levels:")
        for price in test_prices:
            amount, mult, level = strategy.calculate_dca_amount(100, price)
            print(f"    ${price:.4f}: {mult}x ({level}) = ${amount:.2f}")

    print("\n" + "=" * 60)
    print("Strategy ready for paper trading")
    print("=" * 60)
