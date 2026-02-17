"""
Cross-Platform Risk Management System

Monitors all trading bots and enforces risk limits.
Implements circuit breakers to halt trading when limits are breached.
Includes correlation monitoring to prevent concentrated risk.

Author: Jacob
Created: January 2026
Updated: January 2026 - Added correlation monitoring
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RiskManager')


class HaltReason(Enum):
    """Reasons for trading halt."""
    DAILY_LOSS = "Daily loss limit exceeded"
    WEEKLY_LOSS = "Weekly loss limit exceeded"
    MONTHLY_LOSS = "Monthly loss limit exceeded"
    MAX_DRAWDOWN = "Maximum drawdown exceeded"
    MANUAL = "Manual halt triggered"
    ERROR = "System error detected"


# =============================================================================
# CENTRALIZED RISK CONFIGURATION
# =============================================================================
# All platforms use these standardized limits for consistency.
# Override via environment variables if needed.

RISK_LIMITS = {
    # Per-trade limits
    "per_trade_risk": float(os.getenv("MAX_RISK_PER_TRADE", "0.02")),  # 2% max per trade
    "max_position_pct": 0.10,  # 10% max in single position

    # Loss limits (circuit breakers)
    "daily_loss_limit": float(os.getenv("DAILY_LOSS_LIMIT", "0.03")),  # 3% daily
    "weekly_loss_limit": float(os.getenv("WEEKLY_LOSS_LIMIT", "0.05")),  # 5% weekly
    "monthly_loss_limit": float(os.getenv("MONTHLY_LOSS_LIMIT", "0.10")),  # 10% monthly
    "max_drawdown": float(os.getenv("MAX_DRAWDOWN", "0.15")),  # 15% max drawdown

    # Concurrent position limits
    "max_concurrent_trades": int(os.getenv("MAX_CONCURRENT_TRADES", "6")),  # 2 per platform
    "max_trades_per_platform": 2,

    # Correlation risk
    "correlation_limit": 0.7,
    "max_correlated_positions": 2,

    # Kelly criterion
    "kelly_fraction": float(os.getenv("KELLY_FRACTION", "0.10")),  # 10% Kelly
    "max_kelly_fraction": 0.25,  # Cap at 25% Kelly
}

# Platform-specific overrides (use RISK_LIMITS as base)
PLATFORM_LIMITS = {
    "alpaca": {
        "max_position_pct": 0.10,
        "per_trade_risk": RISK_LIMITS["per_trade_risk"],
        "max_concurrent": 2,
    },
    "kalshi": {
        "max_position_pct": 0.05,  # More conservative for prediction markets
        "per_trade_risk": 0.01,  # 1% per trade
        "max_concurrent": 2,
        "min_edge": 0.05,  # Require 5% edge minimum
    },
    "oanda": {
        "max_position_pct": 0.10,
        "per_trade_risk": RISK_LIMITS["per_trade_risk"],
        "max_concurrent": 2,
        "max_leverage": 10,  # Cap leverage at 10x
    },
}


def get_platform_limits(platform: str) -> dict:
    """
    Get risk limits for a specific platform.
    Falls back to global RISK_LIMITS for missing keys.

    Args:
        platform: Platform name (alpaca, kalshi, oanda)

    Returns:
        Dict of risk limits for the platform
    """
    base = RISK_LIMITS.copy()
    platform_overrides = PLATFORM_LIMITS.get(platform.lower(), {})
    base.update(platform_overrides)
    return base


@dataclass
class TradeRecord:
    """Record of a single trade."""
    platform: str
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float] = None
    size: float = 0.0
    pnl: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    status: str = "open"


class RiskManager:
    """
    Cross-platform risk management system.

    Features:
    - Track P&L across all platforms
    - Enforce daily/weekly/monthly loss limits
    - Monitor drawdown from peak
    - Circuit breaker to halt trading
    - Trade correlation checks
    """

    def __init__(
        self,
        starting_balance: float = 500,
        alert_callback: Optional[Callable] = None
    ):
        """
        Initialize risk manager.

        Args:
            starting_balance: Total capital across all platforms
            alert_callback: Function to call for alerts (e.g., Telegram)
        """
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.peak_balance = starting_balance
        self.alert_callback = alert_callback

        # P&L tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.total_pnl = 0.0

        # Trade tracking
        self.open_trades: Dict[str, TradeRecord] = {}
        self.trade_history: list = []

        # State
        self.halted = False
        self.halt_reason: Optional[HaltReason] = None
        self.halt_time: Optional[datetime] = None

        # Period tracking
        self.last_daily_reset = datetime.now().date()
        self.last_weekly_reset = datetime.now().date()
        self.last_monthly_reset = datetime.now().date()

    def check_trade(
        self,
        platform: str,
        symbol: str,
        risk_amount: float,
        side: str = "long"
    ) -> tuple[bool, str]:
        """
        Validate a proposed trade against risk limits.

        Args:
            platform: Trading platform name
            symbol: Trading symbol
            risk_amount: Amount at risk in this trade
            side: Trade direction

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Check if halted
        if self.halted:
            return False, f"Trading halted: {self.halt_reason.value if self.halt_reason else 'Unknown'}"

        # Check per-trade risk
        risk_pct = risk_amount / self.current_balance
        if risk_pct > RISK_LIMITS["per_trade_risk"]:
            return False, f"Trade risk {risk_pct:.1%} exceeds limit of {RISK_LIMITS['per_trade_risk']:.1%}"

        # Check max concurrent trades
        if len(self.open_trades) >= RISK_LIMITS["max_concurrent_trades"]:
            return False, f"Max concurrent trades ({RISK_LIMITS['max_concurrent_trades']}) reached"

        # Check correlation (same symbol on multiple platforms)
        same_symbol_count = sum(
            1 for t in self.open_trades.values()
            if t.symbol.split('/')[0] == symbol.split('/')[0] or
               t.symbol.split('_')[0] == symbol.split('_')[0]
        )
        if same_symbol_count >= 2:
            return False, f"Too many correlated positions in {symbol}"

        return True, "OK"

    def register_trade(
        self,
        trade_id: str,
        platform: str,
        symbol: str,
        side: str,
        entry_price: float,
        size: float
    ) -> None:
        """
        Register a new open trade.

        Args:
            trade_id: Unique trade identifier
            platform: Trading platform
            symbol: Trading symbol
            side: Trade direction
            entry_price: Entry price
            size: Position size in dollars
        """
        trade = TradeRecord(
            platform=platform,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size
        )
        self.open_trades[trade_id] = trade

        logger.info(f"Trade registered: {trade_id} - {side} {symbol} @ {entry_price}")

    def close_trade(
        self,
        trade_id: str,
        exit_price: float
    ) -> Optional[float]:
        """
        Close a trade and calculate P&L.

        Args:
            trade_id: Trade identifier
            exit_price: Exit price

        Returns:
            P&L amount or None if trade not found
        """
        if trade_id not in self.open_trades:
            logger.warning(f"Trade {trade_id} not found")
            return None

        trade = self.open_trades[trade_id]
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.status = "closed"

        # Calculate P&L
        if trade.side == "long":
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:  # short
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

        trade.pnl = trade.size * pnl_pct

        # Update P&L tracking
        self.update_pnl(trade.pnl)

        # Move to history
        self.trade_history.append(trade)
        del self.open_trades[trade_id]

        logger.info(f"Trade closed: {trade_id} - P&L: ${trade.pnl:.2f}")
        return trade.pnl

    def update_pnl(self, pnl: float) -> None:
        """
        Update P&L counters and check circuit breakers.

        Args:
            pnl: P&L amount to add
        """
        # Reset periods if needed
        self._check_period_resets()

        # Update counters
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl
        self.total_pnl += pnl
        self.current_balance += pnl

        # Update peak
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        # Check circuit breakers
        self._check_circuit_breakers()

    def _check_period_resets(self) -> None:
        """Reset period counters if new period started."""
        today = datetime.now().date()

        # Daily reset
        if today > self.last_daily_reset:
            logger.info(f"Daily P&L was ${self.daily_pnl:.2f}, resetting")
            self.daily_pnl = 0.0
            self.last_daily_reset = today

            # Un-halt if was daily limit
            if self.halt_reason == HaltReason.DAILY_LOSS:
                self.resume_trading()

        # Weekly reset (Monday)
        if today.weekday() == 0 and today > self.last_weekly_reset:
            logger.info(f"Weekly P&L was ${self.weekly_pnl:.2f}, resetting")
            self.weekly_pnl = 0.0
            self.last_weekly_reset = today

            if self.halt_reason == HaltReason.WEEKLY_LOSS:
                self.resume_trading()

        # Monthly reset (1st of month)
        if today.day == 1 and today > self.last_monthly_reset:
            logger.info(f"Monthly P&L was ${self.monthly_pnl:.2f}, resetting")
            self.monthly_pnl = 0.0
            self.last_monthly_reset = today

            if self.halt_reason == HaltReason.MONTHLY_LOSS:
                self.resume_trading()

    def _check_circuit_breakers(self) -> None:
        """Check if any circuit breakers should be triggered."""
        # Daily loss limit
        daily_loss_pct = abs(min(0, self.daily_pnl)) / self.starting_balance
        if daily_loss_pct >= RISK_LIMITS["daily_loss_limit"]:
            self.halt_trading(HaltReason.DAILY_LOSS)
            return

        # Weekly loss limit
        weekly_loss_pct = abs(min(0, self.weekly_pnl)) / self.starting_balance
        if weekly_loss_pct >= RISK_LIMITS["weekly_loss_limit"]:
            self.halt_trading(HaltReason.WEEKLY_LOSS)
            return

        # Monthly loss limit
        monthly_loss_pct = abs(min(0, self.monthly_pnl)) / self.starting_balance
        if monthly_loss_pct >= RISK_LIMITS["monthly_loss_limit"]:
            self.halt_trading(HaltReason.MONTHLY_LOSS)
            return

        # Max drawdown
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            if drawdown >= RISK_LIMITS["max_drawdown"]:
                self.halt_trading(HaltReason.MAX_DRAWDOWN)
                return

    def halt_trading(self, reason: HaltReason) -> None:
        """
        Halt all trading.

        Args:
            reason: Reason for the halt
        """
        self.halted = True
        self.halt_reason = reason
        self.halt_time = datetime.now()

        message = f"TRADING HALTED: {reason.value}"
        logger.warning(message)

        # Send alert
        if self.alert_callback:
            self.alert_callback(message)

    def resume_trading(self) -> None:
        """Resume trading after a halt."""
        if not self.halted:
            return

        logger.info(f"Resuming trading after {self.halt_reason.value if self.halt_reason else 'halt'}")
        self.halted = False
        self.halt_reason = None
        self.halt_time = None

        if self.alert_callback:
            self.alert_callback("Trading resumed")

    def get_status(self) -> Dict:
        """Get current risk manager status."""
        drawdown = 0.0
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance

        return {
            "halted": self.halted,
            "halt_reason": self.halt_reason.value if self.halt_reason else None,
            "current_balance": self.current_balance,
            "starting_balance": self.starting_balance,
            "peak_balance": self.peak_balance,
            "total_pnl": self.total_pnl,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "monthly_pnl": self.monthly_pnl,
            "drawdown": drawdown,
            "open_trades": len(self.open_trades),
            "total_trades": len(self.trade_history),
        }

    def get_statistics(self) -> Dict:
        """Calculate trading statistics."""
        if not self.trade_history:
            return {}

        wins = [t for t in self.trade_history if t.pnl > 0]
        losses = [t for t in self.trade_history if t.pnl <= 0]

        total_trades = len(self.trade_history)
        win_count = len(wins)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = gross_profit / win_count if win_count > 0 else 0
        avg_loss = gross_loss / len(losses) if losses else 0

        return {
            "total_trades": total_trades,
            "wins": win_count,
            "losses": len(losses),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_profit": self.total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": max((t.pnl for t in wins), default=0),
            "largest_loss": min((t.pnl for t in losses), default=0),
        }


def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    stop_loss_pct: float
) -> float:
    """
    Calculate position size based on risk parameters.

    Args:
        account_balance: Total account value
        risk_per_trade: Decimal risk per trade (e.g., 0.02 for 2%)
        stop_loss_pct: Decimal stop loss (e.g., 0.05 for 5%)

    Returns:
        Maximum position size in dollars
    """
    if stop_loss_pct == 0:
        return 0

    risk_amount = account_balance * risk_per_trade
    position_size = risk_amount / stop_loss_pct

    # Never exceed 50% in one position
    max_position = account_balance * 0.5
    return min(position_size, max_position)


# =============================================================================
# CORRELATION MONITORING
# =============================================================================

# Maximum allowed correlation between positions
MAX_CORRELATION = 0.70


def calculate_rolling_correlation(
    returns_a: pd.Series,
    returns_b: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate rolling correlation between two return series.

    Args:
        returns_a: First return series
        returns_b: Second return series
        window: Rolling window size (default 20 days)

    Returns:
        Rolling correlation series
    """
    # Align the series
    aligned_a, aligned_b = returns_a.align(returns_b, join='inner')

    if len(aligned_a) < window:
        return pd.Series([np.nan])

    # Calculate rolling correlation
    rolling_corr = aligned_a.rolling(window=window).corr(aligned_b)

    return rolling_corr


def calculate_correlation_matrix(
    returns_dict: Dict[str, pd.Series],
    window: int = 20
) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple assets.

    Args:
        returns_dict: Dict mapping symbol to returns series
        window: Rolling window for correlation

    Returns:
        DataFrame correlation matrix
    """
    # Create DataFrame from returns
    returns_df = pd.DataFrame(returns_dict)

    # Calculate correlation matrix using last 'window' periods
    recent_returns = returns_df.tail(window)

    if len(recent_returns) < window:
        # Not enough data, return empty correlations
        return pd.DataFrame()

    return recent_returns.corr()


@dataclass
class CorrelationAlert:
    """Alert for high correlation between positions"""
    symbol_a: str
    symbol_b: str
    correlation: float
    timestamp: datetime
    risk_level: str  # 'warning' or 'critical'


class CorrelationRiskManager:
    """
    Monitors correlation between positions to prevent concentrated risk.

    Features:
    - Track returns for all open positions
    - Calculate rolling correlations
    - Block new positions that are too correlated with existing
    - Alert on high correlation events
    """

    MAX_CORRELATION = 0.70  # Maximum allowed correlation
    WARNING_CORRELATION = 0.50  # Warning threshold
    LOOKBACK_WINDOW = 20  # Days for correlation calculation

    def __init__(
        self,
        max_correlation: float = 0.70,
        lookback_window: int = 20,
        alert_callback: Optional[Callable] = None
    ):
        """
        Initialize correlation risk manager.

        Args:
            max_correlation: Maximum allowed correlation (default 0.70)
            lookback_window: Days for rolling correlation (default 20)
            alert_callback: Function to call for alerts
        """
        self.max_correlation = max_correlation
        self.lookback_window = lookback_window
        self.alert_callback = alert_callback

        # Returns data storage: symbol -> Series of daily returns
        self.returns_data: Dict[str, pd.Series] = {}

        # Active positions
        self.active_positions: Dict[str, Dict] = {}

        # Correlation cache (refreshed periodically)
        self._correlation_cache: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None

        logger.info(f"CorrelationRiskManager initialized (max_corr={max_correlation})")

    def add_returns_data(
        self,
        symbol: str,
        returns: pd.Series
    ) -> None:
        """
        Add or update returns data for a symbol.

        Args:
            symbol: Trading symbol
            returns: Daily returns series
        """
        self.returns_data[symbol] = returns
        self._invalidate_cache()

    def update_return(
        self,
        symbol: str,
        date: datetime,
        daily_return: float
    ) -> None:
        """
        Update single return value for a symbol.

        Args:
            symbol: Trading symbol
            date: Date of the return
            daily_return: Daily return value
        """
        if symbol not in self.returns_data:
            self.returns_data[symbol] = pd.Series(dtype=float)

        self.returns_data[symbol][date] = daily_return
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate the correlation cache"""
        self._correlation_cache = None
        self._cache_time = None

    def _get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix, using cache if available.

        Returns:
            Correlation matrix DataFrame
        """
        # Check cache validity (5 minutes)
        if (self._correlation_cache is not None and
            self._cache_time is not None and
            datetime.now() - self._cache_time < timedelta(minutes=5)):
            return self._correlation_cache

        # Calculate fresh correlation matrix
        if len(self.returns_data) < 2:
            return pd.DataFrame()

        self._correlation_cache = calculate_correlation_matrix(
            self.returns_data,
            window=self.lookback_window
        )
        self._cache_time = datetime.now()

        return self._correlation_cache

    def get_correlation(self, symbol_a: str, symbol_b: str) -> Optional[float]:
        """
        Get correlation between two symbols.

        Args:
            symbol_a: First symbol
            symbol_b: Second symbol

        Returns:
            Correlation value or None if unavailable
        """
        if symbol_a not in self.returns_data or symbol_b not in self.returns_data:
            return None

        corr_matrix = self._get_correlation_matrix()

        if corr_matrix.empty or symbol_a not in corr_matrix.columns or symbol_b not in corr_matrix.columns:
            return None

        return corr_matrix.loc[symbol_a, symbol_b]

    def can_add_position(
        self,
        new_symbol: str,
        existing_positions: Optional[List[str]] = None
    ) -> Tuple[bool, str, List[CorrelationAlert]]:
        """
        Check if a new position can be added based on correlation limits.

        Args:
            new_symbol: Symbol of the new position
            existing_positions: List of existing position symbols
                               (uses self.active_positions if None)

        Returns:
            Tuple of (allowed: bool, reason: str, alerts: list)
        """
        if existing_positions is None:
            existing_positions = list(self.active_positions.keys())

        if not existing_positions:
            return True, "No existing positions", []

        if new_symbol not in self.returns_data:
            return True, f"No returns data for {new_symbol}, allowing with caution", []

        alerts = []
        high_corr_positions = []

        for existing_symbol in existing_positions:
            corr = self.get_correlation(new_symbol, existing_symbol)

            if corr is None:
                continue

            abs_corr = abs(corr)

            # Create alert for significant correlations
            if abs_corr >= self.WARNING_CORRELATION:
                risk_level = "critical" if abs_corr >= self.max_correlation else "warning"
                alert = CorrelationAlert(
                    symbol_a=new_symbol,
                    symbol_b=existing_symbol,
                    correlation=corr,
                    timestamp=datetime.now(),
                    risk_level=risk_level
                )
                alerts.append(alert)

                if abs_corr >= self.max_correlation:
                    high_corr_positions.append((existing_symbol, corr))

        # Check if any correlation exceeds limit
        if high_corr_positions:
            reason_parts = [
                f"{sym} (corr={corr:.2f})" for sym, corr in high_corr_positions
            ]
            reason = f"High correlation with: {', '.join(reason_parts)}"

            # Send alert if callback configured
            if self.alert_callback and alerts:
                alert_msg = f"CORRELATION ALERT: {new_symbol} blocked - {reason}"
                self.alert_callback(alert_msg)

            return False, reason, alerts

        return True, "Correlation check passed", alerts

    def register_position(
        self,
        symbol: str,
        size: float,
        entry_price: float
    ) -> None:
        """
        Register an active position.

        Args:
            symbol: Trading symbol
            size: Position size
            entry_price: Entry price
        """
        self.active_positions[symbol] = {
            'size': size,
            'entry_price': entry_price,
            'entry_time': datetime.now()
        }

    def close_position(self, symbol: str) -> None:
        """Remove a closed position"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]

    def get_portfolio_correlation_risk(self) -> Dict:
        """
        Analyze overall portfolio correlation risk.

        Returns:
            Dict with correlation metrics and warnings
        """
        positions = list(self.active_positions.keys())

        if len(positions) < 2:
            return {
                'position_count': len(positions),
                'max_correlation': 0,
                'avg_correlation': 0,
                'high_correlation_pairs': [],
                'risk_level': 'low'
            }

        corr_matrix = self._get_correlation_matrix()

        if corr_matrix.empty:
            return {
                'position_count': len(positions),
                'max_correlation': None,
                'avg_correlation': None,
                'high_correlation_pairs': [],
                'risk_level': 'unknown'
            }

        # Analyze correlations between positions
        correlations = []
        high_corr_pairs = []

        for i, sym_a in enumerate(positions):
            for sym_b in positions[i+1:]:
                if sym_a in corr_matrix.columns and sym_b in corr_matrix.columns:
                    corr = corr_matrix.loc[sym_a, sym_b]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

                        if abs(corr) >= self.WARNING_CORRELATION:
                            high_corr_pairs.append({
                                'pair': (sym_a, sym_b),
                                'correlation': corr
                            })

        if not correlations:
            return {
                'position_count': len(positions),
                'max_correlation': None,
                'avg_correlation': None,
                'high_correlation_pairs': [],
                'risk_level': 'unknown'
            }

        max_corr = max(correlations)
        avg_corr = np.mean(correlations)

        # Determine risk level
        if max_corr >= self.max_correlation:
            risk_level = 'critical'
        elif max_corr >= self.WARNING_CORRELATION:
            risk_level = 'elevated'
        else:
            risk_level = 'low'

        return {
            'position_count': len(positions),
            'max_correlation': max_corr,
            'avg_correlation': avg_corr,
            'high_correlation_pairs': high_corr_pairs,
            'risk_level': risk_level
        }


# Example usage
if __name__ == "__main__":
    # Create risk manager with $500 starting balance
    rm = RiskManager(starting_balance=500)

    # Check if a trade is allowed
    allowed, reason = rm.check_trade("alpaca", "BTC/USD", risk_amount=10, side="long")
    print(f"Trade allowed: {allowed}, Reason: {reason}")

    # Register a trade
    rm.register_trade(
        trade_id="alpaca_001",
        platform="alpaca",
        symbol="BTC/USD",
        side="long",
        entry_price=42000,
        size=100
    )

    # Close the trade
    pnl = rm.close_trade("alpaca_001", exit_price=43000)
    print(f"Trade P&L: ${pnl:.2f}")

    # Get status
    status = rm.get_status()
    print(f"Status: {status}")

    # Get statistics
    stats = rm.get_statistics()
    print(f"Statistics: {stats}")

    print("\n" + "=" * 50)
    print("CORRELATION RISK MANAGER TEST")
    print("=" * 50)

    # Test correlation risk manager
    crm = CorrelationRiskManager(max_correlation=0.70)

    # Add some sample returns data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

    # Create correlated returns (SPY and QQQ are typically correlated)
    np.random.seed(42)
    spy_returns = pd.Series(np.random.randn(30) * 0.01, index=dates)
    qqq_returns = spy_returns * 0.8 + pd.Series(np.random.randn(30) * 0.005, index=dates)  # High correlation
    gld_returns = pd.Series(np.random.randn(30) * 0.008, index=dates)  # Low correlation

    crm.add_returns_data("SPY", spy_returns)
    crm.add_returns_data("QQQ", qqq_returns)
    crm.add_returns_data("GLD", gld_returns)

    # Register SPY position
    crm.register_position("SPY", size=1000, entry_price=500)

    # Check if we can add QQQ (high correlation with SPY)
    allowed, reason, alerts = crm.can_add_position("QQQ")
    print(f"\nCan add QQQ: {allowed}")
    print(f"Reason: {reason}")
    if alerts:
        print(f"Alerts: {len(alerts)}")
        for alert in alerts:
            print(f"  - {alert.symbol_a} <-> {alert.symbol_b}: {alert.correlation:.2f} ({alert.risk_level})")

    # Check if we can add GLD (low correlation)
    allowed, reason, alerts = crm.can_add_position("GLD")
    print(f"\nCan add GLD: {allowed}")
    print(f"Reason: {reason}")

    # Get portfolio risk
    risk = crm.get_portfolio_correlation_risk()
    print(f"\nPortfolio Correlation Risk: {risk}")
