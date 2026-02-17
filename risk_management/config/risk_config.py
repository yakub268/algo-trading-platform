"""
Advanced Risk Management Configuration
=====================================

Comprehensive configuration for all risk management components.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from decimal import Decimal


class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PortfolioLimits:
    """Portfolio-level risk limits"""
    max_total_exposure: float = 0.95  # 95% max exposure
    max_single_position: float = 0.15  # 15% max per position
    max_sector_exposure: float = 0.25  # 25% max per sector
    max_strategy_exposure: float = 0.40  # 40% max per strategy
    max_correlated_exposure: float = 0.30  # 30% max correlated positions
    min_cash_reserve: float = 0.05  # 5% minimum cash

    # Risk budgets by category
    stock_allocation: float = 0.40  # 40% stocks
    crypto_allocation: float = 0.15  # 15% crypto
    forex_allocation: float = 0.20  # 20% forex
    prediction_allocation: float = 0.20  # 20% prediction markets
    other_allocation: float = 0.05  # 5% other


@dataclass
class VaRConfig:
    """Value at Risk configuration"""
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    time_horizons: List[int] = field(default_factory=lambda: [1, 5, 10])  # days
    monte_carlo_simulations: int = 10000
    historical_window: int = 252  # trading days

    # VaR limits
    daily_var_limit: float = 0.02  # 2% daily VaR limit
    weekly_var_limit: float = 0.05  # 5% weekly VaR limit
    monthly_var_limit: float = 0.10  # 10% monthly VaR limit


@dataclass
class CorrelationLimits:
    """Correlation-based risk limits"""
    max_correlation_threshold: float = 0.70  # Above 0.7 considered highly correlated
    max_correlated_weight: float = 0.30  # 30% max in correlated assets
    correlation_window: int = 60  # 60 days for correlation calculation

    # Correlation groups and limits
    correlation_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        'equity': ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL'],
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
        'crypto': ['BTC', 'ETH', 'SOL', 'ADA'],
        'forex_majors': ['EUR/USD', 'GBP/USD', 'USD/JPY'],
        'commodities': ['GLD', 'SLV', 'USO', 'UNG']
    })

    group_limits: Dict[str, float] = field(default_factory=lambda: {
        'equity': 0.40,
        'tech': 0.25,
        'crypto': 0.15,
        'forex_majors': 0.20,
        'commodities': 0.15
    })


@dataclass
class DrawdownConfig:
    """Drawdown protection settings"""
    max_portfolio_drawdown: float = 0.15  # V4: 15% max drawdown
    max_strategy_drawdown: float = 0.15  # 15% max per strategy

    # Drawdown-based position scaling (V4: tightened to match 15% max)
    drawdown_scaling_tiers: Dict[float, float] = field(default_factory=lambda: {
        0.03: 0.90,  # 3% DD -> 90% position size
        0.06: 0.75,  # 6% DD -> 75% position size
        0.10: 0.50,  # 10% DD -> 50% position size
        0.13: 0.25,  # 13% DD -> 25% position size
        0.15: 0.00   # 15% DD -> halt trading
    })

    # Recovery requirements
    recovery_threshold: float = 0.03  # Must recover 3% to resume normal sizing
    halt_trading_threshold: float = 0.15  # Halt all trading at 15% DD


@dataclass
class VolatilityConfig:
    """Volatility-based risk adjustments"""
    volatility_lookback: int = 20  # 20-day volatility
    vix_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'low': (0, 15),      # VIX 0-15: Low volatility
        'normal': (15, 20),  # VIX 15-20: Normal
        'elevated': (20, 25), # VIX 20-25: Elevated
        'high': (25, 30),    # VIX 25-30: High
        'extreme': (30, 40), # VIX 30-40: Extreme
        'crisis': (40, 100)  # VIX 40+: Crisis
    })

    # Position size multipliers based on volatility regime
    volatility_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'low': 1.0,      # Normal sizing
        'normal': 1.0,   # Normal sizing
        'elevated': 0.75, # 75% sizing
        'high': 0.50,    # 50% sizing
        'extreme': 0.25, # 25% sizing
        'crisis': 0.00   # No new positions
    })


@dataclass
class KellyConfig:
    """Kelly Criterion optimization settings"""
    default_kelly_fraction: float = 0.25  # Quarter Kelly
    min_kelly_fraction: float = 0.10  # Minimum 10%
    max_kelly_fraction: float = 0.50  # Maximum 50%

    # Kelly calculation parameters
    min_trade_sample: int = 30  # Minimum trades for Kelly calculation
    win_rate_confidence: float = 0.90  # 90% confidence interval

    # Strategy-specific Kelly fractions
    strategy_kelly_fractions: Dict[str, float] = field(default_factory=lambda: {
        'rsi_mean_reversion': 0.25,
        'momentum': 0.20,
        'arbitrage': 0.40,  # Higher for low-risk arbitrage
        'prediction_markets': 0.15,  # Lower for binary outcomes
        'default': 0.25
    })


@dataclass
class StressTesting:
    """Stress testing configuration"""
    scenarios: List[Dict[str, float]] = field(default_factory=lambda: [
        {'name': '2008_Crisis', 'equity_shock': -0.50, 'volatility_spike': 3.0},
        {'name': 'Flash_Crash', 'equity_shock': -0.20, 'volatility_spike': 2.0},
        {'name': 'Rate_Shock', 'bond_shock': -0.15, 'currency_shock': 0.10},
        {'name': 'Crypto_Crash', 'crypto_shock': -0.80, 'correlation_spike': 0.90},
        {'name': 'Tail_Risk', 'all_assets_shock': -0.30, 'liquidity_shock': 0.50}
    ])

    stress_frequency: str = 'daily'  # Run stress tests daily
    max_stress_loss: float = 0.25  # 25% max stress loss tolerated


@dataclass
class AlertConfig:
    """Risk alert configuration"""
    # Alert thresholds
    portfolio_heat_warning: float = 0.80  # 80% of max exposure
    portfolio_heat_critical: float = 0.90  # 90% of max exposure

    correlation_warning: float = 0.60  # Correlation above 60%
    correlation_critical: float = 0.80  # Correlation above 80%

    var_warning_multiplier: float = 0.80  # 80% of VaR limit
    var_critical_multiplier: float = 0.95  # 95% of VaR limit

    drawdown_warning: float = 0.10  # 10% drawdown
    drawdown_critical: float = 0.15  # 15% drawdown

    # Alert destinations
    telegram_enabled: bool = True
    email_enabled: bool = False
    discord_enabled: bool = False

    # Alert frequency limits (avoid spam)
    min_alert_interval: int = 300  # 5 minutes between similar alerts
    max_alerts_per_hour: int = 10


@dataclass
class EmergencyProtocols:
    """Emergency stop-loss and circuit breaker protocols"""
    # Automatic position closure triggers
    emergency_drawdown: float = 0.15  # V4: Close all positions at 15% DD
    flash_crash_threshold: float = 0.10  # Close if 10% drop in 15 minutes
    liquidity_crisis_threshold: float = 0.50  # Close if spread > 50bp

    # Manual override capabilities
    allow_manual_override: bool = True
    override_requires_confirmation: bool = True

    # Recovery protocols
    gradual_reentry: bool = True  # Gradually re-enter after emergency stop
    reentry_delay_hours: int = 24  # Wait 24 hours before reentry
    reentry_size_multiplier: float = 0.50  # Start with 50% normal size


@dataclass
class RiskManagementConfig:
    """Master risk management configuration"""
    # Core settings
    risk_level: RiskLevel = RiskLevel.MODERATE
    portfolio_value: float = 10000.0
    update_frequency: int = 60  # Update every 60 seconds

    # Component configurations
    portfolio_limits: PortfolioLimits = field(default_factory=PortfolioLimits)
    var_config: VaRConfig = field(default_factory=VaRConfig)
    correlation_limits: CorrelationLimits = field(default_factory=CorrelationLimits)
    drawdown_config: DrawdownConfig = field(default_factory=DrawdownConfig)
    volatility_config: VolatilityConfig = field(default_factory=VolatilityConfig)
    kelly_config: KellyConfig = field(default_factory=KellyConfig)
    stress_testing: StressTesting = field(default_factory=StressTesting)
    alert_config: AlertConfig = field(default_factory=AlertConfig)
    emergency_protocols: EmergencyProtocols = field(default_factory=EmergencyProtocols)

    def adjust_for_risk_level(self):
        """Adjust all parameters based on risk level"""
        if self.risk_level == RiskLevel.CONSERVATIVE:
            self.portfolio_limits.max_single_position = 0.10
            self.portfolio_limits.max_sector_exposure = 0.20
            self.kelly_config.default_kelly_fraction = 0.15
            self.drawdown_config.max_portfolio_drawdown = 0.10  # V4: tighter for conservative

        elif self.risk_level == RiskLevel.AGGRESSIVE:
            self.portfolio_limits.max_single_position = 0.20
            self.portfolio_limits.max_sector_exposure = 0.35
            self.kelly_config.default_kelly_fraction = 0.35
            self.drawdown_config.max_portfolio_drawdown = 0.20  # V4: capped at 20% even for aggressive

        elif self.risk_level == RiskLevel.EXTREME:
            self.portfolio_limits.max_single_position = 0.25
            self.portfolio_limits.max_sector_exposure = 0.40
            self.kelly_config.default_kelly_fraction = 0.50
            self.drawdown_config.max_portfolio_drawdown = 0.25  # V4: capped at 25% even for extreme

    def load_from_env(self):
        """Load configuration from environment variables"""
        if os.getenv('RISK_LEVEL'):
            self.risk_level = RiskLevel(os.getenv('RISK_LEVEL').lower())

        if os.getenv('PORTFOLIO_VALUE'):
            self.portfolio_value = float(os.getenv('PORTFOLIO_VALUE'))

        if os.getenv('MAX_PORTFOLIO_DRAWDOWN'):
            self.drawdown_config.max_portfolio_drawdown = float(os.getenv('MAX_PORTFOLIO_DRAWDOWN'))

        if os.getenv('MAX_SINGLE_POSITION'):
            self.portfolio_limits.max_single_position = float(os.getenv('MAX_SINGLE_POSITION'))

        # Apply risk level adjustments
        self.adjust_for_risk_level()

        # Crypto-only mode: when only crypto bots are active (stock bots disabled due to PDT),
        # widen limits so crypto positions don't permanently saturate sector/correlation heat.
        if os.getenv('CRYPTO_ONLY_MODE', '').lower() == 'true':
            self.portfolio_limits.crypto_allocation = 0.60
            self.portfolio_limits.max_sector_exposure = 0.50
            self.portfolio_limits.max_correlated_exposure = 0.50
            self.correlation_limits.max_correlated_weight = 0.50
            self.correlation_limits.group_limits['crypto'] = 0.50


def load_risk_config() -> RiskManagementConfig:
    """Load risk management configuration with environment overrides"""
    config = RiskManagementConfig()
    config.load_from_env()
    return config


# Default configuration instance
DEFAULT_RISK_CONFIG = load_risk_config()


if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED RISK MANAGEMENT CONFIGURATION")
    print("=" * 70)

    config = load_risk_config()

    print(f"\nRisk Level: {config.risk_level.value.upper()}")
    print(f"Portfolio Value: ${config.portfolio_value:,.0f}")
    print(f"Max Single Position: {config.portfolio_limits.max_single_position:.0%}")
    print(f"Max Drawdown: {config.drawdown_config.max_portfolio_drawdown:.0%}")
    print(f"Kelly Fraction: {config.kelly_config.default_kelly_fraction:.0%}")
    print(f"VaR Confidence: {config.var_config.confidence_levels}")
    print(f"Monte Carlo Sims: {config.var_config.monte_carlo_simulations:,}")

    print("\n" + "=" * 70)