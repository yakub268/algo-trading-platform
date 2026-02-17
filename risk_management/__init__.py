"""
Advanced Risk Management System
===============================

Comprehensive risk management framework for multi-bot trading system.

Features:
- Portfolio heat monitoring
- Correlation limits
- Value-at-Risk (VaR) calculations
- Dynamic position sizing
- Drawdown protection
- Real-time alerts
- Stress testing
- Kelly Criterion optimization
- Emergency protocols

Author: Trading Bot Arsenal
Created: February 2026
"""

from .core.risk_manager import AdvancedRiskManager
from .core.portfolio_heat import PortfolioHeatMonitor
from .core.correlation_monitor import CorrelationMonitor
from .calculators.var_calculator import VaRCalculator
from .calculators.kelly_optimizer import KellyOptimizer
from .calculators.stress_tester import StressTester
from .monitors.drawdown_protection import DrawdownProtection
from .alerts.risk_alerts import RiskAlertManager
from .config.risk_config import RiskManagementConfig, load_risk_config
from .integration.trading_integration import TradingSystemIntegration, initialize_risk_management
from .integration.dashboard_integration import RiskDashboardAPI

__version__ = "1.0.0"

__all__ = [
    "AdvancedRiskManager",
    "PortfolioHeatMonitor",
    "CorrelationMonitor",
    "VaRCalculator",
    "KellyOptimizer",
    "StressTester",
    "DrawdownProtection",
    "RiskAlertManager",
    "RiskManagementConfig",
    "load_risk_config",
    "TradingSystemIntegration",
    "initialize_risk_management",
    "RiskDashboardAPI"
]

# Convenience function for quick setup
def setup_risk_management(config=None, start_monitoring=True):
    """
    Quick setup for risk management system.

    Args:
        config: RiskManagementConfig (loads default if None)
        start_monitoring: Whether to start background monitoring

    Returns:
        TradingSystemIntegration instance
    """
    if config is None:
        config = load_risk_config()

    integration = TradingSystemIntegration(config)

    if start_monitoring:
        integration.start_monitoring()

    return integration