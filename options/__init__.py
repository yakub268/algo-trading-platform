"""
Advanced Options Trading System

A comprehensive options trading platform with sophisticated strategies,
real-time Greeks calculation, volatility analysis, and institutional-grade
risk management capabilities.

Modules:
- core: Options chain data, pricing models, Greeks calculation
- strategies: Advanced options strategies (spreads, straddles, condors)
- data: Real-time options data integration and historical IV data
- risk: Options-specific risk management and hedging
- analytics: Volatility forecasting, IV rank analysis
- backtesting: Options backtesting framework with historical data
- flow: Options flow analysis and unusual activity detection
- events: Event-driven strategies (earnings, ex-div dates)
"""

__version__ = "1.0.0"
__author__ = "Trading Bot Arsenal"

# Core imports
from .core.option_chain import OptionChain
from .core.greeks_calculator import GreeksCalculator
from .core.pricing_models import BlackScholes, BinomialModel

# Strategy imports
from .strategies.basic import CallStrategy, PutStrategy, CoveredCall
from .strategies.spreads import BullCallSpread, BearPutSpread, IronCondor
from .strategies.volatility import Straddle, Strangle, Calendar

# Risk management imports
from .risk.portfolio_greeks import PortfolioGreeksMonitor
from .risk.delta_hedger import DeltaHedger
from .risk.risk_manager import OptionsRiskManager

__all__ = [
    'OptionChain',
    'GreeksCalculator',
    'BlackScholes',
    'BinomialModel',
    'CallStrategy',
    'PutStrategy',
    'CoveredCall',
    'BullCallSpread',
    'BearPutSpread',
    'IronCondor',
    'Straddle',
    'Strangle',
    'Calendar',
    'PortfolioGreeksMonitor',
    'DeltaHedger',
    'OptionsRiskManager'
]