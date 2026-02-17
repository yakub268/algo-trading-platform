"""
Core Options Trading Infrastructure

This module provides the foundational components for options trading including:
- Options chain data structures and management
- Real-time Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Options pricing models (Black-Scholes, Binomial)
- Implied volatility calculation and analysis
- Options contract specifications and metadata
"""

from .option_chain import OptionChain, OptionContract
from .greeks_calculator import GreeksCalculator, Greeks
from .pricing_models import BlackScholes, BinomialModel, ImpliedVolatilityCalculator
from .option_data_manager import OptionDataManager

__all__ = [
    'OptionChain',
    'OptionContract',
    'GreeksCalculator',
    'Greeks',
    'BlackScholes',
    'BinomialModel',
    'ImpliedVolatilityCalculator',
    'OptionDataManager'
]