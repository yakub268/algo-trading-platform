"""
Trading Dashboard V4.2 - Services Package
"""
from .broker_alpaca import AlpacaService
from .broker_kalshi import KalshiService
from .broker_oanda import OandaService
from .broker_coinbase import CoinbaseService, get_coinbase_service
from .edge_detector import EdgeDetector
from .ai_verifier import AIVerifier
from .strategy_manager import StrategyManager

__all__ = [
    'AlpacaService',
    'KalshiService',
    'OandaService',
    'CoinbaseService',
    'get_coinbase_service',
    'EdgeDetector',
    'AIVerifier',
    'StrategyManager'
]
