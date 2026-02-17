"""
Trading Dashboard - Routes Package

All API endpoints are organized into blueprints:
- v5_bp:       /api/v5/*           V5 dashboard endpoints
- ai_bots_bp:  /api/ai/*           AI trading bot endpoints
- brokers_bp:  /api/alpaca/*, /api/coinbase/*  Broker-specific endpoints
- actions_bp:  /api/action, /api/high-risk/*, /api/refresh, /api/risk-management/*
- legacy_bp:   /api/opportunities, /api/category-stats, /api/paper-*, etc.
"""

from .v5 import v5_bp
from .ai_bots import ai_bots_bp
from .brokers import brokers_bp
from .actions import actions_bp
from .legacy import legacy_bp

__all__ = ['v5_bp', 'ai_bots_bp', 'brokers_bp', 'actions_bp', 'legacy_bp']
