"""
AI Module for Trading Bot
=========================
Provides AI-powered trade filtering, prediction, and market analysis.

Components:
- LLMClient: Multi-provider LLM client (DeepSeek, Claude, Ollama)
- AIVetoLayer: Trade signal evaluation (APPROVE/VETO/REDUCE_SIZE)
- AIFilter: Integration with master orchestrator
- AIMarketAnalyst: Periodic market analysis and directional predictions
- MultiTimeframeEngine: 1h/4h/24h predictions with confluence detection
- SignalInjector: AI-driven signal modification pipeline
- EnsembleCombiner: Multi-source weighted prediction fusion
- AdaptiveParams: Regime-based bot parameter auto-tuning

Author: Trading Bot Arsenal
Created: January 2026 | V6 AI Upgrade: February 2026 | V6+ AI Upgrade: February 2026
"""

from ai.llm_client import LLMClient, LLMProvider, LLMResponse
from ai.veto_layer import AIVetoLayer, VetoDecision, VetoResult

__all__ = [
    'LLMClient', 'LLMProvider', 'LLMResponse',
    'AIVetoLayer', 'VetoDecision', 'VetoResult',
]

# Lazy imports for V6+ components (avoid import errors if deps missing)
def get_multi_timeframe_engine(*args, **kwargs):
    from ai.multi_timeframe import MultiTimeframeEngine
    return MultiTimeframeEngine(*args, **kwargs)

def get_signal_injector(*args, **kwargs):
    from ai.signal_injector import SignalInjector
    return SignalInjector(*args, **kwargs)

def get_ensemble_combiner(*args, **kwargs):
    from ai.ensemble import EnsembleCombiner
    return EnsembleCombiner(*args, **kwargs)

def get_adaptive_params(*args, **kwargs):
    from ai.adaptive_params import AdaptiveParams
    return AdaptiveParams(*args, **kwargs)
