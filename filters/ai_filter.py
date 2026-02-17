"""
AI Filter - Integration with Master Orchestrator
Wraps AIVetoLayer for use in the existing bot framework

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.veto_layer import AIVetoLayer, VetoDecision, VetoResult
from ai.llm_client import LLMClient

logger = logging.getLogger('AIFilter')


@dataclass
class FilterResult:
    """Result from AI filter evaluation"""
    should_execute: bool
    size_multiplier: float  # 0.0 to 1.0
    reasoning: str
    confidence: float
    decision: str  # 'approve', 'veto', 'reduce_size'
    latency_ms: float
    veto_result: Optional[VetoResult] = None


class AIFilter:
    """
    AI Filter for master orchestrator integration.

    Usage:
        filter = AIFilter()
        result = filter.evaluate(signal, context)
        if result.should_execute:
            execute_trade(quantity * result.size_multiplier)
    """

    def __init__(
        self,
        enabled: bool = True,
        dry_run: bool = False,
        min_confidence: float = 0.25
    ):
        self.enabled = enabled
        self.dry_run = dry_run
        self.min_confidence = min_confidence

        # Initialize AI components
        self.llm_client = LLMClient()
        self.veto_layer = AIVetoLayer(
            llm_client=self.llm_client,
            enabled=enabled,
            dry_run=dry_run
        )

        # Cache for market context (updated periodically)
        self._market_context: Dict[str, Any] = {}
        self._context_timestamp: float = 0
        self._context_ttl: float = 300  # 5 minutes

        # Bots to skip (no trades, signal only)
        self.skip_bots = [
            'Market-Scanner',
            'Sentiment-Bot'
        ]

        # Bots that always go through AI (even if global disabled)
        self.force_veto_bots = [
            'Kalshi-Hourly-Crypto',
            'Alpaca-Crypto-RSI'
        ]

        logger.info(f"AIFilter initialized (enabled={enabled}, dry_run={dry_run})")

    def evaluate(self, signal: Dict[str, Any], context: Optional[Dict] = None) -> FilterResult:
        """
        Synchronous wrapper for veto evaluation.
        Call this from master_orchestrator before executing trades.
        """
        bot_name = signal.get('bot_name', '')

        # Skip certain bots
        if bot_name in self.skip_bots:
            return FilterResult(
                should_execute=True,
                size_multiplier=1.0,
                reasoning="Bot in skip list",
                confidence=1.0,
                decision='approve',
                latency_ms=0
            )

        # Check if AI filter is enabled for this bot
        if not self.enabled and bot_name not in self.force_veto_bots:
            return FilterResult(
                should_execute=True,
                size_multiplier=1.0,
                reasoning="AI filter disabled",
                confidence=1.0,
                decision='approve',
                latency_ms=0
            )

        # Merge provided context with cached market context
        full_context = {**self._market_context, **(context or {})}

        # Add timestamp info if not present
        now = datetime.now()
        if 'time_of_day' not in full_context:
            full_context['time_of_day'] = now.strftime('%H:%M')
        if 'day_of_week' not in full_context:
            full_context['day_of_week'] = now.strftime('%A')

        # Run async evaluation
        start_time = time.time()
        try:
            loop = asyncio.get_event_loop()
            # Check if loop is closed (causes "Event loop is closed" errors)
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            # Create fresh event loop if none exists or current one is closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            veto_result = loop.run_until_complete(
                self.veto_layer.evaluate(signal, full_context)
            )
        except Exception as e:
            logger.error(f"AI evaluation failed: {e}")
            # Fail closed: block trade on AI error (safer than letting bad trades through)
            return FilterResult(
                should_execute=False,
                size_multiplier=0.0,
                reasoning=f"AI error, blocking trade (fail-closed): {str(e)[:50]}",
                confidence=0.0,
                decision='veto',
                latency_ms=(time.time() - start_time) * 1000
            )

        latency = (time.time() - start_time) * 1000

        # Convert to FilterResult
        if veto_result.decision == VetoDecision.VETO:
            should_execute = self.dry_run  # Execute anyway if dry_run
            size_mult = 0.0 if not self.dry_run else 1.0
        elif veto_result.decision == VetoDecision.REDUCE_SIZE:
            should_execute = True
            size_mult = veto_result.suggested_size_multiplier
        else:
            should_execute = True
            size_mult = veto_result.suggested_size_multiplier

        return FilterResult(
            should_execute=should_execute,
            size_multiplier=size_mult,
            reasoning=veto_result.reasoning,
            confidence=veto_result.confidence,
            decision=veto_result.decision.value,
            latency_ms=latency,
            veto_result=veto_result
        )

    def update_market_context(self, context: Dict[str, Any]):
        """Update cached market context (call periodically from orchestrator)"""
        self._market_context.update(context)
        self._context_timestamp = time.time()
        logger.debug(f"Market context updated: {list(context.keys())}")

    def set_market_regime(self, regime: str):
        """Quick setter for market regime"""
        self._market_context['market_regime'] = regime

    def set_vix(self, vix: float):
        """Quick setter for VIX"""
        self._market_context['vix'] = vix

    def get_stats(self) -> Dict:
        """Get filter statistics"""
        return self.veto_layer.get_stats()

    def get_recent_decisions(self, limit: int = 20):
        """Get recent AI decisions"""
        return self.veto_layer.get_recent_decisions(limit)


# Global filter instance for easy import
_global_filter: Optional[AIFilter] = None


def get_ai_filter(enabled: bool = True, dry_run: bool = False) -> AIFilter:
    """Get or create global AI filter instance"""
    global _global_filter
    if _global_filter is None:
        _global_filter = AIFilter(enabled=enabled, dry_run=dry_run)
    return _global_filter


def reset_ai_filter():
    """Reset global AI filter (for testing)"""
    global _global_filter
    _global_filter = None


if __name__ == "__main__":
    # Test the AI filter
    print("=" * 60)
    print("AI FILTER TEST")
    print("=" * 60)

    filter = get_ai_filter(enabled=True, dry_run=True)

    # Test signal
    signal = {
        "bot_name": "RSI2-MeanReversion",
        "action": "buy",
        "symbol": "SPY",
        "price": 485.50,
        "quantity": 5,
        "strategy_confidence": 0.8,
        "reason": "RSI(2) = 8, extremely oversold"
    }

    context = {
        "market_regime": "bullish",
        "vix": 16.5,
        "spy_trend": "up"
    }

    print(f"\nTest Signal: {signal['bot_name']} {signal['action'].upper()} {signal['symbol']}")
    print(f"Context: VIX={context['vix']}, Regime={context['market_regime']}")

    result = filter.evaluate(signal, context)

    print(f"\n--- FILTER RESULT ---")
    print(f"Should Execute: {result.should_execute}")
    print(f"Decision: {result.decision.upper()}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Size Multiplier: {result.size_multiplier:.2f}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Latency: {result.latency_ms:.0f}ms")

    print(f"\nFilter Stats: {filter.get_stats()}")
