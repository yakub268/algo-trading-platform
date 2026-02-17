"""
AI Signal Injector
===================
Feeds AI predictions into bot signal generation pipeline.
When analyst predicts bearish >60% confidence, suppress buys and boost sells.
Modifies signal weights based on MTF confluence signals.

This is a FILTER in the pipeline - it modifies signals, never replaces them.

Author: Trading Bot Arsenal
Created: February 2026 | V6+ AI Upgrade
"""

import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger('SignalInjector')


@dataclass
class InjectionResult:
    """Result of signal injection processing"""
    action: str          # 'pass', 'suppress', 'boost', 'reduce'
    original_quantity: float
    adjusted_quantity: float
    size_multiplier: float
    reason: str
    confluence_direction: str   # bullish/bearish/neutral
    confluence_strength: str    # strong/moderate/weak/none
    prediction_confidence: int  # 0-100


# Thresholds for signal modification
BEARISH_SUPPRESS_THRESHOLD = 60    # Suppress buys when bearish confidence > 60%
BULLISH_BOOST_THRESHOLD = 65       # Boost buys when bullish confidence > 65%
STRONG_CONFLUENCE_MULTIPLIER = 1.5 # 50% size boost on strong confluence alignment
MODERATE_CONFLUENCE_MULTIPLIER = 1.2  # 20% boost on moderate alignment
OPPOSING_CONFLUENCE_MULTIPLIER = 0.5  # 50% size cut on opposing confluence
STRONG_OPPOSING_MULTIPLIER = 0.25    # 75% cut on strong opposing signal


class SignalInjector:
    """
    Injects AI predictions into bot signal generation.

    Pipeline position: After bot generates signal, before AI veto layer.

    Logic:
    1. Check market analyst prediction for signal's symbol
    2. Check MTF confluence for signal's symbol
    3. If strong bearish + signal is BUY -> suppress (reduce to 25% or block)
    4. If strong bullish + signal is BUY -> boost (increase up to 150%)
    5. If confluence aligns with signal -> boost proportional to strength
    6. If confluence opposes signal -> reduce proportional to strength
    """

    def __init__(self, market_analyst=None, mtf_engine=None):
        self.market_analyst = market_analyst
        self.mtf_engine = mtf_engine
        self._lock = threading.Lock()

        # Stats
        self.signals_processed = 0
        self.signals_suppressed = 0
        self.signals_boosted = 0
        self.signals_reduced = 0
        self.signals_passed = 0

        logger.info("SignalInjector initialized")

    def inject(self, signal: Dict, bot_name: str = '') -> InjectionResult:
        """
        Process a signal through the AI injection pipeline.

        Args:
            signal: Trade signal dict with 'action', 'symbol', 'quantity', etc.
            bot_name: Name of the originating bot

        Returns:
            InjectionResult with adjusted quantity and reasoning
        """
        self.signals_processed += 1

        symbol = signal.get('symbol', signal.get('ticker', ''))
        action = signal.get('action', '').lower()
        quantity = signal.get('quantity', signal.get('shares', 1))

        if not symbol or action not in ('buy', 'sell', 'long', 'short'):
            self.signals_passed += 1
            return InjectionResult(
                action='pass',
                original_quantity=quantity,
                adjusted_quantity=quantity,
                size_multiplier=1.0,
                reason='Non-trade signal, passing through',
                confluence_direction='neutral',
                confluence_strength='none',
                prediction_confidence=0
            )

        is_buy = action in ('buy', 'long')

        # Gather AI signals
        prediction = self._get_prediction(symbol)
        confluence = self._get_confluence(symbol)

        # Calculate injection
        multiplier = 1.0
        reasons = []
        conf_direction = 'neutral'
        conf_strength = 'none'
        pred_confidence = 0

        # 1. Market analyst prediction check
        if prediction:
            pred_dir = prediction.get('direction', 'neutral')
            pred_conf = prediction.get('confidence', 0)
            pred_confidence = pred_conf

            if pred_dir == 'bearish' and pred_conf >= BEARISH_SUPPRESS_THRESHOLD and is_buy:
                # Bearish prediction + buy signal = suppress
                suppress_factor = min(1.0, (pred_conf - 40) / 60)  # 0-1 scale from 40-100
                multiplier *= (1.0 - suppress_factor * 0.75)  # Up to 75% reduction
                reasons.append(f"Analyst bearish {pred_conf}%: -{suppress_factor*75:.0f}% size")

            elif pred_dir == 'bullish' and pred_conf >= BULLISH_BOOST_THRESHOLD and is_buy:
                # Bullish prediction + buy signal = boost
                boost_factor = min(0.5, (pred_conf - 50) / 100)  # Up to 50% boost
                multiplier *= (1.0 + boost_factor)
                reasons.append(f"Analyst bullish {pred_conf}%: +{boost_factor*100:.0f}% size")

            elif pred_dir == 'bullish' and pred_conf >= BEARISH_SUPPRESS_THRESHOLD and not is_buy:
                # Bullish prediction + sell signal = suppress sell
                suppress_factor = min(1.0, (pred_conf - 40) / 60)
                multiplier *= (1.0 - suppress_factor * 0.5)  # Up to 50% reduction on sells
                reasons.append(f"Analyst bullish {pred_conf}% vs sell: -{suppress_factor*50:.0f}% size")

        # 2. MTF confluence check
        if confluence:
            conf_direction = confluence.get('direction', 'neutral')
            conf_strength = confluence.get('strength', 'none')
            conf_score = confluence.get('weighted_score', 0)

            signal_aligns = (conf_direction == 'bullish' and is_buy) or \
                           (conf_direction == 'bearish' and not is_buy)

            if signal_aligns:
                # Confluence supports the signal
                if conf_strength == 'strong':
                    multiplier *= STRONG_CONFLUENCE_MULTIPLIER
                    reasons.append(f"Strong {conf_direction} confluence: +50% size")
                elif conf_strength == 'moderate':
                    multiplier *= MODERATE_CONFLUENCE_MULTIPLIER
                    reasons.append(f"Moderate {conf_direction} confluence: +20% size")
            else:
                # Confluence opposes the signal
                if conf_strength == 'strong':
                    multiplier *= STRONG_OPPOSING_MULTIPLIER
                    reasons.append(f"Strong opposing {conf_direction} confluence: -75% size")
                elif conf_strength == 'moderate':
                    multiplier *= OPPOSING_CONFLUENCE_MULTIPLIER
                    reasons.append(f"Moderate opposing {conf_direction} confluence: -50% size")

        # Clamp multiplier
        multiplier = max(0.1, min(2.0, multiplier))

        # Calculate adjusted quantity
        adjusted_qty = max(0.001, quantity * multiplier)

        # Determine action type
        if multiplier < 0.3:
            result_action = 'suppress'
            self.signals_suppressed += 1
        elif multiplier > 1.1:
            result_action = 'boost'
            self.signals_boosted += 1
        elif multiplier < 0.9:
            result_action = 'reduce'
            self.signals_reduced += 1
        else:
            result_action = 'pass'
            self.signals_passed += 1

        reason_str = ' | '.join(reasons) if reasons else 'No AI signals to inject'

        if result_action != 'pass':
            logger.info(f"[INJECT] {bot_name} {symbol} {action}: {result_action} "
                        f"(x{multiplier:.2f}) {reason_str}")

        return InjectionResult(
            action=result_action,
            original_quantity=quantity,
            adjusted_quantity=adjusted_qty,
            size_multiplier=multiplier,
            reason=reason_str,
            confluence_direction=conf_direction,
            confluence_strength=conf_strength,
            prediction_confidence=pred_confidence
        )

    def apply_injection(self, signal: Dict, bot_name: str = '') -> Dict:
        """
        Apply signal injection and modify the signal dict in-place.
        Convenience method that wraps inject() and updates the signal.

        Returns the modified signal dict.
        """
        result = self.inject(signal, bot_name)

        if result.action != 'pass':
            original_qty = signal.get('quantity', signal.get('shares', 1))
            signal['si_original_quantity'] = original_qty
            signal['quantity'] = result.adjusted_quantity
            if 'shares' in signal:
                signal['shares'] = result.adjusted_quantity
            signal['si_action'] = result.action
            signal['si_multiplier'] = result.size_multiplier
            signal['si_reason'] = result.reason
            signal['si_confluence'] = f"{result.confluence_direction}/{result.confluence_strength}"

        return signal

    def _get_prediction(self, symbol: str) -> Optional[Dict]:
        """Get market analyst prediction for symbol"""
        if not self.market_analyst:
            return None
        try:
            return self.market_analyst.get_prediction_for_symbol(symbol)
        except Exception:
            return None

    def _get_confluence(self, symbol: str) -> Optional[Dict]:
        """Get MTF confluence signal for symbol"""
        if not self.mtf_engine:
            return None
        try:
            return self.mtf_engine.get_confluence(symbol)
        except Exception:
            return None

    def get_stats(self) -> Dict:
        """Get injection statistics"""
        total = max(1, self.signals_processed)
        return {
            'total_processed': self.signals_processed,
            'suppressed': self.signals_suppressed,
            'boosted': self.signals_boosted,
            'reduced': self.signals_reduced,
            'passed': self.signals_passed,
            'suppress_rate': round(self.signals_suppressed / total * 100, 1),
            'boost_rate': round(self.signals_boosted / total * 100, 1),
        }
