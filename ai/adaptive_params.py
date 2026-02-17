"""
Adaptive Parameter Tuning
===========================
Uses AI market regime detection to auto-adjust bot parameters:
- Tighten stops in volatile markets
- Widen stops in trending markets
- Adjust position sizes based on conviction
- Modify RSI thresholds based on regime

All adjustments are IN-MEMORY only (no file writes).

Author: Trading Bot Arsenal
Created: February 2026 | V6+ AI Upgrade
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger('AdaptiveParams')


@dataclass
class ParamAdjustment:
    """Record of a parameter adjustment"""
    bot_name: str
    param_name: str
    original_value: float
    adjusted_value: float
    reason: str
    regime: str
    timestamp: str


# Regime-based multipliers for different parameter categories
REGIME_ADJUSTMENTS = {
    'volatile': {
        'stop_loss_pct': 0.7,          # Tighter stops (30% tighter)
        'take_profit_pct': 0.8,        # Tighter TP (20% tighter)
        'position_size_pct': 0.6,      # 40% smaller positions
        'rsi_oversold': -5,            # Lower RSI threshold (additive)
        'rsi_overbought': +5,          # Higher RSI threshold (additive)
        'min_confidence': +10,          # Require higher confidence (additive)
        'max_positions': 0.7,          # Fewer concurrent positions
        'cooldown_multiplier': 1.5,    # Longer cooldown between trades
    },
    'bearish': {
        'stop_loss_pct': 0.8,          # Slightly tighter stops
        'take_profit_pct': 0.85,       # Slightly tighter TP
        'position_size_pct': 0.7,      # 30% smaller positions
        'rsi_oversold': -3,
        'rsi_overbought': +3,
        'min_confidence': +5,
        'max_positions': 0.8,
        'cooldown_multiplier': 1.2,
    },
    'bullish': {
        'stop_loss_pct': 1.15,         # Wider stops (let winners run)
        'take_profit_pct': 1.3,        # Wider TP (capture more upside)
        'position_size_pct': 1.2,      # 20% larger positions
        'rsi_oversold': +3,            # Higher entry threshold
        'rsi_overbought': -3,          # Lower exit threshold
        'min_confidence': -5,           # Accept lower confidence
        'max_positions': 1.2,
        'cooldown_multiplier': 0.8,    # Faster re-entry
    },
    'neutral': {
        'stop_loss_pct': 1.0,          # No change
        'take_profit_pct': 1.0,
        'position_size_pct': 1.0,
        'rsi_oversold': 0,
        'rsi_overbought': 0,
        'min_confidence': 0,
        'max_positions': 1.0,
        'cooldown_multiplier': 1.0,
    },
}

# Conviction-based position sizing overlays
CONVICTION_OVERLAYS = {
    'high': 1.3,      # Strong ensemble agreement
    'medium': 1.0,    # Normal
    'low': 0.6,       # Weak/conflicting signals
}


class AdaptiveParams:
    """
    Dynamically adjusts bot parameters based on market regime and AI signals.

    Flow:
    1. Every 15 min, read market regime from analyst
    2. Look up regime-specific multipliers
    3. Apply to active bot configs (in-memory)
    4. Log all adjustments for audit trail

    Safety:
    - All changes are in-memory only (bot configs revert on restart)
    - Hard limits prevent extreme values
    - Changes are logged and can be reviewed in dashboard
    """

    def __init__(self, market_analyst=None, ensemble=None):
        self.market_analyst = market_analyst
        self.ensemble = ensemble
        self._lock = threading.Lock()

        # Track original values for rollback
        self._original_params: Dict[str, Dict[str, float]] = {}
        # History of adjustments
        self._adjustment_history: List[ParamAdjustment] = []
        # Current active regime
        self._current_regime = 'neutral'
        self._last_adjustment = None

        logger.info("AdaptiveParams initialized")

    def adjust_bot_params(self, bots: Dict) -> List[ParamAdjustment]:
        """
        Adjust parameters for all active bots based on current market regime.

        Args:
            bots: Dict of bot_name -> BotState from orchestrator

        Returns:
            List of adjustments made
        """
        regime = self._get_regime()
        adjustments = []

        if regime == self._current_regime and self._last_adjustment:
            # Same regime, skip unless it's been >30 min
            elapsed = (datetime.now(timezone.utc) -
                      datetime.fromisoformat(self._last_adjustment)).total_seconds()
            if elapsed < 1800:  # 30 minutes
                return []

        logger.info(f"[ADAPTIVE] Regime: {regime} (was: {self._current_regime}) - adjusting parameters")
        self._current_regime = regime
        self._last_adjustment = datetime.now(timezone.utc).isoformat()

        regime_mults = REGIME_ADJUSTMENTS.get(regime, REGIME_ADJUSTMENTS['neutral'])

        for bot_name, bot_state in bots.items():
            if not hasattr(bot_state, 'instance') or bot_state.instance is None:
                continue

            bot_adjustments = self._adjust_single_bot(
                bot_name, bot_state.instance, regime, regime_mults
            )
            adjustments.extend(bot_adjustments)

        if adjustments:
            logger.info(f"[ADAPTIVE] Applied {len(adjustments)} parameter adjustments for regime '{regime}'")
            with self._lock:
                self._adjustment_history.extend(adjustments)
                # Keep last 500 adjustments
                if len(self._adjustment_history) > 500:
                    self._adjustment_history = self._adjustment_history[-500:]
        else:
            logger.info(f"[ADAPTIVE] No parameter adjustments needed for regime '{regime}'")

        return adjustments

    def _adjust_single_bot(self, bot_name: str, bot_instance,
                           regime: str, regime_mults: Dict) -> List[ParamAdjustment]:
        """Adjust parameters for a single bot"""
        adjustments = []

        # Save original params if first time
        if bot_name not in self._original_params:
            self._original_params[bot_name] = self._snapshot_params(bot_instance)

        originals = self._original_params[bot_name]

        # Stop loss adjustment
        if hasattr(bot_instance, 'stop_loss_pct') and 'stop_loss_pct' in originals:
            original = originals['stop_loss_pct']
            mult = regime_mults['stop_loss_pct']
            new_val = self._clamp(original * mult, 0.005, 0.15)  # 0.5% to 15%
            if abs(new_val - getattr(bot_instance, 'stop_loss_pct', original)) > 0.001:
                bot_instance.stop_loss_pct = new_val
                adjustments.append(ParamAdjustment(
                    bot_name=bot_name, param_name='stop_loss_pct',
                    original_value=original, adjusted_value=new_val,
                    reason=f"Regime {regime}: x{mult}", regime=regime,
                    timestamp=datetime.now(timezone.utc).isoformat()
                ))

        # Take profit adjustment
        if hasattr(bot_instance, 'take_profit_pct') and 'take_profit_pct' in originals:
            original = originals['take_profit_pct']
            mult = regime_mults['take_profit_pct']
            new_val = self._clamp(original * mult, 0.005, 0.25)  # 0.5% to 25%
            if abs(new_val - getattr(bot_instance, 'take_profit_pct', original)) > 0.001:
                bot_instance.take_profit_pct = new_val
                adjustments.append(ParamAdjustment(
                    bot_name=bot_name, param_name='take_profit_pct',
                    original_value=original, adjusted_value=new_val,
                    reason=f"Regime {regime}: x{mult}", regime=regime,
                    timestamp=datetime.now(timezone.utc).isoformat()
                ))

        # RSI thresholds
        if hasattr(bot_instance, 'rsi_oversold') and 'rsi_oversold' in originals:
            original = originals['rsi_oversold']
            offset = regime_mults['rsi_oversold']
            new_val = self._clamp(original + offset, 15, 45)
            if abs(new_val - getattr(bot_instance, 'rsi_oversold', original)) > 0.5:
                bot_instance.rsi_oversold = new_val
                adjustments.append(ParamAdjustment(
                    bot_name=bot_name, param_name='rsi_oversold',
                    original_value=original, adjusted_value=new_val,
                    reason=f"Regime {regime}: {offset:+d}", regime=regime,
                    timestamp=datetime.now(timezone.utc).isoformat()
                ))

        if hasattr(bot_instance, 'rsi_overbought') and 'rsi_overbought' in originals:
            original = originals['rsi_overbought']
            offset = regime_mults['rsi_overbought']
            new_val = self._clamp(original + offset, 55, 90)
            if abs(new_val - getattr(bot_instance, 'rsi_overbought', original)) > 0.5:
                bot_instance.rsi_overbought = new_val
                adjustments.append(ParamAdjustment(
                    bot_name=bot_name, param_name='rsi_overbought',
                    original_value=original, adjusted_value=new_val,
                    reason=f"Regime {regime}: {offset:+d}", regime=regime,
                    timestamp=datetime.now(timezone.utc).isoformat()
                ))

        # Position size multiplier (applied as an attribute for orchestrator to read)
        size_mult = regime_mults['position_size_pct']

        # Overlay ensemble conviction if available
        if self.ensemble:
            conviction = self._get_conviction_level(bot_name)
            conviction_mult = CONVICTION_OVERLAYS.get(conviction, 1.0)
            size_mult *= conviction_mult

        size_mult = self._clamp(size_mult, 0.3, 2.0)

        # Store as a readable attribute for orchestrator
        if not hasattr(bot_instance, '_adaptive_size_mult') or \
           abs(getattr(bot_instance, '_adaptive_size_mult', 1.0) - size_mult) > 0.01:
            bot_instance._adaptive_size_mult = size_mult
            adjustments.append(ParamAdjustment(
                bot_name=bot_name, param_name='_adaptive_size_mult',
                original_value=1.0, adjusted_value=size_mult,
                reason=f"Regime {regime}: size x{size_mult:.2f}", regime=regime,
                timestamp=datetime.now(timezone.utc).isoformat()
            ))

        return adjustments

    def _snapshot_params(self, bot_instance) -> Dict[str, float]:
        """Capture current bot parameters for later rollback"""
        params = {}
        for attr in ['stop_loss_pct', 'take_profit_pct', 'rsi_oversold',
                     'rsi_overbought', 'max_positions', 'position_size_pct']:
            if hasattr(bot_instance, attr):
                val = getattr(bot_instance, attr)
                if isinstance(val, (int, float)):
                    params[attr] = float(val)
        return params

    def reset_bot_params(self, bot_name: str, bot_instance):
        """Reset a bot's params to original values"""
        if bot_name in self._original_params:
            originals = self._original_params[bot_name]
            for param, value in originals.items():
                if hasattr(bot_instance, param):
                    setattr(bot_instance, param, value)
            logger.info(f"[ADAPTIVE] Reset {bot_name} to original params")

    def reset_all(self, bots: Dict):
        """Reset all bots to original params"""
        for bot_name, bot_state in bots.items():
            if hasattr(bot_state, 'instance') and bot_state.instance:
                self.reset_bot_params(bot_name, bot_state.instance)

    def _get_regime(self) -> str:
        """Get current market regime from analyst"""
        if self.market_analyst:
            try:
                regime = self.market_analyst.get_market_regime()
                if regime and regime != 'unknown':
                    return regime
            except Exception:
                pass
        return 'neutral'

    def _get_conviction_level(self, bot_name: str) -> str:
        """Get conviction level from ensemble predictions"""
        if not self.ensemble:
            return 'medium'
        try:
            # Check if any ensemble prediction is strong for symbols this bot trades
            preds = self.ensemble.get_all_predictions()
            if not preds:
                return 'medium'

            # Average confidence across all predictions
            confidences = [p.get('confidence', 50) for p in preds.values()]
            avg_conf = sum(confidences) / len(confidences) if confidences else 50

            if avg_conf >= 70:
                return 'high'
            elif avg_conf <= 35:
                return 'low'
            return 'medium'

        except Exception:
            return 'medium'

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))

    def get_current_regime(self) -> str:
        """Get currently applied regime"""
        return self._current_regime

    def get_adjustment_history(self, limit: int = 50) -> List[Dict]:
        """Get recent parameter adjustments"""
        with self._lock:
            recent = self._adjustment_history[-limit:]
            return [
                {
                    'bot': a.bot_name,
                    'param': a.param_name,
                    'original': a.original_value,
                    'adjusted': a.adjusted_value,
                    'reason': a.reason,
                    'regime': a.regime,
                    'time': a.timestamp
                }
                for a in recent
            ]

    def get_stats(self) -> Dict:
        """Get adaptive params statistics"""
        return {
            'current_regime': self._current_regime,
            'last_adjustment': self._last_adjustment,
            'total_adjustments': len(self._adjustment_history),
            'bots_tracked': len(self._original_params),
            'regime_history': self._count_regimes(),
        }

    def _count_regimes(self) -> Dict[str, int]:
        """Count regime occurrences in adjustment history"""
        counts = {}
        with self._lock:
            for adj in self._adjustment_history:
                counts[adj.regime] = counts.get(adj.regime, 0) + 1
        return counts
