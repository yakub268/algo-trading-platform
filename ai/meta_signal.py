"""
Meta-Signal Aggregator
=======================
Aggregates signals from all active bots into a consensus score.
"3/3 bots bullish on ETH" = strong conviction signal.
Inter-strategy agreement is one of the most powerful signals in quant trading.

Author: Trading Bot Arsenal
Created: February 2026 | V7 Crystal Ball Upgrade
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger('MetaSignal')


@dataclass
class BotSignal:
    """Signal from an individual bot"""
    bot_name: str
    symbol: str
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0-1
    timestamp: str


class MetaSignalAggregator:
    """
    Aggregates signals across all active bots to detect consensus.

    When multiple independent strategies agree on direction,
    the probability of that move is significantly higher.

    Agreement levels:
    - 3/3 bots agree → STRONG consensus (confidence boost)
    - 2/3 bots agree → MODERATE consensus
    - 1/3 or mixed → WEAK/no consensus
    """

    def __init__(self):
        self._signals: Dict[str, List[BotSignal]] = {}  # symbol -> [signals]
        self._consensus: Dict[str, Dict] = {}  # symbol -> consensus result
        self._lock = threading.Lock()

        logger.info("MetaSignalAggregator initialized")

    def record_signal(self, bot_name: str, symbol: str, direction: str,
                      confidence: float = 0.5):
        """Record a signal from a bot for aggregation."""
        signal = BotSignal(
            bot_name=bot_name,
            symbol=symbol,
            direction=direction,
            confidence=min(1.0, max(0.0, confidence)),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        with self._lock:
            if symbol not in self._signals:
                self._signals[symbol] = []

            # Replace existing signal from same bot
            self._signals[symbol] = [
                s for s in self._signals[symbol] if s.bot_name != bot_name
            ]
            self._signals[symbol].append(signal)

        # Recompute consensus
        self._compute_consensus(symbol)

    def record_bot_signals(self, bot_signals: Dict[str, Dict]):
        """
        Batch record signals from multiple bots.

        Args:
            bot_signals: {bot_name: {symbol: {'direction': ..., 'confidence': ...}}}
        """
        for bot_name, symbols in bot_signals.items():
            for symbol, signal_data in symbols.items():
                self.record_signal(
                    bot_name=bot_name,
                    symbol=symbol,
                    direction=signal_data.get('direction', 'neutral'),
                    confidence=signal_data.get('confidence', 0.5)
                )

    def _compute_consensus(self, symbol: str):
        """Compute consensus for a symbol from all recorded bot signals."""
        with self._lock:
            signals = self._signals.get(symbol, [])

        if not signals:
            return

        bullish_count = 0
        bearish_count = 0
        bullish_conf = []
        bearish_conf = []
        total = len(signals)

        for sig in signals:
            if sig.direction == 'bullish':
                bullish_count += 1
                bullish_conf.append(sig.confidence)
            elif sig.direction == 'bearish':
                bearish_count += 1
                bearish_conf.append(sig.confidence)

        # Determine consensus direction
        if bullish_count > bearish_count:
            direction = 'bullish'
            agreement = bullish_count / total
            avg_conf = sum(bullish_conf) / len(bullish_conf) if bullish_conf else 0
        elif bearish_count > bullish_count:
            direction = 'bearish'
            agreement = bearish_count / total
            avg_conf = sum(bearish_conf) / len(bearish_conf) if bearish_conf else 0
        else:
            direction = 'neutral'
            agreement = 0
            avg_conf = 0

        # Confidence scales with agreement AND individual bot confidence
        # 3/3 agree with high confidence → very strong signal
        confidence = int(min(90, agreement * avg_conf * 100))

        # Agreement bonus
        if agreement >= 1.0 and total >= 3:
            confidence = min(95, confidence + 15)  # Perfect agreement bonus
        elif agreement >= 0.67:
            confidence = min(90, confidence + 5)

        consensus = {
            'direction': direction,
            'confidence': confidence,
            'agreement_ratio': round(agreement, 2),
            'bullish_bots': bullish_count,
            'bearish_bots': bearish_count,
            'total_bots': total,
            'bot_details': [
                {'bot': s.bot_name, 'direction': s.direction,
                 'confidence': round(s.confidence, 2)}
                for s in signals
            ],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        with self._lock:
            self._consensus[symbol] = consensus

    def get_consensus(self, symbol: str) -> Optional[Dict]:
        """Get current consensus for a symbol."""
        with self._lock:
            return self._consensus.get(symbol)

    def get_all_consensus(self) -> Dict[str, Dict]:
        """Get consensus for all symbols."""
        with self._lock:
            return self._consensus.copy()

    def clear_stale_signals(self, max_age_seconds: int = 3600):
        """Remove signals older than max_age."""
        from datetime import datetime as dt
        now = dt.now(timezone.utc)
        with self._lock:
            for symbol in list(self._signals.keys()):
                self._signals[symbol] = [
                    s for s in self._signals[symbol]
                    if (now - dt.fromisoformat(s.timestamp)).total_seconds() < max_age_seconds
                ]
                if not self._signals[symbol]:
                    del self._signals[symbol]
                    if symbol in self._consensus:
                        del self._consensus[symbol]

    def get_stats(self) -> Dict:
        """Get meta-signal statistics."""
        with self._lock:
            return {
                'symbols_tracked': len(self._signals),
                'total_signals': sum(len(v) for v in self._signals.values()),
                'strong_consensus': sum(
                    1 for c in self._consensus.values()
                    if c.get('agreement_ratio', 0) >= 1.0
                ),
                'moderate_consensus': sum(
                    1 for c in self._consensus.values()
                    if 0.67 <= c.get('agreement_ratio', 0) < 1.0
                ),
            }
