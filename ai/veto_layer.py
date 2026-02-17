"""
AI Veto Layer - Core decision engine
Reviews trade signals and returns APPROVE/VETO with confidence

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import asyncio
import json
import logging
import sqlite3
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.llm_client import LLMClient, LLMResponse
from ai.prompts.veto_prompt import VETO_SYSTEM_PROMPT, build_veto_prompt

logger = logging.getLogger('AIVetoLayer')


class VetoDecision(Enum):
    APPROVE = "approve"
    VETO = "veto"
    REDUCE_SIZE = "reduce_size"  # Approve but reduce position


@dataclass
class VetoResult:
    """Result from AI veto evaluation"""
    decision: VetoDecision
    confidence: float  # 0.0 to 1.0
    reasoning: str
    risk_factors: List[str]
    suggested_size_multiplier: float  # 1.0 = full size, 0.5 = half size
    latency_ms: float
    cached: bool


class AIVetoLayer:
    """
    AI-powered trade veto system.

    Integration pattern:
    1. Bot generates signal
    2. VetoLayer.evaluate(signal, context) called
    3. Returns APPROVE/VETO/REDUCE_SIZE with confidence
    4. Master orchestrator acts on decision

    V7: Supports batch evaluation for multiple signals in one LLM call.
    """

    # Confidence thresholds (tuned for 15-30% pass rate, not 99.4% veto)
    APPROVE_THRESHOLD = 0.35      # Above this = approve (was 0.50 - too strict)
    STRONG_APPROVE_THRESHOLD = 0.80  # Above this = full size
    VETO_THRESHOLD = 0.15         # Below this = veto (was 0.25 - only veto very bad signals)

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        enabled: bool = True,
        dry_run: bool = False,  # Log decisions but don't block trades
        log_db_path: str = None
    ):
        self.llm = llm_client or LLMClient()
        self.enabled = enabled
        self.dry_run = dry_run

        # Decision logging database
        if log_db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            log_db_path = os.path.join(base_dir, "cache", "veto_decisions.db")

        self.log_db_path = log_db_path
        self._init_log_db()

        # Statistics
        self.total_evaluations = 0
        self.approvals = 0
        self.vetoes = 0
        self.size_reductions = 0
        self.errors = 0

        logger.info(f"AIVetoLayer initialized (enabled={enabled}, dry_run={dry_run})")

    def _init_log_db(self):
        """Initialize decision logging database"""
        os.makedirs(os.path.dirname(self.log_db_path), exist_ok=True)
        conn = sqlite3.connect(self.log_db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS veto_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                bot_name TEXT,
                symbol TEXT,
                action TEXT,
                decision TEXT,
                confidence REAL,
                reasoning TEXT,
                risk_factors TEXT,
                size_multiplier REAL,
                latency_ms REAL,
                cached INTEGER,
                dry_run INTEGER
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS shadow_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                bot_name TEXT,
                symbol TEXT,
                action TEXT,
                confidence REAL,
                decision TEXT,
                signal_data TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _log_decision(self, signal: Dict, result: VetoResult):
        """Log decision to database for analysis"""
        try:
            conn = sqlite3.connect(self.log_db_path)
            conn.execute('''
                INSERT INTO veto_decisions
                (timestamp, bot_name, symbol, action, decision, confidence, reasoning,
                 risk_factors, size_multiplier, latency_ms, cached, dry_run)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                signal.get('bot_name', 'unknown'),
                signal.get('symbol', 'unknown'),
                signal.get('action', 'unknown'),
                result.decision.value,
                result.confidence,
                result.reasoning,
                json.dumps(result.risk_factors),
                result.suggested_size_multiplier,
                result.latency_ms,
                1 if result.cached else 0,
                1 if self.dry_run else 0
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to log decision: {e}")

    def _log_shadow_trade(self, signal: Dict, decision: str, confidence: float):
        """
        Log ALL evaluated signals to shadow_trades table for opportunity cost analysis.
        This captures both vetoed and approved signals so we can measure what we missed.
        """
        try:
            # Serialize the full signal for later analysis
            signal_data = json.dumps({
                k: v for k, v in signal.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            })
            conn = sqlite3.connect(self.log_db_path)
            conn.execute('''
                INSERT INTO shadow_trades
                (timestamp, bot_name, symbol, action, confidence, decision, signal_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                signal.get('bot_name', 'unknown'),
                signal.get('symbol', 'unknown'),
                signal.get('action', 'unknown'),
                confidence,
                decision,
                signal_data
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to log shadow trade: {e}")

    async def evaluate(
        self,
        signal: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> VetoResult:
        """
        Evaluate a trade signal and return veto decision.

        Args:
            signal: Trade signal from bot
                {
                    "bot_name": "RSI2-MeanReversion",
                    "action": "buy",
                    "symbol": "AAPL",
                    "price": 185.50,
                    "quantity": 10,
                    "strategy_confidence": 0.75,
                    "reason": "RSI < 10, oversold"
                }
            context: Optional market context
                {
                    "market_regime": "volatile",
                    "vix": 22.5,
                    "spy_trend": "down",
                    "recent_news": ["Fed hawkish...", "Earnings miss..."],
                    "sector_sentiment": -0.3,
                    "time_of_day": "14:30",
                    "day_of_week": "Friday"
                }

        Returns:
            VetoResult with decision and confidence
        """
        if not self.enabled:
            return VetoResult(
                decision=VetoDecision.APPROVE,
                confidence=1.0,
                reasoning="AI veto disabled",
                risk_factors=[],
                suggested_size_multiplier=1.0,
                latency_ms=0,
                cached=False
            )

        self.total_evaluations += 1

        # Build prompt
        prompt = build_veto_prompt(signal, context or {})

        # Query LLM
        try:
            response = await self.llm.query(
                prompt=prompt,
                system_prompt=VETO_SYSTEM_PROMPT,
                max_tokens=400,
                temperature=0.1
            )

            # Parse response
            result = self._parse_response(response)

            # Update stats
            if result.decision == VetoDecision.APPROVE:
                self.approvals += 1
            elif result.decision == VetoDecision.VETO:
                self.vetoes += 1
            else:
                self.size_reductions += 1

            # Log decision
            self._log_decision(signal, result)

            # Log shadow trade (ALL signals, approved or vetoed)
            self._log_shadow_trade(signal, result.decision.value, result.confidence)

            # Log to console
            decision_emoji = {
                VetoDecision.APPROVE: "APPROVE",
                VetoDecision.VETO: "VETO",
                VetoDecision.REDUCE_SIZE: "REDUCE"
            }
            dry_prefix = "[DRY-RUN] " if self.dry_run else ""
            logger.info(
                f"{dry_prefix}AI {decision_emoji[result.decision]}: "
                f"{signal.get('bot_name')} {signal.get('action', '').upper()} {signal.get('symbol')} "
                f"({result.confidence:.0%} confidence)"
            )

            return result

        except Exception as e:
            self.errors += 1
            logger.error(f"AI Veto error: {e}")

            # Default to REDUCE_SIZE on error (not VETO - don't block all trades on AI failure)
            result = VetoResult(
                decision=VetoDecision.REDUCE_SIZE,
                confidence=0.2,
                reasoning=f"AI evaluation failed, reducing size as precaution: {str(e)[:100]}",
                risk_factors=["ai_error"],
                suggested_size_multiplier=0.5,
                latency_ms=0,
                cached=False
            )

            self._log_decision(signal, result)
            self._log_shadow_trade(signal, "error_reduce", 0.2)
            return result

    def _parse_response(self, response: LLMResponse) -> VetoResult:
        """Parse LLM response into VetoResult"""
        content = response.content.strip()

        # Try to parse as JSON first
        try:
            # Find JSON in response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(content[json_start:json_end])

                decision_str = data.get("decision", "approve").lower()
                if "veto" in decision_str:
                    decision = VetoDecision.VETO
                elif "reduce" in decision_str:
                    decision = VetoDecision.REDUCE_SIZE
                else:
                    decision = VetoDecision.APPROVE

                confidence = float(data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1

                # Apply thresholds
                if confidence < self.VETO_THRESHOLD:
                    decision = VetoDecision.VETO
                elif confidence < self.APPROVE_THRESHOLD:
                    decision = VetoDecision.REDUCE_SIZE

                # Calculate size multiplier
                if decision == VetoDecision.VETO:
                    size_mult = 0.0
                elif decision == VetoDecision.REDUCE_SIZE:
                    # Range 0.5-0.75 for reduced size trades (more permissive)
                    size_mult = 0.5 + (confidence - self.VETO_THRESHOLD) / (self.APPROVE_THRESHOLD - self.VETO_THRESHOLD) * 0.25
                elif confidence >= self.STRONG_APPROVE_THRESHOLD:
                    size_mult = 1.0
                else:
                    # Scale between 0.5 and 1.0 based on confidence
                    size_mult = 0.5 + (confidence - self.APPROVE_THRESHOLD) / (1.0 - self.APPROVE_THRESHOLD) * 0.5

                risk_factors = data.get("risk_factors", [])
                if isinstance(risk_factors, str):
                    risk_factors = [risk_factors]

                return VetoResult(
                    decision=decision,
                    confidence=confidence,
                    reasoning=data.get("reasoning", "No reasoning provided")[:200],
                    risk_factors=risk_factors[:5],  # Max 5 factors
                    suggested_size_multiplier=min(1.0, max(0.0, size_mult)),
                    latency_ms=response.latency_ms,
                    cached=response.cached
                )

        except json.JSONDecodeError:
            pass

        # Fallback: parse text response
        content_lower = content.lower()

        if "veto" in content_lower or "reject" in content_lower or "do not" in content_lower:
            decision = VetoDecision.VETO
            confidence = 0.3
        elif "reduce" in content_lower or "caution" in content_lower or "half" in content_lower:
            decision = VetoDecision.REDUCE_SIZE
            confidence = 0.5
        else:
            decision = VetoDecision.APPROVE
            confidence = 0.7

        return VetoResult(
            decision=decision,
            confidence=confidence,
            reasoning=content[:200],
            risk_factors=[],
            suggested_size_multiplier=1.0 if decision == VetoDecision.APPROVE else 0.5,
            latency_ms=response.latency_ms,
            cached=response.cached
        )

    async def evaluate_batch(
        self,
        signals: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[VetoResult]:
        """
        V7: Evaluate multiple signals in a single LLM call (cost optimization).
        Falls back to individual evaluation if batch parsing fails.
        """
        if not signals:
            return []

        if len(signals) == 1:
            result = await self.evaluate(signals[0], context)
            return [result]

        if not self.enabled:
            return [VetoResult(
                decision=VetoDecision.APPROVE, confidence=1.0,
                reasoning="AI veto disabled", risk_factors=[],
                suggested_size_multiplier=1.0, latency_ms=0, cached=False
            ) for _ in signals]

        try:
            from ai.prompts.veto_prompt import build_batch_veto_prompt
            prompt = build_batch_veto_prompt(signals, context or {})

            response = await self.llm.query(
                prompt=prompt,
                system_prompt="""You are a senior risk manager evaluating multiple trade signals.
For each signal, decide: approve, veto, or reduce_size.
Respond with a JSON array: [{"signal_id": 1, "decision": "approve", "confidence": 0.7, "reasoning": "..."}, ...]""",
                max_tokens=600,
                temperature=0.1
            )

            content = response.content.strip()
            json_start = content.find('[')
            json_end = content.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                batch_results = json.loads(content[json_start:json_end])

                results = []
                for i, signal in enumerate(signals):
                    # Find matching result
                    matching = next(
                        (r for r in batch_results if r.get('signal_id') == i + 1),
                        batch_results[i] if i < len(batch_results) else {}
                    )

                    decision_str = matching.get('decision', 'approve').lower()
                    if 'veto' in decision_str:
                        decision = VetoDecision.VETO
                    elif 'reduce' in decision_str:
                        decision = VetoDecision.REDUCE_SIZE
                    else:
                        decision = VetoDecision.APPROVE

                    confidence = float(matching.get('confidence', 0.5))
                    result = VetoResult(
                        decision=decision,
                        confidence=confidence,
                        reasoning=matching.get('reasoning', '')[:200],
                        risk_factors=[],
                        suggested_size_multiplier=1.0 if decision == VetoDecision.APPROVE else 0.5,
                        latency_ms=response.latency_ms / len(signals),
                        cached=response.cached
                    )
                    results.append(result)

                    self.total_evaluations += 1
                    if decision == VetoDecision.APPROVE:
                        self.approvals += 1
                    elif decision == VetoDecision.VETO:
                        self.vetoes += 1
                    else:
                        self.size_reductions += 1

                    self._log_decision(signal, result)
                    self._log_shadow_trade(signal, decision.value, confidence)

                logger.info(f"Batch veto: {len(results)} signals evaluated in 1 LLM call")
                return results

        except Exception as e:
            logger.warning(f"Batch veto failed, falling back to individual: {e}")

        # Fallback: evaluate individually
        results = []
        for signal in signals:
            result = await self.evaluate(signal, context)
            results.append(result)
        return results

    def get_stats(self) -> Dict:
        """Get veto layer statistics"""
        total = self.total_evaluations or 1
        return {
            "total_evaluations": self.total_evaluations,
            "approvals": self.approvals,
            "vetoes": self.vetoes,
            "size_reductions": self.size_reductions,
            "errors": self.errors,
            "approval_rate": round(self.approvals / total, 3),
            "veto_rate": round(self.vetoes / total, 3),
            "reduce_rate": round(self.size_reductions / total, 3),
            "llm_stats": self.llm.get_session_stats()
        }

    def get_recent_decisions(self, limit: int = 20) -> List[Dict]:
        """Get recent decisions from log database"""
        try:
            conn = sqlite3.connect(self.log_db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, bot_name, symbol, action, decision, confidence, reasoning
                FROM veto_decisions
                ORDER BY id DESC
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    "timestamp": row[0],
                    "bot_name": row[1],
                    "symbol": row[2],
                    "action": row[3],
                    "decision": row[4],
                    "confidence": row[5],
                    "reasoning": row[6]
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get recent decisions: {e}")
            return []


# Synchronous wrapper
def evaluate_sync(
    veto_layer: AIVetoLayer,
    signal: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> VetoResult:
    """Synchronous wrapper for AIVetoLayer.evaluate()"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(veto_layer.evaluate(signal, context))


if __name__ == "__main__":
    # Test the veto layer
    async def test():
        print("=" * 60)
        print("AI VETO LAYER TEST")
        print("=" * 60)

        veto = AIVetoLayer(dry_run=True)

        # Test signal
        signal = {
            "bot_name": "RSI2-MeanReversion",
            "action": "buy",
            "symbol": "AAPL",
            "price": 185.50,
            "quantity": 10,
            "strategy_confidence": 0.75,
            "reason": "RSI(2) < 10, oversold condition"
        }

        context = {
            "market_regime": "bullish",
            "vix": 18.5,
            "spy_trend": "up",
            "time_of_day": "10:30",
            "day_of_week": "Tuesday"
        }

        print(f"\nTest Signal: {signal}")
        print(f"Context: {context}")

        result = await veto.evaluate(signal, context)

        print(f"\n--- RESULT ---")
        print(f"Decision: {result.decision.value}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Size Multiplier: {result.suggested_size_multiplier:.2f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Risk Factors: {result.risk_factors}")
        print(f"Latency: {result.latency_ms:.0f}ms")
        print(f"Cached: {result.cached}")

        print(f"\nStats: {veto.get_stats()}")

    asyncio.run(test())
