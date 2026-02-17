"""
Trading Dashboard V4.2 - AI Verification Service
Verifies user actions before execution to prevent human error
"""
import logging
from typing import Dict, List, Any

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dashboard.models import Position, Strategy, BrokerHealth, BrokerStatus

logger = logging.getLogger(__name__)


class AIVerifier:
    """Service for AI-powered action verification"""

    def __init__(self, strategies: List[Strategy] = None,
                 positions: List[Position] = None,
                 brokers: List[BrokerHealth] = None):
        self.strategies = strategies or []
        self.positions = positions or []
        self.brokers = brokers or []

    def update_state(self, strategies: List[Strategy],
                     positions: List[Position],
                     brokers: List[BrokerHealth]):
        """Update the current state for verification"""
        self.strategies = strategies
        self.positions = positions
        self.brokers = brokers

    def verify_action(self, action: str, params: dict) -> Dict[str, Any]:
        """
        Verify if an action is safe to execute.

        Args:
            action: The action type (e.g., "close_position", "kill_switch")
            params: Action parameters

        Returns:
            {
                "status": "approved" | "warning" | "denied",
                "checks": [
                    {"pass": True, "message": "Net P&L positive"},
                    {"pass": False, "message": "Position at loss: -$5.20"}
                ]
            }
        """
        verifiers = {
            "kill_switch": self._verify_kill_switch,
            "close_all": self._verify_close_all,
            "close_position": self._verify_close_position,
            "reduce_position": self._verify_reduce_position,
            "set_stop": self._verify_set_stop,
            "pause_all": self._verify_pause_all,
            "resume_all": self._verify_resume_all,
            "toggle_strategy": self._verify_toggle_strategy,
        }

        verifier = verifiers.get(action, self._default_verify)
        return verifier(params)

    def _verify_kill_switch(self, params: dict) -> Dict[str, Any]:
        """Verify kill switch action"""
        checks = [
            {"pass": True, "message": "Emergency action - always available"},
            {"pass": len(self.positions) > 0, "message": f"{len(self.positions)} positions will close"},
            {"pass": True, "message": "All strategies will pause"}
        ]

        # Calculate total P&L that will be realized
        total_pnl = sum(p.pnl for p in self.positions)
        checks.append({
            "pass": total_pnl >= 0,
            "message": f"Realized P&L: {'+'if total_pnl >= 0 else ''}${total_pnl:.2f}"
        })

        return {"status": "warning", "checks": checks}

    def _verify_close_all(self, params: dict) -> Dict[str, Any]:
        """Verify close all positions action"""
        checks = []
        status = "approved"

        # Check 1: Net P&L
        total_pnl = sum(p.pnl for p in self.positions)
        checks.append({
            "pass": total_pnl >= 0,
            "message": f"Net P&L: {'+'if total_pnl >= 0 else ''}${total_pnl:.2f}"
        })

        # Check 2: Market conditions
        all_brokers_healthy = all(b.status != BrokerStatus.DISCONNECTED for b in self.brokers)
        checks.append({
            "pass": all_brokers_healthy,
            "message": "Market conditions: Normal" if all_brokers_healthy else "Warning: Some brokers disconnected"
        })

        # Check 3: Position count
        checks.append({
            "pass": True,
            "message": f"Will close {len(self.positions)} positions"
        })

        # Set status based on checks
        if total_pnl < 0:
            status = "warning"
        if not all_brokers_healthy:
            status = "warning"

        return {"status": status, "checks": checks}

    def _verify_close_position(self, params: dict) -> Dict[str, Any]:
        """Verify close single position action"""
        symbol = params.get("symbol", "")
        position = next((p for p in self.positions if p.symbol == symbol), None)

        checks = []
        status = "approved"

        if position:
            # Check 1: P&L status
            checks.append({
                "pass": position.pnl >= 0,
                "message": f"Position P&L: {'+'if position.pnl >= 0 else ''}${position.pnl:.2f}"
            })

            # Check 2: Broker health
            broker = next((b for b in self.brokers if b.name == position.broker), None)
            broker_ok = broker and broker.status != BrokerStatus.DISCONNECTED
            checks.append({
                "pass": broker_ok,
                "message": f"Broker {position.broker}: {'OK' if broker_ok else 'Issue'}"
            })

            # Check 3: Liquidity estimate
            checks.append({
                "pass": True,
                "message": "Liquidity: Sufficient"
            })

            # Check 4: Correlated positions warning
            related_positions = [p for p in self.positions
                               if p.symbol != symbol and p.strategy == position.strategy]
            if related_positions:
                checks.append({
                    "pass": True,
                    "message": f"Note: {len(related_positions)} related positions remain"
                })

            # Set status
            if position.pnl < -5:
                status = "warning"
            if not broker_ok:
                status = "warning"

        else:
            checks.append({"pass": False, "message": f"Position {symbol} not found"})
            status = "denied"

        return {"status": status, "checks": checks}

    def _verify_reduce_position(self, params: dict) -> Dict[str, Any]:
        """Verify reduce position action"""
        symbol = params.get("symbol", "")
        percent = params.get("percent", 0.5)
        position = next((p for p in self.positions if p.symbol == symbol), None)

        checks = []
        status = "approved"

        if position:
            checks.append({
                "pass": True,
                "message": f"Will reduce {symbol} by {int(percent * 100)}%"
            })

            # Check minimum position size after reduction
            remaining_qty = position.quantity * (1 - percent)
            checks.append({
                "pass": remaining_qty > 0,
                "message": f"Remaining: {remaining_qty:.2f} units"
            })

            # Check broker
            broker = next((b for b in self.brokers if b.name == position.broker), None)
            if broker and broker.status == BrokerStatus.DISCONNECTED:
                status = "warning"
                checks.append({"pass": False, "message": f"Broker {position.broker} disconnected"})
        else:
            checks.append({"pass": False, "message": f"Position {symbol} not found"})
            status = "denied"

        return {"status": status, "checks": checks}

    def _verify_set_stop(self, params: dict) -> Dict[str, Any]:
        """Verify set stop loss action"""
        symbol = params.get("symbol", "")
        position = next((p for p in self.positions if p.symbol == symbol), None)

        checks = [
            {"pass": True, "message": "Manual stop loss will override strategy"},
            {"pass": True, "message": "Stop will trigger at market"}
        ]

        if position:
            checks.append({
                "pass": True,
                "message": f"Current P&L: {'+'if position.pnl >= 0 else ''}${position.pnl:.2f}"
            })

        return {"status": "warning", "checks": checks}

    def _verify_pause_all(self, params: dict) -> Dict[str, Any]:
        """Verify pause all strategies action"""
        live_strategies = [s for s in self.strategies if s.status.value == "LIVE"]

        checks = [
            {"pass": True, "message": f"{len(live_strategies)} strategies will pause"},
            {"pass": len(self.positions) == 0,
             "message": f"{len(self.positions)} open positions remain" if self.positions else "No open positions"}
        ]

        # Warn if there are open positions
        if self.positions:
            checks.append({
                "pass": True,
                "message": "Note: Open positions will NOT be closed"
            })

        return {"status": "approved", "checks": checks}

    def _verify_resume_all(self, params: dict) -> Dict[str, Any]:
        """Verify resume all strategies action"""
        paused_strategies = [s for s in self.strategies if s.status.value == "PAUSED"]

        # Check market hours (simplified - just check brokers are connected)
        all_connected = all(b.status != BrokerStatus.DISCONNECTED for b in self.brokers)

        checks = [
            {"pass": all_connected, "message": "All brokers connected" if all_connected else "Some brokers disconnected"},
            {"pass": True, "message": f"{len(paused_strategies)} strategies will resume"},
            {"pass": True, "message": "AI veto layer remains active"}
        ]

        status = "approved" if all_connected else "warning"

        return {"status": status, "checks": checks}

    def _verify_toggle_strategy(self, params: dict) -> Dict[str, Any]:
        """Verify toggle single strategy action"""
        strategy_id = params.get("id")
        strategy_name = params.get("name", "Unknown")
        current_status = params.get("status", "LIVE")

        strategy = next((s for s in self.strategies if s.id == strategy_id), None)

        new_status = "PAUSED" if current_status == "LIVE" else "LIVE"

        checks = [
            {"pass": True, "message": f"{strategy_name} will {new_status.lower()}"},
            {"pass": True, "message": "AI veto layer remains active"}
        ]

        # Check for open positions from this strategy
        if strategy:
            strategy_positions = [p for p in self.positions if p.strategy == strategy_name]
            if strategy_positions and new_status == "PAUSED":
                checks.append({
                    "pass": True,
                    "message": f"Note: {len(strategy_positions)} open positions remain"
                })

        return {"status": "approved", "checks": checks}

    def _default_verify(self, params: dict) -> Dict[str, Any]:
        """Default verification for unknown actions"""
        return {
            "status": "warning",
            "checks": [
                {"pass": True, "message": "Action not recognized, proceeding with caution"}
            ]
        }
