"""
Fleet Adapter — Bridges FleetOrchestrator to master_orchestrator's bot interface.

The master_orchestrator expects bots to implement:
- run_scan() -> List[Dict]  (with keys: action, symbol, price, quantity, confidence, reason)
- check_exits() -> List[Dict] (optional)
- get_status() -> Dict

This adapter wraps FleetOrchestrator to match that interface.
Registered as 'Fleet-Orchestrator' in BOT_REGISTRY.
"""

import os
import sys
import logging
from typing import Dict, List, Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from bots.fleet.fleet_orchestrator import FleetOrchestrator

logger = logging.getLogger('Fleet.Adapter')


class FleetAdapter:
    """
    Adapter between FleetOrchestrator and master_orchestrator.

    The master orchestrator calls:
    - run_scan() every 60 seconds
    - check_exits() periodically
    - get_status() for dashboard

    This adapter delegates to FleetOrchestrator which manages all 19 sub-bots.
    """

    def __init__(self, capital: float = 500.0, paper_mode: bool = None):
        if paper_mode is None:
            paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'

        self.paper_mode = paper_mode
        self.capital = capital
        self._orchestrator = FleetOrchestrator(
            paper_mode=paper_mode,
            fleet_capital=capital,
        )
        self._initialized = False
        logger.info(f"FleetAdapter created (paper={paper_mode}, capital=${capital})")

    def _ensure_init(self):
        if not self._initialized:
            self._orchestrator.initialize()
            self._initialized = True

    def run_scan(self) -> List[Dict]:
        """
        Called by master_orchestrator every 60 seconds.
        Runs the fleet cycle and returns trade signals in the format
        the master orchestrator expects.
        """
        self._ensure_init()

        try:
            trades = self._orchestrator.run_cycle()
            return trades
        except Exception as e:
            logger.error(f"Fleet cycle error: {e}")
            return []

    def check_exits(self) -> List[Dict]:
        """Called by master_orchestrator to check for exits."""
        self._ensure_init()

        try:
            return self._orchestrator.check_exits()
        except Exception as e:
            logger.error(f"Fleet exit check error: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """Called by master_orchestrator / dashboard."""
        self._ensure_init()

        try:
            status = self._orchestrator.get_fleet_status()
            return {
                'name': 'Fleet-Orchestrator',
                'status': 'running',
                'paper_mode': self.paper_mode,
                'sub_bots': len(self._orchestrator.bots),
                'cycle_count': status.get('cycle_count', 0),
                'total_trades': status.get('total_trades', 0),
                'today_pnl': status.get('today_pnl', 0),
                'open_positions': status.get('open_positions', 0),
                'details': status,
            }
        except Exception as e:
            logger.error(f"Fleet status error: {e}")
            return {'name': 'Fleet-Orchestrator', 'status': 'error', 'error': str(e)}

    # Expose sub-bot management
    def pause_bot(self, bot_name: str) -> bool:
        self._ensure_init()
        return self._orchestrator.pause_bot(bot_name)

    def resume_bot(self, bot_name: str) -> bool:
        self._ensure_init()
        return self._orchestrator.resume_bot(bot_name)

    def emergency_stop(self):
        self._ensure_init()
        self._orchestrator.emergency_stop()

    def get_bot_names(self) -> List[str]:
        self._ensure_init()
        return self._orchestrator.get_bot_names()

    @property
    def orchestrator(self) -> FleetOrchestrator:
        self._ensure_init()
        return self._orchestrator
