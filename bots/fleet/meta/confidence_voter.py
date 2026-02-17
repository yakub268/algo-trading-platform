"""
Confidence Voter — Thompson Sampling capital allocation across fleet bots.

Uses Bayesian bandit approach to dynamically allocate more capital to bots
that are performing well and less to those that are underperforming.

Thompson Sampling: Sample from Beta(alpha, beta) for each bot.
- Win → alpha += 1
- Loss → beta += 1
- Higher alpha/beta ratio → higher allocation

This bot doesn't trade directly — it adjusts max_position_usd for other bots.
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType
from bots.fleet.shared.fleet_db import FleetDB

logger = logging.getLogger('Fleet.ConfidenceVoter')

# Default allocation bounds
MIN_ALLOCATION_USD = 5.0    # Even worst bot gets $5 min
MAX_ALLOCATION_USD = 75.0   # Best bot caps at $75
DEFAULT_ALLOCATION_USD = 25.0
TOTAL_FLEET_CAPITAL = 500.0

# Thompson Sampling parameters
PRIOR_ALPHA = 1.0   # Uniform prior
PRIOR_BETA = 1.0
EXPLORATION_BONUS = 0.1  # Small bonus for under-traded bots


class ConfidenceVoter(FleetBot):
    """
    Thompson Sampling capital allocator.

    Not a trader — it reads fleet_bot_stats and fleet_thompson tables,
    runs Thompson Sampling, and returns allocation adjustments as FleetSignals
    with side='ALLOCATE' (meta signal, not executed as trade).
    """

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name='Confidence-Voter',
                bot_type=BotType.META,
                schedule_seconds=300,
                max_position_usd=0,  # Doesn't trade
                max_daily_trades=100,  # Allocation updates
                min_confidence=0.0,
                enabled=True,
                paper_mode=True,
            )
        super().__init__(config)
        self._db: Optional[FleetDB] = None
        self._allocations: Dict[str, float] = {}

    def set_db(self, db: FleetDB):
        """Inject fleet database reference."""
        self._db = db

    def scan(self) -> List[FleetSignal]:
        """
        Run Thompson Sampling and return allocation signals.
        Each signal represents a capital allocation for a bot.
        """
        if not self._db:
            self.logger.warning("No DB reference — cannot compute allocations")
            return []

        # Get all Thompson states
        states = self._db.get_all_thompson_states()
        if not states:
            self.logger.info("No Thompson data yet — using default allocations")
            return []

        # Sample from Beta distribution for each bot
        samples = {}
        for state in states:
            bot_name = state['bot_name']
            alpha = max(state.get('alpha', PRIOR_ALPHA), PRIOR_ALPHA)
            beta = max(state.get('beta', PRIOR_BETA), PRIOR_BETA)
            total_trades = state.get('total_trades', 0)

            # Thompson sample
            sample = np.random.beta(alpha, beta)

            # Exploration bonus for under-traded bots
            if total_trades < 10:
                sample += EXPLORATION_BONUS * (1 - total_trades / 10)

            samples[bot_name] = sample

        if not samples:
            return []

        # Normalize to allocation percentages
        total_sample = sum(samples.values())
        if total_sample <= 0:
            return []

        signals = []
        # Batch DB updates with a single connection
        db_updates = []
        for bot_name, sample in samples.items():
            pct = sample / total_sample
            allocation = max(MIN_ALLOCATION_USD, min(MAX_ALLOCATION_USD, pct * TOTAL_FLEET_CAPITAL))

            self._allocations[bot_name] = allocation
            db_updates.append((sample, allocation, bot_name))

            signals.append(FleetSignal(
                bot_name=self.name,
                bot_type=BotType.META.value,
                symbol=bot_name,  # "symbol" = target bot name
                side='ALLOCATE',
                entry_price=0,
                position_size_usd=allocation,
                confidence=pct,
                edge=sample,
                reason=f"Thompson α={samples.get(bot_name, 0):.2f} → ${allocation:.0f}",
                metadata={
                    'target_bot': bot_name,
                    'allocation_usd': allocation,
                    'allocation_pct': pct,
                    'thompson_sample': sample,
                },
            ))

        # Commit all Thompson updates in one transaction
        if db_updates:
            try:
                conn = self._db._get_conn()
                with conn:
                    conn.executemany('''
                        UPDATE fleet_thompson
                        SET last_sample = ?, last_allocation = ?, updated_at = datetime('now')
                        WHERE bot_name = ?
                    ''', db_updates)
                conn.close()
            except Exception as e:
                self.logger.warning(f"Failed to update Thompson states: {e}")

        self.logger.info(
            f"Allocations updated for {len(signals)} bots. "
            f"Top: {max(self._allocations, key=self._allocations.get)} "
            f"(${max(self._allocations.values()):.0f})"
        )

        return signals

    def get_allocation(self, bot_name: str) -> float:
        """Get current allocation for a bot. Used by FleetOrchestrator."""
        return self._allocations.get(bot_name, DEFAULT_ALLOCATION_USD)

    def get_all_allocations(self) -> Dict[str, float]:
        """Get all current allocations."""
        return dict(self._allocations)

    def get_status(self) -> Dict:
        """Extended status with allocation info."""
        base = super().get_status()
        base['allocations'] = dict(self._allocations)
        base['total_allocated'] = sum(self._allocations.values())
        base['bot_count'] = len(self._allocations)
        return base
