"""
Fleet Risk Management — 3-layer risk controls.

Layer 1 (Bot-level): Handled in FleetBot.filter_signals()
Layer 2 (Fleet-level): This module — daily loss, exposure caps, concentration, contradictions
Layer 3 (Master-level): Inherited from master_orchestrator (drawdown, regime, AI veto)
"""

import re
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from .fleet_db import FleetDB
from .fleet_bot import FleetSignal

logger = logging.getLogger('Fleet.Risk')


class FleetRisk:
    """Fleet-level risk manager — sits between bot signals and execution."""

    # Fleet-wide limits
    DAILY_LOSS_LIMIT = 30.0        # Max $30 daily loss across all fleet bots
    MAX_OPEN_POSITIONS = 25         # Max concurrent positions
    MAX_TOTAL_EXPOSURE_PCT = 0.80   # Max 80% of fleet capital deployed
    MAX_BOT_TYPE_CONCENTRATION = 0.40  # Max 40% in any single bot_type
    COOLDOWN_AFTER_LOSSES = 3       # 3+ consecutive fleet-wide losses → cooldown
    COOLDOWN_MINUTES = 30

    def __init__(self, db: FleetDB, fleet_capital: float = 500.0):
        self.db = db
        self.fleet_capital = fleet_capital
        self._consecutive_fleet_losses = 0
        self._cooldown_until: Optional[datetime] = None
        self._last_reset_date: Optional[str] = None

    def check_trade(self, signal: FleetSignal) -> Tuple[bool, str]:
        """
        Check if a trade passes fleet-level risk controls.
        Returns (allowed, reason).
        """
        self._reset_daily()

        # 0. Fleet cooldown
        if self._cooldown_until and datetime.now(timezone.utc) < self._cooldown_until:
            return False, f"Fleet cooldown until {self._cooldown_until.strftime('%H:%M:%S')}"

        # 1. Daily loss limit
        today_pnl = self.db.get_today_pnl()
        if today_pnl <= -self.DAILY_LOSS_LIMIT:
            self.db.update_risk_state(risk_blocked=1)
            return False, f"Daily loss limit hit: ${today_pnl:.2f} <= -${self.DAILY_LOSS_LIMIT}"

        # 2. Open position cap
        open_count = self.db.get_open_position_count()
        if open_count >= self.MAX_OPEN_POSITIONS:
            return False, f"Max open positions: {open_count} >= {self.MAX_OPEN_POSITIONS}"

        # 3. Total exposure cap
        exposure = self.db.get_total_exposure()
        max_exposure = self.fleet_capital * self.MAX_TOTAL_EXPOSURE_PCT
        if exposure + signal.position_size_usd > max_exposure:
            return False, f"Exposure limit: ${exposure:.0f} + ${signal.position_size_usd:.0f} > ${max_exposure:.0f}"

        # 4. Bot type concentration
        open_positions = self.db.get_open_positions()
        type_exposure = sum(
            p['position_size_usd'] for p in open_positions
            if p['bot_type'] == signal.bot_type
        )
        max_type_exposure = self.fleet_capital * self.MAX_BOT_TYPE_CONCENTRATION
        if type_exposure + signal.position_size_usd > max_type_exposure:
            return False, (
                f"{signal.bot_type} concentration: ${type_exposure:.0f} + "
                f"${signal.position_size_usd:.0f} > ${max_type_exposure:.0f}"
            )

        # 5. Contradictory position check (same underlying, opposite side)
        underlying = self._extract_underlying(signal.symbol)
        for pos in open_positions:
            pos_underlying = self._extract_underlying(pos['symbol'])
            if pos_underlying == underlying and pos['side'] != signal.side:
                return False, (
                    f"Contradictory: {signal.side} {signal.symbol} vs "
                    f"existing {pos['side']} {pos['symbol']}"
                )

        # 6. Duplicate position check
        if self.db.has_open_position(signal.symbol, signal.bot_name):
            return False, f"Already has open position on {signal.symbol}"

        return True, "OK"

    def record_trade_result(self, won: bool):
        """Track fleet-wide consecutive losses."""
        if won:
            self._consecutive_fleet_losses = 0
        else:
            self._consecutive_fleet_losses += 1
            if self._consecutive_fleet_losses >= self.COOLDOWN_AFTER_LOSSES:
                self._cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=self.COOLDOWN_MINUTES)
                logger.warning(
                    f"{self._consecutive_fleet_losses} consecutive fleet losses — "
                    f"cooldown until {self._cooldown_until.strftime('%H:%M:%S')}"
                )

    def get_risk_summary(self) -> Dict:
        """Current risk state for dashboard."""
        return {
            'daily_pnl': self.db.get_today_pnl(),
            'daily_loss_limit': self.DAILY_LOSS_LIMIT,
            'open_positions': self.db.get_open_position_count(),
            'max_positions': self.MAX_OPEN_POSITIONS,
            'total_exposure': self.db.get_total_exposure(),
            'max_exposure': self.fleet_capital * self.MAX_TOTAL_EXPOSURE_PCT,
            'consecutive_losses': self._consecutive_fleet_losses,
            'cooldown_active': bool(
                self._cooldown_until and datetime.now(timezone.utc) < self._cooldown_until
            ),
            'fleet_capital': self.fleet_capital,
        }

    @staticmethod
    def _extract_underlying(symbol: str) -> str:
        """
        Extract underlying from a symbol.
        Kalshi: KXHIGHCHI-26FEB09-B38.5 → KXHIGHCHI-26FEB09
        Crypto: BTC/USD → BTC/USD
        Forex: EUR_USD → EUR_USD
        """
        # Kalshi ticker pattern: PREFIX-DATE-THRESHOLD
        match = re.match(r'^(KX\w+-\w+)-[BT]', symbol)
        if match:
            return match.group(1)
        return symbol

    def _reset_daily(self):
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if self._last_reset_date != today:
            self._consecutive_fleet_losses = 0
            self._cooldown_until = None
            self._last_reset_date = today
