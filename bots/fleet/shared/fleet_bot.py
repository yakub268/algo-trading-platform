"""
FleetBot — Abstract base class for all fleet trading bots.

Broker-agnostic: supports Kalshi (cents), Alpaca (USD), OANDA (pips).
Every fleet bot implements scan() -> List[FleetSignal].
The FleetOrchestrator handles execution via BrokerRouter.
"""

import os
import sys
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)


class BotType(Enum):
    """Bot classification for risk and routing."""
    KALSHI = "kalshi"
    CRYPTO = "crypto"
    FOREX = "forex"
    PREDICTION = "prediction"
    META = "meta"


@dataclass
class FleetBotConfig:
    """Configuration for a fleet sub-bot."""
    name: str
    bot_type: BotType
    schedule_seconds: int           # How often to run (seconds)
    max_position_usd: float = 50.0  # Max per-trade
    max_daily_trades: int = 10
    min_confidence: float = 0.6
    max_open_positions: int = 10
    symbols: List[str] = field(default_factory=list)
    enabled: bool = True
    paper_mode: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FleetSignal:
    """Universal signal from any fleet bot — broker-agnostic."""
    bot_name: str
    bot_type: str                   # BotType.value
    symbol: str
    side: str                       # BUY/SELL or YES/NO
    entry_price: float              # USD or cents for Kalshi
    target_price: float = 0.0
    stop_loss: float = 0.0
    quantity: float = 0.0           # Units or contracts
    position_size_usd: float = 0.0  # Dollar amount
    confidence: float = 0.0         # 0.0-1.0
    edge: float = 0.0              # Expected edge %
    reason: str = ""
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'bot_name': self.bot_name,
            'bot_type': self.bot_type,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'quantity': self.quantity,
            'position_size_usd': self.position_size_usd,
            'confidence': self.confidence,
            'edge': self.edge,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
        }


class FleetBot(ABC):
    """
    Abstract base class for all fleet bots.

    Subclasses must implement:
    - scan() -> List[FleetSignal]  — detect opportunities

    Optional overrides:
    - check_exits() -> List[FleetSignal]  — exit signals for open positions
    - on_trade_result(signal, success, fill_info) — post-trade callback
    """

    def __init__(self, config: FleetBotConfig):
        self.config = config
        self.name = config.name
        self.bot_type = config.bot_type
        self.paper_mode = config.paper_mode
        self.logger = logging.getLogger(f'Fleet.{config.name}')

        # State tracking
        self.last_run: Optional[datetime] = None
        self.trades_today: int = 0
        self.consecutive_losses: int = 0
        self.daily_pnl: float = 0.0
        self.is_paused: bool = False
        self._last_reset_date: Optional[str] = None
        self._cooldown_until: Optional[datetime] = None
        self._traded_symbols: Dict[str, datetime] = {}  # symbol -> last trade time

    @abstractmethod
    def scan(self) -> List[FleetSignal]:
        """
        Scan for trading opportunities. Called by FleetOrchestrator when this bot is DUE.
        Returns list of FleetSignals (may be empty).
        """
        ...

    def get_open_positions(self) -> List[Dict]:
        """Return open positions for this bot. Override for bot-specific filtering."""
        return []

    def check_exits(self) -> List[FleetSignal]:
        """Check if any open positions should be closed. Override if bot manages its own exits."""
        return []

    def on_trade_result(self, signal: FleetSignal, success: bool, fill_info: Optional[Dict] = None):
        """Called after a trade attempt. Update internal state."""
        if success:
            self.trades_today += 1
            self._traded_symbols[signal.symbol] = datetime.now(timezone.utc)
        self.logger.info(
            f"Trade {'OK' if success else 'FAIL'}: {signal.side} {signal.symbol} "
            f"${signal.position_size_usd:.2f} conf={signal.confidence:.2f}"
        )

    def is_due(self, now: Optional[datetime] = None) -> bool:
        """Check if this bot should run based on its schedule."""
        if self.is_paused or not self.config.enabled:
            return False
        if self._cooldown_until and datetime.now(timezone.utc) < self._cooldown_until:
            return False
        if self.last_run is None:
            return True
        now = now or datetime.now(timezone.utc)
        elapsed = (now - self.last_run).total_seconds()
        return elapsed >= self.config.schedule_seconds

    def pre_scan_checks(self) -> bool:
        """Run before scan(). Returns False to skip this cycle."""
        self._reset_daily_counters()

        if self.trades_today >= self.config.max_daily_trades:
            self.logger.debug(f"Daily trade limit reached ({self.config.max_daily_trades})")
            return False
        return True

    def filter_signals(self, signals: List[FleetSignal]) -> List[FleetSignal]:
        """Apply bot-level filters to signals before fleet-level risk."""
        filtered = []
        for sig in signals:
            # Confidence check
            if sig.confidence < self.config.min_confidence:
                self.logger.debug(f"Low confidence {sig.confidence:.2f} < {self.config.min_confidence} for {sig.symbol}")
                continue

            # Position size cap
            if sig.position_size_usd > self.config.max_position_usd:
                sig.position_size_usd = self.config.max_position_usd

            # 30-min same-symbol cooldown
            last_trade = self._traded_symbols.get(sig.symbol)
            if last_trade and (datetime.now(timezone.utc) - last_trade).total_seconds() < 1800:
                self.logger.debug(f"Cooldown active for {sig.symbol}")
                continue

            # Consecutive loss position reduction
            if self.consecutive_losses >= 5:
                sig.position_size_usd *= 0.5
                sig.metadata['loss_reduction'] = True

            filtered.append(sig)
        return filtered

    def record_loss(self):
        """Record a loss for consecutive tracking."""
        self.consecutive_losses += 1
        if self.consecutive_losses >= 10:
            self._cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=30)
            self.logger.warning(f"10 consecutive losses — 30min cooldown until {self._cooldown_until}")

    def record_win(self):
        """Record a win, reset consecutive loss counter."""
        self.consecutive_losses = 0

    def get_status(self) -> Dict[str, Any]:
        """Return bot status for dashboard."""
        return {
            'name': self.name,
            'bot_type': self.bot_type.value,
            'enabled': self.config.enabled,
            'paused': self.is_paused,
            'schedule_seconds': self.config.schedule_seconds,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'trades_today': self.trades_today,
            'consecutive_losses': self.consecutive_losses,
            'daily_pnl': self.daily_pnl,
            'paper_mode': self.paper_mode,
        }

    def _reset_daily_counters(self):
        """Reset daily counters at midnight UTC."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if self._last_reset_date != today:
            self.trades_today = 0
            self.daily_pnl = 0.0
            self._last_reset_date = today
            self._traded_symbols.clear()
