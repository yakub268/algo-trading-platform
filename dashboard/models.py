"""
Trading Dashboard V4.2 - Data Models
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class StrategyStatus(Enum):
    LIVE = "LIVE"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class Side(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class BrokerStatus(Enum):
    HEALTHY = "HEALTHY"
    SLOW = "SLOW"
    DISCONNECTED = "DISCONNECTED"


class AlertSeverity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Strategy:
    id: int
    name: str
    status: StrategyStatus
    pnl: float
    win_rate: float
    drawdown: float
    last_signal: str
    ai_reasoning: str
    trades_today: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "pnl": self.pnl,
            "win_rate": self.win_rate,
            "drawdown": self.drawdown,
            "last_signal": self.last_signal,
            "ai_reasoning": self.ai_reasoning,
            "trades_today": self.trades_today
        }


@dataclass
class Position:
    symbol: str
    broker: str  # "Alpaca", "Kalshi", "OANDA"
    strategy: str
    side: Side
    entry_price: float
    current_price: float
    quantity: float
    pnl: float
    age: str  # "2h 34m" format
    signal: str  # What triggered entry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "broker": self.broker,
            "strategy": self.strategy,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "age": self.age,
            "signal": self.signal
        }


@dataclass
class BrokerHealth:
    name: str
    status: BrokerStatus
    latency_ms: int
    buying_power: float
    last_ping: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "buying_power": self.buying_power,
            "last_ping": self.last_ping.isoformat() if self.last_ping else None
        }


@dataclass
class Edge:
    title: str
    our_probability: float  # 0-100
    market_probability: float  # 0-100
    edge_percent: float  # Difference
    confidence: str  # "HIGH", "MED", "LOW"
    source: str  # "NWS API", "CME FedWatch", etc.
    expires: str  # "6h", "13d", etc.
    contract_symbol: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "our_probability": self.our_probability,
            "market_probability": self.market_probability,
            "edge_percent": self.edge_percent,
            "confidence": self.confidence,
            "source": self.source,
            "expires": self.expires,
            "contract_symbol": self.contract_symbol
        }


@dataclass
class RecentTrade:
    timestamp: datetime
    symbol: str
    side: Side
    strategy: str
    signal: str
    expected_price: float
    filled_price: float
    slippage: float
    execution_delay_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.strftime("%H:%M") if self.timestamp else "",
            "symbol": self.symbol,
            "side": self.side.value if isinstance(self.side, Side) else self.side,
            "strategy": self.strategy,
            "signal": self.signal,
            "expected_price": self.expected_price,
            "filled_price": self.filled_price,
            "slippage": self.slippage,
            "execution_delay_ms": self.execution_delay_ms
        }


@dataclass
class Alert:
    severity: AlertSeverity
    message: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "time_ago": self._time_ago()
        }

    def _time_ago(self) -> str:
        if not self.timestamp:
            return "Unknown"
        delta = datetime.now() - self.timestamp
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return f"{seconds} sec ago"
        elif seconds < 3600:
            return f"{seconds // 60} min ago"
        elif seconds < 86400:
            return f"{seconds // 3600} hour ago"
        else:
            return f"{seconds // 86400} day ago"


@dataclass
class DashboardData:
    """Aggregated dashboard data for API response"""
    strategies: List[Strategy] = field(default_factory=list)
    positions: List[Position] = field(default_factory=list)
    brokers: List[BrokerHealth] = field(default_factory=list)
    edges: Dict[str, List[Edge]] = field(default_factory=dict)
    alerts: List[Alert] = field(default_factory=list)
    recent_trades: List[RecentTrade] = field(default_factory=list)
    totals: Dict[str, float] = field(default_factory=dict)
    ai_status: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategies": [s.to_dict() for s in self.strategies],
            "positions": [p.to_dict() for p in self.positions],
            "brokers": [b.to_dict() for b in self.brokers],
            "edges": {k: [e.to_dict() for e in v] for k, v in self.edges.items()},
            "alerts": [a.to_dict() for a in self.alerts],
            "recent_trades": [t.to_dict() for t in self.recent_trades],
            "totals": self.totals,
            "ai_status": self.ai_status
        }
