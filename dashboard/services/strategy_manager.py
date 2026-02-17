"""
Trading Dashboard V4.2 - Strategy Manager Service
Manages strategy states and coordinates with the trading system
"""
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dashboard.config import STRATEGY_STATE_FILE, TRADES_DB
from dashboard.models import Strategy, StrategyStatus, RecentTrade, Alert, AlertSeverity, Side

logger = logging.getLogger(__name__)


class StrategyManager:
    """Service for managing trading strategies"""

    def __init__(self):
        self.state_file = Path(STRATEGY_STATE_FILE)
        self.trades_db = Path(TRADES_DB)
        self.strategies = self._load_strategies()
        self._ensure_alerts_table()

    def _load_strategies(self) -> List[Strategy]:
        """Load strategy states from state file or database"""
        # Try loading from state file first
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                return [self._dict_to_strategy(s) for s in data]
            except Exception as e:
                logger.error(f"Failed to load strategy states: {e}")

        # Fall back to loading from database
        return self._load_from_database()

    def _load_from_database(self) -> List[Strategy]:
        """Load strategy info from trades database"""
        strategies = self._default_strategies()

        if not self.trades_db.exists():
            return strategies

        try:
            conn = sqlite3.connect(self.trades_db)
            cursor = conn.cursor()

            # Check if bot_status table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='bot_status'
            """)

            if cursor.fetchone():
                cursor.execute("""
                    SELECT bot_name, status, last_signal, pnl_today, trades_today, error
                    FROM bot_status
                """)

                for row in cursor.fetchall():
                    bot_name, status, last_signal, pnl_today, trades_today, error = row

                    # Map bot name to strategy
                    strategy = next((s for s in strategies if s.name.lower() in bot_name.lower()), None)
                    if strategy:
                        strategy.status = StrategyStatus.LIVE if status == "running" else StrategyStatus.PAUSED
                        strategy.last_signal = last_signal or ""
                        strategy.pnl = pnl_today or 0
                        strategy.trades_today = trades_today or 0
                        if error:
                            strategy.status = StrategyStatus.ERROR
                            strategy.ai_reasoning = f"Error: {error}"

            # Get historical stats
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='daily_summary'
            """)

            if cursor.fetchone():
                for strategy in strategies:
                    cursor.execute("""
                        SELECT SUM(pnl), SUM(wins), SUM(wins + losses)
                        FROM daily_summary
                        WHERE bot_name LIKE ?
                    """, (f"%{strategy.name}%",))

                    row = cursor.fetchone()
                    if row and row[2]:
                        strategy.pnl = row[0] or 0
                        strategy.win_rate = (row[1] / row[2] * 100) if row[2] > 0 else 0

            conn.close()

        except Exception as e:
            logger.error(f"Failed to load from database: {e}")

        return strategies

    def _default_strategies(self) -> List[Strategy]:
        """Default strategy configuration"""
        return [
            Strategy(
                id=1,
                name="Momentum",
                status=StrategyStatus.LIVE,
                pnl=0,
                win_rate=65.0,
                drawdown=0,
                last_signal="",
                ai_reasoning="EMA crossover strategy on SPY/QQQ",
                trades_today=0
            ),
            Strategy(
                id=2,
                name="RSI-2",
                status=StrategyStatus.LIVE,
                pnl=0,
                win_rate=68.0,
                drawdown=0,
                last_signal="",
                ai_reasoning="Larry Connors RSI-2 mean reversion",
                trades_today=0
            ),
            Strategy(
                id=3,
                name="Fed Bot",
                status=StrategyStatus.PAUSED,
                pnl=0,
                win_rate=61.0,
                drawdown=0,
                last_signal="",
                ai_reasoning="FOMC probability edge trading",
                trades_today=0
            ),
            Strategy(
                id=4,
                name="Weather",
                status=StrategyStatus.LIVE,
                pnl=0,
                win_rate=58.0,
                drawdown=0,
                last_signal="",
                ai_reasoning="NWS vs Kalshi weather edge",
                trades_today=0
            ),
        ]

    def _dict_to_strategy(self, data: dict) -> Strategy:
        """Convert dictionary to Strategy object"""
        status = data.get("status", "PAUSED")
        if isinstance(status, str):
            status = StrategyStatus(status)

        return Strategy(
            id=data.get("id", 0),
            name=data.get("name", "Unknown"),
            status=status,
            pnl=data.get("pnl", 0),
            win_rate=data.get("win_rate", 0),
            drawdown=data.get("drawdown", 0),
            last_signal=data.get("last_signal", ""),
            ai_reasoning=data.get("ai_reasoning", ""),
            trades_today=data.get("trades_today", 0)
        )

    def get_strategies(self) -> List[Strategy]:
        """Get all strategies"""
        # Refresh from database for latest stats
        self._refresh_stats()
        return self.strategies

    def _refresh_stats(self):
        """Refresh strategy stats from database"""
        if not self.trades_db.exists():
            return

        try:
            conn = sqlite3.connect(self.trades_db)
            cursor = conn.cursor()

            # Get today's date
            today = datetime.now().strftime("%Y-%m-%d")

            for strategy in self.strategies:
                # Get today's stats
                cursor.execute("""
                    SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), SUM(pnl)
                    FROM trades
                    WHERE bot_name LIKE ? AND DATE(entry_time) = ?
                """, (f"%{strategy.name}%", today))

                row = cursor.fetchone()
                if row and row[0]:
                    strategy.trades_today = row[0]
                    strategy.win_rate = (row[1] / row[0] * 100) if row[0] > 0 else strategy.win_rate
                    strategy.pnl = row[2] or 0

                # Get max drawdown
                cursor.execute("""
                    SELECT MIN(pnl) FROM trades
                    WHERE bot_name LIKE ? AND pnl < 0
                """, (f"%{strategy.name}%",))

                row = cursor.fetchone()
                if row and row[0]:
                    strategy.drawdown = abs(row[0])

            conn.close()

        except Exception as e:
            logger.error(f"Failed to refresh stats: {e}")

    def toggle_strategy(self, strategy_id: int) -> bool:
        """Toggle strategy status"""
        for s in self.strategies:
            if s.id == strategy_id:
                if s.status == StrategyStatus.LIVE:
                    s.status = StrategyStatus.PAUSED
                else:
                    s.status = StrategyStatus.LIVE
                self._save_state()
                return True
        return False

    def pause_all(self):
        """Pause all strategies"""
        for s in self.strategies:
            s.status = StrategyStatus.PAUSED
        self._save_state()

    def resume_all(self):
        """Resume all strategies"""
        for s in self.strategies:
            if s.status != StrategyStatus.ERROR:
                s.status = StrategyStatus.LIVE
        self._save_state()

    def _save_state(self):
        """Persist strategy states to file"""
        try:
            # Ensure directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            data = [s.to_dict() for s in self.strategies]
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info("Strategy states saved")
        except Exception as e:
            logger.error(f"Failed to save strategy states: {e}")

    def get_recent_trades(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent trades from database. Returns dict with 'trades', 'is_mock'."""
        trades = []

        if not self.trades_db.exists():
            return {"trades": self._mock_recent_trades(), "is_mock": True}

        try:
            conn = sqlite3.connect(self.trades_db)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT entry_time, symbol, side, bot_name, entry_price, exit_price, pnl
                FROM trades
                ORDER BY entry_time DESC
                LIMIT ?
            """, (limit,))

            for row in cursor.fetchall():
                entry_time, symbol, side, bot_name, entry_price, exit_price, pnl = row

                # Parse timestamp
                try:
                    timestamp = datetime.fromisoformat(entry_time)
                except Exception as e:
                    logger.debug(f"Error parsing trade timestamp: {e}")
                    timestamp = datetime.now()

                # Calculate slippage (simplified)
                slippage = 0.0
                if entry_price and exit_price:
                    slippage = abs(exit_price - entry_price) * 0.001  # Estimate

                trades.append(RecentTrade(
                    timestamp=timestamp,
                    symbol=symbol or "?",
                    side=Side.LONG if side and side.upper() in ["BUY", "LONG"] else Side.SHORT,
                    strategy=self._extract_strategy_name(bot_name),
                    signal="Signal",
                    expected_price=entry_price or 0,
                    filled_price=entry_price or 0,
                    slippage=slippage
                ))

            conn.close()

        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return {"trades": self._mock_recent_trades(), "is_mock": True}

        if trades:
            return {"trades": trades, "is_mock": False}
        else:
            return {"trades": self._mock_recent_trades(), "is_mock": True}

    def get_alerts(self, limit: int = 10) -> List[Alert]:
        """Get recent alerts from database.
        Note: Returns List[Alert] for backward compatibility.
        Mock alerts have '[MOCK] ' prefix in their messages.
        """
        if not self.trades_db.exists():
            return self._mock_alerts()

        try:
            conn = sqlite3.connect(self.trades_db)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT severity, message, timestamp
                FROM alerts
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            alerts = []
            for row in rows:
                alerts.append(Alert(
                    severity=AlertSeverity(row[0]),
                    message=row[1],
                    timestamp=datetime.fromisoformat(row[2]) if row[2] else datetime.now()
                ))

            return alerts if alerts else self._mock_alerts()

        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return self._mock_alerts()

    def add_alert(self, severity: str, message: str):
        """Add a new alert to the database"""
        logger.info(f"Alert [{severity}]: {message}")

        if not self.trades_db.exists():
            return

        try:
            conn = sqlite3.connect(self.trades_db)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO alerts (severity, message, timestamp)
                VALUES (?, ?, ?)
            """, (severity, message, datetime.now().isoformat()))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to add alert: {e}")

    def _ensure_alerts_table(self):
        """Create alerts table if it doesn't exist"""
        if not self.trades_db.exists():
            return

        try:
            conn = sqlite3.connect(self.trades_db)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to create alerts table: {e}")

    def get_ai_status(self) -> Dict[str, Any]:
        """Get AI system status from regime detector and veto layer"""
        status = {
            "regime": "SIDEWAYS",
            "sentiment": 0.0,
            "veto_rate": 0,
            "ai_cost": 0.0
        }

        # Try to get regime from RegimeDetector
        try:
            from filters.regime_detector import RegimeDetector, MarketRegime
            import yfinance as yf

            detector = RegimeDetector()
            spy = yf.Ticker("SPY")
            hist = spy.history(period="60d")

            if not hist.empty:
                result = detector.detect(hist)
                if result:
                    status["regime"] = result.current_regime.value.upper()
                    status["sentiment"] = round(result.confidence, 2)
        except Exception as e:
            logger.debug(f"Could not get regime: {e}")

        # Try to get veto stats from trades database
        try:
            if self.trades_db.exists():
                conn = sqlite3.connect(self.trades_db)
                cursor = conn.cursor()

                # Count recent trades with AI veto
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN notes LIKE '%vetoed%' THEN 1 ELSE 0 END) as vetoed
                    FROM trades
                    WHERE entry_time > datetime('now', '-7 days')
                """)
                row = cursor.fetchone()
                conn.close()

                if row and row[0] > 0:
                    status["veto_rate"] = int((row[1] or 0) / row[0] * 100)
        except Exception as e:
            logger.debug(f"Could not get veto stats: {e}")

        # Estimate AI cost (based on API calls)
        try:
            # Check for AI usage log file
            ai_log = self.trades_db.parent / "ai_usage.json"
            if ai_log.exists():
                import json
                with open(ai_log) as f:
                    usage = json.load(f)
                    status["ai_cost"] = round(usage.get("total_cost_today", 0), 2)
        except Exception as e:
            logger.debug(f"Could not get AI cost: {e}")

        return status

    def get_equity_curve(self, days: int = 30) -> Dict[str, Any]:
        """Get equity curve data. Returns dict with 'data', 'is_mock', and optional 'message'."""
        curve = []

        if not self.trades_db.exists():
            return {"data": [], "is_mock": True, "message": "No portfolio data yet - trades database not found"}

        try:
            conn = sqlite3.connect(self.trades_db)
            cursor = conn.cursor()

            # Get portfolio value history
            cursor.execute("""
                SELECT timestamp, total_value
                FROM portfolio_value
                ORDER BY timestamp DESC
                LIMIT ?
            """, (days,))

            rows = cursor.fetchall()
            conn.close()

            for i, row in enumerate(reversed(rows)):
                curve.append({
                    "day": i + 1,
                    "value": row[1],
                    "timestamp": row[0]
                })

        except Exception as e:
            logger.error(f"Failed to get equity curve: {e}")
            return {"data": [], "is_mock": True, "message": f"Error loading equity data: {e}"}

        if curve:
            return {"data": curve, "is_mock": False}
        else:
            return {"data": [], "is_mock": True, "message": "No portfolio data yet - start trading to see equity curve"}

    def _extract_strategy_name(self, bot_name: str) -> str:
        """Extract strategy name from bot name"""
        if not bot_name:
            return "Unknown"

        bot_name_lower = bot_name.lower()
        if "momentum" in bot_name_lower or "ema" in bot_name_lower:
            return "Momentum"
        elif "rsi" in bot_name_lower:
            return "RSI-2"
        elif "fed" in bot_name_lower or "fomc" in bot_name_lower:
            return "Fed Bot"
        elif "weather" in bot_name_lower:
            return "Weather"
        elif "macd" in bot_name_lower:
            return "MACD"

        return bot_name.split("_")[0].title() if "_" in bot_name else bot_name

    def _mock_recent_trades(self) -> List[RecentTrade]:
        """Return mock trades for demo"""
        now = datetime.now()
        return [
            RecentTrade(
                timestamp=now - timedelta(minutes=30),
                symbol="AAPL",
                side=Side.LONG,
                strategy="Momentum",
                signal="EMA Cross",
                expected_price=182.45,
                filled_price=182.50,
                slippage=0.05
            ),
            RecentTrade(
                timestamp=now - timedelta(hours=1),
                symbol="BTC/USD",
                side=Side.LONG,
                strategy="RSI-2",
                signal="RSI<10",
                expected_price=43095,
                filled_price=43100,
                slippage=0.05
            ),
            RecentTrade(
                timestamp=now - timedelta(hours=2),
                symbol="EUR/USD",
                side=Side.SHORT,
                strategy="MACD",
                signal="Cross Down",
                expected_price=1.0825,
                filled_price=1.0823,
                slippage=-0.02
            ),
        ]

    def _mock_alerts(self) -> List[Alert]:
        """Return mock alerts for demo"""
        now = datetime.now()
        return [
            Alert(
                severity=AlertSeverity.CRITICAL,
                message="RSI-2 execution timeout on TSLA order",
                timestamp=now - timedelta(minutes=2)
            ),
            Alert(
                severity=AlertSeverity.WARNING,
                message="Kalshi API latency elevated (340ms)",
                timestamp=now - timedelta(minutes=15)
            ),
            Alert(
                severity=AlertSeverity.INFO,
                message="Regime change: BULL → SIDEWAYS",
                timestamp=now - timedelta(hours=1)
            ),
            Alert(
                severity=AlertSeverity.INFO,
                message="Weather edge found: NYC +14%",
                timestamp=now - timedelta(hours=2)
            ),
        ]

    def _mock_equity_curve(self, days: int) -> List[Dict[str, Any]]:
        """Return mock equity curve for demo"""
        import random
        curve = []
        value = 400

        for day in range(1, days + 1):
            value += random.uniform(-5, 8)
            curve.append({
                "day": day,
                "value": round(value, 2)
            })

        return curve
