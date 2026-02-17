"""
Paper Trading Validation System

2-week validation protocol for all strategies before live deployment.

GO/NO-GO Criteria:
- Minimum 100 trades total
- Minimum 45% win rate
- Minimum 1.0 Sharpe ratio
- Maximum 15% drawdown
- Minimum 95% uptime

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PaperTrading')


class ValidationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class PaperTrade:
    """Single paper trade record"""
    id: Optional[int]
    strategy: str
    symbol: str
    side: str  # buy, sell
    entry_time: datetime
    entry_price: float
    quantity: float
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    pnl: Optional[float]
    pnl_pct: Optional[float]
    status: str  # open, closed, stopped
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float
    reasoning: str


@dataclass
class StrategyMetrics:
    """Aggregated metrics for a strategy"""
    strategy: str
    total_trades: int
    open_trades: int
    closed_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    validation_status: str
    last_updated: datetime


@dataclass
class ValidationCriteria:
    """GO/NO-GO validation criteria"""
    min_trades: int = 100
    min_win_rate: float = 0.45
    min_sharpe: float = 1.0
    max_drawdown: float = 0.15
    min_profit_factor: float = 1.2
    validation_days: int = 14


class PaperTradingTracker:
    """
    Tracks paper trades and validates strategy performance.
    
    Usage:
        tracker = PaperTradingTracker()
        tracker.record_entry('rsi2', 'SPY', 'buy', 450.0, 100, 0.85, 'RSI oversold')
        tracker.record_exit('rsi2', 'SPY', 455.0, 'take_profit')
        metrics = tracker.get_strategy_metrics('rsi2')
        
        if tracker.check_go_no_go('rsi2'):
            print("Strategy validated! Ready for live trading.")
    """
    
    def __init__(self, db_path: str = "data/paper_trading.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.criteria = ValidationCriteria()
        self._init_schema()
        
        logger.info(f"PaperTradingTracker initialized: {db_path}")
    
    def _init_schema(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Paper trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                exit_time TEXT,
                exit_price REAL,
                pnl REAL,
                pnl_pct REAL,
                status TEXT NOT NULL DEFAULT 'open',
                stop_loss REAL,
                take_profit REAL,
                confidence REAL,
                reasoning TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Daily P&L tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                strategy TEXT NOT NULL,
                starting_equity REAL NOT NULL,
                ending_equity REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                daily_return REAL NOT NULL,
                trades_count INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                losses INTEGER NOT NULL,
                max_drawdown REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, strategy)
            )
        """)
        
        # Strategy validation status
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL UNIQUE,
                start_date TEXT NOT NULL,
                end_date TEXT,
                total_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                status TEXT DEFAULT 'pending',
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Equity curve tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                equity REAL NOT NULL,
                peak_equity REAL NOT NULL,
                drawdown REAL NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def start_validation(self, strategy: str, initial_equity: float = 10000.0) -> bool:
        """Start validation period for a strategy"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        end_date = (datetime.now(timezone.utc) + timedelta(days=self.criteria.validation_days)).isoformat()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO validation_status 
                (strategy, start_date, end_date, status, updated_at)
                VALUES (?, ?, ?, 'running', ?)
            """, (strategy, now, end_date, now))
            
            # Initialize equity curve
            cursor.execute("""
                INSERT INTO equity_curve (timestamp, strategy, equity, peak_equity, drawdown)
                VALUES (?, ?, ?, ?, 0)
            """, (now, strategy, initial_equity, initial_equity))
            
            conn.commit()
            logger.info(f"[{strategy}] Validation started - {self.criteria.validation_days} day period")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start validation: {e}")
            return False
        finally:
            conn.close()
    
    def record_entry(
        self,
        strategy: str,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        confidence: float = 0.5,
        reasoning: str = "",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> int:
        """Record a new paper trade entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        
        cursor.execute("""
            INSERT INTO paper_trades 
            (strategy, symbol, side, entry_time, entry_price, quantity, 
             status, stop_loss, take_profit, confidence, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)
        """, (strategy, symbol, side, now, price, quantity, 
              stop_loss, take_profit, confidence, reasoning))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"[{strategy}] ENTRY: {side.upper()} {quantity} {symbol} @ ${price:.2f}")
        return trade_id
    
    def record_exit(
        self,
        strategy: str,
        symbol: str,
        exit_price: float,
        exit_reason: str = "manual"
    ) -> Optional[Dict]:
        """Record exit for open paper trade"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find open trade
        cursor.execute("""
            SELECT id, side, entry_price, quantity, entry_time
            FROM paper_trades
            WHERE strategy = ? AND symbol = ? AND status = 'open'
            ORDER BY entry_time DESC
            LIMIT 1
        """, (strategy, symbol))
        
        row = cursor.fetchone()
        if not row:
            logger.warning(f"No open trade found for {strategy}/{symbol}")
            conn.close()
            return None
        
        trade_id, side, entry_price, quantity, entry_time = row
        
        # Calculate P&L
        if side == 'buy':
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # sell/short
            pnl = (entry_price - exit_price) * quantity
            pnl_pct = (entry_price - exit_price) / entry_price
        
        now = datetime.now(timezone.utc).isoformat()
        
        cursor.execute("""
            UPDATE paper_trades
            SET exit_time = ?, exit_price = ?, pnl = ?, pnl_pct = ?, status = ?
            WHERE id = ?
        """, (now, exit_price, pnl, pnl_pct, exit_reason, trade_id))
        
        conn.commit()
        conn.close()
        
        emoji = "✅" if pnl >= 0 else "❌"
        logger.info(f"[{strategy}] EXIT {emoji}: {symbol} @ ${exit_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.2%})")
        
        # Update equity curve
        self._update_equity_curve(strategy, pnl)
        
        return {
            'trade_id': trade_id,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason
        }
    
    def check_stops(self, strategy: str, current_prices: Dict[str, float]) -> List[Dict]:
        """Check if any open trades hit stop loss or take profit"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, symbol, side, entry_price, quantity, stop_loss, take_profit
            FROM paper_trades
            WHERE strategy = ? AND status = 'open'
        """, (strategy,))
        
        closed_trades = []
        
        for row in cursor.fetchall():
            trade_id, symbol, side, entry_price, quantity, stop_loss, take_profit = row
            
            if symbol not in current_prices:
                continue
            
            price = current_prices[symbol]
            exit_reason = None
            
            if side == 'buy':
                if stop_loss and price <= stop_loss:
                    exit_reason = 'stop_loss'
                elif take_profit and price >= take_profit:
                    exit_reason = 'take_profit'
            else:  # sell/short
                if stop_loss and price >= stop_loss:
                    exit_reason = 'stop_loss'
                elif take_profit and price <= take_profit:
                    exit_reason = 'take_profit'
            
            if exit_reason:
                conn.close()
                result = self.record_exit(strategy, symbol, price, exit_reason)
                if result:
                    closed_trades.append(result)
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
        
        conn.close()
        return closed_trades
    
    def _update_equity_curve(self, strategy: str, pnl: float):
        """Update equity curve after trade"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get latest equity
        cursor.execute("""
            SELECT equity, peak_equity FROM equity_curve
            WHERE strategy = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (strategy,))
        
        row = cursor.fetchone()
        if row:
            equity, peak_equity = row
            new_equity = equity + pnl
            new_peak = max(peak_equity, new_equity)
            drawdown = (new_peak - new_equity) / new_peak if new_peak > 0 else 0
            
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute("""
                INSERT INTO equity_curve (timestamp, strategy, equity, peak_equity, drawdown)
                VALUES (?, ?, ?, ?, ?)
            """, (now, strategy, new_equity, new_peak, drawdown))
            
            conn.commit()
        
        conn.close()
    
    def get_strategy_metrics(self, strategy: str) -> Optional[StrategyMetrics]:
        """Calculate comprehensive metrics for a strategy"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get trade statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_trades,
                SUM(CASE WHEN status != 'open' THEN 1 ELSE 0 END) as closed,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 AND status != 'open' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN status != 'open' THEN pnl ELSE 0 END) as total_pnl,
                AVG(CASE WHEN status != 'open' THEN pnl ELSE NULL END) as avg_pnl,
                AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as avg_loss
            FROM paper_trades
            WHERE strategy = ?
        """, (strategy,))
        
        row = cursor.fetchone()
        if not row or row[0] == 0:
            conn.close()
            return None
        
        total, open_trades, closed, wins, losses, total_pnl, avg_pnl, avg_win, avg_loss = row
        
        # Handle None values
        wins = wins or 0
        losses = losses or 0
        total_pnl = total_pnl or 0
        avg_pnl = avg_pnl or 0
        avg_win = avg_win or 0
        avg_loss = avg_loss or 0
        
        win_rate = wins / closed if closed > 0 else 0
        profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else 0
        
        # Get returns for Sharpe calculation
        cursor.execute("""
            SELECT pnl_pct FROM paper_trades
            WHERE strategy = ? AND status != 'open' AND pnl_pct IS NOT NULL
        """, (strategy,))
        
        returns = [r[0] for r in cursor.fetchall()]
        
        if len(returns) > 1:
            returns_array = np.array(returns)
            sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
        else:
            sharpe = 0
        
        # Get max drawdown
        cursor.execute("""
            SELECT MAX(drawdown), drawdown FROM equity_curve
            WHERE strategy = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (strategy,))
        
        dd_row = cursor.fetchone()
        max_drawdown = dd_row[0] if dd_row and dd_row[0] else 0
        current_drawdown = dd_row[1] if dd_row and dd_row[1] else 0
        
        # Get validation status
        cursor.execute("""
            SELECT status FROM validation_status WHERE strategy = ?
        """, (strategy,))
        
        status_row = cursor.fetchone()
        validation_status = status_row[0] if status_row else 'pending'
        
        conn.close()
        
        return StrategyMetrics(
            strategy=strategy,
            total_trades=total,
            open_trades=open_trades or 0,
            closed_trades=closed or 0,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            validation_status=validation_status,
            last_updated=datetime.now(timezone.utc)
        )
    
    def check_go_no_go(self, strategy: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if strategy passes GO/NO-GO validation criteria.
        
        Returns:
            (passed: bool, details: dict)
        """
        metrics = self.get_strategy_metrics(strategy)
        
        if not metrics:
            return False, {'error': 'No metrics available'}
        
        checks = {
            'min_trades': {
                'required': self.criteria.min_trades,
                'actual': metrics.closed_trades,
                'passed': metrics.closed_trades >= self.criteria.min_trades
            },
            'win_rate': {
                'required': f"{self.criteria.min_win_rate:.0%}",
                'actual': f"{metrics.win_rate:.1%}",
                'passed': metrics.win_rate >= self.criteria.min_win_rate
            },
            'sharpe_ratio': {
                'required': self.criteria.min_sharpe,
                'actual': round(metrics.sharpe_ratio, 2),
                'passed': metrics.sharpe_ratio >= self.criteria.min_sharpe
            },
            'max_drawdown': {
                'required': f"<{self.criteria.max_drawdown:.0%}",
                'actual': f"{metrics.max_drawdown:.1%}",
                'passed': metrics.max_drawdown <= self.criteria.max_drawdown
            },
            'profit_factor': {
                'required': self.criteria.min_profit_factor,
                'actual': round(metrics.profit_factor, 2),
                'passed': metrics.profit_factor >= self.criteria.min_profit_factor
            }
        }
        
        all_passed = all(c['passed'] for c in checks.values())
        
        # Update validation status
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        status = 'passed' if all_passed else 'running'
        
        # Check if validation period ended
        cursor.execute("SELECT end_date FROM validation_status WHERE strategy = ?", (strategy,))
        row = cursor.fetchone()
        if row and row[0]:
            end_date = datetime.fromisoformat(row[0].replace('Z', '+00:00'))
            if datetime.now(timezone.utc) > end_date and not all_passed:
                status = 'failed'
        
        cursor.execute("""
            UPDATE validation_status
            SET total_trades = ?, win_rate = ?, sharpe_ratio = ?, 
                max_drawdown = ?, profit_factor = ?, status = ?, updated_at = ?
            WHERE strategy = ?
        """, (metrics.closed_trades, metrics.win_rate, metrics.sharpe_ratio,
              metrics.max_drawdown, metrics.profit_factor, status,
              datetime.now(timezone.utc).isoformat(), strategy))
        
        conn.commit()
        conn.close()
        
        return all_passed, {
            'strategy': strategy,
            'status': status,
            'checks': checks,
            'metrics': asdict(metrics)
        }
    
    def get_all_strategies_status(self) -> List[Dict]:
        """Get validation status for all strategies"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT strategy FROM paper_trades
            UNION
            SELECT strategy FROM validation_status
        """)
        
        strategies = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        results = []
        for strategy in strategies:
            passed, details = self.check_go_no_go(strategy)
            results.append(details)
        
        return results
    
    def get_open_positions(self, strategy: Optional[str] = None) -> List[Dict]:
        """Get all open paper positions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if strategy:
            cursor.execute("""
                SELECT * FROM paper_trades
                WHERE status = 'open' AND strategy = ?
            """, (strategy,))
        else:
            cursor.execute("SELECT * FROM paper_trades WHERE status = 'open'")
        
        columns = [desc[0] for desc in cursor.description]
        positions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return positions
    
    def generate_report(self) -> str:
        """Generate validation status report"""
        report = []
        report.append("=" * 60)
        report.append("PAPER TRADING VALIDATION REPORT")
        report.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        report.append("=" * 60)
        
        statuses = self.get_all_strategies_status()
        
        for status in statuses:
            strategy = status.get('strategy', 'Unknown')
            validation = status.get('status', 'unknown')
            checks = status.get('checks', {})
            
            emoji = "✅" if validation == 'passed' else "❌" if validation == 'failed' else "🔄"
            report.append(f"\n{emoji} {strategy.upper()}: {validation.upper()}")
            report.append("-" * 40)
            
            for check_name, check_data in checks.items():
                passed = "✓" if check_data['passed'] else "✗"
                report.append(f"  [{passed}] {check_name}: {check_data['actual']} (req: {check_data['required']})")
        
        report.append("\n" + "=" * 60)
        
        # Open positions
        open_pos = self.get_open_positions()
        report.append(f"\nOPEN POSITIONS: {len(open_pos)}")
        for pos in open_pos[:5]:  # Show first 5
            report.append(f"  - {pos['strategy']}: {pos['side'].upper()} {pos['symbol']} @ ${pos['entry_price']:.2f}")
        
        return "\n".join(report)


if __name__ == "__main__":
    print("=" * 60)
    print("PAPER TRADING TRACKER - TEST")
    print("=" * 60)
    
    tracker = PaperTradingTracker()
    
    # Start validation for test strategies
    for strategy in ['rsi2', 'macd_rsi', 'bollinger_squeeze']:
        tracker.start_validation(strategy)
    
    # Simulate some trades
    print("\nSimulating paper trades...")
    
    # RSI-2 trades
    tracker.record_entry('rsi2', 'SPY', 'buy', 450.0, 100, 0.85, 'RSI oversold', 445.0, 460.0)
    tracker.record_exit('rsi2', 'SPY', 455.0, 'take_profit')
    
    tracker.record_entry('rsi2', 'QQQ', 'buy', 380.0, 50, 0.75, 'RSI oversold', 375.0, 390.0)
    tracker.record_exit('rsi2', 'QQQ', 378.0, 'stop_loss')
    
    # Get metrics
    print("\nStrategy Metrics:")
    metrics = tracker.get_strategy_metrics('rsi2')
    if metrics:
        print(f"  Total Trades: {metrics.total_trades}")
        print(f"  Win Rate: {metrics.win_rate:.1%}")
        print(f"  Total P&L: ${metrics.total_pnl:.2f}")
        print(f"  Sharpe: {metrics.sharpe_ratio:.2f}")
    
    # GO/NO-GO check
    print("\nGO/NO-GO Check:")
    passed, details = tracker.check_go_no_go('rsi2')
    print(f"  Passed: {passed}")
    for check, data in details.get('checks', {}).items():
        status = "✓" if data['passed'] else "✗"
        print(f"  [{status}] {check}: {data['actual']} (req: {data['required']})")
    
    # Full report
    print("\n" + tracker.generate_report())
