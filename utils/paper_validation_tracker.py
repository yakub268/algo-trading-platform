"""
Paper Trading Validation Tracker

Tracks paper trading performance against GO/NO-GO criteria.
Required validation period: 2 weeks with specific metrics.

GO/NO-GO Criteria:
- 100+ trades minimum
- 45%+ win rate (target: 68%)
- Sharpe ratio > 1.0
- Max drawdown < 15%

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PaperTracker')


@dataclass
class ValidationCriteria:
    """GO/NO-GO criteria for paper trading validation"""
    min_trades: int = 100
    min_win_rate: float = 0.45
    min_sharpe: float = 1.0
    max_drawdown: float = 0.15
    validation_days: int = 14


@dataclass
class TradeRecord:
    """Single trade record"""
    trade_id: str
    symbol: str
    strategy: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: Optional[float]
    shares: float
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl: float
    pnl_pct: float
    tier: Optional[str]  # For scaled exits
    status: str  # 'open', 'closed'


@dataclass
class DailyStats:
    """Daily performance statistics"""
    date: str
    trades: int
    wins: int
    losses: int
    pnl: float
    portfolio_value: float
    drawdown: float
    win_rate: float


@dataclass
class ValidationStatus:
    """Current validation status"""
    start_date: datetime
    current_date: datetime
    days_elapsed: int
    days_remaining: int
    
    # Metrics
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    total_pnl: float
    
    # GO/NO-GO
    trades_passed: bool
    win_rate_passed: bool
    sharpe_passed: bool
    drawdown_passed: bool
    overall_passed: bool
    
    # Recommendation
    recommendation: str


class PaperTradingValidator:
    """
    Validates paper trading performance before live deployment.
    
    Features:
    - Trade logging
    - Daily statistics
    - Real-time metrics calculation
    - GO/NO-GO criteria checking
    - Validation report generation
    """
    
    def __init__(
        self,
        db_path: str = "data/paper_validation.db",
        starting_balance: float = 10000.0,
        criteria: Optional[ValidationCriteria] = None
    ):
        """
        Initialize paper trading validator.
        
        Args:
            db_path: Path to SQLite database
            starting_balance: Initial capital
            criteria: Validation criteria (uses defaults if None)
        """
        self.db_path = db_path
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.criteria = criteria or ValidationCriteria()
        
        self.start_date: Optional[datetime] = None
        self.high_water_mark = starting_balance
        self.max_drawdown = 0.0
        
        self._init_database()
        self._load_state()
        
        logger.info(f"PaperTradingValidator initialized")
        logger.info(f"Starting balance: ${starting_balance:,.2f}")
        logger.info(f"Validation period: {self.criteria.validation_days} days")
    
    def _init_database(self):
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                strategy TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                shares REAL,
                entry_time TEXT,
                exit_time TEXT,
                pnl REAL,
                pnl_pct REAL,
                tier TEXT,
                status TEXT
            )
        ''')
        
        # Daily stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                pnl REAL,
                portfolio_value REAL,
                drawdown REAL,
                win_rate REAL
            )
        ''')
        
        # State table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_state(self):
        """Load state from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load start date
        cursor.execute("SELECT value FROM state WHERE key = 'start_date'")
        row = cursor.fetchone()
        if row:
            self.start_date = datetime.fromisoformat(row[0])
        
        # Load current balance
        cursor.execute("SELECT value FROM state WHERE key = 'current_balance'")
        row = cursor.fetchone()
        if row:
            self.current_balance = float(row[0])
        
        # Load high water mark
        cursor.execute("SELECT value FROM state WHERE key = 'high_water_mark'")
        row = cursor.fetchone()
        if row:
            self.high_water_mark = float(row[0])
        
        # Load max drawdown
        cursor.execute("SELECT value FROM state WHERE key = 'max_drawdown'")
        row = cursor.fetchone()
        if row:
            self.max_drawdown = float(row[0])
        
        conn.close()
    
    def _save_state(self):
        """Save state to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        state_items = [
            ('start_date', self.start_date.isoformat() if self.start_date else None),
            ('current_balance', str(self.current_balance)),
            ('high_water_mark', str(self.high_water_mark)),
            ('max_drawdown', str(self.max_drawdown)),
        ]
        
        for key, value in state_items:
            if value:
                cursor.execute('''
                    INSERT OR REPLACE INTO state (key, value)
                    VALUES (?, ?)
                ''', (key, value))
        
        conn.commit()
        conn.close()
    
    def start_validation(self):
        """Start the validation period."""
        self.start_date = datetime.now()
        self._save_state()
        logger.info(f"Validation started: {self.start_date.isoformat()}")
    
    def log_trade(
        self,
        symbol: str,
        strategy: str,
        side: str,
        entry_price: float,
        shares: float,
        exit_price: Optional[float] = None,
        pnl: float = 0.0,
        tier: Optional[str] = None
    ) -> str:
        """
        Log a trade.
        
        Args:
            symbol: Trading symbol
            strategy: Strategy name
            side: 'buy' or 'sell'
            entry_price: Entry price
            shares: Number of shares
            exit_price: Exit price (if closed)
            pnl: Profit/loss
            tier: Tier for scaled exits
            
        Returns:
            Trade ID
        """
        if not self.start_date:
            self.start_validation()
        
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        entry_time = datetime.now()
        
        status = "closed" if exit_price else "open"
        pnl_pct = (exit_price - entry_price) / entry_price if exit_price else 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (trade_id, symbol, strategy, side, entry_price,
                               exit_price, shares, entry_time, exit_time,
                               pnl, pnl_pct, tier, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_id, symbol, strategy, side, entry_price,
            exit_price, shares, entry_time.isoformat(),
            entry_time.isoformat() if exit_price else None,
            pnl, pnl_pct, tier, status
        ))
        
        conn.commit()
        conn.close()
        
        # Update balance
        if exit_price:
            self.current_balance += pnl
            self._update_drawdown()
            self._save_state()
        
        logger.info(f"Trade logged: {trade_id} ({status})")
        return trade_id
    
    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float
    ):
        """Close an open trade."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        pnl_pct = pnl / self.current_balance
        exit_time = datetime.now()
        
        cursor.execute('''
            UPDATE trades
            SET exit_price = ?, exit_time = ?, pnl = ?, pnl_pct = ?, status = 'closed'
            WHERE trade_id = ?
        ''', (exit_price, exit_time.isoformat(), pnl, pnl_pct, trade_id))
        
        conn.commit()
        conn.close()
        
        # Update balance
        self.current_balance += pnl
        self._update_drawdown()
        self._save_state()
        
        logger.info(f"Trade closed: {trade_id}, P&L: ${pnl:.2f}")
    
    def _update_drawdown(self):
        """Update high water mark and max drawdown."""
        if self.current_balance > self.high_water_mark:
            self.high_water_mark = self.current_balance
        
        current_dd = (self.high_water_mark - self.current_balance) / self.high_water_mark
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
    
    def get_trades(self, status: Optional[str] = None) -> List[Dict]:
        """Get all trades, optionally filtered by status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if status:
            cursor.execute("SELECT * FROM trades WHERE status = ?", (status,))
        else:
            cursor.execute("SELECT * FROM trades")
        
        columns = [desc[0] for desc in cursor.description]
        trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return trades
    
    def calculate_metrics(self) -> Dict:
        """Calculate current performance metrics."""
        trades = self.get_trades(status='closed')
        
        if not trades:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'current_balance': self.current_balance,
                'total_return': 0.0
            }
        
        # Basic stats
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        total_pnl = sum(t['pnl'] for t in trades)
        
        # Win rate
        win_rate = len(wins) / len(trades) if trades else 0
        
        # Returns for Sharpe
        returns = [t['pnl_pct'] for t in trades]
        
        # Sharpe ratio (annualized, assuming daily)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Total return
        total_return = (self.current_balance - self.starting_balance) / self.starting_balance
        
        return {
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(trades) if trades else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': self.max_drawdown,
            'current_balance': self.current_balance,
            'total_return': total_return
        }
    
    def get_validation_status(self) -> ValidationStatus:
        """Get current validation status with GO/NO-GO assessment."""
        if not self.start_date:
            self.start_validation()
        
        metrics = self.calculate_metrics()
        
        now = datetime.now()
        days_elapsed = (now - self.start_date).days
        days_remaining = max(0, self.criteria.validation_days - days_elapsed)
        
        # Check criteria
        trades_passed = metrics['total_trades'] >= self.criteria.min_trades
        win_rate_passed = metrics['win_rate'] >= self.criteria.min_win_rate
        sharpe_passed = metrics['sharpe_ratio'] >= self.criteria.min_sharpe
        drawdown_passed = metrics['max_drawdown'] <= self.criteria.max_drawdown
        
        overall_passed = all([
            trades_passed, win_rate_passed, sharpe_passed, drawdown_passed
        ])
        
        # Recommendation
        if days_remaining > 0:
            if overall_passed:
                recommendation = "ON TRACK - Continue validation"
            else:
                failed = []
                if not trades_passed:
                    failed.append(f"trades ({metrics['total_trades']}/{self.criteria.min_trades})")
                if not win_rate_passed:
                    failed.append(f"win rate ({metrics['win_rate']:.1%}/{self.criteria.min_win_rate:.0%})")
                if not sharpe_passed:
                    failed.append(f"Sharpe ({metrics['sharpe_ratio']:.2f}/{self.criteria.min_sharpe})")
                if not drawdown_passed:
                    failed.append(f"drawdown ({metrics['max_drawdown']:.1%}/{self.criteria.max_drawdown:.0%})")
                recommendation = f"NEEDS IMPROVEMENT: {', '.join(failed)}"
        else:
            if overall_passed:
                recommendation = "✅ GO - Ready for live trading"
            else:
                recommendation = "❌ NO-GO - Do not deploy live"
        
        return ValidationStatus(
            start_date=self.start_date,
            current_date=now,
            days_elapsed=days_elapsed,
            days_remaining=days_remaining,
            total_trades=metrics['total_trades'],
            win_rate=metrics['win_rate'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            total_pnl=metrics['total_pnl'],
            trades_passed=trades_passed,
            win_rate_passed=win_rate_passed,
            sharpe_passed=sharpe_passed,
            drawdown_passed=drawdown_passed,
            overall_passed=overall_passed,
            recommendation=recommendation
        )
    
    def generate_report(self) -> str:
        """Generate validation report."""
        status = self.get_validation_status()
        metrics = self.calculate_metrics()
        
        report = f"""
{'='*60}
PAPER TRADING VALIDATION REPORT
{'='*60}

📅 VALIDATION PERIOD
   Start Date: {status.start_date.strftime('%Y-%m-%d')}
   Days Elapsed: {status.days_elapsed} / {self.criteria.validation_days}
   Days Remaining: {status.days_remaining}

📊 PERFORMANCE METRICS
   Total Trades: {metrics['total_trades']}
   Wins/Losses: {metrics['wins']}/{metrics['losses']}
   Win Rate: {metrics['win_rate']:.1%}
   Total P&L: ${metrics['total_pnl']:+,.2f}
   Total Return: {metrics['total_return']:+.1%}
   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
   Max Drawdown: {metrics['max_drawdown']:.1%}
   Current Balance: ${metrics['current_balance']:,.2f}

✅ GO/NO-GO CRITERIA
   Trades (≥{self.criteria.min_trades}): {'✅ PASS' if status.trades_passed else '❌ FAIL'} ({metrics['total_trades']})
   Win Rate (≥{self.criteria.min_win_rate:.0%}): {'✅ PASS' if status.win_rate_passed else '❌ FAIL'} ({metrics['win_rate']:.1%})
   Sharpe (≥{self.criteria.min_sharpe}): {'✅ PASS' if status.sharpe_passed else '❌ FAIL'} ({metrics['sharpe_ratio']:.2f})
   Drawdown (≤{self.criteria.max_drawdown:.0%}): {'✅ PASS' if status.drawdown_passed else '❌ FAIL'} ({metrics['max_drawdown']:.1%})

🎯 RECOMMENDATION
   {status.recommendation}

{'='*60}
"""
        return report
    
    def print_report(self):
        """Print validation report to console."""
        print(self.generate_report())


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PAPER TRADING VALIDATOR TEST")
    print("=" * 60)
    
    # Create validator
    validator = PaperTradingValidator(
        db_path="data/test_validation.db",
        starting_balance=10000.0
    )
    
    # Simulate some trades
    print("\n📈 Simulating trades...")
    
    import random
    random.seed(42)
    
    for i in range(50):
        # Simulate win/loss (68% win rate target)
        is_win = random.random() < 0.68
        
        entry = 500 + random.uniform(-10, 10)
        if is_win:
            pnl = random.uniform(10, 50)
            exit_price = entry + pnl / 10
        else:
            pnl = -random.uniform(10, 30)
            exit_price = entry + pnl / 10
        
        validator.log_trade(
            symbol="SPY",
            strategy="RSI2-Improved",
            side="buy",
            entry_price=entry,
            shares=10,
            exit_price=exit_price,
            pnl=pnl
        )
    
    # Print report
    validator.print_report()
