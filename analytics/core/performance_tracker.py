"""
Performance Tracker - Comprehensive Real-Time Performance Monitoring
====================================================================

Central hub for all performance tracking including:
- Real-time portfolio value tracking
- Multi-timeframe performance analysis
- Rolling performance windows
- Strategy attribution
- Risk metrics monitoring
- Alert triggering

Author: Trading Bot System
Created: February 2026
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable
from decimal import Decimal
import threading
import time
import sqlite3
import json
from collections import defaultdict, deque

from .pnl_calculator import PnLCalculator, PnLSnapshot
from ..metrics.performance_ratios import PerformanceRatios
from ..metrics.drawdown_monitor import DrawdownMonitor
from ..metrics.trade_statistics import TradeStatistics

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Comprehensive real-time performance tracker that coordinates all analytics modules
    and provides centralized performance monitoring for the trading system.

    Features:
    - Real-time performance updates
    - Multi-timeframe analysis (1d, 7d, 30d, 90d, 1y)
    - Rolling window calculations
    - Strategy attribution analysis
    - Performance alerts and notifications
    - Historical performance data
    - Risk metrics integration
    """

    def __init__(
        self,
        starting_capital: float = 10000.0,
        db_path: str = None,
        update_interval: int = 60,
        alert_callback: Optional[Callable] = None
    ):
        """
        Initialize performance tracker.

        Args:
            starting_capital: Starting portfolio value
            db_path: Database path for persistence
            update_interval: Update interval in seconds
            alert_callback: Callback function for alerts
        """
        self.starting_capital = Decimal(str(starting_capital))
        self.db_path = db_path or "data/performance_tracker.db"
        self.update_interval = update_interval
        self.alert_callback = alert_callback

        # Core components
        self.pnl_calculator = PnLCalculator(self.db_path.replace('.db', '_pnl.db'))
        self.performance_ratios = PerformanceRatios()
        self.drawdown_monitor = DrawdownMonitor()
        self.trade_statistics = TradeStatistics()

        # Performance data storage
        self.performance_history: deque = deque(maxlen=10000)  # Keep last 10k snapshots
        self.rolling_windows = {
            '1d': deque(maxlen=1440),    # 1 minute intervals for 24 hours
            '7d': deque(maxlen=10080),   # 1 minute intervals for 7 days
            '30d': deque(maxlen=43200),  # 1 minute intervals for 30 days
            '90d': deque(maxlen=129600), # 1 minute intervals for 90 days
            '1y': deque(maxlen=525600)   # 1 minute intervals for 1 year
        }

        # Current performance state
        self.current_portfolio_value = self.starting_capital
        self.daily_starting_value = self.starting_capital
        self.last_update = datetime.now()

        # Thread safety
        self._lock = threading.RLock()

        # Auto-update thread
        self._running = False
        self._update_thread = None

        # Alert thresholds
        self.alert_thresholds = {
            'max_drawdown': 0.05,      # 5% max drawdown
            'daily_loss': 0.02,        # 2% daily loss
            'position_concentration': 0.20,  # 20% max position size
            'volatility_spike': 0.30,  # 30% volatility increase
        }

        self._init_database()
        self._load_historical_data()

        logger.info(f"Performance Tracker initialized with ${starting_capital:,.2f}")

    def _init_database(self):
        """Initialize database for performance tracking"""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Performance snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    portfolio_value REAL,
                    daily_return REAL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    max_drawdown REAL,
                    current_drawdown REAL,
                    volatility REAL,
                    position_count INTEGER,
                    open_trades INTEGER,
                    realized_pnl REAL,
                    unrealized_pnl REAL,
                    by_strategy TEXT,
                    by_platform TEXT,
                    by_market TEXT
                )
            ''')

            # Rolling performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rolling_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    window_period TEXT,
                    returns_1d REAL,
                    returns_7d REAL,
                    returns_30d REAL,
                    returns_90d REAL,
                    returns_1y REAL,
                    volatility_1d REAL,
                    volatility_7d REAL,
                    volatility_30d REAL,
                    volatility_90d REAL,
                    volatility_1y REAL,
                    sharpe_1d REAL,
                    sharpe_7d REAL,
                    sharpe_30d REAL,
                    sharpe_90d REAL,
                    sharpe_1y REAL,
                    max_dd_1d REAL,
                    max_dd_7d REAL,
                    max_dd_30d REAL,
                    max_dd_90d REAL,
                    max_dd_1y REAL
                )
            ''')

            # Performance alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    alert_type TEXT,
                    message TEXT,
                    severity TEXT,
                    metric_value REAL,
                    threshold_value REAL,
                    acknowledged INTEGER DEFAULT 0
                )
            ''')

            conn.commit()

    def _load_historical_data(self):
        """Load historical performance data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load recent performance snapshots
                query = '''
                    SELECT * FROM performance_snapshots
                    WHERE timestamp >= datetime('now', '-30 days')
                    ORDER BY timestamp DESC
                    LIMIT 1000
                '''
                df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])

                if len(df) > 0:
                    # Convert to performance history
                    for _, row in df.iterrows():
                        snapshot_data = {
                            'timestamp': row['timestamp'],
                            'portfolio_value': row['portfolio_value'],
                            'daily_return': row['daily_return'],
                            'total_return': row['total_return'],
                            'sharpe_ratio': row['sharpe_ratio'],
                            'sortino_ratio': row['sortino_ratio'],
                            'max_drawdown': row['max_drawdown'],
                            'current_drawdown': row['current_drawdown'],
                            'volatility': row['volatility'],
                            'position_count': row['position_count'],
                            'open_trades': row['open_trades'],
                            'realized_pnl': row['realized_pnl'],
                            'unrealized_pnl': row['unrealized_pnl']
                        }

                        # Parse JSON fields
                        try:
                            snapshot_data['by_strategy'] = json.loads(row['by_strategy'] or '{}')
                            snapshot_data['by_platform'] = json.loads(row['by_platform'] or '{}')
                            snapshot_data['by_market'] = json.loads(row['by_market'] or '{}')
                        except Exception as e:
                            logger.debug(f"Error parsing JSON fields in performance data: {e}")
                            snapshot_data['by_strategy'] = {}
                            snapshot_data['by_platform'] = {}
                            snapshot_data['by_market'] = {}

                        self.performance_history.append(snapshot_data)

                    logger.info(f"Loaded {len(df)} historical performance records")

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")

    def start_monitoring(self):
        """Start real-time performance monitoring"""
        with self._lock:
            if self._running:
                logger.warning("Performance monitoring already running")
                return

            self._running = True
            self._update_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._update_thread.start()

            logger.info("Started real-time performance monitoring")

    def stop_monitoring(self):
        """Stop real-time performance monitoring"""
        with self._lock:
            self._running = False
            if self._update_thread and self._update_thread.is_alive():
                self._update_thread.join(timeout=5)

            logger.info("Stopped performance monitoring")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                self.update_performance()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(min(self.update_interval, 60))  # Don't retry too quickly

    def update_performance(self, force_update: bool = False) -> Dict:
        """
        Update all performance metrics.

        Args:
            force_update: Force update even if within interval

        Returns:
            Current performance snapshot
        """
        with self._lock:
            now = datetime.now()

            # Check if update is needed
            if not force_update:
                time_since_update = (now - self.last_update).total_seconds()
                if time_since_update < self.update_interval:
                    return self.get_current_performance()

            # Get current P&L snapshot
            pnl_snapshot = self.pnl_calculator.get_current_pnl()

            # Calculate portfolio value
            portfolio_value = self.starting_capital + pnl_snapshot.total_pnl
            self.current_portfolio_value = portfolio_value

            # Calculate returns
            daily_return = self._calculate_daily_return(portfolio_value)
            total_return = float((portfolio_value / self.starting_capital) - 1)

            # Get rolling performance
            rolling_perf = self._calculate_rolling_performance()

            # Calculate risk metrics
            sharpe_ratio = rolling_perf.get('sharpe_30d', 0)
            sortino_ratio = rolling_perf.get('sortino_30d', 0)
            volatility = rolling_perf.get('volatility_30d', 0)

            # Update drawdown monitor
            drawdown_info = self.drawdown_monitor.update(portfolio_value)

            # Create performance snapshot
            performance_snapshot = {
                'timestamp': now,
                'portfolio_value': float(portfolio_value),
                'daily_return': daily_return,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': drawdown_info.get('max_drawdown', 0),
                'current_drawdown': drawdown_info.get('current_drawdown', 0),
                'volatility': volatility,
                'position_count': pnl_snapshot.position_count,
                'open_trades': len(self.pnl_calculator.positions),
                'realized_pnl': float(pnl_snapshot.realized_pnl),
                'unrealized_pnl': float(pnl_snapshot.unrealized_pnl),
                'by_strategy': pnl_snapshot.by_strategy,
                'by_platform': pnl_snapshot.by_platform,
                'by_market': pnl_snapshot.by_market,
                'rolling_performance': rolling_perf
            }

            # Add to history
            self.performance_history.append(performance_snapshot)

            # Update rolling windows
            self._update_rolling_windows(performance_snapshot)

            # Save to database
            self._save_performance_snapshot(performance_snapshot)

            # Check for alerts
            self._check_performance_alerts(performance_snapshot)

            self.last_update = now

            return performance_snapshot

    def _calculate_daily_return(self, current_value: Decimal) -> float:
        """Calculate daily return"""
        if self.daily_starting_value > 0:
            return float((current_value / self.daily_starting_value) - 1)
        return 0.0

    def _calculate_rolling_performance(self) -> Dict[str, float]:
        """Calculate rolling performance metrics for different time windows"""
        rolling_perf = {}

        if len(self.performance_history) < 2:
            return rolling_perf

        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(self.performance_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        # Calculate returns
        df['returns'] = df['portfolio_value'].pct_change()

        for period in ['1d', '7d', '30d', '90d', '1y']:
            try:
                # Get data for the period
                if period == '1d':
                    period_data = df.last('1D')
                elif period == '7d':
                    period_data = df.last('7D')
                elif period == '30d':
                    period_data = df.last('30D')
                elif period == '90d':
                    period_data = df.last('90D')
                else:  # 1y
                    period_data = df.last('365D')

                if len(period_data) < 2:
                    continue

                returns = period_data['returns'].dropna()

                if len(returns) > 0:
                    # Calculate metrics
                    period_return = (period_data['portfolio_value'].iloc[-1] /
                                   period_data['portfolio_value'].iloc[0]) - 1
                    volatility = returns.std() * np.sqrt(252 * 24 * 60)  # Annualized for minute data

                    # Sharpe ratio (assuming 0% risk-free rate)
                    if volatility > 0:
                        sharpe = (returns.mean() * 252 * 24 * 60) / volatility
                    else:
                        sharpe = 0

                    # Sortino ratio
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0:
                        downside_std = downside_returns.std() * np.sqrt(252 * 24 * 60)
                        sortino = (returns.mean() * 252 * 24 * 60) / downside_std if downside_std > 0 else 0
                    else:
                        sortino = 0

                    # Maximum drawdown
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    max_drawdown = drawdown.min()

                    rolling_perf.update({
                        f'returns_{period}': period_return,
                        f'volatility_{period}': volatility,
                        f'sharpe_{period}': sharpe,
                        f'sortino_{period}': sortino,
                        f'max_dd_{period}': max_drawdown
                    })

            except Exception as e:
                logger.debug(f"Error calculating {period} performance: {e}")

        return rolling_perf

    def _update_rolling_windows(self, snapshot: Dict):
        """Update rolling window data"""
        for window in self.rolling_windows.keys():
            self.rolling_windows[window].append({
                'timestamp': snapshot['timestamp'],
                'portfolio_value': snapshot['portfolio_value'],
                'daily_return': snapshot['daily_return'],
                'sharpe_ratio': snapshot['sharpe_ratio'],
                'max_drawdown': snapshot['max_drawdown']
            })

    def _check_performance_alerts(self, snapshot: Dict):
        """Check for performance alert conditions"""
        alerts = []

        # Maximum drawdown alert
        if snapshot['max_drawdown'] < -self.alert_thresholds['max_drawdown']:
            alerts.append({
                'type': 'max_drawdown',
                'message': f"Maximum drawdown exceeded: {snapshot['max_drawdown']:.2%}",
                'severity': 'high',
                'metric_value': snapshot['max_drawdown'],
                'threshold': -self.alert_thresholds['max_drawdown']
            })

        # Daily loss alert
        if snapshot['daily_return'] < -self.alert_thresholds['daily_loss']:
            alerts.append({
                'type': 'daily_loss',
                'message': f"Daily loss exceeded threshold: {snapshot['daily_return']:.2%}",
                'severity': 'medium',
                'metric_value': snapshot['daily_return'],
                'threshold': -self.alert_thresholds['daily_loss']
            })

        # Position concentration alert
        for strategy, pnl in snapshot['by_strategy'].items():
            strategy_allocation = abs(float(pnl)) / float(snapshot['portfolio_value'])
            if strategy_allocation > self.alert_thresholds['position_concentration']:
                alerts.append({
                    'type': 'position_concentration',
                    'message': f"High concentration in {strategy}: {strategy_allocation:.1%}",
                    'severity': 'medium',
                    'metric_value': strategy_allocation,
                    'threshold': self.alert_thresholds['position_concentration']
                })

        # Volatility spike alert
        if len(self.performance_history) > 1:
            prev_vol = self.performance_history[-2].get('volatility', 0)
            current_vol = snapshot['volatility']
            if prev_vol > 0 and (current_vol / prev_vol) > (1 + self.alert_thresholds['volatility_spike']):
                alerts.append({
                    'type': 'volatility_spike',
                    'message': f"Volatility spike detected: {(current_vol/prev_vol-1):.1%} increase",
                    'severity': 'medium',
                    'metric_value': current_vol / prev_vol - 1,
                    'threshold': self.alert_thresholds['volatility_spike']
                })

        # Process alerts
        for alert in alerts:
            self._trigger_alert(alert)

    def _trigger_alert(self, alert: Dict):
        """Trigger a performance alert"""
        try:
            # Save alert to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_alerts
                    (timestamp, alert_type, message, severity, metric_value, threshold_value)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    alert['type'],
                    alert['message'],
                    alert['severity'],
                    alert['metric_value'],
                    alert['threshold']
                ))
                conn.commit()

            # Call alert callback if provided
            if self.alert_callback:
                self.alert_callback(alert)

            logger.warning(f"Performance Alert [{alert['severity'].upper()}]: {alert['message']}")

        except Exception as e:
            logger.error(f"Error triggering alert: {e}")

    def _save_performance_snapshot(self, snapshot: Dict):
        """Save performance snapshot to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_snapshots
                    (timestamp, portfolio_value, daily_return, total_return,
                     sharpe_ratio, sortino_ratio, max_drawdown, current_drawdown,
                     volatility, position_count, open_trades, realized_pnl,
                     unrealized_pnl, by_strategy, by_platform, by_market)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot['timestamp'].isoformat(),
                    snapshot['portfolio_value'],
                    snapshot['daily_return'],
                    snapshot['total_return'],
                    snapshot['sharpe_ratio'],
                    snapshot['sortino_ratio'],
                    snapshot['max_drawdown'],
                    snapshot['current_drawdown'],
                    snapshot['volatility'],
                    snapshot['position_count'],
                    snapshot['open_trades'],
                    snapshot['realized_pnl'],
                    snapshot['unrealized_pnl'],
                    json.dumps(snapshot['by_strategy']),
                    json.dumps(snapshot['by_platform']),
                    json.dumps(snapshot['by_market'])
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Error saving performance snapshot: {e}")

    def get_current_performance(self) -> Dict:
        """Get current performance metrics"""
        if self.performance_history:
            return dict(self.performance_history[-1])
        else:
            # Return default performance
            return {
                'timestamp': datetime.now(),
                'portfolio_value': float(self.current_portfolio_value),
                'daily_return': 0.0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'current_drawdown': 0.0,
                'volatility': 0.0,
                'position_count': 0,
                'open_trades': 0,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'by_strategy': {},
                'by_platform': {},
                'by_market': {}
            }

    def get_rolling_performance(self, period: str = '30d') -> Dict:
        """Get rolling performance for specified period"""
        if period not in self.rolling_windows:
            raise ValueError(f"Invalid period: {period}")

        window_data = list(self.rolling_windows[period])
        if not window_data:
            return {}

        # Calculate rolling metrics
        values = [d['portfolio_value'] for d in window_data]
        returns = [d['daily_return'] for d in window_data if d['daily_return'] is not None]

        if len(values) < 2:
            return {}

        # Period return
        period_return = (values[-1] / values[0]) - 1

        # Volatility
        if returns:
            volatility = np.std(returns) * np.sqrt(252)
        else:
            volatility = 0

        # Sharpe ratio (annualized)
        if returns and volatility > 0:
            sharpe = np.mean(returns) * np.sqrt(252) / volatility
        else:
            sharpe = 0

        # Maximum drawdown
        cumulative = np.cumprod(1 + np.array([0] + returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        return {
            'period': period,
            'start_value': values[0],
            'end_value': values[-1],
            'period_return': period_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'data_points': len(window_data)
        }

    def get_strategy_performance(self) -> Dict:
        """Get performance breakdown by strategy"""
        return self.pnl_calculator.get_strategy_attribution()

    def reset_daily_tracking(self):
        """Reset daily tracking (call at market open)"""
        with self._lock:
            self.daily_starting_value = self.current_portfolio_value
            self.drawdown_monitor.reset_daily()
            logger.info(f"Reset daily tracking - Starting value: ${self.daily_starting_value:,.2f}")

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent performance alerts"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM performance_alerts
                    WHERE timestamp >= datetime('now', ?)
                    ORDER BY timestamp DESC
                ''', (f'-{hours} hours',))

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []

    def acknowledge_alert(self, alert_id: int):
        """Acknowledge a performance alert"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE performance_alerts SET acknowledged = 1 WHERE id = ?',
                    (alert_id,)
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")

    def get_performance_report(self, period_days: int = 30) -> Dict:
        """Generate comprehensive performance report"""
        current_perf = self.get_current_performance()
        rolling_perf = self.get_rolling_performance('30d')
        strategy_perf = self.get_strategy_performance()
        recent_alerts = self.get_recent_alerts(24 * period_days)

        return {
            'report_date': datetime.now().isoformat(),
            'period_days': period_days,
            'current_performance': current_perf,
            'rolling_performance': rolling_perf,
            'strategy_attribution': strategy_perf,
            'recent_alerts': recent_alerts,
            'summary': {
                'total_return': current_perf['total_return'],
                'sharpe_ratio': current_perf['sharpe_ratio'],
                'max_drawdown': current_perf['max_drawdown'],
                'position_count': current_perf['position_count'],
                'alert_count': len(recent_alerts)
            }
        }


# Example usage
if __name__ == "__main__":
    def alert_handler(alert):
        print(f"ALERT: {alert['message']}")

    tracker = PerformanceTracker(
        starting_capital=10000,
        update_interval=60,
        alert_callback=alert_handler
    )

    # Start monitoring
    tracker.start_monitoring()

    # Get current performance
    perf = tracker.get_current_performance()
    print(f"Portfolio Value: ${perf['portfolio_value']:,.2f}")
    print(f"Total Return: {perf['total_return']:.2%}")
    print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")

    # Get rolling performance
    rolling = tracker.get_rolling_performance('30d')
    print(f"30-day Return: {rolling.get('period_return', 0):.2%}")

    # Stop monitoring
    # tracker.stop_monitoring()