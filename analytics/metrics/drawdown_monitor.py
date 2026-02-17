"""
Drawdown Monitor - Real-Time Drawdown Tracking and Alerts
========================================================

Advanced drawdown monitoring system with:
- Real-time drawdown calculation
- Maximum drawdown tracking
- Underwater curve analysis
- Duration-based drawdown metrics
- Alert triggering for drawdown thresholds
- Recovery time analysis

Author: Trading Bot System
Created: February 2026
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
import sqlite3
import json
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class DrawdownSnapshot:
    """Snapshot of current drawdown state"""
    timestamp: datetime
    portfolio_value: float
    peak_value: float
    current_drawdown: float
    max_drawdown: float
    drawdown_duration: int  # days
    max_drawdown_duration: int  # days
    underwater_days: int
    recovery_factor: float  # How close to recovery (1.0 = fully recovered)
    is_new_peak: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'peak_value': self.peak_value,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'drawdown_duration': self.drawdown_duration,
            'max_drawdown_duration': self.max_drawdown_duration,
            'underwater_days': self.underwater_days,
            'recovery_factor': self.recovery_factor,
            'is_new_peak': self.is_new_peak
        }


@dataclass
class DrawdownAlert:
    """Drawdown alert configuration"""
    alert_type: str
    threshold: float
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'


class DrawdownMonitor:
    """
    Comprehensive drawdown monitoring system for real-time portfolio tracking.

    Features:
    - Real-time drawdown calculation
    - Peak tracking and recovery analysis
    - Multiple alert thresholds
    - Historical drawdown analysis
    - Underwater curve tracking
    - Duration-based metrics
    """

    def __init__(
        self,
        db_path: str = None,
        alert_callback: Optional[Callable] = None
    ):
        """
        Initialize drawdown monitor.

        Args:
            db_path: Database path for persistence
            alert_callback: Callback function for alerts
        """
        self.db_path = db_path or "data/drawdown_monitor.db"
        self.alert_callback = alert_callback

        # Current state
        self.peak_value = 0.0
        self.all_time_peak = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_duration = 0
        self.current_drawdown_start = None
        self.max_drawdown_start = None
        self.max_drawdown_end = None

        # Historical tracking
        self.value_history: deque = deque(maxlen=10000)
        self.drawdown_history: List[DrawdownSnapshot] = []

        # Alert configuration
        self.alert_thresholds = [
            DrawdownAlert('moderate_drawdown', 0.05, 'Moderate drawdown: {:.1%}', 'low'),
            DrawdownAlert('significant_drawdown', 0.10, 'Significant drawdown: {:.1%}', 'medium'),
            DrawdownAlert('severe_drawdown', 0.15, 'Severe drawdown: {:.1%}', 'high'),
            DrawdownAlert('critical_drawdown', 0.15, 'CRITICAL drawdown: {:.1%}', 'critical'),
            DrawdownAlert('extended_underwater', 30, 'Extended underwater period: {} days', 'medium'),
            DrawdownAlert('very_long_underwater', 60, 'Very long underwater period: {} days', 'high')
        ]

        # Alert state tracking
        self.triggered_alerts = set()
        self.last_alert_check = datetime.now()

        self._init_database()
        self._load_historical_data()

        logger.info("Drawdown Monitor initialized")

    def _init_database(self):
        """Initialize database for drawdown tracking"""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Drawdown snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drawdown_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    portfolio_value REAL,
                    peak_value REAL,
                    current_drawdown REAL,
                    max_drawdown REAL,
                    drawdown_duration INTEGER,
                    max_drawdown_duration INTEGER,
                    underwater_days INTEGER,
                    recovery_factor REAL,
                    is_new_peak INTEGER
                )
            ''')

            # Drawdown periods table (complete drawdown cycles)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drawdown_periods (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_date TEXT,
                    end_date TEXT,
                    peak_value REAL,
                    trough_value REAL,
                    max_drawdown REAL,
                    duration_days INTEGER,
                    recovery_days INTEGER,
                    total_days INTEGER
                )
            ''')

            # Drawdown alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drawdown_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    alert_type TEXT,
                    message TEXT,
                    severity TEXT,
                    drawdown_value REAL,
                    threshold_value REAL,
                    acknowledged INTEGER DEFAULT 0
                )
            ''')

            conn.commit()

    def _load_historical_data(self):
        """Load historical drawdown data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load recent snapshots
                query = '''
                    SELECT * FROM drawdown_snapshots
                    WHERE timestamp >= datetime('now', '-30 days')
                    ORDER BY timestamp DESC
                    LIMIT 1000
                '''
                cursor = conn.cursor()
                cursor.execute(query)

                for row in cursor.fetchall():
                    snapshot = DrawdownSnapshot(
                        timestamp=datetime.fromisoformat(row[1]),
                        portfolio_value=row[2],
                        peak_value=row[3],
                        current_drawdown=row[4],
                        max_drawdown=row[5],
                        drawdown_duration=row[6],
                        max_drawdown_duration=row[7],
                        underwater_days=row[8],
                        recovery_factor=row[9],
                        is_new_peak=bool(row[10])
                    )
                    self.drawdown_history.append(snapshot)

                # Update current state from latest snapshot
                if self.drawdown_history:
                    latest = self.drawdown_history[-1]
                    self.peak_value = latest.peak_value
                    self.all_time_peak = max(s.peak_value for s in self.drawdown_history)
                    self.current_drawdown = latest.current_drawdown
                    self.max_drawdown = latest.max_drawdown

                logger.info(f"Loaded {len(self.drawdown_history)} drawdown records")

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")

    def update(self, portfolio_value: float) -> Dict:
        """
        Update drawdown calculations with new portfolio value.

        Args:
            portfolio_value: Current portfolio value

        Returns:
            Dictionary with current drawdown metrics
        """
        now = datetime.now()
        is_new_peak = False

        # Initialize peak if first value
        if self.peak_value == 0:
            self.peak_value = portfolio_value
            self.all_time_peak = portfolio_value
            is_new_peak = True

        # Update peak if new high
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.all_time_peak = max(self.all_time_peak, portfolio_value)
            is_new_peak = True

            # Reset drawdown if new peak
            if self.current_drawdown < 0:
                # Record drawdown period completion
                self._complete_drawdown_period()

            self.current_drawdown = 0.0
            self.current_drawdown_start = None

        # Calculate current drawdown
        if self.peak_value > 0:
            self.current_drawdown = (portfolio_value - self.peak_value) / self.peak_value

            # Track drawdown start
            if self.current_drawdown < 0 and self.current_drawdown_start is None:
                self.current_drawdown_start = now

            # Update maximum drawdown
            if self.current_drawdown < self.max_drawdown:
                self.max_drawdown = self.current_drawdown
                self.max_drawdown_start = self.current_drawdown_start
                self.max_drawdown_end = now

        # Calculate durations
        drawdown_duration = 0
        if self.current_drawdown_start:
            drawdown_duration = (now - self.current_drawdown_start).days

        underwater_days = self._calculate_underwater_days()
        recovery_factor = self._calculate_recovery_factor(portfolio_value)

        # Create snapshot
        snapshot = DrawdownSnapshot(
            timestamp=now,
            portfolio_value=portfolio_value,
            peak_value=self.peak_value,
            current_drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown,
            drawdown_duration=drawdown_duration,
            max_drawdown_duration=self.max_drawdown_duration,
            underwater_days=underwater_days,
            recovery_factor=recovery_factor,
            is_new_peak=is_new_peak
        )

        # Add to history
        self.value_history.append((now, portfolio_value))
        self.drawdown_history.append(snapshot)

        # Trim history if too long
        if len(self.drawdown_history) > 1000:
            self.drawdown_history = self.drawdown_history[-1000:]

        # Save to database
        self._save_snapshot(snapshot)

        # Check for alerts
        self._check_alerts(snapshot)

        return self._create_metrics_dict(snapshot)

    def _calculate_underwater_days(self) -> int:
        """Calculate total days underwater (below peak)"""
        if len(self.value_history) == 0:
            return 0

        underwater_days = 0
        peak_so_far = 0

        for timestamp, value in self.value_history:
            if value > peak_so_far:
                peak_so_far = value
            elif value < peak_so_far:
                underwater_days += 1

        return underwater_days

    def _calculate_recovery_factor(self, current_value: float) -> float:
        """Calculate how close to recovery (1.0 = fully recovered)"""
        if self.current_drawdown >= 0 or self.peak_value == 0:
            return 1.0

        # Calculate the recovery needed
        recovery_needed = self.peak_value - current_value
        total_drawdown = self.peak_value * abs(self.max_drawdown)

        if total_drawdown == 0:
            return 1.0

        recovery_made = total_drawdown - recovery_needed
        return max(0.0, min(1.0, recovery_made / total_drawdown))

    def _complete_drawdown_period(self):
        """Record completion of a drawdown period"""
        if (self.max_drawdown_start and self.max_drawdown_end and
            self.max_drawdown < 0):

            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    # Find the trough value
                    trough_value = self.peak_value * (1 + self.max_drawdown)
                    duration_days = (self.max_drawdown_end - self.max_drawdown_start).days
                    recovery_days = (datetime.now() - self.max_drawdown_end).days
                    total_days = duration_days + recovery_days

                    cursor.execute('''
                        INSERT INTO drawdown_periods
                        (start_date, end_date, peak_value, trough_value,
                         max_drawdown, duration_days, recovery_days, total_days)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        self.max_drawdown_start.isoformat(),
                        self.max_drawdown_end.isoformat(),
                        self.peak_value,
                        trough_value,
                        self.max_drawdown,
                        duration_days,
                        recovery_days,
                        total_days
                    ))

                    conn.commit()

                    logger.info(f"Recorded drawdown period: {self.max_drawdown:.1%} over {total_days} days")

            except Exception as e:
                logger.error(f"Error recording drawdown period: {e}")

        # Reset maximums for next cycle
        self.max_drawdown = 0.0
        self.max_drawdown_start = None
        self.max_drawdown_end = None

    def _check_alerts(self, snapshot: DrawdownSnapshot):
        """Check for drawdown alert conditions"""
        current_time = snapshot.timestamp

        for alert_config in self.alert_thresholds:
            alert_key = f"{alert_config.alert_type}_{current_time.date()}"

            # Skip if already triggered today
            if alert_key in self.triggered_alerts:
                continue

            should_trigger = False
            metric_value = 0.0

            if alert_config.alert_type.endswith('_drawdown'):
                # Drawdown threshold alerts
                if snapshot.current_drawdown < -alert_config.threshold:
                    should_trigger = True
                    metric_value = abs(snapshot.current_drawdown)

            elif alert_config.alert_type.endswith('_underwater'):
                # Underwater duration alerts
                if snapshot.underwater_days >= alert_config.threshold:
                    should_trigger = True
                    metric_value = snapshot.underwater_days

            if should_trigger:
                self._trigger_alert(alert_config, metric_value, snapshot)
                self.triggered_alerts.add(alert_key)

        # Clean old triggered alerts (keep only last 30 days)
        cutoff_date = current_time.date() - timedelta(days=30)
        self.triggered_alerts = {
            key for key in self.triggered_alerts
            if datetime.strptime(key.split('_')[-1], '%Y-%m-%d').date() >= cutoff_date
        }

    def _trigger_alert(self, alert_config: DrawdownAlert, metric_value: float, snapshot: DrawdownSnapshot):
        """Trigger a drawdown alert"""
        try:
            # Format message
            if alert_config.alert_type.endswith('_drawdown'):
                message = alert_config.message.format(metric_value)
            else:  # underwater alerts
                message = alert_config.message.format(int(metric_value))

            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO drawdown_alerts
                    (timestamp, alert_type, message, severity, drawdown_value, threshold_value)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.timestamp.isoformat(),
                    alert_config.alert_type,
                    message,
                    alert_config.severity,
                    metric_value,
                    alert_config.threshold
                ))
                conn.commit()

            # Call alert callback if provided
            if self.alert_callback:
                alert_data = {
                    'type': alert_config.alert_type,
                    'message': message,
                    'severity': alert_config.severity,
                    'metric_value': metric_value,
                    'threshold': alert_config.threshold,
                    'timestamp': snapshot.timestamp,
                    'portfolio_value': snapshot.portfolio_value,
                    'current_drawdown': snapshot.current_drawdown
                }
                self.alert_callback(alert_data)

            logger.warning(f"Drawdown Alert [{alert_config.severity.upper()}]: {message}")

        except Exception as e:
            logger.error(f"Error triggering alert: {e}")

    def _save_snapshot(self, snapshot: DrawdownSnapshot):
        """Save drawdown snapshot to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO drawdown_snapshots
                    (timestamp, portfolio_value, peak_value, current_drawdown,
                     max_drawdown, drawdown_duration, max_drawdown_duration,
                     underwater_days, recovery_factor, is_new_peak)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.timestamp.isoformat(),
                    snapshot.portfolio_value,
                    snapshot.peak_value,
                    snapshot.current_drawdown,
                    snapshot.max_drawdown,
                    snapshot.drawdown_duration,
                    snapshot.max_drawdown_duration,
                    snapshot.underwater_days,
                    snapshot.recovery_factor,
                    int(snapshot.is_new_peak)
                ))
                conn.commit()

        except Exception as e:
            logger.debug(f"Error saving drawdown snapshot: {e}")

    def _create_metrics_dict(self, snapshot: DrawdownSnapshot) -> Dict:
        """Create metrics dictionary from snapshot"""
        return {
            'current_drawdown': snapshot.current_drawdown,
            'max_drawdown': snapshot.max_drawdown,
            'peak_value': snapshot.peak_value,
            'drawdown_duration': snapshot.drawdown_duration,
            'underwater_days': snapshot.underwater_days,
            'recovery_factor': snapshot.recovery_factor,
            'is_new_peak': snapshot.is_new_peak,
            'all_time_peak': self.all_time_peak
        }

    def get_underwater_curve(self, days: int = 90) -> pd.DataFrame:
        """
        Get underwater curve (drawdown over time).

        Args:
            days: Number of days to include

        Returns:
            DataFrame with underwater curve data
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_snapshots = [
            s for s in self.drawdown_history
            if s.timestamp >= cutoff_date
        ]

        if not recent_snapshots:
            return pd.DataFrame()

        data = []
        for snapshot in recent_snapshots:
            data.append({
                'timestamp': snapshot.timestamp,
                'portfolio_value': snapshot.portfolio_value,
                'peak_value': snapshot.peak_value,
                'drawdown': snapshot.current_drawdown,
                'underwater': snapshot.current_drawdown < 0
            })

        return pd.DataFrame(data)

    def get_drawdown_statistics(self, days: int = 365) -> Dict:
        """
        Get comprehensive drawdown statistics.

        Args:
            days: Period for analysis

        Returns:
            Dictionary with drawdown statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get drawdown periods in the specified timeframe
                cursor.execute('''
                    SELECT * FROM drawdown_periods
                    WHERE start_date >= datetime('now', ?)
                ''', (f'-{days} days',))

                periods = cursor.fetchall()

                if not periods:
                    return {
                        'total_periods': 0,
                        'avg_drawdown': 0.0,
                        'avg_duration': 0,
                        'avg_recovery_days': 0,
                        'worst_drawdown': 0.0,
                        'longest_period': 0,
                        'total_underwater_days': 0
                    }

                # Calculate statistics
                drawdowns = [abs(p[5]) for p in periods]  # max_drawdown column
                durations = [p[6] for p in periods]  # duration_days column
                recovery_times = [p[7] for p in periods]  # recovery_days column
                total_days = [p[8] for p in periods]  # total_days column

                stats = {
                    'total_periods': len(periods),
                    'avg_drawdown': np.mean(drawdowns),
                    'avg_duration': np.mean(durations),
                    'avg_recovery_days': np.mean(recovery_times),
                    'worst_drawdown': max(drawdowns),
                    'longest_period': max(total_days),
                    'total_underwater_days': sum(total_days),
                    'median_drawdown': np.median(drawdowns),
                    'drawdown_frequency': len(periods) / (days / 365) if days >= 365 else len(periods),
                    'recovery_factor_avg': np.mean(recovery_times) / np.mean(durations) if durations else 0
                }

                return stats

        except Exception as e:
            logger.error(f"Error calculating drawdown statistics: {e}")
            return {}

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent drawdown alerts"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM drawdown_alerts
                    WHERE timestamp >= datetime('now', ?)
                    ORDER BY timestamp DESC
                ''', (f'-{hours} hours',))

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []

    def reset_daily(self):
        """Reset daily tracking metrics"""
        # Clear daily alert tracking
        today = datetime.now().date()
        self.triggered_alerts = {
            key for key in self.triggered_alerts
            if not key.endswith(str(today))
        }

        logger.info("Reset daily drawdown tracking")

    def set_alert_thresholds(self, thresholds: Dict[str, float]):
        """
        Update alert thresholds.

        Args:
            thresholds: Dictionary with threshold updates
        """
        for alert_config in self.alert_thresholds:
            if alert_config.alert_type in thresholds:
                alert_config.threshold = thresholds[alert_config.alert_type]

        logger.info(f"Updated alert thresholds: {thresholds}")


# Example usage
if __name__ == "__main__":
    def alert_handler(alert):
        print(f"DRAWDOWN ALERT: {alert['message']} (Severity: {alert['severity']})")

    monitor = DrawdownMonitor(alert_callback=alert_handler)

    # Simulate portfolio values with drawdown
    values = [10000, 10500, 10200, 9800, 9500, 9200, 9800, 10100, 10600, 11000]

    for value in values:
        metrics = monitor.update(value)
        print(f"Value: ${value:,.0f} | Drawdown: {metrics['current_drawdown']:.1%} | "
              f"Max DD: {metrics['max_drawdown']:.1%}")

    # Get statistics
    stats = monitor.get_drawdown_statistics()
    print(f"\nDrawdown Statistics: {stats}")

    # Get underwater curve
    underwater = monitor.get_underwater_curve()
    print(f"\nUnderwater periods: {len(underwater[underwater['underwater']]) if len(underwater) > 0 else 0}")