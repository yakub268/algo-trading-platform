"""
Drawdown Protection System
=========================

Advanced drawdown monitoring and protection with automatic position reduction.

Features:
- Real-time drawdown calculation
- Multi-timeframe drawdown tracking
- Automatic position scaling based on drawdown
- Recovery protocols
- Early warning system
- Stress-based drawdown scenarios

Author: Trading Bot Arsenal
Created: February 2026
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json

from ..config.risk_config import RiskManagementConfig, AlertSeverity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DrawdownProtection')


class DrawdownSeverity(Enum):
    """Drawdown severity levels"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ProtectionAction(Enum):
    """Protection actions"""
    NONE = "none"
    REDUCE_POSITIONS = "reduce_positions"
    HALT_NEW_TRADES = "halt_new_trades"
    EMERGENCY_EXIT = "emergency_exit"
    FULL_LIQUIDATION = "full_liquidation"


@dataclass
class DrawdownSnapshot:
    """Single drawdown measurement"""
    timestamp: datetime
    portfolio_value: float
    peak_value: float
    drawdown_amount: float
    drawdown_percentage: float
    duration_days: int
    recovery_factor: float  # How much recovery is needed to reach new high


@dataclass
class DrawdownPeriod:
    """Complete drawdown period from peak to recovery"""
    start_date: datetime
    end_date: Optional[datetime]
    peak_value: float
    trough_value: float
    max_drawdown_pct: float
    duration_days: int
    recovery_days: Optional[int]
    is_active: bool
    cause: str = "Unknown"


@dataclass
class ProtectionStatus:
    """Current protection system status"""
    current_drawdown: DrawdownSnapshot
    severity_level: DrawdownSeverity
    recommended_action: ProtectionAction
    position_size_multiplier: float
    new_trades_allowed: bool

    # Historical context
    worst_drawdown_30d: float
    avg_drawdown_30d: float
    drawdown_frequency: float

    # Recovery metrics
    days_since_peak: int
    recovery_probability: float
    estimated_recovery_days: int

    # Alerts and recommendations
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class DrawdownProtection:
    """
    Comprehensive drawdown protection system.

    Monitors portfolio drawdowns in real-time and automatically
    implements protection measures based on severity.
    """

    def __init__(
        self,
        config: RiskManagementConfig,
        alert_callback: Optional[Callable] = None
    ):
        """
        Initialize drawdown protection system.

        Args:
            config: Risk management configuration
            alert_callback: Function to call for alerts
        """
        self.config = config
        self.drawdown_config = config.drawdown_config
        self.alert_callback = alert_callback

        # Current state
        self.current_portfolio_value = config.portfolio_value
        self.peak_portfolio_value = config.portfolio_value
        self.current_drawdown_start: Optional[datetime] = None

        # Historical tracking
        self.value_history: List[Tuple[datetime, float]] = []
        self.drawdown_history: List[DrawdownSnapshot] = []
        self.drawdown_periods: List[DrawdownPeriod] = []

        # Protection state
        self.protection_active = False
        self.current_severity = DrawdownSeverity.NORMAL
        self.position_multiplier = 1.0

        # Database for persistence
        self.db_path = "logs/drawdown_protection.db"
        self._init_database()

        logger.info("DrawdownProtection initialized")

    def update_portfolio_value(self, new_value: float, timestamp: datetime = None) -> ProtectionStatus:
        """
        Update portfolio value and recalculate drawdown metrics.

        Args:
            new_value: Current portfolio value
            timestamp: Update timestamp (defaults to now)

        Returns:
            Current protection status
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.current_portfolio_value = new_value

        # Update value history
        self.value_history.append((timestamp, new_value))

        # Keep limited history for performance
        if len(self.value_history) > 10000:
            self.value_history = self.value_history[-5000:]

        # Update peak if necessary
        if new_value > self.peak_portfolio_value:
            self.peak_portfolio_value = new_value
            self._end_current_drawdown_period(timestamp)

        # Calculate current drawdown
        current_drawdown = self._calculate_current_drawdown(timestamp)

        # Update historical tracking
        self.drawdown_history.append(current_drawdown)
        if len(self.drawdown_history) > 1000:
            self.drawdown_history = self.drawdown_history[-500:]

        # Assess severity and determine actions
        protection_status = self._assess_protection_status(current_drawdown)

        # Execute protection actions if necessary
        self._execute_protection_actions(protection_status)

        # Save to database
        self._save_snapshot(current_drawdown)

        return protection_status

    def get_position_size_multiplier(self) -> float:
        """Get current position size multiplier due to drawdown protection"""
        return self.position_multiplier

    def is_new_trades_allowed(self) -> bool:
        """Check if new trades are allowed under current protection status"""
        return self.current_severity not in [DrawdownSeverity.EMERGENCY]

    def force_protection_level(self, severity: DrawdownSeverity, reason: str = "Manual override"):
        """Manually set protection level"""
        self.current_severity = severity
        self.protection_active = True

        if self.alert_callback:
            self.alert_callback(f"⚠️ Drawdown protection manually set to {severity.value}: {reason}")

        logger.warning(f"Protection level manually set to {severity.value}: {reason}")

    def reset_protection(self, reason: str = "Manual reset"):
        """Reset protection to normal status"""
        self.current_severity = DrawdownSeverity.NORMAL
        self.protection_active = False
        self.position_multiplier = 1.0

        if self.alert_callback:
            self.alert_callback(f"✅ Drawdown protection reset: {reason}")

        logger.info(f"Protection reset: {reason}")

    def analyze_drawdown_patterns(self, days_back: int = 90) -> Dict[str, Any]:
        """
        Analyze historical drawdown patterns.

        Args:
            days_back: Days of history to analyze

        Returns:
            Drawdown analysis results
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_snapshots = [
            snapshot for snapshot in self.drawdown_history
            if snapshot.timestamp >= cutoff_date
        ]

        if not recent_snapshots:
            return {"error": "No recent drawdown data available"}

        # Calculate statistics
        drawdowns = [snap.drawdown_percentage for snap in recent_snapshots]
        max_drawdown = max(drawdowns) if drawdowns else 0
        avg_drawdown = np.mean(drawdowns) if drawdowns else 0
        std_drawdown = np.std(drawdowns) if drawdowns else 0

        # Count drawdown events
        significant_drawdowns = [dd for dd in drawdowns if dd > 0.05]  # > 5%
        severe_drawdowns = [dd for dd in drawdowns if dd > 0.10]  # > 10%

        # Recovery analysis
        recovery_times = []
        for period in self.drawdown_periods:
            if period.recovery_days is not None:
                recovery_times.append(period.recovery_days)

        avg_recovery_days = np.mean(recovery_times) if recovery_times else 0

        return {
            'analysis_period_days': days_back,
            'total_snapshots': len(recent_snapshots),
            'max_drawdown': f"{max_drawdown:.2%}",
            'avg_drawdown': f"{avg_drawdown:.2%}",
            'drawdown_volatility': f"{std_drawdown:.2%}",
            'significant_drawdown_count': len(significant_drawdowns),
            'severe_drawdown_count': len(severe_drawdowns),
            'avg_recovery_days': f"{avg_recovery_days:.1f}",
            'drawdown_frequency': f"{len(significant_drawdowns) / max(days_back/30, 1):.1f} per month"
        }

    def _calculate_current_drawdown(self, timestamp: datetime) -> DrawdownSnapshot:
        """Calculate current drawdown metrics"""
        drawdown_amount = self.peak_portfolio_value - self.current_portfolio_value
        drawdown_percentage = drawdown_amount / self.peak_portfolio_value

        # Calculate duration
        if self.current_drawdown_start is None and drawdown_percentage > 0:
            self.current_drawdown_start = timestamp

        duration_days = 0
        if self.current_drawdown_start is not None:
            duration_days = (timestamp - self.current_drawdown_start).days

        # Calculate recovery factor
        recovery_factor = (self.peak_portfolio_value - self.current_portfolio_value) / self.current_portfolio_value

        return DrawdownSnapshot(
            timestamp=timestamp,
            portfolio_value=self.current_portfolio_value,
            peak_value=self.peak_portfolio_value,
            drawdown_amount=drawdown_amount,
            drawdown_percentage=drawdown_percentage,
            duration_days=duration_days,
            recovery_factor=recovery_factor
        )

    def _assess_protection_status(self, drawdown: DrawdownSnapshot) -> ProtectionStatus:
        """Assess current protection status and determine actions"""
        # Determine severity level
        dd_pct = drawdown.drawdown_percentage

        if dd_pct >= self.drawdown_config.max_portfolio_drawdown:
            severity = DrawdownSeverity.EMERGENCY
        elif dd_pct >= self.drawdown_config.max_portfolio_drawdown * 0.8:
            severity = DrawdownSeverity.CRITICAL
        elif dd_pct >= self.drawdown_config.max_portfolio_drawdown * 0.5:
            severity = DrawdownSeverity.WARNING
        else:
            severity = DrawdownSeverity.NORMAL

        self.current_severity = severity

        # Determine recommended action
        if severity == DrawdownSeverity.EMERGENCY:
            action = ProtectionAction.EMERGENCY_EXIT
        elif severity == DrawdownSeverity.CRITICAL:
            action = ProtectionAction.HALT_NEW_TRADES
        elif severity == DrawdownSeverity.WARNING:
            action = ProtectionAction.REDUCE_POSITIONS
        else:
            action = ProtectionAction.NONE

        # Calculate position size multiplier
        multiplier = self._calculate_position_multiplier(dd_pct)
        self.position_multiplier = multiplier

        # New trades allowed?
        new_trades_allowed = severity not in [DrawdownSeverity.EMERGENCY]

        # Historical context
        recent_drawdowns = self._get_recent_drawdown_stats()

        # Recovery metrics
        recovery_metrics = self._calculate_recovery_metrics(drawdown)

        # Generate alerts and recommendations
        alerts = self._generate_alerts(drawdown, severity)
        recommendations = self._generate_recommendations(drawdown, severity)

        return ProtectionStatus(
            current_drawdown=drawdown,
            severity_level=severity,
            recommended_action=action,
            position_size_multiplier=multiplier,
            new_trades_allowed=new_trades_allowed,
            worst_drawdown_30d=recent_drawdowns['worst_30d'],
            avg_drawdown_30d=recent_drawdowns['avg_30d'],
            drawdown_frequency=recent_drawdowns['frequency'],
            days_since_peak=drawdown.duration_days,
            recovery_probability=recovery_metrics['probability'],
            estimated_recovery_days=recovery_metrics['estimated_days'],
            alerts=alerts,
            recommendations=recommendations
        )

    def _calculate_position_multiplier(self, drawdown_pct: float) -> float:
        """Calculate position size multiplier based on drawdown"""
        # Use configured scaling tiers
        for dd_threshold, multiplier in sorted(self.drawdown_config.drawdown_scaling_tiers.items()):
            if drawdown_pct >= dd_threshold:
                return multiplier

        return 1.0  # No drawdown - normal sizing

    def _get_recent_drawdown_stats(self) -> Dict[str, float]:
        """Get recent drawdown statistics"""
        cutoff_30d = datetime.now() - timedelta(days=30)
        recent_snapshots = [
            snap for snap in self.drawdown_history
            if snap.timestamp >= cutoff_30d
        ]

        if not recent_snapshots:
            return {'worst_30d': 0.0, 'avg_30d': 0.0, 'frequency': 0.0}

        drawdowns = [snap.drawdown_percentage for snap in recent_snapshots]
        worst_30d = max(drawdowns)
        avg_30d = np.mean(drawdowns)

        # Count significant drawdown events (> 2%)
        significant_events = len([dd for dd in drawdowns if dd > 0.02])
        frequency = significant_events / 30 * 30  # Events per 30 days

        return {
            'worst_30d': worst_30d,
            'avg_30d': avg_30d,
            'frequency': frequency
        }

    def _calculate_recovery_metrics(self, drawdown: DrawdownSnapshot) -> Dict[str, Any]:
        """Calculate recovery probability and estimates"""
        dd_pct = drawdown.drawdown_percentage

        if dd_pct == 0:
            return {'probability': 1.0, 'estimated_days': 0}

        # Simple heuristic based on historical recovery patterns
        # In practice, would use more sophisticated modeling

        # Recovery probability decreases with drawdown magnitude and duration
        base_probability = max(0.1, 1.0 - dd_pct * 2)  # Larger DD = lower probability

        # Duration penalty
        duration_penalty = min(0.5, drawdown.duration_days / 100)
        probability = max(0.1, base_probability - duration_penalty)

        # Estimated recovery days based on historical patterns
        # Assuming exponential recovery model
        if dd_pct < 0.05:
            estimated_days = 10
        elif dd_pct < 0.10:
            estimated_days = 30
        elif dd_pct < 0.15:
            estimated_days = 90
        else:
            estimated_days = 180

        return {
            'probability': probability,
            'estimated_days': estimated_days
        }

    def _generate_alerts(self, drawdown: DrawdownSnapshot, severity: DrawdownSeverity) -> List[str]:
        """Generate appropriate alerts based on drawdown severity"""
        alerts = []

        if severity == DrawdownSeverity.EMERGENCY:
            alerts.append(f"🚨 EMERGENCY DRAWDOWN: {drawdown.drawdown_percentage:.1%}")
            alerts.append("Emergency protection protocols activated")

        elif severity == DrawdownSeverity.CRITICAL:
            alerts.append(f"⚠️ CRITICAL DRAWDOWN: {drawdown.drawdown_percentage:.1%}")
            alerts.append("New trades halted, consider position reduction")

        elif severity == DrawdownSeverity.WARNING:
            alerts.append(f"⚠️ Warning: Drawdown at {drawdown.drawdown_percentage:.1%}")
            alerts.append("Position sizes reduced automatically")

        # Duration warnings
        if drawdown.duration_days > 30:
            alerts.append(f"Extended drawdown period: {drawdown.duration_days} days")

        return alerts

    def _generate_recommendations(
        self,
        drawdown: DrawdownSnapshot,
        severity: DrawdownSeverity
    ) -> List[str]:
        """Generate recommendations based on current situation"""
        recommendations = []

        if severity == DrawdownSeverity.EMERGENCY:
            recommendations.append("Consider emergency liquidation of positions")
            recommendations.append("Review trading strategy effectiveness")
            recommendations.append("Implement capital preservation mode")

        elif severity == DrawdownSeverity.CRITICAL:
            recommendations.append("Halt all new position entries")
            recommendations.append("Review and reduce existing positions")
            recommendations.append("Analyze causes of current drawdown")

        elif severity == DrawdownSeverity.WARNING:
            recommendations.append("Reduce position sizes temporarily")
            recommendations.append("Focus on high-conviction trades only")
            recommendations.append("Monitor market conditions closely")

        # Duration-based recommendations
        if drawdown.duration_days > 60:
            recommendations.append("Consider strategy review due to extended drawdown")

        # Recovery recommendations
        if drawdown.recovery_factor > 0.5:
            recommendations.append("Significant recovery needed - consider risk reduction")

        return recommendations

    def _execute_protection_actions(self, status: ProtectionStatus):
        """Execute automatic protection actions"""
        action = status.recommended_action

        if action == ProtectionAction.EMERGENCY_EXIT:
            if self.alert_callback:
                self.alert_callback("🚨 EMERGENCY: Maximum drawdown reached - Emergency protocols activated")
            logger.critical("EMERGENCY DRAWDOWN PROTECTION ACTIVATED")

        elif action == ProtectionAction.HALT_NEW_TRADES:
            if not self.protection_active:
                self.protection_active = True
                if self.alert_callback:
                    self.alert_callback("⚠️ New trades halted due to critical drawdown")
                logger.warning("New trades halted due to drawdown")

        elif action == ProtectionAction.REDUCE_POSITIONS:
            if self.alert_callback:
                self.alert_callback(f"📉 Position sizes reduced to {status.position_size_multiplier:.0%} due to drawdown")

    def _end_current_drawdown_period(self, timestamp: datetime):
        """End the current drawdown period (new peak reached)"""
        if self.current_drawdown_start is not None:
            # Complete the current drawdown period
            duration = (timestamp - self.current_drawdown_start).days
            trough_value = min(value for _, value in self.value_history
                              if _ >= self.current_drawdown_start)
            max_dd = (self.peak_portfolio_value - trough_value) / self.peak_portfolio_value

            period = DrawdownPeriod(
                start_date=self.current_drawdown_start,
                end_date=timestamp,
                peak_value=self.peak_portfolio_value,
                trough_value=trough_value,
                max_drawdown_pct=max_dd,
                duration_days=duration,
                recovery_days=duration,
                is_active=False
            )

            self.drawdown_periods.append(period)
            self.current_drawdown_start = None

            # Reset protection if recovering
            if self.protection_active and max_dd < self.drawdown_config.recovery_threshold:
                self.reset_protection("Recovery from drawdown")

    def _init_database(self):
        """Initialize SQLite database for persistence"""
        try:
            import os
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drawdown_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    portfolio_value REAL,
                    peak_value REAL,
                    drawdown_amount REAL,
                    drawdown_percentage REAL,
                    duration_days INTEGER,
                    recovery_factor REAL
                )
            ''')

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def _save_snapshot(self, snapshot: DrawdownSnapshot):
        """Save drawdown snapshot to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO drawdown_snapshots
                (timestamp, portfolio_value, peak_value, drawdown_amount,
                 drawdown_percentage, duration_days, recovery_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.timestamp.isoformat(),
                snapshot.portfolio_value,
                snapshot.peak_value,
                snapshot.drawdown_amount,
                snapshot.drawdown_percentage,
                snapshot.duration_days,
                snapshot.recovery_factor
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive drawdown protection status report"""
        if not self.drawdown_history:
            current_dd = self._calculate_current_drawdown(datetime.now())
        else:
            current_dd = self.drawdown_history[-1]

        status = self._assess_protection_status(current_dd)
        analysis = self.analyze_drawdown_patterns()

        return {
            'timestamp': datetime.now().isoformat(),
            'current_status': {
                'portfolio_value': f"${self.current_portfolio_value:,.2f}",
                'peak_value': f"${self.peak_portfolio_value:,.2f}",
                'current_drawdown': f"{current_dd.drawdown_percentage:.2%}",
                'duration_days': current_dd.duration_days,
                'severity_level': status.severity_level.value,
                'protection_active': self.protection_active
            },
            'protection_measures': {
                'position_multiplier': f"{status.position_size_multiplier:.0%}",
                'new_trades_allowed': status.new_trades_allowed,
                'recommended_action': status.recommended_action.value
            },
            'historical_analysis': analysis,
            'alerts': status.alerts,
            'recommendations': status.recommendations
        }


if __name__ == "__main__":
    from ..config.risk_config import load_risk_config

    # Test drawdown protection
    config = load_risk_config()
    protection = DrawdownProtection(config)

    # Simulate portfolio value changes
    initial_value = 10000
    values = [initial_value]

    # Simulate a drawdown scenario
    for i in range(30):
        if i < 10:
            # Growth phase
            change = np.random.normal(0.001, 0.02)  # Small positive trend
        elif i < 20:
            # Drawdown phase
            change = np.random.normal(-0.02, 0.03)  # Negative trend
        else:
            # Recovery phase
            change = np.random.normal(0.005, 0.02)  # Recovery

        new_value = values[-1] * (1 + change)
        values.append(new_value)

        timestamp = datetime.now() - timedelta(days=30-i)
        status = protection.update_portfolio_value(new_value, timestamp)

        if i % 10 == 0:
            print(f"Day {i}: Value=${new_value:,.2f}, DD={status.current_drawdown.drawdown_percentage:.1%}, "
                  f"Severity={status.severity_level.value}, Multiplier={status.position_size_multiplier:.0%}")

    # Final report
    report = protection.get_status_report()
    print(f"\nFinal Status:")
    print(f"Current Drawdown: {report['current_status']['current_drawdown']}")
    print(f"Severity: {report['current_status']['severity_level']}")
    print(f"Position Multiplier: {report['protection_measures']['position_multiplier']}")
    print(f"Alerts: {report['alerts']}")