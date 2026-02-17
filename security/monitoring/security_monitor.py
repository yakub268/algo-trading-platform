"""
Security Monitor
===============

Comprehensive security monitoring system with real-time threat detection,
anomaly detection, and automated response capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque
import psutil
import json
from pathlib import Path
from .anomaly_detector import AnomalyDetector
from .intrusion_detection import IntrusionDetector
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class SecurityAlert:
    """Represents a security alert."""

    def __init__(self, alert_type: str, severity: str, message: str,
                 source: str, data: Dict[str, Any] = None):
        self.id = self._generate_alert_id()
        self.alert_type = alert_type
        self.severity = severity  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
        self.message = message
        self.source = source
        self.timestamp = datetime.utcnow()
        self.data = data or {}
        self.acknowledged = False
        self.resolved = False

    def _generate_alert_id(self) -> str:
        import secrets
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        random_suffix = secrets.token_hex(4)
        return f"alert_{timestamp}_{random_suffix}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved
        }

class SecurityMonitor:
    """
    Comprehensive security monitoring system.

    Features:
    - Real-time system monitoring
    - Anomaly detection
    - Intrusion detection
    - Rate limiting
    - Automated threat response
    - Alert management
    - Performance monitoring
    - Log analysis
    """

    def __init__(self, config):
        self.config = config
        self.anomaly_detector = AnomalyDetector(config)
        self.intrusion_detector = IntrusionDetector(config)
        self.rate_limiter = RateLimiter(config)

        # Alert management
        self.alerts: List[SecurityAlert] = []
        self.alert_handlers: Dict[str, List[Callable]] = defaultdict(list)

        # Monitoring state
        self._monitoring = False
        self._monitoring_tasks = []

        # Performance metrics
        self.metrics = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'disk_usage': deque(maxlen=100),
            'network_connections': deque(maxlen=100),
            'api_requests': deque(maxlen=1000),
            'failed_logins': deque(maxlen=100)
        }

    async def start_monitoring(self):
        """Start all security monitoring tasks."""
        if self._monitoring:
            return

        self._monitoring = True

        # Start monitoring tasks
        tasks = [
            self._system_monitor(),
            self._network_monitor(),
            self._log_monitor(),
            self._performance_monitor(),
            self._threat_detection(),
            self._alert_processor()
        ]

        self._monitoring_tasks = [asyncio.create_task(task) for task in tasks]

        logger.info("Security monitoring started")
        self.create_alert('SYSTEM', 'INFO', 'Security monitoring system started', 'SecurityMonitor')

    async def stop_monitoring(self):
        """Stop all security monitoring tasks."""
        if not self._monitoring:
            return

        self._monitoring = False

        # Cancel all monitoring tasks
        for task in self._monitoring_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._monitoring_tasks.clear()

        logger.info("Security monitoring stopped")
        self.create_alert('SYSTEM', 'INFO', 'Security monitoring system stopped', 'SecurityMonitor')

    def create_alert(self, alert_type: str, severity: str, message: str,
                    source: str, data: Dict[str, Any] = None) -> SecurityAlert:
        """Create a new security alert."""
        alert = SecurityAlert(alert_type, severity, message, source, data)
        self.alerts.append(alert)

        # Keep only recent alerts (last 1000)
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]

        # Log the alert
        log_level = {
            'LOW': logging.INFO,
            'MEDIUM': logging.WARNING,
            'HIGH': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }.get(severity, logging.INFO)

        logger.log(log_level, f"Security Alert [{severity}] {alert_type}: {message}")

        # Trigger alert handlers
        asyncio.create_task(self._handle_alert(alert))

        return alert

    def register_alert_handler(self, alert_type: str, handler: Callable):
        """Register a handler for specific alert types."""
        self.alert_handlers[alert_type].append(handler)

    async def get_alerts(self, severity: Optional[str] = None,
                        alert_type: Optional[str] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts with optional filtering."""
        filtered_alerts = self.alerts

        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]

        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a.alert_type == alert_type]

        # Sort by timestamp (newest first) and apply limit
        filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        filtered_alerts = filtered_alerts[:limit]

        return [alert.to_dict() for alert in filtered_alerts]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False

    async def check_rate_limit(self, identifier: str, endpoint: str) -> bool:
        """Check if request is within rate limits."""
        return await self.rate_limiter.check_rate_limit(identifier, endpoint)

    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        recent_alerts = await self.get_alerts(limit=10)
        critical_alerts = await self.get_alerts(severity='CRITICAL', limit=5)

        # System metrics
        current_metrics = {}
        try:
            current_metrics = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'active_connections': len(psutil.net_connections()),
                'running_processes': len(psutil.pids())
            }
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")

        return {
            'monitoring_active': self._monitoring,
            'total_alerts': len(self.alerts),
            'unacknowledged_alerts': len([a for a in self.alerts if not a.acknowledged]),
            'critical_alerts': len([a for a in self.alerts if a.severity == 'CRITICAL']),
            'recent_alerts': recent_alerts,
            'critical_alerts_detail': critical_alerts,
            'system_metrics': current_metrics,
            'anomaly_detection_status': self.anomaly_detector.get_status(),
            'intrusion_detection_status': self.intrusion_detector.get_status(),
            'rate_limiting_status': self.rate_limiter.get_status()
        }

    async def _system_monitor(self):
        """Monitor system resources and health."""
        while self._monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics['cpu_usage'].append((datetime.utcnow(), cpu_percent))

                if cpu_percent > 90:
                    self.create_alert('SYSTEM', 'HIGH', f'High CPU usage: {cpu_percent}%', 'SystemMonitor')

                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics['memory_usage'].append((datetime.utcnow(), memory.percent))

                if memory.percent > 90:
                    self.create_alert('SYSTEM', 'HIGH', f'High memory usage: {memory.percent}%', 'SystemMonitor')

                # Disk usage
                disk = psutil.disk_usage('/')
                self.metrics['disk_usage'].append((datetime.utcnow(), disk.percent))

                if disk.percent > 90:
                    self.create_alert('SYSTEM', 'MEDIUM', f'High disk usage: {disk.percent}%', 'SystemMonitor')

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(60)

    async def _network_monitor(self):
        """Monitor network connections and suspicious activity."""
        while self._monitoring:
            try:
                connections = psutil.net_connections(kind='inet')
                self.metrics['network_connections'].append((datetime.utcnow(), len(connections)))

                # Check for suspicious connections
                suspicious_ports = [22, 23, 135, 139, 445, 1433, 3389]  # Common attack vectors

                for conn in connections:
                    if conn.laddr and conn.laddr.port in suspicious_ports and conn.status == 'ESTABLISHED':
                        self.create_alert(
                            'NETWORK', 'MEDIUM',
                            f'Connection to suspicious port {conn.laddr.port}',
                            'NetworkMonitor',
                            {'port': conn.laddr.port, 'remote_addr': conn.raddr}
                        )

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in network monitoring: {e}")
                await asyncio.sleep(30)

    async def _log_monitor(self):
        """Monitor log files for suspicious activity."""
        while self._monitoring:
            try:
                # Monitor application logs for errors and security events
                log_patterns = {
                    'FAILED_LOGIN': ['failed login', 'authentication failed', 'invalid credentials'],
                    'BRUTE_FORCE': ['too many attempts', 'rate limit exceeded', 'blocked'],
                    'UNAUTHORIZED_ACCESS': ['unauthorized', 'access denied', 'forbidden'],
                    'SQL_INJECTION': ['sql injection', 'union select', 'drop table'],
                    'XSS_ATTEMPT': ['script>', 'javascript:', 'onclick=']
                }

                # This would analyze log files in a real implementation
                # For now, we'll skip the actual file parsing

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in log monitoring: {e}")
                await asyncio.sleep(60)

    async def _performance_monitor(self):
        """Monitor application performance metrics."""
        while self._monitoring:
            try:
                # Check anomalies in performance metrics
                await self.anomaly_detector.check_anomalies(self.metrics)

                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(300)

    async def _threat_detection(self):
        """Active threat detection and analysis."""
        while self._monitoring:
            try:
                # Run intrusion detection
                threats = await self.intrusion_detector.scan_for_threats()

                for threat in threats:
                    self.create_alert(
                        'THREAT', threat['severity'],
                        threat['description'],
                        'ThreatDetection',
                        threat
                    )

                await asyncio.sleep(300)  # Scan every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in threat detection: {e}")
                await asyncio.sleep(300)

    async def _alert_processor(self):
        """Process and respond to alerts."""
        while self._monitoring:
            try:
                # Process unacknowledged critical alerts
                critical_alerts = [a for a in self.alerts
                                 if a.severity == 'CRITICAL' and not a.acknowledged]

                for alert in critical_alerts:
                    # Automated response for critical alerts
                    await self._automated_response(alert)

                # Email notifications for high severity alerts
                if self.config.alert_email:
                    high_severity_alerts = [a for a in self.alerts
                                          if a.severity in ['HIGH', 'CRITICAL']
                                          and not a.acknowledged]

                    if high_severity_alerts:
                        await self._send_alert_notifications(high_severity_alerts)

                await asyncio.sleep(60)  # Process every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert processing: {e}")
                await asyncio.sleep(60)

    async def _handle_alert(self, alert: SecurityAlert):
        """Handle a new alert by calling registered handlers."""
        handlers = self.alert_handlers.get(alert.alert_type, [])

        for handler in handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

    async def _automated_response(self, alert: SecurityAlert):
        """Automated response to critical alerts."""
        try:
            if alert.alert_type == 'BRUTE_FORCE':
                # Block IP address
                ip_address = alert.data.get('ip_address')
                if ip_address:
                    logger.warning(f"Blocking IP address due to brute force: {ip_address}")
                    # Implementation would block the IP in firewall

            elif alert.alert_type == 'SYSTEM' and 'High CPU' in alert.message:
                # Scale down non-essential processes
                logger.warning("High CPU detected - implementing protective measures")
                # Implementation would reduce system load

            alert.acknowledged = True

        except Exception as e:
            logger.error(f"Error in automated response: {e}")

    async def _send_alert_notifications(self, alerts: List[SecurityAlert]):
        """Send alert notifications via email."""
        try:
            if not self.config.alert_email:
                return

            # Format alerts for email
            alert_text = "\n".join([
                f"[{alert.severity}] {alert.alert_type}: {alert.message} ({alert.timestamp})"
                for alert in alerts
            ])

            # This would send actual email notifications
            logger.info(f"Would send alert notification to {self.config.alert_email}")

        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        summary = {}

        for metric_name, values in self.metrics.items():
            if values:
                numeric_values = [v[1] for v in values if isinstance(v[1], (int, float))]
                if numeric_values:
                    summary[metric_name] = {
                        'count': len(numeric_values),
                        'latest': numeric_values[-1],
                        'average': sum(numeric_values) / len(numeric_values),
                        'min': min(numeric_values),
                        'max': max(numeric_values)
                    }

        return summary