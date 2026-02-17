"""
Risk Alert Manager
==================

Real-time risk alert system with multiple notification channels.

Features:
- Real-time risk monitoring
- Multiple alert channels (Telegram, Email, Discord, SMS)
- Alert prioritization and throttling
- Escalation procedures
- Alert acknowledgment system
- Custom alert rules

Author: Trading Bot Arsenal
Created: February 2026
"""

import logging
import smtplib
import requests
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import defaultdict, deque
import sqlite3
import os

from ..config.risk_config import RiskManagementConfig, AlertSeverity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RiskAlerts')


class AlertType(Enum):
    """Types of risk alerts"""
    PORTFOLIO_HEAT = "portfolio_heat"
    DRAWDOWN = "drawdown"
    CORRELATION = "correlation"
    VAR_BREACH = "var_breach"
    POSITION_LIMIT = "position_limit"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    SYSTEM_ERROR = "system_error"
    MANUAL_ALERT = "manual_alert"


class AlertChannel(Enum):
    """Alert delivery channels"""
    TELEGRAM = "telegram"
    EMAIL = "email"
    DISCORD = "discord"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG_ONLY = "log_only"


@dataclass
class AlertRule:
    """Custom alert rule definition"""
    rule_id: str
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: str  # Python expression to evaluate
    channels: List[AlertChannel]
    cooldown_minutes: int = 5
    enabled: bool = True


@dataclass
class Alert:
    """Individual alert instance"""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data: Dict[str, Any]

    # Delivery tracking
    channels_sent: List[AlertChannel] = field(default_factory=list)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    # Escalation
    escalated: bool = False
    escalation_level: int = 0


@dataclass
class AlertStats:
    """Alert system statistics"""
    total_alerts_24h: int
    alerts_by_severity: Dict[AlertSeverity, int]
    alerts_by_type: Dict[AlertType, int]
    avg_response_time: float
    unacknowledged_critical: int
    system_health_score: float


class RiskAlertManager:
    """
    Comprehensive risk alert management system.

    Handles real-time risk monitoring, alert generation,
    and multi-channel notification delivery.
    """

    def __init__(self, config: RiskManagementConfig):
        """
        Initialize risk alert manager.

        Args:
            config: Risk management configuration
        """
        self.config = config
        self.alert_config = config.alert_config

        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_rules: Dict[str, AlertRule] = {}

        # Rate limiting and throttling
        self.alert_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_alert_times: Dict[Tuple[AlertType, str], datetime] = {}
        self.alert_repeat_counts: Dict[Tuple[AlertType, str], int] = {}

        # Channel configurations
        self.channel_configs = {
            AlertChannel.TELEGRAM: self._load_telegram_config(),
            AlertChannel.EMAIL: self._load_email_config(),
            AlertChannel.DISCORD: self._load_discord_config(),
            AlertChannel.SLACK: self._load_slack_config()
        }

        # Background processing
        self.alert_queue: deque = deque()
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_processing = threading.Event()

        # Database
        self.db_path = "logs/risk_alerts.db"
        self._init_database()

        # Load default rules
        self._load_default_alert_rules()

        # Start background processor
        self._start_background_processor()

        logger.info("RiskAlertManager initialized")

    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        data: Dict[str, Any] = None,
        channels: List[AlertChannel] = None
    ) -> str:
        """
        Create and send a new alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            title: Alert title
            message: Alert message
            data: Additional alert data
            channels: Override default channels

        Returns:
            Alert ID
        """
        if data is None:
            data = {}

        # Generate unique alert ID
        alert_id = f"{alert_type.value}_{int(time.time() * 1000)}"

        # Check rate limits
        if not self._check_rate_limits(alert_type, title):
            logger.debug(f"Alert rate limited: {alert_type.value} - {title}")
            return alert_id

        # Determine channels
        if channels is None:
            channels = self._get_default_channels(severity)

        # Create alert
        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            data=data
        )

        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Queue for processing
        self.alert_queue.append((alert, channels))

        logger.info(f"Alert created: {severity.value.upper()} - {title}")

        return alert_id

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "System") -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: Who acknowledged the alert

        Returns:
            True if successful
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()

            # Remove from active alerts
            del self.active_alerts[alert_id]

            self._save_alert_to_db(alert)

            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True

        return False

    def escalate_alert(self, alert_id: str, reason: str = "Automatic escalation") -> bool:
        """
        Escalate an alert to higher severity.

        Args:
            alert_id: Alert ID to escalate
            reason: Reason for escalation

        Returns:
            True if successful
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]

            if not alert.escalated:
                alert.escalated = True
                alert.escalation_level += 1

                # Escalate severity
                if alert.severity == AlertSeverity.WARNING:
                    alert.severity = AlertSeverity.CRITICAL
                elif alert.severity == AlertSeverity.CRITICAL:
                    alert.severity = AlertSeverity.EMERGENCY

                # Send escalation alert
                escalation_message = f"🔥 ESCALATED ALERT: {alert.title}\n\nReason: {reason}\n\nOriginal: {alert.message}"

                self.create_alert(
                    alert_type=AlertType.MANUAL_ALERT,
                    severity=alert.severity,
                    title=f"ESCALATED: {alert.title}",
                    message=escalation_message,
                    channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL]
                )

                logger.warning(f"Alert escalated: {alert_id} - {reason}")
                return True

        return False

    def add_custom_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added custom alert rule: {rule.name}")

    def remove_custom_rule(self, rule_id: str) -> bool:
        """Remove custom alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False

    def evaluate_custom_rules(self, context: Dict[str, Any]):
        """
        Evaluate custom alert rules against current context.

        Args:
            context: Current system state and metrics
        """
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            try:
                # Create safe evaluation environment
                safe_globals = {
                    '__builtins__': {},
                    'abs': abs, 'max': max, 'min': min,
                    'len': len, 'sum': sum, 'any': any, 'all': all
                }

                # Evaluate condition
                if eval(rule.condition, safe_globals, context):
                    # Rule triggered - create alert
                    self.create_alert(
                        alert_type=rule.alert_type,
                        severity=rule.severity,
                        title=f"Custom Rule: {rule.name}",
                        message=f"Rule condition met: {rule.condition}",
                        data={'rule_id': rule.rule_id, 'context': context},
                        channels=rule.channels
                    )

            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")

    def get_alert_stats(self, hours_back: int = 24) -> AlertStats:
        """Get alert system statistics"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]

        # Count by severity
        severity_counts = defaultdict(int)
        for alert in recent_alerts:
            severity_counts[alert.severity] += 1

        # Count by type
        type_counts = defaultdict(int)
        for alert in recent_alerts:
            type_counts[alert.alert_type] += 1

        # Calculate response time (simplified)
        response_times = []
        for alert in recent_alerts:
            if alert.acknowledged and alert.acknowledged_at:
                response_time = (alert.acknowledged_at - alert.timestamp).total_seconds()
                response_times.append(response_time)

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        # Unacknowledged critical alerts
        unack_critical = sum(
            1 for alert in self.active_alerts.values()
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        )

        # System health score (0-100)
        health_factors = []

        # Factor 1: Recent alert volume
        alert_volume_factor = max(0, 100 - len(recent_alerts) * 2)
        health_factors.append(alert_volume_factor)

        # Factor 2: Unacknowledged critical alerts
        unack_factor = max(0, 100 - unack_critical * 20)
        health_factors.append(unack_factor)

        # Factor 3: Response time
        response_factor = max(0, 100 - avg_response_time / 60)  # Penalty for slow response
        health_factors.append(response_factor)

        system_health_score = sum(health_factors) / len(health_factors)

        return AlertStats(
            total_alerts_24h=len(recent_alerts),
            alerts_by_severity=dict(severity_counts),
            alerts_by_type=dict(type_counts),
            avg_response_time=avg_response_time,
            unacknowledged_critical=unack_critical,
            system_health_score=system_health_score
        )

    def _check_rate_limits(self, alert_type: AlertType, title: str) -> bool:
        """Check if alert should be rate limited with escalating cooldowns.

        Cooldown tiers based on repeat count for same (alert_type, title):
        - 0 repeats: 5 min (default min_alert_interval)
        - 1-2 repeats: 1 hour
        - 3-5 repeats: 4 hours
        - 6+ repeats: 24 hours
        """
        now = datetime.now()
        key = (alert_type, title)

        # Determine cooldown based on repeat count
        repeat_count = self.alert_repeat_counts.get(key, 0)
        if repeat_count >= 6:
            cooldown_seconds = 86400  # 24 hours
        elif repeat_count >= 3:
            cooldown_seconds = 14400  # 4 hours
        elif repeat_count >= 1:
            cooldown_seconds = 3600   # 1 hour
        else:
            cooldown_seconds = self.alert_config.min_alert_interval  # 5 min

        # Check cooldown
        if key in self.last_alert_times:
            time_since_last = now - self.last_alert_times[key]
            if time_since_last.total_seconds() < cooldown_seconds:
                return False
            # If enough time passed, allow but track repeat
            self.alert_repeat_counts[key] = repeat_count + 1
        else:
            # First time seeing this alert
            self.alert_repeat_counts[key] = 0

        # Check hourly limit
        hour_key = f"{alert_type.value}_{now.hour}"
        self.alert_counts[hour_key].append(now)

        # Clean old entries
        cutoff = now - timedelta(hours=1)
        while self.alert_counts[hour_key] and self.alert_counts[hour_key][0] < cutoff:
            self.alert_counts[hour_key].popleft()

        if len(self.alert_counts[hour_key]) > self.alert_config.max_alerts_per_hour:
            return False

        self.last_alert_times[key] = now
        return True

    def _get_default_channels(self, severity: AlertSeverity) -> List[AlertChannel]:
        """Get default channels based on severity"""
        if severity == AlertSeverity.EMERGENCY:
            return [AlertChannel.TELEGRAM, AlertChannel.EMAIL, AlertChannel.DISCORD]
        elif severity == AlertSeverity.CRITICAL:
            return [AlertChannel.TELEGRAM, AlertChannel.EMAIL]
        elif severity == AlertSeverity.WARNING:
            return [AlertChannel.TELEGRAM]
        else:
            return [AlertChannel.LOG_ONLY]

    def _start_background_processor(self):
        """Start background alert processing thread"""
        def process_alerts():
            while not self.stop_processing.is_set():
                try:
                    if self.alert_queue:
                        alert, channels = self.alert_queue.popleft()
                        self._send_alert(alert, channels)
                    else:
                        time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error processing alert: {e}")

        self.processing_thread = threading.Thread(target=process_alerts, daemon=True)
        self.processing_thread.start()

    def _send_alert(self, alert: Alert, channels: List[AlertChannel]):
        """Send alert through specified channels"""
        for channel in channels:
            try:
                if channel == AlertChannel.TELEGRAM:
                    self._send_telegram_alert(alert)
                elif channel == AlertChannel.EMAIL:
                    self._send_email_alert(alert)
                elif channel == AlertChannel.DISCORD:
                    self._send_discord_alert(alert)
                elif channel == AlertChannel.SLACK:
                    self._send_slack_alert(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._send_webhook_alert(alert)
                elif channel == AlertChannel.LOG_ONLY:
                    self._log_alert(alert)

                alert.channels_sent.append(channel)

            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")

        # Save to database
        self._save_alert_to_db(alert)

    def _send_telegram_alert(self, alert: Alert):
        """Send alert via Telegram"""
        config = self.channel_configs.get(AlertChannel.TELEGRAM)
        if not config or not config.get('enabled'):
            return

        # Format message
        emoji = self._get_severity_emoji(alert.severity)
        message = f"{emoji} {alert.title}\n\n{alert.message}\n\nTime: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

        url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
        data = {
            'chat_id': config['chat_id'],
            'text': message,
            'parse_mode': 'HTML' if '<' in message else 'Markdown'
        }

        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()

    def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        config = self.channel_configs.get(AlertChannel.EMAIL)
        if not config or not config.get('enabled'):
            return

        msg = MIMEMultipart()
        msg['From'] = config['from_email']
        msg['To'] = config['to_email']
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"

        body = f"""
        Risk Alert - {alert.severity.value.upper()}

        Alert Type: {alert.alert_type.value}
        Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

        {alert.message}

        Alert ID: {alert.alert_id}
        """

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['username'], config['password'])
        text = msg.as_string()
        server.sendmail(config['from_email'], config['to_email'], text)
        server.quit()

    def _send_discord_alert(self, alert: Alert):
        """Send alert via Discord webhook"""
        config = self.channel_configs.get(AlertChannel.DISCORD)
        if not config or not config.get('enabled'):
            return

        color = self._get_severity_color(alert.severity)
        data = {
            "embeds": [{
                "title": alert.title,
                "description": alert.message,
                "color": color,
                "timestamp": alert.timestamp.isoformat(),
                "fields": [
                    {"name": "Alert Type", "value": alert.alert_type.value, "inline": True},
                    {"name": "Severity", "value": alert.severity.value.upper(), "inline": True},
                    {"name": "Alert ID", "value": alert.alert_id, "inline": True}
                ]
            }]
        }

        response = requests.post(config['webhook_url'], json=data, timeout=10)
        response.raise_for_status()

    def _send_slack_alert(self, alert: Alert):
        """Send alert via Slack"""
        config = self.channel_configs.get(AlertChannel.SLACK)
        if not config or not config.get('enabled'):
            return

        emoji = self._get_severity_emoji(alert.severity)
        data = {
            "text": f"{emoji} {alert.title}",
            "attachments": [{
                "color": self._get_severity_color_hex(alert.severity),
                "fields": [
                    {"title": "Message", "value": alert.message, "short": False},
                    {"title": "Type", "value": alert.alert_type.value, "short": True},
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                ]
            }]
        }

        response = requests.post(config['webhook_url'], json=data, timeout=10)
        response.raise_for_status()

    def _send_webhook_alert(self, alert: Alert):
        """Send alert via custom webhook"""
        # Generic webhook implementation
        pass

    def _log_alert(self, alert: Alert):
        """Log alert to file/console"""
        severity_map = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }

        level = severity_map.get(alert.severity, logging.INFO)
        logger.log(level, f"ALERT [{alert.alert_type.value}]: {alert.title} - {alert.message}")

    def _get_severity_emoji(self, severity: AlertSeverity) -> str:
        """Get emoji for alert severity"""
        emoji_map = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🔥"
        }
        return emoji_map.get(severity, "📢")

    def _get_severity_color(self, severity: AlertSeverity) -> int:
        """Get color for Discord embeds"""
        color_map = {
            AlertSeverity.INFO: 0x00BFFF,      # Blue
            AlertSeverity.WARNING: 0xFFD700,   # Gold
            AlertSeverity.CRITICAL: 0xFF4500,  # Red-Orange
            AlertSeverity.EMERGENCY: 0xFF0000   # Red
        }
        return color_map.get(severity, 0x808080)

    def _get_severity_color_hex(self, severity: AlertSeverity) -> str:
        """Get hex color for Slack"""
        color_map = {
            AlertSeverity.INFO: "#00BFFF",
            AlertSeverity.WARNING: "#FFD700",
            AlertSeverity.CRITICAL: "#FF4500",
            AlertSeverity.EMERGENCY: "#FF0000"
        }
        return color_map.get(severity, "#808080")

    def _load_telegram_config(self) -> Dict[str, Any]:
        """Load Telegram configuration"""
        return {
            'enabled': self.alert_config.telegram_enabled,
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID')
        }

    def _load_email_config(self) -> Dict[str, Any]:
        """Load email configuration"""
        return {
            'enabled': self.alert_config.email_enabled,
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('SMTP_USERNAME'),
            'password': os.getenv('SMTP_PASSWORD'),
            'from_email': os.getenv('ALERT_FROM_EMAIL'),
            'to_email': os.getenv('ALERT_TO_EMAIL')
        }

    def _load_discord_config(self) -> Dict[str, Any]:
        """Load Discord configuration"""
        return {
            'enabled': self.alert_config.discord_enabled,
            'webhook_url': os.getenv('DISCORD_WEBHOOK_URL')
        }

    def _load_slack_config(self) -> Dict[str, Any]:
        """Load Slack configuration"""
        return {
            'enabled': os.getenv('SLACK_ENABLED', 'false').lower() == 'true',
            'webhook_url': os.getenv('SLACK_WEBHOOK_URL')
        }

    def _load_default_alert_rules(self):
        """Load default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="portfolio_heat_critical",
                name="Portfolio Heat Critical",
                alert_type=AlertType.PORTFOLIO_HEAT,
                severity=AlertSeverity.CRITICAL,
                condition="portfolio_heat > 90",
                channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL]
            ),
            AlertRule(
                rule_id="high_correlation_warning",
                name="High Correlation Warning",
                alert_type=AlertType.CORRELATION,
                severity=AlertSeverity.WARNING,
                condition="max_correlation > 0.8",
                channels=[AlertChannel.TELEGRAM]
            ),
            AlertRule(
                rule_id="var_breach_critical",
                name="VaR Breach Critical",
                alert_type=AlertType.VAR_BREACH,
                severity=AlertSeverity.CRITICAL,
                condition="current_var > var_limit * 0.95",
                channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL]
            )
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

    def _init_database(self):
        """Initialize SQLite database"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    title TEXT,
                    message TEXT,
                    data TEXT,
                    channels_sent TEXT,
                    acknowledged BOOLEAN,
                    acknowledged_by TEXT,
                    acknowledged_at TEXT
                )
            ''')

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def _save_alert_to_db(self, alert: Alert):
        """Save alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO alerts
                (alert_id, timestamp, alert_type, severity, title, message, data,
                 channels_sent, acknowledged, acknowledged_by, acknowledged_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.timestamp.isoformat(),
                alert.alert_type.value,
                alert.severity.value,
                alert.title,
                alert.message,
                json.dumps(alert.data),
                json.dumps([c.value for c in alert.channels_sent]),
                alert.acknowledged,
                alert.acknowledged_by,
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to save alert to database: {e}")

    def shutdown(self):
        """Shutdown alert manager"""
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5)

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive alert system status"""
        stats = self.get_alert_stats()

        return {
            'timestamp': datetime.now().isoformat(),
            'system_health_score': f"{stats.system_health_score:.0f}/100",
            'active_alerts': len(self.active_alerts),
            'alerts_24h': stats.total_alerts_24h,
            'unacknowledged_critical': stats.unacknowledged_critical,
            'avg_response_time': f"{stats.avg_response_time:.1f}s",
            'alerts_by_severity': {
                severity.value: count
                for severity, count in stats.alerts_by_severity.items()
            },
            'channel_status': {
                channel.value: config.get('enabled', False)
                for channel, config in self.channel_configs.items()
                if config
            },
            'alert_rules': {
                rule_id: {
                    'name': rule.name,
                    'enabled': rule.enabled,
                    'severity': rule.severity.value
                }
                for rule_id, rule in self.alert_rules.items()
            }
        }


if __name__ == "__main__":
    from ..config.risk_config import load_risk_config

    # Test alert manager
    config = load_risk_config()
    alert_manager = RiskAlertManager(config)

    # Create test alerts
    alert_manager.create_alert(
        alert_type=AlertType.PORTFOLIO_HEAT,
        severity=AlertSeverity.WARNING,
        title="Portfolio Heat Warning",
        message="Portfolio heat is at 85% of limit",
        data={'heat_level': 0.85}
    )

    alert_manager.create_alert(
        alert_type=AlertType.DRAWDOWN,
        severity=AlertSeverity.CRITICAL,
        title="Critical Drawdown",
        message="Portfolio drawdown has reached 15%",
        data={'drawdown_pct': 0.15}
    )

    # Test custom rule evaluation
    context = {
        'portfolio_heat': 95,
        'max_correlation': 0.85,
        'current_var': 1000,
        'var_limit': 1200
    }

    alert_manager.evaluate_custom_rules(context)

    # Get status report
    status = alert_manager.get_status_report()
    print("Alert Manager Status:")
    print(f"System Health: {status['system_health_score']}")
    print(f"Active Alerts: {status['active_alerts']}")
    print(f"24h Alerts: {status['alerts_24h']}")
    print(f"Unacknowledged Critical: {status['unacknowledged_critical']}")

    # Cleanup
    time.sleep(1)
    alert_manager.shutdown()