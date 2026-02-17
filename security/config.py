"""
Security Configuration Management
================================

Central configuration for all security components with environment-aware settings.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class VaultConfig:
    """API Key Vault Configuration"""
    encryption_key_path: str = "security/vault/master.key"
    storage_path: str = "security/vault/keys.enc"
    rotation_interval_hours: int = 24
    backup_count: int = 5
    auto_rotation_enabled: bool = True
    key_strength: int = 256

@dataclass
class AuthConfig:
    """Authentication Configuration"""
    mfa_enabled: bool = True
    session_timeout_minutes: int = 30
    max_failed_attempts: int = 3
    lockout_duration_minutes: int = 15
    totp_issuer: str = "TradingBot"
    backup_codes_count: int = 10

@dataclass
class AuditConfig:
    """Audit Logging Configuration"""
    log_level: str = "INFO"
    log_file_path: str = "logs/audit/trades.log"
    max_file_size_mb: int = 100
    backup_count: int = 10
    immutable_logging: bool = True
    hash_verification: bool = True

@dataclass
class ComplianceConfig:
    """Compliance Reporting Configuration"""
    reports_path: str = "data/compliance/reports"
    tax_jurisdiction: str = "US"
    enable_1099_generation: bool = True
    data_retention_years: int = 7
    gdpr_compliance: bool = True
    pii_encryption: bool = True

@dataclass
class EncryptionConfig:
    """Database Encryption Configuration"""
    field_level_encryption: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_derivation: str = "PBKDF2"
    key_rotation_days: int = 90
    backup_encryption: bool = True
    in_transit_tls: bool = True

@dataclass
class MonitoringConfig:
    """Security Monitoring Configuration"""
    anomaly_detection: bool = True
    rate_limiting_enabled: bool = True
    max_requests_per_minute: int = 100
    intrusion_detection: bool = True
    alert_email: str = ""
    log_suspicious_activity: bool = True

@dataclass
class BackupConfig:
    """Backup and Disaster Recovery Configuration"""
    enabled: bool = True
    backup_interval_hours: int = 6
    retention_days: int = 30
    cloud_backup_enabled: bool = True
    encryption_enabled: bool = True
    verification_enabled: bool = True

@dataclass
class AccessConfig:
    """Access Control Configuration"""
    rbac_enabled: bool = True
    default_role: str = "viewer"
    session_management: bool = True
    ip_whitelist_enabled: bool = False
    admin_approval_required: bool = True

class SecurityConfig:
    """
    Main security configuration class that loads and validates all security settings.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/security.json"
        self.vault = VaultConfig()
        self.auth = AuthConfig()
        self.audit = AuditConfig()
        self.compliance = ComplianceConfig()
        self.encryption = EncryptionConfig()
        self.monitoring = MonitoringConfig()
        self.backup = BackupConfig()
        self.access = AccessConfig()

        self._load_config()
        self._validate_config()
        self._setup_directories()

    def _load_config(self):
        """Load configuration from file and environment variables."""
        # Load from file if it exists
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                self._apply_config_data(config_data)
                logger.info(f"Loaded security config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")

        # Override with environment variables
        self._load_env_overrides()

    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data to dataclass instances."""
        if 'vault' in config_data:
            self._update_dataclass(self.vault, config_data['vault'])
        if 'auth' in config_data:
            self._update_dataclass(self.auth, config_data['auth'])
        if 'audit' in config_data:
            self._update_dataclass(self.audit, config_data['audit'])
        if 'compliance' in config_data:
            self._update_dataclass(self.compliance, config_data['compliance'])
        if 'encryption' in config_data:
            self._update_dataclass(self.encryption, config_data['encryption'])
        if 'monitoring' in config_data:
            self._update_dataclass(self.monitoring, config_data['monitoring'])
        if 'backup' in config_data:
            self._update_dataclass(self.backup, config_data['backup'])
        if 'access' in config_data:
            self._update_dataclass(self.access, config_data['access'])

    def _update_dataclass(self, instance, data):
        """Update dataclass instance with dictionary data."""
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        # Security-critical overrides
        if os.getenv('SECURITY_MFA_ENABLED'):
            self.auth.mfa_enabled = os.getenv('SECURITY_MFA_ENABLED').lower() == 'true'

        if os.getenv('SECURITY_ENCRYPTION_ENABLED'):
            self.encryption.field_level_encryption = os.getenv('SECURITY_ENCRYPTION_ENABLED').lower() == 'true'

        if os.getenv('SECURITY_MONITORING_EMAIL'):
            self.monitoring.alert_email = os.getenv('SECURITY_MONITORING_EMAIL')

        if os.getenv('SECURITY_VAULT_ROTATION_HOURS'):
            self.vault.rotation_interval_hours = int(os.getenv('SECURITY_VAULT_ROTATION_HOURS'))

    def _validate_config(self):
        """Validate configuration for security compliance."""
        # Validate critical security settings
        if not self.auth.mfa_enabled:
            logger.warning("MFA is disabled - this is not recommended for production")

        if not self.encryption.field_level_encryption:
            logger.warning("Field-level encryption is disabled")

        if not self.monitoring.anomaly_detection:
            logger.warning("Anomaly detection is disabled")

        if self.vault.rotation_interval_hours > 168:  # 1 week
            logger.warning("API key rotation interval is longer than recommended")

        # Validate email for alerts
        if self.monitoring.anomaly_detection and not self.monitoring.alert_email:
            logger.error("Anomaly detection enabled but no alert email configured")

    def _setup_directories(self):
        """Create necessary directories for security components."""
        directories = [
            Path(self.vault.storage_path).parent,
            Path(self.audit.log_file_path).parent,
            Path(self.compliance.reports_path),
            "security/keys",
            "security/certificates",
            "data/backups",
            "logs/security"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        config_data = {
            'vault': asdict(self.vault),
            'auth': asdict(self.auth),
            'audit': asdict(self.audit),
            'compliance': asdict(self.compliance),
            'encryption': asdict(self.encryption),
            'monitoring': asdict(self.monitoring),
            'backup': asdict(self.backup),
            'access': asdict(self.access)
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Security configuration saved to {save_path}")

    def get_production_recommendations(self):
        """Get recommendations for production deployment."""
        recommendations = []

        if not self.auth.mfa_enabled:
            recommendations.append("Enable multi-factor authentication")

        if self.vault.rotation_interval_hours > 24:
            recommendations.append("Reduce API key rotation interval to 24 hours or less")

        if not self.encryption.field_level_encryption:
            recommendations.append("Enable field-level database encryption")

        if not self.monitoring.anomaly_detection:
            recommendations.append("Enable security monitoring and anomaly detection")

        if not self.backup.cloud_backup_enabled:
            recommendations.append("Enable encrypted cloud backups")

        return recommendations

# Environment-specific configurations
class ProductionSecurityConfig(SecurityConfig):
    """Production-hardened security configuration."""

    def __init__(self):
        super().__init__()
        # Override with production-safe defaults
        self.auth.mfa_enabled = True
        self.auth.session_timeout_minutes = 15
        self.vault.rotation_interval_hours = 12
        self.encryption.field_level_encryption = True
        self.monitoring.anomaly_detection = True
        self.backup.cloud_backup_enabled = True
        self.access.admin_approval_required = True

class DevelopmentSecurityConfig(SecurityConfig):
    """Development-friendly security configuration."""

    def __init__(self):
        super().__init__()
        # More relaxed settings for development
        self.auth.session_timeout_minutes = 60
        self.vault.rotation_interval_hours = 168  # 1 week
        self.monitoring.rate_limiting_enabled = False
        self.backup.backup_interval_hours = 24