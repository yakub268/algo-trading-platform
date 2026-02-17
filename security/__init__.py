"""
Enterprise Security Framework for Trading Bot
============================================

This module provides comprehensive security hardening and regulatory compliance
features for institutional-grade trading operations.

Key Features:
- API key vault with automatic rotation
- Trade logging and audit trails
- Multi-factor authentication
- Database encryption (at rest and in transit)
- Network security monitoring
- Backup and disaster recovery
- Compliance reporting (tax/regulatory)
- Role-based access control
- Security monitoring and anomaly detection

Usage:
    from security import SecurityManager, ComplianceManager

    security = SecurityManager()
    compliance = ComplianceManager()
"""

from .vault.api_key_manager import APIKeyVault
from .auth.mfa_manager import MFAManager
from .audit.trade_logger import AuditTradeLogger
from .compliance.reporting import ComplianceReporter
from .encryption.database import EncryptedDB
from .monitoring.security_monitor import SecurityMonitor
from .backup.disaster_recovery import DisasterRecovery
from .access_control.rbac import RoleBasedAccess
from .config import SecurityConfig

__version__ = "1.0.0"
__author__ = "Trading Bot Security Team"

class SecurityManager:
    """
    Central security management class that orchestrates all security components.
    """

    def __init__(self, config_path: str = None):
        self.config = SecurityConfig(config_path)
        self.api_vault = APIKeyVault(self.config.vault)
        self.mfa = MFAManager(self.config.auth)
        self.audit_logger = AuditTradeLogger(self.config.audit)
        self.compliance = ComplianceReporter(self.config.compliance)
        self.encrypted_db = EncryptedDB(self.config.encryption)
        self.security_monitor = SecurityMonitor(self.config.monitoring)
        self.disaster_recovery = DisasterRecovery(self.config.backup)
        self.access_control = RoleBasedAccess(self.config.access)

    async def initialize(self):
        """Initialize all security components."""
        await self.api_vault.initialize()
        await self.encrypted_db.initialize()
        await self.security_monitor.start_monitoring()
        self.audit_logger.start()

    async def shutdown(self):
        """Gracefully shutdown all security components."""
        await self.security_monitor.stop_monitoring()
        await self.audit_logger.stop()
        await self.api_vault.shutdown()

class ComplianceManager:
    """
    Simplified interface for compliance-related operations.
    """

    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        self.reporter = security_manager.compliance
        self.audit_logger = security_manager.audit_logger

    def log_trade(self, trade_data: dict):
        """Log trade for compliance purposes."""
        return self.audit_logger.log_trade(trade_data)

    def generate_tax_report(self, year: int):
        """Generate tax compliance report."""
        return self.reporter.generate_tax_report(year)

    def generate_audit_trail(self, start_date: str, end_date: str):
        """Generate audit trail for specified period."""
        return self.reporter.generate_audit_trail(start_date, end_date)

# Global security instance (lazy initialization)
_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager

def get_compliance_manager() -> ComplianceManager:
    """Get the global compliance manager instance."""
    return ComplianceManager(get_security_manager())

__all__ = [
    'SecurityManager',
    'ComplianceManager',
    'get_security_manager',
    'get_compliance_manager',
    'APIKeyVault',
    'MFAManager',
    'AuditTradeLogger',
    'ComplianceReporter',
    'EncryptedDB',
    'SecurityMonitor',
    'DisasterRecovery',
    'RoleBasedAccess'
]