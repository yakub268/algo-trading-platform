"""
Multi-Factor Authentication Manager
===================================

Comprehensive MFA system with TOTP, backup codes, session management,
and brute-force protection for enterprise security.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import hashlib
import secrets
from dataclasses import dataclass, asdict
from .totp_generator import TOTPGenerator
from ..vault.encryption import AdvancedEncryption

logger = logging.getLogger(__name__)

@dataclass
class MFAUser:
    """Represents a user with MFA configuration."""
    user_id: str
    username: str
    email: str
    totp_secret: Optional[str] = None
    backup_codes: List[str] = None
    is_mfa_enabled: bool = False
    mfa_setup_completed: bool = False
    created_at: datetime = None
    last_totp_used: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.backup_codes is None:
            self.backup_codes = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for field in ['created_at', 'last_totp_used', 'locked_until']:
            if data[field]:
                data[field] = data[field].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MFAUser':
        """Create from dictionary."""
        # Convert ISO strings back to datetime objects
        for field in ['created_at', 'last_totp_used', 'locked_until']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)

    def is_locked(self) -> bool:
        """Check if user is currently locked due to failed attempts."""
        if not self.locked_until:
            return False
        return datetime.utcnow() < self.locked_until

@dataclass
class AuthenticationAttempt:
    """Represents an authentication attempt."""
    user_id: str
    timestamp: datetime
    success: bool
    method: str  # 'totp', 'backup_code', 'password'
    ip_address: str
    user_agent: str
    failure_reason: Optional[str] = None

class MFAManager:
    """
    Comprehensive Multi-Factor Authentication manager.

    Features:
    - TOTP (Time-based One-Time Password) authentication
    - Backup codes for account recovery
    - Brute-force protection with account lockout
    - Session management integration
    - Audit logging of authentication events
    - QR code generation for easy setup
    """

    def __init__(self, config):
        self.config = config
        self.encryption = AdvancedEncryption()
        self.totp_generator = TOTPGenerator(issuer=config.totp_issuer)
        self.users_file = "security/auth/mfa_users.enc"
        self.users: Dict[str, MFAUser] = {}
        self.auth_attempts: List[AuthenticationAttempt] = []
        self._initialized = False

    async def initialize(self):
        """Initialize the MFA manager."""
        if self._initialized:
            return

        # Load master key
        try:
            self.encryption.load_master_key()
        except FileNotFoundError:
            logger.info("Generating new master key for MFA system")
            self.encryption.generate_master_key()

        # Load existing users
        await self._load_users()
        self._initialized = True
        logger.info("MFA Manager initialized successfully")

    def create_user(self, user_id: str, username: str, email: str) -> MFAUser:
        """
        Create a new MFA user.

        Args:
            user_id: Unique user identifier
            username: Username
            email: User email address

        Returns:
            MFAUser: Created user object
        """
        if user_id in self.users:
            raise ValueError(f"User {user_id} already exists")

        user = MFAUser(
            user_id=user_id,
            username=username,
            email=email
        )

        self.users[user_id] = user
        self._save_users()
        logger.info(f"Created MFA user: {username} ({user_id})")
        return user

    def setup_totp(self, user_id: str) -> Tuple[str, str, bytes]:
        """
        Set up TOTP for a user.

        Args:
            user_id: User identifier

        Returns:
            tuple: (secret, provisioning_uri, qr_code_png)
        """
        user = self._get_user(user_id)

        # Generate new TOTP secret
        secret = self.totp_generator.generate_secret()
        user.totp_secret = secret

        # Generate backup codes
        user.backup_codes = self._generate_backup_codes()

        # Create provisioning URI and QR code
        provisioning_uri = self.totp_generator.get_provisioning_uri(secret, user.email)
        qr_code = self.totp_generator.generate_qr_code(secret, user.email)

        self._save_users()
        logger.info(f"TOTP setup initiated for user {user_id}")

        return secret, provisioning_uri, qr_code

    def complete_totp_setup(self, user_id: str, verification_code: str) -> bool:
        """
        Complete TOTP setup by verifying the first code.

        Args:
            user_id: User identifier
            verification_code: TOTP code from authenticator app

        Returns:
            bool: True if setup completed successfully
        """
        user = self._get_user(user_id)

        if not user.totp_secret:
            raise ValueError("TOTP not set up for this user")

        if user.mfa_setup_completed:
            raise ValueError("MFA already set up for this user")

        # Verify the code
        if self.totp_generator.verify_code(user.totp_secret, verification_code):
            user.is_mfa_enabled = True
            user.mfa_setup_completed = True
            user.last_totp_used = datetime.utcnow()
            self._save_users()
            logger.info(f"TOTP setup completed for user {user_id}")
            return True

        logger.warning(f"Invalid verification code during TOTP setup for user {user_id}")
        return False

    def verify_totp(self, user_id: str, totp_code: str, ip_address: str = "unknown",
                   user_agent: str = "unknown") -> bool:
        """
        Verify a TOTP code for authentication.

        Args:
            user_id: User identifier
            totp_code: TOTP code to verify
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            bool: True if code is valid and user is authenticated
        """
        user = self._get_user(user_id)

        # Check if user is locked
        if user.is_locked():
            self._log_auth_attempt(user_id, False, 'totp', ip_address, user_agent, 'Account locked')
            raise ValueError("Account is temporarily locked due to too many failed attempts")

        if not user.is_mfa_enabled or not user.totp_secret:
            self._log_auth_attempt(user_id, False, 'totp', ip_address, user_agent, 'MFA not enabled')
            return False

        # Verify the TOTP code
        if self.totp_generator.verify_code(user.totp_secret, totp_code):
            # Success - reset failed attempts and update last used
            user.failed_attempts = 0
            user.locked_until = None
            user.last_totp_used = datetime.utcnow()
            self._save_users()
            self._log_auth_attempt(user_id, True, 'totp', ip_address, user_agent)
            logger.info(f"Successful TOTP authentication for user {user_id}")
            return True
        else:
            # Failed attempt
            user.failed_attempts += 1
            if user.failed_attempts >= self.config.max_failed_attempts:
                user.locked_until = datetime.utcnow() + timedelta(minutes=self.config.lockout_duration_minutes)
                logger.warning(f"User {user_id} locked after {user.failed_attempts} failed attempts")

            self._save_users()
            self._log_auth_attempt(user_id, False, 'totp', ip_address, user_agent, 'Invalid TOTP code')
            logger.warning(f"Failed TOTP authentication for user {user_id} (attempt {user.failed_attempts})")
            return False

    def verify_backup_code(self, user_id: str, backup_code: str, ip_address: str = "unknown",
                          user_agent: str = "unknown") -> bool:
        """
        Verify a backup code for authentication.

        Args:
            user_id: User identifier
            backup_code: Backup code to verify
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            bool: True if backup code is valid (code is consumed after use)
        """
        user = self._get_user(user_id)

        # Check if user is locked
        if user.is_locked():
            self._log_auth_attempt(user_id, False, 'backup_code', ip_address, user_agent, 'Account locked')
            raise ValueError("Account is temporarily locked due to too many failed attempts")

        if not user.is_mfa_enabled:
            self._log_auth_attempt(user_id, False, 'backup_code', ip_address, user_agent, 'MFA not enabled')
            return False

        # Check if backup code exists
        if backup_code in user.backup_codes:
            # Remove the used backup code
            user.backup_codes.remove(backup_code)
            user.failed_attempts = 0
            user.locked_until = None
            self._save_users()
            self._log_auth_attempt(user_id, True, 'backup_code', ip_address, user_agent)
            logger.info(f"Successful backup code authentication for user {user_id}")
            return True
        else:
            # Failed attempt
            user.failed_attempts += 1
            if user.failed_attempts >= self.config.max_failed_attempts:
                user.locked_until = datetime.utcnow() + timedelta(minutes=self.config.lockout_duration_minutes)

            self._save_users()
            self._log_auth_attempt(user_id, False, 'backup_code', ip_address, user_agent, 'Invalid backup code')
            logger.warning(f"Failed backup code authentication for user {user_id}")
            return False

    def regenerate_backup_codes(self, user_id: str) -> List[str]:
        """
        Regenerate backup codes for a user.

        Args:
            user_id: User identifier

        Returns:
            list: New backup codes
        """
        user = self._get_user(user_id)
        user.backup_codes = self._generate_backup_codes()
        self._save_users()
        logger.info(f"Regenerated backup codes for user {user_id}")
        return user.backup_codes.copy()

    def disable_mfa(self, user_id: str) -> bool:
        """
        Disable MFA for a user.

        Args:
            user_id: User identifier

        Returns:
            bool: True if MFA was disabled
        """
        user = self._get_user(user_id)
        user.is_mfa_enabled = False
        user.mfa_setup_completed = False
        user.totp_secret = None
        user.backup_codes = []
        user.failed_attempts = 0
        user.locked_until = None
        self._save_users()
        logger.info(f"MFA disabled for user {user_id}")
        return True

    def unlock_user(self, user_id: str) -> bool:
        """
        Unlock a user account (admin function).

        Args:
            user_id: User identifier

        Returns:
            bool: True if user was unlocked
        """
        user = self._get_user(user_id)
        user.failed_attempts = 0
        user.locked_until = None
        self._save_users()
        logger.info(f"User {user_id} manually unlocked")
        return True

    def get_user_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get MFA status for a user.

        Args:
            user_id: User identifier

        Returns:
            dict: User MFA status information
        """
        user = self._get_user(user_id)

        return {
            'user_id': user.user_id,
            'username': user.username,
            'email': user.email,
            'mfa_enabled': user.is_mfa_enabled,
            'setup_completed': user.mfa_setup_completed,
            'backup_codes_remaining': len(user.backup_codes),
            'failed_attempts': user.failed_attempts,
            'is_locked': user.is_locked(),
            'locked_until': user.locked_until.isoformat() if user.locked_until else None,
            'last_totp_used': user.last_totp_used.isoformat() if user.last_totp_used else None,
            'created_at': user.created_at.isoformat()
        }

    def get_auth_attempts(self, user_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get authentication attempts.

        Args:
            user_id: Optional user ID to filter by
            limit: Maximum number of attempts to return

        Returns:
            list: Authentication attempt records
        """
        attempts = self.auth_attempts

        if user_id:
            attempts = [a for a in attempts if a.user_id == user_id]

        # Sort by timestamp descending and apply limit
        attempts.sort(key=lambda x: x.timestamp, reverse=True)
        attempts = attempts[:limit]

        return [
            {
                'user_id': a.user_id,
                'timestamp': a.timestamp.isoformat(),
                'success': a.success,
                'method': a.method,
                'ip_address': a.ip_address,
                'user_agent': a.user_agent,
                'failure_reason': a.failure_reason
            }
            for a in attempts
        ]

    def _get_user(self, user_id: str) -> MFAUser:
        """Get user or raise exception if not found."""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        return self.users[user_id]

    def _generate_backup_codes(self, count: int = None) -> List[str]:
        """Generate backup codes."""
        count = count or self.config.backup_codes_count
        return self.totp_generator.get_backup_codes(count)

    def _log_auth_attempt(self, user_id: str, success: bool, method: str,
                         ip_address: str, user_agent: str, failure_reason: str = None):
        """Log an authentication attempt."""
        attempt = AuthenticationAttempt(
            user_id=user_id,
            timestamp=datetime.utcnow(),
            success=success,
            method=method,
            ip_address=ip_address,
            user_agent=user_agent,
            failure_reason=failure_reason
        )

        self.auth_attempts.append(attempt)

        # Keep only recent attempts (last 1000)
        if len(self.auth_attempts) > 1000:
            self.auth_attempts = self.auth_attempts[-1000:]

    async def _load_users(self):
        """Load users from encrypted storage."""
        if not Path(self.users_file).exists():
            logger.info("No existing MFA users found, starting with empty database")
            return

        try:
            with open(self.users_file, 'r') as f:
                encrypted_data = json.load(f)

            # Decrypt the data
            decrypted_data = self.encryption.decrypt_data(encrypted_data)
            users_data = json.loads(decrypted_data.decode())

            # Load users
            for user_data in users_data:
                user = MFAUser.from_dict(user_data)
                self.users[user.user_id] = user

            logger.info(f"Loaded {len(self.users)} MFA users from storage")

        except Exception as e:
            logger.error(f"Failed to load MFA users: {e}")
            raise

    def _save_users(self):
        """Save users to encrypted storage."""
        try:
            # Convert users to serializable format
            users_data = [user.to_dict() for user in self.users.values()]
            data_json = json.dumps(users_data, indent=2)

            # Encrypt the data
            encrypted_data = self.encryption.encrypt_data(data_json)

            # Save to file
            Path(self.users_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.users_file, 'w') as f:
                json.dump(encrypted_data, f, indent=2)

            logger.debug(f"Saved {len(self.users)} MFA users to encrypted storage")

        except Exception as e:
            logger.error(f"Failed to save MFA users: {e}")
            raise