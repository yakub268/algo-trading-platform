"""
API Key Vault Manager
====================

Secure storage and automatic rotation of API keys with enterprise-grade security.
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import secrets
from dataclasses import dataclass, asdict
from .encryption import AdvancedEncryption

logger = logging.getLogger(__name__)

@dataclass
class APIKey:
    """Represents an API key with metadata."""
    key_id: str
    service: str
    key_value: str
    secret_value: Optional[str] = None
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    last_rotated: Optional[datetime] = None
    rotation_count: int = 0
    is_active: bool = True
    permissions: List[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.permissions is None:
            self.permissions = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for field in ['created_at', 'expires_at', 'last_rotated']:
            if data[field]:
                data[field] = data[field].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIKey':
        """Create from dictionary."""
        # Convert ISO strings back to datetime objects
        for field in ['created_at', 'expires_at', 'last_rotated']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)

    def needs_rotation(self, rotation_interval: timedelta) -> bool:
        """Check if the key needs rotation."""
        if not self.last_rotated:
            return True
        return datetime.utcnow() - self.last_rotated > rotation_interval

    def is_expired(self) -> bool:
        """Check if the key has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at

class APIKeyVault:
    """
    Secure vault for storing and managing API keys with automatic rotation.
    """

    def __init__(self, config):
        self.config = config
        self.encryption = AdvancedEncryption(config.encryption_key_path)
        self.storage_path = config.storage_path
        self.rotation_interval = timedelta(hours=config.rotation_interval_hours)
        self.keys: Dict[str, APIKey] = {}
        self._rotation_task = None
        self._initialized = False

    async def initialize(self):
        """Initialize the vault and start rotation scheduler."""
        if self._initialized:
            return

        # Ensure the master key exists
        try:
            self.encryption.load_master_key()
        except FileNotFoundError:
            logger.info("Generating new master key for API vault")
            self.encryption.generate_master_key()

        # Load existing keys
        await self._load_keys()

        # Start automatic rotation if enabled
        if self.config.auto_rotation_enabled:
            self._rotation_task = asyncio.create_task(self._rotation_scheduler())

        self._initialized = True
        logger.info("API Key Vault initialized successfully")

    async def shutdown(self):
        """Gracefully shutdown the vault."""
        if self._rotation_task:
            self._rotation_task.cancel()
            try:
                await self._rotation_task
            except asyncio.CancelledError:
                pass

        # Save current state
        await self._save_keys()
        logger.info("API Key Vault shutdown complete")

    async def store_key(self, service: str, key_value: str, secret_value: Optional[str] = None,
                       expires_at: Optional[datetime] = None, permissions: Optional[List[str]] = None) -> str:
        """
        Store a new API key in the vault.

        Args:
            service: The service this key belongs to (e.g., 'alpaca', 'oanda')
            key_value: The API key value
            secret_value: Optional secret key value
            expires_at: Optional expiration datetime
            permissions: Optional list of permissions

        Returns:
            str: The key ID for retrieval
        """
        key_id = self._generate_key_id()

        api_key = APIKey(
            key_id=key_id,
            service=service,
            key_value=key_value,
            secret_value=secret_value,
            expires_at=expires_at,
            permissions=permissions or []
        )

        self.keys[key_id] = api_key
        await self._save_keys()

        logger.info(f"API key stored for service '{service}' with ID '{key_id}'")
        return key_id

    async def get_key(self, key_id: str) -> Optional[APIKey]:
        """
        Retrieve an API key by ID.

        Args:
            key_id: The key ID

        Returns:
            APIKey: The API key if found and active, None otherwise
        """
        api_key = self.keys.get(key_id)

        if not api_key:
            logger.warning(f"API key '{key_id}' not found")
            return None

        if not api_key.is_active:
            logger.warning(f"API key '{key_id}' is inactive")
            return None

        if api_key.is_expired():
            logger.warning(f"API key '{key_id}' has expired")
            await self.deactivate_key(key_id)
            return None

        return api_key

    async def get_keys_for_service(self, service: str, active_only: bool = True) -> List[APIKey]:
        """
        Get all keys for a specific service.

        Args:
            service: The service name
            active_only: Whether to return only active keys

        Returns:
            List[APIKey]: List of API keys for the service
        """
        keys = []
        for api_key in self.keys.values():
            if api_key.service == service:
                if not active_only or (api_key.is_active and not api_key.is_expired()):
                    keys.append(api_key)
        return keys

    async def rotate_key(self, key_id: str, new_key_value: str, new_secret_value: Optional[str] = None) -> bool:
        """
        Rotate an existing API key.

        Args:
            key_id: The key ID to rotate
            new_key_value: The new API key value
            new_secret_value: The new secret value (if applicable)

        Returns:
            bool: True if rotation successful
        """
        api_key = self.keys.get(key_id)
        if not api_key:
            logger.error(f"Cannot rotate key '{key_id}': key not found")
            return False

        # Update the key
        api_key.key_value = new_key_value
        if new_secret_value:
            api_key.secret_value = new_secret_value
        api_key.last_rotated = datetime.utcnow()
        api_key.rotation_count += 1

        await self._save_keys()
        logger.info(f"API key '{key_id}' rotated successfully (rotation #{api_key.rotation_count})")
        return True

    async def deactivate_key(self, key_id: str) -> bool:
        """
        Deactivate an API key.

        Args:
            key_id: The key ID to deactivate

        Returns:
            bool: True if deactivation successful
        """
        api_key = self.keys.get(key_id)
        if not api_key:
            logger.error(f"Cannot deactivate key '{key_id}': key not found")
            return False

        api_key.is_active = False
        await self._save_keys()
        logger.info(f"API key '{key_id}' deactivated")
        return True

    async def delete_key(self, key_id: str) -> bool:
        """
        Permanently delete an API key.

        Args:
            key_id: The key ID to delete

        Returns:
            bool: True if deletion successful
        """
        if key_id not in self.keys:
            logger.error(f"Cannot delete key '{key_id}': key not found")
            return False

        del self.keys[key_id]
        await self._save_keys()
        logger.info(f"API key '{key_id}' permanently deleted")
        return True

    async def list_keys(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """
        List all keys with metadata (excluding sensitive data).

        Args:
            include_inactive: Whether to include inactive keys

        Returns:
            List[Dict]: List of key metadata
        """
        keys_info = []
        for api_key in self.keys.values():
            if include_inactive or api_key.is_active:
                info = {
                    'key_id': api_key.key_id,
                    'service': api_key.service,
                    'created_at': api_key.created_at.isoformat(),
                    'last_rotated': api_key.last_rotated.isoformat() if api_key.last_rotated else None,
                    'rotation_count': api_key.rotation_count,
                    'is_active': api_key.is_active,
                    'expires_at': api_key.expires_at.isoformat() if api_key.expires_at else None,
                    'is_expired': api_key.is_expired(),
                    'needs_rotation': api_key.needs_rotation(self.rotation_interval),
                    'permissions': api_key.permissions
                }
                keys_info.append(info)

        return keys_info

    async def get_rotation_status(self) -> Dict[str, Any]:
        """
        Get the current rotation status for all keys.

        Returns:
            Dict: Rotation status information
        """
        total_keys = len(self.keys)
        active_keys = sum(1 for k in self.keys.values() if k.is_active)
        expired_keys = sum(1 for k in self.keys.values() if k.is_expired())
        needs_rotation = sum(1 for k in self.keys.values() if k.needs_rotation(self.rotation_interval))

        return {
            'total_keys': total_keys,
            'active_keys': active_keys,
            'expired_keys': expired_keys,
            'needs_rotation': needs_rotation,
            'auto_rotation_enabled': self.config.auto_rotation_enabled,
            'rotation_interval_hours': self.config.rotation_interval_hours,
            'last_check': datetime.utcnow().isoformat()
        }

    def _generate_key_id(self) -> str:
        """Generate a unique key ID."""
        return f"key_{secrets.token_urlsafe(16)}"

    async def _load_keys(self):
        """Load keys from encrypted storage."""
        if not Path(self.storage_path).exists():
            logger.info("No existing key storage found, starting with empty vault")
            return

        try:
            with open(self.storage_path, 'r') as f:
                encrypted_data = json.load(f)

            # Decrypt the data
            decrypted_data = self.encryption.decrypt_data(encrypted_data)
            keys_data = json.loads(decrypted_data.decode())

            # Load keys
            for key_data in keys_data:
                api_key = APIKey.from_dict(key_data)
                self.keys[api_key.key_id] = api_key

            logger.info(f"Loaded {len(self.keys)} API keys from storage")

        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            raise

    async def _save_keys(self):
        """Save keys to encrypted storage."""
        try:
            # Convert keys to serializable format
            keys_data = [key.to_dict() for key in self.keys.values()]
            data_json = json.dumps(keys_data, indent=2)

            # Encrypt the data
            encrypted_data = self.encryption.encrypt_data(data_json)

            # Save to file with backup
            await self._create_backup()
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)

            with open(self.storage_path, 'w') as f:
                json.dump(encrypted_data, f, indent=2)

            logger.debug(f"Saved {len(self.keys)} API keys to encrypted storage")

        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
            raise

    async def _create_backup(self):
        """Create a backup of the current key storage."""
        if not Path(self.storage_path).exists():
            return

        backup_path = f"{self.storage_path}.backup.{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        try:
            import shutil
            shutil.copy2(self.storage_path, backup_path)

            # Clean up old backups
            await self._cleanup_old_backups()

        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    async def _cleanup_old_backups(self):
        """Clean up old backup files."""
        backup_dir = Path(self.storage_path).parent
        pattern = f"{Path(self.storage_path).name}.backup.*"

        backups = list(backup_dir.glob(pattern))
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Keep only the configured number of backups
        for old_backup in backups[self.config.backup_count:]:
            try:
                old_backup.unlink()
                logger.debug(f"Cleaned up old backup: {old_backup}")
            except Exception as e:
                logger.warning(f"Failed to clean up backup {old_backup}: {e}")

    async def _rotation_scheduler(self):
        """Background task for automatic key rotation."""
        logger.info("API key rotation scheduler started")

        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour

                # Check for keys that need rotation
                for api_key in self.keys.values():
                    if api_key.is_active and api_key.needs_rotation(self.rotation_interval):
                        logger.info(f"API key '{api_key.key_id}' for service '{api_key.service}' needs rotation")
                        # Note: Actual rotation requires new keys from the service
                        # This would typically trigger a notification or automated process

                # Check for expired keys
                for api_key in self.keys.values():
                    if api_key.is_active and api_key.is_expired():
                        logger.warning(f"API key '{api_key.key_id}' for service '{api_key.service}' has expired")
                        await self.deactivate_key(api_key.key_id)

            except asyncio.CancelledError:
                logger.info("API key rotation scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Error in rotation scheduler: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying