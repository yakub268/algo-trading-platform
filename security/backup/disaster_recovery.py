"""
Disaster Recovery System
========================

Comprehensive disaster recovery with automated backups, cloud storage,
system state snapshots, and recovery procedures.
"""

import os
import json
import logging
import asyncio
import shutil
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import hashlib
import tarfile
import zipfile
from ..vault.encryption import AdvancedEncryption

logger = logging.getLogger(__name__)

class DisasterRecovery:
    """
    Comprehensive disaster recovery system.

    Features:
    - Automated database backups
    - System configuration snapshots
    - Cloud storage integration
    - Integrity verification
    - Automated recovery procedures
    - Point-in-time recovery
    - Multi-tiered backup strategy
    """

    def __init__(self, config):
        self.config = config
        self.encryption = AdvancedEncryption()
        self.backup_base_path = Path("data/backups")
        self.backup_base_path.mkdir(parents=True, exist_ok=True)

        # Backup categories
        self.backup_categories = {
            'database': {
                'priority': 'CRITICAL',
                'frequency': timedelta(hours=1),
                'retention_days': 30
            },
            'configuration': {
                'priority': 'HIGH',
                'frequency': timedelta(hours=6),
                'retention_days': 14
            },
            'logs': {
                'priority': 'MEDIUM',
                'frequency': timedelta(hours=12),
                'retention_days': 7
            },
            'system_state': {
                'priority': 'HIGH',
                'frequency': timedelta(hours=4),
                'retention_days': 7
            }
        }

        # Recovery procedures
        self.recovery_procedures = {
            'database_corruption': self._recover_database,
            'configuration_loss': self._recover_configuration,
            'system_failure': self._recover_system_state,
            'data_loss': self._recover_full_system
        }

        self._backup_scheduler_task = None
        self._running = False

    async def initialize(self):
        """Initialize disaster recovery system."""
        # Load encryption keys
        try:
            self.encryption.load_master_key()
        except FileNotFoundError:
            logger.info("Generating new master key for disaster recovery")
            self.encryption.generate_master_key()

        # Start backup scheduler
        if self.config.enabled:
            await self.start_backup_scheduler()

        logger.info("Disaster recovery system initialized")

    async def start_backup_scheduler(self):
        """Start automated backup scheduler."""
        if self._running:
            return

        self._running = True
        self._backup_scheduler_task = asyncio.create_task(self._backup_scheduler())
        logger.info("Backup scheduler started")

    async def stop_backup_scheduler(self):
        """Stop automated backup scheduler."""
        if not self._running:
            return

        self._running = False
        if self._backup_scheduler_task:
            self._backup_scheduler_task.cancel()
            try:
                await self._backup_scheduler_task
            except asyncio.CancelledError:
                pass

        logger.info("Backup scheduler stopped")

    async def create_full_backup(self, backup_name: Optional[str] = None) -> str:
        """
        Create a full system backup.

        Args:
            backup_name: Optional custom backup name

        Returns:
            str: Path to the backup archive
        """
        if not backup_name:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_name = f"full_backup_{timestamp}"

        backup_dir = self.backup_base_path / backup_name
        backup_dir.mkdir(exist_ok=True)

        logger.info(f"Creating full backup: {backup_name}")

        try:
            # Backup database
            db_backup_path = await self.backup_database(backup_dir / "database")

            # Backup configuration
            config_backup_path = await self.backup_configuration(backup_dir / "configuration")

            # Backup system state
            state_backup_path = await self.backup_system_state(backup_dir / "system_state")

            # Backup logs
            logs_backup_path = await self.backup_logs(backup_dir / "logs")

            # Create backup manifest
            manifest = {
                'backup_name': backup_name,
                'created_at': datetime.utcnow().isoformat(),
                'backup_type': 'FULL',
                'components': {
                    'database': str(db_backup_path) if db_backup_path else None,
                    'configuration': str(config_backup_path) if config_backup_path else None,
                    'system_state': str(state_backup_path) if state_backup_path else None,
                    'logs': str(logs_backup_path) if logs_backup_path else None
                },
                'verification': await self._create_backup_verification(backup_dir)
            }

            manifest_path = backup_dir / "backup_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

            # Create compressed archive
            archive_path = await self._create_backup_archive(backup_dir, backup_name)

            # Upload to cloud if enabled
            if self.config.cloud_backup_enabled:
                await self._upload_to_cloud(archive_path)

            # Clean up temporary directory
            shutil.rmtree(backup_dir)

            logger.info(f"Full backup completed: {archive_path}")
            return str(archive_path)

        except Exception as e:
            logger.error(f"Failed to create full backup: {e}")
            # Clean up on failure
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            raise

    async def backup_database(self, backup_path: Optional[Path] = None) -> str:
        """
        Backup all databases.

        Args:
            backup_path: Optional custom backup path

        Returns:
            str: Path to database backup
        """
        if not backup_path:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_base_path / f"database_{timestamp}"

        backup_path.mkdir(parents=True, exist_ok=True)

        logger.debug("Starting database backup")

        try:
            # Backup main trading database
            main_db_path = "data/trading_bot_encrypted.db"
            if Path(main_db_path).exists():
                backup_file = backup_path / "trading_bot.db"
                await self._backup_sqlite_db(main_db_path, backup_file)

            # Backup security databases
            security_dbs = [
                "security/auth/mfa_users.enc",
                "security/vault/keys.enc"
            ]

            for db_path in security_dbs:
                if Path(db_path).exists():
                    backup_file = backup_path / Path(db_path).name
                    shutil.copy2(db_path, backup_file)

            # Create database backup manifest
            db_manifest = {
                'backup_type': 'DATABASE',
                'created_at': datetime.utcnow().isoformat(),
                'databases': list(backup_path.glob('*')),
                'integrity_hash': await self._calculate_directory_hash(backup_path)
            }

            with open(backup_path / "db_manifest.json", 'w') as f:
                json.dump(db_manifest, f, indent=2, default=str)

            logger.debug(f"Database backup completed: {backup_path}")
            return str(backup_path)

        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise

    async def backup_configuration(self, backup_path: Optional[Path] = None) -> str:
        """
        Backup system configuration files.

        Args:
            backup_path: Optional custom backup path

        Returns:
            str: Path to configuration backup
        """
        if not backup_path:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_base_path / f"configuration_{timestamp}"

        backup_path.mkdir(parents=True, exist_ok=True)

        logger.debug("Starting configuration backup")

        try:
            # Configuration files to backup
            config_files = [
                ".env",
                "config/security.json",
                "config/freqtrade_config.json",
                "requirements.txt",
                "setup.py",
                "CLAUDE.md"
            ]

            # Configuration directories to backup
            config_dirs = [
                "config",
                ".claude",
                "security/config"
            ]

            # Copy configuration files
            for config_file in config_files:
                source_path = Path(config_file)
                if source_path.exists():
                    dest_path = backup_path / source_path.name
                    shutil.copy2(source_path, dest_path)

            # Copy configuration directories
            for config_dir in config_dirs:
                source_dir = Path(config_dir)
                if source_dir.exists():
                    dest_dir = backup_path / source_dir.name
                    shutil.copytree(source_dir, dest_dir, ignore_dangling_symlinks=True)

            # Create configuration backup manifest
            config_manifest = {
                'backup_type': 'CONFIGURATION',
                'created_at': datetime.utcnow().isoformat(),
                'files_backed_up': [str(f) for f in backup_path.rglob('*') if f.is_file()],
                'integrity_hash': await self._calculate_directory_hash(backup_path)
            }

            with open(backup_path / "config_manifest.json", 'w') as f:
                json.dump(config_manifest, f, indent=2)

            logger.debug(f"Configuration backup completed: {backup_path}")
            return str(backup_path)

        except Exception as e:
            logger.error(f"Configuration backup failed: {e}")
            raise

    async def backup_system_state(self, backup_path: Optional[Path] = None) -> str:
        """
        Backup system state and runtime information.

        Args:
            backup_path: Optional custom backup path

        Returns:
            str: Path to system state backup
        """
        if not backup_path:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_base_path / f"system_state_{timestamp}"

        backup_path.mkdir(parents=True, exist_ok=True)

        logger.debug("Starting system state backup")

        try:
            # Capture system state
            system_state = {
                'timestamp': datetime.utcnow().isoformat(),
                'python_environment': await self._capture_python_environment(),
                'system_info': await self._capture_system_info(),
                'running_processes': await self._capture_process_info(),
                'network_config': await self._capture_network_config(),
                'environment_variables': await self._capture_env_vars(),
                'installed_packages': await self._capture_installed_packages()
            }

            # Save system state
            with open(backup_path / "system_state.json", 'w') as f:
                json.dump(system_state, f, indent=2)

            # Copy critical runtime files
            runtime_files = [
                "logs/trading.log",
                "logs/system.log",
                "data/positions.json" if Path("data/positions.json").exists() else None
            ]

            for runtime_file in runtime_files:
                if runtime_file and Path(runtime_file).exists():
                    dest_path = backup_path / Path(runtime_file).name
                    shutil.copy2(runtime_file, dest_path)

            logger.debug(f"System state backup completed: {backup_path}")
            return str(backup_path)

        except Exception as e:
            logger.error(f"System state backup failed: {e}")
            raise

    async def backup_logs(self, backup_path: Optional[Path] = None) -> str:
        """
        Backup system and application logs.

        Args:
            backup_path: Optional custom backup path

        Returns:
            str: Path to logs backup
        """
        if not backup_path:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_path = self.backup_base_path / f"logs_{timestamp}"

        backup_path.mkdir(parents=True, exist_ok=True)

        logger.debug("Starting logs backup")

        try:
            # Log directories to backup
            log_dirs = ["logs", "data/logs", "security/logs"]

            for log_dir in log_dirs:
                source_dir = Path(log_dir)
                if source_dir.exists():
                    dest_dir = backup_path / source_dir.name
                    shutil.copytree(source_dir, dest_dir, ignore_dangling_symlinks=True)

            logger.debug(f"Logs backup completed: {backup_path}")
            return str(backup_path)

        except Exception as e:
            logger.error(f"Logs backup failed: {e}")
            raise

    async def restore_from_backup(self, backup_path: str, restore_type: str = 'FULL') -> bool:
        """
        Restore system from backup.

        Args:
            backup_path: Path to backup archive or directory
            restore_type: Type of restore ('FULL', 'DATABASE', 'CONFIGURATION')

        Returns:
            bool: True if restore successful
        """
        logger.info(f"Starting {restore_type} restore from {backup_path}")

        try:
            backup_dir = await self._extract_backup_archive(backup_path)

            # Load backup manifest
            manifest_path = backup_dir / "backup_manifest.json"
            if not manifest_path.exists():
                raise ValueError("Invalid backup: manifest not found")

            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            # Verify backup integrity
            if not await self._verify_backup_integrity(backup_dir, manifest):
                raise ValueError("Backup integrity verification failed")

            # Perform restore based on type
            if restore_type == 'FULL':
                await self._restore_full_system(backup_dir, manifest)
            elif restore_type == 'DATABASE':
                await self._restore_database(backup_dir)
            elif restore_type == 'CONFIGURATION':
                await self._restore_configuration(backup_dir)
            else:
                raise ValueError(f"Unsupported restore type: {restore_type}")

            logger.info(f"Restore completed successfully")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    async def get_backup_status(self) -> Dict[str, Any]:
        """
        Get current backup status and statistics.

        Returns:
            dict: Backup status information
        """
        backup_files = list(self.backup_base_path.glob("*.tar.gz"))
        backup_files.extend(list(self.backup_base_path.glob("*.zip")))

        # Calculate total backup size
        total_size = sum(f.stat().st_size for f in backup_files)

        # Get most recent backup
        most_recent = max(backup_files, key=lambda x: x.stat().st_mtime) if backup_files else None

        return {
            'backup_enabled': self.config.enabled,
            'scheduler_running': self._running,
            'total_backups': len(backup_files),
            'total_backup_size_mb': total_size / (1024 * 1024),
            'most_recent_backup': {
                'file': str(most_recent) if most_recent else None,
                'created_at': datetime.fromtimestamp(most_recent.stat().st_mtime).isoformat() if most_recent else None,
                'size_mb': most_recent.stat().st_size / (1024 * 1024) if most_recent else 0
            },
            'backup_schedule': {
                'database_frequency_hours': self.backup_categories['database']['frequency'].total_seconds() / 3600,
                'configuration_frequency_hours': self.backup_categories['configuration']['frequency'].total_seconds() / 3600,
                'next_scheduled_backup': 'Running' if self._running else 'Stopped'
            }
        }

    async def _backup_scheduler(self):
        """Background backup scheduler."""
        logger.info("Backup scheduler running")

        last_backups = {category: datetime.min for category in self.backup_categories}

        while self._running:
            try:
                current_time = datetime.utcnow()

                # Check each backup category
                for category, settings in self.backup_categories.items():
                    if current_time - last_backups[category] >= settings['frequency']:
                        try:
                            if category == 'database':
                                await self.backup_database()
                            elif category == 'configuration':
                                await self.backup_configuration()
                            elif category == 'system_state':
                                await self.backup_system_state()
                            elif category == 'logs':
                                await self.backup_logs()

                            last_backups[category] = current_time
                            logger.info(f"Scheduled {category} backup completed")

                        except Exception as e:
                            logger.error(f"Scheduled {category} backup failed: {e}")

                # Clean up old backups
                await self._cleanup_old_backups()

                # Wait before next check
                await asyncio.sleep(self.config.backup_interval_hours * 3600)

            except asyncio.CancelledError:
                logger.info("Backup scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Error in backup scheduler: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _backup_sqlite_db(self, source_db: str, backup_file: Path):
        """Backup SQLite database using proper locking."""
        source_conn = sqlite3.connect(source_db)
        backup_conn = sqlite3.connect(str(backup_file))

        try:
            source_conn.backup(backup_conn)
        finally:
            source_conn.close()
            backup_conn.close()

    async def _create_backup_archive(self, backup_dir: Path, backup_name: str) -> Path:
        """Create compressed backup archive."""
        archive_path = self.backup_base_path / f"{backup_name}.tar.gz"

        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(backup_dir, arcname=backup_name)

        return archive_path

    async def _extract_backup_archive(self, backup_path: str) -> Path:
        """Extract backup archive to temporary directory."""
        backup_path = Path(backup_path)
        extract_dir = self.backup_base_path / f"restore_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        if backup_path.suffix == '.gz':
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        elif backup_path.suffix == '.zip':
            with zipfile.ZipFile(backup_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

        # Find the actual backup directory (might be nested)
        backup_contents = list(extract_dir.iterdir())
        if len(backup_contents) == 1 and backup_contents[0].is_dir():
            return backup_contents[0]
        else:
            return extract_dir

    async def _create_backup_verification(self, backup_dir: Path) -> Dict[str, str]:
        """Create verification hashes for backup integrity."""
        verification = {}

        for file_path in backup_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(backup_dir)
                file_hash = await self._calculate_file_hash(file_path)
                verification[str(relative_path)] = file_hash

        return verification

    async def _verify_backup_integrity(self, backup_dir: Path, manifest: Dict[str, Any]) -> bool:
        """Verify backup integrity using stored hashes."""
        try:
            verification = manifest.get('verification', {})

            for file_path in backup_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(backup_dir)
                    expected_hash = verification.get(str(relative_path))

                    if expected_hash:
                        actual_hash = await self._calculate_file_hash(file_path)
                        if actual_hash != expected_hash:
                            logger.error(f"Hash mismatch for {relative_path}")
                            return False

            return True

        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False

    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    async def _calculate_directory_hash(self, directory: Path) -> str:
        """Calculate hash of all files in a directory."""
        all_hashes = []

        for file_path in sorted(directory.rglob('*')):
            if file_path.is_file():
                file_hash = await self._calculate_file_hash(file_path)
                all_hashes.append(f"{file_path.relative_to(directory)}:{file_hash}")

        combined_hash = hashlib.sha256('\n'.join(all_hashes).encode()).hexdigest()
        return combined_hash

    async def _cleanup_old_backups(self):
        """Clean up old backup files based on retention policy."""
        for category, settings in self.backup_categories.items():
            retention_days = settings['retention_days']
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

            # Find old backup files
            pattern = f"{category}_*"
            old_files = []

            for backup_file in self.backup_base_path.glob(f"{pattern}.*"):
                if datetime.fromtimestamp(backup_file.stat().st_mtime) < cutoff_date:
                    old_files.append(backup_file)

            # Delete old files
            for old_file in old_files:
                try:
                    old_file.unlink()
                    logger.debug(f"Deleted old backup: {old_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete old backup {old_file}: {e}")

    # System information capture methods
    async def _capture_python_environment(self) -> Dict[str, Any]:
        """Capture Python environment information."""
        import sys
        import platform

        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'executable': sys.executable,
            'path': sys.path
        }

    async def _capture_system_info(self) -> Dict[str, Any]:
        """Capture system information."""
        try:
            import psutil
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_usage': {str(p.mountpoint): p.usage._asdict()
                             for p in psutil.disk_partitions()},
                'boot_time': psutil.boot_time()
            }
        except ImportError:
            return {'error': 'psutil not available'}

    async def _capture_process_info(self) -> List[Dict[str, Any]]:
        """Capture running process information."""
        try:
            import psutil
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return processes
        except ImportError:
            return []

    async def _capture_network_config(self) -> Dict[str, Any]:
        """Capture network configuration."""
        try:
            import psutil
            return {
                'network_interfaces': {name: addr._asdict()
                                     for name, addrs in psutil.net_if_addrs().items()
                                     for addr in addrs},
                'network_stats': psutil.net_io_counters()._asdict()
            }
        except ImportError:
            return {}

    async def _capture_env_vars(self) -> Dict[str, str]:
        """Capture relevant environment variables (filtered for security)."""
        safe_vars = ['PATH', 'HOME', 'USER', 'SHELL', 'LANG', 'TZ']
        return {var: os.getenv(var, '') for var in safe_vars if os.getenv(var)}

    async def _capture_installed_packages(self) -> List[str]:
        """Capture list of installed Python packages."""
        try:
            import pkg_resources
            return [str(d) for d in pkg_resources.working_set]
        except ImportError:
            return []

    # Recovery procedure methods (placeholders)
    async def _restore_full_system(self, backup_dir: Path, manifest: Dict[str, Any]):
        """Restore full system from backup."""
        await self._restore_database(backup_dir)
        await self._restore_configuration(backup_dir)
        logger.info("Full system restore completed")

    async def _restore_database(self, backup_dir: Path):
        """Restore database from backup."""
        db_backup_dir = backup_dir / "database"
        if db_backup_dir.exists():
            # Restore main database
            main_db_backup = db_backup_dir / "trading_bot.db"
            if main_db_backup.exists():
                shutil.copy2(main_db_backup, "data/trading_bot_encrypted.db")
                logger.info("Main database restored")

    async def _restore_configuration(self, backup_dir: Path):
        """Restore configuration from backup."""
        config_backup_dir = backup_dir / "configuration"
        if config_backup_dir.exists():
            # Restore configuration files
            for config_file in config_backup_dir.rglob('*'):
                if config_file.is_file():
                    dest_path = Path(config_file.name)
                    shutil.copy2(config_file, dest_path)
            logger.info("Configuration restored")

    # Placeholder recovery methods
    async def _recover_database(self, **kwargs):
        """Recover from database corruption."""
        logger.info("Recovering from database corruption")

    async def _recover_configuration(self, **kwargs):
        """Recover from configuration loss."""
        logger.info("Recovering from configuration loss")

    async def _recover_system_state(self, **kwargs):
        """Recover from system failure."""
        logger.info("Recovering from system failure")

    async def _recover_full_system(self, **kwargs):
        """Recover from complete data loss."""
        logger.info("Recovering from full system failure")

    async def _upload_to_cloud(self, backup_file: Path):
        """Upload backup to cloud storage (placeholder)."""
        logger.info(f"Would upload backup to cloud: {backup_file}")
        # Implementation would upload to AWS S3, Google Cloud, etc.