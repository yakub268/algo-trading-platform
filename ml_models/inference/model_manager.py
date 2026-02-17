"""
Model Manager
============

Manages model persistence, versioning, and lifecycle.
Handles loading, saving, and updating of ML models.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import json
import shutil
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..predictors.base_predictor import BasePredictor
from ..predictors.price_direction_model import PriceDirectionPredictor
from ..predictors.volatility_model import VolatilityPredictor

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages ML model persistence and versioning.

    Features:
    - Model saving and loading with versioning
    - Model registry and metadata management
    - Automatic model backup and rotation
    - Model performance tracking
    - Hot model updates without downtime
    """

    def __init__(self, model_directory: str = None):
        """
        Initialize model manager.

        Args:
            model_directory: Directory to store models
        """
        self.model_directory = Path(model_directory) if model_directory else Path("models")
        self.model_directory.mkdir(parents=True, exist_ok=True)

        # Model registry file
        self.registry_file = self.model_directory / "model_registry.json"
        self.registry = self._load_registry()

        # Backup directory
        self.backup_directory = self.model_directory / "backups"
        self.backup_directory.mkdir(exist_ok=True)

        logger.info(f"ModelManager initialized: {self.model_directory}")

    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model registry: {e}")

        return {
            'models': {},
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }

    def _save_registry(self):
        """Save model registry to disk"""
        try:
            self.registry['last_updated'] = datetime.now().isoformat()
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")

    def save_model(self,
                   model: BasePredictor,
                   model_name: str,
                   version: str = None,
                   metadata: Dict[str, Any] = None,
                   create_backup: bool = True) -> str:
        """
        Save model to disk with versioning.

        Args:
            model: Model instance to save
            model_name: Name of the model
            version: Version string (auto-generated if None)
            metadata: Additional metadata to store
            create_backup: Whether to backup existing model

        Returns:
            Path to saved model
        """
        if version is None:
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")

        # Create model directory
        model_dir = self.model_directory / model_name
        model_dir.mkdir(exist_ok=True)

        # Model file path
        model_file = model_dir / f"{model_name}_{version}.joblib"

        try:
            # Backup existing model if requested
            if create_backup:
                self._backup_model(model_name)

            # Save model
            model.save_model(str(model_file))

            # Update registry
            model_info = {
                'name': model_name,
                'version': version,
                'type': model.model_type,
                'class': model.__class__.__name__,
                'file_path': str(model_file.relative_to(self.model_directory)),
                'created_at': datetime.now().isoformat(),
                'model_version': getattr(model, 'model_version', '1.0.0'),
                'last_trained': getattr(model, 'last_trained', None),
                'training_metrics': self._serialize_metrics(getattr(model, 'training_metrics', None)),
                'validation_metrics': self._serialize_metrics(getattr(model, 'validation_metrics', None)),
                'feature_count': getattr(model, 'n_features', 0),
                'is_fitted': getattr(model, 'is_fitted', False),
                'metadata': metadata or {}
            }

            # Add to registry
            if model_name not in self.registry['models']:
                self.registry['models'][model_name] = {}

            self.registry['models'][model_name][version] = model_info
            self._save_registry()

            logger.info(f"Model saved: {model_name} v{version}")
            return str(model_file)

        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            raise

    def load_model(self,
                   model_name: str,
                   version: str = "latest") -> Optional[BasePredictor]:
        """
        Load model from disk.

        Args:
            model_name: Name of the model
            version: Version to load ("latest" for most recent)

        Returns:
            Loaded model instance or None if not found
        """
        try:
            # Get model info from registry
            model_info = self._get_model_info(model_name, version)
            if not model_info:
                logger.warning(f"Model {model_name} v{version} not found in registry")
                return None

            # Get model file path
            model_file = self.model_directory / model_info['file_path']
            if not model_file.exists():
                logger.warning(f"Model file not found: {model_file}")
                return None

            # Create model instance based on class
            model_class = self._get_model_class(model_info['class'])
            if not model_class:
                logger.error(f"Unknown model class: {model_info['class']}")
                return None

            # Create instance and load
            model = model_class()
            model.load_model(str(model_file))

            logger.info(f"Model loaded: {model_name} v{model_info['version']}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None

    def load_all_models(self) -> Dict[str, BasePredictor]:
        """
        Load all latest models.

        Returns:
            Dict mapping model names to loaded instances
        """
        models = {}

        for model_name in self.registry['models'].keys():
            try:
                model = self.load_model(model_name, "latest")
                if model is not None:
                    models[model_name] = model
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")

        logger.info(f"Loaded {len(models)} models")
        return models

    def list_models(self, model_name: str = None) -> Dict[str, Any]:
        """
        List available models and versions.

        Args:
            model_name: Specific model name (None for all models)

        Returns:
            Dict with model information
        """
        if model_name and model_name in self.registry['models']:
            return {model_name: self.registry['models'][model_name]}
        elif model_name:
            return {}
        else:
            return self.registry['models']

    def delete_model(self,
                     model_name: str,
                     version: str = None,
                     keep_latest: bool = True) -> bool:
        """
        Delete model(s) from disk and registry.

        Args:
            model_name: Name of the model
            version: Specific version (None to delete all versions)
            keep_latest: Keep latest version when deleting all

        Returns:
            True if successful
        """
        try:
            if model_name not in self.registry['models']:
                logger.warning(f"Model {model_name} not found")
                return False

            model_versions = self.registry['models'][model_name]

            if version:
                # Delete specific version
                if version in model_versions:
                    model_info = model_versions[version]
                    model_file = self.model_directory / model_info['file_path']
                    if model_file.exists():
                        os.remove(model_file)

                    del model_versions[version]
                    logger.info(f"Deleted model: {model_name} v{version}")
                else:
                    logger.warning(f"Version {version} not found for {model_name}")
                    return False

            else:
                # Delete all versions (optionally keep latest)
                if keep_latest and len(model_versions) > 1:
                    # Find latest version
                    latest_version = max(model_versions.keys(),
                                       key=lambda v: model_versions[v]['created_at'])

                    # Delete all except latest
                    versions_to_delete = [v for v in model_versions.keys() if v != latest_version]
                else:
                    versions_to_delete = list(model_versions.keys())

                # Delete files and registry entries
                for v in versions_to_delete:
                    model_info = model_versions[v]
                    model_file = self.model_directory / model_info['file_path']
                    if model_file.exists():
                        os.remove(model_file)
                    del model_versions[v]

                logger.info(f"Deleted {len(versions_to_delete)} versions of {model_name}")

            # Clean up empty model directories
            if not model_versions:
                del self.registry['models'][model_name]
                model_dir = self.model_directory / model_name
                if model_dir.exists() and not any(model_dir.iterdir()):
                    model_dir.rmdir()

            self._save_registry()
            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False

    def _backup_model(self, model_name: str) -> bool:
        """Create backup of existing model"""
        try:
            if model_name not in self.registry['models']:
                return True  # No existing model to backup

            # Get latest model info
            model_info = self._get_model_info(model_name, "latest")
            if not model_info:
                return True

            model_file = self.model_directory / model_info['file_path']
            if not model_file.exists():
                return True

            # Create backup
            backup_name = f"{model_name}_{model_info['version']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            backup_file = self.backup_directory / backup_name

            shutil.copy2(model_file, backup_file)
            logger.debug(f"Model backup created: {backup_name}")

            # Cleanup old backups (keep last 5)
            self._cleanup_backups(model_name)

            return True

        except Exception as e:
            logger.warning(f"Failed to backup model {model_name}: {e}")
            return False

    def _cleanup_backups(self, model_name: str, keep_count: int = 5):
        """Cleanup old backup files"""
        try:
            # Find backup files for this model
            backup_pattern = f"{model_name}_*"
            backup_files = list(self.backup_directory.glob(backup_pattern))

            if len(backup_files) <= keep_count:
                return

            # Sort by modification time (oldest first)
            backup_files.sort(key=lambda f: f.stat().st_mtime)

            # Remove oldest files
            for backup_file in backup_files[:-keep_count]:
                backup_file.unlink()
                logger.debug(f"Removed old backup: {backup_file.name}")

        except Exception as e:
            logger.debug(f"Failed to cleanup backups for {model_name}: {e}")

    def _get_model_info(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get model information from registry"""
        if model_name not in self.registry['models']:
            return None

        model_versions = self.registry['models'][model_name]

        if version == "latest":
            if not model_versions:
                return None
            # Find latest version by creation date
            latest_version = max(model_versions.keys(),
                               key=lambda v: model_versions[v]['created_at'])
            return model_versions[latest_version]
        elif version in model_versions:
            return model_versions[version]
        else:
            return None

    def _get_model_class(self, class_name: str) -> Optional[type]:
        """Get model class by name"""
        class_mapping = {
            'PriceDirectionPredictor': PriceDirectionPredictor,
            'VolatilityPredictor': VolatilityPredictor,
            'BasePredictor': BasePredictor
        }

        return class_mapping.get(class_name)

    def _serialize_metrics(self, metrics) -> Optional[Dict[str, Any]]:
        """Serialize metrics for JSON storage"""
        if metrics is None:
            return None

        try:
            if hasattr(metrics, '__dict__'):
                return {k: v for k, v in metrics.__dict__.items() if v is not None}
            elif isinstance(metrics, dict):
                return metrics
            else:
                return {'value': str(metrics)}
        except Exception as e:
            logger.debug(f"Failed to serialize metrics: {e}")
            return None

    def get_model_performance(self, model_name: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """Get model performance metrics"""
        model_info = self._get_model_info(model_name, version)
        if not model_info:
            return None

        return {
            'training_metrics': model_info.get('training_metrics'),
            'validation_metrics': model_info.get('validation_metrics'),
            'last_trained': model_info.get('last_trained'),
            'version': model_info.get('version'),
            'created_at': model_info.get('created_at')
        }

    def update_model_metadata(self,
                             model_name: str,
                             version: str,
                             metadata: Dict[str, Any]) -> bool:
        """Update model metadata"""
        try:
            if model_name in self.registry['models'] and version in self.registry['models'][model_name]:
                self.registry['models'][model_name][version]['metadata'].update(metadata)
                self._save_registry()
                logger.info(f"Updated metadata for {model_name} v{version}")
                return True
            else:
                logger.warning(f"Model {model_name} v{version} not found")
                return False

        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            return False

    def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics"""
        try:
            total_size = 0
            file_count = 0

            # Calculate size of all model files
            for model_file in self.model_directory.rglob("*.joblib"):
                total_size += model_file.stat().st_size
                file_count += 1

            # Calculate backup size
            backup_size = 0
            backup_count = 0
            for backup_file in self.backup_directory.glob("*.joblib"):
                backup_size += backup_file.stat().st_size
                backup_count += 1

            return {
                'total_size_mb': total_size / (1024 * 1024),
                'model_files': file_count,
                'backup_size_mb': backup_size / (1024 * 1024),
                'backup_files': backup_count,
                'model_count': len(self.registry['models']),
                'directory': str(self.model_directory)
            }

        except Exception as e:
            logger.error(f"Failed to calculate storage usage: {e}")
            return {}


def main():
    """Test model manager"""
    print("Testing Model Manager")
    print("=" * 30)

    # Initialize manager
    manager = ModelManager("test_models")

    # Create test model
    print("Creating test model...")
    model = PriceDirectionPredictor(sequence_length=10, lstm_units=32)

    # Save model
    print("Saving model...")
    model_path = manager.save_model(
        model,
        "test_direction_model",
        metadata={'description': 'Test model for direction prediction'}
    )
    print(f"Model saved to: {model_path}")

    # List models
    print("\nListing models...")
    models = manager.list_models()
    for name, versions in models.items():
        print(f"Model: {name}")
        for version, info in versions.items():
            print(f"  Version: {version} (created: {info['created_at']})")

    # Load model
    print("\nLoading model...")
    loaded_model = manager.load_model("test_direction_model")
    if loaded_model:
        print(f"Model loaded successfully: {loaded_model.model_name}")
    else:
        print("Failed to load model")

    # Get storage usage
    print("\nStorage usage:")
    usage = manager.get_storage_usage()
    for key, value in usage.items():
        print(f"  {key}: {value}")

    # Cleanup
    print("\nCleaning up...")
    manager.delete_model("test_direction_model", keep_latest=False)

    # Remove test directory
    import shutil
    shutil.rmtree("test_models", ignore_errors=True)

    print("Model manager test completed!")


if __name__ == "__main__":
    main()