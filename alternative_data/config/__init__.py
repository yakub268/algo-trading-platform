"""
Alternative Data Configuration
=============================

Configuration management for alternative data sources.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class AltDataConfig:
    """Configuration manager for alternative data sources"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "altdata_config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def is_enabled(self, connector: str) -> bool:
        """Check if a connector is enabled"""
        return self.get(f'{connector}.enabled', False)

    def get_api_key(self, connector: str) -> Optional[str]:
        """Get API key from environment variables"""
        env_var = f"{connector.upper()}_API_KEY"
        return os.getenv(env_var)

    def get_rate_limit(self, connector: str) -> int:
        """Get rate limit for a connector"""
        return self.get(f'{connector}.rate_limit_per_minute', 60)

# Global config instance
config = AltDataConfig()

__all__ = ['AltDataConfig', 'config']