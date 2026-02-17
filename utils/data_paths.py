"""
Data Path Utility - Separates Paper vs Live Trading Data

This module ensures paper trading data is completely separate from live trading data.
All databases and data files should use these paths to avoid mixing results.

Usage:
    from utils.data_paths import get_data_path, get_db_path

    # Get mode-specific database path
    db = get_db_path('trades.db')  # Returns 'data/live/trades.db' or 'data/paper/trades.db'

    # Get mode-specific data directory
    data_dir = get_data_path()  # Returns 'data/live' or 'data/paper'

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
from dotenv import load_dotenv

load_dotenv()


def is_live_mode() -> bool:
    """Check if we're in live trading mode."""
    return os.getenv('PAPER_MODE', 'true').lower() == 'false'


def get_mode_string() -> str:
    """Get current mode as string."""
    return 'live' if is_live_mode() else 'paper'


def get_data_path(subdir: str = None) -> str:
    """
    Get the mode-specific data directory path.

    Args:
        subdir: Optional subdirectory within the mode directory

    Returns:
        Full path to the data directory (e.g., 'data/live' or 'data/paper')
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mode = get_mode_string()
    path = os.path.join(base, 'data', mode)

    if subdir:
        path = os.path.join(path, subdir)

    # Ensure directory exists
    os.makedirs(path, exist_ok=True)
    return path


def get_db_path(db_name: str) -> str:
    """
    Get the mode-specific database path.

    Args:
        db_name: Name of the database file (e.g., 'trades.db')

    Returns:
        Full path to the database (e.g., 'data/live/trades.db')
    """
    return os.path.join(get_data_path(), db_name)


def get_cache_path(cache_name: str) -> str:
    """
    Get path for cache directories (shared between modes).

    Args:
        cache_name: Name of the cache directory

    Returns:
        Path to cache directory
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, 'data', 'cache', cache_name)
    os.makedirs(path, exist_ok=True)
    return path


# Print current mode when imported
if __name__ == "__main__":
    print(f"Current Mode: {get_mode_string().upper()}")
    print(f"Data Path: {get_data_path()}")
    print(f"Example DB: {get_db_path('trading.db')}")
