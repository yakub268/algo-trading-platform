"""
Kalshi API Configuration

Prediction markets trading bot configuration.
Documentation: https://trading-api.readme.io/
"""

import os

KALSHI_CONFIG = {
    "base_url": "https://api.elections.kalshi.com/trade-api/v2",
    "api_key_id": os.getenv("KALSHI_API_KEY", os.getenv("KALSHI_API_KEY_ID", "")),
    "private_key_path": os.path.expanduser(os.getenv("KALSHI_PRIVATE_KEY_PATH", "~/.trading_keys/kalshi_private_key.pem")),
    "paper_mode": os.getenv("KALSHI_PAPER_MODE", "true").lower() == "true",

    # Trading parameters
    "max_position_size": 15,      # $15 max per trade (10% of $150 allocation)
    "max_concurrent_positions": 3,
    "min_probability_edge": 0.05,  # 5% minimum edge vs implied odds (lowered from 10%)

    # Target event_ticker patterns (Kalshi uses event_ticker for market categorization)
    "target_series_patterns": [
        # Sports markets (current primary markets on Kalshi)
        "KXMVE",        # Multi-value extended sports markets (parlays)
        "KXNBA",        # NBA player props and games
        "KXNFL",        # NFL player props and games
        "KXMLB",        # MLB player props and games
        "KXNHL",        # NHL player props and games
        "KXUFC",        # UFC fight markets
        "SPORTS",       # General sports markets
        # Economic/Fed markets
        "FED",          # Fed rate decisions (FOMC)
        "FOMC",         # Federal Open Market Committee
        "KXCPI",        # Consumer Price Index
        "KXGDP",        # GDP data markets
        "KXJOBS",       # Jobs report markets
        "KXUNEMP",      # Unemployment markets
        "INXD",         # Economic index data
        # Weather markets
        "KXHIGH",       # Temperature markets (high temps)
        "KXLOW",        # Temperature markets (low temps)
        "KXPRECIP",     # Precipitation markets
        "TEMP",         # General temp markets
        # Crypto markets
        "KXBTC",        # Bitcoin price markets
        "KXETH",        # Ethereum price markets
        # Politics/Events
        "KXPRES",       # Presidential
        "KXCONGRESS",   # Congress
        "KXSCOTUS",     # Supreme Court
    ],

    # Time constraints
    "min_time_to_expiry_hours": 1,   # Don't trade contracts expiring too soon
    "max_time_to_expiry_days": 7,     # Don't lock capital too long

    # Risk management
    "daily_loss_limit": 10,          # $10 max daily loss
    "stop_trading_on_loss": True,
}

# Rate limits
RATE_LIMITS = {
    "requests_per_second": 10,
    "burst_limit": 20,
}
