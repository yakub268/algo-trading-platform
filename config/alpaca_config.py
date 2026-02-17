"""
Alpaca API Configuration

Direct Alpaca integration for FOMC drift strategy.
Documentation: https://docs.alpaca.markets/
"""

import os

ALPACA_CONFIG = {
    # API endpoints
    "paper_url": "https://paper-api.alpaca.markets",
    "live_url": "https://api.alpaca.markets",
    "data_url": "https://data.alpaca.markets",

    # Credentials from environment
    "api_key": os.getenv("ALPACA_API_KEY", ""),
    "secret_key": os.getenv("ALPACA_SECRET_KEY", os.getenv("ALPACA_API_SECRET", "")),
    "paper_mode": os.getenv("ALPACA_PAPER_MODE", "true").lower() == "true",

    # Trading parameters
    "allocation": 200,  # $200 Alpaca allocation
    "max_risk_per_trade": 0.02,  # 2% risk per trade

    # FOMC Drift Strategy parameters
    "fomc_drift": {
        "symbol": "SPY",
        "position_size_usd": 400,  # Max position size
        "use_fractional": True,    # Enable fractional shares for expensive stocks
        "take_profit_pct": 0.005,  # +0.5%
        "stop_loss_pct": 0.015,    # -1.5%
        "entry_hours_before": 24,  # Enter 24 hours before FOMC
        "exit_minutes_before": 30, # Exit 30 minutes before announcement
    },

    # Risk management
    "daily_loss_limit": 0.03,  # 3% daily max loss
    "max_drawdown": 0.15,      # 15% max drawdown
}

# 2026 FOMC Meeting Schedule (announcement times in ET)
# Format: {"date": "YYYY-MM-DD", "time": "HH:MM", "type": "scheduled/unscheduled"}
FOMC_SCHEDULE_2026 = [
    {"date": "2026-01-28", "time": "14:00", "type": "scheduled"},
    {"date": "2026-01-29", "time": "14:00", "type": "scheduled"},
    {"date": "2026-03-17", "time": "14:00", "type": "scheduled"},
    {"date": "2026-03-18", "time": "14:00", "type": "scheduled"},
    {"date": "2026-05-05", "time": "14:00", "type": "scheduled"},
    {"date": "2026-05-06", "time": "14:00", "type": "scheduled"},
    {"date": "2026-06-16", "time": "14:00", "type": "scheduled"},
    {"date": "2026-06-17", "time": "14:00", "type": "scheduled"},
    {"date": "2026-07-28", "time": "14:00", "type": "scheduled"},
    {"date": "2026-07-29", "time": "14:00", "type": "scheduled"},
    {"date": "2026-09-15", "time": "14:00", "type": "scheduled"},
    {"date": "2026-09-16", "time": "14:00", "type": "scheduled"},
    {"date": "2026-11-03", "time": "14:00", "type": "scheduled"},
    {"date": "2026-11-04", "time": "14:00", "type": "scheduled"},
    {"date": "2026-12-15", "time": "14:00", "type": "scheduled"},
    {"date": "2026-12-16", "time": "14:00", "type": "scheduled"},
]
