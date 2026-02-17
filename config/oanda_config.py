"""
OANDA v20 API Configuration

Forex trading bot configuration.
Documentation: https://developer.oanda.com/rest-live-v20/

Account ID format:
  - Practice accounts start with "001-" (use api-fxpractice.oanda.com)
  - Live accounts start with "101-" (use api-fxtrade.oanda.com)
"""

import os
import logging

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _detect_environment(account_id: str, paper_mode_env: str) -> bool:
    """
    Determine if we should use the practice environment.

    Safety logic:
    1. If account ID starts with "001-", it's a practice account -> force practice mode.
    2. If account ID starts with "101-", it's a live account -> respect OANDA_PAPER_MODE.
    3. If account ID is missing or unrecognized, default to practice for safety.

    Returns True if practice (paper) mode should be used.
    """
    if not account_id:
        logger.warning("OANDA_ACCOUNT_ID is empty -- defaulting to practice mode for safety")
        return True

    if account_id.startswith("001-"):
        if paper_mode_env == "false":
            logger.warning(
                f"Account ID '{account_id[:8]}...' is a PRACTICE account (001-prefix), "
                f"but OANDA_PAPER_MODE=false. Overriding to practice mode to avoid 400 errors."
            )
        return True  # Practice account must use practice endpoint

    if account_id.startswith("101-"):
        return paper_mode_env != "false"  # Respect explicit setting for live accounts

    # Unknown prefix -- default to practice for safety
    logger.warning(
        f"Unrecognized OANDA account ID prefix '{account_id[:4]}'. "
        f"Expected '001-' (practice) or '101-' (live). Defaulting to practice mode."
    )
    return True


_account_id = os.getenv("OANDA_ACCOUNT_ID", "")
_paper_mode_env = os.getenv("OANDA_PAPER_MODE", "true").lower()
_paper_mode = _detect_environment(_account_id, _paper_mode_env)

OANDA_CONFIG = {
    "practice_url": "https://api-fxpractice.oanda.com",
    "live_url": "https://api-fxtrade.oanda.com",
    "api_key": os.getenv("OANDA_API_KEY", ""),
    "account_id": _account_id,
    "paper_mode": _paper_mode,

    # Trading parameters
    "allocation": 150,  # $150 forex allocation
    "max_risk_per_trade": 0.02,  # 2% risk per trade
    "max_concurrent_trades": 2,

    # Strategy parameters
    "timeframe": "H4",  # 4-hour candles
    "fast_ma_period": 10,
    "slow_ma_period": 20,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "atr_period": 14,
    "atr_stop_multiplier": 1.5,
    "atr_profit_multiplier": 3.0,  # 2:1 reward:risk

    # Risk management
    "daily_loss_limit": 0.03,  # 3% daily max loss
    "max_drawdown": 0.15,  # 15% max drawdown
}

# Major forex pairs to trade (best liquidity, tightest spreads)
FOREX_PAIRS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "AUD_USD",
]

# Trading sessions (UTC)
TRADING_SESSIONS = {
    "london": {"start": 8, "end": 16},
    "new_york": {"start": 13, "end": 21},
    "overlap": {"start": 13, "end": 16},  # Best liquidity
}
