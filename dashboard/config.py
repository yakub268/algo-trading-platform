"""
Trading Dashboard V4.2 - Configuration
Loads credentials from the root .env file
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths (use __file__ for portability)
DASHBOARD_DIR = Path(__file__).parent.resolve()
BASE_DIR = DASHBOARD_DIR.parent

# Load environment variables from root .env
load_dotenv(BASE_DIR / ".env")

# Alpaca Configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_PAPER_MODE = os.getenv('PAPER_MODE', 'true').lower() == 'true'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets' if ALPACA_PAPER_MODE else 'https://api.alpaca.markets'

# Kalshi Configuration
# NOTE: Kalshi consolidated APIs - all markets now on elections URL (Jan 2026)
KALSHI_API_KEY = os.getenv('KALSHI_API_KEY', '')
KALSHI_PRIVATE_KEY_PATH = Path(os.path.expanduser(os.getenv('KALSHI_PRIVATE_KEY_PATH', '~/.trading_keys/kalshi_private_key.pem')))
KALSHI_API_BASE = os.getenv('KALSHI_API_BASE', 'https://api.elections.kalshi.com/trade-api/v2')

# OANDA Configuration
OANDA_API_KEY = os.getenv('OANDA_API_KEY', '')
OANDA_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID', '')
OANDA_ENVIRONMENT = os.getenv('OANDA_ENVIRONMENT', 'practice')

# Use endpoint from env (token must match endpoint)
if OANDA_ENVIRONMENT == 'practice':
    OANDA_BASE_URL = 'https://api-fxpractice.oanda.com'
else:
    OANDA_BASE_URL = 'https://api-fxtrade.oanda.com'

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Database paths - ALWAYS use live DB for dashboard (orchestrator writes here)
TRADES_DB = BASE_DIR / "data" / "live" / "trading_master.db"
ALERTS_LOG = BASE_DIR / "logs" / "alerts.log"
STRATEGY_STATE_FILE = BASE_DIR / "data" / "strategy_states.json"

# Capital allocation
TOTAL_CAPITAL = float(os.getenv('TOTAL_CAPITAL', '450'))
ALPACA_ALLOCATION = float(os.getenv('ALPACA_ALLOCATION', '200'))
KALSHI_ALLOCATION = float(os.getenv('KALSHI_ALLOCATION', '150'))
OANDA_ALLOCATION = float(os.getenv('OANDA_ALLOCATION', '100'))

# Risk parameters
MAX_RISK_PER_TRADE = float(os.getenv('MAX_RISK_PER_TRADE', '0.02'))  # 2%
DAILY_LOSS_LIMIT = float(os.getenv('DAILY_LOSS_LIMIT', '0.05'))  # 5%
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '0.15'))  # 15%

# Dashboard settings
DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', '127.0.0.1')
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '5000'))
DASHBOARD_DEBUG = os.getenv('DASHBOARD_DEBUG', 'false').lower() == 'true'
POLL_INTERVAL_SECONDS = 5

# Scraper cache directories
WEATHER_CACHE = BASE_DIR / "data" / "weather_cache"
ECONOMIC_CACHE = BASE_DIR / "data" / "economic_cache"
SPORTS_CACHE = BASE_DIR / "data" / "sports_cache"
