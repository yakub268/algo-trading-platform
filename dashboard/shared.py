"""
Shared state and utilities for dashboard blueprints.

All shared variables, clients, and helper functions live here to avoid
circular imports between app.py and blueprint modules.
"""

import os
import re
import json
import time
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Paper mode flags
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

_paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
_alpaca_paper = os.getenv('ALPACA_PAPER_MODE', 'true').lower() == 'true'

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
try:
    from utils.data_paths import get_db_path as _get_db_path
    DB_PATH = _get_db_path('trading_master.db')
except ImportError:
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'live', 'trading_master.db')


def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Category colours
# ---------------------------------------------------------------------------
CATEGORY_COLORS = {
    'weather': '#3498db',    # Blue
    'fed': '#e74c3c',        # Red
    'crypto': '#f39c12',     # Orange
    'earnings': '#9b59b6',   # Purple
    'economic': '#1abc9c',   # Teal
    'sports': '#2ecc71',     # Green
    'boxoffice': '#e91e63',  # Pink
}

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.alpaca_client import AlpacaClient

alpaca_client = AlpacaClient(paper=_alpaca_paper)

# KalshiClient (optional)
try:
    from bots.kalshi_client import KalshiClient
except ImportError:
    print("Warning: KalshiClient not available for AI bots")
    KalshiClient = None

# Coinbase (optional)
try:
    from dashboard.services.broker_coinbase import get_coinbase_service
    COINBASE_AVAILABLE = True
except ImportError:
    print("Warning: Coinbase service not available")
    COINBASE_AVAILABLE = False
    get_coinbase_service = None

# News Aggregator (optional)
try:
    from news_feeds.aggregator import NewsAggregator
    NEWS_AGGREGATOR_AVAILABLE = True
    _news_aggregator = None  # Lazy load
except ImportError:
    print("Warning: News aggregator not available")
    NEWS_AGGREGATOR_AVAILABLE = False
    _news_aggregator = None


def get_news_aggregator():
    """Get or create NewsAggregator singleton"""
    global _news_aggregator
    if _news_aggregator is None and NEWS_AGGREGATOR_AVAILABLE:
        try:
            _news_aggregator = NewsAggregator()
        except Exception as e:
            print(f"Failed to initialize NewsAggregator: {e}")
    return _news_aggregator


# Risk Management (optional)
try:
    from risk_management.integration.dashboard_integration import create_risk_routes, RiskDashboardAPI
    from risk_management.integration.trading_integration import get_trading_integration
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    print("Warning: Risk management not available")
    RISK_MANAGEMENT_AVAILABLE = False
    create_risk_routes = None
    RiskDashboardAPI = None
    get_trading_integration = None

# ---------------------------------------------------------------------------
# Market scan cache
# ---------------------------------------------------------------------------
_market_scan_cache = {'data': None, 'timestamp': 0, 'lock': threading.Lock()}
SCAN_CACHE_TTL = 60  # seconds


def get_cached_market_scan():
    """Cache market scan results for 60 seconds"""
    from bots.multi_market_strategy import MultiMarketStrategy
    now = time.time()
    with _market_scan_cache['lock']:
        if _market_scan_cache['data'] is not None and (now - _market_scan_cache['timestamp']) < SCAN_CACHE_TTL:
            return _market_scan_cache['data']
    # Cache miss - do the scan outside the lock
    strategy = MultiMarketStrategy(paper_mode=_paper_mode)
    results = strategy.scan_all_markets()
    with _market_scan_cache['lock']:
        _market_scan_cache['data'] = results
        _market_scan_cache['timestamp'] = now
    return results


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
_rate_limits = defaultdict(lambda: {'count': 0, 'reset_time': 0})
RATE_LIMIT_WINDOW = 10  # seconds
RATE_LIMIT_MAX = 5  # max requests per window for expensive endpoints


def rate_limit_check(endpoint_key):
    """Returns True if rate limited, False if OK"""
    now = time.time()
    rl = _rate_limits[endpoint_key]
    if now > rl['reset_time']:
        rl['count'] = 0
        rl['reset_time'] = now + RATE_LIMIT_WINDOW
    rl['count'] += 1
    return rl['count'] > RATE_LIMIT_MAX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_ticker_date(ticker):
    """Extract date from ticker like KXHIGHNY-26JAN27-T23 -> 2026-01-27"""
    date_match = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker.upper())
    if not date_match:
        return None
    year = '20' + date_match.group(1)
    month_abbr = date_match.group(2)
    day = date_match.group(3)
    months = {
        'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
        'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
        'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
    }
    month = months.get(month_abbr)
    if not month:
        return None
    return f'{year}-{month}-{day}'


def _format_time_ago(timestamp):
    """Format a timestamp as relative time"""
    now = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    delta = now - timestamp
    seconds = int(delta.total_seconds())

    if seconds < 60:
        return f"{seconds}s ago"
    elif seconds < 3600:
        return f"{seconds // 60}m ago"
    elif seconds < 86400:
        return f"{seconds // 3600}h ago"
    else:
        return f"{seconds // 86400}d ago"


# ---------------------------------------------------------------------------
# High-risk config helpers
# ---------------------------------------------------------------------------
HIGH_RISK_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'high_risk_config.json')


def load_high_risk_config():
    """Load high risk trading configuration"""
    try:
        if os.path.exists(HIGH_RISK_CONFIG_PATH):
            with open(HIGH_RISK_CONFIG_PATH, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading high risk config: {e}")

    # Return default config if file doesn't exist or error
    return {
        "high_risk_enabled": False,
        "aggressive_bots": [
            "MomentumScalper",
            "BreakoutHunter",
            "MemeCoinSniper",
            "RSIExtremes",
            "MultiMomentum"
        ],
        "last_modified": None,
        "last_modified_by": "system"
    }


def save_high_risk_config(config):
    """Save high risk trading configuration"""
    try:
        # Ensure config directory exists
        os.makedirs(os.path.dirname(HIGH_RISK_CONFIG_PATH), exist_ok=True)
        with open(HIGH_RISK_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving high risk config: {e}")
        return False
