"""
Test Configuration and Fixtures
===============================
Central pytest configuration for the trading bot test suite.
"""

import os
import sys
import pytest
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Standalone scripts — not pytest tests (require live services or manual execution)
_tests_dir = os.path.dirname(__file__)
collect_ignore = [
    os.path.join(_tests_dir, "test_ai_dashboard_integration.py"),
    os.path.join(_tests_dir, "test_arbitrage_with_real_matches.py"),
    os.path.join(_tests_dir, "test_cv_bot_structure.py"),
    os.path.join(_tests_dir, "test_simple_ai.py"),
    os.path.join(_tests_dir, "test_sports_ai.py"),
    os.path.join(_tests_dir, "test_real_kalshi.py"),
    # Standalone Kalshi scripts with module-level API calls / sys.exit()
    os.path.join(_tests_dir, "test_kalshi_debug.py"),
    os.path.join(_tests_dir, "test_kalshi_final.py"),
    os.path.join(_tests_dir, "test_kalshi_find_market.py"),
    os.path.join(_tests_dir, "test_kalshi_only.py"),
]

# Test configuration
TEST_DB_PATH = ":memory:"  # Use in-memory SQLite for tests
MOCK_STARTING_CAPITAL = 10000.0


@pytest.fixture
def mock_env_vars():
    """Mock environment variables"""
    env_vars = {
        "ALPACA_API_KEY": "test_alpaca_key",
        "ALPACA_SECRET_KEY": "test_alpaca_secret",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        "KALSHI_API_KEY_ID": "test_kalshi_key",
        "KALSHI_PRIVATE_KEY_PATH": "test_kalshi_key.pem",
        "OANDA_API_KEY": "test_oanda_key",
        "OANDA_ACCOUNT_ID": "test_oanda_account",
        "FRED_API_KEY": "test_fred_key",
        "TELEGRAM_BOT_TOKEN": "test_telegram_token",
        "TELEGRAM_CHAT_ID": "12345",
        "DEEPSEEK_API_KEY": "test_deepseek_key",
        "ANTHROPIC_API_KEY": "test_anthropic_key",
        "PAPER_MODE": "true",
        "TOTAL_CAPITAL": str(MOCK_STARTING_CAPITAL),
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_db():
    """Create temporary test database"""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create test tables
    cursor.execute("""
        CREATE TABLE trades (
            trade_id TEXT PRIMARY KEY,
            bot_name TEXT,
            market TEXT,
            symbol TEXT,
            side TEXT,
            entry_price REAL,
            exit_price REAL,
            quantity REAL,
            entry_time TEXT,
            exit_time TEXT,
            pnl REAL,
            pnl_pct REAL,
            status TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE bot_status (
            bot_name TEXT PRIMARY KEY,
            status TEXT,
            last_run TEXT,
            last_signal TEXT,
            trades_today INTEGER,
            pnl_today REAL,
            error TEXT
        )
    """)

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def mock_market_data():
    """Mock market data for testing"""
    import pandas as pd
    import numpy as np

    # Generate 100 days of mock OHLCV data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)  # For reproducible tests

    # Create realistic price data with trend and volatility
    base_price = 100
    prices = []
    for i in range(100):
        if i == 0:
            prices.append(base_price)
        else:
            # Add trend + random walk
            change = np.random.normal(0.001, 0.02)  # 0.1% avg return, 2% volatility
            prices.append(prices[-1] * (1 + change))

    high_prices = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    low_prices = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    volumes = [int(np.random.normal(1000000, 200000)) for _ in range(100)]

    return pd.DataFrame(
        {
            "Open": prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": prices,
            "Volume": volumes,
        },
        index=dates,
    )


@pytest.fixture
def mock_kalshi_markets():
    """Mock Kalshi markets data"""
    return [
        {
            "ticker": "FEDRAISE-24FEB28",
            "title": "Will the Fed raise rates by Feb 28?",
            "outcome_1": "YES",
            "outcome_2": "NO",
            "yes_price": 25,
            "no_price": 75,
            "volume": 1000,
            "open_interest": 5000,
            "close_time": "2024-02-28T20:00:00Z",
            "category": "Economics",
        },
        {
            "ticker": "INFLATION-24MAR",
            "title": "Will CPI be above 3% in March?",
            "outcome_1": "YES",
            "outcome_2": "NO",
            "yes_price": 45,
            "no_price": 55,
            "volume": 2000,
            "open_interest": 8000,
            "close_time": "2024-03-15T14:00:00Z",
            "category": "Economics",
        },
        {
            "ticker": "BTCPRICE-24Q1",
            "title": "Will Bitcoin close above $60k in Q1?",
            "outcome_1": "YES",
            "outcome_2": "NO",
            "yes_price": 35,
            "no_price": 65,
            "volume": 3000,
            "open_interest": 12000,
            "close_time": "2024-03-31T23:59:00Z",
            "category": "Crypto",
        },
    ]


@pytest.fixture
def mock_news_feed():
    """Mock news feed data"""
    return [
        {
            "title": "Fed Chair Powell Signals Rate Cut Possible",
            "content": (
                "Federal Reserve Chairman Jerome Powell indicated that interest"
                " rate cuts could be on the table if inflation continues to moderate."
            ),
            "source": "Reuters",
            "timestamp": datetime.now() - timedelta(hours=1),
            "sentiment": 0.3,  # Positive
            "keywords": ["fed", "rate cut", "inflation", "powell"],
            "relevance": 0.95,
        },
        {
            "title": "Bitcoin Surges on ETF Approval Hopes",
            "content": "Bitcoin jumped 5% after reports suggested the SEC might approve spot Bitcoin ETFs soon.",
            "source": "CoinDesk",
            "timestamp": datetime.now() - timedelta(minutes=30),
            "sentiment": 0.7,  # Very positive
            "keywords": ["bitcoin", "etf", "sec", "approval"],
            "relevance": 0.88,
        },
        {
            "title": "Unemployment Claims Rise Unexpectedly",
            "content": "Weekly unemployment claims rose to 250,000, higher than the expected 220,000.",
            "source": "Bloomberg",
            "timestamp": datetime.now() - timedelta(hours=3),
            "sentiment": -0.4,  # Negative
            "keywords": ["unemployment", "claims", "jobs", "economy"],
            "relevance": 0.75,
        },
    ]


@pytest.fixture
def mock_crypto_pairs():
    """Mock cryptocurrency trading pairs"""
    return {
        "BTC/USD": {
            "symbol": "BTCUSD",
            "price": 45000.0,
            "volume_24h": 1000000000,
            "change_24h": 0.025,
            "bid": 44995.0,
            "ask": 45005.0,
        },
        "ETH/USD": {
            "symbol": "ETHUSD",
            "price": 2800.0,
            "volume_24h": 500000000,
            "change_24h": 0.018,
            "bid": 2799.5,
            "ask": 2800.5,
        },
        "SOL/USD": {
            "symbol": "SOLUSD",
            "price": 95.0,
            "volume_24h": 100000000,
            "change_24h": -0.012,
            "bid": 94.98,
            "ask": 95.02,
        },
    }


@pytest.fixture
def mock_forex_pairs():
    """Mock forex trading pairs"""
    return {
        "EUR/USD": {
            "bid": 1.0850,
            "ask": 1.0852,
            "spread": 0.0002,
            "timestamp": datetime.now(),
        },
        "GBP/USD": {
            "bid": 1.2650,
            "ask": 1.2652,
            "spread": 0.0002,
            "timestamp": datetime.now(),
        },
        "USD/JPY": {
            "bid": 148.50,
            "ask": 148.52,
            "spread": 0.02,
            "timestamp": datetime.now(),
        },
    }


@pytest.fixture
def mock_alpaca_client():
    """Mock Alpaca trading client"""
    client = Mock()

    # Mock account info
    client.get_account.return_value = Mock(
        equity=MOCK_STARTING_CAPITAL,
        cash=MOCK_STARTING_CAPITAL,
        buying_power=MOCK_STARTING_CAPITAL * 2,
        portfolio_value=MOCK_STARTING_CAPITAL,
    )

    # Mock positions
    client.list_positions.return_value = []

    # Mock orders
    client.list_orders.return_value = []
    client.submit_order.return_value = Mock(
        id="test_order_123", status="accepted", symbol="SPY"
    )

    return client


@pytest.fixture
def mock_kalshi_client():
    """Mock Kalshi trading client"""
    client = Mock()

    client.get_markets.return_value = []
    client.get_market.return_value = {
        "ticker": "TEST-MARKET",
        "title": "Test Market",
        "yes_price": 50,
        "no_price": 50,
    }
    client.place_order.return_value = {
        "order_id": "test_order_123",
        "status": "resting",
    }

    return client


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for AI testing"""
    from ai.llm_client import LLMResponse, LLMProvider

    client = Mock()

    # Mock successful response
    mock_response = LLMResponse(
        content="This is a test AI response with 75% confidence for BUY signal.",
        provider=LLMProvider.DEEPSEEK,
        latency_ms=250.0,
        tokens_used=50,
        cached=False,
        cost_estimate=0.001,
    )

    client.query = AsyncMock(return_value=mock_response)
    client.get_session_stats.return_value = {
        "total_calls": 1,
        "cache_hits": 0,
        "total_cost_usd": 0.001,
        "avg_cost_per_call": 0.001,
    }

    return client


class AsyncMock(Mock):
    """Mock class for async methods"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


@pytest.fixture
def mock_telegram_bot():
    """Mock Telegram bot for alerts"""
    bot = Mock()
    bot.send_message = AsyncMock()
    return bot


# Test data generators
def generate_trade_signal(
    bot_name: str = "test_bot", action: str = "buy"
) -> Dict[str, Any]:
    """Generate a mock trading signal"""
    return {
        "action": action,
        "symbol": "SPY",
        "price": 450.0,
        "quantity": 10,
        "confidence": 0.75,
        "reasoning": f"Test signal from {bot_name}",
        "timestamp": datetime.now(),
        "stop_loss": 440.0 if action == "buy" else 460.0,
        "take_profit": 460.0 if action == "buy" else 440.0,
    }


def generate_market_opportunity(category: str = "Economics") -> Dict[str, Any]:
    """Generate a mock market opportunity"""
    return {
        "ticker": f"TEST-{category.upper()}-001",
        "title": f"Test {category} Market",
        "our_probability": 0.65,
        "market_price": 0.55,
        "edge": 0.10,
        "side": "YES",
        "confidence": 0.80,
        "volume": 5000,
        "liquidity_score": 0.85,
        "expiration": datetime.now() + timedelta(days=7),
    }


# Performance test utilities
@pytest.fixture
def performance_monitor():
    """Monitor for performance testing"""

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.metrics = {}

        def start(self):
            self.start_time = datetime.now()

        def stop(self):
            if self.start_time:
                duration = (datetime.now() - self.start_time).total_seconds()
                return duration
            return 0

        def record_metric(self, name: str, value: float):
            self.metrics[name] = value

        def get_metrics(self):
            return self.metrics.copy()

    return PerformanceMonitor()


# Error simulation utilities
@pytest.fixture
def error_simulator():
    """Utility to simulate various error conditions"""

    class ErrorSimulator:
        @staticmethod
        def network_error():
            from requests.exceptions import RequestException

            return RequestException("Mock network error")

        @staticmethod
        def api_rate_limit():
            from requests.exceptions import HTTPError

            error = HTTPError("Rate limit exceeded")
            error.response = Mock()
            error.response.status_code = 429
            return error

        @staticmethod
        def insufficient_funds():
            return ValueError("Insufficient funds for trade")

        @staticmethod
        def market_closed():
            return RuntimeError("Market is closed")

        @staticmethod
        def invalid_symbol():
            return ValueError("Invalid trading symbol")

    return ErrorSimulator()


# Cleanup utilities
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Clean up any test files created during tests"""
    yield
    # Cleanup happens after each test
    # Remove any temporary files if created
    import glob

    test_files = (
        glob.glob("test_*.db") + glob.glob("test_*.log") + glob.glob("temp_*.json")
    )
    for file in test_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass
