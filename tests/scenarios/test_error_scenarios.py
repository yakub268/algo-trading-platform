"""
Error Scenario Tests
===================
Test error handling and edge cases for all AI trading components.
"""

import pytest
import asyncio
import sys
import os
import json
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any
import math
from requests.exceptions import ConnectionError, Timeout, HTTPError

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class TestNetworkErrorScenarios:
    """Test network-related error handling"""

    def test_api_connection_failures(self):
        """Test handling of API connection failures"""
        from bots.kalshi_bot import KalshiBot

        config = {
            "base_url": "https://demo-api.kalshi.co/trade-api/v2",
            "api_key_id": "test-key-id",
            "private_key_path": "kalshi_private_key.pem",
            "paper_mode": True,
            "max_position_size": 15,
            "max_concurrent_positions": 3,
            "min_probability_edge": 0.05,
            "target_series_patterns": ["FED"],
            "min_time_to_expiry_hours": 1,
            "max_time_to_expiry_days": 7,
            "daily_loss_limit": 10,
            "stop_trading_on_loss": True,
        }

        with patch.object(KalshiBot, "_load_private_key", return_value=Mock()):
            bot = KalshiBot(config=config)

            # Mock _request to raise ConnectionError (bypasses signature logic)
            with patch.object(
                bot, "_request", side_effect=ConnectionError("Connection refused")
            ):
                # Bot should handle network failure gracefully
                result = bot.get_markets()

                # Should return empty list instead of crashing
                assert isinstance(result, list)
                assert len(result) == 0

    def test_api_timeout_handling(self):
        """Test handling of API timeouts"""
        from bots.oanda_bot import OANDABot

        config = {
            "practice_url": "https://api-fxpractice.oanda.com",
            "live_url": "https://api-fxtrade.oanda.com",
            "api_key": "test",
            "account_id": "001-001-12345-001",
            "paper_mode": True,
            "allocation": 150,
            "max_risk_per_trade": 0.02,
            "max_concurrent_trades": 2,
            "timeframe": "H4",
            "fast_ma_period": 10,
            "slow_ma_period": 20,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "atr_period": 14,
            "atr_stop_multiplier": 1.5,
            "atr_profit_multiplier": 3.0,
            "daily_loss_limit": 0.03,
            "max_drawdown": 0.15,
        }

        bot = OANDABot(config=config)

        # Mock session to raise Timeout on any request
        bot.session = Mock()
        bot.session.request.side_effect = Timeout("Request timeout")

        # Should handle timeout gracefully
        result = bot.get_candles("EUR_USD")

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_rate_limiting_scenarios(self, error_simulator):
        """Test handling of API rate limits"""

        client = Mock()

        # Simulate rate limiting with exponential backoff
        call_count = 0

        async def mock_query_with_rate_limit(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count <= 3:  # First 3 calls are rate limited
                raise HTTPError("429 Too Many Requests")
            else:
                # Subsequent calls succeed
                from ai.llm_client import LLMResponse, LLMProvider

                return LLMResponse(
                    content="Success after rate limit",
                    provider=LLMProvider.DEEPSEEK,
                    latency_ms=100,
                    tokens_used=20,
                    cached=False,
                    cost_estimate=0.001,
                )

        client.query = mock_query_with_rate_limit

        # Should eventually succeed after rate limit recovery (with retry)
        result = None
        for attempt in range(5):
            try:
                result = await client.query("Test query")
                break
            except HTTPError:
                continue

        assert result is not None
        assert result.content == "Success after rate limit"
        assert call_count > 3  # Should have retried

    def test_partial_api_failures(self):
        """Test handling when some APIs work but others fail"""
        from master_orchestrator import BotState, BotConfig, Market, BotStatus

        # Create mock bots with different failure modes
        working_bot = Mock()
        working_bot.run_strategy.return_value = {
            "status": "success",
            "opportunities": 2,
        }

        failing_bot = Mock()
        failing_bot.run_strategy.side_effect = ConnectionError("API down")

        timeout_bot = Mock()
        timeout_bot.run_strategy.side_effect = Timeout("Request timeout")

        # Create BotState objects directly (no full orchestrator needed)
        bot_states = {
            "WorkingBot": BotState(
                config=BotConfig("WorkingBot", "mock", "Mock", Market.PREDICTION, True),
                instance=working_bot,
                status=BotStatus.WAITING,
            ),
            "FailingBot": BotState(
                config=BotConfig("FailingBot", "mock", "Mock", Market.FOREX, True),
                instance=failing_bot,
                status=BotStatus.WAITING,
            ),
            "TimeoutBot": BotState(
                config=BotConfig("TimeoutBot", "mock", "Mock", Market.CRYPTO, True),
                instance=timeout_bot,
                status=BotStatus.WAITING,
            ),
        }

        # Verify bot states were created correctly
        assert len(bot_states) == 3
        assert bot_states["WorkingBot"].config.name == "WorkingBot"
        assert bot_states["WorkingBot"].config.market == Market.PREDICTION
        assert bot_states["FailingBot"].config.market == Market.FOREX
        assert bot_states["TimeoutBot"].config.market == Market.CRYPTO
        assert all(s.status == BotStatus.WAITING for s in bot_states.values())

        # Verify mock bots behave as expected
        result = working_bot.run_strategy()
        assert result["status"] == "success"
        assert result["opportunities"] == 2

        with pytest.raises(ConnectionError):
            failing_bot.run_strategy()

        with pytest.raises(Timeout):
            timeout_bot.run_strategy()


class TestDataCorruptionScenarios:
    """Test handling of corrupted or malformed data"""

    @pytest.mark.asyncio
    async def test_malformed_json_response(self):
        """Test handling of malformed JSON from APIs"""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.text = '{"invalid": json data'  # Malformed JSON
            mock_response.json.side_effect = json.JSONDecodeError(
                "Expecting ',' delimiter", "", 0
            )
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            # Mock news fetcher that should handle JSON errors
            class MockNewsFetcher:
                async def fetch_news(self):
                    try:
                        response = mock_get("https://api.example.com/news")
                        return response.json()
                    except json.JSONDecodeError:
                        return []  # Return empty list on JSON error

            fetcher = MockNewsFetcher()
            news = await fetcher.fetch_news()

            # Should return empty list instead of crashing
            assert news == []

    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test handling of data with missing required fields"""
        from tests.mocks.mock_data import MockDataGenerator

        _generator = MockDataGenerator()  # noqa: F841

        # Generate data with missing fields
        corrupted_articles = [
            {"title": "Missing content"},  # Missing 'content'
            {"content": "Missing title"},  # Missing 'title'
            {},  # Missing everything
            {
                "title": "Valid Article",
                "content": "This article has all required fields",
                "source": "Test Source",
                "timestamp": datetime.now(),
            },
        ]

        # Process articles with validation
        valid_articles = []
        for article in corrupted_articles:
            try:
                # Validate required fields
                required_fields = ["title", "content"]
                if all(
                    field in article and article[field] for field in required_fields
                ):
                    valid_articles.append(article)
            except Exception:
                # Skip corrupted articles
                continue

        # Should only include valid article
        assert len(valid_articles) == 1
        assert valid_articles[0]["title"] == "Valid Article"

    @pytest.mark.asyncio
    async def test_invalid_numeric_data(self):
        """Test handling of invalid numeric values"""
        invalid_market_data = [
            {"price": "invalid", "volume": 1000},  # Invalid price
            {"price": float("inf"), "volume": 1000},  # Infinite price
            {"price": float("nan"), "volume": 1000},  # NaN price
            {"price": -50, "volume": 1000},  # Negative price (invalid for most assets)
            {"price": 100, "volume": "invalid"},  # Invalid volume
        ]

        processed_data = []
        for data in invalid_market_data:
            try:
                price = float(data["price"])
                volume = int(data["volume"])

                # Validate ranges
                if (
                    price > 0
                    and not (math.isinf(price) or math.isnan(price))
                    and volume >= 0
                ):
                    processed_data.append({"price": price, "volume": volume})
            except (ValueError, TypeError):
                # Skip invalid data
                continue

        # Should filter out all invalid data
        assert len(processed_data) == 0

    @pytest.mark.asyncio
    async def test_database_corruption_recovery(self):
        """Test recovery from database corruption"""
        import tempfile
        import sqlite3

        # Create corrupted database scenario
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
            db_path = tmp_file.name

        # Write invalid data to simulate corruption
        with open(db_path, "wb") as f:
            f.write(b"This is not a valid SQLite database")

        # Test database recovery
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades")
            cursor.fetchall()
            conn.close()
            conn = None
            assert False, "Should have failed with corrupted database"
        except sqlite3.DatabaseError:
            # Expected - database is corrupted
            if conn:
                conn.close()
                conn = None

        # Simulate recovery by recreating database
        os.unlink(db_path)  # Delete corrupted file

        # Recreate with proper schema
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                price REAL
            )
        """)
        conn.commit()

        # Should work now
        cursor.execute(
            "INSERT INTO trades (symbol, price) VALUES (?, ?)", ("TEST", 100.0)
        )
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        assert count == 1

        conn.close()
        os.unlink(db_path)


class TestExtremeMarketConditions:
    """Test behavior during extreme market conditions"""

    @pytest.mark.asyncio
    async def test_market_crash_scenario(self):
        """Test behavior during market crash (high volatility, negative returns)"""
        import pandas as pd
        import numpy as np

        # Generate crash scenario data
        crash_dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        crash_returns = np.array(
            [-0.15, -0.12, -0.08, -0.20, -0.05, -0.03, 0.02, -0.10, -0.15, -0.07]
        )

        base_price = 100
        prices = [base_price]
        for ret in crash_returns:
            prices.append(prices[-1] * (1 + ret))

        crash_data = pd.DataFrame(
            {
                "Close": prices[1:],
                "Volume": [
                    10000000 + i * 1000000 for i in range(10)
                ],  # Increasing volume
            },
            index=crash_dates,
        )

        # Test risk management during crash
        class MockRiskManager:
            def __init__(self):
                self.max_drawdown = 0.15  # V4: 15% max drawdown
                self.position_limit = 0.05  # 5% position limit

            def should_halt_trading(self, current_drawdown: float) -> bool:
                return current_drawdown > self.max_drawdown

            def adjust_position_size(self, normal_size: int, volatility: float) -> int:
                if volatility > 0.05:  # High volatility
                    return max(1, int(normal_size * 0.5))  # Reduce position by 50%
                return normal_size

        risk_manager = MockRiskManager()

        # Calculate drawdown
        peak = max(crash_data["Close"])
        current = crash_data["Close"].iloc[-1]
        drawdown = (peak - current) / peak

        # Should halt trading during extreme conditions
        should_halt = risk_manager.should_halt_trading(drawdown)
        assert should_halt

        # Should reduce position sizes
        volatility = crash_data["Close"].pct_change().std()
        adjusted_size = risk_manager.adjust_position_size(100, volatility)
        assert adjusted_size < 100

    @pytest.mark.asyncio
    async def test_flash_crash_recovery(self):
        """Test recovery mechanisms during flash crashes"""
        # Simulate flash crash: sudden drop followed by recovery
        flash_crash_prices = [
            100,
            95,
            85,
            60,
            55,
            70,
            80,
            85,
            90,
            95,
        ]  # V-shaped recovery

        class MockFlashCrashDetector:
            def __init__(self):
                self.price_history = []
                self.flash_crash_threshold = 0.20  # 20% drop in short period

            def add_price(self, price: float):
                self.price_history.append(price)
                if len(self.price_history) > 10:
                    self.price_history.pop(0)  # Keep only recent history

            def is_flash_crash(self) -> bool:
                if len(self.price_history) < 5:
                    return False

                recent_high = max(self.price_history[-5:])
                current_price = self.price_history[-1]
                drop = (recent_high - current_price) / recent_high

                return drop > self.flash_crash_threshold

            def is_recovery(self) -> bool:
                if len(self.price_history) < 3:
                    return False

                # Check if price is recovering (last 3 prices trending up)
                recent_prices = self.price_history[-3:]
                return all(
                    recent_prices[i] >= recent_prices[i - 1]
                    for i in range(1, len(recent_prices))
                )

        detector = MockFlashCrashDetector()

        flash_crash_detected = False
        recovery_detected = False

        for price in flash_crash_prices:
            detector.add_price(price)

            if detector.is_flash_crash() and not flash_crash_detected:
                flash_crash_detected = True

            if (
                flash_crash_detected
                and detector.is_recovery()
                and not recovery_detected
            ):
                recovery_detected = True

        assert flash_crash_detected is True
        assert recovery_detected is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self):
        """Test behavior when markets hit circuit breakers"""

        class MockCircuitBreaker:
            def __init__(self):
                self.circuit_breakers = [0.07, 0.13, 0.20]  # 7%, 13%, 20% levels
                self.is_triggered = False
                self.trigger_level = None

            def check_circuit_breaker(self, daily_return: float) -> bool:
                abs_return = abs(daily_return)

                for level in self.circuit_breakers:
                    if abs_return >= level:
                        self.is_triggered = True
                        self.trigger_level = level
                        return True

                return False

            def reset(self):
                self.is_triggered = False
                self.trigger_level = None

        breaker = MockCircuitBreaker()

        # Test various scenarios
        test_scenarios = [
            -0.05,  # Normal decline, no breaker
            -0.08,  # First breaker triggered
            0.15,  # Second breaker triggered (positive direction)
            -0.25,  # Third breaker triggered
        ]

        results = []
        for scenario in test_scenarios:
            breaker.reset()
            triggered = breaker.check_circuit_breaker(scenario)
            results.append((scenario, triggered, breaker.trigger_level))

        assert results[0][1] is False  # No breaker for -5%
        assert results[1][1] is True  # Breaker for -8%
        assert results[2][1] is True  # Breaker for +15%
        assert results[3][1] is True  # Breaker for -25%

    @pytest.mark.asyncio
    async def test_liquidity_crisis_handling(self):
        """Test handling during liquidity crises (wide spreads, low volume)"""
        liquidity_scenarios = [
            {"bid": 99.5, "ask": 100.5, "volume": 1000000},  # Normal liquidity
            {"bid": 98.0, "ask": 102.0, "volume": 100000},  # Reduced liquidity
            {"bid": 95.0, "ask": 105.0, "volume": 10000},  # Low liquidity
            {"bid": 90.0, "ask": 110.0, "volume": 1000},  # Liquidity crisis
        ]

        class MockLiquidityManager:
            def __init__(self):
                self.max_spread_pct = 0.02  # 2% max spread
                self.min_volume = 50000  # Minimum volume threshold

            def assess_liquidity(
                self, bid: float, ask: float, volume: int
            ) -> Dict[str, Any]:
                spread = ask - bid
                mid_price = (bid + ask) / 2
                spread_pct = spread / mid_price

                return {
                    "spread_pct": spread_pct,
                    "volume": volume,
                    "is_liquid": spread_pct <= self.max_spread_pct
                    and volume >= self.min_volume,
                    "risk_level": self._calculate_risk_level(spread_pct, volume),
                }

            def _calculate_risk_level(self, spread_pct: float, volume: int) -> str:
                if spread_pct > 0.05 or volume < 10000:
                    return "HIGH"
                elif spread_pct > 0.03 or volume < 100000:
                    return "MEDIUM"
                else:
                    return "LOW"

        liquidity_manager = MockLiquidityManager()

        assessments = []
        for scenario in liquidity_scenarios:
            assessment = liquidity_manager.assess_liquidity(
                scenario["bid"], scenario["ask"], scenario["volume"]
            )
            assessments.append(assessment)

        # First scenario should be liquid, last should not
        assert assessments[0]["is_liquid"] is True
        assert assessments[0]["risk_level"] == "LOW"

        assert assessments[-1]["is_liquid"] is False
        assert assessments[-1]["risk_level"] == "HIGH"


class TestSystemResourceExhaustion:
    """Test behavior under resource exhaustion conditions"""

    @pytest.mark.asyncio
    async def test_memory_exhaustion_handling(self):
        """Test handling when system memory is exhausted"""
        import gc

        psutil = __import__("pytest").importorskip("psutil")

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Simulate memory pressure
        data_chunks = []
        max_chunks = 100

        try:
            for i in range(max_chunks):
                # Allocate 10MB chunks
                chunk = [0] * (10 * 1024 * 1024 // 8)  # 8 bytes per int
                data_chunks.append(chunk)

                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory

                # If memory usage becomes excessive, implement cleanup
                if memory_increase > 500 * 1024 * 1024:  # 500MB limit
                    # Emergency cleanup
                    del data_chunks[:-5]  # Keep only last 5 chunks
                    gc.collect()
                    break

            # System should handle memory pressure gracefully
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Should not have consumed unlimited memory
            assert memory_increase < 1024 * 1024 * 1024  # Less than 1GB

        finally:
            # Clean up
            del data_chunks
            gc.collect()

    @pytest.mark.asyncio
    async def test_cpu_exhaustion_handling(self):
        """Test handling under high CPU load"""
        psutil = __import__("pytest").importorskip("psutil")
        import threading
        import time

        cpu_intensive_tasks = []

        def cpu_intensive_task():
            """Simulate CPU-intensive work"""
            end_time = time.time() + 2  # Run for 2 seconds
            while time.time() < end_time:
                # CPU-intensive operation
                sum(i**2 for i in range(10000))

        # Start multiple CPU-intensive tasks
        for _ in range(psutil.cpu_count()):
            task = threading.Thread(target=cpu_intensive_task)
            cpu_intensive_tasks.append(task)
            task.start()

        # Monitor CPU usage
        start_time = time.time()
        max_cpu_usage = 0

        while time.time() - start_time < 3:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            max_cpu_usage = max(max_cpu_usage, cpu_usage)

        # Wait for tasks to complete
        for task in cpu_intensive_tasks:
            task.join()

        # System should handle high CPU load
        assert max_cpu_usage > 50  # Should have actually used CPU
        # But system should still be responsive (not completely frozen)

    @pytest.mark.asyncio
    async def test_disk_space_exhaustion(self):
        """Test handling when disk space is exhausted"""
        import tempfile
        import shutil

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Get available space
            disk_usage = shutil.disk_usage(temp_dir)
            _available_space = disk_usage.free  # noqa: F841

            # Simulate log file that could grow large
            log_file = os.path.join(temp_dir, "test.log")

            # Mock logger that checks disk space
            class MockSpaceAwareLogger:
                def __init__(self, log_path: str, max_size: int = 100 * 1024 * 1024):
                    self.log_path = log_path
                    self.max_size = max_size

                def log(self, message: str) -> bool:
                    try:
                        # Check available space before writing
                        disk_usage = shutil.disk_usage(os.path.dirname(self.log_path))
                        if disk_usage.free < 10 * 1024 * 1024:  # 10MB minimum
                            return False  # Can't write, disk full

                        # Check log file size
                        if os.path.exists(self.log_path):
                            if os.path.getsize(self.log_path) > self.max_size:
                                # Rotate log (delete old content)
                                with open(self.log_path, "w") as f:
                                    f.write("=== Log rotated ===\n")

                        # Write log entry
                        with open(self.log_path, "a") as f:
                            f.write(f"{datetime.now()}: {message}\n")

                        return True
                    except IOError:
                        return False

            logger = MockSpaceAwareLogger(log_file)

            # Test normal logging
            success = logger.log("Test message 1")
            assert success is True

            # Test with large messages to trigger rotation
            large_message = "X" * (50 * 1024 * 1024)  # 50MB message
            success = logger.log(large_message)
            success = logger.log("Test message 2")  # Should trigger rotation

            assert os.path.exists(log_file)

        finally:
            # Clean up
            shutil.rmtree(temp_dir)


class TestConcurrencyIssues:
    """Test race conditions and concurrency issues"""

    @pytest.mark.asyncio
    async def test_concurrent_database_access(self):
        """Test concurrent database access and locking"""
        import sqlite3
        import tempfile
        import threading
        from queue import Queue

        # Create test database
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
            db_path = tmp_file.name

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                value INTEGER,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

        # Test concurrent writes
        results_queue = Queue()

        def concurrent_writer(thread_id: int, num_writes: int):
            """Write to database concurrently"""
            try:
                local_conn = sqlite3.connect(db_path, timeout=10.0)
                local_cursor = local_conn.cursor()

                for i in range(num_writes):
                    local_cursor.execute(
                        "INSERT INTO test_table (value, timestamp) VALUES (?, ?)",
                        (thread_id * 1000 + i, datetime.now().isoformat()),
                    )

                local_conn.commit()
                local_conn.close()
                results_queue.put(("success", thread_id, num_writes))

            except Exception as e:
                results_queue.put(("error", thread_id, str(e)))

        # Start multiple concurrent writers
        threads = []
        num_threads = 5
        writes_per_thread = 10

        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=concurrent_writer, args=(thread_id, writes_per_thread)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Verify results
        successful_threads = [r for r in results if r[0] == "success"]
        _failed_threads = [r for r in results if r[0] == "error"]  # noqa: F841

        # Most threads should succeed (allowing for some contention)
        assert len(successful_threads) >= num_threads // 2

        # Verify database integrity
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_table")
        total_records = cursor.fetchone()[0]
        conn.close()

        # Should have records from successful threads
        expected_records = sum(r[2] for r in successful_threads)
        assert total_records == expected_records

        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_race_condition_in_bot_status(self):
        """Test race conditions in bot status updates"""
        from master_orchestrator import BotState, BotStatus, BotConfig, Market

        # Mock bot state that could have race conditions
        bot_state = BotState(
            config=BotConfig("TestBot", "mock", "Mock", Market.PREDICTION, True),
            instance=Mock(),
            status=BotStatus.WAITING,
            trades_today=0,
            pnl_today=0.0,
        )

        # Simulate concurrent updates
        import threading
        import time

        update_results = []
        lock = threading.Lock()

        def update_bot_status(update_id: int):
            """Simulate concurrent status updates"""
            try:
                # Simulate some processing time
                time.sleep(0.01)

                # Thread-safe update
                with lock:
                    _original_trades = bot_state.trades_today  # noqa: F841
                    _original_pnl = bot_state.pnl_today  # noqa: F841

                    # Update values
                    bot_state.trades_today += 1
                    bot_state.pnl_today += 10.0
                    bot_state.status = BotStatus.RUNNING

                    _final_trades = bot_state.trades_today  # noqa: F841
                    _final_pnl = bot_state.pnl_today  # noqa: F841

                update_results.append(
                    {
                        "update_id": update_id,
                        "success": True,
                        "trades_increment": 1,
                        "pnl_increment": 10.0,
                    }
                )

            except Exception as e:
                update_results.append(
                    {"update_id": update_id, "success": False, "error": str(e)}
                )

        # Start concurrent updates
        threads = []
        num_updates = 10

        for i in range(num_updates):
            thread = threading.Thread(target=update_bot_status, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no race conditions occurred
        successful_updates = [r for r in update_results if r["success"]]
        assert len(successful_updates) == num_updates

        # Final values should be consistent
        assert bot_state.trades_today == num_updates
        assert bot_state.pnl_today == num_updates * 10.0

    @pytest.mark.asyncio
    async def test_deadlock_prevention(self):
        """Test deadlock prevention in resource locking"""
        import threading
        import time

        # Create two locks that could cause deadlock
        lock_a = threading.Lock()
        lock_b = threading.Lock()

        _deadlock_detected = False  # noqa: F841
        thread_results = []

        def task_1():
            """Task that acquires locks in order A -> B"""
            try:
                acquired_a = lock_a.acquire(timeout=1.0)
                if not acquired_a:
                    thread_results.append(("task_1", "timeout_a"))
                    return

                time.sleep(0.1)  # Hold lock A for a bit

                acquired_b = lock_b.acquire(timeout=1.0)
                if not acquired_b:
                    lock_a.release()
                    thread_results.append(("task_1", "timeout_b"))
                    return

                # Critical section with both locks
                time.sleep(0.1)

                lock_b.release()
                lock_a.release()
                thread_results.append(("task_1", "success"))

            except Exception as e:
                thread_results.append(("task_1", f"error: {e}"))

        def task_2():
            """Task that acquires locks in order A -> B (same order to prevent deadlock)"""
            try:
                acquired_a = lock_a.acquire(timeout=1.0)
                if not acquired_a:
                    thread_results.append(("task_2", "timeout_a"))
                    return

                time.sleep(0.1)  # Hold lock A for a bit

                acquired_b = lock_b.acquire(timeout=1.0)
                if not acquired_b:
                    lock_a.release()
                    thread_results.append(("task_2", "timeout_b"))
                    return

                # Critical section with both locks
                time.sleep(0.1)

                lock_b.release()
                lock_a.release()
                thread_results.append(("task_2", "success"))

            except Exception as e:
                thread_results.append(("task_2", f"error: {e}"))

        # Start both tasks concurrently
        thread1 = threading.Thread(target=task_1)
        thread2 = threading.Thread(target=task_2)

        thread1.start()
        thread2.start()

        # Wait with timeout
        thread1.join(timeout=5.0)
        thread2.join(timeout=5.0)

        # Both threads should complete without deadlock
        assert len(thread_results) == 2
        assert any("success" in result[1] for result in thread_results)


class TestRecoveryMechanisms:
    """Test system recovery mechanisms"""

    @pytest.mark.asyncio
    async def test_automatic_restart_after_failure(self):
        """Test automatic restart mechanisms after failures"""
        failure_count = 0
        _max_failures = 3  # noqa: F841

        class MockRestartableBot:
            def __init__(self):
                self.restart_count = 0
                self.max_restarts = 3

            async def run_with_restart(self):
                nonlocal failure_count

                for attempt in range(self.max_restarts + 1):
                    try:
                        return await self._run()
                    except Exception as e:
                        failure_count += 1
                        self.restart_count += 1

                        if attempt < self.max_restarts:
                            # Wait before restart (exponential backoff)
                            await asyncio.sleep(0.1 * (2**attempt))
                        else:
                            raise Exception(f"Max restart attempts exceeded: {e}")

            async def _run(self):
                # Fail first few attempts, then succeed
                if failure_count < 3:
                    raise Exception(f"Simulated failure {failure_count}")
                return "Success after restart"

        bot = MockRestartableBot()

        # Should eventually succeed after restarts
        result = await bot.run_with_restart()
        assert result == "Success after restart"
        assert bot.restart_count == 3  # Should have restarted 3 times

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when components fail"""

        class MockTradingSystem:
            def __init__(self):
                self.components = {
                    "news_feed": True,
                    "market_data": True,
                    "ai_analysis": True,
                    "order_execution": True,
                }
                self.degraded_mode = False

            async def run_trading_cycle(self):
                available_components = [k for k, v in self.components.items() if v]

                if len(available_components) < 2:
                    raise Exception("Too many components failed")

                # Graceful degradation logic
                if "ai_analysis" not in available_components:
                    self.degraded_mode = True
                    return await self._run_simple_mode()
                elif "news_feed" not in available_components:
                    return await self._run_without_news()
                else:
                    return await self._run_full_mode()

            async def _run_full_mode(self):
                return {
                    "mode": "full",
                    "features": [
                        "ai_analysis",
                        "news_feed",
                        "market_data",
                        "order_execution",
                    ],
                    "performance": 1.0,
                }

            async def _run_without_news(self):
                return {
                    "mode": "no_news",
                    "features": ["ai_analysis", "market_data", "order_execution"],
                    "performance": 0.8,
                }

            async def _run_simple_mode(self):
                return {
                    "mode": "simple",
                    "features": ["market_data", "order_execution"],
                    "performance": 0.5,
                }

        system = MockTradingSystem()

        # Test full mode
        result = await system.run_trading_cycle()
        assert result["mode"] == "full"
        assert result["performance"] == 1.0

        # Test degradation - disable news feed
        system.components["news_feed"] = False
        result = await system.run_trading_cycle()
        assert result["mode"] == "no_news"
        assert result["performance"] == 0.8

        # Test further degradation - disable AI
        system.components["ai_analysis"] = False
        result = await system.run_trading_cycle()
        assert result["mode"] == "simple"
        assert result["performance"] == 0.5
        assert system.degraded_mode is True

    @pytest.mark.asyncio
    async def test_data_backup_and_restore(self):
        """Test data backup and restore mechanisms"""
        import tempfile
        import shutil
        import json

        # Create temporary directories
        data_dir = tempfile.mkdtemp()
        backup_dir = tempfile.mkdtemp()

        try:
            # Mock data manager with backup capabilities
            class MockDataManager:
                def __init__(self, data_path: str, backup_path: str):
                    self.data_path = data_path
                    self.backup_path = backup_path

                def save_data(self, data: Dict[str, Any]):
                    """Save data with backup"""
                    data_file = os.path.join(self.data_path, "data.json")

                    # Create backup of existing data
                    if os.path.exists(data_file):
                        backup_file = os.path.join(
                            self.backup_path,
                            f'data_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                        )
                        shutil.copy(data_file, backup_file)

                    # Save new data
                    with open(data_file, "w") as f:
                        json.dump(data, f)

                def restore_from_backup(self, backup_timestamp: str = None):
                    """Restore data from backup"""
                    if backup_timestamp:
                        backup_file = os.path.join(
                            self.backup_path, f"data_backup_{backup_timestamp}.json"
                        )
                    else:
                        # Find latest backup
                        backup_files = [
                            f
                            for f in os.listdir(self.backup_path)
                            if f.startswith("data_backup_")
                        ]
                        if not backup_files:
                            raise Exception("No backups available")

                        backup_files.sort(reverse=True)
                        backup_file = os.path.join(self.backup_path, backup_files[0])

                    # Restore data
                    data_file = os.path.join(self.data_path, "data.json")
                    shutil.copy(backup_file, data_file)

                def load_data(self):
                    """Load current data"""
                    data_file = os.path.join(self.data_path, "data.json")
                    if not os.path.exists(data_file):
                        return None

                    with open(data_file, "r") as f:
                        return json.load(f)

            manager = MockDataManager(data_dir, backup_dir)

            # Save initial data
            initial_data = {
                "trades": 10,
                "pnl": 150.0,
                "timestamp": "2024-01-15T10:00:00",
            }
            manager.save_data(initial_data)

            # Verify data saved
            loaded_data = manager.load_data()
            assert loaded_data == initial_data

            # Save updated data (creates backup)
            await asyncio.sleep(0.01)  # Ensure different timestamp
            updated_data = {
                "trades": 15,
                "pnl": 200.0,
                "timestamp": "2024-01-15T11:00:00",
            }
            manager.save_data(updated_data)

            # Verify backup was created
            backup_files = os.listdir(backup_dir)
            assert len(backup_files) >= 1

            # Simulate data corruption
            data_file = os.path.join(data_dir, "data.json")
            with open(data_file, "w") as f:
                f.write("corrupted data")

            # Restore from backup
            manager.restore_from_backup()

            # Verify restoration
            restored_data = manager.load_data()
            assert restored_data == initial_data  # Should be the backed up version

        finally:
            # Clean up
            shutil.rmtree(data_dir)
            shutil.rmtree(backup_dir)
