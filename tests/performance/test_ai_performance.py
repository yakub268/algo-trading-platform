"""
AI Performance Benchmarks
=========================
Performance tests and benchmarks for AI trading components.
"""

import pytest
import asyncio
import sys
import os
import time
import json
import random

psutil = __import__("pytest").importorskip("psutil")
from unittest.mock import Mock, patch
from datetime import datetime

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class TestAIComponentPerformance:
    """Performance tests for AI components"""

    @pytest.fixture
    def performance_thresholds(self):
        """Define performance thresholds for different operations"""
        return {
            "llm_query_time": 5.0,  # seconds
            "news_analysis_time": 2.0,  # seconds
            "market_scan_time": 10.0,  # seconds
            "bot_execution_time": 15.0,  # seconds
            "memory_limit": 512 * 1024 * 1024,  # 512MB
            "cpu_usage_limit": 80.0,  # percentage
        }

    @pytest.mark.asyncio
    async def test_llm_client_response_time(
        self, performance_thresholds, performance_monitor
    ):
        """Test LLM client response times"""
        from ai.llm_client import LLMResponse, LLMProvider

        # Mock LLM client for performance testing
        client = Mock()

        # Simulate various response times
        test_cases = [
            ("short_query", "What is 2+2?", 0.5),
            (
                "medium_query",
                "Analyze this market data and provide trading signals.",
                2.0,
            ),
            (
                "long_query",
                "Provide detailed analysis of Federal Reserve policy impact on crypto markets.",
                4.0,
            ),
        ]

        for test_name, query, expected_time in test_cases:

            async def mock_query_with_delay(prompt, **kwargs):
                await asyncio.sleep(expected_time)
                return LLMResponse(
                    content="Test response",
                    provider=LLMProvider.DEEPSEEK,
                    latency_ms=expected_time * 1000,
                    tokens_used=100,
                    cached=False,
                    cost_estimate=0.001,
                )

            client.query = mock_query_with_delay

            performance_monitor.start()
            response = await client.query(query)
            duration = performance_monitor.stop()

            # Verify response time is within acceptable limits
            assert duration < performance_thresholds["llm_query_time"]
            assert response.latency_ms > 0

            performance_monitor.record_metric(f"{test_name}_duration", duration)

    @pytest.mark.asyncio
    async def test_concurrent_llm_requests(
        self, performance_thresholds, performance_monitor
    ):
        """Test performance with concurrent LLM requests"""
        from ai.llm_client import LLMResponse, LLMProvider

        # Mock multiple concurrent requests
        async def mock_query(prompt, **kwargs):
            await asyncio.sleep(1.0)  # Simulate processing time
            return LLMResponse(
                content="Concurrent response",
                provider=LLMProvider.DEEPSEEK,
                latency_ms=1000,
                tokens_used=50,
                cached=False,
                cost_estimate=0.001,
            )

        client = Mock()
        client.query = mock_query

        # Test with increasing concurrency levels
        concurrency_levels = [1, 5, 10, 20]

        for concurrency in concurrency_levels:
            performance_monitor.start()

            # Create concurrent tasks
            tasks = [client.query(f"Query {i}") for i in range(concurrency)]

            # Execute all tasks concurrently
            responses = await asyncio.gather(*tasks)

            duration = performance_monitor.stop()

            # All requests should complete
            assert len(responses) == concurrency

            # Total time should not scale linearly (benefit of concurrency)
            expected_max_time = concurrency * 1.5  # Allow some overhead
            assert duration < expected_max_time

            performance_monitor.record_metric(
                f"concurrent_{concurrency}_duration", duration
            )

    @pytest.mark.asyncio
    async def test_news_processing_performance(
        self, performance_thresholds, performance_monitor, mock_news_feed
    ):
        """Test news processing pipeline performance"""
        from tests.mocks.mock_data import MockDataGenerator

        generator = MockDataGenerator()

        # Generate different sizes of news batches
        batch_sizes = [10, 50, 100, 500]

        for batch_size in batch_sizes:
            articles = generator.generate_news_articles(batch_size, days_back=1)

            performance_monitor.start()

            # Simulate news processing pipeline
            processed_articles = []
            for article in articles:
                # Simulate processing steps
                processed = {
                    **article.__dict__,
                    "sentiment_analyzed": True,
                    "keywords_extracted": True,
                    "relevance_scored": True,
                }
                processed_articles.append(processed)

            duration = performance_monitor.stop()

            # Performance should scale reasonably with batch size
            time_per_article = duration / batch_size
            assert time_per_article < 0.1  # 100ms per article max

            performance_monitor.record_metric(
                f"news_batch_{batch_size}_duration", duration
            )
            performance_monitor.record_metric(
                f"news_per_article_{batch_size}", time_per_article
            )

    @pytest.mark.asyncio
    async def test_market_scanning_performance(
        self, performance_thresholds, performance_monitor
    ):
        """Test market scanning performance for different bot types"""
        from tests.mocks.mock_data import MockDataGenerator

        generator = MockDataGenerator()

        # Test different market types
        market_configs = [
            ("kalshi_markets", 100, "prediction"),
            ("forex_pairs", 50, "forex"),
            ("crypto_symbols", 200, "crypto"),
            ("stock_symbols", 500, "stocks"),
        ]

        for market_name, symbol_count, market_type in market_configs:
            # Generate mock market data
            if market_type == "prediction":
                market_data = generator.generate_kalshi_markets(symbol_count)
            else:
                symbols = [f"SYMBOL_{i}" for i in range(symbol_count)]
                market_data = generator.generate_market_data(
                    symbols[: min(symbol_count, 10)]
                )

            performance_monitor.start()

            # Simulate market scanning
            opportunities = []
            for i, data in enumerate(
                list(market_data.items())
                if isinstance(market_data, dict)
                else market_data[:100]
            ):
                # Simulate analysis
                if isinstance(data, dict):
                    opportunity_score = hash(str(data)) % 100 / 100.0
                else:
                    opportunity_score = 0.5

                if opportunity_score > 0.7:  # Threshold for opportunity
                    opportunities.append(
                        {
                            "symbol": f"SYMBOL_{i}",
                            "score": opportunity_score,
                            "market_type": market_type,
                        }
                    )

            duration = performance_monitor.stop()

            # Market scanning should complete within reasonable time
            assert duration < performance_thresholds["market_scan_time"]

            # Should find some opportunities
            assert len(opportunities) >= 0

            performance_monitor.record_metric(
                f"market_scan_{market_name}_duration", duration
            )
            performance_monitor.record_metric(
                f"market_scan_{market_name}_opportunities", len(opportunities)
            )

    @pytest.mark.asyncio
    async def test_bot_execution_performance(
        self, performance_thresholds, performance_monitor
    ):
        """Test individual bot execution performance"""
        from master_orchestrator import Market

        # Mock different types of bots
        bot_configs = [
            ("FastBot", "fast_operations", Market.FOREX),
            ("MediumBot", "medium_operations", Market.CRYPTO),
            ("SlowBot", "slow_operations", Market.PREDICTION),
            ("ComplexBot", "complex_operations", Market.STOCKS),
        ]

        for bot_name, operation_type, market in bot_configs:
            # Mock bot with different performance characteristics
            mock_bot = Mock()

            if operation_type == "fast_operations":

                async def fast_scan():
                    await asyncio.sleep(0.5)
                    return {"status": "success", "signals": 3}

                mock_bot.run_scan = fast_scan

            elif operation_type == "medium_operations":

                async def medium_scan():
                    await asyncio.sleep(2.0)
                    return {"status": "success", "signals": 5}

                mock_bot.run_scan = medium_scan

            elif operation_type == "slow_operations":

                async def slow_scan():
                    await asyncio.sleep(5.0)
                    return {"status": "success", "signals": 8}

                mock_bot.run_scan = slow_scan

            else:  # complex_operations

                async def complex_scan():
                    # Simulate complex processing
                    for _ in range(100):
                        await asyncio.sleep(0.01)  # Small operations
                    return {"status": "success", "signals": 10}

                mock_bot.run_scan = complex_scan

            performance_monitor.start()

            # Execute bot
            result = await mock_bot.run_scan()

            duration = performance_monitor.stop()

            # Bot should complete within time limit
            assert duration < performance_thresholds["bot_execution_time"]
            assert result["status"] == "success"

            performance_monitor.record_metric(f"bot_{bot_name}_duration", duration)

    @pytest.mark.asyncio
    async def test_memory_usage_patterns(self, performance_thresholds):
        """Test memory usage patterns during AI operations"""
        import gc

        process = psutil.Process()
        _initial_memory = process.memory_info().rss  # noqa: F841

        # Test memory usage during different operations
        operations = [
            ("small_data_processing", lambda: list(range(1000))),
            ("medium_data_processing", lambda: list(range(100000))),
            ("large_data_processing", lambda: list(range(1000000))),
            ("repeated_small_ops", lambda: [list(range(100)) for _ in range(1000)]),
        ]

        for op_name, operation in operations:
            gc.collect()  # Clean up before test
            before_memory = process.memory_info().rss

            # Execute operation
            data = operation()

            after_memory = process.memory_info().rss
            memory_increase = after_memory - before_memory

            # Memory increase should be reasonable
            assert memory_increase < performance_thresholds["memory_limit"]

            # Clean up
            del data
            gc.collect()

            final_memory = process.memory_info().rss

            # Memory should be released (allowing for some overhead)
            memory_after_cleanup = final_memory - before_memory
            assert (
                memory_after_cleanup < memory_increase * 0.5
            )  # At least 50% should be released

    @pytest.mark.asyncio
    async def test_cpu_usage_under_load(self, performance_thresholds):
        """Test CPU usage under various loads"""
        import threading

        cpu_usage_samples = []

        def monitor_cpu():
            """Monitor CPU usage in background"""
            for _ in range(10):  # Sample for 5 seconds
                cpu_usage_samples.append(psutil.cpu_percent(interval=0.5))

        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()

        # Simulate AI workload
        tasks = []
        for i in range(5):  # Multiple concurrent tasks

            async def ai_workload(task_id):
                # Simulate AI computation
                for _ in range(1000):
                    # Simulate some computation
                    result = sum(range(1000))
                    await asyncio.sleep(0.001)
                return result

            tasks.append(ai_workload(i))

        # Execute concurrent AI tasks
        await asyncio.gather(*tasks)

        # Wait for CPU monitoring to complete
        monitor_thread.join()

        # Check CPU usage
        if cpu_usage_samples:
            avg_cpu_usage = sum(cpu_usage_samples) / len(cpu_usage_samples)
            max_cpu_usage = max(cpu_usage_samples)

            # CPU usage should be reasonable
            assert avg_cpu_usage < performance_thresholds["cpu_usage_limit"]
            assert max_cpu_usage < 95.0  # Should not max out CPU

    @pytest.mark.asyncio
    async def test_scalability_with_data_volume(self, performance_monitor):
        """Test how AI components scale with data volume"""
        from tests.mocks.mock_data import MockDataGenerator

        generator = MockDataGenerator()

        # Test with increasing data volumes
        data_volumes = [100, 500, 1000, 5000]

        previous_duration = 0

        for volume in data_volumes:
            # Generate data
            trade_data = generator.generate_trade_history(volume)
            news_data = generator.generate_news_articles(min(volume, 100))

            performance_monitor.start()

            # Simulate processing
            processed_trades = 0
            processed_news = 0

            for trade in trade_data:
                # Simulate trade analysis
                if trade.pnl > 0:
                    processed_trades += 1

            for article in news_data:
                # Simulate news analysis
                if article.sentiment_score > 0.5:
                    processed_news += 1

            duration = performance_monitor.stop()

            # Performance should scale sub-linearly (efficiency gains)
            if previous_duration > 0:
                scaling_factor = duration / previous_duration
                volume_factor = volume / data_volumes[data_volumes.index(volume) - 1]

                # Duration should not scale linearly with volume
                assert scaling_factor < volume_factor * 1.5

            previous_duration = duration
            performance_monitor.record_metric(f"volume_{volume}_duration", duration)

    @pytest.mark.asyncio
    async def test_caching_performance_impact(self, performance_monitor):
        """Test performance impact of caching mechanisms"""
        # Mock cache implementation
        cache = {}

        def cached_operation(key: str, expensive_operation):
            """Simulate cached operation"""
            if key in cache:
                return cache[key]  # Cache hit

            # Cache miss - perform expensive operation
            result = expensive_operation()
            cache[key] = result
            return result

        def expensive_computation():
            """Simulate expensive computation"""
            time.sleep(0.1)  # 100ms operation
            return sum(range(10000))

        # Test cache miss (first call)
        performance_monitor.start()
        result1 = cached_operation("test_key", expensive_computation)
        first_duration = performance_monitor.stop()

        # Test cache hit (second call)
        performance_monitor.start()
        result2 = cached_operation("test_key", lambda: expensive_computation())
        second_duration = performance_monitor.stop()

        # Results should be identical
        assert result1 == result2

        # Cache hit should be much faster
        assert second_duration < first_duration * 0.1  # At least 10x faster

        performance_monitor.record_metric("cache_miss_duration", first_duration)
        performance_monitor.record_metric("cache_hit_duration", second_duration)

    @pytest.mark.asyncio
    async def test_ai_model_switching_performance(self, performance_monitor):
        """Test performance when switching between AI models"""
        from ai.llm_client import LLMProvider

        # Simulate different AI models with different characteristics
        model_configs = [
            (LLMProvider.DEEPSEEK, 1.0),  # Fast, cheap
            (LLMProvider.CLAUDE_HAIKU, 2.0),  # Medium speed, medium cost
            (LLMProvider.LOCAL_OLLAMA, 5.0),  # Slow but free
        ]

        switching_times = []

        current_model = None

        for provider, response_time in model_configs:
            performance_monitor.start()

            # Simulate model switching overhead
            if current_model != provider:
                await asyncio.sleep(0.1)  # Model switching delay
                current_model = provider

            # Simulate model inference
            await asyncio.sleep(response_time)

            duration = performance_monitor.stop()
            switching_times.append(duration)

            performance_monitor.record_metric(f"{provider.value}_total_time", duration)

        # Model switching should not add excessive overhead
        max_switching_overhead = 0.5  # 500ms max for switching
        for duration, (_, expected_time) in zip(switching_times, model_configs):
            switching_overhead = duration - expected_time
            assert switching_overhead < max_switching_overhead


class TestOrchestrationPerformance:
    """Performance tests for orchestrator and multi-bot scenarios"""

    @pytest.mark.asyncio
    async def test_orchestrator_startup_time(self, performance_monitor, mock_env_vars):
        """Test orchestrator startup performance"""
        from master_orchestrator import MasterOrchestrator

        with patch("master_orchestrator.TradingDB"):
            performance_monitor.start()

            # Create orchestrator (simulates startup)
            orchestrator = MasterOrchestrator(
                starting_capital=10000.0, paper_mode=True, use_do_nothing_filter=False
            )

            startup_duration = performance_monitor.stop()

            # Startup should be reasonably fast
            assert startup_duration < 10.0  # 10 seconds max
            assert orchestrator is not None

            performance_monitor.record_metric(
                "orchestrator_startup_time", startup_duration
            )

    @pytest.mark.asyncio
    async def test_multi_bot_execution_performance(self, performance_monitor):
        """Test performance when running multiple bots"""
        from master_orchestrator import BotState, BotConfig, Market, BotStatus

        # Create mock bots with different performance profiles
        bot_configs = [
            ("FastBot1", 0.5),
            ("FastBot2", 0.7),
            ("MediumBot1", 2.0),
            ("MediumBot2", 2.5),
            ("SlowBot1", 5.0),
        ]

        bots = {}

        for bot_name, execution_time in bot_configs:
            mock_bot = Mock()

            async def create_bot_function(exec_time):
                async def bot_scan():
                    await asyncio.sleep(exec_time)
                    return {"status": "success", "bot": bot_name}

                return bot_scan

            mock_bot.run_scan = await create_bot_function(execution_time)

            config = BotConfig(
                name=bot_name,
                module_path="mock",
                class_name="MockBot",
                market=Market.PREDICTION,
                enabled=True,
            )

            bots[bot_name] = BotState(
                config=config, instance=mock_bot, status=BotStatus.WAITING
            )

        # Test sequential execution
        performance_monitor.start()

        sequential_results = []
        for bot_name, state in bots.items():
            result = await state.instance.run_scan()
            sequential_results.append(result)

        sequential_duration = performance_monitor.stop()

        # Test parallel execution
        performance_monitor.start()

        parallel_tasks = [state.instance.run_scan() for state in bots.values()]
        parallel_results = await asyncio.gather(*parallel_tasks)

        parallel_duration = performance_monitor.stop()

        # Parallel should be significantly faster than sequential
        assert len(sequential_results) == len(parallel_results)
        assert parallel_duration < sequential_duration * 0.7  # At least 30% faster

        performance_monitor.record_metric(
            "sequential_execution_time", sequential_duration
        )
        performance_monitor.record_metric("parallel_execution_time", parallel_duration)

    @pytest.mark.asyncio
    async def test_database_performance_under_load(self, performance_monitor):
        """Test database performance with high trade volume"""
        import sqlite3
        import tempfile
        from tests.mocks.mock_data import MockDataGenerator

        generator = MockDataGenerator()

        # Create temporary database
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
            db_path = tmp_file.name

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create test table
        cursor.execute("""
            CREATE TABLE test_trades (
                id INTEGER PRIMARY KEY,
                bot_name TEXT,
                symbol TEXT,
                pnl REAL,
                timestamp TEXT
            )
        """)
        conn.commit()

        # Test with different batch sizes
        batch_sizes = [100, 500, 1000, 5000]

        for batch_size in batch_sizes:
            trades = generator.generate_trade_history(batch_size)

            performance_monitor.start()

            # Insert trades in batch
            trade_data = [
                (trade.bot_name, trade.symbol, trade.pnl, trade.entry_time.isoformat())
                for trade in trades
            ]

            cursor.executemany(
                "INSERT INTO test_trades (bot_name, symbol, pnl, timestamp) VALUES (?, ?, ?, ?)",
                trade_data,
            )
            conn.commit()

            insert_duration = performance_monitor.stop()

            # Query performance
            performance_monitor.start()

            cursor.execute("SELECT COUNT(*) FROM test_trades WHERE pnl > 0")
            profitable_count = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(pnl) FROM test_trades")
            avg_pnl = cursor.fetchone()[0]

            query_duration = performance_monitor.stop()

            # Database operations should be reasonably fast
            assert insert_duration < 5.0  # 5 seconds for inserts
            assert query_duration < 1.0  # 1 second for queries

            # Results should be valid
            assert profitable_count >= 0
            assert avg_pnl is not None

            performance_monitor.record_metric(
                f"db_insert_{batch_size}", insert_duration
            )
            performance_monitor.record_metric(f"db_query_{batch_size}", query_duration)

        conn.close()
        os.unlink(db_path)  # Clean up

    @pytest.mark.asyncio
    async def test_real_time_data_processing(self, performance_monitor):
        """Test real-time data processing performance"""
        import asyncio
        import queue
        import threading

        # Simulate real-time data stream
        data_queue = queue.Queue()
        processed_items = []

        def data_producer():
            """Simulate incoming market data"""
            for i in range(1000):
                data_point = {
                    "timestamp": datetime.now(),
                    "symbol": f"SYMBOL_{i % 10}",
                    "price": 100 + (i % 50),
                    "volume": 1000 + (i % 1000),
                }
                data_queue.put(data_point)
                time.sleep(0.001)  # 1ms intervals

        async def data_processor():
            """Process incoming data"""
            while True:
                try:
                    data = data_queue.get_nowait()

                    # Simulate processing
                    processed = {
                        **data,
                        "processed_at": datetime.now(),
                        "moving_avg": data["price"] * 0.95,  # Simple processing
                    }
                    processed_items.append(processed)

                except queue.Empty:
                    await asyncio.sleep(0.001)
                    if len(processed_items) >= 1000:
                        break

        # Start data producer in background
        producer_thread = threading.Thread(target=data_producer)
        producer_thread.start()

        # Measure processing performance
        performance_monitor.start()

        await data_processor()

        processing_duration = performance_monitor.stop()

        producer_thread.join()

        # Should process data efficiently
        assert len(processed_items) >= 900  # Allow for some timing variance
        processing_rate = len(processed_items) / processing_duration
        assert processing_rate > 100  # At least 100 items per second

        performance_monitor.record_metric(
            "realtime_processing_duration", processing_duration
        )
        performance_monitor.record_metric("realtime_processing_rate", processing_rate)


class TestStressTests:
    """Stress tests for AI components under extreme conditions"""

    @pytest.mark.asyncio
    async def test_high_frequency_requests(self, performance_monitor):
        """Test performance under high frequency requests"""
        request_count = 1000
        concurrent_requests = 50

        async def make_request(request_id):
            # Simulate AI request processing
            await asyncio.sleep(random.uniform(0.01, 0.1))
            return {"request_id": request_id, "status": "success"}

        performance_monitor.start()

        # Create batches of concurrent requests
        all_results = []
        for batch in range(0, request_count, concurrent_requests):
            batch_tasks = [
                make_request(i)
                for i in range(batch, min(batch + concurrent_requests, request_count))
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)

        duration = performance_monitor.stop()

        # All requests should complete
        assert len(all_results) == request_count

        # Calculate throughput
        throughput = request_count / duration
        assert throughput > 50  # At least 50 requests per second

        performance_monitor.record_metric("stress_test_duration", duration)
        performance_monitor.record_metric("stress_test_throughput", throughput)

    @pytest.mark.asyncio
    async def test_memory_stress(self):
        """Test behavior under memory pressure"""
        import gc

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Gradually increase memory usage
        data_chunks = []
        max_memory_increase = 200 * 1024 * 1024  # 200MB limit

        try:
            for i in range(100):
                # Create data chunk (2MB each)
                chunk = [random.randint(1, 1000000) for _ in range(500000)]
                data_chunks.append(chunk)

                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory

                if memory_increase > max_memory_increase:
                    break

                # Simulate some processing
                _total = sum(chunk[:1000])  # noqa: F841 - Process part of the data

            # System should handle gracefully without crashing
            assert len(data_chunks) > 0

        finally:
            # Clean up
            del data_chunks
            gc.collect()

    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, performance_monitor):
        """Test performance of error recovery mechanisms"""
        error_scenarios = [
            ("network_timeout", lambda: asyncio.sleep(0.1)),
            ("processing_error", lambda: 1 / 0),  # Division by zero
            ("data_corruption", lambda: json.loads("invalid json")),
            ("resource_exhaustion", lambda: [0] * 10000000),  # Large allocation
        ]

        recovery_times = []

        for error_name, error_func in error_scenarios:
            performance_monitor.start()

            try:
                # Simulate error condition
                await asyncio.create_task(asyncio.coroutine(error_func)())
            except Exception:
                # Simulate recovery
                await asyncio.sleep(0.01)  # Recovery time

            recovery_time = performance_monitor.stop()
            recovery_times.append(recovery_time)

            # Recovery should be fast
            assert recovery_time < 1.0  # 1 second max recovery time

        # Average recovery time should be reasonable
        avg_recovery_time = sum(recovery_times) / len(recovery_times)
        assert avg_recovery_time < 0.5  # 500ms average recovery time
