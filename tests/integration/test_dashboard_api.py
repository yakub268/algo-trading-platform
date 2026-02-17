"""
Dashboard API Integration Tests
==============================
Test dashboard API endpoints, data flow, and frontend integration.
Tests real Flask routes from dashboard/app.py.
"""

import pytest
import asyncio
import sys
import os
import json
import time
import threading
import queue
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class TestDashboardAPIEndpoints:
    """Test all real dashboard API endpoints"""

    @pytest.fixture
    def client(self, mock_env_vars):
        """Create Flask test client with mocked broker clients"""
        # Patch module-level init functions that run at import time
        with patch("dashboard.app.init_paper_trades_table"), patch(
            "dashboard.app.init_scraper_status_table"
        ):
            import dashboard.app as dash_module
            import dashboard.shared

            dash_module.app.config["TESTING"] = True
            dash_module.DASHBOARD_API_KEY = ""  # Disable auth for tests
            # Clear market scan cache between tests
            dashboard.shared._market_scan_cache["data"] = None
            dashboard.shared._market_scan_cache["timestamp"] = 0

            with patch.object(dash_module, "freqtrade_client", create=True) as mock_ft, patch.object(
                dashboard.shared, "alpaca_client"
            ) as mock_alpaca:
                mock_ft.is_connected.return_value = True
                mock_ft.get_profit.return_value = {
                    "profit_closed_coin": 0.5,
                    "trade_count": 10,
                    "winning_trades": 7,
                }
                mock_alpaca.is_connected.return_value = True
                mock_alpaca.paper = True
                mock_alpaca.get_positions.return_value = [{"unrealized_pl": 50.0}]
                mock_alpaca.get_account.return_value = {"equity": 10000.0}
                mock_alpaca.get_fomc_status.return_value = None

                yield dash_module.app.test_client()

    def _make_mock_opportunity(self, **overrides):
        """Create a mock opportunity object with required attributes"""
        opp = Mock()
        opp.ticker = overrides.get("ticker", "KXTEST-27FEB28-T10")
        opp.title = overrides.get("title", "Test Opportunity")
        opp.category = overrides.get("category", "weather")
        opp.side = overrides.get("side", "YES")
        opp.our_probability = overrides.get("our_probability", 0.72)
        opp.market_price = overrides.get("market_price", 0.55)
        opp.edge = overrides.get("edge", 0.17)
        opp.data_source = overrides.get("data_source", "NWS")
        opp.overall_score = overrides.get("overall_score", 0.845)
        opp.reasoning = overrides.get("reasoning", "Strong edge detected")
        return opp

    def test_homepage_redirect(self, client):
        """GET / should redirect (302) to /static/dashboard_v5.html"""
        response = client.get("/")
        assert response.status_code == 302
        assert "/static/dashboard_v5.html" in response.headers.get("Location", "")

    def test_opportunities_endpoint(self, client):
        """GET /api/opportunities returns properly structured JSON with opportunity data"""
        mock_opps = [
            self._make_mock_opportunity(
                ticker="KXTEST-28FEB28-T10",
                title="Will it rain in NYC?",
                category="weather",
                side="YES",
                our_probability=0.72,
                market_price=0.55,
                edge=0.17,
                data_source="NWS",
                overall_score=0.845,
                reasoning="NWS forecast shows 80% chance of rain",
            ),
            self._make_mock_opportunity(
                ticker="KXFED-28FEB28-R1",
                title="Will the Fed raise rates?",
                category="fed",
                side="NO",
                our_probability=0.85,
                market_price=0.70,
                edge=0.15,
                data_source="FRED",
                overall_score=0.790,
                reasoning="Economic indicators suggest hold",
            ),
        ]

        with patch("dashboard.routes.legacy.get_cached_market_scan") as mock_scan:
            mock_scan.return_value = mock_opps

            response = client.get("/api/opportunities")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"] is True
            assert "opportunities" in data
            assert "timestamp" in data
            assert "count" in data
            assert data["count"] == len(data["opportunities"])

            # Check structure of each opportunity
            for opp in data["opportunities"]:
                assert "ticker" in opp
                assert "title" in opp
                assert "category" in opp
                assert "side" in opp
                assert "our_probability" in opp
                assert "market_price" in opp
                assert "edge" in opp
                assert "data_source" in opp
                assert "overall_score" in opp
                assert "reasoning" in opp
                assert "color" in opp

    def test_category_stats_endpoint(self, client):
        """GET /api/category-stats returns all 7 categories with counts and avg_edge"""
        all_categories = [
            "weather",
            "fed",
            "crypto",
            "earnings",
            "economic",
            "sports",
            "boxoffice",
        ]

        mock_opps = [
            self._make_mock_opportunity(
                ticker="KXTEST-28FEB28-T10", category="weather", edge=0.10
            ),
            self._make_mock_opportunity(
                ticker="KXTEST-28FEB28-T11", category="weather", edge=0.20
            ),
            self._make_mock_opportunity(
                ticker="KXFED-28FEB28-R1", category="fed", edge=0.15
            ),
            self._make_mock_opportunity(
                ticker="KXCRYPTO-28FEB28-C1", category="crypto", edge=0.12
            ),
        ]

        with patch("dashboard.routes.legacy.get_cached_market_scan") as mock_scan:
            mock_scan.return_value = mock_opps

            response = client.get("/api/category-stats")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"] is True
            assert "stats" in data
            assert "total" in data

            # All 7 categories must be present (even with 0 count)
            returned_categories = [s["category"] for s in data["stats"]]
            for cat in all_categories:
                assert cat in returned_categories, f"Missing category: {cat}"

            # Each stat has required fields
            for stat in data["stats"]:
                assert "category" in stat
                assert "count" in stat
                assert "avg_edge" in stat
                assert "color" in stat

            # Check that weather has count=2 from our mock data
            weather_stat = next(s for s in data["stats"] if s["category"] == "weather")
            assert weather_stat["count"] == 2

    def test_paper_trades_endpoint(self, client):
        """GET /api/paper-trades returns trades list and summary with P&L"""
        # Build mock cursor results for the two queries in get_paper_trades()
        mock_trade_row = MagicMock()
        mock_trade_row.__getitem__ = lambda self, key: {
            "id": 1,
            "timestamp": "2026-01-15T10:00:00Z",
            "ticker": "KXTEST-28FEB28-T10",
            "category": "weather",
            "side": "YES",
            "entry_price": 0.55,
            "contracts": 10,
            "edge": 0.17,
            "status": "closed",
            "exit_price": 0.72,
            "pnl": 1.70,
        }[key]

        mock_summary_row = MagicMock()
        mock_summary_row.__getitem__ = lambda self, key: {
            "realized_pnl": 1.70,
            "unrealized_pnl": 0.0,
            "total_trades": 1,
            "winning_trades": 1,
            "closed_trades": 1,
        }[key]

        mock_cursor = MagicMock()
        # First execute -> fetchall (trades), second execute -> fetchone (summary)
        mock_cursor.fetchall.return_value = [mock_trade_row]
        mock_cursor.fetchone.return_value = mock_summary_row

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("dashboard.routes.legacy.get_db_connection", return_value=mock_conn):
            response = client.get("/api/paper-trades")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"] is True
            assert "trades" in data
            assert "summary" in data
            assert len(data["trades"]) == 1

            # Summary structure
            summary = data["summary"]
            assert "realized_pnl" in summary
            assert "unrealized_pnl" in summary
            assert "total_trades" in summary
            assert "win_rate" in summary

            # Trade structure
            trade = data["trades"][0]
            assert trade["ticker"] == "KXTEST-28FEB28-T10"
            assert trade["category"] == "weather"
            assert trade["status"] == "closed"

    def test_system_status_endpoint(self, client):
        """GET /api/system-status returns systems dict with kalshi/alpaca"""
        response = client.get("/api/system-status")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["success"] is True
        assert "systems" in data
        assert "timestamp" in data

        systems = data["systems"]
        assert "kalshi" in systems
        assert "alpaca" in systems

        # Kalshi always online in paper mode
        assert systems["kalshi"]["status"] == "online"
        assert systems["kalshi"]["mode"] == "paper"
        assert "color" in systems["kalshi"]

        # Alpaca online (mocked as connected)
        assert systems["alpaca"]["status"] == "online"
        assert systems["alpaca"]["mode"] == "paper"
        assert "color" in systems["alpaca"]

        # FOMC field present
        assert "fomc" in data

    def test_combined_summary_endpoint(self, client):
        """GET /api/combined/summary returns full multi-system summary"""
        response = client.get("/api/combined/summary")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["success"] is True

        # Top-level keys
        assert "kalshi" in data
        assert "alpaca" in data
        assert "combined" in data
        assert "timestamp" in data

        # Kalshi structure
        assert "pnl" in data["kalshi"]
        assert "trades" in data["kalshi"]
        assert "win_rate" in data["kalshi"]
        assert "status" in data["kalshi"]

        # Alpaca structure (connected=True from fixture)
        assert data["alpaca"]["connected"] is True
        assert "pnl" in data["alpaca"]
        assert "equity" in data["alpaca"]
        assert data["alpaca"]["paper_mode"] is True
        assert data["alpaca"]["equity"] == 10000.0
        assert data["alpaca"]["pnl"] == 50.0  # from mock position unrealized_pl

        # Combined structure
        assert "total_pnl" in data["combined"]
        assert "total_trades" in data["combined"]
        assert "best_system" in data["combined"]

    def test_scraper_status_endpoint(self, client):
        """GET /api/scraper-status returns status for all 6 scrapers"""
        mock_summary = {
            "timestamp": "2026-01-15T10:00:00Z",
            "total_estimates": 42,
            "weather": {"status": "online", "estimates": 10},
            "economic": {"status": "online", "estimates": 8},
            "crypto": {"status": "online", "estimates": 7},
            "earnings": {"status": "online", "estimates": 6},
            "sports": {"status": "online", "estimates": 5},
            "boxoffice": {"status": "online", "estimates": 6},
        }

        with patch("scrapers.data_aggregator.DataAggregator") as MockAgg:
            MockAgg.return_value.get_summary.return_value = mock_summary

            response = client.get("/api/scraper-status")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"] is True
            assert "scrapers" in data
            assert "total_estimates" in data
            assert data["total_estimates"] == 42

            # Exactly 6 scrapers
            assert len(data["scrapers"]) == 6

            expected_names = [
                "Weather (NWS)",
                "Economic (FRED)",
                "Crypto (CoinGecko)",
                "Earnings (Yahoo)",
                "Sports (ESPN)",
                "Box Office (BOM)",
            ]
            actual_names = [s["name"] for s in data["scrapers"]]
            for name in expected_names:
                assert name in actual_names, f"Missing scraper: {name}"

            # Each scraper has required fields
            for scraper in data["scrapers"]:
                assert "name" in scraper
                assert "status" in scraper
                assert "records" in scraper
                assert "last_run" in scraper
                assert "color" in scraper

    def test_opportunities_error_handling(self, client):
        """GET /api/opportunities returns success=False when market scan raises"""
        with patch("dashboard.routes.legacy.get_cached_market_scan") as mock_scan:
            mock_scan.side_effect = RuntimeError("Kalshi API timeout")

            response = client.get("/api/opportunities")
            assert (
                response.status_code == 200
            )  # Flask still returns 200, but success=False

            data = json.loads(response.data)
            assert data["success"] is False
            assert "error" in data
            assert "Kalshi API timeout" in data["error"]
            assert data["opportunities"] == []

    def test_system_status_with_disconnected_services(self, mock_env_vars):
        """When brokers are disconnected, system-status reports offline"""
        with patch("dashboard.app.init_paper_trades_table"), patch(
            "dashboard.app.init_scraper_status_table"
        ):
            import dashboard.app as dash_module
            import dashboard.shared

            dash_module.app.config["TESTING"] = True
            dash_module.DASHBOARD_API_KEY = ""  # Disable auth for tests

            with patch.object(dash_module, "freqtrade_client", create=True) as mock_ft, patch.object(
                dashboard.shared, "alpaca_client"
            ) as mock_alpaca:
                mock_ft.is_connected.return_value = False
                mock_alpaca.is_connected.return_value = False
                mock_alpaca.paper = True
                mock_alpaca.get_fomc_status.return_value = None

                test_client = dash_module.app.test_client()
                response = test_client.get("/api/system-status")
                assert response.status_code == 200

                data = json.loads(response.data)
                assert data["success"] is True

                systems = data["systems"]
                # Kalshi always online (paper mode)
                assert systems["kalshi"]["status"] == "online"
                # Alpaca should be offline
                assert systems["alpaca"]["status"] == "offline"

    def test_paper_trades_empty(self, client):
        """GET /api/paper-trades with empty database returns empty trades and zeroed summary"""
        mock_summary_row = MagicMock()
        mock_summary_row.__getitem__ = lambda self, key: {
            "realized_pnl": None,
            "unrealized_pnl": None,
            "total_trades": 0,
            "winning_trades": 0,
            "closed_trades": 0,
        }[key]

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = mock_summary_row

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        with patch("dashboard.routes.legacy.get_db_connection", return_value=mock_conn):
            response = client.get("/api/paper-trades")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"] is True
            assert data["trades"] == []
            assert data["summary"]["realized_pnl"] == 0
            assert data["summary"]["unrealized_pnl"] == 0
            assert data["summary"]["total_trades"] == 0
            assert data["summary"]["win_rate"] == 0


class TestDashboardDataFlow:
    """Test data flow between components and dashboard"""

    @pytest.fixture
    def mock_data_manager(self):
        """Mock dashboard data manager"""

        class MockDataManager:
            def __init__(self):
                self.cache = {}
                self.cache_ttl = 30  # 30 seconds
                self.last_update = {}

            async def get_live_status(self):
                """Get live orchestrator status"""
                if self._is_cache_valid("status"):
                    return self.cache["status"]

                # Simulate fetching from orchestrator
                status = {
                    "timestamp": datetime.now().isoformat(),
                    "total_bots": 8,
                    "active_bots": 6,
                    "total_pnl": 350.0,
                }

                self._update_cache("status", status)
                return status

            async def get_live_trades(self, limit: int = 50):
                """Get recent trades"""
                if self._is_cache_valid("trades"):
                    return self.cache["trades"][:limit]

                # Simulate fetching from database
                trades = [
                    {"trade_id": f"trade_{i}", "pnl": 25.0 * (i % 3)}
                    for i in range(100)
                ]

                self._update_cache("trades", trades)
                return trades[:limit]

            async def get_bot_metrics(self, bot_name: str):
                """Get metrics for specific bot"""
                cache_key = f"bot_metrics_{bot_name}"
                if self._is_cache_valid(cache_key):
                    return self.cache[cache_key]

                # Simulate computing metrics
                metrics = {
                    "name": bot_name,
                    "total_trades": 25,
                    "win_rate": 0.72,
                    "avg_pnl": 15.5,
                    "best_trade": 125.0,
                    "worst_trade": -35.0,
                }

                self._update_cache(cache_key, metrics)
                return metrics

            def _is_cache_valid(self, key: str):
                """Check if cache entry is still valid"""
                if key not in self.cache or key not in self.last_update:
                    return False

                age = (datetime.now() - self.last_update[key]).total_seconds()
                return age < self.cache_ttl

            def _update_cache(self, key: str, data: Any):
                """Update cache with new data"""
                self.cache[key] = data
                self.last_update[key] = datetime.now()

        return MockDataManager()

    @pytest.mark.asyncio
    async def test_live_data_updates(self, mock_data_manager):
        """Test live data updates from orchestrator"""
        # First call should fetch fresh data
        status1 = await mock_data_manager.get_live_status()
        assert status1["total_bots"] == 8

        # Second call within cache TTL should return cached data
        status2 = await mock_data_manager.get_live_status()
        assert status1["timestamp"] == status2["timestamp"]

    @pytest.mark.asyncio
    async def test_cache_expiration(self, mock_data_manager):
        """Test cache expiration and refresh"""
        # Set very short cache TTL for testing
        mock_data_manager.cache_ttl = 0.1

        # Get initial data
        status1 = await mock_data_manager.get_live_status()

        # Wait for cache to expire
        await asyncio.sleep(0.2)

        # Should fetch fresh data
        status2 = await mock_data_manager.get_live_status()
        # Timestamps should be different (fresh fetch)
        assert status1["timestamp"] != status2["timestamp"]

    @pytest.mark.asyncio
    async def test_concurrent_data_access(self, mock_data_manager):
        """Test concurrent access to data"""
        # Make multiple concurrent requests
        tasks = [
            mock_data_manager.get_live_status(),
            mock_data_manager.get_live_trades(20),
            mock_data_manager.get_bot_metrics("Kalshi-Fed"),
            mock_data_manager.get_bot_metrics("OANDA-Forex"),
        ]

        results = await asyncio.gather(*tasks)

        # All requests should complete successfully
        assert len(results) == 4
        assert results[0]["total_bots"] == 8  # Status
        assert len(results[1]) == 20  # Trades
        assert results[2]["name"] == "Kalshi-Fed"  # Bot 1 metrics
        assert results[3]["name"] == "OANDA-Forex"  # Bot 2 metrics

    @pytest.mark.asyncio
    async def test_data_aggregation(self, mock_data_manager):
        """Test data aggregation for dashboard"""
        # Get data for multiple bots
        bot_names = ["Kalshi-Fed", "Sports-AI", "OANDA-Forex"]
        metrics = []

        for bot_name in bot_names:
            bot_metrics = await mock_data_manager.get_bot_metrics(bot_name)
            metrics.append(bot_metrics)

        # Aggregate win rates
        total_trades = sum(m["total_trades"] for m in metrics)
        avg_win_rate = (
            sum(m["win_rate"] * m["total_trades"] for m in metrics) / total_trades
        )

        assert total_trades == 75  # 25 * 3 bots
        assert 0 <= avg_win_rate <= 1


class TestDashboardPerformance:
    """Test dashboard performance and responsiveness"""

    @pytest.fixture
    def client(self, mock_env_vars):
        """Create Flask test client with mocked broker clients"""
        with patch("dashboard.app.init_paper_trades_table"), patch(
            "dashboard.app.init_scraper_status_table"
        ):
            import dashboard.app as dash_module
            import dashboard.shared

            dash_module.app.config["TESTING"] = True
            dash_module.DASHBOARD_API_KEY = ""  # Disable auth for tests
            # Clear market scan cache between tests
            dashboard.shared._market_scan_cache["data"] = None
            dashboard.shared._market_scan_cache["timestamp"] = 0

            with patch.object(dash_module, "freqtrade_client", create=True) as mock_ft, patch.object(
                dashboard.shared, "alpaca_client"
            ) as mock_alpaca:
                mock_ft.is_connected.return_value = True
                mock_ft.get_profit.return_value = {
                    "profit_closed_coin": 0.5,
                    "trade_count": 10,
                    "winning_trades": 7,
                }
                mock_alpaca.is_connected.return_value = True
                mock_alpaca.paper = True
                mock_alpaca.get_positions.return_value = [{"unrealized_pl": 50.0}]
                mock_alpaca.get_account.return_value = {"equity": 10000.0}
                mock_alpaca.get_fomc_status.return_value = None

                yield dash_module.app.test_client()

    def test_api_response_times(self, client):
        """Multiple real endpoints respond within 2 seconds each"""
        endpoints = [
            "/api/system-status",
            "/api/combined/summary",
        ]

        for endpoint in endpoints:
            start = time.time()
            response = client.get(endpoint)
            duration = time.time() - start

            assert (
                response.status_code == 200
            ), f"{endpoint} returned {response.status_code}"
            assert duration < 2.0, f"{endpoint} took {duration:.2f}s (> 2s limit)"

    def test_concurrent_data_access(self, client):
        """Threaded requests to multiple endpoints do not crash"""
        results = queue.Queue()
        errors = queue.Queue()

        def make_request(endpoint):
            try:
                response = client.get(endpoint)
                results.put((endpoint, response.status_code))
            except Exception as exc:
                errors.put((endpoint, str(exc)))

        endpoints = [
            "/api/system-status",
            "/api/combined/summary",
            "/api/system-status",
            "/api/combined/summary",
        ]

        threads = []
        for ep in endpoints:
            t = threading.Thread(target=make_request, args=(ep,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5.0)

        # No threads should have errored
        assert (
            errors.empty()
        ), f"Errors occurred: {[errors.get() for _ in range(errors.qsize())]}"

        # All threads should have completed
        status_codes = []
        while not results.empty():
            ep, code = results.get()
            status_codes.append(code)

        assert len(status_codes) == len(endpoints)
        assert all(code == 200 for code in status_codes)


class TestDashboardSecurity:
    """Test dashboard security basics"""

    @pytest.fixture
    def client(self, mock_env_vars):
        """Create Flask test client with mocked broker clients"""
        with patch("dashboard.app.init_paper_trades_table"), patch(
            "dashboard.app.init_scraper_status_table"
        ):
            import dashboard.app as dash_module
            import dashboard.shared

            dash_module.app.config["TESTING"] = True
            dash_module.DASHBOARD_API_KEY = ""  # Disable auth for tests
            # Clear market scan cache between tests
            dashboard.shared._market_scan_cache["data"] = None
            dashboard.shared._market_scan_cache["timestamp"] = 0

            with patch.object(dash_module, "freqtrade_client", create=True) as mock_ft, patch.object(
                dashboard.shared, "alpaca_client"
            ) as mock_alpaca:
                mock_ft.is_connected.return_value = True
                mock_ft.get_profit.return_value = {
                    "profit_closed_coin": 0.5,
                    "trade_count": 10,
                    "winning_trades": 7,
                }
                mock_alpaca.is_connected.return_value = True
                mock_alpaca.paper = True
                mock_alpaca.get_positions.return_value = []
                mock_alpaca.get_account.return_value = {"equity": 10000.0}
                mock_alpaca.get_fomc_status.return_value = None

                yield dash_module.app.test_client()

    def test_nonexistent_endpoint_returns_404(self, client):
        """GET /api/nonexistent should return 404"""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """POST to a GET-only endpoint should return 405"""
        response = client.post("/api/system-status")
        assert response.status_code == 405
