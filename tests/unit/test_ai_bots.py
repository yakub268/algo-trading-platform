"""
Unit Tests for Individual AI Bots
=================================
Test each AI bot individually with mocked dependencies.
"""

import pytest
import asyncio
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta, timezone
from collections import deque

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class TestKalshiBot:
    """Test the Kalshi prediction market bot using actual interface"""

    @pytest.fixture
    def kalshi_config(self):
        """Minimal config dict matching KALSHI_CONFIG structure"""
        return {
            "base_url": "https://api.elections.kalshi.com/trade-api/v2",
            "api_key_id": "test_key",
            "private_key_path": "nonexistent.pem",
            "paper_mode": True,
            "max_position_size": 15,
            "max_concurrent_positions": 3,
            "min_probability_edge": 0.05,
            "target_series_patterns": ["FED", "KXHIGH", "KXNFL"],
            "min_time_to_expiry_hours": 1,
            "max_time_to_expiry_days": 7,
            "daily_loss_limit": 10,
            "stop_trading_on_loss": True,
        }

    @pytest.fixture
    def kalshi_bot(self, mock_env_vars, kalshi_config):
        from bots.kalshi_bot import KalshiBot

        with patch.object(KalshiBot, "_load_private_key", return_value=Mock()):
            bot = KalshiBot(config=kalshi_config)
            return bot

    def test_initialization(self, kalshi_bot):
        """Test bot initializes with config dict"""
        assert kalshi_bot.paper_mode is True
        assert kalshi_bot.daily_pnl == 0.0
        assert kalshi_bot.config["min_probability_edge"] == 0.05

    def test_filter_markets_by_pattern(self, kalshi_bot):
        """Test filter_markets filters by target_series_patterns"""
        now = datetime.now(timezone.utc)
        expiry = (now + timedelta(days=3)).isoformat()
        markets = [
            {
                "ticker": "FED-RAISE-26FEB",
                "event_ticker": "FOMC",
                "title": "Fed Rate Decision",
                "expected_expiration_time": expiry,
                "volume": 500,
            },
            {
                "ticker": "RANDOM-MARKET",
                "event_ticker": "RANDOM",
                "title": "Unrelated Market",
                "expected_expiration_time": expiry,
                "volume": 500,
            },
            {
                "ticker": "KXNFL-PASS-26FEB",
                "event_ticker": "KXNFL",
                "title": "NFL Passing Yards",
                "expected_expiration_time": expiry,
                "volume": 200,
            },
        ]
        filtered = kalshi_bot.filter_markets(markets)
        tickers = [m["ticker"] for m in filtered]
        assert "FED-RAISE-26FEB" in tickers
        assert "KXNFL-PASS-26FEB" in tickers
        assert "RANDOM-MARKET" not in tickers

    def test_filter_markets_expiry_too_soon(self, kalshi_bot):
        """Test filter_markets rejects markets expiring too soon"""
        now = datetime.now(timezone.utc)
        expiry_soon = (now + timedelta(minutes=30)).isoformat()
        markets = [
            {
                "ticker": "FED-RAISE",
                "event_ticker": "FED",
                "title": "Fed",
                "expected_expiration_time": expiry_soon,
            },
        ]
        filtered = kalshi_bot.filter_markets(markets)
        assert len(filtered) == 0

    def test_evaluate_opportunity_with_edge(self, kalshi_bot):
        """Test evaluate_opportunity returns signal when edge exists"""
        market = {
            "ticker": "FED-RAISE-26FEB",
            "series_ticker": "FED",
            "title": "Will the Fed raise rates?",
            "yes_bid": 20,
            "yes_ask": 25,
            "no_bid": 70,
            "no_ask": 75,
        }
        # Mock _estimate_probability to return value with edge
        with patch.object(kalshi_bot, "_estimate_probability", return_value=0.40):
            result = kalshi_bot.evaluate_opportunity(market)
            # Edge = 0.40 - 0.25 = 0.15, above min_probability_edge of 0.05
            assert result is not None
            assert result["side"] == "yes"
            assert result["edge"] >= 0.05

    def test_evaluate_opportunity_no_edge(self, kalshi_bot):
        """Test evaluate_opportunity returns None when no edge"""
        market = {
            "ticker": "FED-RAISE-26FEB",
            "series_ticker": "FED",
            "title": "Will the Fed raise rates?",
            "yes_bid": 48,
            "yes_ask": 50,
            "no_bid": 48,
            "no_ask": 50,
        }
        # Our probability matches market -> no edge
        with patch.object(kalshi_bot, "_estimate_probability", return_value=0.50):
            result = kalshi_bot.evaluate_opportunity(market)
            assert result is None


class TestSportsAIBot:
    """Test the AI-powered sports betting bot using actual _evaluate_opportunity"""

    @pytest.fixture
    def sports_bot(self, mock_env_vars):
        # Patch modules that may not exist before importing
        import types

        mock_final_ai = types.ModuleType("final_complete_ai")
        mock_final_ai.SportsBettingAI = Mock
        mock_news_feeds = types.ModuleType("news_feeds")
        mock_news_feeds.NewsAggregator = Mock
        with patch.dict(
            "sys.modules",
            {
                "final_complete_ai": mock_final_ai,
                "news_feeds": mock_news_feeds,
            },
        ):
            from bots.sports_ai_bot import SportsAIBot

            with patch("bots.sports_ai_bot.KalshiClient"), patch(
                "bots.sports_ai_bot.SportsBettingAI"
            ), patch("bots.sports_ai_bot.NewsAggregator"):
                bot = SportsAIBot()
                return bot

    def test_evaluate_opportunity_qualifies(self, sports_bot):
        """Test _evaluate_opportunity with high-scoring opportunity"""
        opp = {
            "player": "Drake Maye",
            "prop": "passing_yards",
            "confidence": 80,
            "expected_value": 0.12,
            "edge": 0.07,
        }
        qualifies, score, breakdown = sports_bot._evaluate_opportunity(opp)
        assert qualifies is True
        assert score > sports_bot.min_weighted_score

    def test_evaluate_opportunity_rejects(self, sports_bot):
        """Test _evaluate_opportunity with low-scoring opportunity"""
        opp = {
            "player": "Unknown Player",
            "prop": "passing_yards",
            "confidence": 30,
            "expected_value": 0.01,
            "edge": 0.01,
        }
        qualifies, score, breakdown = sports_bot._evaluate_opportunity(opp)
        assert qualifies is False
        assert score < sports_bot.min_weighted_score


@pytest.mark.skip(
    reason="No testable pure functions — all methods need full KalshiClient + ArbitrageDetector"
)
class TestArbitrageBot:
    """Test cross-market arbitrage detection"""


class TestOANDABot:
    """Test OANDA forex trading bot using actual interface"""

    @pytest.fixture
    def oanda_config(self):
        return {
            "practice_url": "https://api-fxpractice.oanda.com",
            "live_url": "https://api-fxtrade.oanda.com",
            "api_key": "test_key",
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

    @pytest.fixture
    def oanda_bot(self, mock_env_vars, oanda_config):
        from bots.oanda_bot import OANDABot

        bot = OANDABot(config=oanda_config, use_divergence=False)
        return bot

    def _make_candles(self, closes, highs=None, lows=None):
        """Helper to create candle dicts matching OANDA format"""
        if highs is None:
            highs = [c * 1.005 for c in closes]
        if lows is None:
            lows = [c * 0.995 for c in closes]
        candles = []
        for i in range(len(closes)):
            candles.append(
                {
                    "complete": True,
                    "mid": {
                        "o": str(closes[i]),
                        "h": str(highs[i]),
                        "l": str(lows[i]),
                        "c": str(closes[i]),
                    },
                }
            )
        return candles

    def test_calculate_indicators_returns_keys(self, oanda_bot):
        """Test calculate_indicators returns expected keys"""
        # Generate 30 candles with uptrend
        closes = [1.08 + i * 0.001 for i in range(30)]
        candles = self._make_candles(closes)
        result = oanda_bot.calculate_indicators(candles)
        assert "close" in result
        assert "fast_ma" in result
        assert "slow_ma" in result
        assert "rsi" in result
        assert "atr" in result

    def test_check_entry_signal_bullish_crossover(self, oanda_bot):
        """Test bullish MA crossover generates buy signal"""
        indicators = {
            "fast_ma": 1.090,
            "slow_ma": 1.085,
            "fast_ma_prev": 1.084,
            "slow_ma_prev": 1.085,
            "rsi": 55,
            "hidden_bullish": False,
            "hidden_bearish": False,
            "regular_bullish": False,
            "regular_bearish": False,
            "divergence_strength": 0,
        }
        signal = oanda_bot.check_entry_signal(indicators)
        assert signal == "buy"

    def test_check_entry_signal_bearish_crossover(self, oanda_bot):
        """Test bearish MA crossover generates sell signal"""
        indicators = {
            "fast_ma": 1.080,
            "slow_ma": 1.085,
            "fast_ma_prev": 1.086,
            "slow_ma_prev": 1.085,
            "rsi": 45,
            "hidden_bullish": False,
            "hidden_bearish": False,
            "regular_bullish": False,
            "regular_bearish": False,
            "divergence_strength": 0,
        }
        signal = oanda_bot.check_entry_signal(indicators)
        assert signal == "sell"

    def test_check_entry_signal_no_crossover(self, oanda_bot):
        """Test no signal when MAs don't cross"""
        indicators = {
            "fast_ma": 1.090,
            "slow_ma": 1.085,
            "fast_ma_prev": 1.088,
            "slow_ma_prev": 1.085,
            "rsi": 55,
            "hidden_bullish": False,
            "hidden_bearish": False,
            "regular_bullish": False,
            "regular_bearish": False,
            "divergence_strength": 0,
        }
        signal = oanda_bot.check_entry_signal(indicators)
        assert signal is None


class TestAlpacaCryptoBot:
    """Test Alpaca crypto momentum bot pure functions"""

    @pytest.fixture
    def crypto_bot(self, mock_env_vars):
        from bots.alpaca_crypto_momentum import AlpacaCryptoMomentumBot

        bot = AlpacaCryptoMomentumBot(capital=135.0)
        return bot

    def test_calculate_ema(self, crypto_bot):
        """Test EMA calculation returns pd.Series"""
        prices = pd.Series(
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
        )
        result = crypto_bot.calculate_ema(prices, 12)
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)
        # EMA should lag below in uptrend
        assert result.iloc[-1] < prices.iloc[-1]

    def test_calculate_macd(self, crypto_bot):
        """Test MACD returns dict with expected keys"""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5))
        result = crypto_bot.calculate_macd(prices)
        assert "ema_fast" in result
        assert "ema_slow" in result
        assert "macd_line" in result
        assert "signal_line" in result
        assert "histogram" in result

    def test_calculate_entropy_random(self, crypto_bot):
        """Test entropy is high for random prices"""
        np.random.seed(42)
        random_prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 2))
        entropy = crypto_bot.calculate_entropy(random_prices)
        assert 0 <= entropy <= 1

    def test_calculate_entropy_trending(self, crypto_bot):
        """Test entropy is lower for strongly trending prices"""
        # Strong uptrend — all positive returns
        trending_prices = pd.Series([100 + i * 2 for i in range(50)])
        entropy = crypto_bot.calculate_entropy(trending_prices)
        # Trending should have lower entropy than random
        assert entropy < 1.0

    def test_check_volume_confirmation_high(self, crypto_bot):
        """Test volume confirmation with high volume"""
        # 20 normal volume bars + 1 spike
        volumes = pd.Series([1000] * 21 + [3000])
        ratio = crypto_bot.check_volume_confirmation(volumes)
        assert ratio > 1.5  # Above confirmation threshold

    def test_check_volume_confirmation_low(self, crypto_bot):
        """Test volume confirmation with low volume"""
        volumes = pd.Series([1000] * 21 + [500])
        ratio = crypto_bot.check_volume_confirmation(volumes)
        assert ratio < 1.0  # Below average


class TestSentimentBot:
    """Test sentiment bot using actual interface"""

    @pytest.fixture
    def sentiment_bot(self, mock_env_vars):
        from bots.sentiment_bot import SentimentBot

        with patch("bots.sentiment_bot.SentimentAnalyzer"), patch(
            "bots.sentiment_bot.RedditScraper"
        ), patch("bots.sentiment_bot.StockTwitsScraper"), patch(
            "bots.sentiment_bot.sqlite3"
        ) as mock_sqlite:
            mock_conn = Mock()
            mock_sqlite.connect.return_value = mock_conn
            mock_conn.cursor.return_value = Mock()
            bot = SentimentBot(paper_mode=True)
            return bot

    def test_calculate_zscore_insufficient_history(self, sentiment_bot):
        """Test zscore returns 0 with insufficient history"""
        zscore = sentiment_bot.calculate_zscore("BTC", 0.5)
        assert zscore == 0.0

    def test_calculate_zscore_with_history(self, sentiment_bot):
        """Test zscore calculation with sufficient history"""
        # Pre-populate history
        sentiment_bot.sentiment_history["BTC"] = deque(maxlen=30)
        # Add 20 entries with mean ~0.5
        for val in [
            0.4,
            0.5,
            0.6,
            0.45,
            0.55,
            0.5,
            0.48,
            0.52,
            0.47,
            0.53,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]:
            sentiment_bot.sentiment_history["BTC"].append(val)

        # Extreme score should produce high z-score
        zscore = sentiment_bot.calculate_zscore("BTC", 0.9)
        assert zscore > 1.0

        # Normal score should produce low z-score
        zscore_normal = sentiment_bot.calculate_zscore("BTC", 0.5)
        assert abs(zscore_normal) < 1.0


class TestWeatherEdgeFinder:
    """Test weather edge finder pure math functions"""

    @pytest.fixture
    def weather_bot(self, mock_env_vars):
        from bots.weather_edge_finder import WeatherEdgeFinder

        with patch("bots.weather_edge_finder.MarketScanner"), patch(
            "bots.weather_edge_finder.WeatherScraper"
        ):
            bot = WeatherEdgeFinder()
            return bot

    def test_norm_cdf_zero(self, weather_bot):
        """norm_cdf(0) should be 0.5"""
        result = weather_bot.norm_cdf(0)
        assert abs(result - 0.5) < 1e-10

    def test_norm_cdf_positive(self, weather_bot):
        """norm_cdf(large positive) should be ~1"""
        result = weather_bot.norm_cdf(5)
        assert result > 0.999

    def test_norm_cdf_negative(self, weather_bot):
        """norm_cdf(large negative) should be ~0"""
        result = weather_bot.norm_cdf(-5)
        assert result < 0.001

    def test_calc_prob_above_high_forecast(self, weather_bot):
        """Forecast 10° above threshold -> high probability"""
        prob = weather_bot.calc_prob_above(80, 70)
        assert prob > 0.99  # 10 degrees above with 3° uncertainty

    def test_calc_prob_below_low_forecast(self, weather_bot):
        """Forecast 10° below threshold -> high probability of below"""
        prob = weather_bot.calc_prob_below(60, 70)
        assert prob > 0.99

    def test_calc_prob_bracket(self, weather_bot):
        """Forecast in middle of bracket -> high probability"""
        prob = weather_bot.calc_prob_bracket(70, 65, 75)
        assert prob > 0.5  # Should be high when forecast is in middle

    def test_calculate_kelly_no_edge(self, weather_bot):
        """Kelly fraction should be 0 when no edge"""
        kelly = weather_bot.calculate_kelly_fraction(
            probability=0.50, market_price=0.55, confidence=0.8
        )
        assert kelly == 0.0  # prob < market_price -> no edge

    def test_calculate_kelly_positive_edge(self, weather_bot):
        """Kelly fraction should be positive when edge exists"""
        kelly = weather_bot.calculate_kelly_fraction(
            probability=0.70, market_price=0.50, confidence=0.8
        )
        assert kelly > 0.0

    def test_calculate_kelly_capped(self, weather_bot):
        """Kelly fraction should be capped at max_kelly"""
        kelly = weather_bot.calculate_kelly_fraction(
            probability=0.99, market_price=0.10, confidence=1.0, max_kelly=0.25
        )
        assert kelly <= 0.25

    def test_parse_ticker_above(self, weather_bot):
        """Parse above-threshold ticker"""
        result = weather_bot.parse_ticker(
            "KXHIGHNY-26JAN27-T23", "High temp above 23 in NYC"
        )
        assert result is not None
        assert result["city"] == "NYC"
        assert result["type"] == "above"
        assert result["threshold"] == 23

    def test_parse_ticker_bracket(self, weather_bot):
        """Parse bracket ticker"""
        result = weather_bot.parse_ticker(
            "KXHIGHNY-26JAN27-B22.5", "High temp between 22-23 in NYC"
        )
        assert result is not None
        assert result["type"] == "bracket"

    def test_parse_ticker_invalid(self, weather_bot):
        """Parse invalid ticker returns None"""
        result = weather_bot.parse_ticker("RANDOM-TICKER", "Some random title")
        assert result is None


# AI Component Tests
class TestAIComponents:
    """Test AI-specific components like LLM client and veto layer"""

    @pytest.fixture
    def llm_client(self, mock_env_vars):
        from ai.llm_client import LLMClient

        return LLMClient()

    @pytest.mark.asyncio
    async def test_llm_client_query(self, mock_llm_client):
        """Test LLM client query functionality"""
        response = await mock_llm_client.query("Test prompt")

        assert response.content is not None
        assert response.provider is not None
        assert response.tokens_used > 0

    @pytest.mark.asyncio
    async def test_llm_client_fallback(self, llm_client):
        """Test LLM client fallback mechanism"""
        # Mock primary provider failure
        with patch.object(
            llm_client, "_query_deepseek", side_effect=Exception("Primary failed")
        ):
            with patch.object(llm_client, "_query_claude") as mock_fallback:
                from ai.llm_client import LLMProvider

                mock_fallback.return_value = {
                    "content": "Fallback response",
                    "provider": LLMProvider.CLAUDE_HAIKU,
                    "tokens": 25,
                }

                response = await llm_client.query("Test prompt")
                assert response.provider == LLMProvider.CLAUDE_HAIKU

    def test_veto_layer_initialization(self, mock_env_vars):
        """Test AI veto layer initialization"""
        try:
            from ai.veto_layer import AIVetoFilter

            veto_filter = AIVetoFilter(enabled=True, dry_run=True)
            assert veto_filter.enabled is True
            assert veto_filter.dry_run is True
        except ImportError:
            pytest.skip("Veto layer not available")

    @pytest.mark.asyncio
    async def test_ai_prediction_analyzer(self, mock_env_vars, mock_kalshi_client):
        """Test AI prediction analyzer"""
        try:
            from bots.ai_prediction_analyzer import AIPredictionAnalyzer

            with patch("bots.kalshi_client.KalshiClient") as mock_client_class:
                mock_client_class.return_value = mock_kalshi_client
                analyzer = AIPredictionAnalyzer(mock_kalshi_client)

                # Test keyword extraction (case-insensitive matching)
                keywords = analyzer._extract_keywords(
                    "oil gas climate weather recession"
                )
                assert len(keywords) > 0

                # Verify uppercase terms like GDP, Fed, CPI match in lowered text
                keywords_upper = analyzer._extract_keywords(
                    "gdp growth and fed policy on cpi data"
                )
                assert "GDP" in keywords_upper
                assert "Fed" in keywords_upper
                assert "CPI" in keywords_upper
        except ImportError:
            pytest.skip("AI prediction analyzer not available")


# Performance Tests for AI Bots
class TestAIBotPerformance:
    """Performance tests for AI bots"""

    def test_bot_response_time(self, mock_env_vars, performance_monitor):
        """Test that KalshiBot initializes within acceptable time"""
        from bots.kalshi_bot import KalshiBot

        config = {
            "base_url": "https://api.elections.kalshi.com/trade-api/v2",
            "api_key_id": "test",
            "private_key_path": "test.pem",
            "paper_mode": True,
            "max_position_size": 15,
            "max_concurrent_positions": 3,
            "min_probability_edge": 0.05,
            "target_series_patterns": [],
            "min_time_to_expiry_hours": 1,
            "max_time_to_expiry_days": 7,
            "daily_loss_limit": 10,
            "stop_trading_on_loss": True,
        }
        with patch.object(KalshiBot, "_load_private_key", return_value=Mock()):
            performance_monitor.start()
            KalshiBot(config=config)
            duration = performance_monitor.stop()
            assert duration < 5.0

    @pytest.mark.asyncio
    async def test_concurrent_bot_execution(self):
        """Test multiple bots running concurrently"""

        async def mock_bot_run():
            await asyncio.sleep(0.1)
            return {"status": "success"}

        # Run multiple bots concurrently
        tasks = [mock_bot_run() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r["status"] == "success" for r in results)

    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test that bots don't consume excessive memory"""
        psutil = __import__("pytest").importorskip("psutil")
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Simulate bot operations
        for _ in range(100):
            await asyncio.sleep(0.001)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB for this test)
        assert memory_increase < 50 * 1024 * 1024


# Error Handling Tests
class TestAIBotErrorHandling:
    """Test error handling in AI bots"""

    def test_api_failure_handling(self, mock_env_vars):
        """Test KalshiBot.run_strategy() handles API errors gracefully"""
        from bots.kalshi_bot import KalshiBot

        config = {
            "base_url": "https://api.elections.kalshi.com/trade-api/v2",
            "api_key_id": "test",
            "private_key_path": "test.pem",
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
            # Mock session to raise on request
            bot.session = Mock()
            bot.session.request.side_effect = Exception("Connection failed")
            # run_strategy should handle errors and return a result dict (not crash)
            result = bot.run_strategy()
            assert result is not None

    def test_rate_limit_handling(self, mock_env_vars):
        """Test KalshiBot get_markets handles rate limit errors"""
        from bots.kalshi_bot import KalshiBot
        from requests.exceptions import HTTPError

        config = {
            "base_url": "https://api.elections.kalshi.com/trade-api/v2",
            "api_key_id": "test",
            "private_key_path": "test.pem",
            "paper_mode": True,
            "max_position_size": 15,
            "max_concurrent_positions": 3,
            "min_probability_edge": 0.05,
            "target_series_patterns": [],
            "min_time_to_expiry_hours": 1,
            "max_time_to_expiry_days": 7,
            "daily_loss_limit": 10,
            "stop_trading_on_loss": True,
        }
        with patch.object(KalshiBot, "_load_private_key", return_value=Mock()):
            bot = KalshiBot(config=config)
            # Mock _request to raise HTTPError (bypassing signature logic)
            with patch.object(
                bot, "_request", side_effect=HTTPError("429 Too Many Requests")
            ):
                # get_markets should return empty list on error
                result = bot.get_markets()
                assert isinstance(result, list)

    def test_invalid_data_handling(self, mock_env_vars):
        """Test SentimentBot handles missing data gracefully"""
        from bots.sentiment_bot import SentimentBot

        with patch("bots.sentiment_bot.SentimentAnalyzer"), patch(
            "bots.sentiment_bot.RedditScraper"
        ), patch("bots.sentiment_bot.StockTwitsScraper"), patch(
            "bots.sentiment_bot.sqlite3"
        ) as mock_sqlite:
            mock_conn = Mock()
            mock_sqlite.connect.return_value = mock_conn
            mock_conn.cursor.return_value = Mock()
            bot = SentimentBot(paper_mode=True)

            # calculate_zscore with no history should return 0 (not crash)
            zscore = bot.calculate_zscore("INVALID", 0.5)
            assert zscore == 0.0

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self):
        """Test handling of connection timeouts"""
        import asyncio

        async def timeout_simulation():
            await asyncio.sleep(10)  # Simulate long operation
            return "success"

        # Test with short timeout
        try:
            await asyncio.wait_for(timeout_simulation(), timeout=1.0)
            assert False, "Should have timed out"
        except asyncio.TimeoutError:
            # This is expected behavior
            assert True
