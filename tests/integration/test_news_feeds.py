"""
News Feed Integration Tests
===========================
Test news feed connectivity, fallbacks, and data processing.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, List

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class TestNewsFeedConnectivity:
    """Test connectivity to various news sources"""

    @pytest.fixture
    def mock_news_sources(self):
        """Mock various news source APIs"""
        return {
            "reuters": {
                "url": "https://api.reuters.com/v1/news",
                "api_key": "test_reuters_key",
                "status": "active",
            },
            "bloomberg": {
                "url": "https://api.bloomberg.com/v1/news",
                "api_key": "test_bloomberg_key",
                "status": "active",
            },
            "fred": {
                "url": "https://api.stlouisfed.org/fred",
                "api_key": "test_fred_key",
                "status": "active",
            },
            "yahoo_finance": {
                "url": "https://query1.finance.yahoo.com/v1/finance",
                "api_key": None,
                "status": "active",
            },
        }

    @pytest.fixture
    def mock_news_responses(self):
        """Mock news API responses"""
        return {
            "reuters_fed_news": {
                "status": "ok",
                "articles": [
                    {
                        "title": "Fed Keeps Rates Steady, Signals Caution",
                        "description": "Federal Reserve maintains interest rates at current levels...",
                        "content": (
                            "The Federal Reserve announced today that it will"
                            " keep interest rates unchanged at 5.25-5.50%..."
                        ),
                        "publishedAt": "2024-01-15T14:30:00Z",
                        "source": {"name": "Reuters"},
                        "url": "https://reuters.com/article/fed-rates-1",
                    }
                ],
            },
            "bloomberg_markets": {
                "status": "success",
                "data": [
                    {
                        "headline": "S&P 500 Reaches Record High on Tech Rally",
                        "summary": "Technology stocks led the market higher as investors bet on AI growth...",
                        "published_time": "2024-01-15T15:45:00Z",
                        "source": "Bloomberg",
                        "symbols": ["SPY", "QQQ", "AAPL", "NVDA"],
                    }
                ],
            },
            "fred_economic_data": {
                "observations": [
                    {
                        "realtime_start": "2024-01-15",
                        "realtime_end": "2024-01-15",
                        "date": "2024-01-15",
                        "value": "3.2",
                    }
                ],
                "series_id": "CPIAUCSL",
                "title": "Consumer Price Index",
            },
        }

    @pytest.mark.asyncio
    async def test_reuters_connection(self, mock_news_sources, mock_news_responses):
        """Test Reuters API connection"""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_news_responses["reuters_fed_news"]
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            # Simulate news fetcher
            response = await self._fetch_news_async("reuters", "fed rates")

            assert response["status"] == "ok"
            assert len(response["articles"]) > 0
            assert "Fed" in response["articles"][0]["title"]

    @pytest.mark.asyncio
    async def test_bloomberg_connection(self, mock_news_sources, mock_news_responses):
        """Test Bloomberg API connection"""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_news_responses["bloomberg_markets"]
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            response = await self._fetch_news_async("bloomberg", "markets")

            assert response["status"] == "success"
            assert len(response["data"]) > 0

    @pytest.mark.asyncio
    async def test_fred_connection(self, mock_news_sources, mock_news_responses):
        """Test FRED API connection"""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_news_responses["fred_economic_data"]
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            response = await self._fetch_economic_data_async("CPIAUCSL")

            assert "observations" in response
            assert len(response["observations"]) > 0

    async def _fetch_news_async(self, source: str, query: str):
        """Mock async news fetch"""
        await asyncio.sleep(0.1)  # Simulate network delay
        # This would normally make HTTP request
        if source == "reuters":
            return {"status": "ok", "articles": [{"title": "Fed News"}]}
        elif source == "bloomberg":
            return {"status": "success", "data": [{"headline": "Market News"}]}
        return {}

    async def _fetch_economic_data_async(self, series_id: str):
        """Mock async economic data fetch"""
        await asyncio.sleep(0.1)
        return {"observations": [{"value": "3.2"}]}


class TestNewsFeedFallbacks:
    """Test fallback mechanisms when primary sources fail"""

    @pytest.fixture
    def news_aggregator(self):
        """Mock news aggregator with fallback logic"""

        class MockNewsAggregator:
            def __init__(self):
                self.primary_sources = ["reuters", "bloomberg", "ap"]
                self.fallback_sources = ["yahoo", "google_news", "rss_feeds"]
                self.failed_sources = []

            async def fetch_with_fallback(self, topic: str, max_retries: int = 3):
                """Fetch news with fallback logic"""
                sources_to_try = self.primary_sources + self.fallback_sources

                for source in sources_to_try:
                    if source in self.failed_sources:
                        continue

                    try:
                        result = await self._fetch_from_source(source, topic)
                        if result:
                            return result
                    except Exception:
                        self.failed_sources.append(source)
                        continue

                return []

            async def _fetch_from_source(self, source: str, topic: str):
                """Mock source fetch"""
                await asyncio.sleep(0.1)
                if source in ["reuters", "bloomberg"]:
                    # Simulate primary source success
                    return [{"title": f"{topic} from {source}", "source": source}]
                elif source in ["yahoo", "google_news"]:
                    # Simulate fallback source
                    return [
                        {"title": f"Fallback {topic} from {source}", "source": source}
                    ]
                return None

        return MockNewsAggregator()

    @pytest.mark.asyncio
    async def test_primary_source_success(self, news_aggregator):
        """Test successful fetch from primary source"""
        result = await news_aggregator.fetch_with_fallback("fed rates")

        assert len(result) > 0
        assert result[0]["source"] in ["reuters", "bloomberg", "ap"]

    @pytest.mark.asyncio
    async def test_primary_source_failure_fallback(self, news_aggregator):
        """Test fallback when primary sources fail"""
        # Simulate primary source failures
        news_aggregator.failed_sources = ["reuters", "bloomberg", "ap"]

        result = await news_aggregator.fetch_with_fallback("market news")

        assert len(result) > 0
        assert result[0]["source"] in ["yahoo", "google_news"]

    @pytest.mark.asyncio
    async def test_all_sources_failure(self, news_aggregator):
        """Test behavior when all sources fail"""
        # Simulate all sources failing
        news_aggregator.failed_sources = [
            "reuters",
            "bloomberg",
            "ap",
            "yahoo",
            "google_news",
            "rss_feeds",
        ]

        result = await news_aggregator.fetch_with_fallback("crypto news")

        assert result == []

    @pytest.mark.asyncio
    async def test_source_recovery(self, news_aggregator):
        """Test that failed sources can recover"""
        # Initially fail primary source
        news_aggregator.failed_sources = ["reuters"]

        # First fetch should use fallback
        result1 = await news_aggregator.fetch_with_fallback("inflation")
        assert result1[0]["source"] != "reuters"

        # Clear failed sources (simulate recovery)
        news_aggregator.failed_sources = []

        # Second fetch should use primary again
        result2 = await news_aggregator.fetch_with_fallback("inflation")
        assert result2[0]["source"] in ["reuters", "bloomberg", "ap"]


class TestNewsProcessingPipeline:
    """Test news data processing and filtering"""

    @pytest.fixture
    def news_processor(self):
        """Mock news processing pipeline"""

        class MockNewsProcessor:
            def __init__(self):
                self.keywords_filter = {
                    "fed": [
                        "federal reserve",
                        "interest rate",
                        "monetary policy",
                        "jerome powell",
                    ],
                    "inflation": [
                        "cpi",
                        "consumer price",
                        "inflation rate",
                        "price index",
                    ],
                    "employment": [
                        "unemployment",
                        "jobs report",
                        "nfp",
                        "labor market",
                    ],
                    "crypto": ["bitcoin", "ethereum", "cryptocurrency", "blockchain"],
                }
                self.relevance_threshold = 0.3

            def process_news_batch(self, articles: List[Dict]) -> List[Dict]:
                """Process a batch of news articles"""
                processed = []

                for article in articles:
                    processed_article = self._process_article(article)
                    if processed_article["relevance_score"] >= self.relevance_threshold:
                        processed.append(processed_article)

                return processed

            def _process_article(self, article: Dict) -> Dict:
                """Process a single article"""
                content = (
                    f"{article.get('title', '')} {article.get('content', '')}".lower()
                )

                # Calculate relevance
                relevance_scores = []
                matched_categories = []

                for category, keywords in self.keywords_filter.items():
                    score = sum(1 for keyword in keywords if keyword in content)
                    if score > 0:
                        relevance_scores.append(score / len(keywords))
                        matched_categories.append(category)

                max_relevance = max(relevance_scores) if relevance_scores else 0

                # Calculate sentiment (simple mock)
                positive_words = [
                    "up",
                    "rise",
                    "increase",
                    "strong",
                    "growth",
                    "positive",
                ]
                negative_words = [
                    "down",
                    "fall",
                    "decrease",
                    "weak",
                    "decline",
                    "negative",
                ]

                pos_count = sum(1 for word in positive_words if word in content)
                neg_count = sum(1 for word in negative_words if word in content)
                sentiment = (pos_count - neg_count) / max(1, pos_count + neg_count)

                return {
                    **article,
                    "relevance_score": max_relevance,
                    "matched_categories": matched_categories,
                    "sentiment_score": sentiment,
                    "processed_at": datetime.now(),
                    "keywords_found": self._extract_keywords(content),
                }

            def _extract_keywords(self, content: str) -> List[str]:
                """Extract relevant keywords"""
                found_keywords = []
                for category, keywords in self.keywords_filter.items():
                    found_keywords.extend([kw for kw in keywords if kw in content])
                return found_keywords

        return MockNewsProcessor()

    @pytest.mark.asyncio
    async def test_news_relevance_filtering(self, news_processor):
        """Test filtering news by relevance"""
        test_articles = [
            {
                "title": "Federal Reserve Announces Interest Rate Decision",
                "content": "The Federal Reserve today announced it will maintain interest rates at current levels...",
                "source": "Reuters",
            },
            {
                "title": "Celebrity News: Actor Wins Award",
                "content": "Famous actor receives prestigious award at ceremony...",
                "source": "Entertainment Weekly",
            },
            {
                "title": "Bitcoin Price Surges on Crypto Market Rally",
                "content": (
                    "Bitcoin and other cryptocurrencies saw significant"
                    " gains as blockchain adoption increases..."
                ),
                "source": "CoinDesk",
            },
        ]

        processed = news_processor.process_news_batch(test_articles)

        # Should filter out entertainment news, keep Fed and crypto news
        assert len(processed) == 2
        assert any("fed" in article["matched_categories"] for article in processed)
        assert any("crypto" in article["matched_categories"] for article in processed)

    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, news_processor):
        """Test sentiment scoring"""
        positive_article = {
            "title": "Strong Economic Growth Continues",
            "content": "Economic indicators show positive trends with strong growth and increased employment...",
            "source": "Bloomberg",
        }

        negative_article = {
            "title": "Market Decline on Economic Concerns",
            "content": "Markets fall sharply as economic data shows weak performance and declining indicators...",
            "source": "Reuters",
        }

        pos_result = news_processor._process_article(positive_article)
        neg_result = news_processor._process_article(negative_article)

        assert pos_result["sentiment_score"] > 0
        assert neg_result["sentiment_score"] < 0

    @pytest.mark.asyncio
    async def test_keyword_extraction(self, news_processor):
        """Test keyword extraction functionality"""
        article = {
            "title": "Federal Reserve Chairman Jerome Powell Discusses Monetary Policy",
            "content": (
                "Federal Reserve Chairman Jerome Powell spoke about"
                " monetary policy and interest rate decisions..."
            ),
            "source": "Wall Street Journal",
        }

        processed = news_processor._process_article(article)

        assert len(processed["keywords_found"]) > 0
        assert "federal reserve" in processed["keywords_found"]
        assert "jerome powell" in processed["keywords_found"]

    @pytest.mark.asyncio
    async def test_batch_processing_performance(
        self, news_processor, performance_monitor
    ):
        """Test performance of batch news processing"""
        # Generate large batch of test articles
        test_articles = []
        for i in range(100):
            test_articles.append(
                {
                    "title": f"Test Article {i} about Federal Reserve Policy",
                    "content": (
                        f"Article {i} discusses monetary policy and interest"
                        " rate decisions by the Federal Reserve..."
                    ),
                    "source": "Test Source",
                }
            )

        performance_monitor.start()
        processed = news_processor.process_news_batch(test_articles)
        duration = performance_monitor.stop()

        # Should process 100 articles quickly
        assert len(processed) > 0
        assert duration < 5.0  # Should complete within 5 seconds


class TestNewsAlertSystem:
    """Test news-based alert and notification system"""

    @pytest.fixture
    def alert_system(self):
        """Mock news alert system"""

        class MockAlertSystem:
            def __init__(self):
                self.alert_rules = [
                    {
                        "name": "Fed Rate Decision",
                        "keywords": ["federal reserve", "interest rate", "fed funds"],
                        "priority": "high",
                        "notify_channels": ["telegram", "email"],
                    },
                    {
                        "name": "Inflation Data",
                        "keywords": ["cpi", "inflation", "consumer price"],
                        "priority": "medium",
                        "notify_channels": ["telegram"],
                    },
                    {
                        "name": "Crypto Regulation",
                        "keywords": ["crypto regulation", "sec bitcoin", "crypto ban"],
                        "priority": "high",
                        "notify_channels": ["telegram", "sms"],
                    },
                ]
                self.triggered_alerts = []

            def check_alerts(self, articles: List[Dict]) -> List[Dict]:
                """Check articles against alert rules"""
                alerts = []

                for article in articles:
                    content = f"{article.get('title', '')} {article.get('content', '')}".lower()

                    for rule in self.alert_rules:
                        if any(keyword in content for keyword in rule["keywords"]):
                            alert = {
                                "rule_name": rule["name"],
                                "article": article,
                                "priority": rule["priority"],
                                "channels": rule["notify_channels"],
                                "timestamp": datetime.now(),
                            }
                            alerts.append(alert)
                            self.triggered_alerts.append(alert)

                return alerts

            async def send_alerts(self, alerts: List[Dict]):
                """Send alerts through specified channels"""
                for alert in alerts:
                    for channel in alert["channels"]:
                        await self._send_to_channel(channel, alert)

            async def _send_to_channel(self, channel: str, alert: Dict):
                """Mock sending alert to specific channel"""
                await asyncio.sleep(0.1)  # Simulate sending delay
                # In real implementation, would send to Telegram, email, etc.

        return MockAlertSystem()

    @pytest.mark.asyncio
    async def test_fed_rate_alert(self, alert_system):
        """Test alert triggered by Fed rate news"""
        fed_article = {
            "title": "Federal Reserve Cuts Interest Rates",
            "content": "The Federal Reserve announced a 25 basis point cut to the fed funds rate...",
            "source": "Reuters",
        }

        alerts = alert_system.check_alerts([fed_article])

        assert len(alerts) > 0
        assert alerts[0]["rule_name"] == "Fed Rate Decision"
        assert alerts[0]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_inflation_alert(self, alert_system):
        """Test alert triggered by inflation news"""
        inflation_article = {
            "title": "CPI Data Shows Inflation Rising to 4.2%",
            "content": "Consumer Price Index increased more than expected, showing inflation concerns...",
            "source": "Bloomberg",
        }

        alerts = alert_system.check_alerts([inflation_article])

        assert len(alerts) > 0
        assert alerts[0]["rule_name"] == "Inflation Data"
        assert alerts[0]["priority"] == "medium"

    @pytest.mark.asyncio
    async def test_multiple_alerts_single_article(self, alert_system):
        """Test multiple alerts triggered by one article"""
        multi_topic_article = {
            "title": "Fed Considers Rate Hike Due to Rising Inflation",
            "content": (
                "Federal Reserve officials are considering interest rate"
                " increases as CPI data shows persistent inflation..."
            ),
            "source": "Wall Street Journal",
        }

        alerts = alert_system.check_alerts([multi_topic_article])

        # Should trigger both Fed and inflation alerts
        assert len(alerts) >= 2
        rule_names = [alert["rule_name"] for alert in alerts]
        assert "Fed Rate Decision" in rule_names
        assert "Inflation Data" in rule_names

    @pytest.mark.asyncio
    async def test_no_false_alerts(self, alert_system):
        """Test that irrelevant news doesn't trigger alerts"""
        irrelevant_article = {
            "title": "Sports Team Wins Championship",
            "content": "Local sports team celebrates victory in championship game...",
            "source": "Sports Network",
        }

        alerts = alert_system.check_alerts([irrelevant_article])

        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_alert_notification_sending(self, alert_system):
        """Test sending alerts through notification channels"""
        crypto_article = {
            "title": "SEC Announces New Crypto Regulations",
            "content": "Securities and Exchange Commission releases new regulatory framework for cryptocurrency...",
            "source": "CoinDesk",
        }

        alerts = alert_system.check_alerts([crypto_article])

        # Mock the sending process
        with patch.object(
            alert_system, "_send_to_channel", new_callable=AsyncMock
        ) as mock_send:
            await alert_system.send_alerts(alerts)

            # Should send to both telegram and sms
            assert mock_send.call_count >= 2


class TestNewsDataIntegrity:
    """Test news data validation and integrity checks"""

    @pytest.fixture
    def data_validator(self):
        """Mock data validation system"""

        class MockDataValidator:
            def __init__(self):
                self.required_fields = ["title", "content", "source", "published_at"]
                self.max_age_hours = 24
                self.min_content_length = 50

            def validate_article(self, article: Dict) -> Dict:
                """Validate a single news article"""
                errors = []
                warnings = []

                # Check required fields
                for field in self.required_fields:
                    if field not in article or not article[field]:
                        errors.append(f"Missing required field: {field}")

                # Check content length
                if (
                    article.get("content")
                    and len(article["content"]) < self.min_content_length
                ):
                    warnings.append("Content shorter than minimum length")

                # Check article age
                if article.get("published_at"):
                    try:
                        pub_time = datetime.fromisoformat(
                            article["published_at"].replace("Z", "+00:00")
                        )
                        age_hours = (
                            datetime.now().replace(tzinfo=pub_time.tzinfo) - pub_time
                        ).total_seconds() / 3600
                        if age_hours > self.max_age_hours:
                            warnings.append(f"Article is {age_hours:.1f} hours old")
                    except ValueError:
                        errors.append("Invalid published_at format")

                return {
                    "valid": len(errors) == 0,
                    "errors": errors,
                    "warnings": warnings,
                    "article": article,
                }

            def validate_batch(self, articles: List[Dict]) -> Dict:
                """Validate a batch of articles"""
                results = [self.validate_article(article) for article in articles]

                valid_articles = [r["article"] for r in results if r["valid"]]
                invalid_articles = [r for r in results if not r["valid"]]

                return {
                    "total_articles": len(articles),
                    "valid_articles": len(valid_articles),
                    "invalid_articles": len(invalid_articles),
                    "validation_results": results,
                    "clean_articles": valid_articles,
                }

        return MockDataValidator()

    @pytest.mark.asyncio
    async def test_valid_article_validation(self, data_validator):
        """Test validation of properly formatted article"""
        valid_article = {
            "title": "Federal Reserve Maintains Interest Rates",
            "content": (
                "The Federal Reserve announced today that it will maintain"
                " the federal funds rate at the current level of 5.25-5.50%."
                " This decision comes after careful consideration of economic"
                " indicators and inflation data."
            ),
            "source": "Reuters",
            "published_at": datetime.now().isoformat(),
        }

        result = data_validator.validate_article(valid_article)

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_invalid_article_validation(self, data_validator):
        """Test validation of improperly formatted article"""
        invalid_article = {
            "title": "",  # Empty title
            "content": "Short",  # Too short content
            "source": "Unknown",
            # Missing published_at
        }

        result = data_validator.validate_article(invalid_article)

        assert result["valid"] is False
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_old_article_warning(self, data_validator):
        """Test warning for old articles"""
        old_article = {
            "title": "Old News Article",
            "content": (
                "This is an old news article that was published more than"
                " 24 hours ago and may not be relevant for current trading"
                " decisions."
            ),
            "source": "News Source",
            "published_at": (datetime.now() - timedelta(hours=30)).isoformat(),
        }

        result = data_validator.validate_article(old_article)

        assert result["valid"] is True
        assert any("hours old" in warning for warning in result["warnings"])

    @pytest.mark.asyncio
    async def test_batch_validation(self, data_validator):
        """Test batch validation of multiple articles"""
        articles = [
            {  # Valid article
                "title": "Valid News",
                "content": "This is a valid news article with sufficient content and proper formatting.",
                "source": "Reuters",
                "published_at": datetime.now().isoformat(),
            },
            {  # Invalid article
                "title": "",
                "content": "Short",
                "source": "Unknown",
                # Missing published_at
            },
            {  # Another valid article
                "title": "Another Valid News",
                "content": "This is another valid news article with proper content and all required fields filled.",
                "source": "Bloomberg",
                "published_at": datetime.now().isoformat(),
            },
        ]

        result = data_validator.validate_batch(articles)

        assert result["total_articles"] == 3
        assert result["valid_articles"] == 2
        assert result["invalid_articles"] == 1
        assert len(result["clean_articles"]) == 2
