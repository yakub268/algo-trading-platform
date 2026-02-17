"""
Test Suite for Computer Vision Trading Bot

Comprehensive testing for the CV bot including:
- Component initialization
- Windows-MCP integration
- Visual element detection
- Broker interface interaction
- Error handling and edge cases

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import unittest
import time
from unittest.mock import Mock, patch, MagicMock

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has_module(name):
    """Check if a module is importable."""
    import importlib

    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


from bots.computer_vision_bot import (
    ComputerVisionBot,
    WindowsMCPClient,
    ComputerVisionEngine,
    BrokerInterfaceManager,
    TradingOpportunity,
)


class TestComputerVisionBot(unittest.TestCase):
    """Test cases for the main Computer Vision Bot"""

    def setUp(self):
        """Set up test fixtures"""
        self.bot = ComputerVisionBot(paper_mode=True, enabled_brokers=["alpaca"])

    def test_initialization(self):
        """Test bot initialization"""
        self.assertTrue(self.bot.paper_mode)
        self.assertEqual(self.bot.enabled_brokers, ["alpaca"])
        self.assertIsNotNone(self.bot.mcp_client)
        self.assertIsNotNone(self.bot.broker_manager)

    def test_get_status(self):
        """Test status reporting"""
        status = self.bot.get_status()

        self.assertIn("paper_mode", status)
        self.assertIn("enabled_brokers", status)
        self.assertIn("scan_interval", status)
        self.assertIn("confidence_threshold", status)

        self.assertTrue(status["paper_mode"])
        self.assertEqual(status["enabled_brokers"], ["alpaca"])

    @patch("bots.computer_vision_bot.ComputerVisionBot._is_broker_open")
    @patch("bots.computer_vision_bot.BrokerInterfaceManager.navigate_to_broker")
    @patch("bots.computer_vision_bot.BrokerInterfaceManager.scan_for_opportunities")
    def test_run_strategy(self, mock_scan, mock_navigate, mock_is_open):
        """Test strategy execution"""
        # Mock dependencies
        mock_is_open.return_value = False
        mock_navigate.return_value = True
        mock_scan.return_value = []

        result = self.bot.run_strategy()

        self.assertIn("status", result)
        self.assertIn("opportunities_found", result)

    def test_filter_opportunities(self):
        """Test opportunity filtering logic"""
        # Create test opportunities with different confidence levels
        opportunities = [
            TradingOpportunity(
                broker="alpaca",
                symbol="AAPL",
                action="buy",
                price=150.0,
                confidence=0.9,
                visual_evidence="",
                reasoning="test",
                timestamp=None,
            ),
            TradingOpportunity(
                broker="alpaca",
                symbol="AAPL",
                action="sell",
                price=150.0,
                confidence=0.5,
                visual_evidence="",
                reasoning="test",
                timestamp=None,
            ),
            TradingOpportunity(
                broker="kalshi",
                symbol="ELECTION",
                action="buy",
                price=0.6,
                confidence=0.8,
                visual_evidence="",
                reasoning="test",
                timestamp=None,
            ),
        ]

        filtered = self.bot._filter_opportunities(opportunities)

        # Should filter out low confidence (0.5 < 0.7 threshold)
        self.assertEqual(len(filtered), 2)

        # Should be sorted by confidence (0.9, 0.8)
        self.assertEqual(filtered[0].confidence, 0.9)
        self.assertEqual(filtered[1].confidence, 0.8)


class TestWindowsMCPClient(unittest.TestCase):
    """Test cases for Windows-MCP integration"""

    @classmethod
    def setUpClass(cls):
        if "pyautogui" not in sys.modules:
            cls._mock_pyautogui = MagicMock()
            sys.modules["pyautogui"] = cls._mock_pyautogui
            cls._pyautogui_mocked = True
        else:
            cls._pyautogui_mocked = False

    @classmethod
    def tearDownClass(cls):
        if cls._pyautogui_mocked:
            del sys.modules["pyautogui"]

    def setUp(self):
        """Set up test fixtures"""
        self.client = WindowsMCPClient()

    @patch("subprocess.Popen")
    def test_start_server(self, mock_popen):
        """Test MCP server startup"""
        mock_popen.return_value = Mock()

        with patch.object(self.client, "_test_connection", return_value=True):
            result = self.client.start_server()
            self.assertTrue(result)
            self.assertTrue(self.client.is_connected)

    @patch("pyautogui.screenshot")
    def test_take_screenshot(self, mock_screenshot):
        """Test screenshot capture"""
        mock_screenshot.return_value = Mock()

        result = self.client.take_screenshot()
        self.assertIsNotNone(result)

    @patch("pyautogui.click")
    def test_click_element(self, mock_click):
        """Test element clicking"""
        result = self.client.click_element(100, 200, "left")
        self.assertTrue(result)
        mock_click.assert_called_once_with(100, 200, button="left")

    @patch("pyautogui.click")
    @patch("pyautogui.type")
    def test_type_text(self, mock_type, mock_click):
        """Test text typing"""
        result = self.client.type_text(100, 200, "test text", clear=True)
        self.assertTrue(result)
        mock_click.assert_called_once()
        mock_type.assert_called_once_with("test text")


class TestComputerVisionEngine(unittest.TestCase):
    """Test cases for the CV engine"""

    @classmethod
    def setUpClass(cls):
        cls._modules_mocked = False
        cls._cv2_injected = False
        if "cv2" not in sys.modules:
            cls._mock_cv2 = MagicMock()
            sys.modules["cv2"] = cls._mock_cv2
            cls._modules_mocked = True
        # Inject cv2 into bot module namespace if missing
        import bots.computer_vision_bot as cv_mod

        if not hasattr(cv_mod, "cv2"):
            cv_mod.cv2 = sys.modules["cv2"]
            cls._cv2_injected = True

    @classmethod
    def tearDownClass(cls):
        if cls._cv2_injected:
            import bots.computer_vision_bot as cv_mod

            if hasattr(cv_mod, "cv2"):
                delattr(cv_mod, "cv2")
        if cls._modules_mocked:
            del sys.modules["cv2"]

    def setUp(self):
        """Set up test fixtures"""
        self.engine = ComputerVisionEngine()

    def test_initialization(self):
        """Test CV engine initialization"""
        self.assertIsInstance(self.engine.template_cache, dict)
        self.assertIsInstance(self.engine.ocr_confidence_threshold, float)

    def test_extract_number(self):
        """Test number extraction from text"""
        test_cases = [
            ("$123.45", 123.45),
            ("Price: 50.0", 50.0),
            ("10%", 10.0),
            ("No number here", None),
            ("", None),
        ]

        for text, expected in test_cases:
            result = self.engine._extract_number(text)
            self.assertEqual(result, expected)

    @patch("bots.computer_vision_bot.CV_AVAILABLE", True)
    @patch("bots.computer_vision_bot.os.path.exists", return_value=True)
    def test_load_template(self, mock_exists):
        """Test template loading and caching"""
        import bots.computer_vision_bot as cv_mod

        mock_cv2 = cv_mod.cv2
        mock_cv2.imread.reset_mock()
        mock_cv2.imread.return_value = Mock()
        mock_cv2.IMREAD_COLOR = 1
        # Clear cache from prior runs
        self.engine.template_cache.clear()

        # First load
        result1 = self.engine.load_template("test_template.png")

        # Second load (should use cache)
        result2 = self.engine.load_template("test_template.png")

        self.assertEqual(result1, result2)
        mock_cv2.imread.assert_called_once()

    def test_remove_overlaps(self):
        """Test overlap removal in template matching"""
        matches = [(10, 10, 0.9), (15, 15, 0.8), (100, 100, 0.7)]
        template_shape = (50, 50)

        filtered = self.engine._remove_overlaps(matches, template_shape)

        # Should remove overlapping matches (10,10) and (15,15)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0], (10, 10, 0.9))  # Highest confidence kept
        self.assertEqual(filtered[1], (100, 100, 0.7))  # Non-overlapping kept


class TestBrokerInterfaceManager(unittest.TestCase):
    """Test cases for broker interface management"""

    @classmethod
    def setUpClass(cls):
        cls._modules_mocked = False
        if "cv2" not in sys.modules:
            cls._mock_cv2 = MagicMock()
            sys.modules["cv2"] = cls._mock_cv2
            cls._modules_mocked = True

    @classmethod
    def tearDownClass(cls):
        if cls._modules_mocked:
            del sys.modules["cv2"]

    def setUp(self):
        """Set up test fixtures"""
        self.mcp_client = Mock()
        self.manager = BrokerInterfaceManager(self.mcp_client)

    def test_initialization(self):
        """Test broker manager initialization"""
        self.assertIn("alpaca", self.manager.brokers)
        self.assertIn("kalshi", self.manager.brokers)
        self.assertIn("oanda", self.manager.brokers)

    @patch("webbrowser.open")
    def test_navigate_to_broker(self, mock_open):
        """Test broker navigation"""
        result = self.manager.navigate_to_broker("alpaca")
        self.assertTrue(result)
        mock_open.assert_called_once()

    def test_navigate_to_unknown_broker(self):
        """Test navigation to unknown broker"""
        result = self.manager.navigate_to_broker("unknown_broker")
        self.assertFalse(result)

    @patch("cv2.imread")
    def test_scan_for_opportunities(self, mock_imread):
        """Test opportunity scanning"""
        mock_imread.return_value = Mock()
        self.mcp_client.take_screenshot.return_value = "test_screenshot.png"

        with patch.object(
            self.manager.cv_engine, "detect_price_changes", return_value=[]
        ):
            with patch.object(
                self.manager.cv_engine, "find_trading_buttons", return_value=[]
            ):
                opportunities = self.manager.scan_for_opportunities("alpaca")
                self.assertIsInstance(opportunities, list)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and end-to-end workflows"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.bot = ComputerVisionBot(paper_mode=True)

    @patch("bots.computer_vision_bot.ComputerVisionBot._is_broker_open")
    @patch("bots.computer_vision_bot.BrokerInterfaceManager.navigate_to_broker")
    @patch("bots.computer_vision_bot.BrokerInterfaceManager.scan_for_opportunities")
    def test_full_trading_cycle(self, mock_scan, mock_navigate, mock_is_open):
        """Test complete trading cycle from scan to execution"""
        # Setup mocks
        mock_is_open.return_value = False
        mock_navigate.return_value = True

        test_opportunity = TradingOpportunity(
            broker="alpaca",
            symbol="AAPL",
            action="buy",
            price=150.0,
            confidence=0.9,
            visual_evidence="",
            reasoning="Strong buy signal",
            timestamp=None,
        )
        mock_scan.return_value = [test_opportunity]

        # Run strategy
        result = self.bot.run_strategy()

        # Verify results
        self.assertIn(result["status"], ("completed", "success"))
        self.assertIn("opportunities_found", result)

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with broken MCP client
        with patch.object(self.bot.mcp_client, "take_screenshot", return_value=None):
            result = self.bot.run_strategy()
            # Should handle gracefully without crashing

        # Test with invalid broker
        self.bot.enabled_brokers = ["invalid_broker"]
        result = self.bot.run_strategy()
        self.assertIn("status", result)

    def test_performance_under_load(self):
        """Test bot performance with multiple rapid calls"""
        start_time = time.time()

        # Rapid succession calls
        for _ in range(5):
            result = self.bot.run_strategy()
            self.assertIn("status", result)

        end_time = time.time()

        # Should complete within reasonable time (10 seconds for 5 calls)
        self.assertLess(end_time - start_time, 10.0)


class TestOrchestatorIntegration(unittest.TestCase):
    """Test integration with the master orchestrator"""

    def test_orchestrator_compatibility(self):
        """Test that bot works with orchestrator patterns"""
        bot = ComputerVisionBot(paper_mode=True)

        # Test required methods exist
        self.assertTrue(hasattr(bot, "run_strategy"))
        self.assertTrue(hasattr(bot, "get_status"))
        self.assertTrue(hasattr(bot, "find_opportunities"))

        # Test method signatures
        self.assertTrue(callable(bot.run_strategy))
        self.assertTrue(callable(bot.get_status))
        self.assertTrue(callable(bot.find_opportunities))

        # Test return types
        result = bot.run_strategy()
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)

        status = bot.get_status()
        self.assertIsInstance(status, dict)

        opportunities = bot.find_opportunities()
        self.assertIsInstance(opportunities, list)


def run_tests():
    """Run all test suites"""
    print("=" * 60)
    print("COMPUTER VISION BOT TEST SUITE")
    print("=" * 60)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_cases = [
        TestComputerVisionBot,
        TestWindowsMCPClient,
        TestComputerVisionEngine,
        TestBrokerInterfaceManager,
        TestIntegrationScenarios,
        TestOrchestatorIntegration,
    ]

    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%"
    )

    if result.failures:
        print("\nFAILURES:")
        for test, trace in result.failures:
            print(f"- {test}: {trace}")

    if result.errors:
        print("\nERRORS:")
        for test, trace in result.errors:
            print(f"- {test}: {trace}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
