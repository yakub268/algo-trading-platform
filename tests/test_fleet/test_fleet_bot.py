"""
Unit tests for FleetBot base class, FleetSignal, and FleetBotConfig.
"""

import os
import sys
import unittest
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType


class StubBot(FleetBot):
    """Concrete implementation for testing."""

    def __init__(self, config=None, signals=None):
        if config is None:
            config = FleetBotConfig(
                name='Test-Bot', bot_type=BotType.CRYPTO, schedule_seconds=300,
                max_position_usd=50, max_daily_trades=5, min_confidence=0.6,
            )
        super().__init__(config)
        self._signals = signals or []

    def scan(self):
        return self._signals


class TestFleetBotConfig(unittest.TestCase):
    def test_defaults(self):
        config = FleetBotConfig(name='Test', bot_type=BotType.CRYPTO, schedule_seconds=60)
        self.assertEqual(config.max_position_usd, 50.0)
        self.assertEqual(config.max_daily_trades, 10)
        self.assertEqual(config.min_confidence, 0.6)
        self.assertTrue(config.enabled)
        self.assertTrue(config.paper_mode)


class TestFleetSignal(unittest.TestCase):
    def test_to_dict(self):
        sig = FleetSignal(
            bot_name='Test', bot_type='crypto', symbol='BTC/USD',
            side='BUY', entry_price=50000, confidence=0.8, edge=0.05,
            reason='test signal',
        )
        d = sig.to_dict()
        self.assertEqual(d['bot_name'], 'Test')
        self.assertEqual(d['symbol'], 'BTC/USD')
        self.assertEqual(d['side'], 'BUY')
        self.assertIn('trade_id', d)
        self.assertIn('timestamp', d)

    def test_trade_id_unique(self):
        s1 = FleetSignal(bot_name='A', bot_type='crypto', symbol='X', side='BUY', entry_price=1)
        s2 = FleetSignal(bot_name='A', bot_type='crypto', symbol='X', side='BUY', entry_price=1)
        self.assertNotEqual(s1.trade_id, s2.trade_id)


class TestFleetBot(unittest.TestCase):
    def test_is_due_first_run(self):
        bot = StubBot()
        self.assertTrue(bot.is_due())

    def test_is_due_after_run(self):
        bot = StubBot()
        bot.last_run = datetime.now(timezone.utc)
        self.assertFalse(bot.is_due())  # Just ran, not due

    def test_is_due_after_schedule(self):
        bot = StubBot()
        bot.last_run = datetime.now(timezone.utc) - timedelta(seconds=400)
        self.assertTrue(bot.is_due())  # 400s > 300s schedule

    def test_is_due_paused(self):
        bot = StubBot()
        bot.is_paused = True
        self.assertFalse(bot.is_due())

    def test_pre_scan_daily_limit(self):
        bot = StubBot()
        bot._last_reset_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')  # Prevent reset
        bot.trades_today = 5  # At limit
        self.assertFalse(bot.pre_scan_checks())

    def test_filter_low_confidence(self):
        bot = StubBot()
        signals = [FleetSignal(
            bot_name='Test', bot_type='crypto', symbol='BTC/USD',
            side='BUY', entry_price=50000, confidence=0.3,  # Below 0.6 min
        )]
        filtered = bot.filter_signals(signals)
        self.assertEqual(len(filtered), 0)

    def test_filter_passes_good_signal(self):
        bot = StubBot()
        signals = [FleetSignal(
            bot_name='Test', bot_type='crypto', symbol='BTC/USD',
            side='BUY', entry_price=50000, confidence=0.8,
            position_size_usd=30,
        )]
        filtered = bot.filter_signals(signals)
        self.assertEqual(len(filtered), 1)

    def test_filter_caps_position_size(self):
        bot = StubBot()
        signals = [FleetSignal(
            bot_name='Test', bot_type='crypto', symbol='BTC/USD',
            side='BUY', entry_price=50000, confidence=0.8,
            position_size_usd=100,  # Over $50 cap
        )]
        filtered = bot.filter_signals(signals)
        self.assertEqual(filtered[0].position_size_usd, 50)  # Capped

    def test_filter_cooldown(self):
        bot = StubBot()
        bot._traded_symbols['BTC/USD'] = datetime.now(timezone.utc)  # Just traded
        signals = [FleetSignal(
            bot_name='Test', bot_type='crypto', symbol='BTC/USD',
            side='BUY', entry_price=50000, confidence=0.8,
            position_size_usd=30,
        )]
        filtered = bot.filter_signals(signals)
        self.assertEqual(len(filtered), 0)

    def test_consecutive_loss_reduction(self):
        bot = StubBot()
        bot.consecutive_losses = 5
        signals = [FleetSignal(
            bot_name='Test', bot_type='crypto', symbol='BTC/USD',
            side='BUY', entry_price=50000, confidence=0.8,
            position_size_usd=40,
        )]
        filtered = bot.filter_signals(signals)
        self.assertEqual(filtered[0].position_size_usd, 20)  # Halved

    def test_record_win_resets_losses(self):
        bot = StubBot()
        bot.consecutive_losses = 5
        bot.record_win()
        self.assertEqual(bot.consecutive_losses, 0)

    def test_record_loss_cooldown(self):
        bot = StubBot()
        for _ in range(10):
            bot.record_loss()
        self.assertEqual(bot.consecutive_losses, 10)
        self.assertIsNotNone(bot._cooldown_until)

    def test_get_status(self):
        bot = StubBot()
        status = bot.get_status()
        self.assertEqual(status['name'], 'Test-Bot')
        self.assertEqual(status['bot_type'], 'crypto')
        self.assertTrue(status['paper_mode'])


if __name__ == '__main__':
    unittest.main()
