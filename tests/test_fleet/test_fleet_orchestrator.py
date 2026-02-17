"""
Unit tests for FleetOrchestrator — cycle scheduling and trade pipeline.
"""

import os
import sys
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType
from bots.fleet.shared.fleet_db import FleetDB
from bots.fleet.fleet_orchestrator import FleetOrchestrator


class MockBot(FleetBot):
    """Mock bot for testing orchestrator."""

    def __init__(self, name='Mock-Bot', signals=None):
        config = FleetBotConfig(
            name=name, bot_type=BotType.CRYPTO, schedule_seconds=60,
            max_position_usd=50, max_daily_trades=10, min_confidence=0.3,
        )
        super().__init__(config)
        self._signals = signals or []

    def scan(self):
        return self._signals


class TestFleetOrchestrator(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'test_fleet.db')

    def _make_orchestrator(self):
        orch = FleetOrchestrator(paper_mode=True, fleet_capital=500.0)
        orch.db = FleetDB(db_path=self.db_path)
        orch.risk.db = orch.db
        orch._initialized = True
        return orch

    def test_empty_cycle(self):
        orch = self._make_orchestrator()
        # No bots loaded
        trades = orch.run_cycle()
        self.assertEqual(trades, [])
        self.assertEqual(orch.cycle_count, 1)

    def test_bot_not_due(self):
        orch = self._make_orchestrator()
        bot = MockBot(signals=[FleetSignal(
            bot_name='Mock-Bot', bot_type='crypto', symbol='BTC/USD',
            side='BUY', entry_price=50000, confidence=0.8, position_size_usd=25,
        )])
        bot.last_run = datetime.now(timezone.utc)  # Just ran
        orch.bots['Mock-Bot'] = bot

        trades = orch.run_cycle()
        self.assertEqual(trades, [])

    def test_bot_executes_when_due(self):
        orch = self._make_orchestrator()
        sig = FleetSignal(
            bot_name='Mock-Bot', bot_type='crypto', symbol='BTC/USD',
            side='BUY', entry_price=50000, confidence=0.8, position_size_usd=25,
        )
        bot = MockBot(signals=[sig])
        # Never ran — is due
        orch.bots['Mock-Bot'] = bot

        trades = orch.run_cycle()
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]['symbol'], 'BTC/USD')
        self.assertIn('Fleet:Mock-Bot', trades[0]['bot_name'])

    def test_trade_logged_to_db(self):
        orch = self._make_orchestrator()
        sig = FleetSignal(
            bot_name='Mock-Bot', bot_type='crypto', symbol='ETH/USD',
            side='BUY', entry_price=3000, confidence=0.7, position_size_usd=20,
        )
        bot = MockBot(signals=[sig])
        orch.bots['Mock-Bot'] = bot

        orch.run_cycle()
        positions = orch.db.get_open_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['symbol'], 'ETH/USD')

    def test_risk_blocks_trade(self):
        orch = self._make_orchestrator()
        orch.risk.MAX_OPEN_POSITIONS = 0  # Block everything

        sig = FleetSignal(
            bot_name='Mock-Bot', bot_type='crypto', symbol='BTC/USD',
            side='BUY', entry_price=50000, confidence=0.8, position_size_usd=25,
        )
        bot = MockBot(signals=[sig])
        orch.bots['Mock-Bot'] = bot

        trades = orch.run_cycle()
        self.assertEqual(trades, [])
        self.assertEqual(orch.total_risk_blocked, 1)

    def test_pause_resume_bot(self):
        orch = self._make_orchestrator()
        bot = MockBot()
        orch.bots['Mock-Bot'] = bot

        self.assertTrue(orch.pause_bot('Mock-Bot'))
        self.assertTrue(bot.is_paused)

        self.assertTrue(orch.resume_bot('Mock-Bot'))
        self.assertFalse(bot.is_paused)

    def test_emergency_stop(self):
        orch = self._make_orchestrator()
        orch.bots['Bot-A'] = MockBot('Bot-A')
        orch.bots['Bot-B'] = MockBot('Bot-B')

        orch.emergency_stop()
        self.assertTrue(orch.bots['Bot-A'].is_paused)
        self.assertTrue(orch.bots['Bot-B'].is_paused)

    def test_fleet_status(self):
        orch = self._make_orchestrator()
        orch.bots['Mock-Bot'] = MockBot()

        status = orch.get_fleet_status()
        self.assertIn('bots', status)
        self.assertIn('Mock-Bot', status['bots'])
        self.assertIn('risk', status)
        self.assertEqual(status['paper_mode'], True)


if __name__ == '__main__':
    unittest.main()
