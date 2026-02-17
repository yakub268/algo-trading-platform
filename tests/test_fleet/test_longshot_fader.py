"""
Unit tests for Longshot Fader — highest-conviction Kalshi strategy.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bots.fleet.shared.fleet_bot import FleetSignal, BotType


class TestLongshotFader(unittest.TestCase):

    def _get_bot(self):
        """Import and instantiate, skipping if KalshiClient unavailable."""
        try:
            from bots.fleet.kalshi.longshot_fader import LongshotFader
            return LongshotFader()
        except Exception:
            self.skipTest("LongshotFader import failed (missing deps)")

    def test_scan_returns_list(self):
        """Test that scan returns a list (may be empty without live API)."""
        bot = self._get_bot()
        # Without a live Kalshi API connection, scan should return empty list
        signals = bot.scan()
        self.assertIsInstance(signals, list)

    def test_config_defaults(self):
        """Test default configuration."""
        bot = self._get_bot()
        self.assertEqual(bot.name, 'Longshot-Fader')
        self.assertEqual(bot.bot_type, BotType.KALSHI)
        self.assertTrue(bot.paper_mode)


if __name__ == '__main__':
    unittest.main()
