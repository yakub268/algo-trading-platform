"""
Unit tests for FleetRisk — fleet-level risk controls.
"""

import os
import sys
import unittest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bots.fleet.shared.fleet_bot import FleetSignal, BotType
from bots.fleet.shared.fleet_db import FleetDB
from bots.fleet.shared.fleet_risk import FleetRisk


class TestFleetRisk(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'test_fleet.db')
        self.db = FleetDB(db_path=self.db_path)
        self.risk = FleetRisk(self.db, fleet_capital=500.0)

    def _make_signal(self, symbol='BTC/USD', side='BUY', bot_type='crypto', size=25.0):
        return FleetSignal(
            bot_name='Test-Bot', bot_type=bot_type, symbol=symbol,
            side=side, entry_price=50000, position_size_usd=size,
            confidence=0.8,
        )

    def test_allows_normal_trade(self):
        sig = self._make_signal()
        allowed, reason = self.risk.check_trade(sig)
        self.assertTrue(allowed)
        self.assertEqual(reason, "OK")

    def test_blocks_daily_loss_limit(self):
        # Simulate $30+ in losses today
        for i in range(4):
            sig = self._make_signal(symbol=f'SYM{i}')
            self.db.log_trade(sig.to_dict())
            self.db.close_trade(sig.trade_id, 49000, -10.0, -0.20)

        new_sig = self._make_signal(symbol='NEW')
        allowed, reason = self.risk.check_trade(new_sig)
        self.assertFalse(allowed)
        self.assertIn("Daily loss limit", reason)

    def test_blocks_max_open_positions(self):
        self.risk.MAX_OPEN_POSITIONS = 3
        for i in range(3):
            sig = self._make_signal(symbol=f'SYM{i}')
            self.db.log_trade(sig.to_dict())

        new_sig = self._make_signal(symbol='NEW')
        allowed, reason = self.risk.check_trade(new_sig)
        self.assertFalse(allowed)
        self.assertIn("Max open positions", reason)

    def test_blocks_exposure_limit(self):
        self.risk.MAX_TOTAL_EXPOSURE_PCT = 0.10  # Only $50 exposure
        sig = self._make_signal(size=40)
        self.db.log_trade(sig.to_dict())

        new_sig = self._make_signal(symbol='ETH/USD', size=20)
        allowed, reason = self.risk.check_trade(new_sig)
        self.assertFalse(allowed)
        self.assertIn("Exposure limit", reason)

    def test_blocks_contradictory_positions(self):
        # Open a BUY position
        buy = self._make_signal(symbol='KXHIGHCHI-26FEB09-B38.5', side='YES', bot_type='kalshi')
        self.db.log_trade(buy.to_dict())

        # Try to open opposite side on same underlying
        sell = self._make_signal(symbol='KXHIGHCHI-26FEB09-B40.5', side='NO', bot_type='kalshi')
        allowed, reason = self.risk.check_trade(sell)
        self.assertFalse(allowed)
        self.assertIn("Contradictory", reason)

    def test_blocks_duplicate_position(self):
        sig = self._make_signal()
        self.db.log_trade(sig.to_dict())

        dup = self._make_signal()
        allowed, reason = self.risk.check_trade(dup)
        self.assertFalse(allowed)
        self.assertIn("Already has open", reason)

    def test_consecutive_loss_cooldown(self):
        from datetime import datetime, timezone
        self.risk._last_reset_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')  # Prevent reset
        self.risk.COOLDOWN_AFTER_LOSSES = 2
        self.risk.record_trade_result(False)
        self.risk.record_trade_result(False)

        sig = self._make_signal()
        allowed, reason = self.risk.check_trade(sig)
        self.assertFalse(allowed)
        self.assertIn("cooldown", reason)

    def test_extract_underlying_kalshi(self):
        self.assertEqual(
            FleetRisk._extract_underlying('KXHIGHCHI-26FEB09-B38.5'),
            'KXHIGHCHI-26FEB09'
        )

    def test_extract_underlying_crypto(self):
        self.assertEqual(FleetRisk._extract_underlying('BTC/USD'), 'BTC/USD')


if __name__ == '__main__':
    unittest.main()
