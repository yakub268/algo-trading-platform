"""
Tests for Kalshi Infrastructure Layer

Tests:
- KalshiPositionManager: sync, reconcile, update, query
- KalshiFillTracker: new fill detection, dedup
- KalshiSettlementCollector: P&L calculation, settlement detection
- KalshiRiskManager: daily loss, max positions, contradictory blocker
- KalshiInfrastructure: bundle wiring

Uses a temporary SQLite DB per test (no side effects).
"""

import os
import sys
import tempfile
import sqlite3
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bots.kalshi_infrastructure import (
    KalshiPositionManager,
    KalshiFillTracker,
    KalshiSettlementCollector,
    KalshiRiskManager,
    KalshiInfrastructure,
    init_schema,
    _get_db,
)


@pytest.fixture
def db_path(tmp_path):
    """Create a temp DB path for each test."""
    path = str(tmp_path / "test_kalshi.db")
    init_schema(path)
    return path


@pytest.fixture
def mock_client():
    """Create a mock Kalshi client."""
    client = MagicMock()
    client.get_positions.return_value = []
    client.get_fills.return_value = []
    client.get_orders.return_value = []
    client.get_market.return_value = {'status': 'open', 'title': 'Test Market'}
    client.cancel_order.return_value = {}
    client._initialized = True
    return client


@pytest.fixture
def position_manager(mock_client, db_path):
    return KalshiPositionManager(mock_client, db_path)


@pytest.fixture
def fill_tracker(mock_client, position_manager, db_path):
    return KalshiFillTracker(mock_client, position_manager, db_path)


@pytest.fixture
def settlement_collector(mock_client, position_manager, db_path):
    return KalshiSettlementCollector(mock_client, position_manager, db_path)


@pytest.fixture
def risk_manager(position_manager, db_path):
    return KalshiRiskManager(
        position_manager, db_path,
        daily_loss_limit_cents=2000,
        max_open_positions=3,
        max_contracts_per_market=10,
    )


# =============================================================================
# Position Manager Tests
# =============================================================================

class TestPositionManager:

    def test_empty_sync(self, position_manager):
        """Sync with no API positions."""
        stats = position_manager.sync_from_api()
        assert stats['synced'] == 0
        assert stats['added'] == 0

    def test_sync_adds_positions(self, position_manager, mock_client):
        """Sync adds positions from API."""
        mock_client.get_positions.return_value = [
            {'ticker': 'KXFED-26MAR-CUT', 'yes_count': 5, 'no_count': 0, 'average_price': 42},
        ]
        stats = position_manager.sync_from_api()
        assert stats['added'] == 1
        assert stats['synced'] == 1

        pos = position_manager.get_all_positions()
        assert len(pos) == 1
        assert pos[0]['ticker'] == 'KXFED-26MAR-CUT'
        assert pos[0]['side'] == 'yes'
        assert pos[0]['quantity'] == 5

    def test_sync_removes_closed(self, position_manager, mock_client, db_path):
        """Positions not in API get marked closed."""
        # First add a position
        position_manager.update_position('KXFED-26MAR-CUT', 'yes', 5, 42)
        assert position_manager.get_open_position_count() == 1

        # Sync with empty API — should close the position
        mock_client.get_positions.return_value = []
        stats = position_manager.sync_from_api()
        assert stats['removed'] == 1
        assert position_manager.get_open_position_count() == 0

    def test_update_position_buy(self, position_manager):
        """Buying increases position."""
        position_manager.update_position('KXBTC-26FEB-B100K', 'yes', 3, 45)
        pos = position_manager.get_position('KXBTC-26FEB-B100K', 'yes')
        assert pos is not None
        assert pos['quantity'] == 3
        assert pos['avg_price_cents'] == 45

    def test_update_position_sell_closes(self, position_manager):
        """Selling all contracts closes position."""
        position_manager.update_position('KXBTC-26FEB-B100K', 'yes', 5, 45)
        position_manager.update_position('KXBTC-26FEB-B100K', 'yes', -5, 50)
        pos = position_manager.get_position('KXBTC-26FEB-B100K', 'yes')
        assert pos is None  # Closed

    def test_weighted_avg_price(self, position_manager):
        """Weighted average price on additional buys."""
        position_manager.update_position('KXFED-26MAR-CUT', 'yes', 2, 40)
        position_manager.update_position('KXFED-26MAR-CUT', 'yes', 2, 60)
        pos = position_manager.get_position('KXFED-26MAR-CUT', 'yes')
        # (2*40 + 2*60) / 4 = 50
        assert pos['avg_price_cents'] == 50

    def test_reconcile_clean(self, position_manager, mock_client):
        """Reconciliation with matching state returns no discrepancies."""
        mock_client.get_positions.return_value = []
        discrepancies = position_manager.reconcile()
        assert discrepancies == []


# =============================================================================
# Fill Tracker Tests
# =============================================================================

class TestFillTracker:

    def test_new_fill_recorded(self, fill_tracker, mock_client, db_path):
        """New fills get recorded to DB."""
        mock_client.get_fills.return_value = [
            {
                'trade_id': 'fill_001',
                'ticker': 'KXFED-26MAR-CUT',
                'side': 'yes',
                'action': 'buy',
                'yes_price': 42,
                'count': 3,
                'created_time': '2026-02-15T12:00:00Z',
            }
        ]

        new = fill_tracker.poll_fills('test-bot')
        assert len(new) == 1
        assert new[0]['fill_id'] == 'fill_001'
        assert new[0]['count'] == 3

    def test_duplicate_fill_skipped(self, fill_tracker, mock_client):
        """Same fill_id is not recorded twice."""
        fill_data = [{
            'trade_id': 'fill_001',
            'ticker': 'KXFED-26MAR-CUT',
            'side': 'yes',
            'action': 'buy',
            'yes_price': 42,
            'count': 3,
        }]
        mock_client.get_fills.return_value = fill_data

        fills1 = fill_tracker.poll_fills()
        fills2 = fill_tracker.poll_fills()
        assert len(fills1) == 1
        assert len(fills2) == 0  # Deduped

    def test_fill_updates_position(self, fill_tracker, mock_client, position_manager):
        """Fills automatically update position manager."""
        mock_client.get_fills.return_value = [{
            'trade_id': 'fill_002',
            'ticker': 'KXBTC-26FEB-B100K',
            'side': 'yes',
            'action': 'buy',
            'yes_price': 55,
            'count': 2,
        }]

        fill_tracker.poll_fills()
        pos = position_manager.get_position('KXBTC-26FEB-B100K', 'yes')
        assert pos is not None
        assert pos['quantity'] == 2


# =============================================================================
# Settlement Collector Tests
# =============================================================================

class TestSettlementCollector:

    def test_pnl_win(self, settlement_collector):
        """Winning position P&L calculation."""
        pnl = settlement_collector._calculate_pnl('yes', 'yes', 5, 40)
        # Won: (100-40)*5 = 300 cents
        assert pnl == 300

    def test_pnl_loss(self, settlement_collector):
        """Losing position P&L calculation."""
        pnl = settlement_collector._calculate_pnl('no', 'yes', 5, 40)
        # Lost: -40*5 = -200 cents
        assert pnl == -200

    def test_pnl_voided(self, settlement_collector):
        """Voided market returns 0 P&L."""
        pnl = settlement_collector._calculate_pnl('voided', 'yes', 5, 40)
        assert pnl == 0

    def test_collect_with_settled_market(self, settlement_collector, mock_client, position_manager, db_path):
        """Settlement collector records settled markets."""
        # Add a position first
        position_manager.update_position('KXFED-26MAR-CUT', 'yes', 3, 42)

        # Mock API returns settled market
        mock_client.get_market.return_value = {
            'status': 'settled',
            'result': 'yes',
            'title': 'Fed holds rates',
            'close_time': '2026-03-19T18:00:00Z',
        }

        stats = settlement_collector.collect()
        assert stats['checked'] == 1
        assert stats['settled'] == 1

    def test_settlement_stats(self, settlement_collector, db_path):
        """Settlement stats calculation."""
        stats = settlement_collector.get_settlement_stats()
        assert stats['total_settled'] == 0
        assert stats['total_pnl_dollars'] == 0


# =============================================================================
# Risk Manager Tests
# =============================================================================

class TestRiskManager:

    def test_trade_allowed_clean(self, risk_manager):
        """Trade allowed when no limits hit."""
        allowed, reason = risk_manager.check_trade_allowed('KXFED-26MAR-CUT', 'yes', 3)
        assert allowed is True

    def test_daily_loss_blocks(self, risk_manager, db_path):
        """Trade blocked when daily loss limit hit."""
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        conn = _get_db(db_path)
        conn.execute(
            "INSERT INTO daily_pnl (date, realized_pnl_cents, num_fills, num_settlements) VALUES (?, ?, 5, 0)",
            (today, -2500)  # -$25, over $20 limit
        )
        conn.commit()
        conn.close()

        allowed, reason = risk_manager.check_trade_allowed('KXFED-26MAR-CUT', 'yes', 3)
        assert allowed is False
        assert 'loss limit' in reason.lower()

    def test_max_positions_blocks(self, risk_manager, position_manager):
        """Trade blocked when max positions reached."""
        # Add 3 positions (our limit)
        for i in range(3):
            position_manager.update_position(f'TICKER-{i}', 'yes', 1, 50)

        allowed, reason = risk_manager.check_trade_allowed('NEW-TICKER', 'yes', 1)
        assert allowed is False
        assert 'max open positions' in reason.lower()

    def test_per_market_limit(self, risk_manager, position_manager):
        """Trade blocked when per-market contract limit exceeded."""
        position_manager.update_position('KXFED-26MAR-CUT', 'yes', 8, 42)

        # 8 existing + 5 new = 13, over 10 limit
        allowed, reason = risk_manager.check_trade_allowed('KXFED-26MAR-CUT', 'yes', 5)
        assert allowed is False
        assert 'per-market' in reason.lower()

    def test_contradictory_position_blocked(self, risk_manager, position_manager):
        """Trade blocked when it would create contradictory position."""
        position_manager.update_position('KXHIGHCHI-26FEB09-B38.5', 'yes', 3, 55)

        # Different threshold on same underlying should be blocked
        allowed, reason = risk_manager.check_trade_allowed('KXHIGHCHI-26FEB09-B40.5', 'yes', 2)
        assert allowed is False
        assert 'conflict' in reason.lower()

    def test_extract_underlying(self, risk_manager):
        """Underlying extraction from tickers."""
        assert risk_manager._extract_underlying('KXHIGHCHI-26FEB09-B38.5') == 'KXHIGHCHI-26FEB09'
        assert risk_manager._extract_underlying('KXFED-26MAR19-H0') == 'KXFED-26MAR19'

    def test_pause_resume(self, risk_manager):
        """Pause/resume blocks/unblocks trading."""
        risk_manager.pause()
        allowed, _ = risk_manager.check_trade_allowed('KXFED-26MAR-CUT', 'yes', 1)
        assert allowed is False

        risk_manager.resume()
        allowed, _ = risk_manager.check_trade_allowed('KXFED-26MAR-CUT', 'yes', 1)
        assert allowed is True


# =============================================================================
# Infrastructure Bundle Tests
# =============================================================================

class TestKalshiInfrastructure:

    def test_bundle_init(self, mock_client, db_path):
        """Infrastructure bundle creates all components."""
        infra = KalshiInfrastructure(mock_client, db_path)
        assert infra.position_manager is not None
        assert infra.fill_tracker is not None
        assert infra.settlement_collector is not None
        assert infra.risk_manager is not None

    def test_startup_reconciliation(self, mock_client, db_path):
        """Startup reconciliation runs without error."""
        mock_client.get_positions.return_value = []
        infra = KalshiInfrastructure(mock_client, db_path)
        infra.startup_reconciliation()  # Should not raise

    def test_get_status(self, mock_client, db_path):
        """Status returns combined metrics."""
        infra = KalshiInfrastructure(mock_client, db_path)
        status = infra.get_status()
        assert 'open_positions' in status
        assert 'daily_pnl_dollars' in status
        assert 'total_settled' in status
        assert 'fills_today' in status

    def test_cancel_all_orders(self, mock_client, db_path):
        """Cancel all orders calls API correctly."""
        mock_client.get_orders.return_value = [
            {'order_id': 'ord_001'},
            {'order_id': 'ord_002'},
        ]

        infra = KalshiInfrastructure(mock_client, db_path)
        infra.cancel_all_orders()
        assert mock_client.cancel_order.call_count == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
