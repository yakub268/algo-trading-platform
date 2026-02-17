"""
Integration Tests for Master Orchestrator
=========================================
Test the master orchestrator integration with all AI modules.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


class TestMasterOrchestratorIntegration:
    """Test master orchestrator with multiple bots"""

    @pytest.fixture
    def orchestrator(self, mock_env_vars, mock_db):
        """Create orchestrator with all __init__ side effects patched out"""
        from master_orchestrator import (
            MasterOrchestrator,
        )

        with patch("master_orchestrator.TradingDB") as mock_db_class, patch.object(
            MasterOrchestrator, "_init_alerts"
        ), patch.object(MasterOrchestrator, "_init_risk_management"), patch.object(
            MasterOrchestrator, "_init_telegram_command_listener"
        ), patch.object(
            MasterOrchestrator, "_init_bots"
        ), patch.object(
            MasterOrchestrator, "_init_ai_filter"
        ), patch.object(
            MasterOrchestrator, "_load_high_risk_config", return_value=False
        ), patch.object(
            MasterOrchestrator, "_validate_db_file"
        ):

            mock_db_instance = Mock()
            mock_db_class.return_value = mock_db_instance
            mock_db_instance.log_trade = Mock()
            mock_db_instance.update_bot_status = Mock()
            mock_db_instance.get_today_summary.return_value = {
                "trades": 5,
                "wins": 3,
                "total_pnl": 150.0,
            }
            mock_db_instance.get_open_trades.return_value = []

            orch = MasterOrchestrator(
                starting_capital=10000.0,
                paper_mode=True,
                use_do_nothing_filter=False,
                enable_risk_management=False,
            )

            # Ensure bots dict is empty (we inject what we need per test)
            orch.bots = {}
            orch.ai_filter = None
            orch.alerts = None
            orch.risk_integration = None
            orch.telegram_command_listener = None
            orch.high_risk_enabled = False

            return orch

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes with correct properties"""
        assert orchestrator.starting_capital == 10000.0
        assert orchestrator.paper_mode is True
        assert orchestrator.current_capital == 10000.0

    def test_orchestrator_bots_dict_empty(self, orchestrator):
        """Test orchestrator starts with empty bots dict (we patched _init_bots)"""
        assert len(orchestrator.bots) == 0

    def test_run_single_bot(self, orchestrator):
        """Test running a single bot via _run_bot"""
        from master_orchestrator import BotState, BotConfig, Market, BotStatus

        mock_bot = Mock()
        mock_bot.run_strategy.return_value = {
            "status": "success",
            "action": "buy",
            "symbol": "FED-RAISE",
            "price": 0.45,
            "quantity": 10,
            "confidence": 0.75,
        }

        config = BotConfig(
            name="MockBot",
            module_path="mock.module",
            class_name="MockBot",
            market=Market.PREDICTION,
            enabled=True,
        )

        orchestrator.bots["MockBot"] = BotState(
            config=config, instance=mock_bot, status=BotStatus.WAITING
        )

        result = orchestrator._run_bot("MockBot")
        assert result is not None
        mock_bot.run_strategy.assert_called_once()

    def test_run_bot_not_found(self, orchestrator):
        """Test _run_bot returns None for non-existent bot"""
        result = orchestrator._run_bot("NonExistentBot")
        assert result is None

    def test_run_bot_error_state(self, orchestrator):
        """Test _run_bot returns error for bot in ERROR state"""
        from master_orchestrator import BotState, BotConfig, Market, BotStatus

        config = BotConfig(
            name="ErrorBot",
            module_path="mock",
            class_name="Mock",
            market=Market.CRYPTO,
            enabled=True,
        )
        orchestrator.bots["ErrorBot"] = BotState(
            config=config, instance=None, status=BotStatus.ERROR, error="Previous crash"
        )

        result = orchestrator._run_bot("ErrorBot")
        assert result is not None
        assert result["status"] == "error"

    def test_run_bot_exception_handling(self, orchestrator):
        """Test _run_bot handles exceptions from bot methods"""
        from master_orchestrator import BotState, BotConfig, Market, BotStatus

        mock_bot = Mock()
        mock_bot.run_strategy.side_effect = Exception("Bot crashed!")

        config = BotConfig(
            name="CrashBot",
            module_path="mock",
            class_name="Mock",
            market=Market.PREDICTION,
            enabled=True,
        )
        orchestrator.bots["CrashBot"] = BotState(
            config=config, instance=mock_bot, status=BotStatus.WAITING
        )

        result = orchestrator._run_bot("CrashBot")
        # Should not raise, should return error dict or handle gracefully
        assert result is not None

    def test_run_all_once(self, orchestrator):
        """Test running multiple bots via run_all_once"""
        from master_orchestrator import BotState, BotConfig, Market, BotStatus

        for i in range(3):
            mock_bot = Mock()
            mock_bot.run_strategy.return_value = {"status": "success", "bot_id": i}

            config = BotConfig(
                name=f"MockBot{i}",
                module_path="mock.module",
                class_name="MockBot",
                market=Market.PREDICTION,
                enabled=True,
            )

            orchestrator.bots[f"MockBot{i}"] = BotState(
                config=config, instance=mock_bot, status=BotStatus.WAITING
            )

        # Patch check_do_nothing_filter to return False
        with patch.object(orchestrator, "check_do_nothing_filter", return_value=False):
            results = orchestrator.run_all_once()

        assert len(results) == 3
        for name in results:
            assert results[name] is not None

    def test_status_reporting(self, orchestrator):
        """Test comprehensive status reporting"""
        from master_orchestrator import BotState, BotConfig, Market, BotStatus

        bot_configs = [
            ("RunningBot", BotStatus.RUNNING, Market.PREDICTION),
            ("WaitingBot", BotStatus.WAITING, Market.STOCKS),
            ("ErrorBot", BotStatus.ERROR, Market.CRYPTO),
        ]

        for name, status, market in bot_configs:
            config = BotConfig(
                name=name,
                module_path="mock",
                class_name="MockBot",
                market=market,
                enabled=True,
            )
            orchestrator.bots[name] = BotState(
                config=config,
                instance=Mock(),
                status=status,
                trades_today=2 if status == BotStatus.RUNNING else 0,
                pnl_today=100.0 if status == BotStatus.RUNNING else 0.0,
            )

        status = orchestrator.get_status()

        assert status["total_bots"] == 3
        assert status["paper_mode"] is True
        assert status["starting_capital"] == 10000.0
        assert "by_market" in status
        assert "prediction" in status["by_market"]
        assert "stocks" in status["by_market"]
        assert "crypto" in status["by_market"]

    def test_capital_tracking(self, orchestrator):
        """Test capital tracking and allocation"""
        initial_capital = orchestrator.current_capital

        orchestrator.current_capital += 500
        assert orchestrator.current_capital == initial_capital + 500

        orchestrator.current_capital -= 200
        assert orchestrator.current_capital == initial_capital + 300

    def test_schedule_setup(self, orchestrator):
        """Test schedule setup for different bot types"""
        from master_orchestrator import BotState, BotConfig, Market, BotStatus
        import schedule as sched_module

        sched_module.clear()

        configs = [
            BotConfig(
                name="IntervalBot",
                module_path="mock",
                class_name="MockBot",
                market=Market.PREDICTION,
                enabled=True,
                schedule_type="interval",
                schedule_value=300,
            ),
            BotConfig(
                name="DailyBot",
                module_path="mock",
                class_name="MockBot",
                market=Market.STOCKS,
                enabled=True,
                schedule_type="daily",
                schedule_value="09:30",
            ),
        ]

        for config in configs:
            orchestrator.bots[config.name] = BotState(
                config=config, instance=Mock(), status=BotStatus.WAITING
            )

        # Patch the extra scheduled methods so they don't break
        with patch.object(orchestrator, "_check_momentum_exits"), patch.object(
            orchestrator, "_force_close_all_stale_positions"
        ), patch.object(orchestrator, "_check_drawdown_emergency_close"), patch.object(
            orchestrator, "_reconcile_exchange_positions"
        ), patch.object(
            orchestrator, "_retry_close_pending_trades"
        ), patch.object(
            orchestrator, "_daily_reconciliation_report"
        ):
            orchestrator.setup_schedule()

        assert len(sched_module.jobs) > 0
        sched_module.clear()

    def test_bot_state_dataclass(self):
        """Test BotState and BotConfig dataclass creation"""
        from master_orchestrator import BotState, BotConfig, Market, BotStatus

        config = BotConfig(
            name="TestBot",
            module_path="test.module",
            class_name="TestBot",
            market=Market.PREDICTION,
            enabled=True,
            schedule_type="interval",
            schedule_value=300,
            allocation_pct=5.0,
            description="Test bot",
        )

        state = BotState(
            config=config,
            instance=Mock(),
            status=BotStatus.WAITING,
            trades_today=0,
            pnl_today=0.0,
        )

        assert state.config.name == "TestBot"
        assert state.config.market == Market.PREDICTION
        assert state.status == BotStatus.WAITING

    def test_bot_restart_after_failure(self, orchestrator):
        """Test that failed bots can be reset and retried"""
        from master_orchestrator import BotState, BotConfig, Market, BotStatus

        mock_bot = Mock()
        mock_bot.run_strategy.side_effect = [
            Exception("First failure"),
            {"status": "success", "action": "buy"},
        ]

        config = BotConfig(
            name="RestartBot",
            module_path="mock",
            class_name="Mock",
            market=Market.PREDICTION,
            enabled=True,
        )
        orchestrator.bots["RestartBot"] = BotState(
            config=config, instance=mock_bot, status=BotStatus.WAITING
        )

        # First run fails
        result1 = orchestrator._run_bot("RestartBot")
        assert result1 is not None

        # Reset status to allow retry
        orchestrator.bots["RestartBot"].status = BotStatus.WAITING

        # Second run succeeds
        result2 = orchestrator._run_bot("RestartBot")
        assert result2 is not None


@pytest.mark.skip(
    reason="Depends on orchestrator fixture — see TestMasterOrchestratorIntegration skip reason"
)
class TestOrchestratorScheduling:
    """Test orchestrator scheduling functionality — covered in TestMasterOrchestratorIntegration.test_schedule_setup"""


@pytest.mark.skip(
    reason="psutil dependency and tests covered by TestMasterOrchestratorIntegration"
)
class TestOrchestratorPerformance:
    """Test orchestrator performance — requires psutil"""


@pytest.mark.skip(
    reason="Tests requiring deep internal wiring (DB failure, Telegram alerts) — too coupled to internals"
)
class TestOrchestratorRecovery:
    """Test orchestrator error recovery"""
