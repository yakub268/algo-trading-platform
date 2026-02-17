"""
Fleet Orchestrator — Sub-bot lifecycle management, scheduling, and trade execution.

This is the brain of the fleet system. It:
1. Maintains all 19 sub-bot instances
2. Checks which bots are DUE each cycle (timestamp-based scheduling)
3. Calls bot.scan() → List[FleetSignal] for due bots
4. Applies Thompson Sampling allocation (from ConfidenceVoter)
5. Runs fleet-level risk checks (FleetRisk)
6. Routes to broker (BrokerRouter)
7. Logs everything to fleet_trades.db (FleetDB)

Called by FleetAdapter.run_scan() every 60 seconds from master_orchestrator.
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType
from bots.fleet.shared.fleet_db import FleetDB
from bots.fleet.shared.fleet_risk import FleetRisk
from bots.fleet.shared.broker_router import BrokerRouter

logger = logging.getLogger('Fleet.Orchestrator')

# Log directory
LOG_DIR = os.path.join(_PROJECT_ROOT, 'logs', 'fleet')
os.makedirs(LOG_DIR, exist_ok=True)


def _safe_import_bot(module_path: str, class_name: str):
    """Safely import a bot class, returning None on failure."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except Exception as e:
        logger.warning(f"Failed to import {module_path}.{class_name}: {e}")
        return None


# Bot registry: (module_path, class_name, default_config)
FLEET_BOT_REGISTRY = [
    # Kalshi bots
    ('bots.fleet.kalshi.longshot_fader', 'LongshotFader', FleetBotConfig(
        name='Longshot-Fader', bot_type=BotType.KALSHI, schedule_seconds=300,
        max_position_usd=15, max_daily_trades=20, min_confidence=0.55,
    )),
    ('bots.fleet.kalshi.weather_prophet', 'WeatherProphet', FleetBotConfig(
        name='Weather-Prophet', bot_type=BotType.KALSHI, schedule_seconds=1800,
        max_position_usd=30, max_daily_trades=5, min_confidence=0.6,
    )),
    ('bots.fleet.kalshi.econ_event_trader', 'EconEventTrader', FleetBotConfig(
        name='Econ-Event-Trader', bot_type=BotType.KALSHI, schedule_seconds=3600,
        max_position_usd=40, max_daily_trades=5, min_confidence=0.6,
    )),
    ('bots.fleet.kalshi.tail_risk_hedger', 'TailRiskHedger', FleetBotConfig(
        name='Tail-Risk-Hedger', bot_type=BotType.KALSHI, schedule_seconds=86400,
        max_position_usd=5, max_daily_trades=3, min_confidence=0.3,
    )),

    # Simple crypto bots
    ('bots.fleet.crypto.dca_accumulator', 'DCAAccumulator', FleetBotConfig(
        name='DCA-Accumulator', bot_type=BotType.CRYPTO, schedule_seconds=900,
        max_position_usd=50, max_daily_trades=20, min_confidence=0.3,
    )),
    ('bots.fleet.crypto.mean_reversion_sniper', 'MeanReversionSniper', FleetBotConfig(
        name='Mean-Reversion-Sniper', bot_type=BotType.CRYPTO, schedule_seconds=300,
        max_position_usd=50, max_daily_trades=10, min_confidence=0.6,
    )),
    ('bots.fleet.crypto.sentiment_surfer', 'SentimentSurfer', FleetBotConfig(
        name='Sentiment-Surfer', bot_type=BotType.CRYPTO, schedule_seconds=3600,
        max_position_usd=40, max_daily_trades=5, min_confidence=0.5,
    )),
    ('bots.fleet.crypto.momentum_weekly', 'MomentumWeekly', FleetBotConfig(
        name='Momentum-Weekly', bot_type=BotType.CRYPTO, schedule_seconds=604800,
        max_position_usd=50, max_daily_trades=3, min_confidence=0.6,
    )),

    # Complex crypto bots
    ('bots.fleet.crypto.volatility_breakout', 'VolatilityBreakoutBot', FleetBotConfig(
        name='Volatility-Breakout', bot_type=BotType.CRYPTO, schedule_seconds=300,
        max_position_usd=40, max_daily_trades=8, min_confidence=0.6,
    )),
    ('bots.fleet.crypto.grid_trader', 'GridTraderBot', FleetBotConfig(
        name='Grid-Trader', bot_type=BotType.CRYPTO, schedule_seconds=60,
        max_position_usd=10, max_daily_trades=50, min_confidence=0.3,
    )),
    ('bots.fleet.crypto.pairs_trader', 'PairsTraderBot', FleetBotConfig(
        name='Pairs-Trader', bot_type=BotType.CRYPTO, schedule_seconds=300,
        max_position_usd=40, max_daily_trades=10, min_confidence=0.6,
    )),
    ('bots.fleet.crypto.funding_rate_harvester', 'FundingRateHarvesterBot', FleetBotConfig(
        name='Funding-Rate-Harvester', bot_type=BotType.CRYPTO, schedule_seconds=300,
        max_position_usd=35, max_daily_trades=8, min_confidence=0.5,
    )),
    ('bots.fleet.crypto.fomc_drift_trader', 'FOMCDriftTraderBot', FleetBotConfig(
        name='FOMC-Drift-Trader', bot_type=BotType.CRYPTO, schedule_seconds=86400,
        max_position_usd=50, max_daily_trades=2, min_confidence=0.7,
    )),
    ('bots.fleet.crypto.ml_ensemble', 'MLEnsembleBot', FleetBotConfig(
        name='ML-Ensemble', bot_type=BotType.CRYPTO, schedule_seconds=600,
        max_position_usd=35, max_daily_trades=10, min_confidence=0.6,
    )),

    # Prediction bots
    ('bots.fleet.prediction.sports_edge_enhanced', 'SportsEdgeEnhancedBot', FleetBotConfig(
        name='Sports-Edge-Enhanced', bot_type=BotType.PREDICTION, schedule_seconds=900,
        max_position_usd=25, max_daily_trades=8, min_confidence=0.55,
    )),
    ('bots.fleet.prediction.entertainment_oracle', 'EntertainmentOracleBot', FleetBotConfig(
        name='Entertainment-Oracle', bot_type=BotType.PREDICTION, schedule_seconds=86400,
        max_position_usd=8, max_daily_trades=3, min_confidence=0.5,
    )),

    # Forex
    ('bots.fleet.forex.news_momentum_fx', 'NewsMomentumFXBot', FleetBotConfig(
        name='News-Momentum-FX', bot_type=BotType.FOREX, schedule_seconds=300,
        max_position_usd=30, max_daily_trades=5, min_confidence=0.6,
    )),

    # Meta-controllers
    ('bots.fleet.meta.confidence_voter', 'ConfidenceVoter', FleetBotConfig(
        name='Confidence-Voter', bot_type=BotType.META, schedule_seconds=300,
        max_position_usd=0, max_daily_trades=100, min_confidence=0.0,
    )),
    ('bots.fleet.meta.cross_asset_momentum', 'CrossAssetMomentum', FleetBotConfig(
        name='Cross-Asset-Momentum', bot_type=BotType.META, schedule_seconds=1800,
        max_position_usd=0, max_daily_trades=50, min_confidence=0.0,
    )),
]


class FleetOrchestrator:
    """
    Manages all fleet sub-bots. Called by FleetAdapter every 60 seconds.

    Each cycle:
    1. Check which bots are DUE (per their schedule)
    2. Call scan() on due bots
    3. Apply bot-level filters
    4. Apply Thompson Sampling position sizing
    5. Apply fleet-level risk checks
    6. Apply cross-asset scaling
    7. Execute via BrokerRouter
    8. Log to FleetDB
    """

    def __init__(self, paper_mode: bool = True, fleet_capital: float = 500.0):
        self.paper_mode = paper_mode
        self.fleet_capital = fleet_capital

        # Core infrastructure
        self.db = FleetDB()
        self.risk = FleetRisk(self.db, fleet_capital)
        self.router = BrokerRouter(paper_mode=paper_mode)

        # Sub-bots
        self.bots: Dict[str, FleetBot] = {}
        self._confidence_voter = None
        self._cross_asset = None

        # Stats
        self.cycle_count = 0
        self.total_signals_generated = 0
        self.total_trades_executed = 0
        self.total_risk_blocked = 0
        self._initialized = False

        logger.info(f"FleetOrchestrator created (paper={paper_mode}, capital=${fleet_capital})")

    def initialize(self):
        """Load all sub-bots. Called once at startup."""
        if self._initialized:
            return

        loaded = 0
        failed = 0

        for module_path, class_name, default_config in FLEET_BOT_REGISTRY:
            try:
                bot_class = _safe_import_bot(module_path, class_name)
                if bot_class is None:
                    failed += 1
                    continue

                # Override paper_mode from orchestrator
                default_config.paper_mode = self.paper_mode

                bot = bot_class(config=default_config)
                self.bots[default_config.name] = bot

                # Give all bots DB access for position queries and exits
                bot.set_db(self.db)

                # Track meta-controller references
                if default_config.name == 'Confidence-Voter':
                    self._confidence_voter = bot
                elif default_config.name == 'Cross-Asset-Momentum':
                    self._cross_asset = bot

                loaded += 1
                logger.debug(f"Loaded: {default_config.name}")
            except Exception as e:
                failed += 1
                logger.error(f"Failed to load {class_name}: {e}")

        self._initialized = True
        logger.info(f"Fleet initialized: {loaded} bots loaded, {failed} failed")

    def run_cycle(self) -> List[Dict]:
        """
        Run one fleet cycle. Called every 60 seconds by FleetAdapter.
        Returns list of executed trade dicts for master_orchestrator logging.
        """
        if not self._initialized:
            self.initialize()

        self.cycle_count += 1
        now = datetime.now(timezone.utc)
        cycle_start = time.time()
        executed_trades = []
        cycle_signals = 0
        cycle_blocked = 0

        # Collect signals from all DUE bots
        for bot_name, bot in self.bots.items():
            if not bot.is_due(now):
                continue

            try:
                # Pre-scan checks
                if not bot.pre_scan_checks():
                    bot.last_run = now
                    continue

                # Scan for signals
                signals = bot.scan()
                bot.last_run = now

                if not signals:
                    continue

                # Meta bots don't produce tradeable signals
                if bot.bot_type == BotType.META:
                    cycle_signals += len(signals)
                    continue

                # Bot-level filter
                filtered = bot.filter_signals(signals)
                cycle_signals += len(filtered)

                for signal in filtered:
                    # Apply Thompson Sampling position sizing
                    if self._confidence_voter:
                        allocation = self._confidence_voter.get_allocation(bot_name)
                        signal.position_size_usd = min(signal.position_size_usd, allocation)

                    # Apply cross-asset scaling for crypto bots
                    if bot.bot_type == BotType.CRYPTO and self._cross_asset:
                        scale = self._cross_asset.get_scale_factor()
                        signal.position_size_usd *= scale
                        signal.metadata['risk_scale'] = scale

                    # Fleet-level risk check
                    allowed, reason = self.risk.check_trade(signal)
                    if not allowed:
                        cycle_blocked += 1
                        self.total_risk_blocked += 1
                        logger.info(f"BLOCKED: {bot_name} {signal.side} {signal.symbol}: {reason}")
                        continue

                    # Execute via broker router
                    result = self.router.execute(signal)

                    if result['success']:
                        # Log to fleet DB
                        self.db.log_trade(signal.to_dict())

                        # Update Thompson Sampling
                        self.db.update_thompson(bot_name, True, 0)  # P&L updated on close

                        # Notify bot
                        bot.on_trade_result(signal, True, result)
                        self.total_trades_executed += 1

                        # Format for master_orchestrator
                        executed_trades.append({
                            'action': signal.side,
                            'symbol': signal.symbol,
                            'price': result['fill_price'],
                            'quantity': result['fill_quantity'],
                            'confidence': signal.confidence,
                            'reason': f"[Fleet:{bot_name}] {signal.reason}",
                            'bot_name': f'Fleet:{bot_name}',
                            'order_id': result['order_id'],
                        })
                    else:
                        bot.on_trade_result(signal, False, result)
                        logger.warning(
                            f"Execution failed: {bot_name} {signal.side} {signal.symbol}: "
                            f"{result['error']}"
                        )

            except Exception as e:
                logger.error(f"Error in {bot_name}: {e}\n{traceback.format_exc()}")
                continue

        # Update risk state
        self.db.update_risk_state(
            daily_pnl=self.db.get_today_pnl(),
            daily_trades=len(self.db.get_today_trades()),
            open_positions=self.db.get_open_position_count(),
            total_exposure=self.db.get_total_exposure(),
        )

        elapsed = time.time() - cycle_start
        self.total_signals_generated += cycle_signals

        if cycle_signals > 0 or executed_trades:
            logger.info(
                f"Cycle #{self.cycle_count}: {cycle_signals} signals, "
                f"{len(executed_trades)} executed, {cycle_blocked} blocked "
                f"({elapsed:.1f}s)"
            )

        return executed_trades

    def check_exits(self) -> List[Dict]:
        """
        Check all bots for exit signals. Called by FleetAdapter.check_exits().
        """
        if not self._initialized:
            return []

        exit_trades = []
        open_positions = self.db.get_open_positions()

        for bot_name, bot in self.bots.items():
            if bot.bot_type == BotType.META:
                continue

            try:
                exit_signals = bot.check_exits()
                for signal in exit_signals:
                    exit_type = signal.metadata.get('exit_type', '')
                    trade_id = signal.metadata.get('trade_id', '')

                    # Kalshi expiry: market settled, just close in DB (no broker call needed)
                    if exit_type == 'expiry' and bot.bot_type == BotType.KALSHI:
                        matching = [
                            p for p in open_positions
                            if p.get('trade_id') == trade_id or
                               (p['symbol'] == signal.symbol and p['bot_name'] == bot_name)
                        ]
                        for pos in matching:
                            # Expired Kalshi = total loss (settled at 0 for our side)
                            pnl = -pos['position_size_usd']
                            pnl_pct = -1.0
                            self.db.close_trade(pos['trade_id'], 0.0, pnl, pnl_pct)
                            self.db.update_bot_stats(bot_name, pnl, False, pos.get('edge', 0))
                            self.db.update_thompson(bot_name, False, pnl)
                            self.risk.record_trade_result(False)
                            bot.record_loss()
                            exit_trades.append({
                                'action': 'EXPIRED',
                                'symbol': signal.symbol,
                                'price': 0.0,
                                'pnl': pnl,
                                'bot_name': f'Fleet:{bot_name}',
                                'reason': signal.reason,
                            })
                        continue

                    # For crypto/forex: execute sell via broker
                    result = self.router.execute(signal)
                    if result['success']:
                        matching = [
                            p for p in open_positions
                            if p.get('trade_id') == trade_id or
                               (p['symbol'] == signal.symbol and p['bot_name'] == bot_name)
                        ]
                        for pos in matching:
                            entry = pos['entry_price']
                            exit_p = result['fill_price']
                            if pos['side'] in ('BUY', 'YES'):
                                pnl = (exit_p - entry) * pos['quantity']
                            else:
                                pnl = (entry - exit_p) * pos['quantity']
                            pnl_pct = pnl / pos['position_size_usd'] if pos['position_size_usd'] > 0 else 0

                            self.db.close_trade(pos['trade_id'], exit_p, pnl, pnl_pct)
                            self.db.update_bot_stats(bot_name, pnl, pnl > 0, pos.get('edge', 0))
                            self.db.update_thompson(bot_name, pnl > 0, pnl)
                            self.risk.record_trade_result(pnl > 0)

                            if pnl > 0:
                                bot.record_win()
                            else:
                                bot.record_loss()

                            exit_trades.append({
                                'action': signal.side,
                                'symbol': signal.symbol,
                                'price': exit_p,
                                'pnl': pnl,
                                'bot_name': f'Fleet:{bot_name}',
                                'reason': signal.reason,
                            })

            except Exception as e:
                logger.error(f"Exit check error for {bot_name}: {e}")

        if exit_trades:
            logger.info(f"Exits: {len(exit_trades)} positions closed")

        return exit_trades

    def pause_bot(self, bot_name: str) -> bool:
        """Pause a specific sub-bot."""
        if bot_name in self.bots:
            self.bots[bot_name].is_paused = True
            logger.info(f"Paused: {bot_name}")
            return True
        return False

    def resume_bot(self, bot_name: str) -> bool:
        """Resume a specific sub-bot."""
        if bot_name in self.bots:
            self.bots[bot_name].is_paused = False
            logger.info(f"Resumed: {bot_name}")
            return True
        return False

    def emergency_stop(self):
        """Stop all fleet trading immediately."""
        for bot in self.bots.values():
            bot.is_paused = True
        logger.warning("EMERGENCY STOP — all fleet bots paused")

    def get_fleet_status(self) -> Dict[str, Any]:
        """Full fleet status for dashboard."""
        return {
            'cycle_count': self.cycle_count,
            'total_signals': self.total_signals_generated,
            'total_trades': self.total_trades_executed,
            'total_blocked': self.total_risk_blocked,
            'paper_mode': self.paper_mode,
            'fleet_capital': self.fleet_capital,
            'bots': {name: bot.get_status() for name, bot in self.bots.items()},
            'risk': self.risk.get_risk_summary(),
            'open_positions': self.db.get_open_position_count(),
            'today_pnl': self.db.get_today_pnl(),
        }

    def get_bot_names(self) -> List[str]:
        """Get all sub-bot names."""
        return list(self.bots.keys())
