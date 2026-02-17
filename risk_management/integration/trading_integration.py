"""
Trading System Integration
=========================

Integration module connecting the advanced risk management system
with the existing trading bots and master orchestrator.

Features:
- Seamless integration with existing trading framework
- Real-time risk monitoring for all bots
- Automatic position sizing and risk controls
- Trading approval/rejection system
- Emergency protection protocols

Author: Trading Bot Arsenal
Created: February 2026
"""

import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import threading
import time
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.risk_manager import AdvancedRiskManager, TradingDecision, RiskAssessment
from ..config.risk_config import RiskManagementConfig, load_risk_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TradingIntegration')


@dataclass
class BotRiskProfile:
    """Risk profile for individual trading bot"""
    bot_name: str
    strategy_type: str
    max_position_size: float
    kelly_fraction: float
    risk_multiplier: float
    enabled: bool = True
    last_trade: Optional[datetime] = None
    total_trades: int = 0
    wins: int = 0
    losses: int = 0


@dataclass
class TradeRequest:
    """Trade request from trading bot"""
    bot_name: str
    symbol: str
    action: str  # 'buy', 'sell', 'close'
    size: float
    price: Optional[float]
    strategy: str
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeResponse:
    """Response to trade request"""
    request_id: str
    approved: bool
    recommended_size: float
    rejection_reasons: List[str]
    risk_adjustments: Dict[str, float]
    risk_score: float
    max_allowed_size: float


class TradingSystemIntegration:
    """
    Integration layer between risk management and trading system.

    Provides risk-controlled trading interface for all bots.
    """

    def __init__(self, config: RiskManagementConfig = None):
        """
        Initialize trading system integration.

        Args:
            config: Risk management configuration
        """
        if config is None:
            config = load_risk_config()

        self.config = config
        self.risk_manager = AdvancedRiskManager(config, self._handle_alert)

        # Bot management
        self.registered_bots: Dict[str, BotRiskProfile] = {}
        self.active_positions: Dict[str, Dict] = {}  # bot_name -> positions

        # Trade tracking
        self.pending_trades: Dict[str, TradeRequest] = {}
        self.trade_history: List[Dict] = []

        # Integration state
        self.integration_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()

        # Performance tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.start_balance = config.portfolio_value

        # Callbacks
        self.trade_callbacks: List[callable] = []
        self.alert_callbacks: List[callable] = []

        # Broker clients for emergency position closure
        # These get registered by the dashboard or orchestrator at startup
        self._broker_clients: Dict[str, Any] = {}  # e.g. {'alpaca': AlpacaService, 'kalshi': KalshiService, ...}
        self._emergency_close_callbacks: List[callable] = []

        self._load_bot_profiles()
        logger.info("TradingSystemIntegration initialized")

    def register_bot(
        self,
        bot_name: str,
        strategy_type: str,
        max_position_size: float = None,
        kelly_fraction: float = None
    ):
        """
        Register trading bot with risk management.

        Args:
            bot_name: Name of the trading bot
            strategy_type: Type of strategy (momentum, mean_reversion, etc.)
            max_position_size: Maximum position size for this bot
            kelly_fraction: Kelly fraction for this bot
        """
        if max_position_size is None:
            max_position_size = self.config.portfolio_limits.max_single_position * self.config.portfolio_value

        if kelly_fraction is None:
            kelly_fraction = self.config.kelly_config.strategy_kelly_fractions.get(
                strategy_type, self.config.kelly_config.default_kelly_fraction
            )

        risk_profile = BotRiskProfile(
            bot_name=bot_name,
            strategy_type=strategy_type,
            max_position_size=max_position_size,
            kelly_fraction=kelly_fraction,
            risk_multiplier=1.0
        )

        self.registered_bots[bot_name] = risk_profile
        self.active_positions[bot_name] = {}

        logger.info(f"Registered bot: {bot_name} ({strategy_type})")

    def request_trade(
        self,
        bot_name: str,
        symbol: str,
        action: str,
        size: float,
        price: float = None,
        confidence: float = 1.0
    ) -> TradeResponse:
        """
        Request trade approval from risk management.

        Args:
            bot_name: Name of requesting bot
            symbol: Trading symbol
            action: Trade action ('buy', 'sell', 'close')
            size: Requested position size
            price: Trade price (optional)
            confidence: Trade confidence (0-1)

        Returns:
            Trade response with approval/rejection
        """
        try:
            # Validate bot registration
            if bot_name not in self.registered_bots:
                return TradeResponse(
                    request_id=f"INVALID_{int(time.time())}",
                    approved=False,
                    recommended_size=0.0,
                    rejection_reasons=[f"Bot {bot_name} not registered"],
                    risk_adjustments={},
                    risk_score=100.0,
                    max_allowed_size=0.0
                )

            bot_profile = self.registered_bots[bot_name]
            if not bot_profile.enabled:
                return TradeResponse(
                    request_id=f"DISABLED_{int(time.time())}",
                    approved=False,
                    recommended_size=0.0,
                    rejection_reasons=[f"Bot {bot_name} is disabled"],
                    risk_adjustments={},
                    risk_score=100.0,
                    max_allowed_size=0.0
                )

            # Create trade request
            request = TradeRequest(
                bot_name=bot_name,
                symbol=symbol,
                action=action,
                size=size,
                price=price,
                strategy=bot_profile.strategy_type,
                confidence=confidence
            )

            request_id = f"{bot_name}_{symbol}_{int(time.time() * 1000)}"
            self.pending_trades[request_id] = request

            # Apply bot-specific limits
            adjusted_size = min(size, bot_profile.max_position_size)

            # Get risk management decision
            decision = self.risk_manager.evaluate_trade(
                symbol=symbol,
                action=action,
                proposed_size=adjusted_size,
                strategy=bot_profile.strategy_type,
                confidence=confidence
            )

            # Apply bot-specific risk multiplier
            final_size = decision.recommended_size * bot_profile.risk_multiplier

            response = TradeResponse(
                request_id=request_id,
                approved=decision.approved,
                recommended_size=final_size,
                rejection_reasons=decision.rejection_reasons,
                risk_adjustments=decision.risk_adjustments,
                risk_score=decision.risk_score,
                max_allowed_size=decision.max_allowed_size
            )

            # Log decision
            if decision.approved:
                logger.info(f"Trade approved: {bot_name} {action} {symbol} ${final_size:,.0f}")
            else:
                logger.warning(f"Trade rejected: {bot_name} {action} {symbol} - {decision.rejection_reasons}")

            # Clean up old pending trades
            self._cleanup_pending_trades()

            return response

        except Exception as e:
            logger.error(f"Trade request failed: {e}")
            return TradeResponse(
                request_id=f"ERROR_{int(time.time())}",
                approved=False,
                recommended_size=0.0,
                rejection_reasons=[f"System error: {str(e)}"],
                risk_adjustments={},
                risk_score=100.0,
                max_allowed_size=0.0
            )

    def execute_trade(
        self,
        request_id: str,
        executed_size: float,
        executed_price: float,
        success: bool = True
    ):
        """
        Report trade execution back to risk management.

        Args:
            request_id: Trade request ID
            executed_size: Actual executed size
            executed_price: Actual executed price
            success: Whether trade was successful
        """
        try:
            if request_id not in self.pending_trades:
                logger.warning(f"Unknown trade request ID: {request_id}")
                return

            request = self.pending_trades[request_id]
            bot_name = request.bot_name

            if success:
                # Update position tracking
                if request.action.lower() in ['buy', 'sell']:
                    self._add_position(
                        bot_name=bot_name,
                        symbol=request.symbol,
                        size=executed_size if request.action.lower() == 'buy' else -executed_size,
                        price=executed_price,
                        strategy=request.strategy
                    )
                elif request.action.lower() == 'close':
                    self._close_position(bot_name, request.symbol, executed_price)

                # Update bot statistics
                bot_profile = self.registered_bots[bot_name]
                bot_profile.total_trades += 1
                bot_profile.last_trade = datetime.now()

                # Record trade for Kelly optimization
                if request.action.lower() == 'close' and request.symbol in self.active_positions[bot_name]:
                    position = self.active_positions[bot_name][request.symbol]
                    entry_price = position['entry_price']

                    self.risk_manager.kelly_optimizer.add_trade_result(
                        symbol=request.symbol,
                        entry_price=entry_price,
                        exit_price=executed_price,
                        position_size=abs(executed_size),
                        trade_date=datetime.now()
                    )

                    # Update win/loss stats
                    is_win = (executed_price > entry_price) if position['size'] > 0 else (executed_price < entry_price)
                    if is_win:
                        bot_profile.wins += 1
                    else:
                        bot_profile.losses += 1

                logger.info(f"Trade executed: {bot_name} {request.action} {request.symbol} "
                           f"${executed_size:,.0f} @ ${executed_price:.2f}")

            else:
                logger.warning(f"Trade execution failed: {request_id}")

            # Remove from pending
            del self.pending_trades[request_id]

            # Record in history
            self.trade_history.append({
                'request_id': request_id,
                'bot_name': bot_name,
                'symbol': request.symbol,
                'action': request.action,
                'requested_size': request.size,
                'executed_size': executed_size,
                'executed_price': executed_price,
                'success': success,
                'timestamp': datetime.now()
            })

            # Keep limited history
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-500:]

            # Trigger callbacks
            for callback in self.trade_callbacks:
                try:
                    callback(request, executed_size, executed_price, success)
                except Exception as e:
                    logger.error(f"Trade callback failed: {e}")

        except Exception as e:
            logger.error(f"Trade execution reporting failed: {e}")

    def update_portfolio_value(self, new_value: float):
        """
        Update portfolio value and trigger risk assessment.

        Args:
            new_value: New portfolio value
        """
        old_value = self.config.portfolio_value
        self.config.portfolio_value = new_value

        # Update P&L tracking
        pnl_change = new_value - old_value
        self.daily_pnl += pnl_change
        self.total_pnl = new_value - self.start_balance

        # Update risk management
        protection_status = self.risk_manager.update_portfolio_value(new_value)

        # Check for risk alerts
        if protection_status.severity_level.value in ['critical', 'emergency']:
            self._handle_critical_drawdown(protection_status)

        logger.debug(f"Portfolio value updated: ${new_value:,.2f} (P&L: ${pnl_change:+,.2f})")

    def start_monitoring(self):
        """Start background risk monitoring"""
        self.integration_active = True
        self.risk_manager.start_monitoring()

        def monitor_positions():
            while not self.stop_monitoring.is_set():
                try:
                    self._update_position_tracking()
                    self._perform_periodic_assessment()
                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    logger.error(f"Position monitoring error: {e}")
                    time.sleep(60)

        self.monitoring_thread = threading.Thread(target=monitor_positions, daemon=True)
        self.monitoring_thread.start()

        logger.info("Risk monitoring started")

    def stop_monitoring(self):
        """Stop background risk monitoring"""
        self.integration_active = False
        self.stop_monitoring.set()

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)

        self.risk_manager.stop_monitoring()
        logger.info("Risk monitoring stopped")

    def enable_bot(self, bot_name: str):
        """Enable trading for specific bot"""
        if bot_name in self.registered_bots:
            self.registered_bots[bot_name].enabled = True
            logger.info(f"Bot enabled: {bot_name}")

    def disable_bot(self, bot_name: str):
        """Disable trading for specific bot"""
        if bot_name in self.registered_bots:
            self.registered_bots[bot_name].enabled = False
            logger.warning(f"Bot disabled: {bot_name}")

    def register_broker(self, broker_name: str, broker_client: Any):
        """
        Register a broker client for emergency position closure.

        Call this from your dashboard or orchestrator after initializing broker services.
        Example:
            integration.register_broker('alpaca', alpaca_service)
            integration.register_broker('kalshi', kalshi_service)
            integration.register_broker('oanda', oanda_service)

        Args:
            broker_name: Identifier for the broker (e.g. 'alpaca', 'kalshi', 'oanda')
            broker_client: The broker service object (must have close_all_positions or close_position)
        """
        self._broker_clients[broker_name] = broker_client
        logger.info(f"Broker registered for emergency closure: {broker_name}")

    def unregister_broker(self, broker_name: str):
        """Remove a broker client registration"""
        if broker_name in self._broker_clients:
            del self._broker_clients[broker_name]
            logger.info(f"Broker unregistered: {broker_name}")

    def register_emergency_close_callback(self, callback: callable):
        """
        Register a callback that will be invoked during emergency_stop_all()
        to close positions. Use this if broker clients aren't directly accessible.

        The callback receives the reason string as its argument.
        Example:
            def close_all_positions(reason):
                services['alpaca'].close_all_positions()
            integration.register_emergency_close_callback(close_all_positions)
        """
        self._emergency_close_callbacks.append(callback)
        logger.info("Emergency close callback registered")

    def _close_all_broker_positions(self, reason: str) -> Dict[str, Any]:
        """
        Close all positions across all registered brokers.

        Each broker is wrapped in its own try/except so a failure on one
        broker does not prevent the others from closing.

        Args:
            reason: The emergency reason string for logging

        Returns:
            Dict with per-broker results: {'broker_name': {'success': bool, 'error': str|None}}
        """
        results = {}

        # --- Alpaca ---
        if 'alpaca' in self._broker_clients and self._broker_clients['alpaca'] is not None:
            try:
                client = self._broker_clients['alpaca']
                client.close_all_positions()
                results['alpaca'] = {'success': True, 'error': None}
                logger.critical(f"EMERGENCY: Closed all Alpaca positions - {reason}")
            except Exception as e:
                results['alpaca'] = {'success': False, 'error': str(e)}
                logger.error(f"EMERGENCY: Failed to close Alpaca positions: {e}")

        # --- Kalshi ---
        if 'kalshi' in self._broker_clients and self._broker_clients['kalshi'] is not None:
            try:
                client = self._broker_clients['kalshi']
                # Kalshi has no close_all_positions(); iterate and close each position
                try:
                    positions = client.get_positions()
                except Exception:
                    positions = []
                closed_count = 0
                failed_count = 0
                for pos in positions:
                    try:
                        symbol = pos.symbol if hasattr(pos, 'symbol') else pos.get('symbol', '')
                        if symbol:
                            client.close_position(symbol)
                            closed_count += 1
                    except Exception as pos_e:
                        failed_count += 1
                        logger.error(f"EMERGENCY: Failed to close Kalshi position {pos}: {pos_e}")
                results['kalshi'] = {
                    'success': failed_count == 0,
                    'closed': closed_count,
                    'failed': failed_count,
                    'error': None if failed_count == 0 else f"{failed_count} positions failed to close"
                }
                logger.critical(f"EMERGENCY: Closed {closed_count} Kalshi positions "
                               f"({failed_count} failed) - {reason}")
            except Exception as e:
                results['kalshi'] = {'success': False, 'error': str(e)}
                logger.error(f"EMERGENCY: Failed to close Kalshi positions: {e}")

        # --- OANDA ---
        if 'oanda' in self._broker_clients and self._broker_clients['oanda'] is not None:
            try:
                client = self._broker_clients['oanda']
                client.close_all_positions()
                results['oanda'] = {'success': True, 'error': None}
                logger.critical(f"EMERGENCY: Closed all OANDA positions - {reason}")
            except Exception as e:
                results['oanda'] = {'success': False, 'error': str(e)}
                logger.error(f"EMERGENCY: Failed to close OANDA positions: {e}")

        # --- Any other registered brokers ---
        for broker_name, client in self._broker_clients.items():
            if broker_name in ('alpaca', 'kalshi', 'oanda') or client is None:
                continue  # Already handled above
            try:
                if hasattr(client, 'close_all_positions'):
                    client.close_all_positions()
                    results[broker_name] = {'success': True, 'error': None}
                    logger.critical(f"EMERGENCY: Closed all {broker_name} positions - {reason}")
                else:
                    results[broker_name] = {'success': False, 'error': 'No close_all_positions method'}
                    logger.warning(f"EMERGENCY: Broker {broker_name} has no close_all_positions method")
            except Exception as e:
                results[broker_name] = {'success': False, 'error': str(e)}
                logger.error(f"EMERGENCY: Failed to close {broker_name} positions: {e}")

        # --- Fire emergency close callbacks ---
        for i, callback in enumerate(self._emergency_close_callbacks):
            try:
                callback(reason)
                results[f'callback_{i}'] = {'success': True, 'error': None}
            except Exception as e:
                results[f'callback_{i}'] = {'success': False, 'error': str(e)}
                logger.error(f"EMERGENCY: Emergency close callback {i} failed: {e}")

        return results

    def emergency_stop_all(self, reason: str = "Manual emergency stop"):
        """
        Emergency stop all trading activity AND close all positions.

        Execution order:
        1. Disable all bots (prevent new trades immediately)
        2. Close all positions on all registered brokers
        3. Activate emergency mode in risk manager
        4. Send alerts with closure results
        """
        logger.critical(f"EMERGENCY STOP: {reason}")

        # Step 1: Disable all bots first to prevent any new trades
        for bot_name in self.registered_bots:
            self.disable_bot(bot_name)

        # Step 2: Close all positions on all registered brokers
        # Each broker is independently wrapped - one failure won't block others
        closure_results = {}
        if self._broker_clients or self._emergency_close_callbacks:
            logger.critical("EMERGENCY: Closing all positions across all brokers...")
            closure_results = self._close_all_broker_positions(reason)

            # Build summary for alert
            successes = [b for b, r in closure_results.items() if r.get('success')]
            failures = [b for b, r in closure_results.items() if not r.get('success')]

            if successes:
                logger.critical(f"EMERGENCY: Position closure succeeded on: {', '.join(successes)}")
            if failures:
                logger.error(f"EMERGENCY: Position closure FAILED on: {', '.join(failures)}")
                for broker in failures:
                    logger.error(f"  {broker}: {closure_results[broker].get('error')}")
        else:
            logger.warning(
                "EMERGENCY STOP: No broker clients registered - positions were NOT closed! "
                "Call register_broker() at startup to enable emergency position closure."
            )

        # Step 3: Activate emergency mode in risk manager
        self.risk_manager._activate_emergency_mode(reason)

        # Step 4: Send alerts with closure results
        alert_parts = [f"EMERGENCY STOP: {reason}"]
        if closure_results:
            successes = [b for b, r in closure_results.items() if r.get('success')]
            failures = [b for b, r in closure_results.items() if not r.get('success')]
            if successes:
                alert_parts.append(f"Positions closed on: {', '.join(successes)}")
            if failures:
                alert_parts.append(f"FAILED to close on: {', '.join(failures)}")
        else:
            alert_parts.append("WARNING: No brokers registered - positions may still be open!")

        self._handle_alert(" | ".join(alert_parts))

    def get_bot_status(self, bot_name: str = None) -> Dict[str, Any]:
        """Get status for specific bot or all bots"""
        if bot_name:
            if bot_name not in self.registered_bots:
                return {'error': f'Bot {bot_name} not registered'}

            bot_profile = self.registered_bots[bot_name]
            positions = self.active_positions[bot_name]

            win_rate = bot_profile.wins / max(bot_profile.total_trades, 1)

            return {
                'bot_name': bot_name,
                'strategy_type': bot_profile.strategy_type,
                'enabled': bot_profile.enabled,
                'total_trades': bot_profile.total_trades,
                'win_rate': f"{win_rate:.1%}",
                'last_trade': bot_profile.last_trade.isoformat() if bot_profile.last_trade else None,
                'active_positions': len(positions),
                'position_value': sum(pos['size'] * pos.get('current_price', pos['entry_price'])
                                    for pos in positions.values()),
                'max_position_size': f"${bot_profile.max_position_size:,.2f}",
                'risk_multiplier': bot_profile.risk_multiplier
            }
        else:
            # Return status for all bots
            return {
                bot_name: self.get_bot_status(bot_name)
                for bot_name in self.registered_bots
            }

    def _add_position(
        self,
        bot_name: str,
        symbol: str,
        size: float,
        price: float,
        strategy: str
    ):
        """Add position to tracking"""
        position_value = abs(size) * price

        # Add to bot's positions
        if symbol in self.active_positions[bot_name]:
            # Update existing position
            existing = self.active_positions[bot_name][symbol]
            new_size = existing['size'] + size
            new_value = abs(new_size) * price

            if abs(new_size) < 0.01:  # Position effectively closed
                del self.active_positions[bot_name][symbol]
                self.risk_manager.remove_position(symbol)
            else:
                existing['size'] = new_size
                existing['current_price'] = price
        else:
            # New position
            self.active_positions[bot_name][symbol] = {
                'size': size,
                'entry_price': price,
                'current_price': price,
                'strategy': strategy,
                'timestamp': datetime.now()
            }

        # Add to risk manager
        self.risk_manager.add_position(
            symbol=symbol,
            size=abs(size),
            entry_price=price,
            strategy=strategy,
            risk_amount=position_value * 0.02  # Assume 2% risk
        )

    def _close_position(self, bot_name: str, symbol: str, price: float):
        """Close position"""
        if symbol in self.active_positions[bot_name]:
            del self.active_positions[bot_name][symbol]
            self.risk_manager.remove_position(symbol)

    def _update_position_tracking(self):
        """Update position tracking with current prices"""
        # This would typically fetch current prices and update positions
        # For now, we'll use a placeholder
        pass

    def _perform_periodic_assessment(self):
        """Perform periodic risk assessment"""
        try:
            assessment = self.risk_manager.assess_overall_risk()

            # Check for concerning risk levels
            if assessment.overall_risk_score > 80:
                self._handle_high_risk(assessment)

            # Auto-adjust bot risk multipliers based on performance
            self._auto_adjust_bot_risk()

        except Exception as e:
            logger.error(f"Periodic assessment failed: {e}")

    def _handle_high_risk(self, assessment: RiskAssessment):
        """Handle high risk situations"""
        if assessment.overall_risk_score > 90:
            # Disable high-risk bots
            for bot_name, profile in self.registered_bots.items():
                if profile.strategy_type in ['momentum', 'aggressive']:
                    profile.risk_multiplier = 0.5  # Reduce risk
                    logger.warning(f"Reduced risk multiplier for {bot_name} due to high portfolio risk")

    def _handle_critical_drawdown(self, protection_status):
        """Handle critical drawdown situations"""
        if protection_status.severity_level.value == 'emergency':
            # Emergency stop
            self.emergency_stop_all("Maximum drawdown reached")
        elif protection_status.severity_level.value == 'critical':
            # Reduce all bot risk multipliers
            for profile in self.registered_bots.values():
                profile.risk_multiplier = min(profile.risk_multiplier, 0.5)

    def _auto_adjust_bot_risk(self):
        """Auto-adjust bot risk multipliers based on performance"""
        for bot_name, profile in self.registered_bots.items():
            if profile.total_trades >= 20:  # Minimum sample size
                win_rate = profile.wins / profile.total_trades

                if win_rate > 0.70:
                    # High performance - allow slight risk increase
                    profile.risk_multiplier = min(profile.risk_multiplier * 1.05, 1.2)
                elif win_rate < 0.40:
                    # Poor performance - reduce risk
                    profile.risk_multiplier = max(profile.risk_multiplier * 0.95, 0.3)

    def _cleanup_pending_trades(self):
        """Clean up old pending trades"""
        cutoff_time = datetime.now() - timedelta(minutes=30)
        expired_ids = [
            req_id for req_id, req in self.pending_trades.items()
            if req.timestamp < cutoff_time
        ]

        for req_id in expired_ids:
            del self.pending_trades[req_id]

    def _handle_alert(self, message: str):
        """Handle risk management alerts"""
        logger.warning(f"RISK ALERT: {message}")

        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _load_bot_profiles(self):
        """Load bot risk profiles from configuration"""
        # Define default bot profiles based on existing trading bots
        default_profiles = {
            'RSI2-MeanReversion': {
                'strategy_type': 'mean_reversion',
                'kelly_fraction': 0.25,
                'risk_multiplier': 1.0
            },
            'CumulativeRSI': {
                'strategy_type': 'mean_reversion',
                'kelly_fraction': 0.20,
                'risk_multiplier': 0.9
            },
            'DualMomentum': {
                'strategy_type': 'momentum',
                'kelly_fraction': 0.20,
                'risk_multiplier': 0.8
            },
            'SectorRotation': {
                'strategy_type': 'momentum',
                'kelly_fraction': 0.15,
                'risk_multiplier': 0.7
            },
            'KalshiBot': {
                'strategy_type': 'arbitrage',
                'kelly_fraction': 0.40,
                'risk_multiplier': 1.2
            },
            'OANDABot': {
                'strategy_type': 'forex',
                'kelly_fraction': 0.15,
                'risk_multiplier': 0.8
            },
            'CryptoArb': {
                'strategy_type': 'arbitrage',
                'kelly_fraction': 0.30,
                'risk_multiplier': 1.0
            }
        }

        # Register default bots
        for bot_name, profile in default_profiles.items():
            max_position = (self.config.portfolio_limits.max_single_position *
                          self.config.portfolio_value * profile['risk_multiplier'])

            self.register_bot(
                bot_name=bot_name,
                strategy_type=profile['strategy_type'],
                max_position_size=max_position,
                kelly_fraction=profile['kelly_fraction']
            )

    def add_trade_callback(self, callback: callable):
        """Add callback for trade events"""
        self.trade_callbacks.append(callback)

    def add_alert_callback(self, callback: callable):
        """Add callback for alert events"""
        self.alert_callbacks.append(callback)

    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        risk_status = self.risk_manager.get_status_report()

        return {
            'timestamp': datetime.now().isoformat(),
            'integration_active': self.integration_active,
            'registered_bots': len(self.registered_bots),
            'active_bots': sum(1 for bot in self.registered_bots.values() if bot.enabled),
            'pending_trades': len(self.pending_trades),
            'total_positions': sum(len(positions) for positions in self.active_positions.values()),
            'portfolio_metrics': {
                'current_value': f"${self.config.portfolio_value:,.2f}",
                'daily_pnl': f"${self.daily_pnl:+,.2f}",
                'total_pnl': f"${self.total_pnl:+,.2f}",
                'total_pnl_pct': f"{self.total_pnl/self.start_balance:+.1%}"
            },
            'risk_management': risk_status,
            'bot_summary': {
                bot_name: {
                    'enabled': profile.enabled,
                    'trades': profile.total_trades,
                    'win_rate': f"{profile.wins/max(profile.total_trades, 1):.1%}",
                    'risk_multiplier': profile.risk_multiplier
                }
                for bot_name, profile in self.registered_bots.items()
            }
        }


# Global integration instance
_trading_integration: Optional[TradingSystemIntegration] = None


def get_trading_integration() -> TradingSystemIntegration:
    """Get global trading integration instance"""
    global _trading_integration
    if _trading_integration is None:
        _trading_integration = TradingSystemIntegration()
    return _trading_integration


def initialize_risk_management(config: RiskManagementConfig = None) -> TradingSystemIntegration:
    """Initialize risk management integration"""
    global _trading_integration
    _trading_integration = TradingSystemIntegration(config)
    return _trading_integration


# Convenience functions for trading bots
def request_trade_approval(
    bot_name: str,
    symbol: str,
    action: str,
    size: float,
    price: float = None,
    confidence: float = 1.0
) -> TradeResponse:
    """Request trade approval (convenience function)"""
    integration = get_trading_integration()
    return integration.request_trade(bot_name, symbol, action, size, price, confidence)


def report_trade_execution(
    request_id: str,
    executed_size: float,
    executed_price: float,
    success: bool = True
):
    """Report trade execution (convenience function)"""
    integration = get_trading_integration()
    integration.execute_trade(request_id, executed_size, executed_price, success)


if __name__ == "__main__":
    # Test the integration system
    config = load_risk_config()
    integration = TradingSystemIntegration(config)

    # Start monitoring
    integration.start_monitoring()

    # Test trade request
    response = integration.request_trade(
        bot_name='RSI2-MeanReversion',
        symbol='SPY',
        action='buy',
        size=1000,
        confidence=0.8
    )

    print(f"Trade Request Result:")
    print(f"Approved: {response.approved}")
    print(f"Recommended Size: ${response.recommended_size:.2f}")
    print(f"Risk Score: {response.risk_score:.0f}")
    if not response.approved:
        print(f"Rejection Reasons: {response.rejection_reasons}")

    # Test trade execution
    if response.approved:
        integration.execute_trade(
            request_id=response.request_id,
            executed_size=response.recommended_size,
            executed_price=450.0,
            success=True
        )

    # Get status
    status = integration.get_integration_status()
    print(f"\nIntegration Status:")
    print(f"Active Bots: {status['active_bots']}/{status['registered_bots']}")
    print(f"Portfolio Value: {status['portfolio_metrics']['current_value']}")
    print(f"Risk Score: {status['risk_management']['overall_risk_score']}")

    # Cleanup
    time.sleep(1)
    integration.stop_monitoring()