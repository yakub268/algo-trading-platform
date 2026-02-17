"""
Master Options Trading System

Comprehensive integration of all options trading components:
- Real-time options data feeds
- Advanced strategy execution
- Portfolio Greeks monitoring
- Risk management and hedging
- Volatility forecasting
- Options flow analysis
- Backtesting and performance analytics

This is the main orchestrator that brings together all options trading capabilities.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable, Any
from decimal import Decimal
import logging
import json
from pathlib import Path
import warnings

# Core imports
from .core.option_chain import OptionChain, OptionContract
from .core.greeks_calculator import GreeksCalculator
from .core.option_data_manager import OptionDataManager

# Strategy imports
from .strategies.basic import CallStrategy, PutStrategy, CoveredCall, CashSecuredPut
from .strategies.spreads import BullCallSpread, BearPutSpread, IronCondor, Calendar
from .strategies.base_strategy import BaseOptionsStrategy

# Risk management imports
from .risk.portfolio_greeks import PortfolioGreeksMonitor, PortfolioGreeks
from .risk.risk_manager import OptionsRiskManager, RiskLimits

# Analytics imports
from .analytics.volatility_forecaster import VolatilityForecaster, VolatilityModel

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MasterOptionsSystem:
    """
    Master Options Trading System

    Central orchestrator for all options trading operations including:
    - Data management and real-time feeds
    - Strategy construction and execution
    - Risk monitoring and hedging
    - Performance tracking and analytics
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 data_sources: Optional[List[str]] = None,
                 log_level: str = "INFO"):
        """
        Initialize the Master Options Trading System

        Args:
            config: Configuration dictionary
            data_sources: List of data sources to use
            log_level: Logging level
        """
        # Set up logging
        logging.basicConfig(level=getattr(logging, log_level))

        # Configuration
        self.config = config or self._get_default_config()

        # Initialize components
        self.data_manager = OptionDataManager(
            alpaca_api_key=self.config.get('alpaca_api_key'),
            alpaca_secret_key=self.config.get('alpaca_secret_key'),
            polygon_api_key=self.config.get('polygon_api_key'),
            cache_dir=self.config.get('cache_dir', 'data/options_cache')
        )

        self.greeks_monitor = PortfolioGreeksMonitor()
        self.risk_manager = OptionsRiskManager()
        self.vol_forecaster = VolatilityForecaster()

        # Active strategies and positions
        self.active_strategies: Dict[str, BaseOptionsStrategy] = {}
        self.option_chains: Dict[str, OptionChain] = {}

        # Performance tracking
        self.performance_history = []
        self.trade_log = []

        # System state
        self.is_running = False
        self.last_update = None

        # Callbacks
        self.alert_callbacks = []
        self.trade_callbacks = []

        logger.info("Master Options Trading System initialized")

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'data_refresh_interval': 60,  # seconds
            'risk_check_interval': 30,   # seconds
            'max_portfolio_delta': 100,
            'max_portfolio_gamma': 50,
            'max_single_position_size': 10,
            'default_dte_target': 30,
            'volatility_lookback': 252,
            'enable_auto_hedging': False,
            'enable_auto_exit': False,
            'log_trades': True,
            'save_performance_data': True
        }

    async def start_system(self) -> None:
        """Start the options trading system"""
        try:
            self.is_running = True
            logger.info("Starting Master Options Trading System...")

            # Start background tasks
            tasks = [
                self._data_update_loop(),
                self._risk_monitoring_loop(),
                self._performance_tracking_loop()
            ]

            # Run all tasks concurrently
            await asyncio.gather(*tasks)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Error in system operation: {e}")
        finally:
            await self.stop_system()

    async def stop_system(self) -> None:
        """Stop the options trading system"""
        self.is_running = False
        logger.info("Stopping Master Options Trading System...")

        # Save current state
        await self._save_system_state()

        logger.info("System stopped")

    async def _data_update_loop(self) -> None:
        """Background loop for updating options data"""
        while self.is_running:
            try:
                # Update option chains for active underlyings
                active_underlyings = set()
                for strategy in self.active_strategies.values():
                    active_underlyings.add(strategy.symbol)

                # Update data for each underlying
                for underlying in active_underlyings:
                    await self._update_option_chain(underlying)

                # Update portfolio Greeks
                self._update_portfolio_greeks()

                self.last_update = datetime.now()

                # Wait for next update
                await asyncio.sleep(self.config['data_refresh_interval'])

            except Exception as e:
                logger.error(f"Error in data update loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry

    async def _risk_monitoring_loop(self) -> None:
        """Background loop for risk monitoring"""
        while self.is_running:
            try:
                # Calculate current portfolio Greeks
                portfolio_greeks = self.greeks_monitor.calculate_portfolio_greeks()

                # Check risk limits
                risk_alerts = self.risk_manager.check_risk_limits(portfolio_greeks)

                # Handle alerts
                for alert in risk_alerts:
                    await self._handle_risk_alert(alert, portfolio_greeks)

                # Auto-hedging if enabled
                if self.config.get('enable_auto_hedging', False):
                    await self._auto_hedge_portfolio(portfolio_greeks)

                await asyncio.sleep(self.config['risk_check_interval'])

            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _performance_tracking_loop(self) -> None:
        """Background loop for performance tracking"""
        while self.is_running:
            try:
                # Calculate current performance metrics
                performance = await self._calculate_performance_metrics()

                if performance:
                    self.performance_history.append({
                        'timestamp': datetime.now(),
                        'metrics': performance
                    })

                # Trim history if too long
                if len(self.performance_history) > 10000:
                    self.performance_history = self.performance_history[-5000:]

                # Save performance data if configured
                if self.config.get('save_performance_data', True):
                    await self._save_performance_data()

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(60)

    async def _update_option_chain(self, symbol: str) -> None:
        """Update option chain for symbol"""
        try:
            chain = await self.data_manager.get_option_chain(
                symbol=symbol,
                preferred_source='alpaca',
                include_greeks=True
            )

            if chain:
                self.option_chains[symbol] = chain
                logger.debug(f"Updated option chain for {symbol}: {len(chain)} contracts")
            else:
                logger.warning(f"Failed to get option chain for {symbol}")

        except Exception as e:
            logger.error(f"Error updating option chain for {symbol}: {e}")

    def _update_portfolio_greeks(self) -> None:
        """Update portfolio Greeks with current positions"""
        try:
            # Clear existing positions and re-add from strategies
            # This ensures Greeks are calculated with latest data

            for strategy_id, strategy in self.active_strategies.items():
                for leg in strategy.legs:
                    # Get updated contract data
                    updated_chain = self.option_chains.get(strategy.symbol)
                    if updated_chain:
                        updated_contract = updated_chain.get_contract(leg.contract.symbol)
                        if updated_contract:
                            # Update the position in Greeks monitor
                            self.greeks_monitor.add_position(
                                symbol=leg.contract.symbol,
                                underlying=strategy.symbol,
                                contract=updated_contract,
                                position_size=leg.quantity,
                                spot_price=float(updated_chain.spot_price)
                            )

        except Exception as e:
            logger.error(f"Error updating portfolio Greeks: {e}")

    async def create_strategy(self,
                             strategy_type: str,
                             symbol: str,
                             parameters: Dict,
                             strategy_id: Optional[str] = None) -> Optional[str]:
        """
        Create and execute an options strategy

        Args:
            strategy_type: Type of strategy ('call', 'put', 'iron_condor', etc.)
            symbol: Underlying symbol
            parameters: Strategy parameters
            strategy_id: Optional custom strategy ID

        Returns:
            Strategy ID if successful, None if failed
        """
        try:
            # Generate strategy ID if not provided
            if strategy_id is None:
                strategy_id = f"{strategy_type}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Get current option chain
            option_chain = await self.data_manager.get_option_chain(
                symbol=symbol,
                include_greeks=True
            )

            if not option_chain:
                logger.error(f"Could not get option chain for {symbol}")
                return None

            # Create strategy based on type
            strategy = self._create_strategy_instance(strategy_type, symbol)
            if not strategy:
                logger.error(f"Unknown strategy type: {strategy_type}")
                return None

            # Construct the strategy
            success = strategy.construct_strategy(
                option_chain=option_chain,
                **parameters
            )

            if not success:
                logger.error(f"Failed to construct {strategy_type} strategy for {symbol}")
                return None

            # Validate strategy
            is_valid, issues = strategy.validate_strategy()
            if not is_valid:
                logger.error(f"Strategy validation failed: {issues}")
                return None

            # Check risk limits before adding
            risk_check = await self._check_strategy_risk(strategy)
            if not risk_check['approved']:
                logger.error(f"Strategy rejected by risk check: {risk_check['reason']}")
                return None

            # Add to active strategies
            self.active_strategies[strategy_id] = strategy

            # Add positions to Greeks monitor
            for leg in strategy.legs:
                self.greeks_monitor.add_position(
                    symbol=leg.contract.symbol,
                    underlying=symbol,
                    contract=leg.contract,
                    position_size=leg.quantity,
                    spot_price=float(option_chain.spot_price)
                )

            # Log the trade
            if self.config.get('log_trades', True):
                await self._log_trade('OPEN', strategy_id, strategy)

            # Trigger callbacks
            for callback in self.trade_callbacks:
                try:
                    await callback('strategy_created', strategy_id, strategy)
                except Exception as e:
                    logger.error(f"Error in trade callback: {e}")

            logger.info(f"Created strategy {strategy_id}: {strategy_type} on {symbol}")
            return strategy_id

        except Exception as e:
            logger.error(f"Error creating strategy: {e}")
            return None

    def _create_strategy_instance(self, strategy_type: str, symbol: str) -> Optional[BaseOptionsStrategy]:
        """Create strategy instance based on type"""
        strategy_map = {
            'long_call': lambda: CallStrategy(symbol, is_long=True),
            'short_call': lambda: CallStrategy(symbol, is_long=False),
            'long_put': lambda: PutStrategy(symbol, is_long=True),
            'short_put': lambda: PutStrategy(symbol, is_long=False),
            'covered_call': lambda: CoveredCall(symbol),
            'cash_secured_put': lambda: CashSecuredPut(symbol),
            'bull_call_spread': lambda: BullCallSpread(symbol),
            'bear_put_spread': lambda: BearPutSpread(symbol),
            'iron_condor': lambda: IronCondor(symbol),
            'calendar': lambda: Calendar(symbol)
        }

        strategy_constructor = strategy_map.get(strategy_type.lower())
        if strategy_constructor:
            return strategy_constructor()
        return None

    async def _check_strategy_risk(self, strategy: BaseOptionsStrategy) -> Dict[str, Any]:
        """Check if strategy passes risk limits"""
        try:
            # Calculate strategy metrics
            current_chain = self.option_chains.get(strategy.symbol)
            if not current_chain:
                return {'approved': False, 'reason': 'No market data available'}

            metrics = strategy.calculate_strategy_metrics(current_chain.spot_price)

            # Check position size limits
            max_position_size = self.config.get('max_single_position_size', 10)
            total_contracts = sum(abs(leg.quantity) for leg in strategy.legs)
            if total_contracts > max_position_size:
                return {'approved': False, 'reason': f'Position size {total_contracts} exceeds limit {max_position_size}'}

            # Check portfolio-level impacts
            temp_greeks = self.greeks_monitor.calculate_portfolio_greeks()

            # Add strategy Greeks to current portfolio
            strategy_greeks = strategy.calculate_strategy_greeks()
            projected_delta = temp_greeks.total_delta + strategy_greeks.delta
            projected_gamma = temp_greeks.total_gamma + strategy_greeks.gamma

            # Check limits
            max_delta = self.config.get('max_portfolio_delta', 100)
            max_gamma = self.config.get('max_portfolio_gamma', 50)

            if abs(projected_delta) > max_delta:
                return {'approved': False, 'reason': f'Projected delta {projected_delta} exceeds limit {max_delta}'}

            if abs(projected_gamma) > max_gamma:
                return {'approved': False, 'reason': f'Projected gamma {projected_gamma} exceeds limit {max_gamma}'}

            return {'approved': True, 'metrics': metrics}

        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return {'approved': False, 'reason': f'Risk check error: {str(e)}'}

    async def close_strategy(self, strategy_id: str, reason: str = "Manual close") -> bool:
        """
        Close an active strategy

        Args:
            strategy_id: ID of strategy to close
            reason: Reason for closing

        Returns:
            True if successfully closed
        """
        try:
            if strategy_id not in self.active_strategies:
                logger.error(f"Strategy {strategy_id} not found")
                return False

            strategy = self.active_strategies[strategy_id]

            # Remove positions from Greeks monitor
            for leg in strategy.legs:
                self.greeks_monitor.remove_position(leg.contract.symbol)

            # Calculate final P&L
            current_chain = self.option_chains.get(strategy.symbol)
            if current_chain:
                final_pnl = strategy.calculate_pnl(current_chain.spot_price)
            else:
                final_pnl = Decimal('0')

            # Log the trade
            if self.config.get('log_trades', True):
                await self._log_trade('CLOSE', strategy_id, strategy, pnl=final_pnl, reason=reason)

            # Remove from active strategies
            del self.active_strategies[strategy_id]

            # Trigger callbacks
            for callback in self.trade_callbacks:
                try:
                    await callback('strategy_closed', strategy_id, strategy, final_pnl)
                except Exception as e:
                    logger.error(f"Error in trade callback: {e}")

            logger.info(f"Closed strategy {strategy_id}: P&L ${final_pnl:.2f}, Reason: {reason}")
            return True

        except Exception as e:
            logger.error(f"Error closing strategy {strategy_id}: {e}")
            return False

    async def _handle_risk_alert(self, alert: Dict, portfolio_greeks: PortfolioGreeks) -> None:
        """Handle risk management alerts"""
        logger.warning(f"Risk alert: {alert}")

        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert, portfolio_greeks)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        # Auto-close strategies if configured and alert is severe
        if (self.config.get('enable_auto_exit', False) and
            alert.get('severity') == 'high'):

            logger.warning("High severity risk alert - initiating auto-exit procedures")
            await self._emergency_exit_procedures()

    async def _auto_hedge_portfolio(self, portfolio_greeks: PortfolioGreeks) -> None:
        """Auto-hedge portfolio if delta/gamma limits exceeded"""
        try:
            max_delta = self.config.get('max_portfolio_delta', 100)

            if abs(portfolio_greeks.total_delta) > max_delta:
                logger.info(f"Auto-hedging: Portfolio delta {portfolio_greeks.total_delta} exceeds limit")

                # Simple delta hedging - would be more sophisticated in practice
                # This is a placeholder for actual hedging logic
                hedge_amount = -portfolio_greeks.total_delta

                logger.info(f"Would hedge {hedge_amount} delta (auto-hedging not fully implemented)")

        except Exception as e:
            logger.error(f"Error in auto-hedging: {e}")

    async def _emergency_exit_procedures(self) -> None:
        """Emergency procedures to close all positions"""
        logger.critical("Executing emergency exit procedures")

        strategies_to_close = list(self.active_strategies.keys())
        for strategy_id in strategies_to_close:
            await self.close_strategy(strategy_id, "Emergency exit")

    async def _log_trade(self, action: str, strategy_id: str, strategy: BaseOptionsStrategy,
                        pnl: Optional[Decimal] = None, reason: str = "") -> None:
        """Log trade to trade log"""
        trade_log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'strategy_id': strategy_id,
            'strategy_type': strategy.strategy_name,
            'symbol': strategy.symbol,
            'legs': [
                {
                    'contract_symbol': leg.contract.symbol,
                    'strike': float(leg.contract.strike),
                    'expiry': leg.contract.expiry.isoformat(),
                    'option_type': leg.leg_type,
                    'quantity': leg.quantity,
                    'entry_price': float(leg.entry_price)
                }
                for leg in strategy.legs
            ],
            'net_premium': float(strategy.calculate_net_premium()),
            'pnl': float(pnl) if pnl is not None else None,
            'reason': reason
        }

        self.trade_log.append(trade_log_entry)

    async def _calculate_performance_metrics(self) -> Optional[Dict]:
        """Calculate current performance metrics"""
        try:
            if not self.active_strategies:
                return None

            total_pnl = Decimal('0')
            total_premium = Decimal('0')
            winning_trades = 0
            losing_trades = 0

            # Calculate current P&L for all active strategies
            for strategy in self.active_strategies.values():
                current_chain = self.option_chains.get(strategy.symbol)
                if current_chain:
                    strategy_pnl = strategy.calculate_pnl(current_chain.spot_price)
                    total_pnl += strategy_pnl

                    if strategy_pnl > 0:
                        winning_trades += 1
                    elif strategy_pnl < 0:
                        losing_trades += 1

                total_premium += abs(strategy.calculate_net_premium())

            # Calculate portfolio Greeks
            portfolio_greeks = self.greeks_monitor.calculate_portfolio_greeks()

            metrics = {
                'total_pnl': float(total_pnl),
                'total_premium_at_risk': float(total_premium),
                'return_on_premium': float(total_pnl / total_premium) if total_premium > 0 else 0,
                'winning_strategies': winning_trades,
                'losing_strategies': losing_trades,
                'total_strategies': len(self.active_strategies),
                'portfolio_delta': portfolio_greeks.total_delta,
                'portfolio_gamma': portfolio_greeks.total_gamma,
                'portfolio_theta': portfolio_greeks.total_theta,
                'portfolio_vega': portfolio_greeks.total_vega,
                'portfolio_value': portfolio_greeks.portfolio_value
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return None

    async def _save_performance_data(self) -> None:
        """Save performance data to file"""
        try:
            if not self.performance_history:
                return

            performance_file = Path('data/performance_history.json')
            performance_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert to JSON serializable format
            data_to_save = []
            for entry in self.performance_history[-100:]:  # Save last 100 entries
                serializable_entry = {
                    'timestamp': entry['timestamp'].isoformat(),
                    'metrics': entry['metrics']
                }
                data_to_save.append(serializable_entry)

            with open(performance_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving performance data: {e}")

    async def _save_system_state(self) -> None:
        """Save current system state"""
        try:
            state_file = Path('data/system_state.json')
            state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                'timestamp': datetime.now().isoformat(),
                'active_strategies': {
                    strategy_id: strategy.get_strategy_summary()
                    for strategy_id, strategy in self.active_strategies.items()
                },
                'portfolio_greeks': self.greeks_monitor.calculate_portfolio_greeks().to_dict(),
                'performance_summary': await self._calculate_performance_metrics()
            }

            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)

            logger.info("System state saved")

        except Exception as e:
            logger.error(f"Error saving system state: {e}")

    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback for risk alerts"""
        self.alert_callbacks.append(callback)

    def add_trade_callback(self, callback: Callable) -> None:
        """Add callback for trade events"""
        self.trade_callbacks.append(callback)

    def get_strategy_summary(self) -> Dict:
        """Get summary of all active strategies"""
        return {
            strategy_id: strategy.get_strategy_summary()
            for strategy_id, strategy in self.active_strategies.items()
        }

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        portfolio_greeks = self.greeks_monitor.calculate_portfolio_greeks()
        performance = asyncio.run(self._calculate_performance_metrics()) if self.active_strategies else {}

        return {
            'portfolio_greeks': portfolio_greeks.to_dict(),
            'performance_metrics': performance,
            'active_strategies_count': len(self.active_strategies),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'system_running': self.is_running
        }

    async def run_strategy_scan(self, symbols: List[str], scan_parameters: Dict) -> List[Dict]:
        """
        Scan for strategy opportunities across multiple symbols

        Args:
            symbols: List of symbols to scan
            scan_parameters: Parameters for opportunity scanning

        Returns:
            List of identified opportunities
        """
        opportunities = []

        for symbol in symbols:
            try:
                # Get option chain
                option_chain = await self.data_manager.get_option_chain(symbol, include_greeks=True)
                if not option_chain:
                    continue

                # Get volatility forecast
                # This would require historical price data in practice

                # Example opportunity identification (simplified)
                # Look for high IV rank opportunities
                contracts = list(option_chain.contracts.values())
                if contracts:
                    avg_iv = np.mean([c.implied_volatility for c in contracts if c.implied_volatility])

                    if avg_iv > scan_parameters.get('min_iv', 0.3):
                        opportunity = {
                            'symbol': symbol,
                            'opportunity_type': 'high_iv',
                            'current_iv': avg_iv,
                            'spot_price': float(option_chain.spot_price),
                            'suggested_strategies': ['iron_condor', 'short_strangle'],
                            'confidence': 'medium'
                        }
                        opportunities.append(opportunity)

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue

        return opportunities

    def __repr__(self) -> str:
        return (f"MasterOptionsSystem(active_strategies={len(self.active_strategies)}, "
                f"running={self.is_running}, last_update={self.last_update})")


# Example usage and integration
async def main():
    """Example usage of the Master Options System"""

    # Initialize system
    config = {
        'max_portfolio_delta': 50,
        'max_portfolio_gamma': 25,
        'enable_auto_hedging': False,
        'log_trades': True
    }

    system = MasterOptionsSystem(config=config)

    # Add alert callback
    async def risk_alert_handler(alert, portfolio_greeks):
        print(f"RISK ALERT: {alert}")

    system.add_alert_callback(risk_alert_handler)

    # Example: Create an iron condor strategy
    strategy_params = {
        'target_dte': 30,
        'call_width': Decimal('5'),
        'put_width': Decimal('5'),
        'target_delta': 0.15
    }

    strategy_id = await system.create_strategy(
        strategy_type='iron_condor',
        symbol='SPY',
        parameters=strategy_params
    )

    if strategy_id:
        print(f"Created strategy: {strategy_id}")

        # Get portfolio summary
        summary = system.get_portfolio_summary()
        print(f"Portfolio Summary: {summary}")

    # In a real application, you would start the system:
    # await system.start_system()


if __name__ == "__main__":
    asyncio.run(main())