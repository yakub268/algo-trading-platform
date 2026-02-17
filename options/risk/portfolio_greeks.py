"""
Portfolio Greeks Monitoring

Real-time calculation and monitoring of aggregate portfolio Greeks
with alerts and automated hedging triggers.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from decimal import Decimal
import logging
from collections import defaultdict

from ..core.greeks_calculator import Greeks, GreeksCalculator
from ..core.option_chain import OptionContract

logger = logging.getLogger(__name__)


@dataclass
class PortfolioGreeks:
    """Container for portfolio-level Greeks"""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float

    # By expiry breakdown
    greeks_by_expiry: Dict[str, Greeks] = field(default_factory=dict)

    # By underlying breakdown
    greeks_by_underlying: Dict[str, Greeks] = field(default_factory=dict)

    # Risk metrics
    delta_dollars: float = 0.0  # Delta in dollar terms
    gamma_dollars: float = 0.0  # Gamma in dollar terms
    theta_dollars: float = 0.0  # Theta in dollar terms
    vega_dollars: float = 0.0   # Vega in dollar terms

    # Portfolio value
    portfolio_value: float = 0.0
    underlying_exposure: float = 0.0

    # Timestamp
    calculation_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'total_delta': self.total_delta,
            'total_gamma': self.total_gamma,
            'total_theta': self.total_theta,
            'total_vega': self.total_vega,
            'total_rho': self.total_rho,
            'delta_dollars': self.delta_dollars,
            'gamma_dollars': self.gamma_dollars,
            'theta_dollars': self.theta_dollars,
            'vega_dollars': self.vega_dollars,
            'portfolio_value': self.portfolio_value,
            'underlying_exposure': self.underlying_exposure,
            'calculation_time': self.calculation_time.isoformat(),
            'greeks_by_expiry': {k: v.to_dict() for k, v in self.greeks_by_expiry.items()},
            'greeks_by_underlying': {k: v.to_dict() for k, v in self.greeks_by_underlying.items()}
        }


@dataclass
class PositionGreeks:
    """Greeks for individual position"""
    symbol: str
    underlying: str
    position_size: int
    contract_greeks: Greeks
    position_greeks: Greeks  # Adjusted for position size
    market_value: float
    expiry_date: datetime


class PortfolioGreeksMonitor:
    """
    Real-time portfolio Greeks monitoring and calculation

    Provides continuous monitoring of portfolio Greeks with
    alerts and automatic rebalancing triggers.
    """

    def __init__(self,
                 spot_price_feed: Optional[Callable] = None,
                 alert_thresholds: Optional[Dict] = None):
        """
        Initialize portfolio Greeks monitor

        Args:
            spot_price_feed: Function to get current spot prices
            alert_thresholds: Dictionary of alert thresholds for Greeks
        """
        self.positions = {}  # symbol -> PositionGreeks
        self.spot_prices = {}  # underlying -> current price
        self.spot_price_feed = spot_price_feed

        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'delta': {'max': 100, 'min': -100},  # Total delta limits
            'gamma': {'max': 50, 'min': -50},    # Total gamma limits
            'theta': {'max': 0, 'min': -500},    # Daily theta limit
            'vega': {'max': 1000, 'min': -1000}, # Vega exposure limit
        }

        self.greeks_calc = GreeksCalculator()
        self.historical_greeks = []  # Store historical snapshots
        self.max_history = 1000  # Maximum historical records

        # Callbacks
        self.alert_callbacks = []  # Functions to call on alerts
        self.hedge_callbacks = []  # Functions to call for hedging

        logger.info("Initialized PortfolioGreeksMonitor")

    def add_position(self,
                    symbol: str,
                    underlying: str,
                    contract: OptionContract,
                    position_size: int,
                    spot_price: Optional[float] = None) -> None:
        """
        Add options position to portfolio

        Args:
            symbol: Option symbol
            underlying: Underlying symbol
            contract: Option contract details
            position_size: Number of contracts (positive for long, negative for short)
            spot_price: Current underlying price
        """
        try:
            # Update spot price
            if spot_price is not None:
                self.spot_prices[underlying] = spot_price
            elif self.spot_price_feed:
                self.spot_prices[underlying] = self.spot_price_feed(underlying)

            # Get contract Greeks or calculate them
            if all([contract.delta is not None, contract.gamma is not None,
                   contract.theta is not None, contract.vega is not None,
                   contract.rho is not None]):
                contract_greeks = Greeks(
                    delta=contract.delta,
                    gamma=contract.gamma,
                    theta=contract.theta,
                    vega=contract.vega,
                    rho=contract.rho
                )
            else:
                # Calculate Greeks if not available
                current_spot = self.spot_prices.get(underlying, float(contract.strike))
                contract_greeks = self.greeks_calc.calculate_greeks(
                    spot_price=current_spot,
                    strike_price=float(contract.strike),
                    time_to_expiry=contract.time_to_expiry,
                    risk_free_rate=0.02,  # Default risk-free rate
                    volatility=contract.implied_volatility or 0.3,
                    option_type=contract.option_type
                )

            # Calculate position Greeks (scaled by position size)
            position_greeks = Greeks(
                delta=contract_greeks.delta * position_size,
                gamma=contract_greeks.gamma * position_size,
                theta=contract_greeks.theta * position_size,
                vega=contract_greeks.vega * position_size,
                rho=contract_greeks.rho * position_size
            )

            # Calculate market value
            market_value = float(contract.mid_price) * position_size * 100  # Options are per 100 shares

            # Create position object
            position = PositionGreeks(
                symbol=symbol,
                underlying=underlying,
                position_size=position_size,
                contract_greeks=contract_greeks,
                position_greeks=position_greeks,
                market_value=market_value,
                expiry_date=contract.expiry
            )

            self.positions[symbol] = position

            logger.info(f"Added position: {symbol} ({position_size} contracts)")

        except Exception as e:
            logger.error(f"Error adding position {symbol}: {e}")

    def remove_position(self, symbol: str) -> bool:
        """
        Remove position from portfolio

        Args:
            symbol: Option symbol to remove

        Returns:
            True if position was removed
        """
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"Removed position: {symbol}")
            return True
        else:
            logger.warning(f"Position {symbol} not found")
            return False

    def update_position_size(self, symbol: str, new_size: int) -> bool:
        """
        Update position size for existing position

        Args:
            symbol: Option symbol
            new_size: New position size

        Returns:
            True if position was updated
        """
        if symbol not in self.positions:
            logger.warning(f"Position {symbol} not found")
            return False

        try:
            position = self.positions[symbol]

            # Recalculate position Greeks with new size
            position.position_size = new_size
            position.position_greeks = Greeks(
                delta=position.contract_greeks.delta * new_size,
                gamma=position.contract_greeks.gamma * new_size,
                theta=position.contract_greeks.theta * new_size,
                vega=position.contract_greeks.vega * new_size,
                rho=position.contract_greeks.rho * new_size
            )

            # Update market value (would need current price in real implementation)
            # position.market_value = current_price * new_size * 100

            logger.info(f"Updated position {symbol} to {new_size} contracts")
            return True

        except Exception as e:
            logger.error(f"Error updating position {symbol}: {e}")
            return False

    def calculate_portfolio_greeks(self) -> PortfolioGreeks:
        """
        Calculate aggregate portfolio Greeks

        Returns:
            PortfolioGreeks object with all metrics
        """
        try:
            # Initialize totals
            total_delta = 0.0
            total_gamma = 0.0
            total_theta = 0.0
            total_vega = 0.0
            total_rho = 0.0

            portfolio_value = 0.0
            underlying_exposure = 0.0

            # Breakdown dictionaries
            greeks_by_expiry = defaultdict(lambda: {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0})
            greeks_by_underlying = defaultdict(lambda: {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0})

            # Sum up all positions
            for symbol, position in self.positions.items():
                # Add to totals
                total_delta += position.position_greeks.delta
                total_gamma += position.position_greeks.gamma
                total_theta += position.position_greeks.theta
                total_vega += position.position_greeks.vega
                total_rho += position.position_greeks.rho

                portfolio_value += position.market_value

                # Calculate underlying exposure (delta equivalent shares)
                current_spot = self.spot_prices.get(position.underlying, 100.0)
                underlying_exposure += position.position_greeks.delta * current_spot

                # Breakdown by expiry
                expiry_key = position.expiry_date.strftime('%Y-%m-%d')
                greeks_by_expiry[expiry_key]['delta'] += position.position_greeks.delta
                greeks_by_expiry[expiry_key]['gamma'] += position.position_greeks.gamma
                greeks_by_expiry[expiry_key]['theta'] += position.position_greeks.theta
                greeks_by_expiry[expiry_key]['vega'] += position.position_greeks.vega
                greeks_by_expiry[expiry_key]['rho'] += position.position_greeks.rho

                # Breakdown by underlying
                greeks_by_underlying[position.underlying]['delta'] += position.position_greeks.delta
                greeks_by_underlying[position.underlying]['gamma'] += position.position_greeks.gamma
                greeks_by_underlying[position.underlying]['theta'] += position.position_greeks.theta
                greeks_by_underlying[position.underlying]['vega'] += position.position_greeks.vega
                greeks_by_underlying[position.underlying]['rho'] += position.position_greeks.rho

            # Convert breakdowns to Greeks objects
            greeks_by_expiry_dict = {}
            for expiry, greeks_dict in greeks_by_expiry.items():
                greeks_by_expiry_dict[expiry] = Greeks(**greeks_dict)

            greeks_by_underlying_dict = {}
            for underlying, greeks_dict in greeks_by_underlying.items():
                greeks_by_underlying_dict[underlying] = Greeks(**greeks_dict)

            # Calculate dollar Greeks (approximate)
            # This would be more accurate with current underlying prices
            avg_underlying_price = np.mean(list(self.spot_prices.values())) if self.spot_prices else 100.0

            delta_dollars = total_delta * avg_underlying_price
            gamma_dollars = total_gamma * avg_underlying_price  # Simplified
            theta_dollars = total_theta * 100  # Theta per day in dollars
            vega_dollars = total_vega  # Vega is already in dollar terms

            # Create portfolio Greeks object
            portfolio_greeks = PortfolioGreeks(
                total_delta=total_delta,
                total_gamma=total_gamma,
                total_theta=total_theta,
                total_vega=total_vega,
                total_rho=total_rho,
                greeks_by_expiry=greeks_by_expiry_dict,
                greeks_by_underlying=greeks_by_underlying_dict,
                delta_dollars=delta_dollars,
                gamma_dollars=gamma_dollars,
                theta_dollars=theta_dollars,
                vega_dollars=vega_dollars,
                portfolio_value=portfolio_value,
                underlying_exposure=underlying_exposure,
                calculation_time=datetime.now()
            )

            # Store in history
            self.historical_greeks.append(portfolio_greeks)
            if len(self.historical_greeks) > self.max_history:
                self.historical_greeks.pop(0)

            # Check alerts
            self._check_alerts(portfolio_greeks)

            return portfolio_greeks

        except Exception as e:
            logger.error(f"Error calculating portfolio Greeks: {e}")
            return PortfolioGreeks(0, 0, 0, 0, 0)

    def _check_alerts(self, portfolio_greeks: PortfolioGreeks) -> None:
        """Check if any Greeks exceed alert thresholds"""
        alerts = []

        # Check delta
        if (portfolio_greeks.total_delta > self.alert_thresholds['delta']['max'] or
            portfolio_greeks.total_delta < self.alert_thresholds['delta']['min']):
            alerts.append({
                'type': 'delta_limit',
                'current_value': portfolio_greeks.total_delta,
                'threshold': self.alert_thresholds['delta'],
                'severity': 'high' if abs(portfolio_greeks.total_delta) > abs(self.alert_thresholds['delta']['max']) * 1.5 else 'medium'
            })

        # Check gamma
        if (portfolio_greeks.total_gamma > self.alert_thresholds['gamma']['max'] or
            portfolio_greeks.total_gamma < self.alert_thresholds['gamma']['min']):
            alerts.append({
                'type': 'gamma_limit',
                'current_value': portfolio_greeks.total_gamma,
                'threshold': self.alert_thresholds['gamma'],
                'severity': 'medium'
            })

        # Check theta
        if portfolio_greeks.total_theta < self.alert_thresholds['theta']['min']:
            alerts.append({
                'type': 'theta_decay',
                'current_value': portfolio_greeks.total_theta,
                'threshold': self.alert_thresholds['theta']['min'],
                'severity': 'low'
            })

        # Check vega
        if (abs(portfolio_greeks.total_vega) > self.alert_thresholds['vega']['max']):
            alerts.append({
                'type': 'vega_exposure',
                'current_value': portfolio_greeks.total_vega,
                'threshold': self.alert_thresholds['vega']['max'],
                'severity': 'medium'
            })

        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert, portfolio_greeks)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback function for risk alerts"""
        self.alert_callbacks.append(callback)

    def add_hedge_callback(self, callback: Callable) -> None:
        """Add callback function for hedge triggers"""
        self.hedge_callbacks.append(callback)

    def get_delta_exposure_by_underlying(self) -> Dict[str, float]:
        """Get delta exposure broken down by underlying"""
        delta_exposure = defaultdict(float)

        for position in self.positions.values():
            delta_exposure[position.underlying] += position.position_greeks.delta

        return dict(delta_exposure)

    def get_positions_expiring_soon(self, days_threshold: int = 7) -> List[PositionGreeks]:
        """Get positions expiring within threshold days"""
        expiring_soon = []
        cutoff_date = datetime.now() + timedelta(days=days_threshold)

        for position in self.positions.values():
            if position.expiry_date <= cutoff_date:
                expiring_soon.append(position)

        return expiring_soon

    def calculate_stress_scenarios(self,
                                  price_shocks: List[float] = None,
                                  vol_shocks: List[float] = None) -> Dict:
        """
        Calculate portfolio Greeks under stress scenarios

        Args:
            price_shocks: List of price shock percentages (e.g., [-0.1, 0.1])
            vol_shocks: List of volatility shock percentages

        Returns:
            Dictionary of scenario results
        """
        if price_shocks is None:
            price_shocks = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]  # Default shocks

        if vol_shocks is None:
            vol_shocks = [-0.5, -0.25, 0.25, 0.5]  # Vol shocks in percentage points

        scenarios = {}

        # Price shock scenarios
        for shock in price_shocks:
            scenario_name = f"price_shock_{shock:+.1%}"
            scenario_greeks = self._calculate_scenario_greeks(price_shock=shock)
            scenarios[scenario_name] = scenario_greeks

        # Volatility shock scenarios
        for shock in vol_shocks:
            scenario_name = f"vol_shock_{shock:+.2f}"
            scenario_greeks = self._calculate_scenario_greeks(vol_shock=shock)
            scenarios[scenario_name] = scenario_greeks

        return scenarios

    def _calculate_scenario_greeks(self,
                                  price_shock: float = 0.0,
                                  vol_shock: float = 0.0) -> PortfolioGreeks:
        """Calculate portfolio Greeks under specific scenario"""
        # This is a simplified implementation
        # In practice, you'd recalculate Greeks with shocked parameters

        current_greeks = self.calculate_portfolio_greeks()

        # Approximate impact (this could be more sophisticated)
        shocked_delta = current_greeks.total_delta
        shocked_gamma = current_greeks.total_gamma
        shocked_theta = current_greeks.total_theta
        shocked_vega = current_greeks.total_vega + vol_shock * abs(current_greeks.total_vega)
        shocked_rho = current_greeks.total_rho

        # Price shock affects delta through gamma
        if price_shock != 0:
            # Simplified: delta change = gamma * price_shock * spot_price
            avg_spot = np.mean(list(self.spot_prices.values())) if self.spot_prices else 100.0
            delta_change = current_greeks.total_gamma * price_shock * avg_spot
            shocked_delta += delta_change

        return PortfolioGreeks(
            total_delta=shocked_delta,
            total_gamma=shocked_gamma,
            total_theta=shocked_theta,
            total_vega=shocked_vega,
            total_rho=shocked_rho,
            calculation_time=datetime.now()
        )

    def get_historical_greeks(self,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> List[PortfolioGreeks]:
        """Get historical Greeks within time range"""
        if start_time is None and end_time is None:
            return self.historical_greeks

        filtered_greeks = []
        for greeks in self.historical_greeks:
            if start_time and greeks.calculation_time < start_time:
                continue
            if end_time and greeks.calculation_time > end_time:
                continue
            filtered_greeks.append(greeks)

        return filtered_greeks

    def calculate_greeks_pnl(self,
                           current_prices: Dict[str, float],
                           time_decay_days: float = 0.0) -> Dict[str, float]:
        """
        Calculate P&L attribution from Greeks

        Args:
            current_prices: Dictionary of underlying -> current price
            time_decay_days: Number of days that have passed (for theta)

        Returns:
            Dictionary of P&L by Greek
        """
        portfolio_greeks = self.calculate_portfolio_greeks()
        pnl_attribution = {
            'delta_pnl': 0.0,
            'gamma_pnl': 0.0,
            'theta_pnl': 0.0,
            'vega_pnl': 0.0,
            'total_pnl': 0.0
        }

        for underlying, current_price in current_prices.items():
            if underlying not in self.spot_prices:
                continue

            price_change = current_price - self.spot_prices[underlying]
            price_change_pct = price_change / self.spot_prices[underlying]

            # Get Greeks for this underlying
            underlying_greeks = portfolio_greeks.greeks_by_underlying.get(underlying)
            if not underlying_greeks:
                continue

            # Calculate P&L components
            delta_pnl = underlying_greeks.delta * price_change
            gamma_pnl = 0.5 * underlying_greeks.gamma * (price_change ** 2)
            theta_pnl = underlying_greeks.theta * time_decay_days

            # Aggregate
            pnl_attribution['delta_pnl'] += delta_pnl
            pnl_attribution['gamma_pnl'] += gamma_pnl
            pnl_attribution['theta_pnl'] += theta_pnl

        pnl_attribution['total_pnl'] = (pnl_attribution['delta_pnl'] +
                                       pnl_attribution['gamma_pnl'] +
                                       pnl_attribution['theta_pnl'])

        return pnl_attribution

    def __len__(self) -> int:
        """Return number of positions"""
        return len(self.positions)

    def __repr__(self) -> str:
        greeks = self.calculate_portfolio_greeks()
        return (f"PortfolioGreeksMonitor(positions={len(self.positions)}, "
                f"delta={greeks.total_delta:.2f}, gamma={greeks.total_gamma:.2f}, "
                f"theta={greeks.total_theta:.2f}, vega={greeks.total_vega:.2f})")