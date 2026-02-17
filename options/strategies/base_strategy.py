"""
Base Options Strategy Framework

Provides the foundation for all options strategies with:
- Common interface and methods
- P&L calculation and risk metrics
- Greeks aggregation
- Strategy validation and analysis
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
from decimal import Decimal, getcontext
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging

from ..core.option_chain import OptionContract
from ..core.greeks_calculator import Greeks, GreeksCalculator

# Set decimal precision for financial calculations
getcontext().prec = 28

logger = logging.getLogger(__name__)


@dataclass
class StrategyLeg:
    """Individual leg of an options strategy"""
    contract: OptionContract
    quantity: int  # Positive for long, negative for short
    entry_price: Decimal
    leg_type: str  # 'call', 'put', 'stock'

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of the leg"""
        return abs(self.quantity) * self.entry_price * 100  # Options are per 100 shares

    @property
    def is_long(self) -> bool:
        """Check if this is a long position"""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if this is a short position"""
        return self.quantity < 0


@dataclass
class StrategyMetrics:
    """Risk and return metrics for an options strategy"""

    # P&L metrics
    max_profit: Optional[Decimal] = None
    max_loss: Optional[Decimal] = None
    breakeven_points: List[Decimal] = field(default_factory=list)
    probability_of_profit: Optional[float] = None

    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Risk metrics
    margin_requirement: Optional[Decimal] = None
    capital_requirement: Optional[Decimal] = None
    return_on_capital: Optional[float] = None
    sharpe_ratio: Optional[float] = None

    # Time decay metrics
    theta_decay_1d: Optional[Decimal] = None
    theta_decay_7d: Optional[Decimal] = None
    theta_decay_30d: Optional[Decimal] = None

    # Volatility metrics
    vega_1pct: Optional[Decimal] = None
    iv_break_even: Optional[float] = None

    # Additional metrics
    days_to_expiry: Optional[int] = None
    moneyness: Optional[float] = None
    implied_volatility: Optional[float] = None


class BaseOptionsStrategy(ABC):
    """
    Abstract base class for all options strategies

    Provides common functionality for strategy construction, analysis,
    and risk management.
    """

    def __init__(self, symbol: str, strategy_name: str):
        """
        Initialize base strategy

        Args:
            symbol: Underlying symbol
            strategy_name: Name of the strategy
        """
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.legs: List[StrategyLeg] = []
        self.entry_date = datetime.now()
        self.target_dte: Optional[int] = None  # Days to expiry target
        self.greeks_calc = GreeksCalculator()

        # Strategy state
        self.is_constructed = False
        self.spot_price_at_entry: Optional[Decimal] = None
        self.iv_at_entry: Optional[float] = None

        # Risk management parameters
        self.max_loss_threshold: Optional[Decimal] = None
        self.profit_target: Optional[Decimal] = None
        self.stop_loss: Optional[Decimal] = None

        logger.info(f"Initialized {strategy_name} strategy for {symbol}")

    @abstractmethod
    def construct_strategy(self, **kwargs) -> bool:
        """
        Construct the strategy by selecting appropriate contracts

        Returns:
            True if strategy constructed successfully
        """
        pass

    @abstractmethod
    def calculate_pnl(self, spot_price: Decimal, new_contracts: Optional[List[OptionContract]] = None) -> Decimal:
        """
        Calculate current P&L of the strategy

        Args:
            spot_price: Current underlying price
            new_contracts: Updated option contracts with current prices

        Returns:
            Current P&L
        """
        pass

    def add_leg(self, contract: OptionContract, quantity: int, entry_price: Optional[Decimal] = None) -> None:
        """
        Add a leg to the strategy

        Args:
            contract: Option contract
            quantity: Number of contracts (positive for long, negative for short)
            entry_price: Entry price (uses mid-price if not provided)
        """
        if entry_price is None:
            entry_price = contract.mid_price

        leg_type = contract.option_type if hasattr(contract, 'option_type') else 'stock'

        leg = StrategyLeg(
            contract=contract,
            quantity=quantity,
            entry_price=entry_price,
            leg_type=leg_type
        )

        self.legs.append(leg)
        logger.debug(f"Added {quantity} {leg_type} leg at ${entry_price}")

    def remove_leg(self, leg_index: int) -> None:
        """Remove a leg from the strategy"""
        if 0 <= leg_index < len(self.legs):
            removed_leg = self.legs.pop(leg_index)
            logger.debug(f"Removed leg: {removed_leg.leg_type}")

    def calculate_strategy_greeks(self) -> Greeks:
        """Calculate aggregate Greeks for the entire strategy"""
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0

        for leg in self.legs:
            if hasattr(leg.contract, 'delta') and leg.contract.delta is not None:
                # Weight Greeks by quantity
                total_delta += leg.contract.delta * leg.quantity
                total_gamma += (leg.contract.gamma or 0.0) * leg.quantity
                total_theta += (leg.contract.theta or 0.0) * leg.quantity
                total_vega += (leg.contract.vega or 0.0) * leg.quantity
                total_rho += (leg.contract.rho or 0.0) * leg.quantity

        return Greeks(
            delta=total_delta,
            gamma=total_gamma,
            theta=total_theta,
            vega=total_vega,
            rho=total_rho
        )

    def calculate_net_premium(self) -> Decimal:
        """Calculate net premium paid/received for the strategy"""
        net_premium = Decimal('0')

        for leg in self.legs:
            # Positive quantity = premium paid (debit)
            # Negative quantity = premium received (credit)
            premium_flow = leg.entry_price * leg.quantity
            net_premium += premium_flow

        return net_premium

    def calculate_max_profit_loss(self, spot_range: Tuple[Decimal, Decimal], num_points: int = 100) -> Dict:
        """
        Calculate maximum profit and loss across a range of spot prices

        Args:
            spot_range: (min_spot, max_spot) range to analyze
            num_points: Number of price points to analyze

        Returns:
            Dictionary with max profit, max loss, and breakeven points
        """
        min_spot, max_spot = spot_range
        spot_prices = np.linspace(float(min_spot), float(max_spot), num_points)

        pnl_values = []
        for spot in spot_prices:
            pnl = self.calculate_pnl_at_expiry(Decimal(str(spot)))
            pnl_values.append(float(pnl))

        max_profit = max(pnl_values) if pnl_values else 0
        max_loss = min(pnl_values) if pnl_values else 0

        # Find breakeven points (where P&L crosses zero)
        breakeven_points = []
        for i in range(len(pnl_values) - 1):
            if (pnl_values[i] <= 0 <= pnl_values[i + 1]) or (pnl_values[i] >= 0 >= pnl_values[i + 1]):
                # Linear interpolation to find exact breakeven
                spot_low = spot_prices[i]
                spot_high = spot_prices[i + 1]
                pnl_low = pnl_values[i]
                pnl_high = pnl_values[i + 1]

                if pnl_high != pnl_low:
                    breakeven_spot = spot_low - pnl_low * (spot_high - spot_low) / (pnl_high - pnl_low)
                    breakeven_points.append(Decimal(str(breakeven_spot)))

        return {
            'max_profit': Decimal(str(max_profit)),
            'max_loss': Decimal(str(max_loss)),
            'breakeven_points': breakeven_points
        }

    def calculate_pnl_at_expiry(self, spot_price: Decimal) -> Decimal:
        """
        Calculate P&L at expiration for a given spot price

        Args:
            spot_price: Underlying price at expiration

        Returns:
            P&L at expiration
        """
        total_pnl = Decimal('0')

        for leg in self.legs:
            if leg.leg_type in ['call', 'put']:
                # Calculate intrinsic value at expiry
                if leg.leg_type == 'call':
                    intrinsic = max(spot_price - leg.contract.strike, Decimal('0'))
                else:  # put
                    intrinsic = max(leg.contract.strike - spot_price, Decimal('0'))

                # P&L = (current value - entry price) * quantity
                leg_pnl = (intrinsic - leg.entry_price) * leg.quantity
                total_pnl += leg_pnl

            elif leg.leg_type == 'stock':
                # Stock leg P&L
                leg_pnl = (spot_price - leg.entry_price) * leg.quantity / 100  # Adjust for options scaling
                total_pnl += leg_pnl

        return total_pnl

    def calculate_strategy_metrics(self, current_spot: Decimal, iv_environment: Optional[float] = None) -> StrategyMetrics:
        """
        Calculate comprehensive strategy metrics

        Args:
            current_spot: Current underlying price
            iv_environment: Current IV environment for analysis

        Returns:
            StrategyMetrics object with all calculated metrics
        """
        # Basic P&L analysis
        spot_range = (current_spot * Decimal('0.7'), current_spot * Decimal('1.3'))
        pnl_analysis = self.calculate_max_profit_loss(spot_range)

        # Greeks
        greeks = self.calculate_strategy_greeks()

        # Time to expiry (use shortest expiry in strategy)
        min_dte = None
        for leg in self.legs:
            if hasattr(leg.contract, 'days_to_expiry'):
                if min_dte is None or leg.contract.days_to_expiry < min_dte:
                    min_dte = leg.contract.days_to_expiry

        # Calculate metrics
        net_premium = self.calculate_net_premium()

        metrics = StrategyMetrics(
            max_profit=pnl_analysis['max_profit'],
            max_loss=pnl_analysis['max_loss'],
            breakeven_points=pnl_analysis['breakeven_points'],
            delta=greeks.delta,
            gamma=greeks.gamma,
            theta=greeks.theta,
            vega=greeks.vega,
            rho=greeks.rho,
            days_to_expiry=min_dte,
            capital_requirement=abs(net_premium) if net_premium < 0 else self._estimate_margin_requirement()
        )

        # Calculate additional derived metrics
        if metrics.capital_requirement and metrics.capital_requirement > 0:
            if metrics.max_profit:
                metrics.return_on_capital = float(metrics.max_profit / metrics.capital_requirement)

        # Theta decay projections
        if greeks.theta != 0:
            metrics.theta_decay_1d = Decimal(str(greeks.theta))
            metrics.theta_decay_7d = Decimal(str(greeks.theta * 7))
            metrics.theta_decay_30d = Decimal(str(greeks.theta * 30))

        # Vega sensitivity
        if greeks.vega != 0:
            metrics.vega_1pct = Decimal(str(greeks.vega))  # Already per 1% vol change

        return metrics

    def _estimate_margin_requirement(self) -> Decimal:
        """Estimate margin requirement for short options positions"""
        margin = Decimal('0')

        for leg in self.legs:
            if leg.quantity < 0 and hasattr(leg.contract, 'strike'):  # Short option
                if leg.leg_type == 'call':
                    # Short call margin: 20% of underlying + premium - OTM amount
                    underlying_value = leg.contract.strike * 100  # Assume near ATM for estimation
                    margin_req = underlying_value * Decimal('0.2') + leg.entry_price * 100
                    otm_amount = max(leg.contract.strike - (self.spot_price_at_entry or leg.contract.strike), Decimal('0'))
                    margin_req -= otm_amount * 100
                    margin += max(margin_req, leg.entry_price * 100)

                elif leg.leg_type == 'put':
                    # Short put margin: 20% of underlying + premium - OTM amount
                    underlying_value = leg.contract.strike * 100
                    margin_req = underlying_value * Decimal('0.2') + leg.entry_price * 100
                    otm_amount = max((self.spot_price_at_entry or leg.contract.strike) - leg.contract.strike, Decimal('0'))
                    margin_req -= otm_amount * 100
                    margin += max(margin_req, leg.entry_price * 100)

        return margin

    def generate_pnl_chart_data(self, spot_range: Tuple[Decimal, Decimal], num_points: int = 50) -> pd.DataFrame:
        """
        Generate P&L chart data across a range of spot prices

        Args:
            spot_range: (min_spot, max_spot) price range
            num_points: Number of data points

        Returns:
            DataFrame with spot prices and P&L values
        """
        min_spot, max_spot = spot_range
        spot_prices = np.linspace(float(min_spot), float(max_spot), num_points)

        data = []
        for spot in spot_prices:
            spot_decimal = Decimal(str(spot))
            pnl_at_expiry = self.calculate_pnl_at_expiry(spot_decimal)
            current_pnl = self.calculate_pnl(spot_decimal)

            data.append({
                'spot_price': spot,
                'pnl_at_expiry': float(pnl_at_expiry),
                'current_pnl': float(current_pnl),
                'spot_change_pct': (spot - float(self.spot_price_at_entry or spot)) / float(self.spot_price_at_entry or spot) * 100
            })

        return pd.DataFrame(data)

    def check_exit_conditions(self, current_spot: Decimal, current_pnl: Decimal) -> Dict[str, bool]:
        """
        Check if any exit conditions are met

        Args:
            current_spot: Current underlying price
            current_pnl: Current P&L

        Returns:
            Dictionary of exit condition flags
        """
        conditions = {
            'profit_target_hit': False,
            'stop_loss_hit': False,
            'max_loss_threshold_hit': False,
            'time_decay_exit': False
        }

        # Profit target
        if self.profit_target and current_pnl >= self.profit_target:
            conditions['profit_target_hit'] = True

        # Stop loss
        if self.stop_loss and current_pnl <= -self.stop_loss:
            conditions['stop_loss_hit'] = True

        # Max loss threshold
        if self.max_loss_threshold and current_pnl <= -self.max_loss_threshold:
            conditions['max_loss_threshold_hit'] = True

        # Time-based exit (e.g., close at 50% of time to expiry)
        for leg in self.legs:
            if hasattr(leg.contract, 'days_to_expiry') and leg.contract.days_to_expiry <= 7:
                conditions['time_decay_exit'] = True
                break

        return conditions

    def get_strategy_summary(self) -> Dict:
        """Get a summary of the strategy structure and key metrics"""
        summary = {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'entry_date': self.entry_date.strftime('%Y-%m-%d'),
            'is_constructed': self.is_constructed,
            'num_legs': len(self.legs),
            'net_premium': float(self.calculate_net_premium()),
            'strategy_type': 'debit' if self.calculate_net_premium() > 0 else 'credit',
            'legs': []
        }

        for i, leg in enumerate(self.legs):
            leg_info = {
                'leg_number': i + 1,
                'contract_symbol': leg.contract.symbol if hasattr(leg.contract, 'symbol') else 'N/A',
                'strike': float(leg.contract.strike) if hasattr(leg.contract, 'strike') else 'N/A',
                'expiry': leg.contract.expiry.strftime('%Y-%m-%d') if hasattr(leg.contract, 'expiry') else 'N/A',
                'option_type': leg.leg_type,
                'quantity': leg.quantity,
                'entry_price': float(leg.entry_price),
                'position_type': 'long' if leg.quantity > 0 else 'short'
            }
            summary['legs'].append(leg_info)

        return summary

    def validate_strategy(self) -> Tuple[bool, List[str]]:
        """
        Validate the strategy construction

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check if strategy has legs
        if not self.legs:
            issues.append("Strategy has no legs")

        # Check for valid contracts
        for i, leg in enumerate(self.legs):
            if not hasattr(leg.contract, 'strike'):
                issues.append(f"Leg {i+1} has invalid contract")

            if leg.quantity == 0:
                issues.append(f"Leg {i+1} has zero quantity")

            if leg.entry_price <= 0:
                issues.append(f"Leg {i+1} has invalid entry price")

        # Check expiry alignment (if applicable)
        expiry_dates = set()
        for leg in self.legs:
            if hasattr(leg.contract, 'expiry'):
                expiry_dates.add(leg.contract.expiry.date())

        # Strategy-specific validations can be added in subclasses

        is_valid = len(issues) == 0
        return is_valid, issues

    def __repr__(self) -> str:
        return f"{self.strategy_name}({self.symbol}, legs={len(self.legs)}, premium=${self.calculate_net_premium():.2f})"