"""
Basic Options Strategies

Implementation of fundamental options strategies:
- Long/Short Calls
- Long/Short Puts
- Covered Calls
- Cash-Secured Puts
- Protective Puts
"""

from decimal import Decimal, getcontext
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import logging

from .base_strategy import BaseOptionsStrategy, StrategyLeg
from ..core.option_chain import OptionChain, OptionContract

# Set decimal precision
getcontext().prec = 28

logger = logging.getLogger(__name__)


class CallStrategy(BaseOptionsStrategy):
    """
    Long or Short Call Strategy

    Long Call: Bullish strategy with unlimited upside, limited downside
    Short Call: Bearish/neutral strategy with limited upside, unlimited downside
    """

    def __init__(self, symbol: str, is_long: bool = True):
        strategy_name = f"{'Long' if is_long else 'Short'} Call"
        super().__init__(symbol, strategy_name)
        self.is_long = is_long
        self.target_delta = 0.5 if is_long else -0.3  # Default target deltas

    def construct_strategy(self,
                          option_chain: OptionChain,
                          target_dte: int = 30,
                          strike_selection: str = 'atm',  # 'atm', 'otm', 'itm'
                          target_delta: Optional[float] = None,
                          **kwargs) -> bool:
        """
        Construct call strategy

        Args:
            option_chain: Available options chain
            target_dte: Target days to expiry
            strike_selection: Strike selection method
            target_delta: Target delta for contract selection

        Returns:
            True if strategy constructed successfully
        """
        try:
            # Find appropriate expiry
            target_expiry = None
            min_dte_diff = float('inf')

            for expiry in option_chain.get_expiry_dates():
                dte = (expiry.date() - datetime.now().date()).days
                if abs(dte - target_dte) < min_dte_diff:
                    min_dte_diff = abs(dte - target_dte)
                    target_expiry = expiry

            if not target_expiry:
                logger.error("No suitable expiry found")
                return False

            # Get call contracts for the expiry
            contracts = option_chain.get_contracts_by_expiry(target_expiry)
            call_contracts = [c for c in contracts if c.option_type.lower() == 'call']

            if not call_contracts:
                logger.error("No call contracts found")
                return False

            # Select contract based on criteria
            selected_contract = self._select_contract(
                call_contracts, option_chain.spot_price, strike_selection, target_delta
            )

            if not selected_contract:
                logger.error("Could not select suitable contract")
                return False

            # Add the leg
            quantity = 1 if self.is_long else -1
            self.add_leg(selected_contract, quantity)

            self.is_constructed = True
            self.spot_price_at_entry = option_chain.spot_price
            self.target_dte = target_dte

            logger.info(f"Constructed {self.strategy_name} at strike ${selected_contract.strike}")
            return True

        except Exception as e:
            logger.error(f"Error constructing call strategy: {e}")
            return False

    def _select_contract(self,
                        call_contracts: List[OptionContract],
                        spot_price: Decimal,
                        strike_selection: str,
                        target_delta: Optional[float]) -> Optional[OptionContract]:
        """Select the most appropriate call contract"""

        if target_delta is not None:
            # Select by delta
            best_contract = None
            min_delta_diff = float('inf')

            for contract in call_contracts:
                if contract.delta is not None:
                    delta_diff = abs(contract.delta - target_delta)
                    if delta_diff < min_delta_diff:
                        min_delta_diff = delta_diff
                        best_contract = contract

            return best_contract

        # Select by strike relative to spot
        if strike_selection == 'atm':
            # Find closest to at-the-money
            best_contract = None
            min_strike_diff = float('inf')

            for contract in call_contracts:
                strike_diff = abs(contract.strike - spot_price)
                if strike_diff < min_strike_diff:
                    min_strike_diff = strike_diff
                    best_contract = contract

            return best_contract

        elif strike_selection == 'otm':
            # Find first out-of-the-money strike
            otm_contracts = [c for c in call_contracts if c.strike > spot_price]
            if otm_contracts:
                return min(otm_contracts, key=lambda c: c.strike)

        elif strike_selection == 'itm':
            # Find first in-the-money strike
            itm_contracts = [c for c in call_contracts if c.strike < spot_price]
            if itm_contracts:
                return max(itm_contracts, key=lambda c: c.strike)

        # Fallback to ATM
        return min(call_contracts, key=lambda c: abs(c.strike - spot_price))

    def calculate_pnl(self, spot_price: Decimal, new_contracts: Optional[List[OptionContract]] = None) -> Decimal:
        """Calculate current P&L"""
        if not self.legs:
            return Decimal('0')

        leg = self.legs[0]  # Single leg strategy

        if new_contracts:
            # Use updated contract price
            updated_contract = next(
                (c for c in new_contracts if c.symbol == leg.contract.symbol), None
            )
            if updated_contract:
                current_price = updated_contract.mid_price
            else:
                # Estimate based on intrinsic value
                current_price = max(spot_price - leg.contract.strike, Decimal('0'))
        else:
            # Estimate based on intrinsic value (simplified)
            current_price = max(spot_price - leg.contract.strike, Decimal('0'))

        # P&L = (current_price - entry_price) * quantity * 100
        pnl = (current_price - leg.entry_price) * leg.quantity * 100
        return pnl


class PutStrategy(BaseOptionsStrategy):
    """
    Long or Short Put Strategy

    Long Put: Bearish strategy with unlimited downside, limited upside
    Short Put: Bullish/neutral strategy with limited downside, unlimited upside
    """

    def __init__(self, symbol: str, is_long: bool = True):
        strategy_name = f"{'Long' if is_long else 'Short'} Put"
        super().__init__(symbol, strategy_name)
        self.is_long = is_long
        self.target_delta = -0.5 if is_long else 0.3  # Default target deltas

    def construct_strategy(self,
                          option_chain: OptionChain,
                          target_dte: int = 30,
                          strike_selection: str = 'atm',  # 'atm', 'otm', 'itm'
                          target_delta: Optional[float] = None,
                          **kwargs) -> bool:
        """Construct put strategy - similar logic to CallStrategy"""
        try:
            # Find appropriate expiry (same logic as call)
            target_expiry = None
            min_dte_diff = float('inf')

            for expiry in option_chain.get_expiry_dates():
                dte = (expiry.date() - datetime.now().date()).days
                if abs(dte - target_dte) < min_dte_diff:
                    min_dte_diff = abs(dte - target_dte)
                    target_expiry = expiry

            if not target_expiry:
                logger.error("No suitable expiry found")
                return False

            # Get put contracts
            contracts = option_chain.get_contracts_by_expiry(target_expiry)
            put_contracts = [c for c in contracts if c.option_type.lower() == 'put']

            if not put_contracts:
                logger.error("No put contracts found")
                return False

            # Select contract
            selected_contract = self._select_put_contract(
                put_contracts, option_chain.spot_price, strike_selection, target_delta
            )

            if not selected_contract:
                logger.error("Could not select suitable contract")
                return False

            # Add the leg
            quantity = 1 if self.is_long else -1
            self.add_leg(selected_contract, quantity)

            self.is_constructed = True
            self.spot_price_at_entry = option_chain.spot_price
            self.target_dte = target_dte

            logger.info(f"Constructed {self.strategy_name} at strike ${selected_contract.strike}")
            return True

        except Exception as e:
            logger.error(f"Error constructing put strategy: {e}")
            return False

    def _select_put_contract(self,
                           put_contracts: List[OptionContract],
                           spot_price: Decimal,
                           strike_selection: str,
                           target_delta: Optional[float]) -> Optional[OptionContract]:
        """Select the most appropriate put contract"""

        if target_delta is not None:
            # Select by delta
            best_contract = None
            min_delta_diff = float('inf')

            for contract in put_contracts:
                if contract.delta is not None:
                    delta_diff = abs(contract.delta - target_delta)
                    if delta_diff < min_delta_diff:
                        min_delta_diff = delta_diff
                        best_contract = contract

            return best_contract

        # Select by strike relative to spot (opposite of calls)
        if strike_selection == 'atm':
            # Find closest to at-the-money
            return min(put_contracts, key=lambda c: abs(c.strike - spot_price))

        elif strike_selection == 'otm':
            # For puts, OTM means strike < spot
            otm_contracts = [c for c in put_contracts if c.strike < spot_price]
            if otm_contracts:
                return max(otm_contracts, key=lambda c: c.strike)

        elif strike_selection == 'itm':
            # For puts, ITM means strike > spot
            itm_contracts = [c for c in put_contracts if c.strike > spot_price]
            if itm_contracts:
                return min(itm_contracts, key=lambda c: c.strike)

        # Fallback to ATM
        return min(put_contracts, key=lambda c: abs(c.strike - spot_price))

    def calculate_pnl(self, spot_price: Decimal, new_contracts: Optional[List[OptionContract]] = None) -> Decimal:
        """Calculate current P&L"""
        if not self.legs:
            return Decimal('0')

        leg = self.legs[0]  # Single leg strategy

        if new_contracts:
            # Use updated contract price
            updated_contract = next(
                (c for c in new_contracts if c.symbol == leg.contract.symbol), None
            )
            if updated_contract:
                current_price = updated_contract.mid_price
            else:
                # Estimate based on intrinsic value
                current_price = max(leg.contract.strike - spot_price, Decimal('0'))
        else:
            # Estimate based on intrinsic value (simplified)
            current_price = max(leg.contract.strike - spot_price, Decimal('0'))

        # P&L = (current_price - entry_price) * quantity * 100
        pnl = (current_price - leg.entry_price) * leg.quantity * 100
        return pnl


class CoveredCall(BaseOptionsStrategy):
    """
    Covered Call Strategy

    Long stock + Short call
    Income-generating strategy with limited upside, limited downside protection
    """

    def __init__(self, symbol: str):
        super().__init__(symbol, "Covered Call")
        self.shares_per_contract = 100  # Standard option contract size

    def construct_strategy(self,
                          option_chain: OptionChain,
                          stock_price: Decimal,
                          target_dte: int = 30,
                          strike_selection: str = 'otm',
                          target_delta: float = 0.3,
                          num_contracts: int = 1,
                          **kwargs) -> bool:
        """
        Construct covered call strategy

        Args:
            option_chain: Available options chain
            stock_price: Current stock price (assuming we own stock)
            target_dte: Target days to expiry
            strike_selection: 'otm' or delta-based
            target_delta: Target delta for call (typically 0.2-0.4)
            num_contracts: Number of covered call contracts
        """
        try:
            # Find appropriate expiry
            target_expiry = None
            min_dte_diff = float('inf')

            for expiry in option_chain.get_expiry_dates():
                dte = (expiry.date() - datetime.now().date()).days
                if abs(dte - target_dte) < min_dte_diff:
                    min_dte_diff = abs(dte - target_dte)
                    target_expiry = expiry

            if not target_expiry:
                logger.error("No suitable expiry found")
                return False

            # Get call contracts
            contracts = option_chain.get_contracts_by_expiry(target_expiry)
            call_contracts = [c for c in contracts if c.option_type.lower() == 'call']

            if not call_contracts:
                logger.error("No call contracts found")
                return False

            # Select OTM call with appropriate delta
            selected_call = None

            if strike_selection == 'otm':
                # Find calls with strike > current price and target delta
                otm_calls = [c for c in call_contracts if c.strike > stock_price]
                if otm_calls:
                    # Select by delta if available, otherwise closest strike
                    if any(c.delta is not None for c in otm_calls):
                        selected_call = min(
                            [c for c in otm_calls if c.delta is not None],
                            key=lambda c: abs(c.delta - target_delta)
                        )
                    else:
                        selected_call = min(otm_calls, key=lambda c: c.strike)
            else:
                # Select by delta regardless of moneyness
                if any(c.delta is not None for c in call_contracts):
                    selected_call = min(
                        [c for c in call_contracts if c.delta is not None],
                        key=lambda c: abs(c.delta - target_delta)
                    )

            if not selected_call:
                logger.error("Could not find suitable call to sell")
                return False

            # Add stock leg (long stock)
            # Note: This is conceptual - in practice, you'd already own the stock
            stock_contract = type('StockContract', (), {
                'symbol': f"{self.symbol}_STOCK",
                'strike': stock_price,
                'expiry': datetime.now() + timedelta(days=365),  # No expiry for stock
                'option_type': 'stock'
            })()

            self.add_leg(stock_contract, num_contracts * self.shares_per_contract, stock_price)

            # Add short call leg
            self.add_leg(selected_call, -num_contracts)

            self.is_constructed = True
            self.spot_price_at_entry = stock_price
            self.target_dte = target_dte

            logger.info(f"Constructed Covered Call: Long {num_contracts * self.shares_per_contract} shares, Short {num_contracts} calls at ${selected_call.strike}")
            return True

        except Exception as e:
            logger.error(f"Error constructing covered call: {e}")
            return False

    def calculate_pnl(self, spot_price: Decimal, new_contracts: Optional[List[OptionContract]] = None) -> Decimal:
        """Calculate current P&L for covered call"""
        if len(self.legs) != 2:
            return Decimal('0')

        total_pnl = Decimal('0')

        for leg in self.legs:
            if leg.leg_type == 'stock':
                # Stock P&L
                stock_pnl = (spot_price - leg.entry_price) * leg.quantity
                total_pnl += stock_pnl

            elif leg.leg_type == 'call':
                # Call P&L
                if new_contracts:
                    updated_contract = next(
                        (c for c in new_contracts if c.symbol == leg.contract.symbol), None
                    )
                    if updated_contract:
                        current_call_price = updated_contract.mid_price
                    else:
                        current_call_price = max(spot_price - leg.contract.strike, Decimal('0'))
                else:
                    current_call_price = max(spot_price - leg.contract.strike, Decimal('0'))

                call_pnl = (current_call_price - leg.entry_price) * leg.quantity * 100
                total_pnl += call_pnl

        return total_pnl


class CashSecuredPut(BaseOptionsStrategy):
    """
    Cash-Secured Put Strategy

    Short put + cash to secure the put
    Income-generating strategy used to acquire stock at a discount
    """

    def __init__(self, symbol: str):
        super().__init__(symbol, "Cash-Secured Put")

    def construct_strategy(self,
                          option_chain: OptionChain,
                          target_dte: int = 30,
                          strike_selection: str = 'otm',
                          target_delta: float = -0.3,
                          num_contracts: int = 1,
                          **kwargs) -> bool:
        """
        Construct cash-secured put strategy

        Args:
            option_chain: Available options chain
            target_dte: Target days to expiry
            strike_selection: Strike selection method
            target_delta: Target delta (typically -0.2 to -0.4)
            num_contracts: Number of contracts
        """
        try:
            # Find appropriate expiry
            target_expiry = None
            min_dte_diff = float('inf')

            for expiry in option_chain.get_expiry_dates():
                dte = (expiry.date() - datetime.now().date()).days
                if abs(dte - target_dte) < min_dte_diff:
                    min_dte_diff = abs(dte - target_dte)
                    target_expiry = expiry

            if not target_expiry:
                logger.error("No suitable expiry found")
                return False

            # Get put contracts
            contracts = option_chain.get_contracts_by_expiry(target_expiry)
            put_contracts = [c for c in contracts if c.option_type.lower() == 'put']

            if not put_contracts:
                logger.error("No put contracts found")
                return False

            # Select OTM put with appropriate delta
            selected_put = None

            if strike_selection == 'otm':
                # Find puts with strike < current price
                otm_puts = [c for c in put_contracts if c.strike < option_chain.spot_price]
                if otm_puts:
                    if any(c.delta is not None for c in otm_puts):
                        selected_put = min(
                            [c for c in otm_puts if c.delta is not None],
                            key=lambda c: abs(c.delta - target_delta)
                        )
                    else:
                        selected_put = max(otm_puts, key=lambda c: c.strike)
            else:
                # Select by delta
                if any(c.delta is not None for c in put_contracts):
                    selected_put = min(
                        [c for c in put_contracts if c.delta is not None],
                        key=lambda c: abs(c.delta - target_delta)
                    )

            if not selected_put:
                logger.error("Could not find suitable put to sell")
                return False

            # Add short put leg
            self.add_leg(selected_put, -num_contracts)

            # Calculate cash requirement
            cash_requirement = selected_put.strike * 100 * num_contracts

            self.is_constructed = True
            self.spot_price_at_entry = option_chain.spot_price
            self.target_dte = target_dte

            # Store cash requirement for margin calculations
            self.cash_requirement = cash_requirement

            logger.info(f"Constructed Cash-Secured Put: Short {num_contracts} puts at ${selected_put.strike}, Cash required: ${cash_requirement}")
            return True

        except Exception as e:
            logger.error(f"Error constructing cash-secured put: {e}")
            return False

    def calculate_pnl(self, spot_price: Decimal, new_contracts: Optional[List[OptionContract]] = None) -> Decimal:
        """Calculate current P&L for cash-secured put"""
        if not self.legs:
            return Decimal('0')

        leg = self.legs[0]  # Single leg strategy

        if new_contracts:
            updated_contract = next(
                (c for c in new_contracts if c.symbol == leg.contract.symbol), None
            )
            if updated_contract:
                current_price = updated_contract.mid_price
            else:
                current_price = max(leg.contract.strike - spot_price, Decimal('0'))
        else:
            current_price = max(leg.contract.strike - spot_price, Decimal('0'))

        # P&L = (current_price - entry_price) * quantity * 100
        # For short put, quantity is negative
        pnl = (current_price - leg.entry_price) * leg.quantity * 100
        return pnl