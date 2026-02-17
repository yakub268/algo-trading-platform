"""
Spread Options Strategies

Implementation of various spread strategies:
- Bull/Bear Call Spreads
- Bull/Bear Put Spreads
- Iron Condors and Iron Butterflies
- Calendar Spreads
- Diagonal Spreads
- Ratio Spreads
"""

from decimal import Decimal, getcontext
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import logging

from .base_strategy import BaseOptionsStrategy
from ..core.option_chain import OptionChain, OptionContract

# Set decimal precision
getcontext().prec = 28

logger = logging.getLogger(__name__)


class BullCallSpread(BaseOptionsStrategy):
    """
    Bull Call Spread Strategy

    Long lower strike call + Short higher strike call
    Limited profit, limited loss, bullish strategy
    """

    def __init__(self, symbol: str):
        super().__init__(symbol, "Bull Call Spread")

    def construct_strategy(self,
                          option_chain: OptionChain,
                          target_dte: int = 30,
                          width: Decimal = Decimal('5'),
                          target_delta_long: float = 0.7,
                          target_delta_short: float = 0.3,
                          **kwargs) -> bool:
        """
        Construct bull call spread

        Args:
            option_chain: Available options chain
            target_dte: Target days to expiry
            width: Strike width between long and short calls
            target_delta_long: Target delta for long call
            target_delta_short: Target delta for short call
        """
        try:
            # Find appropriate expiry
            target_expiry = self._find_target_expiry(option_chain, target_dte)
            if not target_expiry:
                return False

            # Get call contracts
            contracts = option_chain.get_contracts_by_expiry(target_expiry)
            call_contracts = [c for c in contracts if c.option_type.lower() == 'call']

            if len(call_contracts) < 2:
                logger.error("Insufficient call contracts for spread")
                return False

            # Find strikes for the spread
            long_call, short_call = self._select_spread_strikes(
                call_contracts, option_chain.spot_price, width, target_delta_long, target_delta_short
            )

            if not long_call or not short_call:
                logger.error("Could not find suitable strikes for bull call spread")
                return False

            # Validate spread
            if long_call.strike >= short_call.strike:
                logger.error("Invalid bull call spread: long strike must be lower than short strike")
                return False

            # Add legs
            self.add_leg(long_call, 1)    # Buy lower strike call
            self.add_leg(short_call, -1)  # Sell higher strike call

            self.is_constructed = True
            self.spot_price_at_entry = option_chain.spot_price
            self.target_dte = target_dte

            logger.info(f"Constructed Bull Call Spread: Long ${long_call.strike} / Short ${short_call.strike}")
            return True

        except Exception as e:
            logger.error(f"Error constructing bull call spread: {e}")
            return False

    def _find_target_expiry(self, option_chain: OptionChain, target_dte: int) -> Optional[datetime]:
        """Find the best expiry date matching target DTE"""
        best_expiry = None
        min_dte_diff = float('inf')

        for expiry in option_chain.get_expiry_dates():
            dte = (expiry.date() - datetime.now().date()).days
            dte_diff = abs(dte - target_dte)
            if dte_diff < min_dte_diff:
                min_dte_diff = dte_diff
                best_expiry = expiry

        return best_expiry

    def _select_spread_strikes(self,
                              call_contracts: List[OptionContract],
                              spot_price: Decimal,
                              width: Decimal,
                              target_delta_long: float,
                              target_delta_short: float) -> Tuple[Optional[OptionContract], Optional[OptionContract]]:
        """Select appropriate strikes for the spread"""

        # Method 1: Try to find by delta
        contracts_with_delta = [c for c in call_contracts if c.delta is not None]

        if contracts_with_delta:
            # Find long call (higher delta, lower strike)
            long_candidates = [c for c in contracts_with_delta if c.delta >= target_delta_long - 0.1]
            if long_candidates:
                long_call = min(long_candidates, key=lambda c: abs(c.delta - target_delta_long))

                # Find short call (lower delta, higher strike)
                short_candidates = [
                    c for c in contracts_with_delta
                    if c.strike >= long_call.strike + width and c.delta <= target_delta_short + 0.1
                ]
                if short_candidates:
                    short_call = min(short_candidates, key=lambda c: abs(c.delta - target_delta_short))
                    return long_call, short_call

        # Method 2: Strike-based selection (fallback)
        # Find ATM or slightly OTM for long call
        atm_strike = self._find_closest_strike(call_contracts, spot_price)
        long_call = next((c for c in call_contracts if c.strike == atm_strike), None)

        if long_call:
            # Find short call at specified width
            target_short_strike = long_call.strike + width
            short_call = self._find_closest_strike_contract(call_contracts, target_short_strike)
            return long_call, short_call

        return None, None

    def _find_closest_strike(self, contracts: List[OptionContract], target_price: Decimal) -> Decimal:
        """Find the strike closest to target price"""
        return min(contracts, key=lambda c: abs(c.strike - target_price)).strike

    def _find_closest_strike_contract(self, contracts: List[OptionContract], target_strike: Decimal) -> Optional[OptionContract]:
        """Find contract with strike closest to target"""
        return min(contracts, key=lambda c: abs(c.strike - target_strike))

    def calculate_pnl(self, spot_price: Decimal, new_contracts: Optional[List[OptionContract]] = None) -> Decimal:
        """Calculate current P&L for bull call spread"""
        if len(self.legs) != 2:
            return Decimal('0')

        total_pnl = Decimal('0')

        for leg in self.legs:
            if new_contracts:
                updated_contract = next(
                    (c for c in new_contracts if c.symbol == leg.contract.symbol), None
                )
                if updated_contract:
                    current_price = updated_contract.mid_price
                else:
                    current_price = max(spot_price - leg.contract.strike, Decimal('0'))
            else:
                current_price = max(spot_price - leg.contract.strike, Decimal('0'))

            leg_pnl = (current_price - leg.entry_price) * leg.quantity * 100
            total_pnl += leg_pnl

        return total_pnl


class BearPutSpread(BaseOptionsStrategy):
    """
    Bear Put Spread Strategy

    Long higher strike put + Short lower strike put
    Limited profit, limited loss, bearish strategy
    """

    def __init__(self, symbol: str):
        super().__init__(symbol, "Bear Put Spread")

    def construct_strategy(self,
                          option_chain: OptionChain,
                          target_dte: int = 30,
                          width: Decimal = Decimal('5'),
                          target_delta_long: float = -0.7,
                          target_delta_short: float = -0.3,
                          **kwargs) -> bool:
        """Construct bear put spread"""
        try:
            target_expiry = self._find_target_expiry(option_chain, target_dte)
            if not target_expiry:
                return False

            contracts = option_chain.get_contracts_by_expiry(target_expiry)
            put_contracts = [c for c in contracts if c.option_type.lower() == 'put']

            if len(put_contracts) < 2:
                logger.error("Insufficient put contracts for spread")
                return False

            long_put, short_put = self._select_put_spread_strikes(
                put_contracts, option_chain.spot_price, width, target_delta_long, target_delta_short
            )

            if not long_put or not short_put:
                logger.error("Could not find suitable strikes for bear put spread")
                return False

            # Validate spread
            if long_put.strike <= short_put.strike:
                logger.error("Invalid bear put spread: long strike must be higher than short strike")
                return False

            # Add legs
            self.add_leg(long_put, 1)    # Buy higher strike put
            self.add_leg(short_put, -1)  # Sell lower strike put

            self.is_constructed = True
            self.spot_price_at_entry = option_chain.spot_price
            self.target_dte = target_dte

            logger.info(f"Constructed Bear Put Spread: Long ${long_put.strike} / Short ${short_put.strike}")
            return True

        except Exception as e:
            logger.error(f"Error constructing bear put spread: {e}")
            return False

    def _find_target_expiry(self, option_chain: OptionChain, target_dte: int) -> Optional[datetime]:
        """Find the best expiry date matching target DTE"""
        best_expiry = None
        min_dte_diff = float('inf')

        for expiry in option_chain.get_expiry_dates():
            dte = (expiry.date() - datetime.now().date()).days
            dte_diff = abs(dte - target_dte)
            if dte_diff < min_dte_diff:
                min_dte_diff = dte_diff
                best_expiry = expiry

        return best_expiry

    def _select_put_spread_strikes(self,
                                  put_contracts: List[OptionContract],
                                  spot_price: Decimal,
                                  width: Decimal,
                                  target_delta_long: float,
                                  target_delta_short: float) -> Tuple[Optional[OptionContract], Optional[OptionContract]]:
        """Select appropriate strikes for the put spread"""

        contracts_with_delta = [c for c in put_contracts if c.delta is not None]

        if contracts_with_delta:
            # Find long put (more negative delta, higher strike)
            long_candidates = [c for c in contracts_with_delta if c.delta <= target_delta_long + 0.1]
            if long_candidates:
                long_put = min(long_candidates, key=lambda c: abs(c.delta - target_delta_long))

                # Find short put (less negative delta, lower strike)
                short_candidates = [
                    c for c in contracts_with_delta
                    if c.strike <= long_put.strike - width and c.delta >= target_delta_short - 0.1
                ]
                if short_candidates:
                    short_put = min(short_candidates, key=lambda c: abs(c.delta - target_delta_short))
                    return long_put, short_put

        # Strike-based fallback
        atm_strike = min(put_contracts, key=lambda c: abs(c.strike - spot_price)).strike
        long_put = next((c for c in put_contracts if c.strike == atm_strike), None)

        if long_put:
            target_short_strike = long_put.strike - width
            short_put = min(put_contracts, key=lambda c: abs(c.strike - target_short_strike))
            return long_put, short_put

        return None, None

    def calculate_pnl(self, spot_price: Decimal, new_contracts: Optional[List[OptionContract]] = None) -> Decimal:
        """Calculate current P&L for bear put spread"""
        if len(self.legs) != 2:
            return Decimal('0')

        total_pnl = Decimal('0')

        for leg in self.legs:
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

            leg_pnl = (current_price - leg.entry_price) * leg.quantity * 100
            total_pnl += leg_pnl

        return total_pnl


class IronCondor(BaseOptionsStrategy):
    """
    Iron Condor Strategy

    Sell OTM call spread + Sell OTM put spread
    Neutral strategy that profits from low volatility
    """

    def __init__(self, symbol: str):
        super().__init__(symbol, "Iron Condor")

    def construct_strategy(self,
                          option_chain: OptionChain,
                          target_dte: int = 30,
                          call_width: Decimal = Decimal('5'),
                          put_width: Decimal = Decimal('5'),
                          target_delta: float = 0.15,  # Target delta for short strikes
                          **kwargs) -> bool:
        """
        Construct iron condor

        Args:
            option_chain: Available options chain
            target_dte: Target days to expiry
            call_width: Width of call spread
            put_width: Width of put spread
            target_delta: Target delta for short strikes (typically 0.10-0.20)
        """
        try:
            target_expiry = self._find_target_expiry(option_chain, target_dte)
            if not target_expiry:
                return False

            contracts = option_chain.get_contracts_by_expiry(target_expiry)
            call_contracts = [c for c in contracts if c.option_type.lower() == 'call']
            put_contracts = [c for c in contracts if c.option_type.lower() == 'put']

            if len(call_contracts) < 2 or len(put_contracts) < 2:
                logger.error("Insufficient contracts for iron condor")
                return False

            # Select strikes for iron condor
            short_call, long_call, short_put, long_put = self._select_condor_strikes(
                call_contracts, put_contracts, option_chain.spot_price,
                call_width, put_width, target_delta
            )

            if not all([short_call, long_call, short_put, long_put]):
                logger.error("Could not find suitable strikes for iron condor")
                return False

            # Add legs (sell inside, buy outside)
            self.add_leg(short_put, -1)   # Sell OTM put
            self.add_leg(long_put, 1)     # Buy further OTM put
            self.add_leg(short_call, -1)  # Sell OTM call
            self.add_leg(long_call, 1)    # Buy further OTM call

            self.is_constructed = True
            self.spot_price_at_entry = option_chain.spot_price
            self.target_dte = target_dte

            logger.info(f"Constructed Iron Condor: "
                       f"Put spread ${long_put.strike}/${short_put.strike}, "
                       f"Call spread ${short_call.strike}/${long_call.strike}")
            return True

        except Exception as e:
            logger.error(f"Error constructing iron condor: {e}")
            return False

    def _find_target_expiry(self, option_chain: OptionChain, target_dte: int) -> Optional[datetime]:
        """Find the best expiry date matching target DTE"""
        best_expiry = None
        min_dte_diff = float('inf')

        for expiry in option_chain.get_expiry_dates():
            dte = (expiry.date() - datetime.now().date()).days
            dte_diff = abs(dte - target_dte)
            if dte_diff < min_dte_diff:
                min_dte_diff = dte_diff
                best_expiry = expiry

        return best_expiry

    def _select_condor_strikes(self,
                              call_contracts: List[OptionContract],
                              put_contracts: List[OptionContract],
                              spot_price: Decimal,
                              call_width: Decimal,
                              put_width: Decimal,
                              target_delta: float) -> Tuple[Optional[OptionContract], ...]:
        """Select the four strikes for iron condor"""

        # Find short call (OTM call with target delta)
        call_candidates = [c for c in call_contracts if c.strike > spot_price and c.delta is not None]
        if call_candidates:
            short_call = min(call_candidates, key=lambda c: abs(c.delta - target_delta))
            # Find long call
            long_call_strike = short_call.strike + call_width
            long_call = min(call_contracts, key=lambda c: abs(c.strike - long_call_strike))
        else:
            return None, None, None, None

        # Find short put (OTM put with target delta)
        put_candidates = [c for c in put_contracts if c.strike < spot_price and c.delta is not None]
        if put_candidates:
            short_put = min(put_candidates, key=lambda c: abs(c.delta + target_delta))  # Put delta is negative
            # Find long put
            long_put_strike = short_put.strike - put_width
            long_put = min(put_contracts, key=lambda c: abs(c.strike - long_put_strike))
        else:
            return None, None, None, None

        return short_call, long_call, short_put, long_put

    def calculate_pnl(self, spot_price: Decimal, new_contracts: Optional[List[OptionContract]] = None) -> Decimal:
        """Calculate current P&L for iron condor"""
        if len(self.legs) != 4:
            return Decimal('0')

        total_pnl = Decimal('0')

        for leg in self.legs:
            if new_contracts:
                updated_contract = next(
                    (c for c in new_contracts if c.symbol == leg.contract.symbol), None
                )
                if updated_contract:
                    current_price = updated_contract.mid_price
                else:
                    # Estimate based on intrinsic value
                    if leg.leg_type == 'call':
                        current_price = max(spot_price - leg.contract.strike, Decimal('0'))
                    else:  # put
                        current_price = max(leg.contract.strike - spot_price, Decimal('0'))
            else:
                # Estimate based on intrinsic value
                if leg.leg_type == 'call':
                    current_price = max(spot_price - leg.contract.strike, Decimal('0'))
                else:  # put
                    current_price = max(leg.contract.strike - spot_price, Decimal('0'))

            leg_pnl = (current_price - leg.entry_price) * leg.quantity * 100
            total_pnl += leg_pnl

        return total_pnl


class IronButterfly(BaseOptionsStrategy):
    """
    Iron Butterfly Strategy

    Sell ATM call + Sell ATM put + Buy OTM call + Buy OTM put
    Neutral strategy with higher profit potential than iron condor
    """

    def __init__(self, symbol: str):
        super().__init__(symbol, "Iron Butterfly")

    def construct_strategy(self,
                          option_chain: OptionChain,
                          target_dte: int = 30,
                          wing_width: Decimal = Decimal('5'),
                          **kwargs) -> bool:
        """
        Construct iron butterfly

        Args:
            option_chain: Available options chain
            target_dte: Target days to expiry
            wing_width: Distance from ATM to wings
        """
        try:
            target_expiry = self._find_target_expiry(option_chain, target_dte)
            if not target_expiry:
                return False

            contracts = option_chain.get_contracts_by_expiry(target_expiry)
            call_contracts = [c for c in contracts if c.option_type.lower() == 'call']
            put_contracts = [c for c in contracts if c.option_type.lower() == 'put']

            # Find ATM strikes
            atm_strike = min(call_contracts, key=lambda c: abs(c.strike - option_chain.spot_price)).strike

            # Find contracts
            atm_call = next((c for c in call_contracts if c.strike == atm_strike), None)
            atm_put = next((c for c in put_contracts if c.strike == atm_strike), None)

            otm_call_strike = atm_strike + wing_width
            otm_put_strike = atm_strike - wing_width

            otm_call = min(call_contracts, key=lambda c: abs(c.strike - otm_call_strike))
            otm_put = min(put_contracts, key=lambda c: abs(c.strike - otm_put_strike))

            if not all([atm_call, atm_put, otm_call, otm_put]):
                logger.error("Could not find all required contracts for iron butterfly")
                return False

            # Add legs
            self.add_leg(atm_call, -1)   # Sell ATM call
            self.add_leg(atm_put, -1)    # Sell ATM put
            self.add_leg(otm_call, 1)    # Buy OTM call
            self.add_leg(otm_put, 1)     # Buy OTM put

            self.is_constructed = True
            self.spot_price_at_entry = option_chain.spot_price
            self.target_dte = target_dte

            logger.info(f"Constructed Iron Butterfly centered at ${atm_strike} with ${wing_width} wings")
            return True

        except Exception as e:
            logger.error(f"Error constructing iron butterfly: {e}")
            return False

    def _find_target_expiry(self, option_chain: OptionChain, target_dte: int) -> Optional[datetime]:
        """Find the best expiry date matching target DTE"""
        best_expiry = None
        min_dte_diff = float('inf')

        for expiry in option_chain.get_expiry_dates():
            dte = (expiry.date() - datetime.now().date()).days
            dte_diff = abs(dte - target_dte)
            if dte_diff < min_dte_diff:
                min_dte_diff = dte_diff
                best_expiry = expiry

        return best_expiry

    def calculate_pnl(self, spot_price: Decimal, new_contracts: Optional[List[OptionContract]] = None) -> Decimal:
        """Calculate current P&L for iron butterfly"""
        if len(self.legs) != 4:
            return Decimal('0')

        total_pnl = Decimal('0')

        for leg in self.legs:
            if new_contracts:
                updated_contract = next(
                    (c for c in new_contracts if c.symbol == leg.contract.symbol), None
                )
                if updated_contract:
                    current_price = updated_contract.mid_price
                else:
                    if leg.leg_type == 'call':
                        current_price = max(spot_price - leg.contract.strike, Decimal('0'))
                    else:
                        current_price = max(leg.contract.strike - spot_price, Decimal('0'))
            else:
                if leg.leg_type == 'call':
                    current_price = max(spot_price - leg.contract.strike, Decimal('0'))
                else:
                    current_price = max(leg.contract.strike - spot_price, Decimal('0'))

            leg_pnl = (current_price - leg.entry_price) * leg.quantity * 100
            total_pnl += leg_pnl

        return total_pnl


class Calendar(BaseOptionsStrategy):
    """
    Calendar Spread Strategy

    Short near-term option + Long far-term option (same strike)
    Profits from time decay and volatility changes
    """

    def __init__(self, symbol: str, option_type: str = 'call'):
        strategy_name = f"Calendar Spread ({option_type.title()})"
        super().__init__(symbol, strategy_name)
        self.option_type = option_type.lower()

    def construct_strategy(self,
                          option_chain: OptionChain,
                          near_dte: int = 30,
                          far_dte: int = 60,
                          strike_selection: str = 'atm',
                          target_delta: float = 0.5,
                          **kwargs) -> bool:
        """
        Construct calendar spread

        Args:
            option_chain: Available options chain
            near_dte: Target DTE for short leg
            far_dte: Target DTE for long leg
            strike_selection: Strike selection method
            target_delta: Target delta for strike selection
        """
        try:
            # Find expiry dates
            near_expiry = self._find_target_expiry(option_chain, near_dte)
            far_expiry = self._find_target_expiry(option_chain, far_dte)

            if not near_expiry or not far_expiry or near_expiry >= far_expiry:
                logger.error("Could not find suitable expiry dates for calendar spread")
                return False

            # Get contracts for both expiries
            near_contracts = option_chain.get_contracts_by_expiry(near_expiry)
            far_contracts = option_chain.get_contracts_by_expiry(far_expiry)

            near_options = [c for c in near_contracts if c.option_type.lower() == self.option_type]
            far_options = [c for c in far_contracts if c.option_type.lower() == self.option_type]

            if not near_options or not far_options:
                logger.error(f"Insufficient {self.option_type} contracts for calendar spread")
                return False

            # Select strike
            target_strike = self._select_calendar_strike(near_options, option_chain.spot_price, strike_selection, target_delta)

            # Find matching contracts
            near_option = min(near_options, key=lambda c: abs(c.strike - target_strike))
            far_option = min(far_options, key=lambda c: abs(c.strike - target_strike))

            # Add legs
            self.add_leg(near_option, -1)  # Sell near-term
            self.add_leg(far_option, 1)    # Buy far-term

            self.is_constructed = True
            self.spot_price_at_entry = option_chain.spot_price
            self.target_dte = near_dte

            logger.info(f"Constructed Calendar Spread: {self.option_type} at ${target_strike}, {near_dte}/{far_dte} DTE")
            return True

        except Exception as e:
            logger.error(f"Error constructing calendar spread: {e}")
            return False

    def _find_target_expiry(self, option_chain: OptionChain, target_dte: int) -> Optional[datetime]:
        """Find the best expiry date matching target DTE"""
        best_expiry = None
        min_dte_diff = float('inf')

        for expiry in option_chain.get_expiry_dates():
            dte = (expiry.date() - datetime.now().date()).days
            dte_diff = abs(dte - target_dte)
            if dte_diff < min_dte_diff:
                min_dte_diff = dte_diff
                best_expiry = expiry

        return best_expiry

    def _select_calendar_strike(self,
                               contracts: List[OptionContract],
                               spot_price: Decimal,
                               strike_selection: str,
                               target_delta: float) -> Decimal:
        """Select strike for calendar spread"""

        if strike_selection == 'atm':
            return min(contracts, key=lambda c: abs(c.strike - spot_price)).strike
        elif strike_selection == 'delta':
            contracts_with_delta = [c for c in contracts if c.delta is not None]
            if contracts_with_delta:
                return min(contracts_with_delta, key=lambda c: abs(c.delta - target_delta)).strike
            else:
                return min(contracts, key=lambda c: abs(c.strike - spot_price)).strike
        else:
            return min(contracts, key=lambda c: abs(c.strike - spot_price)).strike

    def calculate_pnl(self, spot_price: Decimal, new_contracts: Optional[List[OptionContract]] = None) -> Decimal:
        """Calculate current P&L for calendar spread"""
        if len(self.legs) != 2:
            return Decimal('0')

        total_pnl = Decimal('0')

        for leg in self.legs:
            if new_contracts:
                updated_contract = next(
                    (c for c in new_contracts if c.symbol == leg.contract.symbol), None
                )
                if updated_contract:
                    current_price = updated_contract.mid_price
                else:
                    # Simplified intrinsic value calculation
                    if self.option_type == 'call':
                        current_price = max(spot_price - leg.contract.strike, Decimal('0'))
                    else:
                        current_price = max(leg.contract.strike - spot_price, Decimal('0'))
            else:
                if self.option_type == 'call':
                    current_price = max(spot_price - leg.contract.strike, Decimal('0'))
                else:
                    current_price = max(leg.contract.strike - spot_price, Decimal('0'))

            leg_pnl = (current_price - leg.entry_price) * leg.quantity * 100
            total_pnl += leg_pnl

        return total_pnl


class DiagonalSpread(BaseOptionsStrategy):
    """
    Diagonal Spread Strategy

    Different strikes and different expiry dates
    Combines aspects of vertical and calendar spreads
    """

    def __init__(self, symbol: str, option_type: str = 'call'):
        strategy_name = f"Diagonal Spread ({option_type.title()})"
        super().__init__(symbol, strategy_name)
        self.option_type = option_type.lower()

    def construct_strategy(self,
                          option_chain: OptionChain,
                          near_dte: int = 30,
                          far_dte: int = 60,
                          strike_differential: Decimal = Decimal('5'),
                          **kwargs) -> bool:
        """
        Construct diagonal spread

        Args:
            option_chain: Available options chain
            near_dte: Target DTE for short leg
            far_dte: Target DTE for long leg
            strike_differential: Strike difference between legs
        """
        try:
            # Find expiry dates
            near_expiry = self._find_target_expiry(option_chain, near_dte)
            far_expiry = self._find_target_expiry(option_chain, far_dte)

            if not near_expiry or not far_expiry or near_expiry >= far_expiry:
                logger.error("Could not find suitable expiry dates for diagonal spread")
                return False

            # Get contracts
            near_contracts = option_chain.get_contracts_by_expiry(near_expiry)
            far_contracts = option_chain.get_contracts_by_expiry(far_expiry)

            near_options = [c for c in near_contracts if c.option_type.lower() == self.option_type]
            far_options = [c for c in far_contracts if c.option_type.lower() == self.option_type]

            # Select strikes for diagonal spread
            if self.option_type == 'call':
                # For call diagonal: sell lower strike near-term, buy higher strike far-term
                near_strike = min(near_options, key=lambda c: abs(c.strike - option_chain.spot_price)).strike
                far_strike = near_strike + strike_differential
            else:
                # For put diagonal: sell higher strike near-term, buy lower strike far-term
                near_strike = min(near_options, key=lambda c: abs(c.strike - option_chain.spot_price)).strike
                far_strike = near_strike - strike_differential

            # Find matching contracts
            near_option = min(near_options, key=lambda c: abs(c.strike - near_strike))
            far_option = min(far_options, key=lambda c: abs(c.strike - far_strike))

            # Add legs
            self.add_leg(near_option, -1)  # Sell near-term
            self.add_leg(far_option, 1)    # Buy far-term

            self.is_constructed = True
            self.spot_price_at_entry = option_chain.spot_price
            self.target_dte = near_dte

            logger.info(f"Constructed Diagonal Spread: {self.option_type} ${near_strike}/{far_strike}")
            return True

        except Exception as e:
            logger.error(f"Error constructing diagonal spread: {e}")
            return False

    def _find_target_expiry(self, option_chain: OptionChain, target_dte: int) -> Optional[datetime]:
        """Find the best expiry date matching target DTE"""
        best_expiry = None
        min_dte_diff = float('inf')

        for expiry in option_chain.get_expiry_dates():
            dte = (expiry.date() - datetime.now().date()).days
            dte_diff = abs(dte - target_dte)
            if dte_diff < min_dte_diff:
                min_dte_diff = dte_diff
                best_expiry = expiry

        return best_expiry

    def calculate_pnl(self, spot_price: Decimal, new_contracts: Optional[List[OptionContract]] = None) -> Decimal:
        """Calculate current P&L for diagonal spread"""
        if len(self.legs) != 2:
            return Decimal('0')

        total_pnl = Decimal('0')

        for leg in self.legs:
            if new_contracts:
                updated_contract = next(
                    (c for c in new_contracts if c.symbol == leg.contract.symbol), None
                )
                if updated_contract:
                    current_price = updated_contract.mid_price
                else:
                    if self.option_type == 'call':
                        current_price = max(spot_price - leg.contract.strike, Decimal('0'))
                    else:
                        current_price = max(leg.contract.strike - spot_price, Decimal('0'))
            else:
                if self.option_type == 'call':
                    current_price = max(spot_price - leg.contract.strike, Decimal('0'))
                else:
                    current_price = max(leg.contract.strike - spot_price, Decimal('0'))

            leg_pnl = (current_price - leg.entry_price) * leg.quantity * 100
            total_pnl += leg_pnl

        return total_pnl