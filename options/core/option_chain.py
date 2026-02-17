"""
Options Chain Data Management

Handles real-time options chain data, contract specifications,
and provides efficient access to options data for analysis and trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

# Set decimal precision for financial calculations
getcontext().prec = 28

logger = logging.getLogger(__name__)


@dataclass
class OptionContract:
    """Individual options contract data structure"""

    symbol: str
    underlying: str
    strike: Decimal
    expiry: datetime
    option_type: str  # 'call' or 'put'

    # Market data
    bid: Decimal
    ask: Decimal
    last_price: Decimal
    volume: int
    open_interest: int

    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    # Volatility
    implied_volatility: Optional[float] = None
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None

    # Additional metrics
    intrinsic_value: Optional[Decimal] = None
    time_value: Optional[Decimal] = None
    moneyness: Optional[float] = None

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid-point of bid-ask spread"""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread"""
        return self.ask - self.bid

    @property
    def spread_percentage(self) -> float:
        """Calculate spread as percentage of mid price"""
        mid = self.mid_price
        if mid > 0:
            return float(self.spread / mid * 100)
        return 0.0

    @property
    def days_to_expiry(self) -> int:
        """Calculate days until expiration"""
        return (self.expiry.date() - datetime.now().date()).days

    @property
    def time_to_expiry(self) -> float:
        """Calculate time to expiry as fraction of year"""
        return self.days_to_expiry / 365.0

    def is_itm(self, spot_price: Decimal) -> bool:
        """Check if option is in-the-money"""
        if self.option_type.lower() == 'call':
            return spot_price > self.strike
        else:  # put
            return spot_price < self.strike

    def is_otm(self, spot_price: Decimal) -> bool:
        """Check if option is out-of-the-money"""
        return not self.is_itm(spot_price)

    def calculate_moneyness(self, spot_price: Decimal) -> float:
        """Calculate moneyness (S/K for calls, K/S for puts)"""
        if self.option_type.lower() == 'call':
            return float(spot_price / self.strike)
        else:  # put
            return float(self.strike / spot_price)

    def calculate_intrinsic_value(self, spot_price: Decimal) -> Decimal:
        """Calculate intrinsic value of the option"""
        if self.option_type.lower() == 'call':
            return max(spot_price - self.strike, Decimal('0'))
        else:  # put
            return max(self.strike - spot_price, Decimal('0'))

    def calculate_time_value(self, spot_price: Decimal) -> Decimal:
        """Calculate time value (option price - intrinsic value)"""
        intrinsic = self.calculate_intrinsic_value(spot_price)
        return self.mid_price - intrinsic

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'underlying': self.underlying,
            'strike': float(self.strike),
            'expiry': self.expiry.isoformat(),
            'option_type': self.option_type,
            'bid': float(self.bid),
            'ask': float(self.ask),
            'last_price': float(self.last_price),
            'volume': self.volume,
            'open_interest': self.open_interest,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
            'implied_volatility': self.implied_volatility,
            'iv_rank': self.iv_rank,
            'iv_percentile': self.iv_percentile,
            'mid_price': float(self.mid_price),
            'spread': float(self.spread),
            'spread_percentage': self.spread_percentage,
            'days_to_expiry': self.days_to_expiry,
            'time_to_expiry': self.time_to_expiry
        }


class OptionChain:
    """
    Complete options chain for an underlying asset

    Provides efficient access to options data, filtering capabilities,
    and analysis tools for options strategies.
    """

    def __init__(self, underlying: str, spot_price: Decimal):
        self.underlying = underlying
        self.spot_price = spot_price
        self.last_updated = datetime.now()
        self.contracts: Dict[str, OptionContract] = {}

        # Organized views
        self._by_expiry: Dict[str, List[OptionContract]] = {}
        self._by_strike: Dict[Decimal, Dict[str, OptionContract]] = {}

        logger.info(f"Initialized option chain for {underlying} at ${spot_price}")

    def add_contract(self, contract: OptionContract) -> None:
        """Add a contract to the chain"""
        # Update contract metrics
        contract.intrinsic_value = contract.calculate_intrinsic_value(self.spot_price)
        contract.time_value = contract.calculate_time_value(self.spot_price)
        contract.moneyness = contract.calculate_moneyness(self.spot_price)

        # Store contract
        self.contracts[contract.symbol] = contract

        # Update organized views
        expiry_str = contract.expiry.strftime('%Y-%m-%d')
        if expiry_str not in self._by_expiry:
            self._by_expiry[expiry_str] = []
        self._by_expiry[expiry_str].append(contract)

        if contract.strike not in self._by_strike:
            self._by_strike[contract.strike] = {}
        self._by_strike[contract.strike][contract.option_type] = contract

    def get_contract(self, symbol: str) -> Optional[OptionContract]:
        """Get contract by symbol"""
        return self.contracts.get(symbol)

    def get_expiry_dates(self) -> List[datetime]:
        """Get all available expiry dates"""
        return sorted([datetime.strptime(d, '%Y-%m-%d') for d in self._by_expiry.keys()])

    def get_strikes(self, expiry: Optional[datetime] = None) -> List[Decimal]:
        """Get all strikes, optionally filtered by expiry"""
        if expiry is None:
            return sorted(self._by_strike.keys())

        expiry_str = expiry.strftime('%Y-%m-%d')
        if expiry_str not in self._by_expiry:
            return []

        return sorted([c.strike for c in self._by_expiry[expiry_str]])

    def get_contracts_by_expiry(self, expiry: datetime) -> List[OptionContract]:
        """Get all contracts for a specific expiry"""
        expiry_str = expiry.strftime('%Y-%m-%d')
        return self._by_expiry.get(expiry_str, [])

    def get_contracts_by_strike(self, strike: Decimal) -> Dict[str, OptionContract]:
        """Get call/put contracts for a specific strike"""
        return self._by_strike.get(strike, {})

    def get_call_put_pair(self, strike: Decimal, expiry: datetime) -> Tuple[Optional[OptionContract], Optional[OptionContract]]:
        """Get call/put pair for specific strike and expiry"""
        contracts = self.get_contracts_by_expiry(expiry)
        strike_contracts = [c for c in contracts if c.strike == strike]

        call = next((c for c in strike_contracts if c.option_type.lower() == 'call'), None)
        put = next((c for c in strike_contracts if c.option_type.lower() == 'put'), None)

        return call, put

    def get_atm_contracts(self, expiry: datetime, tolerance: float = 0.02) -> List[OptionContract]:
        """Get at-the-money contracts within tolerance"""
        contracts = self.get_contracts_by_expiry(expiry)
        atm_contracts = []

        for contract in contracts:
            if abs(contract.moneyness - 1.0) <= tolerance:
                atm_contracts.append(contract)

        return atm_contracts

    def get_otm_contracts(self, expiry: datetime, min_delta: float = 0.05, max_delta: float = 0.45) -> List[OptionContract]:
        """Get out-of-the-money contracts within delta range"""
        contracts = self.get_contracts_by_expiry(expiry)
        otm_contracts = []

        for contract in contracts:
            if contract.delta is not None:
                abs_delta = abs(contract.delta)
                if min_delta <= abs_delta <= max_delta:
                    otm_contracts.append(contract)

        return otm_contracts

    def get_high_volume_contracts(self, min_volume: int = 100) -> List[OptionContract]:
        """Get contracts with high volume"""
        return [c for c in self.contracts.values() if c.volume >= min_volume]

    def get_high_oi_contracts(self, min_oi: int = 500) -> List[OptionContract]:
        """Get contracts with high open interest"""
        return [c for c in self.contracts.values() if c.open_interest >= min_oi]

    def get_liquid_contracts(self, max_spread_pct: float = 5.0) -> List[OptionContract]:
        """Get liquid contracts (tight bid-ask spreads)"""
        return [c for c in self.contracts.values() if c.spread_percentage <= max_spread_pct]

    def get_contracts_by_iv_rank(self, min_rank: float, max_rank: float) -> List[OptionContract]:
        """Get contracts within IV rank range"""
        contracts = []
        for contract in self.contracts.values():
            if contract.iv_rank is not None:
                if min_rank <= contract.iv_rank <= max_rank:
                    contracts.append(contract)
        return contracts

    def get_earnings_contracts(self, earnings_date: datetime, buffer_days: int = 7) -> List[OptionContract]:
        """Get contracts expiring around earnings"""
        contracts = []
        for contract in self.contracts.values():
            days_to_earnings = (earnings_date.date() - datetime.now().date()).days
            days_to_expiry = contract.days_to_expiry

            if abs(days_to_expiry - days_to_earnings) <= buffer_days:
                contracts.append(contract)

        return contracts

    def to_dataframe(self) -> pd.DataFrame:
        """Convert entire chain to DataFrame"""
        data = []
        for contract in self.contracts.values():
            data.append(contract.to_dict())

        return pd.DataFrame(data)

    def get_summary_stats(self) -> Dict:
        """Get summary statistics for the options chain"""
        contracts = list(self.contracts.values())

        if not contracts:
            return {}

        total_volume = sum(c.volume for c in contracts)
        total_oi = sum(c.open_interest for c in contracts)
        avg_iv = np.mean([c.implied_volatility for c in contracts if c.implied_volatility is not None])

        call_contracts = [c for c in contracts if c.option_type.lower() == 'call']
        put_contracts = [c for c in contracts if c.option_type.lower() == 'put']

        call_volume = sum(c.volume for c in call_contracts)
        put_volume = sum(c.volume for c in put_contracts)

        call_oi = sum(c.open_interest for c in call_contracts)
        put_oi = sum(c.open_interest for c in put_contracts)

        return {
            'underlying': self.underlying,
            'spot_price': float(self.spot_price),
            'total_contracts': len(contracts),
            'call_contracts': len(call_contracts),
            'put_contracts': len(put_contracts),
            'total_volume': total_volume,
            'call_volume': call_volume,
            'put_volume': put_volume,
            'put_call_volume_ratio': put_volume / call_volume if call_volume > 0 else None,
            'total_open_interest': total_oi,
            'call_open_interest': call_oi,
            'put_open_interest': put_oi,
            'put_call_oi_ratio': put_oi / call_oi if call_oi > 0 else None,
            'average_implied_volatility': avg_iv,
            'last_updated': self.last_updated.isoformat()
        }

    def update_spot_price(self, new_spot_price: Decimal) -> None:
        """Update spot price and recalculate dependent metrics"""
        self.spot_price = new_spot_price
        self.last_updated = datetime.now()

        # Recalculate moneyness, intrinsic value, time value for all contracts
        for contract in self.contracts.values():
            contract.moneyness = contract.calculate_moneyness(new_spot_price)
            contract.intrinsic_value = contract.calculate_intrinsic_value(new_spot_price)
            contract.time_value = contract.calculate_time_value(new_spot_price)

        logger.info(f"Updated spot price for {self.underlying} to ${new_spot_price}")

    def __len__(self) -> int:
        """Return number of contracts in chain"""
        return len(self.contracts)

    def __repr__(self) -> str:
        return f"OptionChain(underlying='{self.underlying}', contracts={len(self.contracts)}, spot=${self.spot_price})"