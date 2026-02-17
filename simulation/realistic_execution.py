"""
REALISTIC EXECUTION MODEL
=========================

Simulates real-world trading conditions:
- Slippage (price impact)
- Execution delays
- Partial fills
- Order rejection
- Market impact

Based on research:
- Liquid markets: 0.1-0.5% slippage
- Illiquid markets: 0.5-1%+ slippage
- Execution delays: 50-500ms
- Partial fills: 70-100%

Author: Trading Bot Arsenal
Created: January 2026
"""

import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger('RealisticExecution')


class MarketCondition(Enum):
    """Market liquidity conditions"""
    HIGH_LIQUIDITY = "high_liquidity"      # Major pairs, large caps
    NORMAL = "normal"                       # Standard conditions
    LOW_LIQUIDITY = "low_liquidity"         # Small caps, exotic pairs
    VOLATILE = "volatile"                   # News events, high VIX
    ILLIQUID = "illiquid"                   # After hours, penny stocks


@dataclass
class ExecutionConfig:
    """Configuration for execution simulation"""
    # Slippage settings by market condition
    slippage_config: Dict[MarketCondition, Tuple[float, float]] = None
    
    # Execution delay (milliseconds)
    delay_range: Tuple[int, int] = (50, 500)
    
    # Partial fill probability and range
    partial_fill_probability: float = 0.15  # 15% chance of partial fill
    partial_fill_range: Tuple[float, float] = (0.50, 0.95)
    
    # Order rejection probability
    rejection_probability: float = 0.02  # 2% rejection rate
    
    # Requote probability (price changed)
    requote_probability: float = 0.05
    
    def __post_init__(self):
        if self.slippage_config is None:
            self.slippage_config = {
                MarketCondition.HIGH_LIQUIDITY: (0.0005, 0.001),   # 0.05-0.1%
                MarketCondition.NORMAL: (0.001, 0.003),            # 0.1-0.3%
                MarketCondition.LOW_LIQUIDITY: (0.003, 0.008),     # 0.3-0.8%
                MarketCondition.VOLATILE: (0.005, 0.015),          # 0.5-1.5%
                MarketCondition.ILLIQUID: (0.01, 0.03),            # 1-3%
            }


@dataclass
class ExecutionResult:
    """Result of order execution simulation"""
    original_price: float
    executed_price: float
    requested_quantity: float
    filled_quantity: float
    slippage_pct: float
    slippage_amount: float
    delay_ms: int
    is_partial_fill: bool
    is_rejected: bool
    is_requoted: bool
    requote_price: Optional[float]
    execution_time: datetime
    market_condition: MarketCondition


class RealisticExecutionModel:
    """
    Simulates realistic order execution.
    
    Features:
    - Variable slippage based on market conditions
    - Execution delays
    - Partial fills
    - Order rejections
    - Requotes
    - Market impact (for large orders)
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
    
    def simulate_execution(
        self,
        order_price: float,
        quantity: float,
        side: str,
        market_condition: MarketCondition = MarketCondition.NORMAL,
        order_time: Optional[datetime] = None
    ) -> ExecutionResult:
        """
        Simulate order execution with realistic conditions.
        
        Args:
            order_price: Requested execution price
            quantity: Requested quantity
            side: 'buy' or 'sell'
            market_condition: Current market liquidity
            order_time: Time of order (for timestamp)
        
        Returns:
            ExecutionResult with all execution details
        """
        order_time = order_time or datetime.now()
        
        # Check for rejection
        if random.random() < self.config.rejection_probability:
            return ExecutionResult(
                original_price=order_price,
                executed_price=0,
                requested_quantity=quantity,
                filled_quantity=0,
                slippage_pct=0,
                slippage_amount=0,
                delay_ms=0,
                is_partial_fill=False,
                is_rejected=True,
                is_requoted=False,
                requote_price=None,
                execution_time=order_time,
                market_condition=market_condition
            )
        
        # Calculate slippage
        slip_min, slip_max = self.config.slippage_config[market_condition]
        slippage_pct = np.random.uniform(slip_min, slip_max)
        
        # Slippage direction (always against trader)
        if side == 'buy':
            slippage_amount = order_price * slippage_pct
            executed_price = order_price + slippage_amount
        else:
            slippage_amount = order_price * slippage_pct
            executed_price = order_price - slippage_amount
        
        # Check for requote
        is_requoted = random.random() < self.config.requote_probability
        requote_price = None
        if is_requoted:
            # Requote is usually worse than original
            requote_adjustment = np.random.uniform(0.001, 0.005)
            if side == 'buy':
                requote_price = executed_price * (1 + requote_adjustment)
            else:
                requote_price = executed_price * (1 - requote_adjustment)
            executed_price = requote_price
        
        # Partial fill
        is_partial_fill = random.random() < self.config.partial_fill_probability
        if is_partial_fill:
            fill_pct = np.random.uniform(*self.config.partial_fill_range)
            filled_quantity = quantity * fill_pct
        else:
            filled_quantity = quantity
        
        # Execution delay
        delay_ms = random.randint(*self.config.delay_range)
        execution_time = order_time + timedelta(milliseconds=delay_ms)
        
        return ExecutionResult(
            original_price=order_price,
            executed_price=executed_price,
            requested_quantity=quantity,
            filled_quantity=filled_quantity,
            slippage_pct=slippage_pct,
            slippage_amount=slippage_amount,
            delay_ms=delay_ms,
            is_partial_fill=is_partial_fill,
            is_rejected=False,
            is_requoted=is_requoted,
            requote_price=requote_price,
            execution_time=execution_time,
            market_condition=market_condition
        )
    
    def simulate_market_impact(
        self,
        order_price: float,
        quantity: float,
        average_daily_volume: float,
        side: str
    ) -> float:
        """
        Simulate market impact for large orders.
        
        Based on square-root model:
        Impact = σ * sqrt(Q / ADV) * sign
        
        Args:
            order_price: Current price
            quantity: Order size
            average_daily_volume: Average daily volume
            side: 'buy' or 'sell'
        
        Returns:
            Price after market impact
        """
        if average_daily_volume <= 0:
            return order_price
        
        # Participation rate
        participation = quantity / average_daily_volume
        
        # Assume 2% daily volatility
        volatility = 0.02
        
        # Square-root impact model
        impact = volatility * np.sqrt(participation)
        
        # Apply impact
        if side == 'buy':
            return order_price * (1 + impact)
        else:
            return order_price * (1 - impact)
    
    def get_market_condition(
        self,
        symbol: str,
        time: datetime,
        vix: Optional[float] = None,
        is_after_hours: bool = False
    ) -> MarketCondition:
        """
        Determine market condition based on various factors.
        """
        # Check for after hours
        if is_after_hours:
            return MarketCondition.ILLIQUID
        
        # Check VIX (if available)
        if vix is not None:
            if vix > 30:
                return MarketCondition.VOLATILE
            elif vix > 20:
                return MarketCondition.LOW_LIQUIDITY
        
        # Check time of day (less liquid around open/close)
        hour = time.hour
        if hour < 10 or hour > 15:  # Before 10am or after 3pm
            return MarketCondition.LOW_LIQUIDITY
        
        # Default to normal
        return MarketCondition.NORMAL


class SlippageModel:
    """
    Different slippage models for various scenarios.
    """
    
    @staticmethod
    def fixed_slippage(price: float, bps: float = 5) -> float:
        """Fixed basis points slippage"""
        return price * (bps / 10000)
    
    @staticmethod
    def percentage_slippage(price: float, pct: float = 0.1) -> float:
        """Fixed percentage slippage"""
        return price * (pct / 100)
    
    @staticmethod
    def volume_based_slippage(
        price: float,
        order_volume: float,
        market_volume: float,
        base_slippage: float = 0.001
    ) -> float:
        """
        Slippage increases with order size relative to market volume.
        """
        if market_volume <= 0:
            return price * base_slippage * 10
        
        volume_ratio = order_volume / market_volume
        slippage_multiplier = 1 + (volume_ratio * 10)  # 10x multiplier
        
        return price * base_slippage * slippage_multiplier
    
    @staticmethod
    def volatility_adjusted_slippage(
        price: float,
        volatility: float,
        base_slippage: float = 0.001
    ) -> float:
        """
        Slippage increases with volatility.
        """
        # Assume 20% is "normal" volatility
        vol_multiplier = volatility / 0.20
        return price * base_slippage * vol_multiplier
    
    @staticmethod
    def random_slippage(
        price: float,
        mean: float = 0.001,
        std: float = 0.0005
    ) -> float:
        """
        Random slippage from normal distribution.
        """
        slippage_pct = abs(np.random.normal(mean, std))
        return price * slippage_pct


class SkipTradeSimulator:
    """
    Simulates missed trades due to various reasons:
    - Platform issues
    - Internet outages
    - Slippage too high
    - Insufficient liquidity
    - Human error
    """
    
    def __init__(
        self,
        base_skip_rate: float = 0.05,
        platform_issue_rate: float = 0.02,
        connection_issue_rate: float = 0.01,
        slippage_skip_threshold: float = 0.02  # Skip if slippage > 2%
    ):
        self.base_skip_rate = base_skip_rate
        self.platform_issue_rate = platform_issue_rate
        self.connection_issue_rate = connection_issue_rate
        self.slippage_skip_threshold = slippage_skip_threshold
    
    def should_skip_trade(
        self,
        slippage: Optional[float] = None,
        is_volatile: bool = False
    ) -> Tuple[bool, str]:
        """
        Determine if a trade should be skipped.
        
        Returns:
            Tuple of (should_skip, reason)
        """
        # Base random skip
        if random.random() < self.base_skip_rate:
            return True, "Random execution failure"
        
        # Platform issue
        if random.random() < self.platform_issue_rate:
            return True, "Platform technical issue"
        
        # Connection issue
        if random.random() < self.connection_issue_rate:
            return True, "Connection timeout"
        
        # Slippage too high
        if slippage is not None and slippage > self.slippage_skip_threshold:
            if random.random() < 0.5:  # 50% chance to skip high slippage
                return True, f"Slippage too high ({slippage:.2%})"
        
        # More skips during volatile conditions
        if is_volatile and random.random() < 0.1:
            return True, "Volatile market conditions"
        
        return False, ""


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Testing Realistic Execution Model...")
    
    config = ExecutionConfig()
    model = RealisticExecutionModel(config)
    
    # Simulate 100 orders
    results = []
    for i in range(100):
        condition = random.choice(list(MarketCondition))
        result = model.simulate_execution(
            order_price=100.0,
            quantity=100,
            side='buy',
            market_condition=condition
        )
        results.append(result)
    
    # Analyze results
    executed = [r for r in results if not r.is_rejected]
    rejected = [r for r in results if r.is_rejected]
    partial = [r for r in executed if r.is_partial_fill]
    requoted = [r for r in executed if r.is_requoted]
    
    avg_slippage = np.mean([r.slippage_pct for r in executed]) if executed else 0
    avg_delay = np.mean([r.delay_ms for r in executed]) if executed else 0
    
    print(f"\n📊 EXECUTION SIMULATION RESULTS (100 orders)")
    print(f"   Executed: {len(executed)}")
    print(f"   Rejected: {len(rejected)} ({len(rejected)/100:.0%})")
    print(f"   Partial Fills: {len(partial)} ({len(partial)/100:.0%})")
    print(f"   Requoted: {len(requoted)} ({len(requoted)/100:.0%})")
    print(f"   Average Slippage: {avg_slippage:.2%}")
    print(f"   Average Delay: {avg_delay:.0f}ms")
    
    # Test skip simulator
    print("\n📉 SKIP TRADE SIMULATION (100 trades)")
    skipper = SkipTradeSimulator(base_skip_rate=0.10)
    
    skipped = 0
    reasons = {}
    for _ in range(100):
        should_skip, reason = skipper.should_skip_trade(
            slippage=random.uniform(0, 0.03),
            is_volatile=random.random() < 0.2
        )
        if should_skip:
            skipped += 1
            reasons[reason] = reasons.get(reason, 0) + 1
    
    print(f"   Total Skipped: {skipped}")
    print(f"   Skip Rate: {skipped/100:.0%}")
    print(f"   Reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"      • {reason}: {count}")
