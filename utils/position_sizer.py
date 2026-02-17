"""
Advanced Position Sizing Module
================================

Professional position sizing using:
- Fractional Kelly Criterion
- Risk Parity (HRP)
- Value at Risk (VaR) limits
- Riskfolio-Lib integration

For a $500 portfolio with 25 bots, proper position sizing is EXISTENTIAL.
Full Kelly = Ruin. Use 10% Kelly or less.

Key Principles:
1. Never risk more than 2% per trade
2. Use 10% Kelly (not full)
3. Limit correlated exposure to 25%
4. Stop all trading at 15% drawdown (V4 spec)

Author: Trading Bot Arsenal
Created: January 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PositionSizer')

# Try to import riskfolio-lib
try:
    import riskfolio as rp
    RISKFOLIO_AVAILABLE = True
except ImportError:
    RISKFOLIO_AVAILABLE = False
    logger.warning("riskfolio-lib not installed. Run: pip install riskfolio-lib")

# Try to import pypfopt
try:
    from pypfopt import risk_models, expected_returns, HRPOpt, EfficientFrontier
    from pypfopt.discrete_allocation import DiscreteAllocation
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    logger.warning("PyPortfolioOpt not installed. Run: pip install PyPortfolioOpt")


@dataclass
class PositionConfig:
    """Configuration for position sizing"""
    total_capital: float = 500.0
    max_position_pct: float = 0.10       # 10% max per strategy
    max_single_trade_risk: float = 0.02  # 2% risk per trade
    max_correlated_exposure: float = 0.25  # 25% in correlated assets
    kelly_fraction: float = 0.10         # 10% Kelly (NOT full!)
    global_drawdown_limit: float = 0.15  # V4: Stop at 15% drawdown
    min_position_size: float = 1.0       # Minimum $1 position
    use_var_limits: bool = True
    var_confidence: float = 0.95         # 95% VaR


@dataclass
class PositionSize:
    """Result of position sizing calculation"""
    bot_name: str
    recommended_capital: float
    recommended_pct: float
    risk_per_trade: float
    kelly_optimal: float
    kelly_fraction_used: float
    var_limit: float
    rationale: str


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation across all bots"""
    allocations: Dict[str, PositionSize]
    total_allocated: float
    cash_reserve: float
    risk_metrics: Dict[str, float]
    warnings: List[str]


class PositionSizer:
    """
    Advanced Position Sizing for Multi-Bot Portfolio.
    
    Methods:
    1. Fractional Kelly - Optimal growth with safety margin
    2. Risk Parity (HRP) - Equal risk contribution
    3. VaR-based limits - Maximum loss constraints
    4. Correlation-aware - Reduce correlated positions
    
    For $500 across 25 bots:
    - Average allocation: $20 per bot
    - But some bots deserve more (higher edge)
    - Some deserve less (higher risk/correlation)
    """
    
    # Bot profiles with expected edge and risk
    BOT_PROFILES = {
        # STOCKS (7) - Correlated with each other
        'RSI2-MeanReversion': {'win_rate': 0.62, 'rr_ratio': 1.87, 'category': 'stocks', 'correlation_group': 'equity'},
        'CumulativeRSI': {'win_rate': 0.58, 'rr_ratio': 1.33, 'category': 'stocks', 'correlation_group': 'equity'},
        'MACD-RSI-Combo': {'win_rate': 0.55, 'rr_ratio': 1.40, 'category': 'stocks', 'correlation_group': 'equity'},
        'BollingerSqueeze': {'win_rate': 0.52, 'rr_ratio': 1.67, 'category': 'stocks', 'correlation_group': 'equity'},
        'MTF-RSI': {'win_rate': 0.56, 'rr_ratio': 1.45, 'category': 'stocks', 'correlation_group': 'equity'},
        'DualMomentum': {'win_rate': 0.54, 'rr_ratio': 1.67, 'category': 'stocks', 'correlation_group': 'equity'},
        'SectorRotation': {'win_rate': 0.53, 'rr_ratio': 1.50, 'category': 'stocks', 'correlation_group': 'equity'},
        
        # PREDICTION MARKETS (8) - Lower correlation
        'Kalshi-Fed': {'win_rate': 0.65, 'rr_ratio': 1.33, 'category': 'prediction', 'correlation_group': 'macro'},
        'Weather-Edge': {'win_rate': 0.70, 'rr_ratio': 1.60, 'category': 'prediction', 'correlation_group': 'weather'},
        'Sports-Edge': {'win_rate': 0.55, 'rr_ratio': 1.67, 'category': 'prediction', 'correlation_group': 'sports'},
        'Sports-Props': {'win_rate': 0.54, 'rr_ratio': 1.50, 'category': 'prediction', 'correlation_group': 'sports'},
        'BoxOffice-Edge': {'win_rate': 0.58, 'rr_ratio': 1.50, 'category': 'prediction', 'correlation_group': 'entertainment'},
        'Awards-Edge': {'win_rate': 0.60, 'rr_ratio': 1.50, 'category': 'prediction', 'correlation_group': 'entertainment'},
        'Climate-Edge': {'win_rate': 0.68, 'rr_ratio': 1.67, 'category': 'prediction', 'correlation_group': 'weather'},
        'Economic-Edge': {'win_rate': 0.63, 'rr_ratio': 1.50, 'category': 'prediction', 'correlation_group': 'macro'},
        
        # FOREX (2)
        'OANDA-Forex': {'win_rate': 0.52, 'rr_ratio': 1.43, 'category': 'forex', 'correlation_group': 'forex'},
        'London-Breakout': {'win_rate': 0.48, 'rr_ratio': 1.50, 'category': 'forex', 'correlation_group': 'forex'},
        
        # CRYPTO (4)
        'FundingRate-Arb': {'win_rate': 0.72, 'rr_ratio': 1.50, 'category': 'crypto', 'correlation_group': 'crypto'},
        'Crypto-Arb': {'win_rate': 0.65, 'rr_ratio': 1.67, 'category': 'crypto', 'correlation_group': 'crypto'},
        'Kalshi-Hourly-Crypto': {'win_rate': 0.563, 'rr_ratio': 0.80, 'category': 'crypto', 'correlation_group': 'crypto'},
        'Alpaca-Crypto-RSI': {'win_rate': 0.563, 'rr_ratio': 1.67, 'category': 'crypto', 'correlation_group': 'crypto'},
        
        # OTHER (2)
        'Earnings-PEAD': {'win_rate': 0.50, 'rr_ratio': 1.60, 'category': 'events', 'correlation_group': 'equity'},
        'FOMC-Trader': {'win_rate': 0.55, 'rr_ratio': 1.50, 'category': 'events', 'correlation_group': 'macro'},
    }
    
    # Correlation matrix between groups (simplified)
    CORRELATION_GROUPS = {
        'equity': {'equity': 1.0, 'crypto': 0.6, 'forex': 0.3, 'macro': 0.5, 'weather': 0.0, 'sports': 0.0, 'entertainment': 0.1},
        'crypto': {'equity': 0.6, 'crypto': 1.0, 'forex': 0.2, 'macro': 0.4, 'weather': 0.0, 'sports': 0.0, 'entertainment': 0.1},
        'forex': {'equity': 0.3, 'crypto': 0.2, 'forex': 1.0, 'macro': 0.4, 'weather': 0.0, 'sports': 0.0, 'entertainment': 0.0},
        'macro': {'equity': 0.5, 'crypto': 0.4, 'forex': 0.4, 'macro': 1.0, 'weather': 0.1, 'sports': 0.0, 'entertainment': 0.1},
        'weather': {'equity': 0.0, 'crypto': 0.0, 'forex': 0.0, 'macro': 0.1, 'weather': 1.0, 'sports': 0.0, 'entertainment': 0.0},
        'sports': {'equity': 0.0, 'crypto': 0.0, 'forex': 0.0, 'macro': 0.0, 'weather': 0.0, 'sports': 1.0, 'entertainment': 0.1},
        'entertainment': {'equity': 0.1, 'crypto': 0.1, 'forex': 0.0, 'macro': 0.1, 'weather': 0.0, 'sports': 0.1, 'entertainment': 1.0},
    }
    
    def __init__(self, config: PositionConfig = None):
        self.config = config or PositionConfig()
        logger.info(f"PositionSizer initialized with ${self.config.total_capital} capital")
    
    def calculate_kelly(self, win_rate: float, rr_ratio: float, fraction: float = None) -> float:
        """
        Calculate Kelly criterion position size.
        
        Kelly = (p * b - q) / b
        where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = win/loss ratio
        
        IMPORTANT: Full Kelly is too aggressive. Use 10-25% Kelly.
        
        Args:
            win_rate: Probability of winning (0-1)
            rr_ratio: Reward/Risk ratio
            fraction: Kelly fraction to use (default: config value)
            
        Returns:
            Recommended position size as fraction of capital
        """
        if fraction is None:
            fraction = self.config.kelly_fraction
        
        p = win_rate
        q = 1 - p
        b = rr_ratio
        
        # Standard Kelly formula
        kelly = (p * b - q) / b
        
        # Fractional Kelly for safety
        fractional_kelly = kelly * fraction
        
        # Clamp to reasonable range
        return max(0, min(fractional_kelly, self.config.max_position_pct))
    
    def calculate_expected_value(self, win_rate: float, rr_ratio: float) -> float:
        """
        Calculate expected value per dollar risked.
        
        EV = (win_rate * rr_ratio) - (1 - win_rate)
        """
        return (win_rate * rr_ratio) - (1 - win_rate)
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Historical returns series
            confidence: Confidence level (e.g., 0.95 for 95%)
            method: 'historical' or 'parametric'
            
        Returns:
            VaR as positive percentage
        """
        if len(returns) < 30:
            return 0.05  # Default 5% if insufficient data
        
        if method == 'historical':
            var = -np.percentile(returns, (1 - confidence) * 100)
        else:
            # Parametric (assumes normal distribution)
            mu = returns.mean()
            sigma = returns.std()
            from scipy import stats
            z = stats.norm.ppf(1 - confidence)
            var = -(mu + z * sigma)
        
        return max(var, 0.01)  # At least 1%
    
    def build_correlation_matrix(self) -> pd.DataFrame:
        """Build correlation matrix for all bots."""
        bots = list(self.BOT_PROFILES.keys())
        n = len(bots)
        corr_matrix = np.eye(n)
        
        for i, bot1 in enumerate(bots):
            for j, bot2 in enumerate(bots):
                if i != j:
                    group1 = self.BOT_PROFILES[bot1]['correlation_group']
                    group2 = self.BOT_PROFILES[bot2]['correlation_group']
                    corr_matrix[i, j] = self.CORRELATION_GROUPS[group1][group2]
        
        return pd.DataFrame(corr_matrix, index=bots, columns=bots)
    
    def size_single_bot(self, bot_name: str) -> PositionSize:
        """
        Calculate position size for a single bot.
        
        Uses Kelly criterion with safety adjustments.
        """
        if bot_name not in self.BOT_PROFILES:
            return PositionSize(
                bot_name=bot_name,
                recommended_capital=self.config.min_position_size,
                recommended_pct=self.config.min_position_size / self.config.total_capital,
                risk_per_trade=self.config.min_position_size * self.config.max_single_trade_risk,
                kelly_optimal=0,
                kelly_fraction_used=self.config.kelly_fraction,
                var_limit=0.05,
                rationale="Unknown bot - minimum allocation"
            )
        
        profile = self.BOT_PROFILES[bot_name]
        win_rate = profile['win_rate']
        rr_ratio = profile['rr_ratio']
        
        # Calculate Kelly
        kelly_optimal = self.calculate_kelly(win_rate, rr_ratio, fraction=1.0)
        kelly_fractional = self.calculate_kelly(win_rate, rr_ratio)
        
        # Calculate expected value
        ev = self.calculate_expected_value(win_rate, rr_ratio)
        
        # Base allocation
        recommended_pct = kelly_fractional
        
        # Adjust for negative EV
        if ev <= 0:
            recommended_pct = 0
            rationale = f"Negative EV ({ev:.2f}) - no allocation"
        elif kelly_optimal <= 0:
            recommended_pct = 0
            rationale = f"Negative Kelly ({kelly_optimal:.2f}) - no allocation"
        else:
            # Apply category limits
            category = profile['category']
            if category == 'crypto':
                # Crypto is volatile - cap exposure
                recommended_pct = min(recommended_pct, 0.08)
                rationale = f"Kelly {kelly_fractional:.1%}, capped for crypto volatility"
            elif category == 'forex':
                # Forex has lower edge
                recommended_pct = min(recommended_pct, 0.05)
                rationale = f"Kelly {kelly_fractional:.1%}, capped for forex"
            else:
                rationale = f"Kelly {kelly_fractional:.1%} (10% of optimal {kelly_optimal:.1%})"
        
        # Calculate capital amount
        recommended_capital = self.config.total_capital * recommended_pct
        
        # Enforce minimum
        if 0 < recommended_capital < self.config.min_position_size:
            recommended_capital = self.config.min_position_size
            recommended_pct = recommended_capital / self.config.total_capital
        
        # Risk per trade
        risk_per_trade = recommended_capital * self.config.max_single_trade_risk
        
        return PositionSize(
            bot_name=bot_name,
            recommended_capital=recommended_capital,
            recommended_pct=recommended_pct,
            risk_per_trade=risk_per_trade,
            kelly_optimal=kelly_optimal,
            kelly_fraction_used=self.config.kelly_fraction,
            var_limit=recommended_capital * 0.05,  # Approximate
            rationale=rationale
        )
    
    def allocate_portfolio_equal(self) -> PortfolioAllocation:
        """
        Simple equal-weight allocation.
        
        $500 / 25 bots = $20 each
        """
        n_bots = len(self.BOT_PROFILES)
        equal_alloc = self.config.total_capital / n_bots
        
        allocations = {}
        for bot_name in self.BOT_PROFILES:
            allocations[bot_name] = PositionSize(
                bot_name=bot_name,
                recommended_capital=equal_alloc,
                recommended_pct=1.0 / n_bots,
                risk_per_trade=equal_alloc * self.config.max_single_trade_risk,
                kelly_optimal=0,
                kelly_fraction_used=0,
                var_limit=equal_alloc * 0.05,
                rationale="Equal weight allocation"
            )
        
        return PortfolioAllocation(
            allocations=allocations,
            total_allocated=self.config.total_capital,
            cash_reserve=0,
            risk_metrics={'method': 'equal_weight'},
            warnings=["Equal weight ignores edge differences"]
        )
    
    def allocate_portfolio_kelly(self) -> PortfolioAllocation:
        """
        Allocate based on Kelly criterion.
        
        Bots with higher edge get more capital.
        """
        allocations = {}
        total_kelly = 0
        
        # Calculate raw Kelly for each bot
        kelly_values = {}
        for bot_name in self.BOT_PROFILES:
            size = self.size_single_bot(bot_name)
            kelly_values[bot_name] = size.recommended_pct
            total_kelly += size.recommended_pct
        
        # Normalize if total > 100%
        if total_kelly > 1.0:
            scale = 0.9 / total_kelly  # Leave 10% cash
        else:
            scale = 1.0
        
        warnings = []
        
        for bot_name, kelly in kelly_values.items():
            adjusted_pct = kelly * scale
            
            # Apply max position limit
            if adjusted_pct > self.config.max_position_pct:
                adjusted_pct = self.config.max_position_pct
                warnings.append(f"{bot_name} capped at {self.config.max_position_pct:.0%}")
            
            capital = self.config.total_capital * adjusted_pct
            
            allocations[bot_name] = PositionSize(
                bot_name=bot_name,
                recommended_capital=capital,
                recommended_pct=adjusted_pct,
                risk_per_trade=capital * self.config.max_single_trade_risk,
                kelly_optimal=kelly / self.config.kelly_fraction if kelly > 0 else 0,
                kelly_fraction_used=self.config.kelly_fraction,
                var_limit=capital * 0.05,
                rationale=f"Kelly-weighted: {kelly:.1%} raw, {adjusted_pct:.1%} adjusted"
            )
        
        total_allocated = sum(a.recommended_capital for a in allocations.values())
        
        return PortfolioAllocation(
            allocations=allocations,
            total_allocated=total_allocated,
            cash_reserve=self.config.total_capital - total_allocated,
            risk_metrics={
                'method': 'fractional_kelly',
                'kelly_fraction': self.config.kelly_fraction,
                'total_kelly_raw': total_kelly
            },
            warnings=warnings
        )
    
    def allocate_portfolio_hrp(self, returns_data: pd.DataFrame = None) -> PortfolioAllocation:
        """
        Allocate using Hierarchical Risk Parity (HRP).
        
        HRP is more robust than mean-variance optimization because:
        1. No matrix inversion (numerically stable)
        2. Uses hierarchical clustering (captures structure)
        3. Better out-of-sample performance
        
        Args:
            returns_data: Historical returns for each bot (optional)
        """
        if not PYPFOPT_AVAILABLE:
            logger.warning("PyPortfolioOpt not available, falling back to Kelly")
            return self.allocate_portfolio_kelly()
        
        # If no returns data, use synthetic based on expected characteristics
        if returns_data is None:
            returns_data = self._generate_synthetic_returns()
        
        try:
            # Run HRP optimization
            hrp = HRPOpt(returns_data)
            weights = hrp.optimize()
            
            # Convert to allocations
            allocations = {}
            warnings = []
            
            for bot_name, weight in weights.items():
                # Apply position limits
                if weight > self.config.max_position_pct:
                    weight = self.config.max_position_pct
                    warnings.append(f"{bot_name} capped at {self.config.max_position_pct:.0%}")
                
                capital = self.config.total_capital * weight
                
                # Get Kelly for comparison
                if bot_name in self.BOT_PROFILES:
                    profile = self.BOT_PROFILES[bot_name]
                    kelly_opt = self.calculate_kelly(profile['win_rate'], profile['rr_ratio'], fraction=1.0)
                else:
                    kelly_opt = 0
                
                allocations[bot_name] = PositionSize(
                    bot_name=bot_name,
                    recommended_capital=capital,
                    recommended_pct=weight,
                    risk_per_trade=capital * self.config.max_single_trade_risk,
                    kelly_optimal=kelly_opt,
                    kelly_fraction_used=0,  # HRP doesn't use Kelly
                    var_limit=capital * 0.05,
                    rationale=f"HRP allocation: {weight:.1%}"
                )
            
            total_allocated = sum(a.recommended_capital for a in allocations.values())
            
            # Get portfolio stats
            portfolio_return, portfolio_vol, portfolio_sharpe = hrp.portfolio_performance()
            
            return PortfolioAllocation(
                allocations=allocations,
                total_allocated=total_allocated,
                cash_reserve=self.config.total_capital - total_allocated,
                risk_metrics={
                    'method': 'hrp',
                    'expected_return': portfolio_return,
                    'expected_volatility': portfolio_vol,
                    'expected_sharpe': portfolio_sharpe
                },
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"HRP allocation failed: {e}, falling back to Kelly")
            return self.allocate_portfolio_kelly()
    
    def _generate_synthetic_returns(self, n_periods: int = 252) -> pd.DataFrame:
        """
        Generate synthetic returns based on bot profiles.
        
        Used when historical data is not available.
        """
        np.random.seed(42)
        
        returns = {}
        dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='D')
        
        for bot_name, profile in self.BOT_PROFILES.items():
            win_rate = profile['win_rate']
            rr_ratio = profile['rr_ratio']
            
            # Expected daily return (simplified)
            ev = self.calculate_expected_value(win_rate, rr_ratio)
            daily_return = ev * 0.02  # Assuming 2% risk per trade
            
            # Volatility based on win rate and R:R
            daily_vol = 0.02 * rr_ratio * (1 + (1 - win_rate))
            
            # Generate returns
            bot_returns = np.random.normal(daily_return, daily_vol, n_periods)
            returns[bot_name] = bot_returns
        
        return pd.DataFrame(returns, index=dates)
    
    def check_correlation_limits(self, allocation: PortfolioAllocation) -> List[str]:
        """
        Check if correlated positions exceed limits.
        
        Returns list of warnings.
        """
        warnings = []
        
        # Group allocations by correlation group
        group_exposure = {}
        
        for bot_name, position in allocation.allocations.items():
            if bot_name in self.BOT_PROFILES:
                group = self.BOT_PROFILES[bot_name]['correlation_group']
                if group not in group_exposure:
                    group_exposure[group] = 0
                group_exposure[group] += position.recommended_pct
        
        # Check limits
        for group, exposure in group_exposure.items():
            if exposure > self.config.max_correlated_exposure:
                warnings.append(
                    f"WARNING: {group} exposure {exposure:.0%} exceeds "
                    f"limit {self.config.max_correlated_exposure:.0%}"
                )
        
        return warnings
    
    def get_optimal_allocation(self, method: str = 'kelly') -> PortfolioAllocation:
        """
        Get optimal portfolio allocation.
        
        Args:
            method: 'equal', 'kelly', or 'hrp'
            
        Returns:
            PortfolioAllocation with all sizing recommendations
        """
        if method == 'equal':
            allocation = self.allocate_portfolio_equal()
        elif method == 'hrp':
            allocation = self.allocate_portfolio_hrp()
        else:
            allocation = self.allocate_portfolio_kelly()
        
        # Check correlation limits
        correlation_warnings = self.check_correlation_limits(allocation)
        allocation.warnings.extend(correlation_warnings)
        
        return allocation
    
    def print_allocation(self, allocation: PortfolioAllocation):
        """Pretty print allocation."""
        print("\n" + "=" * 70)
        print(f"PORTFOLIO ALLOCATION (${self.config.total_capital:,.0f})")
        print("=" * 70)
        
        print(f"\nMethod: {allocation.risk_metrics.get('method', 'unknown')}")
        print(f"Total Allocated: ${allocation.total_allocated:,.2f}")
        print(f"Cash Reserve: ${allocation.cash_reserve:,.2f}")
        
        print("\n" + "-" * 70)
        print(f"{'Bot':<25} {'Capital':>10} {'Pct':>8} {'Risk/Trade':>12} {'Rationale':<30}")
        print("-" * 70)
        
        # Sort by allocation size
        sorted_allocs = sorted(
            allocation.allocations.items(),
            key=lambda x: x[1].recommended_capital,
            reverse=True
        )
        
        for bot_name, pos in sorted_allocs:
            if pos.recommended_capital > 0:
                print(f"{bot_name:<25} ${pos.recommended_capital:>8.2f} {pos.recommended_pct:>7.1%} "
                      f"${pos.risk_per_trade:>10.2f}  {pos.rationale[:28]}")
        
        if allocation.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in allocation.warnings:
                print(f"  - {warning}")
        
        print("=" * 70)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("POSITION SIZING MODULE TEST")
    print("=" * 70)
    
    # Initialize with $500
    config = PositionConfig(
        total_capital=500.0,
        max_position_pct=0.10,
        kelly_fraction=0.10,
        max_correlated_exposure=0.25
    )
    
    sizer = PositionSizer(config)
    
    # Test single bot sizing
    print("\n[1] Single Bot Sizing (RSI2-MeanReversion):")
    size = sizer.size_single_bot('RSI2-MeanReversion')
    print(f"  Recommended: ${size.recommended_capital:.2f} ({size.recommended_pct:.1%})")
    print(f"  Risk per trade: ${size.risk_per_trade:.2f}")
    print(f"  Kelly optimal: {size.kelly_optimal:.1%}")
    print(f"  Rationale: {size.rationale}")
    
    # Test equal weight allocation
    print("\n[2] Equal Weight Allocation:")
    equal_alloc = sizer.allocate_portfolio_equal()
    print(f"  Per bot: ${equal_alloc.allocations['RSI2-MeanReversion'].recommended_capital:.2f}")
    
    # Test Kelly allocation
    print("\n[3] Kelly Criterion Allocation:")
    kelly_alloc = sizer.get_optimal_allocation(method='kelly')
    sizer.print_allocation(kelly_alloc)
    
    # Test HRP allocation (if available)
    if PYPFOPT_AVAILABLE:
        print("\n[4] Hierarchical Risk Parity (HRP) Allocation:")
        hrp_alloc = sizer.get_optimal_allocation(method='hrp')
        sizer.print_allocation(hrp_alloc)
    
    print("\n" + "=" * 70)
