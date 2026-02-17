"""
Kalshi Prediction Market Backtester
====================================

Custom backtesting framework for prediction markets (Kalshi).
No existing open-source tools exist for this - built from scratch.

Features:
- Historical contract simulation
- Time decay modeling (theta)
- Bid/ask spread impact
- Brier score tracking (calibration)
- Multiple market types (Fed, Weather, Sports, etc.)

Market Types Supported:
- Fed decisions (rate hikes/cuts)
- Weather (temperature, precipitation)
- Sports (game outcomes, props)
- Economic (GDP, inflation, jobs)
- Entertainment (box office, awards)

Author: Trading Bot Arsenal
Created: January 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KalshiBacktester')


class MarketType(Enum):
    """Kalshi market categories"""
    FED = "fed"
    WEATHER = "weather"
    SPORTS = "sports"
    ECONOMIC = "economic"
    ENTERTAINMENT = "entertainment"
    CRYPTO = "crypto"
    POLITICS = "politics"


class ContractOutcome(Enum):
    """Contract resolution outcomes"""
    YES = "yes"
    NO = "no"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass
class PredictionContract:
    """Represents a single prediction market contract"""
    contract_id: str
    market_type: MarketType
    title: str
    expiry: datetime
    strike: Optional[float]  # For numeric markets (temp > 80, etc.)
    
    # Market state
    yes_price: float  # Current YES price (0.01 - 0.99)
    no_price: float   # Current NO price (0.01 - 0.99)
    volume: int
    open_interest: int
    
    # Resolution
    outcome: ContractOutcome = ContractOutcome.PENDING
    settled_price: float = 0.0  # 1.0 for YES, 0.0 for NO
    
    # Our position
    position: int = 0  # Positive = YES, Negative = NO
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None


@dataclass
class KalshiTrade:
    """Record of a single trade"""
    trade_id: str
    contract_id: str
    timestamp: datetime
    side: str  # 'buy_yes', 'buy_no', 'sell_yes', 'sell_no'
    quantity: int
    price: float
    fees: float
    realized_pnl: float = 0.0


@dataclass
class BacktestConfig:
    """Configuration for Kalshi backtest"""
    initial_capital: float = 500.0
    max_position_per_contract: int = 100  # Max contracts per market
    max_portfolio_exposure: float = 0.25  # Max % of capital in any one market
    fee_rate: float = 0.10  # 10% of profits (Kalshi fee structure)
    min_edge: float = 0.05  # Minimum edge required to trade
    use_spreads: bool = True  # Account for bid/ask spread
    spread_cost: float = 0.02  # 2 cents typical spread


@dataclass
class BacktestResult:
    """Results from Kalshi backtest"""
    total_return: float
    total_pnl: float
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    brier_score: float  # Calibration metric
    total_fees: float
    final_capital: float
    trades: List[KalshiTrade]
    equity_curve: pd.Series
    by_market_type: Dict[str, Dict]


class KalshiBacktester:
    """
    Backtester for Kalshi prediction market strategies.
    
    Key Features:
    1. Time decay modeling - contracts worth more as expiry approaches
    2. Spread impact - realistic execution costs
    3. Brier score tracking - measure prediction calibration
    4. Multiple market types with different characteristics
    
    Usage:
        backtester = KalshiBacktester(config)
        
        # Add historical contracts
        backtester.add_contract(contract)
        
        # Run strategy
        result = backtester.run_strategy(my_strategy_func)
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.contracts: Dict[str, PredictionContract] = {}
        self.trades: List[KalshiTrade] = []
        self.capital = self.config.initial_capital
        self.peak_capital = self.capital
        self.equity_curve = []
        
        # Tracking
        self.predictions: List[Tuple[float, float]] = []  # (predicted_prob, actual_outcome)
        self.trade_counter = 0
        
        logger.info(f"KalshiBacktester initialized with ${self.config.initial_capital} capital")
    
    def add_contract(self, contract: PredictionContract) -> None:
        """Add a contract to the backtest universe."""
        self.contracts[contract.contract_id] = contract
    
    def generate_synthetic_contracts(
        self,
        market_type: MarketType,
        n_contracts: int = 100,
        base_date: datetime = None,
        win_rate_assumption: float = 0.55
    ) -> List[PredictionContract]:
        """
        Generate synthetic contracts for backtesting.
        
        This is necessary because Kalshi historical data is limited.
        Uses realistic distributions based on market type.
        
        Args:
            market_type: Type of market to generate
            n_contracts: Number of contracts
            base_date: Starting date
            win_rate_assumption: Assumed edge for "good" predictions
        """
        if base_date is None:
            base_date = datetime.now() - timedelta(days=365)
        
        contracts = []
        
        # Market-specific parameters
        market_params = {
            MarketType.FED: {
                'expiry_days': (7, 45),  # 1 week to 6 weeks out
                'price_range': (0.20, 0.80),  # Fed decisions often uncertain
                'volume_range': (1000, 50000),
                'edge_potential': 0.08
            },
            MarketType.WEATHER: {
                'expiry_days': (1, 14),  # Short-term
                'price_range': (0.30, 0.70),
                'volume_range': (500, 10000),
                'edge_potential': 0.12  # Weather has good edges
            },
            MarketType.SPORTS: {
                'expiry_days': (1, 7),
                'price_range': (0.25, 0.75),
                'volume_range': (2000, 100000),
                'edge_potential': 0.05  # Sports are efficient
            },
            MarketType.ECONOMIC: {
                'expiry_days': (14, 60),
                'price_range': (0.15, 0.85),
                'volume_range': (500, 20000),
                'edge_potential': 0.07
            },
            MarketType.ENTERTAINMENT: {
                'expiry_days': (7, 90),
                'price_range': (0.10, 0.90),
                'volume_range': (200, 5000),
                'edge_potential': 0.10
            },
            MarketType.CRYPTO: {
                'expiry_days': (1, 7),  # Hourly to weekly
                'price_range': (0.35, 0.65),
                'volume_range': (1000, 30000),
                'edge_potential': 0.06
            }
        }
        
        params = market_params.get(market_type, market_params[MarketType.ECONOMIC])
        
        for i in range(n_contracts):
            # Generate contract parameters
            days_to_expiry = np.random.randint(params['expiry_days'][0], params['expiry_days'][1])
            expiry = base_date + timedelta(days=days_to_expiry + i * 3)  # Stagger contracts
            
            # Initial price
            yes_price = np.random.uniform(params['price_range'][0], params['price_range'][1])
            no_price = 1.0 - yes_price  # Simplified (ignoring spread for generation)
            
            # Determine outcome based on edge
            # If we have edge, our predicted probability should be better than market
            edge = params['edge_potential']
            true_prob = yes_price + np.random.uniform(-edge, edge)
            true_prob = np.clip(true_prob, 0.05, 0.95)
            
            outcome = ContractOutcome.YES if np.random.random() < true_prob else ContractOutcome.NO
            settled_price = 1.0 if outcome == ContractOutcome.YES else 0.0
            
            contract = PredictionContract(
                contract_id=f"{market_type.value}_{i:04d}",
                market_type=market_type,
                title=f"{market_type.value.title()} Contract {i}",
                expiry=expiry,
                strike=None,
                yes_price=yes_price,
                no_price=no_price,
                volume=np.random.randint(params['volume_range'][0], params['volume_range'][1]),
                open_interest=np.random.randint(100, 5000),
                outcome=outcome,
                settled_price=settled_price
            )
            
            contracts.append(contract)
            self.add_contract(contract)
        
        logger.info(f"Generated {n_contracts} synthetic {market_type.value} contracts")
        return contracts
    
    def calculate_time_decay(
        self,
        current_price: float,
        true_value: float,
        days_to_expiry: float,
        total_days: float = 30
    ) -> float:
        """
        Model time decay (theta) for prediction markets.
        
        As expiry approaches, prices converge to true value.
        This creates opportunity for informed traders.
        
        Args:
            current_price: Current market price
            true_value: Our estimate of true probability
            days_to_expiry: Days until settlement
            total_days: Total contract duration
            
        Returns:
            Expected price movement due to time decay
        """
        if days_to_expiry <= 0:
            return true_value - current_price
        
        # Time decay accelerates near expiry
        time_factor = 1 - (days_to_expiry / total_days) ** 0.5
        
        # Price converges to true value
        expected_move = (true_value - current_price) * time_factor
        
        return expected_move
    
    def calculate_edge(
        self,
        contract: PredictionContract,
        predicted_prob: float
    ) -> Tuple[float, str]:
        """
        Calculate trading edge for a contract.
        
        Args:
            contract: The contract to evaluate
            predicted_prob: Our predicted probability of YES
            
        Returns:
            (edge, recommended_side)
        """
        # Account for spread
        if self.config.use_spreads:
            buy_yes_price = contract.yes_price + self.config.spread_cost / 2
            buy_no_price = contract.no_price + self.config.spread_cost / 2
        else:
            buy_yes_price = contract.yes_price
            buy_no_price = contract.no_price
        
        # Edge on YES side
        yes_edge = predicted_prob - buy_yes_price
        
        # Edge on NO side
        no_edge = (1 - predicted_prob) - buy_no_price
        
        if yes_edge > no_edge and yes_edge > self.config.min_edge:
            return yes_edge, 'buy_yes'
        elif no_edge > yes_edge and no_edge > self.config.min_edge:
            return no_edge, 'buy_no'
        else:
            return 0.0, 'no_trade'
    
    def execute_trade(
        self,
        contract: PredictionContract,
        side: str,
        quantity: int,
        timestamp: datetime
    ) -> Optional[KalshiTrade]:
        """
        Execute a trade on a contract.
        
        Args:
            contract: Contract to trade
            side: 'buy_yes', 'buy_no', 'sell_yes', 'sell_no'
            quantity: Number of contracts
            timestamp: Trade timestamp
            
        Returns:
            KalshiTrade record or None if trade fails
        """
        # Determine price
        if side == 'buy_yes':
            price = contract.yes_price + (self.config.spread_cost / 2 if self.config.use_spreads else 0)
        elif side == 'buy_no':
            price = contract.no_price + (self.config.spread_cost / 2 if self.config.use_spreads else 0)
        elif side == 'sell_yes':
            price = contract.yes_price - (self.config.spread_cost / 2 if self.config.use_spreads else 0)
        elif side == 'sell_no':
            price = contract.no_price - (self.config.spread_cost / 2 if self.config.use_spreads else 0)
        else:
            return None
        
        # Check capital
        cost = price * quantity
        if side.startswith('buy') and cost > self.capital * self.config.max_portfolio_exposure:
            # Reduce quantity to fit exposure limit
            quantity = int((self.capital * self.config.max_portfolio_exposure) / price)
            if quantity <= 0:
                return None
            cost = price * quantity
        
        # Check position limits
        if quantity > self.config.max_position_per_contract:
            quantity = self.config.max_position_per_contract
            cost = price * quantity
        
        # Execute
        if side.startswith('buy'):
            self.capital -= cost
        else:
            self.capital += cost
        
        # Update contract position
        if side == 'buy_yes':
            contract.position += quantity
            contract.entry_price = price
            contract.entry_time = timestamp
        elif side == 'buy_no':
            contract.position -= quantity
            contract.entry_price = price
            contract.entry_time = timestamp
        elif side == 'sell_yes':
            contract.position -= quantity
        elif side == 'sell_no':
            contract.position += quantity
        
        self.trade_counter += 1
        trade = KalshiTrade(
            trade_id=f"trade_{self.trade_counter:05d}",
            contract_id=contract.contract_id,
            timestamp=timestamp,
            side=side,
            quantity=quantity,
            price=price,
            fees=0.0  # Fees calculated at settlement
        )
        
        self.trades.append(trade)
        return trade
    
    def settle_contract(self, contract: PredictionContract) -> float:
        """
        Settle a contract and calculate P&L.
        
        Returns realized P&L after fees.
        """
        if contract.position == 0:
            return 0.0
        
        # Calculate gross P&L
        if contract.position > 0:  # Long YES
            if contract.outcome == ContractOutcome.YES:
                gross_pnl = (1.0 - contract.entry_price) * abs(contract.position)
            else:
                gross_pnl = -contract.entry_price * abs(contract.position)
        else:  # Long NO
            if contract.outcome == ContractOutcome.NO:
                gross_pnl = (1.0 - contract.entry_price) * abs(contract.position)
            else:
                gross_pnl = -contract.entry_price * abs(contract.position)
        
        # Apply fees (only on profits)
        if gross_pnl > 0:
            fees = gross_pnl * self.config.fee_rate
            net_pnl = gross_pnl - fees
        else:
            fees = 0.0
            net_pnl = gross_pnl
        
        # Update capital
        self.capital += net_pnl + (contract.entry_price * abs(contract.position))  # Return of principal + P&L
        
        # Track for Brier score
        predicted_prob = contract.entry_price if contract.position > 0 else (1 - contract.entry_price)
        actual_outcome = 1.0 if contract.outcome == ContractOutcome.YES else 0.0
        if contract.position < 0:
            actual_outcome = 1 - actual_outcome
        self.predictions.append((predicted_prob, actual_outcome))
        
        # Update peak capital for drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        
        # Record equity
        self.equity_curve.append({'timestamp': contract.expiry, 'capital': self.capital})
        
        return net_pnl
    
    def calculate_brier_score(self) -> float:
        """
        Calculate Brier score for prediction calibration.
        
        Brier Score = mean((predicted - actual)^2)
        
        Lower is better:
        - 0.0 = Perfect predictions
        - 0.25 = Random guessing
        - 1.0 = Perfectly wrong
        """
        if not self.predictions:
            return 0.25  # Default to random
        
        brier = np.mean([(p - a) ** 2 for p, a in self.predictions])
        return brier
    
    def run_strategy(
        self,
        strategy_func: Callable[[PredictionContract, Dict], Tuple[str, int, float]],
        market_data: Dict = None
    ) -> BacktestResult:
        """
        Run a prediction market strategy.
        
        Args:
            strategy_func: Function that takes (contract, market_data) and returns
                          (side, quantity, predicted_prob) or ('no_trade', 0, 0)
            market_data: Additional data for strategy decisions
            
        Returns:
            BacktestResult with all metrics
        """
        if market_data is None:
            market_data = {}
        
        initial_capital = self.capital
        wins = 0
        losses = 0
        total_profit = 0.0
        total_loss = 0.0
        total_fees = 0.0
        
        # Track by market type
        by_market = {}
        
        # Sort contracts by expiry
        sorted_contracts = sorted(
            self.contracts.values(),
            key=lambda c: c.expiry
        )
        
        for contract in sorted_contracts:
            # Skip already resolved
            if contract.outcome == ContractOutcome.CANCELLED:
                continue
            
            # Get strategy decision
            try:
                side, quantity, predicted_prob = strategy_func(contract, market_data)
            except Exception as e:
                logger.error(f"Strategy error: {e}")
                continue
            
            if side == 'no_trade' or quantity <= 0:
                continue
            
            # Execute trade
            trade = self.execute_trade(
                contract,
                side,
                quantity,
                contract.expiry - timedelta(days=np.random.randint(1, 7))  # Random entry time
            )
            
            if trade is None:
                continue
            
            # Settle at expiry
            pnl = self.settle_contract(contract)
            
            # Track results
            market_key = contract.market_type.value
            if market_key not in by_market:
                by_market[market_key] = {
                    'trades': 0, 'wins': 0, 'pnl': 0.0
                }
            
            by_market[market_key]['trades'] += 1
            by_market[market_key]['pnl'] += pnl
            
            if pnl > 0:
                wins += 1
                total_profit += pnl
                by_market[market_key]['wins'] += 1
                
                # Track fees
                fees = pnl * self.config.fee_rate / (1 - self.config.fee_rate)
                total_fees += fees
            else:
                losses += 1
                total_loss += abs(pnl)
        
        # Calculate final metrics
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0
        avg_win = total_profit / wins if wins > 0 else 0
        avg_loss = total_loss / losses if losses > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Max drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) > 0:
            equity_df['peak'] = equity_df['capital'].cummax()
            equity_df['drawdown'] = (equity_df['peak'] - equity_df['capital']) / equity_df['peak']
            max_drawdown = equity_df['drawdown'].max()
        else:
            max_drawdown = 0.0
        
        # Sharpe (simplified)
        if len(self.equity_curve) > 1:
            returns = pd.Series([e['capital'] for e in self.equity_curve]).pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0.0
        
        # Brier score
        brier_score = self.calculate_brier_score()
        
        return BacktestResult(
            total_return=(self.capital - initial_capital) / initial_capital,
            total_pnl=self.capital - initial_capital,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            brier_score=brier_score,
            total_fees=total_fees,
            final_capital=self.capital,
            trades=self.trades,
            equity_curve=pd.Series([e['capital'] for e in self.equity_curve]),
            by_market_type=by_market
        )


# =============================================================================
# EXAMPLE STRATEGIES
# =============================================================================

def edge_strategy(contract: PredictionContract, market_data: Dict) -> Tuple[str, int, float]:
    """
    Simple edge-based strategy.
    
    Trades when we have sufficient edge vs market price.
    """
    # Get our predicted probability (in real use, this comes from a model)
    # For testing, assume we have slight edge
    edge_pct = market_data.get('edge', 0.08)
    
    # Our prediction = market price + noise + edge
    noise = np.random.uniform(-0.05, 0.05)
    
    # Bias toward correct outcome (simulating edge)
    if contract.outcome == ContractOutcome.YES:
        predicted_prob = contract.yes_price + edge_pct + noise
    else:
        predicted_prob = contract.yes_price - edge_pct + noise
    
    predicted_prob = np.clip(predicted_prob, 0.05, 0.95)
    
    # Calculate edge
    min_edge = market_data.get('min_edge', 0.05)
    
    if predicted_prob > contract.yes_price + min_edge:
        return 'buy_yes', 10, predicted_prob
    elif predicted_prob < contract.yes_price - min_edge:
        return 'buy_no', 10, predicted_prob
    else:
        return 'no_trade', 0, predicted_prob


def kelly_strategy(contract: PredictionContract, market_data: Dict) -> Tuple[str, int, float]:
    """
    Kelly criterion position sizing for prediction markets.
    
    Uses fractional Kelly (25%) for safety.
    """
    # Get predicted probability
    edge_pct = market_data.get('edge', 0.08)
    
    if contract.outcome == ContractOutcome.YES:
        predicted_prob = min(contract.yes_price + edge_pct, 0.95)
    else:
        predicted_prob = max(contract.yes_price - edge_pct, 0.05)
    
    # Kelly calculation
    p = predicted_prob  # Our probability of YES
    q = 1 - p  # Our probability of NO
    
    # Odds offered by market
    if p > contract.yes_price:
        # Bet YES
        b = (1 - contract.yes_price) / contract.yes_price  # Odds
        kelly_fraction = (b * p - q) / b
    else:
        # Bet NO
        b = (1 - contract.no_price) / contract.no_price
        p_no = 1 - p
        q_no = p
        kelly_fraction = (b * p_no - q_no) / b
    
    # Apply fractional Kelly (25%)
    kelly_fraction = kelly_fraction * 0.25
    
    if kelly_fraction <= 0:
        return 'no_trade', 0, predicted_prob
    
    # Calculate position size
    capital = market_data.get('capital', 500)
    position_value = capital * kelly_fraction
    
    if p > contract.yes_price:
        quantity = int(position_value / contract.yes_price)
        return 'buy_yes', max(1, min(quantity, 50)), predicted_prob
    else:
        quantity = int(position_value / contract.no_price)
        return 'buy_no', max(1, min(quantity, 50)), predicted_prob


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("KALSHI PREDICTION MARKET BACKTESTER")
    print("=" * 60)
    
    # Initialize backtester
    config = BacktestConfig(
        initial_capital=500.0,
        max_position_per_contract=50,
        fee_rate=0.10,
        min_edge=0.05
    )
    
    backtester = KalshiBacktester(config)
    
    # Generate synthetic contracts for testing
    print("\nGenerating synthetic contracts...")
    backtester.generate_synthetic_contracts(MarketType.FED, n_contracts=50)
    backtester.generate_synthetic_contracts(MarketType.WEATHER, n_contracts=100)
    backtester.generate_synthetic_contracts(MarketType.SPORTS, n_contracts=100)
    backtester.generate_synthetic_contracts(MarketType.CRYPTO, n_contracts=100)
    
    print(f"Total contracts: {len(backtester.contracts)}")
    
    # Run edge strategy
    print("\n" + "-" * 40)
    print("Running Edge Strategy...")
    result = backtester.run_strategy(
        edge_strategy,
        market_data={'edge': 0.08, 'min_edge': 0.05}
    )
    
    print(f"\nResults:")
    print(f"  Total Return: {result.total_return:.1%}")
    print(f"  Total P&L: ${result.total_pnl:.2f}")
    print(f"  Final Capital: ${result.final_capital:.2f}")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate:.1%}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown:.1%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Brier Score: {result.brier_score:.3f} (lower is better, 0.25 = random)")
    print(f"  Total Fees: ${result.total_fees:.2f}")
    
    print(f"\nBy Market Type:")
    for market, stats in result.by_market_type.items():
        wr = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        print(f"  {market:15s}: {stats['trades']:3d} trades | "
              f"WR: {wr:.0%} | P&L: ${stats['pnl']:+.2f}")
    
    print("\n" + "=" * 60)
