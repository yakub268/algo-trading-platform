"""
Kelly Criterion Optimizer
=========================

Advanced Kelly Criterion optimization with safety features:
- Fractional Kelly sizing
- Multi-asset Kelly optimization
- Risk-adjusted Kelly calculations
- Dynamic Kelly with confidence intervals
- Maximum leverage constraints

Author: Trading Bot Arsenal
Created: February 2026
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import scipy.optimize as opt
import scipy.stats as stats
from scipy.linalg import inv, LinAlgError
import warnings

from ..config.risk_config import RiskManagementConfig, AlertSeverity

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KellyOptimizer')


@dataclass
class KellyResult:
    """Kelly criterion calculation result"""
    symbol: str
    full_kelly_fraction: float
    recommended_kelly_fraction: float
    confidence_interval: Tuple[float, float]

    # Input statistics
    win_rate: float
    average_win: float
    average_loss: float
    reward_risk_ratio: float

    # Risk adjustments
    volatility_adjustment: float
    correlation_adjustment: float
    confidence_adjustment: float
    liquidity_adjustment: float

    # Final recommendation
    recommended_position_size: float
    max_position_size: float
    risk_score: float

    # Metadata
    sample_size: int
    calculation_date: datetime = field(default_factory=datetime.now)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PortfolioKellyResult:
    """Multi-asset Kelly optimization result"""
    individual_kelly: Dict[str, KellyResult]
    optimal_weights: Dict[str, float]
    portfolio_kelly: float

    # Risk metrics
    portfolio_volatility: float
    expected_return: float
    sharpe_ratio: float
    max_drawdown_estimate: float

    # Constraints applied
    leverage_constraint: float
    concentration_constraints: Dict[str, float]
    correlation_adjustments: Dict[str, float]

    # Validation
    kelly_stability_score: float
    recommendation_confidence: float


class KellyOptimizer:
    """
    Advanced Kelly Criterion optimizer with multi-asset support.

    Features:
    - Individual Kelly calculations
    - Portfolio-level Kelly optimization
    - Risk adjustments for volatility, correlation, liquidity
    - Fractional Kelly with safety margins
    - Dynamic Kelly with confidence intervals
    """

    def __init__(self, config: RiskManagementConfig):
        """
        Initialize Kelly optimizer.

        Args:
            config: Risk management configuration
        """
        self.config = config
        self.kelly_config = config.kelly_config

        # Performance tracking
        self.trade_history: Dict[str, List[Dict]] = {}  # symbol -> trades
        self.kelly_history: Dict[str, List[KellyResult]] = {}  # symbol -> kelly results

        # Market data
        self.current_volatilities: Dict[str, float] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None

        logger.info("KellyOptimizer initialized")

    def add_trade_result(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        position_size: float,
        trade_date: datetime
    ):
        """
        Add trade result for Kelly calculation.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size
            trade_date: Trade execution date
        """
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []

        # Calculate return
        trade_return = (exit_price - entry_price) / entry_price

        trade_record = {
            'date': trade_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': trade_return,
            'is_win': trade_return > 0,
            'position_size': position_size,
            'pnl': trade_return * position_size * entry_price
        }

        self.trade_history[symbol].append(trade_record)

        # Keep limited history
        if len(self.trade_history[symbol]) > 1000:
            self.trade_history[symbol] = self.trade_history[symbol][-1000:]

        logger.debug(f"Added trade for {symbol}: {trade_return:+.2%}")

    def calculate_kelly_single_asset(
        self,
        symbol: str,
        min_trades: int = None,
        confidence_level: float = 0.90
    ) -> KellyResult:
        """
        Calculate Kelly criterion for single asset.

        Args:
            symbol: Trading symbol
            min_trades: Minimum trades required (uses config default)
            confidence_level: Confidence level for intervals

        Returns:
            Kelly calculation result
        """
        if min_trades is None:
            min_trades = self.kelly_config.min_trade_sample

        warnings = []

        # Check if we have sufficient trade history
        if symbol not in self.trade_history:
            return self._create_empty_kelly_result(symbol, ["No trade history available"])

        trades = self.trade_history[symbol]

        if len(trades) < min_trades:
            warnings.append(f"Insufficient trades: {len(trades)} < {min_trades}")

        if len(trades) < 5:  # Absolute minimum
            return self._create_empty_kelly_result(symbol, warnings)

        try:
            # Calculate basic statistics
            returns = [trade['return'] for trade in trades]
            wins = [r for r in returns if r > 0]
            losses = [abs(r) for r in returns if r < 0]

            if not wins or not losses:
                warnings.append("No wins or no losses in sample")
                return self._create_empty_kelly_result(symbol, warnings)

            # Core Kelly parameters
            win_rate = len(wins) / len(returns)
            average_win = np.mean(wins)
            average_loss = np.mean(losses)
            reward_risk_ratio = average_win / average_loss

            # Calculate full Kelly
            full_kelly = self._calculate_full_kelly(win_rate, reward_risk_ratio)

            # Risk adjustments
            volatility_adj = self._calculate_volatility_adjustment(symbol, returns)
            correlation_adj = self._calculate_correlation_adjustment(symbol)
            confidence_adj = self._calculate_confidence_adjustment(len(trades), confidence_level)
            liquidity_adj = self._calculate_liquidity_adjustment(symbol)

            # Apply safety fraction
            strategy_fraction = self.kelly_config.strategy_kelly_fractions.get(
                symbol, self.kelly_config.default_kelly_fraction
            )

            # Combined adjustment
            total_adjustment = (volatility_adj * correlation_adj *
                              confidence_adj * liquidity_adj)

            recommended_kelly = full_kelly * strategy_fraction * total_adjustment

            # Calculate confidence interval
            kelly_std = self._calculate_kelly_standard_error(returns, win_rate,
                                                           average_win, average_loss)
            ci_lower = recommended_kelly - 1.96 * kelly_std
            ci_upper = recommended_kelly + 1.96 * kelly_std

            # Position sizing
            max_position = self.config.portfolio_limits.max_single_position
            recommended_position = min(recommended_kelly, max_position)

            # Risk score (higher = riskier)
            risk_score = self._calculate_risk_score(
                full_kelly, recommended_kelly, np.std(returns), len(trades)
            )

            # Validation warnings
            if full_kelly < 0:
                warnings.append("Negative Kelly - strategy may not be profitable")
            if full_kelly > 1.0:
                warnings.append(f"Very high Kelly ({full_kelly:.2%}) - high risk")
            if len(trades) < 50:
                warnings.append("Low sample size - results may be unstable")

            return KellyResult(
                symbol=symbol,
                full_kelly_fraction=full_kelly,
                recommended_kelly_fraction=recommended_kelly,
                confidence_interval=(ci_lower, ci_upper),
                win_rate=win_rate,
                average_win=average_win,
                average_loss=average_loss,
                reward_risk_ratio=reward_risk_ratio,
                volatility_adjustment=volatility_adj,
                correlation_adjustment=correlation_adj,
                confidence_adjustment=confidence_adj,
                liquidity_adjustment=liquidity_adj,
                recommended_position_size=recommended_position,
                max_position_size=max_position,
                risk_score=risk_score,
                sample_size=len(trades),
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Kelly calculation failed for {symbol}: {e}")
            return self._create_empty_kelly_result(symbol, [f"Calculation error: {str(e)}"])

    def optimize_portfolio_kelly(
        self,
        symbols: List[str],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> PortfolioKellyResult:
        """
        Optimize Kelly allocation across multiple assets.

        Args:
            symbols: List of symbols to optimize
            correlation_matrix: Asset correlation matrix

        Returns:
            Portfolio Kelly optimization result
        """
        try:
            # Calculate individual Kelly fractions
            individual_kelly = {}
            for symbol in symbols:
                kelly_result = self.calculate_kelly_single_asset(symbol)
                individual_kelly[symbol] = kelly_result

            # Filter out symbols with insufficient data or negative Kelly
            valid_symbols = [
                symbol for symbol, result in individual_kelly.items()
                if result.recommended_kelly_fraction > 0 and not result.warnings
            ]

            if len(valid_symbols) < 2:
                logger.warning("Insufficient valid symbols for portfolio optimization")
                return self._create_empty_portfolio_kelly(individual_kelly)

            # Set up optimization
            n_assets = len(valid_symbols)

            # Extract expected returns and Kelly fractions
            kelly_fractions = np.array([
                individual_kelly[symbol].recommended_kelly_fraction
                for symbol in valid_symbols
            ])

            expected_returns = np.array([
                individual_kelly[symbol].win_rate * individual_kelly[symbol].average_win -
                (1 - individual_kelly[symbol].win_rate) * individual_kelly[symbol].average_loss
                for symbol in valid_symbols
            ])

            # Build covariance matrix
            if correlation_matrix is not None:
                cov_matrix = self._build_covariance_matrix(valid_symbols, correlation_matrix)
            else:
                # Assume independence (identity correlation matrix)
                volatilities = np.array([
                    self.current_volatilities.get(symbol, 0.20)
                    for symbol in valid_symbols
                ])
                cov_matrix = np.diag(volatilities ** 2)

            # Portfolio Kelly optimization objective
            def portfolio_kelly_objective(weights):
                """Objective function for portfolio Kelly optimization"""
                weights = np.array(weights)

                # Portfolio expected return
                port_return = np.dot(weights, expected_returns)

                # Portfolio variance
                port_variance = np.dot(weights, np.dot(cov_matrix, weights))

                # Kelly fraction for portfolio
                if port_variance > 0:
                    kelly_fraction = port_return / port_variance
                else:
                    kelly_fraction = 0

                # Maximize Kelly (minimize negative Kelly)
                return -kelly_fraction

            # Constraints
            constraints = [
                # Weights sum to target allocation (less than 1 for safety)
                {'type': 'eq', 'fun': lambda x: sum(x) - 0.8},

                # Individual position limits
                *[{'type': 'ineq', 'fun': lambda x, i=i:
                   self.config.portfolio_limits.max_single_position - x[i]}
                  for i in range(n_assets)]
            ]

            # Bounds (0 to max position size)
            bounds = [(0, self.config.portfolio_limits.max_single_position)
                     for _ in range(n_assets)]

            # Initial guess (equal weight within total allocation)
            initial_weights = np.full(n_assets, 0.8 / n_assets)

            # Optimize
            result = opt.minimize(
                portfolio_kelly_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                optimal_weights_array = result.x
                optimal_weights = {
                    symbol: weight
                    for symbol, weight in zip(valid_symbols, optimal_weights_array)
                }
            else:
                logger.warning("Portfolio optimization failed, using individual Kelly")
                # Fallback to scaled individual Kelly fractions
                total_kelly = sum(kelly_fractions)
                scale_factor = 0.8 / max(total_kelly, 0.01)

                optimal_weights = {
                    symbol: individual_kelly[symbol].recommended_kelly_fraction * scale_factor
                    for symbol in valid_symbols
                }

            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                optimal_weights, individual_kelly, correlation_matrix
            )

            return PortfolioKellyResult(
                individual_kelly=individual_kelly,
                optimal_weights=optimal_weights,
                portfolio_kelly=portfolio_metrics['portfolio_kelly'],
                portfolio_volatility=portfolio_metrics['volatility'],
                expected_return=portfolio_metrics['expected_return'],
                sharpe_ratio=portfolio_metrics['sharpe_ratio'],
                max_drawdown_estimate=portfolio_metrics['max_drawdown'],
                leverage_constraint=0.8,
                concentration_constraints={
                    symbol: self.config.portfolio_limits.max_single_position
                    for symbol in symbols
                },
                correlation_adjustments={},
                kelly_stability_score=portfolio_metrics['stability_score'],
                recommendation_confidence=portfolio_metrics['confidence']
            )

        except Exception as e:
            logger.error(f"Portfolio Kelly optimization failed: {e}")
            return self._create_empty_portfolio_kelly({})

    def update_volatilities(self, volatilities: Dict[str, float]):
        """Update current volatility estimates"""
        self.current_volatilities.update(volatilities)

    def update_correlation_matrix(self, correlation_matrix: pd.DataFrame):
        """Update correlation matrix"""
        self.correlation_matrix = correlation_matrix

    def _calculate_full_kelly(self, win_rate: float, reward_risk_ratio: float) -> float:
        """Calculate full Kelly criterion"""
        # Kelly formula: f* = (bp - q) / b
        # where b = reward/risk ratio, p = win rate, q = 1 - p
        p = win_rate
        q = 1 - p
        b = reward_risk_ratio

        if b <= 0:
            return 0.0

        kelly = (b * p - q) / b
        return max(0.0, kelly)

    def _calculate_volatility_adjustment(
        self,
        symbol: str,
        returns: List[float]
    ) -> float:
        """Calculate volatility-based adjustment to Kelly"""
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        # Higher volatility reduces Kelly fraction
        # Adjustment between 0.5 and 1.0
        vol_adjustment = 1.0 / (1.0 + volatility * 2)
        return max(0.5, min(1.0, vol_adjustment))

    def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calculate correlation-based adjustment"""
        if self.correlation_matrix is None:
            return 1.0

        if symbol not in self.correlation_matrix.index:
            return 1.0

        # Average absolute correlation with other assets
        correlations = self.correlation_matrix.loc[symbol].abs()
        avg_correlation = correlations.mean()

        # Higher correlation reduces allocation
        corr_adjustment = 1.0 - (avg_correlation * 0.5)
        return max(0.5, corr_adjustment)

    def _calculate_confidence_adjustment(self, sample_size: int, confidence_level: float) -> float:
        """Calculate confidence-based adjustment for sample size"""
        min_sample = self.kelly_config.min_trade_sample

        if sample_size >= min_sample * 2:
            return 1.0
        elif sample_size >= min_sample:
            # Linear interpolation between 0.7 and 1.0
            return 0.7 + 0.3 * (sample_size - min_sample) / min_sample
        else:
            # Very small sample - significant reduction
            return 0.3 + 0.4 * (sample_size / min_sample)

    @staticmethod
    def _is_prediction_market_ticker(symbol: str) -> bool:
        """Check if a ticker is a prediction market contract (e.g. Kalshi).
        Kalshi tickers start with 'KX' and contain hyphens like 'KXHIGHNY-26FEB04-T42'."""
        s = symbol.upper()
        return s.startswith('KX') or ('-T' in s and '-' in s and any(c.isdigit() for c in s))

    def _calculate_liquidity_adjustment(self, symbol: str) -> float:
        """Calculate liquidity-based adjustment"""
        # Simplified liquidity adjustment
        # In practice, would use bid-ask spreads, volume, etc.

        # Prediction market contracts have lower liquidity than stocks
        if self._is_prediction_market_ticker(symbol):
            return 0.7  # 30% reduction for prediction market contracts

        # Major assets get full allocation
        major_assets = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'EUR/USD', 'BTC']

        if symbol in major_assets:
            return 1.0
        else:
            return 0.8  # 20% reduction for less liquid assets

    def _calculate_kelly_standard_error(
        self,
        returns: List[float],
        win_rate: float,
        average_win: float,
        average_loss: float
    ) -> float:
        """Calculate standard error of Kelly estimate"""
        n = len(returns)

        if n < 10:
            return 0.1  # High uncertainty for small samples

        # Simplified standard error calculation
        # In practice, would use more sophisticated bootstrap methods
        return_std = np.std(returns)
        kelly_se = return_std / np.sqrt(n) * 2  # Rough approximation

        return kelly_se

    def _calculate_risk_score(
        self,
        full_kelly: float,
        recommended_kelly: float,
        volatility: float,
        sample_size: int
    ) -> float:
        """Calculate risk score (0-100, higher = riskier)"""
        risk_factors = []

        # Kelly magnitude risk
        kelly_risk = min(full_kelly * 100, 50)
        risk_factors.append(kelly_risk)

        # Volatility risk
        vol_risk = min(volatility * 100, 30)
        risk_factors.append(vol_risk)

        # Sample size risk
        sample_risk = max(0, 20 - sample_size / 5)
        risk_factors.append(sample_risk)

        return sum(risk_factors)

    def _build_covariance_matrix(
        self,
        symbols: List[str],
        correlation_matrix: pd.DataFrame
    ) -> np.ndarray:
        """Build covariance matrix from correlations and volatilities"""
        n = len(symbols)
        cov_matrix = np.zeros((n, n))

        volatilities = [
            self.current_volatilities.get(symbol, 0.20)
            for symbol in symbols
        ]

        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    cov_matrix[i, j] = volatilities[i] ** 2
                else:
                    if (symbol1 in correlation_matrix.index and
                        symbol2 in correlation_matrix.columns):
                        correlation = correlation_matrix.loc[symbol1, symbol2]
                    else:
                        correlation = 0.0  # Assume independence

                    cov_matrix[i, j] = (volatilities[i] * volatilities[j] * correlation)

        return cov_matrix

    def _calculate_portfolio_metrics(
        self,
        optimal_weights: Dict[str, float],
        individual_kelly: Dict[str, KellyResult],
        correlation_matrix: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        symbols = list(optimal_weights.keys())
        weights = np.array(list(optimal_weights.values()))

        # Expected returns
        expected_returns = np.array([
            individual_kelly[symbol].win_rate * individual_kelly[symbol].average_win -
            (1 - individual_kelly[symbol].win_rate) * individual_kelly[symbol].average_loss
            for symbol in symbols
        ])

        portfolio_return = np.dot(weights, expected_returns)

        # Portfolio volatility (simplified)
        if correlation_matrix is not None:
            cov_matrix = self._build_covariance_matrix(symbols, correlation_matrix)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
        else:
            # Assume independence
            individual_vols = np.array([
                self.current_volatilities.get(symbol, 0.20)
                for symbol in symbols
            ])
            portfolio_volatility = np.sqrt(np.sum((weights * individual_vols) ** 2))

        # Sharpe ratio (assuming 4% risk-free rate)
        risk_free_rate = 0.04
        sharpe_ratio = (portfolio_return - risk_free_rate) / max(portfolio_volatility, 0.001)

        # Portfolio Kelly
        if portfolio_volatility > 0:
            portfolio_kelly = portfolio_return / (portfolio_volatility ** 2)
        else:
            portfolio_kelly = 0.0

        # Max drawdown estimate (rough)
        max_drawdown = portfolio_volatility * 2.5  # Simplified estimate

        # Stability score based on individual Kelly confidence
        stability_scores = [
            1.0 / (1.0 + len(result.warnings))
            for result in individual_kelly.values()
        ]
        stability_score = np.mean(stability_scores)

        # Overall confidence
        sample_sizes = [result.sample_size for result in individual_kelly.values()]
        min_sample = min(sample_sizes) if sample_sizes else 0
        confidence = min(1.0, min_sample / self.kelly_config.min_trade_sample)

        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_kelly': portfolio_kelly,
            'max_drawdown': max_drawdown,
            'stability_score': stability_score,
            'confidence': confidence
        }

    def _create_empty_kelly_result(self, symbol: str, warnings: List[str]) -> KellyResult:
        """Create empty Kelly result for error cases"""
        return KellyResult(
            symbol=symbol,
            full_kelly_fraction=0.0,
            recommended_kelly_fraction=0.0,
            confidence_interval=(0.0, 0.0),
            win_rate=0.0,
            average_win=0.0,
            average_loss=0.0,
            reward_risk_ratio=0.0,
            volatility_adjustment=1.0,
            correlation_adjustment=1.0,
            confidence_adjustment=1.0,
            liquidity_adjustment=1.0,
            recommended_position_size=0.0,
            max_position_size=self.config.portfolio_limits.max_single_position,
            risk_score=100.0,
            sample_size=0,
            warnings=warnings
        )

    def _create_empty_portfolio_kelly(
        self,
        individual_kelly: Dict[str, KellyResult]
    ) -> PortfolioKellyResult:
        """Create empty portfolio Kelly result"""
        return PortfolioKellyResult(
            individual_kelly=individual_kelly,
            optimal_weights={},
            portfolio_kelly=0.0,
            portfolio_volatility=0.0,
            expected_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown_estimate=0.0,
            leverage_constraint=0.8,
            concentration_constraints={},
            correlation_adjustments={},
            kelly_stability_score=0.0,
            recommendation_confidence=0.0
        )

    def get_kelly_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get comprehensive Kelly optimization summary"""
        portfolio_result = self.optimize_portfolio_kelly(symbols)

        return {
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': len(symbols),
            'portfolio_metrics': {
                'portfolio_kelly': f"{portfolio_result.portfolio_kelly:.2%}",
                'expected_return': f"{portfolio_result.expected_return:.2%}",
                'volatility': f"{portfolio_result.portfolio_volatility:.2%}",
                'sharpe_ratio': f"{portfolio_result.sharpe_ratio:.2f}",
                'max_drawdown_estimate': f"{portfolio_result.max_drawdown_estimate:.2%}"
            },
            'optimal_weights': {
                symbol: f"{weight:.1%}"
                for symbol, weight in portfolio_result.optimal_weights.items()
            },
            'individual_kelly': {
                symbol: {
                    'full_kelly': f"{result.full_kelly_fraction:.2%}",
                    'recommended_kelly': f"{result.recommended_kelly_fraction:.2%}",
                    'win_rate': f"{result.win_rate:.1%}",
                    'reward_risk_ratio': f"{result.reward_risk_ratio:.2f}",
                    'sample_size': result.sample_size,
                    'risk_score': f"{result.risk_score:.0f}/100",
                    'warnings': result.warnings
                }
                for symbol, result in portfolio_result.individual_kelly.items()
            },
            'recommendation_confidence': f"{portfolio_result.recommendation_confidence:.0%}",
            'stability_score': f"{portfolio_result.kelly_stability_score:.0%}"
        }


if __name__ == "__main__":
    from ..config.risk_config import load_risk_config

    # Test Kelly optimizer
    config = load_risk_config()
    optimizer = KellyOptimizer(config)

    # Add some sample trades for testing
    test_symbols = ['SPY', 'QQQ', 'AAPL']

    # Simulate trade history
    for symbol in test_symbols:
        for i in range(50):
            # Simulate random trades with slight positive edge
            if np.random.random() < 0.55:  # 55% win rate
                exit_price = 100 * (1 + np.random.uniform(0.01, 0.03))  # 1-3% gain
            else:
                exit_price = 100 * (1 - np.random.uniform(0.01, 0.02))  # 1-2% loss

            optimizer.add_trade_result(
                symbol=symbol,
                entry_price=100.0,
                exit_price=exit_price,
                position_size=1000,
                trade_date=datetime.now() - timedelta(days=i)
            )

    # Calculate Kelly for single asset
    spy_kelly = optimizer.calculate_kelly_single_asset('SPY')
    print(f"SPY Kelly Result:")
    print(f"  Full Kelly: {spy_kelly.full_kelly_fraction:.2%}")
    print(f"  Recommended: {spy_kelly.recommended_kelly_fraction:.2%}")
    print(f"  Win Rate: {spy_kelly.win_rate:.1%}")
    print(f"  Risk Score: {spy_kelly.risk_score:.0f}/100")

    # Portfolio optimization
    portfolio_result = optimizer.optimize_portfolio_kelly(test_symbols)
    print(f"\nPortfolio Kelly Optimization:")
    print(f"  Portfolio Kelly: {portfolio_result.portfolio_kelly:.2%}")
    print(f"  Expected Return: {portfolio_result.expected_return:.2%}")
    print(f"  Sharpe Ratio: {portfolio_result.sharpe_ratio:.2f}")
    print(f"  Optimal Weights:")
    for symbol, weight in portfolio_result.optimal_weights.items():
        print(f"    {symbol}: {weight:.1%}")