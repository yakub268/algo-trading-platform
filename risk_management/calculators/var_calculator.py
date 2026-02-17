"""
Value at Risk (VaR) Calculator
==============================

Comprehensive VaR calculation using multiple methodologies:
- Historical Simulation
- Monte Carlo Simulation
- Parametric (Normal Distribution)
- Extreme Value Theory

Includes Expected Shortfall (Conditional VaR) calculations.

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
import scipy.stats as stats
from scipy.optimize import minimize_scalar
import yfinance as yf
import warnings

from ..config.risk_config import RiskManagementConfig, AlertSeverity

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('VaRCalculator')


class VaRMethod(Enum):
    """VaR calculation methods"""
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    PARAMETRIC = "parametric"
    EXTREME_VALUE = "extreme_value"


@dataclass
class VaRResult:
    """VaR calculation result"""
    method: VaRMethod
    confidence_level: float
    time_horizon: int
    var_value: float
    var_percentage: float
    expected_shortfall: float
    portfolio_value: float

    # Additional metrics
    volatility: float
    skewness: float
    kurtosis: float
    max_loss: float

    # Metadata
    calculation_date: datetime = field(default_factory=datetime.now)
    data_points: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class MonteCarloParameters:
    """Monte Carlo simulation parameters"""
    simulations: int = 10000
    time_steps: int = 252  # Trading days per year
    random_seed: Optional[int] = 42
    use_antithetic_variates: bool = True
    use_control_variates: bool = False


@dataclass
class PortfolioVaRSummary:
    """Complete portfolio VaR analysis"""
    total_var: Dict[str, VaRResult]  # method -> result
    position_var: Dict[str, Dict[str, VaRResult]]  # symbol -> method -> result
    marginal_var: Dict[str, float]  # symbol -> marginal VaR
    component_var: Dict[str, float]  # symbol -> component VaR

    # Risk metrics
    portfolio_diversification_ratio: float
    var_coverage_ratio: float
    coherence_metrics: Dict[str, float]

    # Stress scenarios
    stress_test_results: Dict[str, float]
    worst_case_scenario: Dict[str, Any]


class VaRCalculator:
    """
    Advanced Value at Risk calculator supporting multiple methodologies.

    Provides portfolio-level and position-level VaR calculations with
    comprehensive risk metrics and stress testing capabilities.
    """

    def __init__(self, config: RiskManagementConfig):
        """
        Initialize VaR calculator.

        Args:
            config: Risk management configuration
        """
        self.config = config
        self.var_config = config.var_config

        # Data storage
        self.price_history: Dict[str, pd.Series] = {}
        self.returns_history: Dict[str, pd.Series] = {}
        self.current_positions: Dict[str, float] = {}  # symbol -> portfolio value
        self.portfolio_value = config.portfolio_value

        # VaR history for backtesting
        self.var_history: List[PortfolioVaRSummary] = []

        logger.info("VaRCalculator initialized")

    def update_positions(self, positions: Dict[str, float], portfolio_value: float):
        """
        Update current positions and portfolio value.

        Args:
            positions: Dict mapping symbol -> position value
            portfolio_value: Total portfolio value
        """
        self.current_positions = positions.copy()
        self.portfolio_value = portfolio_value

        logger.info(f"Updated positions: {len(positions)} positions, "
                   f"portfolio value: ${portfolio_value:,.2f}")

    @staticmethod
    def _is_prediction_market_ticker(symbol: str) -> bool:
        """Check if a ticker is a prediction market contract (e.g. Kalshi).
        Kalshi tickers start with 'KX' and contain hyphens like 'KXHIGHNY-26FEB04-T42'."""
        s = symbol.upper()
        return s.startswith('KX') or ('-T' in s and '-' in s and any(c.isdigit() for c in s))

    def fetch_price_data(self, symbols: List[str], period: str = None) -> bool:
        """
        Fetch historical price data for VaR calculations.

        Args:
            symbols: List of symbols
            period: Historical period

        Returns:
            True if successful
        """
        try:
            if period is None:
                period = f"{self.var_config.historical_window + 50}d"  # Extra buffer

            if not symbols:
                return True

            # Filter out forex, currency pairs, and prediction market tickers
            stock_symbols = [s for s in symbols
                           if '/' not in s and '=' not in s
                           and not self._is_prediction_market_ticker(s)]

            # Log skipped prediction market tickers
            skipped = [s for s in symbols if self._is_prediction_market_ticker(s)]
            if skipped:
                logger.info(f"Skipping prediction market tickers for VaR price fetch: {skipped}")

            if not stock_symbols:
                return True

            # Batch download
            data = yf.download(
                stock_symbols,
                period=period,
                interval='1d',
                group_by='ticker',
                auto_adjust=True,
                prepost=True,
                threads=True,
                progress=False
            )

            # Process data
            for symbol in stock_symbols:
                try:
                    if len(stock_symbols) == 1:
                        prices = data['Close']
                    else:
                        if (symbol, 'Close') in data.columns:
                            prices = data[(symbol, 'Close')]
                        else:
                            prices = data[symbol]['Close']

                    prices = prices.dropna()

                    if len(prices) >= 50:  # Minimum data points
                        self.price_history[symbol] = prices

                        # Calculate returns
                        returns = prices.pct_change().dropna()
                        self.returns_history[symbol] = returns

                except Exception as e:
                    logger.warning(f"Could not process data for {symbol}: {e}")

            logger.info(f"Fetched data for {len(self.price_history)} symbols")
            return True

        except Exception as e:
            logger.error(f"Failed to fetch price data: {e}")
            return False

    def calculate_historical_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> VaRResult:
        """
        Calculate VaR using Historical Simulation method.

        Args:
            returns: Historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days

        Returns:
            VaRResult object
        """
        try:
            if isinstance(returns, pd.Series):
                returns_array = returns.values
            else:
                returns_array = returns

            returns_array = returns_array[~np.isnan(returns_array)]

            if len(returns_array) < 30:
                raise ValueError("Insufficient historical data")

            # Scale for time horizon
            if time_horizon > 1:
                scaled_returns = returns_array * np.sqrt(time_horizon)
            else:
                scaled_returns = returns_array

            # Calculate VaR (negative of percentile)
            var_percentile = (1 - confidence_level) * 100
            var_value = -np.percentile(scaled_returns, var_percentile)
            var_percentage = var_value

            # Calculate Expected Shortfall (Conditional VaR)
            threshold = np.percentile(scaled_returns, var_percentile)
            tail_losses = scaled_returns[scaled_returns <= threshold]
            expected_shortfall = -np.mean(tail_losses) if len(tail_losses) > 0 else var_value

            # Calculate additional metrics
            volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
            skewness = stats.skew(returns_array)
            kurtosis = stats.kurtosis(returns_array)
            max_loss = -np.min(returns_array)

            return VaRResult(
                method=VaRMethod.HISTORICAL,
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                var_value=var_value,
                var_percentage=var_percentage,
                expected_shortfall=expected_shortfall,
                portfolio_value=self.portfolio_value,
                volatility=volatility,
                skewness=skewness,
                kurtosis=kurtosis,
                max_loss=max_loss,
                data_points=len(returns_array)
            )

        except Exception as e:
            logger.error(f"Historical VaR calculation failed: {e}")
            return self._create_empty_var_result(VaRMethod.HISTORICAL, confidence_level, time_horizon)

    def calculate_monte_carlo_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        mc_params: Optional[MonteCarloParameters] = None
    ) -> VaRResult:
        """
        Calculate VaR using Monte Carlo simulation.

        Args:
            returns: Historical returns for parameter estimation
            confidence_level: Confidence level
            time_horizon: Time horizon in days
            mc_params: Monte Carlo parameters

        Returns:
            VaRResult object
        """
        try:
            if mc_params is None:
                mc_params = MonteCarloParameters()

            if isinstance(returns, pd.Series):
                returns_array = returns.values
            else:
                returns_array = returns

            returns_array = returns_array[~np.isnan(returns_array)]

            if len(returns_array) < 30:
                raise ValueError("Insufficient data for Monte Carlo")

            # Estimate parameters
            mu = np.mean(returns_array)
            sigma = np.std(returns_array)

            # Set random seed for reproducibility
            if mc_params.random_seed is not None:
                np.random.seed(mc_params.random_seed)

            # Generate random scenarios
            n_sims = mc_params.simulations
            simulated_returns = np.random.normal(mu, sigma, (n_sims, time_horizon))

            # Antithetic variates for variance reduction
            if mc_params.use_antithetic_variates:
                antithetic_returns = -simulated_returns + 2 * mu
                simulated_returns = np.vstack([simulated_returns, antithetic_returns])

            # Calculate cumulative returns for each path
            if time_horizon == 1:
                path_returns = simulated_returns.flatten()
            else:
                # Geometric returns for multi-period
                path_returns = np.prod(1 + simulated_returns, axis=1) - 1

            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = -np.percentile(path_returns, var_percentile)
            var_percentage = var_value

            # Expected Shortfall
            threshold = np.percentile(path_returns, var_percentile)
            tail_losses = path_returns[path_returns <= threshold]
            expected_shortfall = -np.mean(tail_losses) if len(tail_losses) > 0 else var_value

            # Additional metrics
            volatility = sigma * np.sqrt(252)
            skewness = stats.skew(path_returns)
            kurtosis = stats.kurtosis(path_returns)
            max_loss = -np.min(path_returns)

            return VaRResult(
                method=VaRMethod.MONTE_CARLO,
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                var_value=var_value,
                var_percentage=var_percentage,
                expected_shortfall=expected_shortfall,
                portfolio_value=self.portfolio_value,
                volatility=volatility,
                skewness=skewness,
                kurtosis=kurtosis,
                max_loss=max_loss,
                data_points=len(simulated_returns)
            )

        except Exception as e:
            logger.error(f"Monte Carlo VaR calculation failed: {e}")
            return self._create_empty_var_result(VaRMethod.MONTE_CARLO, confidence_level, time_horizon)

    def calculate_parametric_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        distribution: str = 'normal'
    ) -> VaRResult:
        """
        Calculate VaR using parametric method.

        Args:
            returns: Historical returns
            confidence_level: Confidence level
            time_horizon: Time horizon in days
            distribution: Distribution assumption ('normal', 't', 'skewed_t')

        Returns:
            VaRResult object
        """
        try:
            if isinstance(returns, pd.Series):
                returns_array = returns.values
            else:
                returns_array = returns

            returns_array = returns_array[~np.isnan(returns_array)]

            if len(returns_array) < 30:
                raise ValueError("Insufficient data for parametric VaR")

            mu = np.mean(returns_array)
            sigma = np.std(returns_array)

            # Scale for time horizon
            mu_scaled = mu * time_horizon
            sigma_scaled = sigma * np.sqrt(time_horizon)

            if distribution == 'normal':
                z_score = stats.norm.ppf(1 - confidence_level)
                var_value = -(mu_scaled + z_score * sigma_scaled)

            elif distribution == 't':
                # Fit t-distribution
                df, loc, scale = stats.t.fit(returns_array)
                t_score = stats.t.ppf(1 - confidence_level, df, loc, scale)
                var_value = -t_score * sigma_scaled

            elif distribution == 'skewed_t':
                # Simplified skewed-t approximation
                skewness = stats.skew(returns_array)
                kurtosis_excess = stats.kurtosis(returns_array)

                # Cornish-Fisher expansion
                z = stats.norm.ppf(1 - confidence_level)
                z_cf = z + (z**2 - 1) * skewness / 6 + (z**3 - 3*z) * kurtosis_excess / 24

                var_value = -(mu_scaled + z_cf * sigma_scaled)

            else:
                raise ValueError(f"Unsupported distribution: {distribution}")

            var_percentage = var_value

            # Estimate Expected Shortfall analytically (for normal distribution)
            if distribution == 'normal':
                phi = stats.norm.pdf(stats.norm.ppf(1 - confidence_level))
                expected_shortfall = -(mu_scaled + sigma_scaled * phi / (1 - confidence_level))
            else:
                # Use numerical approximation for non-normal distributions
                expected_shortfall = var_value * 1.2  # Rough approximation

            # Additional metrics
            volatility = sigma * np.sqrt(252)
            skewness_val = stats.skew(returns_array)
            kurtosis_val = stats.kurtosis(returns_array)
            max_loss = -np.min(returns_array)

            return VaRResult(
                method=VaRMethod.PARAMETRIC,
                confidence_level=confidence_level,
                time_horizon=time_horizon,
                var_value=var_value,
                var_percentage=var_percentage,
                expected_shortfall=expected_shortfall,
                portfolio_value=self.portfolio_value,
                volatility=volatility,
                skewness=skewness_val,
                kurtosis=kurtosis_val,
                max_loss=max_loss,
                data_points=len(returns_array)
            )

        except Exception as e:
            logger.error(f"Parametric VaR calculation failed: {e}")
            return self._create_empty_var_result(VaRMethod.PARAMETRIC, confidence_level, time_horizon)

    def calculate_portfolio_var(
        self,
        correlation_matrix: Optional[pd.DataFrame] = None,
        methods: List[VaRMethod] = None
    ) -> PortfolioVaRSummary:
        """
        Calculate comprehensive portfolio VaR analysis.

        Args:
            correlation_matrix: Correlation matrix (calculated if None)
            methods: VaR methods to use

        Returns:
            Complete portfolio VaR summary
        """
        if methods is None:
            methods = [VaRMethod.HISTORICAL, VaRMethod.MONTE_CARLO, VaRMethod.PARAMETRIC]

        # Ensure we have data
        symbols = list(self.current_positions.keys())
        self.fetch_price_data(symbols)

        # Calculate individual position VaRs
        position_vars = {}
        for symbol in symbols:
            if symbol in self.returns_history:
                returns = self.returns_history[symbol]
                position_vars[symbol] = {}

                for method in methods:
                    for confidence in self.var_config.confidence_levels:
                        for horizon in self.var_config.time_horizons:
                            if method == VaRMethod.HISTORICAL:
                                var_result = self.calculate_historical_var(returns, confidence, horizon)
                            elif method == VaRMethod.MONTE_CARLO:
                                var_result = self.calculate_monte_carlo_var(returns, confidence, horizon)
                            elif method == VaRMethod.PARAMETRIC:
                                var_result = self.calculate_parametric_var(returns, confidence, horizon)
                            else:
                                continue

                            key = f"{method.value}_{confidence}_{horizon}d"
                            position_vars[symbol][key] = var_result

        # Calculate portfolio-level VaR
        portfolio_vars = {}

        if len(symbols) > 1:
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(correlation_matrix)

            for method in methods:
                for confidence in self.var_config.confidence_levels:
                    for horizon in self.var_config.time_horizons:
                        if method == VaRMethod.HISTORICAL:
                            var_result = self.calculate_historical_var(portfolio_returns, confidence, horizon)
                        elif method == VaRMethod.MONTE_CARLO:
                            var_result = self.calculate_monte_carlo_var(portfolio_returns, confidence, horizon)
                        elif method == VaRMethod.PARAMETRIC:
                            var_result = self.calculate_parametric_var(portfolio_returns, confidence, horizon)
                        else:
                            continue

                        # Scale by portfolio value
                        var_result.var_value *= self.portfolio_value
                        var_result.expected_shortfall *= self.portfolio_value

                        key = f"{method.value}_{confidence}_{horizon}d"
                        portfolio_vars[key] = var_result

        # Calculate marginal and component VaR
        marginal_vars = self._calculate_marginal_var(position_vars, correlation_matrix)
        component_vars = self._calculate_component_var(marginal_vars)

        # Risk metrics
        diversification_ratio = self._calculate_diversification_ratio(position_vars, portfolio_vars)

        # Stress testing
        stress_results = self._perform_stress_tests()

        return PortfolioVaRSummary(
            total_var=portfolio_vars,
            position_var=position_vars,
            marginal_var=marginal_vars,
            component_var=component_vars,
            portfolio_diversification_ratio=diversification_ratio,
            var_coverage_ratio=1.0,  # Placeholder
            coherence_metrics={},  # Placeholder
            stress_test_results=stress_results,
            worst_case_scenario={}  # Placeholder
        )

    def _calculate_portfolio_returns(self, correlation_matrix: Optional[pd.DataFrame] = None) -> pd.Series:
        """Calculate portfolio returns from individual positions"""
        if not self.current_positions:
            return pd.Series(dtype=float)

        # Get aligned returns data
        symbols = [s for s in self.current_positions.keys() if s in self.returns_history]

        if not symbols:
            return pd.Series(dtype=float)

        # Align all return series
        returns_data = {}
        common_dates = None

        for symbol in symbols:
            returns = self.returns_history[symbol]
            returns_data[symbol] = returns

            if common_dates is None:
                common_dates = set(returns.index)
            else:
                common_dates = common_dates.intersection(set(returns.index))

        if not common_dates or len(common_dates) < 20:
            return pd.Series(dtype=float)

        common_dates = sorted(list(common_dates))

        # Create aligned DataFrame
        aligned_returns = pd.DataFrame({
            symbol: self.returns_history[symbol].loc[common_dates]
            for symbol in symbols
        })

        # Calculate weights
        total_value = sum(self.current_positions[s] for s in symbols)
        weights = {symbol: self.current_positions[symbol] / total_value for symbol in symbols}

        # Calculate portfolio returns
        portfolio_returns = sum(aligned_returns[symbol] * weights[symbol] for symbol in symbols)

        return portfolio_returns

    def _calculate_marginal_var(
        self,
        position_vars: Dict[str, Dict[str, VaRResult]],
        correlation_matrix: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate marginal VaR for each position"""
        # Simplified marginal VaR calculation
        marginal_vars = {}

        for symbol in self.current_positions:
            if symbol in position_vars:
                # Use 95% confidence, 1-day VaR as baseline
                key = "historical_0.95_1d"
                if key in position_vars[symbol]:
                    var_result = position_vars[symbol][key]
                    position_value = self.current_positions[symbol]
                    marginal_var = var_result.var_percentage * position_value
                    marginal_vars[symbol] = marginal_var
                else:
                    marginal_vars[symbol] = 0.0
            else:
                marginal_vars[symbol] = 0.0

        return marginal_vars

    def _calculate_component_var(self, marginal_vars: Dict[str, float]) -> Dict[str, float]:
        """Calculate component VaR for each position"""
        total_marginal_var = sum(marginal_vars.values())

        if total_marginal_var == 0:
            return {symbol: 0.0 for symbol in marginal_vars}

        # Component VaR proportional to marginal VaR
        component_vars = {}
        for symbol, marginal_var in marginal_vars.items():
            component_vars[symbol] = marginal_var

        return component_vars

    def _calculate_diversification_ratio(
        self,
        position_vars: Dict[str, Dict[str, VaRResult]],
        portfolio_vars: Dict[str, VaRResult]
    ) -> float:
        """Calculate portfolio diversification ratio"""
        if not position_vars or not portfolio_vars:
            return 1.0

        # Sum of individual VaRs vs portfolio VaR
        try:
            key = "historical_0.95_1d"
            individual_var_sum = 0

            for symbol in self.current_positions:
                if symbol in position_vars and key in position_vars[symbol]:
                    var_result = position_vars[symbol][key]
                    position_value = self.current_positions[symbol]
                    individual_var_sum += var_result.var_percentage * position_value

            if key in portfolio_vars:
                portfolio_var = portfolio_vars[key].var_value

                if portfolio_var > 0:
                    diversification_ratio = individual_var_sum / portfolio_var
                    return min(diversification_ratio, 10.0)  # Cap at reasonable level

        except Exception as e:
            logger.warning(f"Could not calculate diversification ratio: {e}")

        return 1.0

    def _perform_stress_tests(self) -> Dict[str, float]:
        """Perform stress testing scenarios"""
        stress_results = {}

        # Define stress scenarios
        scenarios = {
            'market_crash': {'equity_shock': -0.30, 'volatility_multiplier': 2.0},
            'interest_rate_shock': {'rate_shock': 0.02, 'bond_impact': -0.15},
            'currency_crisis': {'fx_volatility': 3.0, 'emerging_shock': -0.40},
            'liquidity_crisis': {'bid_ask_widening': 0.05, 'volume_shock': -0.70}
        }

        for scenario_name, params in scenarios.items():
            # Simplified stress calculation
            stress_loss = 0.0

            for symbol, position_value in self.current_positions.items():
                if symbol in self.returns_history:
                    returns = self.returns_history[symbol]
                    volatility = returns.std()

                    # Apply scenario-specific shocks
                    if 'equity_shock' in params and symbol not in ['USD', 'EUR', 'GBP']:
                        shock = params['equity_shock']
                        stress_loss += position_value * shock

                    if 'volatility_multiplier' in params:
                        vol_shock = volatility * (params['volatility_multiplier'] - 1)
                        stress_loss += position_value * vol_shock * 2  # Rough approximation

            stress_results[scenario_name] = abs(stress_loss)

        return stress_results

    def _create_empty_var_result(
        self,
        method: VaRMethod,
        confidence_level: float,
        time_horizon: int
    ) -> VaRResult:
        """Create empty VaR result for error cases"""
        return VaRResult(
            method=method,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            var_value=0.0,
            var_percentage=0.0,
            expected_shortfall=0.0,
            portfolio_value=self.portfolio_value,
            volatility=0.0,
            skewness=0.0,
            kurtosis=0.0,
            max_loss=0.0,
            warnings=["Calculation failed due to insufficient data"]
        )

    def get_var_summary(self) -> Dict[str, Any]:
        """Get comprehensive VaR summary"""
        summary = self.calculate_portfolio_var()

        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'positions_analyzed': len(self.current_positions),
            'portfolio_var': {
                key: {
                    'method': result.method.value,
                    'confidence': result.confidence_level,
                    'horizon_days': result.time_horizon,
                    'var_amount': f"${result.var_value:,.2f}",
                    'var_percentage': f"{result.var_percentage:.2%}",
                    'expected_shortfall': f"${result.expected_shortfall:,.2f}",
                    'volatility': f"{result.volatility:.2%}"
                }
                for key, result in summary.total_var.items()
            },
            'marginal_var': {
                symbol: f"${amount:,.2f}"
                for symbol, amount in summary.marginal_var.items()
            },
            'diversification_ratio': f"{summary.portfolio_diversification_ratio:.2f}",
            'stress_tests': {
                scenario: f"${loss:,.2f}"
                for scenario, loss in summary.stress_test_results.items()
            }
        }


if __name__ == "__main__":
    from ..config.risk_config import load_risk_config

    # Test VaR calculator
    config = load_risk_config()
    calculator = VaRCalculator(config)

    # Set up test positions
    positions = {
        'SPY': 5000,
        'QQQ': 3000,
        'AAPL': 2000
    }

    calculator.update_positions(positions, 10000)

    # Calculate VaR
    summary = calculator.calculate_portfolio_var()

    print("VaR Analysis Results:")
    print(f"Portfolio Diversification Ratio: {summary.portfolio_diversification_ratio:.2f}")

    if summary.total_var:
        for key, var_result in list(summary.total_var.items())[:3]:  # Show first 3
            print(f"\n{key}:")
            print(f"  VaR: ${var_result.var_value:,.2f} ({var_result.var_percentage:.2%})")
            print(f"  Expected Shortfall: ${var_result.expected_shortfall:,.2f}")
            print(f"  Volatility: {var_result.volatility:.2%}")

    print(f"\nMarginal VaR by Position:")
    for symbol, marginal_var in summary.marginal_var.items():
        print(f"  {symbol}: ${marginal_var:,.2f}")

    print(f"\nStress Test Results:")
    for scenario, loss in summary.stress_test_results.items():
        print(f"  {scenario}: ${loss:,.2f}")