"""
Stress Testing Module
====================

Comprehensive stress testing for extreme market scenarios.

Features:
- Historical scenario replay
- Monte Carlo stress simulations
- Custom scenario testing
- Liquidity stress tests
- Correlation breakdown scenarios
- Tail risk analysis

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
from scipy.linalg import cholesky
import warnings

from ..config.risk_config import RiskManagementConfig, AlertSeverity

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('StressTester')


class StressScenario(Enum):
    """Predefined stress scenarios"""
    MARKET_CRASH_2008 = "market_crash_2008"
    FLASH_CRASH_2010 = "flash_crash_2010"
    COVID_CRASH_2020 = "covid_crash_2020"
    TECH_BUBBLE_2000 = "tech_bubble_2000"
    RATE_SHOCK = "rate_shock"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    VOLATILITY_SPIKE = "volatility_spike"
    CURRENCY_CRISIS = "currency_crisis"
    CUSTOM = "custom"


@dataclass
class StressShock:
    """Individual stress shock parameters"""
    asset_class: str
    shock_type: str  # 'return', 'volatility', 'correlation'
    shock_value: float
    duration_days: int = 1
    confidence_level: float = 0.95


@dataclass
class StressResult:
    """Stress test result for single scenario"""
    scenario_name: str
    scenario_type: StressScenario

    # Portfolio impact
    portfolio_pnl: float
    portfolio_pnl_pct: float
    worst_position_pnl: float
    best_position_pnl: float

    # Position-level results
    position_pnls: Dict[str, float]
    position_pnl_pcts: Dict[str, float]

    # Risk metrics under stress
    stressed_var: float
    stressed_volatility: float
    stressed_correlation: float
    liquidity_impact: float

    # Scenario parameters
    shocks_applied: List[StressShock]
    probability_estimate: float
    historical_precedent: Optional[str]

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    warnings: List[str] = field(default_factory=list)


@dataclass
class StressTestSuite:
    """Complete stress testing results"""
    portfolio_value: float
    test_date: datetime
    scenarios_tested: List[str]

    # Individual results
    scenario_results: Dict[str, StressResult]

    # Aggregate metrics
    worst_case_scenario: str
    worst_case_loss: float
    expected_tail_loss: float  # Expected loss in worst 5% scenarios
    stress_var_95: float

    # Diversification analysis
    concentration_risk: float
    correlation_risk: float
    liquidity_risk: float

    # Pass/fail summary
    scenarios_passed: int
    scenarios_failed: int
    critical_failures: List[str]


class StressTester:
    """
    Comprehensive stress testing system for portfolio risk analysis.

    Tests portfolio resilience under extreme market conditions.
    """

    # Historical scenario parameters
    HISTORICAL_SCENARIOS = {
        StressScenario.MARKET_CRASH_2008: {
            'name': '2008 Financial Crisis',
            'shocks': [
                StressShock('equity', 'return', -0.50, 90),
                StressShock('credit', 'return', -0.40, 60),
                StressShock('forex', 'volatility', 2.0, 30),
                StressShock('all', 'correlation', 0.85, 90)
            ],
            'probability': 0.02,  # 2% annual probability
            'precedent': 'Lehman Brothers collapse, global financial crisis'
        },

        StressScenario.FLASH_CRASH_2010: {
            'name': '2010 Flash Crash',
            'shocks': [
                StressShock('equity', 'return', -0.09, 1),
                StressShock('equity', 'volatility', 10.0, 1),
                StressShock('all', 'liquidity', 0.1, 1)
            ],
            'probability': 0.05,
            'precedent': 'May 6, 2010 flash crash'
        },

        StressScenario.COVID_CRASH_2020: {
            'name': 'COVID-19 Market Crash',
            'shocks': [
                StressShock('equity', 'return', -0.35, 30),
                StressShock('oil', 'return', -0.60, 45),
                StressShock('credit', 'return', -0.20, 60),
                StressShock('all', 'volatility', 3.0, 30)
            ],
            'probability': 0.01,
            'precedent': 'COVID-19 pandemic market crash March 2020'
        },

        StressScenario.RATE_SHOCK: {
            'name': 'Interest Rate Shock',
            'shocks': [
                StressShock('bonds', 'return', -0.15, 30),
                StressShock('equity', 'return', -0.20, 60),
                StressShock('real_estate', 'return', -0.25, 180),
                StressShock('forex', 'volatility', 1.5, 90)
            ],
            'probability': 0.10,
            'precedent': 'Fed tightening cycles, bond selloffs'
        },

        StressScenario.LIQUIDITY_CRISIS: {
            'name': 'Liquidity Crisis',
            'shocks': [
                StressShock('all', 'liquidity', 0.2, 30),
                StressShock('all', 'volatility', 2.0, 30),
                StressShock('credit', 'return', -0.30, 60)
            ],
            'probability': 0.03,
            'precedent': 'LTCM crisis 1998, repo market freeze 2008'
        }
    }

    def __init__(self, config: RiskManagementConfig):
        """
        Initialize stress tester.

        Args:
            config: Risk management configuration
        """
        self.config = config
        self.stress_config = config.stress_testing

        # Current portfolio state
        self.portfolio_positions: Dict[str, float] = {}
        self.portfolio_value = config.portfolio_value
        self.correlation_matrix: Optional[pd.DataFrame] = None

        # Historical data for scenario calibration
        self.historical_returns: Dict[str, pd.Series] = {}

        # Results storage
        self.last_test_suite: Optional[StressTestSuite] = None

        logger.info("StressTester initialized")

    def update_portfolio(
        self,
        positions: Dict[str, float],
        portfolio_value: float,
        correlation_matrix: Optional[pd.DataFrame] = None
    ):
        """
        Update portfolio for stress testing.

        Args:
            positions: Dict mapping symbol -> position value
            portfolio_value: Total portfolio value
            correlation_matrix: Asset correlation matrix
        """
        self.portfolio_positions = positions.copy()
        self.portfolio_value = portfolio_value
        self.correlation_matrix = correlation_matrix

        logger.info(f"Portfolio updated for stress testing: {len(positions)} positions, "
                   f"${portfolio_value:,.2f} total value")

    def run_scenario_test(self, scenario: StressScenario, custom_shocks: List[StressShock] = None) -> StressResult:
        """
        Run single stress scenario test.

        Args:
            scenario: Stress scenario to test
            custom_shocks: Custom shocks for CUSTOM scenario

        Returns:
            Stress test result
        """
        try:
            # Get scenario parameters
            if scenario == StressScenario.CUSTOM and custom_shocks:
                scenario_params = {
                    'name': 'Custom Scenario',
                    'shocks': custom_shocks,
                    'probability': 0.05,  # Default
                    'precedent': 'User-defined scenario'
                }
            elif scenario in self.HISTORICAL_SCENARIOS:
                scenario_params = self.HISTORICAL_SCENARIOS[scenario]
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            shocks = scenario_params['shocks']

            # Apply shocks to portfolio
            position_pnls = self._apply_shocks_to_portfolio(shocks)

            # Calculate portfolio-level metrics
            portfolio_pnl = sum(position_pnls.values())
            portfolio_pnl_pct = portfolio_pnl / self.portfolio_value if self.portfolio_value > 0 else 0

            # Position-level metrics
            position_pnl_pcts = {
                symbol: pnl / self.portfolio_positions.get(symbol, 1)
                for symbol, pnl in position_pnls.items()
            }

            worst_position_pnl = min(position_pnls.values()) if position_pnls else 0
            best_position_pnl = max(position_pnls.values()) if position_pnls else 0

            # Stressed risk metrics
            stressed_metrics = self._calculate_stressed_metrics(shocks)

            return StressResult(
                scenario_name=scenario_params['name'],
                scenario_type=scenario,
                portfolio_pnl=portfolio_pnl,
                portfolio_pnl_pct=portfolio_pnl_pct,
                worst_position_pnl=worst_position_pnl,
                best_position_pnl=best_position_pnl,
                position_pnls=position_pnls,
                position_pnl_pcts=position_pnl_pcts,
                stressed_var=stressed_metrics['var'],
                stressed_volatility=stressed_metrics['volatility'],
                stressed_correlation=stressed_metrics['correlation'],
                liquidity_impact=stressed_metrics['liquidity'],
                shocks_applied=shocks,
                probability_estimate=scenario_params['probability'],
                historical_precedent=scenario_params['precedent']
            )

        except Exception as e:
            logger.error(f"Stress test failed for {scenario.value}: {e}")
            return self._create_failed_result(scenario, str(e))

    def run_full_stress_suite(self) -> StressTestSuite:
        """
        Run complete stress testing suite.

        Returns:
            Complete stress test results
        """
        try:
            scenario_results = {}
            test_date = datetime.now()

            # Test all predefined scenarios
            scenarios_to_test = [
                StressScenario.MARKET_CRASH_2008,
                StressScenario.FLASH_CRASH_2010,
                StressScenario.COVID_CRASH_2020,
                StressScenario.RATE_SHOCK,
                StressScenario.LIQUIDITY_CRISIS
            ]

            for scenario in scenarios_to_test:
                result = self.run_scenario_test(scenario)
                scenario_results[scenario.value] = result

            # Add custom extreme scenarios
            extreme_scenarios = self._generate_extreme_scenarios()
            for name, shocks in extreme_scenarios.items():
                result = self.run_scenario_test(StressScenario.CUSTOM, shocks)
                result.scenario_name = name
                scenario_results[name] = result

            # Analyze results
            suite_metrics = self._analyze_stress_suite(scenario_results)

            # Create test suite
            test_suite = StressTestSuite(
                portfolio_value=self.portfolio_value,
                test_date=test_date,
                scenarios_tested=list(scenario_results.keys()),
                scenario_results=scenario_results,
                **suite_metrics
            )

            self.last_test_suite = test_suite

            logger.info(f"Stress testing complete: {len(scenario_results)} scenarios tested")
            return test_suite

        except Exception as e:
            logger.error(f"Full stress suite failed: {e}")
            raise

    def run_monte_carlo_stress(
        self,
        num_simulations: int = 1000,
        confidence_levels: List[float] = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo stress testing.

        Args:
            num_simulations: Number of MC simulations
            confidence_levels: Confidence levels for VaR calculation

        Returns:
            Monte Carlo stress results
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]

        try:
            if not self.portfolio_positions:
                raise ValueError("No portfolio positions to stress test")

            # Generate random shocks
            portfolio_pnls = []

            for _ in range(num_simulations):
                # Generate correlated random shocks
                random_shocks = self._generate_random_shocks()
                position_pnls = self._apply_random_shocks(random_shocks)
                portfolio_pnl = sum(position_pnls.values())
                portfolio_pnls.append(portfolio_pnl)

            portfolio_pnls = np.array(portfolio_pnls)

            # Calculate stress VaR at different confidence levels
            stress_vars = {}
            for confidence in confidence_levels:
                var_percentile = (1 - confidence) * 100
                stress_var = -np.percentile(portfolio_pnls, var_percentile)
                stress_vars[f'var_{confidence}'] = stress_var

            # Expected Shortfall (Conditional VaR)
            var_95_threshold = np.percentile(portfolio_pnls, 5)
            tail_losses = portfolio_pnls[portfolio_pnls <= var_95_threshold]
            expected_shortfall = -np.mean(tail_losses) if len(tail_losses) > 0 else 0

            # Distribution statistics
            mean_loss = np.mean(portfolio_pnls)
            std_loss = np.std(portfolio_pnls)
            min_loss = np.min(portfolio_pnls)
            max_gain = np.max(portfolio_pnls)

            # Probability of large losses
            prob_loss_10pct = np.mean(portfolio_pnls < -0.10 * self.portfolio_value)
            prob_loss_20pct = np.mean(portfolio_pnls < -0.20 * self.portfolio_value)

            return {
                'num_simulations': num_simulations,
                'stress_vars': stress_vars,
                'expected_shortfall': expected_shortfall,
                'mean_loss': mean_loss,
                'volatility': std_loss,
                'worst_case_loss': min_loss,
                'best_case_gain': max_gain,
                'prob_loss_10pct': prob_loss_10pct,
                'prob_loss_20pct': prob_loss_20pct,
                'loss_distribution': portfolio_pnls.tolist()
            }

        except Exception as e:
            logger.error(f"Monte Carlo stress testing failed: {e}")
            return {'error': str(e)}

    def _apply_shocks_to_portfolio(self, shocks: List[StressShock]) -> Dict[str, float]:
        """Apply stress shocks to portfolio positions"""
        position_pnls = {}

        for symbol, position_value in self.portfolio_positions.items():
            total_shock = 0.0

            for shock in shocks:
                # Determine if shock applies to this asset
                if self._shock_applies_to_asset(shock, symbol):
                    if shock.shock_type == 'return':
                        total_shock += shock.shock_value
                    elif shock.shock_type == 'volatility':
                        # Volatility shock contributes to return shock (simplified)
                        vol_contribution = -0.02 * (shock.shock_value - 1.0)
                        total_shock += vol_contribution

            # Calculate P&L
            position_pnl = position_value * total_shock
            position_pnls[symbol] = position_pnl

        return position_pnls

    def _shock_applies_to_asset(self, shock: StressShock, symbol: str) -> bool:
        """Determine if shock applies to specific asset"""
        if shock.asset_class == 'all':
            return True

        # Simple asset classification
        asset_class_mapping = {
            'equity': ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'bonds': ['TLT', 'AGG', 'BND', 'LQD'],
            'credit': ['HYG', 'JNK', 'LQD'],
            'forex': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'],
            'commodities': ['GLD', 'SLV', 'USO', 'UNG'],
            'oil': ['USO', 'XLE'],
            'real_estate': ['VNQ', 'XLRE']
        }

        relevant_symbols = asset_class_mapping.get(shock.asset_class, [])
        return symbol.upper() in [s.upper() for s in relevant_symbols]

    def _calculate_stressed_metrics(self, shocks: List[StressShock]) -> Dict[str, float]:
        """Calculate risk metrics under stress"""
        # Base portfolio volatility (simplified)
        base_vol = 0.15  # 15% annual volatility

        # Apply volatility shocks
        stress_vol_multiplier = 1.0
        for shock in shocks:
            if shock.shock_type == 'volatility':
                stress_vol_multiplier *= shock.shock_value

        stressed_volatility = base_vol * stress_vol_multiplier

        # Estimate stressed VaR (simplified)
        stressed_var = stressed_volatility * 2.33 * self.portfolio_value  # 99% VaR approximation

        # Correlation under stress
        base_correlation = 0.3
        stress_correlation = base_correlation

        for shock in shocks:
            if shock.shock_type == 'correlation':
                stress_correlation = max(stress_correlation, shock.shock_value)

        # Liquidity impact
        liquidity_impact = 0.0
        for shock in shocks:
            if shock.asset_class == 'all' and 'liquidity' in shock.shock_type:
                liquidity_impact = shock.shock_value

        return {
            'var': stressed_var,
            'volatility': stressed_volatility,
            'correlation': stress_correlation,
            'liquidity': liquidity_impact
        }

    def _generate_extreme_scenarios(self) -> Dict[str, List[StressShock]]:
        """Generate additional extreme scenarios"""
        return {
            'Extreme_Market_Crash': [
                StressShock('equity', 'return', -0.60, 60),
                StressShock('bonds', 'return', -0.25, 90),
                StressShock('all', 'volatility', 5.0, 30),
                StressShock('all', 'correlation', 0.95, 60)
            ],

            'Hyperinflation_Scenario': [
                StressShock('bonds', 'return', -0.40, 180),
                StressShock('forex', 'return', -0.30, 180),
                StressShock('commodities', 'return', 0.50, 180),
                StressShock('all', 'volatility', 3.0, 180)
            ],

            'Deflationary_Spiral': [
                StressShock('equity', 'return', -0.45, 365),
                StressShock('credit', 'return', -0.50, 365),
                StressShock('commodities', 'return', -0.40, 365),
                StressShock('all', 'correlation', 0.90, 180)
            ],

            'Technology_Crash': [
                StressShock('equity', 'return', -0.70, 90),  # Focused on tech
                StressShock('all', 'volatility', 4.0, 60),
                StressShock('all', 'liquidity', 0.3, 30)
            ]
        }

    def _generate_random_shocks(self) -> Dict[str, float]:
        """Generate random correlated shocks for Monte Carlo"""
        # Simplified random shock generation
        num_assets = len(self.portfolio_positions)

        if num_assets == 0:
            return {}

        # Generate correlated random returns
        if self.correlation_matrix is not None and not self.correlation_matrix.empty:
            try:
                corr_matrix = self.correlation_matrix.values
                L = cholesky(corr_matrix, lower=True)

                # Generate independent random variables
                random_vars = np.random.normal(0, 1, num_assets)

                # Apply correlation structure
                correlated_shocks = L @ random_vars

                symbols = list(self.portfolio_positions.keys())
                return dict(zip(symbols, correlated_shocks))

            except Exception:
                # Fall back to independent shocks
                pass

        # Independent random shocks
        random_shocks = {}
        for symbol in self.portfolio_positions:
            # Fat-tailed distribution for extreme events
            shock = np.random.standard_t(df=3) * 0.02  # t-distribution with 3 df
            random_shocks[symbol] = shock

        return random_shocks

    def _apply_random_shocks(self, random_shocks: Dict[str, float]) -> Dict[str, float]:
        """Apply random shocks to portfolio"""
        position_pnls = {}

        for symbol, position_value in self.portfolio_positions.items():
            shock = random_shocks.get(symbol, 0.0)
            position_pnl = position_value * shock
            position_pnls[symbol] = position_pnl

        return position_pnls

    def _analyze_stress_suite(self, scenario_results: Dict[str, StressResult]) -> Dict[str, Any]:
        """Analyze stress testing suite results"""
        if not scenario_results:
            return {
                'worst_case_scenario': 'None',
                'worst_case_loss': 0.0,
                'expected_tail_loss': 0.0,
                'stress_var_95': 0.0,
                'concentration_risk': 0.0,
                'correlation_risk': 0.0,
                'liquidity_risk': 0.0,
                'scenarios_passed': 0,
                'scenarios_failed': 0,
                'critical_failures': []
            }

        # Find worst case
        worst_scenario = min(scenario_results.items(), key=lambda x: x[1].portfolio_pnl)
        worst_case_scenario = worst_scenario[0]
        worst_case_loss = worst_scenario[1].portfolio_pnl

        # Calculate tail loss (worst 20% scenarios)
        all_losses = [result.portfolio_pnl for result in scenario_results.values()]
        all_losses.sort()
        tail_size = max(1, int(len(all_losses) * 0.2))
        expected_tail_loss = np.mean(all_losses[:tail_size])

        # Stress VaR (95th percentile loss)
        stress_var_95 = -np.percentile(all_losses, 5)

        # Risk concentration analysis (simplified)
        concentration_risk = self._calculate_concentration_risk()
        correlation_risk = self._calculate_correlation_risk()
        liquidity_risk = self._calculate_liquidity_risk(scenario_results)

        # Pass/fail analysis
        loss_threshold = -0.20 * self.portfolio_value  # 20% loss threshold
        scenarios_passed = sum(1 for result in scenario_results.values()
                             if result.portfolio_pnl > loss_threshold)
        scenarios_failed = len(scenario_results) - scenarios_passed

        critical_failures = [
            name for name, result in scenario_results.items()
            if result.portfolio_pnl < -0.25 * self.portfolio_value  # 25% loss
        ]

        return {
            'worst_case_scenario': worst_case_scenario,
            'worst_case_loss': worst_case_loss,
            'expected_tail_loss': expected_tail_loss,
            'stress_var_95': stress_var_95,
            'concentration_risk': concentration_risk,
            'correlation_risk': correlation_risk,
            'liquidity_risk': liquidity_risk,
            'scenarios_passed': scenarios_passed,
            'scenarios_failed': scenarios_failed,
            'critical_failures': critical_failures
        }

    def _calculate_concentration_risk(self) -> float:
        """Calculate portfolio concentration risk"""
        if not self.portfolio_positions:
            return 0.0

        # Herfindahl index
        total_value = sum(self.portfolio_positions.values())
        if total_value == 0:
            return 0.0

        weights = [value / total_value for value in self.portfolio_positions.values()]
        herfindahl = sum(w**2 for w in weights)

        # Normalize to 0-1 scale (1 = maximum concentration)
        n = len(weights)
        max_herfindahl = 1.0
        min_herfindahl = 1.0 / n if n > 0 else 1.0

        if max_herfindahl > min_herfindahl:
            concentration_risk = (herfindahl - min_herfindahl) / (max_herfindahl - min_herfindahl)
        else:
            concentration_risk = 0.0

        return concentration_risk

    def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk"""
        if (self.correlation_matrix is None or
            self.correlation_matrix.empty or
            len(self.correlation_matrix) < 2):
            return 0.0

        # Average absolute correlation
        corr_values = self.correlation_matrix.values
        np.fill_diagonal(corr_values, np.nan)  # Exclude diagonal
        avg_correlation = np.nanmean(np.abs(corr_values))

        return avg_correlation

    def _calculate_liquidity_risk(self, scenario_results: Dict[str, StressResult]) -> float:
        """Calculate liquidity risk from stress scenarios"""
        liquidity_impacts = [
            result.liquidity_impact for result in scenario_results.values()
            if hasattr(result, 'liquidity_impact')
        ]

        if not liquidity_impacts:
            return 0.0

        return max(liquidity_impacts)

    def _create_failed_result(self, scenario: StressScenario, error_msg: str) -> StressResult:
        """Create failed stress result"""
        return StressResult(
            scenario_name=f"FAILED: {scenario.value}",
            scenario_type=scenario,
            portfolio_pnl=0.0,
            portfolio_pnl_pct=0.0,
            worst_position_pnl=0.0,
            best_position_pnl=0.0,
            position_pnls={},
            position_pnl_pcts={},
            stressed_var=0.0,
            stressed_volatility=0.0,
            stressed_correlation=0.0,
            liquidity_impact=0.0,
            shocks_applied=[],
            probability_estimate=0.0,
            historical_precedent=None,
            warnings=[f"Test failed: {error_msg}"]
        )

    def get_stress_report(self) -> Dict[str, Any]:
        """Get comprehensive stress testing report"""
        if not self.last_test_suite:
            return {'error': 'No stress tests have been run'}

        suite = self.last_test_suite

        return {
            'timestamp': suite.test_date.isoformat(),
            'portfolio_value': f"${suite.portfolio_value:,.2f}",
            'scenarios_tested': len(suite.scenarios_tested),
            'scenarios_passed': suite.scenarios_passed,
            'scenarios_failed': suite.scenarios_failed,
            'worst_case': {
                'scenario': suite.worst_case_scenario,
                'loss': f"${suite.worst_case_loss:,.2f}",
                'loss_pct': f"{suite.worst_case_loss/suite.portfolio_value:.1%}"
            },
            'risk_metrics': {
                'stress_var_95': f"${suite.stress_var_95:,.2f}",
                'expected_tail_loss': f"${suite.expected_tail_loss:,.2f}",
                'concentration_risk': f"{suite.concentration_risk:.1%}",
                'correlation_risk': f"{suite.correlation_risk:.1%}",
                'liquidity_risk': f"{suite.liquidity_risk:.1%}"
            },
            'critical_failures': suite.critical_failures,
            'individual_results': {
                name: {
                    'loss': f"${result.portfolio_pnl:,.2f}",
                    'loss_pct': f"{result.portfolio_pnl_pct:.1%}",
                    'probability': f"{result.probability_estimate:.1%}"
                }
                for name, result in suite.scenario_results.items()
            }
        }


if __name__ == "__main__":
    from ..config.risk_config import load_risk_config

    # Test stress tester
    config = load_risk_config()
    stress_tester = StressTester(config)

    # Set up test portfolio
    test_positions = {
        'SPY': 5000,
        'QQQ': 3000,
        'AAPL': 2000
    }

    stress_tester.update_portfolio(test_positions, 10000)

    # Run single scenario test
    crash_result = stress_tester.run_scenario_test(StressScenario.MARKET_CRASH_2008)
    print(f"2008 Crash Test:")
    print(f"Portfolio P&L: ${crash_result.portfolio_pnl:,.2f} ({crash_result.portfolio_pnl_pct:.1%})")
    print(f"Worst Position: ${crash_result.worst_position_pnl:,.2f}")

    # Run full stress suite
    full_suite = stress_tester.run_full_stress_suite()
    print(f"\nFull Stress Suite:")
    print(f"Worst Case Scenario: {full_suite.worst_case_scenario}")
    print(f"Worst Case Loss: ${full_suite.worst_case_loss:,.2f}")
    print(f"Scenarios Passed: {full_suite.scenarios_passed}/{len(full_suite.scenarios_tested)}")
    print(f"Critical Failures: {full_suite.critical_failures}")

    # Run Monte Carlo stress test
    mc_results = stress_tester.run_monte_carlo_stress(num_simulations=1000)
    if 'error' not in mc_results:
        print(f"\nMonte Carlo Stress Test:")
        print(f"VaR 95%: ${mc_results['stress_vars']['var_0.95']:,.2f}")
        print(f"Expected Shortfall: ${mc_results['expected_shortfall']:,.2f}")
        print(f"Prob of 10% Loss: {mc_results['prob_loss_10pct']:.1%}")
        print(f"Prob of 20% Loss: {mc_results['prob_loss_20pct']:.1%}")