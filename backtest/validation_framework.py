"""
Backtest Validation Framework - Prevents Overfitting

Implements research-validated validation methods:
- Combinatorial Purged Cross-Validation (CPCV) - Lopez de Prado
- Deflated Sharpe Ratio (DSR)
- Monte Carlo Permutation Tests
- Walk-Forward Optimization

Key Finding: Quantopian's 888-strategy study showed backtest Sharpe ratios 
had near-zero predictive power for live returns. The more optimized, 
the worse it performed live.

Author: Trading Bot Arsenal
Created: January 2026
Research Base: Advances in Financial Machine Learning (Lopez de Prado)
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BacktestValidation')


# =============================================================================
# COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
# =============================================================================

@dataclass
class CPCVResult:
    """Results from CPCV validation"""
    sharpe_ratios: List[float]
    mean_sharpe: float
    std_sharpe: float
    min_sharpe: float
    max_sharpe: float
    probability_of_loss: float
    probability_of_backtest_overfitting: float
    n_combinations: int
    is_robust: bool  # True if strategy appears robust


def cpcv_split(
    data: np.ndarray,
    n_groups: int = 6,
    test_groups: int = 2,
    embargo_pct: float = 0.01
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate Combinatorial Purged Cross-Validation splits.
    
    CPCV tests multiple historical paths rather than a single train/test split,
    providing a distribution of Sharpe ratios and enabling calculation of
    Probability of Backtest Overfitting (PBO).
    
    Args:
        data: Array of returns or equity curve
        n_groups: Number of groups to split data into (default 6)
        test_groups: Number of groups to use for testing (default 2)
        embargo_pct: Percentage of data to embargo between train/test (default 1%)
        
    Returns:
        List of (train_data, test_data) tuples
    """
    n_samples = len(data)
    group_size = n_samples // n_groups
    embargo_size = int(group_size * embargo_pct)
    
    splits = []
    
    for test_indices in combinations(range(n_groups), test_groups):
        train_indices = [i for i in range(n_groups) if i not in test_indices]
        
        train_data = []
        for i in train_indices:
            start = i * group_size
            end = (i + 1) * group_size
            
            # Apply embargo if adjacent to test set
            if i + 1 in test_indices:
                end -= embargo_size
            if i - 1 in test_indices:
                start += embargo_size
            
            if start < end:
                train_data.append(data[start:end])
        
        test_data = []
        for i in test_indices:
            start = i * group_size
            end = (i + 1) * group_size if i < n_groups - 1 else n_samples
            test_data.append(data[start:end])
        
        if train_data and test_data:
            splits.append((
                np.concatenate(train_data),
                np.concatenate(test_data)
            ))
    
    return splits


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Trading periods per year (252 for daily)
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    
    # Annualize
    return sharpe * np.sqrt(periods_per_year)


def run_cpcv_validation(
    returns: np.ndarray,
    n_groups: int = 6,
    test_groups: int = 2,
    strategy_func: Optional[Callable] = None
) -> CPCVResult:
    """
    Run full CPCV validation on a strategy.
    
    Args:
        returns: Array of strategy returns
        n_groups: Number of groups for CPCV (default 6)
        test_groups: Number of test groups (default 2)
        strategy_func: Optional function to generate returns from data
        
    Returns:
        CPCVResult with validation metrics
    """
    splits = cpcv_split(returns, n_groups, test_groups)
    
    sharpe_ratios = []
    for train_data, test_data in splits:
        if strategy_func:
            # Re-run strategy on test data
            test_returns = strategy_func(test_data)
            sharpe = calculate_sharpe_ratio(test_returns)
        else:
            # Use returns directly
            sharpe = calculate_sharpe_ratio(test_data)
        
        sharpe_ratios.append(sharpe)
    
    sharpe_ratios = np.array(sharpe_ratios)
    
    # Calculate PBO (Probability of Backtest Overfitting)
    # PBO = probability that best in-sample performer is worst out-of-sample
    n_negative = np.sum(sharpe_ratios < 0)
    prob_loss = n_negative / len(sharpe_ratios)
    
    # Simplified PBO estimate based on Sharpe distribution
    # If distribution is centered around 0 or negative, high PBO
    pbo = stats.norm.cdf(0, loc=np.mean(sharpe_ratios), scale=np.std(sharpe_ratios) + 1e-6)
    
    # Strategy is robust if:
    # 1. Mean Sharpe > 0.5
    # 2. Min Sharpe > -0.5
    # 3. PBO < 0.5
    is_robust = (
        np.mean(sharpe_ratios) > 0.5 and
        np.min(sharpe_ratios) > -0.5 and
        pbo < 0.5
    )
    
    return CPCVResult(
        sharpe_ratios=sharpe_ratios.tolist(),
        mean_sharpe=float(np.mean(sharpe_ratios)),
        std_sharpe=float(np.std(sharpe_ratios)),
        min_sharpe=float(np.min(sharpe_ratios)),
        max_sharpe=float(np.max(sharpe_ratios)),
        probability_of_loss=float(prob_loss),
        probability_of_backtest_overfitting=float(pbo),
        n_combinations=len(sharpe_ratios),
        is_robust=is_robust
    )


# =============================================================================
# DEFLATED SHARPE RATIO
# =============================================================================

@dataclass
class DSRResult:
    """Results from Deflated Sharpe Ratio calculation"""
    observed_sharpe: float
    deflated_sharpe: float
    expected_max_sharpe: float
    p_value: float
    is_significant: bool  # True if DSR > 0.95


def calculate_deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    track_record_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float = 0.0
) -> DSRResult:
    """
    Calculate Deflated Sharpe Ratio (DSR) to adjust for multiple testing.
    
    The DSR adjusts the observed Sharpe ratio for:
    - Number of trials/strategies tested (selection bias)
    - Non-normality of returns (skewness/kurtosis)
    - Track record length
    
    Args:
        observed_sharpe: Observed (backtest) Sharpe ratio
        n_trials: Number of strategies/parameters tested
        track_record_length: Number of observations in backtest
        skewness: Skewness of returns (default 0)
        kurtosis: Kurtosis of returns (default 3 = normal)
        benchmark_sharpe: Sharpe ratio of null hypothesis benchmark
        
    Returns:
        DSRResult with deflated metrics
    """
    # Expected maximum Sharpe ratio under null (Bailey & Lopez de Prado)
    # E[max(SR)] ≈ σ(SR) * [(1-γ) * Φ^(-1)(1-1/N) + γ * Φ^(-1)(1-1/(N*e))]
    # where γ ≈ 0.5772 (Euler-Mascheroni constant)
    
    euler_gamma = 0.5772156649
    
    if n_trials <= 1:
        expected_max_sharpe = benchmark_sharpe
    else:
        z1 = stats.norm.ppf(1 - 1/n_trials)
        z2 = stats.norm.ppf(1 - 1/(n_trials * np.e))
        expected_max_sharpe = benchmark_sharpe + (
            (1 - euler_gamma) * z1 + euler_gamma * z2
        ) / np.sqrt(track_record_length)
    
    # Variance of Sharpe ratio (accounting for non-normality)
    # Var(SR) = [1 + 0.5*SR^2 - γ3*SR + ((γ4-3)/4)*SR^2] / T
    sr_var = (
        1 + 0.5 * observed_sharpe**2 
        - skewness * observed_sharpe 
        + ((kurtosis - 3) / 4) * observed_sharpe**2
    ) / track_record_length
    
    # Deflated Sharpe Ratio (probability that observed SR beats expected max)
    if sr_var > 0:
        z_score = (observed_sharpe - expected_max_sharpe) / np.sqrt(sr_var)
        deflated_sharpe = stats.norm.cdf(z_score)
    else:
        deflated_sharpe = 0.5
    
    # P-value: probability of observing this Sharpe by chance
    p_value = 1 - deflated_sharpe
    
    return DSRResult(
        observed_sharpe=observed_sharpe,
        deflated_sharpe=deflated_sharpe,
        expected_max_sharpe=expected_max_sharpe,
        p_value=p_value,
        is_significant=deflated_sharpe > 0.95
    )


# =============================================================================
# MONTE CARLO PERMUTATION TEST
# =============================================================================

@dataclass
class MonteCarloResult:
    """Results from Monte Carlo permutation test"""
    observed_metric: float
    permutation_mean: float
    permutation_std: float
    p_value: float
    percentile: float
    n_permutations: int
    is_significant: bool


def monte_carlo_permutation_test(
    returns: np.ndarray,
    n_permutations: int = 1000,
    metric_func: Optional[Callable] = None,
    random_seed: int = 42
) -> MonteCarloResult:
    """
    Run Monte Carlo permutation test to assess statistical significance.
    
    Generates random permutations of returns to create a null distribution,
    then calculates the probability of observing the actual result by chance.
    
    Args:
        returns: Array of strategy returns
        n_permutations: Number of random permutations (default 1000)
        metric_func: Function to calculate metric (default: Sharpe ratio)
        random_seed: Random seed for reproducibility
        
    Returns:
        MonteCarloResult with significance metrics
    """
    np.random.seed(random_seed)
    
    if metric_func is None:
        metric_func = calculate_sharpe_ratio
    
    # Calculate observed metric
    observed = metric_func(returns)
    
    # Generate permutation distribution
    permutation_metrics = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(returns)
        permutation_metrics.append(metric_func(shuffled))
    
    permutation_metrics = np.array(permutation_metrics)
    
    # Calculate p-value (proportion of permutations >= observed)
    p_value = np.mean(permutation_metrics >= observed)
    
    # Calculate percentile of observed value
    percentile = stats.percentileofscore(permutation_metrics, observed)
    
    return MonteCarloResult(
        observed_metric=float(observed),
        permutation_mean=float(np.mean(permutation_metrics)),
        permutation_std=float(np.std(permutation_metrics)),
        p_value=float(p_value),
        percentile=float(percentile),
        n_permutations=n_permutations,
        is_significant=p_value < 0.05
    )


# =============================================================================
# WALK-FORWARD OPTIMIZATION
# =============================================================================

@dataclass
class WalkForwardResult:
    """Results from walk-forward optimization"""
    in_sample_sharpes: List[float]
    out_of_sample_sharpes: List[float]
    mean_is_sharpe: float
    mean_oos_sharpe: float
    degradation: float  # OOS/IS ratio
    is_consistent: bool  # True if degradation < 50%


def walk_forward_validation(
    returns: np.ndarray,
    n_windows: int = 5,
    train_pct: float = 0.7,
    strategy_func: Optional[Callable] = None
) -> WalkForwardResult:
    """
    Run walk-forward optimization validation.
    
    Divides data into rolling windows, trains on first part, tests on second.
    Measures how much performance degrades from in-sample to out-of-sample.
    
    Args:
        returns: Array of strategy returns
        n_windows: Number of rolling windows (default 5)
        train_pct: Percentage of each window for training (default 70%)
        strategy_func: Optional function to optimize/generate returns
        
    Returns:
        WalkForwardResult with degradation metrics
    """
    window_size = len(returns) // n_windows
    
    in_sample_sharpes = []
    out_of_sample_sharpes = []
    
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        
        if end > len(returns):
            break
        
        window_data = returns[start:end]
        train_size = int(len(window_data) * train_pct)
        
        train_data = window_data[:train_size]
        test_data = window_data[train_size:]
        
        if strategy_func:
            train_returns = strategy_func(train_data)
            test_returns = strategy_func(test_data)
        else:
            train_returns = train_data
            test_returns = test_data
        
        is_sharpe = calculate_sharpe_ratio(train_returns)
        oos_sharpe = calculate_sharpe_ratio(test_returns)
        
        in_sample_sharpes.append(is_sharpe)
        out_of_sample_sharpes.append(oos_sharpe)
    
    mean_is = np.mean(in_sample_sharpes)
    mean_oos = np.mean(out_of_sample_sharpes)
    
    # Degradation: how much worse is OOS vs IS
    if mean_is != 0:
        degradation = mean_oos / mean_is
    else:
        degradation = 0.0
    
    # Strategy is consistent if degradation < 50%
    # Research shows 30-50% degradation is normal
    is_consistent = degradation > 0.5 or (mean_is < 0.5 and mean_oos > 0)
    
    return WalkForwardResult(
        in_sample_sharpes=in_sample_sharpes,
        out_of_sample_sharpes=out_of_sample_sharpes,
        mean_is_sharpe=mean_is,
        mean_oos_sharpe=mean_oos,
        degradation=degradation,
        is_consistent=is_consistent
    )


# =============================================================================
# OVERFITTING DETECTION
# =============================================================================

def calculate_parameter_count_ratio(
    n_parameters: int,
    n_observations: int
) -> Dict:
    """
    Check if strategy has too many parameters for the data.
    
    Lopez de Prado's rule: N_parameters << sqrt(N_observations)
    
    With 2 years of daily data (~500 observations), keep to 3-5 parameters max.
    
    Args:
        n_parameters: Number of strategy parameters
        n_observations: Number of data points in backtest
        
    Returns:
        Dict with assessment
    """
    sqrt_n = np.sqrt(n_observations)
    ratio = n_parameters / sqrt_n
    
    # Classifications
    if ratio < 0.2:
        risk_level = "low"
        recommendation = "Parameter count is appropriate"
    elif ratio < 0.5:
        risk_level = "moderate"
        recommendation = "Consider reducing parameters"
    else:
        risk_level = "high"
        recommendation = "Strategy likely overfit - reduce parameters significantly"
    
    return {
        'n_parameters': n_parameters,
        'n_observations': n_observations,
        'max_recommended': int(sqrt_n * 0.2),
        'ratio': ratio,
        'risk_level': risk_level,
        'recommendation': recommendation
    }


def detect_overfitting_signs(
    returns: np.ndarray,
    in_sample_sharpe: float,
    out_of_sample_sharpe: float,
    n_parameters: int
) -> Dict:
    """
    Comprehensive overfitting detection.
    
    Signs of overfitting:
    1. Smooth equity curve (real strategies have drawdowns)
    2. Too many parameters
    3. High IS but low OOS performance
    4. Strategy fails on similar instruments
    5. Small parameter changes cause large performance swings
    
    Args:
        returns: Strategy returns
        in_sample_sharpe: In-sample Sharpe ratio
        out_of_sample_sharpe: Out-of-sample Sharpe ratio
        n_parameters: Number of parameters
        
    Returns:
        Dict with overfitting assessment
    """
    signs = []
    risk_score = 0
    
    # Check 1: Performance degradation
    if in_sample_sharpe > 0:
        degradation = 1 - (out_of_sample_sharpe / in_sample_sharpe)
        if degradation > 0.5:
            signs.append(f"Severe IS→OOS degradation: {degradation:.0%}")
            risk_score += 3
        elif degradation > 0.3:
            signs.append(f"Moderate IS→OOS degradation: {degradation:.0%}")
            risk_score += 2
    
    # Check 2: Unrealistic Sharpe
    if in_sample_sharpe > 2.5:
        signs.append(f"Unrealistically high IS Sharpe: {in_sample_sharpe:.2f}")
        risk_score += 2
    
    # Check 3: Parameter count
    param_check = calculate_parameter_count_ratio(n_parameters, len(returns))
    if param_check['risk_level'] == 'high':
        signs.append(f"Too many parameters: {n_parameters} (max {param_check['max_recommended']})")
        risk_score += 3
    elif param_check['risk_level'] == 'moderate':
        risk_score += 1
    
    # Check 4: Equity curve smoothness (lack of drawdowns)
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_drawdown = abs(np.min(drawdowns))
    
    if max_drawdown < 0.02 and in_sample_sharpe > 1:
        signs.append(f"Suspiciously low max drawdown: {max_drawdown:.2%}")
        risk_score += 2
    
    # Check 5: Win rate (unrealistic if very high with high Sharpe)
    win_rate = np.mean(returns > 0)
    if win_rate > 0.85 and in_sample_sharpe > 1.5:
        signs.append(f"Unrealistically high win rate: {win_rate:.0%}")
        risk_score += 2
    
    # Overall assessment
    if risk_score >= 6:
        assessment = "HIGH OVERFITTING RISK"
        action = "Do NOT deploy - strategy likely overfit"
    elif risk_score >= 3:
        assessment = "MODERATE OVERFITTING RISK"
        action = "Run additional validation before deployment"
    else:
        assessment = "LOW OVERFITTING RISK"
        action = "Strategy appears robust for paper trading"
    
    return {
        'signs': signs,
        'risk_score': risk_score,
        'max_risk_score': 12,
        'assessment': assessment,
        'action': action,
        'metrics': {
            'in_sample_sharpe': in_sample_sharpe,
            'out_of_sample_sharpe': out_of_sample_sharpe,
            'degradation': 1 - (out_of_sample_sharpe / in_sample_sharpe) if in_sample_sharpe > 0 else 0,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_parameters': n_parameters
        }
    }


# =============================================================================
# COMPLETE VALIDATION REPORT
# =============================================================================

def generate_validation_report(
    returns: np.ndarray,
    strategy_name: str,
    n_parameters: int = 5,
    n_trials: int = 1,
    verbose: bool = True
) -> Dict:
    """
    Generate comprehensive validation report for a strategy.
    
    Runs all validation methods and provides overall assessment.
    
    Args:
        returns: Array of strategy returns
        strategy_name: Name of the strategy
        n_parameters: Number of strategy parameters
        n_trials: Number of strategies/variations tested
        verbose: Print results to console
        
    Returns:
        Dict with complete validation results
    """
    if verbose:
        print("=" * 60)
        print(f"VALIDATION REPORT: {strategy_name}")
        print("=" * 60)
    
    results = {
        'strategy_name': strategy_name,
        'timestamp': datetime.now().isoformat(),
        'n_observations': len(returns)
    }
    
    # 1. Basic metrics
    basic_sharpe = calculate_sharpe_ratio(returns)
    total_return = np.sum(returns)
    max_dd = abs(np.min(np.cumsum(returns) - np.maximum.accumulate(np.cumsum(returns))))
    win_rate = np.mean(returns > 0)
    
    results['basic_metrics'] = {
        'sharpe_ratio': basic_sharpe,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'win_rate': win_rate
    }
    
    if verbose:
        print(f"\n📊 BASIC METRICS:")
        print(f"   Sharpe Ratio: {basic_sharpe:.2f}")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Max Drawdown: {max_dd:.2%}")
        print(f"   Win Rate: {win_rate:.1%}")
    
    # 2. CPCV Validation
    if len(returns) >= 100:
        cpcv_result = run_cpcv_validation(returns)
        results['cpcv'] = {
            'mean_sharpe': cpcv_result.mean_sharpe,
            'std_sharpe': cpcv_result.std_sharpe,
            'min_sharpe': cpcv_result.min_sharpe,
            'pbo': cpcv_result.probability_of_backtest_overfitting,
            'is_robust': cpcv_result.is_robust
        }
        
        if verbose:
            print(f"\n🔬 CPCV VALIDATION ({cpcv_result.n_combinations} combinations):")
            print(f"   Mean Sharpe: {cpcv_result.mean_sharpe:.2f} ± {cpcv_result.std_sharpe:.2f}")
            print(f"   Min/Max: {cpcv_result.min_sharpe:.2f} / {cpcv_result.max_sharpe:.2f}")
            print(f"   PBO: {cpcv_result.probability_of_backtest_overfitting:.1%}")
            print(f"   Robust: {'✅ YES' if cpcv_result.is_robust else '❌ NO'}")
    
    # 3. Deflated Sharpe Ratio
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns) + 3  # Convert to excess kurtosis
    
    dsr_result = calculate_deflated_sharpe_ratio(
        observed_sharpe=basic_sharpe,
        n_trials=n_trials,
        track_record_length=len(returns),
        skewness=skewness,
        kurtosis=kurtosis
    )
    
    results['deflated_sharpe'] = {
        'observed': dsr_result.observed_sharpe,
        'deflated': dsr_result.deflated_sharpe,
        'expected_max': dsr_result.expected_max_sharpe,
        'p_value': dsr_result.p_value,
        'is_significant': dsr_result.is_significant
    }
    
    if verbose:
        print(f"\n📉 DEFLATED SHARPE RATIO:")
        print(f"   Observed Sharpe: {dsr_result.observed_sharpe:.2f}")
        print(f"   Expected Max (null): {dsr_result.expected_max_sharpe:.2f}")
        print(f"   Deflated SR: {dsr_result.deflated_sharpe:.2f}")
        print(f"   Significant: {'✅ YES' if dsr_result.is_significant else '❌ NO'}")
    
    # 4. Monte Carlo Test
    mc_result = monte_carlo_permutation_test(returns)
    results['monte_carlo'] = {
        'observed': mc_result.observed_metric,
        'null_mean': mc_result.permutation_mean,
        'p_value': mc_result.p_value,
        'percentile': mc_result.percentile,
        'is_significant': mc_result.is_significant
    }
    
    if verbose:
        print(f"\n🎲 MONTE CARLO TEST ({mc_result.n_permutations} permutations):")
        print(f"   Observed Sharpe: {mc_result.observed_metric:.2f}")
        print(f"   Null Distribution Mean: {mc_result.permutation_mean:.2f}")
        print(f"   P-value: {mc_result.p_value:.3f}")
        print(f"   Percentile: {mc_result.percentile:.1f}%")
        print(f"   Significant: {'✅ YES' if mc_result.is_significant else '❌ NO'}")
    
    # 5. Walk-Forward Validation
    if len(returns) >= 200:
        wf_result = walk_forward_validation(returns)
        results['walk_forward'] = {
            'is_sharpe': wf_result.mean_is_sharpe,
            'oos_sharpe': wf_result.mean_oos_sharpe,
            'degradation': wf_result.degradation,
            'is_consistent': wf_result.is_consistent
        }
        
        if verbose:
            print(f"\n🚶 WALK-FORWARD VALIDATION ({len(wf_result.in_sample_sharpes)} windows):")
            print(f"   In-Sample Sharpe: {wf_result.mean_is_sharpe:.2f}")
            print(f"   Out-of-Sample Sharpe: {wf_result.mean_oos_sharpe:.2f}")
            print(f"   Degradation: {(1-wf_result.degradation)*100:.0f}%")
            print(f"   Consistent: {'✅ YES' if wf_result.is_consistent else '❌ NO'}")
    
    # 6. Overfitting Detection
    oos_sharpe = results.get('walk_forward', {}).get('oos_sharpe', basic_sharpe * 0.6)
    overfit_check = detect_overfitting_signs(returns, basic_sharpe, oos_sharpe, n_parameters)
    results['overfitting'] = overfit_check
    
    if verbose:
        print(f"\n⚠️ OVERFITTING CHECK:")
        print(f"   Risk Score: {overfit_check['risk_score']}/{overfit_check['max_risk_score']}")
        print(f"   Assessment: {overfit_check['assessment']}")
        if overfit_check['signs']:
            print(f"   Warning Signs:")
            for sign in overfit_check['signs']:
                print(f"      - {sign}")
        print(f"   Action: {overfit_check['action']}")
    
    # 7. Overall Verdict
    passes = 0
    total_checks = 0
    
    if 'cpcv' in results:
        total_checks += 1
        if results['cpcv']['is_robust']:
            passes += 1
    
    if results['deflated_sharpe']['is_significant']:
        passes += 1
    total_checks += 1
    
    if results['monte_carlo']['is_significant']:
        passes += 1
    total_checks += 1
    
    if 'walk_forward' in results:
        total_checks += 1
        if results['walk_forward']['is_consistent']:
            passes += 1
    
    if overfit_check['risk_score'] < 3:
        passes += 1
    total_checks += 1
    
    results['verdict'] = {
        'checks_passed': passes,
        'total_checks': total_checks,
        'pass_rate': passes / total_checks,
        'recommendation': 'DEPLOY' if passes >= total_checks * 0.6 else 'REVIEW' if passes >= total_checks * 0.4 else 'REJECT'
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"📋 FINAL VERDICT: {results['verdict']['recommendation']}")
        print(f"   Checks Passed: {passes}/{total_checks} ({results['verdict']['pass_rate']:.0%})")
        print("=" * 60)
    
    return results


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BACKTEST VALIDATION FRAMEWORK TEST")
    print("=" * 60)
    
    # Generate sample returns (simulating a strategy)
    np.random.seed(42)
    
    # Simulate a moderately profitable strategy
    n_days = 500
    daily_return_mean = 0.0005  # 0.05% daily = ~12.5% annually
    daily_return_std = 0.01    # 1% daily volatility
    
    returns = np.random.normal(daily_return_mean, daily_return_std, n_days)
    
    # Add some structure (autocorrelation) to make it more realistic
    for i in range(1, len(returns)):
        returns[i] += returns[i-1] * 0.05
    
    # Run validation
    report = generate_validation_report(
        returns=returns,
        strategy_name="RSI-2 Mean Reversion",
        n_parameters=5,
        n_trials=10,  # Assume we tested 10 parameter combinations
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Test complete!")
