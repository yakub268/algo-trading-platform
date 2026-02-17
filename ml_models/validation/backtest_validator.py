"""
Backtest Validator
=================

Backtesting framework for validating ML model performance in trading scenarios.
Provides realistic trading simulations with transaction costs, slippage, and market impact.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..predictors.base_predictor import BasePredictor
from .performance_metrics import PerformanceAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    max_position_size: float = 0.1  # 10% of portfolio
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    risk_free_rate: float = 0.02  # 2% annual
    benchmark_symbol: str = "SPY"


@dataclass
class Trade:
    """Individual trade record"""
    timestamp: datetime
    symbol: str
    action: str  # buy, sell, hold
    quantity: float
    price: float
    value: float
    transaction_cost: float
    portfolio_value: float
    model_prediction: Optional[Dict] = None


@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    trades: List[Trade]
    portfolio_values: pd.Series
    daily_returns: pd.Series
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float


class BacktestValidator:
    """
    Comprehensive backtesting framework for ML trading models.

    Features:
    - Realistic trading simulation
    - Transaction costs and slippage
    - Risk management constraints
    - Multiple performance metrics
    - Benchmark comparison
    - Walk-forward validation
    - Monte Carlo analysis
    """

    def __init__(self, config: BacktestConfig = None):
        """Initialize backtest validator"""
        self.config = config or BacktestConfig()
        self.performance_analyzer = PerformanceAnalyzer()

        logger.info("BacktestValidator initialized")

    def run_backtest(self,
                    model: BasePredictor,
                    data: pd.DataFrame,
                    symbol: str = "SPY",
                    start_date: datetime = None,
                    end_date: datetime = None) -> BacktestResults:
        """
        Run comprehensive backtest of ML model.

        Args:
            model: Trained ML model
            data: Historical price data
            symbol: Symbol being traded
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            Detailed backtest results
        """
        logger.info(f"Starting backtest for {symbol}")

        # Prepare data
        backtest_data = self._prepare_backtest_data(data, start_date, end_date)
        if backtest_data.empty:
            raise ValueError("No data available for backtesting")

        # Initialize portfolio
        portfolio_value = self.config.initial_capital
        cash = self.config.initial_capital
        positions = {}
        trades = []
        portfolio_values = []
        daily_returns = []

        # Track performance
        previous_portfolio_value = portfolio_value

        logger.info(f"Backtesting period: {backtest_data.index[0]} to {backtest_data.index[-1]}")
        logger.info(f"Backtest samples: {len(backtest_data)}")

        for date, row in backtest_data.iterrows():
            try:
                # Get current market data up to this date
                current_data = backtest_data.loc[:date]

                # Generate predictions
                predictions = self._get_model_predictions(model, current_data)

                # Make trading decisions
                trade_signals = self._generate_trade_signals(predictions, row)

                # Execute trades
                for signal in trade_signals:
                    trade = self._execute_trade(
                        signal, row, cash, positions, portfolio_value, date
                    )
                    if trade:
                        trades.append(trade)
                        cash -= trade.value + trade.transaction_cost

                # Update portfolio value
                position_value = sum(
                    qty * row['Close'] for qty in positions.values()
                )
                portfolio_value = cash + position_value
                portfolio_values.append(portfolio_value)

                # Calculate daily return
                daily_return = (portfolio_value / previous_portfolio_value) - 1
                daily_returns.append(daily_return)
                previous_portfolio_value = portfolio_value

            except Exception as e:
                logger.warning(f"Backtest error at {date}: {e}")
                # Use previous values
                portfolio_values.append(previous_portfolio_value)
                daily_returns.append(0.0)

        # Create results
        results = self._compile_results(
            backtest_data, symbol, trades, portfolio_values, daily_returns
        )

        logger.info(f"Backtest complete - Total Return: {results.total_return:.2%}")
        return results

    def run_walk_forward_analysis(self,
                                model_class: type,
                                data: pd.DataFrame,
                                symbol: str = "SPY",
                                train_window: int = 252,
                                test_window: int = 63,
                                step_size: int = 21) -> Dict[str, Any]:
        """
        Run walk-forward analysis to validate model robustness.

        Args:
            model_class: ML model class to test
            data: Historical data
            symbol: Symbol being tested
            train_window: Training window in days
            test_window: Testing window in days
            step_size: Step size for walking forward

        Returns:
            Walk-forward analysis results
        """
        logger.info(f"Starting walk-forward analysis for {symbol}")

        all_results = []
        total_data_length = len(data)

        # Walk forward through the data
        start_idx = train_window
        while start_idx + test_window < total_data_length:
            try:
                # Define windows
                train_start = start_idx - train_window
                train_end = start_idx
                test_start = start_idx
                test_end = min(start_idx + test_window, total_data_length)

                # Extract data
                train_data = data.iloc[train_start:train_end]
                test_data = data.iloc[test_start:test_end]

                logger.debug(f"Window {len(all_results)+1}: Train {train_data.index[0]} to {train_data.index[-1]}")

                # Train model
                model = self._train_model_for_period(model_class, train_data, symbol)
                if model is None:
                    logger.warning(f"Model training failed for window {len(all_results)+1}")
                    start_idx += step_size
                    continue

                # Run backtest on test period
                test_results = self.run_backtest(
                    model, test_data, symbol,
                    start_date=test_data.index[0],
                    end_date=test_data.index[-1]
                )

                # Store results
                window_result = {
                    'window': len(all_results) + 1,
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'total_return': test_results.total_return,
                    'sharpe_ratio': test_results.sharpe_ratio,
                    'max_drawdown': test_results.max_drawdown,
                    'win_rate': test_results.win_rate,
                    'total_trades': test_results.total_trades
                }

                all_results.append(window_result)

            except Exception as e:
                logger.warning(f"Walk-forward window failed: {e}")

            start_idx += step_size

        # Aggregate results
        if not all_results:
            return {'success': False, 'error': 'No successful windows'}

        aggregated = self._aggregate_walk_forward_results(all_results)
        logger.info(f"Walk-forward analysis complete: {len(all_results)} windows")

        return {
            'success': True,
            'windows': len(all_results),
            'individual_results': all_results,
            'aggregated_metrics': aggregated
        }

    def monte_carlo_validation(self,
                             model: BasePredictor,
                             data: pd.DataFrame,
                             symbol: str = "SPY",
                             num_simulations: int = 100,
                             bootstrap_length: int = 252) -> Dict[str, Any]:
        """
        Run Monte Carlo validation using bootstrap sampling.

        Args:
            model: Trained model
            data: Historical data
            symbol: Symbol
            num_simulations: Number of Monte Carlo runs
            bootstrap_length: Length of each bootstrap sample

        Returns:
            Monte Carlo validation results
        """
        logger.info(f"Starting Monte Carlo validation: {num_simulations} simulations")

        simulation_results = []

        for sim in range(num_simulations):
            try:
                # Bootstrap sample
                sample_data = self._bootstrap_sample(data, bootstrap_length)

                # Run backtest
                results = self.run_backtest(model, sample_data, symbol)

                simulation_results.append({
                    'simulation': sim + 1,
                    'total_return': results.total_return,
                    'sharpe_ratio': results.sharpe_ratio,
                    'max_drawdown': results.max_drawdown,
                    'win_rate': results.win_rate
                })

            except Exception as e:
                logger.warning(f"Simulation {sim+1} failed: {e}")

        if not simulation_results:
            return {'success': False, 'error': 'All simulations failed'}

        # Calculate statistics
        returns = [r['total_return'] for r in simulation_results]
        sharpes = [r['sharpe_ratio'] for r in simulation_results if not np.isnan(r['sharpe_ratio'])]
        drawdowns = [r['max_drawdown'] for r in simulation_results]

        monte_carlo_stats = {
            'simulations': len(simulation_results),
            'returns': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentile_5': np.percentile(returns, 5),
                'percentile_95': np.percentile(returns, 95)
            },
            'sharpe_ratios': {
                'mean': np.mean(sharpes) if sharpes else 0,
                'std': np.std(sharpes) if sharpes else 0
            },
            'max_drawdowns': {
                'mean': np.mean(drawdowns),
                'worst': np.min(drawdowns)  # Most negative
            },
            'win_probability': np.mean([1 if r > 0 else 0 for r in returns])
        }

        logger.info(f"Monte Carlo complete - Mean Return: {monte_carlo_stats['returns']['mean']:.2%}")

        return {
            'success': True,
            'statistics': monte_carlo_stats,
            'individual_results': simulation_results
        }

    def _prepare_backtest_data(self,
                             data: pd.DataFrame,
                             start_date: datetime = None,
                             end_date: datetime = None) -> pd.DataFrame:
        """Prepare data for backtesting"""
        # Filter by date range if specified
        if start_date or end_date:
            mask = pd.Series(True, index=data.index)
            if start_date:
                mask &= data.index >= start_date
            if end_date:
                mask &= data.index <= end_date
            data = data[mask]

        # Ensure we have required columns
        required_columns = ['Close']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Data missing required columns: {required_columns}")
            return pd.DataFrame()

        # Add derived columns if needed
        if 'Open' not in data.columns:
            data['Open'] = data['Close'].shift(1)  # Approximate
        if 'High' not in data.columns:
            data['High'] = data['Close']  # Approximate
        if 'Low' not in data.columns:
            data['Low'] = data['Close']  # Approximate

        return data.dropna()

    def _get_model_predictions(self, model: BasePredictor, data: pd.DataFrame) -> Optional[Dict]:
        """Get predictions from model"""
        try:
            if len(data) < 50:  # Need minimum data for features
                return None

            # Use last 50 rows for prediction to ensure we have enough data
            recent_data = data.tail(50)

            predictions = model.predict(recent_data)

            # Convert to dict format
            if hasattr(predictions, 'predictions'):
                return {
                    'predictions': predictions.predictions,
                    'confidence': getattr(predictions, 'confidence', None),
                    'probabilities': getattr(predictions, 'probabilities', None)
                }
            else:
                return {'predictions': predictions}

        except Exception as e:
            logger.debug(f"Prediction generation failed: {e}")
            return None

    def _generate_trade_signals(self, predictions: Optional[Dict], row: pd.Series) -> List[Dict]:
        """Generate trade signals from predictions"""
        if predictions is None or 'predictions' not in predictions:
            return []

        signals = []
        pred_array = predictions['predictions']

        if len(pred_array) == 0:
            return signals

        # Use last prediction
        latest_pred = pred_array[-1]
        confidence = None

        if predictions.get('confidence') is not None:
            conf_array = predictions['confidence']
            confidence = conf_array[-1] if len(conf_array) > 0 else 0.5

        # Convert prediction to signal
        if isinstance(latest_pred, (int, float)):
            if latest_pred > 0:  # Positive prediction (buy signal)
                action = 'buy'
                strength = min(abs(latest_pred), 1.0)
            elif latest_pred < 0:  # Negative prediction (sell signal)
                action = 'sell'
                strength = min(abs(latest_pred), 1.0)
            else:  # Neutral
                return signals
        else:
            # Handle classification predictions
            return signals

        # Calculate position size based on confidence and strength
        base_size = self.config.max_position_size
        if confidence is not None:
            position_size = base_size * strength * confidence
        else:
            position_size = base_size * strength * 0.5  # Default confidence

        signals.append({
            'action': action,
            'size': position_size,
            'confidence': confidence or 0.5,
            'strength': strength,
            'price': row['Close']
        })

        return signals

    def _execute_trade(self,
                     signal: Dict,
                     row: pd.Series,
                     cash: float,
                     positions: Dict,
                     portfolio_value: float,
                     date: datetime) -> Optional[Trade]:
        """Execute a trade based on signal"""
        try:
            action = signal['action']
            size = signal['size']
            price = signal['price']

            # Apply slippage
            if action == 'buy':
                execution_price = price * (1 + self.config.slippage)
            else:
                execution_price = price * (1 - self.config.slippage)

            # Calculate trade value
            trade_value = portfolio_value * size
            quantity = trade_value / execution_price

            # Check if we have enough cash for buy orders
            if action == 'buy':
                total_cost = trade_value + (trade_value * self.config.transaction_cost)
                if total_cost > cash:
                    return None  # Not enough cash

            # Calculate transaction cost
            transaction_cost = trade_value * self.config.transaction_cost

            # Update positions
            symbol = 'POSITION'  # Generic position for single-asset backtest
            if action == 'buy':
                positions[symbol] = positions.get(symbol, 0) + quantity
            else:  # sell
                current_pos = positions.get(symbol, 0)
                positions[symbol] = max(0, current_pos - quantity)

            return Trade(
                timestamp=date,
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=execution_price,
                value=trade_value,
                transaction_cost=transaction_cost,
                portfolio_value=portfolio_value,
                model_prediction=signal
            )

        except Exception as e:
            logger.debug(f"Trade execution failed: {e}")
            return None

    def _compile_results(self,
                       data: pd.DataFrame,
                       symbol: str,
                       trades: List[Trade],
                       portfolio_values: List[float],
                       daily_returns: List[float]) -> BacktestResults:
        """Compile comprehensive backtest results"""

        # Convert to pandas series
        portfolio_series = pd.Series(portfolio_values, index=data.index)
        returns_series = pd.Series(daily_returns, index=data.index)

        # Calculate performance metrics
        initial_capital = self.config.initial_capital
        final_capital = portfolio_values[-1] if portfolio_values else initial_capital
        total_return = (final_capital / initial_capital) - 1

        # Annualized return
        days = len(data)
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        # Volatility
        volatility = float(returns_series.std() * np.sqrt(252)) if len(returns_series) > 1 else 0

        # Sharpe ratio
        excess_return = annual_return - self.config.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Maximum drawdown
        running_max = portfolio_series.cummax()
        drawdown = (portfolio_series - running_max) / running_max
        max_drawdown = float(drawdown.min())

        # Trade statistics
        winning_trades = [t for t in trades if t.action == 'sell' and t.value > 0]  # Simplified
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        # Profit factor (simplified)
        profit_factor = 1.0  # Placeholder

        # Benchmark comparison (if available)
        benchmark_return = self._calculate_benchmark_return(data)
        alpha = annual_return - benchmark_return
        beta = 1.0  # Placeholder
        information_ratio = alpha / volatility if volatility > 0 else 0

        return BacktestResults(
            config=self.config,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            trades=trades,
            portfolio_values=portfolio_series,
            daily_returns=returns_series,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio
        )

    def _calculate_benchmark_return(self, data: pd.DataFrame) -> float:
        """Calculate benchmark return"""
        try:
            if len(data) < 2:
                return 0.0

            # Simple buy-and-hold return
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            return (end_price / start_price) - 1

        except Exception as e:
            logger.debug(f"Benchmark calculation failed: {e}")
            return 0.0

    def _train_model_for_period(self, model_class: type, train_data: pd.DataFrame, symbol: str) -> Optional[BasePredictor]:
        """Train model for specific period in walk-forward analysis"""
        try:
            # Simple model training - in practice would use full training pipeline
            model = model_class()

            # This is a simplified training - real implementation would use
            # the full feature engineering and training pipeline
            logger.debug(f"Training {model_class.__name__} on {len(train_data)} samples")

            # Return untrained model for now (placeholder)
            return model

        except Exception as e:
            logger.debug(f"Model training failed: {e}")
            return None

    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate walk-forward analysis results"""
        if not results:
            return {}

        returns = [r['total_return'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results if not np.isnan(r['sharpe_ratio'])]
        drawdowns = [r['max_drawdown'] for r in results]
        win_rates = [r['win_rate'] for r in results]

        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_sharpe': np.mean(sharpes) if sharpes else 0,
            'mean_drawdown': np.mean(drawdowns),
            'worst_drawdown': np.min(drawdowns),
            'mean_win_rate': np.mean(win_rates),
            'consistency': len([r for r in returns if r > 0]) / len(returns)
        }

    def _bootstrap_sample(self, data: pd.DataFrame, length: int) -> pd.DataFrame:
        """Create bootstrap sample of data"""
        if len(data) < length:
            return data

        # Random sampling with replacement
        sample_indices = np.random.choice(len(data), size=length, replace=True)
        sample_data = data.iloc[sample_indices].copy()

        # Reset index to maintain time series structure
        sample_data.index = pd.date_range(
            start=data.index[0],
            periods=length,
            freq='D'
        )

        return sample_data


def main():
    """Test backtest validator"""
    print("Testing Backtest Validator")
    print("=" * 35)

    # Create simple mock model
    class MockModel(BasePredictor):
        def __init__(self):
            super().__init__("MockModel", "classifier")
            self.is_fitted = True

        def _build_model(self, **kwargs):
            return None

        def _prepare_data(self, X, y=None):
            return X, y

        def _fit_model(self, X, y, **kwargs):
            pass

        def _predict_model(self, X):
            # Simple momentum-based predictions
            if hasattr(X, 'Close'):
                returns = float(X['Close'].pct_change().tail(5).mean())
                return np.array([1 if returns > 0 else -1])
            else:
                return np.array([1])

    # Generate test data
    import yfinance as yf

    print("Downloading test data...")
    data = yf.download("SPY", period="1y", progress=False)

    if len(data) == 0:
        print("No data available for testing")
        return

    # Initialize validator
    config = BacktestConfig(
        initial_capital=100000,
        transaction_cost=0.001,
        max_position_size=0.1
    )
    validator = BacktestValidator(config)

    # Create mock model
    model = MockModel()

    # Run backtest
    print("Running backtest...")
    results = validator.run_backtest(model, data, "SPY")

    print(f"\nBacktest Results:")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Annual Return: {results.annual_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.1%}")
    print(f"Total Trades: {results.total_trades}")

    # Test Monte Carlo validation
    print("\nRunning Monte Carlo validation...")
    mc_results = validator.monte_carlo_validation(model, data, "SPY", num_simulations=10)

    if mc_results['success']:
        stats = mc_results['statistics']
        print(f"Monte Carlo Results ({stats['simulations']} simulations):")
        print(f"Mean Return: {stats['returns']['mean']:.2%}")
        print(f"Return Std: {stats['returns']['std']:.2%}")
        print(f"Win Probability: {stats['win_probability']:.1%}")

    print("\nBacktest validator test completed!")


if __name__ == "__main__":
    main()