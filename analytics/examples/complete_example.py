"""
Complete Analytics System Example
================================

Comprehensive example demonstrating all features of the trading analytics system
including real-time monitoring, performance analysis, and reporting.

This example shows how to:
1. Set up the complete analytics system
2. Simulate trading activity
3. Monitor performance in real-time
4. Generate comprehensive reports
5. Use the REST API
6. Handle alerts and notifications

Author: Trading Bot System
Created: February 2026
"""

import asyncio
import time
import threading
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List

# Import analytics components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from analytics_integration import create_analytics_integration
from reporting.performance_reporter import ReportConfig


class TradingSimulator:
    """Simulate realistic trading activity for demonstration"""

    def __init__(self, analytics_integration):
        self.analytics = analytics_integration
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'BTC/USD', 'ETH/USD']
        self.strategies = ['momentum', 'mean_reversion', 'arbitrage', 'crypto_momentum']
        self.platforms = ['alpaca', 'binance', 'kalshi']
        self.trade_id_counter = 0
        self.open_positions = {}

    def generate_trade(self) -> Dict:
        """Generate a realistic trade"""
        self.trade_id_counter += 1
        symbol = np.random.choice(self.symbols)
        strategy = np.random.choice(self.strategies)
        platform = np.random.choice(self.platforms)

        # Determine action based on current positions
        if symbol in self.open_positions:
            # Close existing position
            action = 'sell' if self.open_positions[symbol]['side'] == 'buy' else 'buy'
        else:
            # Open new position
            action = np.random.choice(['buy', 'sell'])

        # Generate realistic price
        base_price = {
            'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500, 'TSLA': 200, 'NVDA': 400,
            'BTC/USD': 45000, 'ETH/USD': 3000
        }.get(symbol, 100)

        price = base_price * (1 + np.random.normal(0, 0.02))  # 2% volatility
        quantity = np.random.randint(10, 200)

        trade = {
            'trade_id': f'trade_{self.trade_id_counter:05d}',
            'symbol': symbol,
            'action': action,
            'price': round(price, 2),
            'quantity': quantity,
            'strategy': strategy,
            'platform': platform,
            'timestamp': datetime.now()
        }

        # Update positions tracking
        if action == 'buy' and symbol not in self.open_positions:
            self.open_positions[symbol] = {'side': 'buy', 'price': price, 'quantity': quantity}
        elif action == 'sell' and symbol in self.open_positions:
            del self.open_positions[symbol]

        return trade

    def generate_price_update(self) -> Dict:
        """Generate realistic price update for existing positions"""
        if not self.open_positions:
            return None

        symbol = np.random.choice(list(self.open_positions.keys()))
        current_price = self.open_positions[symbol]['price']

        # Random walk price movement
        new_price = current_price * (1 + np.random.normal(0, 0.01))  # 1% volatility
        self.open_positions[symbol]['price'] = new_price

        return {
            'symbol': symbol,
            'current_price': round(new_price, 2),
            'timestamp': datetime.now()
        }


def alert_handler(alert: Dict):
    """Handle analytics alerts"""
    print(f"\n🚨 ANALYTICS ALERT 🚨")
    print(f"Type: {alert.get('type', 'unknown')}")
    print(f"Message: {alert.get('message', 'No message')}")
    print(f"Severity: {alert.get('severity', 'unknown')}")
    print(f"Time: {alert.get('timestamp', datetime.now())}")

    # In a real system, you would:
    # - Send to Slack/Discord/Telegram
    # - Store in alerts database
    # - Trigger automated responses
    # - Send email notifications


async def simulate_trading(analytics, duration_minutes: int = 30):
    """Simulate trading activity for the specified duration"""
    print(f"🎯 Starting trading simulation for {duration_minutes} minutes...")

    simulator = TradingSimulator(analytics)
    start_time = datetime.now()
    trade_count = 0

    while (datetime.now() - start_time).total_seconds() < duration_minutes * 60:
        try:
            # Generate random trade (70% probability)
            if np.random.random() < 0.7:
                trade = simulator.generate_trade()
                analytics.on_trade_executed(trade)
                trade_count += 1
                print(f"📈 Trade {trade_count}: {trade['action'].upper()} {trade['quantity']} "
                      f"{trade['symbol']} @ ${trade['price']:.2f} [{trade['strategy']}]")

            # Generate price update (90% probability)
            if np.random.random() < 0.9:
                price_update = simulator.generate_price_update()
                if price_update:
                    analytics.on_position_update(price_update)

            # Wait before next action (simulate realistic trading pace)
            await asyncio.sleep(np.random.uniform(2, 8))

        except Exception as e:
            print(f"❌ Error in simulation: {e}")

    print(f"✅ Simulation completed. Generated {trade_count} trades.")


def demonstrate_performance_tracking(analytics):
    """Demonstrate performance tracking features"""
    print("\n📊 PERFORMANCE TRACKING DEMONSTRATION")
    print("=" * 60)

    # Get current performance
    current_perf = analytics.performance_tracker.get_current_performance()
    print(f"Current Portfolio Value: ${current_perf['portfolio_value']:,.2f}")
    print(f"Total Return: {current_perf['total_return']:.2%}")
    print(f"Daily Return: {current_perf['daily_return']:.2%}")
    print(f"Sharpe Ratio: {current_perf['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {current_perf['max_drawdown']:.2%}")
    print(f"Open Positions: {current_perf['position_count']}")

    # Get P&L breakdown
    current_pnl = analytics.pnl_calculator.get_current_pnl()
    print(f"\nP&L Breakdown:")
    print(f"Total P&L: ${current_pnl.total_pnl:.2f}")
    print(f"Unrealized P&L: ${current_pnl.unrealized_pnl:.2f}")
    print(f"Realized P&L: ${current_pnl.realized_pnl:.2f}")

    if current_pnl.by_strategy:
        print(f"\nStrategy Attribution:")
        for strategy, pnl in current_pnl.by_strategy.items():
            print(f"  {strategy}: ${pnl:.2f}")


def demonstrate_performance_ratios(analytics):
    """Demonstrate performance ratio calculations"""
    print("\n📈 PERFORMANCE RATIOS DEMONSTRATION")
    print("=" * 60)

    # Get recent performance history
    history = list(analytics.performance_tracker.performance_history)

    if len(history) > 10:
        returns = [h['daily_return'] for h in history[-30:]]  # Last 30 observations
        ratios = analytics.performance_ratios.calculate_all_ratios(returns)

        print(f"Sharpe Ratio: {ratios.sharpe_ratio:.3f}")
        print(f"Sortino Ratio: {ratios.sortino_ratio:.3f}")
        print(f"Calmar Ratio: {ratios.calmar_ratio:.3f}")
        print(f"Information Ratio: {ratios.information_ratio:.3f}")
        print(f"Omega Ratio: {ratios.omega_ratio:.3f}")
        print(f"Beta: {ratios.beta:.3f}")
        print(f"Alpha: {ratios.jensen_alpha:.4f}")

        # Get confidence intervals
        confidence_intervals = analytics.performance_ratios.get_ratio_confidence_intervals(returns)
        if confidence_intervals:
            print(f"\n95% Confidence Intervals:")
            for ratio, (lower, upper) in confidence_intervals.items():
                print(f"  {ratio}: [{lower:.3f}, {upper:.3f}]")
    else:
        print("Insufficient data for ratio calculations (need more trading history)")


def demonstrate_benchmark_comparison(analytics):
    """Demonstrate benchmark comparison features"""
    print("\n🎯 BENCHMARK COMPARISON DEMONSTRATION")
    print("=" * 60)

    history = list(analytics.performance_tracker.performance_history)

    if len(history) > 20:
        # Create returns series
        returns = pd.Series([h['daily_return'] for h in history])
        timestamps = pd.DatetimeIndex([h['timestamp'] for h in history])
        returns.index = timestamps

        # Compare against major benchmarks
        benchmarks = ['SPY', 'QQQ', 'BTC-USD']
        print(f"Comparing portfolio against: {', '.join(benchmarks)}")

        try:
            comparisons = analytics.benchmark_analyzer.compare_to_multiple_benchmarks(
                returns, benchmarks
            )

            for symbol, comparison in comparisons.items():
                print(f"\n{symbol} Comparison:")
                print(f"  Alpha: {comparison.alpha:.4f} ({comparison.alpha*100:.1f} bps annually)")
                print(f"  Beta: {comparison.beta:.3f}")
                print(f"  Correlation: {comparison.correlation:.3f}")
                print(f"  Information Ratio: {comparison.information_ratio:.3f}")
                print(f"  Tracking Error: {comparison.tracking_error:.3f}")

            # Get multi-benchmark summary
            summary = analytics.benchmark_analyzer.get_multi_benchmark_summary(comparisons)
            print(f"\nBest Alpha Benchmark: {summary.best_alpha_benchmark} "
                  f"(Alpha: {summary.best_alpha_value:.4f})")
            print(f"Average Beta: {summary.average_beta:.3f}")
            print(f"Average Correlation: {summary.average_correlation:.3f}")

        except Exception as e:
            print(f"Benchmark comparison unavailable: {e}")
    else:
        print("Insufficient data for benchmark comparison (need more trading history)")


def demonstrate_drawdown_analysis(analytics):
    """Demonstrate drawdown monitoring features"""
    print("\n📉 DRAWDOWN ANALYSIS DEMONSTRATION")
    print("=" * 60)

    # Get current drawdown status
    current_value = analytics.performance_tracker.current_portfolio_value
    drawdown_metrics = analytics.drawdown_monitor.update(float(current_value))

    print(f"Current Drawdown: {drawdown_metrics['current_drawdown']:.2%}")
    print(f"Maximum Drawdown: {drawdown_metrics['max_drawdown']:.2%}")
    print(f"Peak Value: ${drawdown_metrics['peak_value']:,.2f}")
    print(f"Underwater Days: {drawdown_metrics['underwater_days']}")
    print(f"Recovery Factor: {drawdown_metrics['recovery_factor']:.2%}")

    # Get drawdown statistics
    stats = analytics.drawdown_monitor.get_drawdown_statistics()
    if stats:
        print(f"\nDrawdown Statistics:")
        print(f"  Total Periods: {stats.get('total_periods', 0)}")
        print(f"  Average Drawdown: {stats.get('avg_drawdown', 0):.2%}")
        print(f"  Average Duration: {stats.get('avg_duration', 0):.0f} days")
        print(f"  Worst Drawdown: {stats.get('worst_drawdown', 0):.2%}")
        print(f"  Longest Period: {stats.get('longest_period', 0):.0f} days")


def demonstrate_reporting(analytics):
    """Demonstrate report generation"""
    print("\n📋 REPORT GENERATION DEMONSTRATION")
    print("=" * 60)

    try:
        # Generate comprehensive report
        report_files = analytics.generate_report_on_demand(
            report_type='comprehensive',
            include_charts=True,
            output_formats=['json', 'html']
        )

        if report_files:
            print("Generated reports:")
            for format_type, file_path in report_files.items():
                print(f"  {format_type.upper()}: {file_path}")
        else:
            print("No reports generated (possible insufficient data)")

        # Create custom report configuration
        config = ReportConfig(
            report_type='custom',
            include_charts=True,
            include_detailed_trades=True,
            include_benchmark_comparison=True,
            output_formats=['json']
        )

        performance_data = analytics.get_comprehensive_performance_data()
        custom_report = analytics.performance_reporter.generate_performance_report(
            performance_data, config, "demo_custom_report"
        )

        if custom_report:
            print(f"\nCustom report generated: {custom_report}")

    except Exception as e:
        print(f"Report generation error: {e}")


def demonstrate_api_integration():
    """Demonstrate REST API integration"""
    print("\n🌐 API INTEGRATION DEMONSTRATION")
    print("=" * 60)

    api_base = "http://localhost:5001"

    try:
        # Health check
        response = requests.get(f"{api_base}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API Health: {health['status']}")
            print(f"   Uptime: {health['uptime']:.1f} seconds")
        else:
            print("❌ API health check failed")
            return

        # Get current performance
        response = requests.get(f"{api_base}/api/performance/current", timeout=5)
        if response.status_code == 200:
            data = response.json()['data']
            print(f"\n📊 Current Performance (via API):")
            print(f"   Portfolio Value: ${data['portfolio_value']:,.2f}")
            print(f"   Daily Return: {data['daily_return']:.2%}")
            print(f"   Sharpe Ratio: {data['sharpe_ratio']:.3f}")

        # Get P&L data
        response = requests.get(f"{api_base}/api/pnl/current", timeout=5)
        if response.status_code == 200:
            pnl_data = response.json()['data']
            print(f"\n💰 P&L Data (via API):")
            print(f"   Total P&L: ${pnl_data['total_pnl']:.2f}")
            print(f"   Unrealized: ${pnl_data['unrealized_pnl']:.2f}")
            print(f"   Realized: ${pnl_data['realized_pnl']:.2f}")

        # Generate report via API
        report_request = {
            'report_type': 'api_demo',
            'include_charts': False,
            'output_formats': ['json']
        }

        response = requests.post(
            f"{api_base}/api/reports/generate",
            json=report_request,
            timeout=10
        )

        if response.status_code == 200:
            report_info = response.json()['data']
            print(f"\n📋 Report Generated via API:")
            print(f"   Report ID: {report_info.get('report_id', 'unknown')}")
            print(f"   Files: {list(report_info.get('output_files', {}).keys())}")

    except requests.exceptions.RequestException as e:
        print(f"❌ API connection failed: {e}")
        print("   Make sure the analytics system is running with API enabled")


def print_system_status(analytics):
    """Print comprehensive system status"""
    print("\n🖥️  SYSTEM STATUS")
    print("=" * 60)

    status = analytics.get_current_status()
    print(f"Analytics Running: {'✅' if status['is_running'] else '❌'}")
    print(f"API Server: {'✅' if status['api_server_enabled'] else '❌'}")

    if status['api_server_enabled']:
        print(f"API Port: {status['api_port']}")

    print(f"Auto Reporting: {'✅' if status['auto_reporting_enabled'] else '❌'}")
    print(f"Last Update: {status['last_update']}")
    print(f"Open Positions: {status['position_count']}")
    print(f"Recent Alerts: {status['alerts_count']}")

    components = status['components_status']
    print(f"\nComponent Status:")
    for component, is_running in components.items():
        status_icon = '✅' if is_running else '❌'
        print(f"  {component}: {status_icon}")


async def main():
    """Main demonstration function"""
    print("🚀 ADVANCED TRADING ANALYTICS SYSTEM - COMPLETE EXAMPLE")
    print("=" * 80)

    # Create analytics integration
    print("Initializing analytics system...")
    analytics = create_analytics_integration(
        starting_capital=10000,
        enable_api=True,
        api_port=5001,
        alert_callback=alert_handler
    )

    # Start analytics system
    print("Starting analytics system...")
    analytics.start_analytics()

    # Wait for system to initialize
    await asyncio.sleep(2)

    # Print initial status
    print_system_status(analytics)

    # Start trading simulation in background
    simulation_task = asyncio.create_task(simulate_trading(analytics, duration_minutes=5))

    # Wait a bit for some trading activity
    await asyncio.sleep(30)

    # Demonstrate various features
    demonstrate_performance_tracking(analytics)
    demonstrate_performance_ratios(analytics)
    demonstrate_drawdown_analysis(analytics)

    # Wait for more trading activity
    await asyncio.sleep(30)

    demonstrate_benchmark_comparison(analytics)
    demonstrate_reporting(analytics)

    # Wait a bit more for API demonstration
    await asyncio.sleep(10)
    demonstrate_api_integration()

    # Wait for simulation to complete
    await simulation_task

    # Final status
    print("\n" + "=" * 80)
    print("📊 FINAL PERFORMANCE SUMMARY")
    print("=" * 80)
    demonstrate_performance_tracking(analytics)

    print("\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("The analytics system is now running and can be accessed via:")
    print("  - Python API: analytics object methods")
    print("  - REST API: http://localhost:5001/api/*")
    print("  - Health check: http://localhost:5001/health")
    print("  - API info: http://localhost:5001/info")

    # Keep system running for additional testing
    print("\n💡 System will continue running for 5 more minutes for testing...")
    print("   Press Ctrl+C to stop early")

    try:
        await asyncio.sleep(300)  # 5 minutes
    except KeyboardInterrupt:
        print("\n👋 Stopping system...")

    # Cleanup
    analytics.stop_analytics()
    print("✅ Analytics system stopped successfully")


if __name__ == "__main__":
    try:
        # Run the complete example
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Example interrupted by user")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()