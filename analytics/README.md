# Advanced Real-Time Performance Analytics System

A comprehensive trading performance monitoring and analytics system providing real-time P&L tracking, advanced performance ratios, risk metrics, and automated reporting.

## Features

### Core Analytics
- **Real-time P&L Tracking**: Live position tracking with unrealized/realized P&L separation
- **Performance Ratios**: Sharpe, Sortino, Calmar, Information, Treynor, Jensen's Alpha, Omega ratios
- **Rolling Performance Windows**: 1d, 7d, 30d, 90d, 1y analysis periods
- **Drawdown Monitoring**: Real-time drawdown tracking with alert thresholds
- **Trade Statistics**: Win rate, profit factor, expectancy, distribution analysis
- **Strategy Attribution**: Performance breakdown by trading strategy
- **Risk-Adjusted Returns**: Alpha/Beta calculations and significance testing

### Benchmarking
- **Multi-Benchmark Comparison**: SPY, QQQ, BTC, bonds, commodities, and custom benchmarks
- **Rolling Alpha/Beta Analysis**: Time-varying risk metrics
- **Correlation Analysis**: Portfolio correlation with various asset classes
- **Performance Attribution**: Selection vs timing return decomposition

### Reporting & Visualization
- **Multi-Format Reports**: JSON, HTML, and PDF output formats
- **Interactive Charts**: Plotly-based dashboards with real-time updates
- **Automated Reporting**: Daily, weekly, monthly report generation
- **Email/Telegram Delivery**: Automated report distribution

### API & Integration
- **REST API**: Comprehensive API for external consumption
- **WebSocket Streaming**: Real-time performance data feeds
- **Dashboard Integration**: Ready-to-use endpoints for web dashboards
- **CORS Support**: Cross-origin resource sharing for web apps

## Quick Start

### Installation

```python
# The system is integrated into the trading bot project
# All dependencies are included in requirements.txt
```

### Basic Usage

```python
from analytics.analytics_integration import create_analytics_integration

# Create analytics integration
analytics = create_analytics_integration(
    starting_capital=10000,
    enable_api=True,
    api_port=5001
)

# Start analytics system
analytics.start_analytics()

# Simulate trade execution
trade_data = {
    'trade_id': 'trade_001',
    'symbol': 'AAPL',
    'action': 'buy',
    'price': 150.0,
    'quantity': 100,
    'strategy': 'momentum',
    'platform': 'alpaca'
}
analytics.on_trade_executed(trade_data)

# Update position price
analytics.on_position_update({
    'symbol': 'AAPL',
    'current_price': 155.0
})

# Generate performance report
report_files = analytics.generate_report_on_demand(
    report_type='daily',
    include_charts=True,
    output_formats=['json', 'html']
)
```

## System Architecture

```
analytics/
├── core/                          # Core analytics modules
│   ├── performance_tracker.py     # Real-time performance monitoring
│   └── pnl_calculator.py         # P&L calculation engine
├── metrics/                       # Performance metrics
│   ├── performance_ratios.py     # Sharpe, Sortino, Calmar, etc.
│   ├── drawdown_monitor.py       # Drawdown tracking & alerts
│   └── trade_statistics.py       # Trade analysis & statistics
├── benchmarks/                    # Benchmark analysis
│   └── benchmark_analyzer.py     # Multi-benchmark comparison
├── reporting/                     # Report generation
│   └── performance_reporter.py   # Multi-format reporting
├── analytics_api.py              # REST API server
├── analytics_integration.py      # Main integration module
└── README.md                     # This file
```

## API Endpoints

### Performance Tracking
- `GET /api/performance/current` - Current performance metrics
- `GET /api/performance/historical` - Historical performance data
- `GET /api/performance/summary` - Performance summary

### P&L Analysis
- `GET /api/pnl/current` - Current P&L snapshot
- `GET /api/pnl/by-strategy` - P&L breakdown by strategy
- `GET /api/pnl/rolling` - Rolling P&L analysis

### Performance Ratios
- `GET /api/ratios/current` - Current performance ratios
- `GET /api/ratios/rolling` - Rolling ratio analysis
- `GET /api/ratios/confidence` - Ratio confidence intervals

### Drawdown Monitoring
- `GET /api/drawdown/current` - Current drawdown status
- `GET /api/drawdown/history` - Drawdown history
- `GET /api/drawdown/statistics` - Drawdown statistics

### Benchmark Comparison
- `GET /api/benchmarks/compare` - Multi-benchmark comparison
- `GET /api/benchmarks/rolling` - Rolling benchmark analysis
- `GET /api/benchmarks/correlation` - Benchmark correlations

### Reporting
- `POST /api/reports/generate` - Generate custom report
- `GET /api/reports/list` - List available reports
- `GET /api/reports/{id}` - Get specific report

### Dashboard
- `GET /api/dashboard/overview` - Dashboard overview data
- `GET /api/dashboard/charts` - Chart data for visualization

## Configuration

### Performance Tracker Configuration

```python
performance_tracker = PerformanceTracker(
    starting_capital=10000.0,
    update_interval=60,  # seconds
    alert_callback=your_alert_handler
)

# Set alert thresholds
performance_tracker.alert_thresholds = {
    'max_drawdown': 0.05,      # 5% max drawdown
    'daily_loss': 0.02,        # 2% daily loss
    'position_concentration': 0.20,  # 20% max position size
    'volatility_spike': 0.30   # 30% volatility increase
}
```

### Report Configuration

```python
from analytics.reporting.performance_reporter import ReportConfig

config = ReportConfig(
    report_type='monthly',
    include_charts=True,
    include_detailed_trades=True,
    include_benchmark_comparison=True,
    output_formats=['json', 'html', 'pdf'],
    email_recipients=['trader@example.com'],
    telegram_chat_ids=['123456789']
)
```

### Drawdown Monitor Configuration

```python
# Set custom alert thresholds
drawdown_monitor.set_alert_thresholds({
    'moderate_drawdown': 0.03,     # 3%
    'significant_drawdown': 0.07,  # 7%
    'severe_drawdown': 0.12,       # 12%
    'critical_drawdown': 0.18,     # 18%
    'extended_underwater': 20,     # 20 days
    'very_long_underwater': 45     # 45 days
})
```

## Examples

### 1. Real-Time Performance Monitoring

```python
import asyncio
from analytics.core.performance_tracker import PerformanceTracker

# Create performance tracker
tracker = PerformanceTracker(starting_capital=50000)

# Start monitoring
tracker.start_monitoring()

# Get real-time updates
while True:
    performance = tracker.get_current_performance()
    print(f"Portfolio Value: ${performance['portfolio_value']:,.2f}")
    print(f"Daily Return: {performance['daily_return']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")

    await asyncio.sleep(5)  # Update every 5 seconds
```

### 2. Multi-Strategy Attribution Analysis

```python
from analytics.core.pnl_calculator import PnLCalculator

calculator = PnLCalculator()

# Open positions for different strategies
calculator.open_position('momentum_1', 'AAPL', 'momentum', 'alpaca', 'long', 150.0, 100)
calculator.open_position('mean_rev_1', 'MSFT', 'mean_reversion', 'alpaca', 'long', 300.0, 50)

# Update prices
calculator.update_price('AAPL', 155.0)
calculator.update_price('MSFT', 295.0)

# Get strategy attribution
attribution = calculator.get_strategy_attribution()
for strategy, metrics in attribution.items():
    print(f"{strategy}: Total P&L = ${metrics['total_pnl']:.2f}")
```

### 3. Comprehensive Benchmark Analysis

```python
from analytics.benchmarks.benchmark_analyzer import BenchmarkAnalyzer
import pandas as pd

analyzer = BenchmarkAnalyzer()

# Create sample portfolio returns
portfolio_returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01] * 50)

# Compare against multiple benchmarks
benchmarks = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'BTC-USD']
comparisons = analyzer.compare_to_multiple_benchmarks(portfolio_returns, benchmarks)

# Display results
for symbol, comparison in comparisons.items():
    print(f"\n{symbol} Comparison:")
    print(f"  Alpha: {comparison.alpha:.3f}")
    print(f"  Beta: {comparison.beta:.3f}")
    print(f"  Information Ratio: {comparison.information_ratio:.3f}")
    print(f"  Correlation: {comparison.correlation:.3f}")
```

### 4. Automated Report Generation

```python
from analytics.reporting.performance_reporter import PerformanceReporter, ReportConfig

reporter = PerformanceReporter()

# Configure report
config = ReportConfig(
    report_type='weekly',
    include_charts=True,
    include_benchmark_comparison=True,
    output_formats=['html', 'json']
)

# Generate report
performance_data = {
    'overall_metrics': {
        'total_return': 0.15,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.08
    }
}

report_files = reporter.generate_performance_report(performance_data, config)
print(f"Generated reports: {report_files}")
```

### 5. REST API Integration

```python
import requests

# Get current performance via API
response = requests.get('http://localhost:5001/api/performance/current')
performance = response.json()

print(f"Portfolio Value: ${performance['data']['portfolio_value']:,.2f}")
print(f"Total Return: {performance['data']['total_return']:.1%}")

# Generate custom report
report_request = {
    'report_type': 'custom',
    'include_charts': True,
    'output_formats': ['html']
}

response = requests.post(
    'http://localhost:5001/api/reports/generate',
    json=report_request
)
report_info = response.json()
print(f"Report generated: {report_info['data']['output_files']}")
```

## Alert System

The analytics system includes a comprehensive alert framework:

### Performance Alerts
- Maximum drawdown exceeded
- Daily loss threshold breached
- Position concentration too high
- Volatility spike detected

### Drawdown Alerts
- Moderate/significant/severe/critical drawdown levels
- Extended underwater periods
- Recovery factor changes

### Custom Alert Handlers

```python
def custom_alert_handler(alert):
    print(f"Alert Type: {alert['type']}")
    print(f"Message: {alert['message']}")
    print(f"Severity: {alert['severity']}")

    # Send to Slack, email, etc.
    if alert['severity'] == 'critical':
        send_emergency_notification(alert)

analytics = create_analytics_integration(alert_callback=custom_alert_handler)
```

## Database Schema

### Performance Snapshots
- Portfolio value tracking
- Daily/MTD/YTD returns
- Risk metrics (Sharpe, drawdown, volatility)
- Strategy/platform/market attribution

### P&L Tracking
- Position-level tracking
- Realized vs unrealized P&L
- Currency conversion support
- Historical snapshots

### Drawdown Monitoring
- Drawdown periods and recovery
- Underwater curve tracking
- Alert history

### Trade Statistics
- Trade-level metrics
- Strategy performance
- Distribution analysis

## Integration with Trading System

### Master Orchestrator Integration

```python
# In your master_orchestrator.py, add:
from analytics.analytics_integration import create_analytics_integration

class MasterOrchestrator:
    def __init__(self):
        # ... existing code ...

        # Initialize analytics
        self.analytics = create_analytics_integration(
            master_orchestrator=self,
            starting_capital=self.starting_capital,
            enable_api=True
        )

    def start(self):
        # ... existing code ...

        # Start analytics
        self.analytics.start_analytics()

    def _log_trade_from_signal(self, bot_name: str, signal: Dict):
        # ... existing code ...

        # Notify analytics of trade
        self.analytics.on_trade_executed({
            'trade_id': f"{bot_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'symbol': signal.get('symbol', 'UNKNOWN'),
            'action': signal.get('action', 'unknown'),
            'price': signal.get('price', 0),
            'quantity': signal.get('quantity', 1),
            'strategy': bot_name,
            'platform': 'alpaca'  # or determine from bot
        })
```

## Performance Considerations

- **Memory Usage**: Maintains rolling windows with configurable sizes
- **CPU Usage**: Optimized calculations with caching
- **Database**: SQLite for local storage, easily upgradeable to PostgreSQL
- **Network**: Efficient API with pagination and filtering
- **Scalability**: Modular design allows for distributed deployment

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- yfinance: Market data fetching
- flask: REST API server
- scipy: Statistical calculations
- matplotlib/plotly: Chart generation (optional)

## License

This analytics system is part of the trading bot project and follows the same licensing terms.

## Support

For questions or issues with the analytics system:
1. Check the examples in this README
2. Review the API documentation at `/info` endpoint
3. Enable debug logging for detailed troubleshooting
4. Check the database for data integrity issues