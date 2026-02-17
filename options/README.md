# Advanced Options Trading System

A comprehensive, institutional-grade options trading platform with sophisticated strategies, real-time risk management, and advanced analytics.

## 🚀 Features

### Core Capabilities
- **Real-time Options Data Integration** - Multiple data sources (Alpaca, Polygon, Yahoo Finance)
- **Advanced Greeks Calculation** - Delta, Gamma, Theta, Vega, Rho with second-order Greeks
- **Comprehensive Strategy Suite** - From basic calls/puts to complex multi-leg strategies
- **Portfolio Risk Management** - Real-time monitoring with automated alerts and hedging
- **Volatility Forecasting** - GARCH, LSTM, and ensemble models
- **Options Flow Analysis** - Unusual activity detection and sentiment analysis
- **Professional Backtesting** - Historical IV data and realistic execution modeling

### Strategy Coverage

#### Basic Strategies
- Long/Short Calls and Puts
- Covered Calls and Cash-Secured Puts
- Protective Puts and Collars

#### Spread Strategies
- Bull/Bear Call and Put Spreads
- Iron Condors and Iron Butterflies
- Calendar and Diagonal Spreads
- Ratio Spreads and Backspreads

#### Volatility Strategies
- Long/Short Straddles and Strangles
- Butterflies and Condors
- Volatility Trading Strategies

#### Income Strategies
- Wheel Strategy
- Credit Spread Management
- Covered Call Programs

#### Advanced Strategies
- Synthetic Positions
- Conversion and Reversal Arbitrage
- Multi-expiry Strategies
- Earnings Play Strategies

### Risk Management
- **Real-time Portfolio Greeks** - Continuous monitoring and aggregation
- **Delta Hedging** - Automated delta-neutral portfolio management
- **Gamma Scalping** - Dynamic hedging for gamma exposure
- **Position Sizing** - Volatility-adjusted position sizing algorithms
- **Stress Testing** - Scenario analysis and Monte Carlo simulations
- **Risk Limits** - Customizable limits with automated enforcement

### Analytics & Intelligence
- **Volatility Surface Modeling** - Real-time volatility smile analysis
- **IV Rank and Percentiles** - Historical volatility context
- **Term Structure Analysis** - Volatility term structure monitoring
- **Market Regime Detection** - Adaptive strategy selection
- **Options Flow Analysis** - Large order detection and sentiment

## 📁 System Architecture

```
options/
├── core/                      # Core options infrastructure
│   ├── option_chain.py        # Options chain data management
│   ├── greeks_calculator.py   # Real-time Greeks calculation
│   ├── pricing_models.py      # Black-Scholes, Binomial, Monte Carlo
│   └── option_data_manager.py # Multi-source data integration
│
├── strategies/                # Strategy implementations
│   ├── basic.py              # Calls, puts, covered calls
│   ├── spreads.py            # Bull/bear spreads, condors
│   ├── volatility.py         # Straddles, strangles, butterflies
│   ├── income.py             # Wheel, credit spreads
│   ├── advanced.py           # Synthetics, conversions
│   └── base_strategy.py      # Strategy framework
│
├── risk/                     # Risk management
│   ├── portfolio_greeks.py   # Portfolio Greeks monitoring
│   ├── delta_hedger.py       # Automated delta hedging
│   ├── risk_manager.py       # Risk limits and alerts
│   └── position_sizer.py     # Position sizing algorithms
│
├── analytics/                # Advanced analytics
│   ├── volatility_forecaster.py  # Vol forecasting models
│   ├── iv_analyzer.py            # IV analysis and ranking
│   ├── volatility_surface.py     # Vol surface modeling
│   └── market_regime.py          # Regime detection
│
├── flow/                     # Options flow analysis
│   ├── flow_analyzer.py      # Options flow detection
│   ├── unusual_activity.py   # Large order identification
│   └── sentiment_analyzer.py # Options sentiment
│
├── events/                   # Event-driven strategies
│   ├── earnings_strategy.py  # Earnings play strategies
│   ├── dividend_strategy.py  # Ex-dividend strategies
│   └── event_scanner.py      # Event identification
│
├── backtesting/             # Backtesting framework
│   ├── options_backtester.py # Historical strategy testing
│   ├── iv_data_manager.py    # Historical IV data
│   └── performance_analyzer.py # Performance metrics
│
└── master_options_system.py # Main system orchestrator
```

## 🚦 Quick Start

### 1. Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Install TA-Lib (for technical indicators)
# Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# macOS: brew install ta-lib && pip install TA-Lib
# Linux: apt-get install ta-lib && pip install TA-Lib

# Optional: Install ARCH for GARCH models
pip install arch

# Optional: Install TensorFlow for LSTM models
pip install tensorflow
```

### 2. Configuration

```python
# Set environment variables
export ALPACA_API_KEY="your_alpaca_key"
export ALPACA_SECRET_KEY="your_alpaca_secret"
export POLYGON_API_KEY="your_polygon_key"
```

### 3. Basic Usage

```python
import asyncio
from options import MasterOptionsSystem

async def main():
    # Initialize system
    config = {
        'max_portfolio_delta': 100,
        'max_portfolio_gamma': 50,
        'enable_auto_hedging': True,
        'data_refresh_interval': 30
    }

    system = MasterOptionsSystem(config=config)

    # Create an iron condor strategy
    strategy_id = await system.create_strategy(
        strategy_type='iron_condor',
        symbol='SPY',
        parameters={
            'target_dte': 30,
            'target_delta': 0.15,
            'call_width': 5,
            'put_width': 5
        }
    )

    # Monitor portfolio
    portfolio = system.get_portfolio_summary()
    print(f"Portfolio Delta: {portfolio['portfolio_greeks']['total_delta']}")

    # Start real-time monitoring
    await system.start_system()

asyncio.run(main())
```

### 4. Strategy Examples

#### Iron Condor
```python
# High probability, neutral strategy
await system.create_strategy(
    strategy_type='iron_condor',
    symbol='SPY',
    parameters={
        'target_dte': 45,
        'target_delta': 0.16,  # ~16 delta short strikes
        'call_width': 10,
        'put_width': 10
    }
)
```

#### Covered Call
```python
# Income generation on stock holdings
await system.create_strategy(
    strategy_type='covered_call',
    symbol='AAPL',
    parameters={
        'target_dte': 30,
        'target_delta': 0.30,  # 30 delta call
        'num_contracts': 5
    }
)
```

#### Calendar Spread
```python
# Time decay and volatility strategy
await system.create_strategy(
    strategy_type='calendar',
    symbol='TSLA',
    parameters={
        'near_dte': 30,
        'far_dte': 60,
        'strike_selection': 'atm'
    }
)
```

## 🎯 Advanced Features

### Real-time Greeks Monitoring
```python
from options.risk import PortfolioGreeksMonitor

monitor = PortfolioGreeksMonitor()

# Add alert thresholds
monitor.alert_thresholds = {
    'delta': {'max': 50, 'min': -50},
    'gamma': {'max': 25, 'min': -25},
    'theta': {'max': 0, 'min': -200}
}

# Real-time portfolio metrics
greeks = monitor.calculate_portfolio_greeks()
print(f"Portfolio Delta: {greeks.total_delta}")
print(f"Daily Theta Decay: ${greeks.total_theta:.2f}")
```

### Volatility Forecasting
```python
from options.analytics import VolatilityForecaster, VolatilityModel

forecaster = VolatilityForecaster()

# Multiple forecasting models
forecast = forecaster.generate_forecast(
    price_data=price_df,
    model_type=VolatilityModel.ENSEMBLE
)

print(f"30-day volatility forecast: {forecast.forecast_values}")
```

### Options Flow Analysis
```python
from options.flow import FlowAnalyzer

flow_analyzer = FlowAnalyzer()

# Detect unusual options activity
unusual_activity = flow_analyzer.detect_unusual_activity('SPY')
for activity in unusual_activity:
    print(f"Large {activity['option_type']} order: {activity['volume']} contracts")
```

### Automated Risk Management
```python
# Set up automated hedging
system.config['enable_auto_hedging'] = True
system.config['delta_hedge_threshold'] = 25

# Custom risk alert handler
async def risk_handler(alert, portfolio_greeks):
    if alert['severity'] == 'high':
        print(f"HIGH RISK: {alert['message']}")
        # Custom logic here

system.add_alert_callback(risk_handler)
```

## 📊 Performance Analytics

### Strategy Performance
```python
# Get detailed performance metrics
performance = await system._calculate_performance_metrics()

print(f"Total P&L: ${performance['total_pnl']:.2f}")
print(f"Return on Premium: {performance['return_on_premium']:.1%}")
print(f"Win Rate: {performance['winning_strategies']/(performance['total_strategies'] or 1):.1%}")
```

### Historical Analysis
```python
# Analyze strategy performance over time
history = system.performance_history

pnl_series = [entry['metrics']['total_pnl'] for entry in history]
returns = pd.Series(pnl_series).pct_change()

print(f"Sharpe Ratio: {returns.mean()/returns.std()*np.sqrt(252):.2f}")
print(f"Max Drawdown: {(returns.cumsum().cummax() - returns.cumsum()).max():.1%}")
```

## 🔧 Integration with Existing Trading Bot

### Add to Main Trading System
```python
# In your main trading bot
from options import MasterOptionsSystem

class EnhancedTradingBot:
    def __init__(self):
        self.stock_strategies = StockStrategies()
        self.options_system = MasterOptionsSystem()

    async def run_enhanced_strategies(self):
        # Run stock strategies
        stock_signals = await self.stock_strategies.get_signals()

        # Generate options strategies based on stock signals
        for signal in stock_signals:
            if signal['confidence'] > 0.8:
                # High confidence - use options for leverage
                await self.options_system.create_strategy(
                    strategy_type='bull_call_spread' if signal['direction'] == 'bullish' else 'bear_put_spread',
                    symbol=signal['symbol'],
                    parameters={'target_dte': 30, 'target_delta': 0.7}
                )
```

## 📈 Risk Management Best Practices

### Position Sizing
```python
# Volatility-adjusted position sizing
from options.risk import OptionsPositionSizer

sizer = OptionsPositionSizer()
position_size = sizer.calculate_position_size(
    account_value=100000,
    risk_per_trade=0.02,  # 2% risk
    strategy_max_loss=500,
    volatility_adjustment=True
)
```

### Portfolio Limits
```python
# Set comprehensive risk limits
risk_limits = {
    'max_portfolio_delta': 100,
    'max_portfolio_gamma': 50,
    'max_portfolio_theta': -500,
    'max_portfolio_vega': 1000,
    'max_single_position_size': 10,
    'max_concentration_per_underlying': 0.25
}

system.risk_manager.set_limits(risk_limits)
```

## 🎛️ Configuration Options

### System Configuration
```python
config = {
    # Data settings
    'data_refresh_interval': 30,        # seconds
    'data_sources': ['alpaca', 'polygon'],
    'cache_expiry': 300,               # seconds

    # Risk settings
    'max_portfolio_delta': 100,
    'max_portfolio_gamma': 50,
    'max_single_position_size': 10,
    'enable_auto_hedging': True,
    'hedge_threshold': 25,

    # Strategy settings
    'default_dte_target': 30,
    'min_iv_rank': 20,                # Minimum IV rank for strategies
    'max_bid_ask_spread': 0.05,       # Maximum spread as % of price

    # Performance settings
    'save_performance_data': True,
    'performance_history_length': 1000,
    'enable_trade_logging': True
}
```

## 🔍 Monitoring and Alerts

### Real-time Dashboard Integration
```python
# Integration with existing dashboard
from options import MasterOptionsSystem

@app.route('/api/options/portfolio')
async def options_portfolio():
    summary = options_system.get_portfolio_summary()
    return jsonify(summary)

@app.route('/api/options/strategies')
async def active_strategies():
    strategies = options_system.get_strategy_summary()
    return jsonify(strategies)

# WebSocket for real-time updates
@socketio.on('subscribe_greeks')
def handle_greeks_subscription():
    # Send real-time Greeks updates
    greeks = options_system.greeks_monitor.calculate_portfolio_greeks()
    emit('greeks_update', greeks.to_dict())
```

## 🏗️ System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB+ recommended for ML models
- **Storage**: SSD recommended for options data caching
- **Network**: Stable internet for real-time data feeds

### Software Requirements
- **Python**: 3.8+
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Dependencies**: See requirements.txt

### API Requirements
- **Alpaca API** (recommended for options data)
- **Polygon.io API** (alternative data source)
- **Yahoo Finance** (free backup source)

## 🚨 Important Disclaimers

1. **Paper Trading First**: Always paper trade strategies before risking real capital
2. **Risk Management**: Options trading involves substantial risk of loss
3. **Market Data**: Real-time data feeds require paid subscriptions
4. **Regulatory Compliance**: Ensure compliance with local trading regulations
5. **Tax Implications**: Consult tax professionals for options trading implications

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your enhancement
4. Add comprehensive tests
5. Submit a pull request

## 📞 Support

For support, questions, or feature requests:
- Create an issue on GitHub
- Check the documentation
- Review example implementations

---

**Built for professional options traders who demand institutional-grade tools and risk management.**