# Advanced Risk Management System

Comprehensive risk management framework for multi-bot trading system with sophisticated risk controls, real-time monitoring, and automated protection protocols.

## 🚀 Features

### Core Risk Management
- **Portfolio Heat Monitoring** - Real-time tracking of portfolio risk exposure
- **Correlation Analysis** - Dynamic correlation clustering and concentration limits
- **Value-at-Risk (VaR)** - Multi-method VaR with Monte Carlo simulation
- **Dynamic Position Sizing** - Kelly Criterion optimization with safety margins
- **Drawdown Protection** - Automatic position reduction during losses
- **Real-time Alerts** - Multi-channel alert system (Telegram, Email, Discord)

### Advanced Components
- **Stress Testing** - Extreme market scenario analysis
- **Kelly Optimization** - Multi-asset Kelly criterion with constraints
- **Emergency Protocols** - Automatic halt and liquidation procedures
- **Risk Budget Allocation** - Strategy-based risk allocation
- **Trading Integration** - Seamless bot integration with approval system
- **Dashboard Integration** - Real-time monitoring interface

## 📁 Directory Structure

```
risk_management/
├── config/
│   └── risk_config.py          # Comprehensive configuration
├── core/
│   ├── portfolio_heat.py       # Portfolio heat monitoring
│   ├── correlation_monitor.py  # Correlation analysis
│   └── risk_manager.py         # Main risk manager
├── calculators/
│   ├── var_calculator.py       # VaR calculations
│   ├── kelly_optimizer.py      # Kelly criterion optimization
│   └── stress_tester.py        # Stress testing engine
├── monitors/
│   └── drawdown_protection.py  # Drawdown protection system
├── alerts/
│   └── risk_alerts.py          # Alert management system
├── integration/
│   ├── trading_integration.py  # Trading system integration
│   └── dashboard_integration.py # Dashboard API
└── __init__.py                 # Main exports
```

## 🔧 Quick Start

### Basic Usage

```python
from risk_management import AdvancedRiskManager
from risk_management.config.risk_config import load_risk_config

# Initialize risk management
config = load_risk_config()
risk_manager = AdvancedRiskManager(config)

# Add positions
risk_manager.add_position(
    symbol='SPY',
    size=1000,
    entry_price=450.0,
    strategy='momentum',
    risk_amount=500
)

# Assess risk
assessment = risk_manager.assess_overall_risk()
print(f"Risk Score: {assessment.overall_risk_score}/100")
print(f"New trades allowed: {assessment.new_trades_allowed}")

# Evaluate trade
decision = risk_manager.evaluate_trade(
    symbol='QQQ',
    action='buy',
    proposed_size=800,
    strategy='mean_reversion'
)

if decision.approved:
    print(f"Trade approved: ${decision.recommended_size:,.0f}")
else:
    print(f"Trade rejected: {decision.rejection_reasons}")
```

### Trading Integration

```python
from risk_management.integration.trading_integration import (
    initialize_risk_management,
    request_trade_approval,
    report_trade_execution
)

# Initialize integration
integration = initialize_risk_management()
integration.start_monitoring()

# Request trade from bot
response = request_trade_approval(
    bot_name='RSI2-MeanReversion',
    symbol='AAPL',
    action='buy',
    size=1200,
    confidence=0.85
)

if response.approved:
    # Execute trade and report back
    report_trade_execution(
        request_id=response.request_id,
        executed_size=response.recommended_size,
        executed_price=180.0,
        success=True
    )
```

### Dashboard Integration

```python
from risk_management.integration.dashboard_integration import (
    RiskDashboardAPI,
    create_risk_routes
)

# Flask app integration
from flask import Flask
app = Flask(__name__)

# Create dashboard API
dashboard_api = RiskDashboardAPI()

# Add routes to Flask app
create_risk_routes(app)

# Available endpoints:
# /api/risk/metrics       - Real-time metrics
# /api/risk/heatmap       - Position heatmap
# /api/risk/correlation   - Correlation matrix
# /api/risk/var           - VaR breakdown
# /api/risk/stress        - Stress test results
# /api/risk/alerts        - Alert dashboard
# /api/risk/bots          - Bot performance
```

## 📊 Risk Components

### 1. Portfolio Heat Monitor

Tracks concentration and exposure across all positions:

```python
from risk_management.core.portfolio_heat import PortfolioHeatMonitor

monitor = PortfolioHeatMonitor(config)
monitor.add_position('SPY', 100, 450.0, 'momentum', 1000)

metrics = monitor.calculate_heat_metrics()
print(f"Portfolio Heat: {metrics.overall_heat}/100")
print(f"Max Single Position: {metrics.max_single_position_pct:.1%}")
```

### 2. Correlation Monitor

Analyzes correlation patterns and clustering:

```python
from risk_management.core.correlation_monitor import CorrelationMonitor

corr_monitor = CorrelationMonitor(config)
corr_monitor.update_positions({'SPY': 0.3, 'QQQ': 0.2, 'AAPL': 0.15})

metrics = corr_monitor.analyze_correlations()
print(f"Max Correlation: {metrics.max_pairwise_correlation:.2%}")
print(f"Clusters Found: {len(metrics.clusters)}")
```

### 3. VaR Calculator

Multi-method Value-at-Risk calculations:

```python
from risk_management.calculators.var_calculator import VaRCalculator

var_calc = VaRCalculator(config)
var_calc.update_positions({'SPY': 5000, 'QQQ': 3000}, 10000)

# Calculate VaR
summary = var_calc.calculate_portfolio_var()
print(f"95% VaR: ${summary.total_var['historical_0.95_1d'].var_value:,.2f}")
```

### 4. Kelly Optimizer

Advanced position sizing with Kelly Criterion:

```python
from risk_management.calculators.kelly_optimizer import KellyOptimizer

kelly = KellyOptimizer(config)

# Single asset Kelly
result = kelly.calculate_kelly_single_asset('SPY')
print(f"Recommended Kelly: {result.recommended_kelly_fraction:.1%}")

# Portfolio optimization
portfolio_result = kelly.optimize_portfolio_kelly(['SPY', 'QQQ', 'AAPL'])
print(f"Optimal Weights: {portfolio_result.optimal_weights}")
```

### 5. Drawdown Protection

Automatic position reduction during drawdowns:

```python
from risk_management.monitors.drawdown_protection import DrawdownProtection

protection = DrawdownProtection(config)

# Update portfolio value
status = protection.update_portfolio_value(9500)  # 5% drawdown
print(f"Drawdown: {status.current_drawdown.drawdown_percentage:.1%}")
print(f"Position Multiplier: {status.position_size_multiplier:.0%}")
```

### 6. Stress Testing

Extreme scenario analysis:

```python
from risk_management.calculators.stress_tester import StressTester, StressScenario

stress_tester = StressTester(config)
stress_tester.update_portfolio({'SPY': 5000, 'QQQ': 3000}, 10000)

# Test 2008 crisis scenario
result = stress_tester.run_scenario_test(StressScenario.MARKET_CRASH_2008)
print(f"2008 Crisis Loss: ${result.portfolio_pnl:,.2f}")

# Full stress suite
suite = stress_tester.run_full_stress_suite()
print(f"Worst Case: {suite.worst_case_scenario}")
print(f"Scenarios Passed: {suite.scenarios_passed}/{len(suite.scenarios_tested)}")
```

## ⚙️ Configuration

### Risk Management Config

```python
from risk_management.config.risk_config import RiskManagementConfig, RiskLevel

config = RiskManagementConfig()

# Adjust risk level
config.risk_level = RiskLevel.CONSERVATIVE
config.adjust_for_risk_level()

# Portfolio limits
config.portfolio_limits.max_single_position = 0.10  # 10% max
config.portfolio_limits.max_sector_exposure = 0.25   # 25% max

# Drawdown protection
config.drawdown_config.max_portfolio_drawdown = 0.15  # V4: 15% max

# VaR settings
config.var_config.daily_var_limit = 0.02  # 2% daily VaR limit
config.var_config.monte_carlo_simulations = 10000

# Kelly settings
config.kelly_config.default_kelly_fraction = 0.25  # Quarter Kelly
```

### Environment Variables

```bash
# Risk management
RISK_LEVEL=moderate
PORTFOLIO_VALUE=10000
MAX_PORTFOLIO_DRAWDOWN=0.15

# Alerts
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
DISCORD_WEBHOOK_URL=your_webhook

# Email alerts
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=your_email
SMTP_PASSWORD=your_password
```

## 🔔 Alert System

### Multi-Channel Alerts

The system supports multiple notification channels:

- **Telegram** - Real-time trading alerts
- **Email** - Critical system alerts
- **Discord** - Team notifications
- **Slack** - Workplace integration
- **Webhook** - Custom integrations

### Alert Types

- `PORTFOLIO_HEAT` - Portfolio concentration alerts
- `DRAWDOWN` - Drawdown protection alerts
- `CORRELATION` - High correlation warnings
- `VAR_BREACH` - VaR limit violations
- `POSITION_LIMIT` - Position size violations
- `VOLATILITY_SPIKE` - Market volatility alerts
- `SYSTEM_ERROR` - System malfunction alerts

### Custom Alert Rules

```python
from risk_management.alerts.risk_alerts import AlertRule, AlertType, AlertSeverity

rule = AlertRule(
    rule_id="custom_heat_warning",
    name="Custom Heat Warning",
    alert_type=AlertType.PORTFOLIO_HEAT,
    severity=AlertSeverity.WARNING,
    condition="portfolio_heat > 80",  # When portfolio heat > 80%
    channels=[AlertChannel.TELEGRAM, AlertChannel.EMAIL]
)

risk_manager.alert_manager.add_custom_rule(rule)
```

## 🏗️ Integration with Existing System

### Master Orchestrator Integration

```python
# In your master_orchestrator.py
from risk_management.integration.trading_integration import initialize_risk_management

# Initialize at startup
risk_integration = initialize_risk_management()
risk_integration.start_monitoring()

# Before executing trades
def execute_trade_with_risk_check(bot_name, symbol, action, size):
    response = risk_integration.request_trade(
        bot_name=bot_name,
        symbol=symbol,
        action=action,
        size=size
    )

    if response.approved:
        # Execute with recommended size
        actual_size = response.recommended_size
        # ... execute trade

        # Report execution
        risk_integration.execute_trade(
            request_id=response.request_id,
            executed_size=actual_size,
            executed_price=execution_price,
            success=True
        )
    else:
        logger.warning(f"Trade rejected: {response.rejection_reasons}")
```

### Bot Registration

```python
# Register each bot with risk management
risk_integration.register_bot(
    bot_name='RSI2-MeanReversion',
    strategy_type='mean_reversion',
    max_position_size=5000,
    kelly_fraction=0.25
)
```

## 📈 Performance Monitoring

### Key Metrics Tracked

- **Overall Risk Score** (0-100)
- **Portfolio Heat** - Concentration metrics
- **Correlation Risk** - Diversification analysis
- **VaR Compliance** - Risk limit adherence
- **Drawdown Tracking** - Real-time drawdown monitoring
- **Bot Performance** - Individual bot statistics

### Dashboard Views

1. **Risk Overview** - Real-time risk dashboard
2. **Position Heatmap** - Visual position risk analysis
3. **Correlation Matrix** - Interactive correlation view
4. **VaR Analysis** - Detailed VaR breakdown
5. **Stress Testing** - Scenario analysis results
6. **Alert Center** - Alert management interface
7. **Bot Performance** - Trading bot monitoring

## 🚨 Emergency Protocols

### Automatic Triggers

- **Emergency Drawdown** (25%) - Immediate position liquidation
- **VaR Breach** (95%+ of limit) - Halt new trades
- **Correlation Spike** (>90%) - Reduce correlated positions
- **System Error** - Emergency stop all trading

### Manual Controls

```python
# Emergency stop all trading
risk_integration.emergency_stop_all("Market conditions")

# Disable specific bot
risk_integration.disable_bot("RSI2-MeanReversion")

# Reset emergency mode
risk_manager._deactivate_emergency_mode("Manual reset")
```

## 🧪 Testing

### Unit Tests

```bash
cd risk_management
python -m pytest tests/ -v
```

### Integration Tests

```python
# Test full system integration
from risk_management.tests.integration_test import run_full_integration_test

results = run_full_integration_test()
print(f"Tests passed: {results.passed}/{results.total}")
```

### Stress Testing

```python
# Run stress tests
suite = stress_tester.run_full_stress_suite()
print(f"Critical failures: {suite.critical_failures}")
```

## 📚 API Reference

### Core Classes

- `AdvancedRiskManager` - Main risk management interface
- `PortfolioHeatMonitor` - Portfolio concentration monitoring
- `CorrelationMonitor` - Correlation analysis and clustering
- `VaRCalculator` - Value-at-Risk calculations
- `KellyOptimizer` - Position sizing optimization
- `DrawdownProtection` - Drawdown monitoring and protection
- `RiskAlertManager` - Alert system management

### Integration Classes

- `TradingSystemIntegration` - Trading system interface
- `RiskDashboardAPI` - Dashboard API interface

### Configuration Classes

- `RiskManagementConfig` - Master configuration
- `PortfolioLimits` - Portfolio limits configuration
- `VaRConfig` - VaR calculation settings
- `AlertConfig` - Alert system settings

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Ensure proper path setup
   import sys
   sys.path.append('/path/to/trading_bot')
   ```

2. **Database Connection Issues**
   ```bash
   # Check logs directory exists
   mkdir -p logs/
   chmod 755 logs/
   ```

3. **Alert Delivery Issues**
   ```bash
   # Verify environment variables
   echo $TELEGRAM_BOT_TOKEN
   echo $TELEGRAM_CHAT_ID
   ```

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('risk_management')
```

## 📄 License

Created for Trading Bot Arsenal - February 2026

## 🤝 Contributing

1. Follow existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure all risk limits are configurable
5. Test integration with existing trading system

## 📞 Support

For technical support or questions about the risk management system, please review the code documentation and test examples provided in each module.