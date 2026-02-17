# AI Trading Bot Test Suite
**Comprehensive Testing Framework for All AI Trading Components**

## 📁 Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                   # Pytest configuration and fixtures
├── README.md                     # This documentation
│
├── unit/                         # Unit Tests (Individual Components)
│   ├── __init__.py
│   └── test_ai_bots.py          # Individual AI bot testing
│
├── integration/                  # Integration Tests (Component Interaction)
│   ├── __init__.py
│   ├── test_orchestrator.py     # Master orchestrator integration
│   ├── test_dashboard_api.py    # Dashboard API endpoints
│   └── test_news_feeds.py       # News feed connectivity & fallbacks
│
├── performance/                  # Performance & Benchmarks
│   ├── __init__.py
│   └── test_ai_performance.py   # AI component performance testing
│
├── scenarios/                    # Error Scenarios & Edge Cases
│   ├── __init__.py
│   └── test_error_scenarios.py  # Network failures, data corruption, etc.
│
└── mocks/                        # Mock Data & Test Utilities
    ├── __init__.py
    └── mock_data.py              # Centralized mock data generation
```

## 🚀 Quick Start

### Run All Tests
```bash
# Run complete test suite
python run_tests.py

# Run with coverage report
python run_tests.py --coverage

# Run with HTML report
python run_tests.py --html --coverage
```

### Run Specific Test Categories
```bash
# Unit tests only
python run_tests.py --unit

# Integration tests only
python run_tests.py --integration

# Performance tests only
python run_tests.py --performance

# Error scenarios only
python run_tests.py --scenarios
```

### Fast Testing (Skip Slow Tests)
```bash
# Quick test run (skips performance tests and slow markers)
python run_tests.py --fast

# Quick unit tests with coverage
python run_tests.py --unit --fast --coverage
```

## 🧪 Test Categories

### 1. Unit Tests (`tests/unit/`)
Test individual AI bots and components in isolation:

- **Kalshi Bot**: Fed prediction markets, paper trading
- **Sports AI Bot**: AI-powered sports betting analysis
- **Arbitrage Bot**: Cross-market arbitrage detection
- **OANDA Bot**: Forex trading signals and execution
- **Alpaca Crypto Bot**: RSI-based crypto trading
- **Sentiment Bot**: News sentiment analysis and signals
- **Weather Bot**: Weather prediction market edges
- **AI Components**: LLM client, veto layer, prediction analyzer

**Key Features:**
- Mocked external dependencies (APIs, databases)
- Individual bot functionality validation
- Error handling for each component
- Performance baseline testing

### 2. Integration Tests (`tests/integration/`)
Test component interaction and data flow:

- **Master Orchestrator**: Multi-bot coordination and scheduling
- **Dashboard API**: All Flask endpoints and data flow
- **News Feeds**: Connectivity, fallbacks, data processing pipeline
- **Database Integration**: Trade logging, bot status updates

**Key Features:**
- Real component interaction (with mocked external services)
- End-to-end workflow validation
- API endpoint comprehensive testing
- Data consistency verification

### 3. Performance Tests (`tests/performance/`)
Benchmark AI components and system performance:

- **AI Response Times**: LLM query performance, concurrent requests
- **Market Scanning**: Performance across different market types
- **News Processing**: Batch processing performance
- **Memory Usage**: Memory patterns and efficiency
- **Concurrent Execution**: Multi-bot performance testing
- **Database Performance**: High-volume trade logging

**Key Features:**
- Performance thresholds and SLA validation
- Scalability testing with increasing data volumes
- Memory and CPU usage monitoring
- Stress testing under extreme conditions

### 4. Error Scenarios (`tests/scenarios/`)
Test error handling and edge cases:

- **Network Failures**: API timeouts, connection errors, rate limiting
- **Data Corruption**: Malformed JSON, missing fields, invalid data
- **Resource Exhaustion**: Memory limits, CPU overload, disk space
- **Market Conditions**: Flash crashes, circuit breakers, liquidity crises
- **Concurrency Issues**: Race conditions, deadlocks, database contention
- **Recovery Mechanisms**: Automatic restart, graceful degradation, backups

**Key Features:**
- Comprehensive error simulation
- Recovery mechanism validation
- System resilience testing
- Failure mode analysis

## 🛠️ Test Configuration

### Pytest Configuration (`pytest.ini`)
- Test discovery patterns
- Marker definitions for test categorization
- Timeout configurations
- Coverage settings
- Parallel execution options
- Warning filters

### Fixtures (`conftest.py`)
**Global Fixtures Available to All Tests:**

- `mock_env_vars`: Mock environment variables
- `mock_db`: In-memory SQLite database
- `mock_market_data`: OHLCV data for backtesting
- `mock_kalshi_markets`: Prediction market data
- `mock_news_feed`: News articles with sentiment
- `mock_crypto_pairs`: Cryptocurrency market data
- `mock_forex_pairs`: Forex pair data
- `mock_llm_client`: AI client with mocked responses
- `performance_monitor`: Performance measurement utilities
- `error_simulator`: Error condition simulation

### Mock Data (`tests/mocks/`)
Centralized mock data generation:

- `MockDataGenerator`: Realistic test data generation
- Trade history with proper win rates and PnL patterns
- Market data with realistic price movements
- News articles with sentiment and relevance scoring
- Kalshi markets with proper pricing
- Performance metrics and bot status data

## 📊 Test Reporting

### HTML Reports
Comprehensive HTML reports with:
- Overall test summary with visual status indicators
- Category-wise results breakdown
- Test execution timeline
- Coverage information
- Performance metrics
- Environment details

### JSON Reports
Machine-readable reports for CI/CD integration:
- Detailed test results per category
- Coverage percentages
- Performance benchmarks
- Error details and stack traces

### Coverage Reports
- Line-by-line coverage highlighting
- Branch coverage analysis
- Missing coverage identification
- Coverage trend tracking

## 🎯 Testing Best Practices

### 1. Test Isolation
- Each test should be independent
- Use fixtures for setup and teardown
- Mock external dependencies
- Clean up resources after tests

### 2. Realistic Test Data
- Use `MockDataGenerator` for consistent test data
- Simulate real market conditions
- Include edge cases in test datasets
- Maintain data consistency across tests

### 3. Performance Awareness
- Set appropriate timeouts for different test types
- Monitor resource usage in performance tests
- Use markers to categorize slow tests
- Optimize test execution time

### 4. Error Testing
- Test both success and failure paths
- Simulate realistic error conditions
- Verify error recovery mechanisms
- Test system behavior under stress

### 5. Maintainability
- Use descriptive test names
- Document test purposes and expectations
- Keep tests simple and focused
- Regular review and cleanup of obsolete tests

## 🔧 Advanced Usage

### Custom Test Markers
```bash
# Run only AI-related tests
pytest -m ai

# Run tests requiring network (skip in offline mode)
pytest -m "not network"

# Run database tests only
pytest -m database
```

### Parallel Test Execution
```bash
# Auto-detect CPU count for parallel execution
pytest -n auto

# Specify number of parallel workers
pytest -n 4
```

### Debugging Failed Tests
```bash
# Stop on first failure with detailed output
pytest -x -vs

# Run only previously failed tests
pytest --lf

# Drop into debugger on failures
pytest --pdb
```

### Coverage Analysis
```bash
# Generate coverage with missing lines
pytest --cov=bots --cov-report=term-missing

# HTML coverage report
pytest --cov=bots --cov-report=html

# Fail if coverage below threshold
pytest --cov=bots --cov-fail-under=80
```

## 🚨 Continuous Integration

### GitHub Actions Integration
```yaml
# .github/workflows/tests.yml example
- name: Run Tests
  run: python run_tests.py --ci --coverage --html

- name: Upload Coverage
  uses: codecov/codecov-action@v1

- name: Archive Test Reports
  uses: actions/upload-artifact@v2
  with:
    name: test-reports
    path: test_reports/
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: run-tests
      name: Run Fast Tests
      entry: python run_tests.py --fast
      language: system
      pass_filenames: false
```

## 📈 Performance Benchmarks

### Expected Performance Targets
- **Unit Test Suite**: < 2 minutes
- **Integration Tests**: < 5 minutes
- **Individual Bot Execution**: < 15 seconds
- **LLM Query Response**: < 5 seconds
- **News Processing**: < 100ms per article
- **Market Scanning**: < 10 seconds per category

### Memory Usage Limits
- **Test Suite Total**: < 512MB
- **Individual Test**: < 50MB increase
- **Bot Instances**: < 100MB per bot

## 🆘 Troubleshooting

### Common Issues

**ImportError: No module named 'bots'**
```bash
# Ensure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Tests hanging or timing out**
```bash
# Run with verbose output to identify hanging test
pytest -v --timeout=30
```

**Database connection errors**
```bash
# Clear any stale database connections
rm -f test_reports/test_*.db
```

**Mock data issues**
```bash
# Regenerate mock data with fixed seed
python -c "from tests.mocks.mock_data import MockDataGenerator; MockDataGenerator(42)"
```

### Performance Issues
- Use `--fast` mode for quick feedback
- Skip performance tests during development: `pytest -m "not performance"`
- Run performance tests separately: `python run_tests.py --performance`

### Coverage Issues
- Ensure all source directories are included in coverage config
- Check for file path issues in coverage reports
- Use `--cov-report=term-missing` to identify uncovered lines

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/) for async test support
- [pytest-cov](https://pytest-cov.readthedocs.io/) for coverage reporting
- [pytest-xdist](https://pytest-xdist.readthedocs.io/) for parallel execution

---

**Last Updated**: January 2026
**Test Suite Version**: 1.0
**Python Requirements**: 3.9+