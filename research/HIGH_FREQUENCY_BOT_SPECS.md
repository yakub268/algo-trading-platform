# HIGH-FREQUENCY BOT IMPLEMENTATION PLAN
## Desktop ↔ Code Coordination Document
### January 29, 2026

---

# OVERVIEW

**Goal:** Build 4 new high-frequency bots to achieve 200 trades in 2-3 weeks

| Bot | Platform | Expected Trades/Week | Priority |
|-----|----------|---------------------|----------|
| Kalshi Hourly Crypto | Kalshi | 90-120 | 🔴 P1 |
| Alpaca Crypto RSI | Alpaca | 35-50 | 🟠 P2 |
| OANDA Forex Session | OANDA | 50-75 | 🟡 P3 |
| Kalshi S&P Hourly | Kalshi | 35 | 🟢 P4 |

**Combined Target:** 200+ trades/week (vs current ~30-50)

---

# BOT 1: KALSHI HOURLY CRYPTO PREDICTOR
## Priority: 🔴 HIGHEST (Build First)

### Overview
Trade Kalshi's hourly crypto markets (KXBTCD, KXETHD) that resolve every hour, 24/7.

### Market Details
- **Ticker:** KXBTCD (Bitcoin), KXETHD (Ethereum)
- **Resolution:** Every hour on the hour
- **Contract Type:** Binary (Yes/No) - Will price be above X at resolution?
- **Trading Hours:** 24/7/365
- **Potential:** 24 trades/day × 7 days = **168 opportunities/week**

### Strategy Logic
```
ENTRY RULES:
1. Check current BTC/ETH price vs contract strike prices
2. Calculate momentum (last 1-4 hours trend)
3. Check if price is significantly above/below nearest strike
4. If momentum aligns with price position → Enter

SIGNALS:
- BULLISH: Price trending up + currently above strike → Buy YES
- BEARISH: Price trending down + currently below strike → Buy NO

POSITION SIZING:
- Risk 2% of Kalshi capital per trade ($4 on $200)
- Use limit orders ONLY (0% maker fee vs 1-7% taker)

EXIT RULES:
- Hold to resolution (hourly) OR
- Exit early if 50%+ profit available OR
- Exit if momentum reverses strongly
```

### Technical Requirements
```python
# File: bots/kalshi_hourly_crypto.py

class KalshiHourlyCryptoBot:
    def __init__(self):
        self.markets = ['KXBTCD', 'KXETHD']  # Bitcoin, Ethereum hourly
        self.capital = 200  # Allocated capital
        self.risk_per_trade = 0.02  # 2%
        self.min_edge = 0.10  # Minimum 10% edge to enter
        
    def get_hourly_markets(self):
        """Fetch current hourly markets from Kalshi API"""
        # Get markets expiring in next 1-2 hours
        pass
    
    def calculate_momentum(self, symbol, lookback_hours=4):
        """Calculate price momentum from exchange data"""
        # Use Alpaca/Coinbase for real-time crypto prices
        pass
    
    def estimate_probability(self, current_price, strike_price, momentum):
        """Estimate probability price will be above/below strike"""
        # Simple model: distance from strike + momentum direction
        pass
    
    def find_edge(self, market_price, estimated_prob):
        """Calculate edge: our estimate vs market price"""
        # Edge = estimated_prob - market_price (for YES)
        # Edge = (1 - estimated_prob) - (1 - market_price) (for NO)
        pass
    
    def generate_signal(self):
        """Main signal generation"""
        signals = []
        for market in self.get_hourly_markets():
            edge = self.find_edge(...)
            if abs(edge) > self.min_edge:
                signals.append({
                    'market': market,
                    'side': 'YES' if edge > 0 else 'NO',
                    'edge': edge,
                    'size': self.calculate_position_size(edge)
                })
        return signals
```

### API Endpoints Needed
```
GET /trade-api/v2/markets?ticker=KXBTCD  # Get Bitcoin hourly markets
GET /trade-api/v2/markets?ticker=KXETHD  # Get Ethereum hourly markets
POST /trade-api/v2/portfolio/orders      # Place order (use limit!)
GET /trade-api/v2/portfolio/positions    # Check positions
```

### Data Sources
- **Crypto Prices:** Alpaca API or Coinbase API (already integrated)
- **Kalshi Markets:** Kalshi API (already integrated)

---

# BOT 2: ALPACA CRYPTO RSI SCALPER
## Priority: 🟠 HIGH (No PDT Limit!)

### Overview
Apply the validated RSI strategy to BTC/ETH on Alpaca. Crypto is EXEMPT from PDT rule = unlimited day trades!

### Strategy Logic
```
ENTRY RULES (same as validated stock RSI):
1. RSI(2) < 10 → BUY signal
2. RSI(2) > 90 → SELL signal (or short if enabled)
3. Confirm with RSI(14) direction

POSITION SIZING:
- Risk 2% of Alpaca crypto allocation per trade
- Use market orders for speed (crypto moves fast)

EXIT RULES (your validated exits):
- 2x ATR trailing stop
- Scaled exits at 1R, 2R, 3R
- Time stop: Exit if no profit after 4 hours

FREQUENCY TARGET: 10-20 trades/day on BTC + ETH
```

### Technical Requirements
```python
# File: bots/alpaca_crypto_rsi.py

class AlpacaCryptoRSIBot:
    def __init__(self):
        self.symbols = ['BTC/USD', 'ETH/USD']
        self.timeframe = '5Min'  # 5-minute bars
        self.rsi_period_fast = 2
        self.rsi_period_slow = 14
        self.rsi_oversold = 10
        self.rsi_overbought = 90
        self.capital = 150
        self.risk_per_trade = 0.02
        
    def calculate_rsi(self, prices, period):
        """Calculate RSI"""
        pass
    
    def generate_signal(self):
        """Generate trading signals for crypto"""
        signals = []
        for symbol in self.symbols:
            bars = self.get_crypto_bars(symbol, self.timeframe, limit=100)
            rsi_fast = self.calculate_rsi(bars, self.rsi_period_fast)
            rsi_slow = self.calculate_rsi(bars, self.rsi_period_slow)
            
            if rsi_fast < self.rsi_oversold and rsi_slow < 50:
                signals.append({'symbol': symbol, 'side': 'buy', 'rsi': rsi_fast})
            elif rsi_fast > self.rsi_overbought and rsi_slow > 50:
                signals.append({'symbol': symbol, 'side': 'sell', 'rsi': rsi_fast})
        
        return signals
```

### Key Advantage
- **NO PDT RULE** - Trade as much as you want
- **24/7 Market** - Crypto never closes
- **Uses validated strategy** - Same RSI logic that produced 56.3% win rate

---

# BOT 3: OANDA FOREX SESSION BOT
## Priority: 🟡 MEDIUM

### Overview
Scalp EUR/USD and USD/JPY during high-liquidity sessions when spreads are tightest.

### Session Schedule
| Session | Time (EST) | Pairs | Spread |
|---------|------------|-------|--------|
| London Open | 3-4 AM | EUR/USD | 0.8-1.0 pip |
| London-NY Overlap | 8 AM-12 PM | EUR/USD | 0.6-1.0 pip |
| NY Session | 12-5 PM | USD/JPY | 1.0-1.5 pip |

### Strategy Logic
```
ENTRY RULES:
1. Only trade during specified sessions
2. RSI(14) < 30 on 5-min chart → BUY
3. RSI(14) > 70 on 5-min chart → SELL
4. Confirm trend on 15-min chart

POSITION SIZING:
- Risk 1% per trade ($1.50 on $150)
- Use micro lots (1,000 units) for precise sizing
- Calculate lots based on pip risk

EXIT RULES:
- Target: 10-15 pips
- Stop: 8-12 pips
- Time stop: Close if flat after 30 minutes

FREQUENCY TARGET: 5-15 trades per session
```

### Technical Requirements
```python
# File: bots/oanda_forex_session.py

class OANDAForexSessionBot:
    def __init__(self):
        self.pairs = ['EUR_USD', 'USD_JPY']
        self.sessions = {
            'london_open': {'start': 3, 'end': 4, 'pairs': ['EUR_USD']},
            'london_ny': {'start': 8, 'end': 12, 'pairs': ['EUR_USD']},
            'ny': {'start': 12, 'end': 17, 'pairs': ['USD_JPY']}
        }
        self.capital = 150
        self.risk_per_trade = 0.01  # 1% for forex (tighter)
        
    def is_session_active(self):
        """Check if we're in a valid trading session"""
        pass
    
    def get_current_spread(self, pair):
        """Get current spread - skip if too wide"""
        pass
    
    def generate_signal(self):
        """Generate forex signals during active sessions"""
        if not self.is_session_active():
            return []
        
        signals = []
        for pair in self.get_session_pairs():
            spread = self.get_current_spread(pair)
            if spread > 1.5:  # Skip if spread too wide
                continue
            
            rsi = self.calculate_rsi(pair, '5min', 14)
            trend = self.get_trend(pair, '15min')
            
            if rsi < 30 and trend == 'up':
                signals.append({'pair': pair, 'side': 'buy', 'rsi': rsi})
            elif rsi > 70 and trend == 'down':
                signals.append({'pair': pair, 'side': 'sell', 'rsi': rsi})
        
        return signals
```

---

# BOT 4: KALSHI S&P HOURLY BOT
## Priority: 🟢 LOWER (Add After Others Working)

### Overview
Trade S&P 500 and NASDAQ hourly price markets during US market hours.

### Market Details
- **Tickers:** INXU (S&P 500), NDQU (NASDAQ)
- **Resolution:** Every hour during market hours (9:30 AM - 4 PM EST)
- **Potential:** 7 resolutions × 2 indices = **14 trades/day**, **70/week**

### Strategy Logic
```
ENTRY RULES:
1. Check SPY/QQQ momentum (last 1-2 hours)
2. Compare to Kalshi contract prices
3. Enter if momentum suggests mispricing
4. Use VIX as volatility filter

FILTERS:
- Skip first hour (too volatile)
- Skip last 30 minutes (low liquidity)
- Skip if VIX > 25 (unpredictable)

POSITION SIZING:
- Same as Kalshi crypto: 2% risk, limit orders only
```

---

# WORK SPLIT: DESKTOP vs CODE

## CLAUDE CODE TASKS (Implementation)

### Immediate (Today):
1. ✅ Create `bots/kalshi_hourly_crypto.py` - Full implementation
2. ✅ Create `bots/alpaca_crypto_rsi.py` - Port RSI strategy to crypto
3. ✅ Update `master_orchestrator.py` to include new bots
4. ✅ Add Telegram alerts for new bot signals

### Next:
5. Create `bots/oanda_forex_session.py`
6. Create `bots/kalshi_sp_hourly.py`
7. Integration testing with paper trading

## CLAUDE DESKTOP TASKS (Research & Specs)

### Immediate (Today):
1. ✅ Research Kalshi hourly market structure - DONE
2. ✅ Define strategy specifications - DONE (this document)
3. ✅ Calculate expected trade frequencies - DONE
4. Monitor simulation results for validation

### Next:
5. Research additional Kalshi markets to add
6. Analyze backtest results when CC provides them
7. Optimize parameters based on paper trading

---

# INTEGRATION WITH EXISTING SYSTEM

## Files to Modify
```
trading_bot/
├── bots/
│   ├── kalshi_hourly_crypto.py    # NEW
│   ├── alpaca_crypto_rsi.py       # NEW
│   ├── oanda_forex_session.py     # NEW
│   └── kalshi_sp_hourly.py        # NEW
├── master_orchestrator.py          # UPDATE - add new bots
├── config/trading_config.py        # UPDATE - add new bot configs
└── utils/trading_alerts.py         # UPDATE - add alert templates
```

## New Bot Registry Entries
```python
# Add to BOT_REGISTRY in master_orchestrator.py

'kalshi_hourly_crypto': {
    'name': 'Kalshi Hourly Crypto',
    'module': 'bots.kalshi_hourly_crypto',
    'class': 'KalshiHourlyCryptoBot',
    'market': 'prediction',
    'enabled': True,
    'frequency': 'hourly',
    'capital_allocation': 200
},
'alpaca_crypto_rsi': {
    'name': 'Alpaca Crypto RSI',
    'module': 'bots.alpaca_crypto_rsi',
    'class': 'AlpacaCryptoRSIBot',
    'market': 'crypto',
    'enabled': True,
    'frequency': '5min',
    'capital_allocation': 150
},
'oanda_forex_session': {
    'name': 'OANDA Forex Session',
    'module': 'bots.oanda_forex_session',
    'class': 'OANDAForexSessionBot',
    'market': 'forex',
    'enabled': True,
    'frequency': '5min',
    'capital_allocation': 150
},
'kalshi_sp_hourly': {
    'name': 'Kalshi S&P Hourly',
    'module': 'bots.kalshi_sp_hourly',
    'class': 'KalshiSPHourlyBot',
    'market': 'prediction',
    'enabled': True,
    'frequency': 'hourly',
    'capital_allocation': 100
}
```

---

# EXPECTED RESULTS

## Trade Frequency After Implementation

| Bot | Daily | Weekly | Monthly |
|-----|-------|--------|---------|
| Kalshi Hourly Crypto | 12-18 | 90-120 | 360-500 |
| Alpaca Crypto RSI | 5-10 | 35-70 | 150-280 |
| OANDA Forex Session | 8-12 | 50-75 | 200-300 |
| Kalshi S&P Hourly | 5-7 | 25-35 | 100-140 |
| **Existing 23 Bots** | 5-10 | 35-70 | 150-280 |
| **TOTAL** | **35-57** | **235-370** | **960-1500** |

## Time to 200 Trades
- **Before:** 5-8 weeks
- **After:** **5-6 days** 🚀

---

# VALIDATION REQUIREMENTS

Before going live with real money:

1. **Paper trade each new bot for 48 hours minimum**
2. **Verify API connections work reliably**
3. **Confirm position sizing is correct**
4. **Test exit logic triggers properly**
5. **Monitor for rate limit issues**

## Success Criteria
- [ ] Each bot generates signals as expected
- [ ] Orders execute without errors
- [ ] Risk limits are respected
- [ ] Telegram alerts fire correctly
- [ ] No duplicate orders

---

# COMMAND FOR CLAUDE CODE

Copy and paste this to CC:

```
BUILD HIGH-FREQUENCY BOTS - Implementation Task

Read the spec document at: research/HIGH_FREQUENCY_BOT_SPECS.md

Priority order:
1. bots/kalshi_hourly_crypto.py - Kalshi BTC/ETH hourly markets
2. bots/alpaca_crypto_rsi.py - RSI strategy on Alpaca crypto (no PDT!)
3. Update master_orchestrator.py with new bots
4. Add Telegram alert templates for new bots

Key requirements:
- Use existing Kalshi API integration
- Use existing Alpaca API integration  
- Apply the validated exit rules (2x ATR trailing, scaled exits)
- Risk 2% per trade
- Limit orders only on Kalshi (0% maker fee)
- Log all signals to paper_trading_master.db

Start with kalshi_hourly_crypto.py - this is highest priority.
Target: 200+ trades per week across all new bots.

GO!
```

---

*Document created by Claude Desktop - January 29, 2026*
*For coordination with Claude Code implementation*
