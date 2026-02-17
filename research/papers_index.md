# Academic Papers Index for Algorithmic Trading

A curated collection of peer-reviewed research with implementation notes.

---

## Momentum & Trend Following

### 1. Returns to Buying Winners and Selling Losers
**Authors:** Jegadeesh, N., & Titman, S.  
**Year:** 1993 (with 2023 follow-up)  
**Journal:** Journal of Finance  
**Key Findings:**
- Stocks that performed well (poorly) over 3-12 months continue to outperform (underperform)
- ~1% monthly excess returns (12% annual)
- 2023 follow-up confirms 0.74-0.89% monthly returns persist internationally

**Implementation Notes:**
- Rank stocks by past 12-month returns
- Go long top decile, short bottom decile
- Rebalance monthly
- Exclude most recent month (short-term reversal)
- Works best with 6-12 month formation periods

**Risk Warning:** Momentum crashes can be severe (2009: ~50% drawdown)

---

### 2. Time Series Momentum
**Authors:** Moskowitz, T., Ooi, Y.H., & Pedersen, L.H.  
**Year:** 2012  
**Journal:** Journal of Financial Economics  
**Key Findings:**
- Sharpe ratio of 1.1 across 58 liquid futures contracts
- Significant in equity indices, currencies, commodities, bonds
- 12-month lookback with volatility-scaling optimal
- Provides "crisis alpha" - performs well during market stress

**Implementation Notes:**
- Signal = sign(12-month return)
- Position size = (target_vol / realized_vol) * signal
- Target volatility: 10-15% annualized
- Rebalance monthly

**Code Snippet:**
```python
lookback = 252  # ~12 months
signal = np.sign(close.pct_change(lookback))
position_size = target_vol / close.pct_change().rolling(20).std() * signal
```

---

## Mean Reversion

### 3. Short-Term Reversal (RSI-2)
**Authors:** Connors, L., & Alvarez, C.  
**Year:** 2009  
**Source:** "High Probability ETF Trading" (Book)  
**Key Findings:**
- RSI(2) < 5 in uptrends shows 75-91% win rate
- Average holding period: 4 days
- Works on liquid ETFs (SPY, QQQ, IWM)

**Implementation Notes:**
- Entry: RSI(2) < 10 AND Close > SMA(200)
- Exit: Close > SMA(5) OR RSI(2) > 90
- Stop loss: 3% from entry
- Position size: Use 2% account risk rule

**Documented Performance:**
- Sharpe: 1.05-1.18
- Annual return: 25-30%
- Max drawdown: 35%

---

### 4. Cumulative RSI Strategy
**Authors:** Various (Quantitativo Research)  
**Year:** 2020  
**Source:** Quantitativo Blog  
**Key Findings:**
- Sum of RSI(2) over 2 days < 10 outperforms single-day RSI
- 30.3% annual return vs 11% buy-and-hold
- Sharpe ratio: 1.18

**Implementation:**
```python
cumulative_rsi = rsi_2.rolling(2).sum()
entry = (cumulative_rsi < 10) & (close > sma_200)
exit = cumulative_rsi > 65
```

---

## Event-Driven

### 5. Post-Earnings Announcement Drift (PEAD)
**Authors:** Bernard, V., & Thomas, J.  
**Year:** 1989, 1990  
**Journal:** Journal of Accounting and Economics  
**Key Findings:**
- Stocks drift 2-6% in direction of earnings surprise over 60 days
- Effect strongest in first 5 days
- Larger for small-caps with low analyst coverage

**Modern Update (2022):**
- Traditional PEAD declining in developed markets
- Text-based PEAD (earnings call analysis) shows 6.15% drift
- Chinese market: 6.78% quarterly excess returns still persist

**Implementation Notes:**
- Surprise = (Actual EPS - Estimate) / |Estimate|
- Entry: Day after announcement if surprise > 5%
- Exit: 5 trading days later
- Filter: Market cap > $1B, volume > 500K

---

### 6. Pre-FOMC Announcement Drift
**Authors:** Lucca, D., & Moench, E.  
**Year:** 2015  
**Journal:** Journal of Finance (NY Fed working paper)  
**Key Findings:**
- 80% of equity risk premium realized in 24 hours before FOMC
- ~50 bps per event (8 events/year = 4% annual)
- Sharpe ratio: 1.14

**Update (2021):**
- Effect diminished post-2016 (Kurov study)
- Still present but smaller (~30 bps per event)
- Works better with 3x leveraged ETFs for small accounts

**Implementation:**
- Enter SPY long 24 hours before FOMC announcement
- Exit at announcement time
- Only trade when VIX < 25

---

## Factor Investing

### 7. Betting Against Beta
**Authors:** Frazzini, A., & Pedersen, L.H.  
**Year:** 2014  
**Journal:** Journal of Financial Economics  
**Key Findings:**
- Low-beta stocks generate higher risk-adjusted returns than CAPM predicts
- "Leverage constraint" explanation: investors overweight high-beta stocks
- Alpha of 6.4% per year on beta-sorted portfolios

**Implementation:**
- Go long low-beta stocks (bottom quintile)
- Go short high-beta stocks (top quintile)
- Leverage low-beta leg to match beta exposure
- ETF proxies: SPLV, USMV (long only)

**Caveat:** Underperformed significantly in 2010-2020 bull market

---

## Market Microstructure & Arbitrage

### 8. Limits of Arbitrage in Prediction Markets
**Authors:** Various (2025 arXiv preprint)  
**Year:** 2025  
**Source:** arXiv  
**Key Findings:**
- $40M profits extracted from Polymarket (Apr 2024-Apr 2025)
- Median profit ~0.60 per dollar during peak (2024 election)
- Binary arbitrage: YES + NO prices ≠ $1.00

**Implementation Notes:**
- Monitor for YES_price + NO_price > 1.01 (sell both)
- Monitor for YES_price + NO_price < 0.99 (buy both)
- Account for fees (typically 2-5%)
- Rapidly compressing - ICE $2B investment signals institutional entry

---

## Volatility

### 9. Bollinger Band Trading
**Authors:** Bollinger, J.  
**Year:** 2001  
**Source:** "Bollinger on Bollinger Bands" (Book)  
**Key Findings:**
- Pure BB strategies underperform (documented "total loss" in some studies)
- BB + Supertrend + ATR stops: 87.65% profit, 54% hit ratio
- Squeeze (low width) precedes volatility expansion

**Implementation:**
- Entry: Width at 6-month low → price breaks upper band with volume
- Exit: Price below middle band
- Best on: SMH semiconductor ETF (1.4% avg/trade, 78% win rate)

---

## Crypto-Specific

### 10. Funding Rate Arbitrage
**Authors:** Various crypto research  
**Year:** 2020-2024  
**Source:** Exchange documentation, independent research  
**Key Findings:**
- Positive funding = longs pay shorts (market bullish)
- Delta-neutral strategy: Long spot + Short perpetual
- APY: 15-50% favorable periods, 15-30% realistic long-term
- Max documented single-period loss: 1.92%

**Implementation:**
- Entry: Funding rate > 0.02% per 8hr (~27% annualized)
- Position: Equal notional long spot / short perp
- Exit: Funding turns negative OR spread > 0.5%
- Risk: Exchange counterparty (use 2-3 exchanges)

**Monitoring:**
- CoinGlass.com
- ArbitrageScanner.io

---

## References by Category

### Must-Read Papers (in order)
1. Jegadeesh & Titman (1993) - Momentum
2. Moskowitz et al. (2012) - Time Series Momentum
3. Bernard & Thomas (1989) - PEAD
4. Frazzini & Pedersen (2014) - Betting Against Beta
5. Lucca & Moench (2015) - Pre-FOMC Drift

### Implementation Resources
- QuantConnect documentation
- VectorBT tutorials
- Backtrader documentation
- CCXT library (crypto)

### Data Sources
- Yahoo Finance (free equity data)
- FRED (economic data)
- Alternative.me (crypto Fear & Greed)
- Earnings Whispers (earnings calendar)

---

*Last updated: January 2026*
*Maintained by: Trading Bot Arsenal*
