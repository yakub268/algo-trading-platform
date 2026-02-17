"""
Dual Momentum Strategy - Research Validated

Gary Antonacci's Dual Momentum strategy combining:
- Relative Momentum: Compare asset performance against alternatives
- Absolute Momentum: Compare against risk-free rate (trend filter)

Key Research Findings:
- 270 basis points/year outperformance over buy-and-hold
- Maximum drawdown ~28% vs 56% for buy-and-hold
- Works with $5,000+ capital
- Monthly rebalancing only (low maintenance)
- Negatively correlated with mean reversion (diversification)

Implementation:
- Compare SPY vs EFA (US vs International)
- Buy the better performer if its 12-month return > T-bills
- Otherwise hold AGG (bonds)

Author: Trading Bot Arsenal
Created: January 2026
Research Base: Dual Momentum Investing (Gary Antonacci)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import volume-price divergence filter
try:
    from filters.volume_price_divergence import VolumePriceDivergenceFilter
    DIVERGENCE_FILTER_AVAILABLE = True
except ImportError:
    DIVERGENCE_FILTER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DualMomentum')


class HoldingType(Enum):
    """Current holding type"""
    US_EQUITY = "SPY"       # US stocks
    INTL_EQUITY = "EFA"     # International stocks  
    BONDS = "AGG"           # Bonds (safe haven)
    CASH = "CASH"           # Cash (waiting)


@dataclass
class MomentumSignal:
    """Dual momentum trading signal"""
    timestamp: datetime
    current_holding: HoldingType
    recommended_holding: HoldingType
    action: str  # "hold", "switch", "initial"
    
    # Momentum scores
    spy_momentum: float
    efa_momentum: float
    tbill_rate: float
    
    # Decision factors
    relative_winner: str
    passes_absolute: bool
    confidence: float
    reasoning: str


@dataclass
class MomentumMetrics:
    """Current momentum metrics for all assets"""
    spy_12m_return: float
    efa_12m_return: float
    agg_12m_return: float
    tbill_rate: float
    spy_above_sma: bool
    efa_above_sma: bool
    timestamp: datetime


class DualMomentumStrategy:
    """
    Dual Momentum Strategy Implementation
    
    Rules:
    1. Calculate 12-month returns for SPY (US) and EFA (International)
    2. Compare returns (Relative Momentum)
    3. If winner's return > T-bill rate (Absolute Momentum):
       - Buy the winner
    4. If winner's return < T-bill rate:
       - Hold AGG (bonds)
    5. Rebalance monthly on first trading day
    
    Expected Performance:
    - CAGR: 10-15%
    - Sharpe: 0.8-1.2
    - Max Drawdown: ~28%
    - Win Rate (monthly): ~60%
    """
    
    # Core ETFs
    US_EQUITY = "SPY"      # US stocks (S&P 500)
    INTL_EQUITY = "EFA"    # International developed (EAFE)
    BONDS = "AGG"          # US aggregate bonds
    
    # Parameters
    LOOKBACK_MONTHS = 12   # 12-month momentum
    SMA_PERIOD = 200       # Optional trend filter
    
    # Current T-bill rate (update periodically)
    TBILL_RATE = 0.045     # ~4.5% as of early 2026
    
    def __init__(
        self,
        paper_mode: bool = None,
        db_path: str = None
    ):
        """
        Initialize Dual Momentum strategy.

        Args:
            paper_mode: Paper trading mode (reads from PAPER_MODE env if None)
            db_path: Path to SQLite database
        """
        # Safe default: read from environment, default to PAPER if not set
        if paper_mode is None:
            paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
        # Use mode-specific path
        try:
            from utils.data_paths import get_db_path
            db_path = db_path or get_db_path("dual_momentum.db")
        except ImportError:
            db_path = db_path or "data/dual_momentum.db"
        self.paper_mode = paper_mode
        self.db_path = db_path
        self.current_holding = HoldingType.CASH
        
        # Initialize database
        self._init_database()
        
        # Load current state
        self._load_state()
        
        logger.info(f"DualMomentumStrategy initialized (paper={paper_mode})")
        logger.info(f"Current holding: {self.current_holding.value}")
    
    def _init_database(self):
        """Initialize SQLite database for tracking."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Holdings history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                holding TEXT,
                entry_price REAL,
                spy_momentum REAL,
                efa_momentum REAL,
                reasoning TEXT
            )
        ''')
        
        # Trade history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                action TEXT,
                from_holding TEXT,
                to_holding TEXT,
                price REAL,
                pnl_pct REAL,
                reasoning TEXT
            )
        ''')
        
        # Monthly metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monthly_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                month TEXT UNIQUE,
                spy_return REAL,
                efa_return REAL,
                agg_return REAL,
                holding TEXT,
                portfolio_return REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_state(self):
        """Load current holding state from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT holding FROM holdings 
                ORDER BY timestamp DESC LIMIT 1
            ''')
            row = cursor.fetchone()
            
            if row:
                self.current_holding = HoldingType(row[0])
            
            conn.close()
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
    
    def fetch_data(self, period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Fetch price data for all assets.
        
        Args:
            period: Data period (default 2 years)
            
        Returns:
            Dict of symbol -> DataFrame with OHLCV data
        """
        data = {}
        symbols = [self.US_EQUITY, self.INTL_EQUITY, self.BONDS]
        
        for symbol in symbols:
            try:
                df = yf.download(symbol, period=period, interval="1d", progress=False)
                if len(df) > 0:
                    # Normalize columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0].lower() for col in df.columns]
                    else:
                        df.columns = df.columns.str.lower()
                    data[symbol] = df
                    logger.debug(f"Fetched {len(df)} rows for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        
        return data
    
    def calculate_momentum(self, prices: pd.Series, months: int = 12) -> float:
        """
        Calculate momentum (total return) over specified period.
        
        Args:
            prices: Price series
            months: Lookback period in months
            
        Returns:
            Momentum (percentage return)
        """
        trading_days = months * 21  # Approximate trading days per month
        
        if len(prices) < trading_days:
            logger.warning(f"Insufficient data: {len(prices)} < {trading_days}")
            return 0.0
        
        current_price = prices.iloc[-1]
        past_price = prices.iloc[-trading_days]
        
        momentum = (current_price - past_price) / past_price
        return float(momentum)
    
    def get_tbill_rate(self) -> float:
        """
        Get current T-bill rate.
        
        In production, this would fetch from FRED or similar.
        For now, uses a static rate.
        
        Returns:
            Annualized T-bill rate
        """
        try:
            # Try to fetch 13-week T-bill rate from Yahoo Finance
            tbill = yf.Ticker("^IRX")
            hist = tbill.history(period="5d")
            if len(hist) > 0:
                # IRX is quoted in percentage points
                return float(hist['Close'].iloc[-1] / 100)
        except Exception as e:
            logger.debug(f"Could not fetch T-bill rate: {e}")
        
        return self.TBILL_RATE
    
    def get_metrics(self) -> MomentumMetrics:
        """
        Calculate current momentum metrics for all assets.
        
        Returns:
            MomentumMetrics with all current values
        """
        data = self.fetch_data()
        
        if not all(s in data for s in [self.US_EQUITY, self.INTL_EQUITY, self.BONDS]):
            raise ValueError("Could not fetch data for all assets")
        
        spy_data = data[self.US_EQUITY]
        efa_data = data[self.INTL_EQUITY]
        agg_data = data[self.BONDS]
        
        # Calculate 12-month momentum
        spy_momentum = self.calculate_momentum(spy_data['close'])
        efa_momentum = self.calculate_momentum(efa_data['close'])
        agg_momentum = self.calculate_momentum(agg_data['close'])
        
        # Calculate SMA for trend filter
        spy_sma = spy_data['close'].rolling(self.SMA_PERIOD).mean().iloc[-1]
        efa_sma = efa_data['close'].rolling(self.SMA_PERIOD).mean().iloc[-1]
        
        spy_above_sma = spy_data['close'].iloc[-1] > spy_sma
        efa_above_sma = efa_data['close'].iloc[-1] > efa_sma
        
        return MomentumMetrics(
            spy_12m_return=spy_momentum,
            efa_12m_return=efa_momentum,
            agg_12m_return=agg_momentum,
            tbill_rate=self.get_tbill_rate(),
            spy_above_sma=spy_above_sma,
            efa_above_sma=efa_above_sma,
            timestamp=datetime.now()
        )
    
    def generate_signal(self) -> MomentumSignal:
        """
        Generate trading signal based on dual momentum rules.
        
        Rules:
        1. Compare SPY vs EFA (relative momentum)
        2. Winner must beat T-bill rate (absolute momentum)
        3. If neither beats T-bills, hold bonds
        
        Returns:
            MomentumSignal with recommendation
        """
        metrics = self.get_metrics()
        
        # Step 1: Relative momentum - which equity is stronger?
        if metrics.spy_12m_return > metrics.efa_12m_return:
            relative_winner = self.US_EQUITY
            winner_return = metrics.spy_12m_return
        else:
            relative_winner = self.INTL_EQUITY
            winner_return = metrics.efa_12m_return
        
        # Step 2: Absolute momentum - does winner beat risk-free rate?
        passes_absolute = winner_return > metrics.tbill_rate
        
        # Step 3: Determine recommended holding
        if passes_absolute:
            if relative_winner == self.US_EQUITY:
                recommended = HoldingType.US_EQUITY
            else:
                recommended = HoldingType.INTL_EQUITY
        else:
            recommended = HoldingType.BONDS
        
        # Determine action
        if self.current_holding == HoldingType.CASH:
            action = "initial"
        elif self.current_holding == recommended:
            action = "hold"
        else:
            action = "switch"
        
        # Calculate confidence
        confidence = self._calculate_confidence(metrics, winner_return)
        
        # Build reasoning
        reasoning = self._build_reasoning(
            metrics, relative_winner, passes_absolute, recommended
        )
        
        return MomentumSignal(
            timestamp=metrics.timestamp,
            current_holding=self.current_holding,
            recommended_holding=recommended,
            action=action,
            spy_momentum=metrics.spy_12m_return,
            efa_momentum=metrics.efa_12m_return,
            tbill_rate=metrics.tbill_rate,
            relative_winner=relative_winner,
            passes_absolute=passes_absolute,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _calculate_confidence(
        self,
        metrics: MomentumMetrics,
        winner_return: float
    ) -> float:
        """Calculate confidence in the signal."""
        confidence = 0.5
        
        # Stronger momentum = higher confidence
        if abs(metrics.spy_12m_return - metrics.efa_12m_return) > 0.10:
            confidence += 0.15  # Clear winner
        elif abs(metrics.spy_12m_return - metrics.efa_12m_return) > 0.05:
            confidence += 0.10
        
        # Clear absolute momentum signal
        if winner_return > metrics.tbill_rate + 0.10:
            confidence += 0.15  # Strongly beats T-bills
        elif winner_return > metrics.tbill_rate + 0.05:
            confidence += 0.10
        elif winner_return < metrics.tbill_rate - 0.05:
            confidence += 0.10  # Clearly below T-bills (bonds signal)
        
        # Trend alignment (optional confirmation)
        if metrics.spy_12m_return > 0 and metrics.spy_above_sma:
            confidence += 0.05
        if metrics.efa_12m_return > 0 and metrics.efa_above_sma:
            confidence += 0.05
        
        return min(confidence, 0.95)
    
    def _build_reasoning(
        self,
        metrics: MomentumMetrics,
        relative_winner: str,
        passes_absolute: bool,
        recommended: HoldingType
    ) -> str:
        """Build human-readable reasoning for the signal."""
        parts = []
        
        # Relative momentum
        spy_str = f"SPY: {metrics.spy_12m_return:+.1%}"
        efa_str = f"EFA: {metrics.efa_12m_return:+.1%}"
        parts.append(f"12M Returns: {spy_str}, {efa_str}")
        parts.append(f"Relative winner: {relative_winner}")
        
        # Absolute momentum
        winner_return = metrics.spy_12m_return if relative_winner == self.US_EQUITY else metrics.efa_12m_return
        tbill_str = f"T-bill: {metrics.tbill_rate:.1%}"
        if passes_absolute:
            parts.append(f"Passes absolute momentum ({winner_return:.1%} > {tbill_str})")
        else:
            parts.append(f"Fails absolute momentum ({winner_return:.1%} < {tbill_str})")
        
        # Recommendation
        parts.append(f"Recommendation: Hold {recommended.value}")
        
        return " | ".join(parts)
    
    def execute_rebalance(self, signal: MomentumSignal) -> Dict:
        """
        Execute rebalancing based on signal.
        
        Args:
            signal: MomentumSignal with recommendation
            
        Returns:
            Execution result
        """
        if signal.action == "hold":
            logger.info(f"No change needed - holding {self.current_holding.value}")
            return {
                'action': 'hold',
                'holding': self.current_holding.value,
                'reasoning': signal.reasoning
            }
        
        # Record the switch
        old_holding = self.current_holding
        new_holding = signal.recommended_holding
        
        # Get current price for the new holding
        try:
            ticker = yf.Ticker(new_holding.value)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
        except Exception as e:
            logger.debug(f"Error fetching price for {new_holding.value}: {e}")
            current_price = 0.0
        
        # Update state
        self.current_holding = new_holding
        
        # Log to database
        self._log_trade(signal, old_holding, new_holding, current_price)
        
        result = {
            'action': signal.action,
            'from_holding': old_holding.value,
            'to_holding': new_holding.value,
            'price': current_price,
            'reasoning': signal.reasoning,
            'paper_mode': self.paper_mode
        }
        
        logger.info(f"Rebalanced: {old_holding.value} -> {new_holding.value}")
        
        return result
    
    def _log_trade(
        self,
        signal: MomentumSignal,
        old_holding: HoldingType,
        new_holding: HoldingType,
        price: float
    ):
        """Log trade to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Log to trades
        cursor.execute('''
            INSERT INTO trades (timestamp, action, from_holding, to_holding, price, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            signal.timestamp.isoformat(),
            signal.action,
            old_holding.value,
            new_holding.value,
            price,
            signal.reasoning
        ))
        
        # Update holdings
        cursor.execute('''
            INSERT INTO holdings (timestamp, holding, entry_price, spy_momentum, efa_momentum, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            signal.timestamp.isoformat(),
            new_holding.value,
            price,
            signal.spy_momentum,
            signal.efa_momentum,
            signal.reasoning
        ))
        
        conn.commit()
        conn.close()
    
    def is_rebalance_day(self) -> bool:
        """
        Check if today is a rebalance day.
        
        Rebalance on first trading day of each month.
        
        Returns:
            True if should rebalance today
        """
        today = datetime.now()
        
        # Check if first trading day of month
        # For simplicity, rebalance on days 1-3 of each month
        if today.day <= 3:
            # Check if market is open (not weekend)
            if today.weekday() < 5:
                return True
        
        return False
    
    def run_monthly_check(self) -> Optional[Dict]:
        """
        Run monthly rebalancing check.
        
        Returns:
            Execution result if rebalanced, None otherwise
        """
        if not self.is_rebalance_day():
            logger.debug("Not a rebalance day")
            return None
        
        logger.info("Running monthly momentum check...")
        
        signal = self.generate_signal()
        
        logger.info(f"Signal: {signal.action} - {signal.reasoning}")
        logger.info(f"Confidence: {signal.confidence:.0%}")
        
        if signal.action != "hold":
            return self.execute_rebalance(signal)
        
        return {
            'action': 'hold',
            'holding': self.current_holding.value,
            'reasoning': signal.reasoning
        }
    
    def get_status(self) -> Dict:
        """Get current strategy status."""
        try:
            metrics = self.get_metrics()
            signal = self.generate_signal()
            
            return {
                'current_holding': self.current_holding.value,
                'recommended': signal.recommended_holding.value,
                'action_needed': signal.action != "hold",
                'metrics': {
                    'spy_12m_return': metrics.spy_12m_return,
                    'efa_12m_return': metrics.efa_12m_return,
                    'agg_12m_return': metrics.agg_12m_return,
                    'tbill_rate': metrics.tbill_rate
                },
                'confidence': signal.confidence,
                'reasoning': signal.reasoning,
                'paper_mode': self.paper_mode,
                'is_rebalance_day': self.is_rebalance_day()
            }
        except Exception as e:
            return {
                'current_holding': self.current_holding.value,
                'error': str(e),
                'paper_mode': self.paper_mode
            }
    
    def backtest(
        self,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        initial_capital: float = 10000.0
    ) -> Dict:
        """
        Backtest dual momentum strategy.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date (default: today)
            initial_capital: Starting capital
            
        Returns:
            Backtest results
        """
        logger.info(f"Running backtest from {start_date}...")
        
        # Fetch all data
        symbols = [self.US_EQUITY, self.INTL_EQUITY, self.BONDS]
        data = {}
        
        for symbol in symbols:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = df.columns.str.lower()
            data[symbol] = df
        
        # Align all data
        dates = data[self.US_EQUITY].index
        
        # Initialize tracking
        capital = initial_capital
        holding = HoldingType.BONDS
        trades = []
        monthly_returns = []
        
        # Process month by month
        for i in range(252, len(dates)):  # Start after 1 year of data
            current_date = dates[i]
            
            # Check if first day of month
            if i > 0 and current_date.month != dates[i-1].month:
                # Calculate 12-month momentum
                lookback = 252  # Trading days in a year
                
                spy_now = data[self.US_EQUITY]['close'].iloc[i]
                spy_past = data[self.US_EQUITY]['close'].iloc[i - lookback]
                spy_momentum = (spy_now - spy_past) / spy_past
                
                efa_now = data[self.INTL_EQUITY]['close'].iloc[i]
                efa_past = data[self.INTL_EQUITY]['close'].iloc[i - lookback]
                efa_momentum = (efa_now - efa_past) / efa_past
                
                # Determine winner
                if spy_momentum > efa_momentum:
                    relative_winner = HoldingType.US_EQUITY
                    winner_momentum = spy_momentum
                else:
                    relative_winner = HoldingType.INTL_EQUITY
                    winner_momentum = efa_momentum
                
                # Absolute momentum check (assume 3% risk-free)
                tbill_annual = 0.03
                
                if winner_momentum > tbill_annual:
                    new_holding = relative_winner
                else:
                    new_holding = HoldingType.BONDS
                
                # Execute switch if needed
                if new_holding != holding:
                    trades.append({
                        'date': current_date,
                        'from': holding.value,
                        'to': new_holding.value,
                        'spy_momentum': spy_momentum,
                        'efa_momentum': efa_momentum
                    })
                    holding = new_holding
            
            # Calculate daily return for current holding
            if holding == HoldingType.US_EQUITY:
                prices = data[self.US_EQUITY]['close']
            elif holding == HoldingType.INTL_EQUITY:
                prices = data[self.INTL_EQUITY]['close']
            else:
                prices = data[self.BONDS]['close']
            
            if i > 0:
                daily_return = (prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]
                capital *= (1 + daily_return)
        
        # Calculate statistics
        total_return = (capital - initial_capital) / initial_capital
        years = len(dates) / 252
        cagr = (capital / initial_capital) ** (1 / years) - 1
        
        # Calculate buy-and-hold comparison
        spy_return = (data[self.US_EQUITY]['close'].iloc[-1] / data[self.US_EQUITY]['close'].iloc[252] - 1)
        
        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'cagr': cagr,
            'years': years,
            'total_trades': len(trades),
            'spy_buy_hold_return': spy_return,
            'outperformance': total_return - spy_return,
            'trades': trades[-10:]  # Last 10 trades
        }


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DUAL MOMENTUM STRATEGY TEST")
    print("=" * 60)
    
    # Initialize strategy
    strategy = DualMomentumStrategy(paper_mode=True)
    
    # Get current status
    status = strategy.get_status()
    print(f"\n📊 Current Status:")
    print(f"   Holding: {status['current_holding']}")
    print(f"   Recommended: {status.get('recommended', 'N/A')}")
    
    if 'metrics' in status:
        print(f"\n📈 Momentum Metrics:")
        print(f"   SPY 12M Return: {status['metrics']['spy_12m_return']:+.1%}")
        print(f"   EFA 12M Return: {status['metrics']['efa_12m_return']:+.1%}")
        print(f"   AGG 12M Return: {status['metrics']['agg_12m_return']:+.1%}")
        print(f"   T-bill Rate: {status['metrics']['tbill_rate']:.1%}")
    
    if 'reasoning' in status:
        print(f"\n💡 Reasoning: {status['reasoning']}")
    
    # Generate signal
    print("\n" + "-" * 40)
    signal = strategy.generate_signal()
    print(f"📡 Signal Generated:")
    print(f"   Action: {signal.action}")
    print(f"   Current: {signal.current_holding.value}")
    print(f"   Recommended: {signal.recommended_holding.value}")
    print(f"   Confidence: {signal.confidence:.0%}")
    
    # Run backtest
    print("\n" + "=" * 60)
    print("BACKTEST")
    print("=" * 60)
    
    try:
        results = strategy.backtest(start_date="2015-01-01")
        print(f"\n📊 Backtest Results (2015-present):")
        print(f"   Initial: ${results['initial_capital']:,.0f}")
        print(f"   Final: ${results['final_capital']:,.0f}")
        print(f"   Total Return: {results['total_return']:+.1%}")
        print(f"   CAGR: {results['cagr']:+.1%}")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   SPY Buy&Hold: {results['spy_buy_hold_return']:+.1%}")
        print(f"   Outperformance: {results['outperformance']:+.1%}")
    except Exception as e:
        print(f"Backtest error: {e}")
    
    print("\n" + "=" * 60)
