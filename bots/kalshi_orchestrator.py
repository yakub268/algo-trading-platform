"""
Kalshi Arbitrage Orchestrator

Master controller that coordinates all Kalshi arbitrage strategies:
- Probability Arbitrage (YES + NO < $1.00)
- Sum-to-One Arbitrage (Multi-outcome events < $1.00)
- Cross-Platform Arbitrage (Kalshi vs ForecastEx)
- Market Making (Spread capture with 0% maker fees)

Features:
- Unified risk management across all strategies
- Single balance/position tracking
- Coordinated execution to prevent conflicts
- Real-time dashboard integration
- Telegram alerts for opportunities

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import time
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Tuple
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient

# Import strategy modules
try:
    from bots.kalshi_probability_arb import KalshiProbabilityArbitrage
    PROB_ARB_AVAILABLE = True
except ImportError:
    PROB_ARB_AVAILABLE = False

try:
    from bots.kalshi_sum_to_one_arb import KalshiSumToOneArbitrage
    SUM_TO_ONE_AVAILABLE = True
except ImportError:
    SUM_TO_ONE_AVAILABLE = False

try:
    from bots.kalshi_cross_platform_arb import KalshiCrossPlatformArbitrage
    CROSS_PLATFORM_AVAILABLE = True
except ImportError:
    CROSS_PLATFORM_AVAILABLE = False

try:
    from bots.kalshi_market_maker import KalshiMarketMaker
    MARKET_MAKER_AVAILABLE = True
except ImportError:
    MARKET_MAKER_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('KalshiOrchestrator')


@dataclass
class StrategyStatus:
    """Status of a single strategy"""
    name: str
    enabled: bool
    running: bool
    last_scan: Optional[datetime]
    opportunities_found: int
    trades_executed: int
    pnl: float
    error: Optional[str] = None


@dataclass
class RiskLimits:
    """Global risk limits"""
    max_daily_loss: float = 50.0
    max_position_value: float = 200.0
    max_single_trade: float = 25.0
    max_concurrent_positions: int = 10
    min_profit_threshold: float = 0.02  # 2%


@dataclass 
class Opportunity:
    """Unified opportunity format"""
    strategy: str
    ticker: str
    title: str
    profit_pct: float
    profit_dollars: float
    confidence: str
    details: Dict
    timestamp: datetime


class KalshiOrchestrator:
    """
    Master controller for all Kalshi arbitrage strategies.
    
    Provides:
    - Unified risk management
    - Coordinated scanning (prevents API rate limit issues)
    - Single source of truth for positions/balance
    - Dashboard integration
    - Alert system
    
    Usage:
        orchestrator = KalshiOrchestrator(paper_mode=True)
        orchestrator.start()  # Begins scanning all strategies
        
        # Get status
        status = orchestrator.get_status()
        
        # Manual execution
        orchestrator.execute_opportunity(opportunity, size=10)
        
        # Stop
        orchestrator.stop()
    """
    
    def __init__(self, paper_mode: bool = True, config: Dict = None):
        """
        Initialize the orchestrator.
        
        Args:
            paper_mode: If True, simulate all trades
            config: Optional configuration overrides
        """
        self.paper_mode = paper_mode
        self.config = config or {}
        
        # Risk limits
        self.risk_limits = RiskLimits(
            max_daily_loss=self.config.get('max_daily_loss', 50.0),
            max_position_value=self.config.get('max_position_value', 200.0),
            max_single_trade=self.config.get('max_single_trade', 25.0),
            max_concurrent_positions=self.config.get('max_concurrent_positions', 10),
            min_profit_threshold=self.config.get('min_profit_threshold', 0.02)
        )
        
        # Initialize Kalshi client
        try:
            self.client = KalshiClient()
            self._connected = True
            logger.info("Kalshi client connected")
        except Exception as e:
            logger.warning(f"Kalshi client not available: {e}")
            self.client = None
            self._connected = False
        
        # Strategy status (must be initialized before _init_strategies)
        self._strategy_status: Dict[str, StrategyStatus] = {}
        
        # Initialize strategies
        self.strategies: Dict[str, Any] = {}
        self._init_strategies()
        
        # State tracking
        self._running = False
        self._scan_thread = None
        self._opportunity_queue: Queue = Queue()
        self._daily_pnl = 0.0
        self._positions: List[Dict] = []
        self._paper_balance = 500.0
        
        # Database
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'kalshi_orchestrator.db'
        )
        self._init_database()
        
        logger.info(f"Orchestrator initialized (paper_mode={paper_mode})")
        logger.info(f"Available strategies: {list(self.strategies.keys())}")
    
    def _init_strategies(self):
        """Initialize all available strategies."""
        
        if PROB_ARB_AVAILABLE:
            try:
                self.strategies['probability_arb'] = KalshiProbabilityArbitrage(
                    paper_mode=self.paper_mode
                )
                self._strategy_status['probability_arb'] = StrategyStatus(
                    name='Probability Arbitrage',
                    enabled=True,
                    running=False,
                    last_scan=None,
                    opportunities_found=0,
                    trades_executed=0,
                    pnl=0.0
                )
                logger.info("✓ Probability Arbitrage loaded")
            except Exception as e:
                logger.error(f"Failed to load Probability Arbitrage: {e}")
        
        if SUM_TO_ONE_AVAILABLE:
            try:
                self.strategies['sum_to_one'] = KalshiSumToOneArbitrage(
                    paper_mode=self.paper_mode
                )
                self._strategy_status['sum_to_one'] = StrategyStatus(
                    name='Sum-to-One Arbitrage',
                    enabled=True,
                    running=False,
                    last_scan=None,
                    opportunities_found=0,
                    trades_executed=0,
                    pnl=0.0
                )
                logger.info("✓ Sum-to-One Arbitrage loaded")
            except Exception as e:
                logger.error(f"Failed to load Sum-to-One Arbitrage: {e}")
        
        if CROSS_PLATFORM_AVAILABLE:
            try:
                self.strategies['cross_platform'] = KalshiCrossPlatformArbitrage(
                    paper_mode=self.paper_mode
                )
                self._strategy_status['cross_platform'] = StrategyStatus(
                    name='Cross-Platform Arbitrage',
                    enabled=True,
                    running=False,
                    last_scan=None,
                    opportunities_found=0,
                    trades_executed=0,
                    pnl=0.0
                )
                logger.info("✓ Cross-Platform Arbitrage loaded")
            except Exception as e:
                logger.error(f"Failed to load Cross-Platform Arbitrage: {e}")
        
        if MARKET_MAKER_AVAILABLE:
            try:
                self.strategies['market_maker'] = KalshiMarketMaker(
                    paper_mode=self.paper_mode
                )
                self._strategy_status['market_maker'] = StrategyStatus(
                    name='Market Maker',
                    enabled=False,  # Disabled by default - requires more capital
                    running=False,
                    last_scan=None,
                    opportunities_found=0,
                    trades_executed=0,
                    pnl=0.0
                )
                logger.info("✓ Market Maker loaded (disabled by default)")
            except Exception as e:
                logger.error(f"Failed to load Market Maker: {e}")
    
    def _init_database(self):
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                strategy TEXT,
                ticker TEXT,
                title TEXT,
                profit_pct REAL,
                profit_dollars REAL,
                confidence TEXT,
                executed INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                strategy TEXT,
                ticker TEXT,
                contracts INTEGER,
                cost REAL,
                expected_profit REAL,
                status TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE,
                total_scans INTEGER,
                opportunities_found INTEGER,
                trades_executed INTEGER,
                total_pnl REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ============== Risk Management ==============
    
    def check_risk_limits(self, trade_cost: float) -> Tuple[bool, str]:
        """
        Check if a trade passes risk limits.
        
        Returns:
            (allowed, reason) tuple
        """
        # Check daily loss limit
        if self._daily_pnl <= -self.risk_limits.max_daily_loss:
            return False, f"Daily loss limit reached (${self.risk_limits.max_daily_loss})"
        
        # Check single trade size
        if trade_cost > self.risk_limits.max_single_trade:
            return False, f"Trade exceeds max single trade (${self.risk_limits.max_single_trade})"
        
        # Check total position value
        current_position_value = sum(p.get('value', 0) for p in self._positions)
        if current_position_value + trade_cost > self.risk_limits.max_position_value:
            return False, f"Would exceed max position value (${self.risk_limits.max_position_value})"
        
        # Check concurrent positions
        if len(self._positions) >= self.risk_limits.max_concurrent_positions:
            return False, f"Max concurrent positions reached ({self.risk_limits.max_concurrent_positions})"
        
        # Check paper balance
        if self.paper_mode and trade_cost > self._paper_balance:
            return False, f"Insufficient paper balance (${self._paper_balance:.2f})"
        
        return True, "OK"
    
    # ============== Scanning ==============
    
    def scan_all_strategies(self) -> List[Opportunity]:
        """
        Run all enabled strategies and collect opportunities.
        
        Returns:
            List of opportunities sorted by profit potential
        """
        all_opportunities = []
        
        for strategy_name, strategy in self.strategies.items():
            status = self._strategy_status.get(strategy_name)
            if not status or not status.enabled:
                continue
            
            try:
                status.running = True
                logger.info(f"Scanning {strategy_name}...")
                
                # Each strategy has a scan method
                if hasattr(strategy, 'scan_all_markets'):
                    opps = strategy.scan_all_markets(
                        min_profit_pct=self.risk_limits.min_profit_threshold
                    )
                elif hasattr(strategy, 'scan_all_events'):
                    opps = strategy.scan_all_events(
                        min_profit_pct=self.risk_limits.min_profit_threshold
                    )
                elif hasattr(strategy, 'scan_opportunities'):
                    opps = strategy.scan_opportunities(
                        min_spread=self.risk_limits.min_profit_threshold
                    )
                else:
                    opps = []
                
                # Convert to unified format
                for opp in opps:
                    unified = self._convert_opportunity(strategy_name, opp)
                    if unified:
                        all_opportunities.append(unified)
                        self._log_opportunity(unified)
                
                status.last_scan = datetime.now(timezone.utc)
                status.opportunities_found += len(opps)
                status.running = False
                status.error = None
                
                logger.info(f"  {strategy_name}: Found {len(opps)} opportunities")
                
            except Exception as e:
                status.running = False
                status.error = str(e)
                logger.error(f"Error scanning {strategy_name}: {e}")
        
        # Sort by profit potential
        all_opportunities.sort(key=lambda x: x.profit_pct, reverse=True)
        
        return all_opportunities
    
    def _convert_opportunity(self, strategy_name: str, opp: Any) -> Optional[Opportunity]:
        """Convert strategy-specific opportunity to unified format."""
        try:
            # Handle different opportunity types
            if hasattr(opp, 'ticker'):
                ticker = opp.ticker
            elif hasattr(opp, 'event_ticker'):
                ticker = opp.event_ticker
            else:
                ticker = 'UNKNOWN'
            
            if hasattr(opp, 'title'):
                title = opp.title
            elif hasattr(opp, 'event_title'):
                title = opp.event_title
            else:
                title = ticker
            
            profit_pct = getattr(opp, 'profit_pct', 0)
            
            # Estimate profit in dollars (assume $10 position)
            profit_dollars = profit_pct * 10
            
            confidence = getattr(opp, 'confidence', 'MEDIUM')
            if isinstance(confidence, (int, float)):
                if confidence > 0.8:
                    confidence = 'HIGH'
                elif confidence > 0.5:
                    confidence = 'MEDIUM'
                else:
                    confidence = 'LOW'
            
            return Opportunity(
                strategy=strategy_name,
                ticker=ticker,
                title=title[:100] if title else ticker,
                profit_pct=profit_pct,
                profit_dollars=profit_dollars,
                confidence=confidence,
                details=asdict(opp) if hasattr(opp, '__dataclass_fields__') else {},
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.warning(f"Failed to convert opportunity: {e}")
            return None
    
    def _log_opportunity(self, opp: Opportunity):
        """Log opportunity to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO opportunities 
                (timestamp, strategy, ticker, title, profit_pct, profit_dollars, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                opp.timestamp.isoformat(),
                opp.strategy,
                opp.ticker,
                opp.title,
                opp.profit_pct,
                opp.profit_dollars,
                opp.confidence
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log opportunity: {e}")
    
    # ============== Execution ==============
    
    def execute_opportunity(self, opp: Opportunity, contracts: int = 10) -> bool:
        """
        Execute an opportunity through the appropriate strategy.
        
        Args:
            opp: The opportunity to execute
            contracts: Number of contracts
            
        Returns:
            True if execution successful
        """
        # Estimate cost
        estimated_cost = contracts * (1 - opp.profit_pct)  # Rough estimate
        
        # Check risk limits
        allowed, reason = self.check_risk_limits(estimated_cost)
        if not allowed:
            logger.warning(f"Trade blocked: {reason}")
            return False
        
        strategy = self.strategies.get(opp.strategy)
        if not strategy:
            logger.error(f"Strategy not found: {opp.strategy}")
            return False
        
        try:
            # Each strategy has an execute method
            if hasattr(strategy, 'execute'):
                # Find the original opportunity object
                result = strategy.execute(opp.details, contracts=contracts)
            else:
                logger.error(f"Strategy {opp.strategy} has no execute method")
                return False
            
            if result:
                # Update stats
                status = self._strategy_status.get(opp.strategy)
                if status:
                    status.trades_executed += 1
                
                # Log execution
                self._log_execution(opp, contracts, estimated_cost)
                
                # Update paper balance
                if self.paper_mode:
                    self._paper_balance -= estimated_cost
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return False
    
    def _log_execution(self, opp: Opportunity, contracts: int, cost: float):
        """Log execution to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO executions 
                (timestamp, strategy, ticker, contracts, cost, expected_profit, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(timezone.utc).isoformat(),
                opp.strategy,
                opp.ticker,
                contracts,
                cost,
                opp.profit_dollars * contracts / 10,
                'executed'
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log execution: {e}")
    
    # ============== Control ==============
    
    def start(self, scan_interval: int = 60):
        """
        Start the orchestrator scanning loop.
        
        Args:
            scan_interval: Seconds between scans
        """
        if self._running:
            logger.warning("Orchestrator already running")
            return
        
        self._running = True
        
        def scan_loop():
            while self._running:
                try:
                    opportunities = self.scan_all_strategies()
                    
                    # Auto-execute high-confidence opportunities
                    for opp in opportunities:
                        if opp.confidence == 'HIGH' and opp.profit_pct >= 0.03:
                            logger.info(f"Auto-executing: {opp.ticker} ({opp.profit_pct:.2%})")
                            self.execute_opportunity(opp, contracts=5)
                    
                    time.sleep(scan_interval)
                    
                except Exception as e:
                    logger.error(f"Scan loop error: {e}")
                    time.sleep(scan_interval * 2)
        
        self._scan_thread = threading.Thread(target=scan_loop, daemon=True)
        self._scan_thread.start()
        
        logger.info(f"Orchestrator started (scan_interval={scan_interval}s)")
    
    def stop(self):
        """Stop the orchestrator."""
        self._running = False
        if self._scan_thread:
            self._scan_thread.join(timeout=5)
        logger.info("Orchestrator stopped")
    
    def enable_strategy(self, strategy_name: str):
        """Enable a strategy."""
        if strategy_name in self._strategy_status:
            self._strategy_status[strategy_name].enabled = True
            logger.info(f"Enabled {strategy_name}")
    
    def disable_strategy(self, strategy_name: str):
        """Disable a strategy."""
        if strategy_name in self._strategy_status:
            self._strategy_status[strategy_name].enabled = False
            logger.info(f"Disabled {strategy_name}")
    
    # ============== Status & Reporting ==============
    
    def get_status(self) -> Dict:
        """Get current orchestrator status."""
        return {
            'running': self._running,
            'connected': self._connected,
            'paper_mode': self.paper_mode,
            'paper_balance': self._paper_balance,
            'daily_pnl': self._daily_pnl,
            'positions': len(self._positions),
            'strategies': {
                name: asdict(status) 
                for name, status in self._strategy_status.items()
            },
            'risk_limits': asdict(self.risk_limits),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_dashboard_data(self) -> Dict:
        """Get data formatted for dashboard display."""
        status = self.get_status()
        
        # Get recent opportunities from database
        recent_opps = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT strategy, ticker, title, profit_pct, confidence, timestamp
                FROM opportunities
                ORDER BY timestamp DESC
                LIMIT 10
            ''')
            for row in cursor.fetchall():
                recent_opps.append({
                    'strategy': row[0],
                    'ticker': row[1],
                    'title': row[2],
                    'profit_pct': row[3],
                    'confidence': row[4],
                    'timestamp': row[5]
                })
            conn.close()
        except Exception as e:
            logger.debug(f"Error fetching recent opportunities: {e}")

        return {
            **status,
            'recent_opportunities': recent_opps,
            'strategy_summary': [
                {
                    'name': s.name,
                    'status': 'LIVE' if s.enabled and s.running else ('PAUSED' if s.enabled else 'DISABLED'),
                    'opportunities': s.opportunities_found,
                    'trades': s.trades_executed,
                    'pnl': s.pnl
                }
                for s in self._strategy_status.values()
            ]
        }


def main():
    """Test the orchestrator."""
    print("=" * 70)
    print("KALSHI ARBITRAGE ORCHESTRATOR")
    print("=" * 70)
    print("""
Master controller for all Kalshi arbitrage strategies:
  • Probability Arbitrage (YES + NO < $1.00)
  • Sum-to-One Arbitrage (Multi-outcome < $1.00)  
  • Cross-Platform Arbitrage (Kalshi vs ForecastEx)
  • Market Making (Spread capture)
""")
    
    # Initialize orchestrator
    orchestrator = KalshiOrchestrator(paper_mode=True)
    
    # Show status
    print("\n" + "=" * 70)
    print("📊 ORCHESTRATOR STATUS")
    print("=" * 70)
    
    status = orchestrator.get_status()
    print(f"\nConnected: {status['connected']}")
    print(f"Paper Mode: {status['paper_mode']}")
    print(f"Paper Balance: ${status['paper_balance']:.2f}")
    print(f"\nStrategies:")
    for name, strat_status in status['strategies'].items():
        enabled = "✓" if strat_status['enabled'] else "✗"
        print(f"  {enabled} {strat_status['name']}")
    
    # Run one scan
    print("\n" + "=" * 70)
    print("🔍 RUNNING SCAN")
    print("=" * 70)
    
    opportunities = orchestrator.scan_all_strategies()
    
    if opportunities:
        print(f"\n🎯 Found {len(opportunities)} total opportunities:\n")
        for opp in opportunities[:10]:
            print(f"  [{opp.strategy}] {opp.ticker}")
            print(f"    {opp.title[:50]}...")
            print(f"    Profit: {opp.profit_pct:.2%} | Confidence: {opp.confidence}")
            print()
    else:
        print("\n❌ No opportunities found above threshold")
    
    # Dashboard data
    print("\n" + "=" * 70)
    print("📈 DASHBOARD DATA")
    print("=" * 70)
    
    dashboard = orchestrator.get_dashboard_data()
    print(f"\nStrategy Summary:")
    for s in dashboard['strategy_summary']:
        print(f"  {s['name']}: {s['status']} | {s['opportunities']} opps | {s['trades']} trades")
    
    print("\n✅ Orchestrator test complete!")
    print("\nTo run continuously:")
    print("  orchestrator.start(scan_interval=60)")


if __name__ == '__main__':
    main()
