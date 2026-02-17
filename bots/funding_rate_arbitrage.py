"""
Crypto Funding Rate Arbitrage Scanner

Delta-neutral strategy: Long spot + Short perpetual futures
Collects funding payments every 8 hours when rates are positive.

Strategy Overview:
- When longs pay shorts (positive funding), we go long spot and short perp
- Position is delta-neutral: market direction doesn't matter
- Profit comes from funding payments, not price movement

Expected APR: 10-30% depending on market conditions
Risk: Exchange counterparty, funding can turn negative, spread widening

Exchanges Supported:
- Binance Futures (primary)
- Can expand to OKX, Bybit, Gate.io

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
import sqlite3
import requests
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FundingArb')


@dataclass
class FundingOpportunity:
    """Represents a funding rate arbitrage opportunity"""
    symbol: str
    base_asset: str           # BTC, ETH, etc.
    exchange: str
    funding_rate: float       # Per 8-hour period
    funding_rate_annual: float  # Annualized
    mark_price: float
    index_price: float
    premium: float            # (mark - index) / index
    next_funding_time: datetime
    time_to_funding_hours: float
    estimated_daily_profit: float  # Per $1000 position
    confidence: float
    volume_24h: float
    open_interest: float
    timestamp: datetime


@dataclass 
class ArbitragePosition:
    """Tracks an open arbitrage position"""
    symbol: str
    exchange: str
    spot_entry_price: float
    spot_quantity: float
    perp_entry_price: float
    perp_quantity: float
    entry_time: datetime
    total_funding_collected: float
    funding_events: int
    status: str  # 'open', 'closing', 'closed'
    pnl: float


class FundingRateScanner:
    """
    Scans crypto exchanges for funding rate arbitrage opportunities.
    
    Key Features:
    - Real-time funding rate monitoring
    - Historical rate analysis for stability
    - Position sizing recommendations
    - APR calculations
    - Paper trading support
    
    Usage:
        scanner = FundingRateScanner(paper_mode=True)
        opportunities = scanner.scan_all()
        
        # Get top opportunities
        top = scanner.get_top_opportunities(n=5)
    """
    
    # Binance Futures API endpoints
    BINANCE_BASE = "https://fapi.binance.com"
    BINANCE_FUNDING = "/fapi/v1/fundingRate"
    BINANCE_PREMIUM = "/fapi/v1/premiumIndex"
    BINANCE_TICKER = "/fapi/v1/ticker/24hr"
    BINANCE_MARK_PRICE = "/fapi/v1/premiumIndex"
    
    # Thresholds
    MIN_FUNDING_RATE = 0.0001      # 0.01% per 8hr = ~10.9% APR
    MIN_ANNUAL_RATE = 0.10         # 10% APR minimum
    MAX_SPREAD = 0.003             # 0.3% max entry spread
    MIN_VOLUME_24H = 10_000_000    # $10M minimum 24h volume
    MIN_OPEN_INTEREST = 5_000_000  # $5M minimum open interest
    
    # Watchlist - top perpetual pairs
    WATCHLIST = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
        'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'BNBUSDT', 'DOTUSDT',
        'MATICUSDT', 'LTCUSDT', 'ATOMUSDT', 'UNIUSDT', 'NEARUSDT',
        'APTUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'INJUSDT',
        'SEIUSDT', 'TIAUSDT', 'WLDUSDT', 'FETUSDT', 'ONDOUSDT'
    ]
    
    def __init__(self, paper_mode: bool = None, db_path: str = None):
        # Safe default: read from environment, default to PAPER if not set
        if paper_mode is None:
            paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
        self.paper_mode = paper_mode
        # Use mode-specific path
        try:
            from utils.data_paths import get_db_path
            self.db_path = db_path or get_db_path("funding_arbitrage.db")
        except ImportError:
            self.db_path = db_path or "data/funding_arbitrage.db"
        self.opportunities: List[FundingOpportunity] = []
        self.positions: Dict[str, ArbitragePosition] = {}

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()
        
        logger.info(f"FundingRateScanner initialized (paper={paper_mode})")
        logger.info(f"  Min funding rate: {self.MIN_FUNDING_RATE:.4%} per 8hr")
        logger.info(f"  Min APR: {self.MIN_ANNUAL_RATE:.1%}")
        logger.info(f"  Watching {len(self.WATCHLIST)} pairs")
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS funding_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                funding_rate REAL,
                funding_rate_annual REAL,
                mark_price REAL,
                index_price REAL,
                premium REAL,
                next_funding_time TEXT,
                estimated_daily_profit REAL,
                confidence REAL,
                volume_24h REAL,
                open_interest REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS funding_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                funding_rate REAL,
                funding_time TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arb_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                spot_entry REAL,
                spot_qty REAL,
                perp_entry REAL,
                perp_qty REAL,
                entry_time TEXT,
                funding_collected REAL DEFAULT 0,
                funding_events INTEGER DEFAULT 0,
                status TEXT DEFAULT 'open',
                exit_time TEXT,
                pnl REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request to Binance"""
        try:
            url = f"{self.BINANCE_BASE}{endpoint}"
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_premium_index(self) -> List[Dict]:
        """Get premium index (includes funding rate, mark price, index price)"""
        data = self._api_request(self.BINANCE_PREMIUM)
        return data if data else []
    
    def get_24h_tickers(self) -> Dict[str, Dict]:
        """Get 24h trading stats for all symbols"""
        data = self._api_request(self.BINANCE_TICKER)
        if not data:
            return {}
        
        return {item['symbol']: item for item in data}
    
    def get_historical_funding(self, symbol: str, limit: int = 30) -> List[Dict]:
        """Get historical funding rates for stability analysis"""
        params = {'symbol': symbol, 'limit': limit}
        data = self._api_request(self.BINANCE_FUNDING, params)
        return data if data else []
    
    def calculate_annual_rate(self, funding_rate: float) -> float:
        """Convert 8-hour funding rate to annualized rate"""
        # 3 funding periods per day * 365 days
        return funding_rate * 3 * 365
    
    def analyze_funding_stability(self, historical: List[Dict]) -> Tuple[float, float, float]:
        """
        Analyze historical funding for stability and consistency.
        
        Returns:
            (avg_rate, consistency_score, volatility)
        """
        if not historical:
            return 0.0, 0.0, 0.0
        
        rates = [float(h['fundingRate']) for h in historical]
        
        # Average rate
        avg_rate = sum(rates) / len(rates)
        
        # Consistency: % of periods with positive funding
        positive_periods = sum(1 for r in rates if r > 0)
        consistency = positive_periods / len(rates)
        
        # Volatility: standard deviation
        if len(rates) > 1:
            mean = avg_rate
            variance = sum((r - mean) ** 2 for r in rates) / len(rates)
            volatility = variance ** 0.5
        else:
            volatility = 0.0
        
        return avg_rate, consistency, volatility
    
    def scan_symbol(self, symbol: str, premium_data: Dict, ticker_data: Dict) -> Optional[FundingOpportunity]:
        """Scan a single symbol for arbitrage opportunity"""
        if symbol not in premium_data:
            return None
        
        prem = premium_data[symbol]
        tick = ticker_data.get(symbol, {})
        
        # Extract data
        funding_rate = float(prem.get('lastFundingRate', 0))
        mark_price = float(prem.get('markPrice', 0))
        index_price = float(prem.get('indexPrice', 0))
        
        # Skip if data is missing
        if mark_price == 0 or index_price == 0:
            return None
        
        # Calculate premium
        premium = (mark_price - index_price) / index_price
        
        # Get next funding time
        next_funding_ts = int(prem.get('nextFundingTime', 0))
        if next_funding_ts:
            next_funding_time = datetime.fromtimestamp(next_funding_ts / 1000, tz=timezone.utc)
            time_to_funding = (next_funding_time - datetime.now(timezone.utc)).total_seconds() / 3600
        else:
            next_funding_time = datetime.now(timezone.utc)
            time_to_funding = 8.0
        
        # Get volume and open interest
        volume_24h = float(tick.get('quoteVolume', 0))
        # Note: Open interest requires separate API call - approximating from volume
        open_interest = volume_24h * 0.1  # Rough estimate
        
        # Calculate annualized rate
        annual_rate = self.calculate_annual_rate(funding_rate)
        
        # Calculate estimated daily profit per $1000 position
        daily_profit = funding_rate * 3 * 1000  # 3 funding periods per day * $1000
        
        # Get historical funding for confidence calculation
        historical = self.get_historical_funding(symbol, limit=30)
        avg_rate, consistency, volatility = self.analyze_funding_stability(historical)
        
        # Calculate confidence score (0-1)
        # Based on: consistency, rate stability, volume
        confidence = 0.0
        if consistency >= 0.7:
            confidence += 0.4
        elif consistency >= 0.5:
            confidence += 0.2
        
        if funding_rate > self.MIN_FUNDING_RATE * 2:
            confidence += 0.3
        elif funding_rate > self.MIN_FUNDING_RATE:
            confidence += 0.15
        
        if volume_24h >= self.MIN_VOLUME_24H * 2:
            confidence += 0.3
        elif volume_24h >= self.MIN_VOLUME_24H:
            confidence += 0.15
        
        confidence = min(confidence, 1.0)
        
        # Check if meets criteria
        if (funding_rate >= self.MIN_FUNDING_RATE and 
            annual_rate >= self.MIN_ANNUAL_RATE and
            abs(premium) <= self.MAX_SPREAD and
            volume_24h >= self.MIN_VOLUME_24H):
            
            # Extract base asset (remove USDT)
            base_asset = symbol.replace('USDT', '')
            
            return FundingOpportunity(
                symbol=symbol,
                base_asset=base_asset,
                exchange='binance',
                funding_rate=funding_rate,
                funding_rate_annual=annual_rate,
                mark_price=mark_price,
                index_price=index_price,
                premium=premium,
                next_funding_time=next_funding_time,
                time_to_funding_hours=time_to_funding,
                estimated_daily_profit=daily_profit,
                confidence=confidence,
                volume_24h=volume_24h,
                open_interest=open_interest,
                timestamp=datetime.now(timezone.utc)
            )
        
        return None
    
    def scan_all(self) -> List[FundingOpportunity]:
        """Scan all watchlist symbols for opportunities"""
        logger.info("Scanning for funding rate arbitrage opportunities...")
        
        # Fetch market data
        premium_list = self.get_premium_index()
        tickers = self.get_24h_tickers()
        
        if not premium_list:
            logger.error("Failed to fetch premium index data")
            return []
        
        # Convert to dict for easier lookup
        premium_data = {item['symbol']: item for item in premium_list}
        
        self.opportunities = []
        
        for symbol in self.WATCHLIST:
            opp = self.scan_symbol(symbol, premium_data, tickers)
            if opp:
                self.opportunities.append(opp)
                logger.info(f"  ✓ {symbol}: {opp.funding_rate:.4%}/8hr ({opp.funding_rate_annual:.1%} APR)")
        
        # Sort by APR
        self.opportunities.sort(key=lambda x: x.funding_rate_annual, reverse=True)
        
        # Store in database
        self._store_opportunities(self.opportunities)
        
        logger.info(f"Found {len(self.opportunities)} opportunities meeting criteria")
        
        return self.opportunities
    
    def _store_opportunities(self, opportunities: List[FundingOpportunity]):
        """Store opportunities in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for opp in opportunities:
            cursor.execute("""
                INSERT INTO funding_opportunities
                (timestamp, symbol, exchange, funding_rate, funding_rate_annual,
                 mark_price, index_price, premium, next_funding_time,
                 estimated_daily_profit, confidence, volume_24h, open_interest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                opp.timestamp.isoformat(),
                opp.symbol,
                opp.exchange,
                opp.funding_rate,
                opp.funding_rate_annual,
                opp.mark_price,
                opp.index_price,
                opp.premium,
                opp.next_funding_time.isoformat(),
                opp.estimated_daily_profit,
                opp.confidence,
                opp.volume_24h,
                opp.open_interest
            ))
        
        conn.commit()
        conn.close()
    
    def get_top_opportunities(self, n: int = 5) -> List[FundingOpportunity]:
        """Get top N opportunities by APR"""
        if not self.opportunities:
            self.scan_all()
        
        return self.opportunities[:n]
    
    def calculate_portfolio_allocation(self, capital: float, max_positions: int = 3, 
                                        max_per_position: float = 0.25) -> Dict:
        """
        Calculate recommended portfolio allocation.
        
        Args:
            capital: Total capital available
            max_positions: Maximum number of positions
            max_per_position: Maximum allocation per position (0.25 = 25%)
        
        Returns:
            Dict with allocation recommendations
        """
        opportunities = self.get_top_opportunities(max_positions)
        
        if not opportunities:
            return {'error': 'No opportunities available'}
        
        allocations = []
        position_size = capital * max_per_position
        total_deployed = 0
        total_expected_daily = 0
        
        for opp in opportunities:
            # For delta-neutral, need to split between spot and perp
            spot_allocation = position_size / 2
            perp_allocation = position_size / 2
            
            daily_return = (opp.funding_rate * 3) * spot_allocation
            
            allocations.append({
                'symbol': opp.symbol,
                'base_asset': opp.base_asset,
                'spot_allocation': spot_allocation,
                'perp_allocation': perp_allocation,
                'total_allocation': position_size,
                'funding_rate': opp.funding_rate,
                'apr': opp.funding_rate_annual,
                'expected_daily': daily_return,
                'confidence': opp.confidence
            })
            
            total_deployed += position_size
            total_expected_daily += daily_return
        
        portfolio_apr = (total_expected_daily * 365) / total_deployed if total_deployed > 0 else 0
        
        return {
            'capital': capital,
            'deployed': total_deployed,
            'cash_reserve': capital - total_deployed,
            'positions': allocations,
            'portfolio_apr': portfolio_apr,
            'expected_daily': total_expected_daily,
            'expected_monthly': total_expected_daily * 30,
            'expected_annual': total_expected_daily * 365,
            'num_positions': len(allocations)
        }
    
    def get_dashboard_data(self) -> Dict:
        """Get data formatted for dashboard display"""
        opportunities = self.scan_all()
        
        # Calculate portfolio estimate for $5000
        allocation = self.calculate_portfolio_allocation(5000)
        
        return {
            'strategy_name': 'Funding Rate Arbitrage',
            'exchange': 'Binance',
            'last_scan': datetime.now(timezone.utc).isoformat(),
            'opportunities_found': len(opportunities),
            'top_opportunities': [
                {
                    'symbol': o.symbol,
                    'base_asset': o.base_asset,
                    'funding_rate': f"{o.funding_rate:.4%}",
                    'apr': f"{o.funding_rate_annual:.1%}",
                    'premium': f"{o.premium:.3%}",
                    'time_to_funding': f"{o.time_to_funding_hours:.1f}h",
                    'daily_profit_per_1k': f"${o.estimated_daily_profit:.2f}",
                    'confidence': f"{o.confidence:.0%}"
                }
                for o in opportunities[:10]
            ],
            'portfolio_estimate': allocation,
            'thresholds': {
                'min_funding': f"{self.MIN_FUNDING_RATE:.4%}",
                'min_apr': f"{self.MIN_ANNUAL_RATE:.1%}",
                'max_spread': f"{self.MAX_SPREAD:.3%}"
            }
        }


# Simulated data for when API is unavailable
class SimulatedFundingData:
    """Provides simulated data for testing/demo"""
    
    SIMULATED_RATES = {
        'BTCUSDT': 0.00020,
        'ETHUSDT': 0.00015,
        'SOLUSDT': 0.00035,
        'DOGEUSDT': 0.00045,
        'XRPUSDT': 0.00012,
        'ADAUSDT': 0.00010,
        'LINKUSDT': 0.00025,
        'AVAXUSDT': 0.00018,
    }
    
    SIMULATED_PRICES = {
        'BTCUSDT': 97500,
        'ETHUSDT': 3250,
        'SOLUSDT': 225,
        'DOGEUSDT': 0.35,
        'XRPUSDT': 3.10,
        'ADAUSDT': 0.95,
        'LINKUSDT': 25.50,
        'AVAXUSDT': 42.00,
    }
    
    @classmethod
    def get_simulated_opportunities(cls) -> List[FundingOpportunity]:
        """Generate simulated opportunities for testing"""
        opportunities = []
        now = datetime.now(timezone.utc)
        next_funding = now + timedelta(hours=4)
        
        for symbol, rate in cls.SIMULATED_RATES.items():
            if rate >= 0.0001:  # Only include viable opportunities
                price = cls.SIMULATED_PRICES.get(symbol, 100)
                annual = rate * 3 * 365
                
                opportunities.append(FundingOpportunity(
                    symbol=symbol,
                    base_asset=symbol.replace('USDT', ''),
                    exchange='binance_sim',
                    funding_rate=rate,
                    funding_rate_annual=annual,
                    mark_price=price,
                    index_price=price * 0.999,
                    premium=0.001,
                    next_funding_time=next_funding,
                    time_to_funding_hours=4.0,
                    estimated_daily_profit=rate * 3 * 1000,
                    confidence=0.75,
                    volume_24h=100_000_000,
                    open_interest=50_000_000,
                    timestamp=now
                ))
        
        return sorted(opportunities, key=lambda x: x.funding_rate_annual, reverse=True)


if __name__ == "__main__":
    print("=" * 60)
    print("FUNDING RATE ARBITRAGE SCANNER - TEST")
    print("=" * 60)
    
    scanner = FundingRateScanner(paper_mode=True)
    
    # Scan for opportunities
    print("\nScanning Binance Futures...")
    opportunities = scanner.scan_all()
    
    if opportunities:
        print("\n" + "=" * 60)
        print("TOP FUNDING RATE OPPORTUNITIES")
        print("=" * 60)
        print(f"{'Symbol':<12} {'Rate/8h':>10} {'APR':>10} {'Premium':>10} {'Next':>8} {'Conf':>8}")
        print("-" * 60)
        
        for opp in opportunities[:10]:
            print(f"{opp.symbol:<12} {opp.funding_rate:>+9.4%} {opp.funding_rate_annual:>+9.1%} "
                  f"{opp.premium:>+9.3%} {opp.time_to_funding_hours:>6.1f}h {opp.confidence:>7.0%}")
    else:
        print("\nNo opportunities found or API unavailable.")
        print("Showing simulated data for demonstration...\n")
        
        sim_opps = SimulatedFundingData.get_simulated_opportunities()
        
        print(f"{'Symbol':<12} {'Rate/8h':>10} {'APR':>10} {'Premium':>10}")
        print("-" * 50)
        
        for opp in sim_opps:
            print(f"{opp.symbol:<12} {opp.funding_rate:>+9.4%} {opp.funding_rate_annual:>+9.1%} "
                  f"{opp.premium:>+9.3%}")
    
    # Portfolio allocation
    print("\n" + "=" * 60)
    print("PORTFOLIO ALLOCATION ($5,000 account)")
    print("=" * 60)
    
    allocation = scanner.calculate_portfolio_allocation(5000)
    
    if 'error' not in allocation:
        print(f"\nTotal Capital: ${allocation['capital']:,.0f}")
        print(f"Deployed: ${allocation['deployed']:,.0f}")
        print(f"Cash Reserve: ${allocation['cash_reserve']:,.0f}")
        print(f"\nPortfolio APR: {allocation['portfolio_apr']:.1%}")
        print(f"Expected Daily: ${allocation['expected_daily']:.2f}")
        print(f"Expected Monthly: ${allocation['expected_monthly']:.2f}")
        print(f"Expected Annual: ${allocation['expected_annual']:.2f}")
        
        print("\nPositions:")
        for pos in allocation['positions']:
            print(f"  {pos['symbol']}: ${pos['total_allocation']:,.0f} @ {pos['apr']:.1%} APR")
    else:
        print(f"Error: {allocation['error']}")
    
    # Dashboard data
    print("\n" + "=" * 60)
    print("DASHBOARD DATA")
    print("=" * 60)
    
    dashboard = scanner.get_dashboard_data()
    print(f"Strategy: {dashboard['strategy_name']}")
    print(f"Exchange: {dashboard['exchange']}")
    print(f"Opportunities Found: {dashboard['opportunities_found']}")
