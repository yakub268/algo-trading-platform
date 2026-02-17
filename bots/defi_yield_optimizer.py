"""
DeFi Yield Optimizer Bot

Automatically finds and rotates between the best yield opportunities in DeFi.
Monitors lending protocols, liquidity pools, and staking to maximize returns.

Strategy:
1. Scan yield rates across protocols (Aave, Compound, Curve, Convex, etc.)
2. Calculate real APY after fees and impermanent loss risk
3. Automatically move capital to highest-yielding opportunities
4. Compound rewards to maximize returns

Expected APY: 3-15% (varies by market conditions and risk tolerance)
Risk: Smart contract risk, impermanent loss, protocol exploits

Protocols Supported:
- Lending: Aave, Compound, Morpho
- DEX LPs: Uniswap, Curve, Balancer
- Yield Aggregators: Yearn, Convex, Beefy

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
import sqlite3
import json
import time
import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from decimal import Decimal

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DeFiYield')


class Protocol(Enum):
    AAVE = "aave"
    COMPOUND = "compound"
    CURVE = "curve"
    CONVEX = "convex"
    YEARN = "yearn"
    UNISWAP = "uniswap"
    LIDO = "lido"
    ROCKETPOOL = "rocketpool"


class Chain(Enum):
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    BASE = "base"


class YieldType(Enum):
    LENDING = "lending"  # Supply assets to earn interest
    BORROWING = "borrowing"  # Borrow against collateral
    LP = "liquidity_pool"  # Provide liquidity to DEX
    STAKING = "staking"  # Stake tokens for rewards
    VAULT = "vault"  # Yield aggregator vault


@dataclass
class YieldOpportunity:
    """Represents a yield-generating opportunity"""
    protocol: Protocol
    chain: Chain
    yield_type: YieldType
    asset: str  # e.g., "USDC", "ETH", "USDC-ETH LP"
    base_apy: float  # Base yield from protocol
    reward_apy: float  # Additional token rewards
    total_apy: float  # base + rewards
    tvl: float  # Total value locked
    utilization: float  # For lending protocols
    risk_score: int  # 1-10, higher = riskier
    min_deposit: float
    lockup_days: int  # 0 if no lockup
    url: str  # Link to protocol
    timestamp: datetime


@dataclass
class Position:
    """Represents a deposited position"""
    opportunity: YieldOpportunity
    principal: float
    current_value: float
    earned: float
    entry_time: datetime
    last_compound: datetime
    status: str  # 'active', 'withdrawing', 'closed'


@dataclass
class RebalanceAction:
    """Recommended rebalancing action"""
    from_position: Optional[Position]
    to_opportunity: YieldOpportunity
    amount: float
    expected_gain_apy: float  # APY improvement
    reason: str


class DefiLlamaClient:
    """
    Client for DeFi Llama API - comprehensive DeFi data aggregator.
    
    Provides:
    - Protocol TVL data
    - Yield/APY data across protocols
    - Historical rates
    """
    
    BASE_URL = "https://yields.llama.fi"
    
    def __init__(self):
        self.session = None
    
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def get_pools(self) -> List[Dict]:
        """Get all yield pools from DeFi Llama"""
        await self._ensure_session()
        
        try:
            async with self.session.get(
                f"{self.BASE_URL}/pools",
                timeout=30
            ) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                return data.get('data', [])
                
        except Exception as e:
            logger.error(f"DeFi Llama API error: {e}")
            return []
    
    async def close(self):
        if self.session:
            await self.session.close()


class DeFiYieldOptimizer:
    """
    Main DeFi yield optimization bot.
    
    Workflow:
    1. Fetch yield data from DeFi Llama and direct protocol APIs
    2. Filter by risk tolerance and minimum APY
    3. Calculate optimal allocation
    4. Execute rebalancing when beneficial
    
    Key Parameters:
    - MIN_APY: Minimum APY to consider (default 3%)
    - MAX_RISK: Maximum risk score (1-10, default 5)
    - REBALANCE_THRESHOLD: Minimum APY improvement to rebalance (default 1%)
    - CHAINS: Chains to monitor
    
    Usage:
        optimizer = DeFiYieldOptimizer(paper_mode=True)
        opportunities = await optimizer.scan()
        actions = optimizer.get_rebalance_recommendations()
        await optimizer.execute_rebalance(actions)
    """
    
    # Default parameters
    MIN_APY = 0.03  # 3% minimum
    MAX_RISK = 5  # Conservative risk tolerance
    REBALANCE_THRESHOLD = 0.01  # 1% APY improvement needed
    
    # Protocol risk scores (1-10, higher = riskier)
    RISK_SCORES = {
        Protocol.AAVE: 2,
        Protocol.COMPOUND: 2,
        Protocol.LIDO: 3,
        Protocol.CURVE: 4,
        Protocol.CONVEX: 5,
        Protocol.YEARN: 4,
        Protocol.UNISWAP: 6,
        Protocol.ROCKETPOOL: 3,
    }
    
    # Supported stablecoins for yield farming
    STABLECOINS = ['USDC', 'USDT', 'DAI', 'FRAX', 'LUSD']
    
    def __init__(self, 
                 paper_mode: bool = True,
                 chains: List[Chain] = None):
        """
        Initialize DeFi yield optimizer.
        
        Args:
            paper_mode: If True, simulate transactions
            chains: Chains to monitor (default: Ethereum, Arbitrum)
        """
        self.paper_mode = paper_mode
        self.chains = chains or [Chain.ETHEREUM, Chain.ARBITRUM, Chain.POLYGON]
        
        # Data client
        self.llama_client = DefiLlamaClient()
        
        # State
        self._opportunities: List[YieldOpportunity] = []
        self._positions: List[Position] = []
        self._paper_balance = 10000.0  # $10k starting balance
        
        # Database
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'defi_yield.db'
        )
        self._init_database()
        
        logger.info(f"DeFi Yield Optimizer initialized (paper_mode={paper_mode})")
    
    def _init_database(self):
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                protocol TEXT,
                chain TEXT,
                asset TEXT,
                total_apy REAL,
                tvl REAL,
                risk_score INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time TEXT,
                protocol TEXT,
                asset TEXT,
                principal REAL,
                current_value REAL,
                earned REAL,
                status TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rebalances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                from_protocol TEXT,
                to_protocol TEXT,
                amount REAL,
                apy_improvement REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ============== Yield Scanning ==============
    
    async def scan(self) -> List[YieldOpportunity]:
        """
        Scan for yield opportunities across DeFi.
        
        Returns:
            List of opportunities sorted by APY
        """
        logger.info("Scanning for DeFi yield opportunities...")
        
        # Fetch from DeFi Llama
        pools = await self.llama_client.get_pools()
        
        opportunities = []
        
        for pool in pools:
            try:
                opp = self._parse_pool(pool)
                if opp and self._passes_filters(opp):
                    opportunities.append(opp)
                    
            except Exception as e:
                logger.debug(f"Failed to parse pool: {e}")
        
        # Sort by APY
        opportunities.sort(key=lambda x: x.total_apy, reverse=True)
        
        self._opportunities = opportunities
        self._log_opportunities(opportunities[:20])
        
        logger.info(f"Found {len(opportunities)} opportunities above {self.MIN_APY:.0%} APY")
        return opportunities
    
    def _parse_pool(self, pool: Dict) -> Optional[YieldOpportunity]:
        """Parse a DeFi Llama pool into our opportunity format."""
        
        # Map protocol name
        protocol_name = pool.get('project', '').lower()
        protocol = None
        for p in Protocol:
            if p.value in protocol_name:
                protocol = p
                break
        
        if not protocol:
            return None
        
        # Map chain
        chain_name = pool.get('chain', '').lower()
        chain = None
        for c in Chain:
            if c.value in chain_name:
                chain = c
                break
        
        if chain not in self.chains:
            return None
        
        # Get APY
        apy = pool.get('apy', 0) or 0
        if apy < self.MIN_APY * 100:  # API returns percentage
            return None
        
        return YieldOpportunity(
            protocol=protocol,
            chain=chain,
            yield_type=YieldType.LENDING if 'lend' in protocol_name else YieldType.LP,
            asset=pool.get('symbol', 'Unknown'),
            base_apy=pool.get('apyBase', 0) / 100 if pool.get('apyBase') else 0,
            reward_apy=pool.get('apyReward', 0) / 100 if pool.get('apyReward') else 0,
            total_apy=apy / 100,
            tvl=pool.get('tvlUsd', 0),
            utilization=pool.get('utilization', 0) / 100 if pool.get('utilization') else 0,
            risk_score=self.RISK_SCORES.get(protocol, 5),
            min_deposit=0,
            lockup_days=0,
            url=pool.get('url', ''),
            timestamp=datetime.now(timezone.utc)
        )
    
    def _passes_filters(self, opp: YieldOpportunity) -> bool:
        """Check if opportunity passes our filters."""
        if opp.total_apy < self.MIN_APY:
            return False
        
        if opp.risk_score > self.MAX_RISK:
            return False
        
        if opp.tvl < 1_000_000:  # Min $1M TVL
            return False
        
        return True
    
    def _log_opportunities(self, opportunities: List[YieldOpportunity]):
        """Log opportunities to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for opp in opportunities:
                cursor.execute('''
                    INSERT INTO opportunities 
                    (timestamp, protocol, chain, asset, total_apy, tvl, risk_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    opp.timestamp.isoformat(),
                    opp.protocol.value,
                    opp.chain.value,
                    opp.asset,
                    opp.total_apy,
                    opp.tvl,
                    opp.risk_score
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log opportunities: {e}")
    
    # ============== Rebalancing ==============
    
    def get_rebalance_recommendations(self) -> List[RebalanceAction]:
        """
        Get recommendations for rebalancing positions.
        
        Compares current positions to available opportunities
        and recommends moves that would increase yield.
        """
        recommendations = []
        
        if not self._opportunities:
            return recommendations
        
        best_opp = self._opportunities[0]
        
        # Check each position
        for position in self._positions:
            current_apy = position.opportunity.total_apy
            improvement = best_opp.total_apy - current_apy
            
            if improvement > self.REBALANCE_THRESHOLD:
                recommendations.append(RebalanceAction(
                    from_position=position,
                    to_opportunity=best_opp,
                    amount=position.current_value,
                    expected_gain_apy=improvement,
                    reason=f"APY improvement: {current_apy:.1%} → {best_opp.total_apy:.1%}"
                ))
        
        # Recommend deploying idle capital
        idle_capital = self._paper_balance if self.paper_mode else 0
        if idle_capital > 100:
            recommendations.append(RebalanceAction(
                from_position=None,
                to_opportunity=best_opp,
                amount=idle_capital,
                expected_gain_apy=best_opp.total_apy,
                reason=f"Deploy idle capital at {best_opp.total_apy:.1%} APY"
            ))
        
        return recommendations
    
    async def execute_rebalance(self, actions: List[RebalanceAction]) -> List[Position]:
        """
        Execute rebalancing actions.
        
        Args:
            actions: List of rebalance actions
            
        Returns:
            List of new/updated positions
        """
        if self.paper_mode:
            return self._paper_rebalance(actions)
        
        # Real execution would call Web3/protocol APIs
        logger.warning("Real DeFi execution not yet implemented")
        return []
    
    def _paper_rebalance(self, actions: List[RebalanceAction]) -> List[Position]:
        """Simulate rebalancing in paper mode."""
        new_positions = []
        
        for action in actions:
            # Close old position if exists
            if action.from_position:
                action.from_position.status = 'closed'
                self._paper_balance += action.from_position.current_value
            
            # Open new position
            if action.amount > self._paper_balance:
                action.amount = self._paper_balance
            
            if action.amount < 100:
                continue
            
            position = Position(
                opportunity=action.to_opportunity,
                principal=action.amount,
                current_value=action.amount,
                earned=0,
                entry_time=datetime.now(timezone.utc),
                last_compound=datetime.now(timezone.utc),
                status='active'
            )
            
            self._positions.append(position)
            new_positions.append(position)
            self._paper_balance -= action.amount
            
            logger.info(f"""
[PAPER] DeFi Position Opened:
  Protocol: {action.to_opportunity.protocol.value}
  Chain: {action.to_opportunity.chain.value}
  Asset: {action.to_opportunity.asset}
  Amount: ${action.amount:,.2f}
  APY: {action.to_opportunity.total_apy:.2%}
  Risk Score: {action.to_opportunity.risk_score}/10
""")
        
        return new_positions
    
    # ============== Yield Simulation ==============
    
    def simulate_yield(self, days: int = 30) -> Dict:
        """
        Simulate yield earnings over a period.
        
        Args:
            days: Number of days to simulate
            
        Returns:
            Simulation results
        """
        total_earned = 0
        
        for position in self._positions:
            if position.status != 'active':
                continue
            
            daily_rate = position.opportunity.total_apy / 365
            earned = position.principal * daily_rate * days
            
            position.earned += earned
            position.current_value = position.principal + position.earned
            total_earned += earned
        
        return {
            'days_simulated': days,
            'total_earned': total_earned,
            'total_value': sum(p.current_value for p in self._positions if p.status == 'active'),
            'active_positions': len([p for p in self._positions if p.status == 'active']),
            'avg_apy': sum(p.opportunity.total_apy for p in self._positions if p.status == 'active') / max(len(self._positions), 1)
        }
    
    # ============== Reporting ==============
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary."""
        active_positions = [p for p in self._positions if p.status == 'active']
        
        return {
            'total_value': sum(p.current_value for p in active_positions),
            'total_earned': sum(p.earned for p in active_positions),
            'num_positions': len(active_positions),
            'idle_capital': self._paper_balance,
            'weighted_avg_apy': sum(p.opportunity.total_apy * p.current_value for p in active_positions) / max(sum(p.current_value for p in active_positions), 1),
            'paper_mode': self.paper_mode
        }
    
    def get_top_opportunities(self, n: int = 10) -> List[YieldOpportunity]:
        """Get top N opportunities by APY."""
        return self._opportunities[:n]
    
    async def close(self):
        """Close connections."""
        await self.llama_client.close()


# ============== Main Entry Point ==============

def main():
    """Test DeFi yield optimizer."""
    print("=" * 70)
    print("DEFI YIELD OPTIMIZER BOT")
    print("=" * 70)
    print("""
This bot automatically finds and rotates to the best yield in DeFi.

Supported Protocols:
  - Lending: Aave, Compound
  - Staking: Lido, Rocket Pool  
  - LPs: Curve, Uniswap
  - Aggregators: Yearn, Convex

Strategy:
  1. Scan all protocols for yield rates
  2. Filter by risk tolerance
  3. Auto-rotate to highest yields
  4. Compound rewards
""")
    
    # Initialize
    optimizer = DeFiYieldOptimizer(paper_mode=True)
    
    # Scan for opportunities
    print("\n" + "=" * 70)
    print("📊 SCANNING DEFI YIELDS")
    print("=" * 70)
    
    async def run():
        opportunities = await optimizer.scan()
        
        if opportunities:
            print(f"\nTop 10 Opportunities:")
            for i, opp in enumerate(opportunities[:10], 1):
                risk_emoji = "🟢" if opp.risk_score <= 3 else "🟡" if opp.risk_score <= 5 else "🔴"
                print(f"""
  {i}. {opp.asset} on {opp.protocol.value.capitalize()} ({opp.chain.value})
     APY: {opp.total_apy:.2%} (Base: {opp.base_apy:.2%} + Rewards: {opp.reward_apy:.2%})
     TVL: ${opp.tvl:,.0f}
     Risk: {risk_emoji} {opp.risk_score}/10
""")
        
        # Get rebalance recommendations
        print("\n" + "=" * 70)
        print("📋 REBALANCE RECOMMENDATIONS")
        print("=" * 70)
        
        recommendations = optimizer.get_rebalance_recommendations()
        
        for action in recommendations:
            print(f"\n  → {action.reason}")
            print(f"    Amount: ${action.amount:,.2f}")
            print(f"    Target: {action.to_opportunity.protocol.value} - {action.to_opportunity.asset}")
        
        # Execute rebalancing
        if recommendations:
            print("\n🚀 Executing rebalance...")
            await optimizer.execute_rebalance(recommendations)
        
        # Simulate 30 days of yield
        print("\n" + "=" * 70)
        print("📈 30-DAY YIELD SIMULATION")
        print("=" * 70)
        
        simulation = optimizer.simulate_yield(days=30)
        print(f"""
  Days: {simulation['days_simulated']}
  Total Earned: ${simulation['total_earned']:.2f}
  Portfolio Value: ${simulation['total_value']:.2f}
  Active Positions: {simulation['active_positions']}
  Average APY: {simulation['avg_apy']:.2%}
""")
        
        # Portfolio summary
        print("\n" + "=" * 70)
        print("💰 PORTFOLIO SUMMARY")
        print("=" * 70)
        
        summary = optimizer.get_portfolio_summary()
        for key, value in summary.items():
            if isinstance(value, float):
                if key.endswith('apy'):
                    print(f"  {key}: {value:.2%}")
                else:
                    print(f"  {key}: ${value:,.2f}")
            else:
                print(f"  {key}: {value}")
        
        await optimizer.close()
    
    asyncio.run(run())
    
    print("\n✅ DeFi yield optimizer test complete!")


if __name__ == '__main__':
    main()
