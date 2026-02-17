"""
3-Tier Scaled Exit Manager

Research Finding: 28% of trades "exited early" - significant alpha left on table.

Instead of all-or-nothing exits, use scaled exits:
- Tier 1: Exit 33% at 1R (1x risk/reward)
- Tier 2: Exit 33% at 2R (2x risk/reward)  
- Tier 3: Trail remaining 34% with 1.5 ATR stop

This locks in profits early while letting winners run.

Expected Impact:
- Convert 10-15% of early exits to +2-3% additional gains
- Reduce round-trip losses
- Better risk-adjusted returns

Author: Trading Bot Arsenal
Created: January 2026
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ScaledExitManager')


class ExitTier(Enum):
    """Exit tier levels"""
    TIER_1 = "tier_1"  # 33% at 1R
    TIER_2 = "tier_2"  # 33% at 2R
    TIER_3 = "tier_3"  # 34% trailing stop
    FULL = "full"       # Emergency/stop loss exit


@dataclass
class ScaledPosition:
    """Position with scaled exit tracking"""
    symbol: str
    entry_price: float
    total_shares: float
    stop_loss: float
    atr: float
    entry_time: datetime
    
    # Tier tracking
    tier1_shares: float = 0.0
    tier2_shares: float = 0.0
    tier3_shares: float = 0.0
    
    tier1_exited: bool = False
    tier2_exited: bool = False
    tier3_exited: bool = False
    
    # Trailing stop for tier 3
    trailing_stop: float = 0.0
    highest_price: float = 0.0
    
    # P&L tracking
    realized_pnl: float = 0.0
    tier1_exit_price: float = 0.0
    tier2_exit_price: float = 0.0
    tier3_exit_price: float = 0.0
    
    def __post_init__(self):
        """Initialize tier allocations"""
        self.tier1_shares = self.total_shares * 0.33
        self.tier2_shares = self.total_shares * 0.33
        self.tier3_shares = self.total_shares * 0.34
        self.trailing_stop = self.stop_loss
        self.highest_price = self.entry_price


@dataclass
class ExitSignal:
    """Signal to exit a tier"""
    symbol: str
    tier: ExitTier
    shares: float
    exit_price: float
    reason: str
    pnl: float
    pnl_pct: float


class ScaledExitManager:
    """
    Manages 3-tier scaled exits for positions.
    
    Exit Rules:
    - Tier 1 (33%): Exit when price reaches 1R (entry + 1x risk)
    - Tier 2 (33%): Exit when price reaches 2R (entry + 2x risk)
    - Tier 3 (34%): Trail with 1.5 ATR stop, let winners run
    
    Risk (R) = Entry Price - Stop Loss
    
    Example with $100 entry, $97 stop (3% risk = $3 per share):
    - Tier 1 target: $103 (1R = +$3)
    - Tier 2 target: $106 (2R = +$6)
    - Tier 3: Trail with 1.5 ATR (~$4.50) from highest price
    """
    
    # Exit configuration
    TIER1_R_MULTIPLE = 1.0   # Exit 33% at 1R
    TIER2_R_MULTIPLE = 2.0   # Exit 33% at 2R
    TIER3_ATR_TRAIL = 1.5    # Trail with 1.5 ATR
    
    # Allocation percentages
    TIER1_PCT = 0.33
    TIER2_PCT = 0.33
    TIER3_PCT = 0.34
    
    def __init__(self, db_path: str = "data/scaled_exits.db"):
        """Initialize scaled exit manager."""
        self.db_path = db_path
        self.positions: Dict[str, ScaledPosition] = {}
        
        self._init_database()
        self._load_positions()
        
        logger.info("ScaledExitManager initialized")
    
    def _init_database(self):
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                entry_price REAL,
                total_shares REAL,
                stop_loss REAL,
                atr REAL,
                entry_time TEXT,
                tier1_shares REAL,
                tier2_shares REAL,
                tier3_shares REAL,
                tier1_exited INTEGER,
                tier2_exited INTEGER,
                tier3_exited INTEGER,
                trailing_stop REAL,
                highest_price REAL,
                realized_pnl REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exit_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                tier TEXT,
                shares REAL,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                pnl_pct REAL,
                reason TEXT,
                exit_time TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_positions(self):
        """Load open positions from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM positions')
        rows = cursor.fetchall()
        
        for row in rows:
            pos = ScaledPosition(
                symbol=row[0],
                entry_price=row[1],
                total_shares=row[2],
                stop_loss=row[3],
                atr=row[4],
                entry_time=datetime.fromisoformat(row[5]),
                tier1_shares=row[6],
                tier2_shares=row[7],
                tier3_shares=row[8],
                tier1_exited=bool(row[9]),
                tier2_exited=bool(row[10]),
                tier3_exited=bool(row[11]),
                trailing_stop=row[12],
                highest_price=row[13],
                realized_pnl=row[14]
            )
            self.positions[pos.symbol] = pos
        
        conn.close()
        logger.info(f"Loaded {len(self.positions)} open positions")
    
    def _save_position(self, pos: ScaledPosition):
        """Save position to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pos.symbol, pos.entry_price, pos.total_shares, pos.stop_loss,
            pos.atr, pos.entry_time.isoformat(),
            pos.tier1_shares, pos.tier2_shares, pos.tier3_shares,
            int(pos.tier1_exited), int(pos.tier2_exited), int(pos.tier3_exited),
            pos.trailing_stop, pos.highest_price, pos.realized_pnl
        ))
        
        conn.commit()
        conn.close()
    
    def _remove_position(self, symbol: str):
        """Remove closed position from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
        conn.commit()
        conn.close()
        
        if symbol in self.positions:
            del self.positions[symbol]
    
    def _log_exit(self, signal: ExitSignal, entry_price: float):
        """Log exit to history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO exit_history (symbol, tier, shares, entry_price, exit_price, pnl, pnl_pct, reason, exit_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.symbol, signal.tier.value, signal.shares,
            entry_price, signal.exit_price, signal.pnl, signal.pnl_pct,
            signal.reason, datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def open_position(
        self,
        symbol: str,
        entry_price: float,
        total_shares: float,
        stop_loss: float,
        atr: float
    ) -> ScaledPosition:
        """
        Open a new scaled position.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            total_shares: Total position size in shares
            stop_loss: Initial stop loss price
            atr: Current ATR value
            
        Returns:
            ScaledPosition object
        """
        pos = ScaledPosition(
            symbol=symbol,
            entry_price=entry_price,
            total_shares=total_shares,
            stop_loss=stop_loss,
            atr=atr,
            entry_time=datetime.now()
        )
        
        self.positions[symbol] = pos
        self._save_position(pos)
        
        # Calculate targets
        risk = entry_price - stop_loss
        tier1_target = entry_price + (risk * self.TIER1_R_MULTIPLE)
        tier2_target = entry_price + (risk * self.TIER2_R_MULTIPLE)
        
        logger.info(f"Opened scaled position: {symbol}")
        logger.info(f"  Entry: ${entry_price:.2f}, Stop: ${stop_loss:.2f}")
        logger.info(f"  Risk (R): ${risk:.2f}")
        logger.info(f"  Tier 1 target (1R): ${tier1_target:.2f} ({pos.tier1_shares:.1f} shares)")
        logger.info(f"  Tier 2 target (2R): ${tier2_target:.2f} ({pos.tier2_shares:.1f} shares)")
        logger.info(f"  Tier 3: Trailing 1.5 ATR (${atr * 1.5:.2f}) ({pos.tier3_shares:.1f} shares)")
        
        return pos
    
    def check_exits(self, symbol: str, current_price: float) -> List[ExitSignal]:
        """
        Check if any exit conditions are met.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            List of ExitSignal objects for triggered exits
        """
        if symbol not in self.positions:
            return []
        
        pos = self.positions[symbol]
        signals = []
        
        # Calculate risk (R)
        risk = pos.entry_price - pos.stop_loss
        
        # Update highest price for trailing stop
        if current_price > pos.highest_price:
            pos.highest_price = current_price
            # Update trailing stop for tier 3
            new_trail = current_price - (pos.atr * self.TIER3_ATR_TRAIL)
            if new_trail > pos.trailing_stop:
                pos.trailing_stop = new_trail
                logger.debug(f"Updated trailing stop for {symbol}: ${pos.trailing_stop:.2f}")
        
        # Check stop loss (exit all remaining)
        if current_price <= pos.stop_loss:
            remaining_shares = 0
            if not pos.tier1_exited:
                remaining_shares += pos.tier1_shares
            if not pos.tier2_exited:
                remaining_shares += pos.tier2_shares
            if not pos.tier3_exited:
                remaining_shares += pos.tier3_shares
            
            if remaining_shares > 0:
                pnl = remaining_shares * (current_price - pos.entry_price)
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                
                signal = ExitSignal(
                    symbol=symbol,
                    tier=ExitTier.FULL,
                    shares=remaining_shares,
                    exit_price=current_price,
                    reason=f"Stop loss hit: ${current_price:.2f} <= ${pos.stop_loss:.2f}",
                    pnl=pnl,
                    pnl_pct=pnl_pct
                )
                signals.append(signal)
                
                # Mark all tiers exited
                pos.tier1_exited = True
                pos.tier2_exited = True
                pos.tier3_exited = True
                pos.realized_pnl += pnl
                
                self._log_exit(signal, pos.entry_price)
                self._remove_position(symbol)
                
                logger.info(f"STOP LOSS EXIT: {symbol} - {remaining_shares} shares @ ${current_price:.2f}, P&L: ${pnl:.2f}")
                return signals
        
        # Check Tier 1: Exit at 1R
        if not pos.tier1_exited:
            tier1_target = pos.entry_price + (risk * self.TIER1_R_MULTIPLE)
            
            if current_price >= tier1_target:
                pnl = pos.tier1_shares * (current_price - pos.entry_price)
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                
                signal = ExitSignal(
                    symbol=symbol,
                    tier=ExitTier.TIER_1,
                    shares=pos.tier1_shares,
                    exit_price=current_price,
                    reason=f"Tier 1 target (1R) hit: ${current_price:.2f} >= ${tier1_target:.2f}",
                    pnl=pnl,
                    pnl_pct=pnl_pct
                )
                signals.append(signal)
                
                pos.tier1_exited = True
                pos.tier1_exit_price = current_price
                pos.realized_pnl += pnl
                
                self._log_exit(signal, pos.entry_price)
                logger.info(f"TIER 1 EXIT: {symbol} - {pos.tier1_shares:.1f} shares @ ${current_price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:+.1%})")
        
        # Check Tier 2: Exit at 2R
        if not pos.tier2_exited:
            tier2_target = pos.entry_price + (risk * self.TIER2_R_MULTIPLE)
            
            if current_price >= tier2_target:
                pnl = pos.tier2_shares * (current_price - pos.entry_price)
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                
                signal = ExitSignal(
                    symbol=symbol,
                    tier=ExitTier.TIER_2,
                    shares=pos.tier2_shares,
                    exit_price=current_price,
                    reason=f"Tier 2 target (2R) hit: ${current_price:.2f} >= ${tier2_target:.2f}",
                    pnl=pnl,
                    pnl_pct=pnl_pct
                )
                signals.append(signal)
                
                pos.tier2_exited = True
                pos.tier2_exit_price = current_price
                pos.realized_pnl += pnl
                
                self._log_exit(signal, pos.entry_price)
                logger.info(f"TIER 2 EXIT: {symbol} - {pos.tier2_shares:.1f} shares @ ${current_price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:+.1%})")
        
        # Check Tier 3: Trailing stop
        if not pos.tier3_exited and (pos.tier1_exited or pos.tier2_exited):
            # Only activate tier 3 trailing after at least tier 1 exit
            if current_price <= pos.trailing_stop:
                pnl = pos.tier3_shares * (current_price - pos.entry_price)
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                
                signal = ExitSignal(
                    symbol=symbol,
                    tier=ExitTier.TIER_3,
                    shares=pos.tier3_shares,
                    exit_price=current_price,
                    reason=f"Tier 3 trailing stop hit: ${current_price:.2f} <= ${pos.trailing_stop:.2f}",
                    pnl=pnl,
                    pnl_pct=pnl_pct
                )
                signals.append(signal)
                
                pos.tier3_exited = True
                pos.tier3_exit_price = current_price
                pos.realized_pnl += pnl
                
                self._log_exit(signal, pos.entry_price)
                logger.info(f"TIER 3 EXIT: {symbol} - {pos.tier3_shares:.1f} shares @ ${current_price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:+.1%})")
        
        # Check if position is fully closed
        if pos.tier1_exited and pos.tier2_exited and pos.tier3_exited:
            self._remove_position(symbol)
            logger.info(f"Position {symbol} fully closed. Total P&L: ${pos.realized_pnl:.2f}")
        else:
            self._save_position(pos)
        
        return signals
    
    def get_position_status(self, symbol: str) -> Optional[Dict]:
        """Get status of a position."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        risk = pos.entry_price - pos.stop_loss
        
        return {
            'symbol': symbol,
            'entry_price': pos.entry_price,
            'stop_loss': pos.stop_loss,
            'atr': pos.atr,
            'risk_per_share': risk,
            'tiers': {
                'tier1': {
                    'shares': pos.tier1_shares,
                    'target': pos.entry_price + (risk * self.TIER1_R_MULTIPLE),
                    'exited': pos.tier1_exited,
                    'exit_price': pos.tier1_exit_price if pos.tier1_exited else None
                },
                'tier2': {
                    'shares': pos.tier2_shares,
                    'target': pos.entry_price + (risk * self.TIER2_R_MULTIPLE),
                    'exited': pos.tier2_exited,
                    'exit_price': pos.tier2_exit_price if pos.tier2_exited else None
                },
                'tier3': {
                    'shares': pos.tier3_shares,
                    'trailing_stop': pos.trailing_stop,
                    'highest_price': pos.highest_price,
                    'exited': pos.tier3_exited,
                    'exit_price': pos.tier3_exit_price if pos.tier3_exited else None
                }
            },
            'realized_pnl': pos.realized_pnl,
            'entry_time': pos.entry_time.isoformat()
        }
    
    def get_all_positions(self) -> List[Dict]:
        """Get status of all open positions."""
        return [self.get_position_status(s) for s in self.positions.keys()]
    
    def get_exit_history(self, limit: int = 50) -> List[Dict]:
        """Get recent exit history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM exit_history ORDER BY exit_time DESC LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'id': row[0],
                'symbol': row[1],
                'tier': row[2],
                'shares': row[3],
                'entry_price': row[4],
                'exit_price': row[5],
                'pnl': row[6],
                'pnl_pct': row[7],
                'reason': row[8],
                'exit_time': row[9]
            })
        
        return history


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("3-TIER SCALED EXIT MANAGER TEST")
    print("=" * 60)
    
    manager = ScaledExitManager()
    
    # Simulate opening a position
    print("\n📈 Opening position...")
    pos = manager.open_position(
        symbol="SPY",
        entry_price=500.00,
        total_shares=100,
        stop_loss=490.00,  # 2% stop = $10 risk per share
        atr=5.00
    )
    
    # Show status
    print("\n📊 Position Status:")
    status = manager.get_position_status("SPY")
    print(f"  Entry: ${status['entry_price']:.2f}")
    print(f"  Stop: ${status['stop_loss']:.2f}")
    print(f"  Risk (R): ${status['risk_per_share']:.2f}")
    print(f"\n  Tier 1: {status['tiers']['tier1']['shares']:.1f} shares, target ${status['tiers']['tier1']['target']:.2f}")
    print(f"  Tier 2: {status['tiers']['tier2']['shares']:.1f} shares, target ${status['tiers']['tier2']['target']:.2f}")
    print(f"  Tier 3: {status['tiers']['tier3']['shares']:.1f} shares, trailing ${status['tiers']['tier3']['trailing_stop']:.2f}")
    
    # Simulate price movement
    print("\n📈 Simulating price movement...")
    
    # Price moves to 510 (Tier 1 target)
    print("\n→ Price hits $510 (1R)...")
    signals = manager.check_exits("SPY", 510.00)
    for s in signals:
        print(f"  EXIT: {s.tier.value} - {s.shares:.1f} shares @ ${s.exit_price:.2f}, P&L: ${s.pnl:.2f}")
    
    # Price moves to 520 (Tier 2 target)
    print("\n→ Price hits $520 (2R)...")
    signals = manager.check_exits("SPY", 520.00)
    for s in signals:
        print(f"  EXIT: {s.tier.value} - {s.shares:.1f} shares @ ${s.exit_price:.2f}, P&L: ${s.pnl:.2f}")
    
    # Price pulls back to trailing stop
    print("\n→ Price pulls back to $512.50 (trailing stop)...")
    signals = manager.check_exits("SPY", 512.50)
    for s in signals:
        print(f"  EXIT: {s.tier.value} - {s.shares:.1f} shares @ ${s.exit_price:.2f}, P&L: ${s.pnl:.2f}")
    
    # Show history
    print("\n📜 Exit History:")
    history = manager.get_exit_history(limit=10)
    for h in history:
        print(f"  {h['tier']}: {h['shares']:.1f} shares, P&L ${h['pnl']:.2f} ({h['pnl_pct']:+.1%})")
    
    print("\n" + "=" * 60)
