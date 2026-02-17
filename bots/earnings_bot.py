"""
Earnings Momentum Bot

Exploits Post-Earnings Announcement Drift (PEAD) - an academically documented
anomaly where stocks continue to drift in the direction of earnings surprise.

Strategy Logic:
- Entry: Buy stocks with earnings surprise > 5% on day after announcement
- Exit: Hold for 5 trading days (capturing PEAD drift period)
- Filter: Market cap > $1B, average volume > 500K

Academic Source: Bernard & Thomas (1989), documented 2-6% drift over 60 days
Modern Implementation: Focus on first 5 days where drift is strongest

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import json
import logging
import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EarningsBot')

# Constants
CACHE_DIR = Path(__file__).parent.parent / "data" / "earnings_cache"
DB_PATH = Path(__file__).parent.parent / "data" / "event_trades.db"


@dataclass
class EarningsSurprise:
    """Earnings announcement data"""
    symbol: str
    company_name: str
    announcement_date: datetime
    estimated_eps: float
    actual_eps: float
    surprise_pct: float
    market_cap: float
    avg_volume: float
    price_at_announcement: float
    timestamp: datetime


@dataclass 
class EarningsPosition:
    """Active earnings-based position"""
    symbol: str
    entry_date: datetime
    entry_price: float
    surprise_pct: float
    target_exit_date: datetime
    quantity: int
    status: str = "open"
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    pnl_pct: Optional[float] = None


class EarningsBot:
    """
    Earnings Momentum Strategy Bot
    
    Exploits PEAD (Post-Earnings Announcement Drift):
    - Stocks with positive earnings surprises tend to drift higher
    - Stocks with negative earnings surprises tend to drift lower
    - Effect is strongest in first 5 trading days
    
    Entry Criteria:
    1. Earnings surprise > 5% (actual vs estimate)
    2. Market cap > $1B (liquidity)
    3. Average volume > 500K (tradeable)
    4. Price > $10 (avoid penny stocks)
    
    Exit Rules:
    1. Hold for exactly 5 trading days
    2. Stop loss at -5%
    
    Position Sizing:
    - 5% of portfolio per position
    - Maximum 5 concurrent positions (25% total)
    """
    
    MIN_SURPRISE_PCT = 0.05  # 5% minimum surprise
    MIN_MARKET_CAP = 1_000_000_000  # $1B
    MIN_AVG_VOLUME = 500_000
    MIN_PRICE = 10.0
    HOLDING_DAYS = 5
    STOP_LOSS_PCT = 0.05  # 5% stop loss
    POSITION_SIZE_PCT = 0.05  # 5% per position
    MAX_POSITIONS = 5
    
    def __init__(self, paper_mode: bool = None):
        """Initialize Earnings Bot"""
        # Safe default: read from environment, default to PAPER if not set
        if paper_mode is None:
            paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
        self.paper_mode = paper_mode
        self.positions: List[EarningsPosition] = []
        self.alpaca_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
        self.alpaca_base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        # Create cache directory
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"EarningsBot initialized (paper={paper_mode})")
    
    def _init_database(self):
        """Initialize SQLite database for earnings trades"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS earnings_surprises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                company_name TEXT,
                announcement_date TEXT,
                estimated_eps REAL,
                actual_eps REAL,
                surprise_pct REAL,
                market_cap REAL,
                avg_volume REAL,
                price_at_announcement REAL,
                timestamp TEXT,
                UNIQUE(symbol, announcement_date)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS earnings_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_date TEXT,
                entry_price REAL,
                surprise_pct REAL,
                target_exit_date TEXT,
                quantity INTEGER,
                status TEXT DEFAULT 'open',
                exit_price REAL,
                exit_date TEXT,
                pnl_pct REAL,
                paper_mode INTEGER DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Earnings database initialized")
    
    def fetch_earnings_calendar(self, days_ahead: int = 1) -> List[Dict]:
        """
        Fetch upcoming/recent earnings from Yahoo Finance or alternative source.
        
        Args:
            days_ahead: Days to look back for recent earnings
            
        Returns:
            List of earnings announcements
        """
        earnings = []
        
        # Try Yahoo Finance first
        try:
            earnings = self._fetch_yahoo_earnings(days_ahead)
        except Exception as e:
            logger.warning(f"Yahoo earnings fetch failed: {e}")
        
        # Fallback to Financial Modeling Prep (free tier)
        if not earnings:
            try:
                earnings = self._fetch_fmp_earnings(days_ahead)
            except Exception as e:
                logger.warning(f"FMP earnings fetch failed: {e}")
        
        # Cache results
        if earnings:
            cache_file = CACHE_DIR / f"earnings_{datetime.now().strftime('%Y%m%d')}.json"
            with open(cache_file, 'w') as f:
                json.dump(earnings, f, default=str)
        
        return earnings
    
    def _fetch_yahoo_earnings(self, days_back: int = 1) -> List[Dict]:
        """Fetch earnings from Yahoo Finance"""
        import yfinance as yf
        
        earnings = []
        
        # Get S&P 500 constituents for screening
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            tables = pd.read_html(sp500_url)
            sp500_symbols = tables[0]['Symbol'].tolist()
        except Exception as e:
            logger.debug(f"Failed to fetch S&P 500 list: {e}")
            # Fallback to common large caps
            sp500_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 
                           'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS']
        
        # Check recent earnings for each symbol
        for symbol in sp500_symbols[:50]:  # Limit to 50 for speed
            try:
                ticker = yf.Ticker(symbol)
                cal = ticker.calendar
                
                if cal is not None and not cal.empty:
                    # Get earnings date
                    if 'Earnings Date' in cal.index:
                        earn_date = cal.loc['Earnings Date'].values[0]
                        if isinstance(earn_date, (pd.Timestamp, datetime)):
                            earnings.append({
                                'symbol': symbol,
                                'date': earn_date,
                                'source': 'yahoo'
                            })
            except Exception as e:
                continue
        
        return earnings
    
    def _fetch_fmp_earnings(self, days_back: int = 1) -> List[Dict]:
        """Fetch earnings from Financial Modeling Prep API"""
        api_key = os.getenv('FMP_API_KEY', 'demo')  # Free tier
        
        # Get yesterday and today's earnings
        today = datetime.now()
        
        earnings = []
        for days_ago in range(days_back + 1):
            date = today - timedelta(days=days_ago)
            date_str = date.strftime('%Y-%m-%d')
            
            url = f"https://financialmodelingprep.com/api/v3/earning_calendar?from={date_str}&to={date_str}&apikey={api_key}"
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        if item.get('epsEstimated') and item.get('eps'):
                            earnings.append({
                                'symbol': item['symbol'],
                                'date': item['date'],
                                'estimated_eps': item['epsEstimated'],
                                'actual_eps': item['eps'],
                                'source': 'fmp'
                            })
            except Exception as e:
                logger.warning(f"FMP API error: {e}")
        
        return earnings
    
    def calculate_surprise(self, estimated: float, actual: float) -> float:
        """
        Calculate earnings surprise percentage.
        
        Formula: (Actual - Estimate) / |Estimate|
        """
        if estimated == 0:
            return 0.0
        return (actual - estimated) / abs(estimated)
    
    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """
        Get stock data including market cap, volume, price.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dict with stock metrics or None
        """
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'market_cap': info.get('marketCap', 0),
                'avg_volume': info.get('averageVolume', 0),
                'price': info.get('currentPrice') or info.get('regularMarketPrice', 0),
                'name': info.get('shortName', symbol)
            }
        except Exception as e:
            logger.warning(f"Failed to get data for {symbol}: {e}")
            return None
    
    def screen_for_opportunities(self) -> List[EarningsSurprise]:
        """
        Screen for earnings opportunities meeting our criteria.
        
        Returns:
            List of qualified earnings surprises
        """
        opportunities = []
        
        # Fetch recent earnings
        earnings = self.fetch_earnings_calendar(days_ahead=1)
        
        for earn in earnings:
            symbol = earn.get('symbol', '').replace('.', '-')  # Handle BRK.B -> BRK-B
            
            # Skip if missing EPS data
            if 'estimated_eps' not in earn or 'actual_eps' not in earn:
                continue
            
            # Calculate surprise
            surprise = self.calculate_surprise(
                earn['estimated_eps'],
                earn['actual_eps']
            )
            
            # Check minimum surprise threshold
            if abs(surprise) < self.MIN_SURPRISE_PCT:
                continue
            
            # Get additional stock data
            stock_data = self.get_stock_data(symbol)
            if not stock_data:
                continue
            
            # Apply filters
            if stock_data['market_cap'] < self.MIN_MARKET_CAP:
                logger.debug(f"Skipping {symbol}: market cap too low")
                continue
                
            if stock_data['avg_volume'] < self.MIN_AVG_VOLUME:
                logger.debug(f"Skipping {symbol}: volume too low")
                continue
                
            if stock_data['price'] < self.MIN_PRICE:
                logger.debug(f"Skipping {symbol}: price too low")
                continue
            
            # Create opportunity
            opp = EarningsSurprise(
                symbol=symbol,
                company_name=stock_data.get('name', symbol),
                announcement_date=datetime.fromisoformat(str(earn['date'])[:10]),
                estimated_eps=earn['estimated_eps'],
                actual_eps=earn['actual_eps'],
                surprise_pct=surprise,
                market_cap=stock_data['market_cap'],
                avg_volume=stock_data['avg_volume'],
                price_at_announcement=stock_data['price'],
                timestamp=datetime.now(timezone.utc)
            )
            
            opportunities.append(opp)
            logger.info(f"Found opportunity: {symbol} surprise={surprise:.1%}")
        
        # Sort by surprise magnitude
        opportunities.sort(key=lambda x: abs(x.surprise_pct), reverse=True)
        
        return opportunities
    
    def execute_trade(self, opportunity: EarningsSurprise, account_balance: float) -> Optional[EarningsPosition]:
        """
        Execute trade based on earnings opportunity.
        
        Args:
            opportunity: Earnings surprise data
            account_balance: Current account balance
            
        Returns:
            Position if trade executed, None otherwise
        """
        # Check if we have room for more positions
        open_positions = [p for p in self.positions if p.status == 'open']
        if len(open_positions) >= self.MAX_POSITIONS:
            logger.warning(f"Max positions ({self.MAX_POSITIONS}) reached")
            return None
        
        # Check if we already have position in this symbol
        if any(p.symbol == opportunity.symbol and p.status == 'open' for p in self.positions):
            logger.warning(f"Already have position in {opportunity.symbol}")
            return None
        
        # Calculate position size
        position_value = account_balance * self.POSITION_SIZE_PCT
        quantity = int(position_value / opportunity.price_at_announcement)
        
        if quantity < 1:
            logger.warning(f"Position size too small for {opportunity.symbol}")
            return None
        
        # Determine direction (positive surprise = long, negative = short)
        side = "buy" if opportunity.surprise_pct > 0 else "sell"
        
        # Execute via Alpaca
        if self.paper_mode:
            logger.info(f"[PAPER] Would {side} {quantity} {opportunity.symbol} @ ${opportunity.price_at_announcement:.2f}")
            success = True
        else:
            success = self._execute_alpaca_order(opportunity.symbol, quantity, side)
        
        if success:
            # Calculate target exit date (5 trading days)
            target_exit = self._get_trading_day_offset(datetime.now(), self.HOLDING_DAYS)
            
            position = EarningsPosition(
                symbol=opportunity.symbol,
                entry_date=datetime.now(timezone.utc),
                entry_price=opportunity.price_at_announcement,
                surprise_pct=opportunity.surprise_pct,
                target_exit_date=target_exit,
                quantity=quantity
            )
            
            self.positions.append(position)
            self._save_position(position)
            
            logger.info(f"Opened position: {opportunity.symbol} x{quantity}, exit by {target_exit.date()}")
            return position
        
        return None
    
    def _execute_alpaca_order(self, symbol: str, quantity: int, side: str) -> bool:
        """Execute order via Alpaca API"""
        try:
            url = f"{self.alpaca_base}/v2/orders"
            headers = {
                'APCA-API-KEY-ID': self.alpaca_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret
            }
            payload = {
                'symbol': symbol,
                'qty': quantity,
                'side': side,
                'type': 'market',
                'time_in_force': 'day'
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Alpaca order failed: {e}")
            return False
    
    def _get_trading_day_offset(self, start_date: datetime, days: int) -> datetime:
        """Get date N trading days from start"""
        # Simple approximation: add extra days for weekends
        total_days = days + (days // 5) * 2 + 2
        return start_date + timedelta(days=total_days)
    
    def check_exits(self) -> List[Dict]:
        """
        Check if any positions should be exited.
        
        Returns:
            List of exit results
        """
        exits = []
        today = datetime.now(timezone.utc)
        
        for position in self.positions:
            if position.status != 'open':
                continue
            
            # Get current price
            stock_data = self.get_stock_data(position.symbol)
            if not stock_data:
                continue
            
            current_price = stock_data['price']
            pnl_pct = (current_price - position.entry_price) / position.entry_price
            
            # Adjust PnL for short positions
            if position.surprise_pct < 0:
                pnl_pct = -pnl_pct
            
            should_exit = False
            exit_reason = ""
            
            # Check holding period
            if today >= position.target_exit_date:
                should_exit = True
                exit_reason = f"Holding period complete ({self.HOLDING_DAYS} days)"
            
            # Check stop loss
            elif pnl_pct <= -self.STOP_LOSS_PCT:
                should_exit = True
                exit_reason = f"Stop loss triggered ({pnl_pct:.1%})"
            
            if should_exit:
                # Execute exit
                side = "sell" if position.surprise_pct > 0 else "buy"  # Opposite of entry
                
                if self.paper_mode:
                    logger.info(f"[PAPER] Would {side} {position.quantity} {position.symbol} @ ${current_price:.2f}")
                    success = True
                else:
                    success = self._execute_alpaca_order(position.symbol, position.quantity, side)
                
                if success:
                    position.status = 'closed'
                    position.exit_price = current_price
                    position.exit_date = today
                    position.pnl_pct = pnl_pct
                    
                    self._update_position(position)
                    
                    exits.append({
                        'symbol': position.symbol,
                        'entry_price': position.entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'reason': exit_reason
                    })
                    
                    logger.info(f"Closed {position.symbol}: {pnl_pct:+.1%} - {exit_reason}")
        
        return exits
    
    def _save_position(self, position: EarningsPosition):
        """Save position to database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO earnings_trades (symbol, entry_date, entry_price, surprise_pct,
                                         target_exit_date, quantity, status, paper_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            position.symbol,
            position.entry_date.isoformat(),
            position.entry_price,
            position.surprise_pct,
            position.target_exit_date.isoformat(),
            position.quantity,
            position.status,
            1 if self.paper_mode else 0
        ))
        
        conn.commit()
        conn.close()
    
    def _update_position(self, position: EarningsPosition):
        """Update position in database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE earnings_trades 
            SET status = ?, exit_price = ?, exit_date = ?, pnl_pct = ?
            WHERE symbol = ? AND entry_date = ?
        ''', (
            position.status,
            position.exit_price,
            position.exit_date.isoformat() if position.exit_date else None,
            position.pnl_pct,
            position.symbol,
            position.entry_date.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM earnings_trades WHERE status = 'closed'
        ''')
        
        trades = cursor.fetchall()
        conn.close()
        
        if not trades:
            return {'total_trades': 0}
        
        pnls = [t[10] for t in trades if t[10] is not None]  # pnl_pct column
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        return {
            'total_trades': len(pnls),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(pnls) if pnls else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'total_return': sum(pnls),
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0
        }
    
    def run_scan(self, account_balance: float = 10000) -> Dict:
        """
        Run full earnings scan and execute trades.
        
        Args:
            account_balance: Current account value
            
        Returns:
            Scan results
        """
        logger.info("=" * 50)
        logger.info("EARNINGS BOT SCAN")
        logger.info("=" * 50)
        
        # Check for exits first
        exits = self.check_exits()
        
        # Screen for new opportunities
        opportunities = self.screen_for_opportunities()
        
        # Execute trades on top opportunities
        new_positions = []
        for opp in opportunities[:self.MAX_POSITIONS]:
            position = self.execute_trade(opp, account_balance)
            if position:
                new_positions.append(position)
        
        # Get current stats
        stats = self.get_performance_stats()
        
        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'opportunities_found': len(opportunities),
            'new_positions': len(new_positions),
            'exits': len(exits),
            'open_positions': len([p for p in self.positions if p.status == 'open']),
            'performance': stats,
            'paper_mode': self.paper_mode
        }
        
        logger.info(f"Scan complete: {len(opportunities)} opps, {len(new_positions)} new trades, {len(exits)} exits")
        
        return result


def send_telegram_alert(message: str):
    """Send Telegram alert"""
    try:
        from utils.telegram_bot import TelegramBot
        bot = TelegramBot()
        bot.send_message(message)
    except Exception as e:
        logger.warning(f"Telegram alert failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("EARNINGS MOMENTUM BOT")
    print("=" * 60)
    
    bot = EarningsBot(paper_mode=True)
    
    # Run scan
    result = bot.run_scan(account_balance=10000)
    
    print(f"\nScan Results:")
    print(f"  Opportunities: {result['opportunities_found']}")
    print(f"  New Positions: {result['new_positions']}")
    print(f"  Exits: {result['exits']}")
    print(f"  Open Positions: {result['open_positions']}")
    
    if result['performance']['total_trades'] > 0:
        print(f"\nPerformance:")
        print(f"  Win Rate: {result['performance']['win_rate']:.1%}")
        print(f"  Total Return: {result['performance']['total_return']:.1%}")
