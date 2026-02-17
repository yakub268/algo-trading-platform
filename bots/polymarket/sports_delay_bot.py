"""
Sports Broadcast Delay Bot

Exploits the 15-40 second delay between real-world sports events and 
Polymarket market updates. One bot turned $5 into $3.7M using this method.

How it works:
1. Get real-time sports data from premium feeds (faster than TV)
2. When a scoring event happens, bet on Polymarket before odds update
3. The 15-40 second window allows betting on "known" outcomes

Key insight: TV broadcasts have 15-40 second delays. Premium sports APIs 
have 1-5 second delays. This creates a 10-35 second arbitrage window.

Expected APY: 100-500%+ (highly variable)
Risk: Data feed reliability, execution speed, Polymarket's 3-second order delay

Sports Supported:
- NFL (15-30 second delay)
- NBA (20-40 second delay)
- MLB (15-25 second delay)
- Soccer/Football (20-35 second delay)

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
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bots.polymarket.polymarket_client import PolymarketClient, Market

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SportsDelayBot')


class Sport(Enum):
    NFL = "nfl"
    NBA = "nba"
    MLB = "mlb"
    NHL = "nhl"
    SOCCER = "soccer"
    ESPORTS = "esports"


class EventType(Enum):
    SCORE = "score"
    GAME_END = "game_end"
    QUARTER_END = "quarter_end"
    TIMEOUT = "timeout"
    INJURY = "injury"


@dataclass
class LiveGame:
    """Represents a live game being monitored"""
    game_id: str
    sport: Sport
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    period: str  # "Q1", "1st Half", etc.
    time_remaining: str
    status: str  # "live", "halftime", "final"
    poly_market_id: Optional[str] = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ScoringEvent:
    """Represents a detected scoring event"""
    game: LiveGame
    event_type: EventType
    description: str
    old_score: Tuple[int, int]  # (home, away)
    new_score: Tuple[int, int]
    detected_at: datetime
    confidence: float


@dataclass
class DelayOpportunity:
    """Represents a trading opportunity from broadcast delay"""
    scoring_event: ScoringEvent
    poly_market: Market
    recommended_bet: str  # e.g., "BUY Home Team WIN"
    current_market_price: float
    fair_value: float  # Our calculated fair value post-event
    edge: float
    time_window_seconds: float  # Estimated time before market updates
    executed: bool = False


@dataclass
class TradeResult:
    """Result of a delay arbitrage trade"""
    opportunity: DelayOpportunity
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float]
    exit_time: Optional[datetime]
    profit: Optional[float]
    status: str  # 'open', 'won', 'lost', 'cancelled'


class SportsDataProvider:
    """
    Abstract interface for sports data providers.
    
    Premium providers (faster, paid):
    - Sportradar (~1-2s delay) - $500+/mo
    - Genius Sports (~1-3s delay) - $300+/mo
    - Stats Perform (~2-5s delay) - $200+/mo
    
    Free/cheap providers (slower):
    - ESPN API (~5-15s delay) - Free
    - The Odds API (~10-20s delay) - Free tier
    """
    
    async def get_live_games(self, sport: Sport) -> List[LiveGame]:
        """
        Fetch currently live games for a sport.

        Must be implemented by subclasses for each data provider.

        Args:
            sport: Sport enum (NFL, NBA, MLB, etc.)

        Returns:
            List of LiveGame objects for games currently in progress
        """
        logger.warning(f"get_live_games not implemented for {self.__class__.__name__}")
        return []

    async def subscribe_to_game(self, game_id: str, callback: Callable):
        """
        Subscribe to real-time updates for a specific game.

        Must be implemented by subclasses for each data provider.
        The callback will be invoked with a LiveGame object whenever
        the game state changes (score updates, period changes, etc.)

        Args:
            game_id: Unique identifier for the game
            callback: Async function to call with LiveGame updates
        """
        logger.warning(f"subscribe_to_game not implemented for {self.__class__.__name__}")


class ESPNDataProvider(SportsDataProvider):
    """
    ESPN API data provider (free but slower).

    Delay: ~5-15 seconds from real-time
    Good for: Testing, low-stakes trading

    Uses synchronous requests to avoid Windows asyncio DNS issues.
    """

    ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports"

    SPORT_PATHS = {
        Sport.NFL: "football/nfl",
        Sport.NBA: "basketball/nba",
        Sport.MLB: "baseball/mlb",
        Sport.NHL: "hockey/nhl",
        Sport.SOCCER: "soccer/usa.1",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'TradingBot/1.0'})
        self._subscriptions: Dict[str, Callable] = {}
        self._running = False

    def _fetch_scoreboard(self, sport: Sport) -> List[LiveGame]:
        """Synchronous fetch of ESPN scoreboard."""
        path = self.SPORT_PATHS.get(sport)
        if not path:
            return []

        try:
            url = f"{self.ESPN_API_BASE}/{path}/scoreboard"
            response = self.session.get(url, timeout=10)

            if response.status_code != 200:
                logger.debug(f"ESPN returned status {response.status_code}")
                return []

            data = response.json()
            games = []

            for event in data.get('events', []):
                competition = event.get('competitions', [{}])[0]
                competitors = competition.get('competitors', [])

                if len(competitors) < 2:
                    continue

                home = next((c for c in competitors if c.get('homeAway') == 'home'), competitors[0])
                away = next((c for c in competitors if c.get('homeAway') == 'away'), competitors[1])

                status = event.get('status', {})

                game = LiveGame(
                    game_id=event.get('id', ''),
                    sport=sport,
                    home_team=home.get('team', {}).get('name', 'Unknown'),
                    away_team=away.get('team', {}).get('name', 'Unknown'),
                    home_score=int(home.get('score') or 0),
                    away_score=int(away.get('score') or 0),
                    period=status.get('period', 0),
                    time_remaining=status.get('displayClock', ''),
                    status=status.get('type', {}).get('state', 'unknown'),
                    last_update=datetime.now(timezone.utc)
                )
                games.append(game)

            return games

        except requests.exceptions.Timeout:
            logger.warning("ESPN API timeout")
            return []
        except requests.exceptions.ConnectionError:
            logger.warning("ESPN API connection error")
            return []
        except Exception as e:
            logger.debug(f"ESPN API error: {e}")
            return []

    async def get_live_games(self, sport: Sport) -> List[LiveGame]:
        """Fetch current live games from ESPN (async wrapper)."""
        return await asyncio.to_thread(self._fetch_scoreboard, sport)

    async def subscribe_to_game(self, game_id: str, callback: Callable):
        """Subscribe to updates for a specific game."""
        self._subscriptions[game_id] = callback

    async def start_polling(self, interval: float = 2.0):
        """Start polling for updates."""
        self._running = True

        while self._running:
            for sport in Sport:
                try:
                    games = await self.get_live_games(sport)
                    for game in games:
                        if game.game_id in self._subscriptions:
                            await self._subscriptions[game.game_id](game)
                except Exception as e:
                    logger.debug(f"Polling error: {e}")

            await asyncio.sleep(interval)

    def stop_polling(self):
        self._running = False


class SportsDelayBot:
    """
    Main sports broadcast delay arbitrage bot.
    
    Workflow:
    1. Monitor live games via data provider
    2. Detect scoring events by comparing consecutive updates
    3. When score changes, immediately check Polymarket odds
    4. If odds haven't updated yet, place bet on likely outcome
    5. Close position after market adjusts or game ends
    
    Key Parameters:
    - MIN_EDGE: Minimum edge to trade (default 5%)
    - MAX_POSITION: Maximum position size
    - TIME_WINDOW: Expected arbitrage window in seconds
    
    Usage:
        bot = SportsDelayBot(paper_mode=True)
        await bot.start()
    """
    
    # Trading parameters
    MIN_EDGE = 0.05  # 5% minimum edge
    MAX_POSITION = 100  # $100 max per trade
    TIME_WINDOW = 15  # Expected seconds before market updates
    
    # Polymarket's order delay (as of Jan 2026)
    POLY_ORDER_DELAY = 3  # 3-second delay on sports markets
    
    def __init__(self, 
                 paper_mode: bool = True,
                 data_provider: Optional[SportsDataProvider] = None):
        """
        Initialize sports delay bot.
        
        Args:
            paper_mode: If True, simulate trades
            data_provider: Sports data source (defaults to ESPN)
        """
        self.paper_mode = paper_mode
        self.poly_client = PolymarketClient(paper_mode=paper_mode)
        self.data_provider = data_provider or ESPNDataProvider()
        
        # State tracking
        self._previous_scores: Dict[str, Tuple[int, int]] = {}
        self._active_games: Dict[str, LiveGame] = {}
        self._opportunities: List[DelayOpportunity] = []
        self._trades: List[TradeResult] = []
        
        # Market mapping (game_id -> poly_market_id)
        self._market_mapping: Dict[str, str] = {}
        
        # Database
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'sports_delay_bot.db'
        )
        self._init_database()
        
        logger.info(f"Sports Delay Bot initialized (paper_mode={paper_mode})")
    
    def _init_database(self):
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scoring_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                game_id TEXT,
                sport TEXT,
                home_team TEXT,
                away_team TEXT,
                old_home_score INTEGER,
                old_away_score INTEGER,
                new_home_score INTEGER,
                new_away_score INTEGER,
                detected_latency_ms INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                game_id TEXT,
                market_id TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                size REAL,
                profit REAL,
                status TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    # ============== Game Monitoring ==============
    
    async def monitor_game(self, game: LiveGame):
        """
        Process a game update and detect scoring events.
        
        Called by the data provider when game state changes.
        """
        game_id = game.game_id
        current_score = (game.home_score, game.away_score)
        
        # Check for score change
        if game_id in self._previous_scores:
            old_score = self._previous_scores[game_id]
            
            if current_score != old_score:
                # Score changed! Potential opportunity
                event = ScoringEvent(
                    game=game,
                    event_type=EventType.SCORE,
                    description=f"{game.home_team} {current_score[0]} - {current_score[1]} {game.away_team}",
                    old_score=old_score,
                    new_score=current_score,
                    detected_at=datetime.now(timezone.utc),
                    confidence=0.95
                )
                
                logger.info(f"🏈 SCORING EVENT: {event.description}")
                self._log_scoring_event(event)
                
                # Check for trading opportunity
                await self._evaluate_opportunity(event)
        
        # Update state
        self._previous_scores[game_id] = current_score
        self._active_games[game_id] = game
    
    def _log_scoring_event(self, event: ScoringEvent):
        """Log scoring event to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO scoring_events 
                (timestamp, game_id, sport, home_team, away_team,
                 old_home_score, old_away_score, new_home_score, new_away_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.detected_at.isoformat(),
                event.game.game_id,
                event.game.sport.value,
                event.game.home_team,
                event.game.away_team,
                event.old_score[0],
                event.old_score[1],
                event.new_score[0],
                event.new_score[1]
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    # ============== Opportunity Evaluation ==============
    
    async def _evaluate_opportunity(self, event: ScoringEvent):
        """
        Evaluate if a scoring event creates a trading opportunity.
        
        1. Find matching Polymarket market
        2. Get current market odds
        3. Calculate fair value based on new score
        4. If edge exists, create opportunity
        """
        # Find matching market on Polymarket
        market = await self._find_matching_market(event.game)
        
        if not market:
            logger.debug(f"No matching market for {event.game.home_team} vs {event.game.away_team}")
            return
        
        # Get current market price
        current_price = self._get_market_price(market, event.game.home_team)
        if current_price is None:
            return
        
        # Calculate fair value based on new score
        fair_value = self._calculate_fair_value(event)
        
        # Calculate edge
        edge = fair_value - current_price
        
        if abs(edge) >= self.MIN_EDGE:
            # We have an opportunity!
            recommended_bet = "BUY" if edge > 0 else "SELL"
            
            opportunity = DelayOpportunity(
                scoring_event=event,
                poly_market=market,
                recommended_bet=f"{recommended_bet} {event.game.home_team} WIN",
                current_market_price=current_price,
                fair_value=fair_value,
                edge=edge,
                time_window_seconds=self.TIME_WINDOW - self.POLY_ORDER_DELAY
            )
            
            self._opportunities.append(opportunity)
            
            logger.info(f"""
💰 OPPORTUNITY DETECTED!
  Game: {event.game.home_team} vs {event.game.away_team}
  Score: {event.new_score[0]} - {event.new_score[1]}
  Current Price: {current_price:.2%}
  Fair Value: {fair_value:.2%}
  Edge: {edge:.2%}
  Action: {opportunity.recommended_bet}
  Time Window: {opportunity.time_window_seconds:.1f}s
""")
            
            # Execute trade
            await self._execute_trade(opportunity)
    
    async def _find_matching_market(self, game: LiveGame) -> Optional[Market]:
        """Find Polymarket market matching this game."""
        
        # Check cache first
        if game.game_id in self._market_mapping:
            return self.poly_client.get_market(self._market_mapping[game.game_id])
        
        # Search for matching market
        search_query = f"{game.home_team} {game.away_team}"
        markets = self.poly_client.search_markets(search_query, limit=5)
        
        for market in markets:
            # Check if market question mentions both teams
            if (game.home_team.lower() in market.question.lower() and 
                game.away_team.lower() in market.question.lower()):
                self._market_mapping[game.game_id] = market.condition_id
                return market
        
        return None
    
    def _get_market_price(self, market: Market, team: str) -> Optional[float]:
        """Get current market price for a team winning."""
        for outcome, price in market.outcome_prices.items():
            if team.lower() in outcome.lower():
                return price
        return None
    
    def _calculate_fair_value(self, event: ScoringEvent) -> float:
        """
        Calculate fair win probability based on current score.
        
        Uses simple model:
        - Score differential strongly predicts winner
        - Time remaining matters (larger lead earlier = stronger predictor)
        
        This is simplified - production would use more sophisticated models.
        """
        home_score, away_score = event.new_score
        diff = home_score - away_score
        
        # Base probability from score differential
        # Rough approximation: each point = ~3% swing from 50%
        base_prob = 0.5 + (diff * 0.03)
        
        # Clamp to reasonable range
        fair_value = max(0.1, min(0.9, base_prob))
        
        return fair_value
    
    # ============== Trade Execution ==============
    
    async def _execute_trade(self, opportunity: DelayOpportunity):
        """Execute a trade on the opportunity."""
        
        if opportunity.executed:
            return
        
        market = opportunity.poly_market
        
        # Determine token and side
        if "BUY" in opportunity.recommended_bet:
            side = "BUY"
            price = opportunity.current_market_price + 0.01  # Slightly above to ensure fill
        else:
            side = "SELL"
            price = opportunity.current_market_price - 0.01
        
        # Find correct token
        home_team = opportunity.scoring_event.game.home_team
        token_id = None
        for token in market.tokens:
            if home_team.lower() in token.get('outcome', '').lower():
                token_id = token.get('token_id')
                break
        
        if not token_id:
            logger.error("Could not find token for trade")
            return
        
        # Calculate size
        size = min(self.MAX_POSITION / price, 100)  # Max 100 shares
        
        # Place order
        if self.paper_mode:
            order_id = f"paper_{int(time.time())}"
            logger.info(f"[PAPER] Order placed: {side} {size:.2f} @ ${price:.2f}")
        else:
            order = self.poly_client.place_order(
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                order_type='FOK'  # Fill-or-Kill for speed
            )
            order_id = order.order_id if order else None
        
        if order_id:
            opportunity.executed = True
            
            trade = TradeResult(
                opportunity=opportunity,
                entry_price=price,
                entry_time=datetime.now(timezone.utc),
                exit_price=None,
                exit_time=None,
                profit=None,
                status='open'
            )
            
            self._trades.append(trade)
            self._log_trade(trade)
    
    def _log_trade(self, trade: TradeResult):
        """Log trade to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades 
                (timestamp, game_id, market_id, side, entry_price, size, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.entry_time.isoformat(),
                trade.opportunity.scoring_event.game.game_id,
                trade.opportunity.poly_market.condition_id,
                trade.opportunity.recommended_bet,
                trade.entry_price,
                self.MAX_POSITION,
                trade.status
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
    
    # ============== Bot Control ==============
    
    async def start(self, sports: List[Sport] = None):
        """
        Start the sports delay bot.
        
        Args:
            sports: List of sports to monitor (default: all)
        """
        if sports is None:
            sports = [Sport.NFL, Sport.NBA, Sport.MLB]
        
        logger.info(f"Starting Sports Delay Bot for: {[s.value for s in sports]}")
        
        # Subscribe to game updates
        for sport in sports:
            games = await self.data_provider.get_live_games(sport)
            for game in games:
                if game.status == 'in':  # Only live games
                    await self.data_provider.subscribe_to_game(
                        game.game_id, 
                        self.monitor_game
                    )
                    logger.info(f"Monitoring: {game.home_team} vs {game.away_team}")
        
        # Start polling
        await self.data_provider.start_polling(interval=2.0)
    
    def stop(self):
        """Stop the bot."""
        self.data_provider.stop_polling()
        logger.info("Sports Delay Bot stopped")
    
    # ============== Reporting ==============
    
    def get_statistics(self) -> Dict:
        """Get bot statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM scoring_events')
        events_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*), SUM(profit) FROM trades WHERE status = "won"')
        won = cursor.fetchone()

        cursor.execute('SELECT COUNT(*) FROM trades WHERE status = "lost"')
        lost = cursor.fetchone()[0]

        conn.close()

        return {
            'scoring_events_detected': events_count,
            'opportunities_found': len(self._opportunities),
            'trades_executed': len(self._trades),
            'trades_won': won[0] or 0,
            'trades_lost': lost or 0,
            'total_profit': won[1] or 0,
            'paper_mode': self.paper_mode
        }

    def _scan_live_games_sync(self) -> List[Dict]:
        """Scan for live games and opportunities (synchronous)."""
        signals = []

        try:
            for sport in [Sport.NFL, Sport.NBA, Sport.MLB]:
                games = self.data_provider._fetch_scoreboard(sport)

                for game in games:
                    # Track games
                    self._active_games[game.game_id] = game

                    # Check for score changes
                    prev = self._previous_scores.get(game.game_id)
                    current = (game.home_score, game.away_score)

                    if prev and prev != current:
                        # Score changed! Potential opportunity
                        logger.info(f"Score change detected: {game.away_team}@{game.home_team} {prev} -> {current}")

                        signal = {
                            'game_id': game.game_id,
                            'sport': sport.value,
                            'matchup': f"{game.away_team} @ {game.home_team}",
                            'score_change': f"{prev} -> {current}",
                            'action': 'monitor',
                            'type': 'sports_delay'
                        }
                        signals.append(signal)

                    self._previous_scores[game.game_id] = current

        except Exception as e:
            logger.debug(f"Error scanning live games: {e}")

        return signals

    def run_scan(self) -> List[Dict]:
        """
        Main method called by orchestrator.
        Scans for live games and score changes.
        """
        logger.debug("Starting sports delay scan...")

        try:
            signals = self._scan_live_games_sync()

            logger.debug(f"Sports delay scan complete: {len(self._active_games)} games tracked, {len(signals)} events")

            return signals

        except Exception as e:
            logger.warning(f"Sports delay scan failed: {e}")
            return []


# ============== Main Entry Point ==============

def main():
    """Test sports delay bot."""
    print("=" * 70)
    print("SPORTS BROADCAST DELAY BOT")
    print("=" * 70)
    print("""
This bot exploits the 15-40 second delay between real sports events and
Polymarket odds updates. When a scoring event happens, we bet before
the market has time to adjust.

One trader turned $5 into $3.7M using this method!

Broadcast Delays:
  - NFL: 15-30 seconds
  - NBA: 20-40 seconds  
  - MLB: 15-25 seconds

Premium Data Feeds (faster than TV):
  - Sportradar: ~1-2s delay ($500+/mo)
  - Genius Sports: ~1-3s delay ($300+/mo)
  - ESPN API: ~5-15s delay (FREE)
""")
    
    # Initialize bot
    bot = SportsDelayBot(paper_mode=True)
    
    # Fetch live games
    print("\n" + "=" * 70)
    print("📺 FETCHING LIVE GAMES")
    print("=" * 70)
    
    async def fetch_games():
        provider = ESPNDataProvider()
        
        for sport in [Sport.NFL, Sport.NBA, Sport.MLB]:
            games = await provider.get_live_games(sport)
            print(f"\n{sport.value.upper()} ({len(games)} games):")
            
            for game in games:
                status_emoji = "🔴" if game.status == "in" else "⚪"
                print(f"  {status_emoji} {game.away_team} @ {game.home_team}: {game.away_score}-{game.home_score}")
        
        if provider.session:
            await provider.session.close()
    
    asyncio.run(fetch_games())
    
    # Simulate a scoring event
    print("\n" + "=" * 70)
    print("🏈 SIMULATING SCORING EVENT")
    print("=" * 70)
    
    # Create mock game and event
    mock_game = LiveGame(
        game_id="test123",
        sport=Sport.NFL,
        home_team="Chiefs",
        away_team="Bills",
        home_score=21,
        away_score=14,
        period="Q3",
        time_remaining="8:45",
        status="in"
    )
    
    # Simulate previous score
    bot._previous_scores["test123"] = (14, 14)
    
    # Process the "new" score (Chiefs scored touchdown)
    print(f"\nPrevious: Chiefs 14 - Bills 14")
    print(f"New: Chiefs 21 - Bills 14 (TOUCHDOWN!)")
    
    async def simulate():
        await bot.monitor_game(mock_game)
    
    asyncio.run(simulate())
    
    # Statistics
    print("\n" + "=" * 70)
    print("📈 STATISTICS")
    print("=" * 70)
    
    stats = bot.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("⚠️ IMPORTANT NOTES")
    print("=" * 70)
    print("""
1. This bot uses ESPN's FREE API which has ~5-15 second delay
   For production, you need premium feeds (Sportradar/Genius Sports)

2. Polymarket added a 3-second order delay on sports markets
   This reduces but doesn't eliminate the arbitrage window

3. Start with PAPER TRADING to validate the edge exists

4. Consider running on a VPS in Amsterdam (4-6ms to Polymarket)
""")
    
    print("\n✅ Sports delay bot test complete!")


if __name__ == '__main__':
    main()
