"""
COLORADO SPORTS BETTING ARBITRAGE BOT
=====================================

Scans licensed Colorado sportsbooks for arbitrage opportunities.
Colorado legalized sports betting May 1, 2020.

Licensed CO Sportsbooks Monitored:
- DraftKings
- FanDuel
- BetMGM
- Caesars
- PointsBet
- Barstool/ESPN BET

Strategy:
- Fetch odds from multiple books via The Odds API
- Identify guaranteed profit when combined odds < 100%
- Calculate optimal bet sizing for equal profit
- Send alerts (manual execution required - no sportsbook APIs)

Example:
  DraftKings: Chiefs -110 (implied 52.4%)
  FanDuel: Bills +115 (implied 46.5%)
  Combined: 98.9% = 1.1% guaranteed profit

Requirements:
- The Odds API key (free tier: 500 requests/month)
- Colorado residency for legal betting
- Accounts at multiple sportsbooks

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import json
import logging
import sqlite3
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ColoradoSportsArb')


@dataclass
class BookOdds:
    """Odds from a sportsbook"""
    book: str
    team: str
    odds: int  # American odds
    implied_prob: float
    timestamp: datetime


@dataclass
class ArbOpportunity:
    """Sports arbitrage opportunity"""
    sport: str
    event: str
    commence_time: datetime
    outcome1: str
    outcome2: str
    book1: str
    book1_odds: int
    book1_implied: float
    book2: str
    book2_odds: int
    book2_implied: float
    total_implied: float
    arb_pct: float  # Profit percentage
    bet1_pct: float  # % of bankroll on outcome1
    bet2_pct: float  # % of bankroll on outcome2
    expected_profit: float  # For $100 total wagered


class ColoradoSportsArbBot:
    """
    Finds arbitrage opportunities across Colorado sportsbooks.

    Uses The Odds API to fetch real-time odds from licensed books.
    """

    # Colorado-licensed sportsbooks (The Odds API book keys)
    CO_BOOKS = [
        'draftkings',
        'fanduel',
        'betmgm',
        'caesars',
        'pointsbetus',
        'espnbet',
        'betrivers',
        'wynnbet',
    ]

    # Sports to monitor
    SPORTS = [
        'americanfootball_nfl',
        'americanfootball_ncaaf',
        'basketball_nba',
        'basketball_ncaab',
        'icehockey_nhl',
        'baseball_mlb',
        'mma_mixed_martial_arts',
        'soccer_usa_mls',
    ]

    # Minimum arb percentage to alert
    MIN_ARB_PCT = 0.5  # 0.5% minimum profit

    # The Odds API
    ODDS_API_BASE = "https://api.the-odds-api.com/v4"

    def __init__(self, bankroll: float = 100.0):
        self.bankroll = bankroll
        self.api_key = os.getenv('ODDS_API_KEY', '')
        self.opportunities: List[ArbOpportunity] = []
        self.alerts_sent: List[dict] = []

        # Database for tracking
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'sports_arb.db'
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

        logger.info(f"ColoradoSportsArbBot initialized - Bankroll: ${bankroll}")
        if not self.api_key:
            logger.warning("ODDS_API_KEY not set - using demo mode")

    def _init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arb_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                sport TEXT,
                event TEXT,
                book1 TEXT,
                book1_odds INTEGER,
                book2 TEXT,
                book2_odds INTEGER,
                arb_pct REAL,
                status TEXT DEFAULT 'found'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arb_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                opportunity_id INTEGER,
                timestamp TEXT,
                result TEXT,
                actual_profit REAL,
                FOREIGN KEY (opportunity_id) REFERENCES arb_opportunities(id)
            )
        """)

        conn.commit()
        conn.close()

    def american_to_implied(self, odds: int) -> float:
        """Convert American odds to implied probability"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    def implied_to_american(self, prob: float) -> int:
        """Convert implied probability to American odds"""
        if prob >= 0.5:
            return int(-100 * prob / (1 - prob))
        else:
            return int(100 * (1 - prob) / prob)

    def calculate_arb_stakes(self, prob1: float, prob2: float, total_stake: float) -> Tuple[float, float]:
        """
        Calculate optimal stake distribution for arbitrage.

        Returns (stake1, stake2) that guarantees equal profit regardless of outcome.
        """
        total_implied = prob1 + prob2

        if total_implied >= 1:
            return 0, 0  # No arb

        # Optimal stakes (inversely proportional to implied probability)
        stake1 = total_stake * prob1 / total_implied
        stake2 = total_stake * prob2 / total_implied

        return stake1, stake2

    def fetch_odds(self, sport: str) -> List[dict]:
        """Fetch odds from The Odds API"""
        if not self.api_key:
            return self._get_demo_odds(sport)

        url = f"{self.ODDS_API_BASE}/sports/{sport}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h',  # Moneyline
            'oddsFormat': 'american',
            'bookmakers': ','.join(self.CO_BOOKS),
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Odds API error for {sport}: {e}")
            return []

    def _get_demo_odds(self, sport: str) -> List[dict]:
        """Demo odds for testing without API key"""
        import random

        if 'nfl' not in sport and 'nba' not in sport:
            return []

        # Generate sample game
        teams = {
            'nfl': [('Kansas City Chiefs', 'Buffalo Bills'), ('Philadelphia Eagles', 'Dallas Cowboys')],
            'nba': [('Boston Celtics', 'Miami Heat'), ('Denver Nuggets', 'LA Lakers')],
        }

        sport_key = 'nfl' if 'nfl' in sport else 'nba'
        games = []

        for home, away in teams.get(sport_key, []):
            # Generate realistic odds with slight differences between books
            base_home = random.randint(-200, 200)

            bookmakers = []
            for book in ['draftkings', 'fanduel', 'betmgm']:
                home_odds = base_home + random.randint(-15, 15)
                # Away odds are roughly inverse
                if home_odds > 0:
                    away_odds = random.randint(-home_odds - 50, -home_odds + 50)
                else:
                    away_odds = random.randint(abs(home_odds) - 50, abs(home_odds) + 50)

                bookmakers.append({
                    'key': book,
                    'title': book.title(),
                    'markets': [{
                        'key': 'h2h',
                        'outcomes': [
                            {'name': home, 'price': home_odds},
                            {'name': away, 'price': away_odds},
                        ]
                    }]
                })

            games.append({
                'sport_key': sport,
                'commence_time': (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
                'home_team': home,
                'away_team': away,
                'bookmakers': bookmakers,
            })

        return games

    def find_arbs(self, events: List[dict]) -> List[ArbOpportunity]:
        """Find arbitrage opportunities in events"""
        arbs = []

        for event in events:
            bookmakers = event.get('bookmakers', [])
            if len(bookmakers) < 2:
                continue

            home_team = event.get('home_team', '')
            away_team = event.get('away_team', '')
            commence_time = event.get('commence_time', '')

            # Collect best odds for each outcome
            home_odds = []  # (book, odds, implied)
            away_odds = []

            for book in bookmakers:
                book_key = book.get('key', '')
                if book_key not in self.CO_BOOKS:
                    continue

                for market in book.get('markets', []):
                    if market.get('key') != 'h2h':
                        continue

                    for outcome in market.get('outcomes', []):
                        name = outcome.get('name', '')
                        price = outcome.get('price', 0)

                        if price == 0:
                            continue

                        implied = self.american_to_implied(price)

                        if name == home_team:
                            home_odds.append((book_key, price, implied))
                        elif name == away_team:
                            away_odds.append((book_key, price, implied))

            if not home_odds or not away_odds:
                continue

            # Find best odds for each side
            best_home = max(home_odds, key=lambda x: x[1] if x[1] > 0 else -1/x[1])
            best_away = max(away_odds, key=lambda x: x[1] if x[1] > 0 else -1/x[1])

            # Must be from different books
            if best_home[0] == best_away[0]:
                continue

            # Calculate total implied probability
            total_implied = best_home[2] + best_away[2]

            if total_implied < 1:
                # ARBITRAGE FOUND!
                arb_pct = (1 - total_implied) * 100

                if arb_pct >= self.MIN_ARB_PCT:
                    # Calculate stakes
                    stake1_pct = best_home[2] / total_implied * 100
                    stake2_pct = best_away[2] / total_implied * 100

                    # Expected profit per $100 wagered
                    expected_profit = 100 * (1 / total_implied - 1)

                    arbs.append(ArbOpportunity(
                        sport=event.get('sport_key', ''),
                        event=f"{away_team} @ {home_team}",
                        commence_time=datetime.fromisoformat(commence_time.replace('Z', '+00:00')) if commence_time else datetime.now(timezone.utc),
                        outcome1=home_team,
                        outcome2=away_team,
                        book1=best_home[0],
                        book1_odds=best_home[1],
                        book1_implied=best_home[2],
                        book2=best_away[0],
                        book2_odds=best_away[1],
                        book2_implied=best_away[2],
                        total_implied=total_implied,
                        arb_pct=arb_pct,
                        bet1_pct=stake1_pct,
                        bet2_pct=stake2_pct,
                        expected_profit=expected_profit
                    ))

        return sorted(arbs, key=lambda x: x.arb_pct, reverse=True)

    def scan_all_sports(self) -> List[ArbOpportunity]:
        """Scan all sports for arbitrage"""
        logger.info("Scanning Colorado sportsbooks for arbitrage...")

        all_arbs = []

        for sport in self.SPORTS:
            events = self.fetch_odds(sport)
            arbs = self.find_arbs(events)

            if arbs:
                logger.info(f"  {sport}: Found {len(arbs)} arbs")
                all_arbs.extend(arbs)

        self.opportunities = sorted(all_arbs, key=lambda x: x.arb_pct, reverse=True)

        # Store in database
        self._store_opportunities(self.opportunities)

        logger.info(f"Total: {len(self.opportunities)} arbitrage opportunities")
        return self.opportunities

    def _store_opportunities(self, opportunities: List[ArbOpportunity]):
        """Store opportunities in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for opp in opportunities:
            cursor.execute("""
                INSERT INTO arb_opportunities
                (timestamp, sport, event, book1, book1_odds, book2, book2_odds, arb_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                opp.sport,
                opp.event,
                opp.book1,
                opp.book1_odds,
                opp.book2,
                opp.book2_odds,
                opp.arb_pct
            ))

        conn.commit()
        conn.close()

    def format_alert(self, opp: ArbOpportunity) -> str:
        """Format opportunity as alert message"""
        return f"""
🎯 SPORTS ARB ALERT - {opp.arb_pct:.2f}% GUARANTEED PROFIT

{opp.event}
{opp.sport.upper()}
Game: {opp.commence_time.strftime('%b %d, %I:%M %p')}

📊 BETS TO PLACE:
  {opp.book1.upper()}: {opp.outcome1} ({opp.book1_odds:+d})
    → Bet {opp.bet1_pct:.1f}% of bankroll (${self.bankroll * opp.bet1_pct/100:.2f})

  {opp.book2.upper()}: {opp.outcome2} ({opp.book2_odds:+d})
    → Bet {opp.bet2_pct:.1f}% of bankroll (${self.bankroll * opp.bet2_pct/100:.2f})

💰 EXPECTED PROFIT: ${opp.expected_profit:.2f} per $100 wagered
   Total implied: {opp.total_implied:.2%}
"""

    def get_status(self) -> dict:
        """Get bot status"""
        return {
            'name': 'ColoradoSportsArbBot',
            'bankroll': self.bankroll,
            'api_configured': bool(self.api_key),
            'books_monitored': len(self.CO_BOOKS),
            'sports_monitored': len(self.SPORTS),
            'opportunities_found': len(self.opportunities),
            'last_scan': datetime.now(timezone.utc).isoformat()
        }


if __name__ == "__main__":
    print("=" * 60)
    print("COLORADO SPORTS BETTING ARBITRAGE SCANNER")
    print("=" * 60)

    bot = ColoradoSportsArbBot(bankroll=100.0)

    print(f"\nStatus: {bot.get_status()}")
    print(f"\nMonitoring {len(bot.CO_BOOKS)} CO-licensed sportsbooks:")
    for book in bot.CO_BOOKS:
        print(f"  - {book.title()}")

    print("\n--- Scanning for Arbitrage ---")
    opportunities = bot.scan_all_sports()

    if opportunities:
        print(f"\n🎯 Found {len(opportunities)} arbitrage opportunities!\n")

        for opp in opportunities[:5]:
            print(bot.format_alert(opp))
            print("-" * 40)
    else:
        print("\nNo arbitrage opportunities found at this time.")
        print("(This is normal - arbs are rare and close quickly)")

    print("\n" + "=" * 60)
    print("NOTE: Manual execution required - no sportsbook APIs available")
    print("      Place bets quickly as odds change rapidly")
    print("=" * 60)
