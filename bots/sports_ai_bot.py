"""
Sports AI Bot - Production-ready sports betting bot for master orchestrator

Integrates our completed SportsBettingAI class into the orchestrator framework
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_client import KalshiClient
from final_complete_ai import SportsBettingAI
from news_feeds import NewsAggregator

logger = logging.getLogger(__name__)

@dataclass
class SportsTrade:
    """Sports trade result"""
    player: str
    prop: str
    ticker: str  # Kalshi market ticker for execution
    direction: str  # 'YES' or 'NO'
    confidence: float
    market_price: float
    expected_value: float
    edge: float
    recommendation: str
    timestamp: datetime

class SportsAIBot:
    """Production sports betting bot for the orchestrator"""

    def __init__(self):
        self.name = "SportsAI"
        self.kalshi = KalshiClient()
        self.ai = SportsBettingAI(self.kalshi)
        self.trades = []

        # Configuration - Relaxed thresholds for more trade flow
        self.min_confidence = 55  # Minimum confidence for trades (lowered from 70)
        self.min_expected_value = 0.05  # Minimum 5% expected value (lowered from 10%)
        self.min_edge = 0.03  # Minimum 3% edge (lowered from 5%)

        # Weighted scoring configuration
        # Instead of strict AND logic, use weighted scoring where 2 of 3 strong metrics can qualify
        self.use_weighted_scoring = True
        self.min_weighted_score = 0.65  # Minimum weighted score to qualify (0-1 scale)
        self.weights = {
            'confidence': 0.40,      # 40% weight for confidence
            'expected_value': 0.35,  # 35% weight for EV
            'edge': 0.25             # 25% weight for edge
        }
        # Target values for normalization (what we'd consider "excellent")
        self.targets = {
            'confidence': 80,    # 80% confidence = perfect score for this metric
            'expected_value': 0.15,  # 15% EV = perfect score
            'edge': 0.08         # 8% edge = perfect score
        }

        # Near-miss logging - log opportunities within 10% of thresholds
        self.log_near_misses = True
        self.near_miss_threshold = 0.90  # 90% of minimum = near miss

        # Fallback news for when live feeds fail (always define this)
        self.fallback_news = [
            "Drake Maye completed 28 of 41 passes for 334 yards and 3 touchdowns in Patriots victory",
            "Kenneth Walker III rushed for 145 yards and 2 TDs in Seahawks dominant win",
            "Cooper Kupp catches 9 passes for 125 yards despite Rams loss",
            "Stefon Diggs injured during practice, questionable for Sunday game"
        ]

        # Initialize live news feed
        try:
            self.news_aggregator = NewsAggregator()
            self.use_live_news = True
            logger.info("Live news feed initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize live news feed: {e}. Using fallback data.")
            self.use_live_news = False

    def run_strategy(self) -> Dict[str, Any]:
        """Main entry point called by orchestrator"""

        logger.info(f"Running {self.name} sports betting analysis...")

        results = {
            'bot_name': self.name,
            'timestamp': datetime.now(timezone.utc),
            'trades': [],
            'status': 'success',
            'message': '',
            'total_opportunities': 0,
            'high_value_opportunities': 0
        }

        try:
            # Get current active players from Kalshi markets
            active_players = self._get_active_players()
            logger.info(f"Found {len(active_players)} active players in markets")

            # Get live news data
            news_items = self._get_news_data(active_players)
            logger.info(f"Retrieved {len(news_items)} relevant news items")

            # Analyze each piece of news
            total_opportunities = 0
            high_value_count = 0

            for news_item in news_items:
                # Extract news text (handle both string and NewsArticle objects)
                if hasattr(news_item, 'title') and hasattr(news_item, 'content'):
                    news_text = f"{news_item.title}. {news_item.content}"
                else:
                    news_text = str(news_item)

                opportunities = self._analyze_news(news_text, active_players)

                for opportunity in opportunities:
                    total_opportunities += 1

                    # Check if opportunity qualifies using weighted scoring or threshold logic
                    qualifies, weighted_score, score_breakdown = self._evaluate_opportunity(opportunity)

                    if qualifies:
                        high_value_count += 1

                        # Create trade record
                        trade = SportsTrade(
                            player=opportunity['player'],
                            prop=opportunity['prop'],
                            ticker=opportunity.get('ticker', ''),
                            direction=opportunity['bet_direction'],
                            confidence=opportunity['confidence'],
                            market_price=opportunity['market_price'],
                            expected_value=opportunity['expected_value'],
                            edge=opportunity['edge'],
                            recommendation=opportunity['recommendation'],
                            timestamp=datetime.now(timezone.utc)
                        )

                        self.trades.append(trade)

                        # Format trade for orchestrator execution
                        # Orchestrator expects: action, symbol, side, quantity, price, confidence
                        # Kalshi create_order expects: ticker, side ('yes'/'no'), action ('buy'), count, price (cents 1-99)

                        # Convert direction to side (YES -> 'yes', NO -> 'no')
                        side = 'yes' if trade.direction.upper() == 'YES' else 'no'

                        # Convert market_price to cents (Kalshi uses 1-99 cents)
                        # market_price is typically 0.0-1.0 probability, so multiply by 100
                        price_cents = max(1, min(99, int(trade.market_price * 100)))

                        # Calculate quantity based on confidence and allocation
                        # Default to 5 contracts for high confidence, 2 for moderate
                        base_quantity = 5 if trade.confidence >= 70 else 2

                        results['trades'].append({
                            # Orchestrator-required fields
                            'action': 'buy',  # Always buying positions
                            'symbol': trade.ticker,  # Market ticker
                            'side': side,  # 'yes' or 'no'
                            'quantity': base_quantity,  # Number of contracts
                            'price': price_cents / 100.0,  # Price as decimal for logging
                            'price_cents': price_cents,  # Price in cents for Kalshi API
                            'confidence': trade.confidence / 100.0,  # Float 0-1 for orchestrator

                            # Additional context fields
                            'player': trade.player,
                            'prop': trade.prop,
                            'ticker': trade.ticker,
                            'direction': trade.direction,
                            'expected_value': trade.expected_value,
                            'edge': trade.edge,
                            'recommendation': trade.recommendation,
                            'weighted_score': weighted_score,
                            'score_breakdown': score_breakdown,

                            # Metadata for logging/display
                            'display': {
                                'confidence': f"{trade.confidence:.0f}%",
                                'market_price': f"${trade.market_price:.2f}",
                                'expected_value': f"{trade.expected_value:.3f}",
                                'edge': f"{trade.edge:.1%}"
                            }
                        })

                        logger.info(f"QUALIFIED: {trade.player} {trade.prop} - {trade.direction} | "
                                   f"Conf: {trade.confidence:.0f}%, EV: {trade.expected_value:.1%}, Edge: {trade.edge:.1%} | "
                                   f"Weighted Score: {weighted_score:.2f}")
                    else:
                        # Check for near-misses and log them
                        self._log_near_miss(opportunity, weighted_score, score_breakdown)

            results['total_opportunities'] = total_opportunities
            results['high_value_opportunities'] = high_value_count
            results['message'] = f"Found {high_value_count}/{total_opportunities} high-value betting opportunities"

            logger.info(f"{self.name} completed: {high_value_count} high-value opportunities found")

        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            results['status'] = 'error'
            results['message'] = str(e)
            return results  # Return error status dict

        # If we have trades, return them as a list for orchestrator execution
        # The orchestrator expects either a list of signals or a single dict with 'action'
        if results['trades']:
            logger.info(f"Returning {len(results['trades'])} executable trades to orchestrator")
            return results['trades']  # Return list of trade signals
        else:
            # No trades - return status dict (orchestrator will pass through)
            return results

    def _get_news_data(self, active_players: List[str]) -> List[Any]:
        """Get news data from live feeds or fallback sources"""

        if self.use_live_news and self.news_aggregator:
            try:
                # Fetch sports news mentioning active players
                news_articles = self.news_aggregator.fetch_sports_news(
                    players=active_players,
                    limit=15
                )

                if news_articles:
                    logger.info(f"Retrieved {len(news_articles)} live news articles")
                    return news_articles
                else:
                    logger.warning("No live news articles found, using fallback")

            except Exception as e:
                logger.error(f"Failed to fetch live news: {e}")

        # Use fallback news
        logger.info("Using fallback news data")
        return self.fallback_news

    def _get_active_players(self) -> List[str]:
        """Get list of players who have active markets"""

        try:
            markets = self.kalshi.get_markets(limit=30)

            # Extract player names from market titles
            known_players = [
                "drake maye", "sam darnold", "kenneth walker", "rhamondre stevenson",
                "cooper kupp", "jaxon smith-njigba", "stefon diggs", "hunter henry",
                "kayshon boutte", "aj barner"
            ]

            active_players = []

            for player in known_players:
                for market in markets:
                    title = market.get('title', '').lower()
                    if player in title:
                        if player not in active_players:
                            active_players.append(player)
                        break

            return active_players

        except Exception as e:
            logger.error(f"Error getting active players: {e}")
            return ["drake maye", "kenneth walker"]  # Fallback

    def _evaluate_opportunity(self, opportunity: Dict[str, Any]) -> tuple:
        """
        Evaluate if an opportunity qualifies using weighted scoring or threshold logic.

        Returns:
            tuple: (qualifies: bool, weighted_score: float, score_breakdown: dict)
        """
        confidence = opportunity['confidence']
        expected_value = opportunity['expected_value']
        edge = opportunity['edge']

        # Calculate individual metric scores (0-1 scale, capped at 1.0)
        conf_score = min(1.0, confidence / self.targets['confidence'])
        ev_score = min(1.0, expected_value / self.targets['expected_value'])
        edge_score = min(1.0, edge / self.targets['edge'])

        # Calculate weighted score
        weighted_score = (
            conf_score * self.weights['confidence'] +
            ev_score * self.weights['expected_value'] +
            edge_score * self.weights['edge']
        )

        score_breakdown = {
            'confidence_score': f"{conf_score:.2f}",
            'ev_score': f"{ev_score:.2f}",
            'edge_score': f"{edge_score:.2f}",
            'method': 'weighted' if self.use_weighted_scoring else 'threshold'
        }

        if self.use_weighted_scoring:
            # Weighted scoring: qualify if weighted score meets minimum
            # AND at least 2 of 3 metrics meet their individual thresholds
            meets_confidence = confidence >= self.min_confidence
            meets_ev = expected_value >= self.min_expected_value
            meets_edge = edge >= self.min_edge

            metrics_met = sum([meets_confidence, meets_ev, meets_edge])

            # Qualify if:
            # 1. Weighted score >= minimum AND at least 2 metrics meet thresholds, OR
            # 2. All 3 metrics meet thresholds (original logic still works)
            qualifies = (
                (weighted_score >= self.min_weighted_score and metrics_met >= 2) or
                metrics_met == 3
            )

            score_breakdown['metrics_met'] = f"{metrics_met}/3"
            score_breakdown['meets_confidence'] = meets_confidence
            score_breakdown['meets_ev'] = meets_ev
            score_breakdown['meets_edge'] = meets_edge
        else:
            # Original strict AND logic (fallback)
            qualifies = (
                confidence >= self.min_confidence and
                expected_value >= self.min_expected_value and
                edge >= self.min_edge
            )

        return qualifies, weighted_score, score_breakdown

    def _log_near_miss(self, opportunity: Dict[str, Any], weighted_score: float, score_breakdown: dict):
        """Log opportunities that came close to qualifying (within 10% of thresholds)"""

        if not self.log_near_misses:
            return

        confidence = opportunity['confidence']
        expected_value = opportunity['expected_value']
        edge = opportunity['edge']

        # Check if each metric is a near-miss (within 10% of threshold)
        near_miss_conf = (confidence >= self.min_confidence * self.near_miss_threshold and
                         confidence < self.min_confidence)
        near_miss_ev = (expected_value >= self.min_expected_value * self.near_miss_threshold and
                        expected_value < self.min_expected_value)
        near_miss_edge = (edge >= self.min_edge * self.near_miss_threshold and
                          edge < self.min_edge)

        # Also log if weighted score was close
        near_miss_weighted = (weighted_score >= self.min_weighted_score * self.near_miss_threshold and
                              weighted_score < self.min_weighted_score)

        if near_miss_conf or near_miss_ev or near_miss_edge or near_miss_weighted:
            near_miss_reasons = []
            if near_miss_conf:
                near_miss_reasons.append(f"Conf {confidence:.0f}% (need {self.min_confidence}%)")
            if near_miss_ev:
                near_miss_reasons.append(f"EV {expected_value:.1%} (need {self.min_expected_value:.1%})")
            if near_miss_edge:
                near_miss_reasons.append(f"Edge {edge:.1%} (need {self.min_edge:.1%})")
            if near_miss_weighted:
                near_miss_reasons.append(f"WeightedScore {weighted_score:.2f} (need {self.min_weighted_score:.2f})")

            logger.info(f"NEAR-MISS: {opportunity['player']} {opportunity['prop']} - {opportunity['bet_direction']} | "
                       f"Almost qualified: {', '.join(near_miss_reasons)} | "
                       f"Metrics: Conf={confidence:.0f}%, EV={expected_value:.1%}, Edge={edge:.1%}")

    def _analyze_news(self, news: str, active_players: List[str]) -> List[Dict[str, Any]]:
        """Analyze news for betting opportunities"""

        opportunities = []

        # Check which players are mentioned in this news
        for player in active_players:
            if player.lower() in news.lower():

                try:
                    # Use our AI to analyze this opportunity
                    analysis = self.ai.analyze_betting_opportunity(news, player)

                    # Extract opportunities from recommendations
                    for rec in analysis.get('recommendations', []):
                        opportunities.append({
                            'player': player,
                            'prop': rec['prop'],
                            'ticker': rec.get('ticker', ''),  # Market ticker for execution
                            'bet_direction': rec['bet_direction'],
                            'confidence': rec['confidence'],
                            'market_price': rec['market_price'],
                            'expected_value': rec['expected_value'],
                            'edge': rec['edge'],
                            'recommendation': rec['recommendation'],
                            'news': news[:60] + "..."
                        })

                except Exception as e:
                    logger.warning(f"Failed to analyze {player} in news: {e}")
                    continue

        return opportunities

    def place_order(self, ticker: str, side: str, quantity: int, price: int) -> Dict[str, Any]:
        """
        Place an order via Kalshi API.

        Called by orchestrator's _execute_trade_via_bot for prediction markets.

        Args:
            ticker: Kalshi market ticker (e.g., 'KXPLAYER-...')
            side: 'yes' or 'no'
            quantity: Number of contracts (count)
            price: Price in cents (1-99)

        Returns:
            Order result from Kalshi API
        """
        if not ticker:
            logger.error("Cannot place order: no ticker provided")
            return {'error': 'No ticker provided', 'success': False}

        try:
            logger.info(f"[SportsAI] Placing order: {side.upper()} {quantity} contracts of {ticker} @ {price}c")

            # Use Kalshi client to create order
            # create_order(ticker, side, action, count, price, order_type)
            result = self.kalshi.create_order(
                ticker=ticker,
                side=side.lower(),  # 'yes' or 'no'
                action='buy',       # Always buying for initial positions
                count=quantity,
                price=price,        # In cents (1-99)
                order_type='limit'
            )

            logger.info(f"[SportsAI] Order placed successfully: {result}")
            return {'success': True, 'order': result}

        except Exception as e:
            logger.error(f"[SportsAI] Order failed: {e}")
            return {'error': str(e), 'success': False}

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""

        return {
            'name': self.name,
            'active_trades': len(self.trades),
            'last_run': self.trades[-1].timestamp if self.trades else None,
            'recent_opportunities': len([t for t in self.trades if
                                       (datetime.now(timezone.utc) - t.timestamp).seconds < 3600])
        }

    def close_positions(self):
        """Close all positions (placeholder for production)"""
        logger.info(f"{self.name}: No active positions to close (analysis-only bot)")

        # Shutdown news aggregator if it was initialized
        if hasattr(self, 'news_aggregator') and self.news_aggregator:
            try:
                self.news_aggregator.shutdown()
                logger.info("News aggregator shutdown successfully")
            except Exception as e:
                logger.warning(f"Failed to shutdown news aggregator: {e}")

        return True

def main():
    """Test the bot standalone"""

    logging.basicConfig(level=logging.INFO)

    print("Testing Sports AI Bot...")

    bot = SportsAIBot()
    results = bot.run_strategy()

    print(f"\nResults:")

    # Handle both list (trades found) and dict (status/error) return types
    if isinstance(results, list):
        # Trades were found - returned as list for orchestrator
        print(f"Status: success")
        print(f"Found {len(results)} executable trades")

        print(f"\nExecutable Trades:")
        for i, trade in enumerate(results, 1):
            print(f"{i}. {trade.get('player', 'N/A')} - {trade.get('prop', 'N/A')}")
            print(f"   Ticker: {trade.get('ticker', 'N/A')}")
            print(f"   Action: {trade.get('action')} {trade.get('side').upper()} @ {trade.get('price_cents')}c")
            print(f"   Quantity: {trade.get('quantity')} contracts")
            print(f"   Confidence: {trade.get('confidence', 0):.0%}")
            print(f"   EV: {trade.get('expected_value', 0):.3f}, Edge: {trade.get('edge', 0):.1%}")
            print()
    else:
        # Dict result (no trades or error)
        print(f"Status: {results.get('status', 'unknown')}")
        print(f"Message: {results.get('message', 'N/A')}")
        print(f"Opportunities analyzed: {results.get('total_opportunities', 0)}")
        print(f"High-value opportunities: {results.get('high_value_opportunities', 0)}")

    print(f"\nSports AI Bot test complete!")

if __name__ == "__main__":
    main()