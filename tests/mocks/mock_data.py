"""
Mock Data Generation for Testing
================================
Centralized mock data generation for all test scenarios.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np


@dataclass
class MockTradeData:
    """Mock trade data structure"""

    trade_id: str
    bot_name: str
    market: str
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl: float
    status: str
    reasoning: Optional[str] = None


@dataclass
class MockMarketData:
    """Mock market data structure"""

    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    change_24h: float
    timestamp: datetime


@dataclass
class MockNewsData:
    """Mock news article structure"""

    title: str
    content: str
    source: str
    timestamp: datetime
    sentiment_score: float
    keywords: List[str]
    relevance: float


class MockDataGenerator:
    """Generate realistic mock data for testing"""

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducible tests"""
        random.seed(seed)
        np.random.seed(seed)

        # Bot names from actual system
        self.bot_names = [
            "Kalshi-Fed",
            "Sports-AI",
            "OANDA-Forex",
            "Alpaca-Crypto-Momentum",
            "Cross-Market-Arbitrage",
            "Weather-Edge",
            "Sentiment-Bot",
            "Market-Scanner",
        ]

        # Trading symbols by market
        self.symbols = {
            "stocks": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA"],
            "crypto": ["BTCUSD", "ETHUSD", "SOLUSD", "ADAUSD", "DOTUSD"],
            "forex": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD"],
            "prediction": [
                "FEDRAISE-24FEB28",
                "INFLATION-24MAR",
                "BTCPRICE-24Q1",
                "ELECTION-2024",
            ],
        }

        # News sources
        self.news_sources = [
            "Reuters",
            "Bloomberg",
            "Wall Street Journal",
            "Financial Times",
            "CoinDesk",
        ]

    def generate_trade_history(
        self, num_trades: int = 100, days_back: int = 30
    ) -> List[MockTradeData]:
        """Generate realistic trade history"""
        trades = []
        base_time = datetime.now() - timedelta(days=days_back)

        for i in range(num_trades):
            bot_name = random.choice(self.bot_names)
            market = self._get_market_for_bot(bot_name)
            symbol = random.choice(self.symbols[market])

            # Generate realistic prices based on market
            base_price = self._get_base_price(market, symbol)
            entry_price = base_price * (1 + random.uniform(-0.02, 0.02))

            # Determine if trade is profitable (70% win rate)
            is_profitable = random.random() < 0.70

            if is_profitable:
                exit_price = entry_price * (1 + random.uniform(0.005, 0.05))
            else:
                exit_price = entry_price * (1 - random.uniform(0.005, 0.03))

            # Generate trade timing
            entry_time = base_time + timedelta(
                hours=random.randint(0, days_back * 24), minutes=random.randint(0, 59)
            )

            # 80% of trades are closed
            is_closed = random.random() < 0.80
            exit_time = (
                entry_time
                + timedelta(hours=random.randint(1, 48), minutes=random.randint(0, 59))
                if is_closed
                else None
            )

            # Calculate PnL
            if is_closed and exit_price:
                if market == "forex":
                    pnl = (exit_price - entry_price) * 10000  # Forex position sizing
                elif market == "prediction":
                    pnl = (exit_price - entry_price) * 10  # Prediction market contracts
                else:
                    pnl = (exit_price - entry_price) * 10  # Stock/crypto shares
            else:
                pnl = 0.0

            trades.append(
                MockTradeData(
                    trade_id=f"{bot_name.lower()}_{entry_time.strftime('%Y%m%d_%H%M%S')}_{i}",
                    bot_name=bot_name,
                    market=market,
                    symbol=symbol,
                    side=random.choice(["buy", "sell"]),
                    entry_price=round(entry_price, 4),
                    exit_price=round(exit_price, 4) if exit_price else None,
                    quantity=self._get_position_size(market),
                    entry_time=entry_time,
                    exit_time=exit_time,
                    pnl=round(pnl, 2),
                    status="closed" if is_closed else "open",
                    reasoning=self._generate_trade_reasoning(bot_name, symbol),
                )
            )

        return trades

    def generate_market_data(
        self, symbols: List[str], days: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """Generate realistic OHLCV market data"""
        data = {}

        for symbol in symbols:
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=days), periods=days, freq="D"
            )

            # Base price depends on asset type
            if "BTC" in symbol:
                base_price = 45000
                volatility = 0.04
            elif "ETH" in symbol:
                base_price = 2800
                volatility = 0.05
            elif symbol in ["SPY", "QQQ"]:
                base_price = 450
                volatility = 0.015
            else:
                base_price = 100
                volatility = 0.02

            # Generate price series with realistic characteristics
            returns = np.random.normal(0.0005, volatility, days)  # Small positive drift
            prices = [base_price]

            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            # Generate OHLCV data
            opens = prices[:-1]
            closes = prices[1:]

            highs = [c * (1 + abs(np.random.normal(0, 0.01))) for c in closes]
            lows = [c * (1 - abs(np.random.normal(0, 0.01))) for c in closes]

            volumes = [int(np.random.normal(1000000, 200000)) for _ in range(days)]

            data[symbol] = pd.DataFrame(
                {
                    "Open": opens,
                    "High": highs,
                    "Low": lows,
                    "Close": closes,
                    "Volume": volumes,
                },
                index=dates[1:],
            )

        return data

    def generate_news_articles(
        self, num_articles: int = 50, days_back: int = 7
    ) -> List[MockNewsData]:
        """Generate realistic news articles"""
        articles = []
        base_time = datetime.now() - timedelta(days=days_back)

        # News templates by category
        templates = {
            "fed": {
                "titles": [
                    "Federal Reserve Keeps Interest Rates Steady",
                    "Fed Chair Powell Signals Cautious Approach on Rates",
                    "Central Bank Officials Debate Next Policy Move",
                    "Interest Rate Decision Expected This Week",
                ],
                "keywords": [
                    "federal reserve",
                    "interest rates",
                    "monetary policy",
                    "jerome powell",
                ],
                "sentiment_range": (-0.3, 0.3),
            },
            "inflation": {
                "titles": [
                    "Inflation Data Shows Continued Moderation",
                    "Consumer Prices Rise Less Than Expected",
                    "Core CPI Remains Elevated Despite Progress",
                    "Price Pressures Ease in Key Sectors",
                ],
                "keywords": ["inflation", "cpi", "consumer prices", "price index"],
                "sentiment_range": (-0.4, 0.2),
            },
            "crypto": {
                "titles": [
                    "Bitcoin Rallies on Institutional Demand",
                    "Cryptocurrency Market Shows Resilience",
                    "Ethereum Upgrade Boosts Network Activity",
                    "Regulatory Clarity Drives Crypto Adoption",
                ],
                "keywords": ["bitcoin", "ethereum", "cryptocurrency", "blockchain"],
                "sentiment_range": (-0.2, 0.8),
            },
            "markets": {
                "titles": [
                    "Stock Market Reaches New Highs on Tech Rally",
                    "Market Volatility Increases Amid Economic Uncertainty",
                    "Technology Stocks Lead Broad Market Advance",
                    "Investors Weigh Economic Data and Earnings",
                ],
                "keywords": ["stock market", "technology", "earnings", "investors"],
                "sentiment_range": (-0.3, 0.5),
            },
        }

        for i in range(num_articles):
            category = random.choice(list(templates.keys()))
            template = templates[category]

            title = random.choice(template["titles"])
            content = self._generate_article_content(title, template["keywords"])
            source = random.choice(self.news_sources)

            timestamp = base_time + timedelta(
                hours=random.randint(0, days_back * 24), minutes=random.randint(0, 59)
            )

            sentiment_range = template["sentiment_range"]
            sentiment = random.uniform(sentiment_range[0], sentiment_range[1])

            relevance = (
                random.uniform(0.6, 1.0)
                if category in ["fed", "inflation"]
                else random.uniform(0.3, 0.8)
            )

            articles.append(
                MockNewsData(
                    title=title,
                    content=content,
                    source=source,
                    timestamp=timestamp,
                    sentiment_score=round(sentiment, 3),
                    keywords=template["keywords"],
                    relevance=round(relevance, 3),
                )
            )

        return articles

    def generate_kalshi_markets(self, num_markets: int = 20) -> List[Dict[str, Any]]:
        """Generate realistic Kalshi prediction markets"""
        markets = []

        categories = {
            "Economics": [
                "Will the Fed raise rates by {date}?",
                "Will inflation exceed 3% in {month}?",
                "Will unemployment be below 4% in {quarter}?",
            ],
            "Politics": [
                "Will {candidate} win the {state} primary?",
                "Will Congress pass the {bill} by {date}?",
                "Will the President sign {legislation}?",
            ],
            "Crypto": [
                "Will Bitcoin close above ${price} in {timeframe}?",
                "Will Ethereum reach ${price} by {date}?",
                "Will a Bitcoin ETF be approved in {year}?",
            ],
            "Sports": [
                "Will {team} make the playoffs?",
                "Will {player} score over {points} points this season?",
                "Will {team} win the championship?",
            ],
        }

        for i in range(num_markets):
            category = random.choice(list(categories.keys()))
            template = random.choice(categories[category])

            # Fill in template variables
            title = self._fill_template_variables(template, category)

            # Generate realistic pricing
            true_prob = random.uniform(0.15, 0.85)
            market_inefficiency = random.uniform(-0.15, 0.15)
            yes_price = max(5, min(95, int((true_prob + market_inefficiency) * 100)))
            no_price = 100 - yes_price

            # Generate volume and interest
            volume = random.randint(100, 10000)
            open_interest = random.randint(volume, volume * 5)

            # Generate expiration
            close_time = datetime.now() + timedelta(
                days=random.randint(1, 365), hours=random.randint(0, 23)
            )

            markets.append(
                {
                    "ticker": f"{category.upper()}-{i:03d}",
                    "title": title,
                    "category": category,
                    "outcome_1": "YES",
                    "outcome_2": "NO",
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "volume": volume,
                    "open_interest": open_interest,
                    "close_time": close_time.isoformat(),
                    "true_probability": true_prob,
                    "edge": abs(market_inefficiency),
                }
            )

        return markets

    def generate_bot_status(self) -> Dict[str, Any]:
        """Generate realistic bot status data"""
        by_market = {}
        markets = ["prediction", "forex", "crypto", "stocks", "other"]

        for market in markets:
            market_bots = [
                bot for bot in self.bot_names if self._get_market_for_bot(bot) == market
            ]

            running_bots = []
            for bot_name in market_bots:
                status = random.choice(["running", "waiting", "error"])

                running_bots.append(
                    {
                        "name": bot_name,
                        "status": status,
                        "last_run": (
                            datetime.now() - timedelta(minutes=random.randint(1, 60))
                        ).isoformat(),
                        "trades_today": random.randint(0, 10),
                        "pnl_today": round(random.uniform(-50, 200), 2),
                        "error": "Connection timeout" if status == "error" else None,
                    }
                )

            by_market[market] = {
                "total": len(market_bots),
                "running": len(
                    [b for b in running_bots if b["status"] in ["running", "waiting"]]
                ),
                "error": len([b for b in running_bots if b["status"] == "error"]),
                "bots": running_bots,
            }

        total_trades = sum(
            bot["trades_today"]
            for market_data in by_market.values()
            for bot in market_data["bots"]
        )
        total_pnl = sum(
            bot["pnl_today"]
            for market_data in by_market.values()
            for bot in market_data["bots"]
        )
        wins = int(total_trades * random.uniform(0.6, 0.8))

        return {
            "timestamp": datetime.now().isoformat(),
            "paper_mode": True,
            "starting_capital": 10000.0,
            "current_capital": 10000.0 + total_pnl,
            "total_bots": len(self.bot_names),
            "running": True,
            "by_market": by_market,
            "today": {
                "trades": total_trades,
                "wins": wins,
                "total_pnl": round(total_pnl, 2),
            },
            "open_trades": random.randint(0, 5),
        }

    def generate_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Generate performance metrics"""
        daily_pnl = []
        cumulative_pnl = 0

        for i in range(days):
            date = (datetime.now() - timedelta(days=days - i)).strftime("%Y-%m-%d")
            daily = random.uniform(-100, 300)
            trades = random.randint(0, 15)

            cumulative_pnl += daily

            daily_pnl.append({"date": date, "pnl": round(daily, 2), "trades": trades})

        _total_trades = sum(day["pnl"] for day in daily_pnl if day["pnl"] > 0)  # noqa: F841
        winning_trades = len([day for day in daily_pnl if day["pnl"] > 0])

        return {
            "daily_pnl": daily_pnl,
            "cumulative_pnl": round(cumulative_pnl, 2),
            "total_trades": sum(day["trades"] for day in daily_pnl),
            "win_rate": round(winning_trades / max(1, len(daily_pnl)), 3),
            "sharpe_ratio": round(random.uniform(0.5, 2.5), 2),
            "max_drawdown": round(min(day["pnl"] for day in daily_pnl), 2),
            "best_bot": random.choice(self.bot_names),
            "worst_bot": random.choice(self.bot_names),
        }

    # Helper methods
    def _get_market_for_bot(self, bot_name: str) -> str:
        """Map bot name to market type"""
        if "Kalshi" in bot_name or "Sports" in bot_name or "Weather" in bot_name:
            return "prediction"
        elif "OANDA" in bot_name or "Forex" in bot_name:
            return "forex"
        elif "Crypto" in bot_name or "BTC" in bot_name:
            return "crypto"
        elif "Alpaca" in bot_name or "Stock" in bot_name:
            return "stocks"
        else:
            return "other"

    def _get_base_price(self, market: str, symbol: str) -> float:
        """Get realistic base price for symbol"""
        if market == "forex":
            return 1.0 + random.uniform(-0.5, 0.5)
        elif market == "crypto":
            if "BTC" in symbol:
                return 45000
            elif "ETH" in symbol:
                return 2800
            else:
                return random.uniform(0.1, 100)
        elif market == "prediction":
            return random.uniform(5, 95)  # Prediction market prices in cents
        else:  # stocks
            return random.uniform(10, 500)

    def _get_position_size(self, market: str) -> float:
        """Get realistic position size for market"""
        if market == "forex":
            return random.randint(1000, 100000)  # Forex units
        elif market == "prediction":
            return random.randint(1, 100)  # Prediction market contracts
        else:
            return random.randint(1, 1000)  # Shares

    def _generate_trade_reasoning(self, bot_name: str, symbol: str) -> str:
        """Generate realistic trade reasoning"""
        reasons = {
            "Kalshi-Fed": [
                "Fed rate decision imminent, market mispriced",
                "Economic data suggests policy shift",
                "Market sentiment favors dovish stance",
            ],
            "Sports-AI": [
                "Team performance analytics show edge",
                "Historical matchup data suggests value",
                "Injury reports create opportunity",
            ],
            "OANDA-Forex": [
                "Technical breakout pattern confirmed",
                "Central bank divergence trade",
                "Risk-on sentiment supports position",
            ],
        }

        bot_reasons = reasons.get(
            bot_name, ["Algorithm detected opportunity", "Technical signal triggered"]
        )
        return random.choice(bot_reasons)

    def _generate_article_content(self, title: str, keywords: List[str]) -> str:
        """Generate realistic article content"""
        sentences = [
            f"The {random.choice(keywords)} announcement today has significant implications for market participants.",
            "Financial analysts are closely monitoring the situation for potential trading opportunities.",
            "Economic indicators suggest this development could impact multiple asset classes.",
            "Market volatility has increased following the news, creating both risks and opportunities.",
            "Institutional investors are adjusting their portfolios in response to these developments.",
            "The broader economic implications of this news are still being assessed by experts.",
        ]

        # Use 3-5 sentences for content
        num_sentences = random.randint(3, 5)
        content_sentences = random.sample(sentences, num_sentences)

        return f"{title}. " + " ".join(content_sentences)

    def _fill_template_variables(self, template: str, category: str) -> str:
        """Fill template variables with realistic values"""
        replacements = {
            "{date}": (
                datetime.now() + timedelta(days=random.randint(1, 180))
            ).strftime("%B %d"),
            "{month}": (
                datetime.now() + timedelta(days=random.randint(1, 90))
            ).strftime("%B"),
            "{quarter}": f"Q{random.randint(1, 4)} 2024",
            "{year}": str(datetime.now().year),
            "{timeframe}": random.choice(["Q1 2024", "this quarter", "year-end"]),
            "{price}": str(random.randint(40000, 80000)),
            "{candidate}": random.choice(["Smith", "Johnson", "Williams"]),
            "{state}": random.choice(["California", "Texas", "Florida"]),
            "{bill}": random.choice(
                ["Infrastructure Bill", "Healthcare Act", "Tax Reform"]
            ),
            "{legislation}": random.choice(
                ["the spending bill", "new regulations", "trade agreement"]
            ),
            "{team}": random.choice(["Lakers", "Warriors", "Celtics"]),
            "{player}": random.choice(
                ["LeBron James", "Stephen Curry", "Jayson Tatum"]
            ),
            "{points}": str(random.randint(20, 35)),
        }

        result = template
        for var, value in replacements.items():
            result = result.replace(var, value)

        return result


# Convenience functions for quick data generation
def generate_sample_trades(count: int = 50) -> List[Dict[str, Any]]:
    """Generate sample trades as dictionaries"""
    generator = MockDataGenerator()
    trades = generator.generate_trade_history(count)
    return [asdict(trade) for trade in trades]


def generate_sample_news(count: int = 20) -> List[Dict[str, Any]]:
    """Generate sample news articles as dictionaries"""
    generator = MockDataGenerator()
    articles = generator.generate_news_articles(count)
    return [asdict(article) for article in articles]


def generate_sample_markets(count: int = 15) -> List[Dict[str, Any]]:
    """Generate sample Kalshi markets"""
    generator = MockDataGenerator()
    return generator.generate_kalshi_markets(count)


def generate_sample_status() -> Dict[str, Any]:
    """Generate sample orchestrator status"""
    generator = MockDataGenerator()
    return generator.generate_bot_status()


if __name__ == "__main__":
    # Demo usage
    generator = MockDataGenerator()

    print("Generating sample data...")

    trades = generator.generate_trade_history(10)
    print(f"\nGenerated {len(trades)} trades")
    for trade in trades[:3]:
        print(f"  {trade.bot_name}: {trade.symbol} {trade.side} @ {trade.entry_price}")

    news = generator.generate_news_articles(5)
    print(f"\nGenerated {len(news)} news articles")
    for article in news:
        print(f"  {article.source}: {article.title}")

    markets = generator.generate_kalshi_markets(5)
    print(f"\nGenerated {len(markets)} Kalshi markets")
    for market in markets:
        print(f"  {market['ticker']}: {market['title']}")

    print("\nSample data generation complete!")
