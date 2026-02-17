"""
KALSHI HOURLY CRYPTO BOT
========================
High-frequency trading on Kalshi's hourly Bitcoin and Ethereum markets.

Market Tickers:
- KXBTCD: Bitcoin hourly price brackets
- KXETHD: Ethereum hourly price brackets

Opportunity: 24 resolutions/day x 7 days = 168 trades/week PER ASSET

Strategy:
- Calculate momentum from Alpaca crypto prices (5-min, 15-min, 1-hour)
- If strong momentum UP → Buy YES on higher bracket
- If strong momentum DOWN → Buy YES on lower bracket
- Use LIMIT ORDERS ONLY (0% maker fee)

Risk Management:
- $200 allocation
- 2% risk per trade = $4 max loss
- Kelly criterion position sizing

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Kalshi client
try:
    from bots.kalshi_client import KalshiClient
    KALSHI_AVAILABLE = True
except ImportError:
    KALSHI_AVAILABLE = False
    print("Warning: KalshiClient not available")

# Import Alpaca for crypto prices
try:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_CRYPTO_AVAILABLE = True
except ImportError:
    ALPACA_CRYPTO_AVAILABLE = False
    print("Warning: Alpaca crypto data not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KalshiHourlyCrypto')


class MomentumSignal(Enum):
    STRONG_UP = "strong_up"
    WEAK_UP = "weak_up"
    NEUTRAL = "neutral"
    WEAK_DOWN = "weak_down"
    STRONG_DOWN = "strong_down"


@dataclass
class CryptoMomentum:
    """Momentum analysis for a crypto asset"""
    symbol: str
    current_price: float
    price_5min_ago: float
    price_15min_ago: float
    price_1hr_ago: float
    rsi_5min: float
    momentum_5min: float  # % change
    momentum_15min: float
    momentum_1hr: float
    signal: MomentumSignal
    confidence: float  # 0-1


@dataclass
class KalshiOpportunity:
    """Trading opportunity on Kalshi"""
    ticker: str
    asset: str  # BTC or ETH
    bracket_low: float
    bracket_high: float
    current_price: float
    yes_bid: int  # cents
    yes_ask: int
    no_bid: int
    no_ask: int
    expires_at: datetime
    time_to_expiry_mins: float
    edge: float  # calculated edge
    recommended_side: str  # "yes" or "no"
    recommended_price: int
    contracts: int


class KalshiHourlyCryptoBot:
    """
    High-frequency Kalshi crypto trading bot.

    Trades hourly Bitcoin (KXBTCD) and Ethereum (KXETHD) markets
    using momentum signals from Alpaca crypto data.
    """

    # Series tickers for crypto hourly markets
    CRYPTO_SERIES = {
        'BTC': 'KXBTCD',  # Bitcoin daily/hourly
        'ETH': 'KXETHD',  # Ethereum daily/hourly
    }

    # Alpaca symbols
    ALPACA_SYMBOLS = {
        'BTC': 'BTC/USD',
        'ETH': 'ETH/USD',
    }

    # Strategy parameters
    MOMENTUM_THRESHOLD_STRONG = 0.5  # 0.5% = strong momentum
    MOMENTUM_THRESHOLD_WEAK = 0.2    # 0.2% = weak momentum
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    MIN_EDGE = 5  # Minimum 5% edge to trade
    MAX_CONTRACTS_PER_TRADE = 10

    def __init__(
        self,
        capital: float = 200.0,
        risk_per_trade: float = 0.02,  # 2%
        paper_mode: bool = None
    ):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        # Safe default: read from environment, default to PAPER if not set
        if paper_mode is None:
            paper_mode = os.getenv('PAPER_MODE', 'true').lower() == 'true'
        self.paper_mode = paper_mode
        self.max_loss_per_trade = capital * risk_per_trade  # $4

        # Initialize clients
        self.kalshi = None
        self.alpaca_crypto = None

        if KALSHI_AVAILABLE and not paper_mode:
            try:
                self.kalshi = KalshiClient()
                logger.info("Kalshi client initialized")
            except Exception as e:
                logger.warning(f"Kalshi client failed: {e}")

        if ALPACA_CRYPTO_AVAILABLE:
            try:
                # Crypto data doesn't require API keys
                self.alpaca_crypto = CryptoHistoricalDataClient()
                logger.info("Alpaca crypto client initialized")
            except Exception as e:
                logger.warning(f"Alpaca crypto client failed: {e}")

        # Track positions and trades
        self.positions: Dict[str, Dict] = {}
        self.trades_today: List[Dict] = []
        self.pnl_today: float = 0.0

        logger.info(f"KalshiHourlyCryptoBot initialized - Capital: ${capital}, Paper: {paper_mode}")

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI from price series"""
        if len(prices) < period + 1:
            return 50.0  # Neutral

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    def get_crypto_momentum(self, asset: str) -> Optional[CryptoMomentum]:
        """
        Get momentum analysis for a crypto asset using Alpaca data.
        """
        if not self.alpaca_crypto:
            return self._get_mock_momentum(asset)

        symbol = self.ALPACA_SYMBOLS.get(asset)
        if not symbol:
            return None

        try:
            # Get 5-minute bars for last 2 hours
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=2)

            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,  # 1-min bars
                start=start,
                end=end
            )

            bars = self.alpaca_crypto.get_crypto_bars(request)
            df = bars.df if hasattr(bars, 'df') else pd.DataFrame(bars[symbol])

            if len(df) < 60:
                logger.warning(f"Insufficient data for {asset}")
                return self._get_mock_momentum(asset)

            # Calculate metrics
            current_price = float(df['close'].iloc[-1])
            price_5min_ago = float(df['close'].iloc[-6]) if len(df) > 5 else current_price
            price_15min_ago = float(df['close'].iloc[-16]) if len(df) > 15 else current_price
            price_1hr_ago = float(df['close'].iloc[-61]) if len(df) > 60 else current_price

            momentum_5min = ((current_price - price_5min_ago) / price_5min_ago) * 100
            momentum_15min = ((current_price - price_15min_ago) / price_15min_ago) * 100
            momentum_1hr = ((current_price - price_1hr_ago) / price_1hr_ago) * 100

            # RSI on 5-min closes
            rsi_5min = self.calculate_rsi(df['close'].iloc[-30:], period=14)

            # Determine signal
            signal = self._classify_momentum(momentum_5min, momentum_15min, momentum_1hr, rsi_5min)
            confidence = self._calculate_confidence(momentum_5min, momentum_15min, momentum_1hr, rsi_5min)

            return CryptoMomentum(
                symbol=asset,
                current_price=current_price,
                price_5min_ago=price_5min_ago,
                price_15min_ago=price_15min_ago,
                price_1hr_ago=price_1hr_ago,
                rsi_5min=rsi_5min,
                momentum_5min=momentum_5min,
                momentum_15min=momentum_15min,
                momentum_1hr=momentum_1hr,
                signal=signal,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error getting momentum for {asset}: {e}")
            return self._get_mock_momentum(asset)

    def _get_mock_momentum(self, asset: str) -> CryptoMomentum:
        """Generate mock momentum data for testing"""
        import random

        base_price = 95000 if asset == 'BTC' else 3200
        current = base_price * (1 + random.uniform(-0.01, 0.01))

        momentum = random.uniform(-0.8, 0.8)
        rsi = random.uniform(30, 70)

        return CryptoMomentum(
            symbol=asset,
            current_price=current,
            price_5min_ago=current * (1 - momentum/200),
            price_15min_ago=current * (1 - momentum/100),
            price_1hr_ago=current * (1 - momentum/50),
            rsi_5min=rsi,
            momentum_5min=momentum,
            momentum_15min=momentum * 1.5,
            momentum_1hr=momentum * 2,
            signal=self._classify_momentum(momentum, momentum*1.5, momentum*2, rsi),
            confidence=random.uniform(0.5, 0.9)
        )

    def _classify_momentum(
        self,
        mom_5min: float,
        mom_15min: float,
        mom_1hr: float,
        rsi: float
    ) -> MomentumSignal:
        """Classify momentum into signal categories"""

        # Weight recent momentum more heavily
        weighted_mom = (mom_5min * 0.5) + (mom_15min * 0.3) + (mom_1hr * 0.2)

        # RSI confirmation
        rsi_bullish = rsi < self.RSI_OVERSOLD
        rsi_bearish = rsi > self.RSI_OVERBOUGHT

        if weighted_mom > self.MOMENTUM_THRESHOLD_STRONG:
            return MomentumSignal.STRONG_UP if not rsi_bearish else MomentumSignal.WEAK_UP
        elif weighted_mom > self.MOMENTUM_THRESHOLD_WEAK:
            return MomentumSignal.WEAK_UP
        elif weighted_mom < -self.MOMENTUM_THRESHOLD_STRONG:
            return MomentumSignal.STRONG_DOWN if not rsi_bullish else MomentumSignal.WEAK_DOWN
        elif weighted_mom < -self.MOMENTUM_THRESHOLD_WEAK:
            return MomentumSignal.WEAK_DOWN
        else:
            return MomentumSignal.NEUTRAL

    def _calculate_confidence(
        self,
        mom_5min: float,
        mom_15min: float,
        mom_1hr: float,
        rsi: float
    ) -> float:
        """Calculate confidence score 0-1"""

        # All timeframes agree = high confidence
        signs = [
            1 if mom_5min > 0 else -1,
            1 if mom_15min > 0 else -1,
            1 if mom_1hr > 0 else -1
        ]

        agreement = abs(sum(signs)) / 3  # 0.33, 0.67, or 1.0

        # Stronger momentum = higher confidence
        strength = min(abs(mom_5min) / 1.0, 1.0)

        # RSI extremes add confidence
        rsi_confidence = 0.2 if (rsi < 30 or rsi > 70) else 0

        return min((agreement * 0.5) + (strength * 0.3) + rsi_confidence, 1.0)

    def get_hourly_markets(self, asset: str) -> List[Dict]:
        """
        Get available hourly markets for an asset from Kalshi.
        """
        if not self.kalshi:
            return self._get_mock_markets(asset)

        series_ticker = self.CRYPTO_SERIES.get(asset)
        if not series_ticker:
            return []

        try:
            markets = self.kalshi.get_markets(series_ticker=series_ticker, status='open')

            # Filter for hourly markets expiring soon
            hourly_markets = []
            now = datetime.now(timezone.utc)

            for market in markets:
                # Check if it's an hourly market (expires within 2 hours)
                expiry_str = market.get('close_time') or market.get('expiration_time')
                if expiry_str:
                    expiry = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
                    time_to_expiry = (expiry - now).total_seconds() / 60

                    if 5 < time_to_expiry < 120:  # 5 mins to 2 hours
                        hourly_markets.append({
                            'ticker': market['ticker'],
                            'title': market.get('title', ''),
                            'yes_bid': market.get('yes_bid', 0),
                            'yes_ask': market.get('yes_ask', 100),
                            'no_bid': market.get('no_bid', 0),
                            'no_ask': market.get('no_ask', 100),
                            'expiry': expiry,
                            'time_to_expiry_mins': time_to_expiry
                        })

            return hourly_markets

        except Exception as e:
            logger.error(f"Error getting Kalshi markets: {e}")
            return self._get_mock_markets(asset)

    def _get_mock_markets(self, asset: str) -> List[Dict]:
        """Generate mock market data for testing"""
        import random

        now = datetime.now(timezone.utc)
        markets = []

        # Generate 3 mock hourly brackets
        base = 95000 if asset == 'BTC' else 3200

        for i in range(3):
            expiry = now + timedelta(minutes=30 + i*30)
            bracket = base + (i - 1) * (500 if asset == 'BTC' else 20)

            yes_price = random.randint(30, 70)

            markets.append({
                'ticker': f'KX{asset}D-{int(expiry.timestamp())}',
                'title': f'{asset} > ${bracket:,} at {expiry.strftime("%H:%M")} UTC',
                'bracket_value': bracket,
                'yes_bid': yes_price - 2,
                'yes_ask': yes_price + 2,
                'no_bid': 100 - yes_price - 2,
                'no_ask': 100 - yes_price + 2,
                'expiry': expiry,
                'time_to_expiry_mins': (expiry - now).total_seconds() / 60
            })

        return markets

    def find_opportunities(self, asset: str) -> List[KalshiOpportunity]:
        """
        Find trading opportunities by combining momentum with Kalshi prices.
        """
        opportunities = []

        # Get momentum
        momentum = self.get_crypto_momentum(asset)
        if not momentum:
            return []

        # Get markets
        markets = self.get_hourly_markets(asset)
        if not markets:
            return []

        for market in markets:
            # Parse bracket value from title or ticker
            bracket_value = market.get('bracket_value', momentum.current_price)

            # Calculate fair value based on momentum
            fair_yes = self._estimate_fair_value(
                momentum,
                bracket_value,
                market['time_to_expiry_mins']
            )

            # Check for edge
            yes_ask = market['yes_ask']
            no_ask = market['no_ask']

            yes_edge = fair_yes - yes_ask
            no_edge = (100 - fair_yes) - no_ask

            # Determine best trade
            if yes_edge > self.MIN_EDGE and momentum.signal in [MomentumSignal.STRONG_UP, MomentumSignal.WEAK_UP]:
                edge = yes_edge
                side = 'yes'
                price = yes_ask
            elif no_edge > self.MIN_EDGE and momentum.signal in [MomentumSignal.STRONG_DOWN, MomentumSignal.WEAK_DOWN]:
                edge = no_edge
                side = 'no'
                price = no_ask
            else:
                continue

            # Calculate position size (Kelly-inspired)
            contracts = self._calculate_position_size(edge, price, momentum.confidence)

            if contracts > 0:
                opportunities.append(KalshiOpportunity(
                    ticker=market['ticker'],
                    asset=asset,
                    bracket_low=bracket_value - 500 if asset == 'BTC' else bracket_value - 20,
                    bracket_high=bracket_value,
                    current_price=momentum.current_price,
                    yes_bid=market['yes_bid'],
                    yes_ask=yes_ask,
                    no_bid=market['no_bid'],
                    no_ask=no_ask,
                    expires_at=market['expiry'],
                    time_to_expiry_mins=market['time_to_expiry_mins'],
                    edge=edge,
                    recommended_side=side,
                    recommended_price=price,
                    contracts=contracts
                ))

        # Sort by edge (best first)
        opportunities.sort(key=lambda x: x.edge, reverse=True)

        return opportunities

    def _estimate_fair_value(
        self,
        momentum: CryptoMomentum,
        bracket_value: float,
        time_to_expiry_mins: float
    ) -> float:
        """
        Estimate fair probability that price > bracket at expiry.
        """
        current = momentum.current_price

        # Distance to bracket as % of current price
        distance_pct = (bracket_value - current) / current * 100

        # Base probability (assuming random walk)
        base_prob = 50 - (distance_pct * 10)  # 10% per 1% distance

        # Adjust for momentum
        momentum_adj = 0
        if momentum.signal == MomentumSignal.STRONG_UP:
            momentum_adj = 15 * momentum.confidence
        elif momentum.signal == MomentumSignal.WEAK_UP:
            momentum_adj = 7 * momentum.confidence
        elif momentum.signal == MomentumSignal.STRONG_DOWN:
            momentum_adj = -15 * momentum.confidence
        elif momentum.signal == MomentumSignal.WEAK_DOWN:
            momentum_adj = -7 * momentum.confidence

        # Time decay (mean reversion more likely with more time)
        time_factor = min(time_to_expiry_mins / 60, 1.0)
        momentum_adj *= time_factor

        fair_value = base_prob + momentum_adj
        return max(5, min(95, fair_value))  # Clamp 5-95

    def _calculate_position_size(
        self,
        edge: float,
        price: int,
        confidence: float
    ) -> int:
        """
        Calculate position size based on Kelly criterion (fractional).
        """
        # Max loss per contract = price in cents
        max_loss_per_contract = price / 100 * 1  # $1 per contract at price cents

        # How many contracts can we buy with max loss budget?
        max_contracts_by_risk = int(self.max_loss_per_trade / max_loss_per_contract)

        # Kelly fraction (use 25% Kelly for safety)
        win_prob = (50 + edge) / 100
        lose_prob = 1 - win_prob
        win_amount = (100 - price) / price  # Return on investment
        kelly = (win_prob * win_amount - lose_prob) / win_amount
        kelly_fraction = max(0, kelly * 0.25)  # Quarter Kelly

        # Adjust for confidence
        position_fraction = kelly_fraction * confidence

        # Calculate contracts
        contracts = int(max_contracts_by_risk * position_fraction)

        return min(contracts, self.MAX_CONTRACTS_PER_TRADE)

    def execute_trade(self, opportunity: KalshiOpportunity) -> Optional[Dict]:
        """
        Execute a trade on Kalshi (or paper trade).
        """
        trade_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'ticker': opportunity.ticker,
            'asset': opportunity.asset,
            'side': opportunity.recommended_side,
            'action': 'buy',
            'contracts': opportunity.contracts,
            'price': opportunity.recommended_price,
            'edge': opportunity.edge,
            'current_price': opportunity.current_price,
            'bracket_high': opportunity.bracket_high,
            'expires_at': opportunity.expires_at.isoformat(),
            'paper': self.paper_mode,
            'status': 'pending'
        }

        if self.paper_mode:
            trade_record['status'] = 'filled'
            trade_record['fill_price'] = opportunity.recommended_price
            logger.info(
                f"[PAPER] BUY {opportunity.contracts} {opportunity.recommended_side.upper()} "
                f"@ {opportunity.recommended_price}c on {opportunity.ticker} "
                f"(edge: {opportunity.edge:.1f}%)"
            )
        else:
            try:
                # Ensure price is an integer in valid range (1-99 cents)
                price = int(opportunity.recommended_price)
                if price < 1 or price > 99:
                    logger.warning(f"Price {price} out of range (1-99), skipping order")
                    trade_record['status'] = 'skipped'
                    trade_record['error'] = f'Price {price} out of range'
                    self.trades_today.append(trade_record)
                    return trade_record

                # Ensure contracts is valid
                contracts = int(opportunity.contracts)
                if contracts < 1:
                    logger.warning(f"Invalid contract count {contracts}, skipping order")
                    trade_record['status'] = 'skipped'
                    trade_record['error'] = f'Invalid contracts {contracts}'
                    self.trades_today.append(trade_record)
                    return trade_record

                order = self.kalshi.create_order(
                    ticker=opportunity.ticker,
                    side=opportunity.recommended_side,
                    action='buy',
                    count=contracts,
                    price=price,
                    order_type='limit'
                )
                trade_record['order_id'] = order.get('order_id')
                trade_record['status'] = order.get('status', 'pending')
                logger.info(f"[LIVE] Order placed: {order}")
            except ValueError as e:
                logger.warning(f"Invalid order parameters: {e}")
                trade_record['status'] = 'invalid'
                trade_record['error'] = str(e)
            except Exception as e:
                logger.error(f"Order failed: {e}")
                trade_record['status'] = 'failed'
                trade_record['error'] = str(e)

        self.trades_today.append(trade_record)
        return trade_record

    def run_scan(self) -> List[Dict]:
        """
        Run a full scan for opportunities across all crypto assets.
        Returns list of executed trades.
        """
        executed = []

        for asset in ['BTC', 'ETH']:
            logger.info(f"Scanning {asset} hourly markets...")

            opportunities = self.find_opportunities(asset)

            for opp in opportunities[:2]:  # Max 2 trades per asset per scan
                trade = self.execute_trade(opp)
                if trade and trade['status'] in ['filled', 'pending']:
                    executed.append(trade)

        return executed

    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'name': 'KalshiHourlyCrypto',
            'capital': self.capital,
            'paper_mode': self.paper_mode,
            'trades_today': len(self.trades_today),
            'pnl_today': self.pnl_today,
            'kalshi_connected': self.kalshi is not None,
            'alpaca_connected': self.alpaca_crypto is not None,
            'last_scan': datetime.now(timezone.utc).isoformat()
        }


def main():
    """Test the bot"""
    print("=" * 60)
    print("KALSHI HOURLY CRYPTO BOT - TEST RUN")
    print("=" * 60)

    bot = KalshiHourlyCryptoBot(
        capital=200.0,
        risk_per_trade=0.02,
        paper_mode=True
    )

    print(f"\nBot Status: {bot.get_status()}")

    print("\n--- Running Scan ---")
    trades = bot.run_scan()

    print(f"\nExecuted {len(trades)} trades:")
    for trade in trades:
        print(f"  {trade['asset']}: {trade['side'].upper()} {trade['contracts']} @ {trade['price']}c "
              f"(edge: {trade['edge']:.1f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
