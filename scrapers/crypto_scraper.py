"""
Crypto Data Scraper

Fetches cryptocurrency data for Kalshi crypto price contracts:
- Fear & Greed Index from alternative.me
- BTC/ETH current prices and momentum
- Historical volatility for probability calculations
- Technical indicators (RSI, SMA, EMA, MACD, Bollinger Bands)
- On-chain metrics

Author: Trading Bot
Created: January 2026
"""

import os
import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CryptoScraper')


@dataclass
class CryptoPrice:
    """Current cryptocurrency price data"""
    symbol: str  # BTC, ETH
    price_usd: float
    price_24h_ago: float
    change_24h_pct: float
    high_24h: float
    low_24h: float
    volume_24h: float
    market_cap: float
    fetched_at: datetime


@dataclass
class FearGreedIndex:
    """Crypto Fear & Greed Index"""
    value: int  # 0-100
    classification: str  # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
    timestamp: datetime
    previous_value: int
    previous_classification: str


@dataclass
class CryptoProbability:
    """Probability estimate for crypto price contract"""
    symbol: str
    ticker_pattern: str
    threshold: float
    timeframe: str  # 'daily', 'weekly', 'monthly'
    direction: str  # 'above', 'below'
    our_probability: float
    current_price: float
    reasoning: str


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators for a cryptocurrency"""
    symbol: str
    current_price: float

    # Moving averages
    sma_7: Optional[float] = None   # 7-day simple moving average
    sma_20: Optional[float] = None  # 20-day SMA
    sma_50: Optional[float] = None  # 50-day SMA
    sma_200: Optional[float] = None # 200-day SMA
    ema_12: Optional[float] = None  # 12-day exponential moving average
    ema_26: Optional[float] = None  # 26-day EMA

    # RSI (Relative Strength Index)
    rsi_14: Optional[float] = None  # 14-period RSI

    # MACD
    macd_line: Optional[float] = None      # MACD line (12 EMA - 26 EMA)
    macd_signal: Optional[float] = None    # 9-day EMA of MACD
    macd_histogram: Optional[float] = None # MACD - Signal

    # Bollinger Bands (20-day, 2 std dev)
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None

    # Signals
    trend: Optional[str] = None      # 'bullish', 'bearish', 'neutral'
    rsi_signal: Optional[str] = None # 'overbought', 'oversold', 'neutral'
    ma_signal: Optional[str] = None  # 'bullish_cross', 'bearish_cross', 'above_ma', 'below_ma'

    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CryptoTechnicalEstimate:
    """Enhanced probability estimate using technical analysis"""
    symbol: str
    threshold: float
    timeframe_days: int
    direction: str  # 'above', 'below'
    base_probability: float      # From price model
    technical_adjustment: float  # Adjustment from technicals
    final_probability: float     # Base + adjustment
    confidence: str              # 'HIGH', 'MEDIUM', 'LOW'
    signals: Dict                 # Technical signals used
    reasoning: str


class CryptoScraper:
    """
    Scrapes cryptocurrency data from free APIs.

    Sources:
    - Alternative.me Fear & Greed Index
    - CoinGecko API (free tier)
    """

    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    COINGECKO_BASE = "https://api.coingecko.com/api/v3"

    CACHE_DURATION = timedelta(minutes=15)

    def __init__(self, cache_dir: str = "data/crypto_cache"):
        """Initialize the crypto scraper"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingBot/1.0',
            'Accept': 'application/json'
        })

        self.cache: Dict[str, Tuple[datetime, any]] = {}
        logger.info("CryptoScraper initialized")

    def _get_cache_path(self, key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{key}.json")

    def _load_cache(self, key: str) -> Optional[any]:
        """Load from cache if valid"""
        if key in self.cache:
            cached_time, data = self.cache[key]
            if datetime.now(timezone.utc) - cached_time < self.CACHE_DURATION:
                return data

        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                cached_time = datetime.fromisoformat(data.get('cached_at', ''))
                if datetime.now(timezone.utc) - cached_time < self.CACHE_DURATION:
                    self.cache[key] = (cached_time, data.get('data'))
                    return data.get('data')
            except Exception as e:
                logger.debug(f"Cache load error for {key}: {e}")
        return None

    def _save_cache(self, key: str, data: any):
        """Save to cache"""
        now = datetime.now(timezone.utc)
        self.cache[key] = (now, data)

        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'cached_at': now.isoformat(),
                    'data': data
                }, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Cache save error: {e}")

    def fetch_fear_greed_index(self) -> Optional[FearGreedIndex]:
        """
        Fetch the Crypto Fear & Greed Index.

        Returns:
            FearGreedIndex object or None
        """
        cache_key = "fear_greed"
        cached = self._load_cache(cache_key)
        if cached:
            return FearGreedIndex(**cached)

        try:
            # Get current and previous day
            response = self.session.get(
                self.FEAR_GREED_URL,
                params={'limit': 2},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            if data.get('data') and len(data['data']) >= 2:
                current = data['data'][0]
                previous = data['data'][1]

                result = FearGreedIndex(
                    value=int(current['value']),
                    classification=current['value_classification'],
                    timestamp=datetime.fromtimestamp(int(current['timestamp']), tz=timezone.utc),
                    previous_value=int(previous['value']),
                    previous_classification=previous['value_classification']
                )

                self._save_cache(cache_key, {
                    'value': result.value,
                    'classification': result.classification,
                    'timestamp': result.timestamp.isoformat(),
                    'previous_value': result.previous_value,
                    'previous_classification': result.previous_classification
                })

                return result

        except Exception as e:
            logger.error(f"Fear & Greed API error: {e}")

        return None

    def fetch_crypto_prices(self, symbols: List[str] = None) -> Dict[str, CryptoPrice]:
        """
        Fetch current prices from CoinGecko.

        Args:
            symbols: List of symbols (default: BTC, ETH)

        Returns:
            Dict mapping symbol to CryptoPrice
        """
        if symbols is None:
            symbols = ['BTC', 'ETH']

        cache_key = "prices_" + "_".join(symbols)
        cached = self._load_cache(cache_key)
        if cached:
            return {k: CryptoPrice(**v) for k, v in cached.items()}

        # CoinGecko uses different IDs
        coingecko_ids = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'XRP': 'ripple',
        }

        ids = [coingecko_ids.get(s, s.lower()) for s in symbols]

        try:
            response = self.session.get(
                f"{self.COINGECKO_BASE}/coins/markets",
                params={
                    'vs_currency': 'usd',
                    'ids': ','.join(ids),
                    'order': 'market_cap_desc',
                    'sparkline': 'false',
                    'price_change_percentage': '24h'
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            prices = {}
            for coin in data:
                symbol = coin['symbol'].upper()

                price_data = CryptoPrice(
                    symbol=symbol,
                    price_usd=coin['current_price'],
                    price_24h_ago=coin['current_price'] / (1 + coin['price_change_percentage_24h'] / 100),
                    change_24h_pct=coin['price_change_percentage_24h'],
                    high_24h=coin['high_24h'],
                    low_24h=coin['low_24h'],
                    volume_24h=coin['total_volume'],
                    market_cap=coin['market_cap'],
                    fetched_at=datetime.now(timezone.utc)
                )
                prices[symbol] = price_data

            # Cache the results
            cache_data = {k: {
                'symbol': v.symbol,
                'price_usd': v.price_usd,
                'price_24h_ago': v.price_24h_ago,
                'change_24h_pct': v.change_24h_pct,
                'high_24h': v.high_24h,
                'low_24h': v.low_24h,
                'volume_24h': v.volume_24h,
                'market_cap': v.market_cap,
                'fetched_at': v.fetched_at.isoformat()
            } for k, v in prices.items()}
            self._save_cache(cache_key, cache_data)

            time.sleep(0.5)  # Rate limiting
            return prices

        except Exception as e:
            logger.error(f"CoinGecko API error: {e}")
            return {}

    def calculate_price_probability(
        self,
        current_price: float,
        threshold: float,
        direction: str = 'above',
        timeframe_days: int = 1,
        daily_volatility: float = 0.03  # ~3% daily volatility for BTC
    ) -> float:
        """
        Calculate probability that price will be above/below threshold.

        Uses log-normal distribution assumption for crypto prices.

        Args:
            current_price: Current price
            threshold: Price threshold
            direction: 'above' or 'below'
            timeframe_days: Number of days until expiry
            daily_volatility: Daily return volatility (default 3% for BTC)

        Returns:
            Probability (0-1)
        """
        if current_price <= 0 or threshold <= 0:
            return 0.5

        # Calculate drift (assume 0 for short-term)
        drift = 0

        # Scale volatility by sqrt of time
        total_volatility = daily_volatility * math.sqrt(timeframe_days)

        # Log price ratio
        log_ratio = math.log(threshold / current_price)

        # Z-score in log space
        z = (log_ratio - drift * timeframe_days) / total_volatility

        # Standard normal CDF
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        if direction == 'above':
            # P(S_T >= threshold) = 1 - CDF(z)
            return 1 - norm_cdf(z)
        else:
            # P(S_T <= threshold)
            return norm_cdf(z)

    def _generate_kalshi_ticker(
        self,
        symbol: str,
        threshold: float,
        target_date: datetime = None,
        hour: int = 16
    ) -> str:
        """
        Generate Kalshi-format crypto ticker.

        Format: KX{SYMBOL}-{YY}{MMM}{DD}{HH}-T{threshold}
        Example: KXBTC-26JAN3116-T91499.99

        Args:
            symbol: BTC or ETH
            threshold: Price threshold (e.g., 91499.99)
            target_date: Target date for the contract (default: today)
            hour: Hour of resolution (default: 16 = 4pm EST)

        Returns:
            Kalshi ticker string
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc)

        # Format: YYMMMDD (e.g., 26JAN31)
        year_short = target_date.strftime('%y')  # 26
        month_abbr = target_date.strftime('%b').upper()  # JAN
        day = target_date.strftime('%d')  # 31

        # Format threshold with proper precision
        # Kalshi uses decimals for crypto thresholds
        if threshold == int(threshold):
            threshold_str = f"{int(threshold)}"
        else:
            threshold_str = f"{threshold:.2f}"

        return f"KX{symbol}-{year_short}{month_abbr}{day}{hour:02d}-T{threshold_str}"

    def generate_probability_estimates(
        self,
        prices: Dict[str, CryptoPrice],
        fear_greed: Optional[FearGreedIndex] = None
    ) -> List[CryptoProbability]:
        """
        Generate probability estimates for crypto price contracts.

        Args:
            prices: Dict of current crypto prices
            fear_greed: Fear & Greed Index (for sentiment adjustment)

        Returns:
            List of CryptoProbability estimates
        """
        estimates = []

        # Volatility estimates (daily)
        volatility = {
            'BTC': 0.025,  # ~2.5% daily
            'ETH': 0.035,  # ~3.5% daily
        }

        # Adjust volatility based on Fear & Greed
        if fear_greed:
            # Extreme fear/greed = higher volatility
            fg_adjustment = abs(fear_greed.value - 50) / 100  # 0 to 0.5
            for symbol in volatility:
                volatility[symbol] *= (1 + fg_adjustment)

        # Get today and next 7 days for weekly contracts
        today = datetime.now(timezone.utc)

        for symbol, price_data in prices.items():
            current = price_data.price_usd
            vol = volatility.get(symbol, 0.03)

            # Common threshold levels for each crypto
            if symbol == 'BTC':
                # Round to nearest 5000
                base = round(current / 5000) * 5000
                thresholds = [base - 10000, base - 5000, base, base + 5000, base + 10000]
            elif symbol == 'ETH':
                # Round to nearest 100
                base = round(current / 100) * 100
                thresholds = [base - 200, base - 100, base, base + 100, base + 200]
            else:
                continue

            for threshold in thresholds:
                if threshold <= 0:
                    continue

                # Daily contracts (today at 4pm EST)
                prob_above = self.calculate_price_probability(
                    current, threshold, 'above', 1, vol
                )

                # Generate Kalshi-format ticker for today
                # Format: KXBTC-26JAN3116-T91499.99
                daily_ticker = self._generate_kalshi_ticker(
                    symbol=symbol,
                    threshold=threshold,
                    target_date=today,
                    hour=16  # 4pm EST resolution
                )

                estimates.append(CryptoProbability(
                    symbol=symbol,
                    ticker_pattern=daily_ticker,
                    threshold=threshold,
                    timeframe='daily',
                    direction='above',
                    our_probability=prob_above,
                    current_price=current,
                    reasoning=f"{symbol} at ${current:,.0f}, {1-prob_above:.0%} prob below ${threshold:,}"
                ))

                # Weekly contracts (7 days out)
                prob_above_weekly = self.calculate_price_probability(
                    current, threshold, 'above', 7, vol
                )

                weekly_date = today + timedelta(days=7)
                weekly_ticker = self._generate_kalshi_ticker(
                    symbol=symbol,
                    threshold=threshold,
                    target_date=weekly_date,
                    hour=16
                )

                estimates.append(CryptoProbability(
                    symbol=symbol,
                    ticker_pattern=weekly_ticker,
                    threshold=threshold,
                    timeframe='weekly',
                    direction='above',
                    our_probability=prob_above_weekly,
                    current_price=current,
                    reasoning=f"{symbol} weekly: ${current:,.0f}, prob above ${threshold:,}"
                ))

        return estimates

    def get_market_summary(self) -> Dict:
        """Get a summary of current crypto market conditions"""
        prices = self.fetch_crypto_prices(['BTC', 'ETH'])
        fear_greed = self.fetch_fear_greed_index()

        summary = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'fear_greed': None,
            'prices': {},
            'market_sentiment': 'neutral'
        }

        if fear_greed:
            summary['fear_greed'] = {
                'value': fear_greed.value,
                'classification': fear_greed.classification,
                'change': fear_greed.value - fear_greed.previous_value
            }

            # Determine overall sentiment
            if fear_greed.value < 25:
                summary['market_sentiment'] = 'extreme_fear'
            elif fear_greed.value < 45:
                summary['market_sentiment'] = 'fear'
            elif fear_greed.value < 55:
                summary['market_sentiment'] = 'neutral'
            elif fear_greed.value < 75:
                summary['market_sentiment'] = 'greed'
            else:
                summary['market_sentiment'] = 'extreme_greed'

        for symbol, data in prices.items():
            summary['prices'][symbol] = {
                'price': data.price_usd,
                'change_24h': data.change_24h_pct,
                'high_24h': data.high_24h,
                'low_24h': data.low_24h
            }

        return summary

    def fetch_historical_prices(self, symbol: str, days: int = 200) -> List[float]:
        """
        Fetch historical daily close prices from CoinGecko.

        Args:
            symbol: BTC, ETH, etc.
            days: Number of days of history

        Returns:
            List of closing prices (oldest to newest)
        """
        cache_key = f"history_{symbol}_{days}"
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        coingecko_ids = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'XRP': 'ripple',
        }

        coin_id = coingecko_ids.get(symbol, symbol.lower())

        try:
            response = self.session.get(
                f"{self.COINGECKO_BASE}/coins/{coin_id}/market_chart",
                params={
                    'vs_currency': 'usd',
                    'days': days,
                    'interval': 'daily'
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            # Extract prices (each entry is [timestamp, price])
            prices = [p[1] for p in data.get('prices', [])]

            if prices:
                self._save_cache(cache_key, prices)

            time.sleep(0.5)  # Rate limiting
            return prices

        except Exception as e:
            logger.error(f"Error fetching historical prices: {e}")
            return []

    def calculate_sma(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    def calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None

        multiplier = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        if len(prices) < period + 1:
            return None

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change >= 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        # Use last 'period' values for RSI calculation
        recent_gains = gains[-period:]
        recent_losses = losses[-period:]

        avg_gain = sum(recent_gains) / period
        avg_loss = sum(recent_losses) / period

        if avg_loss == 0:
            return 100  # No losses = max RSI

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(self, prices: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        if len(prices) < 35:  # Need enough data for 26 EMA + 9 signal
            return None, None, None

        # Calculate EMAs
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)

        if ema_12 is None or ema_26 is None:
            return None, None, None

        # MACD line
        macd_line = ema_12 - ema_26

        # For signal line, we need MACD history
        # Simplified: use recent price changes to estimate signal
        macd_values = []
        for i in range(26, len(prices)):
            subset = prices[:i + 1]
            e12 = self.calculate_ema(subset, 12)
            e26 = self.calculate_ema(subset, 26)
            if e12 and e26:
                macd_values.append(e12 - e26)

        if len(macd_values) >= 9:
            # Signal is 9-period EMA of MACD
            multiplier = 2 / 10
            signal = macd_values[0]
            for mv in macd_values[1:]:
                signal = (mv - signal) * multiplier + signal
        else:
            signal = macd_line

        histogram = macd_line - signal

        return macd_line, signal, histogram

    def calculate_bollinger_bands(
        self,
        prices: List[float],
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate Bollinger Bands.

        Returns:
            Tuple of (upper, middle, lower)
        """
        if len(prices) < period:
            return None, None, None

        recent = prices[-period:]
        middle = sum(recent) / period

        # Standard deviation
        variance = sum((p - middle) ** 2 for p in recent) / period
        std = math.sqrt(variance)

        upper = middle + std_dev * std
        lower = middle - std_dev * std

        return upper, middle, lower

    def fetch_technical_indicators(self, symbol: str) -> Optional[TechnicalIndicators]:
        """
        Fetch and calculate all technical indicators for a symbol.

        Args:
            symbol: BTC, ETH, etc.

        Returns:
            TechnicalIndicators object or None
        """
        cache_key = f"technicals_{symbol}"
        cached = self._load_cache(cache_key)
        if cached:
            return TechnicalIndicators(**cached)

        # Get current price
        prices_data = self.fetch_crypto_prices([symbol])
        if symbol not in prices_data:
            return None

        current_price = prices_data[symbol].price_usd

        # Get historical prices
        history = self.fetch_historical_prices(symbol, 200)
        if len(history) < 50:
            logger.warning(f"Insufficient history for {symbol} technicals")
            return None

        # Calculate moving averages
        sma_7 = self.calculate_sma(history, 7)
        sma_20 = self.calculate_sma(history, 20)
        sma_50 = self.calculate_sma(history, 50)
        sma_200 = self.calculate_sma(history, 200) if len(history) >= 200 else None
        ema_12 = self.calculate_ema(history, 12)
        ema_26 = self.calculate_ema(history, 26)

        # Calculate RSI
        rsi_14 = self.calculate_rsi(history, 14)

        # Calculate MACD
        macd_line, macd_signal, macd_histogram = self.calculate_macd(history)

        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(history, 20)

        # Determine trend
        trend = 'neutral'
        if sma_20 and sma_50:
            if current_price > sma_20 > sma_50:
                trend = 'bullish'
            elif current_price < sma_20 < sma_50:
                trend = 'bearish'

        # RSI signal
        rsi_signal = 'neutral'
        if rsi_14:
            if rsi_14 >= 70:
                rsi_signal = 'overbought'
            elif rsi_14 <= 30:
                rsi_signal = 'oversold'

        # MA signal
        ma_signal = 'neutral'
        if sma_20 and sma_50:
            if sma_20 > sma_50:
                ma_signal = 'bullish_cross' if abs(sma_20 - sma_50) / sma_50 < 0.02 else 'above_ma'
            else:
                ma_signal = 'bearish_cross' if abs(sma_20 - sma_50) / sma_50 < 0.02 else 'below_ma'

        indicators = TechnicalIndicators(
            symbol=symbol,
            current_price=current_price,
            sma_7=sma_7,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            ema_12=ema_12,
            ema_26=ema_26,
            rsi_14=rsi_14,
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            trend=trend,
            rsi_signal=rsi_signal,
            ma_signal=ma_signal
        )

        # Cache the result
        cache_data = {
            'symbol': indicators.symbol,
            'current_price': indicators.current_price,
            'sma_7': indicators.sma_7,
            'sma_20': indicators.sma_20,
            'sma_50': indicators.sma_50,
            'sma_200': indicators.sma_200,
            'ema_12': indicators.ema_12,
            'ema_26': indicators.ema_26,
            'rsi_14': indicators.rsi_14,
            'macd_line': indicators.macd_line,
            'macd_signal': indicators.macd_signal,
            'macd_histogram': indicators.macd_histogram,
            'bb_upper': indicators.bb_upper,
            'bb_middle': indicators.bb_middle,
            'bb_lower': indicators.bb_lower,
            'trend': indicators.trend,
            'rsi_signal': indicators.rsi_signal,
            'ma_signal': indicators.ma_signal,
            'fetched_at': indicators.fetched_at.isoformat()
        }
        self._save_cache(cache_key, cache_data)

        return indicators

    def calculate_technical_probability(
        self,
        symbol: str,
        threshold: float,
        direction: str = 'above',
        timeframe_days: int = 1
    ) -> Optional[CryptoTechnicalEstimate]:
        """
        Calculate probability with technical analysis adjustment.

        Combines base probability model with technical indicators.

        Args:
            symbol: BTC, ETH, etc.
            threshold: Price threshold
            direction: 'above' or 'below'
            timeframe_days: Days until expiry

        Returns:
            CryptoTechnicalEstimate or None
        """
        # Get technical indicators
        technicals = self.fetch_technical_indicators(symbol)
        if not technicals:
            return None

        current_price = technicals.current_price

        # Base probability from price model
        volatility = {'BTC': 0.025, 'ETH': 0.035}.get(symbol, 0.03)
        base_prob = self.calculate_price_probability(
            current_price, threshold, direction, timeframe_days, volatility
        )

        # Technical adjustment factors
        adjustment = 0.0
        signals = {}

        # RSI adjustment
        if technicals.rsi_14:
            signals['rsi'] = technicals.rsi_14
            if technicals.rsi_signal == 'overbought':
                # Less likely to go higher
                if direction == 'above':
                    adjustment -= 0.05
                else:
                    adjustment += 0.05
            elif technicals.rsi_signal == 'oversold':
                # Less likely to go lower
                if direction == 'below':
                    adjustment -= 0.05
                else:
                    adjustment += 0.05

        # Trend adjustment
        if technicals.trend:
            signals['trend'] = technicals.trend
            if technicals.trend == 'bullish':
                if direction == 'above':
                    adjustment += 0.03
                else:
                    adjustment -= 0.03
            elif technicals.trend == 'bearish':
                if direction == 'below':
                    adjustment += 0.03
                else:
                    adjustment -= 0.03

        # MACD adjustment
        if technicals.macd_histogram:
            signals['macd_histogram'] = technicals.macd_histogram
            if technicals.macd_histogram > 0:
                # Bullish momentum
                if direction == 'above':
                    adjustment += 0.02
                else:
                    adjustment -= 0.02
            else:
                # Bearish momentum
                if direction == 'below':
                    adjustment += 0.02
                else:
                    adjustment -= 0.02

        # Bollinger Band adjustment
        if technicals.bb_upper and technicals.bb_lower:
            signals['bb_position'] = 'middle'
            if current_price >= technicals.bb_upper:
                signals['bb_position'] = 'upper'
                # At upper band, less likely to go higher
                if direction == 'above':
                    adjustment -= 0.03
            elif current_price <= technicals.bb_lower:
                signals['bb_position'] = 'lower'
                # At lower band, less likely to go lower
                if direction == 'below':
                    adjustment -= 0.03

        # Final probability
        final_prob = max(0.05, min(0.95, base_prob + adjustment))

        # Determine confidence
        signal_count = sum(1 for v in signals.values() if v is not None)
        if signal_count >= 4 and abs(adjustment) >= 0.05:
            confidence = 'HIGH'
        elif signal_count >= 2 and abs(adjustment) >= 0.03:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        # Build reasoning
        reasoning_parts = [f"Base prob: {base_prob:.0%}"]
        if technicals.trend != 'neutral':
            reasoning_parts.append(f"Trend: {technicals.trend}")
        if technicals.rsi_signal != 'neutral':
            reasoning_parts.append(f"RSI: {technicals.rsi_signal} ({technicals.rsi_14:.0f})")
        if technicals.macd_histogram:
            macd_dir = "bullish" if technicals.macd_histogram > 0 else "bearish"
            reasoning_parts.append(f"MACD: {macd_dir}")

        return CryptoTechnicalEstimate(
            symbol=symbol,
            threshold=threshold,
            timeframe_days=timeframe_days,
            direction=direction,
            base_probability=base_prob,
            technical_adjustment=adjustment,
            final_probability=final_prob,
            confidence=confidence,
            signals=signals,
            reasoning=', '.join(reasoning_parts)
        )


def main():
    """Test the crypto scraper"""
    print("=" * 60)
    print("CRYPTO SCRAPER TEST")
    print("=" * 60)

    scraper = CryptoScraper()

    print("\n[1] Fear & Greed Index...")
    print("-" * 40)
    fg = scraper.fetch_fear_greed_index()
    if fg:
        print(f"  Current: {fg.value} ({fg.classification})")
        print(f"  Previous: {fg.previous_value} ({fg.previous_classification})")
        change = fg.value - fg.previous_value
        print(f"  Change: {change:+d}")

    print("\n[2] Crypto Prices...")
    print("-" * 40)
    prices = scraper.fetch_crypto_prices(['BTC', 'ETH'])
    for symbol, data in prices.items():
        print(f"  {symbol}: ${data.price_usd:,.2f}")
        print(f"    24h: {data.change_24h_pct:+.1f}%")
        print(f"    Range: ${data.low_24h:,.0f} - ${data.high_24h:,.0f}")

    print("\n[3] Technical Indicators...")
    print("-" * 40)
    for symbol in ['BTC', 'ETH']:
        technicals = scraper.fetch_technical_indicators(symbol)
        if technicals:
            print(f"\n  {symbol}:")
            print(f"    Price: ${technicals.current_price:,.0f}")
            print(f"    SMA-20: ${technicals.sma_20:,.0f}" if technicals.sma_20 else "    SMA-20: N/A")
            print(f"    SMA-50: ${technicals.sma_50:,.0f}" if technicals.sma_50 else "    SMA-50: N/A")
            print(f"    RSI-14: {technicals.rsi_14:.1f}" if technicals.rsi_14 else "    RSI-14: N/A")
            print(f"    MACD: {technicals.macd_line:.2f}" if technicals.macd_line else "    MACD: N/A")
            print(f"    Trend: {technicals.trend}")
            print(f"    RSI Signal: {technicals.rsi_signal}")
            if technicals.bb_upper:
                print(f"    Bollinger: ${technicals.bb_lower:,.0f} - ${technicals.bb_upper:,.0f}")

    print("\n[4] Technical Probability Estimates...")
    print("-" * 40)
    if 'BTC' in prices:
        btc_price = prices['BTC'].price_usd
        thresholds = [round(btc_price / 5000) * 5000 + i * 5000 for i in range(-2, 3)]

        for threshold in thresholds:
            estimate = scraper.calculate_technical_probability('BTC', threshold, 'above', 1)
            if estimate:
                print(f"  BTC > ${threshold:,} (1 day):")
                print(f"    Final: {estimate.final_probability:.0%} (base: {estimate.base_probability:.0%})")
                print(f"    Adjustment: {estimate.technical_adjustment:+.1%}")
                print(f"    {estimate.reasoning}")

    print("\n[5] Basic Probability Estimates...")
    print("-" * 40)
    estimates = scraper.generate_probability_estimates(prices, fg)
    print(f"Generated {len(estimates)} estimates")

    btc_daily = [e for e in estimates if e.symbol == 'BTC' and e.timeframe == 'daily']
    for est in btc_daily[:5]:
        print(f"  {est.ticker_pattern}: {est.our_probability:.0%}")

    print("\n[6] Market Summary...")
    print("-" * 40)
    summary = scraper.get_market_summary()
    print(f"  Sentiment: {summary['market_sentiment']}")
    if summary['fear_greed']:
        print(f"  Fear & Greed: {summary['fear_greed']['value']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
