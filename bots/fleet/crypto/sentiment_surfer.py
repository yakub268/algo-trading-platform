"""
Sentiment Surfer — Fear & Greed contrarian trader.

Strategy:
- Fetch Fear & Greed Index from API
- F&G < 20 (Extreme Fear): BUY with high confidence, $40 position
- F&G 20-25: BUY with medium confidence, $25 position
- F&G > 75 (Extreme Greed): Flag for caution (metadata only, no short)
- Confirmation: RSI < 50 (avoid buying into strong uptrend)
- Targets BTC/USD primarily

Schedule: 3600s (hourly) — F&G updates daily
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType

try:
    from bots.alpaca_crypto_client import AlpacaCryptoClient
except ImportError:
    AlpacaCryptoClient = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class SentimentSurfer(FleetBot):
    """Contrarian trader based on Fear & Greed Index."""

    FNG_API_URL = "https://api.alternative.me/fng/?limit=1"

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="Sentiment-Surfer",
                bot_type=BotType.CRYPTO,
                schedule_seconds=3600,  # 1 hour
                max_position_usd=40.0,
                max_daily_trades=10,
                min_confidence=0.60,
                symbols=["BTC/USD", "ETH/USD"],
                enabled=True,
                paper_mode=True,
                extra={
                    'extreme_fear_threshold': 20,
                    'fear_threshold': 25,
                    'extreme_greed_threshold': 75,
                    'extreme_fear_position': 40.0,
                    'fear_position': 25.0,
                    'rsi_period': 14,
                    'rsi_max': 50,
                }
            )
        super().__init__(config)

        self.client = AlpacaCryptoClient() if AlpacaCryptoClient else None
        self.extreme_fear_threshold = config.extra.get('extreme_fear_threshold', 20)
        self.fear_threshold = config.extra.get('fear_threshold', 25)
        self.extreme_greed_threshold = config.extra.get('extreme_greed_threshold', 75)
        self.extreme_fear_position = config.extra.get('extreme_fear_position', 40.0)
        self.fear_position = config.extra.get('fear_position', 25.0)
        self.rsi_period = config.extra.get('rsi_period', 14)
        self.rsi_max = config.extra.get('rsi_max', 50)

        # Cache F&G data (updates once daily)
        self._fng_cache: Optional[Dict] = None
        self._fng_cache_time: Optional[datetime] = None

    def scan(self) -> List[FleetSignal]:
        """Scan for sentiment-based opportunities."""
        if not self.client or not self.client._initialized:
            self.logger.warning("Alpaca client not initialized")
            return []

        if not REQUESTS_AVAILABLE:
            self.logger.error("requests library not available")
            return []

        # Fetch Fear & Greed Index
        fng_data = self._get_fear_greed_index()
        if not fng_data:
            return []

        fng_value = fng_data['value']
        fng_classification = fng_data['classification']

        self.logger.info(f"Fear & Greed Index: {fng_value} ({fng_classification})")

        # Only trade on fear (contrarian buy)
        if fng_value >= self.fear_threshold:
            self.logger.debug(f"F&G {fng_value} >= {self.fear_threshold}, no buy signal")
            return []

        signals = []

        # Analyze each symbol
        for symbol in self.config.symbols:
            try:
                signal = self._analyze_symbol(symbol, fng_data)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                continue

        return signals

    def _analyze_symbol(self, symbol: str, fng_data: Dict) -> Optional[FleetSignal]:
        """Analyze symbol with F&G sentiment."""
        fng_value = fng_data['value']
        fng_classification = fng_data['classification']

        # Get current price
        current_price = self.client.get_price(symbol)
        if not current_price or current_price <= 0:
            self.logger.warning(f"Invalid price for {symbol}: {current_price}")
            return None

        # Get candles for RSI confirmation
        candles = self._get_candles(symbol, limit=50)
        if not candles or len(candles) < self.rsi_period + 1:
            self.logger.warning(f"Insufficient candles for {symbol}")
            return None

        # Calculate RSI
        rsi = self._calculate_rsi(candles)
        if rsi is None:
            return None

        # RSI confirmation: only buy if RSI < max (avoid buying into strong rally)
        if rsi >= self.rsi_max:
            self.logger.debug(f"{symbol}: RSI {rsi:.1f} >= {self.rsi_max}, skipping")
            return None

        # Determine position size and confidence based on F&G level
        if fng_value < self.extreme_fear_threshold:
            # Extreme Fear
            position_size = self.extreme_fear_position
            confidence = 0.85 + (0.10 * (1 - fng_value / self.extreme_fear_threshold))
            confidence = min(confidence, 0.95)
            reason = f"EXTREME FEAR: F&G={fng_value} ({fng_classification}), RSI={rsi:.1f}"
        else:
            # Normal Fear (20-25)
            position_size = self.fear_position
            confidence = 0.65 + (0.15 * (1 - (fng_value - self.extreme_fear_threshold) /
                                         (self.fear_threshold - self.extreme_fear_threshold)))
            reason = f"Fear: F&G={fng_value} ({fng_classification}), RSI={rsi:.1f}"

        # Cap position size
        position_size = min(position_size, self.config.max_position_usd)

        # Edge estimate: inverse of F&G value (lower F&G = higher edge)
        edge = (100 - fng_value) / 10  # Scale to ~0-10%

        signal = FleetSignal(
            bot_name=self.name,
            bot_type=self.bot_type.value,
            symbol=symbol,
            side="BUY",
            entry_price=current_price,
            target_price=0.0,  # No fixed target
            stop_loss=0.0,     # Let orchestrator manage
            quantity=position_size / current_price,
            position_size_usd=position_size,
            confidence=confidence,
            edge=edge,
            reason=reason,
            metadata={
                'fng_value': fng_value,
                'fng_classification': fng_classification,
                'fng_timestamp': fng_data['timestamp'],
                'rsi': rsi,
                'strategy': 'sentiment_surfer',
            }
        )

        self.logger.info(
            f"Sentiment Signal: {symbol} ${position_size:.2f} | "
            f"F&G={fng_value} ({fng_classification}) | RSI={rsi:.1f} | conf={confidence:.2f}"
        )

        return signal

    def _get_fear_greed_index(self) -> Optional[Dict]:
        """Fetch Fear & Greed Index from API with caching."""
        # Check cache (valid for 1 hour)
        now = datetime.now(timezone.utc)
        if self._fng_cache and self._fng_cache_time:
            age_seconds = (now - self._fng_cache_time).total_seconds()
            if age_seconds < 3600:
                self.logger.debug(f"Using cached F&G data (age: {age_seconds:.0f}s)")
                return self._fng_cache

        # Fetch from API
        try:
            response = requests.get(self.FNG_API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()

            if 'data' not in data or not data['data']:
                self.logger.error("Invalid F&G API response")
                return None

            fng_entry = data['data'][0]
            fng_result = {
                'value': int(fng_entry['value']),
                'classification': fng_entry['value_classification'],
                'timestamp': fng_entry.get('timestamp', ''),
            }

            # Update cache
            self._fng_cache = fng_result
            self._fng_cache_time = now

            return fng_result

        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch F&G Index: {e}")
            return None
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to parse F&G data: {e}")
            return None

    def _get_candles(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Fetch candles from Alpaca."""
        try:
            candles = self.client.get_candles(symbol, granularity='1h', limit=limit)
            if not candles:
                return []
            return candles
        except Exception as e:
            self.logger.error(f"Error fetching candles for {symbol}: {e}")
            return []

    def _calculate_rsi(self, candles: List[Dict]) -> Optional[float]:
        """Calculate RSI from candles."""
        if len(candles) < self.rsi_period + 1:
            return None

        try:
            closes = [c['close'] for c in candles[-self.rsi_period-1:]]
            deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]

            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]

            avg_gain = sum(gains) / self.rsi_period
            avg_loss = sum(losses) / self.rsi_period

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        except Exception as e:
            self.logger.error(f"RSI calculation error: {e}")
            return None
