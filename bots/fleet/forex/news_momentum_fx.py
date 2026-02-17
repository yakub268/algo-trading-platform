"""
News Momentum FX Bot - Fleet Trading System

Exploits USD pair momentum following major economic releases.
NFP/CPI/FOMC surprises create 4-hour momentum windows.

Author: Fleet Trading Bot System
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum

# Project root path resolution
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

try:
    from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType
except ImportError as e:
    logging.error(f"Failed to import FleetBot base: {e}")
    raise

try:
    import requests
except ImportError as e:
    logging.error(f"Failed to import requests: {e}")
    raise


class ReleaseType(Enum):
    """Economic release types."""
    NFP = "NFP"  # Non-Farm Payrolls
    CPI = "CPI"  # Consumer Price Index
    FOMC = "FOMC"  # Federal Open Market Committee


class NewsMomentumFXBot(FleetBot):
    """
    Forex momentum bot trading USD pairs after major economic releases.

    Strategy:
    - Monitor NFP (1st Friday), CPI (monthly), FOMC (8x/year)
    - Compare actual vs consensus via FRED API
    - Positive surprise → BUY USD pairs (EUR_USD SELL, USD_JPY BUY)
    - Negative surprise → SELL USD pairs (EUR_USD BUY, USD_JPY SELL)
    - Hold for 4 hours post-release (momentum window)
    - Position size: 1000 units (micro lots)
    """

    # OANDA config
    OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com"

    # USD pairs to trade
    USD_PAIRS = {
        'EUR_USD': {'direction': 'inverse', 'pip_value': 0.0001},  # USD strong = sell EUR_USD
        'USD_JPY': {'direction': 'direct', 'pip_value': 0.01},     # USD strong = buy USD_JPY
        'GBP_USD': {'direction': 'inverse', 'pip_value': 0.0001},  # USD strong = sell GBP_USD
        'USD_CHF': {'direction': 'direct', 'pip_value': 0.0001},   # USD strong = buy USD_CHF
    }

    # Position sizing
    POSITION_SIZE = 1000  # 1000 units (micro lot for paper)

    # Momentum window (4 hours after release)
    MOMENTUM_WINDOW_HOURS = 4

    # Minimum surprise threshold (% deviation from consensus)
    MIN_SURPRISE_PCT = 0.20  # 0.2% minimum deviation

    # Edge calculation
    MIN_EDGE = 0.005  # 0.5% minimum edge (50 pips on typical pair)
    MIN_CONFIDENCE = 0.50

    # Max positions
    MAX_POSITIONS = 2  # Max 2 concurrent FX positions

    # 2026 Economic Release Schedule (hardcoded)
    # Format: (month, day, hour_utc, release_type)
    RELEASE_SCHEDULE_2026 = [
        # NFP - First Friday of each month at 13:30 UTC
        (1, 2, 13, 30, ReleaseType.NFP),
        (2, 6, 13, 30, ReleaseType.NFP),
        (3, 6, 13, 30, ReleaseType.NFP),
        (4, 3, 13, 30, ReleaseType.NFP),
        (5, 1, 13, 30, ReleaseType.NFP),
        (6, 5, 13, 30, ReleaseType.NFP),
        (7, 3, 13, 30, ReleaseType.NFP),
        (8, 7, 13, 30, ReleaseType.NFP),
        (9, 4, 13, 30, ReleaseType.NFP),
        (10, 2, 13, 30, ReleaseType.NFP),
        (11, 6, 13, 30, ReleaseType.NFP),
        (12, 4, 13, 30, ReleaseType.NFP),

        # CPI - Mid-month around 13th at 13:30 UTC
        (1, 15, 13, 30, ReleaseType.CPI),
        (2, 12, 13, 30, ReleaseType.CPI),
        (3, 12, 13, 30, ReleaseType.CPI),
        (4, 15, 13, 30, ReleaseType.CPI),
        (5, 13, 13, 30, ReleaseType.CPI),
        (6, 10, 13, 30, ReleaseType.CPI),
        (7, 15, 13, 30, ReleaseType.CPI),
        (8, 12, 13, 30, ReleaseType.CPI),
        (9, 15, 13, 30, ReleaseType.CPI),
        (10, 14, 13, 30, ReleaseType.CPI),
        (11, 12, 13, 30, ReleaseType.CPI),
        (12, 10, 13, 30, ReleaseType.CPI),

        # FOMC - 8 meetings per year at 19:00 UTC
        (1, 29, 19, 0, ReleaseType.FOMC),
        (3, 19, 19, 0, ReleaseType.FOMC),
        (5, 7, 19, 0, ReleaseType.FOMC),
        (6, 18, 19, 0, ReleaseType.FOMC),
        (7, 30, 19, 0, ReleaseType.FOMC),
        (9, 17, 19, 0, ReleaseType.FOMC),
        (11, 5, 19, 0, ReleaseType.FOMC),
        (12, 17, 19, 0, ReleaseType.FOMC),
    ]

    # FRED API series IDs
    FRED_SERIES = {
        ReleaseType.NFP: 'PAYEMS',      # Non-Farm Payrolls
        ReleaseType.CPI: 'CPIAUCSL',    # CPI All Urban Consumers
        ReleaseType.FOMC: 'DFF',        # Federal Funds Rate
    }

    def __init__(self, config: Optional[FleetBotConfig] = None):
        """
        Initialize News Momentum FX bot.

        Args:
            config: Fleet bot configuration. Uses defaults if None.
        """
        if config is None:
            config = FleetBotConfig(
                name="News-Momentum-FX",
                bot_type=BotType.FOREX,
                schedule_seconds=300,  # 5 minutes
                max_position_usd=0.0,  # FX uses units not USD
                max_daily_trades=4,
                min_confidence=self.MIN_CONFIDENCE,
                symbols=list(self.USD_PAIRS.keys()),
                enabled=True,
                paper_mode=True,
                extra={'min_edge': self.MIN_EDGE, 'position_size_units': self.POSITION_SIZE}
            )

        super().__init__(config)

        # Get OANDA credentials from environment
        self.oanda_api_key = os.getenv('OANDA_API_KEY')
        self.oanda_account_id = os.getenv('OANDA_ACCOUNT_ID')

        if not self.oanda_api_key or not self.oanda_account_id:
            self.logger.warning("OANDA credentials not found in environment")

        # Get FRED API key from environment
        self.fred_api_key = os.getenv('FRED_API_KEY')

        if not self.fred_api_key:
            self.logger.warning("FRED API key not found in environment")

        # Track active momentum windows
        self.active_windows: List[Dict] = []

        self.logger.info(f"NewsMomentumFXBot initialized: {self.MAX_POSITIONS} max positions, {self.MOMENTUM_WINDOW_HOURS}h momentum window")

    def scan(self) -> List[FleetSignal]:
        """
        Scan for forex momentum opportunities after economic releases.

        Returns:
            List of FleetSignal objects representing trading opportunities
        """
        signals = []

        try:
            # Check for credentials
            if not self.oanda_api_key or not self.fred_api_key:
                self.logger.warning("Missing API credentials, skipping scan")
                return signals

            # Clean up expired momentum windows
            self._clean_expired_windows()

            # Check if we're in an active momentum window
            current_window = self._get_current_momentum_window()

            if not current_window:
                # Check for new releases
                self.logger.info("Checking for recent economic releases...")
                new_window = self._check_for_new_release()

                if new_window:
                    self.logger.info(f"New release detected: {new_window['release_type'].value} with {new_window['surprise_pct']:.2f}% surprise")
                    self.active_windows.append(new_window)
                    current_window = new_window
                else:
                    self.logger.info("No active momentum window, no recent releases")
                    return signals

            # Generate signals for current momentum window
            self.logger.info(f"Active momentum window: {current_window['release_type'].value} ({current_window['direction']})")
            signals = self._generate_momentum_signals(current_window)

            self.logger.info(f"Generated {len(signals)} momentum signals")

        except Exception as e:
            self.logger.error(f"Error in FX momentum scan: {e}", exc_info=True)

        return signals

    def _clean_expired_windows(self):
        """Remove expired momentum windows from tracking."""
        now = datetime.now(timezone.utc)

        self.active_windows = [
            w for w in self.active_windows
            if (now - w['release_time']).total_seconds() < (self.MOMENTUM_WINDOW_HOURS * 3600)
        ]

    def _get_current_momentum_window(self) -> Optional[Dict]:
        """
        Get current active momentum window if any.

        Returns:
            Window dict or None
        """
        if not self.active_windows:
            return None

        # Return most recent window
        return self.active_windows[-1]

    def _check_for_new_release(self) -> Optional[Dict]:
        """
        Check if a major economic release occurred recently.

        Returns:
            Window dict if new release found, None otherwise
        """
        now = datetime.now(timezone.utc)

        # Check releases in past hour
        for month, day, hour, minute, release_type in self.RELEASE_SCHEDULE_2026:
            release_time = datetime(2026, month, day, hour, minute, tzinfo=timezone.utc)

            # Check if release was in the past hour
            time_since_release = (now - release_time).total_seconds()

            if 0 <= time_since_release <= 3600:  # Past hour
                # Check if we already have this window
                if any(w['release_time'] == release_time for w in self.active_windows):
                    continue

                # Fetch actual vs consensus data
                surprise_pct, direction = self._get_release_surprise(release_type, release_time)

                if surprise_pct is None:
                    self.logger.warning(f"Could not get surprise data for {release_type.value}")
                    continue

                # Check if surprise is significant enough
                if abs(surprise_pct) < self.MIN_SURPRISE_PCT:
                    self.logger.info(f"{release_type.value} surprise {surprise_pct:.2f}% too small (< {self.MIN_SURPRISE_PCT}%)")
                    continue

                # Create momentum window
                window = {
                    'release_type': release_type,
                    'release_time': release_time,
                    'surprise_pct': surprise_pct,
                    'direction': direction,  # 'bullish_usd' or 'bearish_usd'
                }

                return window

        return None

    def _get_release_surprise(
        self,
        release_type: ReleaseType,
        release_time: datetime
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Get release surprise (actual vs consensus) from FRED.

        Args:
            release_type: Type of economic release
            release_time: Time of release

        Returns:
            Tuple of (surprise_pct, direction) or (None, None)
        """
        try:
            series_id = self.FRED_SERIES[release_type]

            # Fetch latest data from FRED
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'limit': 2,  # Get latest 2 observations
                'sort_order': 'desc'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'observations' not in data or len(data['observations']) < 2:
                self.logger.warning(f"Insufficient FRED data for {series_id}")
                return None, None

            # Get latest two values
            latest = float(data['observations'][0]['value'])
            previous = float(data['observations'][1]['value'])

            # Calculate change
            if release_type == ReleaseType.NFP:
                # NFP: absolute change in thousands
                change = latest - previous
                surprise_pct = (change / abs(previous)) * 100 if previous != 0 else 0
            elif release_type == ReleaseType.CPI:
                # CPI: month-over-month % change
                surprise_pct = ((latest / previous) - 1) * 100 if previous != 0 else 0
            elif release_type == ReleaseType.FOMC:
                # Fed Funds: basis point change
                change = latest - previous
                surprise_pct = (change / previous) * 100 if previous != 0 else 0
            else:
                surprise_pct = 0

            # Determine USD direction
            # Positive surprise (actual > expected) = bullish USD
            # Negative surprise (actual < expected) = bearish USD
            direction = 'bullish_usd' if surprise_pct > 0 else 'bearish_usd'

            self.logger.info(f"{release_type.value}: {surprise_pct:.2f}% surprise ({direction})")

            return abs(surprise_pct), direction

        except Exception as e:
            self.logger.error(f"Error getting release surprise from FRED: {e}")
            return None, None

    def _generate_momentum_signals(self, window: Dict) -> List[FleetSignal]:
        """
        Generate FX momentum signals for active window.

        Args:
            window: Active momentum window

        Returns:
            List of FleetSignal objects
        """
        signals = []

        direction = window['direction']
        surprise_pct = window['surprise_pct']
        release_type = window['release_type']

        # Trade all USD pairs based on direction
        for pair, pair_info in self.USD_PAIRS.items():
            try:
                # Determine trade side based on USD direction and pair structure
                pair_direction = pair_info['direction']

                if direction == 'bullish_usd':
                    # USD strong
                    side = 'sell' if pair_direction == 'inverse' else 'buy'
                else:
                    # USD weak
                    side = 'buy' if pair_direction == 'inverse' else 'sell'

                # Get current price from OANDA
                current_price = self._get_oanda_price(pair)
                if not current_price:
                    self.logger.warning(f"Could not get price for {pair}")
                    continue

                # Calculate edge based on surprise magnitude
                # Larger surprise = larger expected move = higher edge
                edge_pct = min(2.0, surprise_pct * 0.10)  # 0.1% surprise = 0.01% edge, cap at 2%

                # Check minimum edge
                if edge_pct < self.MIN_EDGE * 100:
                    continue

                # Calculate confidence based on release type and surprise size
                confidence = self._calculate_confidence(release_type, surprise_pct)

                # Check minimum confidence
                if confidence < self.MIN_CONFIDENCE:
                    continue

                # Build reasoning
                reasoning = self._build_reasoning(
                    pair, side, release_type, direction, surprise_pct, edge_pct, confidence
                )

                # Metadata
                metadata = {
                    'pair': pair,
                    'release_type': release_type.value,
                    'surprise_pct': surprise_pct,
                    'usd_direction': direction,
                    'current_price': current_price,
                    'release_time': window['release_time'].isoformat()
                }

                # Create signal
                signal = FleetSignal(
                    bot_name=self.name,
                    bot_type=self.bot_type.value,
                    symbol=pair,
                    side=side.upper(),
                    entry_price=current_price,
                    target_price=0.0,  # FX doesn't use fixed targets
                    stop_loss=0.0,  # FX uses trailing stops
                    quantity=self.POSITION_SIZE,  # Fixed size for FX (units, not USD)
                    position_size_usd=0.0,  # FX uses units
                    confidence=confidence,
                    edge=edge_pct,
                    reason=reasoning,
                    metadata=metadata
                )

                signals.append(signal)

                self.logger.info(
                    f"Generated FX signal: {pair} {side.upper()} | "
                    f"Edge: {edge_pct:.2f}% | Conf: {confidence:.2f} | Size: {self.POSITION_SIZE} units"
                )

            except Exception as e:
                self.logger.error(f"Error generating signal for {pair}: {e}")
                continue

        return signals

    def _get_oanda_price(self, pair: str) -> Optional[float]:
        """
        Get current price from OANDA.

        Args:
            pair: Currency pair (e.g., 'EUR_USD')

        Returns:
            Current mid price or None
        """
        try:
            url = f"{self.OANDA_PRACTICE_URL}/v3/instruments/{pair}/candles"
            params = {
                'count': 1,
                'granularity': 'M5'  # 5-minute candles
            }
            headers = {
                'Authorization': f'Bearer {self.oanda_api_key}'
            }

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'candles' not in data or not data['candles']:
                return None

            candle = data['candles'][0]
            mid = candle.get('mid', {})

            close_price = float(mid.get('c', 0))

            return close_price if close_price > 0 else None

        except Exception as e:
            self.logger.error(f"Error getting OANDA price for {pair}: {e}")
            return None

    def _calculate_confidence(self, release_type: ReleaseType, surprise_pct: float) -> float:
        """
        Calculate confidence based on release type and surprise magnitude.

        Args:
            release_type: Type of release
            surprise_pct: Surprise percentage

        Returns:
            Confidence score (0-1)
        """
        # Base confidence by release type (impact strength)
        base_confidence = {
            ReleaseType.NFP: 0.70,   # NFP has strongest FX impact
            ReleaseType.FOMC: 0.65,  # FOMC strong but less predictable
            ReleaseType.CPI: 0.60,   # CPI moderate impact
        }

        confidence = base_confidence.get(release_type, 0.50)

        # Add confidence based on surprise magnitude
        # Larger surprise = higher confidence
        surprise_contrib = min(0.20, surprise_pct * 0.05)  # +5% per 1% surprise, max +20%
        confidence += surprise_contrib

        # Cap at 0.85 (FX is volatile)
        return min(0.85, confidence)

    def _build_reasoning(
        self,
        pair: str,
        side: str,
        release_type: ReleaseType,
        direction: str,
        surprise_pct: float,
        edge_pct: float,
        confidence: float
    ) -> str:
        """Build human-readable reasoning for the trade."""
        reasoning_parts = [
            f"FX Momentum: {pair}",
            f"Release: {release_type.value}",
            f"Surprise: {surprise_pct:.2f}% ({direction})",
            f"Direction: {side.upper()}",
            f"Edge: {edge_pct:.2f}%",
            f"Confidence: {confidence:.2f}",
            f"Window: {self.MOMENTUM_WINDOW_HOURS}h post-release"
        ]

        return " | ".join(reasoning_parts)


# Entry point for testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = NewsMomentumFXBot()

    print("\n=== News Momentum FX Bot Test ===\n")
    print(f"Bot ID: {bot.config.bot_id}")
    print(f"Bot Type: {bot.config.bot_type.value}")
    print(f"Schedule: {bot.config.schedule_seconds}s")
    print(f"Max Positions: {bot.config.max_positions}")
    print(f"Momentum Window: {bot.MOMENTUM_WINDOW_HOURS}h")
    print(f"Min Edge: {bot.config.min_edge_pct}%")
    print(f"Confidence Threshold: {bot.config.confidence_threshold}")
    print(f"\nOANDA API Key: {'SET' if bot.oanda_api_key else 'MISSING'}")
    print(f"FRED API Key: {'SET' if bot.fred_api_key else 'MISSING'}")
    print("\nScanning for opportunities...\n")

    signals = bot.scan()

    print(f"\nFound {len(signals)} signals:\n")
    for sig in signals:
        print(f"  {sig.ticker} {sig.side.upper()}")
        print(f"    Edge: {sig.edge_pct:.2f}% | Confidence: {sig.confidence:.2f} | Size: {sig.position_size} units")
        print(f"    {sig.reasoning}")
        print()
