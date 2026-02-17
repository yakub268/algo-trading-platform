"""
FOMC Drift Trader Bot
Strategy: Pre-FOMC announcement drift (documented anomaly since 1994)
- 24 hours before FOMC → BUY BTC (risk assets drift up)
- 1 hour after FOMC (14:00 ET → 15:00 ET) → SELL (take profit)
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import traceback

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType

try:
    from bots.alpaca_crypto_client import AlpacaCryptoClient
except ImportError:
    AlpacaCryptoClient = None

try:
    import pytz
except ImportError:
    pytz = None


class FOMCDriftTraderBot(FleetBot):
    """
    FOMC pre-announcement drift trader.

    Logic:
    1. Hardcoded 2026 FOMC dates (typically 14:00 ET)
    2. 24 hours before FOMC → BUY BTC
    3. 1 hour after FOMC (15:00 ET) → SELL (exit)
    4. Well-documented anomaly, high confidence
    """

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="FOMC-Drift-Trader",
                bot_type=BotType.CRYPTO,
                symbols=["BTC/USD"],  # Most correlated with macro
                schedule_seconds=3600,  # Check hourly
                max_position_usd=50.0,
                max_daily_trades=2,
                min_confidence=0.65,
                enabled=True,
                paper_mode=True,
                extra={'base_position_size': 40.0}
            )

        super().__init__(config)

        # 2026 FOMC meeting dates (announcements typically at 14:00 ET)
        # Format: (year, month, day, hour=14, minute=0)
        self.fomc_dates = [
            datetime(2026, 1, 29, 14, 0),
            datetime(2026, 3, 19, 14, 0),
            datetime(2026, 5, 7, 14, 0),
            datetime(2026, 6, 18, 14, 0),
            datetime(2026, 7, 29, 14, 0),
            datetime(2026, 9, 17, 14, 0),
            datetime(2026, 10, 29, 14, 0),
            datetime(2026, 12, 10, 14, 0),
        ]

        # Convert to ET timezone
        if pytz:
            self.et_tz = pytz.timezone('US/Eastern')
            self.fomc_dates = [self.et_tz.localize(dt) for dt in self.fomc_dates]
        else:
            self.logger.warning("pytz not available, using UTC times (may be inaccurate)")
            self.fomc_dates = [dt.replace(tzinfo=timezone.utc) for dt in self.fomc_dates]

        # Strategy parameters
        self.entry_window_hours = 24  # Enter 24h before FOMC
        self.exit_hour_offset = 1  # Exit 1h after FOMC (15:00 ET)

        # Alpaca client
        self.client = None
        if AlpacaCryptoClient:
            try:
                self.client = AlpacaCryptoClient()
            except Exception as e:
                self.logger.warning(f"Failed to initialize AlpacaCryptoClient: {e}")

        # Track active FOMC trade
        self.active_fomc_trade = None  # {'entry_time', 'fomc_date', 'symbol'}

    def scan(self) -> List[FleetSignal]:
        """Scan for FOMC drift opportunities."""
        if not self.client:
            self.logger.error("AlpacaCryptoClient not available")
            return []

        signals = []

        try:
            now = datetime.now(timezone.utc)
            if pytz and self.et_tz:
                now_et = now.astimezone(self.et_tz)
            else:
                now_et = now

            # Find next FOMC date
            upcoming_fomc = None
            for fomc_date in self.fomc_dates:
                if fomc_date > now_et:
                    upcoming_fomc = fomc_date
                    break

            if not upcoming_fomc:
                self.logger.debug("No upcoming FOMC dates in 2026")
                return []

            # Calculate time until FOMC
            time_until_fomc = upcoming_fomc - now_et
            hours_until_fomc = time_until_fomc.total_seconds() / 3600

            # Check if we're in the entry window (24h before FOMC)
            if 0 < hours_until_fomc <= self.entry_window_hours:
                # Entry signal
                if not self.active_fomc_trade:
                    signal = self._generate_entry_signal(upcoming_fomc, hours_until_fomc)
                    if signal:
                        signals.append(signal)
                        self.active_fomc_trade = {
                            'entry_time': now_et,
                            'fomc_date': upcoming_fomc,
                            'symbol': signal.symbol
                        }
                else:
                    self.logger.debug(
                        f"Already have active FOMC trade for {self.active_fomc_trade['fomc_date']}"
                    )

            # Check if we're in the exit window (1h after FOMC)
            elif hours_until_fomc < 0:
                hours_since_fomc = abs(hours_until_fomc)
                if hours_since_fomc <= self.exit_hour_offset:
                    # Exit signal
                    if self.active_fomc_trade and self.active_fomc_trade['fomc_date'] == upcoming_fomc:
                        signal = self._generate_exit_signal()
                        if signal:
                            signals.append(signal)
                            self.active_fomc_trade = None
                    else:
                        self.logger.debug("Exit window but no active trade")
                elif hours_since_fomc > self.exit_hour_offset and self.active_fomc_trade:
                    # Past exit window, force close
                    self.logger.warning("Past exit window, clearing active trade state")
                    self.active_fomc_trade = None

        except Exception as e:
            self.logger.error(f"Error in scan: {e}")
            self.logger.debug(traceback.format_exc())

        return signals

    def _generate_entry_signal(self, fomc_date: datetime, hours_until: float) -> Optional[FleetSignal]:
        """Generate entry signal for FOMC drift."""
        try:
            symbol = self.config.symbols[0]  # BTC/USD

            # Get current price
            current_price = self.client.get_price(symbol)

            # Confidence based on timing (earlier = better)
            # Best confidence when 20-24h before, lower as we get closer
            time_score = min(hours_until / self.entry_window_hours, 1.0)
            confidence = 0.65 + (0.15 * time_score)

            # Position size
            position_size = (self.config.max_position_usd * 0.5) + (
                (self.config.max_position_usd - (self.config.max_position_usd * 0.5)) * time_score
            )

            # Stop loss: 3% below entry (tight, documented effect)
            stop_loss = current_price * 0.97

            # Take profit: 4% above entry (typical pre-FOMC drift)
            take_profit = current_price * 1.04

            signal = FleetSignal(
                bot_name=self.name,
                bot_type=self.bot_type.value,
                symbol=symbol,
                side='BUY',
                entry_price=current_price,
                target_price=take_profit,
                stop_loss=stop_loss,
                quantity=position_size / current_price,
                position_size_usd=position_size,
                confidence=confidence,
                edge=4.0,  # typical 4% pre-FOMC drift
                reason='24h pre-FOMC drift anomaly (documented since 1994)',
                metadata={
                    'fomc_date': fomc_date.isoformat(),
                    'hours_until_fomc': float(hours_until),
                    'strategy': 'pre_fomc_drift',
                    'rationale': '24h pre-FOMC drift anomaly (documented since 1994)'
                }
            )

            self.logger.info(
                f"FOMC ENTRY: {symbol} @ ${current_price:.2f} | "
                f"FOMC in {hours_until:.1f}h ({fomc_date.strftime('%Y-%m-%d %H:%M %Z')}) | "
                f"Conf: {confidence:.2%} | Size: ${position_size:.2f}"
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error generating entry signal: {e}")
            return None

    def _generate_exit_signal(self) -> Optional[FleetSignal]:
        """Generate exit signal for FOMC drift."""
        try:
            if not self.active_fomc_trade:
                return None

            symbol = self.active_fomc_trade['symbol']
            current_price = self.client.get_price(symbol)

            signal = FleetSignal(
                bot_name=self.name,
                bot_type=self.bot_type.value,
                symbol=symbol,
                side='SELL',
                entry_price=current_price,
                target_price=0.0,
                stop_loss=0.0,
                quantity=0.0,  # Exit full position
                position_size_usd=0.0,
                confidence=0.85,  # High confidence exit
                edge=0.0,
                reason='Exit 1h after FOMC announcement',
                metadata={
                    'fomc_date': self.active_fomc_trade['fomc_date'].isoformat(),
                    'entry_time': self.active_fomc_trade['entry_time'].isoformat(),
                    'strategy': 'post_fomc_exit',
                    'rationale': 'Exit 1h after FOMC announcement',
                    'exit_signal': True
                }
            )

            self.logger.info(
                f"FOMC EXIT: {symbol} @ ${current_price:.2f} | "
                f"Exiting post-FOMC ({self.active_fomc_trade['fomc_date'].strftime('%Y-%m-%d %H:%M %Z')})"
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error generating exit signal: {e}")
            return None


def main():
    """Test runner."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = FOMCDriftTraderBot()

    print(f"\n{'='*60}")
    print(f"FOMC Drift Trader Bot - Test Scan")
    print(f"{'='*60}\n")

    # Print FOMC dates
    print("2026 FOMC Dates:")
    for i, fomc_date in enumerate(bot.fomc_dates, 1):
        print(f"  {i}. {fomc_date.strftime('%Y-%m-%d %H:%M %Z')}")
    print()

    signals = bot.scan()

    if signals:
        print(f"Found {len(signals)} signal(s):\n")
        for sig in signals:
            print(f"Symbol: {sig.symbol}")
            print(f"Direction: {sig.direction.upper()}")
            print(f"Confidence: {sig.confidence:.2%}")
            print(f"Entry: ${sig.entry_price:.2f}")
            print(f"Stop Loss: ${sig.stop_loss:.2f if sig.stop_loss else 'N/A'}")
            print(f"Take Profit: ${sig.take_profit:.2f if sig.take_profit else 'N/A'}")
            print(f"Position Size: ${sig.position_size:.2f}")
            print(f"Metadata: {sig.metadata}")
            print(f"{'-'*60}\n")
    else:
        print("No signals generated (not in FOMC window).\n")


if __name__ == "__main__":
    main()
