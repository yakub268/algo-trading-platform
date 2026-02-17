"""
Funding Rate Harvester Bot
Strategy: Extreme perp funding rates → contrarian spot signals
- Fetches funding rates from Bybit via CCXT
- Extreme positive funding → overheated → SELL spot
- Extreme negative funding → oversold → BUY spot
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
    import ccxt
except ImportError:
    ccxt = None

try:
    import numpy as np
except ImportError:
    np = None


class FundingRateHarvesterBot(FleetBot):
    """
    Contrarian funding rate trader.

    Logic:
    1. Fetch perpetual funding rates from Bybit
    2. Extreme positive (>0.05%): shorts paying longs → overheated → SELL spot
    3. Extreme negative (<-0.05%): longs paying shorts → oversold → BUY spot
    4. Execute on Alpaca spot
    """

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name="Funding-Rate-Harvester",
                bot_type=BotType.CRYPTO,
                symbols=["BTC/USD", "ETH/USD", "SOL/USD"],
                schedule_seconds=300,
                max_position_usd=40.0,
                max_daily_trades=10,
                min_confidence=0.55,
                enabled=True,
                paper_mode=True,
                extra={'base_position_size': 20.0}
            )

        super().__init__(config)

        # Strategy parameters
        self.extreme_positive_threshold = 0.0005  # 0.05% (50 bps)
        self.extreme_negative_threshold = -0.0005  # -0.05%
        self.moderate_positive_threshold = 0.0003  # 0.03%
        self.moderate_negative_threshold = -0.0003  # -0.03%

        # Bybit exchange
        self.exchange = None
        if ccxt:
            try:
                self.exchange = ccxt.bybit({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'swap',  # Perpetuals
                    }
                })
                self.exchange.load_markets()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Bybit exchange: {e}")

        # Alpaca client
        self.client = None
        if AlpacaCryptoClient:
            try:
                self.client = AlpacaCryptoClient()
            except Exception as e:
                self.logger.warning(f"Failed to initialize AlpacaCryptoClient: {e}")

        # Symbol mapping: Alpaca -> Bybit
        self.symbol_map = {
            "BTC/USD": "BTC/USDT:USDT",
            "ETH/USD": "ETH/USDT:USDT",
            "SOL/USD": "SOL/USDT:USDT"
        }

        # Cache for funding rates (to avoid rate limits)
        self.funding_cache = {}  # symbol -> {'rate': float, 'timestamp': datetime}
        self.cache_ttl = 300  # 5 minutes

    def scan(self) -> List[FleetSignal]:
        """Scan for funding rate extremes."""
        if not self.exchange:
            self.logger.error("CCXT exchange not available")
            return []

        if not self.client:
            self.logger.error("AlpacaCryptoClient not available")
            return []

        signals = []

        for symbol in self.config.symbols:
            try:
                signal = self._scan_symbol(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
                self.logger.debug(traceback.format_exc())

        return signals

    def _scan_symbol(self, symbol: str) -> Optional[FleetSignal]:
        """Scan single symbol for funding rate signal."""
        try:
            # Get Bybit symbol
            bybit_symbol = self.symbol_map.get(symbol)
            if not bybit_symbol:
                self.logger.debug(f"No Bybit mapping for {symbol}")
                return None

            # Check cache
            now = datetime.now(timezone.utc)
            if symbol in self.funding_cache:
                cached = self.funding_cache[symbol]
                age = (now - cached['timestamp']).total_seconds()
                if age < self.cache_ttl:
                    funding_rate = cached['rate']
                    self.logger.debug(f"{symbol}: using cached funding rate {funding_rate:.4%}")
                else:
                    funding_rate = self._fetch_funding_rate(bybit_symbol)
                    if funding_rate is None:
                        return None
                    self.funding_cache[symbol] = {'rate': funding_rate, 'timestamp': now}
            else:
                funding_rate = self._fetch_funding_rate(bybit_symbol)
                if funding_rate is None:
                    return None
                self.funding_cache[symbol] = {'rate': funding_rate, 'timestamp': now}

            # Get current spot price from Alpaca
            try:
                spot_price = self.client.get_price(symbol)
            except Exception as e:
                self.logger.error(f"Failed to get spot price for {symbol}: {e}")
                return None

            # Analyze funding rate
            signal = None

            if funding_rate >= self.extreme_positive_threshold:
                # Extreme positive funding: shorts paying longs → overheated
                # Contrarian: SELL spot (but we can't short on Alpaca)
                # Instead, treat as strong exit signal for any longs
                confidence = 0.55 + min((funding_rate - self.extreme_positive_threshold) / 0.001, 0.30)

                signal = FleetSignal(
                    bot_name=self.name,
                    bot_type=self.bot_type.value,
                    symbol=symbol,
                    side='SELL',
                    entry_price=spot_price,
                    target_price=0.0,
                    stop_loss=0.0,
                    quantity=0.0,  # Exit signal
                    position_size_usd=0.0,
                    confidence=confidence,
                    edge=0.0,
                    reason='Overheated market, shorts paying longs',
                    metadata={
                        'funding_rate': float(funding_rate),
                        'funding_pct': float(funding_rate * 100),
                        'signal_type': 'extreme_positive',
                        'rationale': 'Overheated market, shorts paying longs',
                        'exit_signal': True
                    }
                )

                self.logger.info(
                    f"FUNDING SIGNAL: {symbol} OVERHEATED | "
                    f"Rate: {funding_rate:.4%} | Spot: ${spot_price:.2f} | EXIT longs"
                )

            elif funding_rate <= self.extreme_negative_threshold:
                # Extreme negative funding: longs paying shorts → oversold
                # Contrarian: BUY spot
                confidence = 0.55 + min((abs(funding_rate) - abs(self.extreme_negative_threshold)) / 0.001, 0.35)

                # Position size based on confidence
                position_size = self._calculate_position_size(confidence)

                # Stop loss: 5% below entry
                stop_loss = spot_price * 0.95

                # Take profit: 8% above entry (expecting bounce)
                take_profit = spot_price * 1.08

                signal = FleetSignal(
                    bot_name=self.name,
                    bot_type=self.bot_type.value,
                    symbol=symbol,
                    side='BUY',
                    entry_price=spot_price,
                    target_price=take_profit,
                    stop_loss=stop_loss,
                    quantity=position_size / spot_price,
                    position_size_usd=position_size,
                    confidence=confidence,
                    edge=abs(funding_rate) * 1000,  # funding rate as edge
                    reason='Oversold market, longs paying shorts',
                    metadata={
                        'funding_rate': float(funding_rate),
                        'funding_pct': float(funding_rate * 100),
                        'signal_type': 'extreme_negative',
                        'rationale': 'Oversold market, longs paying shorts'
                    }
                )

                self.logger.info(
                    f"FUNDING SIGNAL: {symbol} OVERSOLD | "
                    f"Rate: {funding_rate:.4%} | Spot: ${spot_price:.2f} | "
                    f"Conf: {confidence:.2%} | Size: ${position_size:.2f}"
                )

            elif funding_rate >= self.moderate_positive_threshold:
                # Moderate positive: caution signal
                self.logger.debug(
                    f"{symbol}: Moderate positive funding {funding_rate:.4%}, no action"
                )

            elif funding_rate <= self.moderate_negative_threshold:
                # Moderate negative: weak buy signal
                confidence = 0.50 + (abs(funding_rate) - abs(self.moderate_negative_threshold)) / 0.001 * 0.15

                if confidence >= self.config.min_confidence:
                    position_size = (self.config.max_position_usd * 0.5)
                    stop_loss = spot_price * 0.96
                    take_profit = spot_price * 1.05

                    signal = FleetSignal(
                        bot_name=self.name,
                        bot_type=self.bot_type.value,
                        symbol=symbol,
                        side='BUY',
                        entry_price=spot_price,
                        target_price=take_profit,
                        stop_loss=stop_loss,
                        quantity=position_size / spot_price,
                        position_size_usd=position_size,
                        confidence=confidence,
                        edge=abs(funding_rate) * 1000,
                        reason='Mildly oversold conditions',
                        metadata={
                            'funding_rate': float(funding_rate),
                            'funding_pct': float(funding_rate * 100),
                            'signal_type': 'moderate_negative',
                            'rationale': 'Mildly oversold conditions'
                        }
                    )

                    self.logger.info(
                        f"FUNDING SIGNAL: {symbol} mild oversold | "
                        f"Rate: {funding_rate:.4%} | Conf: {confidence:.2%}"
                    )

            return signal

        except Exception as e:
            self.logger.error(f"Error in _scan_symbol for {symbol}: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def _fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """Fetch current funding rate from Bybit."""
        try:
            # Fetch funding rate history (last entry is current/next)
            funding_history = self.exchange.fetch_funding_rate_history(
                symbol=symbol,
                limit=1
            )

            if not funding_history:
                self.logger.warning(f"No funding rate data for {symbol}")
                return None

            # Get the most recent funding rate
            latest = funding_history[-1]
            funding_rate = latest.get('fundingRate')

            if funding_rate is None:
                self.logger.warning(f"Funding rate is None for {symbol}")
                return None

            return float(funding_rate)

        except Exception as e:
            self.logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return None

    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence."""
        base = (self.config.max_position_usd * 0.5)
        max_size = self.config.max_position_usd

        size = base + (max_size - base) * (confidence - self.config.min_confidence) / (1.0 - self.config.min_confidence)

        return round(size, 2)


def main():
    """Test runner."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = FundingRateHarvesterBot()

    print(f"\n{'='*60}")
    print(f"Funding Rate Harvester Bot - Test Scan")
    print(f"{'='*60}\n")

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
        print("No signals generated.\n")


if __name__ == "__main__":
    main()
