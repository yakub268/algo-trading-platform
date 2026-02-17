"""
OANDA Forex Trading Bot

Simple moving average crossover strategy for forex trading.
Uses OANDA v20 REST API.

Author: Jacob
Created: January 2026
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

import requests
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env so environment variables are available when running standalone
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

from config.oanda_config import OANDA_CONFIG, FOREX_PAIRS, TRADING_SESSIONS

# Import RSI divergence detector
try:
    from filters.rsi_divergence import RSIDivergenceDetector
    RSI_DIVERGENCE_AVAILABLE = True
except ImportError:
    RSI_DIVERGENCE_AVAILABLE = False

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/oanda.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OANDABot')


class OANDABot:
    """
    OANDA Forex trading bot using v20 API.

    Strategy: Moving Average Crossover with RSI Hidden Divergence
    - Entry: Fast MA crosses above Slow MA + RSI hidden bullish divergence (or RSI 40-60 fallback)
    - Exit: Fast MA crosses below Slow MA or stop loss/take profit hit

    UPGRADED: RSI now uses hidden divergence detection (70% win rate documented)
    instead of simple level-based filtering.
    """

    def __init__(self, config: Dict = None, use_divergence: bool = True):
        """Initialize the OANDA bot with configuration."""
        self.config = config or OANDA_CONFIG
        self.paper_mode = self.config["paper_mode"]

        # Set base URL based on mode
        if self.paper_mode:
            self.base_url = self.config["practice_url"]
        else:
            self.base_url = self.config["live_url"]

        self.account_id = self.config["account_id"]
        self.api_key = self.config["api_key"]

        # --- Startup validation ---
        if not self.api_key:
            logger.error("OANDA_API_KEY is empty. Set it in your .env file.")
        if not self.account_id:
            logger.error("OANDA_ACCOUNT_ID is empty. Set it in your .env file.")

        # Warn if account ID prefix doesn't match environment
        if self.account_id.startswith("001-") and "fxtrade" in self.base_url:
            logger.warning(
                "Account ID starts with '001-' (practice) but base_url is the LIVE "
                "endpoint. This WILL cause 400 errors. The config auto-detection should "
                "have caught this -- check oanda_config.py."
            )
        elif self.account_id.startswith("101-") and "fxpractice" in self.base_url:
            logger.warning(
                "Account ID starts with '101-' (live) but base_url is the PRACTICE "
                "endpoint. This may cause authentication failures."
            )

        logger.info(f"OANDA base URL: {self.base_url}")
        logger.info(f"OANDA account ID: {self.account_id[:8]}... (truncated)")
        logger.info(f"OANDA paper mode: {self.paper_mode}")

        # Set up session with auth headers
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

        # Trading state
        self.positions: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.peak_balance = self.config["allocation"]

        # Initialize RSI divergence detector
        self.use_divergence = use_divergence and RSI_DIVERGENCE_AVAILABLE
        if self.use_divergence:
            self.divergence_detector = RSIDivergenceDetector(
                rsi_period=self.config.get("rsi_period", 14),
                swing_lookback=10,
                min_rsi_diff=5.0,
                min_price_diff=0.003  # 0.3% for forex
            )
            logger.info("OANDA Bot initialized with RSI Hidden Divergence detection")
        else:
            self.divergence_detector = None
            logger.info("OANDA Bot initialized with level-based RSI")

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make request to OANDA API with detailed error reporting."""
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)

        if not response.ok:
            # OANDA returns detailed error info in the response body -- log it
            try:
                error_body = response.json()
                error_msg = error_body.get("errorMessage", response.text[:500])
            except Exception:
                error_msg = response.text[:500]

            logger.error(
                f"OANDA API error: {response.status_code} {response.reason}\n"
                f"  URL: {method} {url}\n"
                f"  Error: {error_msg}"
            )

            # Provide actionable hints for common 400 errors
            if response.status_code == 400:
                if "Invalid value" in str(error_msg) and "AccountID" in str(error_msg):
                    logger.error(
                        "HINT: Account ID format may be wrong. Practice accounts start "
                        "with '001-', live accounts start with '101-'. Check OANDA_ACCOUNT_ID in .env"
                    )
                elif self.account_id and self.account_id.startswith("001-") and "fxtrade" in self.base_url:
                    logger.error(
                        "HINT: Practice account ID (001-prefix) is being sent to the LIVE "
                        "endpoint (api-fxtrade). Set OANDA_PAPER_MODE=true in .env or use "
                        "a live account ID (101-prefix)."
                    )

            response.raise_for_status()

        return response.json()

    def get_account(self) -> Optional[Dict]:
        """Get account details including balance."""
        try:
            data = self._request("GET", f"/v3/accounts/{self.account_id}")
            return data.get("account")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get account: {e}")
            return None

    def get_candles(
        self,
        instrument: str,
        granularity: str = "H4",
        count: int = 100
    ) -> List[Dict]:
        """
        Fetch historical candles for an instrument.

        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            granularity: Timeframe (e.g., "H4", "D", "M15")
            count: Number of candles to fetch

        Returns:
            List of candle dictionaries
        """
        try:
            data = self._request(
                "GET",
                f"/v3/instruments/{instrument}/candles",
                params={
                    "granularity": granularity,
                    "count": count,
                    "price": "MBA"  # Mid, Bid, Ask
                }
            )
            return data.get("candles", [])

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get candles for {instrument}: {e}")
            return []

    def calculate_indicators(self, candles: List[Dict]) -> Dict:
        """
        Calculate technical indicators from candles.

        Args:
            candles: List of candle dictionaries

        Returns:
            Dictionary with indicator values
        """
        if len(candles) < self.config["slow_ma_period"]:
            return {}

        # Extract closing prices
        closes = np.array([float(c["mid"]["c"]) for c in candles if c.get("complete", True)])
        highs = np.array([float(c["mid"]["h"]) for c in candles if c.get("complete", True)])
        lows = np.array([float(c["mid"]["l"]) for c in candles if c.get("complete", True)])

        if len(closes) < self.config["slow_ma_period"]:
            return {}

        # Calculate SMAs
        fast_ma = np.convolve(
            closes,
            np.ones(self.config["fast_ma_period"]) / self.config["fast_ma_period"],
            mode='valid'
        )
        slow_ma = np.convolve(
            closes,
            np.ones(self.config["slow_ma_period"]) / self.config["slow_ma_period"],
            mode='valid'
        )

        # Calculate RSI
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-self.config["rsi_period"]:])
        avg_loss = np.mean(losses[-self.config["rsi_period"]:])

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Calculate ATR
        tr_list = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        atr = np.mean(tr_list[-self.config["atr_period"]:])

        # Build result dict
        result = {
            "close": closes[-1],
            "fast_ma": fast_ma[-1] if len(fast_ma) > 0 else None,
            "fast_ma_prev": fast_ma[-2] if len(fast_ma) > 1 else None,
            "slow_ma": slow_ma[-1] if len(slow_ma) > 0 else None,
            "slow_ma_prev": slow_ma[-2] if len(slow_ma) > 1 else None,
            "rsi": rsi,
            "atr": atr,
        }

        # Add hidden divergence detection if available
        if self.use_divergence and self.divergence_detector:
            # Create DataFrame for divergence detector
            df = pd.DataFrame({
                'close': closes,
                'high': highs,
                'low': lows
            })
            divergence = self.divergence_detector.detect_divergence(df)
            result.update({
                "hidden_bullish": divergence.hidden_bullish,
                "hidden_bearish": divergence.hidden_bearish,
                "regular_bullish": divergence.regular_bullish,
                "regular_bearish": divergence.regular_bearish,
                "divergence_signal": divergence.signal,
                "divergence_strength": divergence.strength,
                "divergence_reasoning": divergence.reasoning
            })
        else:
            result.update({
                "hidden_bullish": False,
                "hidden_bearish": False,
                "regular_bullish": False,
                "regular_bearish": False,
                "divergence_signal": "none",
                "divergence_strength": 0.0,
                "divergence_reasoning": "Divergence detection not enabled"
            })

        return result

    def calculate_position_size(self, atr: float, current_price: float) -> int:
        """
        Calculate position size based on risk management.

        Args:
            atr: Average True Range value
            current_price: Current price of the instrument

        Returns:
            Position size in units
        """
        account = self.get_account()
        if not account:
            return 0

        balance = float(account.get("balance", 0))
        risk_amount = balance * self.config["max_risk_per_trade"]

        # Stop loss distance based on ATR
        stop_distance = atr * self.config["atr_stop_multiplier"]

        # Position size = Risk Amount / Stop Distance
        if stop_distance == 0:
            return 0

        position_size = int(risk_amount / stop_distance)

        # Apply maximum position limit (50% of balance)
        max_position = int(balance * 0.5 / current_price * 10000)  # Convert to units
        return min(position_size, max_position, 10000)  # Cap at 10k units for safety

    def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        try:
            data = self._request("GET", f"/v3/accounts/{self.account_id}/openPositions")
            return data.get("positions", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Place a market order.

        Args:
            instrument: Currency pair
            units: Positive for buy, negative for sell
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Order response or None
        """
        # Check daily loss limit
        if self.daily_pnl <= -(self.config["allocation"] * self.config["daily_loss_limit"]):
            logger.warning("Daily loss limit reached, not placing order")
            return None

        if self.paper_mode:
            logger.info(f"[PAPER] Market order: {units} {instrument}")
            logger.info(f"[PAPER] SL: {stop_loss}, TP: {take_profit}")
            return {"paper": True, "units": units, "instrument": instrument}

        try:
            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(units),
                    "timeInForce": "FOK",
                    "positionFill": "DEFAULT"
                }
            }

            if stop_loss:
                order_data["order"]["stopLossOnFill"] = {
                    "price": f"{stop_loss:.5f}"
                }

            if take_profit:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": f"{take_profit:.5f}"
                }

            data = self._request(
                "POST",
                f"/v3/accounts/{self.account_id}/orders",
                json=order_data
            )

            logger.info(f"Order placed: {units} {instrument}")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def close_position(self, instrument: str) -> Optional[Dict]:
        """
        Close all positions for an instrument.

        Args:
            instrument: Currency pair to close

        Returns:
            Close response or None
        """
        if self.paper_mode:
            logger.info(f"[PAPER] Closing position: {instrument}")
            return {"paper": True, "closed": instrument}

        try:
            data = self._request(
                "PUT",
                f"/v3/accounts/{self.account_id}/positions/{instrument}/close",
                json={"longUnits": "ALL", "shortUnits": "ALL"}
            )
            logger.info(f"Position closed: {instrument}")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to close position {instrument}: {e}")
            return None

    def check_entry_signal(self, indicators: Dict) -> Optional[str]:
        """
        Check for entry signal based on strategy rules.

        UPGRADED: Now uses RSI Hidden Divergence detection (70% win rate documented)
        instead of simple level-based RSI filtering.

        Args:
            indicators: Dictionary of indicator values

        Returns:
            "buy", "sell", or None
        """
        if not all(k in indicators for k in ["fast_ma", "slow_ma", "fast_ma_prev", "slow_ma_prev", "rsi"]):
            logger.debug("Missing required indicators for entry signal check")
            return None

        fast_ma = indicators["fast_ma"]
        slow_ma = indicators["slow_ma"]
        fast_ma_prev = indicators["fast_ma_prev"]
        slow_ma_prev = indicators["slow_ma_prev"]
        rsi = indicators["rsi"]

        # Get divergence signals if available
        hidden_bullish = indicators.get("hidden_bullish", False)
        hidden_bearish = indicators.get("hidden_bearish", False)
        regular_bullish = indicators.get("regular_bullish", False)
        regular_bearish = indicators.get("regular_bearish", False)
        divergence_strength = indicators.get("divergence_strength", 0)

        # UPGRADED: Check for hidden divergence signals first (trend continuation)
        if self.use_divergence:
            # Hidden Bullish + MA trending up = strong BUY
            if hidden_bullish and fast_ma > slow_ma:
                logger.info(f"Hidden Bullish Divergence detected (strength={divergence_strength:.2f}) - BUY signal")
                return "buy"

            # Hidden Bearish + MA trending down = strong SELL
            if hidden_bearish and fast_ma < slow_ma:
                logger.info(f"Hidden Bearish Divergence detected (strength={divergence_strength:.2f}) - SELL signal")
                return "sell"

            # Regular divergence can also signal (reversal)
            if regular_bullish and rsi < 35:  # Oversold with bullish divergence
                logger.info(f"Regular Bullish Divergence at oversold - potential reversal BUY")
                return "buy"

            if regular_bearish and rsi > 65:  # Overbought with bearish divergence
                logger.info(f"Regular Bearish Divergence at overbought - potential reversal SELL")
                return "sell"

        # Fallback to level-based RSI if no divergence detected
        # WIDENED: RSI 30-70 range instead of 40-60 to allow more signals
        if rsi < 30 or rsi > 70:
            logger.info(f"RSI={rsi:.1f} outside 30-70 range - no signal (extreme conditions)")
            return None

        # Check for MA crossover
        ma_bullish_cross = fast_ma > slow_ma and fast_ma_prev <= slow_ma_prev
        ma_bearish_cross = fast_ma < slow_ma and fast_ma_prev >= slow_ma_prev

        # Bullish crossover
        if ma_bullish_cross:
            logger.info(f"Bullish MA crossover detected (RSI={rsi:.1f})")
            return "buy"

        # Bearish crossover
        if ma_bearish_cross:
            logger.info(f"Bearish MA crossover detected (RSI={rsi:.1f})")
            return "sell"

        # Log why no signal was generated
        ma_diff_pct = ((fast_ma - slow_ma) / slow_ma) * 100
        logger.debug(f"No crossover signal: FastMA-SlowMA={ma_diff_pct:.3f}%, RSI={rsi:.1f}")

        return None

    def check_exit_signal(self, instrument: str, indicators: Dict) -> bool:
        """
        Check for exit signal on existing position.

        Args:
            instrument: Currency pair
            indicators: Dictionary of indicator values

        Returns:
            True if should exit
        """
        if not all(k in indicators for k in ["fast_ma", "slow_ma", "fast_ma_prev", "slow_ma_prev"]):
            return False

        fast_ma = indicators["fast_ma"]
        slow_ma = indicators["slow_ma"]
        fast_ma_prev = indicators["fast_ma_prev"]
        slow_ma_prev = indicators["slow_ma_prev"]

        # Check for crossover in opposite direction
        # If long: exit on bearish crossover
        # If short: exit on bullish crossover

        positions = self.get_positions()
        for pos in positions:
            if pos.get("instrument") == instrument:
                long_units = float(pos.get("long", {}).get("units", 0))
                short_units = float(pos.get("short", {}).get("units", 0))

                # Long position: exit on bearish crossover
                if long_units > 0 and fast_ma < slow_ma and fast_ma_prev >= slow_ma_prev:
                    return True

                # Short position: exit on bullish crossover
                if short_units < 0 and fast_ma > slow_ma and fast_ma_prev <= slow_ma_prev:
                    return True

        return False

    def run_strategy(self) -> List[Dict[str, Any]]:
        """
        Run strategy for all forex pairs.

        Returns:
            List of signal dictionaries compatible with the orchestrator.
            Each signal dict has the following structure:
            {
                "action": str,         # "buy", "sell", or "hold" (orchestrator-compatible)
                "symbol": str,         # Currency pair (e.g., "EUR_USD")
                "quantity": int,       # Position size in units
                "price": float,        # Current price
                "stop_loss": float,    # Stop loss price
                "take_profit": float,  # Take profit price
                "reason": str,         # Why this signal was generated
                "indicators": Dict,    # Current indicator values
                "bot_name": str,       # Bot identifier
            }

            Note: Returns signals for orchestrator to execute. Does NOT place orders internally.
        """
        signals = []
        positions = self.get_positions()
        position_instruments = [p["instrument"] for p in positions]

        for pair in FOREX_PAIRS:
            logger.info(f"Analyzing {pair}...")

            signal_data = {
                "action": "hold",
                "symbol": pair,
                "quantity": 0,
                "price": 0.0,
                "stop_loss": None,
                "take_profit": None,
                "reason": "",
                "indicators": {},
                "bot_name": "oanda_bot"
            }

            # Get candles
            candles = self.get_candles(
                pair,
                granularity=self.config["timeframe"],
                count=50
            )

            if not candles:
                signal_data["reason"] = "Failed to fetch candle data"
                signal_data["action"] = "hold"
                logger.warning(f"{pair}: {signal_data['reason']}")
                signals.append(signal_data)
                continue

            # Calculate indicators
            indicators = self.calculate_indicators(candles)
            if not indicators:
                signal_data["reason"] = "Insufficient data for indicator calculation"
                signal_data["action"] = "hold"
                logger.warning(f"{pair}: {signal_data['reason']}")
                signals.append(signal_data)
                continue

            signal_data["indicators"] = indicators
            signal_data["price"] = indicators.get("close", 0.0)

            logger.info(f"{pair}: Close={indicators.get('close'):.5f}, "
                       f"FastMA={indicators.get('fast_ma'):.5f}, "
                       f"SlowMA={indicators.get('slow_ma'):.5f}, "
                       f"RSI={indicators.get('rsi'):.1f}")

            # Check for exit on existing position
            if pair in position_instruments:
                if self.check_exit_signal(pair, indicators):
                    logger.info(f"Exit signal for {pair}")
                    # Return sell signal for orchestrator to close position
                    signal_data["action"] = "sell"
                    signal_data["reason"] = "MA crossover exit signal - close position"
                    signal_data["quantity"] = 0  # Close existing position
                else:
                    signal_data["action"] = "hold"
                    signal_data["reason"] = "Holding existing position"
                signals.append(signal_data)
                continue

            # Check for entry if no position and under max trades
            if len(positions) >= self.config["max_concurrent_trades"]:
                signal_data["action"] = "hold"
                signal_data["reason"] = f"Max concurrent trades ({self.config['max_concurrent_trades']}) reached"
                logger.info(f"{pair}: {signal_data['reason']}")
                signals.append(signal_data)
                continue

            signal = self.check_entry_signal(indicators)
            if signal:
                logger.info(f"Entry signal for {pair}: {signal}")
                signal_data["action"] = signal  # "buy" or "sell"

                # Calculate position size and stop/target
                atr = indicators["atr"]
                current_price = indicators["close"]
                units = self.calculate_position_size(atr, current_price)

                if units == 0:
                    signal_data["action"] = "hold"
                    signal_data["reason"] = "Position size calculation returned 0"
                    logger.warning(f"{pair}: {signal_data['reason']}")
                    signals.append(signal_data)
                    continue

                stop_distance = atr * self.config["atr_stop_multiplier"]
                profit_distance = atr * self.config["atr_profit_multiplier"]

                if signal == "buy":
                    stop_loss = current_price - stop_distance
                    take_profit = current_price + profit_distance
                else:  # sell
                    units = -units
                    stop_loss = current_price + stop_distance
                    take_profit = current_price - profit_distance

                # Return signal for orchestrator to execute (no internal order placement)
                signal_data["quantity"] = abs(units)
                signal_data["price"] = current_price
                signal_data["stop_loss"] = stop_loss
                signal_data["take_profit"] = take_profit
                signal_data["reason"] = f"{signal.upper()} signal from {'divergence' if indicators.get('hidden_bullish') or indicators.get('hidden_bearish') else 'MA crossover'}"
                signal_data["atr"] = atr
                signal_data["units_signed"] = units  # Positive for buy, negative for sell
            else:
                # No entry signal - return hold
                signal_data["action"] = "hold"

                # Log why no signal was generated
                rsi = indicators.get("rsi", 0)
                fast_ma = indicators.get("fast_ma", 0)
                slow_ma = indicators.get("slow_ma", 0)

                if rsi < 30:
                    signal_data["reason"] = f"RSI too low ({rsi:.1f}) - oversold, waiting"
                elif rsi > 70:
                    signal_data["reason"] = f"RSI too high ({rsi:.1f}) - overbought, waiting"
                elif fast_ma > slow_ma:
                    signal_data["reason"] = f"Uptrend but no crossover (FastMA={fast_ma:.5f} > SlowMA={slow_ma:.5f})"
                elif fast_ma < slow_ma:
                    signal_data["reason"] = f"Downtrend but no crossover (FastMA={fast_ma:.5f} < SlowMA={slow_ma:.5f})"
                else:
                    signal_data["reason"] = "MAs converging, waiting for crossover"

                logger.info(f"{pair}: No signal - {signal_data['reason']}")

            signals.append(signal_data)

        # Log summary
        buy_signals = [s for s in signals if s["action"] == "buy"]
        sell_signals = [s for s in signals if s["action"] == "sell"]
        hold_signals = [s for s in signals if s["action"] == "hold"]
        actionable_signals = [s for s in signals if s["action"] in ["buy", "sell"]]

        logger.info(f"Strategy run complete: {len(buy_signals)} buy, {len(sell_signals)} sell, "
                   f"{len(hold_signals)} hold, {len(actionable_signals)} actionable signals")

        return signals

    def execute_signal(self, signal: Dict[str, Any]) -> Optional[Dict]:
        """
        Execute a signal by placing an order. Used for standalone mode.

        Args:
            signal: Signal dictionary with action, symbol, quantity, stop_loss, take_profit

        Returns:
            Order result dict or None if no action taken
        """
        if signal.get("action") not in ["buy", "sell"]:
            return None

        symbol = signal.get("symbol")
        action = signal.get("action")
        quantity = signal.get("quantity", 0)
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        units_signed = signal.get("units_signed", quantity if action == "buy" else -quantity)

        if quantity == 0 and action == "sell":
            # Close position
            return self.close_position(symbol)

        if quantity == 0:
            logger.warning(f"Cannot execute {action} for {symbol}: quantity is 0")
            return None

        return self.market_order(
            instrument=symbol,
            units=units_signed,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def run(self, interval: int = 300):
        """
        Main bot loop.

        Args:
            interval: Seconds between iterations (default 5 min)
        """
        logger.info("Starting OANDA bot...")
        logger.info(f"Paper mode: {self.paper_mode}")
        logger.info(f"Timeframe: {self.config['timeframe']}")
        logger.info(f"Pairs: {FOREX_PAIRS}")

        # Verify connection
        account = self.get_account()
        if account:
            logger.info(f"Account balance: {account.get('balance')}")
        else:
            logger.error("Failed to connect to OANDA, exiting")
            return

        while True:
            try:
                signals = self.run_strategy()

                # Execute signals in standalone mode
                if signals:
                    active_signals = [s for s in signals if s["action"] != "hold"]
                    if active_signals:
                        logger.info(f"Active signals this iteration: {len(active_signals)}")
                        for sig in active_signals:
                            logger.info(f"  {sig['symbol']}: {sig['action'].upper()} - {sig['reason']}")
                            # Execute the signal
                            order_result = self.execute_signal(sig)
                            if order_result:
                                logger.info(f"  Order executed: {order_result}")

                # Log account status
                account = self.get_account()
                if account:
                    balance = float(account.get("balance", 0))
                    unrealized_pnl = float(account.get("unrealizedPL", 0))
                    logger.info(f"Balance: {balance:.2f}, Unrealized P&L: {unrealized_pnl:.2f}")

                time.sleep(interval)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(interval * 2)


def main():
    """Entry point for the OANDA bot."""
    bot = OANDABot()
    bot.run(interval=300)  # Run every 5 minutes


if __name__ == "__main__":
    main()
