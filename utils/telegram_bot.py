"""
Telegram Bot for Trading Alerts
Sends real-time notifications for trades, alerts, and daily summaries.
"""

import os
import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum

import aiohttp
from aiohttp import TCPConnector
from aiohttp.resolver import ThreadedResolver
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TradeAlert:
    symbol: str
    action: str  # BUY, SELL, SHORT, COVER
    price: float
    quantity: float
    strategy: str
    timestamp: datetime
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: Optional[float] = None


@dataclass
class TradeClosedAlert:
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    hold_time: str
    strategy: str
    exit_reason: str
    timestamp: datetime


class TelegramBot:
    """Telegram bot for sending trading alerts and summaries."""

    BASE_URL = "https://api.telegram.org/bot"

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN not set - alerts will be logged only")
        if not self.chat_id:
            logger.warning("TELEGRAM_CHAT_ID not set - alerts will be logged only")

        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def is_configured(self) -> bool:
        """Check if bot is properly configured."""
        return bool(self.bot_token and self.chat_id)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with Windows-compatible DNS resolver."""
        # Check if session exists, is not closed, and its event loop is still running
        need_new_session = False

        if self._session is None:
            need_new_session = True
        elif self._session.closed:
            need_new_session = True
        else:
            # Check if the session's loop is closed (happens with asyncio.run())
            try:
                loop = self._session._loop
                if loop.is_closed():
                    need_new_session = True
            except Exception:
                need_new_session = True

        if need_new_session:
            # Close old session if it exists
            if self._session and not self._session.closed:
                try:
                    await self._session.close()
                except Exception:
                    pass

            # Use ThreadedResolver for Windows DNS compatibility
            resolver = ThreadedResolver()
            connector = TCPConnector(resolver=resolver)
            self._session = aiohttp.ClientSession(connector=connector)

        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False
    ) -> bool:
        """Send a message via Telegram API."""
        if not self.is_configured:
            logger.info(f"[TELEGRAM DISABLED] {text}")
            return False

        url = f"{self.BASE_URL}{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.debug(f"Telegram message sent successfully")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Telegram API error: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Synchronous message sending for quick tests."""
        import requests
        if not self.is_configured:
            logger.info(f"[TELEGRAM DISABLED] {text}")
            return False
        url = f"{self.BASE_URL}{self.bot_token}/sendMessage"
        try:
            response = requests.post(url, data={'chat_id': self.chat_id, 'text': text, 'parse_mode': parse_mode}, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def _format_currency(self, value: float) -> str:
        """Format currency value."""
        return f"${value:,.2f}"

    def _format_percent(self, value: float) -> str:
        """Format percentage value."""
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.2f}%"

    def _get_emoji(self, action: str) -> str:
        """Get emoji for action type."""
        emoji_map = {
            "BUY": "🟢",
            "SELL": "🔴",
            "SHORT": "🔻",
            "COVER": "🔺",
            "PROFIT": "💰",
            "LOSS": "📉",
            "ALERT": "⚠️",
            "CRITICAL": "🚨",
            "INFO": "ℹ️",
            "WEATHER": "🌤️",
            "KALSHI": "🎯",
        }
        return emoji_map.get(action.upper(), "📊")

    async def send_trade_alert(self, alert: TradeAlert) -> bool:
        """
        Send a trade entry alert.

        Args:
            alert: TradeAlert dataclass with trade details

        Returns:
            True if message sent successfully
        """
        emoji = self._get_emoji(alert.action)
        confidence_str = f" ({alert.confidence:.0%} conf)" if alert.confidence else ""

        message = f"""
{emoji} <b>TRADE ALERT: {alert.action}</b>

<b>Symbol:</b> {alert.symbol}
<b>Price:</b> {self._format_currency(alert.price)}
<b>Quantity:</b> {alert.quantity}
<b>Strategy:</b> {alert.strategy}{confidence_str}

<b>Reason:</b> {alert.reason}
"""

        if alert.stop_loss:
            message += f"<b>Stop Loss:</b> {self._format_currency(alert.stop_loss)}\n"
        if alert.take_profit:
            message += f"<b>Take Profit:</b> {self._format_currency(alert.take_profit)}\n"

        message += f"\n<i>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</i>"

        logger.info(f"Trade alert: {alert.action} {alert.symbol} @ {alert.price}")
        return await self._send_message(message.strip())

    async def send_trade_closed_alert(self, alert: TradeClosedAlert) -> bool:
        """
        Send a trade closed alert with P&L.

        Args:
            alert: TradeClosedAlert dataclass with trade closure details

        Returns:
            True if message sent successfully
        """
        is_profit = alert.pnl >= 0
        emoji = self._get_emoji("PROFIT" if is_profit else "LOSS")
        pnl_color = "green" if is_profit else "red"

        message = f"""
{emoji} <b>TRADE CLOSED</b>

<b>Symbol:</b> {alert.symbol}
<b>Entry:</b> {self._format_currency(alert.entry_price)}
<b>Exit:</b> {self._format_currency(alert.exit_price)}
<b>Quantity:</b> {alert.quantity}

<b>P&L:</b> {self._format_currency(alert.pnl)} ({self._format_percent(alert.pnl_percent)})
<b>Hold Time:</b> {alert.hold_time}
<b>Strategy:</b> {alert.strategy}
<b>Exit Reason:</b> {alert.exit_reason}

<i>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</i>
"""

        logger.info(f"Trade closed: {alert.symbol} P&L: {alert.pnl:.2f} ({alert.pnl_percent:.2f}%)")
        return await self._send_message(message.strip())

    async def send_circuit_breaker_alert(
        self,
        reason: str,
        current_drawdown: float,
        daily_loss: float,
        positions_closed: int = 0,
        resume_time: Optional[datetime] = None
    ) -> bool:
        """
        Send circuit breaker triggered alert.

        Args:
            reason: Why circuit breaker was triggered
            current_drawdown: Current portfolio drawdown percentage
            daily_loss: Today's total loss
            positions_closed: Number of positions force-closed
            resume_time: When trading can resume (if known)

        Returns:
            True if message sent successfully
        """
        emoji = self._get_emoji("CRITICAL")

        message = f"""
{emoji} <b>CIRCUIT BREAKER TRIGGERED</b> {emoji}

<b>Reason:</b> {reason}
<b>Current Drawdown:</b> {self._format_percent(current_drawdown)}
<b>Daily Loss:</b> {self._format_currency(daily_loss)}
"""

        if positions_closed > 0:
            message += f"<b>Positions Closed:</b> {positions_closed}\n"

        if resume_time:
            message += f"<b>Resume:</b> {resume_time.strftime('%Y-%m-%d %H:%M')}\n"
        else:
            message += "<b>Status:</b> Manual review required\n"

        message += f"\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"

        logger.warning(f"Circuit breaker: {reason} - Drawdown: {current_drawdown:.2f}%")
        return await self._send_message(message.strip(), disable_notification=False)

    async def send_daily_summary(
        self,
        date: datetime,
        total_trades: int,
        winning_trades: int,
        total_pnl: float,
        best_trade: Optional[Dict[str, Any]] = None,
        worst_trade: Optional[Dict[str, Any]] = None,
        strategies_performance: Optional[Dict[str, Dict[str, Any]]] = None,
        portfolio_value: Optional[float] = None
    ) -> bool:
        """
        Send end-of-day trading summary.

        Args:
            date: Summary date
            total_trades: Number of trades executed
            winning_trades: Number of profitable trades
            total_pnl: Total P&L for the day
            best_trade: Best performing trade details
            worst_trade: Worst performing trade details
            strategies_performance: Performance breakdown by strategy
            portfolio_value: End-of-day portfolio value

        Returns:
            True if message sent successfully
        """
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        emoji = "📈" if total_pnl >= 0 else "📉"

        message = f"""
{emoji} <b>DAILY SUMMARY - {date.strftime('%Y-%m-%d')}</b>

<b>Total Trades:</b> {total_trades}
<b>Win Rate:</b> {win_rate:.1f}% ({winning_trades}/{total_trades})
<b>Daily P&L:</b> {self._format_currency(total_pnl)}
"""

        if portfolio_value:
            message += f"<b>Portfolio Value:</b> {self._format_currency(portfolio_value)}\n"

        if best_trade:
            message += f"\n<b>Best Trade:</b> {best_trade.get('symbol', 'N/A')} ({self._format_currency(best_trade.get('pnl', 0))})"

        if worst_trade:
            message += f"\n<b>Worst Trade:</b> {worst_trade.get('symbol', 'N/A')} ({self._format_currency(worst_trade.get('pnl', 0))})"

        if strategies_performance:
            message += "\n\n<b>By Strategy:</b>"
            for strategy, perf in strategies_performance.items():
                strat_pnl = perf.get('pnl', 0)
                strat_trades = perf.get('trades', 0)
                message += f"\n• {strategy}: {self._format_currency(strat_pnl)} ({strat_trades} trades)"

        message += f"\n\n<i>Generated at {datetime.now().strftime('%H:%M:%S')}</i>"

        logger.info(f"Daily summary: {total_trades} trades, P&L: {total_pnl:.2f}")
        return await self._send_message(message.strip(), disable_notification=True)

    async def send_opportunity_alert(
        self,
        source: str,  # KALSHI, WEATHER, RSI2, etc.
        symbol: str,
        opportunity_type: str,
        edge: float,
        confidence: float,
        details: str,
        expires: Optional[datetime] = None,
        priority: AlertPriority = AlertPriority.MEDIUM
    ) -> bool:
        """
        Send opportunity/signal alert.

        Args:
            source: Signal source (KALSHI, WEATHER, RSI2, etc.)
            symbol: Ticker/market symbol
            opportunity_type: Type of opportunity (LONG, SHORT, YES, NO)
            edge: Expected edge/alpha percentage
            confidence: Confidence level 0-1
            details: Description of the opportunity
            expires: When the opportunity expires
            priority: Alert priority level

        Returns:
            True if message sent successfully
        """
        source_emoji = self._get_emoji(source)
        priority_indicator = "🔥" if priority == AlertPriority.HIGH else ""
        if priority == AlertPriority.CRITICAL:
            priority_indicator = "🚨"

        message = f"""
{source_emoji} <b>OPPORTUNITY: {source}</b> {priority_indicator}

<b>Symbol:</b> {symbol}
<b>Type:</b> {opportunity_type}
<b>Edge:</b> {self._format_percent(edge)}
<b>Confidence:</b> {confidence:.0%}

<b>Details:</b> {details}
"""

        if expires:
            message += f"<b>Expires:</b> {expires.strftime('%Y-%m-%d %H:%M')}\n"

        message += f"\n<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"

        # Only send notification for high priority
        silent = priority not in [AlertPriority.HIGH, AlertPriority.CRITICAL]

        logger.info(f"Opportunity alert: {source} {symbol} {opportunity_type} - Edge: {edge:.2f}%")
        return await self._send_message(message.strip(), disable_notification=silent)


class TelegramCommandListener:
    """
    Polls Telegram for incoming commands and dispatches them.

    Security:
    - Only responds to messages from the configured TELEGRAM_CHAT_ID
    - Ignores all other users/chats
    - Logs all command attempts (authorized and unauthorized)
    - Requires exact command text (no fuzzy matching)

    Supported commands:
    - /emergency_stop  - Halt all trading and close all positions
    - /kill            - Alias for /emergency_stop
    - /status          - Reply with current bot status summary
    """

    POLL_INTERVAL = 3  # seconds between getUpdates calls
    POLL_TIMEOUT = 10  # long-polling timeout for getUpdates
    CONFLICT_BACKOFF_INIT = 5  # initial backoff on 409 (seconds)
    CONFLICT_BACKOFF_MAX = 60  # max backoff on 409 (seconds)

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token or not self.chat_id:
            logger.warning("TelegramCommandListener: Missing bot token or chat_id - listener disabled")

        self._base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else ""
        self._last_update_id: Optional[int] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callback for emergency stop - set via register_emergency_callback()
        self._emergency_stop_callback: Optional[Callable] = None
        # Callback for status - set via register_status_callback()
        self._status_callback: Optional[Callable] = None
        # Exponential backoff state for 409 conflicts
        self._conflict_backoff = self.CONFLICT_BACKOFF_INIT

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def register_emergency_callback(self, callback: Callable):
        """
        Register the function to call when /emergency_stop or /kill is received.

        The callback should accept a single string argument (the reason).
        Typically this is TradingSystemIntegration.emergency_stop_all
        or MasterOrchestrator's stop method.
        """
        self._emergency_stop_callback = callback
        logger.info("TelegramCommandListener: Emergency stop callback registered")

    def register_status_callback(self, callback: Callable):
        """
        Register the function to call when /status is received.

        The callback should return a string or dict with the current status.
        """
        self._status_callback = callback
        logger.info("TelegramCommandListener: Status callback registered")

    def _send_reply(self, text: str):
        """Send a reply message back to the authorized chat."""
        if not self.is_configured:
            return
        url = f"{self._base_url}/sendMessage"
        try:
            requests.post(url, data={
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'HTML',
            }, timeout=10)
        except Exception as e:
            logger.error(f"TelegramCommandListener: Failed to send reply: {e}")

    def _get_updates(self) -> List[Dict]:
        """Fetch new messages from Telegram using getUpdates with long polling."""
        if not self.is_configured:
            return []

        url = f"{self._base_url}/getUpdates"
        params: Dict[str, Any] = {
            'timeout': self.POLL_TIMEOUT,
        }
        if self._last_update_id is not None:
            params['offset'] = self._last_update_id + 1

        try:
            response = requests.get(url, params=params, timeout=self.POLL_TIMEOUT + 5)
            if response.status_code == 409:
                # Conflict — another getUpdates caller or stale webhook
                logger.warning(
                    f"TelegramCommandListener: 409 conflict - clearing webhook, "
                    f"backing off {self._conflict_backoff}s"
                )
                self._delete_webhook()
                time.sleep(self._conflict_backoff)
                # Exponential backoff: 5 → 10 → 20 → 40 → 60 (capped)
                self._conflict_backoff = min(
                    self._conflict_backoff * 2, self.CONFLICT_BACKOFF_MAX
                )
                return []
            if response.status_code != 200:
                logger.warning(f"TelegramCommandListener: getUpdates returned {response.status_code}")
                return []

            data = response.json()
            if not data.get('ok'):
                logger.warning(f"TelegramCommandListener: getUpdates not ok: {data}")
                return []

            # Successful poll — reset backoff
            self._conflict_backoff = self.CONFLICT_BACKOFF_INIT
            return data.get('result', [])

        except requests.exceptions.Timeout:
            # Normal for long polling - just means no new messages
            return []
        except Exception as e:
            logger.error(f"TelegramCommandListener: getUpdates error: {e}")
            return []

    def _process_update(self, update: Dict):
        """Process a single Telegram update."""
        update_id = update.get('update_id')
        if update_id is not None:
            self._last_update_id = update_id

        message = update.get('message')
        if not message:
            return

        # Extract sender info for security check
        chat = message.get('chat', {})
        from_user = message.get('from', {})
        message_chat_id = str(chat.get('id', ''))
        sender_username = from_user.get('username', 'unknown')
        sender_id = from_user.get('id', 'unknown')
        text = (message.get('text') or '').strip()

        # SECURITY: Only process commands from the authorized chat_id
        if message_chat_id != str(self.chat_id):
            logger.warning(
                f"UNAUTHORIZED command attempt from chat_id={message_chat_id} "
                f"user={sender_username} (id={sender_id}): {text!r}"
            )
            return

        # Log authorized command
        logger.info(f"Telegram command from authorized user ({sender_username}): {text!r}")

        # Dispatch exact commands only
        if text == '/emergency_stop' or text == '/kill':
            self._handle_emergency_stop(sender_username)
        elif text == '/status':
            self._handle_status(sender_username)
        elif text.startswith('/'):
            # Known command prefix but not recognized
            self._send_reply(
                "Unknown command. Available commands:\n"
                "/emergency_stop - Halt all trading and close positions\n"
                "/kill - Same as /emergency_stop\n"
                "/status - Get current system status"
            )

    def _handle_emergency_stop(self, sender: str):
        """Execute the emergency stop sequence."""
        logger.critical(f"EMERGENCY STOP triggered via Telegram by {sender}")

        self._send_reply(
            "\xf0\x9f\x9a\xa8 <b>EMERGENCY STOP INITIATED</b>\n\n"
            "Disabling all bots and closing all positions...\n"
            "Please wait for confirmation."
        )

        if self._emergency_stop_callback is None:
            error_msg = (
                "ERROR: No emergency stop callback registered.\n"
                "The command listener is running but emergency_stop_all() is not connected.\n"
                "Bots may still be trading!"
            )
            logger.error(f"TelegramCommandListener: {error_msg}")
            self._send_reply(f"\xe2\x9a\xa0\xef\xb8\x8f {error_msg}")
            return

        try:
            reason = f"Telegram /emergency_stop by {sender} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self._emergency_stop_callback(reason)

            self._send_reply(
                "\xe2\x9c\x85 <b>EMERGENCY STOP COMPLETE</b>\n\n"
                "All bots disabled.\n"
                "All broker positions closed (or attempted).\n"
                f"Triggered by: {sender}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            logger.critical("EMERGENCY STOP completed successfully via Telegram")

        except Exception as e:
            error_msg = f"Emergency stop callback FAILED: {e}"
            logger.error(f"TelegramCommandListener: {error_msg}")
            self._send_reply(f"\xe2\x9d\x8c <b>EMERGENCY STOP ERROR</b>\n\n{error_msg}")

    def _handle_status(self, sender: str):
        """Return current system status."""
        if self._status_callback is None:
            self._send_reply("Status callback not registered. System is running but status unavailable.")
            return

        try:
            status = self._status_callback()
            if isinstance(status, dict):
                import json
                # Build a readable summary
                lines = [
                    "<b>System Status</b>",
                    f"Running: {status.get('running', 'unknown')}",
                    f"Paper Mode: {status.get('paper_mode', 'unknown')}",
                    f"Total Bots: {status.get('total_bots', '?')}",
                    f"Capital: ${status.get('current_capital', 0):,.0f}",
                    f"Open Trades: {status.get('open_trades', 0)}",
                ]
                today = status.get('today', {})
                if today:
                    lines.append(f"Today's Trades: {today.get('trades', 0)}")
                    lines.append(f"Today's P&L: ${today.get('total_pnl', 0):,.2f}")
                self._send_reply('\n'.join(lines))
            else:
                self._send_reply(str(status))
        except Exception as e:
            logger.error(f"TelegramCommandListener: Status callback failed: {e}")
            self._send_reply(f"Error getting status: {e}")

    def _delete_webhook(self, drop_pending: bool = True):
        """Delete any existing webhook so getUpdates works (they're mutually exclusive)."""
        if not self.is_configured:
            return
        try:
            url = f"{self._base_url}/deleteWebhook"
            resp = requests.post(url, json={"drop_pending_updates": drop_pending}, timeout=10)
            if resp.status_code == 200 and resp.json().get("ok"):
                logger.info("TelegramCommandListener: Webhook cleared (getUpdates mode)")
            else:
                logger.warning(f"TelegramCommandListener: deleteWebhook response: {resp.text}")
        except Exception as e:
            logger.warning(f"TelegramCommandListener: deleteWebhook failed: {e}")

    def _poll_loop(self):
        """Main polling loop - runs in background thread."""
        logger.info("TelegramCommandListener: Polling loop started")

        # Clear any existing webhook — webhook + getUpdates conflict causes 409
        self._delete_webhook()

        # On startup, consume any old pending updates so we don't
        # process stale /emergency_stop commands from before boot
        try:
            old_updates = self._get_updates()
            if old_updates:
                # Just advance the offset past all existing messages
                last = old_updates[-1]
                self._last_update_id = last.get('update_id')
                logger.info(
                    f"TelegramCommandListener: Skipped {len(old_updates)} old messages on startup"
                )
        except Exception as e:
            logger.warning(f"TelegramCommandListener: Failed to clear old updates: {e}")

        while self._running:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._process_update(update)
            except Exception as e:
                logger.error(f"TelegramCommandListener: Poll loop error: {e}")
                time.sleep(5)  # Back off on error

            # Small sleep between poll cycles (long-polling already waits POLL_TIMEOUT)
            if self._running:
                time.sleep(self.POLL_INTERVAL)

        logger.info("TelegramCommandListener: Polling loop stopped")

    def start(self):
        """Start the command listener in a background daemon thread."""
        if not self.is_configured:
            logger.warning("TelegramCommandListener: Not starting - not configured")
            return

        if self._running:
            logger.warning("TelegramCommandListener: Already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="TelegramCommandListener",
            daemon=True,
        )
        self._thread.start()
        logger.info("TelegramCommandListener: Started (background thread)")

    def stop(self):
        """Stop the command listener."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.POLL_TIMEOUT + 5)
        logger.info("TelegramCommandListener: Stopped")


# Global command listener instance
_command_listener: Optional[TelegramCommandListener] = None


def get_command_listener() -> TelegramCommandListener:
    """Get or create global command listener instance."""
    global _command_listener
    if _command_listener is None:
        _command_listener = TelegramCommandListener()
    return _command_listener


# Synchronous wrapper functions for easy integration
def _run_async(coro):
    """Run async function synchronously (Python 3.10+ compatible)."""
    try:
        loop = asyncio.get_running_loop()
        # We're already in an event loop, can't use run_until_complete
        raise RuntimeError("Cannot use _run_async from within an event loop")
    except RuntimeError as e:
        if "no running event loop" in str(e).lower():
            # No event loop running, safe to create one
            return asyncio.run(coro)
        else:
            # Already in a loop
            raise


# Global bot instance
_bot: Optional[TelegramBot] = None


def get_bot() -> TelegramBot:
    """Get or create global bot instance."""
    global _bot
    if _bot is None:
        _bot = TelegramBot()
    return _bot


def send_trade_alert(
    symbol: str,
    action: str,
    price: float,
    quantity: float,
    strategy: str,
    reason: str,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    confidence: Optional[float] = None
) -> bool:
    """Synchronous wrapper for send_trade_alert."""
    alert = TradeAlert(
        symbol=symbol,
        action=action,
        price=price,
        quantity=quantity,
        strategy=strategy,
        timestamp=datetime.now(),
        reason=reason,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence
    )
    return _run_async(get_bot().send_trade_alert(alert))


def send_trade_closed_alert(
    symbol: str,
    entry_price: float,
    exit_price: float,
    quantity: float,
    pnl: float,
    pnl_percent: float,
    hold_time: str,
    strategy: str,
    exit_reason: str
) -> bool:
    """Synchronous wrapper for send_trade_closed_alert."""
    alert = TradeClosedAlert(
        symbol=symbol,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        pnl=pnl,
        pnl_percent=pnl_percent,
        hold_time=hold_time,
        strategy=strategy,
        exit_reason=exit_reason,
        timestamp=datetime.now()
    )
    return _run_async(get_bot().send_trade_closed_alert(alert))


def send_circuit_breaker_alert(
    reason: str,
    current_drawdown: float,
    daily_loss: float,
    positions_closed: int = 0,
    resume_time: Optional[datetime] = None
) -> bool:
    """Synchronous wrapper for send_circuit_breaker_alert."""
    return _run_async(get_bot().send_circuit_breaker_alert(
        reason=reason,
        current_drawdown=current_drawdown,
        daily_loss=daily_loss,
        positions_closed=positions_closed,
        resume_time=resume_time
    ))


def send_daily_summary(
    date: datetime,
    total_trades: int,
    winning_trades: int,
    total_pnl: float,
    best_trade: Optional[Dict[str, Any]] = None,
    worst_trade: Optional[Dict[str, Any]] = None,
    strategies_performance: Optional[Dict[str, Dict[str, Any]]] = None,
    portfolio_value: Optional[float] = None
) -> bool:
    """Synchronous wrapper for send_daily_summary."""
    return _run_async(get_bot().send_daily_summary(
        date=date,
        total_trades=total_trades,
        winning_trades=winning_trades,
        total_pnl=total_pnl,
        best_trade=best_trade,
        worst_trade=worst_trade,
        strategies_performance=strategies_performance,
        portfolio_value=portfolio_value
    ))


def send_opportunity_alert(
    source: str,
    symbol: str,
    opportunity_type: str,
    edge: float,
    confidence: float,
    details: str,
    expires: Optional[datetime] = None,
    priority: str = "medium"
) -> bool:
    """Synchronous wrapper for send_opportunity_alert."""
    priority_enum = AlertPriority(priority.lower())
    return _run_async(get_bot().send_opportunity_alert(
        source=source,
        symbol=symbol,
        opportunity_type=opportunity_type,
        edge=edge,
        confidence=confidence,
        details=details,
        expires=expires,
        priority=priority_enum
    ))


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test trade alert
    send_trade_alert(
        symbol="AAPL",
        action="BUY",
        price=185.50,
        quantity=100,
        strategy="RSI2_MeanReversion",
        reason="RSI(2) dropped below 10, above 200 SMA",
        stop_loss=179.74,
        confidence=0.85
    )

    # Test trade closed alert
    send_trade_closed_alert(
        symbol="AAPL",
        entry_price=185.50,
        exit_price=192.30,
        quantity=100,
        pnl=680.00,
        pnl_percent=3.67,
        hold_time="2d 4h",
        strategy="RSI2_MeanReversion",
        exit_reason="RSI(2) > 90"
    )

    # Test opportunity alert
    send_opportunity_alert(
        source="KALSHI",
        symbol="KXBTC-26JAN27-B102000",
        opportunity_type="YES",
        edge=8.5,
        confidence=0.72,
        details="Bitcoin likely to close above $102k based on momentum",
        priority="high"
    )

    # Test daily summary
    send_daily_summary(
        date=datetime.now(),
        total_trades=15,
        winning_trades=11,
        total_pnl=1250.75,
        best_trade={"symbol": "TSLA", "pnl": 450.00},
        worst_trade={"symbol": "NVDA", "pnl": -120.00},
        strategies_performance={
            "RSI2_MeanReversion": {"pnl": 850.00, "trades": 8},
            "EMA_RSI": {"pnl": 400.75, "trades": 7}
        },
        portfolio_value=52500.00
    )

    print("Test alerts sent (check logs if Telegram not configured)")
