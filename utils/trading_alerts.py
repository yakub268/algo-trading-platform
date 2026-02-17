"""
Enhanced Telegram Alerts for Trading Bot

Sends formatted alerts for:
- Buy/Sell signals
- Tier exits (3-tier system)
- Risk events (circuit breakers, VIX)
- Daily summaries
- System status

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import requests
import logging
from datetime import datetime
from typing import Optional
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TelegramAlerts')


class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradingAlerts:
    """Enhanced Telegram alerts for trading signals."""
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None
    ):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if self.enabled:
            logger.info("TradingAlerts initialized")
        else:
            logger.warning("TradingAlerts disabled - missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    
    def _send(self, text: str, silent: bool = False) -> bool:
        """Send message to Telegram."""
        if not self.enabled:
            logger.debug(f"Alert (disabled): {text[:50]}...")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'HTML',
                'disable_notification': silent
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False
    
    # =========================================================================
    # SIGNAL ALERTS
    # =========================================================================
    
    def buy_signal(
        self,
        symbol: str,
        strategy: str,
        price: float,
        stop_loss: float,
        confidence: float,
        position_size: float,
        reason: str = ""
    ):
        """Send buy signal alert."""
        risk_pct = (price - stop_loss) / price * 100
        
        text = f"""
🟢 <b>BUY SIGNAL: {symbol}</b>

📊 Strategy: {strategy}
💵 Entry: ${price:.2f}
🛑 Stop: ${stop_loss:.2f} ({risk_pct:.1f}%)
📈 Confidence: {confidence:.0%}
💰 Size: ${position_size:.2f}

{f'📝 {reason}' if reason else ''}

<i>Using 3-tier scaled exits</i>
"""
        return self._send(text.strip())
    
    def sell_signal(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        reason: str = ""
    ):
        """Send sell/exit alert."""
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        emoji = "💰" if pnl > 0 else "📉"
        
        text = f"""
🔴 <b>POSITION CLOSED: {symbol}</b>

{emoji} P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)
💵 Entry: ${entry_price:.2f}
💵 Exit: ${exit_price:.2f}

{f'📝 {reason}' if reason else ''}
"""
        return self._send(text.strip())

    def trade_closed(
        self,
        symbol: str,
        strategy: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        hold_time: str = ""
    ):
        """Send trade closed alert."""
        emoji = "💰" if pnl > 0 else "📉"
        result = "WIN" if pnl > 0 else "LOSS"

        text = f"""
{emoji} <b>TRADE CLOSED: {symbol}</b>

📊 Strategy: {strategy}
🎯 Result: {result}
💵 Entry: ${entry_price:.2f}
💵 Exit: ${exit_price:.2f}
📈 P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)
{f'⏱️ Duration: {hold_time}' if hold_time else ''}
"""
        return self._send(text.strip())

    # =========================================================================
    # TIER EXIT ALERTS
    # =========================================================================
    
    def tier_exit(
        self,
        symbol: str,
        tier: str,
        shares: float,
        exit_price: float,
        pnl: float,
        reason: str
    ):
        """Send tier exit alert."""
        tier_emoji = {'tier_1': '1️⃣', 'tier_2': '2️⃣', 'tier_3': '3️⃣'}.get(tier, '🔹')
        pnl_emoji = "✅" if pnl > 0 else "❌"
        
        text = f"""
{tier_emoji} <b>TIER EXIT: {symbol}</b>

{pnl_emoji} P&L: ${pnl:.2f}
📊 Shares: {shares:.1f}
💵 Price: ${exit_price:.2f}
📝 {reason}
"""
        self._send(text.strip())
    
    # =========================================================================
    # RISK ALERTS
    # =========================================================================
    
    def circuit_breaker(self, reason: str, action: str):
        """Send circuit breaker alert."""
        text = f"""
🛑 <b>CIRCUIT BREAKER TRIGGERED</b>

⚠️ Reason: {reason}
⚡ Action: {action}

<i>Trading halted until reset</i>
"""
        self._send(text.strip())
    
    def vix_alert(self, vix: float, regime: str, position_mult: float):
        """Send VIX regime alert."""
        text = f"""
🌡️ <b>VIX ALERT</b>

📊 VIX: {vix:.1f}
🎯 Regime: {regime}
📉 Position Size: {position_mult:.0%}

<i>Positions adjusted automatically</i>
"""
        self._send(text.strip())
    
    def consecutive_losses(self, count: int, action: str):
        """Send consecutive losses warning."""
        text = f"""
⚠️ <b>CONSECUTIVE LOSSES: {count}</b>

⚡ Action: {action}

<i>Review strategy performance</i>
"""
        self._send(text.strip())
    
    # =========================================================================
    # MOMENTUM ALERTS
    # =========================================================================
    
    def dual_momentum_rebalance(
        self,
        from_holding: str,
        to_holding: str,
        spy_momentum: float,
        efa_momentum: float
    ):
        """Send dual momentum rebalance alert."""
        text = f"""
🔄 <b>DUAL MOMENTUM REBALANCE</b>

📊 Switch: {from_holding} → {to_holding}
📈 SPY 12M: {spy_momentum:+.1%}
📈 EFA 12M: {efa_momentum:+.1%}

<i>Monthly rebalance complete</i>
"""
        self._send(text.strip())
    
    # =========================================================================
    # SUMMARY ALERTS
    # =========================================================================
    
    def daily_summary(
        self,
        trades: int,
        wins: int,
        pnl: float,
        open_positions: int,
        portfolio_value: float
    ):
        """Send daily summary."""
        win_rate = (wins / trades * 100) if trades > 0 else 0
        pnl_emoji = "💰" if pnl >= 0 else "📉"
        
        text = f"""
📊 <b>DAILY SUMMARY</b>

📈 Trades: {trades} ({wins} wins, {win_rate:.0f}%)
{pnl_emoji} Day P&L: ${pnl:+.2f}
📂 Open Positions: {open_positions}
💰 Portfolio: ${portfolio_value:,.2f}
"""
        return self._send(text.strip(), silent=True)
    
    # =========================================================================
    # SYSTEM ALERTS
    # =========================================================================
    
    def system_started(self, mode: str = "Paper"):
        """Send system started alert."""
        text = f"""
🚀 <b>TRADING BOT STARTED</b>

📊 Mode: {mode} Trading
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

✅ All systems operational
"""
        return self._send(text.strip())

    def system_stopped(self, reason: str = "Manual"):
        """Send system stopped alert."""
        text = f"""
⛔ <b>TRADING BOT STOPPED</b>

📝 Reason: {reason}
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        self._send(text.strip())
    
    def system_error(self, error: str, component: str = "Unknown"):
        """Send error alert."""
        text = f"""
❌ <b>ERROR</b>

🔧 Component: {component}
📝 Error: {error}
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        self._send(text.strip())
    
    # =========================================================================
    # GENERIC
    # =========================================================================
    
    def send(self, message: str, silent: bool = False):
        """Send generic message."""
        self._send(message, silent)


# Global instance
_alerts: Optional[TradingAlerts] = None

def get_alerts() -> TradingAlerts:
    """Get global alerts instance."""
    global _alerts
    if _alerts is None:
        _alerts = TradingAlerts()
    return _alerts

def send_alert(message: str):
    """Quick send."""
    get_alerts().send(message)


if __name__ == "__main__":
    alerts = TradingAlerts()
    
    print("Testing alerts...")
    
    alerts.buy_signal(
        symbol="SPY",
        strategy="RSI2-Enhanced",
        price=500.00,
        stop_loss=490.00,
        confidence=0.85,
        position_size=1000,
        reason="RSI < 5, 3+ bars, above 200 SMA"
    )
    
    print("Test complete!")
