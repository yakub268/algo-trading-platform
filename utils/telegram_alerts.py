"""
Extended Telegram Alerts for All Strategies

Provides specialized alert formatting for each strategy type:
- Technical signals (RSI-2, MACD+RSI, Bollinger, MTF RSI)
- Event-driven (Earnings, Sentiment)
- Arbitrage (Crypto cross-exchange)

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from utils.telegram_bot import TelegramBot, AlertPriority

logger = logging.getLogger('TelegramAlerts')


class ExtendedTelegramAlerts:
    """
    Extended alert system for all trading strategies.
    
    Alert Types:
    - Strategy signals (buy/sell)
    - Arbitrage opportunities
    - Earnings surprises
    - Sentiment extremes
    - Daily summaries
    - Error notifications
    """
    
    # Emoji mapping for strategies
    STRATEGY_EMOJI = {
        'rsi2': '📊',
        'cumulative_rsi': '📈',
        'macd_rsi': '🔀',
        'bollinger_squeeze': '💥',
        'mtf_rsi': '🎯',
        'earnings': '📰',
        'sentiment': '💬',
        'crypto_arb': '⚡',
        'fomc': '🏛️',
        'weather': '🌤️',
        'sports': '🏈',
    }
    
    SIGNAL_EMOJI = {
        'buy': '🟢',
        'sell': '🔴',
        'hold': '⏸️',
        'bullish': '🟢',
        'bearish': '🔴',
        'neutral': '⚪',
    }
    
    def __init__(self, bot: Optional[TelegramBot] = None):
        self.bot = bot or TelegramBot()
        
        if not self.bot.is_configured:
            logger.warning("Telegram not configured - alerts will be logged only")
    
    async def send_strategy_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Send alert for trading signal.
        
        Args:
            signal: Dict with keys: strategy, symbol, signal, price, confidence, reasoning, etc.
        """
        strategy = signal.get('strategy', 'unknown')
        symbol = signal.get('symbol', '???')
        signal_type = signal.get('signal', 'hold')
        price = signal.get('price', 0)
        confidence = signal.get('confidence', 0)
        reasoning = signal.get('reasoning', '')
        
        strat_emoji = self.STRATEGY_EMOJI.get(strategy, '📌')
        sig_emoji = self.SIGNAL_EMOJI.get(signal_type.lower(), '⚪')
        
        # Build message
        message = f"{strat_emoji} <b>[{strategy.upper()}]</b> {sig_emoji} {signal_type.upper()}\n\n"
        message += f"<b>Symbol:</b> {symbol}\n"
        
        if price:
            message += f"<b>Price:</b> ${price:.2f}\n"
        
        if confidence:
            conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            message += f"<b>Confidence:</b> {confidence:.0%} [{conf_bar}]\n"
        
        # Strategy-specific fields
        if 'rsi2' in signal:
            message += f"<b>RSI(2):</b> {signal['rsi2']:.1f}\n"
        if 'cumulative_rsi' in signal:
            message += f"<b>Cum RSI:</b> {signal['cumulative_rsi']:.1f}\n"
        if 'daily_rsi' in signal:
            message += f"<b>Daily RSI:</b> {signal['daily_rsi']:.1f}\n"
        if 'bb_width_pctl' in signal:
            message += f"<b>BB Width %ile:</b> {signal['bb_width_pctl']:.0%}\n"
        
        if signal.get('stop_loss'):
            message += f"<b>Stop Loss:</b> ${signal['stop_loss']:.2f}\n"
        
        if reasoning:
            message += f"\n<i>{reasoning[:200]}</i>\n"
        
        message += f"\n<code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</code>"
        
        return await self._send(message, priority=AlertPriority.HIGH)
    
    async def send_arb_opportunity(self, arb: Dict[str, Any]) -> bool:
        """Send alert for arbitrage opportunity"""
        symbol = arb.get('symbol', '???')
        buy_exchange = arb.get('buy_exchange', '?')
        sell_exchange = arb.get('sell_exchange', '?')
        spread = arb.get('spread_pct', 0)
        profit = arb.get('profit_pct', 0)
        confidence = arb.get('confidence', 'medium')
        
        message = f"⚡ <b>[ARBITRAGE]</b> {symbol}\n\n"
        message += f"<b>Buy:</b> {buy_exchange}\n"
        message += f"<b>Sell:</b> {sell_exchange}\n"
        message += f"<b>Spread:</b> {spread:.2%}\n"
        message += f"<b>Est. Profit:</b> {profit:.2%}\n"
        message += f"<b>Confidence:</b> {confidence.upper()}\n"
        
        # Priority based on spread
        priority = AlertPriority.CRITICAL if spread >= 0.02 else AlertPriority.HIGH
        
        return await self._send(message, priority=priority)
    
    async def send_earnings_alert(self, earnings: Dict[str, Any]) -> bool:
        """Send alert for earnings surprise"""
        symbol = earnings.get('symbol', '???')
        surprise = earnings.get('surprise_pct', 0)
        
        direction = "BEAT" if surprise > 0 else "MISS"
        emoji = "🚀" if surprise > 0 else "📉"
        
        message = f"📰 <b>[EARNINGS]</b> {emoji} {symbol} {direction}\n\n"
        message += f"<b>Surprise:</b> {surprise:+.1%}\n"
        
        if earnings.get('expected_eps'):
            message += f"<b>Expected EPS:</b> ${earnings['expected_eps']:.2f}\n"
        if earnings.get('actual_eps'):
            message += f"<b>Actual EPS:</b> ${earnings['actual_eps']:.2f}\n"
        
        message += f"\n<i>PEAD Strategy: {'BUY drift expected' if surprise > 0.05 else 'Watch for reversal'}</i>"
        
        return await self._send(message, priority=AlertPriority.HIGH)
    
    async def send_sentiment_alert(self, sentiment: Dict[str, Any]) -> bool:
        """Send alert for extreme sentiment"""
        symbol = sentiment.get('symbol', '???')
        z_score = sentiment.get('z_score', 0)
        direction = sentiment.get('signal', 'neutral')
        
        emoji = "🔥" if abs(z_score) > 2.5 else "💬"
        
        message = f"💬 <b>[SENTIMENT]</b> {emoji} {symbol}\n\n"
        message += f"<b>Direction:</b> {direction.upper()}\n"
        message += f"<b>Z-Score:</b> {z_score:+.2f}\n"
        
        if sentiment.get('reddit_mentions'):
            message += f"<b>Reddit Mentions:</b> {sentiment['reddit_mentions']}\n"
        if sentiment.get('fear_greed'):
            message += f"<b>Fear & Greed:</b> {sentiment['fear_greed']}\n"
        
        if sentiment.get('reasoning'):
            message += f"\n<i>{sentiment['reasoning'][:150]}</i>"
        
        return await self._send(message, priority=AlertPriority.MEDIUM)
    
    async def send_daily_summary(self, summary: Dict[str, Any]) -> bool:
        """Send end-of-day trading summary"""
        date = summary.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        message = f"📊 <b>DAILY SUMMARY - {date}</b>\n\n"
        
        # Signals generated
        if 'signals' in summary:
            message += "<b>Signals Generated:</b>\n"
            for strategy, count in summary['signals'].items():
                emoji = self.STRATEGY_EMOJI.get(strategy, '📌')
                message += f"  {emoji} {strategy}: {count}\n"
        
        # Trades executed
        if 'trades' in summary:
            message += f"\n<b>Trades Executed:</b> {summary['trades']}\n"
        
        # P&L
        if 'pnl' in summary:
            pnl = summary['pnl']
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            message += f"<b>P&L:</b> {pnl_emoji} ${pnl:+,.2f}\n"
        
        # Win rate
        if 'win_rate' in summary:
            message += f"<b>Win Rate:</b> {summary['win_rate']:.0%}\n"
        
        # Arb opportunities
        if 'arb_opportunities' in summary:
            message += f"<b>Arb Opps Found:</b> {summary['arb_opportunities']}\n"
        
        # Portfolio value
        if 'portfolio_value' in summary:
            message += f"\n<b>Portfolio Value:</b> ${summary['portfolio_value']:,.2f}\n"
        
        return await self._send(message, priority=AlertPriority.MEDIUM)
    
    async def send_error_alert(self, error: str, strategy: str = "system") -> bool:
        """Send error notification"""
        message = f"🚨 <b>[ERROR]</b> {strategy.upper()}\n\n"
        message += f"<code>{error[:500]}</code>\n"
        message += f"\n<i>{datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>"
        
        return await self._send(message, priority=AlertPriority.CRITICAL)
    
    async def send_paper_trade_update(self, trade: Dict[str, Any]) -> bool:
        """Send paper trading update"""
        symbol = trade.get('symbol', '???')
        action = trade.get('action', 'unknown')
        strategy = trade.get('strategy', 'unknown')
        
        emoji = "📝"
        if 'pnl' in trade:
            emoji = "✅" if trade['pnl'] >= 0 else "❌"
        
        message = f"{emoji} <b>[PAPER]</b> {strategy.upper()}\n\n"
        message += f"<b>Action:</b> {action.upper()} {symbol}\n"
        
        if trade.get('entry_price'):
            message += f"<b>Entry:</b> ${trade['entry_price']:.2f}\n"
        if trade.get('exit_price'):
            message += f"<b>Exit:</b> ${trade['exit_price']:.2f}\n"
        if 'pnl' in trade:
            message += f"<b>P&L:</b> ${trade['pnl']:+.2f} ({trade.get('pnl_pct', 0):+.1%})\n"
        
        return await self._send(message, priority=AlertPriority.LOW)
    
    async def _send(self, message: str, priority: AlertPriority = AlertPriority.MEDIUM) -> bool:
        """Send message via Telegram bot"""
        try:
            if not self.bot.is_configured:
                logger.info(f"[ALERT] {message[:100]}...")
                return True
            
            return await self.bot._send_message(message, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False


# Helper function for safe event loop management
def _get_or_create_event_loop():
    """Get existing event loop or create new one if closed/missing."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


# Convenience functions for synchronous usage
def send_strategy_alert(signal: Dict[str, Any]) -> bool:
    """Synchronous wrapper for strategy signal alert"""
    alerts = ExtendedTelegramAlerts()
    loop = _get_or_create_event_loop()
    return loop.run_until_complete(alerts.send_strategy_signal(signal))


def send_arb_alert(arb: Dict[str, Any]) -> bool:
    """Synchronous wrapper for arb alert"""
    alerts = ExtendedTelegramAlerts()
    loop = _get_or_create_event_loop()
    return loop.run_until_complete(alerts.send_arb_opportunity(arb))


def send_earnings_alert(earnings: Dict[str, Any]) -> bool:
    """Synchronous wrapper for earnings alert"""
    alerts = ExtendedTelegramAlerts()
    loop = _get_or_create_event_loop()
    return loop.run_until_complete(alerts.send_earnings_alert(earnings))


def send_daily_summary(summary: Dict[str, Any]) -> bool:
    """Synchronous wrapper for daily summary"""
    alerts = ExtendedTelegramAlerts()
    loop = _get_or_create_event_loop()
    return loop.run_until_complete(alerts.send_daily_summary(summary))


if __name__ == "__main__":
    import asyncio
    
    print("Testing Extended Telegram Alerts")
    print("=" * 50)
    
    # Test signal
    test_signal = {
        'strategy': 'rsi2',
        'symbol': 'SPY',
        'signal': 'buy',
        'price': 450.25,
        'confidence': 0.85,
        'rsi2': 8.5,
        'reasoning': 'RSI(2) oversold with price above 200 SMA',
        'stop_loss': 445.00
    }
    
    alerts = ExtendedTelegramAlerts()
    
    async def test():
        print("\nSending test strategy signal...")
        await alerts.send_strategy_signal(test_signal)
        
        print("\nSending test arb alert...")
        await alerts.send_arb_opportunity({
            'symbol': 'BTC/USDT',
            'buy_exchange': 'Binance',
            'sell_exchange': 'Coinbase',
            'spread_pct': 0.015,
            'profit_pct': 0.012,
            'confidence': 'high'
        })
        
        print("\nSending test daily summary...")
        await alerts.send_daily_summary({
            'date': '2026-01-28',
            'signals': {'rsi2': 3, 'macd_rsi': 2, 'bollinger_squeeze': 1},
            'trades': 5,
            'pnl': 125.50,
            'win_rate': 0.80,
            'portfolio_value': 10125.50
        })
    
    asyncio.run(test())
    print("\nDone!")


# Alias for backward compatibility
send_alert = send_strategy_alert

# Alias
send_alert = send_strategy_alert
