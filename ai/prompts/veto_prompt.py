"""
Optimized prompts for AI veto decisions.
Based on research: structured output, low temperature, explicit JSON format

Author: Trading Bot Arsenal
Created: January 2026
"""

VETO_SYSTEM_PROMPT = """You are a senior risk manager at a quantitative trading firm. Your job is to review trade signals and decide whether to APPROVE, VETO, or REDUCE_SIZE.

CORE PRINCIPLES:
1. Protect capital - avoid trades with high downside risk
2. Filtering bad trades > finding perfect entries
3. When uncertain, reduce size rather than veto completely
4. Consider market regime, news sentiment, and timing

OUTPUT FORMAT (JSON only, no other text):
{
    "decision": "approve" | "veto" | "reduce_size",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation (max 100 words)",
    "risk_factors": ["factor1", "factor2"]
}

DECISION GUIDELINES:
- APPROVE (confidence > 0.35): Signal is reasonable, no major red flags. Most signals should pass.
- REDUCE_SIZE (confidence 0.15-0.35): Significant concerns but trade still has merit at smaller size
- VETO (confidence < 0.15): Critical red flags only - obvious bad trades, broken signals, or dangerous conditions

BIAS: You should APPROVE most trades. The bot strategies have already passed backtesting and validation.
Your job is to catch obviously bad situations, NOT to second-guess every signal. When in doubt, REDUCE_SIZE rather than VETO.

VETO ONLY WHEN (strict criteria - all other trades should pass):
- Trading against a STRONG trend in EXTREMELY volatile market (VIX > 35)
- Signal is clearly broken or contradictory (e.g., buy signal with bearish reasoning)
- Market is in crisis mode (flash crash, circuit breakers)

DO NOT VETO FOR:
- Friday afternoon trades (prediction markets and crypto trade 24/7)
- Moderate volatility (VIX 20-30 is normal, not dangerous)
- Minor news events
- Slightly below-average strategy confidence
- Crypto or prediction market trades during off-hours (these markets are always open)

AUTOMATIC APPROVE TRIGGERS:
- Signal aligns with trend direction
- Low to moderate VIX environment (<25)
- Strategy confidence > 0.5
- Prediction market trades with positive expected value
- Crypto trades with clear technical signal"""


def build_veto_prompt(signal: dict, context: dict) -> str:
    """
    Build the evaluation prompt for AI veto decision.

    Args:
        signal: Trade signal from bot
        context: Market context information

    Returns:
        Formatted prompt string
    """

    # Format signal
    signal_str = f"""
TRADE SIGNAL:
- Bot: {signal.get('bot_name', 'Unknown')}
- Action: {signal.get('action', 'Unknown').upper()}
- Symbol: {signal.get('symbol', 'Unknown')}
- Price: ${signal.get('price', 0):.2f}
- Quantity: {signal.get('quantity', signal.get('shares', 'N/A'))}
- Strategy Confidence: {signal.get('strategy_confidence', signal.get('confidence', 0.5)):.0%}
- Reason: {signal.get('reason', signal.get('reasoning', 'No reason provided'))}
"""

    # Format context
    context_str = f"""
MARKET CONTEXT:
- Market Regime: {context.get('market_regime', 'unknown')}
- Risk Level: {context.get('risk_level', 'N/A')}
- VIX: {context.get('vix', 'N/A')}
- SPY Trend: {context.get('spy_trend', 'unknown')}
- Time: {context.get('time_of_day', 'unknown')} on {context.get('day_of_week', 'unknown')}
- Portfolio P&L Today: ${context.get('portfolio_pnl', 0):.2f}
- Trades Today: {context.get('trades_today', 0)}
- Open Positions: {context.get('open_positions', 0)}
"""

    # V6: Add AI analyst summary and prediction
    if context.get('ai_summary'):
        context_str += f"\nAI ANALYST ASSESSMENT:\n{context.get('ai_summary', '')}\n"
        if context.get('key_risks'):
            context_str += f"Key Risks: {context.get('key_risks', '')}\n"

    # Add news if available
    news = context.get('recent_news', [])
    if news:
        context_str += "\nRecent News:\n"
        for i, item in enumerate(news[:3]):
            item_str = str(item)[:100]
            context_str += f"  {i+1}. {item_str}...\n"

    # Add bot-specific context for prediction markets
    bot_name = str(signal.get('bot_name', '')).lower()
    market_type = signal.get('market', '')

    if 'kalshi' in bot_name or 'prediction' in bot_name or market_type == 'prediction':
        context_str += f"""
PREDICTION MARKET CONTEXT:
- Market Price: {context.get('market_price', signal.get('price', 'N/A'))}c
- Our Estimate: {context.get('estimated_probability', 'N/A')}%
- Edge: {context.get('edge', 'N/A')}%
- Resolution Time: {context.get('resolution_time', context.get('expires_at', 'unknown'))}
"""

    # Add crypto-specific context
    if 'crypto' in bot_name or signal.get('symbol', '').endswith('USD'):
        context_str += f"""
CRYPTO CONTEXT:
- 24h Volume: {context.get('volume_24h', 'N/A')}
- Funding Rate: {context.get('funding_rate', 'N/A')}
- BTC Correlation: {context.get('btc_correlation', 'N/A')}
"""

    # Add forex-specific context
    if 'forex' in bot_name or 'oanda' in bot_name:
        context_str += f"""
FOREX CONTEXT:
- Session: {context.get('forex_session', 'N/A')}
- Spread: {context.get('spread', 'N/A')} pips
- Economic Calendar: {context.get('upcoming_events', 'None')}
"""

    return f"""{signal_str}
{context_str}

Evaluate this trade and respond with JSON only:"""


def build_batch_veto_prompt(signals: list, context: dict) -> str:
    """
    Build prompt for evaluating multiple signals at once (cost optimization).

    Args:
        signals: List of trade signals
        context: Shared market context

    Returns:
        Formatted prompt string
    """

    signals_str = "TRADE SIGNALS TO EVALUATE:\n"
    for i, signal in enumerate(signals):
        signals_str += f"""
Signal {i+1}:
- Bot: {signal.get('bot_name', 'Unknown')}
- Action: {signal.get('action', 'Unknown').upper()}
- Symbol: {signal.get('symbol', 'Unknown')}
- Price: ${signal.get('price', 0):.2f}
- Confidence: {signal.get('strategy_confidence', 0.5):.0%}
"""

    context_str = f"""
MARKET CONTEXT (applies to all signals):
- Market Regime: {context.get('market_regime', 'unknown')}
- VIX: {context.get('vix', 'N/A')}
- Time: {context.get('time_of_day', 'unknown')} on {context.get('day_of_week', 'unknown')}
"""

    return f"""{signals_str}
{context_str}

Evaluate each signal and respond with JSON array:
[
    {{"signal_id": 1, "decision": "approve|veto|reduce_size", "confidence": 0.0-1.0, "reasoning": "..."}},
    ...
]"""


# Pre-built prompts for common scenarios
QUICK_APPROVE_CONTEXT = {
    "market_regime": "bullish",
    "vix": 15,
    "spy_trend": "up",
    "day_of_week": "Tuesday",
    "time_of_day": "10:30"
}

QUICK_VETO_CONTEXT = {
    "market_regime": "volatile",
    "vix": 32,
    "spy_trend": "down",
    "day_of_week": "Friday",
    "time_of_day": "15:30"
}
