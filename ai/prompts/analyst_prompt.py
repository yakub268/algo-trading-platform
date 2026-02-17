"""
AI Market Analyst Prompts
=========================
System prompts and builders for market analysis, prediction, and trade reasoning.

Author: Trading Bot Arsenal
Created: February 2026
"""


ANALYST_SYSTEM_PROMPT = """You are a senior quantitative portfolio manager analyzing real-time market data for a crypto + prediction market trading system.

Your job: synthesize technical signals, news sentiment, and market conditions into ACTIONABLE predictions.

CRITICAL RULES:
1. Base analysis ONLY on data provided. Never fabricate data points.
2. When data is insufficient, REDUCE confidence accordingly.
3. You are assessing PROBABILITY of directional moves, not predicting exact prices.
4. Be contrarian when sentiment is extreme - crowds are often wrong at extremes.
5. Weight recent data more heavily than older data.
6. Consider cross-asset correlations (BTC leads alts, VIX inversely correlated with risk assets).

OUTPUT FORMAT (JSON only):
{
    "market_regime": "risk_on" | "risk_off" | "choppy" | "trending_up" | "trending_down",
    "risk_level": "low" | "medium" | "high" | "extreme",
    "predictions": [
        {
            "symbol": "BTC/USD",
            "direction": "bullish" | "bearish" | "neutral",
            "confidence": 0-100,
            "timeframe": "4h",
            "key_factors": ["factor1", "factor2"],
            "price_target_pct": 1.5,
            "stop_loss_pct": -2.0
        }
    ],
    "portfolio_recommendations": {
        "overall_exposure": "increase" | "maintain" | "reduce" | "close_all",
        "bot_adjustments": [
            {"bot": "Momentum-Scalper", "action": "description of adjustment"}
        ]
    },
    "key_risks": ["risk1", "risk2"],
    "conviction_trade": {
        "symbol": "BTC/USD",
        "direction": "long" | "short" | "none",
        "reasoning": "Brief explanation"
    },
    "summary": "2-3 sentence market outlook"
}"""


PREDICTION_SYSTEM_PROMPT = """You are a quantitative analyst making a directional prediction for a specific asset.

Given the technical data, news sentiment, and market context, predict the most likely price direction over the next 1-4 hours.

OUTPUT FORMAT (JSON only):
{
    "direction": "bullish" | "bearish" | "neutral",
    "confidence": 0-100,
    "reasoning": "Brief explanation (max 50 words)",
    "key_signal": "The single most important signal driving this prediction",
    "risk_factors": ["factor1", "factor2"],
    "suggested_action": "buy" | "sell" | "hold" | "reduce_exposure"
}

CONFIDENCE CALIBRATION:
- 80-100: Strong multi-signal confluence (trend + momentum + sentiment aligned)
- 60-79: Moderate signals, some confluence
- 40-59: Mixed signals, low conviction
- 20-39: Weak or contradictory signals
- 0-19: No actionable signal, stay flat"""


TRADE_REASONING_PROMPT = """You are explaining a trade decision to the portfolio manager.

Given the trade signal and market context, provide a concise 1-2 sentence explanation of WHY this trade makes sense (or doesn't).

Be specific: mention the actual technical levels, sentiment scores, and market conditions.
Keep it under 40 words. No hedging language. Be direct."""


def build_analysis_prompt(crypto_data: dict, news_summary: str, portfolio_state: dict,
                          recent_predictions: list = None, data_hub_context: str = None) -> str:
    """Build comprehensive market analysis prompt."""

    # Format crypto data
    crypto_section = "CURRENT CRYPTO MARKET DATA:\n"
    for symbol, data in crypto_data.items():
        crypto_section += f"\n{symbol}:\n"
        crypto_section += f"  Price: ${data.get('price', 'N/A')}\n"
        crypto_section += f"  24h Change: {data.get('change_24h', 'N/A')}%\n"
        crypto_section += f"  RSI(14): {data.get('rsi', 'N/A')}\n"
        crypto_section += f"  EMA(12) vs EMA(26): {data.get('ema_signal', 'N/A')}\n"
        crypto_section += f"  Volume vs Avg: {data.get('volume_ratio', 'N/A')}x\n"
        if data.get('support'):
            crypto_section += f"  Support: ${data['support']}, Resistance: ${data.get('resistance', 'N/A')}\n"

    # Format portfolio
    portfolio_section = f"""
PORTFOLIO STATE:
  Total Capital: ${portfolio_state.get('total_capital', 'N/A')}
  Current P&L: ${portfolio_state.get('current_pnl', 'N/A')}
  Open Positions: {portfolio_state.get('open_positions', 0)}
  Today's Trades: {portfolio_state.get('trades_today', 0)}
  Win Rate (recent): {portfolio_state.get('win_rate', 'N/A')}%
  Active Bots: {portfolio_state.get('active_bots', 'N/A')}
"""

    # Format market context
    market_section = f"""
MARKET CONTEXT:
  VIX: {portfolio_state.get('vix', 'N/A')}
  SPY Trend: {portfolio_state.get('spy_trend', 'N/A')}
  Day/Time: {portfolio_state.get('day_of_week', 'N/A')} {portfolio_state.get('time', 'N/A')}
  BTC Dominance: {portfolio_state.get('btc_dominance', 'N/A')}%
"""

    # V7: Data Hub context (macro, cross-asset, derivatives, on-chain)
    hub_section = ""
    if data_hub_context:
        hub_section = f"\nEXTERNAL DATA (REAL-TIME):\n{data_hub_context}\n"

    # News summary
    news_section = f"\nNEWS & SENTIMENT:\n{news_summary}\n" if news_summary else "\nNEWS: No recent news available.\n"

    # Previous prediction accuracy
    accuracy_section = ""
    if recent_predictions:
        accuracy_section = "\nPREVIOUS PREDICTION ACCURACY (last 24h):\n"
        for pred in recent_predictions[-5:]:
            accuracy_section += f"  {pred['symbol']}: predicted {pred['direction']}, actual {pred.get('actual', 'pending')} "
            accuracy_section += f"({'correct' if pred.get('correct') else 'wrong' if pred.get('correct') is False else 'pending'})\n"

    return f"""{crypto_section}
{market_section}
{hub_section}
{portfolio_section}
{news_section}
{accuracy_section}
Analyze the market and provide predictions in JSON format:"""


def build_prediction_prompt(symbol: str, technical_data: dict, news_sentiment: float,
                           market_regime: str, recent_trades: list = None) -> str:
    """Build prediction prompt for a specific symbol."""

    prompt = f"""PREDICTION REQUEST: {symbol}

TECHNICAL DATA:
  Current Price: ${technical_data.get('price', 'N/A')}
  RSI(14): {technical_data.get('rsi', 'N/A')}
  RSI(2): {technical_data.get('rsi_2', 'N/A')}
  EMA(12): ${technical_data.get('ema_12', 'N/A')}
  EMA(26): ${technical_data.get('ema_26', 'N/A')}
  EMA Signal: {technical_data.get('ema_signal', 'N/A')}
  MACD: {technical_data.get('macd', 'N/A')}
  24h Change: {technical_data.get('change_24h', 'N/A')}%
  24h Volume: ${technical_data.get('volume_24h', 'N/A')}
  Volume Ratio (vs avg): {technical_data.get('volume_ratio', 'N/A')}x
  ATR(14): {technical_data.get('atr', 'N/A')}
  Bollinger Position: {technical_data.get('bb_position', 'N/A')}

CONTEXT:
  News Sentiment: {news_sentiment:.2f} (-1 bearish to +1 bullish)
  Market Regime: {market_regime}
"""

    if recent_trades:
        prompt += "\nRECENT TRADES ON THIS SYMBOL:\n"
        for trade in recent_trades[-3:]:
            prompt += f"  {trade.get('action', '?')} @ ${trade.get('price', '?')} - {trade.get('result', 'open')}\n"

    prompt += "\nPredict direction for the next 1-4 hours. Respond in JSON only:"
    return prompt


def build_trade_reasoning_prompt(signal: dict, market_analysis: dict = None) -> str:
    """Build prompt for explaining a trade decision."""

    prompt = f"""TRADE SIGNAL:
  Bot: {signal.get('bot_name', 'Unknown')}
  Action: {signal.get('action', 'Unknown').upper()}
  Symbol: {signal.get('symbol', 'Unknown')}
  Price: ${signal.get('price', 0):.2f}
  Confidence: {signal.get('strategy_confidence', signal.get('confidence', 'N/A'))}
  Reason: {signal.get('reason', signal.get('reasoning', 'N/A'))}
"""

    if market_analysis:
        prompt += f"""
MARKET CONTEXT:
  Regime: {market_analysis.get('market_regime', 'unknown')}
  Risk Level: {market_analysis.get('risk_level', 'unknown')}
  AI Prediction for {signal.get('symbol', '?')}: {market_analysis.get('prediction_direction', 'N/A')} ({market_analysis.get('prediction_confidence', 'N/A')}% conf)
"""

    prompt += "\nExplain this trade in 1-2 sentences (under 40 words):"
    return prompt
