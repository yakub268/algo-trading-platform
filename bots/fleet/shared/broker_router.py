"""
Broker Router — Dispatches fleet trades to the correct broker.

Supports:
- Kalshi: prediction market orders (price in cents)
- Alpaca: crypto orders (USD)
- OANDA: forex orders (units)
"""

import os
import sys
import logging
from typing import Dict, Optional, Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))

from bots.fleet.shared.fleet_bot import FleetSignal, BotType

logger = logging.getLogger('Fleet.BrokerRouter')

# Lazy-load broker clients
_kalshi_client = None
_alpaca_client = None
_oanda_config = None


def _get_kalshi():
    global _kalshi_client
    if _kalshi_client is None:
        try:
            from bots.kalshi_client import KalshiClient
            _kalshi_client = KalshiClient()
            logger.info("Kalshi client initialized")
        except Exception as e:
            logger.error(f"Kalshi client init failed: {e}")
    return _kalshi_client


def _get_alpaca():
    global _alpaca_client
    if _alpaca_client is None:
        try:
            from bots.alpaca_crypto_client import AlpacaCryptoClient
            _alpaca_client = AlpacaCryptoClient()
            logger.info("Alpaca client initialized")
        except Exception as e:
            logger.error(f"Alpaca client init failed: {e}")
    return _alpaca_client


def _get_oanda_config():
    global _oanda_config
    if _oanda_config is None:
        try:
            from config.oanda_config import OANDA_CONFIG
            _oanda_config = OANDA_CONFIG
            logger.info("OANDA config loaded")
        except Exception as e:
            logger.error(f"OANDA config load failed: {e}")
    return _oanda_config


class BrokerRouter:
    """Routes fleet signals to the correct broker for execution."""

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode

    def execute(self, signal: FleetSignal) -> Dict[str, Any]:
        """
        Execute a trade signal via the appropriate broker.
        Returns {'success': bool, 'fill_price': float, 'fill_quantity': float, 'order_id': str, 'error': str}
        """
        bot_type = BotType(signal.bot_type) if isinstance(signal.bot_type, str) else signal.bot_type

        if self.paper_mode:
            return self._paper_execute(signal)

        if bot_type in (BotType.KALSHI, BotType.PREDICTION):
            return self._execute_kalshi(signal)
        elif bot_type == BotType.CRYPTO:
            return self._execute_alpaca(signal)
        elif bot_type == BotType.FOREX:
            return self._execute_oanda(signal)
        elif bot_type == BotType.META:
            # Meta bots don't trade directly — they adjust allocations
            return {'success': True, 'fill_price': 0, 'fill_quantity': 0, 'order_id': 'meta', 'error': ''}
        else:
            return {'success': False, 'fill_price': 0, 'fill_quantity': 0, 'order_id': '', 'error': f'Unknown bot_type: {bot_type}'}

    def _paper_execute(self, signal: FleetSignal) -> Dict[str, Any]:
        """Paper trade — always succeeds at the signal price."""
        logger.info(
            f"[PAPER] {signal.bot_name}: {signal.side} {signal.symbol} "
            f"qty={signal.quantity} @ ${signal.entry_price:.4f} "
            f"(${signal.position_size_usd:.2f})"
        )
        return {
            'success': True,
            'fill_price': signal.entry_price,
            'fill_quantity': signal.quantity,
            'order_id': f'paper-{signal.trade_id}',
            'error': '',
        }

    def _execute_kalshi(self, signal: FleetSignal) -> Dict[str, Any]:
        """Execute on Kalshi — price in cents, limit orders preferred."""
        client = _get_kalshi()
        if not client:
            return {'success': False, 'fill_price': 0, 'fill_quantity': 0, 'order_id': '', 'error': 'Kalshi client unavailable'}

        try:
            price_cents = int(signal.entry_price * 100) if signal.entry_price < 1 else int(signal.entry_price)
            quantity = max(1, int(signal.quantity))

            result = client.place_order(
                ticker=signal.symbol,
                side=signal.side.lower(),
                quantity=quantity,
                price=price_cents,
                order_type='limit',
            )

            order_id = result.get('order', {}).get('order_id', '')
            return {
                'success': True,
                'fill_price': price_cents / 100.0,
                'fill_quantity': quantity,
                'order_id': order_id,
                'error': '',
            }
        except Exception as e:
            logger.error(f"Kalshi execution failed: {e}")
            return {'success': False, 'fill_price': 0, 'fill_quantity': 0, 'order_id': '', 'error': str(e)}

    def _execute_alpaca(self, signal: FleetSignal) -> Dict[str, Any]:
        """Execute on Alpaca — crypto market orders."""
        client = _get_alpaca()
        if not client:
            return {'success': False, 'fill_price': 0, 'fill_quantity': 0, 'order_id': '', 'error': 'Alpaca client unavailable'}

        try:
            side_str = signal.side.upper()
            alpaca_symbol = client.to_alpaca_symbol(signal.symbol)
            notional = signal.position_size_usd

            result = client.create_market_order(
                symbol=alpaca_symbol,
                side=side_str,
                notional=notional,
            )

            if result and result.get('id'):
                return {
                    'success': True,
                    'fill_price': signal.entry_price,
                    'fill_quantity': notional / signal.entry_price if signal.entry_price > 0 else 0,
                    'order_id': result.get('id', ''),
                    'error': '',
                }
            return {'success': False, 'fill_price': 0, 'fill_quantity': 0, 'order_id': '', 'error': 'No order ID returned'}
        except Exception as e:
            logger.error(f"Alpaca execution failed: {e}")
            return {'success': False, 'fill_price': 0, 'fill_quantity': 0, 'order_id': '', 'error': str(e)}

    def _execute_oanda(self, signal: FleetSignal) -> Dict[str, Any]:
        """Execute on OANDA — forex market orders."""
        config = _get_oanda_config()
        if not config:
            return {'success': False, 'fill_price': 0, 'fill_quantity': 0, 'order_id': '', 'error': 'OANDA config unavailable'}

        try:
            import requests

            account_id = config.get('account_id', os.getenv('OANDA_ACCOUNT_ID', ''))
            api_key = config.get('api_key', os.getenv('OANDA_API_KEY', ''))
            api_url = config.get('api_url', 'https://api-fxpractice.oanda.com')

            units = int(signal.quantity)
            if signal.side.upper() == 'SELL':
                units = -units

            url = f"{api_url}/v3/accounts/{account_id}/orders"
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            }
            data = {
                'order': {
                    'type': 'MARKET',
                    'instrument': signal.symbol,
                    'units': str(units),
                    'timeInForce': 'FOK',
                }
            }

            resp = requests.post(url, json=data, headers=headers, timeout=10)
            if resp.status_code in (200, 201):
                result = resp.json()
                fill = result.get('orderFillTransaction', {})
                return {
                    'success': True,
                    'fill_price': float(fill.get('price', signal.entry_price)),
                    'fill_quantity': abs(units),
                    'order_id': fill.get('id', ''),
                    'error': '',
                }
            return {'success': False, 'fill_price': 0, 'fill_quantity': 0, 'order_id': '', 'error': resp.text[:200]}
        except Exception as e:
            logger.error(f"OANDA execution failed: {e}")
            return {'success': False, 'fill_price': 0, 'fill_quantity': 0, 'order_id': '', 'error': str(e)}
