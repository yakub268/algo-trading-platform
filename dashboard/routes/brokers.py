"""
Broker-specific API endpoints - /api/alpaca/*, /api/coinbase/*
"""

from datetime import datetime, timezone
from flask import Blueprint, jsonify, request

from dashboard.shared import (
    alpaca_client, COINBASE_AVAILABLE, get_coinbase_service,
)

brokers_bp = Blueprint('brokers', __name__)


# ============================================================================
# ALPACA ENDPOINTS
# ============================================================================

@brokers_bp.route('/api/alpaca/account')
def get_alpaca_account():
    """Get Alpaca account information"""
    try:
        if not alpaca_client.is_connected():
            return jsonify({
                'success': False,
                'connected': False,
                'error': 'Alpaca not connected'
            })

        account = alpaca_client.get_account()

        if account:
            return jsonify({
                'success': True,
                'connected': True,
                'account': {
                    'equity': float(account.get('equity', 0)),
                    'cash': float(account.get('cash', 0)),
                    'buying_power': float(account.get('buying_power', 0)),
                    'portfolio_value': float(account.get('portfolio_value', 0)),
                    'day_trade_count': account.get('daytrade_count', 0),
                    'account_blocked': account.get('account_blocked', False),
                    'trading_blocked': account.get('trading_blocked', False)
                },
                'paper_mode': alpaca_client.paper,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No account data'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@brokers_bp.route('/api/alpaca/positions')
def get_alpaca_positions():
    """Get Alpaca positions"""
    try:
        if not alpaca_client.is_connected():
            return jsonify({
                'success': False,
                'connected': False,
                'error': 'Alpaca not connected',
                'positions': []
            })

        positions = alpaca_client.get_positions()
        positions_data = []

        for pos in positions:
            positions_data.append({
                'symbol': pos.get('symbol'),
                'qty': float(pos.get('qty', 0)),
                'side': pos.get('side'),
                'market_value': float(pos.get('market_value', 0)),
                'cost_basis': float(pos.get('cost_basis', 0)),
                'avg_entry_price': float(pos.get('avg_entry_price', 0)),
                'current_price': float(pos.get('current_price', 0)),
                'unrealized_pl': float(pos.get('unrealized_pl', 0)),
                'unrealized_plpc': round(float(pos.get('unrealized_plpc', 0)) * 100, 2),
                'change_today': round(float(pos.get('change_today', 0)) * 100, 2)
            })

        return jsonify({
            'success': True,
            'connected': True,
            'positions': positions_data,
            'count': len(positions_data),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'positions': []
        })


@brokers_bp.route('/api/alpaca/orders')
def get_alpaca_orders():
    """Get Alpaca orders"""
    try:
        if not alpaca_client.is_connected():
            return jsonify({
                'success': False,
                'connected': False,
                'error': 'Alpaca not connected',
                'orders': []
            })

        status = request.args.get('status', 'all')
        limit = request.args.get('limit', 20, type=int)
        orders = alpaca_client.get_orders(status=status, limit=limit)

        orders_data = []
        for order in orders:
            orders_data.append({
                'id': order.get('id'),
                'symbol': order.get('symbol'),
                'side': order.get('side'),
                'type': order.get('type'),
                'qty': order.get('qty'),
                'filled_qty': order.get('filled_qty'),
                'filled_avg_price': order.get('filled_avg_price'),
                'status': order.get('status'),
                'submitted_at': order.get('submitted_at'),
                'filled_at': order.get('filled_at')
            })

        return jsonify({
            'success': True,
            'connected': True,
            'orders': orders_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'orders': []
        })


@brokers_bp.route('/api/alpaca/fomc')
def get_alpaca_fomc():
    """Get FOMC strategy status"""
    try:
        fomc = alpaca_client.get_fomc_status()
        spy_position = None

        if alpaca_client.is_connected():
            spy = alpaca_client.get_spy_position()
            if spy:
                spy_position = {
                    'qty': float(spy.get('qty', 0)),
                    'market_value': float(spy.get('market_value', 0)),
                    'unrealized_pl': float(spy.get('unrealized_pl', 0)),
                    'unrealized_plpc': round(float(spy.get('unrealized_plpc', 0)) * 100, 2)
                }

        clock = alpaca_client.get_clock() if alpaca_client.is_connected() else None
        market_open = clock.get('is_open', False) if clock else False

        return jsonify({
            'success': True,
            'connected': alpaca_client.is_connected(),
            'fomc': fomc,
            'spy_position': spy_position,
            'market_open': market_open,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@brokers_bp.route('/api/alpaca/summary')
def get_alpaca_summary():
    """Get complete Alpaca summary"""
    try:
        summary = alpaca_client.get_summary()
        return jsonify({
            'success': True,
            **summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'connected': False
        })


# ============================================================================
# COINBASE ENDPOINTS
# ============================================================================

@brokers_bp.route('/api/coinbase/positions')
def get_coinbase_positions():
    """Get Coinbase crypto positions"""
    try:
        if not COINBASE_AVAILABLE:
            return jsonify({
                'success': False,
                'connected': False,
                'error': 'Coinbase service not available',
                'positions': []
            })

        coinbase = get_coinbase_service()
        if not coinbase._initialized:
            return jsonify({
                'success': False,
                'connected': False,
                'error': 'Coinbase not connected',
                'positions': []
            })

        positions = coinbase.get_positions()
        positions_data = [p.to_dict() for p in positions]

        return jsonify({
            'success': True,
            'connected': True,
            'positions': positions_data,
            'count': len(positions_data),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'positions': []
        })


@brokers_bp.route('/api/coinbase/balances')
def get_coinbase_balances():
    """Get all Coinbase balances (detailed view)"""
    try:
        if not COINBASE_AVAILABLE:
            return jsonify({
                'success': False,
                'connected': False,
                'error': 'Coinbase service not available',
                'balances': []
            })

        coinbase = get_coinbase_service()
        if not coinbase._initialized:
            return jsonify({
                'success': False,
                'connected': False,
                'error': 'Coinbase not connected',
                'balances': []
            })

        balances = coinbase.get_balances()

        return jsonify({
            'success': True,
            'connected': True,
            'balances': balances,
            'count': len(balances),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'balances': []
        })


@brokers_bp.route('/api/coinbase/account')
def get_coinbase_account():
    """Get Coinbase account summary"""
    try:
        if not COINBASE_AVAILABLE:
            return jsonify({
                'success': False,
                'connected': False,
                'error': 'Coinbase service not available'
            })

        coinbase = get_coinbase_service()
        info = coinbase.get_account_info()

        return jsonify({
            'success': True,
            **info,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'connected': False
        })


@brokers_bp.route('/api/coinbase/health')
def get_coinbase_health():
    """Get Coinbase connection health"""
    try:
        if not COINBASE_AVAILABLE:
            return jsonify({
                'success': False,
                'status': 'DISCONNECTED',
                'latency_ms': 0
            })

        coinbase = get_coinbase_service()
        health = coinbase.get_health()

        return jsonify({
            'success': True,
            'status': health.status.value,
            'latency_ms': health.latency_ms,
            'buying_power': health.buying_power,
            'last_ping': health.last_ping.isoformat() if health.last_ping else None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'status': 'DISCONNECTED'
        })
