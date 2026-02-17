"""
Fleet Dashboard API Routes
==========================
Endpoints for monitoring and controlling the fleet trading system.

Blueprint: fleet_bp, prefix /api/fleet
"""

import logging
from flask import Blueprint, jsonify, request

logger = logging.getLogger('Dashboard.Fleet')

fleet_bp = Blueprint('fleet', __name__, url_prefix='/api/fleet')

# Fleet adapter reference — set by app.py at registration time
_fleet_adapter = None


def set_fleet_adapter(adapter):
    """Called by dashboard/app.py to inject the fleet adapter reference."""
    global _fleet_adapter
    _fleet_adapter = adapter


def _get_adapter():
    """Get the fleet adapter, trying orchestrator if direct ref not set."""
    if _fleet_adapter is not None:
        return _fleet_adapter

    # Try to get from master orchestrator's bot instances
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from bots.fleet.fleet_adapter import FleetAdapter
        adapter = FleetAdapter()
        return adapter
    except Exception as e:
        logger.error(f"Cannot get fleet adapter: {e}")
        return None


@fleet_bp.route('/status', methods=['GET'])
def fleet_status():
    """GET /api/fleet/status — All 19 sub-bot statuses."""
    adapter = _get_adapter()
    if not adapter:
        return jsonify({'error': 'Fleet not initialized'}), 503

    try:
        status = adapter.get_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@fleet_bp.route('/trades', methods=['GET'])
def fleet_trades():
    """GET /api/fleet/trades — Recent trades. Supports ?bot=&limit=&offset="""
    adapter = _get_adapter()
    if not adapter:
        return jsonify({'error': 'Fleet not initialized'}), 503

    try:
        bot = request.args.get('bot', None)
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))

        trades = adapter.orchestrator.db.get_recent_trades(bot_name=bot, limit=limit, offset=offset)
        return jsonify({'trades': trades, 'count': len(trades)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@fleet_bp.route('/performance', methods=['GET'])
def fleet_performance():
    """GET /api/fleet/performance — Per-bot P&L, win rates, Sharpe ratios."""
    adapter = _get_adapter()
    if not adapter:
        return jsonify({'error': 'Fleet not initialized'}), 503

    try:
        bot = request.args.get('bot', None)
        perf = adapter.orchestrator.db.get_bot_performance(bot_name=bot)
        return jsonify({'performance': perf})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@fleet_bp.route('/allocation', methods=['GET'])
def fleet_allocation():
    """GET /api/fleet/allocation — Thompson Sampling current capital splits."""
    adapter = _get_adapter()
    if not adapter:
        return jsonify({'error': 'Fleet not initialized'}), 503

    try:
        thompson_states = adapter.orchestrator.db.get_all_thompson_states()
        return jsonify({'allocations': thompson_states})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@fleet_bp.route('/risk', methods=['GET'])
def fleet_risk():
    """GET /api/fleet/risk — Current risk state."""
    adapter = _get_adapter()
    if not adapter:
        return jsonify({'error': 'Fleet not initialized'}), 503

    try:
        risk = adapter.orchestrator.risk.get_risk_summary()
        return jsonify(risk)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@fleet_bp.route('/pause/<bot_name>', methods=['POST'])
def fleet_pause_bot(bot_name: str):
    """POST /api/fleet/pause/<bot> — Pause specific sub-bot."""
    adapter = _get_adapter()
    if not adapter:
        return jsonify({'error': 'Fleet not initialized'}), 503

    success = adapter.pause_bot(bot_name)
    if success:
        return jsonify({'status': 'paused', 'bot': bot_name})
    return jsonify({'error': f'Bot {bot_name} not found'}), 404


@fleet_bp.route('/resume/<bot_name>', methods=['POST'])
def fleet_resume_bot(bot_name: str):
    """POST /api/fleet/resume/<bot> — Resume specific sub-bot."""
    adapter = _get_adapter()
    if not adapter:
        return jsonify({'error': 'Fleet not initialized'}), 503

    success = adapter.resume_bot(bot_name)
    if success:
        return jsonify({'status': 'resumed', 'bot': bot_name})
    return jsonify({'error': f'Bot {bot_name} not found'}), 404


@fleet_bp.route('/emergency_stop', methods=['POST'])
def fleet_emergency_stop():
    """POST /api/fleet/emergency_stop — Stop all fleet trading."""
    adapter = _get_adapter()
    if not adapter:
        return jsonify({'error': 'Fleet not initialized'}), 503

    adapter.emergency_stop()
    return jsonify({'status': 'emergency_stop', 'message': 'All fleet bots paused'})
