"""
Kalshi Dashboard API

Flask API server that provides real-time data for the trading dashboard.
Integrates with the Kalshi Orchestrator to expose:
- Strategy status
- Opportunities
- Positions
- Execution controls

Run with: python -m bots.kalshi_dashboard_api
Access at: http://localhost:5001

Author: Trading Bot Arsenal
Created: January 2026

SECURITY: This API requires authentication via API key in X-API-Key header.
Set DASHBOARD_API_KEY environment variable to enable authentication.
"""

import os
import sys
import json
import logging
import re
import secrets
from functools import wraps
from datetime import datetime, timezone
from flask import Flask, jsonify, request, abort
from flask_cors import CORS

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.kalshi_orchestrator import KalshiOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('KalshiDashboardAPI')

app = Flask(__name__)

# Security: Restrict CORS to localhost only (configure DASHBOARD_CORS_ORIGINS for other origins)
ALLOWED_ORIGINS = os.getenv('DASHBOARD_CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000,http://localhost:5000,http://127.0.0.1:5000').split(',')
CORS(app, origins=ALLOWED_ORIGINS)

# API Key authentication
API_KEY = os.getenv('DASHBOARD_API_KEY')
if not API_KEY:
    # Generate a random key if not set (for development) - log it once
    API_KEY = secrets.token_urlsafe(32)
    logger.warning(f"DASHBOARD_API_KEY not set. Generated temporary key: {API_KEY}")
    logger.warning("Set DASHBOARD_API_KEY environment variable for production!")


def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        provided_key = request.headers.get('X-API-Key')
        if not provided_key or not secrets.compare_digest(provided_key, API_KEY):
            logger.warning(f"Unauthorized API access attempt from {request.remote_addr}")
            abort(401)
        return f(*args, **kwargs)
    return decorated_function


def sanitize_error(e: Exception) -> str:
    """Sanitize error messages to avoid leaking internal details."""
    # Only return generic error type, not full message
    return f"Internal error: {type(e).__name__}"


# Valid strategy name pattern (alphanumeric, underscores, hyphens)
VALID_STRATEGY_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,50}$')

# Global orchestrator instance
orchestrator = None


def get_orchestrator():
    """Get or create orchestrator instance."""
    global orchestrator
    if orchestrator is None:
        orchestrator = KalshiOrchestrator(paper_mode=True)
    return orchestrator


@app.route('/api/status', methods=['GET'])
@require_api_key
def get_status():
    """Get overall system status."""
    try:
        orch = get_orchestrator()
        status = orch.get_status()
        return jsonify({
            'success': True,
            'data': status
        })
    except Exception as e:
        logger.error(f"Error in get_status: {e}")
        return jsonify({
            'success': False,
            'error': sanitize_error(e)
        }), 500


@app.route('/api/dashboard', methods=['GET'])
@require_api_key
def get_dashboard():
    """Get full dashboard data."""
    try:
        orch = get_orchestrator()
        data = orch.get_dashboard_data()
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        logger.error(f"Error in get_dashboard: {e}")
        return jsonify({
            'success': False,
            'error': sanitize_error(e)
        }), 500


@app.route('/api/scan', methods=['POST'])
@require_api_key
def run_scan():
    """Trigger a scan of all strategies."""
    try:
        orch = get_orchestrator()
        opportunities = orch.scan_all_strategies()

        # Convert to JSON-serializable format
        opps_data = []
        for opp in opportunities:
            opps_data.append({
                'strategy': opp.strategy,
                'ticker': opp.ticker,
                'title': opp.title,
                'profit_pct': opp.profit_pct,
                'profit_dollars': opp.profit_dollars,
                'confidence': opp.confidence,
                'timestamp': opp.timestamp.isoformat()
            })

        return jsonify({
            'success': True,
            'count': len(opps_data),
            'opportunities': opps_data
        })
    except Exception as e:
        logger.error(f"Error in run_scan: {e}")
        return jsonify({
            'success': False,
            'error': sanitize_error(e)
        }), 500


@app.route('/api/strategies', methods=['GET'])
@require_api_key
def get_strategies():
    """Get all strategy statuses."""
    try:
        orch = get_orchestrator()
        status = orch.get_status()
        return jsonify({
            'success': True,
            'strategies': status['strategies']
        })
    except Exception as e:
        logger.error(f"Error in get_strategies: {e}")
        return jsonify({
            'success': False,
            'error': sanitize_error(e)
        }), 500


@app.route('/api/strategies/<strategy_name>/enable', methods=['POST'])
@require_api_key
def enable_strategy(strategy_name):
    """Enable a strategy."""
    # Validate strategy name to prevent injection
    if not VALID_STRATEGY_PATTERN.match(strategy_name):
        return jsonify({
            'success': False,
            'error': 'Invalid strategy name format'
        }), 400

    try:
        orch = get_orchestrator()
        orch.enable_strategy(strategy_name)
        return jsonify({
            'success': True,
            'message': f'{strategy_name} enabled'
        })
    except Exception as e:
        logger.error(f"Error enabling strategy {strategy_name}: {e}")
        return jsonify({
            'success': False,
            'error': sanitize_error(e)
        }), 500


@app.route('/api/strategies/<strategy_name>/disable', methods=['POST'])
@require_api_key
def disable_strategy(strategy_name):
    """Disable a strategy."""
    # Validate strategy name to prevent injection
    if not VALID_STRATEGY_PATTERN.match(strategy_name):
        return jsonify({
            'success': False,
            'error': 'Invalid strategy name format'
        }), 400

    try:
        orch = get_orchestrator()
        orch.disable_strategy(strategy_name)
        return jsonify({
            'success': True,
            'message': f'{strategy_name} disabled'
        })
    except Exception as e:
        logger.error(f"Error disabling strategy {strategy_name}: {e}")
        return jsonify({
            'success': False,
            'error': sanitize_error(e)
        }), 500


@app.route('/api/execute', methods=['POST'])
@require_api_key
def execute_trade():
    """Execute a specific opportunity."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # This would need the full opportunity object
        # For now, just acknowledge
        return jsonify({
            'success': True,
            'message': 'Execution endpoint ready (implement with full opp data)'
        })
    except Exception as e:
        logger.error(f"Error in execute_trade: {e}")
        return jsonify({
            'success': False,
            'error': sanitize_error(e)
        }), 500


@app.route('/api/start', methods=['POST'])
@require_api_key
def start_orchestrator():
    """Start continuous scanning."""
    try:
        data = request.get_json() or {}
        interval = data.get('interval', 60)

        # Validate interval is reasonable (10s to 1 hour)
        if not isinstance(interval, (int, float)) or interval < 10 or interval > 3600:
            return jsonify({
                'success': False,
                'error': 'Invalid interval (must be 10-3600 seconds)'
            }), 400

        orch = get_orchestrator()
        orch.start(scan_interval=int(interval))

        return jsonify({
            'success': True,
            'message': f'Orchestrator started (interval={interval}s)'
        })
    except Exception as e:
        logger.error(f"Error in start_orchestrator: {e}")
        return jsonify({
            'success': False,
            'error': sanitize_error(e)
        }), 500


@app.route('/api/stop', methods=['POST'])
@require_api_key
def stop_orchestrator():
    """Stop continuous scanning."""
    try:
        orch = get_orchestrator()
        orch.stop()

        return jsonify({
            'success': True,
            'message': 'Orchestrator stopped'
        })
    except Exception as e:
        logger.error(f"Error in stop_orchestrator: {e}")
        return jsonify({
            'success': False,
            'error': sanitize_error(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'service': 'kalshi-dashboard-api'
    })


def main():
    """Run the API server."""
    print("=" * 60)
    print("KALSHI DASHBOARD API")
    print("=" * 60)
    print("""
SECURITY: All endpoints (except /api/health) require authentication.
Include header: X-API-Key: <your-api-key>

Endpoints:
  GET  /api/health      - Health check (no auth required)
  GET  /api/status      - System status
  GET  /api/dashboard   - Full dashboard data
  POST /api/scan        - Trigger scan
  GET  /api/strategies  - Strategy statuses
  POST /api/strategies/<name>/enable  - Enable strategy
  POST /api/strategies/<name>/disable - Disable strategy
  POST /api/start       - Start continuous scanning
  POST /api/stop        - Stop scanning

Environment Variables:
  DASHBOARD_API_KEY     - API key for authentication (required for production)
  DASHBOARD_CORS_ORIGINS - Comma-separated allowed origins (default: localhost)
""")

    # Initialize orchestrator on startup
    get_orchestrator()

    # Security: Bind to localhost only by default
    # Set DASHBOARD_HOST=0.0.0.0 to expose externally (NOT recommended)
    host = os.getenv('DASHBOARD_HOST', '127.0.0.1')
    port = int(os.getenv('DASHBOARD_PORT', '5001'))

    if host == '0.0.0.0':
        logger.warning("WARNING: Server bound to all interfaces (0.0.0.0). This is a security risk!")

    print(f"\nStarting server on http://{host}:{port}")
    print("Press Ctrl+C to stop\n")

    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    main()
