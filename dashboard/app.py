"""
Unified Trading Dashboard

Live web dashboard for monitoring all trading systems:
- Kalshi (Prediction Markets)
- Alpaca (Stock/Crypto Trading)
- OANDA (Forex)
- Coinbase (Crypto)

Features:
- Live opportunity table (sortable, color-coded)
- Category breakdown charts (pie + bar)
- Paper trade log with running P&L
- Alpaca account status and FOMC strategy
- Auto-refresh (Kalshi: 60s, Alpaca: 60s)
- Combined P&L tracking across all systems
- API key authentication for API routes

Tech: Flask + Bootstrap 5 + Chart.js

Author: Trading Bot
Created: January 2026
"""

import os
import sys
import time

# Fix Unicode encoding for Windows console
if sys.platform.startswith('win'):
    import codecs
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import logging
from flask import Flask, jsonify, request, redirect

# Suppress noisy oandapyV20 library errors (401s logged at ERROR level internally)
logging.getLogger('oandapyV20.oandapyV20').setLevel(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Shared state is initialized on import (clients, caches, helpers)
# ---------------------------------------------------------------------------
from dashboard.shared import (
    get_db_connection,
    RISK_MANAGEMENT_AVAILABLE, create_risk_routes, get_trading_integration,
)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

DASHBOARD_API_KEY = os.getenv('DASHBOARD_API_KEY', '')


@app.before_request
def check_api_key():
    """Require API key for all API routes"""
    if not request.path.startswith('/api/'):
        return  # Static files don't need auth
    if not DASHBOARD_API_KEY:
        return  # No key configured = auth disabled (development mode)
    provided_key = request.headers.get('X-API-Key') or request.args.get('api_key')
    if provided_key != DASHBOARD_API_KEY:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401


@app.before_request
def start_timer():
    """Start request timer"""
    request._start_time = time.time()


@app.after_request
def log_request(response):
    """Log slow API requests"""
    if request.path.startswith('/api/'):
        duration = getattr(request, '_start_time', None)
        if duration:
            elapsed = time.time() - duration
            if elapsed > 1.0:  # Log requests slower than 1 second
                print(f"SLOW REQUEST: {request.method} {request.path} took {elapsed:.2f}s")
    return response


# ---------------------------------------------------------------------------
# Table initialization
# ---------------------------------------------------------------------------

def init_paper_trades_table():
    """Initialize paper trades table if not exists"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            category TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            contracts INTEGER NOT NULL,
            edge REAL NOT NULL,
            status TEXT DEFAULT 'open',
            exit_price REAL,
            pnl REAL,
            resolved_at TEXT
        )
    ''')
    conn.commit()
    conn.close()


def init_scraper_status_table():
    """Initialize scraper status table if not exists"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scraper_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scraper_name TEXT UNIQUE NOT NULL,
            last_run TEXT,
            status TEXT DEFAULT 'unknown',
            records_fetched INTEGER DEFAULT 0,
            error_message TEXT
        )
    ''')
    conn.commit()
    conn.close()


# Initialize tables on startup
init_paper_trades_table()
init_scraper_status_table()

# ---------------------------------------------------------------------------
# Risk management routes (registered directly on the app, not a blueprint)
# ---------------------------------------------------------------------------
if RISK_MANAGEMENT_AVAILABLE:
    try:
        trading_integration = get_trading_integration()
        create_risk_routes(app, trading_integration)
        print("Risk management dashboard routes initialized")
    except Exception as e:
        print(f"Risk management routes failed to initialize: {e}")
        # Update shared module flag so blueprints see the real status
        import dashboard.shared as _shared
        _shared.RISK_MANAGEMENT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Register blueprints
# ---------------------------------------------------------------------------
from dashboard.routes import v5_bp, ai_bots_bp, brokers_bp, actions_bp, legacy_bp
from dashboard.routes.fleet import fleet_bp

app.register_blueprint(v5_bp)
app.register_blueprint(ai_bots_bp)
app.register_blueprint(brokers_bp)
app.register_blueprint(actions_bp)
app.register_blueprint(legacy_bp)
app.register_blueprint(fleet_bp)


# ---------------------------------------------------------------------------
# Page routes (kept in app.py)
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Main dashboard page - redirects to V5 (fast version)"""
    return redirect('/static/dashboard_v5_fast.html')


@app.route('/v4')
def index_v4():
    """V4 dashboard (legacy)"""
    return redirect('/static/dashboard_v4.html')


@app.route('/v5')
def index_v5():
    """V5 dashboard - optimized version (fast)"""
    return redirect('/static/dashboard_v5_fast.html')

@app.route('/v5/dev')
def index_v5_dev():
    """V5 dashboard - development version (with Babel)"""
    return redirect('/static/dashboard_v5.html')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 70)
    print("TRADING BOT DASHBOARD V5 + ADVANCED RISK MANAGEMENT")
    print("=" * 70)
    print("Starting server at http://localhost:5000")
    print("Main Dashboard:")
    print("  - V5 Dashboard: http://localhost:5000/")
    print("  - V4 Dashboard: http://localhost:5000/v4")
    print("")

    if RISK_MANAGEMENT_AVAILABLE:
        print("Risk Management API Endpoints:")
        print("  - Risk Metrics:     /api/risk/metrics")
        print("  - Position Heatmap: /api/risk/heatmap")
        print("  - Correlation:      /api/risk/correlation")
        print("  - VaR Analysis:     /api/risk/var")
        print("  - Stress Tests:     /api/risk/stress")
        print("  - Alert Dashboard:  /api/risk/alerts")
        print("  - Bot Performance:  /api/risk/bots")
        print("  - Emergency Controls: /api/risk/emergency (POST)")
        print("  - Risk Summary:     /api/risk-management/summary")
        print("")
        print("Advanced Risk Management: ENABLED")
    else:
        print("Advanced Risk Management: DISABLED (module not available)")

    print("")
    print("Press Ctrl+C to stop")
    print("=" * 70)

    app.run(host=os.getenv('DASHBOARD_HOST', '127.0.0.1'), port=5000, debug=os.getenv('DASHBOARD_DEBUG', 'false').lower() == 'true')
