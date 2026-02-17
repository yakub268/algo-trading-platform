"""
Mobile Trading Dashboard
========================

Comprehensive mobile-first dashboard for real-time trading portfolio management.

Features:
- Real-time portfolio monitoring with live P&L updates
- Push notifications for trade alerts and risk warnings
- Interactive performance charts and visualizations
- Bot control panel with start/stop/configure functionality
- Risk management dashboard with live metrics
- Trade execution interface for manual overrides
- Responsive design optimized for mobile, tablet, and desktop
- Progressive Web App (PWA) with offline capabilities
- WebSocket integration for real-time updates
- Voice alerts for critical events
- Dark/light mode toggle
- Portfolio heatmap visualization

Author: Trading Bot System
Created: February 2026
"""

import os
import sys
import json
import sqlite3
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
import logging

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import jwt as pyjwt
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import redis

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import trading bot modules
try:
    from dashboard.app import app as desktop_app  # Import existing dashboard
    from dashboard.freqtrade_client import FreqtradeClient
    from dashboard.alpaca_client import AlpacaClient
    from bots.multi_market_strategy import MultiMarketStrategy
    from scrapers.data_aggregator import DataAggregator
    from utils.telegram_alerts import TelegramAlerter
    from risk_management.integration.dashboard_integration import RiskDashboardAPI
    TRADING_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some trading modules not available: {e}")
    TRADING_MODULES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mobile_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'mobile-trading-dashboard-2026')

# Enable CORS and SocketIO
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Redis for session management and caching
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=0,
        decode_responses=True
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connection established")
except Exception as e:
    REDIS_AVAILABLE = False
    logger.warning(f"Redis not available - using in-memory storage: {e}")

# Initialize trading clients if available
clients = {}
if TRADING_MODULES_AVAILABLE:
    try:
        clients['freqtrade'] = FreqtradeClient()
        clients['alpaca'] = AlpacaClient()
        clients['strategy'] = MultiMarketStrategy()
        clients['aggregator'] = DataAggregator()
        clients['telegram'] = TelegramAlerter()
        clients['risk'] = RiskDashboardAPI()
        logger.info("Trading clients initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize trading clients: {e}")

# Global state for real-time updates
active_connections = set()
portfolio_cache = {}
alerts_cache = []
performance_cache = {}

class MobileDashboardAPI:
    """Main API class for mobile dashboard operations"""

    def __init__(self):
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'mobile_dashboard.db'
        )
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for mobile dashboard data"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'viewer',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    theme TEXT DEFAULT 'dark',
                    notifications_enabled BOOLEAN DEFAULT 1,
                    voice_alerts_enabled BOOLEAN DEFAULT 0,
                    refresh_interval INTEGER DEFAULT 30,
                    dashboard_layout TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data TEXT DEFAULT '{}',
                    acknowledged BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS trade_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity DECIMAL NOT NULL,
                    price DECIMAL,
                    type TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    exchange TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    executed_at TIMESTAMP
                )
            ''')

            conn.commit()

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            summary = {
                'total_value': 0,
                'day_pnl': 0,
                'day_pnl_percent': 0,
                'positions': [],
                'cash_balance': 0,
                'buying_power': 0,
                'exchanges': {},
                'last_updated': datetime.now(timezone.utc).isoformat()
            }

            # Get Alpaca portfolio data
            if 'alpaca' in clients:
                try:
                    alpaca_data = clients['alpaca'].get_account_summary()
                    if alpaca_data:
                        summary['exchanges']['alpaca'] = {
                            'equity': float(alpaca_data.get('equity', 0)),
                            'day_pnl': float(alpaca_data.get('unrealized_pl', 0)),
                            'positions': alpaca_data.get('positions', [])
                        }
                        summary['total_value'] += float(alpaca_data.get('equity', 0))
                        summary['day_pnl'] += float(alpaca_data.get('unrealized_pl', 0))
                except Exception as e:
                    logger.error(f"Failed to get Alpaca data: {e}")

            # Get Freqtrade data
            if 'freqtrade' in clients:
                try:
                    ft_data = clients['freqtrade'].get_summary()
                    if ft_data:
                        summary['exchanges']['freqtrade'] = {
                            'balance': ft_data.get('total_balance', 0),
                            'profit': ft_data.get('profit_total', 0),
                            'open_trades': ft_data.get('open_trades', [])
                        }
                        summary['total_value'] += float(ft_data.get('total_balance', 0))
                        summary['day_pnl'] += float(ft_data.get('profit_today', 0))
                except Exception as e:
                    logger.error(f"Failed to get Freqtrade data: {e}")

            # Calculate percentages
            if summary['total_value'] > 0:
                summary['day_pnl_percent'] = (summary['day_pnl'] / summary['total_value']) * 100

            return summary

        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            return {
                'total_value': 0,
                'day_pnl': 0,
                'day_pnl_percent': 0,
                'positions': [],
                'error': str(e),
                'last_updated': datetime.now(timezone.utc).isoformat()
            }

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts and notifications"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT id, type, severity, title, message, data, acknowledged, created_at
                    FROM alerts
                    WHERE acknowledged = 0
                    ORDER BY created_at DESC
                    LIMIT 50
                ''')

                alerts = []
                for row in cursor.fetchall():
                    alerts.append({
                        'id': row[0],
                        'type': row[1],
                        'severity': row[2],
                        'title': row[3],
                        'message': row[4],
                        'data': json.loads(row[5]) if row[5] else {},
                        'acknowledged': bool(row[6]),
                        'created_at': row[7]
                    })

                return alerts

        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []

    def create_alert(self, alert_type: str, severity: str, title: str, message: str, data: Dict = None):
        """Create a new alert"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO alerts (type, severity, title, message, data)
                    VALUES (?, ?, ?, ?, ?)
                ''', (alert_type, severity, title, message, json.dumps(data or {})))
                conn.commit()

                # Emit real-time alert
                socketio.emit('new_alert', {
                    'type': alert_type,
                    'severity': severity,
                    'title': title,
                    'message': message,
                    'data': data or {},
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

        except Exception as e:
            logger.error(f"Failed to create alert: {e}")

# Initialize mobile dashboard API
mobile_api = MobileDashboardAPI()

# --- Authentication helpers ---

def token_required(f):
    """Decorator to require a valid JWT token for protected routes."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]

        if not token:
            return jsonify({'error': 'Authentication token required'}), 401

        try:
            payload = pyjwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.current_user = payload
        except pyjwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except pyjwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        return f(*args, **kwargs)
    return decorated

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Register a new user. First user gets 'admin' role."""
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400

    try:
        with sqlite3.connect(mobile_api.db_path) as conn:
            # First user becomes admin
            user_count = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
            role = 'admin' if user_count == 0 else 'viewer'

            conn.execute(
                'INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
                (username, generate_password_hash(password), role)
            )
            conn.commit()

        return jsonify({'message': f'User {username} registered', 'role': role}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 409

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Authenticate and return a JWT token."""
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    with sqlite3.connect(mobile_api.db_path) as conn:
        row = conn.execute(
            'SELECT id, username, password_hash, role FROM users WHERE username = ?',
            (username,)
        ).fetchone()

    if not row or not check_password_hash(row[2], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    token = pyjwt.encode(
        {
            'user_id': row[0],
            'username': row[1],
            'role': row[3],
            'exp': datetime.now(timezone.utc) + timedelta(hours=24)
        },
        app.config['SECRET_KEY'],
        algorithm='HS256'
    )

    return jsonify({'token': token, 'username': row[1], 'role': row[3]})

@app.route('/')
def index():
    """Main mobile dashboard page"""
    return render_template('mobile_dashboard.html')

@app.route('/manifest.json')
def manifest():
    """PWA manifest file"""
    return send_from_directory('.', 'manifest.json')

@app.route('/service-worker.js')
def service_worker():
    """Service worker for PWA functionality"""
    return send_from_directory('.', 'service-worker.js')

# API Routes
@app.route('/api/portfolio')
def api_portfolio():
    """Get portfolio summary"""
    return jsonify(mobile_api.get_portfolio_summary())

@app.route('/api/alerts')
def api_alerts():
    """Get active alerts"""
    return jsonify({
        'alerts': mobile_api.get_active_alerts(),
        'count': len(mobile_api.get_active_alerts())
    })

@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def api_acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    try:
        with sqlite3.connect(mobile_api.db_path) as conn:
            conn.execute(
                'UPDATE alerts SET acknowledged = 1 WHERE id = ?',
                (alert_id,)
            )
            conn.commit()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bots/status')
def api_bots_status():
    """Get status of all trading bots"""
    try:
        status = {}

        if 'freqtrade' in clients:
            status['freqtrade'] = clients['freqtrade'].get_status()

        if 'strategy' in clients:
            status['strategy'] = {
                'active': True,
                'last_run': datetime.now().isoformat()
            }

        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bots/<bot_name>/control', methods=['POST'])
@token_required
def api_bot_control(bot_name):
    """Control bot (start/stop/restart) - requires authentication"""
    try:
        action = request.json.get('action')

        if bot_name == 'freqtrade' and 'freqtrade' in clients:
            if action == 'start':
                result = clients['freqtrade'].start_bot()
            elif action == 'stop':
                result = clients['freqtrade'].stop_bot()
            elif action == 'restart':
                result = clients['freqtrade'].restart_bot()
            else:
                return jsonify({'error': 'Invalid action'}), 400

            return jsonify(result)

        return jsonify({'error': 'Bot not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def api_performance():
    """Get performance metrics"""
    try:
        performance = {
            'daily_returns': [],
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_return': 0,
            'win_rate': 0,
            'profit_factor': 0
        }

        # Get performance data from trading systems
        if 'freqtrade' in clients:
            ft_performance = clients['freqtrade'].get_performance()
            if ft_performance:
                performance.update(ft_performance)

        return jsonify(performance)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trade/execute', methods=['POST'])
@token_required
def api_execute_trade():
    """Execute manual trade (requires authentication)"""
    try:
        trade_data = request.json

        # Validate required fields
        required_fields = ['symbol', 'side', 'quantity', 'type', 'exchange']
        for field in required_fields:
            if field not in trade_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Store trade in database with authenticated user
        with sqlite3.connect(mobile_api.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO trade_executions
                (symbol, side, quantity, price, type, exchange, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data['symbol'],
                trade_data['side'],
                trade_data['quantity'],
                trade_data.get('price'),
                trade_data['type'],
                trade_data['exchange'],
                request.current_user['username']
            ))
            trade_id = cursor.lastrowid
            conn.commit()

        # Execute trade based on exchange
        result = {'trade_id': trade_id, 'status': 'submitted'}

        if trade_data['exchange'] == 'alpaca' and 'alpaca' in clients:
            # Execute with Alpaca
            alpaca_result = clients['alpaca'].place_order(trade_data)
            result.update(alpaca_result)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        return jsonify({'error': str(e)}), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    join_room('dashboard_updates')
    active_connections.add(request.sid)
    emit('connected', {'message': 'Connected to trading dashboard'})

    # Send initial data
    emit('portfolio_update', mobile_api.get_portfolio_summary())
    emit('alerts_update', mobile_api.get_active_alerts())

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    leave_room('dashboard_updates')
    active_connections.discard(request.sid)

@socketio.on('subscribe_updates')
def handle_subscribe_updates(data):
    """Subscribe to specific update types"""
    update_types = data.get('types', [])
    join_room(f"updates_{request.sid}")
    emit('subscribed', {'types': update_types})

# Background tasks for real-time updates
def background_portfolio_updates():
    """Background task to send portfolio updates"""
    while True:
        try:
            if active_connections:
                portfolio_data = mobile_api.get_portfolio_summary()
                socketio.emit('portfolio_update', portfolio_data, room='dashboard_updates')

            socketio.sleep(30)  # Update every 30 seconds
        except Exception as e:
            logger.error(f"Portfolio update failed: {e}")
            socketio.sleep(60)

def background_alerts_monitor():
    """Background task to monitor and send alerts"""
    while True:
        try:
            if active_connections:
                # Check for new risk conditions
                if 'risk' in clients:
                    risk_metrics = clients['risk'].get_current_metrics()
                    if risk_metrics:
                        for metric, value in risk_metrics.items():
                            if metric == 'portfolio_drawdown' and value < -0.05:  # 5% drawdown
                                mobile_api.create_alert(
                                    'risk',
                                    'high',
                                    'Portfolio Drawdown Alert',
                                    f'Portfolio down {value:.2%}',
                                    {'metric': metric, 'value': value}
                                )

            socketio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Alerts monitor failed: {e}")
            socketio.sleep(60)

# Start background tasks
if __name__ == '__main__':
    # Start background tasks
    socketio.start_background_task(background_portfolio_updates)
    socketio.start_background_task(background_alerts_monitor)

    # Run the app
    port = int(os.getenv('MOBILE_DASHBOARD_PORT', 5001))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)