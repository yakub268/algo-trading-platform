"""
Trading Dashboard V4.2 - REST API Endpoints
"""
import os
import logging
from flask import Blueprint, jsonify, request
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Read paper mode from environment
PAPER_MODE = os.getenv('PAPER_MODE', 'true').lower() == 'true'

from dashboard.services.broker_alpaca import AlpacaService
from dashboard.services.broker_kalshi import KalshiService
from dashboard.services.broker_oanda import OandaService
from dashboard.services.broker_coinbase import CoinbaseService, get_coinbase_service
from dashboard.services.edge_detector import EdgeDetector
from dashboard.services.ai_verifier import AIVerifier
from dashboard.services.strategy_manager import StrategyManager
from dashboard.models import DashboardData

logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize services (lazy loading)
_services = {}


def get_services():
    """Lazy load services"""
    global _services

    if not _services:
        logger.info("Initializing dashboard services...")

        try:
            _services['alpaca'] = AlpacaService()
        except Exception as e:
            logger.error(f"Failed to init Alpaca: {e}")
            _services['alpaca'] = None

        try:
            _services['kalshi'] = KalshiService()
        except Exception as e:
            logger.error(f"Failed to init Kalshi: {e}")
            _services['kalshi'] = None

        try:
            _services['oanda'] = OandaService()
        except Exception as e:
            logger.error(f"Failed to init OANDA: {e}")
            _services['oanda'] = None

        try:
            _services['coinbase'] = get_coinbase_service()
        except Exception as e:
            logger.error(f"Failed to init Coinbase: {e}")
            _services['coinbase'] = None

        _services['edge_detector'] = EdgeDetector(_services.get('kalshi'))
        _services['strategy_manager'] = StrategyManager()
        _services['ai_verifier'] = AIVerifier()

        # Register broker clients with risk management for emergency position closure
        # This ensures emergency_stop_all() can actually close positions, not just block new trades
        try:
            from risk_management.integration.trading_integration import get_trading_integration
            trading_integration = get_trading_integration()
            for broker_name in ('alpaca', 'kalshi', 'oanda', 'coinbase'):
                if _services.get(broker_name) is not None:
                    trading_integration.register_broker(broker_name, _services[broker_name])
            logger.info("Broker clients registered with risk management for emergency closure")
        except Exception as e:
            logger.warning(f"Could not register brokers with risk management: {e}")

        logger.info("Dashboard services initialized")

    return _services


@api_bp.route('/dashboard', methods=['GET'])
def get_dashboard_data():
    """
    Main endpoint - returns all dashboard data in one call.
    Frontend polls this every 5 seconds.
    """
    services = get_services()

    try:
        # Aggregate positions from all brokers
        positions = []

        if services.get('alpaca'):
            try:
                positions.extend(services['alpaca'].get_positions())
            except Exception as e:
                logger.error(f"Failed to get Alpaca positions: {e}")

        if services.get('kalshi'):
            try:
                positions.extend(services['kalshi'].get_positions())
            except Exception as e:
                logger.error(f"Failed to get Kalshi positions: {e}")

        if services.get('oanda'):
            try:
                positions.extend(services['oanda'].get_positions())
            except Exception as e:
                logger.error(f"Failed to get OANDA positions: {e}")

        if services.get('coinbase'):
            try:
                positions.extend(services['coinbase'].get_positions())
            except Exception as e:
                logger.error(f"Failed to get Coinbase positions: {e}")

        # Get broker health
        brokers = []

        if services.get('alpaca'):
            try:
                brokers.append(services['alpaca'].get_health())
            except Exception as e:
                logger.error(f"Failed to get Alpaca health: {e}")

        if services.get('kalshi'):
            try:
                brokers.append(services['kalshi'].get_health())
            except Exception as e:
                logger.error(f"Failed to get Kalshi health: {e}")

        if services.get('oanda'):
            try:
                brokers.append(services['oanda'].get_health())
            except Exception as e:
                logger.error(f"Failed to get OANDA health: {e}")

        if services.get('coinbase'):
            try:
                brokers.append(services['coinbase'].get_health())
            except Exception as e:
                logger.error(f"Failed to get Coinbase health: {e}")

        # Get strategies
        strategy_manager = services['strategy_manager']
        strategies = strategy_manager.get_strategies()

        # Get edges
        edge_detector = services['edge_detector']
        edges_raw = edge_detector.get_all_edges()
        edges = {k: [e.to_dict() for e in v] for k, v in edges_raw.items()}

        # Get alerts and recent trades
        alerts = strategy_manager.get_alerts()
        trades_result = strategy_manager.get_recent_trades()
        recent_trades = trades_result.get('trades', []) if isinstance(trades_result, dict) else trades_result

        # Get AI status
        ai_status = strategy_manager.get_ai_status()

        # Calculate totals
        total_pnl = sum(s.pnl for s in strategies)
        total_invested = sum(p.quantity * p.entry_price for p in positions)
        total_available = sum(b.buying_power for b in brokers)
        total_capital = total_invested + total_available

        exposure = (total_invested / total_capital * 100) if total_capital > 0 else 0
        avg_win_rate = sum(s.win_rate for s in strategies) / len(strategies) if strategies else 0
        max_drawdown = max((s.drawdown for s in strategies), default=0)

        # Build response
        data = DashboardData(
            strategies=strategies,
            positions=positions,
            brokers=brokers,
            edges=edges_raw,
            alerts=alerts,
            recent_trades=recent_trades,
            totals={
                "pnl": total_pnl,
                "invested": total_invested,
                "available": total_available,
                "exposure": exposure,
                "win_rate": avg_win_rate,
                "drawdown": max_drawdown,
                "paper_mode": PAPER_MODE
            },
            ai_status=ai_status
        )

        return jsonify(data.to_dict())

    except Exception as e:
        logger.exception(f"Dashboard data fetch failed: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route('/action', methods=['POST'])
def execute_action():
    """
    Execute a user action with verification.

    Body:
    {
        "action": "close_position",
        "params": {"symbol": "AAPL"},
        "verify_only": false
    }
    """
    services = get_services()

    try:
        data = request.json
        action = data.get('action')
        params = data.get('params', {})
        verify_only = data.get('verify_only', False)

        # Get current state for verification
        positions = []
        if services.get('alpaca'):
            positions.extend(services['alpaca'].get_positions())
        if services.get('kalshi'):
            positions.extend(services['kalshi'].get_positions())
        if services.get('oanda'):
            positions.extend(services['oanda'].get_positions())
        if services.get('coinbase'):
            positions.extend(services['coinbase'].get_positions())

        strategies = services['strategy_manager'].get_strategies()

        brokers = []
        if services.get('alpaca'):
            brokers.append(services['alpaca'].get_health())
        if services.get('kalshi'):
            brokers.append(services['kalshi'].get_health())
        if services.get('oanda'):
            brokers.append(services['oanda'].get_health())
        if services.get('coinbase'):
            brokers.append(services['coinbase'].get_health())

        # Verify action
        verifier = services['ai_verifier']
        verifier.update_state(strategies, positions, brokers)
        verification = verifier.verify_action(action, params)

        # If just verifying, return verification result
        if verify_only:
            return jsonify(verification)

        # Execute action
        success = False
        message = ""

        if action == "close_position":
            symbol = params.get('symbol')
            pos = next((p for p in positions if p.symbol == symbol), None)

            if pos:
                if pos.broker == "Alpaca" and services.get('alpaca'):
                    success = services['alpaca'].close_position(symbol)
                elif pos.broker == "Kalshi" and services.get('kalshi'):
                    success = services['kalshi'].close_position(symbol)
                elif pos.broker == "OANDA" and services.get('oanda'):
                    success = services['oanda'].close_position(symbol)

                message = f"Closed {symbol}" if success else f"Failed to close {symbol}"
            else:
                message = f"Position {symbol} not found"

        elif action == "close_all":
            brokers_acted = []
            if services.get('alpaca'):
                services['alpaca'].close_all_positions()
                brokers_acted.append('Alpaca')
            if services.get('kalshi'):
                kalshi_positions = [p for p in positions if p.broker == "Kalshi"]
                for pos in kalshi_positions:
                    services['kalshi'].close_position(pos.symbol)
                if kalshi_positions:
                    brokers_acted.append('Kalshi')
            if services.get('oanda'):
                services['oanda'].close_all_positions()
                brokers_acted.append('OANDA')

            success = True
            if brokers_acted:
                message = f"Closed all positions on: {', '.join(brokers_acted)}"
            else:
                message = "No broker connections available - no positions to close"

        elif action == "pause_all":
            services['strategy_manager'].pause_all()
            success = True
            message = "Paused all strategies"

        elif action == "resume_all":
            services['strategy_manager'].resume_all()
            success = True
            message = "Resumed all strategies"

        elif action == "toggle_strategy":
            success = services['strategy_manager'].toggle_strategy(params.get('id'))
            message = f"Toggled strategy {params.get('name')}"

        elif action == "kill_switch":
            # Pause all strategies
            services['strategy_manager'].pause_all()

            # Close all positions
            brokers_killed = []
            if services.get('alpaca'):
                services['alpaca'].close_all_positions()
                brokers_killed.append('Alpaca')
            if services.get('kalshi'):
                kalshi_positions = [p for p in positions if p.broker == "Kalshi"]
                for pos in kalshi_positions:
                    services['kalshi'].close_position(pos.symbol)
                if kalshi_positions:
                    brokers_killed.append('Kalshi')
            if services.get('oanda'):
                services['oanda'].close_all_positions()
                brokers_killed.append('OANDA')

            success = True
            broker_info = f" (closed on: {', '.join(brokers_killed)})" if brokers_killed else " (no broker connections active)"
            message = f"KILL SWITCH EXECUTED - All trading halted{broker_info}"

            # Send Telegram alert
            _send_telegram_alert("🚨 KILL SWITCH ACTIVATED - All trading halted")

        elif action == "reduce_position":
            symbol = params.get('symbol')
            pos = next((p for p in positions if p.symbol == symbol), None)

            if pos and pos.broker == "Alpaca" and services.get('alpaca'):
                success = services['alpaca'].reduce_position(symbol, 0.5)
                message = f"Reduced {symbol} by 50%"
            else:
                message = f"Cannot reduce position {symbol}"

        elif action == "set_stop":
            symbol = params.get('symbol')
            stop_price = params.get('stop_price')

            if not symbol or not stop_price:
                success = False
                message = "Missing symbol or stop_price parameter"
            else:
                stop_price = float(stop_price)
                pos = next((p for p in positions if p.symbol == symbol), None)

                if pos and pos.broker == "Alpaca" and services.get('alpaca'):
                    success = services['alpaca'].set_stop_loss(symbol, stop_price)
                    message = f"Stop loss set for {symbol} at ${stop_price}" if success else f"Failed to set stop loss for {symbol}"
                else:
                    success = False
                    message = f"Cannot set stop loss for {symbol} - position not found or broker unavailable"

        # Log action
        _log_action(action, params, success, message)

        return jsonify({
            "success": success,
            "message": message,
            "verification": verification
        })

    except Exception as e:
        logger.exception(f"Action execution failed: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@api_bp.route('/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts"""
    services = get_services()
    alerts = services['strategy_manager'].get_alerts()
    return jsonify([a.to_dict() for a in alerts])


@api_bp.route('/trades', methods=['GET'])
def get_trades():
    """Get recent trades"""
    services = get_services()
    limit = request.args.get('limit', 10, type=int)
    result = services['strategy_manager'].get_recent_trades(limit)
    trades = result.get('trades', []) if isinstance(result, dict) else result
    is_mock = result.get('is_mock', False) if isinstance(result, dict) else False
    return jsonify({
        "trades": [t.to_dict() for t in trades],
        "is_mock": is_mock
    })


@api_bp.route('/equity', methods=['GET'])
def get_equity_curve_legacy():
    """Get equity curve data"""
    services = get_services()
    days = request.args.get('days', 30, type=int)
    result = services['strategy_manager'].get_equity_curve(days)
    return jsonify(result)


@api_bp.route('/health', methods=['GET'])
def get_health():
    """Health check endpoint"""
    services = get_services()

    brokers = []
    if services.get('alpaca'):
        brokers.append(services['alpaca'].get_health().to_dict())
    if services.get('kalshi'):
        brokers.append(services['kalshi'].get_health().to_dict())
    if services.get('oanda'):
        brokers.append(services['oanda'].get_health().to_dict())
    if services.get('coinbase'):
        brokers.append(services['coinbase'].get_health().to_dict())

    return jsonify({
        "status": "ok",
        "brokers": brokers
    })


def _send_telegram_alert(message: str):
    """Send a Telegram alert"""
    try:
        from dashboard.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        import requests

        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }, timeout=5)
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")


def _log_action(action: str, params: dict, success: bool, message: str):
    """Log an action to the alert log"""
    logger.info(f"Action: {action} | Params: {params} | Success: {success} | {message}")
