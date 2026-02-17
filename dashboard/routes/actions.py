"""
Action endpoints - /api/action, /api/high-risk/*, /api/refresh, /api/risk-management/*
"""

import os
import json
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request

from dashboard.shared import (
    RISK_MANAGEMENT_AVAILABLE, RiskDashboardAPI, get_trading_integration,
    load_high_risk_config, save_high_risk_config,
)

actions_bp = Blueprint('actions', __name__)


@actions_bp.route('/api/refresh')
def refresh_data():
    """Force refresh all data sources"""
    try:
        from scrapers.data_aggregator import DataAggregator
        aggregator = DataAggregator()
        data = aggregator.fetch_all(use_cache=False)

        return jsonify({
            'success': True,
            'message': 'Data refreshed successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


# ============================================================================
# RISK MANAGEMENT INTEGRATION ENDPOINTS
# ============================================================================

@actions_bp.route('/api/risk-management/status')
def get_risk_management_status():
    """Get risk management system status"""
    try:
        if not RISK_MANAGEMENT_AVAILABLE:
            return jsonify({
                'success': False,
                'available': False,
                'error': 'Risk management system not available'
            })

        trading_integration = get_trading_integration()
        status = trading_integration.get_integration_status()

        return jsonify({
            'success': True,
            'available': True,
            'status': status,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'available': False,
            'error': str(e)
        })


@actions_bp.route('/api/risk-management/summary')
def get_risk_management_summary():
    """Get risk management summary for main dashboard"""
    try:
        if not RISK_MANAGEMENT_AVAILABLE:
            return jsonify({
                'success': False,
                'available': False,
                'summary': {
                    'overall_risk_score': 0,
                    'emergency_mode': False,
                    'active_alerts': 0,
                    'portfolio_heat': 0
                }
            })

        # Get risk dashboard API
        dashboard_api = RiskDashboardAPI()
        metrics = dashboard_api.get_realtime_metrics()

        if 'error' in metrics:
            return jsonify({
                'success': False,
                'available': True,
                'error': metrics['error'],
                'summary': {
                    'overall_risk_score': 0,
                    'emergency_mode': False,
                    'active_alerts': 0,
                    'portfolio_heat': 0
                }
            })

        # Extract summary data
        summary = {
            'overall_risk_score': metrics.get('overall_risk', {}).get('score', 0),
            'risk_level': metrics.get('overall_risk', {}).get('level', 'MINIMAL'),
            'emergency_mode': metrics.get('overall_risk', {}).get('emergency_mode', False),
            'new_trades_allowed': metrics.get('overall_risk', {}).get('new_trades_allowed', True),
            'portfolio_heat': metrics.get('portfolio_heat', {}).get('score', 0),
            'active_positions': metrics.get('portfolio', {}).get('positions_count', 0),
            'portfolio_value': metrics.get('portfolio', {}).get('value', 0),
            'daily_pnl': metrics.get('portfolio', {}).get('daily_pnl', 0),
            'total_pnl': metrics.get('portfolio', {}).get('total_pnl', 0),
            'drawdown_pct': metrics.get('drawdown', {}).get('current_pct', 0),
            'var_95': metrics.get('var_metrics', {}).get('portfolio_var_95', 0)
        }

        # Get alert count
        alerts = dashboard_api.get_alert_dashboard()
        if 'active_alerts' in alerts:
            summary['active_alerts'] = len(alerts['active_alerts'])
        else:
            summary['active_alerts'] = 0

        return jsonify({
            'success': True,
            'available': True,
            'summary': summary,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'available': False,
            'error': str(e),
            'summary': {
                'overall_risk_score': 0,
                'emergency_mode': False,
                'active_alerts': 0,
                'portfolio_heat': 0
            }
        })


# ============================================================================
# HIGH RISK TRADING TOGGLE API
# ============================================================================

@actions_bp.route('/api/high-risk/status')
def get_high_risk_status():
    """Get current high risk trading status"""
    try:
        config = load_high_risk_config()
        return jsonify({
            'success': True,
            'high_risk_enabled': config.get('high_risk_enabled', False),
            'aggressive_bots': config.get('aggressive_bots', []),
            'last_modified': config.get('last_modified'),
            'last_modified_by': config.get('last_modified_by'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'high_risk_enabled': False
        })


@actions_bp.route('/api/high-risk/toggle', methods=['POST'])
def toggle_high_risk():
    """Toggle high risk trading mode on/off"""
    try:
        config = load_high_risk_config()

        # Toggle the status
        new_status = not config.get('high_risk_enabled', False)
        config['high_risk_enabled'] = new_status
        config['last_modified'] = datetime.now(timezone.utc).isoformat()
        config['last_modified_by'] = 'dashboard'

        if save_high_risk_config(config):
            status_text = "ENABLED" if new_status else "DISABLED"
            return jsonify({
                'success': True,
                'high_risk_enabled': new_status,
                'message': f'High Risk Trades {status_text}',
                'aggressive_bots': config.get('aggressive_bots', []),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to save configuration'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@actions_bp.route('/api/high-risk/set', methods=['POST'])
def set_high_risk():
    """Explicitly set high risk trading mode"""
    try:
        data = request.get_json()
        new_status = data.get('enabled', False)

        config = load_high_risk_config()
        config['high_risk_enabled'] = new_status
        config['last_modified'] = datetime.now(timezone.utc).isoformat()
        config['last_modified_by'] = 'dashboard'

        if save_high_risk_config(config):
            status_text = "ENABLED" if new_status else "DISABLED"
            return jsonify({
                'success': True,
                'high_risk_enabled': new_status,
                'message': f'High Risk Trades {status_text}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to save configuration'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@actions_bp.route('/api/action', methods=['POST'])
def execute_action():
    """Execute various trading actions"""
    try:
        data = request.get_json()
        action = data.get('action')
        params = data.get('params', {})

        # Import strategy manager
        try:
            from services.strategy_manager import StrategyManager
            strategy_manager = StrategyManager()
        except ImportError:
            # If strategy manager not available, return mock success for now
            return jsonify({
                'success': True,
                'message': f'Action {action} executed (mock mode)',
                'action': action
            })

        if action == 'pause_all':
            strategy_manager.pause_all()
            return jsonify({
                'success': True,
                'message': 'All strategies paused',
                'action': action
            })
        elif action == 'resume_all':
            strategy_manager.resume_all()
            return jsonify({
                'success': True,
                'message': 'All strategies resumed',
                'action': action
            })
        elif action == 'toggle_strategy':
            strategy_id = params.get('id')
            strategy_name = params.get('name', f'Strategy {strategy_id}')
            if strategy_manager.toggle_strategy(strategy_id):
                return jsonify({
                    'success': True,
                    'message': f'Strategy {strategy_name} toggled',
                    'action': action
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Strategy {strategy_id} not found',
                    'action': action
                })
        elif action == 'kill_switch':
            # Emergency stop - pause all strategies and close ALL positions
            strategy_manager.pause_all()

            close_errors = []

            # Close all Alpaca positions
            try:
                from dashboard.services.broker_alpaca import AlpacaService
                alpaca_svc = AlpacaService()
                alpaca_svc.close_all_positions()
            except Exception as e:
                close_errors.append(f"Alpaca: {e}")

            # Close all Kalshi positions
            try:
                from dashboard.services.broker_kalshi import KalshiService
                kalshi_svc = KalshiService()
                for pos in kalshi_svc.get_positions():
                    kalshi_svc.close_position(pos.symbol)
            except Exception as e:
                close_errors.append(f"Kalshi: {e}")

            # Close all OANDA positions
            try:
                from dashboard.services.broker_oanda import OandaService
                oanda_svc = OandaService()
                oanda_svc.close_all_positions()
            except Exception as e:
                close_errors.append(f"OANDA: {e}")

            # Send Telegram alert
            try:
                from dashboard.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
                import requests as tg_requests
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    tg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                    tg_requests.post(tg_url, json={
                        "chat_id": TELEGRAM_CHAT_ID,
                        "text": "\U0001f6a8 KILL SWITCH ACTIVATED - All trading halted",
                        "parse_mode": "HTML"
                    }, timeout=5)
            except Exception:
                pass  # Telegram alert is best-effort

            error_detail = f" (errors: {'; '.join(close_errors)})" if close_errors else ""
            return jsonify({
                'success': True,
                'message': f'KILL SWITCH EXECUTED - All trading halted{error_detail}',
                'action': action
            })
        elif action == 'close_position':
            symbol = params.get('symbol')
            return jsonify({
                'success': True,
                'message': f'Position {symbol} closed (mock)',
                'action': action
            })
        elif action == 'close_all':
            return jsonify({
                'success': True,
                'message': 'All positions closed (mock)',
                'action': action
            })
        elif action == 'force_run_all':
            # Force all bots to run immediately - runs in background to avoid crashing dashboard
            try:
                import subprocess
                import sys

                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                script_path = os.path.join(script_dir, 'scripts', 'force_run_all_bots.py')
                log_path = os.path.join(script_dir, 'logs', 'force_run.log')

                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                capital = os.environ.get('TOTAL_CAPITAL', '500')

                # Run in background (non-blocking) to avoid crashing dashboard
                with open(log_path, 'w') as log_file:
                    subprocess.Popen([
                        sys.executable, script_path,
                        '--live', '--confirm-live', '--capital', capital
                    ], stdout=log_file, stderr=subprocess.STDOUT, env=env)

                return jsonify({
                    'success': True,
                    'message': f'Force run started in background. Check orchestrator window or logs/force_run.log for progress.',
                    'action': action
                })

            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Could not start force run: {str(e)}',
                    'action': action
                })
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown action: {action}',
                'action': action
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'action': data.get('action', 'unknown') if 'data' in locals() else 'unknown'
        })
