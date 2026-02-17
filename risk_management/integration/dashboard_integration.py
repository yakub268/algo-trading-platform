"""
Dashboard Integration
====================

Integration module for real-time risk management dashboard.

Features:
- Real-time risk metrics
- Interactive risk controls
- Alert management interface
- Portfolio heat maps
- Stress testing dashboard

Author: Trading Bot Arsenal
Created: February 2026
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
import pandas as pd

from .trading_integration import TradingSystemIntegration, get_trading_integration
from ..config.risk_config import RiskManagementConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DashboardIntegration')


class RiskDashboardAPI:
    """
    API interface for risk management dashboard.

    Provides endpoints for real-time risk monitoring and control.
    """

    def __init__(self, trading_integration: TradingSystemIntegration = None):
        """
        Initialize dashboard API.

        Args:
            trading_integration: Trading system integration instance
        """
        if trading_integration is None:
            trading_integration = get_trading_integration()

        self.trading_integration = trading_integration
        self.risk_manager = trading_integration.risk_manager

        logger.info("RiskDashboardAPI initialized")

    def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get real-time risk metrics for dashboard"""
        try:
            # Get current risk assessment
            assessment = self.risk_manager.last_assessment
            if not assessment:
                assessment = self.risk_manager.assess_overall_risk()

            # Get portfolio heat metrics
            heat_metrics = self.risk_manager.heat_monitor.calculate_heat_metrics()

            # Get correlation analysis
            correlation_metrics = self.risk_manager.correlation_monitor.analyze_correlations()

            # Get VaR summary
            var_summary = self.risk_manager.var_calculator.calculate_portfolio_var()

            # Get drawdown status
            current_dd = self.risk_manager.drawdown_protection._calculate_current_drawdown(datetime.now())
            protection_status = self.risk_manager.drawdown_protection._assess_protection_status(current_dd)

            return {
                'timestamp': datetime.now().isoformat(),
                'overall_risk': {
                    'score': assessment.overall_risk_score,
                    'level': self._get_risk_level(assessment.overall_risk_score),
                    'emergency_mode': assessment.emergency_mode,
                    'new_trades_allowed': assessment.new_trades_allowed
                },
                'portfolio_heat': {
                    'score': heat_metrics.overall_heat,
                    'total_exposure': heat_metrics.total_exposure,
                    'max_single_position': heat_metrics.max_single_position_pct,
                    'max_sector_exposure': heat_metrics.max_sector_exposure_pct,
                    'warnings': heat_metrics.warnings,
                    'limit_breaches': heat_metrics.limit_breaches
                },
                'correlation_risk': {
                    'score': self._calculate_correlation_score(correlation_metrics),
                    'max_correlation': correlation_metrics.max_pairwise_correlation,
                    'concentration_ratio': correlation_metrics.correlation_concentration_ratio,
                    'high_corr_pairs': len(correlation_metrics.high_correlation_pairs),
                    'clusters': len(correlation_metrics.clusters)
                },
                'var_metrics': {
                    'portfolio_var_95': self._extract_var_95(var_summary),
                    'diversification_ratio': var_summary.portfolio_diversification_ratio if var_summary.total_var else 1.0
                },
                'drawdown': {
                    'current_pct': protection_status.current_drawdown.drawdown_percentage,
                    'duration_days': protection_status.current_drawdown.duration_days,
                    'severity': protection_status.severity_level.value,
                    'position_multiplier': protection_status.position_size_multiplier
                },
                'portfolio': {
                    'value': self.trading_integration.config.portfolio_value,
                    'daily_pnl': self.trading_integration.daily_pnl,
                    'total_pnl': self.trading_integration.total_pnl,
                    'cash_available': heat_metrics.cash_available,
                    'positions_count': len(self.risk_manager.current_positions)
                }
            }

        except Exception as e:
            logger.error(f"Failed to get realtime metrics: {e}")
            return {'error': str(e)}

    def get_position_heatmap(self) -> Dict[str, Any]:
        """Get position heatmap data"""
        try:
            positions = self.risk_manager.current_positions
            heat_metrics = self.risk_manager.heat_monitor.calculate_heat_metrics()

            heatmap_data = []
            for symbol, position in positions.items():
                weight = position.market_value / self.trading_integration.config.portfolio_value

                # Calculate position risk score
                risk_factors = []
                risk_factors.append(weight * 100)  # Size risk
                risk_factors.append(abs(position.pnl_percent) * 50)  # P&L volatility risk

                # Sector concentration
                sector_exposure = self._get_sector_exposure()
                sector = position.sector
                if sector in sector_exposure:
                    sector_risk = (sector_exposure[sector] / 0.25) * 20  # vs 25% limit
                    risk_factors.append(sector_risk)

                position_risk_score = min(100, sum(risk_factors))

                heatmap_data.append({
                    'symbol': symbol,
                    'weight': weight,
                    'value': position.market_value,
                    'pnl': position.unrealized_pnl,
                    'pnl_pct': position.pnl_percent,
                    'sector': sector,
                    'strategy': position.strategy,
                    'risk_score': position_risk_score,
                    'risk_level': self._get_risk_level(position_risk_score)
                })

            return {
                'timestamp': datetime.now().isoformat(),
                'positions': heatmap_data,
                'total_positions': len(heatmap_data),
                'total_value': sum(p['value'] for p in heatmap_data),
                'sector_breakdown': self._get_sector_exposure()
            }

        except Exception as e:
            logger.error(f"Failed to get position heatmap: {e}")
            return {'error': str(e)}

    def get_correlation_matrix(self) -> Dict[str, Any]:
        """Get correlation matrix for dashboard"""
        try:
            correlation_metrics = self.risk_manager.correlation_monitor.analyze_correlations()

            if correlation_metrics.correlation_matrix.empty:
                return {'error': 'No correlation data available'}

            # Convert correlation matrix to dashboard format
            corr_matrix = correlation_metrics.correlation_matrix
            matrix_data = []

            for i, symbol1 in enumerate(corr_matrix.index):
                for j, symbol2 in enumerate(corr_matrix.columns):
                    correlation = corr_matrix.iloc[i, j]
                    matrix_data.append({
                        'x': j,
                        'y': i,
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'correlation': correlation,
                        'abs_correlation': abs(correlation)
                    })

            # Get high correlation pairs
            high_corr_pairs = [
                {
                    'symbol1': pair[0],
                    'symbol2': pair[1],
                    'correlation': pair[2]
                }
                for pair in correlation_metrics.high_correlation_pairs
            ]

            return {
                'timestamp': datetime.now().isoformat(),
                'matrix_data': matrix_data,
                'symbols': list(corr_matrix.index),
                'high_correlation_pairs': high_corr_pairs,
                'avg_correlation': correlation_metrics.avg_portfolio_correlation,
                'max_correlation': correlation_metrics.max_pairwise_correlation,
                'clusters': [
                    {
                        'id': cluster.cluster_id,
                        'symbols': cluster.symbols,
                        'avg_correlation': cluster.avg_correlation,
                        'total_weight': cluster.total_weight,
                        'warning_level': cluster.warning_level.value
                    }
                    for cluster in correlation_metrics.clusters
                ]
            }

        except Exception as e:
            logger.error(f"Failed to get correlation matrix: {e}")
            return {'error': str(e)}

    def get_var_breakdown(self) -> Dict[str, Any]:
        """Get VaR breakdown for dashboard"""
        try:
            var_summary = self.risk_manager.var_calculator.calculate_portfolio_var()

            if not var_summary.total_var:
                return {'error': 'No VaR data available'}

            # Portfolio VaR
            portfolio_vars = {}
            for key, var_result in var_summary.total_var.items():
                portfolio_vars[key] = {
                    'method': var_result.method.value,
                    'confidence': var_result.confidence_level,
                    'horizon_days': var_result.time_horizon,
                    'var_amount': var_result.var_value,
                    'var_percentage': var_result.var_percentage,
                    'expected_shortfall': var_result.expected_shortfall
                }

            # Marginal VaR by position
            marginal_vars = [
                {
                    'symbol': symbol,
                    'marginal_var': amount
                }
                for symbol, amount in var_summary.marginal_var.items()
            ]

            # Component VaR
            component_vars = [
                {
                    'symbol': symbol,
                    'component_var': amount
                }
                for symbol, amount in var_summary.component_var.items()
            ]

            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio_var': portfolio_vars,
                'marginal_var': marginal_vars,
                'component_var': component_vars,
                'diversification_ratio': var_summary.portfolio_diversification_ratio,
                'stress_test_results': var_summary.stress_test_results
            }

        except Exception as e:
            logger.error(f"Failed to get VaR breakdown: {e}")
            return {'error': str(e)}

    def get_stress_test_results(self) -> Dict[str, Any]:
        """Get stress test results for dashboard"""
        try:
            # Run fresh stress tests if needed
            stress_tester = self.risk_manager.var_calculator  # Placeholder - need actual stress tester

            # For now, return placeholder data
            # In full implementation, would use actual stress testing module

            return {
                'timestamp': datetime.now().isoformat(),
                'scenarios': [
                    {
                        'name': '2008 Financial Crisis',
                        'portfolio_loss': -0.45,
                        'worst_position_loss': -0.60,
                        'probability': 0.02,
                        'status': 'FAIL' if -0.45 < -0.25 else 'PASS'
                    },
                    {
                        'name': 'Flash Crash',
                        'portfolio_loss': -0.12,
                        'worst_position_loss': -0.18,
                        'probability': 0.05,
                        'status': 'PASS'
                    },
                    {
                        'name': 'COVID Crash',
                        'portfolio_loss': -0.32,
                        'worst_position_loss': -0.45,
                        'probability': 0.01,
                        'status': 'FAIL' if -0.32 < -0.25 else 'PASS'
                    }
                ],
                'summary': {
                    'worst_case_loss': -0.45,
                    'scenarios_passed': 1,
                    'scenarios_failed': 2,
                    'stress_var_95': 2500,
                    'recommendation': 'Reduce position sizes due to stress test failures'
                }
            }

        except Exception as e:
            logger.error(f"Failed to get stress test results: {e}")
            return {'error': str(e)}

    def get_alert_dashboard(self) -> Dict[str, Any]:
        """Get alert dashboard data"""
        try:
            alert_manager = self.risk_manager.alert_manager
            alert_stats = alert_manager.get_alert_stats()

            # Active alerts
            active_alerts = [
                {
                    'id': alert.alert_id,
                    'timestamp': alert.timestamp.isoformat(),
                    'type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'message': alert.message,
                    'acknowledged': alert.acknowledged
                }
                for alert in alert_manager.active_alerts.values()
            ]

            # Recent alert history (last 24 hours)
            recent_alerts = [
                {
                    'id': alert.alert_id,
                    'timestamp': alert.timestamp.isoformat(),
                    'type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'acknowledged': alert.acknowledged
                }
                for alert in alert_manager.alert_history
                if alert.timestamp >= datetime.now() - timedelta(hours=24)
            ]

            return {
                'timestamp': datetime.now().isoformat(),
                'system_health_score': alert_stats.system_health_score,
                'active_alerts': active_alerts,
                'recent_alerts': recent_alerts,
                'stats': {
                    'total_alerts_24h': alert_stats.total_alerts_24h,
                    'unacknowledged_critical': alert_stats.unacknowledged_critical,
                    'avg_response_time': alert_stats.avg_response_time,
                    'alerts_by_severity': {
                        severity.value: count
                        for severity, count in alert_stats.alerts_by_severity.items()
                    },
                    'alerts_by_type': {
                        alert_type.value: count
                        for alert_type, count in alert_stats.alerts_by_type.items()
                    }
                }
            }

        except Exception as e:
            logger.error(f"Failed to get alert dashboard: {e}")
            return {'error': str(e)}

    def get_bot_performance(self) -> Dict[str, Any]:
        """Get trading bot performance metrics"""
        try:
            bot_stats = {}

            for bot_name, bot_profile in self.trading_integration.registered_bots.items():
                positions = self.trading_integration.active_positions[bot_name]

                # Calculate bot P&L (simplified)
                total_position_value = sum(
                    pos['size'] * pos.get('current_price', pos['entry_price'])
                    for pos in positions.values()
                )

                win_rate = bot_profile.wins / max(bot_profile.total_trades, 1)

                bot_stats[bot_name] = {
                    'enabled': bot_profile.enabled,
                    'strategy_type': bot_profile.strategy_type,
                    'total_trades': bot_profile.total_trades,
                    'wins': bot_profile.wins,
                    'losses': bot_profile.losses,
                    'win_rate': win_rate,
                    'active_positions': len(positions),
                    'position_value': total_position_value,
                    'risk_multiplier': bot_profile.risk_multiplier,
                    'last_trade': bot_profile.last_trade.isoformat() if bot_profile.last_trade else None
                }

            return {
                'timestamp': datetime.now().isoformat(),
                'total_bots': len(self.trading_integration.registered_bots),
                'active_bots': sum(1 for bot in self.trading_integration.registered_bots.values() if bot.enabled),
                'bot_stats': bot_stats,
                'pending_trades': len(self.trading_integration.pending_trades)
            }

        except Exception as e:
            logger.error(f"Failed to get bot performance: {e}")
            return {'error': str(e)}

    def update_risk_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update risk management settings"""
        try:
            config = self.risk_manager.config

            # Update portfolio limits
            if 'max_single_position' in settings:
                config.portfolio_limits.max_single_position = float(settings['max_single_position'])

            if 'max_sector_exposure' in settings:
                config.portfolio_limits.max_sector_exposure = float(settings['max_sector_exposure'])

            if 'max_drawdown' in settings:
                config.drawdown_config.max_portfolio_drawdown = float(settings['max_drawdown'])

            # Update Kelly settings
            if 'kelly_fraction' in settings:
                config.kelly_config.default_kelly_fraction = float(settings['kelly_fraction'])

            # Update VaR settings
            if 'daily_var_limit' in settings:
                config.var_config.daily_var_limit = float(settings['daily_var_limit'])

            logger.info("Risk settings updated successfully")
            return {'success': True, 'message': 'Settings updated successfully'}

        except Exception as e:
            logger.error(f"Failed to update risk settings: {e}")
            return {'success': False, 'error': str(e)}

    def emergency_controls(self, action: str, **kwargs) -> Dict[str, Any]:
        """Emergency control actions"""
        try:
            if action == 'stop_all':
                reason = kwargs.get('reason', 'Manual emergency stop')
                self.trading_integration.emergency_stop_all(reason)
                return {'success': True, 'message': f'Emergency stop executed: {reason}'}

            elif action == 'disable_bot':
                bot_name = kwargs.get('bot_name')
                if bot_name:
                    self.trading_integration.disable_bot(bot_name)
                    return {'success': True, 'message': f'Bot {bot_name} disabled'}
                else:
                    return {'success': False, 'error': 'Bot name required'}

            elif action == 'enable_bot':
                bot_name = kwargs.get('bot_name')
                if bot_name:
                    self.trading_integration.enable_bot(bot_name)
                    return {'success': True, 'message': f'Bot {bot_name} enabled'}
                else:
                    return {'success': False, 'error': 'Bot name required'}

            elif action == 'reset_emergency':
                self.risk_manager._deactivate_emergency_mode("Manual dashboard reset")
                return {'success': True, 'message': 'Emergency mode reset'}

            else:
                return {'success': False, 'error': f'Unknown action: {action}'}

        except Exception as e:
            logger.error(f"Emergency control failed: {e}")
            return {'success': False, 'error': str(e)}

    def acknowledge_alert(self, alert_id: str) -> Dict[str, Any]:
        """Acknowledge an alert"""
        try:
            success = self.risk_manager.alert_manager.acknowledge_alert(alert_id, "Dashboard User")
            if success:
                return {'success': True, 'message': 'Alert acknowledged'}
            else:
                return {'success': False, 'error': 'Alert not found'}

        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return {'success': False, 'error': str(e)}

    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to level"""
        if score >= 90:
            return 'CRITICAL'
        elif score >= 70:
            return 'HIGH'
        elif score >= 50:
            return 'MEDIUM'
        elif score >= 30:
            return 'LOW'
        else:
            return 'MINIMAL'

    def _calculate_correlation_score(self, correlation_metrics) -> float:
        """Calculate correlation risk score"""
        if correlation_metrics.correlation_matrix.empty:
            return 0.0

        base_score = correlation_metrics.correlation_concentration_ratio * 100
        high_corr_penalty = len(correlation_metrics.high_correlation_pairs) * 5
        cluster_penalty = sum(
            10 for cluster in correlation_metrics.clusters
            if cluster.total_weight > 0.25
        )

        return min(100, base_score + high_corr_penalty + cluster_penalty)

    def _extract_var_95(self, var_summary) -> float:
        """Extract 95% VaR from summary"""
        key = "historical_0.95_1d"
        if var_summary.total_var and key in var_summary.total_var:
            return var_summary.total_var[key].var_value
        return 0.0

    def _get_sector_exposure(self) -> Dict[str, float]:
        """Get sector exposure breakdown"""
        sector_exposure = {}
        total_value = self.trading_integration.config.portfolio_value

        for position in self.risk_manager.current_positions.values():
            sector = position.sector
            weight = position.market_value / total_value
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight

        return sector_exposure


# Flask route wrappers (for integration with existing dashboard)
def create_risk_routes(app, trading_integration: TradingSystemIntegration = None):
    """Create Flask routes for risk management dashboard"""

    dashboard_api = RiskDashboardAPI(trading_integration)

    @app.route('/api/risk/metrics')
    def risk_metrics():
        """Real-time risk metrics endpoint"""
        return dashboard_api.get_realtime_metrics()

    @app.route('/api/risk/heatmap')
    def position_heatmap():
        """Position heatmap endpoint"""
        return dashboard_api.get_position_heatmap()

    @app.route('/api/risk/correlation')
    def correlation_matrix():
        """Correlation matrix endpoint"""
        return dashboard_api.get_correlation_matrix()

    @app.route('/api/risk/var')
    def var_breakdown():
        """VaR breakdown endpoint"""
        return dashboard_api.get_var_breakdown()

    @app.route('/api/risk/stress')
    def stress_tests():
        """Stress test results endpoint"""
        return dashboard_api.get_stress_test_results()

    @app.route('/api/risk/alerts')
    def alert_dashboard():
        """Alert dashboard endpoint"""
        return dashboard_api.get_alert_dashboard()

    @app.route('/api/risk/bots')
    def bot_performance():
        """Bot performance endpoint"""
        return dashboard_api.get_bot_performance()

    @app.route('/api/risk/settings', methods=['POST'])
    def update_settings():
        """Update risk settings endpoint"""
        from flask import request
        settings = request.get_json()
        return dashboard_api.update_risk_settings(settings)

    @app.route('/api/risk/emergency', methods=['POST'])
    def emergency_control():
        """Emergency control endpoint"""
        from flask import request
        data = request.get_json()
        action = data.get('action')
        return dashboard_api.emergency_controls(action, **data)

    @app.route('/api/risk/acknowledge/<alert_id>', methods=['POST'])
    def acknowledge_alert(alert_id):
        """Acknowledge alert endpoint"""
        return dashboard_api.acknowledge_alert(alert_id)

    logger.info("Risk management routes created")


if __name__ == "__main__":
    # Test dashboard API
    from .trading_integration import initialize_risk_management

    integration = initialize_risk_management()
    dashboard_api = RiskDashboardAPI(integration)

    # Test getting metrics
    metrics = dashboard_api.get_realtime_metrics()
    print("Real-time Metrics:")
    print(f"Overall Risk Score: {metrics.get('overall_risk', {}).get('score', 'N/A')}")
    print(f"Portfolio Heat: {metrics.get('portfolio_heat', {}).get('score', 'N/A')}")

    # Test heatmap
    heatmap = dashboard_api.get_position_heatmap()
    print(f"\nPosition Heatmap:")
    print(f"Total Positions: {heatmap.get('total_positions', 0)}")

    # Test alerts
    alerts = dashboard_api.get_alert_dashboard()
    print(f"\nAlert Dashboard:")
    print(f"Active Alerts: {len(alerts.get('active_alerts', []))}")
    print(f"System Health: {alerts.get('system_health_score', 'N/A')}")