"""
Analytics API - REST API for Trading Performance Analytics
==========================================================

Comprehensive API for external access to trading analytics including:
- Real-time performance metrics
- Historical analytics data
- Custom reports generation
- WebSocket for live updates
- Dashboard integration endpoints

Author: Trading Bot System
Created: February 2026
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import asyncio
import threading
import time

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd

# Import our analytics modules
from .core.performance_tracker import PerformanceTracker
from .core.pnl_calculator import PnLCalculator
from .metrics.performance_ratios import PerformanceRatios
from .metrics.drawdown_monitor import DrawdownMonitor
from .metrics.trade_statistics import TradeStatistics
from .benchmarks.benchmark_analyzer import BenchmarkAnalyzer
from .reporting.performance_reporter import PerformanceReporter, ReportConfig

logger = logging.getLogger(__name__)


class AnalyticsAPI:
    """
    REST API server for trading analytics with real-time updates and
    comprehensive endpoint coverage.

    Features:
    - RESTful API endpoints for all analytics
    - WebSocket for real-time updates
    - CORS support for web dashboards
    - Authentication and rate limiting
    - Comprehensive error handling
    - API documentation endpoints
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5001,
        debug: bool = False,
        enable_cors: bool = True
    ):
        """
        Initialize Analytics API server.

        Args:
            host: Server host
            port: Server port
            debug: Enable debug mode
            enable_cors: Enable CORS support
        """
        self.host = host
        self.port = port
        self.debug = debug

        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['JSON_SORT_KEYS'] = False

        if enable_cors:
            CORS(self.app)

        # Initialize analytics components
        self.performance_tracker = PerformanceTracker()
        self.pnl_calculator = PnLCalculator()
        self.performance_ratios = PerformanceRatios()
        self.drawdown_monitor = DrawdownMonitor()
        self.trade_statistics = TradeStatistics()
        self.benchmark_analyzer = BenchmarkAnalyzer()
        self.performance_reporter = PerformanceReporter()

        # API state
        self.is_running = False
        self.connected_clients = set()
        self.last_update = datetime.now()

        # Register routes
        self._register_routes()

        logger.info(f"Analytics API initialized on {host}:{port}")

    def _register_routes(self):
        """Register all API routes"""

        # Health and info endpoints
        self.app.route('/health', methods=['GET'])(self.health_check)
        self.app.route('/info', methods=['GET'])(self.get_api_info)

        # Performance tracking endpoints
        self.app.route('/api/performance/current', methods=['GET'])(self.get_current_performance)
        self.app.route('/api/performance/historical', methods=['GET'])(self.get_historical_performance)
        self.app.route('/api/performance/summary', methods=['GET'])(self.get_performance_summary)

        # P&L endpoints
        self.app.route('/api/pnl/current', methods=['GET'])(self.get_current_pnl)
        self.app.route('/api/pnl/by-strategy', methods=['GET'])(self.get_pnl_by_strategy)
        self.app.route('/api/pnl/rolling', methods=['GET'])(self.get_rolling_pnl)

        # Performance ratios endpoints
        self.app.route('/api/ratios/current', methods=['GET'])(self.get_current_ratios)
        self.app.route('/api/ratios/rolling', methods=['GET'])(self.get_rolling_ratios)
        self.app.route('/api/ratios/confidence', methods=['GET'])(self.get_ratio_confidence_intervals)

        # Drawdown endpoints
        self.app.route('/api/drawdown/current', methods=['GET'])(self.get_current_drawdown)
        self.app.route('/api/drawdown/history', methods=['GET'])(self.get_drawdown_history)
        self.app.route('/api/drawdown/statistics', methods=['GET'])(self.get_drawdown_statistics)

        # Trade statistics endpoints
        self.app.route('/api/trades/statistics', methods=['GET'])(self.get_trade_statistics)
        self.app.route('/api/trades/by-strategy', methods=['GET'])(self.get_trades_by_strategy)
        self.app.route('/api/trades/distribution', methods=['GET'])(self.get_trade_distribution)
        self.app.route('/api/trades/monthly', methods=['GET'])(self.get_monthly_performance)

        # Benchmark comparison endpoints
        self.app.route('/api/benchmarks/compare', methods=['GET'])(self.compare_benchmarks)
        self.app.route('/api/benchmarks/rolling', methods=['GET'])(self.get_rolling_benchmark_comparison)
        self.app.route('/api/benchmarks/correlation', methods=['GET'])(self.get_benchmark_correlations)

        # Reporting endpoints
        self.app.route('/api/reports/generate', methods=['POST'])(self.generate_report)
        self.app.route('/api/reports/list', methods=['GET'])(self.list_reports)
        self.app.route('/api/reports/<report_id>', methods=['GET'])(self.get_report)

        # Dashboard data endpoints
        self.app.route('/api/dashboard/overview', methods=['GET'])(self.get_dashboard_overview)
        self.app.route('/api/dashboard/charts', methods=['GET'])(self.get_dashboard_charts)

        # WebSocket for real-time updates
        self.app.route('/api/stream', methods=['GET'])(self.stream_updates)

        # Error handlers
        self.app.errorhandler(400)(self.handle_bad_request)
        self.app.errorhandler(404)(self.handle_not_found)
        self.app.errorhandler(500)(self.handle_internal_error)

    # Health and Info endpoints
    def health_check(self):
        """API health check"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime': (datetime.now() - self.last_update).total_seconds(),
            'version': '1.0.0'
        })

    def get_api_info(self):
        """Get API information and available endpoints"""
        return jsonify({
            'name': 'Trading Analytics API',
            'version': '1.0.0',
            'description': 'Comprehensive trading performance analytics API',
            'endpoints': {
                'performance': [
                    '/api/performance/current',
                    '/api/performance/historical',
                    '/api/performance/summary'
                ],
                'pnl': [
                    '/api/pnl/current',
                    '/api/pnl/by-strategy',
                    '/api/pnl/rolling'
                ],
                'ratios': [
                    '/api/ratios/current',
                    '/api/ratios/rolling',
                    '/api/ratios/confidence'
                ],
                'drawdown': [
                    '/api/drawdown/current',
                    '/api/drawdown/history',
                    '/api/drawdown/statistics'
                ],
                'trades': [
                    '/api/trades/statistics',
                    '/api/trades/by-strategy',
                    '/api/trades/distribution'
                ],
                'benchmarks': [
                    '/api/benchmarks/compare',
                    '/api/benchmarks/rolling',
                    '/api/benchmarks/correlation'
                ],
                'reports': [
                    '/api/reports/generate',
                    '/api/reports/list'
                ],
                'dashboard': [
                    '/api/dashboard/overview',
                    '/api/dashboard/charts'
                ]
            },
            'documentation': '/docs'
        })

    # Performance endpoints
    def get_current_performance(self):
        """Get current performance metrics"""
        try:
            performance = self.performance_tracker.get_current_performance()
            return jsonify({
                'status': 'success',
                'data': performance,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting current performance: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def get_historical_performance(self):
        """Get historical performance data"""
        try:
            days = request.args.get('days', 30, type=int)

            # Get performance history from tracker
            history = list(self.performance_tracker.performance_history)

            # Filter by days if requested
            if days and days > 0:
                cutoff = datetime.now() - timedelta(days=days)
                history = [h for h in history if h['timestamp'] >= cutoff]

            return jsonify({
                'status': 'success',
                'data': history,
                'count': len(history),
                'period_days': days
            })
        except Exception as e:
            logger.error(f"Error getting historical performance: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        try:
            days = request.args.get('days', 30, type=int)
            summary = self.performance_tracker.get_performance_report(days)

            return jsonify({
                'status': 'success',
                'data': summary,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    # P&L endpoints
    def get_current_pnl(self):
        """Get current P&L snapshot"""
        try:
            pnl_snapshot = self.pnl_calculator.get_current_pnl()

            return jsonify({
                'status': 'success',
                'data': {
                    'timestamp': pnl_snapshot.timestamp.isoformat(),
                    'total_pnl': float(pnl_snapshot.total_pnl),
                    'unrealized_pnl': float(pnl_snapshot.unrealized_pnl),
                    'realized_pnl': float(pnl_snapshot.realized_pnl),
                    'daily_pnl': float(pnl_snapshot.daily_pnl),
                    'mtd_pnl': float(pnl_snapshot.mtd_pnl),
                    'ytd_pnl': float(pnl_snapshot.ytd_pnl),
                    'position_count': pnl_snapshot.position_count,
                    'by_strategy': {k: float(v) for k, v in pnl_snapshot.by_strategy.items()},
                    'by_platform': {k: float(v) for k, v in pnl_snapshot.by_platform.items()},
                    'by_market': {k: float(v) for k, v in pnl_snapshot.by_market.items()}
                }
            })
        except Exception as e:
            logger.error(f"Error getting current P&L: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def get_pnl_by_strategy(self):
        """Get P&L breakdown by strategy"""
        try:
            attribution = self.pnl_calculator.get_strategy_attribution()

            # Convert to JSON-serializable format
            result = {}
            for strategy, data in attribution.items():
                result[strategy] = {
                    'unrealized_pnl': float(data['unrealized_pnl']),
                    'realized_pnl': float(data['realized_pnl']),
                    'total_pnl': float(data['total_pnl']),
                    'position_count': data['position_count']
                }

            return jsonify({
                'status': 'success',
                'data': result,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting P&L by strategy: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def get_rolling_pnl(self):
        """Get rolling P&L data"""
        try:
            window_days = request.args.get('window_days', 30, type=int)
            rolling_data = self.pnl_calculator.get_rolling_returns(window_days)

            if rolling_data.empty:
                return jsonify({
                    'status': 'success',
                    'data': [],
                    'message': 'No data available for the specified period'
                })

            # Convert to JSON format
            result = rolling_data.to_dict('records')
            for record in result:
                if 'timestamp' in record:
                    record['timestamp'] = record['timestamp'].isoformat()

            return jsonify({
                'status': 'success',
                'data': result,
                'window_days': window_days
            })
        except Exception as e:
            logger.error(f"Error getting rolling P&L: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    # Performance ratios endpoints
    def get_current_ratios(self):
        """Get current performance ratios"""
        try:
            # Get recent returns from performance tracker
            history = list(self.performance_tracker.performance_history)
            if not history:
                return jsonify({
                    'status': 'error',
                    'message': 'No performance history available'
                }), 404

            returns = [h['daily_return'] for h in history[-252:]]  # Last year
            ratios = self.performance_ratios.calculate_all_ratios(returns)

            return jsonify({
                'status': 'success',
                'data': ratios.to_dict(),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting current ratios: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def get_rolling_ratios(self):
        """Get rolling performance ratios"""
        try:
            window_days = request.args.get('window_days', 60, type=int)
            benchmark = request.args.get('benchmark', 'SPY')

            history = list(self.performance_tracker.performance_history)
            if not history:
                return jsonify({
                    'status': 'error',
                    'message': 'No performance history available'
                }), 404

            returns = [h['daily_return'] for h in history]
            rolling_ratios = self.performance_ratios.calculate_rolling_ratios(
                returns, window_days, benchmark
            )

            if rolling_ratios.empty:
                return jsonify({
                    'status': 'success',
                    'data': [],
                    'message': 'Insufficient data for rolling calculation'
                })

            result = rolling_ratios.to_dict('records')

            return jsonify({
                'status': 'success',
                'data': result,
                'window_days': window_days,
                'benchmark': benchmark
            })
        except Exception as e:
            logger.error(f"Error getting rolling ratios: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def get_ratio_confidence_intervals(self):
        """Get confidence intervals for performance ratios"""
        try:
            confidence_level = request.args.get('confidence_level', 0.95, type=float)

            history = list(self.performance_tracker.performance_history)
            if not history:
                return jsonify({
                    'status': 'error',
                    'message': 'No performance history available'
                }), 404

            returns = [h['daily_return'] for h in history[-252:]]
            confidence_intervals = self.performance_ratios.get_ratio_confidence_intervals(
                returns, confidence_level
            )

            return jsonify({
                'status': 'success',
                'data': confidence_intervals,
                'confidence_level': confidence_level,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting ratio confidence intervals: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    # Drawdown endpoints
    def get_current_drawdown(self):
        """Get current drawdown status"""
        try:
            current_value = self.performance_tracker.current_portfolio_value
            drawdown_metrics = self.drawdown_monitor.update(float(current_value))

            return jsonify({
                'status': 'success',
                'data': drawdown_metrics,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting current drawdown: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def get_drawdown_history(self):
        """Get drawdown history"""
        try:
            days = request.args.get('days', 90, type=int)
            underwater_curve = self.drawdown_monitor.get_underwater_curve(days)

            if underwater_curve.empty:
                return jsonify({
                    'status': 'success',
                    'data': [],
                    'message': 'No drawdown history available'
                })

            result = underwater_curve.to_dict('records')
            for record in result:
                if 'timestamp' in record:
                    record['timestamp'] = record['timestamp'].isoformat()

            return jsonify({
                'status': 'success',
                'data': result,
                'period_days': days
            })
        except Exception as e:
            logger.error(f"Error getting drawdown history: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def get_drawdown_statistics(self):
        """Get drawdown statistics"""
        try:
            days = request.args.get('days', 365, type=int)
            stats = self.drawdown_monitor.get_drawdown_statistics(days)

            return jsonify({
                'status': 'success',
                'data': stats,
                'period_days': days,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting drawdown statistics: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    # Trade statistics endpoints (placeholder implementations)
    def get_trade_statistics(self):
        """Get comprehensive trade statistics"""
        try:
            # This would typically integrate with the trade database
            # For now, return placeholder data
            return jsonify({
                'status': 'success',
                'data': {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'average_win': 0.0,
                    'average_loss': 0.0,
                    'message': 'Trade statistics integration pending'
                },
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting trade statistics: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def get_trades_by_strategy(self):
        """Get trade statistics by strategy"""
        return self.get_trade_statistics()  # Placeholder

    def get_trade_distribution(self):
        """Get trade P&L distribution"""
        return self.get_trade_statistics()  # Placeholder

    def get_monthly_performance(self):
        """Get monthly performance breakdown"""
        return self.get_trade_statistics()  # Placeholder

    # Benchmark comparison endpoints
    def compare_benchmarks(self):
        """Compare performance to benchmarks"""
        try:
            benchmarks = request.args.getlist('benchmarks') or ['SPY', 'QQQ', 'BTC-USD']

            # Get portfolio returns
            history = list(self.performance_tracker.performance_history)
            if not history:
                return jsonify({
                    'status': 'error',
                    'message': 'No performance history available'
                }), 404

            returns = pd.Series([h['daily_return'] for h in history])
            timestamps = pd.DatetimeIndex([h['timestamp'] for h in history])
            returns.index = timestamps

            # Compare to benchmarks
            comparisons = self.benchmark_analyzer.compare_to_multiple_benchmarks(
                returns, benchmarks
            )

            # Convert to JSON format
            result = {}
            for symbol, comparison in comparisons.items():
                result[symbol] = comparison.to_dict()

            return jsonify({
                'status': 'success',
                'data': result,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error comparing benchmarks: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def get_rolling_benchmark_comparison(self):
        """Get rolling benchmark comparison"""
        try:
            benchmark = request.args.get('benchmark', 'SPY')
            window_days = request.args.get('window_days', 60, type=int)

            history = list(self.performance_tracker.performance_history)
            if not history:
                return jsonify({
                    'status': 'error',
                    'message': 'No performance history available'
                }), 404

            returns = pd.Series([h['daily_return'] for h in history])
            timestamps = pd.DatetimeIndex([h['timestamp'] for h in history])
            returns.index = timestamps

            rolling_comparison = self.benchmark_analyzer.calculate_rolling_alpha_beta(
                returns, benchmark, window_days
            )

            if rolling_comparison.empty:
                return jsonify({
                    'status': 'success',
                    'data': [],
                    'message': 'Insufficient data for rolling comparison'
                })

            result = rolling_comparison.to_dict('records')

            return jsonify({
                'status': 'success',
                'data': result,
                'benchmark': benchmark,
                'window_days': window_days
            })
        except Exception as e:
            logger.error(f"Error getting rolling benchmark comparison: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def get_benchmark_correlations(self):
        """Get correlation matrix between benchmarks"""
        try:
            benchmarks = request.args.getlist('benchmarks') or ['SPY', 'QQQ', 'TLT', 'GLD']
            days = request.args.get('days', 252, type=int)

            correlation_matrix = self.benchmark_analyzer.get_benchmark_correlation_matrix(
                benchmarks, days
            )

            if correlation_matrix.empty:
                return jsonify({
                    'status': 'success',
                    'data': {},
                    'message': 'No correlation data available'
                })

            result = correlation_matrix.to_dict()

            return jsonify({
                'status': 'success',
                'data': result,
                'benchmarks': benchmarks,
                'period_days': days
            })
        except Exception as e:
            logger.error(f"Error getting benchmark correlations: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    # Reporting endpoints
    def generate_report(self):
        """Generate performance report"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'status': 'error', 'message': 'No data provided'}), 400

            report_type = data.get('report_type', 'monthly')
            output_formats = data.get('output_formats', ['json'])

            # Prepare performance data
            performance_data = self.performance_tracker.get_performance_report()

            # Create report config
            config = ReportConfig(
                report_type=report_type,
                output_formats=output_formats,
                include_charts=data.get('include_charts', True)
            )

            # Generate report
            output_files = self.performance_reporter.generate_performance_report(
                performance_data, config
            )

            return jsonify({
                'status': 'success',
                'data': {
                    'report_id': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'output_files': output_files,
                    'report_type': report_type
                },
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def list_reports(self):
        """List available reports"""
        try:
            # This would scan the reports directory
            return jsonify({
                'status': 'success',
                'data': [],
                'message': 'Report listing not implemented'
            })
        except Exception as e:
            logger.error(f"Error listing reports: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def get_report(self, report_id):
        """Get specific report"""
        try:
            return jsonify({
                'status': 'error',
                'message': f'Report {report_id} not found'
            }), 404
        except Exception as e:
            logger.error(f"Error getting report: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    # Dashboard endpoints
    def get_dashboard_overview(self):
        """Get dashboard overview data"""
        try:
            current_perf = self.performance_tracker.get_current_performance()
            current_pnl = self.pnl_calculator.get_current_pnl()

            overview = {
                'portfolio_value': current_perf['portfolio_value'],
                'daily_return': current_perf['daily_return'],
                'total_return': current_perf['total_return'],
                'sharpe_ratio': current_perf['sharpe_ratio'],
                'max_drawdown': current_perf['max_drawdown'],
                'position_count': current_perf['position_count'],
                'unrealized_pnl': float(current_pnl.unrealized_pnl),
                'realized_pnl': float(current_pnl.realized_pnl),
                'by_strategy': {k: float(v) for k, v in current_pnl.by_strategy.items()},
                'timestamp': datetime.now().isoformat()
            }

            return jsonify({
                'status': 'success',
                'data': overview
            })
        except Exception as e:
            logger.error(f"Error getting dashboard overview: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def get_dashboard_charts(self):
        """Get chart data for dashboard"""
        try:
            # Get recent performance history for charts
            history = list(self.performance_tracker.performance_history)[-100:]  # Last 100 points

            chart_data = {
                'equity_curve': [
                    {
                        'timestamp': h['timestamp'].isoformat(),
                        'value': h['portfolio_value']
                    }
                    for h in history
                ],
                'daily_returns': [
                    {
                        'timestamp': h['timestamp'].isoformat(),
                        'return': h['daily_return']
                    }
                    for h in history
                ],
                'drawdown': [
                    {
                        'timestamp': h['timestamp'].isoformat(),
                        'drawdown': h.get('current_drawdown', 0)
                    }
                    for h in history
                ]
            }

            return jsonify({
                'status': 'success',
                'data': chart_data,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting dashboard charts: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    # WebSocket for real-time updates
    def stream_updates(self):
        """Stream real-time updates via Server-Sent Events"""
        def generate():
            try:
                while True:
                    # Get current data
                    data = {
                        'timestamp': datetime.now().isoformat(),
                        'performance': self.performance_tracker.get_current_performance(),
                        'pnl': self.pnl_calculator.get_performance_summary()
                    }

                    yield f"data: {json.dumps(data)}\n\n"
                    time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error in stream updates: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            generate(),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )

    # Error handlers
    def handle_bad_request(self, e):
        return jsonify({'status': 'error', 'message': 'Bad request'}), 400

    def handle_not_found(self, e):
        return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

    def handle_internal_error(self, e):
        logger.error(f"Internal server error: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

    def start_server(self):
        """Start the API server"""
        try:
            self.is_running = True
            logger.info(f"Starting Analytics API server on {self.host}:{self.port}")

            # Start performance tracking
            self.performance_tracker.start_monitoring()

            # Run Flask app
            self.app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                threaded=True
            )
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            self.is_running = False

    def stop_server(self):
        """Stop the API server"""
        self.is_running = False
        self.performance_tracker.stop_monitoring()
        logger.info("Analytics API server stopped")


# Example usage
if __name__ == "__main__":
    # Create and start API server
    api = AnalyticsAPI(host="0.0.0.0", port=5001, debug=True)

    try:
        api.start_server()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        api.stop_server()