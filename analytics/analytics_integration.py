"""
Analytics Integration - Main Integration Module
==============================================

Central integration point for analytics with the trading system including:
- Master orchestrator integration
- Real-time data feeds
- Automated reporting
- Dashboard integration
- Alert system coordination

Author: Trading Bot System
Created: February 2026
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import json

# Import analytics modules
from .core.performance_tracker import PerformanceTracker
from .core.pnl_calculator import PnLCalculator
from .metrics.performance_ratios import PerformanceRatios
from .metrics.drawdown_monitor import DrawdownMonitor
from .metrics.trade_statistics import TradeStatistics
from .benchmarks.benchmark_analyzer import BenchmarkAnalyzer
from .reporting.performance_reporter import PerformanceReporter, ReportConfig
from .analytics_api import AnalyticsAPI

logger = logging.getLogger(__name__)


class AnalyticsIntegration:
    """
    Main analytics integration class that coordinates all analytics components
    with the trading system and provides a unified interface.

    Features:
    - Seamless integration with master orchestrator
    - Real-time data synchronization
    - Automated reporting schedules
    - Alert coordination
    - Dashboard data feeds
    - API server management
    """

    def __init__(
        self,
        starting_capital: float = 10000.0,
        enable_api_server: bool = True,
        api_port: int = 5001,
        auto_reporting: bool = True,
        alert_callback: Optional[Callable] = None
    ):
        """
        Initialize analytics integration.

        Args:
            starting_capital: Starting portfolio value
            enable_api_server: Whether to start API server
            api_port: API server port
            auto_reporting: Enable automatic reporting
            alert_callback: Callback for alerts
        """
        self.starting_capital = starting_capital
        self.enable_api_server = enable_api_server
        self.api_port = api_port
        self.auto_reporting = auto_reporting
        self.alert_callback = alert_callback

        # Initialize all analytics components
        self.performance_tracker = PerformanceTracker(
            starting_capital=starting_capital,
            alert_callback=self._handle_performance_alert
        )

        self.pnl_calculator = PnLCalculator()
        self.performance_ratios = PerformanceRatios()
        self.drawdown_monitor = DrawdownMonitor(alert_callback=self._handle_drawdown_alert)
        self.trade_statistics = TradeStatistics()
        self.benchmark_analyzer = BenchmarkAnalyzer()
        self.performance_reporter = PerformanceReporter()

        # API server (optional)
        self.api_server = None
        if enable_api_server:
            self.api_server = AnalyticsAPI(port=api_port)

        # Integration state
        self.is_running = False
        self._update_thread = None
        self._reporting_thread = None

        # Trading system integration
        self.master_orchestrator = None
        self.trade_data_source = None

        # Last sync timestamps
        self.last_trade_sync = datetime.now()
        self.last_position_sync = datetime.now()
        self.last_report_time = datetime.now()

        logger.info("Analytics Integration initialized")

    def integrate_with_orchestrator(self, master_orchestrator):
        """
        Integrate with the master orchestrator.

        Args:
            master_orchestrator: Master orchestrator instance
        """
        self.master_orchestrator = master_orchestrator

        # Hook into orchestrator events
        if hasattr(master_orchestrator, 'add_trade_callback'):
            master_orchestrator.add_trade_callback(self.on_trade_executed)

        if hasattr(master_orchestrator, 'add_position_callback'):
            master_orchestrator.add_position_callback(self.on_position_update)

        logger.info("Integrated with master orchestrator")

    def start_analytics(self):
        """Start all analytics components"""
        try:
            self.is_running = True

            # Start performance tracking
            self.performance_tracker.start_monitoring()

            # Start API server if enabled
            if self.api_server:
                api_thread = threading.Thread(target=self.api_server.start_server, daemon=True)
                api_thread.start()
                logger.info(f"Started API server on port {self.api_port}")

            # Start integration update thread
            self._update_thread = threading.Thread(target=self._integration_loop, daemon=True)
            self._update_thread.start()

            # Start automated reporting if enabled
            if self.auto_reporting:
                self._reporting_thread = threading.Thread(target=self._reporting_loop, daemon=True)
                self._reporting_thread.start()

            logger.info("Analytics system started successfully")

        except Exception as e:
            logger.error(f"Error starting analytics: {e}")
            self.stop_analytics()

    def stop_analytics(self):
        """Stop all analytics components"""
        self.is_running = False

        # Stop performance tracking
        self.performance_tracker.stop_monitoring()

        # Stop API server
        if self.api_server:
            self.api_server.stop_server()

        # Wait for threads to finish
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=5)

        if self._reporting_thread and self._reporting_thread.is_alive():
            self._reporting_thread.join(timeout=5)

        logger.info("Analytics system stopped")

    def _integration_loop(self):
        """Main integration loop for syncing data"""
        while self.is_running:
            try:
                # Sync trading data
                self._sync_trading_data()

                # Update performance metrics
                self.performance_tracker.update_performance(force_update=True)

                # Check for alerts
                self._check_alerts()

                # Sleep for next update
                time.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error in integration loop: {e}")
                time.sleep(10)

    def _sync_trading_data(self):
        """Sync trading data from the trading system"""
        if not self.master_orchestrator:
            return

        try:
            # Get current positions from orchestrator
            current_positions = self._get_current_positions()

            # Update P&L calculator with current positions
            for position_id, position_data in current_positions.items():
                self.pnl_calculator.update_price(
                    position_data['symbol'],
                    position_data['current_price']
                )

            # Get recent trades
            recent_trades = self._get_recent_trades()

            # Update trade statistics
            if recent_trades:
                # This would integrate with actual trade data
                pass

            self.last_trade_sync = datetime.now()

        except Exception as e:
            logger.error(f"Error syncing trading data: {e}")

    def _get_current_positions(self) -> Dict:
        """Get current positions from trading system"""
        # This would integrate with the actual trading system
        # For now, return placeholder
        return {}

    def _get_recent_trades(self) -> List[Dict]:
        """Get recent trades from trading system"""
        # This would integrate with the actual trading system
        # For now, return placeholder
        return []

    def _check_alerts(self):
        """Check for and process alerts"""
        try:
            # Check performance alerts
            recent_alerts = self.performance_tracker.get_recent_alerts(1)

            for alert in recent_alerts:
                if not alert.get('acknowledged', False):
                    self._handle_alert(alert)

            # Check drawdown alerts
            drawdown_alerts = self.drawdown_monitor.get_recent_alerts(1)

            for alert in drawdown_alerts:
                if not alert.get('acknowledged', False):
                    self._handle_alert(alert)

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    def _handle_alert(self, alert: Dict):
        """Handle an alert from any analytics component"""
        try:
            logger.warning(f"Analytics Alert: {alert}")

            # Call external alert callback if provided
            if self.alert_callback:
                self.alert_callback(alert)

            # Send to API clients if API server is running
            if self.api_server:
                # This would broadcast to connected WebSocket clients
                pass

        except Exception as e:
            logger.error(f"Error handling alert: {e}")

    def _handle_performance_alert(self, alert: Dict):
        """Handle performance-specific alerts"""
        alert['source'] = 'performance_tracker'
        self._handle_alert(alert)

    def _handle_drawdown_alert(self, alert: Dict):
        """Handle drawdown-specific alerts"""
        alert['source'] = 'drawdown_monitor'
        self._handle_alert(alert)

    def _reporting_loop(self):
        """Automated reporting loop"""
        while self.is_running:
            try:
                now = datetime.now()

                # Daily report at 9 AM
                if (now.hour == 9 and now.minute == 0 and
                    (now - self.last_report_time).days >= 1):
                    self._generate_daily_report()
                    self.last_report_time = now

                # Weekly report on Monday at 9 AM
                if (now.weekday() == 0 and now.hour == 9 and now.minute == 0 and
                    (now - self.last_report_time).days >= 7):
                    self._generate_weekly_report()

                # Monthly report on 1st of month at 9 AM
                if (now.day == 1 and now.hour == 9 and now.minute == 0 and
                    (now - self.last_report_time).days >= 28):
                    self._generate_monthly_report()

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in reporting loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _generate_daily_report(self):
        """Generate daily performance report"""
        try:
            config = ReportConfig(
                report_type='daily',
                include_charts=True,
                output_formats=['json', 'html']
            )

            performance_data = self.get_comprehensive_performance_data()
            report_files = self.performance_reporter.generate_performance_report(
                performance_data, config, f"daily_report_{datetime.now().strftime('%Y%m%d')}"
            )

            logger.info(f"Generated daily report: {report_files}")

        except Exception as e:
            logger.error(f"Error generating daily report: {e}")

    def _generate_weekly_report(self):
        """Generate weekly performance report"""
        try:
            config = ReportConfig(
                report_type='weekly',
                include_charts=True,
                include_benchmark_comparison=True,
                output_formats=['json', 'html']
            )

            performance_data = self.get_comprehensive_performance_data()
            report_files = self.performance_reporter.generate_performance_report(
                performance_data, config, f"weekly_report_{datetime.now().strftime('%Y%W')}"
            )

            logger.info(f"Generated weekly report: {report_files}")

        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")

    def _generate_monthly_report(self):
        """Generate monthly performance report"""
        try:
            config = ReportConfig(
                report_type='monthly',
                include_charts=True,
                include_benchmark_comparison=True,
                include_detailed_trades=True,
                output_formats=['json', 'html', 'pdf']
            )

            performance_data = self.get_comprehensive_performance_data()
            report_files = self.performance_reporter.generate_performance_report(
                performance_data, config, f"monthly_report_{datetime.now().strftime('%Y%m')}"
            )

            logger.info(f"Generated monthly report: {report_files}")

        except Exception as e:
            logger.error(f"Error generating monthly report: {e}")

    # Trading system event handlers
    def on_trade_executed(self, trade_data: Dict):
        """Handle trade execution event from trading system"""
        try:
            # Open or close position in P&L calculator
            if trade_data.get('action') in ['buy', 'long', 'yes']:
                self.pnl_calculator.open_position(
                    position_id=trade_data['trade_id'],
                    symbol=trade_data['symbol'],
                    strategy=trade_data.get('strategy', 'unknown'),
                    platform=trade_data.get('platform', 'unknown'),
                    side=trade_data['action'],
                    entry_price=trade_data['price'],
                    quantity=trade_data.get('quantity', 1)
                )
            elif trade_data.get('action') in ['sell', 'short', 'no', 'close']:
                self.pnl_calculator.close_position(
                    position_id=trade_data['trade_id'],
                    exit_price=trade_data['price']
                )

            logger.info(f"Processed trade: {trade_data['symbol']} {trade_data['action']}")

        except Exception as e:
            logger.error(f"Error processing trade: {e}")

    def on_position_update(self, position_data: Dict):
        """Handle position update event from trading system"""
        try:
            # Update current price for the position
            self.pnl_calculator.update_price(
                position_data['symbol'],
                position_data['current_price']
            )

            logger.debug(f"Updated position: {position_data['symbol']} @ {position_data['current_price']}")

        except Exception as e:
            logger.error(f"Error updating position: {e}")

    # Public API methods
    def get_comprehensive_performance_data(self) -> Dict:
        """Get comprehensive performance data for reporting"""
        try:
            # Get current performance
            current_perf = self.performance_tracker.get_current_performance()

            # Get P&L data
            current_pnl = self.pnl_calculator.get_current_pnl()
            strategy_attribution = self.pnl_calculator.get_strategy_attribution()

            # Get performance ratios
            history = list(self.performance_tracker.performance_history)
            if history:
                returns = [h['daily_return'] for h in history[-252:]]
                ratios = self.performance_ratios.calculate_all_ratios(returns)
            else:
                ratios = self.performance_ratios.calculate_all_ratios([])

            # Get benchmark comparison
            if history:
                returns_series = pd.Series([h['daily_return'] for h in history])
                benchmarks = ['SPY', 'QQQ', 'BTC-USD']
                benchmark_comparison = self.benchmark_analyzer.compare_to_multiple_benchmarks(
                    returns_series, benchmarks
                )
            else:
                benchmark_comparison = {}

            # Get drawdown analysis
            drawdown_stats = self.drawdown_monitor.get_drawdown_statistics()

            return {
                'overall_metrics': current_perf,
                'pnl_data': {
                    'current': {
                        'total_pnl': float(current_pnl.total_pnl),
                        'unrealized_pnl': float(current_pnl.unrealized_pnl),
                        'realized_pnl': float(current_pnl.realized_pnl),
                        'daily_pnl': float(current_pnl.daily_pnl),
                        'by_strategy': {k: float(v) for k, v in current_pnl.by_strategy.items()},
                        'by_platform': {k: float(v) for k, v in current_pnl.by_platform.items()},
                        'by_market': {k: float(v) for k, v in current_pnl.by_market.items()}
                    },
                    'strategy_attribution': {
                        k: {
                            'total_pnl': float(v['total_pnl']),
                            'unrealized_pnl': float(v['unrealized_pnl']),
                            'realized_pnl': float(v['realized_pnl']),
                            'position_count': v['position_count']
                        }
                        for k, v in strategy_attribution.items()
                    }
                },
                'performance_ratios': ratios.to_dict(),
                'benchmark_comparison': {
                    k: v.to_dict() for k, v in benchmark_comparison.items()
                },
                'risk_metrics': {
                    'drawdown_statistics': drawdown_stats,
                    'current_drawdown': self.drawdown_monitor.current_drawdown,
                    'max_drawdown': self.drawdown_monitor.max_drawdown
                },
                'metadata': {
                    'calculation_time': datetime.now().isoformat(),
                    'starting_capital': float(self.starting_capital),
                    'current_value': current_perf['portfolio_value']
                }
            }

        except Exception as e:
            logger.error(f"Error getting comprehensive performance data: {e}")
            return {}

    def get_current_status(self) -> Dict:
        """Get current analytics system status"""
        return {
            'is_running': self.is_running,
            'api_server_enabled': self.enable_api_server,
            'api_port': self.api_port if self.enable_api_server else None,
            'auto_reporting_enabled': self.auto_reporting,
            'last_update': self.performance_tracker.last_update.isoformat(),
            'position_count': len(self.pnl_calculator.positions),
            'alerts_count': len(self.performance_tracker.get_recent_alerts(24)),
            'components_status': {
                'performance_tracker': self.performance_tracker.running,
                'api_server': self.api_server.is_running if self.api_server else False
            }
        }

    def generate_report_on_demand(
        self,
        report_type: str = 'custom',
        include_charts: bool = True,
        output_formats: List[str] = None
    ) -> Dict[str, str]:
        """Generate a performance report on demand"""
        try:
            if output_formats is None:
                output_formats = ['json', 'html']

            config = ReportConfig(
                report_type=report_type,
                include_charts=include_charts,
                output_formats=output_formats
            )

            performance_data = self.get_comprehensive_performance_data()
            report_files = self.performance_reporter.generate_performance_report(
                performance_data, config
            )

            return report_files

        except Exception as e:
            logger.error(f"Error generating on-demand report: {e}")
            return {}


# Integration helper function
def create_analytics_integration(
    master_orchestrator=None,
    starting_capital: float = 10000.0,
    enable_api: bool = True,
    api_port: int = 5001,
    alert_callback: Optional[Callable] = None
) -> AnalyticsIntegration:
    """
    Create and configure analytics integration.

    Args:
        master_orchestrator: Master orchestrator instance to integrate with
        starting_capital: Starting portfolio value
        enable_api: Enable API server
        api_port: API server port
        alert_callback: Alert callback function

    Returns:
        Configured AnalyticsIntegration instance
    """
    integration = AnalyticsIntegration(
        starting_capital=starting_capital,
        enable_api_server=enable_api,
        api_port=api_port,
        alert_callback=alert_callback
    )

    if master_orchestrator:
        integration.integrate_with_orchestrator(master_orchestrator)

    return integration


# Example usage
if __name__ == "__main__":
    def alert_handler(alert):
        print(f"ANALYTICS ALERT: {alert['message']}")

    # Create analytics integration
    analytics = create_analytics_integration(
        starting_capital=10000,
        enable_api=True,
        alert_callback=alert_handler
    )

    # Start analytics
    analytics.start_analytics()

    try:
        # Simulate some trading activity
        trade_data = {
            'trade_id': 'test_001',
            'symbol': 'AAPL',
            'action': 'buy',
            'price': 150.0,
            'quantity': 100,
            'strategy': 'momentum',
            'platform': 'alpaca'
        }
        analytics.on_trade_executed(trade_data)

        # Update position price
        position_data = {
            'symbol': 'AAPL',
            'current_price': 155.0
        }
        analytics.on_position_update(position_data)

        # Get status
        status = analytics.get_current_status()
        print(f"Analytics Status: {status}")

        # Generate report
        report_files = analytics.generate_report_on_demand()
        print(f"Generated reports: {report_files}")

        # Keep running
        print("Analytics system running... Press Ctrl+C to stop")
        while True:
            time.sleep(10)

    except KeyboardInterrupt:
        print("\nStopping analytics...")
        analytics.stop_analytics()