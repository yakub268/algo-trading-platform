"""
Advanced Risk Manager
====================

Comprehensive risk management system that integrates all risk components.

Features:
- Portfolio heat monitoring
- Correlation analysis
- VaR calculations
- Dynamic position sizing
- Drawdown protection
- Real-time alerts
- Emergency protocols

Author: Trading Bot Arsenal
Created: February 2026
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

from ..config.risk_config import RiskManagementConfig, AlertSeverity
from .portfolio_heat import PortfolioHeatMonitor, Position, HeatMetrics
from .correlation_monitor import CorrelationMonitor, CorrelationMetrics
from ..calculators.var_calculator import VaRCalculator, PortfolioVaRSummary
from ..calculators.kelly_optimizer import KellyOptimizer, PortfolioKellyResult
from ..monitors.drawdown_protection import DrawdownProtection, ProtectionStatus
from ..alerts.risk_alerts import RiskAlertManager, AlertType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AdvancedRiskManager')


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result"""
    timestamp: datetime
    overall_risk_score: float  # 0-100, higher = riskier

    # Component scores
    heat_score: float
    correlation_score: float
    var_score: float
    drawdown_score: float
    volatility_score: float

    # Position sizing recommendations
    recommended_sizes: Dict[str, float]
    position_multipliers: Dict[str, float]

    # Trading permissions
    new_trades_allowed: bool
    max_position_size: float
    emergency_mode: bool

    # Risk metrics
    portfolio_var_95: float
    max_correlation: float
    current_drawdown: float
    portfolio_heat: float

    # Alerts and warnings
    active_warnings: List[str]
    critical_alerts: List[str]
    recommendations: List[str]


@dataclass
class TradingDecision:
    """Trading decision with risk approval"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    recommended_size: float
    max_allowed_size: float
    risk_score: float
    approved: bool
    rejection_reasons: List[str]
    risk_adjustments: Dict[str, float]


class AdvancedRiskManager:
    """
    Master risk management system integrating all risk components.

    Provides unified risk assessment, position sizing, and trading approval.
    """

    def __init__(
        self,
        config: RiskManagementConfig,
        alert_callback: Optional[callable] = None
    ):
        """
        Initialize advanced risk manager.

        Args:
            config: Risk management configuration
            alert_callback: Function to call for alerts
        """
        self.config = config
        self.alert_callback = alert_callback

        # Initialize components
        self.heat_monitor = PortfolioHeatMonitor(config)
        self.correlation_monitor = CorrelationMonitor(config)
        self.var_calculator = VaRCalculator(config)
        self.kelly_optimizer = KellyOptimizer(config)
        self.drawdown_protection = DrawdownProtection(config, alert_callback)
        self.alert_manager = RiskAlertManager(config)

        # Current state
        self.portfolio_value = config.portfolio_value
        self.current_positions: Dict[str, Position] = {}
        self.last_assessment: Optional[RiskAssessment] = None

        # Emergency state
        self.emergency_mode = False
        self.emergency_reason = ""

        # Background monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()

        # Performance tracking
        self.assessment_history: List[RiskAssessment] = []

        logger.info("AdvancedRiskManager initialized")

    def add_position(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        strategy: str,
        risk_amount: float
    ) -> bool:
        """
        Add new position to risk monitoring.

        Args:
            symbol: Trading symbol
            size: Position size
            entry_price: Entry price
            strategy: Strategy name
            risk_amount: Risk amount for position

        Returns:
            True if position added successfully
        """
        try:
            # Add to heat monitor
            success = self.heat_monitor.add_position(
                symbol, size, entry_price, strategy, risk_amount
            )

            if success:
                # Update other components
                positions = {sym: pos.market_value for sym, pos in self.heat_monitor.positions.items()}
                self.correlation_monitor.update_positions(
                    {sym: val / self.portfolio_value for sym, val in positions.items()}
                )
                self.var_calculator.update_positions(positions, self.portfolio_value)

                # Store position reference
                self.current_positions[symbol] = self.heat_monitor.positions[symbol]

                logger.info(f"Position added to risk monitoring: {symbol}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to add position to risk monitoring: {e}")
            return False

    def remove_position(self, symbol: str) -> bool:
        """
        Remove position from risk monitoring.

        Args:
            symbol: Symbol to remove

        Returns:
            True if successful
        """
        try:
            # Remove from heat monitor
            position = self.heat_monitor.remove_position(symbol)

            if position:
                # Update other components
                positions = {sym: pos.market_value for sym, pos in self.heat_monitor.positions.items()}
                self.correlation_monitor.update_positions(
                    {sym: val / self.portfolio_value for sym, val in positions.items()}
                )
                self.var_calculator.update_positions(positions, self.portfolio_value)

                # Remove from local tracking
                if symbol in self.current_positions:
                    del self.current_positions[symbol]

                logger.info(f"Position removed from risk monitoring: {symbol}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to remove position from risk monitoring: {e}")
            return False

    def update_portfolio_value(self, new_value: float) -> ProtectionStatus:
        """
        Update portfolio value and trigger risk assessment.

        Args:
            new_value: New portfolio value

        Returns:
            Drawdown protection status
        """
        self.portfolio_value = new_value

        # Update drawdown protection
        protection_status = self.drawdown_protection.update_portfolio_value(new_value)

        # Check for emergency conditions
        if protection_status.severity_level.value == 'emergency' and not self.emergency_mode:
            self._activate_emergency_mode("Maximum drawdown reached")

        return protection_status

    def assess_overall_risk(self, update_prices: bool = True) -> RiskAssessment:
        """
        Perform comprehensive risk assessment.

        Args:
            update_prices: Whether to update prices first

        Returns:
            Complete risk assessment
        """
        timestamp = datetime.now()

        try:
            # Update prices if requested
            if update_prices and self.current_positions:
                symbols = list(self.current_positions.keys())
                self.heat_monitor.update_prices()

            # Get component assessments (each individually wrapped so one failure
            # doesn't trigger emergency mode for the entire portfolio)
            heat_metrics = self.heat_monitor.calculate_heat_metrics()
            correlation_metrics = self.correlation_monitor.analyze_correlations()

            try:
                var_summary = self.var_calculator.calculate_portfolio_var()
            except Exception as ve:
                logger.warning(f"VaR calculation failed (non-fatal): {ve}")
                var_summary = None

            protection_status = self.drawdown_protection._assess_protection_status(
                self.drawdown_protection._calculate_current_drawdown(timestamp)
            )

            # Calculate component scores (0-100)
            heat_score = heat_metrics.overall_heat
            correlation_score = self._calculate_correlation_score(correlation_metrics)
            var_score = self._calculate_var_score(var_summary) if var_summary else 0.0
            drawdown_score = protection_status.current_drawdown.drawdown_percentage * 500  # Scale to 0-100
            volatility_score = self._calculate_volatility_score()

            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(
                heat_score, correlation_score, var_score, drawdown_score, volatility_score
            )

            # Generate position sizing recommendations
            recommended_sizes, position_multipliers = self._calculate_position_recommendations()

            # Determine trading permissions
            new_trades_allowed = self._are_new_trades_allowed(
                heat_metrics, correlation_metrics, protection_status
            )

            max_position_size = self._calculate_max_position_size(overall_risk_score)

            # Collect warnings and alerts
            warnings, critical_alerts, recommendations = self._collect_risk_alerts(
                heat_metrics, correlation_metrics, var_summary, protection_status
            )

            # Create assessment
            assessment = RiskAssessment(
                timestamp=timestamp,
                overall_risk_score=overall_risk_score,
                heat_score=heat_score,
                correlation_score=correlation_score,
                var_score=var_score,
                drawdown_score=drawdown_score,
                volatility_score=volatility_score,
                recommended_sizes=recommended_sizes,
                position_multipliers=position_multipliers,
                new_trades_allowed=new_trades_allowed,
                max_position_size=max_position_size,
                emergency_mode=self.emergency_mode,
                portfolio_var_95=self._extract_var_95(var_summary),
                max_correlation=correlation_metrics.max_pairwise_correlation,
                current_drawdown=protection_status.current_drawdown.drawdown_percentage,
                portfolio_heat=heat_metrics.overall_heat,
                active_warnings=warnings,
                critical_alerts=critical_alerts,
                recommendations=recommendations
            )

            self.last_assessment = assessment

            # Store in history
            self.assessment_history.append(assessment)
            if len(self.assessment_history) > 100:
                self.assessment_history = self.assessment_history[-50:]

            # Send alerts if necessary
            self._send_risk_alerts(assessment)

            return assessment

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return self._create_emergency_assessment(f"Assessment error: {str(e)}")

    def evaluate_trade(
        self,
        symbol: str,
        action: str,
        proposed_size: float,
        strategy: str = "unknown",
        confidence: float = 1.0
    ) -> TradingDecision:
        """
        Evaluate a proposed trade against risk limits.

        Args:
            symbol: Trading symbol
            action: 'buy' or 'sell'
            proposed_size: Proposed position size
            strategy: Strategy name
            confidence: Trade confidence (0-1)

        Returns:
            Trading decision with approval/rejection
        """
        # Ensure we have current assessment
        if not self.last_assessment or (datetime.now() - self.last_assessment.timestamp).seconds > 300:
            self.assess_overall_risk()

        rejection_reasons = []
        risk_adjustments = {}

        # Emergency mode check
        if self.emergency_mode:
            rejection_reasons.append(f"Emergency mode active: {self.emergency_reason}")

        # New trades allowed check
        if action.lower() == 'buy' and not self.last_assessment.new_trades_allowed:
            rejection_reasons.append("New trades not allowed due to risk conditions")

        # Position size checks
        max_allowed = self.last_assessment.max_position_size
        if proposed_size > max_allowed:
            risk_adjustments['size_reduction'] = max_allowed / proposed_size
            rejection_reasons.append(f"Position size exceeds limit: ${proposed_size:,.0f} > ${max_allowed:,.0f}")

        # Strategy-specific Kelly sizing
        if symbol in self.last_assessment.recommended_sizes:
            kelly_size = self.last_assessment.recommended_sizes[symbol]
            if proposed_size > kelly_size * 1.5:  # Allow 50% above Kelly
                risk_adjustments['kelly_adjustment'] = kelly_size / proposed_size
                rejection_reasons.append(f"Position exceeds Kelly recommendation")

        # Correlation checks
        if action.lower() == 'buy':
            position_weight = proposed_size / self.portfolio_value
            allowed, corr_warnings = self.correlation_monitor.check_correlation_limits(symbol, position_weight)
            if not allowed:
                rejection_reasons.extend(corr_warnings)

        # Heat check
        if self.last_assessment.heat_score > 90:
            rejection_reasons.append("Portfolio heat too high")

        # Drawdown check
        if self.last_assessment.current_drawdown > self.config.drawdown_config.max_portfolio_drawdown * 0.8:
            rejection_reasons.append("Approaching maximum drawdown limit")

        # Calculate recommended size with all adjustments
        recommended_size = proposed_size
        for adjustment_type, multiplier in risk_adjustments.items():
            recommended_size *= multiplier

        # Apply position multipliers
        if symbol in self.last_assessment.position_multipliers:
            recommended_size *= self.last_assessment.position_multipliers[symbol]

        # Risk score calculation
        risk_score = self._calculate_trade_risk_score(
            symbol, action, proposed_size, strategy, confidence
        )

        # Final approval decision
        approved = len(rejection_reasons) == 0 and risk_score < 80

        decision = TradingDecision(
            symbol=symbol,
            action=action,
            recommended_size=recommended_size,
            max_allowed_size=max_allowed,
            risk_score=risk_score,
            approved=approved,
            rejection_reasons=rejection_reasons,
            risk_adjustments=risk_adjustments
        )

        # Log decision
        if approved:
            logger.info(f"Trade approved: {action} {symbol} ${recommended_size:,.0f} (Risk: {risk_score:.0f})")
        else:
            logger.warning(f"Trade rejected: {action} {symbol} - {'; '.join(rejection_reasons)}")

        return decision

    def start_monitoring(self):
        """Start background risk monitoring"""
        def monitor():
            while not self.stop_monitoring.is_set():
                try:
                    self.assess_overall_risk()
                    time.sleep(self.config.update_frequency)
                except Exception as e:
                    logger.error(f"Error in background monitoring: {e}")
                    time.sleep(60)  # Wait longer on error

        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
        logger.info("Background risk monitoring started")

    def stop_monitoring(self):
        """Stop background risk monitoring"""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=10)
            logger.info("Background risk monitoring stopped")

    def _activate_emergency_mode(self, reason: str):
        """Activate emergency mode"""
        self.emergency_mode = True
        self.emergency_reason = reason

        # Send emergency alert
        self.alert_manager.create_alert(
            alert_type=AlertType.SYSTEM_ERROR,
            severity=AlertSeverity.EMERGENCY,
            title="EMERGENCY MODE ACTIVATED",
            message=f"Emergency mode activated: {reason}",
            data={'reason': reason, 'timestamp': datetime.now().isoformat()}
        )

        logger.critical(f"EMERGENCY MODE ACTIVATED: {reason}")

    def _deactivate_emergency_mode(self, reason: str = "Manual override"):
        """Deactivate emergency mode"""
        self.emergency_mode = False
        self.emergency_reason = ""

        self.alert_manager.create_alert(
            alert_type=AlertType.SYSTEM_ERROR,
            severity=AlertSeverity.WARNING,
            title="Emergency mode deactivated",
            message=f"Emergency mode deactivated: {reason}",
            data={'reason': reason, 'timestamp': datetime.now().isoformat()}
        )

        logger.info(f"Emergency mode deactivated: {reason}")

    def _calculate_correlation_score(self, metrics: CorrelationMetrics) -> float:
        """Calculate correlation risk score (0-100)"""
        if metrics.correlation_matrix.empty:
            return 0.0

        # Base score from concentration ratio
        concentration_score = metrics.correlation_concentration_ratio * 100

        # Penalty for high correlations
        high_corr_penalty = len(metrics.high_correlation_pairs) * 10

        # Penalty for cluster concentration
        cluster_penalty = sum(
            20 for cluster in metrics.clusters
            if cluster.total_weight > self.config.correlation_limits.max_correlated_weight
        )

        total_score = concentration_score + high_corr_penalty + cluster_penalty
        return min(100, total_score)

    def _calculate_var_score(self, var_summary: PortfolioVaRSummary) -> float:
        """Calculate VaR risk score (0-100)"""
        if not var_summary.total_var:
            return 0.0

        # Get 95% 1-day VaR
        key = "historical_0.95_1d"
        if key in var_summary.total_var:
            var_result = var_summary.total_var[key]
            var_pct = var_result.var_percentage

            # Score based on VaR as percentage of portfolio
            score = (var_pct / self.config.var_config.daily_var_limit) * 100
            return min(100, score)

        return 50  # Default moderate score if no data

    def _calculate_volatility_score(self) -> float:
        """Calculate volatility risk score (0-100)"""
        # Simplified volatility scoring
        # In practice, would analyze current vs historical volatility
        return 30  # Placeholder

    def _calculate_overall_risk_score(
        self,
        heat_score: float,
        correlation_score: float,
        var_score: float,
        drawdown_score: float,
        volatility_score: float
    ) -> float:
        """Calculate weighted overall risk score"""
        weights = {
            'heat': 0.25,
            'correlation': 0.20,
            'var': 0.20,
            'drawdown': 0.25,
            'volatility': 0.10
        }

        weighted_score = (
            heat_score * weights['heat'] +
            correlation_score * weights['correlation'] +
            var_score * weights['var'] +
            drawdown_score * weights['drawdown'] +
            volatility_score * weights['volatility']
        )

        return min(100, weighted_score)

    def _calculate_position_recommendations(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate position size recommendations"""
        recommended_sizes = {}
        multipliers = {}

        if not self.current_positions:
            return recommended_sizes, multipliers

        # Get Kelly recommendations
        symbols = list(self.current_positions.keys())
        kelly_result = self.kelly_optimizer.optimize_portfolio_kelly(symbols)

        for symbol in symbols:
            # Base Kelly size
            if symbol in kelly_result.optimal_weights:
                kelly_weight = kelly_result.optimal_weights[symbol]
                kelly_size = kelly_weight * self.portfolio_value
            else:
                kelly_size = self.config.portfolio_value * 0.05  # Default 5%

            recommended_sizes[symbol] = kelly_size

            # Calculate overall multiplier
            multiplier = 1.0

            # Drawdown protection multiplier
            if self.last_assessment:
                protection_status = self.drawdown_protection._assess_protection_status(
                    self.drawdown_protection._calculate_current_drawdown(datetime.now())
                )
                multiplier *= protection_status.position_size_multiplier

            # Volatility adjustment (simplified)
            # In practice, would use actual volatility data
            multiplier *= 0.9

            multipliers[symbol] = multiplier

        return recommended_sizes, multipliers

    def _are_new_trades_allowed(
        self,
        heat_metrics: HeatMetrics,
        correlation_metrics: CorrelationMetrics,
        protection_status: ProtectionStatus
    ) -> bool:
        """Determine if new trades are allowed"""
        # Emergency mode
        if self.emergency_mode:
            return False

        # Drawdown protection
        if not protection_status.new_trades_allowed:
            return False

        # Heat limits
        if heat_metrics.overall_heat > 90:
            return False

        # No blocking conditions found
        return True

    def _calculate_max_position_size(self, overall_risk_score: float) -> float:
        """Calculate maximum allowed position size"""
        base_max = self.config.portfolio_limits.max_single_position * self.portfolio_value

        # Reduce based on overall risk
        if overall_risk_score > 80:
            multiplier = 0.5
        elif overall_risk_score > 60:
            multiplier = 0.75
        else:
            multiplier = 1.0

        return base_max * multiplier

    def _collect_risk_alerts(
        self,
        heat_metrics: HeatMetrics,
        correlation_metrics: CorrelationMetrics,
        var_summary: PortfolioVaRSummary,
        protection_status: ProtectionStatus
    ) -> Tuple[List[str], List[str], List[str]]:
        """Collect warnings, alerts, and recommendations"""
        warnings = []
        critical_alerts = []
        recommendations = []

        # Heat warnings
        warnings.extend(heat_metrics.warnings)
        critical_alerts.extend(heat_metrics.limit_breaches)

        # Correlation warnings
        warnings.extend(correlation_metrics.concentration_warnings)

        # Drawdown alerts
        warnings.extend(protection_status.alerts)
        recommendations.extend(protection_status.recommendations)

        return warnings, critical_alerts, recommendations

    def _extract_var_95(self, var_summary: PortfolioVaRSummary) -> float:
        """Extract 95% VaR from summary"""
        key = "historical_0.95_1d"
        if key in var_summary.total_var:
            return var_summary.total_var[key].var_value
        return 0.0

    def _send_risk_alerts(self, assessment: RiskAssessment):
        """Send appropriate risk alerts"""
        # High risk score alert
        if assessment.overall_risk_score > 80:
            self.alert_manager.create_alert(
                alert_type=AlertType.PORTFOLIO_HEAT,
                severity=AlertSeverity.CRITICAL,
                title="High Portfolio Risk",
                message=f"Overall risk score: {assessment.overall_risk_score:.0f}/100",
                data={'risk_score': assessment.overall_risk_score}
            )

        # Critical alerts
        for alert in assessment.critical_alerts:
            self.alert_manager.create_alert(
                alert_type=AlertType.POSITION_LIMIT,
                severity=AlertSeverity.CRITICAL,
                title=f"Risk Limit Breach: {alert[:50]}",
                message=alert
            )

    def _calculate_trade_risk_score(
        self,
        symbol: str,
        action: str,
        size: float,
        strategy: str,
        confidence: float
    ) -> float:
        """Calculate risk score for specific trade"""
        risk_factors = []

        # Size risk
        size_pct = size / self.portfolio_value
        size_risk = (size_pct / self.config.portfolio_limits.max_single_position) * 30
        risk_factors.append(size_risk)

        # Overall portfolio risk
        if self.last_assessment:
            portfolio_risk = self.last_assessment.overall_risk_score * 0.3
            risk_factors.append(portfolio_risk)

        # Confidence adjustment
        confidence_risk = (1 - confidence) * 20
        risk_factors.append(confidence_risk)

        # Strategy risk (simplified)
        strategy_risk = 10  # Default moderate risk
        risk_factors.append(strategy_risk)

        return sum(risk_factors)

    def _create_emergency_assessment(self, reason: str) -> RiskAssessment:
        """Create emergency risk assessment"""
        return RiskAssessment(
            timestamp=datetime.now(),
            overall_risk_score=100.0,
            heat_score=100.0,
            correlation_score=0.0,
            var_score=0.0,
            drawdown_score=0.0,
            volatility_score=0.0,
            recommended_sizes={},
            position_multipliers={},
            new_trades_allowed=False,
            max_position_size=0.0,
            emergency_mode=True,
            portfolio_var_95=0.0,
            max_correlation=0.0,
            current_drawdown=0.0,
            portfolio_heat=100.0,
            active_warnings=[reason],
            critical_alerts=[f"EMERGENCY: {reason}"],
            recommendations=["Immediate manual review required"]
        )

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive risk management status"""
        if not self.last_assessment:
            self.assess_overall_risk()

        assessment = self.last_assessment

        return {
            'timestamp': assessment.timestamp.isoformat(),
            'overall_risk_score': f"{assessment.overall_risk_score:.0f}/100",
            'emergency_mode': assessment.emergency_mode,
            'new_trades_allowed': assessment.new_trades_allowed,
            'portfolio_metrics': {
                'value': f"${self.portfolio_value:,.2f}",
                'heat': f"{assessment.portfolio_heat:.0f}/100",
                'var_95': f"${assessment.portfolio_var_95:,.2f}",
                'max_correlation': f"{assessment.max_correlation:.2%}",
                'current_drawdown': f"{assessment.current_drawdown:.2%}"
            },
            'risk_scores': {
                'overall': f"{assessment.overall_risk_score:.0f}/100",
                'heat': f"{assessment.heat_score:.0f}/100",
                'correlation': f"{assessment.correlation_score:.0f}/100",
                'var': f"{assessment.var_score:.0f}/100",
                'drawdown': f"{assessment.drawdown_score:.0f}/100",
                'volatility': f"{assessment.volatility_score:.0f}/100"
            },
            'position_limits': {
                'max_single_position': f"${assessment.max_position_size:,.2f}",
                'positions_monitored': len(self.current_positions)
            },
            'alerts': {
                'warnings': assessment.active_warnings,
                'critical': assessment.critical_alerts,
                'recommendations': assessment.recommendations
            },
            'component_status': {
                'heat_monitor': 'Active' if self.heat_monitor else 'Inactive',
                'correlation_monitor': 'Active' if self.correlation_monitor else 'Inactive',
                'var_calculator': 'Active' if self.var_calculator else 'Inactive',
                'kelly_optimizer': 'Active' if self.kelly_optimizer else 'Inactive',
                'drawdown_protection': 'Active' if self.drawdown_protection else 'Inactive',
                'alert_manager': 'Active' if self.alert_manager else 'Inactive'
            }
        }


if __name__ == "__main__":
    from ..config.risk_config import load_risk_config

    # Test the advanced risk manager
    config = load_risk_config()
    risk_manager = AdvancedRiskManager(config)

    # Add some test positions
    risk_manager.add_position('SPY', 1000, 450.0, 'momentum', 500)
    risk_manager.add_position('QQQ', 500, 350.0, 'mean_reversion', 300)
    risk_manager.add_position('AAPL', 200, 180.0, 'momentum', 200)

    # Update portfolio value
    risk_manager.update_portfolio_value(10500)

    # Perform risk assessment
    assessment = risk_manager.assess_overall_risk()

    print("Risk Assessment Results:")
    print(f"Overall Risk Score: {assessment.overall_risk_score:.0f}/100")
    print(f"New Trades Allowed: {assessment.new_trades_allowed}")
    print(f"Max Position Size: ${assessment.max_position_size:,.2f}")
    print(f"Emergency Mode: {assessment.emergency_mode}")

    # Test trade evaluation
    decision = risk_manager.evaluate_trade('MSFT', 'buy', 800, 'momentum', 0.8)
    print(f"\nTrade Decision for MSFT:")
    print(f"Approved: {decision.approved}")
    print(f"Recommended Size: ${decision.recommended_size:.2f}")
    print(f"Risk Score: {decision.risk_score:.0f}")
    if decision.rejection_reasons:
        print(f"Rejection Reasons: {decision.rejection_reasons}")

    # Get status report
    status = risk_manager.get_status_report()
    print(f"\nRisk Manager Status:")
    print(f"Overall Risk: {status['overall_risk_score']}")
    print(f"Portfolio Heat: {status['portfolio_metrics']['heat']}")
    print(f"Current Drawdown: {status['portfolio_metrics']['current_drawdown']}")
    print(f"Active Warnings: {len(status['alerts']['warnings'])}")
    print(f"Critical Alerts: {len(status['alerts']['critical'])}")
