"""
Trade Audit Logger
==================

Specialized logging system for trading activities with regulatory compliance features.
Provides immutable audit trail for all trading-related activities.
"""

import csv
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from decimal import Decimal
from dataclasses import dataclass, asdict
from .immutable_log import ImmutableLogger
from ..vault.encryption import AdvancedEncryption

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Represents a complete trade record for audit purposes."""
    trade_id: str
    timestamp: datetime
    exchange: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: Decimal
    price: Decimal
    total_value: Decimal
    fees: Decimal
    strategy: str
    order_type: str
    execution_time_ms: int
    slippage: Optional[Decimal] = None
    market_conditions: Optional[Dict[str, Any]] = None
    risk_metrics: Optional[Dict[str, Any]] = None
    compliance_flags: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert Decimal objects to strings for JSON serialization
        for field in ['quantity', 'price', 'total_value', 'fees', 'slippage']:
            if data[field] is not None:
                data[field] = str(data[field])
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeRecord':
        """Create from dictionary."""
        # Convert string amounts back to Decimal
        for field in ['quantity', 'price', 'total_value', 'fees', 'slippage']:
            if data[field] is not None:
                data[field] = Decimal(data[field])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class AuditTradeLogger:
    """
    Comprehensive trade logging system for regulatory compliance.

    Features:
    - Immutable trade logging with cryptographic integrity
    - Regulatory reporting capabilities
    - Real-time compliance monitoring
    - Audit trail generation
    - Risk metric tracking
    """

    def __init__(self, config):
        self.config = config
        self.encryption = AdvancedEncryption() if config.hash_verification else None
        self.immutable_logger = ImmutableLogger(
            config.log_file_path,
            self.encryption if config.immutable_logging else None
        )
        self._running = False

        # Event types for different trading activities
        self.EVENT_TYPES = {
            'TRADE_EXECUTED': 'Trade executed successfully',
            'TRADE_FAILED': 'Trade execution failed',
            'ORDER_PLACED': 'Order placed on exchange',
            'ORDER_CANCELLED': 'Order cancelled',
            'POSITION_OPENED': 'New position opened',
            'POSITION_CLOSED': 'Position closed',
            'STOP_LOSS_TRIGGERED': 'Stop loss order triggered',
            'TAKE_PROFIT_TRIGGERED': 'Take profit order triggered',
            'MARGIN_CALL': 'Margin call received',
            'COMPLIANCE_VIOLATION': 'Compliance rule violation',
            'RISK_LIMIT_EXCEEDED': 'Risk limit exceeded',
            'SYSTEM_ERROR': 'System error during trading',
            'API_ERROR': 'Exchange API error',
            'CONNECTION_LOST': 'Connection to exchange lost',
            'BALANCE_UPDATE': 'Account balance updated'
        }

    def start(self):
        """Start the audit logger."""
        if not self._running:
            self.immutable_logger.initialize()
            self._running = True
            self.log_system_event('AUDIT_LOGGER_STARTED', {'timestamp': datetime.utcnow().isoformat()})
            logger.info("Audit trade logger started")

    async def stop(self):
        """Stop the audit logger."""
        if self._running:
            self.log_system_event('AUDIT_LOGGER_STOPPED', {'timestamp': datetime.utcnow().isoformat()})
            self._running = False
            logger.info("Audit trade logger stopped")

    def log_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Log a completed trade with full audit trail.

        Args:
            trade_data: Dictionary containing trade information

        Returns:
            str: Event ID for the logged trade
        """
        if not self._running:
            self.start()

        # Create trade record
        trade_record = self._create_trade_record(trade_data)

        # Log the trade event
        event_data = {
            'trade_record': trade_record.to_dict(),
            'compliance_status': self._check_compliance(trade_record),
            'risk_assessment': self._assess_trade_risk(trade_record),
            'market_context': self._capture_market_context(trade_record)
        }

        event_id = self.immutable_logger.log_event('TRADE_EXECUTED', event_data)

        # Log additional compliance information if required
        if trade_record.compliance_flags:
            self.log_compliance_event('TRADE_FLAGS_DETECTED', {
                'trade_id': trade_record.trade_id,
                'flags': trade_record.compliance_flags,
                'timestamp': trade_record.timestamp.isoformat()
            })

        logger.info(f"Trade logged: {trade_record.trade_id} - {trade_record.side} {trade_record.quantity} {trade_record.symbol} @ {trade_record.price}")
        return event_id

    def log_failed_trade(self, trade_data: Dict[str, Any], error: str) -> str:
        """
        Log a failed trade attempt.

        Args:
            trade_data: Dictionary containing attempted trade information
            error: Error message describing the failure

        Returns:
            str: Event ID for the logged failed trade
        """
        event_data = {
            'attempted_trade': trade_data,
            'error': error,
            'timestamp': datetime.utcnow().isoformat(),
            'exchange': trade_data.get('exchange'),
            'symbol': trade_data.get('symbol'),
            'strategy': trade_data.get('strategy')
        }

        return self.immutable_logger.log_event('TRADE_FAILED', event_data)

    def log_order_event(self, order_data: Dict[str, Any], event_type: str) -> str:
        """
        Log order-related events (placed, cancelled, etc.).

        Args:
            order_data: Dictionary containing order information
            event_type: Type of order event

        Returns:
            str: Event ID for the logged order event
        """
        if event_type not in self.EVENT_TYPES:
            raise ValueError(f"Unknown event type: {event_type}")

        event_data = {
            'order_id': order_data.get('order_id'),
            'exchange': order_data.get('exchange'),
            'symbol': order_data.get('symbol'),
            'side': order_data.get('side'),
            'quantity': str(order_data.get('quantity', 0)),
            'price': str(order_data.get('price', 0)),
            'order_type': order_data.get('order_type'),
            'strategy': order_data.get('strategy'),
            'timestamp': datetime.utcnow().isoformat()
        }

        return self.immutable_logger.log_event(event_type, event_data)

    def log_compliance_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """
        Log compliance-related events.

        Args:
            event_type: Type of compliance event
            data: Event data

        Returns:
            str: Event ID for the logged compliance event
        """
        enhanced_data = {
            **data,
            'compliance_timestamp': datetime.utcnow().isoformat(),
            'severity': self._determine_compliance_severity(event_type, data)
        }

        return self.immutable_logger.log_event(event_type, enhanced_data)

    def log_system_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """
        Log system-related events.

        Args:
            event_type: Type of system event
            data: Event data

        Returns:
            str: Event ID for the logged system event
        """
        system_data = {
            **data,
            'system_timestamp': datetime.utcnow().isoformat(),
            'log_level': self.config.log_level
        }

        return self.immutable_logger.log_event(event_type, system_data)

    def generate_audit_report(self, start_date: datetime, end_date: datetime,
                            include_failed: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive audit report for a date range.

        Args:
            start_date: Start date for the report
            end_date: End date for the report
            include_failed: Whether to include failed trades

        Returns:
            dict: Comprehensive audit report
        """
        # Get all relevant entries
        entries = self.immutable_logger.get_entries(start_date, end_date)

        # Filter trade entries
        trade_entries = [e for e in entries if e.event_type in ['TRADE_EXECUTED', 'TRADE_FAILED']]
        successful_trades = [e for e in trade_entries if e.event_type == 'TRADE_EXECUTED']
        failed_trades = [e for e in trade_entries if e.event_type == 'TRADE_FAILED']

        # Calculate summary statistics
        total_trades = len(successful_trades)
        total_volume = Decimal('0')
        total_fees = Decimal('0')
        exchanges = set()
        symbols = set()
        strategies = set()

        for entry in successful_trades:
            trade_data = entry.data.get('trade_record', {})
            total_volume += Decimal(trade_data.get('total_value', '0'))
            total_fees += Decimal(trade_data.get('fees', '0'))
            exchanges.add(trade_data.get('exchange'))
            symbols.add(trade_data.get('symbol'))
            strategies.add(trade_data.get('strategy'))

        # Compliance summary
        compliance_entries = [e for e in entries if 'COMPLIANCE' in e.event_type or 'VIOLATION' in e.event_type]

        report = {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'generated_at': datetime.utcnow().isoformat()
            },
            'trade_summary': {
                'total_successful_trades': total_trades,
                'total_failed_trades': len(failed_trades),
                'success_rate': (total_trades / (total_trades + len(failed_trades)) * 100) if (total_trades + len(failed_trades)) > 0 else 0,
                'total_volume': str(total_volume),
                'total_fees': str(total_fees),
                'unique_exchanges': list(exchanges),
                'unique_symbols': list(symbols),
                'unique_strategies': list(strategies)
            },
            'compliance_summary': {
                'total_compliance_events': len(compliance_entries),
                'compliance_events_by_type': self._group_events_by_type(compliance_entries)
            },
            'integrity_check': self.immutable_logger.verify_integrity(),
            'detailed_trades': [entry.to_dict() for entry in successful_trades] if total_trades <= 1000 else [],
            'detailed_failures': [entry.to_dict() for entry in failed_trades] if include_failed and len(failed_trades) <= 100 else []
        }

        return report

    def export_for_compliance(self, start_date: datetime, end_date: datetime,
                            output_format: str = 'json') -> str:
        """
        Export trade data in compliance-friendly format.

        Args:
            start_date: Start date for export
            end_date: End date for export
            output_format: Output format ('json', 'csv')

        Returns:
            str: Path to the exported file
        """
        report = self.generate_audit_report(start_date, end_date)

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"compliance_export_{timestamp}.{output_format}"
        filepath = self.config.log_file_path.replace('.log', f'_{filename}')

        if output_format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            rows = report.get('detailed_trades', []) + report.get('detailed_failures', [])
            if rows:
                fieldnames = sorted(set().union(*(r.keys() for r in rows)))
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    for row in rows:
                        flat = {}
                        for k, v in row.items():
                            flat[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
                        writer.writerow(flat)
            else:
                with open(filepath, 'w', newline='') as f:
                    f.write('No trades in report period\n')

        logger.info(f"Compliance export generated: {filepath}")
        return filepath

    def _create_trade_record(self, trade_data: Dict[str, Any]) -> TradeRecord:
        """Create a TradeRecord from trade data."""
        return TradeRecord(
            trade_id=trade_data.get('trade_id', ''),
            timestamp=datetime.fromisoformat(trade_data.get('timestamp', datetime.utcnow().isoformat())),
            exchange=trade_data.get('exchange', ''),
            symbol=trade_data.get('symbol', ''),
            side=trade_data.get('side', ''),
            quantity=Decimal(str(trade_data.get('quantity', 0))),
            price=Decimal(str(trade_data.get('price', 0))),
            total_value=Decimal(str(trade_data.get('total_value', 0))),
            fees=Decimal(str(trade_data.get('fees', 0))),
            strategy=trade_data.get('strategy', ''),
            order_type=trade_data.get('order_type', ''),
            execution_time_ms=trade_data.get('execution_time_ms', 0),
            slippage=Decimal(str(trade_data.get('slippage', 0))) if trade_data.get('slippage') else None,
            market_conditions=trade_data.get('market_conditions'),
            risk_metrics=trade_data.get('risk_metrics'),
            compliance_flags=trade_data.get('compliance_flags', [])
        )

    def _check_compliance(self, trade_record: TradeRecord) -> Dict[str, Any]:
        """Check trade compliance against rules."""
        compliance_status = {
            'compliant': True,
            'flags': [],
            'warnings': [],
            'checked_at': datetime.utcnow().isoformat()
        }

        # Example compliance checks
        if trade_record.total_value > Decimal('100000'):  # Large trade threshold
            compliance_status['flags'].append('LARGE_TRADE')

        if trade_record.slippage and trade_record.slippage > Decimal('0.05'):  # High slippage
            compliance_status['warnings'].append('HIGH_SLIPPAGE')

        if trade_record.compliance_flags:
            compliance_status['flags'].extend(trade_record.compliance_flags)

        compliance_status['compliant'] = len(compliance_status['flags']) == 0

        return compliance_status

    def _assess_trade_risk(self, trade_record: TradeRecord) -> Dict[str, Any]:
        """Assess risk metrics for the trade based on actual trade data."""
        # Position size risk: based on total trade value
        total_val = float(trade_record.total_value)
        if total_val >= 50000:
            position_size_risk = 'CRITICAL'
        elif total_val >= 25000:
            position_size_risk = 'HIGH'
        elif total_val >= 10000:
            position_size_risk = 'MEDIUM'
        else:
            position_size_risk = 'LOW'

        # Liquidity risk: slow execution suggests thin order book
        exec_ms = trade_record.execution_time_ms
        if exec_ms >= 5000:
            liquidity_risk = 'HIGH'
        elif exec_ms >= 2000:
            liquidity_risk = 'MEDIUM'
        else:
            liquidity_risk = 'LOW'

        # Execution risk: based on slippage
        if trade_record.slippage is not None:
            slip = abs(float(trade_record.slippage))
            if slip >= 0.05:
                execution_risk = 'HIGH'
            elif slip >= 0.02:
                execution_risk = 'MEDIUM'
            else:
                execution_risk = 'LOW'
        else:
            execution_risk = 'LOW'

        # Overall risk: highest of the three
        risk_levels = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
        max_risk = max(
            risk_levels[position_size_risk],
            risk_levels[liquidity_risk],
            risk_levels[execution_risk]
        )
        overall = {v: k for k, v in risk_levels.items()}[max_risk]

        return {
            'position_size_risk': position_size_risk,
            'liquidity_risk': liquidity_risk,
            'execution_risk': execution_risk,
            'overall_risk': overall,
            'details': {
                'trade_value': total_val,
                'execution_time_ms': exec_ms,
                'slippage': float(trade_record.slippage) if trade_record.slippage else None,
            },
            'assessed_at': datetime.utcnow().isoformat()
        }

    def _capture_market_context(self, trade_record: TradeRecord) -> Dict[str, Any]:
        """Capture market context at time of trade."""
        now_utc = datetime.now(timezone.utc)
        exchange = trade_record.exchange.lower()

        market_hours = self._get_market_hours(exchange, now_utc)

        # Volatility heuristic: market open/close windows tend to be volatile
        et_hour = (now_utc.hour - 5) % 24  # rough ET offset (EST)
        if et_hour in (9, 10, 15, 16):  # open/close windows
            volatility_level = 'HIGH'
        elif et_hour < 9 or et_hour >= 17:
            volatility_level = 'LOW'
        else:
            volatility_level = 'NORMAL'

        return {
            'market_hours': market_hours,
            'volatility_level': volatility_level,
            'exchange': exchange,
            'utc_hour': now_utc.hour,
            'day_of_week': now_utc.strftime('%A'),
            'captured_at': now_utc.isoformat()
        }

    @staticmethod
    def _get_market_hours(exchange: str, now_utc: datetime) -> str:
        """Determine if the market is open for the given exchange."""
        weekday = now_utc.weekday()  # 0=Mon, 6=Sun
        et_hour = (now_utc.hour - 5) % 24  # rough ET (EST, not accounting for DST)
        et_minute = now_utc.minute

        # Crypto exchanges: 24/7
        crypto_exchanges = ('alpaca_crypto', 'coinbase', 'binance', 'defi', 'crypto')
        if any(c in exchange for c in crypto_exchanges):
            return 'OPEN'

        # Prediction markets (Kalshi, Polymarket): ~24/7 with brief maintenance
        if any(p in exchange for p in ('kalshi', 'polymarket', 'prediction')):
            return 'OPEN'

        # Forex (OANDA): Sun 5 PM ET - Fri 5 PM ET
        if any(f in exchange for f in ('oanda', 'forex')):
            if weekday == 6:  # Sunday
                return 'OPEN' if et_hour >= 17 else 'CLOSED'
            elif weekday == 5:  # Saturday
                return 'CLOSED'
            elif weekday == 4:  # Friday
                return 'CLOSED' if et_hour >= 17 else 'OPEN'
            return 'OPEN'

        # Stock exchanges (Alpaca, default): 9:30 AM - 4:00 PM ET, Mon-Fri
        if weekday >= 5:  # Weekend
            return 'CLOSED'
        time_val = et_hour * 60 + et_minute
        if time_val < 4 * 60:  # Before 4:00 AM ET
            return 'CLOSED'
        elif time_val < 9 * 60 + 30:  # 4:00 AM - 9:30 AM ET
            return 'PRE_MARKET'
        elif time_val < 16 * 60:  # 9:30 AM - 4:00 PM ET
            return 'OPEN'
        elif time_val < 20 * 60:  # 4:00 PM - 8:00 PM ET
            return 'AFTER_HOURS'
        return 'CLOSED'

    def _determine_compliance_severity(self, event_type: str, data: Dict[str, Any]) -> str:
        """Determine severity level for compliance events."""
        if 'VIOLATION' in event_type:
            return 'HIGH'
        elif 'LIMIT_EXCEEDED' in event_type:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _group_events_by_type(self, entries: List) -> Dict[str, int]:
        """Group events by type and count them."""
        event_counts = {}
        for entry in entries:
            event_type = entry.event_type
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        return event_counts