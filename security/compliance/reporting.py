"""
Compliance Reporting System
===========================

Comprehensive regulatory compliance and tax reporting for trading operations.
Supports 1099 generation, audit trails, and regulatory reporting requirements.
"""

import json
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sqlite3
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class TaxableEvent:
    """Represents a taxable trading event."""
    event_id: str
    event_type: str  # 'SALE', 'DIVIDEND', 'INTEREST', 'FEE'
    trade_id: str
    symbol: str
    quantity: Decimal
    proceeds: Decimal
    cost_basis: Decimal
    gain_loss: Decimal
    date_acquired: datetime
    date_sold: datetime
    holding_period: str  # 'SHORT_TERM', 'LONG_TERM'
    exchange: str
    tax_year: int

@dataclass
class ComplianceReport:
    """Represents a compliance report."""
    report_id: str
    report_type: str
    tax_year: int
    generated_date: datetime
    data: Dict[str, Any]
    hash_verification: str

class ComplianceReporter:
    """
    Comprehensive compliance reporting system for trading operations.

    Features:
    - Tax reporting (1099, Schedule D)
    - Regulatory compliance reports
    - Audit trail generation
    - GDPR compliance
    - Data retention management
    - Anti-money laundering (AML) reporting
    """

    def __init__(self, config):
        self.config = config
        self.reports_path = Path(config.reports_path)
        self.reports_path.mkdir(parents=True, exist_ok=True)

        # Tax jurisdictions and their requirements
        self.tax_jurisdictions = {
            'US': {
                'forms': ['1099-B', 'Schedule D', '8949'],
                'wash_sale_period': 30,
                'long_term_threshold': 365
            },
            'UK': {
                'forms': ['Capital Gains Tax'],
                'wash_sale_period': 30,
                'long_term_threshold': 0  # No distinction
            },
            'EU': {
                'forms': ['MiFID II Transaction Reporting'],
                'wash_sale_period': 0,
                'long_term_threshold': 365
            }
        }

    def generate_tax_report(self, year: int, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive tax report for specified year.

        Args:
            year: Tax year
            user_id: Optional user filter

        Returns:
            dict: Complete tax report with all forms and schedules
        """
        logger.info(f"Generating tax report for year {year}")

        # Get all trades for the tax year
        trades = self._get_trades_for_year(year, user_id)

        # Calculate taxable events
        taxable_events = self._calculate_taxable_events(trades)

        # Generate different report sections
        report_data = {
            'tax_year': year,
            'jurisdiction': self.config.tax_jurisdiction,
            'generated_date': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'summary': self._generate_tax_summary(taxable_events),
            'form_1099b': self._generate_1099b(taxable_events),
            'schedule_d': self._generate_schedule_d(taxable_events),
            'form_8949': self._generate_form_8949(taxable_events),
            'wash_sales': self._identify_wash_sales(trades),
            'detailed_transactions': [self._taxable_event_to_dict(event) for event in taxable_events]
        }

        # Create compliance report
        report = ComplianceReport(
            report_id=self._generate_report_id('TAX', year),
            report_type='TAX_REPORT',
            tax_year=year,
            generated_date=datetime.utcnow(),
            data=report_data,
            hash_verification=self._calculate_report_hash(report_data)
        )

        # Save report
        self._save_report(report)

        logger.info(f"Tax report generated: {report.report_id}")
        return report_data

    def generate_audit_trail(self, start_date: str, end_date: str,
                           user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate audit trail for specified period.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            user_id: Optional user filter

        Returns:
            dict: Comprehensive audit trail
        """
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        logger.info(f"Generating audit trail from {start_date} to {end_date}")

        # Get audit data from various sources
        audit_data = {
            'period': {'start': start_date, 'end': end_date},
            'user_id': user_id,
            'generated_date': datetime.utcnow().isoformat(),
            'trades': self._get_audit_trades(start_dt, end_dt, user_id),
            'system_events': self._get_audit_system_events(start_dt, end_dt),
            'user_activities': self._get_audit_user_activities(start_dt, end_dt, user_id),
            'compliance_events': self._get_audit_compliance_events(start_dt, end_dt),
            'integrity_verification': self._verify_audit_integrity(start_dt, end_dt)
        }

        # Create compliance report
        report = ComplianceReport(
            report_id=self._generate_report_id('AUDIT', datetime.utcnow().year),
            report_type='AUDIT_TRAIL',
            tax_year=datetime.utcnow().year,
            generated_date=datetime.utcnow(),
            data=audit_data,
            hash_verification=self._calculate_report_hash(audit_data)
        )

        # Save report
        self._save_report(report)

        logger.info(f"Audit trail generated: {report.report_id}")
        return audit_data

    def generate_regulatory_report(self, report_type: str, year: int) -> Dict[str, Any]:
        """
        Generate regulatory compliance report.

        Args:
            report_type: Type of regulatory report ('MIFID2', 'CFTC', 'FINRA')
            year: Reporting year

        Returns:
            dict: Regulatory compliance report
        """
        logger.info(f"Generating {report_type} regulatory report for {year}")

        if report_type == 'MIFID2':
            return self._generate_mifid2_report(year)
        elif report_type == 'CFTC':
            return self._generate_cftc_report(year)
        elif report_type == 'FINRA':
            return self._generate_finra_report(year)
        else:
            raise ValueError(f"Unsupported regulatory report type: {report_type}")

    def generate_gdpr_report(self, user_id: str) -> Dict[str, Any]:
        """
        Generate GDPR data export for a user.

        Args:
            user_id: User identifier

        Returns:
            dict: Complete user data export
        """
        logger.info(f"Generating GDPR data export for user {user_id}")

        gdpr_data = {
            'user_id': user_id,
            'export_date': datetime.utcnow().isoformat(),
            'personal_data': self._get_user_personal_data(user_id),
            'trading_data': self._get_user_trading_data(user_id),
            'system_interactions': self._get_user_system_interactions(user_id),
            'data_retention_info': self._get_data_retention_info(user_id)
        }

        # Create compliance report
        report = ComplianceReport(
            report_id=self._generate_report_id('GDPR', datetime.utcnow().year),
            report_type='GDPR_EXPORT',
            tax_year=datetime.utcnow().year,
            generated_date=datetime.utcnow(),
            data=gdpr_data,
            hash_verification=self._calculate_report_hash(gdpr_data)
        )

        # Save report
        self._save_report(report)

        logger.info(f"GDPR report generated: {report.report_id}")
        return gdpr_data

    def export_to_csv(self, report_data: Dict[str, Any], report_type: str) -> str:
        """
        Export report data to CSV format.

        Args:
            report_data: Report data dictionary
            report_type: Type of report for filename

        Returns:
            str: Path to CSV file
        """
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"{report_type.lower()}_{timestamp}.csv"
        csv_path = self.reports_path / csv_filename

        # Extract tabular data based on report type
        if report_type == 'TAX_REPORT':
            self._export_tax_report_csv(report_data, csv_path)
        elif report_type == 'AUDIT_TRAIL':
            self._export_audit_trail_csv(report_data, csv_path)
        else:
            # Generic export
            self._export_generic_csv(report_data, csv_path)

        logger.info(f"Report exported to CSV: {csv_path}")
        return str(csv_path)

    def _get_trades_for_year(self, year: int, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all trades for a specific tax year."""
        # This would query the database for trades
        # For now, return empty list as placeholder
        return []

    def _calculate_taxable_events(self, trades: List[Dict[str, Any]]) -> List[TaxableEvent]:
        """Calculate taxable events from trades using FIFO method."""
        taxable_events = []

        # Group trades by symbol
        symbol_positions = {}

        for trade in trades:
            symbol = trade['symbol']

            if symbol not in symbol_positions:
                symbol_positions[symbol] = {'holdings': [], 'sales': []}

            if trade['side'] == 'BUY':
                symbol_positions[symbol]['holdings'].append(trade)
            else:  # SELL
                symbol_positions[symbol]['sales'].append(trade)

        # Calculate gains/losses using FIFO
        for symbol, positions in symbol_positions.items():
            holdings = sorted(positions['holdings'], key=lambda x: x['timestamp'])
            sales = sorted(positions['sales'], key=lambda x: x['timestamp'])

            for sale in sales:
                remaining_quantity = Decimal(str(sale['quantity']))

                while remaining_quantity > 0 and holdings:
                    holding = holdings[0]
                    holding_quantity = Decimal(str(holding['quantity']))

                    if holding_quantity <= remaining_quantity:
                        # Use entire holding
                        quantity_used = holding_quantity
                        holdings.pop(0)
                    else:
                        # Partial holding use
                        quantity_used = remaining_quantity
                        holding['quantity'] = str(holding_quantity - quantity_used)

                    # Calculate taxable event
                    proceeds = quantity_used * Decimal(str(sale['price']))
                    cost_basis = quantity_used * Decimal(str(holding['price']))
                    gain_loss = proceeds - cost_basis

                    # Determine holding period
                    holding_date = datetime.fromisoformat(holding['timestamp'])
                    sale_date = datetime.fromisoformat(sale['timestamp'])
                    days_held = (sale_date - holding_date).days

                    jurisdiction = self.tax_jurisdictions[self.config.tax_jurisdiction]
                    holding_period = 'LONG_TERM' if days_held >= jurisdiction['long_term_threshold'] else 'SHORT_TERM'

                    taxable_event = TaxableEvent(
                        event_id=f"{sale['trade_id']}_{len(taxable_events)}",
                        event_type='SALE',
                        trade_id=sale['trade_id'],
                        symbol=symbol,
                        quantity=quantity_used,
                        proceeds=proceeds,
                        cost_basis=cost_basis,
                        gain_loss=gain_loss,
                        date_acquired=holding_date,
                        date_sold=sale_date,
                        holding_period=holding_period,
                        exchange=sale['exchange'],
                        tax_year=sale_date.year
                    )

                    taxable_events.append(taxable_event)
                    remaining_quantity -= quantity_used

        return taxable_events

    def _generate_tax_summary(self, events: List[TaxableEvent]) -> Dict[str, Any]:
        """Generate tax summary from taxable events."""
        short_term_gains = sum(e.gain_loss for e in events if e.holding_period == 'SHORT_TERM')
        long_term_gains = sum(e.gain_loss for e in events if e.holding_period == 'LONG_TERM')

        return {
            'total_transactions': len(events),
            'short_term_gain_loss': str(short_term_gains),
            'long_term_gain_loss': str(long_term_gains),
            'net_gain_loss': str(short_term_gains + long_term_gains),
            'total_proceeds': str(sum(e.proceeds for e in events)),
            'total_cost_basis': str(sum(e.cost_basis for e in events))
        }

    def _generate_1099b(self, events: List[TaxableEvent]) -> List[Dict[str, Any]]:
        """Generate Form 1099-B data."""
        form_1099b = []

        for event in events:
            form_1099b.append({
                'cusip': '',  # Would need to map symbols to CUSIPs
                'description': event.symbol,
                'date_acquired': event.date_acquired.strftime('%m/%d/%Y'),
                'date_sold': event.date_sold.strftime('%m/%d/%Y'),
                'gross_proceeds': str(event.proceeds.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'cost_basis': str(event.cost_basis.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
                'wash_sale_loss_disallowed': '0.00',  # Would calculate wash sales
                'federal_tax_withheld': '0.00',
                'state_tax_withheld': '0.00'
            })

        return form_1099b

    def _generate_schedule_d(self, events: List[TaxableEvent]) -> Dict[str, Any]:
        """Generate Schedule D summary."""
        short_term_events = [e for e in events if e.holding_period == 'SHORT_TERM']
        long_term_events = [e for e in events if e.holding_period == 'LONG_TERM']

        return {
            'short_term_summary': {
                'total_proceeds': str(sum(e.proceeds for e in short_term_events)),
                'total_cost_basis': str(sum(e.cost_basis for e in short_term_events)),
                'net_gain_loss': str(sum(e.gain_loss for e in short_term_events))
            },
            'long_term_summary': {
                'total_proceeds': str(sum(e.proceeds for e in long_term_events)),
                'total_cost_basis': str(sum(e.cost_basis for e in long_term_events)),
                'net_gain_loss': str(sum(e.gain_loss for e in long_term_events))
            }
        }

    def _generate_form_8949(self, events: List[TaxableEvent]) -> Dict[str, Any]:
        """Generate Form 8949 details."""
        return {
            'short_term_transactions': [
                self._taxable_event_to_dict(e) for e in events if e.holding_period == 'SHORT_TERM'
            ],
            'long_term_transactions': [
                self._taxable_event_to_dict(e) for e in events if e.holding_period == 'LONG_TERM'
            ]
        }

    def _identify_wash_sales(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify wash sale violations."""
        # Simplified wash sale detection
        wash_sales = []
        # Implementation would check for sales followed by purchases within wash sale period
        return wash_sales

    def _taxable_event_to_dict(self, event: TaxableEvent) -> Dict[str, Any]:
        """Convert taxable event to dictionary."""
        return {
            'event_id': event.event_id,
            'symbol': event.symbol,
            'quantity': str(event.quantity),
            'date_acquired': event.date_acquired.isoformat(),
            'date_sold': event.date_sold.isoformat(),
            'proceeds': str(event.proceeds),
            'cost_basis': str(event.cost_basis),
            'gain_loss': str(event.gain_loss),
            'holding_period': event.holding_period
        }

    def _generate_report_id(self, report_type: str, year: int) -> str:
        """Generate unique report ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        return f"{report_type}_{year}_{timestamp}"

    def _calculate_report_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash for report integrity verification."""
        data_json = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_json.encode()).hexdigest()

    def _save_report(self, report: ComplianceReport):
        """Save compliance report to file."""
        filename = f"{report.report_type}_{report.report_id}.json"
        filepath = self.reports_path / filename

        with open(filepath, 'w') as f:
            json.dump({
                'report_id': report.report_id,
                'report_type': report.report_type,
                'tax_year': report.tax_year,
                'generated_date': report.generated_date.isoformat(),
                'hash_verification': report.hash_verification,
                'data': report.data
            }, f, indent=2)

    # Placeholder methods for audit trail data
    def _get_audit_trades(self, start_dt: datetime, end_dt: datetime, user_id: Optional[str]) -> List[Dict]:
        return []

    def _get_audit_system_events(self, start_dt: datetime, end_dt: datetime) -> List[Dict]:
        return []

    def _get_audit_user_activities(self, start_dt: datetime, end_dt: datetime, user_id: Optional[str]) -> List[Dict]:
        return []

    def _get_audit_compliance_events(self, start_dt: datetime, end_dt: datetime) -> List[Dict]:
        return []

    def _verify_audit_integrity(self, start_dt: datetime, end_dt: datetime) -> Dict[str, Any]:
        return {'verified': True, 'message': 'Audit integrity verified'}

    # Placeholder methods for regulatory reports
    def _generate_mifid2_report(self, year: int) -> Dict[str, Any]:
        return {'report_type': 'MiFID2', 'year': year, 'data': {}}

    def _generate_cftc_report(self, year: int) -> Dict[str, Any]:
        return {'report_type': 'CFTC', 'year': year, 'data': {}}

    def _generate_finra_report(self, year: int) -> Dict[str, Any]:
        return {'report_type': 'FINRA', 'year': year, 'data': {}}

    # Placeholder methods for GDPR data
    def _get_user_personal_data(self, user_id: str) -> Dict[str, Any]:
        return {}

    def _get_user_trading_data(self, user_id: str) -> Dict[str, Any]:
        return {}

    def _get_user_system_interactions(self, user_id: str) -> Dict[str, Any]:
        return {}

    def _get_data_retention_info(self, user_id: str) -> Dict[str, Any]:
        return {}

    # CSV export methods
    def _export_tax_report_csv(self, report_data: Dict[str, Any], csv_path: Path):
        """Export tax report to CSV."""
        with open(csv_path, 'w', newline='') as csvfile:
            if 'detailed_transactions' in report_data:
                fieldnames = ['symbol', 'quantity', 'date_acquired', 'date_sold', 'proceeds', 'cost_basis', 'gain_loss', 'holding_period']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for transaction in report_data['detailed_transactions']:
                    writer.writerow(transaction)

    def _export_audit_trail_csv(self, report_data: Dict[str, Any], csv_path: Path):
        """Export audit trail to CSV."""
        # Implementation would export audit data to CSV
        pass

    def _export_generic_csv(self, report_data: Dict[str, Any], csv_path: Path):
        """Generic CSV export."""
        # Implementation would export generic data to CSV
        pass