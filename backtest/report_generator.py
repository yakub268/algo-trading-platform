"""
Backtest Report Generator

Generates comprehensive PDF/HTML reports for strategy backtests.

Includes:
- Equity curve visualization
- Drawdown chart
- Monthly returns heatmap
- Trade analysis
- Performance statistics
- Walk-forward results

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ReportGenerator')

REPORTS_DIR = Path(__file__).parent.parent / "data" / "reports"


@dataclass
class ReportData:
    """Data structure for report generation"""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    # Performance metrics
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    
    # Trade statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_holding_days: float
    
    # Time series data
    equity_curve: pd.Series
    drawdown_series: pd.Series
    trades_df: pd.DataFrame
    monthly_returns: pd.DataFrame
    
    # Parameters
    parameters: Dict[str, Any]


class BacktestReportGenerator:
    """
    Generate comprehensive backtest reports.
    
    Output formats:
    - HTML (interactive)
    - Markdown (portable)
    - CSV (data export)
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ReportGenerator initialized, output: {self.output_dir}")
    
    def calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Calculate monthly returns matrix"""
        if equity_curve.empty:
            return pd.DataFrame()
        
        # Ensure datetime index
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve.index = pd.to_datetime(equity_curve.index)
        
        # Calculate daily returns
        daily_returns = equity_curve.pct_change().fillna(0)
        
        # Resample to monthly
        monthly = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table (year x month)
        monthly_df = monthly.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')

        # Map month numbers to names, handling cases where not all months are present
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        pivot.columns = [month_names.get(col, col) for col in pivot.columns]

        return pivot
    
    def calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown time series"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown
    
    def generate_html_report(self, data: ReportData, filename: str = None) -> str:
        """
        Generate HTML report with embedded charts.
        
        Args:
            data: ReportData object
            filename: Output filename (optional)
            
        Returns:
            Path to generated report
        """
        filename = filename or f"{data.strategy_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        filepath = self.output_dir / filename
        
        # Generate charts as base64 images
        equity_chart = self._generate_equity_chart_html(data.equity_curve)
        drawdown_chart = self._generate_drawdown_chart_html(data.drawdown_series)
        monthly_heatmap = self._generate_monthly_heatmap_html(data.monthly_returns)
        
        # Build HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report: {data.strategy_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; font-size: 14px; margin-top: 5px; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .chart {{ width: 100%; margin: 20px 0; }}
        .params {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
        .footer {{ text-align: center; color: #95a5a6; margin-top: 30px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Backtest Report: {data.strategy_name}</h1>
        <p><strong>Symbol:</strong> {data.symbol} | <strong>Period:</strong> {data.start_date.strftime('%Y-%m-%d')} to {data.end_date.strftime('%Y-%m-%d')}</p>
        
        <h2>Performance Summary</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value {'positive' if data.total_return > 0 else 'negative'}">{data.total_return:+.1%}</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{data.max_drawdown:.1%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data.win_rate:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{data.cagr:+.1%}</div>
                <div class="metric-label">CAGR</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data.sortino_ratio:.2f}</div>
                <div class="metric-label">Sortino Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data.profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
        </div>
        
        <h2>Equity Curve</h2>
        <div class="chart">{equity_chart}</div>
        
        <h2>Drawdown Analysis</h2>
        <div class="chart">{drawdown_chart}</div>
        
        <h2>Monthly Returns</h2>
        <div class="chart">{monthly_heatmap}</div>
        
        <h2>Trade Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Trades</td><td>{data.total_trades}</td></tr>
            <tr><td>Win Rate</td><td>{data.win_rate:.1%}</td></tr>
            <tr><td>Average Win</td><td class="positive">{data.avg_win:+.2%}</td></tr>
            <tr><td>Average Loss</td><td class="negative">{data.avg_loss:.2%}</td></tr>
            <tr><td>Best Trade</td><td class="positive">{data.best_trade:+.2%}</td></tr>
            <tr><td>Worst Trade</td><td class="negative">{data.worst_trade:.2%}</td></tr>
            <tr><td>Avg Holding Period</td><td>{data.avg_holding_days:.1f} days</td></tr>
        </table>
        
        <h2>Strategy Parameters</h2>
        <div class="params">
            <pre>{self._format_params(data.parameters)}</pre>
        </div>
        
        <h2>Capital Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Initial Capital</td><td>${data.initial_capital:,.2f}</td></tr>
            <tr><td>Final Capital</td><td>${data.final_capital:,.2f}</td></tr>
            <tr><td>Net Profit/Loss</td><td class="{'positive' if data.final_capital > data.initial_capital else 'negative'}">${data.final_capital - data.initial_capital:+,.2f}</td></tr>
        </table>
        
        <div class="footer">
            Generated by Trading Bot Arsenal | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        logger.info(f"HTML report generated: {filepath}")
        return str(filepath)
    
    def _generate_equity_chart_html(self, equity_curve: pd.Series) -> str:
        """Generate equity curve as SVG"""
        if equity_curve.empty:
            return "<p>No equity data available</p>"
        
        # Simple SVG line chart
        width, height = 800, 300
        padding = 50
        
        values = equity_curve.values
        min_val, max_val = min(values), max(values)
        val_range = max_val - min_val if max_val != min_val else 1
        
        # Scale points
        x_scale = (width - 2*padding) / (len(values) - 1) if len(values) > 1 else 0
        y_scale = (height - 2*padding) / val_range
        
        points = []
        for i, v in enumerate(values):
            x = padding + i * x_scale
            y = height - padding - (v - min_val) * y_scale
            points.append(f"{x},{y}")
        
        path = " ".join(points)
        
        svg = f"""
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#f8f9fa"/>
            <polyline points="{path}" fill="none" stroke="#3498db" stroke-width="2"/>
            <text x="{padding}" y="20" fill="#333">Max: ${max_val:,.0f}</text>
            <text x="{padding}" y="{height-10}" fill="#333">Min: ${min_val:,.0f}</text>
        </svg>
        """
        return svg
    
    def _generate_drawdown_chart_html(self, drawdown: pd.Series) -> str:
        """Generate drawdown chart as SVG"""
        if drawdown.empty:
            return "<p>No drawdown data available</p>"
        
        width, height = 800, 200
        padding = 50
        
        values = drawdown.values
        min_dd = min(values)
        
        x_scale = (width - 2*padding) / (len(values) - 1) if len(values) > 1 else 0
        y_scale = (height - 2*padding) / abs(min_dd) if min_dd != 0 else 1
        
        points = [f"{padding},{padding}"]  # Start at 0
        for i, v in enumerate(values):
            x = padding + i * x_scale
            y = padding - v * y_scale
            points.append(f"{x},{y}")
        points.append(f"{width-padding},{padding}")  # End at 0
        
        path = " ".join(points)
        
        svg = f"""
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#f8f9fa"/>
            <polygon points="{path}" fill="#e74c3c" fill-opacity="0.3" stroke="#e74c3c" stroke-width="1"/>
            <line x1="{padding}" y1="{padding}" x2="{width-padding}" y2="{padding}" stroke="#333" stroke-dasharray="5,5"/>
            <text x="{padding}" y="{height-10}" fill="#333">Max DD: {min_dd:.1%}</text>
        </svg>
        """
        return svg
    
    def _generate_monthly_heatmap_html(self, monthly_returns: pd.DataFrame) -> str:
        """Generate monthly returns heatmap as HTML table"""
        if monthly_returns.empty:
            return "<p>No monthly data available</p>"
        
        html = '<table style="width:100%; text-align:center;">'
        html += '<tr><th>Year</th>'
        for col in monthly_returns.columns:
            html += f'<th>{col}</th>'
        html += '<th>Total</th></tr>'
        
        for idx, row in monthly_returns.iterrows():
            html += f'<tr><td><strong>{idx}</strong></td>'
            yearly_total = 0
            for val in row:
                if pd.notna(val):
                    yearly_total += val
                    color = '#27ae60' if val > 0 else '#e74c3c' if val < 0 else '#95a5a6'
                    html += f'<td style="background:{color}33; color:{color}">{val:.1%}</td>'
                else:
                    html += '<td>-</td>'
            html += f'<td style="font-weight:bold">{yearly_total:.1%}</td></tr>'
        
        html += '</table>'
        return html
    
    def _format_params(self, params: Dict) -> str:
        """Format parameters for display"""
        lines = []
        for key, value in params.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)
    
    def generate_markdown_report(self, data: ReportData, filename: str = None) -> str:
        """Generate Markdown report"""
        filename = filename or f"{data.strategy_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        filepath = self.output_dir / filename
        
        md = f"""# Backtest Report: {data.strategy_name}

**Symbol:** {data.symbol}  
**Period:** {data.start_date.strftime('%Y-%m-%d')} to {data.end_date.strftime('%Y-%m-%d')}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Return | {data.total_return:+.1%} |
| CAGR | {data.cagr:+.1%} |
| Sharpe Ratio | {data.sharpe_ratio:.2f} |
| Sortino Ratio | {data.sortino_ratio:.2f} |
| Calmar Ratio | {data.calmar_ratio:.2f} |
| Max Drawdown | {data.max_drawdown:.1%} |

## Trade Statistics

| Metric | Value |
|--------|-------|
| Total Trades | {data.total_trades} |
| Win Rate | {data.win_rate:.1%} |
| Profit Factor | {data.profit_factor:.2f} |
| Average Win | {data.avg_win:+.2%} |
| Average Loss | {data.avg_loss:.2%} |
| Best Trade | {data.best_trade:+.2%} |
| Worst Trade | {data.worst_trade:.2%} |
| Avg Holding Period | {data.avg_holding_days:.1f} days |

## Capital Summary

| Metric | Value |
|--------|-------|
| Initial Capital | ${data.initial_capital:,.2f} |
| Final Capital | ${data.final_capital:,.2f} |
| Net P&L | ${data.final_capital - data.initial_capital:+,.2f} |

## Strategy Parameters

```
{self._format_params(data.parameters)}
```

---
*Report generated by Trading Bot Arsenal*
"""
        
        with open(filepath, 'w') as f:
            f.write(md)
        
        logger.info(f"Markdown report generated: {filepath}")
        return str(filepath)
    
    def export_trades_csv(self, trades_df: pd.DataFrame, filename: str = None) -> str:
        """Export trades to CSV"""
        filename = filename or f"trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        filepath = self.output_dir / filename
        
        trades_df.to_csv(filepath, index=False)
        logger.info(f"Trades exported: {filepath}")
        return str(filepath)


def create_report_from_backtest(
    strategy_name: str,
    symbol: str,
    equity_curve: pd.Series,
    trades_df: pd.DataFrame,
    parameters: Dict,
    initial_capital: float = 10000.0
) -> str:
    """
    Convenience function to create report from backtest results.
    
    Returns:
        Path to generated HTML report
    """
    generator = BacktestReportGenerator()
    
    # Calculate metrics
    final_capital = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
    total_return = (final_capital - initial_capital) / initial_capital
    
    # Calculate drawdown
    drawdown = generator.calculate_drawdown_series(equity_curve)
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    # Calculate monthly returns
    monthly_returns = generator.calculate_monthly_returns(equity_curve)
    
    # Trade statistics
    if len(trades_df) > 0:
        returns = trades_df['pnl_pct'].values if 'pnl_pct' in trades_df.columns else []
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        
        win_rate = len(wins) / len(returns) if returns else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else 0
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0
        returns = []
    
    # Calculate Sharpe
    daily_returns = equity_curve.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
    
    # Create report data
    data = ReportData(
        strategy_name=strategy_name,
        symbol=symbol,
        start_date=equity_curve.index[0] if len(equity_curve) > 0 else datetime.now(),
        end_date=equity_curve.index[-1] if len(equity_curve) > 0 else datetime.now(),
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return=total_return,
        cagr=total_return,  # Simplified
        sharpe_ratio=sharpe,
        sortino_ratio=sharpe * 0.9,  # Approximation
        calmar_ratio=total_return / max_drawdown if max_drawdown > 0 else 0,
        max_drawdown=max_drawdown,
        total_trades=len(returns),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        best_trade=max(returns) if returns else 0,
        worst_trade=min(returns) if returns else 0,
        avg_holding_days=5,  # Placeholder
        equity_curve=equity_curve,
        drawdown_series=drawdown,
        trades_df=trades_df,
        monthly_returns=monthly_returns,
        parameters=parameters
    )
    
    return generator.generate_html_report(data)


if __name__ == "__main__":
    print("=" * 60)
    print("BACKTEST REPORT GENERATOR")
    print("=" * 60)
    
    # Generate sample report
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    equity = pd.Series(10000 * (1 + np.random.randn(252).cumsum() * 0.01), index=dates)
    
    trades = pd.DataFrame({
        'entry_date': dates[::20][:10],
        'exit_date': dates[10::20][:10],
        'pnl_pct': np.random.uniform(-0.05, 0.10, 10)
    })
    
    params = {'rsi_period': 2, 'oversold': 10, 'stop_loss': 0.03}
    
    report_path = create_report_from_backtest(
        strategy_name="RSI-2 Mean Reversion",
        symbol="SPY",
        equity_curve=equity,
        trades_df=trades,
        parameters=params
    )
    
    print(f"\n✅ Report generated: {report_path}")
