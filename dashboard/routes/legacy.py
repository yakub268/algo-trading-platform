"""
Legacy API endpoints - /api/opportunities, /api/category-stats, /api/paper-trades,
/api/paper-trade, /api/scraper-status, /api/live-trades, /api/combined/summary,
/api/system-status
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from flask import Blueprint, jsonify, request

from dashboard.shared import (
    get_db_connection, CATEGORY_COLORS, extract_ticker_date,
    get_cached_market_scan, rate_limit_check, alpaca_client,
)

legacy_bp = Blueprint('legacy', __name__)


@legacy_bp.route('/api/opportunities')
def get_opportunities():
    """Get current opportunities from multi-market scan"""
    if rate_limit_check('opportunities'):
        return jsonify({'success': False, 'error': 'Rate limited', 'opportunities': []})
    try:
        opportunities = get_cached_market_scan()

        # Get today's date for filtering
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        # Convert to JSON-serializable format, filtering out past markets
        opps_data = []
        for opp in opportunities:
            # Try to extract date from ticker and filter out past markets
            ticker_date = extract_ticker_date(opp.ticker)
            if ticker_date and ticker_date < today:
                continue  # Skip past-dated markets

            opps_data.append({
                'ticker': opp.ticker,
                'title': opp.title,
                'category': opp.category,
                'side': opp.side,
                'our_probability': round(opp.our_probability * 100, 1),
                'market_price': round(opp.market_price * 100, 1),
                'edge': round(opp.edge * 100, 1),
                'data_source': opp.data_source,
                'overall_score': round(opp.overall_score, 3),
                'reasoning': opp.reasoning,
                'color': CATEGORY_COLORS.get(opp.category, '#95a5a6')
            })

        return jsonify({
            'success': True,
            'opportunities': opps_data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'count': len(opps_data)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'opportunities': []
        })


@legacy_bp.route('/api/category-stats')
def get_category_stats():
    """Get opportunity breakdown by category"""
    try:
        opportunities = get_cached_market_scan()

        # Get today's date for filtering
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        # Initialize all categories (even if 0 opportunities)
        all_categories = ['weather', 'fed', 'crypto', 'earnings', 'economic', 'sports', 'boxoffice']
        category_counts = {cat: 0 for cat in all_categories}
        category_edges = {cat: [] for cat in all_categories}

        filtered_count = 0
        for opp in opportunities:
            # Filter out past-dated markets
            ticker_date = extract_ticker_date(opp.ticker)
            if ticker_date and ticker_date < today:
                continue

            filtered_count += 1
            cat = opp.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
            if cat not in category_edges:
                category_edges[cat] = []
            category_edges[cat].append(opp.edge)

        # Calculate averages - include all categories
        stats = []
        for cat in all_categories:
            count = category_counts.get(cat, 0)
            edges = category_edges.get(cat, [])
            avg_edge = sum(edges) / len(edges) if edges else 0
            stats.append({
                'category': cat,
                'count': count,
                'avg_edge': round(avg_edge * 100, 1),
                'color': CATEGORY_COLORS.get(cat, '#95a5a6')
            })

        return jsonify({
            'success': True,
            'stats': stats,
            'total': filtered_count
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'stats': []
        })


@legacy_bp.route('/api/paper-trades')
def get_paper_trades():
    """Get paper trade history with P&L"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM paper_trades
            ORDER BY timestamp DESC
            LIMIT 100
        ''')
        trades = cursor.fetchall()

        # Calculate running P&L
        cursor.execute('''
            SELECT
                SUM(CASE WHEN status = 'closed' THEN pnl ELSE 0 END) as realized_pnl,
                SUM(CASE WHEN status = 'open' THEN (edge * contracts * entry_price) ELSE 0 END) as unrealized_pnl,
                COUNT(*) as total_trades,
                SUM(CASE WHEN status = 'closed' AND pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) as closed_trades
            FROM paper_trades
        ''')
        summary = cursor.fetchone()
        conn.close()

        trades_data = []
        for trade in trades:
            trades_data.append({
                'id': trade['id'],
                'timestamp': trade['timestamp'],
                'ticker': trade['ticker'],
                'category': trade['category'],
                'side': trade['side'],
                'entry_price': trade['entry_price'],
                'contracts': trade['contracts'],
                'edge': round(trade['edge'] * 100, 1),
                'status': trade['status'],
                'exit_price': trade['exit_price'],
                'pnl': trade['pnl'],
                'color': CATEGORY_COLORS.get(trade['category'], '#95a5a6')
            })

        win_rate = 0
        if summary['closed_trades'] and summary['closed_trades'] > 0:
            win_rate = round(summary['winning_trades'] / summary['closed_trades'] * 100, 1)

        return jsonify({
            'success': True,
            'trades': trades_data,
            'summary': {
                'realized_pnl': round(summary['realized_pnl'] or 0, 2),
                'unrealized_pnl': round(summary['unrealized_pnl'] or 0, 2),
                'total_trades': summary['total_trades'] or 0,
                'win_rate': win_rate
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'trades': [],
            'summary': {}
        })


@legacy_bp.route('/api/live-trades')
def get_live_trades():
    """Get live trades from orchestrator's trades table"""
    try:
        # Always use live DB (orchestrator writes here)
        db_path = Path(__file__).parent.parent.parent / "data" / "live" / "trading_master.db"

        if not db_path.exists():
            return jsonify({'success': True, 'trades': [], 'summary': {'total': 0}})

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT trade_id, bot_name, market, symbol, side, entry_price,
                   exit_price, quantity, entry_time, exit_time, pnl, status
            FROM trades
            ORDER BY entry_time DESC
            LIMIT 100
        ''')
        trades = cursor.fetchall()

        trades_data = []
        for t in trades:
            trades_data.append({
                'id': t['trade_id'],
                'bot_name': t['bot_name'],
                'market': t['market'],
                'symbol': t['symbol'][:50] if t['symbol'] else '',
                'side': t['side'],
                'entry_price': t['entry_price'],
                'exit_price': t['exit_price'],
                'quantity': t['quantity'],
                'entry_time': t['entry_time'],
                'exit_time': t['exit_time'],
                'pnl': t['pnl'],
                'status': t['status']
            })

        conn.close()

        return jsonify({
            'success': True,
            'trades': trades_data,
            'summary': {
                'total': len(trades_data),
                'open': sum(1 for t in trades_data if t['status'] == 'open'),
                'closed': sum(1 for t in trades_data if t['status'] == 'closed')
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trades': []})


@legacy_bp.route('/api/paper-trade', methods=['POST'])
def create_paper_trade():
    """Create a new paper trade from an opportunity"""
    try:
        data = request.json
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO paper_trades (timestamp, ticker, category, side, entry_price, contracts, edge, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'open')
        ''', (
            datetime.now(timezone.utc).isoformat(),
            data['ticker'],
            data['category'],
            data['side'],
            data['entry_price'] / 100,  # Convert from cents
            data.get('contracts', 10),
            data['edge'] / 100  # Convert from percentage
        ))

        conn.commit()
        trade_id = cursor.lastrowid
        conn.close()

        return jsonify({
            'success': True,
            'trade_id': trade_id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@legacy_bp.route('/api/scraper-status')
def get_scraper_status():
    """Get status of all scrapers"""
    try:
        from scrapers.data_aggregator import DataAggregator
        aggregator = DataAggregator()
        summary = aggregator.get_summary()

        scrapers = [
            {
                'name': 'Weather (NWS)',
                'status': summary['weather']['status'],
                'records': summary['weather']['estimates'],
                'last_run': summary['timestamp'],
                'color': '#3498db'
            },
            {
                'name': 'Economic (FRED)',
                'status': summary['economic']['status'],
                'records': summary['economic']['estimates'],
                'last_run': summary['timestamp'],
                'color': '#1abc9c'
            },
            {
                'name': 'Crypto (CoinGecko)',
                'status': summary['crypto']['status'],
                'records': summary['crypto']['estimates'],
                'last_run': summary['timestamp'],
                'color': '#f39c12'
            },
            {
                'name': 'Earnings (Yahoo)',
                'status': summary['earnings']['status'],
                'records': summary['earnings']['estimates'],
                'last_run': summary['timestamp'],
                'color': '#9b59b6'
            },
            {
                'name': 'Sports (ESPN)',
                'status': summary['sports']['status'],
                'records': summary['sports']['estimates'],
                'last_run': summary['timestamp'],
                'color': '#2ecc71'
            },
            {
                'name': 'Box Office (BOM)',
                'status': summary['boxoffice']['status'],
                'records': summary['boxoffice']['estimates'],
                'last_run': summary['timestamp'],
                'color': '#e91e63'
            }
        ]

        return jsonify({
            'success': True,
            'scrapers': scrapers,
            'total_estimates': summary['total_estimates']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'scrapers': []
        })


@legacy_bp.route('/api/combined/summary')
def get_combined_summary():
    """Get combined summary across all LIVE trading systems (excludes paper trades)"""
    try:
        # Kalshi LIVE P&L - paper trades excluded
        # Would connect to Kalshi API for live data when integrated
        kalshi_pnl = 0
        kalshi_trades = 0
        kalshi_win_rate = 0

        # Alpaca P&L
        alpaca_pnl = 0
        alpaca_equity = 0
        alpaca_connected = alpaca_client.is_connected()
        if alpaca_connected:
            positions = alpaca_client.get_positions()
            for pos in positions:
                alpaca_pnl += float(pos.get('unrealized_pl', 0))
            account = alpaca_client.get_account()
            if account:
                alpaca_equity = float(account.get('equity', 0))

        # Combined metrics
        total_pnl = kalshi_pnl + alpaca_pnl
        total_trades = kalshi_trades

        # Determine best system
        systems_pnl = {
            'kalshi': kalshi_pnl,
            'alpaca': alpaca_pnl
        }
        best_system = max(systems_pnl, key=systems_pnl.get)

        return jsonify({
            'success': True,
            'kalshi': {
                'pnl': round(kalshi_pnl, 2),
                'trades': kalshi_trades,
                'win_rate': kalshi_win_rate,
                'status': 'paper'
            },
            'alpaca': {
                'connected': alpaca_connected,
                'pnl': round(alpaca_pnl, 2),
                'equity': round(alpaca_equity, 2),
                'paper_mode': alpaca_client.paper
            },
            'combined': {
                'total_pnl': round(total_pnl, 2),
                'total_trades': total_trades,
                'best_system': best_system
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@legacy_bp.route('/api/system-status')
def get_system_status():
    """Get status of all trading systems"""
    try:
        # Check each system
        kalshi_ok = True  # Kalshi client always works in paper mode

        alpaca_connected = alpaca_client.is_connected()

        # Get FOMC status
        fomc = alpaca_client.get_fomc_status()

        return jsonify({
            'success': True,
            'systems': {
                'kalshi': {
                    'status': 'online',
                    'mode': 'paper',
                    'color': '#3498db'
                },
                'alpaca': {
                    'status': 'online' if alpaca_connected else 'offline',
                    'mode': 'paper' if alpaca_client.paper else 'live',
                    'color': '#2ecc71'
                }
            },
            'fomc': fomc,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
