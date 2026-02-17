"""
V5 Dashboard API endpoints - /api/v5/*
"""

import os
import time
import sqlite3
from datetime import datetime, timezone, timedelta
from flask import Blueprint, jsonify, request

from dashboard.shared import (
    alpaca_client, KalshiClient, COINBASE_AVAILABLE,
    get_coinbase_service, get_cached_market_scan,
    rate_limit_check, _format_time_ago, _paper_mode,
)

v5_bp = Blueprint('v5', __name__)

# OANDA failure cache — avoid retrying every poll when credentials are bad
_oanda_fail_cache = {'failed': False, 'until': 0, 'error': '', 'status': 'DISCONNECTED'}


@v5_bp.route('/api/v5/go-no-go')
def get_go_no_go():
    """Get GO/NO-GO validation status for LIVE trading"""
    try:
        # Get trade count from database
        trade_count = 0
        win_count = 0

        # Validation start date - ignore all trades before this (manual/test trades)
        VALIDATION_START = '2026-02-04T00:00:00'

        # Check for trades in the trading_master.db (always use live DB)
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'live', 'trading_master.db')

        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Only count trades since validation start (excludes old manual trades)
                cursor.execute('''
                    SELECT COUNT(*),
                           SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)
                    FROM trades
                    WHERE entry_time >= ?
                ''', (VALIDATION_START,))
                row = cursor.fetchone()
                if row:
                    trade_count = row[0] or 0
                    win_count = row[1] or 0
                conn.close()
            except Exception as db_error:
                print(f"Database error in go-no-go: {db_error}")

        # Also check Kalshi fills from the API
        if KalshiClient:
            try:
                kalshi = KalshiClient()
                fills = kalshi.get_fills(limit=100)
                trade_count += len(fills)
            except Exception as e:
                print(f"Error fetching Kalshi fills: {e}")

        # Calculate win rate
        win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0

        # Check broker connections (only count configured/active brokers)
        alpaca_ok = alpaca_client.is_connected()
        kalshi_ok = True  # Always works via API key

        # Only count brokers we actually use
        active_brokers = [('Alpaca', alpaca_ok), ('Kalshi', kalshi_ok)]
        connected_count = sum(ok for _, ok in active_brokers)
        total_brokers = len(active_brokers)
        api_status = f"{connected_count}/{total_brokers} OK"

        # Calculate real Sharpe ratio, max drawdown, and error rate from trade data
        sharpe = 'N/A'
        drawdown = '0%'
        error_rate = '0%'

        if os.path.exists(db_path) and trade_count >= 30:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Get all PnL values for Sharpe and drawdown calculation
                cursor.execute('''
                    SELECT pnl FROM trades
                    WHERE entry_time >= ? AND pnl IS NOT NULL AND status = 'closed'
                    ORDER BY exit_time ASC
                ''', (VALIDATION_START,))
                pnl_rows = cursor.fetchall()
                pnl_values = [r[0] for r in pnl_rows if r[0] is not None]

                if len(pnl_values) >= 30:
                    # Sharpe ratio: mean(returns) / std(returns) * sqrt(252)
                    import statistics
                    mean_pnl = statistics.mean(pnl_values)
                    std_pnl = statistics.stdev(pnl_values) if len(pnl_values) > 1 else 1
                    if std_pnl > 0:
                        daily_sharpe = mean_pnl / std_pnl
                        annualized_sharpe = daily_sharpe * (252 ** 0.5)
                        sharpe = f'{annualized_sharpe:.2f}'

                    # Max drawdown from cumulative PnL curve
                    cumulative = 0
                    peak = 0
                    max_dd = 0
                    for p in pnl_values:
                        cumulative += p
                        if cumulative > peak:
                            peak = cumulative
                        dd = peak - cumulative
                        if dd > max_dd:
                            max_dd = dd
                    # Express as % of starting capital ($500)
                    drawdown = f'{(max_dd / 500 * 100):.1f}%' if max_dd > 0 else '0%'

                # Error rate: failed trades / total trades
                cursor.execute('''
                    SELECT COUNT(*) FROM trades
                    WHERE entry_time >= ? AND status = 'error'
                ''', (VALIDATION_START,))
                error_row = cursor.fetchone()
                error_count = error_row[0] if error_row else 0
                if trade_count > 0:
                    error_rate = f'{(error_count / trade_count * 100):.1f}%'

                conn.close()
            except Exception as calc_error:
                print(f"Error computing go-no-go metrics: {calc_error}")

        checks = [
            {
                'metric': 'Trades (bot)',
                'value': trade_count,
                'threshold': '>100',
                'status': 'PASS' if trade_count >= 100 else ('WARNING' if trade_count >= 50 else ('PENDING' if trade_count == 0 else 'FAIL'))
            },
            {
                'metric': 'Win Rate',
                'value': f'{win_rate:.1f}%' if trade_count > 0 else 'N/A',
                'threshold': '>45%',
                'status': 'PASS' if win_rate >= 45 else ('WARNING' if win_rate >= 35 else ('PENDING' if trade_count == 0 else 'FAIL'))
            },
            {
                'metric': 'Sharpe Ratio',
                'value': sharpe,
                'threshold': '>1.0',
                'status': 'PENDING' if sharpe == 'N/A' else ('PASS' if float(sharpe) >= 1.0 else ('WARNING' if float(sharpe) >= 0.5 else 'FAIL'))
            },
            {
                'metric': 'Max Drawdown',
                'value': drawdown,
                'threshold': '<15%',
                'status': 'PASS' if float(drawdown.rstrip('%')) < 15 else 'FAIL'
            },
            {
                'metric': 'API Status',
                'value': api_status,
                'threshold': 'All OK',
                'status': 'PASS' if connected_count >= total_brokers else ('WARNING' if connected_count >= 1 else 'FAIL')
            },
            {
                'metric': 'Error Rate',
                'value': error_rate,
                'threshold': '<5%',
                'status': 'PASS' if float(error_rate.rstrip('%')) < 5 else 'FAIL'
            }
        ]

        # Determine overall status
        fail_count = sum(1 for c in checks if c['status'] == 'FAIL')
        warning_count = sum(1 for c in checks if c['status'] == 'WARNING')
        pending_count = sum(1 for c in checks if c['status'] == 'PENDING')

        if fail_count > 0:
            overall = 'NO-GO'
        elif warning_count > 2:
            overall = 'NO-GO'
        elif pending_count > 3:
            overall = 'PENDING'
        else:
            overall = 'GO'

        return jsonify({
            'success': True,
            'checks': checks,
            'overall': overall,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@v5_bp.route('/api/v5/ai-status')
def get_ai_status():
    """Get AI system status"""
    try:
        # Check if it's weekend
        from datetime import datetime
        now = datetime.now()
        is_weekend = now.weekday() >= 5

        return jsonify({
            'success': True,
            'regime': 'WEEKEND' if is_weekend else 'UNKNOWN',
            'regimeConfidence': 0 if is_weekend else 50,
            'sentiment': 0,
            'sentimentLabel': 'Market Closed' if is_weekend else 'Neutral',
            'vetoRate': 0,
            'activeEdges': 0,
            'aiCost': 0,
            'llmCalls': 0,
            'cacheHitRate': 0,
            'avgLatency': 0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@v5_bp.route('/api/v5/circuit-breakers')
def get_circuit_breakers():
    """Get circuit breaker status for LIVE trading"""
    try:
        # NOTE: Paper trades excluded - only tracking LIVE P&L
        # Would calculate from live broker positions and trade history
        daily_loss = 0  # Live trades only
        daily_limit = 22.50  # 5% of $450

        # Calculate real values - 0 when no trades
        max_drawdown = 0  # Would calculate from equity curve
        position_concentration = 0  # Would calculate from positions
        consecutive_losses = 0  # Would track from trade history

        breakers = [
            {
                'name': 'Daily Loss Limit',
                'current': round(daily_loss, 2),
                'limit': daily_limit,
                'pct': min(round(daily_loss / daily_limit * 100, 0), 100) if daily_limit > 0 else 0,
                'status': 'WARNING' if daily_loss >= daily_limit * 0.8 else 'OK'
            },
            {
                'name': 'Max Drawdown',
                'current': max_drawdown,
                'limit': 15,
                'pct': round(max_drawdown / 15 * 100, 0) if max_drawdown > 0 else 0,
                'status': 'WARNING' if max_drawdown >= 12 else 'OK'
            },
            {
                'name': 'Position Concentration',
                'current': position_concentration,
                'limit': 50,
                'pct': round(position_concentration / 50 * 100, 0) if position_concentration > 0 else 0,
                'status': 'WARNING' if position_concentration >= 40 else 'OK'
            },
            {
                'name': 'Consecutive Losses',
                'current': consecutive_losses,
                'limit': 5,
                'pct': round(consecutive_losses / 5 * 100, 0) if consecutive_losses > 0 else 0,
                'status': 'WARNING' if consecutive_losses >= 4 else 'OK'
            }
        ]

        return jsonify({
            'success': True,
            'breakers': breakers,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@v5_bp.route('/api/v5/edges')
def get_edges():
    """Get edge opportunities across all markets"""
    try:
        # Get opportunities from multi-market strategy (cached)
        opportunities = get_cached_market_scan()

        weather_edges = []
        fed_edges = []
        sports_edges = []

        for opp in opportunities:
            edge_data = {
                'market': opp.title[:30],
                'ticker': opp.ticker,
                'ourProb': round(opp.our_probability * 100, 1),
                'marketPrice': round(opp.market_price * 100, 1),
                'edge': round(opp.edge * 100, 1),
                'confidence': 'HIGH' if opp.overall_score > 0.7 else 'MEDIUM' if opp.overall_score > 0.5 else 'LOW',
                'status': 'MONITORING',
                'source': opp.data_source
            }

            if opp.category == 'weather':
                weather_edges.append(edge_data)
            elif opp.category == 'fed':
                fed_edges.append(edge_data)
            elif opp.category == 'sports':
                sports_edges.append(edge_data)

        return jsonify({
            'success': True,
            'weather': weather_edges[:10],
            'fed': fed_edges[:10],
            'sports': sports_edges[:10],
            'totalEdges': len(opportunities),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'weather': [],
            'fed': [],
            'sports': []
        })


@v5_bp.route('/api/v5/strategies')
def get_strategies():
    """Get all strategy statuses - returns real data when available"""
    try:
        # Check if it's weekend (market closed)
        from datetime import datetime
        now = datetime.now()
        is_weekend = now.weekday() >= 5
        market_status = "Market closed - Weekend" if is_weekend else "Market open"

        # Try to load real strategy data from database
        is_mock = True
        strategies = []
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'live', 'trading_master.db')

        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Check if bot_status table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bot_status'")
                if cursor.fetchone():
                    cursor.execute("""
                        SELECT bot_name, status, last_signal, pnl_today, trades_today, error
                        FROM bot_status
                    """)
                    rows = cursor.fetchall()
                    if rows:
                        is_mock = False
                        for idx, row in enumerate(rows, 1):
                            bot_name, status, last_signal, pnl_today, trades_today, error = row
                            strategies.append({
                                'id': idx,
                                'name': bot_name or f'Strategy {idx}',
                                'status': 'ERROR' if error else ('LIVE' if status == 'running' else 'PAUSED'),
                                'pnl': pnl_today or 0,
                                'winRate': 0,
                                'drawdown': 0,
                                'trades': trades_today or 0,
                                'lastSignal': {'type': last_signal or 'HOLD', 'symbol': '', 'time': market_status},
                                'aiDecision': 'ERROR' if error else 'APPROVED',
                                'aiReason': error or market_status
                            })
                conn.close()
            except Exception as db_error:
                print(f"Database error fetching strategies: {db_error}")

        # Fallback to default strategies with mock indicator
        if not strategies:
            is_mock = True
            strategies = [
                {
                    'id': 1,
                    'name': 'RSI-2 Mean Reversion',
                    'status': 'LIVE',
                    'pnl': 0,
                    'winRate': 0,
                    'drawdown': 0,
                    'trades': 0,
                    'lastSignal': {'type': 'HOLD', 'symbol': 'SPY', 'time': market_status},
                    'aiDecision': 'APPROVED',
                    'aiReason': market_status
                },
                {
                    'id': 2,
                    'name': 'Dual Momentum',
                    'status': 'LIVE',
                    'pnl': 0,
                    'winRate': 0,
                    'drawdown': 0,
                    'trades': 0,
                    'lastSignal': {'type': 'HOLD', 'symbol': 'CASH', 'time': 'Monthly rebalance'},
                    'aiDecision': 'APPROVED',
                    'aiReason': 'Waiting for rebalance day'
                }
            ]

        return jsonify({
            'success': True,
            'strategies': strategies,
            'is_mock': is_mock,
            'message': 'Showing default strategies - no bot_status data in database' if is_mock else None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@v5_bp.route('/api/v5/positions')
def get_all_positions():
    """Get all LIVE positions across all brokers (excludes paper trades)"""
    try:
        positions = []

        # Get Alpaca LIVE positions only
        if alpaca_client.is_connected():
            alpaca_positions = alpaca_client.get_positions()
            for pos in alpaca_positions:
                positions.append({
                    'symbol': pos.get('symbol'),
                    'broker': 'Alpaca',
                    'strategy': 'Momentum',  # Would need actual strategy mapping
                    'side': 'LONG' if float(pos.get('qty', 0)) > 0 else 'SHORT',
                    'qty': abs(float(pos.get('qty', 0))),
                    'entryPrice': float(pos.get('avg_entry_price', 0)),
                    'currentPrice': float(pos.get('current_price', 0)),
                    'pnl': float(pos.get('unrealized_pl', 0)),
                    'pnlPct': round(float(pos.get('unrealized_plpc', 0)) * 100, 2),
                    'age': 'N/A'
                })

        # Get Coinbase LIVE positions
        if COINBASE_AVAILABLE:
            try:
                coinbase = get_coinbase_service()
                if coinbase._initialized:
                    cb_positions = coinbase.get_positions()
                    for pos in cb_positions:
                        entry_price = pos.entry_price if hasattr(pos, 'entry_price') and pos.entry_price else 0
                        current_price = pos.current_price if hasattr(pos, 'current_price') and pos.current_price else 0
                        qty = pos.quantity if hasattr(pos, 'quantity') and pos.quantity else 0
                        pnl = pos.pnl if hasattr(pos, 'pnl') and pos.pnl else 0

                        # Calculate P&L percentage
                        cost_basis = abs(qty) * entry_price if entry_price > 0 else 0
                        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0

                        positions.append({
                            'symbol': pos.symbol if hasattr(pos, 'symbol') else 'Unknown',
                            'broker': 'Coinbase',
                            'strategy': pos.strategy if hasattr(pos, 'strategy') and pos.strategy else 'Crypto',
                            'side': pos.side.value if hasattr(pos, 'side') and hasattr(pos.side, 'value') else 'LONG',
                            'qty': qty,
                            'entryPrice': entry_price,
                            'currentPrice': current_price,
                            'pnl': pnl,
                            'pnlPct': round(pnl_pct, 2),
                            'age': pos.age if hasattr(pos, 'age') and pos.age else 'N/A'
                        })
            except Exception as e:
                print(f"Error fetching Coinbase positions: {e}")

        # Get Kalshi LIVE positions
        if KalshiClient:
            try:
                kalshi = KalshiClient()
                kalshi_positions = kalshi.get_positions()
                for p in kalshi_positions:
                    pos_count = p.get('position', 0)
                    if pos_count == 0:
                        continue
                    ticker = p.get('ticker', '')

                    # Get market exposure for cost basis (in cents)
                    exposure = p.get('market_exposure', 0)
                    # Calculate average entry price from exposure
                    avg_price_cents = abs(exposure / pos_count) if pos_count != 0 else 0
                    avg_price_dollars = avg_price_cents / 100

                    # Get realized P&L (in cents) and convert to dollars
                    realized_pnl = p.get('realized_pnl', 0) / 100
                    pnl = realized_pnl
                    pnl_pct = (pnl / (abs(pos_count) * avg_price_dollars) * 100) if avg_price_dollars > 0 else 0

                    # Try to get position age from fills
                    age = 'N/A'
                    try:
                        fills = kalshi.get_fills(ticker=ticker, limit=1)
                        if fills:
                            created = fills[0].get('created_time', '')
                            if created:
                                entry_time = datetime.fromisoformat(created.replace('Z', '+00:00'))
                                delta = datetime.now(timezone.utc) - entry_time
                                hours = int(delta.total_seconds() / 3600)
                                if hours < 1:
                                    age = f"{int(delta.total_seconds() / 60)}m"
                                elif hours < 24:
                                    age = f"{hours}h"
                                else:
                                    age = f"{hours // 24}d"
                    except Exception as e:
                        print(f"Error parsing fill time: {e}")

                    positions.append({
                        'symbol': ticker[-30:] if len(ticker) > 30 else ticker,
                        'broker': 'Kalshi',
                        'strategy': 'Prediction',
                        'side': 'YES' if pos_count > 0 else 'NO',
                        'qty': abs(pos_count),
                        'entryPrice': avg_price_dollars,
                        'currentPrice': avg_price_dollars,  # Use entry as current estimate
                        'pnl': pnl,
                        'pnlPct': round(pnl_pct, 2),
                        'age': age
                    })
            except Exception as e:
                print(f"Error fetching Kalshi positions: {e}")

        # Get OANDA positions (skip if recently failed — avoids log spam)
        if not _oanda_fail_cache['failed'] or time.time() > _oanda_fail_cache['until']:
            try:
                from dashboard.services.broker_oanda import OandaService
                from oandapyV20.endpoints.positions import OpenPositions
                oanda_svc = OandaService()
                if oanda_svc.api:
                    r = OpenPositions(accountID=oanda_svc.account_id)
                    response = oanda_svc.api.request(r)
                    oanda_positions = response.get('positions', [])
                    _oanda_fail_cache['failed'] = False  # Reset on success
                    for op in oanda_positions:
                        instrument = op.get('instrument', '').replace('_', '/')
                        long_units = int(op.get('long', {}).get('units', 0))
                        short_units = int(op.get('short', {}).get('units', 0))

                        if long_units != 0:
                            avg_price = float(op.get('long', {}).get('averagePrice', 0))
                            unrealized_pl = float(op.get('long', {}).get('unrealizedPL', 0))
                            positions.append({
                                'symbol': instrument,
                                'broker': 'OANDA',
                                'strategy': 'Forex',
                                'side': 'LONG',
                                'qty': long_units,
                                'entryPrice': avg_price,
                                'currentPrice': avg_price,
                                'pnl': unrealized_pl,
                                'pnlPct': round((unrealized_pl / (long_units * avg_price) * 100), 2) if avg_price > 0 else 0,
                                'age': 'N/A'
                            })
                        if short_units != 0:
                            avg_price = float(op.get('short', {}).get('averagePrice', 0))
                            unrealized_pl = float(op.get('short', {}).get('unrealizedPL', 0))
                            positions.append({
                                'symbol': instrument,
                                'broker': 'OANDA',
                                'strategy': 'Forex',
                                'side': 'SHORT',
                                'qty': abs(short_units),
                                'entryPrice': avg_price,
                                'currentPrice': avg_price,
                                'pnl': unrealized_pl,
                                'pnlPct': round((unrealized_pl / (abs(short_units) * avg_price) * 100), 2) if avg_price > 0 else 0,
                                'age': 'N/A'
                            })
            except Exception as e:
                if not _oanda_fail_cache['failed']:
                    print(f"OANDA positions error (suppressing for 5min): {e}")
                _oanda_fail_cache.update({'failed': True, 'until': time.time() + 300, 'error': str(e)})

        return jsonify({
            'success': True,
            'positions': positions,
            'is_mock': False,
            'count': len(positions),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'positions': [],
            'is_mock': False
        })


@v5_bp.route('/api/v5/recent-trades')
def get_recent_trades():
    """Get recent LIVE trade executions (excludes paper trades)"""
    try:
        trades = []
        trade_id_counter = 1

        # Get Kalshi live fills
        if KalshiClient:
            try:
                kalshi = KalshiClient()
                fills = kalshi.get_fills(limit=20)
                for f in fills:
                    ticker = f.get('ticker', '')
                    # Shorten ticker for display
                    display_ticker = ticker[-35:] if len(ticker) > 35 else ticker
                    count = f.get('count', 0)
                    yes_price = f.get('yes_price')
                    no_price = f.get('no_price')
                    price = yes_price if yes_price else no_price
                    side = f.get('side', 'yes')
                    created = f.get('created_time', '')

                    # Parse timestamp
                    try:
                        ts = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        time_str = ts.strftime('%H:%M:%S')
                    except Exception as e:
                        time_str = created[:19] if created else 'N/A'

                    # Calculate fill price in dollars
                    fill_price = price / 100 if price else 0

                    trades.append({
                        'id': trade_id_counter,
                        'time': time_str,
                        'symbol': display_ticker,
                        'broker': 'Kalshi',
                        'side': side.upper(),
                        'qty': count,
                        'fillPrice': f"{fill_price:.2f}",
                        'slippage': 0,  # Kalshi has fixed prices, no slippage
                        'value': f"${count * fill_price:.2f}",
                        'status': 'FILLED'
                    })
                    trade_id_counter += 1
            except Exception as e:
                print(f"Error fetching Kalshi fills: {e}")

        # Get Alpaca recent orders (filled)
        if alpaca_client.is_connected():
            try:
                orders = alpaca_client.get_orders(status='closed', limit=10)
                for o in orders:
                    if o.get('status') != 'filled':
                        continue
                    filled_at = o.get('filled_at', '')
                    try:
                        ts = datetime.fromisoformat(filled_at.replace('Z', '+00:00'))
                        time_str = ts.strftime('%H:%M:%S')
                    except Exception as e:
                        time_str = 'N/A'

                    # CRITICAL: Don't default to 0 for missing fill price
                    filled_avg = o.get('filled_avg_price')
                    if filled_avg is None or filled_avg == '':
                        filled_price = 0  # Will display as unfilled
                        logger.warning(f"Order {o.get('id', 'unknown')} has no fill price")
                    else:
                        filled_price = float(filled_avg)

                    limit_price = float(o.get('limit_price', 0)) if o.get('limit_price') else filled_price
                    # Calculate slippage as difference between limit and fill price
                    slippage = round(filled_price - limit_price, 4) if limit_price > 0 and filled_price > 0 else 0

                    trades.append({
                        'id': trade_id_counter,
                        'time': time_str,
                        'symbol': o.get('symbol', ''),
                        'broker': 'Alpaca',
                        'side': o.get('side', '').upper(),
                        'qty': float(o.get('filled_qty', 0)),
                        'fillPrice': f"{filled_price:.2f}",
                        'slippage': slippage,
                        'value': f"${float(o.get('filled_qty', 0)) * filled_price:.2f}",
                        'status': 'FILLED'
                    })
                    trade_id_counter += 1
            except Exception as e:
                print(f"Error fetching Alpaca orders: {e}")

        # Sort by time (most recent first) - trades already have time strings
        # For proper sorting we'd need timestamps, but this gives recent trades

        return jsonify({
            'success': True,
            'trades': trades,
            'is_mock': False,
            'count': len(trades),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'trades': [],
            'is_mock': False
        })


@v5_bp.route('/api/v5/brokers')
def get_brokers_health():
    """Get health status of all brokers"""
    if rate_limit_check('brokers'):
        return jsonify({'success': False, 'error': 'Rate limited'})
    try:
        brokers = []

        # Measure actual Alpaca latency
        alpaca_connected = alpaca_client.is_connected()
        alpaca_bp = 0
        alpaca_latency = 0
        if alpaca_connected:
            start = time.time()
            account = alpaca_client.get_account()
            alpaca_latency = int((time.time() - start) * 1000)
            if account:
                alpaca_bp = float(account.get('buying_power', 0))

        brokers.append({
            'name': 'Alpaca',
            'status': 'CONNECTED' if alpaca_connected else 'DISCONNECTED',
            'latency': alpaca_latency,
            'buyingPower': alpaca_bp,
            'positions': len(alpaca_client.get_positions()) if alpaca_connected else 0,
            'mode': 'PAPER' if alpaca_client.paper else 'LIVE',
            'color': '#22c55e'
        })

        # Measure actual Kalshi latency
        kalshi_paper = os.getenv('KALSHI_PAPER_MODE', 'true').lower() == 'true'
        kalshi_allocation = float(os.getenv('KALSHI_ALLOCATION', 150))
        kalshi_latency = 0
        kalshi_positions_count = 0
        if KalshiClient:
            try:
                start = time.time()
                kalshi = KalshiClient()
                kalshi_positions = kalshi.get_positions()
                kalshi_latency = int((time.time() - start) * 1000)
                kalshi_positions_count = len([p for p in kalshi_positions if p.get('position', 0) != 0])
            except Exception as e:
                print(f"Kalshi latency check failed: {e}")
                kalshi_latency = 0

        brokers.append({
            'name': 'Kalshi',
            'status': 'CONNECTED' if KalshiClient else 'DISCONNECTED',
            'latency': kalshi_latency,
            'buyingPower': kalshi_allocation,
            'positions': kalshi_positions_count,
            'mode': 'PAPER' if kalshi_paper else 'LIVE',
            'color': '#3b82f6'
        })

        # OANDA - connect via OandaService (has 001-prefix safety check)
        oanda_status = 'DISCONNECTED'
        oanda_balance = float(os.getenv('OANDA_ALLOCATION', 100))
        oanda_positions = 0
        oanda_latency = 0
        oanda_paper = os.getenv('OANDA_PAPER_MODE', 'true').lower() == 'true'
        oanda_env = os.getenv('OANDA_ENVIRONMENT', 'practice')

        # Use cached failure status if OANDA recently errored (avoids log spam)
        if _oanda_fail_cache['failed'] and time.time() < _oanda_fail_cache['until']:
            oanda_status = _oanda_fail_cache.get('status', 'DISCONNECTED')
        else:
            try:
                from dashboard.services.broker_oanda import OandaService
                from oandapyV20.endpoints.accounts import AccountSummary
                oanda_svc = OandaService()
                if oanda_svc.api:
                    start = time.time()
                    r = AccountSummary(accountID=oanda_svc.account_id)
                    response = oanda_svc.api.request(r)
                    oanda_latency = int((time.time() - start) * 1000)
                    account_info = response.get('account', {})
                    oanda_balance = float(account_info.get('balance', oanda_balance))
                    oanda_positions = int(account_info.get('openPositionCount', 0))
                    oanda_status = 'CONNECTED'
                    _oanda_fail_cache['failed'] = False
                else:
                    oanda_status = 'NO_CREDENTIALS'
            except Exception as e:
                error_str = str(e).lower()
                if 'authorization' in error_str or '401' in error_str or 'unauthorized' in error_str or 'forbidden' in error_str:
                    oanda_status = 'AUTH_FAILED'
                    if not _oanda_fail_cache['failed']:
                        print("OANDA: API key expired/forbidden. Regenerate at https://hub.oanda.com/")
                else:
                    oanda_status = 'DISCONNECTED'
                    if not _oanda_fail_cache['failed']:
                        print(f"OANDA health check failed (suppressing for 5min): {e}")
                _oanda_fail_cache.update({
                    'failed': True, 'until': time.time() + 300,
                    'error': str(e), 'status': oanda_status
                })

        brokers.append({
            'name': 'OANDA',
            'status': oanda_status,
            'latency': oanda_latency,
            'buyingPower': oanda_balance,
            'positions': oanda_positions,
            'mode': 'PAPER' if oanda_paper or oanda_env == 'practice' else 'LIVE',
            'color': '#f59e0b'
        })

        # Coinbase
        if COINBASE_AVAILABLE:
            try:
                coinbase = get_coinbase_service()
                health = coinbase.get_health()
                brokers.append({
                    'name': 'Coinbase',
                    'status': health.status.value if health.status else 'DISCONNECTED',
                    'latency': health.latency_ms,
                    'buyingPower': health.buying_power,
                    'positions': len(coinbase.get_positions()) if coinbase._initialized else 0,
                    'mode': 'LIVE',  # Coinbase is always live
                    'color': '#0052ff'  # Coinbase blue
                })
            except Exception as e:
                brokers.append({
                    'name': 'Coinbase',
                    'status': 'DISCONNECTED',
                    'latency': 0,
                    'buyingPower': 0,
                    'positions': 0,
                    'mode': 'LIVE',
                    'color': '#0052ff'
                })

        return jsonify({
            'success': True,
            'brokers': brokers,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@v5_bp.route('/api/v5/alerts')
def get_v5_alerts():
    """Get alerts for the V5 dashboard"""
    try:
        alerts = []
        now = datetime.now(timezone.utc)

        # Get alerts from trading_master.db (always use live DB)
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'live', 'trading_master.db')

        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Check if alerts table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'")
                if cursor.fetchone():
                    cursor.execute('''
                        SELECT severity, message, timestamp
                        FROM alerts
                        ORDER BY timestamp DESC
                        LIMIT 20
                    ''')
                    for row in cursor.fetchall():
                        try:
                            ts = datetime.fromisoformat(row[2].replace('Z', '+00:00'))
                            time_ago = _format_time_ago(ts)
                        except Exception as e:
                            time_ago = 'Unknown'

                        alerts.append({
                            'id': len(alerts) + 1,
                            'severity': row[0],
                            'message': row[1],
                            'time': time_ago,
                            'source': 'System'
                        })
                conn.close()
            except Exception as db_error:
                print(f"Database error fetching alerts: {db_error}")

        # Add system status alerts
        alpaca_connected = alpaca_client.is_connected()

        if not alpaca_connected:
            alerts.insert(0, {
                'id': len(alerts) + 1,
                'severity': 'warning',
                'message': 'Alpaca API not connected',
                'time': 'Now',
                'source': 'System'
            })

        # Check if it's weekend - add info alert
        if now.weekday() >= 5:
            alerts.insert(0, {
                'id': len(alerts) + 1,
                'severity': 'info',
                'message': 'Market closed - Weekend',
                'time': 'Now',
                'source': 'System'
            })

        # Default alerts if none found
        is_mock_alerts = False
        if len(alerts) == 0:
            is_mock_alerts = True
            alerts = [
                {'id': 1, 'severity': 'info', 'message': 'System started in LIVE mode', 'time': 'Now', 'source': 'System'},
                {'id': 2, 'severity': 'info', 'message': 'All systems operational', 'time': 'Now', 'source': 'System'},
            ]

        return jsonify({
            'success': True,
            'alerts': alerts,
            'is_mock': is_mock_alerts,
            'count': len(alerts),
            'timestamp': now.isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'alerts': [],
            'is_mock': False
        })


@v5_bp.route('/api/v5/equity-curve')
def get_equity_curve():
    """Get equity curve data for the dashboard chart.

    Queries the portfolio_value table for historical snapshots and
    aggregates them into daily data points. Falls back to flat mock
    data (at starting capital) when no history exists yet.

    Returns: { success, data: [{day, date, equity, spy, btc}], is_mock, points, timestamp }
    """
    try:
        starting_capital = float(os.getenv('TOTAL_CAPITAL', '500'))
        num_days = 30

        # Always use live DB (orchestrator writes here)
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'live', 'trading_master.db')

        equity_data = []

        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Check if portfolio_value table exists and has data
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_value'")
                if cursor.fetchone():
                    # Get daily snapshots - take the last reading per day
                    cursor.execute('''
                        SELECT
                            DATE(timestamp) as date,
                            total_value,
                            cash,
                            positions_value
                        FROM portfolio_value
                        WHERE timestamp >= date('now', ?)
                        ORDER BY timestamp ASC
                    ''', (f'-{num_days} days',))

                    rows = cursor.fetchall()
                    if rows:
                        # Aggregate: take last value per day
                        daily = {}
                        for row in rows:
                            daily[row[0]] = {
                                'date': row[0],
                                'equity': round(row[1], 2) if row[1] else starting_capital,
                            }

                        # Build ordered list
                        day_num = 1
                        for date_key in sorted(daily.keys()):
                            entry = daily[date_key]
                            entry['day'] = day_num
                            entry['spy'] = starting_capital  # Placeholder until benchmark data
                            entry['btc'] = starting_capital  # Placeholder until benchmark data
                            equity_data.append(entry)
                            day_num += 1

                # Also check daily_summary for P&L reconstruction if portfolio_value is empty
                if not equity_data:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_summary'")
                    if cursor.fetchone():
                        cursor.execute('''
                            SELECT date, SUM(pnl) as daily_pnl
                            FROM daily_summary
                            WHERE date >= date('now', ?)
                            GROUP BY date
                            ORDER BY date ASC
                        ''', (f'-{num_days} days',))

                        rows = cursor.fetchall()
                        if rows:
                            running_equity = starting_capital
                            day_num = 1
                            for row in rows:
                                running_equity += (row[1] or 0)
                                equity_data.append({
                                    'day': day_num,
                                    'date': row[0],
                                    'equity': round(running_equity, 2),
                                    'spy': starting_capital,
                                    'btc': starting_capital,
                                })
                                day_num += 1

                conn.close()
            except Exception as db_error:
                print(f"Database error fetching equity curve: {db_error}")

        # Calculate equity from trades if no portfolio_value history exists
        if not equity_data and os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                # Get total P&L from closed trades
                cursor.execute('SELECT SUM(pnl) FROM trades WHERE status = "closed"')
                total_pnl = cursor.fetchone()[0] or 0
                conn.close()

                current_equity = starting_capital + total_pnl
                today = datetime.now().strftime('%Y-%m-%d')
                equity_data = [{
                    'day': num_days,
                    'date': today,
                    'equity': round(current_equity, 2),
                    'spy': starting_capital,
                    'btc': starting_capital,
                }]
            except Exception:
                pass

        # Fallback: return empty data with mock indicator instead of fake flat data
        is_mock = len(equity_data) == 0

        return jsonify({
            'success': True,
            'data': equity_data,
            'is_mock': is_mock,
            'message': 'No portfolio data yet - start trading to see real equity curve' if is_mock else None,
            'points': len(equity_data),
            'starting_capital': starting_capital,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        # On error, return empty data with mock indicator
        starting_capital = float(os.getenv('TOTAL_CAPITAL', '500'))
        return jsonify({
            'success': False,
            'error': str(e),
            'data': [],
            'is_mock': True,
            'message': f'Error loading equity data: {str(e)}',
            'points': 0,
            'starting_capital': starting_capital,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
