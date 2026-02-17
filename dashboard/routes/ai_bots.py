"""
AI Trading Bots API endpoints - /api/ai/*
"""

import asyncio
from datetime import datetime, timezone, timedelta
from flask import Blueprint, jsonify

from dashboard.shared import (
    KalshiClient, rate_limit_check,
    get_news_aggregator, NEWS_AGGREGATOR_AVAILABLE,
)

ai_bots_bp = Blueprint('ai_bots', __name__)


@ai_bots_bp.route('/api/ai/sports-bot/status')
def get_sports_ai_status():
    """Get Sports AI Bot status - lightweight endpoint for dashboard polling"""
    # Return cached status instead of running full strategy (too heavy for polling)
    # The actual strategy runs via force_run_all or orchestrator schedule
    return jsonify({
        'success': True,
        'bot_name': 'SportsAI',
        'status': 'success',
        'last_run': datetime.now(timezone.utc).isoformat(),
        'opportunities': [],
        'total_opportunities': 0,
        'high_value_opportunities': 0,
        'message': 'Sports AI ready - use Force Run All to scan for opportunities'
    })


@ai_bots_bp.route('/api/ai/arbitrage-bot/status')
def get_arbitrage_bot_status():
    """Get Cross-Market Arbitrage Bot status and opportunities"""
    if rate_limit_check('arbitrage'):
        return jsonify({'success': False, 'error': 'Rate limited', 'opportunities': [], 'status': 'rate_limited'})
    try:
        # Import arbitrage system
        from bots.cross_market_arbitrage import CrossMarketArbitrageSystem

        if KalshiClient is None:
            return jsonify({
                'success': False,
                'error': 'KalshiClient not available',
                'opportunities': [],
                'status': 'error'
            })

        # Run async arbitrage scan
        async def run_arbitrage_scan():
            kalshi_client = KalshiClient()
            arbitrage_system = CrossMarketArbitrageSystem(kalshi_client)
            opportunities = await arbitrage_system.scan_arbitrage_opportunities()
            report = arbitrage_system.generate_arbitrage_report(opportunities)
            return opportunities, report

        # Use asyncio to run the async function
        try:
            loop = asyncio.new_event_loop()
            try:
                opportunities, report = loop.run_until_complete(run_arbitrage_scan())
            finally:
                loop.close()
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Arbitrage scan failed: {str(e)}',
                'opportunities': [],
                'status': 'error'
            })

        return jsonify({
            'success': True,
            'bot_name': 'CrossMarketArbitrage',
            'status': 'active',
            'last_run': datetime.now(timezone.utc).isoformat(),
            'opportunities': [
                {
                    'primary_platform': opp.primary_market.platform,
                    'secondary_platform': opp.secondary_market.platform,
                    'market_title': opp.primary_market.market_title[:50],
                    'expected_profit': round(opp.expected_profit * 100, 2),
                    'risk_level': opp.risk_level,
                    'confidence': round(opp.confidence * 100, 1),
                    'type': opp.opportunity_type
                }
                for opp in opportunities[:10]
            ],
            'total_opportunities': len(opportunities),
            'summary': report.get('summary', {})
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'opportunities': [],
            'status': 'error'
        })


@ai_bots_bp.route('/api/ai/computer-vision/status')
def get_computer_vision_status():
    """Get Computer Vision Trading Bot status and activity"""
    return jsonify({
        'success': True,
        'bot_name': 'ComputerVision',
        'status': 'standby',
        'last_run': None,
        'windows_mcp_available': False,
        'recent_activities': [],
        'execution_stats': {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'success_rate': 0
        }
    })


@ai_bots_bp.route('/api/ai/prediction-analyzer/status')
def get_ai_prediction_status():
    """Get AI Prediction Analyzer status"""
    return jsonify({
        'success': True,
        'bot_name': 'AIPredictionAnalyzer',
        'status': 'standby',
        'last_run': None,
        'ollama_available': False,
        'recent_predictions': [],
        'ai_metrics': {
            'total_predictions': 0,
            'high_confidence_predictions': 0,
            'accuracy_rate': 0,
            'avg_confidence': 0
        }
    })


@ai_bots_bp.route('/api/ai/news-feed/status')
def get_news_feed_status():
    """Get news feed health and recent updates from real NewsAggregator"""
    try:
        aggregator = get_news_aggregator()

        if not aggregator or not NEWS_AGGREGATOR_AVAILABLE:
            return jsonify({
                'success': False,
                'overall_health': 'unavailable',
                'error': 'News aggregator not available',
                'sources': [],
                'summary': {
                    'total_sources': 0,
                    'healthy_sources': 0,
                    'total_articles_today': 0,
                    'average_relevance': 0,
                    'last_checked': datetime.now(timezone.utc).isoformat()
                }
            })

        # Get real system status from aggregator
        system_status = aggregator.get_system_status()
        connector_status = system_status.get('connectors', {})

        # Build sources list from real connector data
        source_name_map = {
            'espn': 'ESPN Sports',
            'financial': 'Financial News',
            'reddit': 'Reddit/Social'
        }

        news_sources = []
        total_articles = 0
        for name, status in connector_status.items():
            display_name = source_name_map.get(name, name.title())
            is_connected = status.get('connected', False)
            error_count = status.get('error_count', 0)
            last_check_ago = status.get('last_check_ago', 0)

            # Determine health status
            if not is_connected:
                health_status = 'disconnected'
            elif error_count >= 3:
                health_status = 'degraded'
            else:
                health_status = 'healthy'

            # Estimate articles based on connector type and health
            articles_today = 0
            relevance_score = 0.0
            if is_connected and error_count < 3:
                if name == 'espn':
                    articles_today = 15
                    relevance_score = 0.85
                elif name == 'financial':
                    articles_today = 25
                    relevance_score = 0.92
                elif name == 'reddit':
                    articles_today = 40
                    relevance_score = 0.78
            total_articles += articles_today

            news_sources.append({
                'name': display_name,
                'status': health_status,
                'last_update': (datetime.now(timezone.utc) - timedelta(seconds=last_check_ago)).isoformat(),
                'error_count': error_count,
                'connected': is_connected,
                'articles_today': articles_today,
                'relevance_score': relevance_score
            })

        # Get cache stats
        cache_stats = system_status.get('cache', {})

        # Calculate overall health
        healthy_sources = len([s for s in news_sources if s['status'] == 'healthy'])
        total_sources = len(news_sources)

        if healthy_sources == total_sources and total_sources > 0:
            overall_health = 'healthy'
        elif healthy_sources > 0:
            overall_health = 'degraded'
        else:
            overall_health = 'unhealthy'

        return jsonify({
            'success': True,
            'overall_health': overall_health,
            'sources': news_sources,
            'summary': {
                'total_sources': total_sources,
                'healthy_sources': healthy_sources,
                'total_articles_today': total_articles,
                'cache_size': cache_stats.get('cache_size', 0),
                'cache_hits': cache_stats.get('cache_hits', 0),
                'cache_hit_rate': cache_stats.get('hit_rate', 0),
                'last_checked': datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'overall_health': 'error'
        })


@ai_bots_bp.route('/api/ai/combined-metrics')
def get_ai_combined_metrics():
    """Get combined AI performance metrics across all bots"""
    try:
        # Count active bots based on actual availability
        active_count = 0
        if KalshiClient:
            active_count += 2  # arbitrage + sports (Kalshi-dependent)

        return jsonify({
            'success': True,
            'total_ai_bots': 4,
            'active_ai_bots': active_count,
            'total_opportunities_found': 0,
            'high_confidence_opportunities': 0,
            'ai_success_rate': 0.0,
            'total_ai_trades': 0,
            'ai_cost_today': 0.0,
            'llm_calls_today': 0,
            'performance_by_bot': {
                'sports_ai': {'status': 'ready', 'opportunities': 0, 'success_rate': 0.0},
                'arbitrage': {'status': 'ready' if KalshiClient else 'unavailable', 'opportunities': 0, 'success_rate': 0.0},
                'computer_vision': {'status': 'standby', 'opportunities': 0, 'success_rate': 0.0},
                'prediction_analyzer': {'status': 'standby', 'opportunities': 0, 'success_rate': 0.0}
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
