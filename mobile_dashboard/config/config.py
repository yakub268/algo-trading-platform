"""
Mobile Dashboard Configuration
=============================

Configuration settings for the mobile trading dashboard including
security, performance, and feature settings.

Author: Trading Bot System
Created: February 2026
"""

import os
from typing import Dict, Any

class Config:
    """Base configuration class"""

    # Flask Configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'mobile-trading-dashboard-secret-2026')

    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///mobile_dashboard.db')

    # Redis Configuration (for session management and caching)
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))

    # WebSocket Configuration
    SOCKETIO_ASYNC_MODE = 'threading'
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"

    # Security Configuration
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-2026')
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    JWT_REFRESH_TOKEN_EXPIRES = 2592000  # 30 days

    # Rate Limiting
    RATELIMIT_STORAGE_URL = "redis://localhost:6379"
    RATELIMIT_DEFAULT = "100 per hour"

    # CORS Settings
    CORS_ORIGINS = ["*"]
    CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_HEADERS = ["Content-Type", "Authorization"]

    # Mobile Dashboard Features
    FEATURES = {
        'real_time_updates': True,
        'push_notifications': True,
        'voice_alerts': True,
        'offline_mode': True,
        'dark_mode': True,
        'portfolio_heatmap': True,
        'trade_execution': True,
        'bot_control': True,
        'risk_management': True,
        'performance_analytics': True
    }

    # Update Intervals (in seconds)
    UPDATE_INTERVALS = {
        'portfolio': 30,
        'positions': 15,
        'alerts': 10,
        'performance': 60,
        'bot_status': 30
    }

    # Risk Management Settings
    RISK_SETTINGS = {
        'max_position_size': 0.05,  # 5% of portfolio
        'max_portfolio_exposure': 0.95,  # 95% of portfolio
        'max_daily_loss': 0.02,  # 2% max daily loss
        'max_drawdown_alert': 0.05,  # 5% drawdown alert
        'position_size_alert': 0.03,  # 3% position size alert
        'emergency_stop_loss': 0.10  # 10% portfolio loss triggers emergency stop
    }

    # Notification Settings
    NOTIFICATION_SETTINGS = {
        'push_notifications': {
            'enabled': True,
            'vapid_public_key': os.getenv('VAPID_PUBLIC_KEY', ''),
            'vapid_private_key': os.getenv('VAPID_PRIVATE_KEY', ''),
            'vapid_claims': {
                'sub': 'mailto:admin@tradingbot.com'
            }
        },
        'voice_alerts': {
            'enabled': True,
            'volume': 0.8,
            'rate': 1.0,
            'pitch': 1.0
        },
        'telegram': {
            'enabled': os.getenv('TELEGRAM_BOT_TOKEN') is not None,
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
        }
    }

    # Performance Settings
    PERFORMANCE_SETTINGS = {
        'cache_timeout': 300,  # 5 minutes
        'max_chart_points': 500,
        'chart_update_interval': 30,
        'background_sync_interval': 60,
        'offline_cache_size': 10  # MB
    }

    # Exchange API Settings
    EXCHANGE_SETTINGS = {
        'alpaca': {
            'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            'api_key': os.getenv('ALPACA_API_KEY', ''),
            'api_secret': os.getenv('ALPACA_SECRET_KEY', ''),
            'timeout': 30,
            'max_retries': 3
        },
        'freqtrade': {
            'base_url': os.getenv('FREQTRADE_API_URL', 'http://localhost:8080'),
            'username': os.getenv('FREQTRADE_USERNAME', ''),
            'password': os.getenv('FREQTRADE_PASSWORD', ''),
            'timeout': 30,
            'max_retries': 3
        }
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

    # Relaxed security for development
    CORS_ORIGINS = ["*"]

    # More frequent updates for development
    UPDATE_INTERVALS = {
        'portfolio': 10,
        'positions': 5,
        'alerts': 5,
        'performance': 30,
        'bot_status': 10
    }

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

    # Strict security for production
    CORS_ORIGINS = ["https://your-domain.com"]

    # SSL and Security Headers
    SSL_REDIRECT = True
    SECURITY_HEADERS = {
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' fonts.googleapis.com; font-src 'self' fonts.gstatic.com;"
    }

    # Production logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

    # Use in-memory database for testing
    DATABASE_URL = 'sqlite:///:memory:'

    # Disable external services for testing
    FEATURES = {
        'real_time_updates': False,
        'push_notifications': False,
        'voice_alerts': False,
        'offline_mode': False,
        'dark_mode': True,
        'portfolio_heatmap': True,
        'trade_execution': False,
        'bot_control': False,
        'risk_management': True,
        'performance_analytics': True
    }

# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(environment: str = None) -> Config:
    """Get configuration based on environment"""
    if environment is None:
        environment = os.getenv('FLASK_ENV', 'development')

    config_class = config_map.get(environment, DevelopmentConfig)
    return config_class()

# Feature flags for runtime configuration
FEATURE_FLAGS: Dict[str, bool] = {
    'enable_trade_execution': os.getenv('ENABLE_TRADE_EXECUTION', 'true').lower() == 'true',
    'enable_bot_control': os.getenv('ENABLE_BOT_CONTROL', 'true').lower() == 'true',
    'enable_push_notifications': os.getenv('ENABLE_PUSH_NOTIFICATIONS', 'true').lower() == 'true',
    'enable_voice_alerts': os.getenv('ENABLE_VOICE_ALERTS', 'false').lower() == 'true',
    'enable_offline_mode': os.getenv('ENABLE_OFFLINE_MODE', 'true').lower() == 'true',
    'enable_real_time_updates': os.getenv('ENABLE_REAL_TIME_UPDATES', 'true').lower() == 'true',
    'enable_risk_management': os.getenv('ENABLE_RISK_MANAGEMENT', 'true').lower() == 'true',
    'enable_performance_analytics': os.getenv('ENABLE_PERFORMANCE_ANALYTICS', 'true').lower() == 'true',
    'enable_portfolio_heatmap': os.getenv('ENABLE_PORTFOLIO_HEATMAP', 'true').lower() == 'true',
    'enable_emergency_stop': os.getenv('ENABLE_EMERGENCY_STOP', 'true').lower() == 'true'
}

# Theme configuration
THEME_CONFIG = {
    'default_theme': 'dark',
    'available_themes': ['dark', 'light'],
    'theme_persistence': True,
    'automatic_theme_switching': {
        'enabled': False,
        'dark_hours': [19, 7],  # 7 PM to 7 AM
        'light_hours': [7, 19]  # 7 AM to 7 PM
    }
}

# Chart configuration
CHART_CONFIG = {
    'default_chart_type': 'line',
    'available_chart_types': ['line', 'candlestick', 'bar', 'area'],
    'default_timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
    'max_data_points': 1000,
    'animation_enabled': True,
    'responsive': True,
    'color_schemes': {
        'dark': {
            'background': '#1a1a2e',
            'grid': 'rgba(255, 255, 255, 0.1)',
            'text': '#eee',
            'positive': '#00d4aa',
            'negative': '#ff6b6b',
            'warning': '#feca57',
            'info': '#48cae4'
        },
        'light': {
            'background': '#ffffff',
            'grid': 'rgba(0, 0, 0, 0.1)',
            'text': '#333',
            'positive': '#00d4aa',
            'negative': '#ff6b6b',
            'warning': '#feca57',
            'info': '#48cae4'
        }
    }
}

# Mobile-specific configuration
MOBILE_CONFIG = {
    'touch_optimization': True,
    'gesture_navigation': True,
    'haptic_feedback': True,
    'orientation_lock': False,
    'full_screen_mode': True,
    'status_bar_style': 'dark-content',
    'splash_screen': {
        'enabled': True,
        'duration': 2000,  # 2 seconds
        'logo': '/static/icons/icon-192x192.png'
    },
    'bottom_navigation': {
        'enabled': True,
        'hide_on_scroll': False,
        'items': ['portfolio', 'positions', 'alerts', 'bots', 'performance']
    }
}

# PWA configuration
PWA_CONFIG = {
    'name': 'Trading Bot Mobile Dashboard',
    'short_name': 'Trading Bot',
    'description': 'Comprehensive mobile trading dashboard for real-time portfolio management',
    'start_url': '/',
    'display': 'standalone',
    'background_color': '#1a1a2e',
    'theme_color': '#0f0f23',
    'orientation': 'portrait-primary',
    'categories': ['finance', 'productivity'],
    'lang': 'en-US',
    'icons': [
        {'src': '/static/icons/icon-72x72.png', 'sizes': '72x72', 'type': 'image/png'},
        {'src': '/static/icons/icon-96x96.png', 'sizes': '96x96', 'type': 'image/png'},
        {'src': '/static/icons/icon-128x128.png', 'sizes': '128x128', 'type': 'image/png'},
        {'src': '/static/icons/icon-144x144.png', 'sizes': '144x144', 'type': 'image/png'},
        {'src': '/static/icons/icon-152x152.png', 'sizes': '152x152', 'type': 'image/png'},
        {'src': '/static/icons/icon-192x192.png', 'sizes': '192x192', 'type': 'image/png'},
        {'src': '/static/icons/icon-384x384.png', 'sizes': '384x384', 'type': 'image/png'},
        {'src': '/static/icons/icon-512x512.png', 'sizes': '512x512', 'type': 'image/png'}
    ]
}