"""
Computer Vision Bot Configuration

Configuration settings and parameters for the Computer Vision Trading Bot.
Includes broker-specific settings, visual recognition parameters, and safety limits.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class CVConfig:
    """Computer Vision Bot Configuration"""

    # General Settings
    PAPER_MODE: bool = True
    ENABLED_BROKERS: List[str] = None
    SCAN_INTERVAL_SECONDS: int = 1800  # 30 minutes

    # Visual Recognition Settings
    TEMPLATE_MATCH_THRESHOLD: float = 0.8
    OCR_CONFIDENCE_THRESHOLD: float = 0.6
    PRICE_DETECTION_THRESHOLD: float = 0.7

    # Trading Parameters
    MAX_CONCURRENT_TRADES: int = 3
    POSITION_SIZE_USD: float = 50.0  # Conservative for CV trading
    CONFIDENCE_THRESHOLD: float = 0.7

    # Safety Limits
    MAX_DAILY_TRADES: int = 10
    MAX_DAILY_LOSS_USD: float = 200.0
    STOP_LOSS_PERCENT: float = 0.05  # 5% stop loss

    # Screen Resolution and UI Settings
    SCREENSHOT_TIMEOUT_SECONDS: int = 10
    CLICK_DELAY_MS: int = 500
    TYPE_DELAY_MS: int = 100
    RETRY_ATTEMPTS: int = 3

    # Windows-MCP Settings
    MCP_SERVER_TIMEOUT: int = 30
    MCP_SERVER_PATH: str = r"C:\dev\projects\Windows-MCP"

    def __post_init__(self):
        """Post-initialization setup"""
        if self.ENABLED_BROKERS is None:
            # Default to paper-mode safe brokers
            self.ENABLED_BROKERS = ['alpaca'] if self.PAPER_MODE else ['alpaca', 'kalshi']


# Broker-specific configurations
BROKER_CONFIGS = {
    'alpaca': {
        'name': 'Alpaca Markets',
        'url': 'https://app.alpaca.markets',
        'login_url': 'https://app.alpaca.markets/login',
        'trading_url': 'https://app.alpaca.markets/trading',

        # UI element selectors/coordinates (relative to common screen resolutions)
        'elements': {
            'login_username': {'type': 'input', 'placeholder': 'email'},
            'login_password': {'type': 'input', 'placeholder': 'password'},
            'login_button': {'type': 'button', 'text': 'sign in'},

            'buy_button': {'type': 'button', 'text': 'buy', 'color': 'green'},
            'sell_button': {'type': 'button', 'text': 'sell', 'color': 'red'},
            'quantity_input': {'type': 'input', 'label': 'qty'},
            'price_input': {'type': 'input', 'label': 'limit price'},
            'submit_order': {'type': 'button', 'text': 'place order'},
        },

        # Price pattern recognition
        'price_patterns': [
            r'\$[\d,]+\.\d{2}',  # $123.45 format
            r'[\d,]+\.\d{2}',    # 123.45 format
        ],

        # Chart/visual indicators
        'visual_indicators': {
            'price_up': {'color': 'green', 'symbols': ['+', '↑', '⬆']},
            'price_down': {'color': 'red', 'symbols': ['-', '↓', '⬇']},
        }
    },

    'kalshi': {
        'name': 'Kalshi',
        'url': 'https://kalshi.com',
        'login_url': 'https://kalshi.com/sign-in',
        'trading_url': 'https://kalshi.com/markets',

        'elements': {
            'login_email': {'type': 'input', 'placeholder': 'email'},
            'login_password': {'type': 'input', 'placeholder': 'password'},
            'login_button': {'type': 'button', 'text': 'sign in'},

            'yes_button': {'type': 'button', 'text': 'yes', 'color': 'green'},
            'no_button': {'type': 'button', 'text': 'no', 'color': 'red'},
            'quantity_input': {'type': 'input', 'label': 'quantity'},
            'price_input': {'type': 'input', 'label': 'price'},
            'place_order': {'type': 'button', 'text': 'place order'},
        },

        'price_patterns': [
            r'\d+¢',           # 50¢ format
            r'\$\d+\.\d{2}',   # $0.50 format
            r'0\.\d{2}',       # 0.50 format
        ],

        'visual_indicators': {
            'price_up': {'color': 'green', 'symbols': ['+', '↑']},
            'price_down': {'color': 'red', 'symbols': ['-', '↓']},
        }
    },

    'oanda': {
        'name': 'OANDA',
        'url': 'https://trade.oanda.com',
        'login_url': 'https://www.oanda.com/account/login',
        'trading_url': 'https://trade.oanda.com',

        'elements': {
            'login_username': {'type': 'input', 'placeholder': 'username'},
            'login_password': {'type': 'input', 'placeholder': 'password'},
            'login_button': {'type': 'button', 'text': 'log in'},

            'buy_button': {'type': 'button', 'text': 'buy', 'color': 'blue'},
            'sell_button': {'type': 'button', 'text': 'sell', 'color': 'orange'},
            'units_input': {'type': 'input', 'label': 'units'},
            'price_input': {'type': 'input', 'label': 'rate'},
            'create_order': {'type': 'button', 'text': 'create order'},
        },

        'price_patterns': [
            r'\d+\.\d{4,5}',   # 1.2345 forex format
            r'\d+\.\d{2}',     # 123.45 format
        ],

        'visual_indicators': {
            'price_up': {'color': 'green', 'symbols': ['+', '↑']},
            'price_down': {'color': 'red', 'symbols': ['-', '↓']},
        }
    }
}


# Template file mappings
TEMPLATE_MAPPINGS = {
    'buttons': {
        'generic_login': 'templates/generic_login_button.png',
        'generic_buy': 'templates/generic_buy_button.png',
        'generic_sell': 'templates/generic_sell_button.png',
    },

    'alpaca': {
        'buy_button': 'templates/alpaca_buy_button.png',
        'sell_button': 'templates/alpaca_sell_button.png',
        'order_form': 'templates/alpaca_order_form.png',
    },

    'kalshi': {
        'yes_button': 'templates/kalshi_yes_button.png',
        'no_button': 'templates/kalshi_no_button.png',
        'market_card': 'templates/kalshi_market_card.png',
    },

    'oanda': {
        'buy_button': 'templates/oanda_buy_button.png',
        'sell_button': 'templates/oanda_sell_button.png',
        'order_ticket': 'templates/oanda_order_ticket.png',
    }
}


# Color definitions for visual recognition (HSV color space)
COLOR_RANGES = {
    'green': {
        'lower': (40, 40, 40),
        'upper': (80, 255, 255)
    },
    'red': {
        'lower1': (0, 40, 40),
        'upper1': (10, 255, 255),
        'lower2': (160, 40, 40),
        'upper2': (180, 255, 255)
    },
    'blue': {
        'lower': (100, 40, 40),
        'upper': (140, 255, 255)
    },
    'orange': {
        'lower': (10, 40, 40),
        'upper': (25, 255, 255)
    }
}


# Screen resolution presets for different common setups
SCREEN_RESOLUTIONS = {
    '1080p': (1920, 1080),
    '1440p': (2560, 1440),
    '4k': (3840, 2160),
    'laptop_13': (1366, 768),
    'laptop_15': (1920, 1080),
}


# Risk management settings
RISK_MANAGEMENT = {
    'max_position_size_percent': 5.0,  # 5% of account per position
    'daily_loss_limit_percent': 10.0,  # 10% daily loss limit
    'win_rate_threshold': 0.3,  # Stop if win rate drops below 30%
    'consecutive_losses_limit': 5,  # Stop after 5 consecutive losses
    'cooldown_period_hours': 2,  # 2-hour cooldown after stop
}


# Performance monitoring
PERFORMANCE_METRICS = {
    'track_execution_time': True,
    'track_success_rate': True,
    'track_detection_accuracy': True,
    'log_screenshots': True,  # For debugging and improvement
    'alert_on_errors': True,
}


def get_config() -> CVConfig:
    """Get the current CV bot configuration"""
    config = CVConfig()

    # Override with environment variables if present
    config.PAPER_MODE = os.getenv('CV_PAPER_MODE', 'true').lower() == 'true'
    config.SCAN_INTERVAL_SECONDS = int(os.getenv('CV_SCAN_INTERVAL', config.SCAN_INTERVAL_SECONDS))
    config.MAX_CONCURRENT_TRADES = int(os.getenv('CV_MAX_TRADES', config.MAX_CONCURRENT_TRADES))

    # Parse enabled brokers from environment
    brokers_env = os.getenv('CV_ENABLED_BROKERS')
    if brokers_env:
        config.ENABLED_BROKERS = [b.strip() for b in brokers_env.split(',')]

    return config


def get_broker_config(broker_name: str) -> Dict[str, Any]:
    """Get configuration for a specific broker"""
    return BROKER_CONFIGS.get(broker_name, {})


def get_template_path(category: str, name: str) -> str:
    """Get the file path for a template image"""
    template_map = TEMPLATE_MAPPINGS.get(category, {})
    return template_map.get(name, '')


def validate_config() -> List[str]:
    """Validate the current configuration and return any warnings"""
    warnings = []
    config = get_config()

    # Check template files exist
    for category, templates in TEMPLATE_MAPPINGS.items():
        for name, path in templates.items():
            if not os.path.exists(path):
                warnings.append(f"Missing template file: {path}")

    # Check broker configurations
    for broker in config.ENABLED_BROKERS:
        if broker not in BROKER_CONFIGS:
            warnings.append(f"Unknown broker configuration: {broker}")

    # Validate safety limits
    if config.MAX_DAILY_TRADES > 50:
        warnings.append("MAX_DAILY_TRADES is very high - consider reducing for safety")

    if config.POSITION_SIZE_USD > 1000 and config.PAPER_MODE is False:
        warnings.append("POSITION_SIZE_USD is high for live trading - consider reducing")

    # Check Windows-MCP server path
    if not os.path.exists(config.MCP_SERVER_PATH):
        warnings.append(f"Windows-MCP server path not found: {config.MCP_SERVER_PATH}")

    return warnings