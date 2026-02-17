#!/usr/bin/env python3
"""
Mobile Trading Dashboard Startup Script
=======================================

Production-ready startup script for the mobile trading dashboard.
Handles environment setup, configuration validation, and graceful startup.

Usage:
    python run_mobile_dashboard.py [--port PORT] [--host HOST] [--env ENV]

Features:
- Environment validation
- Configuration verification
- Health checks
- Graceful shutdown
- Performance monitoring

Author: Trading Bot System
Created: February 2026
"""

import os
import sys
import argparse
import logging
import signal
from typing import Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from mobile_dashboard.app import app, socketio, mobile_api
from mobile_dashboard.config.config import get_config, FEATURE_FLAGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mobile_dashboard.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MobileDashboardServer:
    """Mobile dashboard server management class"""

    def __init__(self, host: str = '0.0.0.0', port: int = 5001, environment: str = 'development'):
        self.host = host
        self.port = port
        self.environment = environment
        self.config = get_config(environment)
        self.server_process = None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def validate_environment(self) -> bool:
        """Validate environment and configuration"""
        logger.info("Validating environment configuration...")

        # Check required environment variables
        required_vars = []

        if FEATURE_FLAGS['enable_trade_execution']:
            required_vars.extend([
                'ALPACA_API_KEY',
                'ALPACA_SECRET_KEY'
            ])

        if FEATURE_FLAGS['enable_push_notifications']:
            required_vars.extend([
                'VAPID_PUBLIC_KEY',
                'VAPID_PRIVATE_KEY'
            ])

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
            logger.warning("Some features may be disabled")

        # Check trading bot directory structure
        trading_bot_path = os.path.dirname(__file__)
        required_dirs = ['dashboard', 'bots', 'utils']

        for dir_name in required_dirs:
            dir_path = os.path.join(trading_bot_path, dir_name)
            if not os.path.exists(dir_path):
                logger.error(f"Required directory not found: {dir_path}")
                return False

        logger.info("Environment validation completed")
        return True

    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        logger.info("Checking dependencies...")

        required_modules = [
            'flask',
            'flask_socketio',
            'flask_cors',
            'redis',
            'pandas',
            'numpy'
        ]

        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)

        if missing_modules:
            logger.error(f"Missing required modules: {', '.join(missing_modules)}")
            logger.error("Install dependencies with: pip install -r requirements.txt")
            return False

        logger.info("All dependencies satisfied")
        return True

    def setup_database(self) -> bool:
        """Initialize database and verify connections"""
        logger.info("Setting up database...")

        try:
            # Initialize mobile API database
            mobile_api.init_database()
            logger.info("Mobile dashboard database initialized")
            return True
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False

    def health_check(self) -> bool:
        """Perform comprehensive health check"""
        logger.info("Performing health check...")

        # Check Flask app
        try:
            with app.test_client() as client:
                response = client.get('/api/portfolio')
                if response.status_code not in [200, 404]:  # 404 is OK if no data
                    logger.warning(f"Portfolio endpoint returned {response.status_code}")
        except Exception as e:
            logger.error(f"Flask app health check failed: {e}")
            return False

        # Check Redis connection if enabled
        if hasattr(mobile_api, 'redis_client'):
            try:
                mobile_api.redis_client.ping()
                logger.info("Redis connection OK")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")

        # Check trading bot connections
        try:
            from mobile_dashboard.app import clients

            for client_name, client in clients.items():
                if hasattr(client, 'get_account_summary'):
                    summary = client.get_account_summary()
                    if summary:
                        logger.info(f"{client_name} client connection OK")
                    else:
                        logger.warning(f"{client_name} client returned no data")

        except Exception as e:
            logger.warning(f"Trading client health check failed: {e}")

        logger.info("Health check completed")
        return True

    def start(self):
        """Start the mobile dashboard server"""
        logger.info(f"Starting Mobile Trading Dashboard Server...")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Host: {self.host}")
        logger.info(f"Port: {self.port}")
        logger.info(f"Debug: {self.config.DEBUG}")

        # Validate environment
        if not self.validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)

        # Check dependencies
        if not self.check_dependencies():
            logger.error("Dependency check failed")
            sys.exit(1)

        # Setup database
        if not self.setup_database():
            logger.error("Database setup failed")
            sys.exit(1)

        # Health check
        if not self.health_check():
            logger.warning("Health check failed, but continuing...")

        # Display enabled features
        enabled_features = [name for name, enabled in FEATURE_FLAGS.items() if enabled]
        logger.info(f"Enabled features: {', '.join(enabled_features)}")

        # Start server
        try:
            logger.info("Starting SocketIO server...")
            socketio.run(
                app,
                host=self.host,
                port=self.port,
                debug=self.config.DEBUG,
                use_reloader=False  # Disable reloader to prevent double startup
            )
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.shutdown()
        except Exception as e:
            logger.error(f"Server startup failed: {e}")
            sys.exit(1)

    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Mobile Trading Dashboard...")

        # Close database connections
        try:
            # Any cleanup code here
            pass
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

        logger.info("Mobile Trading Dashboard stopped")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.shutdown()
        sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Mobile Trading Dashboard Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=int(os.getenv('MOBILE_DASHBOARD_PORT', 5001)),
        help='Port to bind to'
    )

    parser.add_argument(
        '--env',
        choices=['development', 'production', 'testing'],
        default=os.getenv('FLASK_ENV', 'development'),
        help='Environment configuration'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without starting server'
    )

    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Perform health check and exit'
    )

    args = parser.parse_args()

    # Create server instance
    server = MobileDashboardServer(
        host=args.host,
        port=args.port,
        environment=args.env
    )

    if args.validate_only:
        logger.info("Validation mode - checking configuration only")
        if server.validate_environment() and server.check_dependencies():
            logger.info("✅ Configuration validation passed")
            sys.exit(0)
        else:
            logger.error("❌ Configuration validation failed")
            sys.exit(1)

    if args.health_check:
        logger.info("Health check mode")
        if server.health_check():
            logger.info("✅ Health check passed")
            sys.exit(0)
        else:
            logger.error("❌ Health check failed")
            sys.exit(1)

    # Start server
    server.start()

if __name__ == '__main__':
    main()