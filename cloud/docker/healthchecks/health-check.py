#!/usr/bin/env python3
"""
ENTERPRISE HEALTH CHECK SYSTEM
============================

Comprehensive health monitoring for trading bot microservices.
Checks system health, database connections, API endpoints, and critical services.

Exit codes:
- 0: Healthy
- 1: Unhealthy
- 2: Warning (degraded but functional)

Features:
- Multi-service health validation
- Database connection testing
- API endpoint verification
- Resource utilization monitoring
- Trading system specific checks
"""

import sys
import os
import time
import json
import sqlite3
import requests
import psutil
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Suppress urllib3 warnings for self-signed certificates
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class HealthChecker:
    def __init__(self):
        self.logger = self._setup_logging()
        self.config = self._load_config()
        self.checks = []

    def _setup_logging(self):
        """Configure logging for health checks."""
        logger = logging.getLogger('health-check')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _load_config(self) -> Dict:
        """Load health check configuration."""
        default_config = {
            "max_memory_percent": 90,
            "max_cpu_percent": 95,
            "max_disk_percent": 90,
            "required_services": ["master_orchestrator", "dashboard"],
            "database_timeout": 5,
            "api_timeout": 10,
            "critical_endpoints": [
                "http://localhost:5000/health",
                "http://localhost:8080/api/health"
            ]
        }

        config_path = "/app/config/health-config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}")

        return default_config

    def check_system_resources(self) -> Tuple[bool, str]:
        """Check system resource utilization."""
        try:
            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > self.config["max_memory_percent"]:
                return False, f"High memory usage: {memory.percent}%"

            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config["max_cpu_percent"]:
                return False, f"High CPU usage: {cpu_percent}%"

            # Disk check
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > self.config["max_disk_percent"]:
                return False, f"High disk usage: {disk_percent:.1f}%"

            return True, f"Resources OK (CPU: {cpu_percent}%, MEM: {memory.percent}%, DISK: {disk_percent:.1f}%)"

        except Exception as e:
            return False, f"Resource check failed: {e}"

    def check_database_connections(self) -> Tuple[bool, str]:
        """Check database connectivity."""
        try:
            # SQLite database check
            db_path = "/app/data/trading_bot.db"
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path, timeout=self.config["database_timeout"])
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                conn.close()

            # PostgreSQL check (if configured)
            pg_host = os.getenv("POSTGRES_HOST")
            if pg_host:
                import psycopg2
                conn_string = (
                    f"host={pg_host} "
                    f"port={os.getenv('POSTGRES_PORT', '5432')} "
                    f"dbname={os.getenv('POSTGRES_DB', 'tradingbot')} "
                    f"user={os.getenv('POSTGRES_USER', 'tradingbot')} "
                    f"password={os.getenv('POSTGRES_PASSWORD', '')}"
                )
                conn = psycopg2.connect(conn_string, connect_timeout=self.config["database_timeout"])
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                conn.close()

            # Redis check (if configured)
            redis_host = os.getenv("REDIS_HOST")
            if redis_host:
                import redis
                r = redis.Redis(
                    host=redis_host,
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    socket_connect_timeout=self.config["database_timeout"]
                )
                r.ping()

            return True, "All databases accessible"

        except Exception as e:
            return False, f"Database check failed: {e}"

    def check_api_endpoints(self) -> Tuple[bool, str]:
        """Check critical API endpoints."""
        failed_endpoints = []

        for endpoint in self.config["critical_endpoints"]:
            try:
                response = requests.get(
                    endpoint,
                    timeout=self.config["api_timeout"],
                    verify=False  # Allow self-signed certificates in dev
                )

                if response.status_code not in [200, 201]:
                    failed_endpoints.append(f"{endpoint} -> {response.status_code}")

            except Exception as e:
                failed_endpoints.append(f"{endpoint} -> {str(e)}")

        if failed_endpoints:
            return False, f"Failed endpoints: {', '.join(failed_endpoints)}"

        return True, f"All {len(self.config['critical_endpoints'])} endpoints healthy"

    def check_trading_system(self) -> Tuple[bool, str]:
        """Check trading system specific health."""
        try:
            issues = []

            # Check log files for recent errors
            log_dir = "/app/logs"
            if os.path.exists(log_dir):
                error_count = 0
                cutoff_time = datetime.now() - timedelta(minutes=5)

                for log_file in os.listdir(log_dir):
                    if log_file.endswith('.log'):
                        log_path = os.path.join(log_dir, log_file)
                        try:
                            with open(log_path, 'r') as f:
                                for line in f:
                                    if 'ERROR' in line or 'CRITICAL' in line:
                                        error_count += 1
                        except Exception as e:
                            continue

                if error_count > 10:  # More than 10 errors in 5 minutes
                    issues.append(f"High error rate: {error_count} errors")

            # Check process files
            pid_dir = "/app/pids"
            if os.path.exists(pid_dir):
                for service in self.config["required_services"]:
                    pid_file = os.path.join(pid_dir, f"{service}.pid")
                    if not os.path.exists(pid_file):
                        issues.append(f"Missing PID file for {service}")
                        continue

                    try:
                        with open(pid_file, 'r') as f:
                            pid = int(f.read().strip())

                        if not psutil.pid_exists(pid):
                            issues.append(f"Process {service} (PID {pid}) not running")
                    except Exception as e:
                        issues.append(f"Invalid PID file for {service}: {e}")

            if issues:
                return False, f"Trading system issues: {'; '.join(issues)}"

            return True, "Trading system healthy"

        except Exception as e:
            return False, f"Trading system check failed: {e}"

    def check_market_connectivity(self) -> Tuple[bool, str]:
        """Check connectivity to trading venues."""
        try:
            # Test key trading APIs
            test_endpoints = [
                ("Alpaca", "https://paper-api.alpaca.markets/v2/account"),
                ("Yahoo Finance", "https://query1.finance.yahoo.com/v8/finance/chart/SPY"),
                ("Alpha Vantage", "https://www.alphavantage.co/"),
            ]

            failed_apis = []

            for name, url in test_endpoints:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code >= 500:  # Server errors only
                        failed_apis.append(f"{name} ({response.status_code})")
                except requests.RequestException:
                    failed_apis.append(f"{name} (timeout/error)")

            if len(failed_apis) >= len(test_endpoints):  # All failed
                return False, f"No market connectivity: {', '.join(failed_apis)}"
            elif failed_apis:
                return True, f"Partial connectivity issues: {', '.join(failed_apis)}"

            return True, "Market connectivity OK"

        except Exception as e:
            return False, f"Market connectivity check failed: {e}"

    def run_all_checks(self) -> int:
        """Run all health checks and return exit code."""
        self.logger.info("Starting health check...")

        checks = [
            ("System Resources", self.check_system_resources),
            ("Database Connections", self.check_database_connections),
            ("API Endpoints", self.check_api_endpoints),
            ("Trading System", self.check_trading_system),
            ("Market Connectivity", self.check_market_connectivity),
        ]

        failed_checks = []
        warning_checks = []

        for check_name, check_func in checks:
            try:
                is_healthy, message = check_func()

                if is_healthy:
                    self.logger.info(f"✓ {check_name}: {message}")
                else:
                    self.logger.error(f"✗ {check_name}: {message}")

                    # Determine if this is critical or warning
                    if check_name in ["System Resources", "Database Connections"]:
                        failed_checks.append(f"{check_name}: {message}")
                    else:
                        warning_checks.append(f"{check_name}: {message}")

            except Exception as e:
                error_msg = f"{check_name}: Unexpected error - {e}"
                self.logger.error(f"✗ {error_msg}")
                failed_checks.append(error_msg)

        # Determine overall health status
        if failed_checks:
            self.logger.error(f"UNHEALTHY - Critical failures: {len(failed_checks)}")
            for failure in failed_checks:
                self.logger.error(f"  - {failure}")
            return 1  # Unhealthy
        elif warning_checks:
            self.logger.warning(f"DEGRADED - Non-critical issues: {len(warning_checks)}")
            for warning in warning_checks:
                self.logger.warning(f"  - {warning}")
            return 2  # Warning
        else:
            self.logger.info("HEALTHY - All checks passed")
            return 0  # Healthy

def main():
    """Main health check entry point."""
    try:
        checker = HealthChecker()
        exit_code = checker.run_all_checks()
        sys.exit(exit_code)
    except Exception as e:
        print(f"CRITICAL: Health check system failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()