"""
API Retry Utilities
===================
Reusable retry logic with exponential backoff for all trading bots.

Features:
- Configurable retry attempts
- Exponential backoff with jitter
- Logging of retry attempts
- Optional Telegram alerts on final failure
- Circuit breaker integration

Usage:
    from utils.api_retry import retry_api_call, APIRetryConfig
    
    @retry_api_call()
    def make_api_request():
        return requests.get(url)
    
    # Or with custom config:
    @retry_api_call(max_attempts=5, base_delay=2.0)
    def critical_order():
        return place_order(...)

Author: Trading Bot Arsenal
Created: January 2026
"""

import time
import random
import logging
import functools
import threading
from typing import Callable, Optional, Tuple, Type, Union
from dataclasses import dataclass
import requests

logger = logging.getLogger('APIRetry')


# Exceptions that should trigger a retry
RETRYABLE_EXCEPTIONS = (
    requests.exceptions.RequestException,
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
    ConnectionError,
    TimeoutError,
    OSError,  # Covers network-level errors
)

# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {
    408,  # Request Timeout
    429,  # Too Many Requests (rate limit)
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}


@dataclass
class APIRetryConfig:
    """Configuration for API retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    jitter_range: Tuple[float, float] = (0.5, 1.5)
    retryable_exceptions: Tuple[Type[Exception], ...] = RETRYABLE_EXCEPTIONS
    retryable_status_codes: set = None  # Will use default if None
    log_retries: bool = True
    raise_on_final_failure: bool = True
    
    def __post_init__(self):
        if self.retryable_status_codes is None:
            self.retryable_status_codes = RETRYABLE_STATUS_CODES


# Default config instance
DEFAULT_CONFIG = APIRetryConfig()


def calculate_delay(
    attempt: int,
    config: APIRetryConfig
) -> float:
    """
    Calculate delay before next retry using exponential backoff.
    
    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    # Exponential backoff
    delay = config.base_delay * (config.exponential_base ** attempt)
    
    # Cap at max delay
    delay = min(delay, config.max_delay)
    
    # Add jitter to prevent thundering herd
    if config.jitter:
        jitter_multiplier = random.uniform(*config.jitter_range)
        delay *= jitter_multiplier
    
    return delay


def should_retry_response(response: requests.Response, config: APIRetryConfig) -> bool:
    """Check if response status code warrants a retry."""
    return response.status_code in config.retryable_status_codes


def retry_api_call(
    max_attempts: int = None,
    base_delay: float = None,
    max_delay: float = None,
    config: APIRetryConfig = None,
    on_retry: Callable[[int, Exception], None] = None,
    on_final_failure: Callable[[Exception], None] = None,
):
    """
    Decorator for retrying API calls with exponential backoff.
    
    Args:
        max_attempts: Override default max attempts
        base_delay: Override default base delay
        max_delay: Override default max delay  
        config: Full config object (overrides individual params)
        on_retry: Callback called on each retry (attempt_num, exception)
        on_final_failure: Callback called when all retries exhausted
        
    Returns:
        Decorated function
        
    Example:
        @retry_api_call(max_attempts=5)
        def fetch_data():
            return requests.get(url, timeout=10)
    """
    # Build config
    if config is None:
        config = APIRetryConfig(
            max_attempts=max_attempts or DEFAULT_CONFIG.max_attempts,
            base_delay=base_delay or DEFAULT_CONFIG.base_delay,
            max_delay=max_delay or DEFAULT_CONFIG.max_delay,
        )
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    
                    # Check if result is a Response object with retryable status
                    if isinstance(result, requests.Response):
                        if should_retry_response(result, config):
                            if attempt < config.max_attempts - 1:
                                delay = calculate_delay(attempt, config)
                                if config.log_retries:
                                    logger.warning(
                                        f"{func.__name__}: HTTP {result.status_code}, "
                                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{config.max_attempts})"
                                    )
                                time.sleep(delay)
                                continue
                            else:
                                # Final attempt failed
                                result.raise_for_status()
                    
                    # Success!
                    if attempt > 0 and config.log_retries:
                        logger.info(f"{func.__name__}: Succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = calculate_delay(attempt, config)
                        
                        if config.log_retries:
                            logger.warning(
                                f"{func.__name__}: {type(e).__name__}: {str(e)[:100]}, "
                                f"retrying in {delay:.1f}s (attempt {attempt + 1}/{config.max_attempts})"
                            )
                        
                        if on_retry:
                            on_retry(attempt + 1, e)
                        
                        time.sleep(delay)
                    else:
                        # Final attempt
                        if config.log_retries:
                            logger.error(
                                f"{func.__name__}: All {config.max_attempts} attempts failed. "
                                f"Last error: {type(e).__name__}: {str(e)[:200]}"
                            )
                        
                        if on_final_failure:
                            on_final_failure(e)
                        
                        if config.raise_on_final_failure:
                            raise
                        
                        return None
            
            # Shouldn't reach here, but just in case
            if last_exception and config.raise_on_final_failure:
                raise last_exception
            return None
        
        return wrapper
    return decorator


class RetrySession(requests.Session):
    """
    A requests.Session subclass with built-in retry logic.
    
    Usage:
        session = RetrySession(max_attempts=3, timeout=10)
        response = session.get('https://api.example.com/data')
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        timeout: Union[float, Tuple[float, float]] = (5, 30),
        **kwargs
    ):
        super().__init__()
        self.retry_config = APIRetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
        )
        self.default_timeout = timeout
        
        # Apply any session kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def request(self, method, url, **kwargs):
        """Override request to add retry logic and default timeout."""
        # Add default timeout if not specified
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.default_timeout
        
        @retry_api_call(config=self.retry_config)
        def _do_request():
            return super(RetrySession, self).request(method, url, **kwargs)
        
        return _do_request()


# Convenience function for one-off retries
def with_retry(
    func: Callable,
    *args,
    max_attempts: int = 3,
    **kwargs
):
    """
    Execute a function with retry logic.
    
    Usage:
        result = with_retry(requests.get, url, timeout=10, max_attempts=5)
    """
    @retry_api_call(max_attempts=max_attempts)
    def wrapper():
        return func(*args, **kwargs)
    
    return wrapper()


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker to prevent repeated failures from overwhelming the system.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests blocked
    - HALF_OPEN: Testing if service recovered
    
    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        if breaker.can_execute():
            try:
                result = make_api_call()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
    """
    
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0
        self._half_open_calls = 0
        self._lock = threading.Lock()
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        if self._state == self.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
                logger.info(f"CircuitBreaker[{self.name}]: OPEN -> HALF_OPEN")
        
        return self._state
    
    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        state = self.state  # This may transition OPEN -> HALF_OPEN
        
        if state == self.CLOSED:
            return True
        elif state == self.HALF_OPEN:
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
        else:  # OPEN
            return False
    
    def record_success(self):
        """Record a successful call."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(f"CircuitBreaker[{self.name}]: HALF_OPEN -> CLOSED (recovered)")
            elif self._state == self.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def record_failure(self):
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._state = self.OPEN
                self._success_count = 0
                logger.warning(f"CircuitBreaker[{self.name}]: HALF_OPEN -> OPEN (failure during recovery)")
            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN
                    logger.warning(
                        f"CircuitBreaker[{self.name}]: CLOSED -> OPEN "
                        f"(failures: {self._failure_count}/{self.failure_threshold})"
                    )
    
    def reset(self):
        """Manually reset the circuit breaker."""
        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        logger.info(f"CircuitBreaker[{self.name}]: Manually reset to CLOSED")
    
    def get_status(self) -> dict:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "time_since_last_failure": time.time() - self._last_failure_time if self._last_failure_time else None,
        }


# Global circuit breakers for each platform
CIRCUIT_BREAKERS = {
    "alpaca": CircuitBreaker(name="alpaca", failure_threshold=5, recovery_timeout=60),
    "kalshi": CircuitBreaker(name="kalshi", failure_threshold=5, recovery_timeout=60),
    "oanda": CircuitBreaker(name="oanda", failure_threshold=5, recovery_timeout=60),
}


def get_circuit_breaker(platform: str) -> CircuitBreaker:
    """Get or create circuit breaker for a platform."""
    if platform not in CIRCUIT_BREAKERS:
        CIRCUIT_BREAKERS[platform] = CircuitBreaker(name=platform)
    return CIRCUIT_BREAKERS[platform]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import requests
    
    print("=" * 60)
    print("API RETRY UTILITIES TEST")
    print("=" * 60)
    
    # Test 1: Basic retry decorator
    print("\n[1] Testing retry decorator...")
    
    @retry_api_call(max_attempts=3, base_delay=0.5)
    def fetch_example():
        return requests.get("https://httpstat.us/503", timeout=5)
    
    try:
        result = fetch_example()
        print(f"Result: {result.status_code}")
    except Exception as e:
        print(f"Failed after retries: {e}")
    
    # Test 2: Circuit breaker
    print("\n[2] Testing circuit breaker...")
    
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
    
    for i in range(5):
        if breaker.can_execute():
            print(f"  Attempt {i+1}: Executing...")
            breaker.record_failure()
        else:
            print(f"  Attempt {i+1}: BLOCKED (circuit open)")
        print(f"  State: {breaker.state}")
    
    print("\n[3] Waiting for recovery...")
    time.sleep(6)
    print(f"  State after timeout: {breaker.state}")
    
    if breaker.can_execute():
        print("  Half-open test succeeded")
        breaker.record_success()
        breaker.record_success()
        breaker.record_success()
        print(f"  Final state: {breaker.state}")
    
    print("\n" + "=" * 60)
    print("Tests complete!")
