"""
Rate Limiter
============

Intelligent rate limiting for API calls with burst handling
and adaptive rate adjustment.
"""

import time
import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import deque


class RateLimiter:
    """
    Token bucket rate limiter with adaptive burst handling

    Features:
    - Token bucket algorithm
    - Burst handling
    - Adaptive rate adjustment
    - Health monitoring
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None,
        adaptive: bool = True
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or max(10, requests_per_minute // 6)
        self.adaptive = adaptive

        # Token bucket
        self.tokens = float(self.burst_size)
        self.max_tokens = float(self.burst_size)
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        self.last_refill = time.time()

        # Request tracking
        self.request_times = deque(maxlen=requests_per_minute)
        self.total_requests = 0
        self.blocked_requests = 0

        # Health monitoring
        self.consecutive_blocks = 0
        self.adaptive_multiplier = 1.0

        self.logger = logging.getLogger("altdata.ratelimiter")

    async def acquire(self) -> None:
        """
        Acquire a token for making a request
        Blocks until token is available
        """
        while not self._try_acquire():
            # Calculate wait time
            wait_time = self._calculate_wait_time()

            # Log if significant delay
            if wait_time > 1.0:
                self.logger.debug(f"Rate limit hit, waiting {wait_time:.1f}s")

            await asyncio.sleep(wait_time)

        # Record successful request
        self.request_times.append(time.time())
        self.total_requests += 1
        self.consecutive_blocks = 0

    def _try_acquire(self) -> bool:
        """Try to acquire a token without blocking"""
        self._refill_tokens()

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True

        self.blocked_requests += 1
        self.consecutive_blocks += 1

        # Adaptive rate reduction if many consecutive blocks
        if self.adaptive and self.consecutive_blocks > 5:
            self._reduce_rate()

        return False

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate * self.adaptive_multiplier
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)

        self.last_refill = now

    def _calculate_wait_time(self) -> float:
        """Calculate optimal wait time"""
        # Base wait time for one token
        base_wait = 1.0 / (self.refill_rate * self.adaptive_multiplier)

        # Add small random jitter to prevent thundering herd
        import random
        jitter = random.uniform(0.1, 0.3)

        return base_wait + jitter

    def _reduce_rate(self) -> None:
        """Adaptively reduce rate due to consecutive blocks"""
        if self.adaptive_multiplier > 0.5:
            self.adaptive_multiplier *= 0.9
            self.logger.info(
                f"Adaptive rate reduction: {self.adaptive_multiplier:.2f}x"
            )

    def remaining_requests(self) -> int:
        """Get remaining requests in current minute"""
        now = time.time()
        # Count requests in last minute
        recent_requests = sum(
            1 for req_time in self.request_times
            if now - req_time < 60
        )
        return max(0, self.requests_per_minute - recent_requests)

    def get_stats(self) -> Dict[str, any]:
        """Get rate limiter statistics"""
        success_rate = (
            (self.total_requests - self.blocked_requests) / self.total_requests
            if self.total_requests > 0 else 1.0
        )

        return {
            'requests_per_minute': self.requests_per_minute,
            'current_tokens': round(self.tokens, 2),
            'max_tokens': self.max_tokens,
            'total_requests': self.total_requests,
            'blocked_requests': self.blocked_requests,
            'success_rate': success_rate,
            'remaining_requests': self.remaining_requests(),
            'adaptive_multiplier': self.adaptive_multiplier,
            'consecutive_blocks': self.consecutive_blocks
        }

    def reset(self) -> None:
        """Reset rate limiter state"""
        self.tokens = float(self.max_tokens)
        self.request_times.clear()
        self.total_requests = 0
        self.blocked_requests = 0
        self.consecutive_blocks = 0
        self.adaptive_multiplier = 1.0
        self.last_refill = time.time()

        self.logger.info("Rate limiter reset")