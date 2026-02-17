"""
LLM Client - Multi-provider with automatic fallback and caching
Primary: DeepSeek (~$0.001/call)
Backup: Claude Haiku (~$0.003/call)

Author: Trading Bot Arsenal
Created: January 2026
"""

import os
import sys
import json
import time
import hashlib
import sqlite3
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger('LLMClient')


class LLMProvider(Enum):
    DEEPSEEK = "deepseek"
    CLAUDE_HAIKU = "claude_haiku"
    LOCAL_OLLAMA = "ollama"


@dataclass
class LLMResponse:
    """Response from LLM query"""
    content: str
    provider: LLMProvider
    latency_ms: float
    tokens_used: int
    cached: bool
    cost_estimate: float


class LLMClient:
    """
    Multi-provider LLM client with caching and fallback.

    Features:
    - DeepSeek as primary (cheapest)
    - Claude Haiku as backup
    - SQLite response caching (1 hour TTL)
    - Cost tracking
    - Async HTTP with httpx
    """

    PRICING = {
        LLMProvider.DEEPSEEK: {"input": 0.00014, "output": 0.00028},  # per 1K tokens
        LLMProvider.CLAUDE_HAIKU: {"input": 0.00025, "output": 0.00125},
        LLMProvider.LOCAL_OLLAMA: {"input": 0, "output": 0},
    }

    def __init__(
        self,
        deepseek_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        cache_db_path: str = None,
        cache_ttl_seconds: int = 3600,  # 1 hour cache
        primary_provider: LLMProvider = LLMProvider.DEEPSEEK
    ):
        self.deepseek_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        self.anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        # Set cache path
        if cache_db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cache_db_path = os.path.join(base_dir, "cache", "response_cache.db")

        self.cache_path = cache_db_path
        self.cache_ttl = cache_ttl_seconds
        self.primary = primary_provider

        # Initialize cache
        os.makedirs(os.path.dirname(cache_db_path), exist_ok=True)
        self._init_cache()

        # Track costs
        self.session_costs = 0.0
        self.session_calls = 0
        self.cache_hits = 0

        # Persistent HTTP clients (connection pooling, TLS session reuse)
        self._http_clients = {}

        # Check API keys
        if not self.deepseek_key:
            logger.warning("DEEPSEEK_API_KEY not found - DeepSeek provider unavailable")
        if not self.anthropic_key:
            logger.warning("ANTHROPIC_API_KEY not found - Claude backup unavailable")

        logger.info(f"LLMClient initialized: primary={primary_provider.value}, cache_ttl={cache_ttl_seconds}s")

    def _init_cache(self):
        """Initialize SQLite cache"""
        conn = sqlite3.connect(self.cache_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS response_cache (
                hash TEXT PRIMARY KEY,
                response TEXT,
                provider TEXT,
                timestamp REAL,
                tokens_used INTEGER
            )
        ''')
        conn.commit()
        conn.close()

    def _get_cache_key(self, prompt: str, system: str) -> str:
        """Generate cache key from prompt"""
        content = f"{system}|{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if response is cached and valid"""
        try:
            conn = sqlite3.connect(self.cache_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response, provider, timestamp, tokens_used FROM response_cache WHERE hash = ?",
                (cache_key,)
            )
            row = cursor.fetchone()
            conn.close()

            if row and (time.time() - row[2]) < self.cache_ttl:
                return {
                    "response": row[0],
                    "provider": row[1],
                    "tokens_used": row[3]
                }
        except Exception as e:
            logger.debug(f"Cache check error: {e}")
        return None

    def _save_cache(self, cache_key: str, response: str, provider: str, tokens: int):
        """Save response to cache"""
        try:
            conn = sqlite3.connect(self.cache_path)
            conn.execute(
                "INSERT OR REPLACE INTO response_cache VALUES (?, ?, ?, ?, ?)",
                (cache_key, response, provider, time.time(), tokens)
            )
            # Prune stale entries (>7 days old) and cap at 10K entries
            conn.execute("DELETE FROM response_cache WHERE ? - timestamp > 604800", (time.time(),))
            count = conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0]
            if count > 10000:
                conn.execute(
                    "DELETE FROM response_cache WHERE hash IN "
                    "(SELECT hash FROM response_cache ORDER BY timestamp ASC LIMIT ?)",
                    (count - 10000,)
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Cache save error: {e}")

    async def query(
        self,
        prompt: str,
        system_prompt: str = "You are a trading analysis assistant.",
        max_tokens: int = 500,
        temperature: float = 0.1,  # Low for consistency
        use_cache: bool = True
    ) -> LLMResponse:
        """Query LLM with automatic caching and fallback"""

        # Check cache first
        cache_key = self._get_cache_key(prompt, system_prompt)
        if use_cache:
            cached = self._check_cache(cache_key)
            if cached:
                self.cache_hits += 1
                return LLMResponse(
                    content=cached["response"],
                    provider=LLMProvider(cached["provider"]),
                    latency_ms=0,
                    tokens_used=cached["tokens_used"],
                    cached=True,
                    cost_estimate=0
                )

        # Try primary provider
        start = time.time()
        response = None
        error_msg = None

        try:
            if self.primary == LLMProvider.DEEPSEEK and self.deepseek_key:
                response = await self._query_deepseek(prompt, system_prompt, max_tokens, temperature)
            elif self.primary == LLMProvider.CLAUDE_HAIKU and self.anthropic_key:
                response = await self._query_claude(prompt, system_prompt, max_tokens, temperature)
            elif self.primary == LLMProvider.LOCAL_OLLAMA:
                response = await self._query_ollama(prompt, system_prompt, max_tokens, temperature)
            else:
                raise ValueError(f"Primary provider {self.primary} not available")

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Primary provider failed: {e}")

            # Fallback chain: DeepSeek → Haiku → Local Ollama
            try:
                if self.primary == LLMProvider.DEEPSEEK and self.anthropic_key:
                    logger.info("Falling back to Claude Haiku")
                    response = await self._query_claude(prompt, system_prompt, max_tokens, temperature)
                elif self.primary == LLMProvider.CLAUDE_HAIKU and self.deepseek_key:
                    logger.info("Falling back to DeepSeek")
                    response = await self._query_deepseek(prompt, system_prompt, max_tokens, temperature)
                else:
                    raise ValueError("Cloud providers unavailable")
            except Exception as e2:
                logger.warning(f"Cloud fallback failed: {e2}, trying local Ollama")
                # Last resort: local Ollama (free, works offline)
                try:
                    response = await self._query_ollama(prompt, system_prompt, max_tokens, temperature)
                except Exception as e3:
                    logger.error(f"All providers failed: primary={error_msg}, cloud={e2}, local={e3}")
                    raise ValueError(f"All providers failed: primary={error_msg}, cloud={e2}, local={e3}")

        latency = (time.time() - start) * 1000

        # Calculate cost
        cost = self._estimate_cost(response["provider"], response["tokens"])
        self.session_costs += cost
        self.session_calls += 1

        # Cache response
        if use_cache:
            self._save_cache(cache_key, response["content"], response["provider"].value, response["tokens"])

        return LLMResponse(
            content=response["content"],
            provider=response["provider"],
            latency_ms=latency,
            tokens_used=response["tokens"],
            cached=False,
            cost_estimate=cost
        )

    async def _get_client(self, provider: str) -> 'httpx.AsyncClient':
        """Get or create httpx.AsyncClient per event loop (thread-safe for ThreadPoolExecutor)."""
        import httpx
        import asyncio

        loop_id = id(asyncio.get_event_loop())
        key = f"{provider}_{loop_id}"

        if key not in self._http_clients or self._http_clients[key].is_closed:
            self._http_clients[key] = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                http2=True
            )
        return self._http_clients[key]

    async def _query_deepseek(self, prompt: str, system: str, max_tokens: int, temp: float) -> Dict:
        """Query DeepSeek API"""
        client = await self._get_client("deepseek")

        response = await client.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {self.deepseek_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temp
            }
        )

        if response.status_code != 200:
            raise ValueError(f"DeepSeek API error: {response.status_code} - {response.text}")

        data = response.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "provider": LLMProvider.DEEPSEEK,
            "tokens": data["usage"]["total_tokens"]
        }

    async def _query_claude(self, prompt: str, system: str, max_tokens: int, temp: float) -> Dict:
        """Query Claude Haiku API"""
        client = await self._get_client("claude")

        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}]
            }
        )

        if response.status_code != 200:
            raise ValueError(f"Claude API error: {response.status_code} - {response.text}")

        data = response.json()
        return {
            "content": data["content"][0]["text"],
            "provider": LLMProvider.CLAUDE_HAIKU,
            "tokens": data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
        }

    async def _query_ollama(self, prompt: str, system: str, max_tokens: int, temp: float) -> Dict:
        """Query local Ollama (Mistral 7B)"""
        client = await self._get_client("ollama")

        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:7b",
                "prompt": f"{system}\n\nUser: {prompt}\n\nAssistant:",
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": temp}
            },
            timeout=60.0
        )

        if response.status_code != 200:
            raise ValueError(f"Ollama API error: {response.status_code}")

        data = response.json()
        return {
            "content": data["response"],
            "provider": LLMProvider.LOCAL_OLLAMA,
            "tokens": len(prompt.split()) + len(data["response"].split())  # Approximate
        }

    def _estimate_cost(self, provider: LLMProvider, tokens: int) -> float:
        """Estimate cost in USD"""
        pricing = self.PRICING[provider]
        # Assume 50/50 input/output split
        return (tokens / 2000) * (pricing["input"] + pricing["output"])

    def get_session_stats(self) -> Dict:
        """Get session statistics"""
        return {
            "total_calls": self.session_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hits / max(1, self.session_calls + self.cache_hits), 3),
            "total_cost_usd": round(self.session_costs, 4),
            "avg_cost_per_call": round(self.session_costs / max(1, self.session_calls), 5)
        }

    def clear_cache(self):
        """Clear the response cache"""
        conn = sqlite3.connect(self.cache_path)
        conn.execute("DELETE FROM response_cache")
        conn.commit()
        conn.close()
        logger.info("Cache cleared")


# Synchronous wrapper for non-async code
def query_sync(
    client: LLMClient,
    prompt: str,
    system_prompt: str = "You are a trading analysis assistant.",
    max_tokens: int = 500,
    temperature: float = 0.1,
    use_cache: bool = True
) -> LLMResponse:
    """Synchronous wrapper for LLMClient.query()"""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        client.query(prompt, system_prompt, max_tokens, temperature, use_cache)
    )


if __name__ == "__main__":
    # Test the client
    import asyncio

    async def test():
        client = LLMClient()
        print(f"Testing LLMClient...")
        print(f"DeepSeek key: {'Set' if client.deepseek_key else 'NOT SET'}")
        print(f"Anthropic key: {'Set' if client.anthropic_key else 'NOT SET'}")

        if client.deepseek_key or client.anthropic_key:
            response = await client.query("Say 'Hello, trading bot!' in exactly those words.")
            print(f"\nResponse: {response.content}")
            print(f"Provider: {response.provider.value}")
            print(f"Latency: {response.latency_ms:.0f}ms")
            print(f"Tokens: {response.tokens_used}")
            print(f"Cost: ${response.cost_estimate:.5f}")
            print(f"Cached: {response.cached}")

            # Test cache
            print("\nTesting cache (same query)...")
            response2 = await client.query("Say 'Hello, trading bot!' in exactly those words.")
            print(f"Cached: {response2.cached}")
            print(f"Latency: {response2.latency_ms:.0f}ms")

            print(f"\nSession stats: {client.get_session_stats()}")
        else:
            print("\nNo API keys set. Add DEEPSEEK_API_KEY to .env file.")

    asyncio.run(test())
