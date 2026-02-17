"""
Liquidation Data Scraper
=========================
Fetches BTC/ETH long/short liquidation data from Coinglass API (free tier).
Liquidation cascades are the highest-signal event in crypto.

Author: Trading Bot Arsenal
Created: February 2026 | V7 Crystal Ball Upgrade
"""

import os
import time
import json
import logging
import threading
from typing import Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger('LiquidationScraper')

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'crypto_cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'liquidations.json')
CACHE_TTL = 300  # 5 minutes


class LiquidationScraper:
    """
    Fetches liquidation data from Coinglass public API.

    Signals:
    - Large long liquidations → more downside (bearish)
    - Large short liquidations → short squeeze (bullish)
    - Long dominance > 65% → bearish pressure
    - Short dominance > 65% → bullish pressure
    """

    def __init__(self):
        self._cache = {}
        self._last_fetch = 0
        self._lock = threading.Lock()

    def get_latest(self) -> Dict:
        """Get latest liquidation data. Uses cache if fresh."""
        if time.time() - self._last_fetch < CACHE_TTL:
            with self._lock:
                if self._cache:
                    return self._cache.copy()

        # Try loading from cache file
        data = self._load_cache()
        if data and time.time() - data.get('_fetched_at', 0) < CACHE_TTL:
            with self._lock:
                self._cache = data
                self._last_fetch = data.get('_fetched_at', time.time())
            return data

        # Fetch fresh data
        data = self._fetch()
        if data:
            with self._lock:
                self._cache = data
                self._last_fetch = time.time()
            self._save_cache(data)
        return data or {}

    def _fetch(self) -> Optional[Dict]:
        """Fetch liquidation data from Coinglass public API."""
        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            '_fetched_at': time.time()
        }

        try:
            import httpx

            # Coinglass public liquidation data
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'TradingBot/1.0'
            }

            # Try Coinglass public API endpoint
            try:
                resp = httpx.get(
                    'https://open-api.coinglass.com/public/v2/liquidation_history?time_type=all&symbol=BTC',
                    headers=headers,
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('success') and data.get('data'):
                        liq_data = data['data']
                        if isinstance(liq_data, list) and liq_data:
                            latest = liq_data[0]
                            result['btc'] = {
                                'long_usd': latest.get('longVolUsd', 0),
                                'short_usd': latest.get('shortVolUsd', 0),
                                'total_usd': latest.get('longVolUsd', 0) + latest.get('shortVolUsd', 0),
                            }
            except Exception as e:
                logger.debug(f"Coinglass BTC liquidation fetch failed: {e}")

            # Fallback: estimate from CCXT exchange data
            if 'btc' not in result:
                result.update(self._estimate_from_ccxt())

            # Calculate dominance
            for coin in ['btc', 'eth']:
                if coin in result:
                    total = result[coin].get('total_usd', 0)
                    if total > 0:
                        long_pct = (result[coin].get('long_usd', 0) / total) * 100
                        result[coin]['long_pct'] = round(long_pct, 1)
                        result[coin]['short_pct'] = round(100 - long_pct, 1)

            # Overall long dominance across all coins
            total_long = sum(result.get(c, {}).get('long_usd', 0) for c in ['btc', 'eth'])
            total_all = sum(result.get(c, {}).get('total_usd', 0) for c in ['btc', 'eth'])
            if total_all > 0:
                result['long_dominance'] = round((total_long / total_all) * 100, 1)
            else:
                result['long_dominance'] = 50.0  # Neutral default

            return result

        except ImportError:
            logger.debug("httpx not available for liquidation scraping")
            return None
        except Exception as e:
            logger.debug(f"Liquidation fetch failed: {e}")
            return None

    def _estimate_from_ccxt(self) -> Dict:
        """Fallback: estimate liquidation pressure from CCXT open interest data."""
        try:
            import ccxt
            exchange = ccxt.bybit({'enableRateLimit': True})

            estimates = {}
            for symbol_pair, coin in [('BTC/USDT:USDT', 'btc'), ('ETH/USDT:USDT', 'eth')]:
                try:
                    oi = exchange.fetch_open_interest(symbol_pair)
                    if oi:
                        oi_value = oi.get('openInterestAmount', 0)
                        # Can't get long/short split from OI alone, but high OI = high liq risk
                        estimates[coin] = {
                            'open_interest': oi_value,
                            'long_usd': 0,
                            'short_usd': 0,
                            'total_usd': 0,
                            'note': 'estimated_from_oi'
                        }
                except Exception:
                    pass

            return estimates
        except ImportError:
            return {}
        except Exception:
            return {}

    def _load_cache(self) -> Optional[Dict]:
        """Load from disk cache."""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _save_cache(self, data: Dict):
        """Save to disk cache."""
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(CACHE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save liquidation cache: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = LiquidationScraper()
    data = scraper.get_latest()
    print(json.dumps(data, indent=2))
