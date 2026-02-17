"""
On-Chain Metrics Scraper
=========================
Fetches exchange inflows/outflows and active addresses using free APIs.
Large exchange inflows = sell pressure (bearish).
Growing active addresses = network health (bullish).

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

logger = logging.getLogger('OnchainScraper')

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'crypto_cache')
CACHE_FILE = os.path.join(CACHE_DIR, 'onchain.json')
CACHE_TTL = 600  # 10 minutes


class OnchainScraper:
    """
    Fetches on-chain metrics from free APIs.

    Sources:
    - Blockchain.com (BTC data, no key needed)
    - Blockchair (multi-chain, free tier)

    Signals:
    - Exchange net flow negative = outflow = accumulation = bullish
    - Exchange net flow positive = inflow = sell pressure = bearish
    - Active addresses growing = bullish
    - Active addresses declining = bearish
    """

    def __init__(self):
        self._cache = {}
        self._last_fetch = 0
        self._lock = threading.Lock()

    def get_latest(self) -> Dict:
        """Get latest on-chain data. Uses cache if fresh."""
        if time.time() - self._last_fetch < CACHE_TTL:
            with self._lock:
                if self._cache:
                    return self._cache.copy()

        data = self._load_cache()
        if data and time.time() - data.get('_fetched_at', 0) < CACHE_TTL:
            with self._lock:
                self._cache = data
                self._last_fetch = data.get('_fetched_at', time.time())
            return data

        data = self._fetch()
        if data:
            with self._lock:
                self._cache = data
                self._last_fetch = time.time()
            self._save_cache(data)
        return data or {}

    def _fetch(self) -> Optional[Dict]:
        """Fetch on-chain data from free APIs."""
        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            '_fetched_at': time.time()
        }

        try:
            import httpx

            # 1. Blockchain.com - BTC stats (free, no key)
            try:
                # Active addresses (unique addresses used in last 24h)
                resp = httpx.get(
                    'https://api.blockchain.info/charts/n-unique-addresses?timespan=7days&format=json',
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    values = data.get('values', [])
                    if len(values) >= 2:
                        latest = values[-1].get('y', 0)
                        previous = values[-2].get('y', 0)
                        change_pct = ((latest - previous) / max(previous, 1)) * 100
                        result['btc_active_addresses'] = latest
                        result['active_address_change_pct'] = round(change_pct, 2)
            except Exception as e:
                logger.debug(f"Blockchain.com active addresses failed: {e}")

            # 2. Blockchain.com - Transaction volume
            try:
                resp = httpx.get(
                    'https://api.blockchain.info/charts/estimated-transaction-volume-usd?timespan=7days&format=json',
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    values = data.get('values', [])
                    if values:
                        result['btc_tx_volume_usd'] = values[-1].get('y', 0)
            except Exception as e:
                logger.debug(f"Blockchain.com tx volume failed: {e}")

            # 3. Blockchain.com - Mempool size (proxy for network congestion)
            try:
                resp = httpx.get(
                    'https://api.blockchain.info/charts/mempool-size?timespan=7days&format=json',
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    values = data.get('values', [])
                    if values:
                        result['btc_mempool_bytes'] = values[-1].get('y', 0)
            except Exception as e:
                logger.debug(f"Blockchain.com mempool failed: {e}")

            # 4. Exchange net flow estimate
            # Without CryptoQuant API key, estimate from large transactions
            try:
                resp = httpx.get(
                    'https://api.blockchain.info/charts/output-volume?timespan=2days&format=json',
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    values = data.get('values', [])
                    if len(values) >= 2:
                        latest_vol = values[-1].get('y', 0)
                        prev_vol = values[-2].get('y', 0)
                        # Net flow proxy: increasing output volume = potential exchange inflow
                        net_flow = latest_vol - prev_vol
                        result['exchange_net_flow'] = net_flow
                        result['exchange_net_flow_signal'] = (
                            'bearish' if net_flow > 0 else
                            'bullish' if net_flow < 0 else 'neutral'
                        )
            except Exception as e:
                logger.debug(f"Exchange flow estimate failed: {e}")

            # 5. Hash rate (Blockchain.com)
            try:
                resp = httpx.get(
                    'https://api.blockchain.info/charts/hash-rate?timespan=30days&format=json',
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    values = data.get('values', [])
                    if len(values) >= 2:
                        latest_hr = values[-1].get('y', 0)
                        week_ago_hr = values[-7].get('y', latest_hr) if len(values) >= 7 else latest_hr
                        hr_change = ((latest_hr - week_ago_hr) / max(week_ago_hr, 1)) * 100
                        result['btc_hash_rate'] = latest_hr
                        result['hash_rate_change_pct'] = round(hr_change, 2)
            except Exception as e:
                logger.debug(f"Hash rate fetch failed: {e}")

            return result

        except ImportError:
            logger.debug("httpx not available for on-chain scraping")
            return None
        except Exception as e:
            logger.debug(f"On-chain fetch failed: {e}")
            return None

    def _load_cache(self) -> Optional[Dict]:
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _save_cache(self, data: Dict):
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(CACHE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save on-chain cache: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = OnchainScraper()
    data = scraper.get_latest()
    print(json.dumps(data, indent=2, default=str))
