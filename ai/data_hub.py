"""
Data Hub - Central Data Aggregator
====================================
Collects all cached data sources into a single dict for AI components.
Reads from existing caches (Fear & Greed, FRED, FedWatch, news sentiment)
and fetches cross-asset data (BTC dominance, ETH/BTC, DXY).
Adds crypto-specific predictive data (funding rates, liquidations, on-chain).

15-minute refresh cycle, in-memory cache.

Author: Trading Bot Arsenal
Created: February 2026 | V7 Crystal Ball Upgrade
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger('DataHub')

# Paths to existing cached data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRYPTO_CACHE_DIR = os.path.join(BASE_DIR, 'data', 'crypto_cache')
ECON_CACHE_DIR = os.path.join(BASE_DIR, 'data', 'economic_cache')
FED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'fed_data_latest.json')

# FRED series mapping to human-readable names
FRED_SERIES = {
    'DFF': 'fed_funds_rate',
    'CPIAUCSL': 'cpi',
    'UNRATE': 'unemployment',
    'GDP': 'gdp',
    'DGS10': 'treasury_10y',
    'DGS2': 'treasury_2y',
    'T10Y2Y': 'yield_curve_spread',
    'PCEPI': 'pce_inflation',
    'PCEPILFE': 'core_pce',
    'A191RL1Q225SBEA': 'real_gdp_growth',
    'PAYEMS': 'nonfarm_payrolls',
    'RSAFS': 'retail_sales',
    'CPIAUCNS': 'cpi_unadjusted',
}

# Cache duration
REFRESH_INTERVAL = 900  # 15 minutes


class DataHub:
    """
    Central data aggregator for all AI components.

    Data sources:
    - Fear & Greed Index (from crypto_cache)
    - 13 FRED indicators (from economic_cache)
    - CME FedWatch rate probabilities (from fed_data_latest.json)
    - News sentiment (from news_feeds.aggregator)
    - BTC dominance, ETH/BTC ratio, DXY (via yfinance)
    - Funding rates (via CCXT - Bybit/OKX perpetual swaps)
    - Liquidation data (via Coinglass API)
    - On-chain metrics (via free APIs)
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._last_refresh = 0
        self._lock = threading.Lock()
        self._news_agg = None

        # Initialize with cached data immediately
        self.refresh()
        logger.info("DataHub initialized with all cached data sources")

    def get_data(self) -> Dict[str, Any]:
        """Get all aggregated data. Auto-refreshes if stale."""
        if time.time() - self._last_refresh > REFRESH_INTERVAL:
            self.refresh()
        with self._lock:
            return self._data.copy()

    def get(self, key: str, default=None):
        """Get a specific data key."""
        data = self.get_data()
        return data.get(key, default)

    def refresh(self):
        """Refresh all data sources."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        start = time.time()
        data = {}

        # Parallelize independent data loaders
        loaders = {
            'fear_greed': self._load_fear_greed,
            'fred': self._load_fred_data,
            'fedwatch': self._load_fedwatch,
            'news_sentiment': self._load_news_sentiment,
            'cross_asset': self._load_cross_asset_data,
            'funding_rates': self._load_funding_rates,
            'liquidations': self._load_liquidation_data,
            'onchain': self._load_onchain_data,
        }

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fn): name for name, fn in loaders.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result(timeout=30)
                    if name == 'cross_asset':
                        data['btc_dominance'] = result.get('btc_dominance')
                        data['eth_btc_ratio'] = result.get('eth_btc_ratio')
                        data['dxy'] = result.get('dxy')
                    else:
                        data[name] = result
                except Exception as e:
                    logger.debug(f"DataHub loader {name} failed: {e}")
                    data[name] = {} if name != 'cross_asset' else None

        # Derived signals
        data['macro_signal'] = self._compute_macro_signal(data)
        data['cross_asset_signal'] = self._compute_cross_asset_signal(data)
        data['crypto_derivatives_signal'] = self._compute_derivatives_signal(data)
        data['onchain_signal'] = self._compute_onchain_signal(data)

        data['last_refresh'] = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._data = data
            self._last_refresh = time.time()

        elapsed = time.time() - start
        logger.info(f"DataHub refreshed in {elapsed:.1f}s | "
                    f"F&G={data['fear_greed'].get('value', 'N/A')} | "
                    f"FedFunds={data['fred'].get('fed_funds_rate', 'N/A')} | "
                    f"BTC.D={data.get('btc_dominance', 'N/A')} | "
                    f"DXY={data.get('dxy', 'N/A')}")

    # ---- Data Loaders ----

    def _load_fear_greed(self) -> Dict:
        """Load Fear & Greed Index from crypto_cache."""
        path = os.path.join(CRYPTO_CACHE_DIR, 'fear_greed.json')
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    raw = json.load(f)
                return raw.get('data', {})
        except Exception as e:
            logger.debug(f"Fear & Greed load failed: {e}")
        return {}

    def _load_fred_data(self) -> Dict:
        """Load all FRED indicators from economic_cache."""
        fred = {}
        for series_id, name in FRED_SERIES.items():
            path = os.path.join(ECON_CACHE_DIR, f'fred_{series_id}.json')
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        raw = json.load(f)
                    data_points = raw.get('data', [])
                    if data_points:
                        # Most recent value
                        fred[name] = data_points[0].get('value')
                        fred[f'{name}_date'] = data_points[0].get('date')
            except Exception as e:
                logger.debug(f"FRED {series_id} load failed: {e}")
        return fred

    def _load_fedwatch(self) -> Dict:
        """Load CME FedWatch rate probabilities."""
        try:
            if os.path.exists(FED_DATA_PATH):
                with open(FED_DATA_PATH, 'r') as f:
                    raw = json.load(f)
                fedwatch = raw.get('cme_fedwatch', {})
                if fedwatch.get('success'):
                    return fedwatch.get('data', {})
        except Exception as e:
            logger.debug(f"FedWatch load failed: {e}")
        return {}

    def _load_news_sentiment(self) -> Dict:
        """Load news sentiment from aggregator."""
        try:
            if self._news_agg is None:
                from news_feeds.aggregator import NewsAggregator
                self._news_agg = NewsAggregator(cache_ttl=600)

            articles = self._news_agg.fetch_financial_news(
                symbols=['bitcoin', 'ethereum', 'crypto', 'fed', 'inflation'],
                include_crypto=True, limit=15
            )
            if articles:
                sentiments = [a.sentiment for a in articles if a.sentiment != 0]
                avg = sum(sentiments) / len(sentiments) if sentiments else 0
                return {
                    'avg_sentiment': round(avg, 3),
                    'article_count': len(articles),
                    'bullish_count': sum(1 for s in sentiments if s > 0.15),
                    'bearish_count': sum(1 for s in sentiments if s < -0.15),
                    'label': 'bullish' if avg > 0.15 else 'bearish' if avg < -0.15 else 'neutral'
                }
        except Exception as e:
            logger.debug(f"News sentiment load failed: {e}")
        return {}

    def _load_cross_asset_data(self) -> Dict:
        """Load BTC dominance, ETH/BTC ratio, DXY via yfinance."""
        result = {}
        try:
            import yfinance as yf

            # BTC dominance - approximate from BTC market cap vs total
            try:
                btc = yf.Ticker('BTC-USD')
                btc_info = btc.history(period='1d')
                if len(btc_info) > 0:
                    btc_price = float(btc_info['Close'].iloc[-1])
                    # Approximate BTC dominance (BTC mcap / total crypto mcap)
                    # Use ^CMC200 or hardcode approximate supply
                    btc_mcap = btc_price * 19_800_000  # ~19.8M BTC in circulation
                    # Total crypto market cap estimate (BTC is typically 50-60%)
                    # We'll get a better estimate from other sources
                    result['btc_price'] = btc_price
            except Exception:
                pass

            # ETH/BTC ratio
            try:
                eth = yf.Ticker('ETH-USD')
                eth_hist = eth.history(period='1d')
                if len(eth_hist) > 0 and result.get('btc_price'):
                    eth_price = float(eth_hist['Close'].iloc[-1])
                    result['eth_btc_ratio'] = round(eth_price / result['btc_price'], 5)
                    result['eth_price'] = eth_price
            except Exception:
                pass

            # DXY (Dollar Index)
            try:
                dxy = yf.Ticker('DX-Y.NYB')
                dxy_hist = dxy.history(period='5d')
                if len(dxy_hist) > 0:
                    result['dxy'] = round(float(dxy_hist['Close'].iloc[-1]), 2)
                    if len(dxy_hist) >= 2:
                        result['dxy_change'] = round(
                            (dxy_hist['Close'].iloc[-1] - dxy_hist['Close'].iloc[-2]) /
                            dxy_hist['Close'].iloc[-2] * 100, 3
                        )
            except Exception:
                pass

            # BTC dominance via CoinGecko (free, no key needed)
            try:
                import httpx
                resp = httpx.get(
                    'https://api.coingecko.com/api/v3/global',
                    timeout=10.0,
                    headers={'Accept': 'application/json'}
                )
                if resp.status_code == 200:
                    global_data = resp.json().get('data', {})
                    market_cap_pct = global_data.get('market_cap_percentage', {})
                    result['btc_dominance'] = round(market_cap_pct.get('btc', 0), 1)
                    result['eth_dominance'] = round(market_cap_pct.get('eth', 0), 1)
                    result['total_market_cap'] = global_data.get('total_market_cap', {}).get('usd')
            except Exception:
                pass

        except ImportError:
            logger.debug("yfinance not available for cross-asset data")
        return result

    def _load_funding_rates(self) -> Dict:
        """Load perpetual funding rates via CCXT (Bybit)."""
        rates = {}
        try:
            import ccxt
            exchange = ccxt.bybit({'enableRateLimit': True})
            symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']

            for symbol in symbols:
                try:
                    funding = exchange.fetch_funding_rate(symbol)
                    base = symbol.split('/')[0]
                    rates[base] = {
                        'rate': funding.get('fundingRate'),
                        'next_time': funding.get('fundingDatetime'),
                        'signal': 'bearish' if (funding.get('fundingRate') or 0) > 0.0005
                                  else 'bullish' if (funding.get('fundingRate') or 0) < -0.0005
                                  else 'neutral'
                    }
                except Exception:
                    pass
        except ImportError:
            logger.debug("ccxt not available for funding rates")
        except Exception as e:
            logger.debug(f"Funding rates load failed: {e}")
        return rates

    def _load_liquidation_data(self) -> Dict:
        """Load liquidation data from scrapers if available."""
        try:
            from scrapers.liquidation_scraper import LiquidationScraper
            scraper = LiquidationScraper()
            return scraper.get_latest()
        except ImportError:
            logger.debug("LiquidationScraper not available yet")
        except Exception as e:
            logger.debug(f"Liquidation data load failed: {e}")
        return {}

    def _load_onchain_data(self) -> Dict:
        """Load on-chain metrics from scrapers if available."""
        try:
            from scrapers.onchain_scraper import OnchainScraper
            scraper = OnchainScraper()
            return scraper.get_latest()
        except ImportError:
            logger.debug("OnchainScraper not available yet")
        except Exception as e:
            logger.debug(f"On-chain data load failed: {e}")
        return {}

    # ---- Signal Computers ----

    def _compute_macro_signal(self, data: Dict) -> Dict:
        """Combine Fear/Greed + FRED + FedWatch into macro directional signal."""
        bullish_points = 0
        bearish_points = 0
        total = 0

        # Fear & Greed (contrarian: extreme fear = bullish, extreme greed = bearish)
        fg_value = data.get('fear_greed', {}).get('value')
        if fg_value is not None:
            total += 2
            if fg_value < 20:
                bullish_points += 2  # Extreme fear = contrarian bullish
            elif fg_value < 40:
                bullish_points += 1
            elif fg_value > 80:
                bearish_points += 2  # Extreme greed = contrarian bearish
            elif fg_value > 60:
                bearish_points += 1

        # FedWatch (rate cut expectations = bullish for risk assets)
        fedwatch = data.get('fedwatch', {})
        if fedwatch:
            total += 1
            cut_prob = fedwatch.get('cut_25', 0) + fedwatch.get('cut_50', 0)
            hike_prob = fedwatch.get('hike_25', 0) + fedwatch.get('hike_50', 0)
            if cut_prob > 0.3:
                bullish_points += 1
            elif hike_prob > 0.3:
                bearish_points += 1

        # Yield curve (inverted = recession risk = bearish)
        fred = data.get('fred', {})
        spread = fred.get('yield_curve_spread')
        if spread is not None:
            total += 1
            if spread < -0.2:
                bearish_points += 1  # Inverted yield curve
            elif spread > 0.5:
                bullish_points += 1  # Normal/steep curve

        # News sentiment
        news = data.get('news_sentiment', {})
        if news.get('label'):
            total += 1
            if news['label'] == 'bullish':
                bullish_points += 1
            elif news['label'] == 'bearish':
                bearish_points += 1

        if total == 0:
            return {'direction': 'neutral', 'confidence': 0, 'source': 'macro'}

        if bullish_points > bearish_points:
            direction = 'bullish'
            confidence = int((bullish_points / total) * 100)
        elif bearish_points > bullish_points:
            direction = 'bearish'
            confidence = int((bearish_points / total) * 100)
        else:
            direction = 'neutral'
            confidence = 25

        return {'direction': direction, 'confidence': min(85, confidence), 'source': 'macro'}

    def _compute_cross_asset_signal(self, data: Dict) -> Dict:
        """DXY inverse, ETH/BTC ratio, BTC dominance → directional signal."""
        bullish = 0
        bearish = 0
        total = 0

        # DXY inverse correlation (dollar down = crypto up)
        dxy_change = data.get('dxy_change')
        if dxy_change is not None:
            total += 1
            if dxy_change < -0.2:
                bullish += 1  # Dollar weakening = bullish crypto
            elif dxy_change > 0.2:
                bearish += 1  # Dollar strengthening = bearish crypto

        # ETH/BTC ratio (rising = risk-on alt season, falling = flight to BTC safety)
        eth_btc = data.get('eth_btc_ratio')
        if eth_btc is not None:
            total += 1
            if eth_btc > 0.055:
                bullish += 1  # Alt season / risk-on
            elif eth_btc < 0.035:
                bearish += 1  # Flight to BTC safety

        # BTC dominance (high = fear, low = greed/alts)
        btc_dom = data.get('btc_dominance')
        if btc_dom is not None:
            total += 1
            if btc_dom > 60:
                bearish += 1  # Flight to BTC = risk-off
            elif btc_dom < 45:
                bullish += 1  # Alts pumping = risk-on

        if total == 0:
            return {'direction': 'neutral', 'confidence': 0, 'source': 'cross_asset'}

        if bullish > bearish:
            direction = 'bullish'
            confidence = int((bullish / total) * 100)
        elif bearish > bullish:
            direction = 'bearish'
            confidence = int((bearish / total) * 100)
        else:
            direction = 'neutral'
            confidence = 25

        return {'direction': direction, 'confidence': min(80, confidence), 'source': 'cross_asset'}

    def _compute_derivatives_signal(self, data: Dict) -> Dict:
        """Funding rates + liquidations → directional signal."""
        bullish = 0
        bearish = 0
        total = 0

        # Funding rates (positive = overleveraged longs → bearish, negative = bullish)
        funding = data.get('funding_rates', {})
        for coin, info in funding.items():
            if info.get('signal'):
                total += 1
                if info['signal'] == 'bullish':
                    bullish += 1
                elif info['signal'] == 'bearish':
                    bearish += 1

        # Liquidation data
        liqs = data.get('liquidations', {})
        if liqs.get('long_dominance') is not None:
            total += 1
            # If mostly longs liquidated = more downside ahead (bearish)
            # If mostly shorts liquidated = short squeeze fuel (bullish)
            if liqs['long_dominance'] > 65:
                bearish += 1
            elif liqs['long_dominance'] < 35:
                bullish += 1

        if total == 0:
            return {'direction': 'neutral', 'confidence': 0, 'source': 'crypto_derivatives'}

        if bullish > bearish:
            direction = 'bullish'
            confidence = int((bullish / total) * 100)
        elif bearish > bullish:
            direction = 'bearish'
            confidence = int((bearish / total) * 100)
        else:
            direction = 'neutral'
            confidence = 25

        return {'direction': direction, 'confidence': min(80, confidence), 'source': 'crypto_derivatives'}

    def _compute_onchain_signal(self, data: Dict) -> Dict:
        """On-chain metrics → directional signal."""
        onchain = data.get('onchain', {})
        if not onchain:
            return {'direction': 'neutral', 'confidence': 0, 'source': 'onchain'}

        bullish = 0
        bearish = 0
        total = 0

        # Exchange net flow (negative = outflow = bullish, positive = inflow = bearish)
        net_flow = onchain.get('exchange_net_flow')
        if net_flow is not None:
            total += 1
            if net_flow < 0:
                bullish += 1  # Outflows = accumulation
            elif net_flow > 0:
                bearish += 1  # Inflows = sell pressure

        # Active addresses (growing = healthy = bullish)
        addr_change = onchain.get('active_address_change_pct')
        if addr_change is not None:
            total += 1
            if addr_change > 5:
                bullish += 1
            elif addr_change < -5:
                bearish += 1

        if total == 0:
            return {'direction': 'neutral', 'confidence': 0, 'source': 'onchain'}

        if bullish > bearish:
            direction = 'bullish'
            confidence = int((bullish / total) * 100)
        elif bearish > bullish:
            direction = 'bearish'
            confidence = int((bearish / total) * 100)
        else:
            direction = 'neutral'
            confidence = 25

        return {'direction': direction, 'confidence': min(75, confidence), 'source': 'onchain'}

    def get_prompt_context(self) -> str:
        """Format data hub contents for LLM prompt injection."""
        data = self.get_data()
        parts = []

        # Fear & Greed
        fg = data.get('fear_greed', {})
        if fg.get('value') is not None:
            parts.append(f"Fear & Greed Index: {fg['value']} ({fg.get('classification', 'N/A')})")

        # FRED key indicators
        fred = data.get('fred', {})
        if fred:
            parts.append(f"Fed Funds Rate: {fred.get('fed_funds_rate', 'N/A')}%")
            parts.append(f"CPI: {fred.get('cpi', 'N/A')}")
            parts.append(f"Unemployment: {fred.get('unemployment', 'N/A')}%")
            parts.append(f"10Y Treasury: {fred.get('treasury_10y', 'N/A')}%")
            parts.append(f"2Y Treasury: {fred.get('treasury_2y', 'N/A')}%")
            parts.append(f"Yield Curve Spread: {fred.get('yield_curve_spread', 'N/A')}%")

        # FedWatch
        fw = data.get('fedwatch', {})
        if fw:
            parts.append(f"FedWatch: Hold {fw.get('hold', 0):.0%}, "
                        f"Cut {fw.get('cut_25', 0) + fw.get('cut_50', 0):.0%}, "
                        f"Hike {fw.get('hike_25', 0) + fw.get('hike_50', 0):.0%}")

        # Cross-asset
        if data.get('btc_dominance'):
            parts.append(f"BTC Dominance: {data['btc_dominance']}%")
        if data.get('eth_btc_ratio'):
            parts.append(f"ETH/BTC: {data['eth_btc_ratio']}")
        if data.get('dxy'):
            chg = f" ({data['dxy_change']:+.2f}%)" if data.get('dxy_change') is not None else ""
            parts.append(f"DXY: {data['dxy']}{chg}")

        # Funding rates
        funding = data.get('funding_rates', {})
        if funding:
            fr_parts = [f"{coin}: {info.get('rate', 0):.4%} ({info.get('signal', 'N/A')})"
                       for coin, info in funding.items() if info.get('rate') is not None]
            if fr_parts:
                parts.append(f"Funding Rates: {', '.join(fr_parts)}")

        # News sentiment
        news = data.get('news_sentiment', {})
        if news.get('label'):
            parts.append(f"News Sentiment: {news['label']} (avg={news.get('avg_sentiment', 0):.2f}, "
                        f"{news.get('article_count', 0)} articles)")

        # Derived signals
        macro = data.get('macro_signal', {})
        if macro.get('direction'):
            parts.append(f"Macro Signal: {macro['direction']} ({macro.get('confidence', 0)}%)")

        cross = data.get('cross_asset_signal', {})
        if cross.get('direction'):
            parts.append(f"Cross-Asset Signal: {cross['direction']} ({cross.get('confidence', 0)}%)")

        derivs = data.get('crypto_derivatives_signal', {})
        if derivs.get('direction'):
            parts.append(f"Derivatives Signal: {derivs['direction']} ({derivs.get('confidence', 0)}%)")

        return '\n'.join(parts) if parts else 'No external data available.'


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s %(name)s %(levelname)s: %(message)s')
    hub = DataHub()
    data = hub.get_data()
    print("\n=== Data Hub Contents ===")
    for key, value in data.items():
        if isinstance(value, dict) and len(str(value)) > 100:
            print(f"  {key}: {type(value).__name__} ({len(value)} items)")
        else:
            print(f"  {key}: {value}")
    print("\n=== Prompt Context ===")
    print(hub.get_prompt_context())
