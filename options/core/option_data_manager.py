"""
Options Data Manager

Handles real-time options data integration from multiple sources:
- Alpaca Options API
- Interactive Brokers
- Yahoo Finance
- Polygon.io
- CBOE data feeds

Provides unified interface for options chain data, historical IV, and market data.
"""

import asyncio
import aiohttp
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Dict, List, Optional, Union, Callable, Any
import logging
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
from pathlib import Path

from .option_chain import OptionChain, OptionContract
from .greeks_calculator import GreeksCalculator
from .pricing_models import ImpliedVolatilityCalculator

logger = logging.getLogger(__name__)


class OptionDataManager:
    """
    Unified options data manager for multiple data sources
    """

    def __init__(self,
                 alpaca_api_key: Optional[str] = None,
                 alpaca_secret_key: Optional[str] = None,
                 polygon_api_key: Optional[str] = None,
                 cache_dir: str = "data/options_cache"):
        """
        Initialize data manager with API credentials

        Args:
            alpaca_api_key: Alpaca API key
            alpaca_secret_key: Alpaca secret key
            polygon_api_key: Polygon.io API key
            cache_dir: Directory for caching historical data
        """
        # API credentials
        self.alpaca_api_key = alpaca_api_key or os.getenv('ALPACA_API_KEY')
        self.alpaca_secret_key = alpaca_secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.polygon_api_key = polygon_api_key or os.getenv('POLYGON_API_KEY')

        # Data sources configuration
        self.data_sources = {
            'alpaca': self._fetch_alpaca_options,
            'polygon': self._fetch_polygon_options,
            'yahoo': self._fetch_yahoo_options,
            'cboe': self._fetch_cboe_options
        }

        # Caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry = 300  # 5 minutes default cache

        # Rate limiting
        self.rate_limits = {
            'alpaca': 200,    # requests per minute
            'polygon': 1000,   # requests per minute
            'yahoo': 2000,     # requests per minute
            'cboe': 100        # requests per minute
        }
        self.request_timestamps = {source: [] for source in self.data_sources}

        # Greeks calculator
        self.greeks_calc = GreeksCalculator()

        # Database for historical data
        self.db_path = self.cache_dir / "options_data.db"
        self._init_database()

        logger.info("Initialized OptionDataManager with multiple data sources")

    def _init_database(self):
        """Initialize SQLite database for historical data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Options contracts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_contracts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                underlying TEXT NOT NULL,
                strike REAL NOT NULL,
                expiry DATE NOT NULL,
                option_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                bid REAL,
                ask REAL,
                last_price REAL,
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                rho REAL,
                data_source TEXT,
                UNIQUE(symbol, timestamp)
            )
        ''')

        # Historical IV table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_iv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                underlying TEXT NOT NULL,
                date DATE NOT NULL,
                expiry DATE NOT NULL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                implied_volatility REAL,
                iv_rank REAL,
                iv_percentile REAL,
                UNIQUE(underlying, date, expiry, strike, option_type)
            )
        ''')

        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                price REAL,
                volume INTEGER,
                realized_volatility REAL,
                UNIQUE(symbol, timestamp)
            )
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_symbol_expiry ON options_contracts(symbol, expiry)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_underlying_timestamp ON options_contracts(underlying, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_iv_underlying_date ON historical_iv(underlying, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)')

        conn.commit()
        conn.close()

    async def get_option_chain(self,
                              symbol: str,
                              expiry_date: Optional[Union[str, date]] = None,
                              preferred_source: str = 'alpaca',
                              include_greeks: bool = True) -> Optional[OptionChain]:
        """
        Get complete options chain for a symbol

        Args:
            symbol: Underlying symbol (e.g., 'AAPL')
            expiry_date: Specific expiry date (optional)
            preferred_source: Preferred data source
            include_greeks: Whether to calculate Greeks

        Returns:
            OptionChain object or None if error
        """
        try:
            # Get current stock price
            spot_price = await self._get_spot_price(symbol)
            if spot_price is None:
                logger.error(f"Could not get spot price for {symbol}")
                return None

            # Create option chain object
            chain = OptionChain(symbol, Decimal(str(spot_price)))

            # Try preferred source first
            contracts = await self._fetch_options_data(symbol, preferred_source, expiry_date)

            # Fallback to other sources if needed
            if not contracts:
                for source in self.data_sources:
                    if source != preferred_source:
                        logger.info(f"Trying fallback source: {source}")
                        contracts = await self._fetch_options_data(symbol, source, expiry_date)
                        if contracts:
                            break

            if not contracts:
                logger.error(f"No options data found for {symbol}")
                return None

            # Process contracts and add to chain
            risk_free_rate = await self._get_risk_free_rate()

            for contract_data in contracts:
                try:
                    contract = await self._process_contract_data(
                        contract_data, spot_price, risk_free_rate, include_greeks
                    )
                    if contract:
                        chain.add_contract(contract)
                except Exception as e:
                    logger.warning(f"Error processing contract {contract_data}: {e}")
                    continue

            logger.info(f"Retrieved option chain for {symbol} with {len(chain)} contracts")

            # Cache the data
            await self._cache_option_chain(chain)

            return chain

        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {e}")
            return None

    async def _fetch_options_data(self,
                                 symbol: str,
                                 source: str,
                                 expiry_date: Optional[Union[str, date]] = None) -> List[Dict]:
        """Fetch options data from specific source"""
        if not self._check_rate_limit(source):
            logger.warning(f"Rate limit exceeded for {source}")
            return []

        try:
            fetch_func = self.data_sources.get(source)
            if fetch_func:
                return await fetch_func(symbol, expiry_date)
            else:
                logger.error(f"Unknown data source: {source}")
                return []
        except Exception as e:
            logger.error(f"Error fetching from {source}: {e}")
            return []

    async def _fetch_alpaca_options(self, symbol: str, expiry_date: Optional[Union[str, date]] = None) -> List[Dict]:
        """Fetch options data from Alpaca"""
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            logger.warning("Alpaca credentials not available")
            return []

        try:
            headers = {
                'APCA-API-KEY-ID': self.alpaca_api_key,
                'APCA-API-SECRET-KEY': self.alpaca_secret_key
            }

            base_url = "https://paper-api.alpaca.markets/v2"  # Use paper trading endpoint

            # Get options contracts
            url = f"{base_url}/options/contracts"
            params = {
                'underlying_symbol': symbol,
                'status': 'active'
            }

            if expiry_date:
                if isinstance(expiry_date, str):
                    params['expiration_date'] = expiry_date
                else:
                    params['expiration_date'] = expiry_date.strftime('%Y-%m-%d')

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        contracts = data.get('option_contracts', [])

                        # Get current quotes for each contract
                        enriched_contracts = []
                        for contract in contracts:
                            quote_data = await self._get_alpaca_option_quote(session, headers, contract['symbol'])
                            if quote_data:
                                contract.update(quote_data)
                                enriched_contracts.append(contract)

                        return enriched_contracts
                    else:
                        logger.error(f"Alpaca API error: {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Error fetching Alpaca options data: {e}")
            return []

    async def _get_alpaca_option_quote(self, session: aiohttp.ClientSession, headers: Dict, option_symbol: str) -> Optional[Dict]:
        """Get real-time quote for Alpaca option"""
        try:
            url = f"https://paper-api.alpaca.markets/v2/options/quotes/latest"
            params = {'symbols': option_symbol}

            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    quotes = data.get('quotes', {})
                    return quotes.get(option_symbol, {})

        except Exception as e:
            logger.warning(f"Error getting Alpaca quote for {option_symbol}: {e}")

        return None

    async def _fetch_polygon_options(self, symbol: str, expiry_date: Optional[Union[str, date]] = None) -> List[Dict]:
        """Fetch options data from Polygon.io"""
        if not self.polygon_api_key:
            logger.warning("Polygon.io API key not available")
            return []

        try:
            base_url = "https://api.polygon.io"

            # Get options contracts
            url = f"{base_url}/v3/reference/options/contracts"
            params = {
                'underlying_ticker': symbol,
                'contract_type': 'option',
                'apikey': self.polygon_api_key,
                'limit': 1000
            }

            if expiry_date:
                if isinstance(expiry_date, str):
                    params['expiration_date'] = expiry_date
                else:
                    params['expiration_date'] = expiry_date.strftime('%Y-%m-%d')

            async with aiohttp.ClientSession() as session:
                contracts = []
                next_url = url

                while next_url:
                    async with session.get(next_url, params=params if next_url == url else None) as response:
                        if response.status == 200:
                            data = await response.json()
                            batch_contracts = data.get('results', [])

                            # Get quotes for this batch
                            symbols = [contract['ticker'] for contract in batch_contracts]
                            if symbols:
                                quotes = await self._get_polygon_option_quotes(session, symbols)

                                # Merge contract data with quotes
                                for contract in batch_contracts:
                                    quote_data = quotes.get(contract['ticker'], {})
                                    contract.update(quote_data)
                                    contracts.append(contract)

                            # Check for next page
                            next_url = data.get('next_url')
                            params = None  # Clear params for next URL
                        else:
                            logger.error(f"Polygon API error: {response.status}")
                            break

                return contracts

        except Exception as e:
            logger.error(f"Error fetching Polygon options data: {e}")
            return []

    async def _get_polygon_option_quotes(self, session: aiohttp.ClientSession, symbols: List[str]) -> Dict:
        """Get real-time quotes for Polygon options"""
        quotes = {}

        try:
            # Polygon API allows up to 5 symbols per request for quotes
            symbol_chunks = [symbols[i:i+5] for i in range(0, len(symbols), 5)]

            for chunk in symbol_chunks:
                url = f"https://api.polygon.io/v2/last/nbbo/{','.join(chunk)}"
                params = {'apikey': self.polygon_api_key}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', {})

                        for symbol, quote in results.items():
                            quotes[symbol] = quote
                    else:
                        logger.warning(f"Polygon quotes error: {response.status}")

        except Exception as e:
            logger.warning(f"Error getting Polygon quotes: {e}")

        return quotes

    async def _fetch_yahoo_options(self, symbol: str, expiry_date: Optional[Union[str, date]] = None) -> List[Dict]:
        """Fetch options data from Yahoo Finance"""
        try:
            import yfinance as yf

            # Yahoo Finance doesn't support async, so we'll run in executor
            loop = asyncio.get_event_loop()

            def get_yahoo_options():
                ticker = yf.Ticker(symbol)

                # Get available expiry dates
                try:
                    expiry_dates = ticker.options
                    if not expiry_dates:
                        return []

                    # Filter by specific expiry if provided
                    if expiry_date:
                        if isinstance(expiry_date, str):
                            target_date = expiry_date
                        else:
                            target_date = expiry_date.strftime('%Y-%m-%d')

                        if target_date not in expiry_dates:
                            return []
                        expiry_dates = [target_date]

                    all_contracts = []

                    for exp_date in expiry_dates:
                        try:
                            option_chain = ticker.option_chain(exp_date)

                            # Process calls
                            calls = option_chain.calls.to_dict('records')
                            for call in calls:
                                call['option_type'] = 'call'
                                call['expiry'] = exp_date
                                call['underlying'] = symbol
                                all_contracts.append(call)

                            # Process puts
                            puts = option_chain.puts.to_dict('records')
                            for put in puts:
                                put['option_type'] = 'put'
                                put['expiry'] = exp_date
                                put['underlying'] = symbol
                                all_contracts.append(put)

                        except Exception as e:
                            logger.warning(f"Error getting Yahoo options for {exp_date}: {e}")
                            continue

                    return all_contracts

                except Exception as e:
                    logger.error(f"Error getting Yahoo options expiry dates: {e}")
                    return []

            # Run in thread pool to avoid blocking
            with ThreadPoolExecutor() as executor:
                contracts = await loop.run_in_executor(executor, get_yahoo_options)

            return contracts

        except ImportError:
            logger.warning("yfinance not installed, skipping Yahoo Finance data")
            return []
        except Exception as e:
            logger.error(f"Error fetching Yahoo options data: {e}")
            return []

    async def _fetch_cboe_options(self, symbol: str, expiry_date: Optional[Union[str, date]] = None) -> List[Dict]:
        """Fetch options data from CBOE (delayed/limited data)"""
        try:
            # CBOE provides delayed options data through their market data API
            # This is a simplified implementation
            base_url = "https://cdn.cboe.com/api/global/delayed_quotes/options"

            params = {
                'symbol': symbol
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}.json", params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Process CBOE data format
                        contracts = []
                        for item in data.get('data', []):
                            # Convert CBOE format to standard format
                            contract = self._normalize_cboe_data(item, symbol)
                            if contract:
                                contracts.append(contract)

                        return contracts
                    else:
                        logger.warning(f"CBOE API response: {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Error fetching CBOE options data: {e}")
            return []

    def _normalize_cboe_data(self, cboe_data: Dict, symbol: str) -> Optional[Dict]:
        """Normalize CBOE data format to standard format"""
        try:
            # This would need to be implemented based on actual CBOE API response format
            return {
                'symbol': cboe_data.get('symbol', ''),
                'underlying': symbol,
                'strike': cboe_data.get('strike', 0),
                'expiry': cboe_data.get('expiry', ''),
                'option_type': cboe_data.get('type', '').lower(),
                'bid': cboe_data.get('bid', 0),
                'ask': cboe_data.get('ask', 0),
                'last_price': cboe_data.get('last', 0),
                'volume': cboe_data.get('volume', 0),
                'open_interest': cboe_data.get('open_interest', 0),
                'implied_volatility': cboe_data.get('iv', None)
            }
        except Exception:
            return None

    async def _process_contract_data(self,
                                   contract_data: Dict,
                                   spot_price: float,
                                   risk_free_rate: float,
                                   include_greeks: bool = True) -> Optional[OptionContract]:
        """Process raw contract data into OptionContract object"""
        try:
            # Normalize field names from different sources
            symbol = contract_data.get('symbol') or contract_data.get('ticker') or contract_data.get('contractSymbol', '')
            underlying = contract_data.get('underlying') or contract_data.get('underlying_symbol', '')

            # Parse strike price
            strike = contract_data.get('strike') or contract_data.get('strike_price', 0)
            if isinstance(strike, str):
                strike = float(strike.replace('$', '').replace(',', ''))

            # Parse expiry date
            expiry_str = contract_data.get('expiry') or contract_data.get('expiration_date') or contract_data.get('expiration', '')
            if isinstance(expiry_str, str):
                expiry = datetime.strptime(expiry_str[:10], '%Y-%m-%d')
            else:
                expiry = expiry_str

            # Option type
            option_type = contract_data.get('option_type') or contract_data.get('type', '').lower()
            if option_type not in ['call', 'put']:
                # Try to parse from symbol if available
                if 'C' in symbol or 'call' in symbol.lower():
                    option_type = 'call'
                elif 'P' in symbol or 'put' in symbol.lower():
                    option_type = 'put'
                else:
                    return None

            # Prices
            bid = float(contract_data.get('bid') or contract_data.get('bidPrice', 0))
            ask = float(contract_data.get('ask') or contract_data.get('askPrice', 0))
            last_price = float(contract_data.get('last_price') or contract_data.get('lastPrice') or
                              contract_data.get('last', 0))

            # Volume and OI
            volume = int(contract_data.get('volume', 0))
            open_interest = int(contract_data.get('open_interest') or contract_data.get('openInterest', 0))

            # Create contract
            contract = OptionContract(
                symbol=symbol,
                underlying=underlying,
                strike=Decimal(str(strike)),
                expiry=expiry,
                option_type=option_type,
                bid=Decimal(str(bid)),
                ask=Decimal(str(ask)),
                last_price=Decimal(str(last_price)),
                volume=volume,
                open_interest=open_interest
            )

            # Calculate Greeks and IV if requested
            if include_greeks and contract.time_to_expiry > 0:
                # Try to get IV from data first
                iv = contract_data.get('implied_volatility') or contract_data.get('impliedVolatility')

                if iv is None or iv <= 0:
                    # Calculate IV from market price
                    market_price = float(last_price) if last_price > 0 else float((bid + ask) / 2) if (bid + ask) > 0 else 0
                    if market_price > 0:
                        iv = ImpliedVolatilityCalculator.calculate_iv_adaptive(
                            market_price, spot_price, float(strike),
                            contract.time_to_expiry, risk_free_rate, option_type
                        )

                if iv and iv > 0:
                    contract.implied_volatility = float(iv)

                    # Calculate Greeks
                    greeks = self.greeks_calc.calculate_greeks(
                        spot_price=spot_price,
                        strike_price=float(strike),
                        time_to_expiry=contract.time_to_expiry,
                        risk_free_rate=risk_free_rate,
                        volatility=float(iv),
                        option_type=option_type
                    )

                    contract.delta = greeks.delta
                    contract.gamma = greeks.gamma
                    contract.theta = greeks.theta
                    contract.vega = greeks.vega
                    contract.rho = greeks.rho

            return contract

        except Exception as e:
            logger.warning(f"Error processing contract data: {e}")
            return None

    async def _get_spot_price(self, symbol: str) -> Optional[float]:
        """Get current spot price for underlying"""
        try:
            # Try Alpaca first
            if self.alpaca_api_key and self.alpaca_secret_key:
                headers = {
                    'APCA-API-KEY-ID': self.alpaca_api_key,
                    'APCA-API-SECRET-KEY': self.alpaca_secret_key
                }

                url = f"https://paper-api.alpaca.markets/v2/stocks/quotes/latest"
                params = {'symbols': symbol}

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            quotes = data.get('quotes', {})
                            quote = quotes.get(symbol, {})
                            bid = quote.get('bid_price', 0)
                            ask = quote.get('ask_price', 0)
                            if bid > 0 and ask > 0:
                                return (bid + ask) / 2

            # Fallback to Yahoo Finance
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                info = ticker.info
                return info.get('currentPrice') or info.get('regularMarketPrice')
            except Exception as e:
                logger.debug(f"Error fetching spot price from yfinance for {symbol}: {e}")

            return None

        except Exception as e:
            logger.error(f"Error getting spot price for {symbol}: {e}")
            return None

    async def _get_risk_free_rate(self) -> float:
        """Get current risk-free rate (10-year Treasury)"""
        try:
            # Try to get from FRED API if available
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'GS10',
                'api_key': os.getenv('FRED_API_KEY', ''),
                'limit': 1,
                'sort_order': 'desc',
                'file_type': 'json'
            }

            if params['api_key']:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            observations = data.get('observations', [])
                            if observations:
                                rate_str = observations[0].get('value', '2.0')
                                return float(rate_str) / 100  # Convert percentage to decimal

            # Default fallback rate
            return 0.02  # 2%

        except Exception as e:
            logger.warning(f"Error getting risk-free rate: {e}")
            return 0.02

    def _check_rate_limit(self, source: str) -> bool:
        """Check if request is within rate limits"""
        now = time.time()
        minute_ago = now - 60

        # Clean old timestamps
        self.request_timestamps[source] = [
            ts for ts in self.request_timestamps[source] if ts > minute_ago
        ]

        # Check limit
        limit = self.rate_limits.get(source, 100)
        if len(self.request_timestamps[source]) >= limit:
            return False

        # Add current timestamp
        self.request_timestamps[source].append(now)
        return True

    async def _cache_option_chain(self, chain: OptionChain):
        """Cache option chain data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            timestamp = datetime.now()

            for contract in chain.contracts.values():
                cursor.execute('''
                    INSERT OR REPLACE INTO options_contracts
                    (symbol, underlying, strike, expiry, option_type, timestamp,
                     bid, ask, last_price, volume, open_interest, implied_volatility,
                     delta, gamma, theta, vega, rho, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    contract.symbol, contract.underlying, float(contract.strike),
                    contract.expiry.date(), contract.option_type, timestamp,
                    float(contract.bid), float(contract.ask), float(contract.last_price),
                    contract.volume, contract.open_interest, contract.implied_volatility,
                    contract.delta, contract.gamma, contract.theta, contract.vega, contract.rho,
                    'live_data'
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error caching option chain: {e}")

    async def get_historical_iv_data(self,
                                    symbol: str,
                                    start_date: Union[str, date],
                                    end_date: Union[str, date],
                                    expiry_filter: Optional[int] = None) -> pd.DataFrame:
        """
        Get historical implied volatility data

        Args:
            symbol: Underlying symbol
            start_date: Start date
            end_date: End date
            expiry_filter: Days to expiry filter (e.g., 30 for monthly options)

        Returns:
            DataFrame with historical IV data
        """
        try:
            conn = sqlite3.connect(self.db_path)

            query = '''
                SELECT date, expiry, strike, option_type, implied_volatility, iv_rank, iv_percentile
                FROM historical_iv
                WHERE underlying = ? AND date BETWEEN ? AND ?
            '''
            params = [symbol, start_date, end_date]

            if expiry_filter:
                query += ' AND (julianday(expiry) - julianday(date)) BETWEEN ? AND ?'
                params.extend([expiry_filter - 5, expiry_filter + 5])

            query += ' ORDER BY date, expiry, strike'

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            return df

        except Exception as e:
            logger.error(f"Error getting historical IV data: {e}")
            return pd.DataFrame()

    async def calculate_iv_rank(self, symbol: str, current_iv: float, lookback_days: int = 252) -> Optional[float]:
        """
        Calculate IV rank (percentile of current IV vs historical range)

        Args:
            symbol: Underlying symbol
            current_iv: Current implied volatility
            lookback_days: Historical lookback period

        Returns:
            IV rank as percentile (0-100)
        """
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)

            df = await self.get_historical_iv_data(symbol, start_date, end_date)

            if df.empty:
                return None

            historical_ivs = df['implied_volatility'].dropna()

            if len(historical_ivs) < 10:  # Need sufficient data
                return None

            # Calculate percentile
            rank = np.percentile(historical_ivs, current_iv * 100)
            return float(rank)

        except Exception as e:
            logger.error(f"Error calculating IV rank: {e}")
            return None

    def get_cached_option_chain(self, symbol: str, max_age_minutes: int = 5) -> Optional[OptionChain]:
        """
        Get cached option chain if available and fresh

        Args:
            symbol: Underlying symbol
            max_age_minutes: Maximum age of cached data

        Returns:
            Cached OptionChain or None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)

            cursor.execute('''
                SELECT DISTINCT underlying FROM options_contracts
                WHERE underlying = ? AND timestamp > ?
            ''', (symbol, cutoff_time))

            if not cursor.fetchone():
                conn.close()
                return None

            # Get all contracts for this symbol
            cursor.execute('''
                SELECT * FROM options_contracts
                WHERE underlying = ? AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (symbol, cutoff_time))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return None

            # Get latest spot price (simplified)
            latest_row = rows[0]
            # This would need to be enhanced to get actual spot price
            spot_price = Decimal('100.0')  # Placeholder

            chain = OptionChain(symbol, spot_price)

            for row in rows:
                contract = OptionContract(
                    symbol=row[1],
                    underlying=row[2],
                    strike=Decimal(str(row[3])),
                    expiry=datetime.strptime(row[4], '%Y-%m-%d'),
                    option_type=row[5],
                    bid=Decimal(str(row[7])),
                    ask=Decimal(str(row[8])),
                    last_price=Decimal(str(row[9])),
                    volume=row[10],
                    open_interest=row[11],
                    implied_volatility=row[12],
                    delta=row[13],
                    gamma=row[14],
                    theta=row[15],
                    vega=row[16],
                    rho=row[17]
                )
                chain.add_contract(contract)

            return chain

        except Exception as e:
            logger.error(f"Error getting cached option chain: {e}")
            return None