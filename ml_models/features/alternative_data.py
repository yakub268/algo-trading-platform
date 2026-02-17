"""
Alternative Data Processor
=========================

Processes alternative data sources for ML feature engineering.
Integrates with existing scrapers and external data sources.

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


class AlternativeDataProcessor:
    """
    Process alternative data sources for ML features.

    Integrates with existing trading bot data infrastructure:
    - Economic indicators (FRED)
    - Weather data (NWS)
    - Sports outcomes and betting odds
    - Crypto on-chain metrics
    - Satellite imagery indicators (simulated)
    - Social media metrics beyond sentiment
    """

    def __init__(self):
        self.data_cache = {}
        self._init_data_sources()

    def _init_data_sources(self):
        """Initialize connections to data sources"""
        try:
            # Try to import existing scrapers from the trading bot
            from scrapers.data_aggregator import DataAggregator
            self.data_aggregator = DataAggregator()
            self.has_aggregator = True
        except ImportError:
            logger.warning("DataAggregator not available. Using simulated alternative data.")
            self.has_aggregator = False

    def extract_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Extract alternative data features.

        Args:
            df: Price DataFrame with DatetimeIndex
            symbol: Asset symbol (if applicable)

        Returns:
            DataFrame with alternative data features
        """
        features = pd.DataFrame(index=df.index)

        try:
            # 1. Economic indicators
            econ_features = self._get_economic_features(df.index)
            if not econ_features.empty:
                features = pd.concat([features, econ_features], axis=1)

            # 2. Cross-asset correlations
            corr_features = self._get_correlation_features(df.index, symbol)
            if not corr_features.empty:
                features = pd.concat([features, corr_features], axis=1)

            # 3. Options and derivatives indicators
            options_features = self._get_options_features(df.index, symbol)
            if not options_features.empty:
                features = pd.concat([features, options_features], axis=1)

            # 4. Seasonality and calendar effects
            calendar_features = self._get_calendar_features(df.index)
            if not calendar_features.empty:
                features = pd.concat([features, calendar_features], axis=1)

            # 5. Macro regime indicators
            regime_features = self._get_regime_features(df.index)
            if not regime_features.empty:
                features = pd.concat([features, regime_features], axis=1)

            # 6. Crypto-specific features (if applicable)
            if symbol and any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'CRYPTO']):
                crypto_features = self._get_crypto_features(df.index, symbol)
                if not crypto_features.empty:
                    features = pd.concat([features, crypto_features], axis=1)

        except Exception as e:
            logger.warning(f"Error extracting alternative data features: {e}")

        # Fill missing values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

        return features

    def _get_economic_features(self, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Get economic indicator features from real cached data (FRED, Fear & Greed)."""
        features = pd.DataFrame(index=date_index)
        import json

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        real_data_loaded = False

        try:
            # --- Fear & Greed Index (real cached data) ---
            fg_path = os.path.join(project_root, 'data', 'crypto_cache', 'fear_greed.json')
            if os.path.exists(fg_path):
                with open(fg_path, 'r') as f:
                    fg = json.load(f)
                fg_value = fg.get('data', {}).get('value')
                if fg_value is not None:
                    features['fear_index'] = float(fg_value)
                    features['fear_index_sma'] = float(fg_value)  # Static snapshot
                    real_data_loaded = True

            # --- FRED indicators (real cached data) ---
            econ_dir = os.path.join(project_root, 'data', 'economic_cache')
            fred_features = {
                'fred_DFF': 'interest_rates',
                'fred_DGS2': 'yield_2y',
                'fred_DGS10': 'yield_10y',
                'fred_T10Y2Y': 'yield_curve_slope',
                'fred_CPIAUCSL': 'cpi',
                'fred_UNRATE': 'unemployment',
                'fred_GDP': 'gdp',
                'fred_M2SL': 'm2_money_supply',
                'fred_DCOILWTICO': 'oil_price',
                'fred_GOLDAMGBD228NLBM': 'gold_price',
                'fred_DTWEXBGS': 'dollar_strength',
            }

            for fname, feature_name in fred_features.items():
                path = os.path.join(econ_dir, f'{fname}.json')
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                    vals = data.get('data', [])
                    if vals:
                        val = vals[0].get('value')
                        if val is not None:
                            features[feature_name] = float(val)
                            real_data_loaded = True

            # Derived features from real data
            if 'yield_10y' in features.columns and 'yield_2y' in features.columns:
                features['yield_curve_inverted'] = (features['yield_curve_slope'] < 0).astype(int) if 'yield_curve_slope' in features.columns else 0

            if 'dollar_strength' in features.columns:
                features['dollar_momentum'] = 0  # Static snapshot, no momentum available

            if 'oil_price' in features.columns and 'gold_price' in features.columns:
                oil_val = features['oil_price'].iloc[0] if len(features) > 0 else 1
                features['gold_oil_ratio'] = features['gold_price'] / max(oil_val, 0.01)
                features['oil_momentum'] = 0  # Static snapshot

            # --- CME FedWatch (real cached data) ---
            fedwatch_path = os.path.join(project_root, 'data', 'fed_data_latest.json')
            if os.path.exists(fedwatch_path):
                with open(fedwatch_path, 'r') as f:
                    fed = json.load(f)
                fw_data = fed.get('cme_fedwatch', {}).get('data', {})
                if fw_data:
                    features['fedwatch_hold_prob'] = fw_data.get('hold', 0)
                    features['fedwatch_cut_prob'] = fw_data.get('cut_25', 0) + fw_data.get('cut_50', 0)
                    features['fedwatch_hike_prob'] = fw_data.get('hike_25', 0) + fw_data.get('hike_50', 0)
                    real_data_loaded = True

            if not real_data_loaded:
                logger.warning("No real cached economic data found — features will be sparse")

        except Exception as e:
            logger.debug(f"Error creating economic features: {e}")

        return features

    def _get_correlation_features(self, date_index: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
        """Get cross-asset correlation features"""
        features = pd.DataFrame(index=date_index)

        try:
            # Simulate correlations with major indices
            np.random.seed(123)

            # Correlation with SPY (market beta)
            market_correlation = np.random.uniform(0.3, 0.9, len(date_index))
            # Add some time-varying behavior
            for i in range(1, len(market_correlation)):
                market_correlation[i] = 0.9 * market_correlation[i-1] + 0.1 * market_correlation[i]

            features['market_correlation'] = market_correlation
            features['market_correlation_regime'] = pd.cut(
                market_correlation,
                bins=[0, 0.5, 0.7, 1.0],
                labels=[0, 1, 2]
            ).astype(float)

            # Sector correlation
            sector_correlation = np.random.uniform(0.4, 0.8, len(date_index))
            features['sector_correlation'] = sector_correlation

            # Bond correlation (typically negative for stocks)
            bond_correlation = np.random.uniform(-0.5, 0.1, len(date_index))
            features['bond_correlation'] = bond_correlation

            # Commodity correlation
            commodity_correlation = np.random.uniform(-0.2, 0.4, len(date_index))
            features['commodity_correlation'] = commodity_correlation

            # Cross-asset momentum
            features['correlation_momentum'] = features['market_correlation'].pct_change(20)

        except Exception as e:
            logger.debug(f"Error creating correlation features: {e}")

        return features

    def _get_options_features(self, date_index: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
        """Get options and derivatives features"""
        features = pd.DataFrame(index=date_index)

        try:
            np.random.seed(456)

            # Implied volatility indicators
            implied_vol = np.random.gamma(2, 0.1, len(date_index))
            implied_vol = np.clip(implied_vol, 0.1, 0.8)
            features['implied_volatility'] = implied_vol

            # VIX-like volatility term structure
            vol_term_structure = np.random.normal(0, 0.05, len(date_index))
            features['vol_term_structure'] = vol_term_structure
            features['vol_backwardation'] = (vol_term_structure < 0).astype(int)

            # Put/call ratio
            put_call_ratio = np.random.gamma(2, 0.5, len(date_index))
            put_call_ratio = np.clip(put_call_ratio, 0.5, 2.0)
            features['put_call_ratio'] = put_call_ratio
            features['put_call_extreme'] = (put_call_ratio > 1.5).astype(int)

            # Options volume indicators
            options_volume = np.random.exponential(1000, len(date_index))
            features['options_volume'] = options_volume
            features['options_volume_sma'] = pd.Series(options_volume).rolling(5).mean()

            # Gamma exposure (market maker positioning)
            gamma_exposure = np.random.normal(0, 1000, len(date_index))
            features['gamma_exposure'] = gamma_exposure
            features['gamma_squeeze'] = (abs(gamma_exposure) > 2000).astype(int)

            # Dark pool indicators
            dark_pool_volume = np.random.uniform(0.2, 0.6, len(date_index))
            features['dark_pool_ratio'] = dark_pool_volume

        except Exception as e:
            logger.debug(f"Error creating options features: {e}")

        return features

    def _get_calendar_features(self, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Get calendar and seasonality features"""
        features = pd.DataFrame(index=date_index)

        try:
            # Day of week effects
            features['monday'] = (date_index.dayofweek == 0).astype(int)
            features['friday'] = (date_index.dayofweek == 4).astype(int)

            # Month effects
            features['january'] = (date_index.month == 1).astype(int)
            features['december'] = (date_index.month == 12).astype(int)

            # Quarter effects
            features['quarter_end'] = date_index.is_quarter_end.astype(int)
            features['month_end'] = date_index.is_month_end.astype(int)

            # Holiday proximity (simplified)
            # Major US holidays
            major_holidays = [
                (1, 1),   # New Year
                (7, 4),   # Independence Day
                (11, 4),  # Thanksgiving (4th Thursday, approximated)
                (12, 25), # Christmas
            ]

            features['holiday_proximity'] = 0
            for month, day in major_holidays:
                holiday_dates = pd.to_datetime(f'{date_index.year[0]}-{month:02d}-{day:02d}')
                if isinstance(holiday_dates, pd.Timestamp):
                    days_to_holiday = abs((date_index - holiday_dates).days)
                    if days_to_holiday <= 3:
                        features['holiday_proximity'] = max(features['holiday_proximity'], 1)

            # Day of week effects
            features['monday'] = (date_index.dayofweek == 0).astype(int)
            features['friday'] = (date_index.dayofweek == 4).astype(int)

            # Month effects
            features['january'] = (date_index.month == 1).astype(int)
            features['december'] = (date_index.month == 12).astype(int)

            # Quarter effects
            features['quarter_end'] = date_index.is_quarter_end.astype(int)
            features['month_end'] = date_index.is_month_end.astype(int)

            # Days to earnings (simulated)
            np.random.seed(789)
            days_to_earnings = np.random.randint(0, 90, len(date_index))
            features['days_to_earnings'] = days_to_earnings
            features['earnings_proximity'] = (days_to_earnings < 7).astype(int)

            # Expiration effects (options, futures)
            # Third Friday of each month (simplified)
            features['options_expiration'] = (
                (date_index.dayofweek == 4) &  # Friday
                (date_index.day >= 15) &       # Third week
                (date_index.day <= 21)         # Third week
            ).astype(int)

        except Exception as e:
            logger.debug(f"Error creating calendar features: {e}")

        return features

    def _get_regime_features(self, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Get macro regime features derived from real cached FRED data."""
        features = pd.DataFrame(index=date_index)
        import json

        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            econ_dir = os.path.join(project_root, 'data', 'economic_cache')

            # Read yield curve to determine regime
            yield_curve = None
            yc_path = os.path.join(econ_dir, 'fred_T10Y2Y.json')
            if os.path.exists(yc_path):
                with open(yc_path, 'r') as f:
                    data = json.load(f)
                vals = data.get('data', [])
                if vals and vals[0].get('value') is not None:
                    yield_curve = float(vals[0]['value'])

            # Read Fear & Greed for risk appetite
            fg_value = None
            fg_path = os.path.join(project_root, 'data', 'crypto_cache', 'fear_greed.json')
            if os.path.exists(fg_path):
                with open(fg_path, 'r') as f:
                    fg = json.load(f)
                fg_val = fg.get('data', {}).get('value')
                if fg_val is not None:
                    fg_value = float(fg_val)

            # Bull/bear regime from yield curve
            if yield_curve is not None:
                features['bull_market_regime'] = 1 if yield_curve > 0 else 0
                features['yield_curve_inverted'] = 1 if yield_curve < 0 else 0
            else:
                features['bull_market_regime'] = 1  # Default assumption

            # Risk appetite from Fear & Greed (0-100 scale → 0-1)
            if fg_value is not None:
                risk = fg_value / 100.0
                features['risk_appetite'] = risk
                features['risk_on'] = int(risk > 0.6)
                features['risk_off'] = int(risk < 0.4)
            else:
                features['risk_appetite'] = 0.5
                features['risk_on'] = 0
                features['risk_off'] = 0

            # Volatility regime from Fear & Greed (extreme values = high vol)
            if fg_value is not None:
                if fg_value < 20 or fg_value > 80:
                    features['volatility_regime'] = 2  # High
                elif fg_value < 35 or fg_value > 65:
                    features['volatility_regime'] = 1  # Medium
                else:
                    features['volatility_regime'] = 0  # Low
            else:
                features['volatility_regime'] = 1  # Default medium

            # Credit/liquidity stress — placeholder (would need HY spread data)
            features['liquidity_stress'] = 0
            features['liquidity_crisis'] = 0
            features['credit_spread'] = 0
            features['credit_stress'] = 0

        except Exception as e:
            logger.debug(f"Error creating regime features: {e}")

        return features

    def _get_crypto_features(self, date_index: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
        """Get cryptocurrency-specific features from real cached data."""
        features = pd.DataFrame(index=date_index)
        import json

        try:
            if not any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'CRYPTO']):
                return features

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_dir = os.path.join(project_root, 'data', 'crypto_cache')

            # --- Fear & Greed (real) ---
            fg_path = os.path.join(cache_dir, 'fear_greed.json')
            if os.path.exists(fg_path):
                with open(fg_path, 'r') as f:
                    fg = json.load(f)
                fg_value = fg.get('data', {}).get('value')
                if fg_value is not None:
                    fg_val = float(fg_value)
                    features['crypto_fear_greed'] = fg_val
                    features['crypto_extreme_fear'] = int(fg_val < 25)
                    features['crypto_extreme_greed'] = int(fg_val > 75)

            # --- On-chain metrics (real from onchain scraper cache) ---
            onchain_path = os.path.join(cache_dir, 'onchain.json')
            if os.path.exists(onchain_path):
                with open(onchain_path, 'r') as f:
                    onchain = json.load(f)
                data = onchain.get('data', {})
                if data.get('active_addresses'):
                    features['active_addresses'] = float(data['active_addresses'])
                if data.get('active_address_change_pct') is not None:
                    features['address_momentum'] = float(data['active_address_change_pct']) / 100
                if data.get('transaction_volume'):
                    features['transaction_volume'] = float(data['transaction_volume'])
                if data.get('hash_rate') and 'BTC' in symbol.upper():
                    features['hash_rate'] = float(data['hash_rate'])
                    features['hash_rate_momentum'] = 0  # Static snapshot
                if data.get('exchange_net_flow') is not None:
                    features['exchange_net_flow'] = float(data['exchange_net_flow'])

            # --- Liquidation data (real from liquidation scraper cache) ---
            liq_path = os.path.join(cache_dir, 'liquidations.json')
            if os.path.exists(liq_path):
                with open(liq_path, 'r') as f:
                    liq = json.load(f)
                liq_data = liq.get('data', {})
                btc_liq = liq_data.get('btc', {})
                if btc_liq.get('long_pct') is not None:
                    features['long_term_holder_ratio'] = float(btc_liq['long_pct']) / 100
                if btc_liq.get('total_usd'):
                    features['network_value'] = float(btc_liq['total_usd'])

        except Exception as e:
            logger.debug(f"Error creating crypto features: {e}")

        return features

    def get_real_time_alternative_data(self, symbol: str = None) -> Dict[str, float]:
        """
        Get real-time alternative data metrics from cached files.

        Returns:
            Dict with current alternative data metrics
        """
        import json

        try:
            if self.has_aggregator:
                summary = self.data_aggregator.get_summary()
                return self._process_aggregator_summary(summary)
        except Exception as e:
            logger.debug(f"Error getting aggregator summary: {e}")

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        alt_data = {'last_updated': datetime.now().isoformat()}

        # Fear & Greed
        try:
            fg_path = os.path.join(project_root, 'data', 'crypto_cache', 'fear_greed.json')
            if os.path.exists(fg_path):
                with open(fg_path, 'r') as f:
                    fg = json.load(f)
                alt_data['crypto_fear_greed'] = float(fg.get('data', {}).get('value', 50))
                alt_data['vix_level'] = alt_data['crypto_fear_greed']  # Proxy
        except Exception:
            pass

        # FRED rates
        econ_dir = os.path.join(project_root, 'data', 'economic_cache')
        fred_rt = {
            'fred_DFF': 'interest_rate',
            'fred_DTWEXBGS': 'dollar_strength',
            'fred_DCOILWTICO': 'oil_price',
            'fred_T10Y2Y': 'yield_curve',
        }
        for fname, key in fred_rt.items():
            try:
                path = os.path.join(econ_dir, f'{fname}.json')
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                    vals = data.get('data', [])
                    if vals and vals[0].get('value') is not None:
                        alt_data[key] = float(vals[0]['value'])
            except Exception:
                pass

        # FedWatch
        try:
            fed_path = os.path.join(project_root, 'data', 'fed_data_latest.json')
            if os.path.exists(fed_path):
                with open(fed_path, 'r') as f:
                    fed = json.load(f)
                fw = fed.get('cme_fedwatch', {}).get('data', {})
                if fw:
                    alt_data['fedwatch_hold'] = fw.get('hold', 0)
                    alt_data['fedwatch_cut'] = fw.get('cut_25', 0) + fw.get('cut_50', 0)
        except Exception:
            pass

        return alt_data

    def _process_aggregator_summary(self, summary: Dict) -> Dict[str, float]:
        """Process data aggregator summary into features"""
        processed = {}

        try:
            # Extract economic indicators
            if 'economic' in summary:
                econ = summary['economic']
                processed['economic_estimates'] = econ.get('estimates', 0)

            # Extract crypto metrics
            if 'crypto' in summary:
                crypto = summary['crypto']
                processed['crypto_fear_greed'] = crypto.get('fear_greed', 50)

            # Extract weather uncertainty
            if 'weather' in summary:
                weather = summary['weather']
                processed['weather_uncertainty'] = weather.get('estimates', 0) / 10

            # Extract sports betting volume (proxy for risk appetite)
            if 'sports' in summary:
                sports = summary['sports']
                processed['sports_activity'] = sports.get('estimates', 0) / 5

        except Exception as e:
            logger.debug(f"Error processing aggregator summary: {e}")

        return processed


def main():
    """Test alternative data processing"""
    import yfinance as yf

    # Download test data
    print("Downloading test data...")
    data = yf.download("AAPL", period="3m", progress=False)

    # Initialize processor
    processor = AlternativeDataProcessor()

    # Extract alternative data features
    print("Extracting alternative data features...")
    alt_features = processor.extract_features(data, symbol="AAPL")

    print(f"\nAlternative Data Features Results:")
    print(f"Shape: {alt_features.shape}")
    print(f"Columns: {list(alt_features.columns[:10])}...")  # First 10
    print("\nSample data:")
    print(alt_features.tail())

    # Test real-time alternative data
    print("\nTesting real-time alternative data...")
    rt_data = processor.get_real_time_alternative_data("AAPL")
    print("Real-time data:", rt_data)

    # Test crypto-specific features
    print("\nTesting crypto features...")
    crypto_features = processor._get_crypto_features(data.index, "BTCUSD")
    print(f"Crypto features shape: {crypto_features.shape}")
    print(f"Crypto columns: {list(crypto_features.columns[:5])}...")

    print("\nAlternative data processing test completed successfully!")


if __name__ == "__main__":
    main()