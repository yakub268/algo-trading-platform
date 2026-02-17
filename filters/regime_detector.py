"""
Market Regime Detector using Hidden Markov Models
==================================================

Detects market regimes (bull, bear, sideways, high-vol) using HMM.
Integrates with DoNothingFilter to prevent trading in unfavorable regimes.

Research Base:
- HMMs capture the "hidden" state of the market
- Different strategies work in different regimes
- Trading against the regime is a losing game

Usage:
    detector = RegimeDetector()
    regime = detector.detect(df)
    
    if regime in ['bear', 'high_volatility']:
        logger.info("Unfavorable regime - reducing exposure")

Author: Trading Bot Arsenal
Created: January 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RegimeDetector')

# Try to import hmmlearn
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. Run: pip install hmmlearn")

# Try to import ruptures for change point detection
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    logger.warning("ruptures not installed. Run: pip install ruptures")


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeResult:
    """Result of regime detection"""
    current_regime: MarketRegime
    confidence: float
    regime_duration_days: int
    transition_probability: Dict[str, float]
    all_regimes: List[str]
    regime_history: pd.Series
    features_used: Dict[str, float]
    recommendation: str


class RegimeDetector:
    """
    Market Regime Detector using Hidden Markov Models.
    
    Detects 4 primary regimes:
    1. Bull - Strong uptrend, low volatility
    2. Bear - Downtrend, rising volatility  
    3. Sideways - No clear trend, normal volatility
    4. High Volatility - Choppy, dangerous conditions
    
    Features used:
    - Returns (momentum)
    - Volatility (risk)
    - Volume ratio (participation)
    - Trend strength (ADX-like)
    
    Parameters:
    - n_regimes: Number of hidden states (default: 4)
    - lookback: Days of history to use (default: 100)
    - retrain_frequency: Retrain model every N days (default: 21)
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        lookback: int = 100,
        retrain_frequency: int = 21,
        random_state: int = 42
    ):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.retrain_frequency = retrain_frequency
        self.random_state = random_state
        
        self.model = None
        self.last_train_date = None
        self.regime_labels = {}
        
        if not HMM_AVAILABLE:
            logger.warning("HMM not available - using fallback regime detection")
        
        logger.info(
            f"RegimeDetector initialized: "
            f"n_regimes={n_regimes}, lookback={lookback}"
        )
    
    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns to lowercase."""
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
        else:
            df.columns = df.columns.str.lower()
        return df
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for regime detection.
        
        Features:
        1. Returns (5-day)
        2. Volatility (20-day rolling std)
        3. Volume ratio (current / 20-day avg)
        4. Trend strength (price vs SMA ratio)
        """
        df = self.normalize_columns(df)
        
        features = pd.DataFrame(index=df.index)
        
        # 5-day returns
        features['returns'] = df['close'].pct_change(5)
        
        # 20-day volatility
        features['volatility'] = df['close'].pct_change().rolling(20).std()
        
        # Volume ratio (if available)
        if 'volume' in df.columns:
            avg_volume = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / avg_volume
        else:
            features['volume_ratio'] = 1.0
        
        # Trend strength: price / SMA50 - 1
        sma50 = df['close'].rolling(50).mean()
        features['trend'] = (df['close'] / sma50) - 1
        
        # Normalize features
        for col in features.columns:
            mean = features[col].rolling(self.lookback).mean()
            std = features[col].rolling(self.lookback).std()
            features[col] = (features[col] - mean) / (std + 1e-8)
        
        return features.dropna()
    
    def _train_hmm(self, features: pd.DataFrame) -> None:
        """Train HMM model on features."""
        if not HMM_AVAILABLE:
            return
        
        X = features.values
        
        # Initialize and train HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state
        )
        
        try:
            self.model.fit(X)
            self.last_train_date = features.index[-1]
            
            # Label regimes based on characteristics
            self._label_regimes(features)
            
            logger.info(f"HMM trained on {len(features)} samples")
        except Exception as e:
            logger.error(f"HMM training failed: {e}")
            self.model = None
    
    def _label_regimes(self, features: pd.DataFrame) -> None:
        """
        Label regimes based on their characteristics.
        
        High return + low vol = Bull
        Low return + high vol = Bear
        Low vol + no trend = Sideways
        High vol = High Volatility
        """
        if self.model is None:
            return
        
        X = features.values
        states = self.model.predict(X)
        
        # Calculate mean characteristics for each state
        state_chars = {}
        for state in range(self.n_regimes):
            mask = states == state
            if mask.sum() > 0:
                state_chars[state] = {
                    'return': features.loc[mask, 'returns'].mean() if 'returns' in features.columns else 0,
                    'volatility': features.loc[mask, 'volatility'].mean() if 'volatility' in features.columns else 0,
                    'trend': features.loc[mask, 'trend'].mean() if 'trend' in features.columns else 0
                }
        
        # Assign labels based on characteristics
        self.regime_labels = {}
        
        # Sort by return
        sorted_by_return = sorted(state_chars.items(), key=lambda x: x[1]['return'], reverse=True)
        
        # Highest return = Bull
        if len(sorted_by_return) > 0:
            self.regime_labels[sorted_by_return[0][0]] = MarketRegime.BULL
        
        # Lowest return = Bear
        if len(sorted_by_return) > 1:
            self.regime_labels[sorted_by_return[-1][0]] = MarketRegime.BEAR
        
        # Sort remaining by volatility
        remaining = [s for s in state_chars.keys() if s not in self.regime_labels]
        if remaining:
            sorted_by_vol = sorted(remaining, key=lambda x: state_chars[x]['volatility'], reverse=True)
            
            # Highest vol = High Volatility
            if len(sorted_by_vol) > 0:
                self.regime_labels[sorted_by_vol[0]] = MarketRegime.HIGH_VOLATILITY
            
            # Lowest vol = Sideways/Low Vol
            if len(sorted_by_vol) > 1:
                self.regime_labels[sorted_by_vol[-1]] = MarketRegime.SIDEWAYS
        
        # Fill any remaining
        for state in range(self.n_regimes):
            if state not in self.regime_labels:
                self.regime_labels[state] = MarketRegime.UNKNOWN
        
        logger.debug(f"Regime labels: {self.regime_labels}")
    
    def _fallback_detection(self, df: pd.DataFrame) -> RegimeResult:
        """
        Simple rule-based regime detection when HMM unavailable.
        """
        df = self.normalize_columns(df)
        
        # Calculate simple metrics
        returns_20d = df['close'].pct_change(20).iloc[-1]
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
        vol_percentile = df['close'].pct_change().rolling(20).std().rank(pct=True).iloc[-1]
        
        # SMA trend
        sma50 = df['close'].rolling(50).mean().iloc[-1]
        sma200 = df['close'].rolling(200).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Determine regime
        if vol_percentile > 0.8:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = vol_percentile
            recommendation = "REDUCE exposure - high volatility regime"
        elif returns_20d > 0.05 and current_price > sma50 > sma200:
            regime = MarketRegime.BULL
            confidence = min(returns_20d * 10, 0.9)
            recommendation = "NORMAL trading - bull regime"
        elif returns_20d < -0.05 and current_price < sma50:
            regime = MarketRegime.BEAR
            confidence = min(abs(returns_20d) * 10, 0.9)
            recommendation = "REDUCE exposure - bear regime"
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.6
            recommendation = "SELECTIVE trading - sideways regime"
        
        # Create regime history (simple)
        regime_history = pd.Series(
            [regime.value] * len(df),
            index=df.index
        )
        
        return RegimeResult(
            current_regime=regime,
            confidence=confidence,
            regime_duration_days=20,  # Approximate
            transition_probability={},
            all_regimes=[r.value for r in MarketRegime],
            regime_history=regime_history,
            features_used={
                'returns_20d': returns_20d,
                'volatility': volatility,
                'vol_percentile': vol_percentile,
                'price_vs_sma50': current_price / sma50 - 1 if sma50 else 0
            },
            recommendation=recommendation
        )
    
    def detect(self, df: pd.DataFrame, force_retrain: bool = False) -> RegimeResult:
        """
        Detect current market regime.
        
        Args:
            df: DataFrame with OHLCV data
            force_retrain: Force model retraining
            
        Returns:
            RegimeResult with current regime and analysis
        """
        if len(df) < self.lookback:
            logger.warning(f"Insufficient data ({len(df)} < {self.lookback})")
            return self._fallback_detection(df)
        
        # Use fallback if HMM not available
        if not HMM_AVAILABLE:
            return self._fallback_detection(df)
        
        # Calculate features
        features = self._calculate_features(df)
        
        if len(features) < 50:
            return self._fallback_detection(df)
        
        # Check if retraining needed
        current_date = features.index[-1]
        needs_retrain = (
            force_retrain or
            self.model is None or
            self.last_train_date is None or
            (current_date - self.last_train_date).days >= self.retrain_frequency
        )
        
        if needs_retrain:
            self._train_hmm(features)
        
        if self.model is None:
            return self._fallback_detection(df)
        
        # Predict current regime
        try:
            X = features.values
            states = self.model.predict(X)
            current_state = states[-1]
            current_regime = self.regime_labels.get(current_state, MarketRegime.UNKNOWN)
            
            # Calculate confidence (posterior probability)
            posteriors = self.model.predict_proba(X)
            confidence = posteriors[-1, current_state]
            
            # Calculate regime duration
            regime_duration = 1
            for i in range(len(states) - 2, -1, -1):
                if states[i] == current_state:
                    regime_duration += 1
                else:
                    break
            
            # Get transition probabilities
            trans_probs = {}
            for i, regime in self.regime_labels.items():
                trans_probs[regime.value] = float(self.model.transmat_[current_state, i])
            
            # Create regime history
            regime_history = pd.Series(
                [self.regime_labels.get(s, MarketRegime.UNKNOWN).value for s in states],
                index=features.index
            )
            
            # Generate recommendation
            if current_regime == MarketRegime.BULL:
                recommendation = "NORMAL trading - bull regime favors momentum strategies"
            elif current_regime == MarketRegime.BEAR:
                recommendation = "REDUCE exposure - bear regime, favor defensive strategies"
            elif current_regime == MarketRegime.HIGH_VOLATILITY:
                recommendation = "CAUTION - high volatility, reduce position sizes"
            elif current_regime == MarketRegime.SIDEWAYS:
                recommendation = "SELECTIVE trading - sideways regime, favor mean reversion"
            else:
                recommendation = "UNKNOWN regime - proceed with caution"
            
            # Get current feature values
            features_used = {
                'returns': float(features['returns'].iloc[-1]),
                'volatility': float(features['volatility'].iloc[-1]),
                'volume_ratio': float(features['volume_ratio'].iloc[-1]),
                'trend': float(features['trend'].iloc[-1])
            }
            
            return RegimeResult(
                current_regime=current_regime,
                confidence=float(confidence),
                regime_duration_days=regime_duration,
                transition_probability=trans_probs,
                all_regimes=[r.value for r in MarketRegime],
                regime_history=regime_history,
                features_used=features_used,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return self._fallback_detection(df)
    
    def get_regime_filter(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get trading filter based on current regime.
        
        Returns dict with:
        - should_trade: bool
        - position_size_multiplier: float (0.0 - 1.0)
        - allowed_strategies: List[str]
        - reason: str
        """
        result = self.detect(df)
        
        regime_filters = {
            MarketRegime.BULL: {
                'should_trade': True,
                'position_size_multiplier': 1.0,
                'allowed_strategies': ['momentum', 'trend', 'breakout', 'mean_reversion'],
                'reason': 'Bull regime - full trading allowed'
            },
            MarketRegime.BEAR: {
                'should_trade': True,
                'position_size_multiplier': 0.5,
                'allowed_strategies': ['mean_reversion', 'defensive'],
                'reason': 'Bear regime - reduced exposure, defensive strategies only'
            },
            MarketRegime.SIDEWAYS: {
                'should_trade': True,
                'position_size_multiplier': 0.75,
                'allowed_strategies': ['mean_reversion', 'range'],
                'reason': 'Sideways regime - favor mean reversion'
            },
            MarketRegime.HIGH_VOLATILITY: {
                'should_trade': True,
                'position_size_multiplier': 0.25,
                'allowed_strategies': ['volatility'],
                'reason': 'High volatility - minimal exposure'
            },
            MarketRegime.LOW_VOLATILITY: {
                'should_trade': True,
                'position_size_multiplier': 1.0,
                'allowed_strategies': ['momentum', 'trend', 'breakout'],
                'reason': 'Low volatility - good for trend following'
            },
            MarketRegime.UNKNOWN: {
                'should_trade': True,
                'position_size_multiplier': 0.5,
                'allowed_strategies': ['mean_reversion'],
                'reason': 'Unknown regime - proceed with caution'
            }
        }
        
        filter_config = regime_filters.get(result.current_regime, regime_filters[MarketRegime.UNKNOWN])
        filter_config['current_regime'] = result.current_regime.value
        filter_config['confidence'] = result.confidence
        filter_config['regime_duration_days'] = result.regime_duration_days
        
        return filter_config


class ChangePointDetector:
    """
    Detect regime change points using ruptures library.
    
    Useful for:
    - Identifying when market regime has changed
    - Adjusting strategy parameters at breakpoints
    - Backtesting regime-aware strategies
    """
    
    def __init__(self, model: str = 'rbf', min_size: int = 20, penalty: float = 3.0):
        """
        Initialize change point detector.
        
        Args:
            model: Detection model ('rbf', 'l2', 'linear')
            min_size: Minimum segment size
            penalty: Penalty for adding breakpoints (higher = fewer breaks)
        """
        self.model = model
        self.min_size = min_size
        self.penalty = penalty
        
        if not RUPTURES_AVAILABLE:
            logger.warning("ruptures not available for change point detection")
    
    def detect(self, df: pd.DataFrame, n_breakpoints: int = None) -> List[int]:
        """
        Detect change points in price series.
        
        Args:
            df: DataFrame with OHLCV data
            n_breakpoints: Number of breakpoints (None = auto)
            
        Returns:
            List of change point indices
        """
        if not RUPTURES_AVAILABLE:
            return []
        
        df_norm = RegimeDetector.normalize_columns(df)
        
        # Calculate returns for change detection
        returns = df_norm['close'].pct_change().dropna().values
        
        if len(returns) < self.min_size * 2:
            return []
        
        # Create detector
        if self.model == 'rbf':
            algo = rpt.KernelCPD(kernel="rbf", min_size=self.min_size)
        elif self.model == 'l2':
            algo = rpt.Pelt(model="l2", min_size=self.min_size)
        else:
            algo = rpt.Pelt(model="linear", min_size=self.min_size)
        
        try:
            algo.fit(returns.reshape(-1, 1))
            
            if n_breakpoints:
                breakpoints = algo.predict(n_bkps=n_breakpoints)
            else:
                breakpoints = algo.predict(pen=self.penalty)
            
            # Convert to original index positions (account for pct_change dropping first)
            breakpoints = [bp + 1 for bp in breakpoints if bp < len(df) - 1]
            
            logger.info(f"Detected {len(breakpoints)} change points")
            return breakpoints
            
        except Exception as e:
            logger.error(f"Change point detection failed: {e}")
            return []
    
    def get_breakpoint_dates(self, df: pd.DataFrame, n_breakpoints: int = None) -> List[datetime]:
        """Get change points as dates."""
        df_norm = RegimeDetector.normalize_columns(df)
        breakpoints = self.detect(df, n_breakpoints)
        
        dates = []
        for bp in breakpoints:
            if bp < len(df_norm):
                dates.append(df_norm.index[bp])
        
        return dates


# Example usage
if __name__ == "__main__":
    import yfinance as yf
    
    print("=" * 60)
    print("MARKET REGIME DETECTOR TEST")
    print("=" * 60)
    
    # Download SPY data
    try:
        spy = yf.download("SPY", period="2y", interval="1d", progress=False)
        
        if len(spy) > 0:
            print(f"\nLoaded {len(spy)} days of SPY data")
            
            # Test regime detector
            detector = RegimeDetector()
            result = detector.detect(spy)
            
            print(f"\nRegime Detection Results:")
            print(f"  Current Regime: {result.current_regime.value}")
            print(f"  Confidence: {result.confidence:.1%}")
            print(f"  Duration: {result.regime_duration_days} days")
            print(f"  Recommendation: {result.recommendation}")
            
            print(f"\nFeatures:")
            for feat, val in result.features_used.items():
                print(f"  {feat}: {val:.3f}")
            
            if result.transition_probability:
                print(f"\nTransition Probabilities:")
                for regime, prob in result.transition_probability.items():
                    print(f"  → {regime}: {prob:.1%}")
            
            # Get trading filter
            filter_config = detector.get_regime_filter(spy)
            print(f"\nTrading Filter:")
            print(f"  Should Trade: {filter_config['should_trade']}")
            print(f"  Position Size Multiplier: {filter_config['position_size_multiplier']:.0%}")
            print(f"  Allowed Strategies: {filter_config['allowed_strategies']}")
            
            # Test change point detection
            if RUPTURES_AVAILABLE:
                print(f"\n" + "-" * 40)
                print("Change Point Detection:")
                cp_detector = ChangePointDetector()
                breakpoint_dates = cp_detector.get_breakpoint_dates(spy, n_breakpoints=5)
                
                print(f"  Detected {len(breakpoint_dates)} change points:")
                for date in breakpoint_dates:
                    print(f"    - {date.date()}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
