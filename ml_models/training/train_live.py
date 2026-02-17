"""
Live XGBoost/LightGBM Training Script
========================================
Trains gradient-boosted models on historical crypto data with real features
from the feature_engineer and alternative_data pipelines.

XGBoost is fastest to deploy and best for tabular financial data.

Usage:
    python ml_models/training/train_live.py --symbol BTC-USD --period 1y

Author: Trading Bot Arsenal
Created: February 2026 | V7 Crystal Ball Upgrade
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger('TrainLive')


def fetch_training_data(symbol: str = 'BTC-USD', period: str = '1y',
                        interval: str = '1d') -> pd.DataFrame:
    """Fetch historical data for training."""
    import yfinance as yf
    logger.info(f"Fetching {symbol} data: period={period}, interval={interval}")
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    # Flatten MultiIndex columns from newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    logger.info(f"Downloaded {len(data)} bars for {symbol}")
    return data


def create_labels(df: pd.DataFrame, horizon: int = 5,
                  threshold_pct: float = 1.0) -> pd.Series:
    """
    Create direction labels for supervised learning.

    Args:
        df: Price DataFrame with 'Close' column
        horizon: Number of bars to look ahead
        threshold_pct: Minimum move % to count as bullish/bearish (V7: raised to 1.0%)

    Returns:
        Series with labels: 1 (bullish), -1 (bearish), 0 (neutral)
    """
    future_return = df['Close'].pct_change(horizon).shift(-horizon) * 100

    labels = pd.Series(0, index=df.index, dtype=int)
    labels[future_return > threshold_pct] = 1   # bullish
    labels[future_return < -threshold_pct] = -1  # bearish

    return labels


def engineer_features(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """Engineer features using the project's feature pipeline + real cached data."""
    features = pd.DataFrame(index=df.index)

    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    volumes = df['Volume'].values

    # Technical indicators
    for period in [7, 14, 21]:
        # RSI
        deltas = df['Close'].diff()
        gains = deltas.where(deltas > 0, 0).rolling(period).mean()
        losses = (-deltas.where(deltas < 0, 0)).rolling(period).mean()
        rs = gains / losses.replace(0, 1e-10)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # EMAs
    for period in [8, 12, 21, 26, 50]:
        features[f'ema_{period}'] = df['Close'].ewm(span=period).mean()

    # EMA crossovers (as numeric)
    features['ema_12_26_cross'] = (features['ema_12'] - features['ema_26']) / df['Close'] * 100
    features['ema_8_21_cross'] = (features['ema_8'] - features['ema_21']) / df['Close'] * 100

    # MACD
    features['macd'] = features['ema_12'] - features['ema_26']
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']

    # Bollinger Bands
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    features['bb_upper'] = sma_20 + 2 * std_20
    features['bb_lower'] = sma_20 - 2 * std_20
    features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower']).replace(0, 1)
    features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20

    # Volume features
    features['volume_sma_20'] = df['Volume'].rolling(20).mean()
    features['volume_ratio'] = df['Volume'] / features['volume_sma_20'].replace(0, 1)
    features['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean().replace(0, 1)

    # ATR
    tr = pd.DataFrame({
        'hl': df['High'] - df['Low'],
        'hc': abs(df['High'] - df['Close'].shift(1)),
        'lc': abs(df['Low'] - df['Close'].shift(1))
    }).max(axis=1)
    features['atr_14'] = tr.rolling(14).mean()
    features['atr_pct'] = features['atr_14'] / df['Close'] * 100

    # Returns at various lookbacks
    for period in [1, 3, 5, 10, 20]:
        features[f'return_{period}d'] = df['Close'].pct_change(period) * 100

    # Momentum
    features['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    features['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1

    # Volatility
    features['volatility_10'] = df['Close'].pct_change().rolling(10).std() * 100
    features['volatility_20'] = df['Close'].pct_change().rolling(20).std() * 100

    # Support/Resistance distance
    features['dist_to_high_20'] = (df['High'].rolling(20).max() - df['Close']) / df['Close'] * 100
    features['dist_to_low_20'] = (df['Close'] - df['Low'].rolling(20).min()) / df['Close'] * 100

    # Day of week / hour features (for intraday)
    if hasattr(df.index, 'dayofweek'):
        features['day_of_week'] = df.index.dayofweek
        features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Load real cached data if available
    features = _add_cached_macro_features(features)

    # Drop NaN rows from rolling windows
    features = features.fillna(method='ffill').fillna(0)

    return features


def _add_cached_macro_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add real macro data from cached FRED/Fear&Greed files."""
    try:
        # Fear & Greed
        fg_path = os.path.join(PROJECT_ROOT, 'data', 'crypto_cache', 'fear_greed.json')
        if os.path.exists(fg_path):
            with open(fg_path, 'r') as f:
                fg = json.load(f)
            fg_value = fg.get('data', {}).get('value')
            if fg_value is not None:
                features['fear_greed'] = fg_value  # Static for now, will be time-series later

        # FRED indicators
        econ_dir = os.path.join(PROJECT_ROOT, 'data', 'economic_cache')
        fred_map = {
            'fred_DFF': 'fed_funds_rate',
            'fred_DGS10': 'treasury_10y',
            'fred_DGS2': 'treasury_2y',
            'fred_T10Y2Y': 'yield_curve',
        }
        for fname, feature_name in fred_map.items():
            path = os.path.join(econ_dir, f'{fname}.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                vals = data.get('data', [])
                if vals:
                    features[feature_name] = vals[0].get('value', 0)

    except Exception as e:
        logger.debug(f"Failed to add cached macro features: {e}")

    return features


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series) -> Tuple:
    """Train XGBoost classifier."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed. Run: pip install xgboost")
        return None, {}

    # Map labels: -1,0,1 → 0,1,2 for multiclass
    label_map = {-1: 0, 0: 1, 1: 2}
    y_train_mapped = y_train.map(label_map)
    y_val_mapped = y_val.map(label_map)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        X_train, y_train_mapped,
        eval_set=[(X_val, y_val_mapped)],
        verbose=False
    )

    # Evaluate
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val_mapped, y_pred)

    # Feature importance
    importance = dict(zip(X_train.columns, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:15]

    metrics = {
        'accuracy': round(accuracy, 4),
        'top_features': top_features,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_features': len(X_train.columns),
        'label_distribution': y_train.value_counts().to_dict(),
    }

    logger.info(f"XGBoost trained: accuracy={accuracy:.3f}, "
                f"features={len(X_train.columns)}, "
                f"train={len(X_train)}, val={len(X_val)}")
    logger.info(f"Top 5 features: {[f[0] for f in top_features[:5]]}")

    return model, metrics


def save_model(model, metrics: Dict, symbol: str, model_dir: str = None):
    """Save trained model and metadata."""
    if model_dir is None:
        model_dir = os.path.join(PROJECT_ROOT, 'ml_models', 'models')

    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, f'xgboost_{symbol.replace("/", "_")}.json')
    model.save_model(model_path)

    # Save metadata
    meta_path = os.path.join(model_dir, f'xgboost_{symbol.replace("/", "_")}_meta.json')
    meta = {
        'symbol': symbol,
        'model_type': 'XGBoost',
        'trained_at': datetime.now().isoformat(),
        'metrics': metrics,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"Model saved: {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost on crypto data')
    parser.add_argument('--symbol', default='BTC-USD', help='Symbol to train on')
    parser.add_argument('--period', default='1y', help='Data period (1y, 2y, etc)')
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon (bars)')
    parser.add_argument('--threshold', type=float, default=1.0, help='Direction threshold %')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s %(name)s %(levelname)s: %(message)s')

    print("=" * 60)
    print(f"TRAINING XGBoost: {args.symbol}")
    print(f"Period: {args.period} | Horizon: {args.horizon} bars | Threshold: {args.threshold}%")
    print("=" * 60)

    # 1. Fetch data
    df = fetch_training_data(args.symbol, args.period)
    if len(df) < 100:
        print(f"Insufficient data: {len(df)} bars (need 100+)")
        return

    # 2. Create features and labels
    features = engineer_features(df, args.symbol)
    labels = create_labels(df, args.horizon, args.threshold)

    # Align
    common_idx = features.index.intersection(labels.dropna().index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    # Remove any remaining NaN
    valid = features.notna().all(axis=1) & labels.notna()
    features = features[valid]
    labels = labels[valid]

    print(f"\nDataset: {len(features)} samples, {len(features.columns)} features")
    print(f"Label distribution: {labels.value_counts().to_dict()}")

    # 3. Train/val split (time-series: last 20% for validation)
    split_idx = int(len(features) * 0.8)
    X_train, X_val = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_val = labels.iloc[:split_idx], labels.iloc[split_idx:]

    print(f"Train: {len(X_train)} | Validation: {len(X_val)}")

    # 4. Train
    model, metrics = train_xgboost(X_train, y_train, X_val, y_val)
    if model is None:
        return

    # 5. Save
    model_path = save_model(model, metrics, args.symbol)

    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Features: {metrics['n_features']}")
    print(f"  Train/Val: {metrics['n_train']}/{metrics['n_val']}")
    print(f"  Model: {model_path}")
    print(f"\nTop 10 Features:")
    for feat, imp in metrics['top_features'][:10]:
        print(f"  {feat}: {imp:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
