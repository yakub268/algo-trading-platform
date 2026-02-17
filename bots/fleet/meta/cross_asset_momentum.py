"""
Cross-Asset Momentum — Multi-asset regime overlay for crypto sizing.

Monitors SPY, VIX, DXY equivalents to create a risk score.
When risk-off: reduce crypto position sizes across the fleet.
When risk-on: allow full position sizes.

Since we don't have direct SPY/VIX data via Alpaca crypto,
we use proxy signals:
- BTC 7-day momentum as risk sentiment proxy
- Fear & Greed Index for crypto-specific sentiment
- BTC-ETH correlation for market stress
"""

import os
import sys
import logging
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from bots.fleet.shared.fleet_bot import FleetBot, FleetSignal, FleetBotConfig, BotType

logger = logging.getLogger('Fleet.CrossAssetMomentum')

# Risk score thresholds
RISK_OFF_THRESHOLD = 0.3   # Below this → reduce positions
RISK_ON_THRESHOLD = 0.7    # Above this → full positions
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"

# Regime scaling factors
RISK_SCALE = {
    'risk_off': 0.3,    # 30% of normal sizing
    'cautious': 0.6,    # 60%
    'neutral': 1.0,     # 100%
    'risk_on': 1.2,     # 120% (slight boost)
}


class CrossAssetMomentum(FleetBot):
    """
    Cross-asset regime overlay. Doesn't trade directly — produces risk score
    that FleetOrchestrator uses to scale crypto position sizes.
    """

    def __init__(self, config: FleetBotConfig = None):
        if config is None:
            config = FleetBotConfig(
                name='Cross-Asset-Momentum',
                bot_type=BotType.META,
                schedule_seconds=1800,
                max_position_usd=0,
                max_daily_trades=50,
                min_confidence=0.0,
                enabled=True,
                paper_mode=True,
            )
        super().__init__(config)
        self._alpaca = None
        self._risk_score: float = 0.5  # Neutral default
        self._regime: str = 'neutral'
        self._scale_factor: float = 1.0
        self._components: Dict[str, float] = {}
        self._cache: Dict[str, Tuple[datetime, Any]] = {}

    def _get_alpaca(self):
        if self._alpaca is None:
            try:
                from bots.alpaca_crypto_client import AlpacaCryptoClient
                self._alpaca = AlpacaCryptoClient()
            except Exception as e:
                self.logger.error(f"Alpaca init failed: {e}")
        return self._alpaca

    def scan(self) -> List[FleetSignal]:
        """
        Compute risk score from multiple signals.
        Returns a single meta-signal with the risk assessment.
        """
        components = {}

        # Component 1: BTC 7-day momentum (0-1)
        btc_momentum = self._get_btc_momentum()
        if btc_momentum is not None:
            # Normalize: -20% to +20% range → 0-1
            components['btc_momentum'] = max(0, min(1, (btc_momentum + 0.20) / 0.40))

        # Component 2: Fear & Greed Index (0-1)
        fng = self._get_fear_greed()
        if fng is not None:
            components['fear_greed'] = fng / 100.0

        # Component 3: BTC-ETH correlation stability (0-1)
        corr_score = self._get_correlation_score()
        if corr_score is not None:
            components['btc_eth_corr'] = corr_score

        # Component 4: BTC volatility (inverse — high vol = lower score)
        vol_score = self._get_volatility_score()
        if vol_score is not None:
            components['volatility'] = vol_score

        if not components:
            self.logger.warning("No risk components available")
            return []

        # Weighted average
        weights = {
            'btc_momentum': 0.30,
            'fear_greed': 0.30,
            'btc_eth_corr': 0.20,
            'volatility': 0.20,
        }

        total_weight = sum(weights[k] for k in components)
        self._risk_score = sum(
            components[k] * weights[k] for k in components
        ) / total_weight if total_weight > 0 else 0.5

        # Determine regime
        if self._risk_score < RISK_OFF_THRESHOLD:
            self._regime = 'risk_off'
        elif self._risk_score < 0.45:
            self._regime = 'cautious'
        elif self._risk_score < RISK_ON_THRESHOLD:
            self._regime = 'neutral'
        else:
            self._regime = 'risk_on'

        self._scale_factor = RISK_SCALE[self._regime]
        self._components = components

        self.logger.info(
            f"Risk score: {self._risk_score:.2f} → {self._regime} "
            f"(scale={self._scale_factor}x) components={components}"
        )

        return [FleetSignal(
            bot_name=self.name,
            bot_type=BotType.META.value,
            symbol='RISK_SCORE',
            side='ASSESS',
            entry_price=0,
            confidence=self._risk_score,
            edge=self._scale_factor,
            reason=f"Regime: {self._regime}, score={self._risk_score:.2f}",
            metadata={
                'risk_score': self._risk_score,
                'regime': self._regime,
                'scale_factor': self._scale_factor,
                'components': components,
            },
        )]

    def get_scale_factor(self) -> float:
        """Get current position scale factor. Used by FleetOrchestrator for crypto bots."""
        return self._scale_factor

    def get_regime(self) -> str:
        """Get current risk regime."""
        return self._regime

    def _get_btc_momentum(self) -> Optional[float]:
        """7-day BTC return."""
        try:
            alpaca = self._get_alpaca()
            if not alpaca:
                return None
            candles = alpaca.get_candles('BTC/USD', 'day', 8)
            if not candles or len(candles) < 2:
                return None
            current = candles[-1]['close']
            week_ago = candles[0]['close']
            return (current - week_ago) / week_ago
        except Exception as e:
            self.logger.debug(f"BTC momentum fetch failed: {e}")
            return None

    def _get_fear_greed(self) -> Optional[float]:
        """Fetch Fear & Greed Index (0-100)."""
        # 30-min cache
        cached = self._cache.get('fng')
        if cached and (datetime.now(timezone.utc) - cached[0]).total_seconds() < 1800:
            return cached[1]

        try:
            resp = requests.get(FEAR_GREED_URL, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                value = float(data['data'][0]['value'])
                self._cache['fng'] = (datetime.now(timezone.utc), value)
                return value
        except Exception as e:
            self.logger.debug(f"F&G fetch failed: {e}")
        return None

    def _get_correlation_score(self) -> Optional[float]:
        """BTC-ETH correlation. High correlation = normal. Low = stress."""
        try:
            alpaca = self._get_alpaca()
            if not alpaca:
                return None
            btc_candles = alpaca.get_candles('BTC/USD', 'hour', 48)
            eth_candles = alpaca.get_candles('ETH/USD', 'hour', 48)
            if not btc_candles or not eth_candles:
                return None

            btc_returns = []
            eth_returns = []
            min_len = min(len(btc_candles), len(eth_candles))
            for i in range(1, min_len):
                btc_returns.append(
                    (btc_candles[i]['close'] - btc_candles[i-1]['close']) / btc_candles[i-1]['close']
                )
                eth_returns.append(
                    (eth_candles[i]['close'] - eth_candles[i-1]['close']) / eth_candles[i-1]['close']
                )

            if len(btc_returns) < 10:
                return None

            corr = np.corrcoef(btc_returns, eth_returns)[0, 1]
            # Normal correlation ~0.8-0.95. Below 0.5 = stress
            return max(0, min(1, (corr - 0.3) / 0.6))
        except Exception as e:
            self.logger.debug(f"Correlation calc failed: {e}")
            return None

    def _get_volatility_score(self) -> Optional[float]:
        """Inverse BTC volatility score. Low vol = high score."""
        try:
            alpaca = self._get_alpaca()
            if not alpaca:
                return None
            candles = alpaca.get_candles('BTC/USD', 'hour', 48)
            if not candles or len(candles) < 20:
                return None

            returns = []
            for i in range(1, len(candles)):
                returns.append(
                    (candles[i]['close'] - candles[i-1]['close']) / candles[i-1]['close']
                )

            vol = np.std(returns) * np.sqrt(24)  # Annualize hourly to daily
            # Typical BTC daily vol: 2-8%. Map 1-10% → 1-0
            score = max(0, min(1, 1 - (vol - 0.01) / 0.09))
            return score
        except Exception as e:
            self.logger.debug(f"Volatility calc failed: {e}")
            return None

    def get_status(self) -> Dict:
        base = super().get_status()
        base['risk_score'] = self._risk_score
        base['regime'] = self._regime
        base['scale_factor'] = self._scale_factor
        base['components'] = self._components
        return base
