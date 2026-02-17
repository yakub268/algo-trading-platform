"""
Portfolio Heat Monitor
=====================

Real-time monitoring of portfolio risk exposure across all positions.
Tracks risk concentration, sector allocation, and overall portfolio heat.

Author: Trading Bot Arsenal
Created: February 2026
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import yfinance as yf

from ..config.risk_config import RiskManagementConfig, AlertSeverity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PortfolioHeat')


@dataclass
class Position:
    """Individual position details"""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    sector: str
    strategy: str
    risk_amount: float
    correlation_group: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return abs(self.size) * self.current_price

    @property
    def weight(self) -> float:
        """Position weight as percentage (needs portfolio value)"""
        return self.market_value

    @property
    def pnl_percent(self) -> float:
        """Unrealized P&L as percentage"""
        return (self.current_price - self.entry_price) / self.entry_price

    def update_price(self, new_price: float):
        """Update current price and recalculate P&L"""
        self.current_price = new_price
        self.unrealized_pnl = self.size * (new_price - self.entry_price)


@dataclass
class HeatMetrics:
    """Portfolio heat analysis metrics"""
    total_exposure: float
    cash_available: float
    portfolio_value: float

    # Concentration metrics
    max_single_position_pct: float
    max_sector_exposure_pct: float
    max_strategy_exposure_pct: float
    max_correlation_group_pct: float

    # Risk metrics
    total_risk_amount: float
    portfolio_beta: float
    diversification_ratio: float

    # Heat scores (0-100)
    concentration_heat: float
    sector_heat: float
    correlation_heat: float
    overall_heat: float

    # Warnings and limits
    warnings: List[str] = field(default_factory=list)
    limit_breaches: List[str] = field(default_factory=list)


class PortfolioHeatMonitor:
    """
    Real-time portfolio heat monitoring system.

    Tracks:
    - Position concentration
    - Sector exposure
    - Strategy allocation
    - Correlation clustering
    - Risk budget utilization
    """

    # Sector mapping for common symbols
    SECTOR_MAPPING = {
        # Broad Market ETFs
        'SPY': 'Broad Market', 'QQQ': 'Technology', 'IWM': 'Small Cap',
        'VOO': 'Broad Market', 'VTI': 'Broad Market', 'VEA': 'International',

        # Sector ETFs
        'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare',
        'XLE': 'Energy', 'XLI': 'Industrials', 'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples', 'XLU': 'Utilities', 'XLRE': 'Real Estate',
        'XLC': 'Communication Services', 'XLB': 'Materials',

        # Individual Stocks
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
        'META': 'Technology', 'NVDA': 'Technology', 'JPM': 'Financials',
        'BAC': 'Financials', 'JNJ': 'Healthcare', 'PG': 'Consumer Staples',

        # Crypto
        'BTC': 'Cryptocurrency', 'ETH': 'Cryptocurrency', 'SOL': 'Cryptocurrency',

        # Forex (simplified)
        'EUR/USD': 'Forex', 'GBP/USD': 'Forex', 'USD/JPY': 'Forex',
        'AUD/USD': 'Forex', 'USD/CHF': 'Forex', 'USD/CAD': 'Forex'
    }

    # Correlation groups
    CORRELATION_GROUPS = {
        'US_LARGE_CAP': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        'TECH_STOCKS': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN'],
        'CRYPTO': ['BTC', 'ETH', 'SOL', 'ADA', 'DOT'],
        'FOREX_MAJORS': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'],
        'ENERGY': ['XLE', 'XOM', 'CVX', 'COP'],
        'FINANCIALS': ['XLF', 'JPM', 'BAC', 'GS', 'WFC']
    }

    def __init__(self, config: RiskManagementConfig):
        """
        Initialize portfolio heat monitor.

        Args:
            config: Risk management configuration
        """
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = config.portfolio_value
        self.last_update = datetime.now()

        # Historical tracking
        self.heat_history: List[HeatMetrics] = []
        self.max_heat_lookback = 100

        logger.info("PortfolioHeatMonitor initialized")

    def add_position(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        strategy: str,
        risk_amount: float
    ) -> bool:
        """
        Add new position to monitoring.

        Args:
            symbol: Trading symbol
            size: Position size (positive for long, negative for short)
            entry_price: Entry price
            strategy: Strategy name
            risk_amount: Risk amount for this position

        Returns:
            True if position added successfully
        """
        try:
            # Get current price
            current_price = self._get_current_price(symbol)
            if current_price is None:
                current_price = entry_price
                logger.warning(f"Could not get current price for {symbol}, using entry price")

            # Determine sector and correlation group
            sector = self._get_sector(symbol)
            correlation_group = self._get_correlation_group(symbol)

            # Calculate unrealized P&L
            unrealized_pnl = size * (current_price - entry_price)

            # Create position
            position = Position(
                symbol=symbol,
                size=size,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                sector=sector,
                strategy=strategy,
                risk_amount=risk_amount,
                correlation_group=correlation_group
            )

            self.positions[symbol] = position

            logger.info(f"Added position: {symbol} {size:+.2f} @ ${entry_price:.2f} (Risk: ${risk_amount:.2f})")

            # Update portfolio metrics
            self._update_portfolio_value()

            return True

        except Exception as e:
            logger.error(f"Failed to add position {symbol}: {e}")
            return False

    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove position from monitoring"""
        if symbol in self.positions:
            position = self.positions.pop(symbol)
            logger.info(f"Removed position: {symbol}")
            self._update_portfolio_value()
            return position
        return None

    def update_prices(self, price_data: Dict[str, float] = None) -> bool:
        """
        Update all position prices.

        Args:
            price_data: Optional dict of symbol -> price, otherwise fetches from API

        Returns:
            True if update successful
        """
        try:
            if not self.positions:
                return True

            if price_data is None:
                price_data = self._fetch_current_prices()

            for symbol, position in self.positions.items():
                if symbol in price_data:
                    position.update_price(price_data[symbol])
                else:
                    logger.warning(f"No price data for {symbol}")

            self._update_portfolio_value()
            self.last_update = datetime.now()

            return True

        except Exception as e:
            logger.error(f"Failed to update prices: {e}")
            return False

    def calculate_heat_metrics(self) -> HeatMetrics:
        """
        Calculate comprehensive portfolio heat metrics.

        Returns:
            HeatMetrics with all portfolio risk measures
        """
        if not self.positions:
            return HeatMetrics(
                total_exposure=0.0,
                cash_available=self.portfolio_value,
                portfolio_value=self.portfolio_value,
                max_single_position_pct=0.0,
                max_sector_exposure_pct=0.0,
                max_strategy_exposure_pct=0.0,
                max_correlation_group_pct=0.0,
                total_risk_amount=0.0,
                portfolio_beta=1.0,
                diversification_ratio=1.0,
                concentration_heat=0.0,
                sector_heat=0.0,
                correlation_heat=0.0,
                overall_heat=0.0
            )

        # Calculate basic metrics
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_risk_amount = sum(pos.risk_amount for pos in self.positions.values())
        cash_available = self.portfolio_value - total_market_value

        # Position concentration
        position_weights = {symbol: pos.market_value / self.portfolio_value
                          for symbol, pos in self.positions.items()}
        max_single_position_pct = max(position_weights.values()) if position_weights else 0.0

        # Sector exposure
        sector_exposure = self._calculate_sector_exposure()
        max_sector_exposure_pct = max(sector_exposure.values()) if sector_exposure else 0.0

        # Strategy exposure
        strategy_exposure = self._calculate_strategy_exposure()
        max_strategy_exposure_pct = max(strategy_exposure.values()) if strategy_exposure else 0.0

        # Correlation group exposure
        correlation_exposure = self._calculate_correlation_exposure()
        max_correlation_group_pct = max(correlation_exposure.values()) if correlation_exposure else 0.0

        # Portfolio beta (simplified)
        portfolio_beta = self._calculate_portfolio_beta()

        # Diversification ratio
        diversification_ratio = self._calculate_diversification_ratio()

        # Heat scores (0-100)
        concentration_heat = self._calculate_concentration_heat(position_weights)
        sector_heat = self._calculate_sector_heat(sector_exposure)
        correlation_heat = self._calculate_correlation_heat(correlation_exposure)
        overall_heat = (concentration_heat + sector_heat + correlation_heat) / 3

        # Check for warnings and limit breaches
        warnings = []
        limit_breaches = []

        # Check concentration limits
        if max_single_position_pct > self.config.portfolio_limits.max_single_position:
            limit_breaches.append(f"Single position limit breached: {max_single_position_pct:.1%}")
        elif max_single_position_pct > self.config.portfolio_limits.max_single_position * 0.8:
            warnings.append(f"Approaching single position limit: {max_single_position_pct:.1%}")

        if max_sector_exposure_pct > self.config.portfolio_limits.max_sector_exposure:
            limit_breaches.append(f"Sector exposure limit breached: {max_sector_exposure_pct:.1%}")
        elif max_sector_exposure_pct > self.config.portfolio_limits.max_sector_exposure * 0.8:
            warnings.append(f"Approaching sector limit: {max_sector_exposure_pct:.1%}")

        # Check total exposure
        total_exposure_pct = total_market_value / self.portfolio_value
        if total_exposure_pct > self.config.portfolio_limits.max_total_exposure:
            limit_breaches.append(f"Total exposure limit breached: {total_exposure_pct:.1%}")

        # Check cash reserve
        cash_pct = cash_available / self.portfolio_value
        if cash_pct < self.config.portfolio_limits.min_cash_reserve:
            warnings.append(f"Below minimum cash reserve: {cash_pct:.1%}")

        metrics = HeatMetrics(
            total_exposure=total_exposure_pct,
            cash_available=cash_available,
            portfolio_value=self.portfolio_value,
            max_single_position_pct=max_single_position_pct,
            max_sector_exposure_pct=max_sector_exposure_pct,
            max_strategy_exposure_pct=max_strategy_exposure_pct,
            max_correlation_group_pct=max_correlation_group_pct,
            total_risk_amount=total_risk_amount,
            portfolio_beta=portfolio_beta,
            diversification_ratio=diversification_ratio,
            concentration_heat=concentration_heat,
            sector_heat=sector_heat,
            correlation_heat=correlation_heat,
            overall_heat=overall_heat,
            warnings=warnings,
            limit_breaches=limit_breaches
        )

        # Store in history
        self.heat_history.append(metrics)
        if len(self.heat_history) > self.max_heat_lookback:
            self.heat_history = self.heat_history[-self.max_heat_lookback:]

        return metrics

    def get_position_recommendations(self) -> Dict[str, str]:
        """
        Get recommendations for position adjustments.

        Returns:
            Dict mapping symbols to recommendation strings
        """
        recommendations = {}
        metrics = self.calculate_heat_metrics()

        # Concentration recommendations
        for symbol, position in self.positions.items():
            weight = position.market_value / self.portfolio_value

            if weight > self.config.portfolio_limits.max_single_position:
                recommendations[symbol] = f"REDUCE: Position too large ({weight:.1%})"
            elif weight > self.config.portfolio_limits.max_single_position * 0.8:
                recommendations[symbol] = f"MONITOR: Approaching limit ({weight:.1%})"

        return recommendations

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol"""
        if self._is_prediction_market_ticker(symbol):
            return 'Prediction Markets'
        return self.SECTOR_MAPPING.get(symbol.upper(), 'Unknown')

    def _get_correlation_group(self, symbol: str) -> str:
        """Get correlation group for a symbol"""
        if self._is_prediction_market_ticker(symbol):
            return 'PREDICTION_MARKETS'
        symbol = symbol.upper()
        for group, symbols in self.CORRELATION_GROUPS.items():
            if symbol in symbols:
                return group
        return 'OTHER'

    @staticmethod
    def _is_prediction_market_ticker(symbol: str) -> bool:
        """Check if a ticker is a prediction market contract (e.g. Kalshi).
        Kalshi tickers start with 'KX' and contain hyphens like 'KXHIGHNY-26FEB04-T42'."""
        s = symbol.upper()
        return s.startswith('KX') or ('-T' in s and '-' in s and any(c.isdigit() for c in s))

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price for symbol"""
        try:
            # Handle forex symbols
            if '/' in symbol:
                # For forex, return a mock price (in real implementation, use forex API)
                return 1.0

            # Prediction market contracts (Kalshi etc.) don't exist on yfinance
            if self._is_prediction_market_ticker(symbol):
                logger.debug(f"Skipping yfinance lookup for prediction market ticker: {symbol}")
                return None

            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            if len(data) > 0:
                return float(data['Close'].iloc[-1])
            return None

        except Exception as e:
            logger.debug(f"Could not fetch price for {symbol}: {e}")
            return None

    def _fetch_current_prices(self) -> Dict[str, float]:
        """Fetch current prices for all positions"""
        price_data = {}
        symbols = list(self.positions.keys())

        try:
            # Separate symbols by type
            stock_symbols = [s for s in symbols
                           if '/' not in s and not self._is_prediction_market_ticker(s)]
            forex_symbols = [s for s in symbols if '/' in s]
            prediction_symbols = [s for s in symbols if self._is_prediction_market_ticker(s)]

            # Fetch stock prices in batch
            if stock_symbols:
                tickers = yf.Tickers(' '.join(stock_symbols))
                for symbol in stock_symbols:
                    try:
                        data = tickers.tickers[symbol].history(period='1d')
                        if len(data) > 0:
                            price_data[symbol] = float(data['Close'].iloc[-1])
                    except Exception as e:
                        logger.debug(f"Error fetching price for {symbol}: {e}")

            # Handle forex symbols (mock prices for now)
            for symbol in forex_symbols:
                price_data[symbol] = 1.0  # Mock price

            # For prediction market contracts, use the current_price already on the position
            for symbol in prediction_symbols:
                if symbol in self.positions:
                    price_data[symbol] = self.positions[symbol].current_price
                    logger.debug(f"Using existing price for prediction market ticker: {symbol}")

        except Exception as e:
            logger.error(f"Error fetching batch prices: {e}")

        return price_data

    def _update_portfolio_value(self):
        """Update total portfolio value including unrealized P&L"""
        if not self.positions:
            return

        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        # Note: This is a simplified calculation
        # In reality, you'd need to track cash flows and realized P&L

    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate exposure by sector"""
        sector_exposure = {}

        for position in self.positions.values():
            sector = position.sector
            weight = position.market_value / self.portfolio_value
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight

        return sector_exposure

    def _calculate_strategy_exposure(self) -> Dict[str, float]:
        """Calculate exposure by strategy"""
        strategy_exposure = {}

        for position in self.positions.values():
            strategy = position.strategy
            weight = position.market_value / self.portfolio_value
            strategy_exposure[strategy] = strategy_exposure.get(strategy, 0) + weight

        return strategy_exposure

    def _calculate_correlation_exposure(self) -> Dict[str, float]:
        """Calculate exposure by correlation group"""
        correlation_exposure = {}

        for position in self.positions.values():
            group = position.correlation_group
            weight = position.market_value / self.portfolio_value
            correlation_exposure[group] = correlation_exposure.get(group, 0) + weight

        return correlation_exposure

    def _calculate_portfolio_beta(self) -> float:
        """Calculate simplified portfolio beta"""
        # Simplified beta calculation - in practice would use regression vs market
        total_weight = 0
        weighted_beta = 0

        # Assign approximate betas by sector
        sector_betas = {
            'Technology': 1.3,
            'Broad Market': 1.0,
            'Healthcare': 0.9,
            'Utilities': 0.7,
            'Financials': 1.1,
            'Energy': 1.2,
            'Consumer Staples': 0.8,
            'Cryptocurrency': 2.0,
            'Forex': 0.5,
            'Prediction Markets': 0.1  # Largely uncorrelated with equity markets
        }

        for position in self.positions.values():
            weight = position.market_value / self.portfolio_value
            beta = sector_betas.get(position.sector, 1.0)
            weighted_beta += weight * beta
            total_weight += weight

        return weighted_beta / max(total_weight, 0.001)

    def _calculate_diversification_ratio(self) -> float:
        """Calculate portfolio diversification ratio"""
        if len(self.positions) <= 1:
            return 1.0

        # Simplified diversification metric
        # In practice, would use correlation matrix and volatilities
        n_positions = len(self.positions)
        equal_weight_score = 1.0 / n_positions

        # Calculate how close to equal weight
        weights = [pos.market_value / self.portfolio_value for pos in self.positions.values()]
        avg_weight = np.mean(weights)
        weight_concentration = np.sum([(w - avg_weight)**2 for w in weights])

        # Normalize to 0-1 range
        max_concentration = (n_positions - 1) * equal_weight_score**2
        diversification = 1 - (weight_concentration / max(max_concentration, 0.001))

        return max(0, min(1, diversification))

    def _calculate_concentration_heat(self, position_weights: Dict[str, float]) -> float:
        """Calculate concentration heat score (0-100)"""
        if not position_weights:
            return 0.0

        max_weight = max(position_weights.values())
        heat = (max_weight / self.config.portfolio_limits.max_single_position) * 100
        return min(heat, 100.0)

    def _calculate_sector_heat(self, sector_exposure: Dict[str, float]) -> float:
        """Calculate sector concentration heat score (0-100)"""
        if not sector_exposure:
            return 0.0

        max_exposure = max(sector_exposure.values())
        heat = (max_exposure / self.config.portfolio_limits.max_sector_exposure) * 100
        return min(heat, 100.0)

    def _calculate_correlation_heat(self, correlation_exposure: Dict[str, float]) -> float:
        """Calculate correlation heat score (0-100)"""
        if not correlation_exposure:
            return 0.0

        max_exposure = max(correlation_exposure.values())
        heat = (max_exposure / self.config.portfolio_limits.max_correlated_exposure) * 100
        return min(heat, 100.0)

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        metrics = self.calculate_heat_metrics()

        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'positions': len(self.positions),
            'total_exposure': f"{metrics.total_exposure:.1%}",
            'cash_available': f"${metrics.cash_available:,.2f}",
            'heat_scores': {
                'overall': f"{metrics.overall_heat:.1f}/100",
                'concentration': f"{metrics.concentration_heat:.1f}/100",
                'sector': f"{metrics.sector_heat:.1f}/100",
                'correlation': f"{metrics.correlation_heat:.1f}/100"
            },
            'risk_metrics': {
                'total_risk_amount': f"${metrics.total_risk_amount:,.2f}",
                'portfolio_beta': f"{metrics.portfolio_beta:.2f}",
                'diversification_ratio': f"{metrics.diversification_ratio:.2f}"
            },
            'warnings': metrics.warnings,
            'limit_breaches': metrics.limit_breaches,
            'last_update': self.last_update.isoformat()
        }


if __name__ == "__main__":
    from ..config.risk_config import load_risk_config

    # Test the portfolio heat monitor
    config = load_risk_config()
    monitor = PortfolioHeatMonitor(config)

    # Add some test positions
    monitor.add_position('SPY', 100, 450.0, 'momentum', 1000)
    monitor.add_position('QQQ', 50, 350.0, 'rsi_mean_reversion', 800)
    monitor.add_position('AAPL', 25, 180.0, 'momentum', 600)

    # Get metrics
    metrics = monitor.calculate_heat_metrics()
    print("Portfolio Heat Metrics:")
    print(f"Overall Heat: {metrics.overall_heat:.1f}/100")
    print(f"Concentration Heat: {metrics.concentration_heat:.1f}/100")
    print(f"Sector Heat: {metrics.sector_heat:.1f}/100")
    print(f"Warnings: {metrics.warnings}")
    print(f"Limit Breaches: {metrics.limit_breaches}")