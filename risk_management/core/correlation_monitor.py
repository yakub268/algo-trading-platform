"""
Correlation Monitor
==================

Advanced correlation analysis and monitoring system to prevent
over-concentration in correlated assets.

Features:
- Real-time correlation calculation
- Dynamic correlation clustering
- Risk-adjusted correlation limits
- Correlation-based position sizing
- Early warning system

Author: Trading Bot Arsenal
Created: February 2026
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import yfinance as yf
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import warnings

from ..config.risk_config import RiskManagementConfig, AlertSeverity

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CorrelationMonitor')


@dataclass
class CorrelationCluster:
    """Correlation cluster information"""
    cluster_id: int
    symbols: List[str]
    avg_correlation: float
    max_correlation: float
    total_weight: float
    risk_contribution: float
    warning_level: AlertSeverity = AlertSeverity.INFO


@dataclass
class CorrelationAlert:
    """Correlation-based alert"""
    alert_type: str
    severity: AlertSeverity
    symbols: List[str]
    correlation_value: float
    limit: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CorrelationMetrics:
    """Comprehensive correlation analysis results"""
    correlation_matrix: pd.DataFrame
    clusters: List[CorrelationCluster]
    alerts: List[CorrelationAlert]

    # Summary metrics
    avg_portfolio_correlation: float
    max_pairwise_correlation: float
    correlation_concentration_ratio: float

    # Risk metrics
    correlation_adjusted_risk: float
    diversification_benefit: float

    # Warnings
    high_correlation_pairs: List[Tuple[str, str, float]]
    concentration_warnings: List[str]


class CorrelationMonitor:
    """
    Advanced correlation monitoring for portfolio risk management.

    Monitors correlation patterns across all positions and provides
    early warnings when correlation risk becomes excessive.
    """

    def __init__(self, config: RiskManagementConfig):
        """
        Initialize correlation monitor.

        Args:
            config: Risk management configuration
        """
        self.config = config
        self.correlation_window = config.correlation_limits.correlation_window
        self.max_correlation_threshold = config.correlation_limits.max_correlation_threshold

        # Data storage
        self.price_history: Dict[str, pd.Series] = {}
        self.correlation_history: List[pd.DataFrame] = []
        self.cluster_history: List[List[CorrelationCluster]] = []

        # Current state
        self.current_positions: Dict[str, float] = {}  # symbol -> weight
        self.last_correlation_matrix: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None

        logger.info("CorrelationMonitor initialized")

    def update_positions(self, positions: Dict[str, float]):
        """
        Update current position weights.

        Args:
            positions: Dict mapping symbol -> portfolio weight
        """
        self.current_positions = positions.copy()
        logger.info(f"Updated positions: {len(positions)} symbols")

    @staticmethod
    def _is_prediction_market_ticker(symbol: str) -> bool:
        """Check if a ticker is a prediction market contract (e.g. Kalshi).
        Kalshi tickers start with 'KX' and contain hyphens like 'KXHIGHNY-26FEB04-T42'."""
        s = symbol.upper()
        return s.startswith('KX') or ('-T' in s and '-' in s and any(c.isdigit() for c in s))

    def fetch_price_data(self, symbols: List[str], period: str = None) -> bool:
        """
        Fetch historical price data for correlation calculation.

        Args:
            symbols: List of symbols to fetch
            period: Historical period (default: correlation_window days)

        Returns:
            True if successful
        """
        try:
            if period is None:
                period = f"{self.correlation_window}d"

            # Fetch data for all symbols
            if not symbols:
                return True

            # Filter out forex, currency pairs, and prediction market tickers
            stock_symbols = [s for s in symbols
                           if '/' not in s and '=' not in s
                           and not self._is_prediction_market_ticker(s)]

            # Log skipped prediction market tickers
            skipped = [s for s in symbols if self._is_prediction_market_ticker(s)]
            if skipped:
                logger.info(f"Skipping prediction market tickers for correlation fetch: {skipped}")

            if not stock_symbols:
                return True

            # Batch download
            data = yf.download(
                stock_symbols,
                period=period,
                interval='1d',
                group_by='ticker',
                auto_adjust=True,
                prepost=True,
                threads=True
            )

            # Process data
            for symbol in stock_symbols:
                try:
                    if len(stock_symbols) == 1:
                        prices = data['Close']
                    else:
                        prices = data[symbol]['Close'] if (symbol, 'Close') in data.columns else data[(symbol, 'Close')]

                    # Clean data
                    prices = prices.dropna()
                    if len(prices) > 10:  # Minimum data points
                        self.price_history[symbol] = prices

                except Exception as e:
                    logger.warning(f"Could not process data for {symbol}: {e}")

            logger.info(f"Fetched price data for {len(self.price_history)} symbols")
            return True

        except Exception as e:
            logger.error(f"Failed to fetch price data: {e}")
            return False

    def calculate_correlations(self) -> Optional[pd.DataFrame]:
        """
        Calculate correlation matrix from price data.

        Returns:
            Correlation matrix or None if insufficient data
        """
        try:
            if len(self.price_history) < 2:
                logger.debug("Insufficient symbols for correlation calculation (need 2+ positions)")
                return None

            # Align all price series by date
            symbols = list(self.price_history.keys())
            aligned_data = {}

            # Find common date range
            all_dates = None
            for symbol, prices in self.price_history.items():
                if all_dates is None:
                    all_dates = set(prices.index)
                else:
                    all_dates = all_dates.intersection(set(prices.index))

            if len(all_dates) < 20:  # Minimum observations
                logger.warning("Insufficient overlapping data for correlation")
                return None

            common_dates = sorted(list(all_dates))

            # Create aligned price matrix
            for symbol in symbols:
                aligned_prices = self.price_history[symbol].loc[common_dates]
                aligned_data[symbol] = aligned_prices

            price_df = pd.DataFrame(aligned_data)

            # Calculate returns
            returns = price_df.pct_change().dropna()

            if len(returns) < 20:
                logger.warning("Insufficient return data for correlation")
                return None

            # Calculate correlation matrix
            correlation_matrix = returns.corr()

            # Store for history
            self.last_correlation_matrix = correlation_matrix
            self.correlation_history.append(correlation_matrix)

            # Keep limited history
            if len(self.correlation_history) > 50:
                self.correlation_history = self.correlation_history[-50:]

            logger.info(f"Calculated correlations for {len(symbols)} symbols")
            return correlation_matrix

        except Exception as e:
            logger.error(f"Failed to calculate correlations: {e}")
            return None

    def perform_cluster_analysis(
        self,
        correlation_matrix: pd.DataFrame,
        n_clusters: int = None
    ) -> List[CorrelationCluster]:
        """
        Perform hierarchical clustering on correlation matrix.

        Args:
            correlation_matrix: Correlation matrix
            n_clusters: Number of clusters (auto-determined if None)

        Returns:
            List of correlation clusters
        """
        try:
            symbols = correlation_matrix.index.tolist()

            if len(symbols) < 2:
                return []

            # Convert correlation to distance matrix
            distance_matrix = 1 - correlation_matrix.abs()

            # Perform hierarchical clustering
            if n_clusters is None:
                # Auto-determine number of clusters based on correlation threshold
                n_clusters = max(2, min(len(symbols),
                    len(symbols) // 2))  # Heuristic

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward',
                metric='euclidean'
            )

            # Fit clustering on distance matrix
            condensed_dist = squareform(distance_matrix.values)
            linkage_matrix = linkage(condensed_dist, method='ward')

            # Get cluster labels
            cluster_labels = clustering.fit_predict(distance_matrix.values)

            # Create cluster objects
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_symbols = [symbols[i] for i, label in enumerate(cluster_labels)
                                 if label == cluster_id]

                if len(cluster_symbols) < 2:
                    continue

                # Calculate cluster statistics
                cluster_corr_values = []
                for i, sym1 in enumerate(cluster_symbols):
                    for j, sym2 in enumerate(cluster_symbols):
                        if i < j:
                            corr = correlation_matrix.loc[sym1, sym2]
                            cluster_corr_values.append(abs(corr))

                avg_correlation = np.mean(cluster_corr_values) if cluster_corr_values else 0
                max_correlation = np.max(cluster_corr_values) if cluster_corr_values else 0

                # Calculate total weight and risk contribution
                total_weight = sum(self.current_positions.get(sym, 0) for sym in cluster_symbols)
                risk_contribution = total_weight * avg_correlation

                # Determine warning level
                warning_level = AlertSeverity.INFO
                if total_weight > self.config.correlation_limits.max_correlated_weight:
                    warning_level = AlertSeverity.CRITICAL
                elif total_weight > self.config.correlation_limits.max_correlated_weight * 0.8:
                    warning_level = AlertSeverity.WARNING

                clusters.append(CorrelationCluster(
                    cluster_id=cluster_id,
                    symbols=cluster_symbols,
                    avg_correlation=avg_correlation,
                    max_correlation=max_correlation,
                    total_weight=total_weight,
                    risk_contribution=risk_contribution,
                    warning_level=warning_level
                ))

            # Sort by risk contribution
            clusters.sort(key=lambda x: x.risk_contribution, reverse=True)

            # Store in history
            self.cluster_history.append(clusters)
            if len(self.cluster_history) > 20:
                self.cluster_history = self.cluster_history[-20:]

            logger.info(f"Identified {len(clusters)} correlation clusters")
            return clusters

        except Exception as e:
            logger.error(f"Failed to perform cluster analysis: {e}")
            return []

    def analyze_correlations(self) -> CorrelationMetrics:
        """
        Perform comprehensive correlation analysis.

        Returns:
            Complete correlation analysis results
        """
        # Ensure we have current data
        if self.current_positions:
            symbols = list(self.current_positions.keys())
            self.fetch_price_data(symbols)

        # Calculate correlation matrix
        correlation_matrix = self.calculate_correlations()

        if correlation_matrix is None:
            # Return empty metrics
            return CorrelationMetrics(
                correlation_matrix=pd.DataFrame(),
                clusters=[],
                alerts=[],
                avg_portfolio_correlation=0.0,
                max_pairwise_correlation=0.0,
                correlation_concentration_ratio=0.0,
                correlation_adjusted_risk=0.0,
                diversification_benefit=0.0,
                high_correlation_pairs=[],
                concentration_warnings=[]
            )

        # Perform cluster analysis
        clusters = self.perform_cluster_analysis(correlation_matrix)

        # Calculate summary metrics
        correlations = correlation_matrix.values
        np.fill_diagonal(correlations, np.nan)  # Exclude diagonal

        avg_correlation = np.nanmean(np.abs(correlations))
        max_correlation = np.nanmax(np.abs(correlations))

        # Portfolio weighted correlation
        portfolio_correlation = self._calculate_portfolio_correlation(correlation_matrix)

        # Concentration ratio
        concentration_ratio = self._calculate_concentration_ratio(clusters)

        # Risk metrics
        correlation_adjusted_risk = self._calculate_correlation_adjusted_risk(
            correlation_matrix, clusters
        )
        diversification_benefit = self._calculate_diversification_benefit(correlation_matrix)

        # Generate alerts
        alerts = self._generate_correlation_alerts(correlation_matrix, clusters)

        # Find high correlation pairs
        high_correlation_pairs = self._find_high_correlation_pairs(correlation_matrix)

        # Generate warnings
        concentration_warnings = self._generate_concentration_warnings(clusters)

        metrics = CorrelationMetrics(
            correlation_matrix=correlation_matrix,
            clusters=clusters,
            alerts=alerts,
            avg_portfolio_correlation=avg_correlation,
            max_pairwise_correlation=max_correlation,
            correlation_concentration_ratio=concentration_ratio,
            correlation_adjusted_risk=correlation_adjusted_risk,
            diversification_benefit=diversification_benefit,
            high_correlation_pairs=high_correlation_pairs,
            concentration_warnings=concentration_warnings
        )

        self.last_update = datetime.now()
        return metrics

    def check_correlation_limits(self, new_symbol: str, new_weight: float) -> Tuple[bool, List[str]]:
        """
        Check if adding a new position would violate correlation limits.

        Args:
            new_symbol: Symbol to add
            new_weight: Portfolio weight of new position

        Returns:
            Tuple of (allowed, list_of_warnings)
        """
        warnings = []

        if self.last_correlation_matrix is None:
            return True, ["No correlation data available"]

        # Check direct correlations with existing positions
        high_correlations = []
        for existing_symbol, existing_weight in self.current_positions.items():
            if existing_symbol in self.last_correlation_matrix.index and \
               new_symbol in self.last_correlation_matrix.index:

                corr = abs(self.last_correlation_matrix.loc[existing_symbol, new_symbol])

                if corr > self.max_correlation_threshold:
                    combined_weight = existing_weight + new_weight
                    high_correlations.append((existing_symbol, corr, combined_weight))

        # Check if any highly correlated combinations exceed limits
        for existing_symbol, corr, combined_weight in high_correlations:
            if combined_weight > self.config.correlation_limits.max_correlated_weight:
                warnings.append(
                    f"High correlation with {existing_symbol} ({corr:.2f}) "
                    f"would create {combined_weight:.1%} exposure"
                )

        # Check cluster-based limits
        temp_positions = self.current_positions.copy()
        temp_positions[new_symbol] = new_weight

        # Quick cluster check (simplified)
        for cluster in self.cluster_history[-1] if self.cluster_history else []:
            if new_symbol in cluster.symbols:
                new_cluster_weight = cluster.total_weight + new_weight
                if new_cluster_weight > self.config.correlation_limits.max_correlated_weight:
                    warnings.append(
                        f"Would exceed cluster weight limit: {new_cluster_weight:.1%}"
                    )

        allowed = len(warnings) == 0
        return allowed, warnings

    def get_correlation_adjusted_position_size(
        self,
        base_size: float,
        symbol: str
    ) -> float:
        """
        Adjust position size based on correlation risk.

        Args:
            base_size: Base position size
            symbol: Symbol for the position

        Returns:
            Correlation-adjusted position size
        """
        if self.last_correlation_matrix is None:
            return base_size

        # Calculate correlation penalty
        penalty_factor = 1.0

        for existing_symbol, existing_weight in self.current_positions.items():
            if existing_symbol in self.last_correlation_matrix.index and \
               symbol in self.last_correlation_matrix.index:

                corr = abs(self.last_correlation_matrix.loc[existing_symbol, symbol])

                if corr > self.max_correlation_threshold:
                    # Apply penalty based on correlation strength and existing weight
                    penalty = (corr - self.max_correlation_threshold) * existing_weight * 2
                    penalty_factor *= (1 - penalty)

        # Ensure penalty doesn't make size negative
        penalty_factor = max(0.1, penalty_factor)

        adjusted_size = base_size * penalty_factor

        if adjusted_size < base_size * 0.9:
            logger.info(f"Correlation adjustment for {symbol}: "
                       f"{base_size:.1%} -> {adjusted_size:.1%}")

        return adjusted_size

    def _calculate_portfolio_correlation(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate portfolio-weighted average correlation"""
        total_correlation = 0.0
        total_weight_pairs = 0.0

        for symbol1 in correlation_matrix.index:
            for symbol2 in correlation_matrix.index:
                if symbol1 != symbol2:
                    weight1 = self.current_positions.get(symbol1, 0)
                    weight2 = self.current_positions.get(symbol2, 0)
                    corr = abs(correlation_matrix.loc[symbol1, symbol2])

                    pair_weight = weight1 * weight2
                    total_correlation += corr * pair_weight
                    total_weight_pairs += pair_weight

        return total_correlation / max(total_weight_pairs, 0.001)

    def _calculate_concentration_ratio(self, clusters: List[CorrelationCluster]) -> float:
        """Calculate correlation concentration ratio"""
        if not clusters:
            return 0.0

        # Herfindahl index for cluster weights
        total_weight = sum(cluster.total_weight for cluster in clusters)
        if total_weight == 0:
            return 0.0

        concentration = sum((cluster.total_weight / total_weight) ** 2 for cluster in clusters)
        return concentration

    def _calculate_correlation_adjusted_risk(
        self,
        correlation_matrix: pd.DataFrame,
        clusters: List[CorrelationCluster]
    ) -> float:
        """Calculate risk adjusted for correlations"""
        # Simplified risk calculation considering correlations
        base_risk = sum(abs(weight) for weight in self.current_positions.values())

        # Correlation adjustment factor
        avg_correlation = np.nanmean(np.abs(correlation_matrix.values))
        correlation_multiplier = 1 + avg_correlation

        return base_risk * correlation_multiplier

    def _calculate_diversification_benefit(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate diversification benefit (reduction in risk due to correlations < 1)"""
        if correlation_matrix.empty:
            return 0.0

        # Perfect correlation risk vs actual correlation risk
        n_assets = len(correlation_matrix)
        perfect_correlation_variance = 1.0  # Normalized

        # Portfolio variance with actual correlations (simplified)
        weights = np.array([self.current_positions.get(sym, 0)
                          for sym in correlation_matrix.index])

        if np.sum(weights) == 0:
            return 0.0

        weights = weights / np.sum(weights)  # Normalize

        # Assume equal volatilities for simplification
        portfolio_variance = np.dot(weights, np.dot(correlation_matrix.values, weights))

        diversification_benefit = (perfect_correlation_variance - portfolio_variance) / perfect_correlation_variance
        return max(0, diversification_benefit)

    def _generate_correlation_alerts(
        self,
        correlation_matrix: pd.DataFrame,
        clusters: List[CorrelationCluster]
    ) -> List[CorrelationAlert]:
        """Generate correlation-based alerts"""
        alerts = []

        # High correlation alerts
        for i, sym1 in enumerate(correlation_matrix.index):
            for j, sym2 in enumerate(correlation_matrix.index):
                if i < j:
                    corr = abs(correlation_matrix.loc[sym1, sym2])

                    if corr > self.max_correlation_threshold:
                        combined_weight = (self.current_positions.get(sym1, 0) +
                                         self.current_positions.get(sym2, 0))

                        severity = AlertSeverity.WARNING
                        if corr > 0.9 or combined_weight > 0.3:
                            severity = AlertSeverity.CRITICAL

                        alerts.append(CorrelationAlert(
                            alert_type="high_correlation",
                            severity=severity,
                            symbols=[sym1, sym2],
                            correlation_value=corr,
                            limit=self.max_correlation_threshold,
                            message=f"High correlation ({corr:.2f}) between {sym1} and {sym2}"
                        ))

        # Cluster concentration alerts
        for cluster in clusters:
            if cluster.warning_level in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]:
                alerts.append(CorrelationAlert(
                    alert_type="cluster_concentration",
                    severity=cluster.warning_level,
                    symbols=cluster.symbols,
                    correlation_value=cluster.avg_correlation,
                    limit=self.config.correlation_limits.max_correlated_weight,
                    message=f"Cluster {cluster.cluster_id} weight ({cluster.total_weight:.1%}) "
                           f"exceeds limit ({self.config.correlation_limits.max_correlated_weight:.1%})"
                ))

        return alerts

    def _find_high_correlation_pairs(
        self,
        correlation_matrix: pd.DataFrame
    ) -> List[Tuple[str, str, float]]:
        """Find pairs with correlation above threshold"""
        pairs = []

        for i, sym1 in enumerate(correlation_matrix.index):
            for j, sym2 in enumerate(correlation_matrix.index):
                if i < j:
                    corr = correlation_matrix.loc[sym1, sym2]
                    if abs(corr) > self.max_correlation_threshold:
                        pairs.append((sym1, sym2, corr))

        # Sort by correlation strength
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs

    def _generate_concentration_warnings(self, clusters: List[CorrelationCluster]) -> List[str]:
        """Generate concentration warnings"""
        warnings = []

        for cluster in clusters:
            if cluster.total_weight > self.config.correlation_limits.max_correlated_weight:
                warnings.append(
                    f"Cluster {cluster.cluster_id} ({', '.join(cluster.symbols[:3])}) "
                    f"exceeds weight limit: {cluster.total_weight:.1%}"
                )
            elif cluster.total_weight > self.config.correlation_limits.max_correlated_weight * 0.8:
                warnings.append(
                    f"Cluster {cluster.cluster_id} ({', '.join(cluster.symbols[:3])}) "
                    f"approaching weight limit: {cluster.total_weight:.1%}"
                )

        return warnings

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive correlation status report"""
        if not self.current_positions:
            return {
                'status': 'No positions to analyze',
                'timestamp': datetime.now().isoformat()
            }

        metrics = self.analyze_correlations()

        return {
            'timestamp': self.last_update.isoformat() if self.last_update else None,
            'positions_analyzed': len(self.current_positions),
            'correlation_metrics': {
                'avg_correlation': f"{metrics.avg_portfolio_correlation:.3f}",
                'max_correlation': f"{metrics.max_pairwise_correlation:.3f}",
                'concentration_ratio': f"{metrics.correlation_concentration_ratio:.3f}",
                'diversification_benefit': f"{metrics.diversification_benefit:.1%}"
            },
            'clusters': [
                {
                    'id': cluster.cluster_id,
                    'symbols': cluster.symbols,
                    'avg_correlation': f"{cluster.avg_correlation:.3f}",
                    'total_weight': f"{cluster.total_weight:.1%}",
                    'warning_level': cluster.warning_level.value
                }
                for cluster in metrics.clusters
            ],
            'high_correlation_pairs': [
                {
                    'symbols': [pair[0], pair[1]],
                    'correlation': f"{pair[2]:.3f}"
                }
                for pair in metrics.high_correlation_pairs[:5]  # Top 5
            ],
            'alerts': [
                {
                    'type': alert.alert_type,
                    'severity': alert.severity.value,
                    'symbols': alert.symbols,
                    'correlation': f"{alert.correlation_value:.3f}",
                    'message': alert.message
                }
                for alert in metrics.alerts
            ],
            'warnings': metrics.concentration_warnings
        }


if __name__ == "__main__":
    from ..config.risk_config import load_risk_config

    # Test the correlation monitor
    config = load_risk_config()
    monitor = CorrelationMonitor(config)

    # Set up test positions
    positions = {
        'SPY': 0.3,
        'QQQ': 0.2,
        'AAPL': 0.15,
        'MSFT': 0.1,
        'GOOGL': 0.1
    }

    monitor.update_positions(positions)

    # Analyze correlations
    metrics = monitor.analyze_correlations()

    print("Correlation Analysis Results:")
    print(f"Average Correlation: {metrics.avg_portfolio_correlation:.3f}")
    print(f"Max Correlation: {metrics.max_pairwise_correlation:.3f}")
    print(f"Number of Clusters: {len(metrics.clusters)}")
    print(f"High Correlation Pairs: {len(metrics.high_correlation_pairs)}")
    print(f"Alerts: {len(metrics.alerts)}")

    for alert in metrics.alerts:
        print(f"  {alert.severity.value.upper()}: {alert.message}")