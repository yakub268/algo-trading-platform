"""
Data Quality Scorer
==================

Comprehensive data quality assessment for alternative data sources.
Provides confidence intervals and quality metrics.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics

from .base_connector import DataPoint, DataSource


class QualityDimension(Enum):
    """Different dimensions of data quality"""
    TIMELINESS = "timeliness"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    RELEVANCE = "relevance"
    RELIABILITY = "reliability"


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for a data point"""
    timeliness_score: float = 0.0
    accuracy_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    relevance_score: float = 0.0
    reliability_score: float = 0.0

    overall_score: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    quality_grade: str = "F"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timeliness': self.timeliness_score,
            'accuracy': self.accuracy_score,
            'completeness': self.completeness_score,
            'consistency': self.consistency_score,
            'relevance': self.relevance_score,
            'reliability': self.reliability_score,
            'overall': self.overall_score,
            'confidence_interval': self.confidence_interval,
            'grade': self.quality_grade
        }


class DataQualityScorer:
    """
    Advanced data quality assessment system

    Features:
    - Multi-dimensional quality scoring
    - Source-specific quality rules
    - Historical quality tracking
    - Confidence intervals
    - Quality grading system
    """

    def __init__(self):
        self.logger = logging.getLogger("altdata.quality")
        self.historical_scores: Dict[DataSource, List[float]] = {}
        self.source_weights = self._initialize_source_weights()
        self.dimension_weights = {
            QualityDimension.TIMELINESS: 0.25,
            QualityDimension.ACCURACY: 0.20,
            QualityDimension.COMPLETENESS: 0.15,
            QualityDimension.CONSISTENCY: 0.15,
            QualityDimension.RELEVANCE: 0.15,
            QualityDimension.RELIABILITY: 0.10
        }

    def _initialize_source_weights(self) -> Dict[DataSource, Dict[QualityDimension, float]]:
        """Initialize quality dimension weights for each data source"""
        return {
            DataSource.TWITTER: {
                QualityDimension.TIMELINESS: 0.30,
                QualityDimension.RELEVANCE: 0.25,
                QualityDimension.COMPLETENESS: 0.15,
                QualityDimension.ACCURACY: 0.10,
                QualityDimension.CONSISTENCY: 0.10,
                QualityDimension.RELIABILITY: 0.10
            },
            DataSource.FRED: {
                QualityDimension.ACCURACY: 0.35,
                QualityDimension.RELIABILITY: 0.25,
                QualityDimension.TIMELINESS: 0.15,
                QualityDimension.COMPLETENESS: 0.15,
                QualityDimension.CONSISTENCY: 0.05,
                QualityDimension.RELEVANCE: 0.05
            },
            DataSource.OPTIONS_FLOW: {
                QualityDimension.TIMELINESS: 0.35,
                QualityDimension.ACCURACY: 0.25,
                QualityDimension.COMPLETENESS: 0.20,
                QualityDimension.CONSISTENCY: 0.10,
                QualityDimension.RELEVANCE: 0.05,
                QualityDimension.RELIABILITY: 0.05
            },
            # Add weights for other sources...
        }

    def score_data_point(self, data_point: DataPoint) -> QualityMetrics:
        """
        Score a single data point across all quality dimensions

        Args:
            data_point: DataPoint to score

        Returns:
            QualityMetrics with comprehensive scores
        """
        metrics = QualityMetrics()

        # Score each dimension
        metrics.timeliness_score = self._score_timeliness(data_point)
        metrics.accuracy_score = self._score_accuracy(data_point)
        metrics.completeness_score = self._score_completeness(data_point)
        metrics.consistency_score = self._score_consistency(data_point)
        metrics.relevance_score = self._score_relevance(data_point)
        metrics.reliability_score = self._score_reliability(data_point)

        # Calculate weighted overall score
        weights = self.source_weights.get(
            data_point.source,
            self.dimension_weights
        )

        scores = [
            metrics.timeliness_score * weights.get(QualityDimension.TIMELINESS, 0.25),
            metrics.accuracy_score * weights.get(QualityDimension.ACCURACY, 0.20),
            metrics.completeness_score * weights.get(QualityDimension.COMPLETENESS, 0.15),
            metrics.consistency_score * weights.get(QualityDimension.CONSISTENCY, 0.15),
            metrics.relevance_score * weights.get(QualityDimension.RELEVANCE, 0.15),
            metrics.reliability_score * weights.get(QualityDimension.RELIABILITY, 0.10)
        ]

        metrics.overall_score = sum(scores)

        # Calculate confidence interval
        metrics.confidence_interval = self._calculate_confidence_interval(
            data_point.source, metrics.overall_score
        )

        # Assign quality grade
        metrics.quality_grade = self._assign_grade(metrics.overall_score)

        # Store for historical tracking
        self._update_historical_scores(data_point.source, metrics.overall_score)

        return metrics

    def _score_timeliness(self, data_point: DataPoint) -> float:
        """Score data timeliness based on latency and freshness"""
        now = datetime.utcnow()
        data_age_minutes = (now - data_point.timestamp).total_seconds() / 60

        # Latency score (lower is better)
        latency_score = max(0, 1 - (data_point.latency_ms / 5000))  # 5s max

        # Freshness score (source-dependent acceptable delay)
        max_age_by_source = {
            DataSource.TWITTER: 15,      # 15 minutes max
            DataSource.OPTIONS_FLOW: 5,   # 5 minutes max
            DataSource.NEWS: 30,         # 30 minutes max
            DataSource.FRED: 1440,       # 24 hours max
            DataSource.SATELLITE: 10080  # 1 week max
        }

        max_age = max_age_by_source.get(data_point.source, 60)
        freshness_score = max(0, 1 - (data_age_minutes / max_age))

        return (latency_score + freshness_score) / 2

    def _score_accuracy(self, data_point: DataPoint) -> float:
        """Score data accuracy based on source reliability and validation"""
        # Source reliability baseline
        source_reliability = {
            DataSource.FRED: 0.95,
            DataSource.SEC_FILINGS: 0.90,
            DataSource.SATELLITE: 0.85,
            DataSource.OPTIONS_FLOW: 0.80,
            DataSource.NEWS: 0.75,
            DataSource.BLOCKCHAIN: 0.95,
            DataSource.WEATHER: 0.85,
            DataSource.TWITTER: 0.60,
            DataSource.REDDIT: 0.55,
            DataSource.DISCORD: 0.50
        }

        base_score = source_reliability.get(data_point.source, 0.5)

        # Data validation checks
        validation_score = self._run_data_validations(data_point)

        return (base_score + validation_score) / 2

    def _score_completeness(self, data_point: DataPoint) -> float:
        """Score data completeness based on required fields"""
        required_fields_by_source = {
            DataSource.TWITTER: ['text', 'user', 'created_at'],
            DataSource.OPTIONS_FLOW: ['strike', 'expiry', 'volume', 'type'],
            DataSource.NEWS: ['title', 'content', 'source', 'published_at'],
            DataSource.FRED: ['value', 'date', 'series_id']
        }

        required = required_fields_by_source.get(data_point.source, [])
        if not required:
            return 0.8  # Default score for unknown source

        present_fields = sum(
            1 for field in required
            if field in data_point.raw_data and data_point.raw_data[field] is not None
        )

        return present_fields / len(required)

    def _score_consistency(self, data_point: DataPoint) -> float:
        """Score data consistency with historical patterns"""
        # Get recent scores for this source
        recent_scores = self.historical_scores.get(data_point.source, [])

        if len(recent_scores) < 5:
            return 0.8  # Default for insufficient history

        # Calculate coefficient of variation
        mean_score = statistics.mean(recent_scores[-10:])
        std_score = statistics.stdev(recent_scores[-10:])

        if mean_score == 0:
            return 0.5

        cv = std_score / mean_score
        # Lower CV means higher consistency
        consistency_score = max(0, 1 - cv)

        return consistency_score

    def _score_relevance(self, data_point: DataPoint) -> float:
        """Score data relevance to trading decisions"""
        # Symbol-specific relevance
        symbol_relevance = 1.0 if data_point.symbol else 0.7

        # Asset class relevance to data source
        relevance_matrix = {
            DataSource.TWITTER: {'crypto': 0.9, 'stocks': 0.8, 'forex': 0.6},
            DataSource.OPTIONS_FLOW: {'stocks': 0.95, 'crypto': 0.8, 'forex': 0.3},
            DataSource.FRED: {'forex': 0.9, 'stocks': 0.8, 'crypto': 0.7},
            DataSource.SATELLITE: {'commodities': 0.95, 'stocks': 0.7, 'crypto': 0.3}
        }

        source_matrix = relevance_matrix.get(data_point.source, {})
        asset_relevance = source_matrix.get(data_point.asset_class, 0.5)

        return (symbol_relevance + asset_relevance) / 2

    def _score_reliability(self, data_point: DataPoint) -> float:
        """Score data source reliability based on historical performance"""
        source_uptime = self._calculate_source_uptime(data_point.source)
        historical_accuracy = self._get_historical_accuracy(data_point.source)

        return (source_uptime + historical_accuracy) / 2

    def _run_data_validations(self, data_point: DataPoint) -> float:
        """Run source-specific data validation checks"""
        validations_passed = 0
        total_validations = 0

        # Basic validations for all sources
        total_validations += 3
        if data_point.raw_data:
            validations_passed += 1
        if data_point.timestamp:
            validations_passed += 1
        if data_point.symbol or data_point.asset_class:
            validations_passed += 1

        # Source-specific validations
        if data_point.source == DataSource.TWITTER:
            total_validations += 2
            text = data_point.raw_data.get('text', '')
            if len(text) > 10:  # Not empty or too short
                validations_passed += 1
            if not any(spam_word in text.lower() for spam_word in ['spam', 'bot', 'fake']):
                validations_passed += 1

        elif data_point.source == DataSource.OPTIONS_FLOW:
            total_validations += 3
            if isinstance(data_point.raw_data.get('volume'), (int, float)) and data_point.raw_data['volume'] > 0:
                validations_passed += 1
            if data_point.raw_data.get('strike') and data_point.raw_data.get('expiry'):
                validations_passed += 1
            if data_point.raw_data.get('type') in ['call', 'put']:
                validations_passed += 1

        return validations_passed / total_validations if total_validations > 0 else 0

    def _calculate_confidence_interval(
        self,
        source: DataSource,
        score: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval for quality score"""
        historical = self.historical_scores.get(source, [])

        if len(historical) < 5:
            # Wide interval for insufficient data
            margin = 0.3
            return (max(0, score - margin), min(1, score + margin))

        # Calculate standard error
        std_error = np.std(historical[-20:]) / np.sqrt(len(historical[-20:]))

        # 95% confidence interval
        margin = 1.96 * std_error

        return (
            max(0, score - margin),
            min(1, score + margin)
        )

    def _assign_grade(self, score: float) -> str:
        """Assign letter grade based on quality score"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.55:
            return "C"
        elif score >= 0.5:
            return "C-"
        else:
            return "F"

    def _calculate_source_uptime(self, source: DataSource) -> float:
        """Calculate source uptime/availability"""
        # This would integrate with monitoring system
        # For now, return reasonable defaults
        uptime_by_source = {
            DataSource.FRED: 0.99,
            DataSource.SEC_FILINGS: 0.95,
            DataSource.TWITTER: 0.98,
            DataSource.REDDIT: 0.97,
            DataSource.OPTIONS_FLOW: 0.93,
            DataSource.NEWS: 0.95,
            DataSource.BLOCKCHAIN: 0.99,
            DataSource.WEATHER: 0.96,
            DataSource.SATELLITE: 0.90
        }
        return uptime_by_source.get(source, 0.90)

    def _get_historical_accuracy(self, source: DataSource) -> float:
        """Get historical accuracy for source"""
        historical = self.historical_scores.get(source, [])
        if not historical:
            return 0.7  # Default

        return statistics.mean(historical[-50:]) if len(historical) >= 5 else 0.7

    def _update_historical_scores(self, source: DataSource, score: float) -> None:
        """Update historical score tracking"""
        if source not in self.historical_scores:
            self.historical_scores[source] = []

        self.historical_scores[source].append(score)

        # Keep only recent scores (last 1000)
        if len(self.historical_scores[source]) > 1000:
            self.historical_scores[source] = self.historical_scores[source][-1000:]

    def get_source_quality_summary(self, source: DataSource) -> Dict[str, Any]:
        """Get quality summary for a data source"""
        historical = self.historical_scores.get(source, [])

        if not historical:
            return {
                'source': source.value,
                'no_data': True
            }

        recent_scores = historical[-50:]

        return {
            'source': source.value,
            'total_points': len(historical),
            'recent_average': statistics.mean(recent_scores),
            'recent_median': statistics.median(recent_scores),
            'recent_std': statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0,
            'recent_min': min(recent_scores),
            'recent_max': max(recent_scores),
            'trend': self._calculate_trend(historical),
            'grade': self._assign_grade(statistics.mean(recent_scores))
        }

    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate quality trend"""
        if len(scores) < 10:
            return "insufficient_data"

        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg + 0.05:
            return "improving"
        elif second_avg < first_avg - 0.05:
            return "declining"
        else:
            return "stable"