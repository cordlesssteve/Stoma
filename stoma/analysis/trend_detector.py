"""Cross-domain trend detection algorithms for research intelligence."""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
import statistics

from ..storage.database import DatabaseStorage
from .nlp_storage import NLPStorage

logger = logging.getLogger(__name__)


@dataclass
class TrendSignal:
    """Represents a detected trend signal."""
    
    keyword: str
    trend_type: str  # 'emerging', 'declining', 'stable', 'volatile'
    strength: float  # 0.0 to 1.0
    velocity: float  # rate of change
    timeframe: str  # '7d', '30d', '90d'
    domains: List[str]  # domains where trend is detected
    supporting_papers: List[int]  # paper IDs supporting the trend
    first_seen: datetime
    peak_date: Optional[datetime] = None
    metadata: Dict = None


@dataclass
class CrossDomainTrend:
    """Represents a trend spanning multiple domains."""
    
    trend_id: str
    primary_keywords: List[str]
    domains: Dict[str, float]  # domain -> strength
    correlation_score: float
    emergence_date: datetime
    papers_count: int
    trend_narrative: str
    related_trends: List[str] = None


class TrendDetector:
    """Detects emerging trends and patterns across domains."""
    
    def __init__(self, 
                 db_storage: Optional[DatabaseStorage] = None,
                 nlp_storage: Optional[NLPStorage] = None):
        """
        Initialize trend detector.
        
        Args:
            db_storage: Database storage instance
            nlp_storage: NLP storage instance
        """
        self.db = db_storage or DatabaseStorage()
        self.nlp_storage = nlp_storage or NLPStorage(self.db)
        
        # Configuration
        self.min_papers_for_trend = 5
        self.min_domains_for_cross_trend = 2
        self.trend_window_days = [7, 30, 90]
        self.velocity_threshold = 0.1
        
    def detect_keyword_trends(self, 
                            timeframe_days: int = 30,
                            min_frequency: int = 3) -> List[TrendSignal]:
        """
        Detect trending keywords over time.
        
        Args:
            timeframe_days: Time window for trend analysis
            min_frequency: Minimum keyword frequency to consider
            
        Returns:
            List of trend signals
        """
        trends = []
        
        try:
            # Get keyword frequency over time
            keyword_timeline = self._get_keyword_timeline(timeframe_days)
            
            for keyword, timeline in keyword_timeline.items():
                if len(timeline) < min_frequency:
                    continue
                
                # Calculate trend metrics
                trend_signal = self._analyze_keyword_trend(keyword, timeline, timeframe_days)
                
                if trend_signal and trend_signal.strength > 0.3:
                    trends.append(trend_signal)
            
            # Sort by trend strength
            trends.sort(key=lambda x: x.strength, reverse=True)
            
            logger.info(f"Detected {len(trends)} keyword trends")
            return trends
            
        except Exception as e:
            logger.error(f"Error detecting keyword trends: {e}")
            return []
    
    def detect_cross_domain_trends(self, 
                                  timeframe_days: int = 60) -> List[CrossDomainTrend]:
        """
        Detect trends that span multiple domains.
        
        Args:
            timeframe_days: Time window for analysis
            
        Returns:
            List of cross-domain trends
        """
        cross_trends = []
        
        try:
            # Get domain-specific keyword trends
            domain_trends = self._get_domain_keyword_trends(timeframe_days)
            
            # Find correlations across domains
            correlations = self._find_cross_domain_correlations(domain_trends)
            
            for correlation in correlations:
                if correlation.correlation_score > 0.6:
                    cross_trends.append(correlation)
            
            logger.info(f"Detected {len(cross_trends)} cross-domain trends")
            return cross_trends
            
        except Exception as e:
            logger.error(f"Error detecting cross-domain trends: {e}")
            return []
    
    def detect_emerging_topics(self, 
                              lookback_days: int = 90,
                              emergence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect emerging topics using topic evolution analysis.
        
        Args:
            lookback_days: How far back to look for baseline
            emergence_threshold: Minimum emergence score
            
        Returns:
            List of emerging topics with metadata
        """
        emerging_topics = []
        
        try:
            # Get topic frequency over time
            topic_timeline = self._get_topic_timeline(lookback_days)
            
            # Analyze each topic for emergence patterns
            for topic, timeline in topic_timeline.items():
                emergence_score = self._calculate_emergence_score(timeline)
                
                if emergence_score > emergence_threshold:
                    emerging_topics.append({
                        'topic': topic,
                        'emergence_score': emergence_score,
                        'first_significant_date': self._find_emergence_date(timeline),
                        'recent_frequency': timeline[-7:] if len(timeline) >= 7 else timeline,
                        'papers_count': sum(timeline),
                        'growth_rate': self._calculate_growth_rate(timeline)
                    })
            
            # Sort by emergence score
            emerging_topics.sort(key=lambda x: x['emergence_score'], reverse=True)
            
            logger.info(f"Detected {len(emerging_topics)} emerging topics")
            return emerging_topics
            
        except Exception as e:
            logger.error(f"Error detecting emerging topics: {e}")
            return []
    
    def _get_keyword_timeline(self, days: int) -> Dict[str, List[Tuple[datetime, int]]]:
        """Get keyword frequency timeline."""
        query = """
        SELECT 
            k.keyword,
            DATE(a.created_at) as date,
            COUNT(*) as frequency
        FROM extracted_keywords k
        JOIN nlp_analysis a ON k.analysis_id = a.id
        WHERE a.created_at >= %s
        GROUP BY k.keyword, DATE(a.created_at)
        ORDER BY k.keyword, DATE(a.created_at)
        """
        
        cutoff_date = datetime.now() - timedelta(days=days)
        timeline = defaultdict(list)
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (cutoff_date,))
                    results = cur.fetchall()
                    
                    for keyword, date, frequency in results:
                        timeline[keyword].append((date, frequency))
            
            return dict(timeline)
            
        except Exception as e:
            logger.error(f"Error getting keyword timeline: {e}")
            return {}
    
    def _analyze_keyword_trend(self, 
                              keyword: str, 
                              timeline: List[Tuple[datetime, int]], 
                              timeframe_days: int) -> Optional[TrendSignal]:
        """Analyze trend for a specific keyword."""
        if len(timeline) < 3:
            return None
        
        # Extract frequencies and dates
        dates = [t[0] for t in timeline]
        frequencies = [t[1] for t in timeline]
        
        # Calculate trend metrics
        velocity = self._calculate_velocity(frequencies)
        strength = self._calculate_trend_strength(frequencies)
        trend_type = self._classify_trend_type(frequencies, velocity)
        
        # Get supporting information
        domains = self._get_keyword_domains(keyword, timeframe_days)
        supporting_papers = self._get_keyword_papers(keyword, timeframe_days)
        
        return TrendSignal(
            keyword=keyword,
            trend_type=trend_type,
            strength=strength,
            velocity=velocity,
            timeframe=f"{timeframe_days}d",
            domains=domains,
            supporting_papers=supporting_papers,
            first_seen=min(dates),
            peak_date=dates[frequencies.index(max(frequencies))] if frequencies else None,
            metadata={
                'total_frequency': sum(frequencies),
                'max_daily_frequency': max(frequencies),
                'days_active': len(timeline)
            }
        )
    
    def _calculate_velocity(self, frequencies: List[int]) -> float:
        """Calculate trend velocity (rate of change)."""
        if len(frequencies) < 2:
            return 0.0
        
        # Use linear regression slope as velocity measure
        n = len(frequencies)
        x_values = list(range(n))
        
        # Calculate slope
        sum_x = sum(x_values)
        sum_y = sum(frequencies)
        sum_xy = sum(x * y for x, y in zip(x_values, frequencies))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Normalize by mean frequency
        mean_freq = statistics.mean(frequencies) if frequencies else 1
        return slope / max(mean_freq, 1)
    
    def _calculate_trend_strength(self, frequencies: List[int]) -> float:
        """Calculate overall trend strength."""
        if len(frequencies) < 2:
            return 0.0
        
        # Combine multiple factors
        velocity = abs(self._calculate_velocity(frequencies))
        consistency = 1.0 - (statistics.stdev(frequencies) / max(statistics.mean(frequencies), 1))
        magnitude = min(1.0, max(frequencies) / 10)  # Normalize to 0-1
        
        # Weighted combination
        strength = 0.4 * velocity + 0.3 * consistency + 0.3 * magnitude
        return min(1.0, max(0.0, strength))
    
    def _classify_trend_type(self, frequencies: List[int], velocity: float) -> str:
        """Classify the type of trend."""
        if abs(velocity) < self.velocity_threshold:
            return 'stable'
        elif velocity > self.velocity_threshold:
            return 'emerging'
        elif velocity < -self.velocity_threshold:
            return 'declining'
        else:
            # Check for volatility
            if len(frequencies) > 3:
                variance = statistics.variance(frequencies)
                mean_freq = statistics.mean(frequencies)
                if variance > mean_freq * 2:
                    return 'volatile'
            return 'stable'
    
    def _get_keyword_domains(self, keyword: str, days: int) -> List[str]:
        """Get domains where keyword appears."""
        query = """
        SELECT DISTINCT
            CASE 
                WHEN p.title ILIKE '%quantum%' OR p.abstract ILIKE '%quantum%' THEN 'quantum_computing'
                WHEN p.title ILIKE '%machine learning%' OR p.title ILIKE '%AI%' THEN 'artificial_intelligence'
                WHEN p.title ILIKE '%blockchain%' OR p.title ILIKE '%crypto%' THEN 'blockchain'
                WHEN p.title ILIKE '%climate%' OR p.title ILIKE '%environment%' THEN 'climate_science'
                WHEN p.title ILIKE '%bio%' OR p.title ILIKE '%medical%' THEN 'biotechnology'
                ELSE 'general'
            END as domain
        FROM extracted_keywords k
        JOIN nlp_analysis a ON k.analysis_id = a.id
        LEFT JOIN papers p ON a.paper_id = p.id
        WHERE k.keyword = %s
        AND a.created_at >= %s
        """
        
        cutoff_date = datetime.now() - timedelta(days=days)
        domains = []
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (keyword, cutoff_date))
                    results = cur.fetchall()
                    domains = [r[0] for r in results if r[0] != 'general']
                    
        except Exception as e:
            logger.error(f"Error getting keyword domains: {e}")
        
        return domains or ['general']
    
    def _get_keyword_papers(self, keyword: str, days: int) -> List[int]:
        """Get papers containing the keyword."""
        query = """
        SELECT DISTINCT a.paper_id
        FROM extracted_keywords k
        JOIN nlp_analysis a ON k.analysis_id = a.id
        WHERE k.keyword = %s
        AND a.created_at >= %s
        AND a.paper_id IS NOT NULL
        """
        
        cutoff_date = datetime.now() - timedelta(days=days)
        papers = []
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (keyword, cutoff_date))
                    results = cur.fetchall()
                    papers = [r[0] for r in results]
                    
        except Exception as e:
            logger.error(f"Error getting keyword papers: {e}")
        
        return papers
    
    def _get_domain_keyword_trends(self, days: int) -> Dict[str, Dict[str, List]]:
        """Get keyword trends per domain."""
        # This would implement domain-specific trend analysis
        # For now, return a simplified structure
        return {}
    
    def _find_cross_domain_correlations(self, domain_trends: Dict) -> List[CrossDomainTrend]:
        """Find correlations between domain trends."""
        # This would implement cross-correlation analysis
        # For now, return empty list
        return []
    
    def _get_topic_timeline(self, days: int) -> Dict[str, List[int]]:
        """Get topic frequency timeline."""
        query = """
        SELECT 
            topic,
            DATE(a.created_at) as date,
            COUNT(*) as frequency
        FROM (
            SELECT 
                a.id,
                a.created_at,
                jsonb_array_elements_text(a.topics) as topic
            FROM nlp_analysis a
            WHERE a.created_at >= %s
            AND a.topics IS NOT NULL
        ) topics_expanded
        JOIN nlp_analysis a ON topics_expanded.id = a.id
        GROUP BY topic, DATE(a.created_at)
        ORDER BY topic, DATE(a.created_at)
        """
        
        cutoff_date = datetime.now() - timedelta(days=days)
        timeline = defaultdict(list)
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (cutoff_date,))
                    results = cur.fetchall()
                    
                    # Convert to daily frequency arrays
                    topic_dates = defaultdict(dict)
                    for topic, date, frequency in results:
                        topic_dates[topic][date] = frequency
                    
                    # Fill in missing dates with 0
                    start_date = cutoff_date.date()
                    end_date = datetime.now().date()
                    date_range = []
                    current = start_date
                    while current <= end_date:
                        date_range.append(current)
                        current += timedelta(days=1)
                    
                    for topic, date_freq in topic_dates.items():
                        freq_array = []
                        for date in date_range:
                            freq_array.append(date_freq.get(date, 0))
                        timeline[topic] = freq_array
            
            return dict(timeline)
            
        except Exception as e:
            logger.error(f"Error getting topic timeline: {e}")
            return {}
    
    def _calculate_emergence_score(self, timeline: List[int]) -> float:
        """Calculate how much a topic is emerging."""
        if len(timeline) < 14:  # Need at least 2 weeks of data
            return 0.0
        
        # Split into baseline (first half) and recent (second half)
        mid_point = len(timeline) // 2
        baseline = timeline[:mid_point]
        recent = timeline[mid_point:]
        
        baseline_avg = statistics.mean(baseline) if baseline else 0
        recent_avg = statistics.mean(recent) if recent else 0
        
        if baseline_avg == 0:
            return 1.0 if recent_avg > 0 else 0.0
        
        # Calculate emergence as ratio with smoothing
        emergence_ratio = recent_avg / baseline_avg
        
        # Apply sigmoid transformation to normalize
        emergence_score = 1 / (1 + math.exp(-2 * (emergence_ratio - 1)))
        
        return emergence_score
    
    def _find_emergence_date(self, timeline: List[int]) -> Optional[datetime]:
        """Find when a topic first started emerging significantly."""
        if len(timeline) < 7:
            return None
        
        # Look for the first sustained increase
        baseline = statistics.mean(timeline[:len(timeline)//3])
        
        for i in range(len(timeline)//3, len(timeline)):
            if timeline[i] > baseline * 1.5:  # 50% increase threshold
                emergence_date = datetime.now() - timedelta(days=len(timeline)-i)
                return emergence_date
        
        return None
    
    def _calculate_growth_rate(self, timeline: List[int]) -> float:
        """Calculate average growth rate over timeline."""
        if len(timeline) < 2:
            return 0.0
        
        growth_rates = []
        for i in range(1, len(timeline)):
            if timeline[i-1] > 0:
                rate = (timeline[i] - timeline[i-1]) / timeline[i-1]
                growth_rates.append(rate)
        
        return statistics.mean(growth_rates) if growth_rates else 0.0