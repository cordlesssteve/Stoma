"""Multi-paper correlation analysis for research intelligence."""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import statistics

from ..storage.database import DatabaseStorage
from .nlp_storage import NLPStorage

logger = logging.getLogger(__name__)


@dataclass
class PaperCorrelation:
    """Represents correlation between papers."""
    
    paper1_id: int
    paper2_id: int
    correlation_score: float
    correlation_type: str  # 'keyword', 'entity', 'topic', 'semantic'
    shared_elements: List[str]
    confidence: float
    metadata: Dict = None


@dataclass
class TopicCluster:
    """Represents a cluster of related papers by topic."""
    
    cluster_id: str
    primary_topic: str
    related_topics: List[str]
    paper_ids: List[int]
    coherence_score: float
    timespan_days: int
    central_keywords: List[str]
    summary: str = ""


@dataclass
class ResearchConnection:
    """Represents a connection between research areas."""
    
    connection_id: str
    area1: str
    area2: str
    strength: float
    bridge_papers: List[int]
    bridge_keywords: List[str]
    emergence_date: datetime
    connection_type: str  # 'interdisciplinary', 'convergence', 'divergence'


class CorrelationAnalyzer:
    """Analyzes correlations and connections between papers and topics."""
    
    def __init__(self, 
                 db_storage: Optional[DatabaseStorage] = None,
                 nlp_storage: Optional[NLPStorage] = None):
        """
        Initialize correlation analyzer.
        
        Args:
            db_storage: Database storage instance
            nlp_storage: NLP storage instance
        """
        self.db = db_storage or DatabaseStorage()
        self.nlp_storage = nlp_storage or NLPStorage(self.db)
        
        # Configuration
        self.min_correlation_score = 0.3
        self.min_cluster_size = 3
        self.max_cluster_size = 50
        
    def find_paper_correlations(self, 
                               paper_ids: Optional[List[int]] = None,
                               correlation_threshold: float = 0.3,
                               max_correlations: int = 100) -> List[PaperCorrelation]:
        """
        Find correlations between papers based on various similarity metrics.
        
        Args:
            paper_ids: Specific papers to analyze, or None for all
            correlation_threshold: Minimum correlation score
            max_correlations: Maximum correlations to return
            
        Returns:
            List of paper correlations
        """
        correlations = []
        
        try:
            # Get paper analysis data
            papers_data = self._get_papers_analysis_data(paper_ids)
            
            if len(papers_data) < 2:
                logger.warning("Need at least 2 papers for correlation analysis")
                return []
            
            # Calculate correlations between all paper pairs
            paper_list = list(papers_data.keys())
            
            for i in range(len(paper_list)):
                for j in range(i + 1, len(paper_list)):
                    paper1_id = paper_list[i]
                    paper2_id = paper_list[j]
                    
                    correlation = self._calculate_paper_correlation(
                        paper1_id, papers_data[paper1_id],
                        paper2_id, papers_data[paper2_id]
                    )
                    
                    if correlation and correlation.correlation_score >= correlation_threshold:
                        correlations.append(correlation)
            
            # Sort by correlation score and limit results
            correlations.sort(key=lambda x: x.correlation_score, reverse=True)
            correlations = correlations[:max_correlations]
            
            logger.info(f"Found {len(correlations)} paper correlations")
            return correlations
            
        except Exception as e:
            logger.error(f"Error finding paper correlations: {e}")
            return []
    
    def cluster_papers_by_topic(self, 
                               timeframe_days: int = 90,
                               min_cluster_size: int = 3) -> List[TopicCluster]:
        """
        Cluster papers by topic similarity.
        
        Args:
            timeframe_days: Time window for analysis
            min_cluster_size: Minimum papers per cluster
            
        Returns:
            List of topic clusters
        """
        clusters = []
        
        try:
            # Get recent papers with topics
            papers_topics = self._get_papers_topics(timeframe_days)
            
            if len(papers_topics) < min_cluster_size:
                logger.warning("Insufficient papers for clustering")
                return []
            
            # Group papers by primary topic
            topic_groups = defaultdict(list)
            
            for paper_id, topics in papers_topics.items():
                if topics:
                    primary_topic = topics[0]  # Use first topic as primary
                    topic_groups[primary_topic].append(paper_id)
            
            # Create clusters from groups
            for topic, paper_ids in topic_groups.items():
                if len(paper_ids) >= min_cluster_size:
                    cluster = self._create_topic_cluster(topic, paper_ids, timeframe_days)
                    if cluster:
                        clusters.append(cluster)
            
            # Sort by coherence score
            clusters.sort(key=lambda x: x.coherence_score, reverse=True)
            
            logger.info(f"Created {len(clusters)} topic clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering papers by topic: {e}")
            return []
    
    def find_research_connections(self, 
                                 timeframe_days: int = 180) -> List[ResearchConnection]:
        """
        Find connections between different research areas.
        
        Args:
            timeframe_days: Time window for analysis
            
        Returns:
            List of research connections
        """
        connections = []
        
        try:
            # Get domain classifications for papers
            domain_papers = self._classify_papers_by_domain(timeframe_days)
            
            # Find bridge papers (papers that span domains)
            bridge_papers = self._find_bridge_papers(domain_papers)
            
            # Analyze connections between domains
            for (domain1, domain2), papers in bridge_papers.items():
                if len(papers) >= 2:  # Need at least 2 bridge papers
                    connection = self._analyze_domain_connection(
                        domain1, domain2, papers, timeframe_days
                    )
                    if connection:
                        connections.append(connection)
            
            # Sort by connection strength
            connections.sort(key=lambda x: x.strength, reverse=True)
            
            logger.info(f"Found {len(connections)} research connections")
            return connections
            
        except Exception as e:
            logger.error(f"Error finding research connections: {e}")
            return []
    
    def analyze_citation_network(self, paper_ids: List[int]) -> Dict:
        """
        Analyze citation patterns and network structure.
        
        Args:
            paper_ids: Papers to analyze
            
        Returns:
            Citation network analysis results
        """
        # This would implement citation network analysis
        # For now, return basic structure
        return {
            'nodes': len(paper_ids),
            'edges': 0,
            'clusters': [],
            'centrality_scores': {},
            'influential_papers': []
        }
    
    def _get_papers_analysis_data(self, paper_ids: Optional[List[int]]) -> Dict:
        """Get analysis data for papers."""
        if paper_ids:
            paper_filter = "AND a.paper_id = ANY(%s)"
            params = [paper_ids]
        else:
            paper_filter = ""
            params = []
        
        query = f"""
        SELECT 
            a.paper_id,
            a.keywords,
            a.entities,
            a.topics,
            a.sentiment,
            p.title,
            p.abstract
        FROM nlp_analysis a
        LEFT JOIN papers p ON a.paper_id = p.id
        WHERE a.paper_id IS NOT NULL
        {paper_filter}
        ORDER BY a.created_at DESC
        """
        
        papers_data = {}
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
                    
                    for row in results:
                        paper_id = row[0]
                        papers_data[paper_id] = {
                            'keywords': row[1] or [],
                            'entities': row[2] or {},
                            'topics': row[3] or [],
                            'sentiment': row[4] or {},
                            'title': row[5] or '',
                            'abstract': row[6] or ''
                        }
                        
        except Exception as e:
            logger.error(f"Error getting papers analysis data: {e}")
        
        return papers_data
    
    def _calculate_paper_correlation(self, 
                                   paper1_id: int, paper1_data: Dict,
                                   paper2_id: int, paper2_data: Dict) -> Optional[PaperCorrelation]:
        """Calculate correlation between two papers."""
        
        # Calculate different types of similarity
        keyword_sim = self._calculate_keyword_similarity(
            paper1_data['keywords'], paper2_data['keywords']
        )
        
        entity_sim = self._calculate_entity_similarity(
            paper1_data['entities'], paper2_data['entities']
        )
        
        topic_sim = self._calculate_topic_similarity(
            paper1_data['topics'], paper2_data['topics']
        )
        
        # Combine similarities with weights
        overall_score = (0.4 * keyword_sim + 0.3 * entity_sim + 0.3 * topic_sim)
        
        if overall_score < self.min_correlation_score:
            return None
        
        # Determine primary correlation type
        scores = {'keyword': keyword_sim, 'entity': entity_sim, 'topic': topic_sim}
        correlation_type = max(scores, key=scores.get)
        
        # Get shared elements
        shared_elements = self._get_shared_elements(
            paper1_data, paper2_data, correlation_type
        )
        
        # Calculate confidence based on amount of shared data
        confidence = min(1.0, len(shared_elements) / 5.0)
        
        return PaperCorrelation(
            paper1_id=paper1_id,
            paper2_id=paper2_id,
            correlation_score=overall_score,
            correlation_type=correlation_type,
            shared_elements=shared_elements,
            confidence=confidence,
            metadata={
                'keyword_similarity': keyword_sim,
                'entity_similarity': entity_sim,
                'topic_similarity': topic_sim
            }
        )
    
    def _calculate_keyword_similarity(self, keywords1: List, keywords2: List) -> float:
        """Calculate keyword similarity using Jaccard similarity."""
        if not keywords1 or not keywords2:
            return 0.0
        
        # Extract keyword strings
        kw1 = set()
        kw2 = set()
        
        for kw in keywords1:
            if isinstance(kw, dict) and 'keyword' in kw:
                kw1.add(kw['keyword'].lower())
            elif isinstance(kw, str):
                kw1.add(kw.lower())
        
        for kw in keywords2:
            if isinstance(kw, dict) and 'keyword' in kw:
                kw2.add(kw['keyword'].lower())
            elif isinstance(kw, str):
                kw2.add(kw.lower())
        
        if not kw1 or not kw2:
            return 0.0
        
        intersection = len(kw1.intersection(kw2))
        union = len(kw1.union(kw2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_entity_similarity(self, entities1: Dict, entities2: Dict) -> float:
        """Calculate entity similarity."""
        if not entities1 or not entities2:
            return 0.0
        
        # Flatten entity lists
        ent1 = set()
        ent2 = set()
        
        for entity_type, entity_list in entities1.items():
            if isinstance(entity_list, list):
                for entity in entity_list:
                    ent1.add(entity.lower())
        
        for entity_type, entity_list in entities2.items():
            if isinstance(entity_list, list):
                for entity in entity_list:
                    ent2.add(entity.lower())
        
        if not ent1 or not ent2:
            return 0.0
        
        intersection = len(ent1.intersection(ent2))
        union = len(ent1.union(ent2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_topic_similarity(self, topics1: List, topics2: List) -> float:
        """Calculate topic similarity."""
        if not topics1 or not topics2:
            return 0.0
        
        t1 = set(t.lower() for t in topics1 if isinstance(t, str))
        t2 = set(t.lower() for t in topics2 if isinstance(t, str))
        
        if not t1 or not t2:
            return 0.0
        
        intersection = len(t1.intersection(t2))
        union = len(t1.union(t2))
        
        return intersection / union if union > 0 else 0.0
    
    def _get_shared_elements(self, data1: Dict, data2: Dict, element_type: str) -> List[str]:
        """Get shared elements of specified type."""
        shared = []
        
        if element_type == 'keyword':
            kw1 = self._extract_keywords_list(data1['keywords'])
            kw2 = self._extract_keywords_list(data2['keywords'])
            shared = list(set(kw1).intersection(set(kw2)))
            
        elif element_type == 'entity':
            ent1 = self._extract_entities_list(data1['entities'])
            ent2 = self._extract_entities_list(data2['entities'])
            shared = list(set(ent1).intersection(set(ent2)))
            
        elif element_type == 'topic':
            t1 = [t.lower() for t in data1['topics'] if isinstance(t, str)]
            t2 = [t.lower() for t in data2['topics'] if isinstance(t, str)]
            shared = list(set(t1).intersection(set(t2)))
        
        return shared[:10]  # Limit to top 10
    
    def _extract_keywords_list(self, keywords: List) -> List[str]:
        """Extract keyword strings from keyword data."""
        kw_list = []
        for kw in keywords:
            if isinstance(kw, dict) and 'keyword' in kw:
                kw_list.append(kw['keyword'].lower())
            elif isinstance(kw, str):
                kw_list.append(kw.lower())
        return kw_list
    
    def _extract_entities_list(self, entities: Dict) -> List[str]:
        """Extract entity strings from entity data."""
        ent_list = []
        for entity_type, entity_list in entities.items():
            if isinstance(entity_list, list):
                for entity in entity_list:
                    ent_list.append(entity.lower())
        return ent_list
    
    def _get_papers_topics(self, days: int) -> Dict[int, List[str]]:
        """Get papers and their topics."""
        query = """
        SELECT a.paper_id, a.topics
        FROM nlp_analysis a
        WHERE a.paper_id IS NOT NULL
        AND a.created_at >= %s
        AND a.topics IS NOT NULL
        """
        
        cutoff_date = datetime.now() - timedelta(days=days)
        papers_topics = {}
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (cutoff_date,))
                    results = cur.fetchall()
                    
                    for paper_id, topics in results:
                        if topics and isinstance(topics, list):
                            papers_topics[paper_id] = topics
                            
        except Exception as e:
            logger.error(f"Error getting papers topics: {e}")
        
        return papers_topics
    
    def _create_topic_cluster(self, topic: str, paper_ids: List[int], timeframe_days: int) -> Optional[TopicCluster]:
        """Create a topic cluster from papers."""
        if len(paper_ids) < self.min_cluster_size:
            return None
        
        # Get central keywords for this cluster
        central_keywords = self._get_cluster_keywords(paper_ids)
        
        # Calculate coherence score
        coherence_score = self._calculate_cluster_coherence(paper_ids)
        
        # Get related topics
        related_topics = self._get_related_topics(paper_ids, topic)
        
        cluster_id = f"cluster_{topic.replace(' ', '_')}_{len(paper_ids)}"
        
        return TopicCluster(
            cluster_id=cluster_id,
            primary_topic=topic,
            related_topics=related_topics,
            paper_ids=paper_ids,
            coherence_score=coherence_score,
            timespan_days=timeframe_days,
            central_keywords=central_keywords,
            summary=f"Cluster of {len(paper_ids)} papers focused on {topic}"
        )
    
    def _get_cluster_keywords(self, paper_ids: List[int]) -> List[str]:
        """Get central keywords for a cluster of papers."""
        query = """
        SELECT k.keyword, COUNT(*) as frequency
        FROM extracted_keywords k
        JOIN nlp_analysis a ON k.analysis_id = a.id
        WHERE a.paper_id = ANY(%s)
        GROUP BY k.keyword
        ORDER BY frequency DESC
        LIMIT 10
        """
        
        keywords = []
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (paper_ids,))
                    results = cur.fetchall()
                    keywords = [r[0] for r in results]
                    
        except Exception as e:
            logger.error(f"Error getting cluster keywords: {e}")
        
        return keywords
    
    def _calculate_cluster_coherence(self, paper_ids: List[int]) -> float:
        """Calculate how coherent a cluster is."""
        # Simple coherence based on shared keywords
        # More sophisticated methods could use topic modeling
        
        if len(paper_ids) < 2:
            return 1.0
        
        # Get all keyword sets for papers in cluster
        keyword_sets = []
        
        for paper_id in paper_ids:
            keywords = self._get_paper_keywords(paper_id)
            keyword_sets.append(set(keywords))
        
        if not keyword_sets:
            return 0.0
        
        # Calculate average pairwise Jaccard similarity
        similarities = []
        
        for i in range(len(keyword_sets)):
            for j in range(i + 1, len(keyword_sets)):
                intersection = len(keyword_sets[i].intersection(keyword_sets[j]))
                union = len(keyword_sets[i].union(keyword_sets[j]))
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        return statistics.mean(similarities) if similarities else 0.0
    
    def _get_paper_keywords(self, paper_id: int) -> List[str]:
        """Get keywords for a specific paper."""
        query = """
        SELECT k.keyword
        FROM extracted_keywords k
        JOIN nlp_analysis a ON k.analysis_id = a.id
        WHERE a.paper_id = %s
        """
        
        keywords = []
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (paper_id,))
                    results = cur.fetchall()
                    keywords = [r[0] for r in results]
                    
        except Exception as e:
            logger.error(f"Error getting paper keywords: {e}")
        
        return keywords
    
    def _get_related_topics(self, paper_ids: List[int], primary_topic: str) -> List[str]:
        """Get topics related to the primary topic in this cluster."""
        query = """
        SELECT DISTINCT topic
        FROM (
            SELECT jsonb_array_elements_text(a.topics) as topic
            FROM nlp_analysis a
            WHERE a.paper_id = ANY(%s)
            AND a.topics IS NOT NULL
        ) t
        WHERE topic != %s
        ORDER BY topic
        LIMIT 5
        """
        
        related_topics = []
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (paper_ids, primary_topic))
                    results = cur.fetchall()
                    related_topics = [r[0] for r in results]
                    
        except Exception as e:
            logger.error(f"Error getting related topics: {e}")
        
        return related_topics
    
    def _classify_papers_by_domain(self, days: int) -> Dict[str, List[int]]:
        """Classify papers into research domains."""
        # This would implement domain classification
        # For now, return empty structure
        return {}
    
    def _find_bridge_papers(self, domain_papers: Dict) -> Dict[Tuple[str, str], List[int]]:
        """Find papers that bridge multiple domains."""
        # This would implement bridge paper detection
        # For now, return empty structure
        return {}
    
    def _analyze_domain_connection(self, domain1: str, domain2: str, 
                                 papers: List[int], timeframe_days: int) -> Optional[ResearchConnection]:
        """Analyze connection between two domains."""
        # This would implement domain connection analysis
        # For now, return None
        return None