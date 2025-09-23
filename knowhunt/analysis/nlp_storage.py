"""Database storage for NLP analysis results."""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor

from ..storage.database import DatabaseStorage
from .nlp_analyzer import AnalysisResult

logger = logging.getLogger(__name__)


class NLPStorage:
    """Storage handler for NLP analysis results."""
    
    def __init__(self, db_storage: Optional[DatabaseStorage] = None):
        """
        Initialize NLP storage.
        
        Args:
            db_storage: Database storage instance
        """
        self.db = db_storage or DatabaseStorage()
        # Only ensure tables if database connection works
        try:
            self._ensure_tables()
        except Exception as e:
            logger.warning(f"Database not available during initialization: {e}")
    
    def _ensure_tables(self):
        """Create NLP analysis tables if they don't exist."""
        create_analysis_table = """
        CREATE TABLE IF NOT EXISTS nlp_analysis (
            id SERIAL PRIMARY KEY,
            document_id VARCHAR(500) NOT NULL,
            paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
            analysis_type VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Core analysis results
            summary TEXT,
            keywords JSONB,
            entities JSONB,
            sentiment JSONB,
            topics JSONB,
            
            -- Statistics
            word_count INTEGER,
            sentence_count INTEGER,
            readability_score FLOAT,
            
            -- Metadata
            metadata JSONB,
            processing_time_ms INTEGER,
            analyzer_version VARCHAR(20) DEFAULT '1.0.0',
            
            -- Indexing
            UNIQUE(document_id, analysis_type),
            INDEX idx_nlp_paper_id (paper_id),
            INDEX idx_nlp_created_at (created_at),
            INDEX idx_nlp_analysis_type (analysis_type)
        );
        """
        
        create_keywords_table = """
        CREATE TABLE IF NOT EXISTS extracted_keywords (
            id SERIAL PRIMARY KEY,
            analysis_id INTEGER REFERENCES nlp_analysis(id) ON DELETE CASCADE,
            keyword VARCHAR(200) NOT NULL,
            score FLOAT NOT NULL,
            frequency INTEGER DEFAULT 1,
            
            INDEX idx_keyword (keyword),
            INDEX idx_keyword_score (score DESC),
            UNIQUE(analysis_id, keyword)
        );
        """
        
        create_entities_table = """
        CREATE TABLE IF NOT EXISTS extracted_entities (
            id SERIAL PRIMARY KEY,
            analysis_id INTEGER REFERENCES nlp_analysis(id) ON DELETE CASCADE,
            entity_type VARCHAR(50) NOT NULL,
            entity_text VARCHAR(500) NOT NULL,
            confidence FLOAT DEFAULT 1.0,
            
            INDEX idx_entity_type (entity_type),
            INDEX idx_entity_text (entity_text),
            UNIQUE(analysis_id, entity_type, entity_text)
        );
        """
        
        # Create full-text search column
        alter_analysis_table = """
        ALTER TABLE nlp_analysis 
        ADD COLUMN IF NOT EXISTS search_vector tsvector 
        GENERATED ALWAYS AS (
            to_tsvector('english', 
                COALESCE(summary, '') || ' ' || 
                COALESCE(topics::text, '')
            )
        ) STORED;
        
        CREATE INDEX IF NOT EXISTS idx_nlp_search_vector 
        ON nlp_analysis USING GIN(search_vector);
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_analysis_table)
                    cur.execute(create_keywords_table)
                    cur.execute(create_entities_table)
                    cur.execute(alter_analysis_table)
                    conn.commit()
                    
            logger.info("NLP analysis tables created/verified")
            
        except Exception as e:
            logger.error(f"Error creating NLP tables: {e}")
            raise
    
    def store_analysis(self, 
                       result: AnalysisResult,
                       paper_id: Optional[int] = None,
                       processing_time_ms: int = 0) -> int:
        """
        Store NLP analysis results in database.
        
        Args:
            result: Analysis result object
            paper_id: Optional reference to papers table
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            Analysis record ID
        """
        insert_query = """
        INSERT INTO nlp_analysis (
            document_id, paper_id, analysis_type,
            summary, keywords, entities, sentiment, topics,
            word_count, sentence_count, readability_score,
            metadata, processing_time_ms
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (document_id, analysis_type) 
        DO UPDATE SET
            summary = EXCLUDED.summary,
            keywords = EXCLUDED.keywords,
            entities = EXCLUDED.entities,
            sentiment = EXCLUDED.sentiment,
            topics = EXCLUDED.topics,
            word_count = EXCLUDED.word_count,
            sentence_count = EXCLUDED.sentence_count,
            readability_score = EXCLUDED.readability_score,
            metadata = EXCLUDED.metadata,
            processing_time_ms = EXCLUDED.processing_time_ms,
            created_at = CURRENT_TIMESTAMP
        RETURNING id;
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Store main analysis
                    cur.execute(insert_query, (
                        result.document_id,
                        paper_id,
                        'comprehensive',  # analysis type
                        result.summary,
                        json.dumps([{"keyword": k, "score": s} for k, s in result.keywords]),
                        json.dumps(result.entities),
                        json.dumps(result.sentiment),
                        json.dumps(result.topics),
                        result.word_count,
                        result.sentence_count,
                        result.readability_score,
                        json.dumps(result.metadata),
                        processing_time_ms
                    ))
                    
                    analysis_id = cur.fetchone()[0]
                    
                    # Store keywords in normalized table
                    self._store_keywords(cur, analysis_id, result.keywords)
                    
                    # Store entities in normalized table
                    self._store_entities(cur, analysis_id, result.entities)
                    
                    conn.commit()
                    
            logger.info(f"Stored NLP analysis for document {result.document_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Error storing analysis: {e}")
            raise
    
    def _store_keywords(self, cursor, analysis_id: int, keywords: List[tuple]):
        """Store keywords in normalized table."""
        if not keywords:
            return
        
        delete_query = "DELETE FROM extracted_keywords WHERE analysis_id = %s"
        cursor.execute(delete_query, (analysis_id,))
        
        insert_query = """
        INSERT INTO extracted_keywords (analysis_id, keyword, score)
        VALUES (%s, %s, %s)
        ON CONFLICT (analysis_id, keyword) DO UPDATE SET score = EXCLUDED.score
        """
        
        for keyword, score in keywords:
            cursor.execute(insert_query, (analysis_id, keyword, score))
    
    def _store_entities(self, cursor, analysis_id: int, entities: Dict[str, List[str]]):
        """Store entities in normalized table."""
        if not entities:
            return
        
        delete_query = "DELETE FROM extracted_entities WHERE analysis_id = %s"
        cursor.execute(delete_query, (analysis_id,))
        
        insert_query = """
        INSERT INTO extracted_entities (analysis_id, entity_type, entity_text)
        VALUES (%s, %s, %s)
        ON CONFLICT (analysis_id, entity_type, entity_text) DO NOTHING
        """
        
        for entity_type, entity_list in entities.items():
            for entity_text in entity_list:
                cursor.execute(insert_query, (analysis_id, entity_type, entity_text))
    
    def get_analysis(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve analysis for a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Analysis data or None
        """
        query = """
        SELECT 
            a.*,
            p.title as paper_title,
            p.authors as paper_authors,
            array_agg(
                DISTINCT jsonb_build_object(
                    'keyword', k.keyword, 
                    'score', k.score
                )
            ) FILTER (WHERE k.keyword IS NOT NULL) as keyword_list,
            array_agg(
                DISTINCT jsonb_build_object(
                    'type', e.entity_type,
                    'text', e.entity_text
                )
            ) FILTER (WHERE e.entity_text IS NOT NULL) as entity_list
        FROM nlp_analysis a
        LEFT JOIN papers p ON a.paper_id = p.id
        LEFT JOIN extracted_keywords k ON k.analysis_id = a.id
        LEFT JOIN extracted_entities e ON e.analysis_id = a.id
        WHERE a.document_id = %s
        GROUP BY a.id, p.id
        ORDER BY a.created_at DESC
        LIMIT 1;
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (document_id,))
                    result = cur.fetchone()
                    
            return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Error retrieving analysis: {e}")
            return None
    
    def search_by_keywords(self, keywords: List[str], limit: int = 10) -> List[Dict]:
        """
        Search analyses by keywords.
        
        Args:
            keywords: Keywords to search for
            limit: Maximum results
            
        Returns:
            List of matching analyses
        """
        query = """
        SELECT DISTINCT
            a.*,
            p.title as paper_title,
            COUNT(DISTINCT k.keyword) as matching_keywords,
            AVG(k.score) as avg_keyword_score
        FROM nlp_analysis a
        LEFT JOIN papers p ON a.paper_id = p.id
        INNER JOIN extracted_keywords k ON k.analysis_id = a.id
        WHERE k.keyword = ANY(%s)
        GROUP BY a.id, p.id
        ORDER BY matching_keywords DESC, avg_keyword_score DESC
        LIMIT %s;
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (keywords, limit))
                    results = cur.fetchall()
                    
            return [dict(r) for r in results]
            
        except Exception as e:
            logger.error(f"Error searching by keywords: {e}")
            return []
    
    def search_by_entities(self, 
                          entity_type: str, 
                          entity_text: Optional[str] = None,
                          limit: int = 10) -> List[Dict]:
        """
        Search analyses by entities.
        
        Args:
            entity_type: Type of entity
            entity_text: Optional specific entity text
            limit: Maximum results
            
        Returns:
            List of matching analyses
        """
        if entity_text:
            query = """
            SELECT DISTINCT
                a.*,
                p.title as paper_title,
                e.entity_text,
                e.entity_type
            FROM nlp_analysis a
            LEFT JOIN papers p ON a.paper_id = p.id
            INNER JOIN extracted_entities e ON e.analysis_id = a.id
            WHERE e.entity_type = %s AND e.entity_text ILIKE %s
            ORDER BY a.created_at DESC
            LIMIT %s;
            """
            params = (entity_type, f"%{entity_text}%", limit)
        else:
            query = """
            SELECT DISTINCT
                a.*,
                p.title as paper_title,
                array_agg(DISTINCT e.entity_text) as entities
            FROM nlp_analysis a
            LEFT JOIN papers p ON a.paper_id = p.id
            INNER JOIN extracted_entities e ON e.analysis_id = a.id
            WHERE e.entity_type = %s
            GROUP BY a.id, p.id
            ORDER BY a.created_at DESC
            LIMIT %s;
            """
            params = (entity_type, limit)
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
                    
            return [dict(r) for r in results]
            
        except Exception as e:
            logger.error(f"Error searching by entities: {e}")
            return []
    
    def get_sentiment_distribution(self, 
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> Dict:
        """
        Get sentiment distribution across analyses.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Sentiment distribution statistics
        """
        query = """
        SELECT 
            COUNT(*) as total,
            AVG((sentiment->>'polarity')::float) as avg_polarity,
            AVG((sentiment->>'subjectivity')::float) as avg_subjectivity,
            SUM(CASE WHEN (sentiment->>'sentiment_label') = 'positive' THEN 1 ELSE 0 END) as positive_count,
            SUM(CASE WHEN (sentiment->>'sentiment_label') = 'negative' THEN 1 ELSE 0 END) as negative_count,
            SUM(CASE WHEN (sentiment->>'sentiment_label') = 'neutral' THEN 1 ELSE 0 END) as neutral_count
        FROM nlp_analysis
        WHERE sentiment IS NOT NULL
        """
        
        params = []
        if start_date:
            query += " AND created_at >= %s"
            params.append(start_date)
        if end_date:
            query += " AND created_at <= %s"
            params.append(end_date)
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    result = cur.fetchone()
                    
            return dict(result) if result else {}
            
        except Exception as e:
            logger.error(f"Error getting sentiment distribution: {e}")
            return {}
    
    def get_top_keywords(self, limit: int = 20) -> List[Dict]:
        """
        Get most frequent keywords across all analyses.
        
        Args:
            limit: Maximum keywords to return
            
        Returns:
            List of top keywords with frequencies
        """
        query = """
        SELECT 
            keyword,
            COUNT(DISTINCT analysis_id) as document_count,
            AVG(score) as avg_score,
            MAX(score) as max_score
        FROM extracted_keywords
        GROUP BY keyword
        ORDER BY document_count DESC, avg_score DESC
        LIMIT %s;
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (limit,))
                    results = cur.fetchall()
                    
            return [dict(r) for r in results]
            
        except Exception as e:
            logger.error(f"Error getting top keywords: {e}")
            return []
    
    def get_top_entities(self, entity_type: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """
        Get most frequent entities.
        
        Args:
            entity_type: Optional filter by entity type
            limit: Maximum entities to return
            
        Returns:
            List of top entities with frequencies
        """
        if entity_type:
            query = """
            SELECT 
                entity_text,
                entity_type,
                COUNT(DISTINCT analysis_id) as document_count
            FROM extracted_entities
            WHERE entity_type = %s
            GROUP BY entity_text, entity_type
            ORDER BY document_count DESC
            LIMIT %s;
            """
            params = (entity_type, limit)
        else:
            query = """
            SELECT 
                entity_text,
                entity_type,
                COUNT(DISTINCT analysis_id) as document_count
            FROM extracted_entities
            GROUP BY entity_text, entity_type
            ORDER BY document_count DESC
            LIMIT %s;
            """
            params = (limit,)
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
                    
            return [dict(r) for r in results]
            
        except Exception as e:
            logger.error(f"Error getting top entities: {e}")
            return []