"""Synchronous database storage for NLP and LLM analysis."""

import json
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    logger.warning("psycopg2 not available - PostgreSQL storage will be disabled")
    PSYCOPG2_AVAILABLE = False


class DatabaseStorage:
    """Synchronous PostgreSQL storage for NLP analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize database storage.

        Args:
            config: Database configuration dict
        """
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL storage")

        if not config:
            # Default configuration for local development
            config = {
                "host": "localhost",
                "port": 5433,
                "database": "stoma",
                "user": "postgres"
            }

        self.config = config
        self._connection_params = self._build_connection_params()
        
    def _build_connection_params(self) -> Dict[str, Any]:
        """Build psycopg2 connection parameters."""
        params = {
            "database": self.config.get("database", "stoma"),
            "user": self.config.get("user", "postgres"),
            "port": self.config.get("port", 5433)
        }
        
        # Add host if specified (for TCP connections)
        if "host" in self.config:
            params["host"] = self.config["host"]
        
        # Add password if specified
        if "password" in self.config:
            params["password"] = self.config["password"]
        
        return params
    
    @contextmanager
    def get_connection(self):
        """Get a database connection context manager."""
        conn = None
        try:
            conn = psycopg2.connect(**self._connection_params)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def create_llm_analysis_tables(self):
        """Create tables for LLM analysis storage."""
        create_tables_sql = """
        -- Main LLM analysis results table
        CREATE TABLE IF NOT EXISTS llm_analysis_reports (
            id SERIAL PRIMARY KEY,
            document_id TEXT UNIQUE NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            input_text_hash TEXT,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            research_quality_score REAL,

            -- Analysis results as JSONB for flexible querying
            novel_contributions JSONB,
            technical_innovations JSONB,
            business_implications JSONB,
            research_significance JSONB,
            methodology_assessment JSONB,
            impact_prediction JSONB,
            research_gaps_identified JSONB,
            related_work_connections JSONB,
            concept_keywords JSONB,

            -- Usage statistics
            tokens_used INTEGER,
            analysis_duration_seconds REAL,

            -- File reference
            file_path TEXT,

            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Index for common queries
        CREATE INDEX IF NOT EXISTS idx_llm_reports_document_id ON llm_analysis_reports(document_id);
        CREATE INDEX IF NOT EXISTS idx_llm_reports_provider ON llm_analysis_reports(provider);
        CREATE INDEX IF NOT EXISTS idx_llm_reports_timestamp ON llm_analysis_reports(timestamp);
        CREATE INDEX IF NOT EXISTS idx_llm_reports_quality_score ON llm_analysis_reports(research_quality_score);

        -- GIN index for JSONB keyword searches
        CREATE INDEX IF NOT EXISTS idx_llm_reports_keywords ON llm_analysis_reports USING GIN (concept_keywords);
        CREATE INDEX IF NOT EXISTS idx_llm_reports_contributions ON llm_analysis_reports USING GIN (novel_contributions);
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_tables_sql)
                    conn.commit()
                logger.info("LLM analysis tables created successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to create LLM analysis tables: {e}")
            return False

    def store_llm_analysis(self, report_data: Dict[str, Any]) -> Optional[int]:
        """Store LLM analysis report in PostgreSQL."""

        analysis = report_data.get('analysis', {})
        usage_stats = report_data.get('usage_statistics', {})

        insert_sql = """
        INSERT INTO llm_analysis_reports (
            document_id, provider, model, input_text_hash, timestamp,
            research_quality_score, novel_contributions, technical_innovations,
            business_implications, research_significance, methodology_assessment,
            impact_prediction, research_gaps_identified, related_work_connections,
            concept_keywords, tokens_used, file_path
        ) VALUES (
            %(document_id)s, %(provider)s, %(model)s, %(input_text_hash)s, %(timestamp)s,
            %(research_quality_score)s, %(novel_contributions)s, %(technical_innovations)s,
            %(business_implications)s, %(research_significance)s, %(methodology_assessment)s,
            %(impact_prediction)s, %(research_gaps_identified)s, %(related_work_connections)s,
            %(concept_keywords)s, %(tokens_used)s, %(file_path)s
        )
        ON CONFLICT (document_id)
        DO UPDATE SET
            updated_at = NOW(),
            research_quality_score = EXCLUDED.research_quality_score,
            novel_contributions = EXCLUDED.novel_contributions,
            technical_innovations = EXCLUDED.technical_innovations,
            business_implications = EXCLUDED.business_implications,
            tokens_used = EXCLUDED.tokens_used
        RETURNING id;
        """

        # Generate input text hash
        import hashlib
        input_text = report_data.get('input_text', '')
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()[:16]

        params = {
            'document_id': analysis.get('document_id', ''),
            'provider': report_data.get('provider', ''),
            'model': report_data.get('model', ''),
            'input_text_hash': input_hash,
            'timestamp': report_data.get('timestamp'),
            'research_quality_score': analysis.get('research_quality_score'),
            'novel_contributions': json.dumps(analysis.get('novel_contributions', [])),
            'technical_innovations': json.dumps(analysis.get('technical_innovations', [])),
            'business_implications': json.dumps(analysis.get('business_implications', [])),
            'research_significance': json.dumps(analysis.get('research_significance', {})),
            'methodology_assessment': json.dumps(analysis.get('methodology_assessment', {})),
            'impact_prediction': json.dumps(analysis.get('impact_prediction', {})),
            'research_gaps_identified': json.dumps(analysis.get('research_gaps_identified', [])),
            'related_work_connections': json.dumps(analysis.get('related_work_connections', [])),
            'concept_keywords': json.dumps(analysis.get('concept_keywords', [])),
            'tokens_used': usage_stats.get('total_tokens'),
            'file_path': report_data.get('_file_path')
        }

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, params)
                    result = cur.fetchone()
                    conn.commit()

                    if result:
                        logger.info(f"Stored LLM analysis: {params['document_id']}")
                        return result[0]

        except Exception as e:
            logger.error(f"Failed to store LLM analysis: {e}")

        return None

    def search_llm_analysis(self, query: Optional[str] = None,
                           provider: Optional[str] = None,
                           min_quality_score: Optional[float] = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """Search LLM analysis reports in PostgreSQL."""

        base_sql = """
        SELECT
            id, document_id, provider, model, timestamp, research_quality_score,
            novel_contributions, technical_innovations, business_implications,
            concept_keywords, tokens_used, file_path
        FROM llm_analysis_reports
        WHERE 1=1
        """

        params = {}
        conditions = []

        if query:
            conditions.append("""
                (document_id ILIKE %(query)s
                 OR concept_keywords::text ILIKE %(query)s
                 OR novel_contributions::text ILIKE %(query)s)
            """)
            params['query'] = f"%{query}%"

        if provider:
            conditions.append("provider = %(provider)s")
            params['provider'] = provider

        if min_quality_score is not None:
            conditions.append("research_quality_score >= %(min_quality_score)s")
            params['min_quality_score'] = min_quality_score

        if conditions:
            base_sql += " AND " + " AND ".join(conditions)

        base_sql += " ORDER BY timestamp DESC LIMIT %(limit)s"
        params['limit'] = limit

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(base_sql, params)
                    results = cur.fetchall()

                    # Convert RealDictRow to regular dict and parse JSON fields
                    processed_results = []
                    for row in results:
                        result_dict = dict(row)

                        # Parse JSON fields back to Python objects
                        json_fields = ['novel_contributions', 'technical_innovations',
                                     'business_implications', 'concept_keywords']
                        for field in json_fields:
                            if result_dict.get(field):
                                try:
                                    result_dict[field] = json.loads(result_dict[field])
                                except (json.JSONDecodeError, TypeError):
                                    result_dict[field] = []

                        processed_results.append(result_dict)

                    return processed_results

        except Exception as e:
            logger.error(f"Failed to search LLM analysis: {e}")
            return []

    def get_llm_analysis_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get specific LLM analysis report by document ID."""

        sql = """
        SELECT * FROM llm_analysis_reports
        WHERE document_id = %(document_id)s
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, {'document_id': document_id})
                    row = cur.fetchone()

                    if row:
                        result = dict(row)

                        # Parse JSON fields
                        json_fields = [
                            'novel_contributions', 'technical_innovations',
                            'business_implications', 'research_significance',
                            'methodology_assessment', 'impact_prediction',
                            'research_gaps_identified', 'related_work_connections',
                            'concept_keywords'
                        ]

                        for field in json_fields:
                            if result.get(field):
                                try:
                                    result[field] = json.loads(result[field])
                                except (json.JSONDecodeError, TypeError):
                                    result[field] = [] if field.endswith('s') else {}

                        return result

        except Exception as e:
            logger.error(f"Failed to get LLM analysis by ID: {e}")

        return None

    def get_llm_analysis_stats(self) -> Dict[str, Any]:
        """Get statistics about stored LLM analysis reports."""

        stats_sql = """
        SELECT
            COUNT(*) as total_reports,
            COUNT(CASE WHEN research_quality_score >= 8 THEN 1 END) as high_quality,
            COUNT(CASE WHEN research_quality_score >= 5 AND research_quality_score < 8 THEN 1 END) as medium_quality,
            COUNT(CASE WHEN research_quality_score < 5 THEN 1 END) as low_quality,
            AVG(research_quality_score) as avg_quality_score,
            COUNT(CASE WHEN timestamp >= NOW() - INTERVAL '30 days' THEN 1 END) as recent_reports,
            SUM(tokens_used) as total_tokens_used
        FROM llm_analysis_reports;
        """

        provider_stats_sql = """
        SELECT provider, COUNT(*) as count, AVG(research_quality_score) as avg_score
        FROM llm_analysis_reports
        GROUP BY provider
        ORDER BY count DESC;
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get overall stats
                    cur.execute(stats_sql)
                    overall_stats = dict(cur.fetchone())

                    # Get provider stats
                    cur.execute(provider_stats_sql)
                    provider_stats = [dict(row) for row in cur.fetchall()]

                    return {
                        'total_reports': overall_stats.get('total_reports', 0),
                        'quality_distribution': {
                            'high_quality': overall_stats.get('high_quality', 0),
                            'medium_quality': overall_stats.get('medium_quality', 0),
                            'low_quality': overall_stats.get('low_quality', 0),
                            'avg_quality_score': float(overall_stats.get('avg_quality_score', 0)) if overall_stats.get('avg_quality_score') else 0
                        },
                        'recent_activity': overall_stats.get('recent_reports', 0),
                        'total_tokens_used': overall_stats.get('total_tokens_used', 0),
                        'by_provider': provider_stats
                    }

        except Exception as e:
            logger.error(f"Failed to get LLM analysis stats: {e}")
            return {
                'total_reports': 0,
                'quality_distribution': {'high_quality': 0, 'medium_quality': 0, 'low_quality': 0, 'avg_quality_score': 0},
                'recent_activity': 0,
                'total_tokens_used': 0,
                'by_provider': []
            }