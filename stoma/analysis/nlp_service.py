"""Service layer for NLP analysis pipeline."""

import logging
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..storage.database import DatabaseStorage
from .nlp_analyzer import NLPAnalyzer, AnalysisResult
from .nlp_storage import NLPStorage

logger = logging.getLogger(__name__)


class NLPService:
    """Service for managing NLP analysis workflows."""
    
    def __init__(self, 
                 db_storage: Optional[DatabaseStorage] = None):
        """
        Initialize NLP service.
        
        Args:
            db_storage: Database storage instance
        """
        self.db = db_storage or DatabaseStorage()
        self.nlp_storage = NLPStorage(self.db)
        self.analyzer = NLPAnalyzer()
        
        logger.info("NLP Service initialized")
    
    def analyze_paper(self, paper_id: int) -> Optional[AnalysisResult]:
        """
        Analyze a specific paper from the database.
        
        Args:
            paper_id: Paper ID in database
            
        Returns:
            Analysis result or None if paper not found
        """
        try:
            # Get paper content
            paper = self._get_paper_with_content(paper_id)
            if not paper:
                logger.warning(f"Paper {paper_id} not found")
                return None
            
            # Use full text if available, otherwise use abstract
            text = paper.get('full_text') or paper.get('abstract', '')
            
            if not text:
                logger.warning(f"No text content for paper {paper_id}")
                return None
            
            # Perform analysis
            start_time = time.time()
            result = self.analyzer.analyze(
                text=text,
                document_id=f"paper_{paper_id}"
            )
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Add paper metadata to result
            result.metadata.update({
                'paper_id': paper_id,
                'title': paper.get('title'),
                'authors': paper.get('authors'),
                'publication_date': str(paper.get('publication_date', ''))
            })
            
            # Store analysis
            self.nlp_storage.store_analysis(
                result=result,
                paper_id=paper_id,
                processing_time_ms=processing_time_ms
            )
            
            logger.info(f"Analyzed paper {paper_id} in {processing_time_ms}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing paper {paper_id}: {e}")
            return None
    
    def analyze_document(self, 
                        file_path: str,
                        document_id: Optional[str] = None) -> Optional[AnalysisResult]:
        """
        Analyze a document from file.
        
        Args:
            file_path: Path to document file
            document_id: Optional document identifier
            
        Returns:
            Analysis result or None if file not found
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                return None
            
            # Read file content
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text:
                logger.warning(f"Empty file: {file_path}")
                return None
            
            # Generate document ID if not provided
            if not document_id:
                document_id = f"file_{path.stem}_{hash(text) % 1000000}"
            
            # Perform analysis
            start_time = time.time()
            result = self.analyzer.analyze(
                text=text,
                document_id=document_id
            )
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Add file metadata
            result.metadata.update({
                'file_path': str(path),
                'file_name': path.name,
                'file_size': path.stat().st_size
            })
            
            # Store analysis
            self.nlp_storage.store_analysis(
                result=result,
                processing_time_ms=processing_time_ms
            )
            
            logger.info(f"Analyzed file {file_path} in {processing_time_ms}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return None
    
    def analyze_text(self, 
                    text: str,
                    document_id: Optional[str] = None,
                    store_result: bool = True) -> Optional[AnalysisResult]:
        """
        Analyze raw text.
        
        Args:
            text: Text to analyze
            document_id: Optional document identifier
            store_result: Whether to store result in database
            
        Returns:
            Analysis result
        """
        try:
            if not text:
                logger.warning("Empty text provided")
                return None
            
            # Generate document ID if not provided
            if not document_id:
                document_id = f"text_{hash(text) % 1000000}"
            
            # Perform analysis
            start_time = time.time()
            result = self.analyzer.analyze(
                text=text,
                document_id=document_id
            )
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Store if requested
            if store_result:
                self.nlp_storage.store_analysis(
                    result=result,
                    processing_time_ms=processing_time_ms
                )
            
            logger.info(f"Analyzed text (ID: {document_id}) in {processing_time_ms}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return None
    
    def batch_analyze_papers(self, 
                            paper_ids: Optional[List[int]] = None,
                            limit: int = 100) -> Dict[str, Any]:
        """
        Analyze multiple papers in batch.
        
        Args:
            paper_ids: Specific paper IDs to analyze, or None for unanalyzed papers
            limit: Maximum papers to process
            
        Returns:
            Summary of batch processing
        """
        summary = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'processing_time_ms': 0,
            'results': []
        }
        
        start_time = time.time()
        
        try:
            # Get papers to analyze
            if paper_ids:
                papers = paper_ids[:limit]
            else:
                papers = self._get_unanalyzed_papers(limit)
            
            logger.info(f"Starting batch analysis of {len(papers)} papers")
            
            for paper_id in papers:
                # Check if already analyzed
                existing = self.nlp_storage.get_analysis(f"paper_{paper_id}")
                if existing:
                    summary['skipped'] += 1
                    continue
                
                # Analyze paper
                result = self.analyze_paper(paper_id)
                
                if result:
                    summary['successful'] += 1
                    summary['results'].append({
                        'paper_id': paper_id,
                        'document_id': result.document_id,
                        'word_count': result.word_count,
                        'keywords': len(result.keywords),
                        'entities': sum(len(v) for v in result.entities.values())
                    })
                else:
                    summary['failed'] += 1
                
                summary['total_processed'] += 1
            
            summary['processing_time_ms'] = int((time.time() - start_time) * 1000)
            
            logger.info(
                f"Batch analysis complete: {summary['successful']} successful, "
                f"{summary['failed']} failed, {summary['skipped']} skipped, "
                f"Time: {summary['processing_time_ms']}ms"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            summary['error'] = str(e)
            return summary
    
    def _get_paper_with_content(self, paper_id: int) -> Optional[Dict]:
        """Get paper with full text content."""
        query = """
        SELECT 
            p.id,
            p.title,
            p.authors,
            p.abstract,
            p.publication_date,
            p.full_text,
            p.pdf_path
        FROM papers p
        WHERE p.id = %s
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (paper_id,))
                    result = cur.fetchone()
                    
                    if result:
                        return {
                            'id': result[0],
                            'title': result[1],
                            'authors': result[2],
                            'abstract': result[3],
                            'publication_date': result[4],
                            'full_text': result[5],
                            'pdf_path': result[6]
                        }
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting paper {paper_id}: {e}")
            return None
    
    def _get_unanalyzed_papers(self, limit: int) -> List[int]:
        """Get papers that haven't been analyzed yet."""
        query = """
        SELECT p.id
        FROM papers p
        LEFT JOIN nlp_analysis a ON a.paper_id = p.id
        WHERE a.id IS NULL
        AND (p.full_text IS NOT NULL OR p.abstract IS NOT NULL)
        ORDER BY p.created_at DESC
        LIMIT %s
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (limit,))
                    results = cur.fetchall()
                    return [r[0] for r in results]
                    
        except Exception as e:
            logger.error(f"Error getting unanalyzed papers: {e}")
            return []
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all analyses."""
        query = """
        SELECT 
            COUNT(*) as total_analyses,
            COUNT(DISTINCT paper_id) as analyzed_papers,
            AVG(word_count) as avg_word_count,
            AVG(readability_score) as avg_readability,
            AVG(processing_time_ms) as avg_processing_time,
            MIN(created_at) as first_analysis,
            MAX(created_at) as last_analysis
        FROM nlp_analysis
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    result = cur.fetchone()
                    
                    if result:
                        return {
                            'total_analyses': result[0],
                            'analyzed_papers': result[1],
                            'avg_word_count': float(result[2]) if result[2] else 0,
                            'avg_readability': float(result[3]) if result[3] else 0,
                            'avg_processing_time_ms': float(result[4]) if result[4] else 0,
                            'first_analysis': str(result[5]) if result[5] else None,
                            'last_analysis': str(result[6]) if result[6] else None
                        }
                    
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting analysis summary: {e}")
            return {}