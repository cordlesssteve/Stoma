"""
Core data pipeline: Collection → Storage → Analysis → Reporting

This is the heart of KnowHunt - a proper data flow that connects all components.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from ..collectors.base import CollectionResult
from ..analysis.nlp_analyzer import AnalysisResult
from .data_types import StoredContent, AnalyzedContent, PipelineState

logger = logging.getLogger(__name__)




class DataPipeline:
    """
    Main data pipeline that orchestrates collection, analysis, and reporting.
    
    This is the central orchestrator that:
    1. Runs collectors to gather data
    2. Stores collected data
    3. Runs analysis on stored data
    4. Provides analyzed data to report generators
    """
    
    def __init__(self, storage_dir: str = "./pipeline_data"):
        """Initialize the data pipeline."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory state (could be replaced with database later)
        self.state = PipelineState()
        
        # Pipeline components will be injected
        self.collectors = {}
        self.analyzer = None
        self.content_enricher = None
        
        # Load existing state if available
        self._load_state()
    
    def register_collector(self, name: str, collector):
        """Register a data collector with the pipeline."""
        self.collectors[name] = collector
        logger.info(f"Registered collector: {name}")
    
    def set_analyzer(self, analyzer):
        """Set the NLP analyzer for the pipeline."""
        self.analyzer = analyzer
        logger.info("NLP analyzer registered with pipeline")
    
    def set_content_enricher(self, enricher):
        """Set the content enricher for the pipeline."""
        self.content_enricher = enricher
        logger.info("Content enricher registered with pipeline")
    
    async def run_collection_cycle(self, 
                                 collector_configs: Optional[Dict[str, Dict]] = None) -> int:
        """
        Run a complete collection cycle across all registered collectors.
        
        Args:
            collector_configs: Optional configurations for each collector
            
        Returns:
            Number of items collected
        """
        if not self.collectors:
            raise ValueError("No collectors registered")
        
        self.state.pipeline_started = datetime.now()
        collected_count = 0
        
        logger.info(f"Starting collection cycle with {len(self.collectors)} collectors")
        
        for collector_name, collector in self.collectors.items():
            try:
                collector_config = collector_configs.get(collector_name, {}) if collector_configs else {}
                
                logger.info(f"Running collector: {collector_name}")
                
                async for result in collector.collect(**collector_config):
                    if result.success and result.data:
                        stored_content = self._store_collection_result(result)
                        if stored_content:
                            self.state.collected_items.append(stored_content)
                            collected_count += 1
                            
                            if collected_count % 10 == 0:
                                logger.info(f"Collected {collected_count} items so far...")
                    
            except Exception as e:
                logger.error(f"Error in collector {collector_name}: {e}")
                continue
        
        self.state.last_collection = datetime.now()
        self._save_state()
        
        logger.info(f"Collection cycle complete: {collected_count} items collected")
        return collected_count
    
    async def run_enrichment_cycle(self, max_items: Optional[int] = None) -> int:
        """
        Run content enrichment on collected content that hasn't been enriched yet.
        
        Args:
            max_items: Optional limit on number of items to enrich
            
        Returns:
            Number of items enriched
        """
        if not self.content_enricher:
            logger.warning("No content enricher registered - skipping enrichment")
            return 0
        
        # Find items that need enrichment
        enriched_content_ids = {item.original_content.id for item in self.state.enriched_items}
        unenriched_items = [
            item for item in self.state.collected_items 
            if item.id not in enriched_content_ids
        ]
        
        if max_items:
            unenriched_items = unenriched_items[:max_items]
        
        if not unenriched_items:
            logger.info("No new items to enrich")
            return 0
        
        logger.info(f"Starting enrichment of {len(unenriched_items)} items")
        
        # Run enrichment in batches
        enrichment_results = await self.content_enricher.enrich_content_batch(
            unenriched_items, max_concurrent=3
        )
        
        # Store enrichment results
        enriched_count = 0
        for result in enrichment_results:
            self.state.enriched_items.append(result)
            if result.enrichment_successful:
                enriched_count += 1
        
        self.state.last_enrichment = datetime.now()
        self._save_state()
        
        logger.info(f"Enrichment cycle complete: {enriched_count}/{len(enrichment_results)} items successfully enriched")
        return enriched_count
    
    async def run_analysis_cycle(self, max_items: Optional[int] = None) -> int:
        """
        Run analysis on all collected content that hasn't been analyzed yet.
        
        Args:
            max_items: Optional limit on number of items to analyze
            
        Returns:
            Number of items analyzed
        """
        if not self.analyzer:
            raise ValueError("No analyzer registered")
        
        # Find items that need analysis
        analyzed_content_ids = {item.content_id for item in self.state.analyzed_items}
        
        # Prefer enriched content over original content for analysis
        items_for_analysis = []
        enriched_content_map = {item.original_content.id: item for item in self.state.enriched_items}
        
        for content_item in self.state.collected_items:
            if content_item.id not in analyzed_content_ids:
                # Use enriched content if available, otherwise original
                if content_item.id in enriched_content_map:
                    enrichment_result = enriched_content_map[content_item.id]
                    if enrichment_result.enrichment_successful:
                        # Create a modified content item with enriched text
                        enriched_content_item = StoredContent(
                            id=content_item.id,
                            source_type=content_item.source_type,
                            source_id=content_item.source_id,
                            title=content_item.title,
                            content=enrichment_result.enriched_content,  # Use enriched content
                            url=content_item.url,
                            author=content_item.author,
                            published_date=content_item.published_date,
                            collected_at=content_item.collected_at,
                            metadata=content_item.metadata,
                            raw_data=content_item.raw_data
                        )
                        items_for_analysis.append(enriched_content_item)
                    else:
                        items_for_analysis.append(content_item)
                else:
                    items_for_analysis.append(content_item)
        
        unanalyzed_items = items_for_analysis
        
        if max_items:
            unanalyzed_items = unanalyzed_items[:max_items]
        
        if not unanalyzed_items:
            logger.info("No new items to analyze")
            return 0
        
        logger.info(f"Starting analysis of {len(unanalyzed_items)} items")
        analyzed_count = 0
        
        for content_item in unanalyzed_items:
            try:
                # Run NLP analysis
                analysis_result = self.analyzer.analyze(content_item.content)
                
                # Store analysis result
                analyzed_content = AnalyzedContent(
                    content_id=content_item.id,
                    analysis_result=analysis_result,
                    analyzed_at=datetime.now(),
                    analysis_metadata={
                        'source_type': content_item.source_type,
                        'title': content_item.title,
                        'content_length': len(content_item.content)
                    }
                )
                
                self.state.analyzed_items.append(analyzed_content)
                analyzed_count += 1
                
                if analyzed_count % 5 == 0:
                    logger.info(f"Analyzed {analyzed_count} items so far...")
                    
            except Exception as e:
                logger.error(f"Error analyzing content {content_item.id}: {e}")
                continue
        
        self.state.last_analysis = datetime.now()
        self._save_state()
        
        logger.info(f"Analysis cycle complete: {analyzed_count} items analyzed")
        return analyzed_count
    
    def get_recent_content(self, 
                          hours_back: int = 24,
                          source_types: Optional[List[str]] = None,
                          limit: Optional[int] = None) -> List[StoredContent]:
        """Get recently collected content."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        filtered_content = [
            item for item in self.state.collected_items
            if item.collected_at >= cutoff_time
        ]
        
        if source_types:
            filtered_content = [
                item for item in filtered_content
                if item.source_type in source_types
            ]
        
        # Sort by collection time (newest first)
        filtered_content.sort(key=lambda x: x.collected_at, reverse=True)
        
        if limit:
            filtered_content = filtered_content[:limit]
        
        return filtered_content
    
    def get_analyzed_content(self,
                           hours_back: int = 24,
                           source_types: Optional[List[str]] = None,
                           limit: Optional[int] = None) -> List[AnalyzedContent]:
        """Get recently analyzed content with analysis results."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Get analyzed content within timeframe
        filtered_analyzed = [
            item for item in self.state.analyzed_items
            if item.analyzed_at >= cutoff_time
        ]
        
        if source_types:
            filtered_analyzed = [
                item for item in filtered_analyzed
                if item.analysis_metadata.get('source_type') in source_types
            ]
        
        # Sort by analysis time (newest first)
        filtered_analyzed.sort(key=lambda x: x.analyzed_at, reverse=True)
        
        if limit:
            filtered_analyzed = filtered_analyzed[:limit]
        
        return filtered_analyzed
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get overall pipeline statistics."""
        now = datetime.now()
        
        # Content by source type
        content_by_source = {}
        for item in self.state.collected_items:
            content_by_source[item.source_type] = content_by_source.get(item.source_type, 0) + 1
        
        # Recent activity (last 24 hours)
        recent_collections = len(self.get_recent_content(24))
        recent_analyses = len(self.get_analyzed_content(24))
        
        # Analysis coverage
        total_collected = len(self.state.collected_items)
        total_analyzed = len(self.state.analyzed_items)
        analysis_coverage = (total_analyzed / total_collected * 100) if total_collected > 0 else 0
        
        return {
            'pipeline_status': {
                'started': self.state.pipeline_started.isoformat() if self.state.pipeline_started else None,
                'last_collection': self.state.last_collection.isoformat() if self.state.last_collection else None,
                'last_analysis': self.state.last_analysis.isoformat() if self.state.last_analysis else None,
            },
            'content_statistics': {
                'total_collected': total_collected,
                'total_analyzed': total_analyzed,
                'analysis_coverage_percent': round(analysis_coverage, 2),
                'content_by_source': content_by_source,
            },
            'recent_activity': {
                'collections_24h': recent_collections,
                'analyses_24h': recent_analyses,
            },
            'registered_collectors': list(self.collectors.keys()),
            'analyzer_available': self.analyzer is not None,
        }
    
    def _store_collection_result(self, result: CollectionResult) -> Optional[StoredContent]:
        """Convert CollectionResult to StoredContent."""
        try:
            # Extract key fields from the collection result
            title = result.data.get('title', 'Untitled')
            content = result.data.get('content', result.data.get('abstract', result.data.get('selftext', '')))
            
            if not content or len(content.strip()) < 10:
                return None  # Skip items with insufficient content
            
            stored_content = StoredContent(
                id=f"{result.source_type.value}_{result.source_id}_{int(result.collected_at.timestamp())}",
                source_type=result.source_type.value,
                source_id=result.source_id,
                title=title,
                content=content,
                url=result.data.get('url', result.data.get('pdf_url')),
                author=result.data.get('author', result.data.get('authors')),
                published_date=self._parse_date(result.data.get('created_date', result.data.get('published'))),
                collected_at=result.collected_at,
                metadata=result.metadata,
                raw_data=result.data
            )
            
            return stored_content
            
        except Exception as e:
            logger.error(f"Error storing collection result: {e}")
            return None
    
    def _parse_date(self, date_str: Any) -> Optional[datetime]:
        """Parse various date formats."""
        if not date_str:
            return None
        
        if isinstance(date_str, datetime):
            return date_str
        
        if isinstance(date_str, list) and date_str:
            date_str = date_str[0]
        
        if isinstance(date_str, str):
            try:
                # Try ISO format first
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except:
                pass
        
        return None
    
    def _save_state(self):
        """Save pipeline state to disk."""
        try:
            state_file = self.storage_dir / "pipeline_state.json"
            
            # Convert state to JSON-serializable format
            state_data = {
                'pipeline_started': self.state.pipeline_started.isoformat() if self.state.pipeline_started else None,
                'last_collection': self.state.last_collection.isoformat() if self.state.last_collection else None,
                'last_analysis': self.state.last_analysis.isoformat() if self.state.last_analysis else None,
                'collected_items': [
                    {
                        'id': item.id,
                        'source_type': item.source_type,
                        'source_id': item.source_id,
                        'title': item.title,
                        'content': item.content,
                        'url': item.url,
                        'author': str(item.author) if item.author else None,
                        'published_date': item.published_date.isoformat() if item.published_date else None,
                        'collected_at': item.collected_at.isoformat(),
                        'metadata': item.metadata,
                        'raw_data': item.raw_data
                    }
                    for item in self.state.collected_items
                ],
                'analyzed_items': [
                    {
                        'content_id': item.content_id,
                        'analyzed_at': item.analyzed_at.isoformat(),
                        'analysis_metadata': item.analysis_metadata,
                        # Store key analysis results
                        'keywords': item.analysis_result.keywords,
                        'word_count': item.analysis_result.word_count,
                        'sentiment': item.analysis_result.sentiment,
                        'entities': item.analysis_result.entities,
                        'summary': item.analysis_result.summary
                    }
                    for item in self.state.analyzed_items
                ]
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving pipeline state: {e}")
    
    def _load_state(self):
        """Load pipeline state from disk."""
        try:
            state_file = self.storage_dir / "pipeline_state.json"
            
            if not state_file.exists():
                return
            
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore pipeline timestamps
            if state_data.get('pipeline_started'):
                self.state.pipeline_started = datetime.fromisoformat(state_data['pipeline_started'])
            if state_data.get('last_collection'):
                self.state.last_collection = datetime.fromisoformat(state_data['last_collection'])
            if state_data.get('last_analysis'):
                self.state.last_analysis = datetime.fromisoformat(state_data['last_analysis'])
            
            # Restore collected items
            for item_data in state_data.get('collected_items', []):
                content = StoredContent(
                    id=item_data['id'],
                    source_type=item_data['source_type'],
                    source_id=item_data['source_id'],
                    title=item_data['title'],
                    content=item_data['content'],
                    url=item_data.get('url'),
                    author=item_data.get('author'),
                    published_date=datetime.fromisoformat(item_data['published_date']) if item_data.get('published_date') else None,
                    collected_at=datetime.fromisoformat(item_data['collected_at']),
                    metadata=item_data.get('metadata', {}),
                    raw_data=item_data.get('raw_data', {})
                )
                self.state.collected_items.append(content)
            
            # Note: We'll skip restoring analyzed items for now to keep this simple
            # In a full implementation, we'd reconstruct AnalysisResult objects
            
            logger.info(f"Loaded pipeline state: {len(self.state.collected_items)} collected items")
            
        except Exception as e:
            logger.error(f"Error loading pipeline state: {e}")