"""
Data types for the KnowHunt pipeline.

This module contains shared data classes to avoid circular imports.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class StoredContent:
    """Represents collected content stored in the pipeline."""
    id: str
    source_type: str
    source_id: str
    title: str
    content: str
    url: Optional[str]
    author: Optional[str]
    published_date: Optional[datetime]
    collected_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyzedContent:
    """Represents content that has been analyzed."""
    content_id: str
    analysis_result: Any  # AnalysisResult from nlp_analyzer
    analyzed_at: datetime
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:
    """Current state of the pipeline."""
    collected_items: List[StoredContent] = field(default_factory=list)
    enriched_items: List[Any] = field(default_factory=list)  # EnrichmentResult
    analyzed_items: List[AnalyzedContent] = field(default_factory=list)
    pipeline_started: Optional[datetime] = None
    last_collection: Optional[datetime] = None
    last_enrichment: Optional[datetime] = None
    last_analysis: Optional[datetime] = None