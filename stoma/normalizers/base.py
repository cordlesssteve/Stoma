"""Base classes for data normalizers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from ..collectors.base import CollectionResult, SourceType


@dataclass
class NormalizedData:
    """Standardized data structure across all sources."""
    # Core identification
    id: str                    # Unique identifier
    source_type: SourceType    # Type of source
    source_id: str            # Original source identifier
    
    # Content
    title: str
    content: str
    summary: Optional[str] = None
    
    # Metadata
    authors: List[str] = None
    published_date: Optional[datetime] = None
    collected_date: datetime = None
    url: Optional[str] = None
    
    # Classification
    keywords: List[str] = None
    categories: List[str] = None
    tags: List[str] = None
    
    # Metrics
    metrics: Dict[str, Any] = None  # view counts, citations, etc.
    
    # Raw data preservation
    raw_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.keywords is None:
            self.keywords = []
        if self.categories is None:
            self.categories = []
        if self.tags is None:
            self.tags = []
        if self.metrics is None:
            self.metrics = {}
        if self.raw_data is None:
            self.raw_data = {}


class BaseNormalizer(ABC):
    """Abstract base class for data normalizers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_source_types = self._get_supported_source_types()
    
    @abstractmethod
    def _get_supported_source_types(self) -> List[SourceType]:
        """Return list of source types this normalizer supports."""
        pass
    
    @abstractmethod
    async def normalize(self, result: CollectionResult) -> NormalizedData:
        """Convert collection result to normalized format."""
        pass
    
    def can_handle(self, source_type: SourceType) -> bool:
        """Check if this normalizer can handle the given source type."""
        return source_type in self.supported_source_types
    
    def _extract_text_content(self, html_content: str) -> str:
        """Extract plain text from HTML content."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(strip=True)
    
    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate a basic summary by truncating content."""
        if len(content) <= max_length:
            return content
        
        # Find the last complete sentence within the limit
        truncated = content[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length // 2:  # Only use if it's not too short
            return truncated[:last_period + 1]
        else:
            return truncated + "..."


class AcademicNormalizer(BaseNormalizer):
    """Normalizer for academic papers and research."""
    
    def _get_supported_source_types(self) -> List[SourceType]:
        return [SourceType.ACADEMIC]
    
    async def normalize(self, result: CollectionResult) -> NormalizedData:
        """Normalize academic paper data."""
        data = result.data
        
        return NormalizedData(
            id=self._generate_id(result),
            source_type=result.source_type,
            source_id=result.source_id,
            title=data.get("title", ""),
            content=data.get("abstract", ""),
            summary=data.get("summary"),
            authors=data.get("authors", []),
            published_date=self._parse_date(data.get("published")),
            collected_date=result.collected_at,
            url=data.get("url"),
            keywords=data.get("keywords", []),
            categories=data.get("categories", []),
            metrics=self._extract_academic_metrics(data),
            raw_data=data
        )
    
    def _generate_id(self, result: CollectionResult) -> str:
        """Generate unique ID for academic papers."""
        import hashlib
        content = f"{result.source_id}_{result.data.get('title', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse various date formats commonly used in academic sources."""
        if not date_str:
            return None
        
        import dateutil.parser
        try:
            return dateutil.parser.parse(date_str)
        except Exception:
            return None
    
    def _extract_academic_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract academic-specific metrics."""
        metrics = {}
        
        if "citation_count" in data:
            metrics["citations"] = data["citation_count"]
        if "download_count" in data:
            metrics["downloads"] = data["download_count"]
        if "journal_impact_factor" in data:
            metrics["impact_factor"] = data["journal_impact_factor"]
            
        return metrics