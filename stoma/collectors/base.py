"""Base classes for data collectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncIterator
from enum import Enum


class SourceType(Enum):
    """Types of data sources."""
    ACADEMIC = "academic"
    PUBLIC_DOCS = "public_docs"
    CORPORATE = "corporate"
    CODE_PROJECTS = "code_projects"
    SOCIAL_PROBLEMS = "social_problems"
    SOCIAL_DISCUSSIONS = "social_discussions"
    TECH_TRENDS = "tech_trends"


@dataclass
class CollectionResult:
    """Result from a data collection operation."""
    source_id: str
    source_type: SourceType
    collected_at: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    raw_content: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class BaseCollector(ABC):
    """Abstract base class for all data collectors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.source_type = self._get_source_type()
        self.rate_limit = config.get("rate_limit", 1.0)  # requests per second
        
    @abstractmethod
    def _get_source_type(self) -> SourceType:
        """Return the source type for this collector."""
        pass
    
    @abstractmethod
    async def collect(self, **kwargs) -> AsyncIterator[CollectionResult]:
        """Collect data from the source."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the data source is accessible."""
        pass
    
    def get_identifier(self) -> str:
        """Get unique identifier for this collector."""
        return f"{self.__class__.__name__}_{self.source_type.value}"


class RSSCollector(BaseCollector):
    """Base class for RSS/feed-based collectors."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.feed_url = config["feed_url"]
        
    async def health_check(self) -> bool:
        """Check if RSS feed is accessible."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.feed_url) as response:
                    return response.status == 200
        except Exception:
            return False


class APICollector(BaseCollector):
    """Base class for API-based collectors."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config["base_url"]
        self.headers = config.get("headers", {})
        
    async def health_check(self) -> bool:
        """Check if API is accessible."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health", 
                    headers=self.headers
                ) as response:
                    return response.status == 200
        except Exception:
            return False


class WebScraperCollector(BaseCollector):
    """Base class for web scraping collectors."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config["base_url"]
        self.user_agent = config.get(
            "user_agent", 
            "Stoma/1.0 (Research Intelligence System)"
        )
        
    async def health_check(self) -> bool:
        """Check if website is accessible."""
        import aiohttp
        try:
            headers = {"User-Agent": self.user_agent}
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, headers=headers) as response:
                    return response.status == 200
        except Exception:
            return False