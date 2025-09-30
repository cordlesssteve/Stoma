"""Data collection modules for various sources."""

from .base import BaseCollector, APICollector, RSSCollector, WebScraperCollector
from .base import CollectionResult, SourceType
from .arxiv import ArXivCollector
from .github import GitHubCollector  
from .sec_edgar import SECEdgarCollector
from .reddit import RedditCollector
from .hackernews import HackerNewsCollector

__all__ = [
    "BaseCollector",
    "APICollector", 
    "RSSCollector",
    "WebScraperCollector",
    "CollectionResult",
    "SourceType",
    "ArXivCollector",
    "GitHubCollector",
    "SECEdgarCollector", 
    "RedditCollector",
    "HackerNewsCollector"
]