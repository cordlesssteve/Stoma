"""ArXiv collector for academic papers."""

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import AsyncIterator, Dict, Any
import aiohttp

from .base import RSSCollector, CollectionResult, SourceType


class ArXivCollector(RSSCollector):
    """Collector for ArXiv academic papers."""
    
    def __init__(self, config: Dict[str, Any]):
        # ArXiv API endpoint
        if "feed_url" not in config:
            config["feed_url"] = "http://export.arxiv.org/api/query"
        super().__init__(config)
        
        # ArXiv specific configuration
        self.max_results = config.get("max_results", 50)
        self.sort_by = config.get("sort_by", "submittedDate")  # submittedDate, lastUpdatedDate, relevance
        self.sort_order = config.get("sort_order", "descending")  # ascending, descending
    
    def _get_source_type(self) -> SourceType:
        return SourceType.ACADEMIC
    
    async def collect(self, 
                     search_query: str = "all",
                     category: str = None,
                     start: int = 0,
                     **kwargs) -> AsyncIterator[CollectionResult]:
        """
        Collect papers from ArXiv.
        
        Args:
            search_query: Search terms (default: "all" for recent papers)
            category: ArXiv category (e.g., "cs.AI", "physics.gen-ph")
            start: Starting index for pagination
        """
        
        # Build query parameters
        params = {
            "search_query": self._build_search_query(search_query, category),
            "start": start,
            "max_results": self.max_results,
            "sortBy": self.sort_by,
            "sortOrder": self.sort_order
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.feed_url, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        async for result in self._parse_arxiv_response(content):
                            yield result
                            # Rate limiting
                            await asyncio.sleep(1.0 / self.rate_limit)
                    else:
                        yield CollectionResult(
                            source_id="arxiv_error",
                            source_type=self.source_type,
                            collected_at=datetime.now(),
                            data={},
                            metadata={"error": f"HTTP {response.status}"},
                            success=False,
                            error_message=f"Failed to fetch from ArXiv: HTTP {response.status}"
                        )
            except Exception as e:
                yield CollectionResult(
                    source_id="arxiv_exception",
                    source_type=self.source_type,
                    collected_at=datetime.now(),
                    data={},
                    metadata={"error": str(e)},
                    success=False,
                    error_message=f"Exception while collecting from ArXiv: {e}"
                )
    
    def _build_search_query(self, search_query: str, category: str = None) -> str:
        """Build ArXiv API search query."""
        if search_query == "all":
            # Get recent papers
            query = "all"
        else:
            # Search in title, abstract, and comments
            query = f"ti:{search_query} OR abs:{search_query} OR co:{search_query}"
        
        if category:
            query = f"cat:{category} AND ({query})"
        
        return query
    
    async def _parse_arxiv_response(self, xml_content: str) -> AsyncIterator[CollectionResult]:
        """Parse ArXiv API XML response."""
        try:
            root = ET.fromstring(xml_content)
            
            # Define XML namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Find all entry elements
            entries = root.findall('.//atom:entry', namespaces)
            
            for entry in entries:
                try:
                    paper_data = await self._parse_arxiv_entry(entry, namespaces)
                    if paper_data:
                        yield CollectionResult(
                            source_id=paper_data["id"],
                            source_type=self.source_type,
                            collected_at=datetime.now(),
                            data=paper_data,
                            metadata={
                                "source": "arxiv",
                                "api_version": "1.0"
                            },
                            raw_content=ET.tostring(entry, encoding='unicode')
                        )
                except Exception as e:
                    # Log error but continue processing other entries
                    print(f"Error parsing ArXiv entry: {e}")
                    continue
                    
        except ET.ParseError as e:
            print(f"Error parsing ArXiv XML: {e}")
    
    async def _parse_arxiv_entry(self, entry, namespaces) -> Dict[str, Any]:
        """Parse individual ArXiv paper entry."""
        
        # Extract basic information
        title_elem = entry.find('atom:title', namespaces)
        title = title_elem.text.strip() if title_elem is not None else ""
        
        summary_elem = entry.find('atom:summary', namespaces)
        abstract = summary_elem.text.strip() if summary_elem is not None else ""
        
        # Extract ArXiv ID from the id field
        id_elem = entry.find('atom:id', namespaces)
        arxiv_url = id_elem.text if id_elem is not None else ""
        arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else ""
        
        # Extract authors
        authors = []
        for author in entry.findall('atom:author', namespaces):
            name_elem = author.find('atom:name', namespaces)
            if name_elem is not None:
                authors.append(name_elem.text.strip())
        
        # Extract dates
        published_elem = entry.find('atom:published', namespaces)
        published_date = published_elem.text if published_elem is not None else None
        
        updated_elem = entry.find('atom:updated', namespaces)
        updated_date = updated_elem.text if updated_elem is not None else None
        
        # Extract categories
        categories = []
        for category in entry.findall('atom:category', namespaces):
            term = category.get('term')
            if term:
                categories.append(term)
        
        # Extract links
        pdf_url = None
        abstract_url = None
        for link in entry.findall('atom:link', namespaces):
            rel = link.get('rel')
            href = link.get('href')
            title = link.get('title')
            
            if title == 'pdf':
                pdf_url = href
            elif rel == 'alternate':
                abstract_url = href
        
        # Extract ArXiv-specific fields
        comment_elem = entry.find('arxiv:comment', namespaces)
        comment = comment_elem.text if comment_elem is not None else None
        
        journal_ref_elem = entry.find('arxiv:journal_ref', namespaces)
        journal_ref = journal_ref_elem.text if journal_ref_elem is not None else None
        
        doi_elem = entry.find('arxiv:doi', namespaces)
        doi = doi_elem.text if doi_elem is not None else None
        
        primary_category_elem = entry.find('arxiv:primary_category', namespaces)
        primary_category = primary_category_elem.get('term') if primary_category_elem is not None else None
        
        return {
            "id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "published": published_date,
            "updated": updated_date,
            "categories": categories,
            "primary_category": primary_category,
            "url": abstract_url,
            "pdf_url": pdf_url,
            "comment": comment,
            "journal_ref": journal_ref,
            "doi": doi,
            "keywords": categories,  # Use categories as keywords for now
        }
    
    async def health_check(self) -> bool:
        """Check if ArXiv API is accessible."""
        try:
            async with aiohttp.ClientSession() as session:
                # Test with a simple query
                params = {
                    "search_query": "all",
                    "start": 0,
                    "max_results": 1
                }
                async with session.get(self.feed_url, params=params) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def get_categories(self) -> Dict[str, str]:
        """Get available ArXiv categories."""
        # Common ArXiv categories
        return {
            "cs.AI": "Artificial Intelligence",
            "cs.CL": "Computation and Language",
            "cs.CV": "Computer Vision and Pattern Recognition",
            "cs.LG": "Machine Learning",
            "cs.NE": "Neural and Evolutionary Computing",
            "cs.RO": "Robotics",
            "cs.SE": "Software Engineering",
            "math.ST": "Statistics Theory",
            "physics.data-an": "Data Analysis, Statistics and Probability",
            "q-bio.QM": "Quantitative Methods",
            "stat.ML": "Machine Learning (Statistics)",
            "econ.EM": "Econometrics",
            "q-fin.CP": "Computational Finance",
        }