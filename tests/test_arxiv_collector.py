"""Tests for ArXiv collector."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from knowhunt.collectors.arxiv import ArXivCollector
from knowhunt.collectors.base import SourceType


@pytest.fixture
def arxiv_collector():
    """Create ArXiv collector instance for testing."""
    config = {
        "max_results": 5,
        "rate_limit": 10.0  # Fast for testing
    }
    return ArXivCollector(config)


@pytest.mark.asyncio
async def test_arxiv_collector_initialization(arxiv_collector):
    """Test ArXiv collector initialization."""
    assert arxiv_collector.source_type == SourceType.ACADEMIC
    assert arxiv_collector.max_results == 5
    assert arxiv_collector.rate_limit == 10.0
    assert "export.arxiv.org" in arxiv_collector.feed_url


@pytest.mark.asyncio
async def test_build_search_query(arxiv_collector):
    """Test search query building."""
    # Test basic query
    query = arxiv_collector._build_search_query("machine learning")
    assert "machine learning" in query
    
    # Test category filter
    query_with_cat = arxiv_collector._build_search_query("deep learning", "cs.AI")
    assert "cat:cs.AI" in query_with_cat
    assert "deep learning" in query_with_cat


@pytest.mark.asyncio
async def test_health_check_success(arxiv_collector):
    """Test successful health check."""
    with patch('aiohttp.ClientSession.get') as mock_get:
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_get.return_value.__aenter__.return_value = mock_response
        
        health = await arxiv_collector.health_check()
        assert health is True


@pytest.mark.asyncio
async def test_health_check_failure(arxiv_collector):
    """Test failed health check."""
    with patch('aiohttp.ClientSession.get') as mock_get:
        # Mock failed response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_get.return_value.__aenter__.return_value = mock_response
        
        health = await arxiv_collector.health_check()
        assert health is False


@pytest.mark.asyncio
async def test_get_categories(arxiv_collector):
    """Test getting ArXiv categories."""
    categories = await arxiv_collector.get_categories()
    
    assert isinstance(categories, dict)
    assert "cs.AI" in categories
    assert "Machine Learning" in categories.get("cs.LG", "")


@pytest.mark.asyncio 
async def test_parse_arxiv_entry(arxiv_collector):
    """Test parsing individual ArXiv entry."""
    # Sample ArXiv XML entry
    xml_entry = '''
    <entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
        <id>http://arxiv.org/abs/2301.00001v1</id>
        <title>Test Paper Title</title>
        <summary>This is a test abstract for the paper.</summary>
        <author>
            <name>John Doe</name>
        </author>
        <author>
            <name>Jane Smith</name>
        </author>
        <published>2023-01-01T00:00:00Z</published>
        <updated>2023-01-01T00:00:00Z</updated>
        <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
        <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
        <link href="http://arxiv.org/abs/2301.00001v1" rel="alternate" type="text/html"/>
        <link title="pdf" href="http://arxiv.org/pdf/2301.00001v1.pdf" rel="related" type="application/pdf"/>
        <arxiv:primary_category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
        <arxiv:comment>10 pages, 3 figures</arxiv:comment>
    </entry>
    '''
    
    import xml.etree.ElementTree as ET
    entry = ET.fromstring(xml_entry)
    
    namespaces = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    
    paper_data = await arxiv_collector._parse_arxiv_entry(entry, namespaces)
    
    assert paper_data["id"] == "2301.00001v1"
    assert paper_data["title"] == "Test Paper Title"
    assert paper_data["abstract"] == "This is a test abstract for the paper."
    assert "John Doe" in paper_data["authors"]
    assert "Jane Smith" in paper_data["authors"]
    assert "cs.AI" in paper_data["categories"]
    assert "cs.LG" in paper_data["categories"]
    assert paper_data["primary_category"] == "cs.AI"
    assert "arxiv.org/abs" in paper_data["url"]
    assert "arxiv.org/pdf" in paper_data["pdf_url"]


if __name__ == "__main__":
    pytest.main([__file__])