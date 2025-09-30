"""
Content enrichment orchestrator for KnowHunt pipeline.

This module coordinates web scraping and PDF extraction to enrich
collected metadata with full content.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from urllib.parse import urlparse

from .web_scraper import RespectfulWebScraper, ScrapedContent
from .pdf_extractor import PDFExtractor, PDFContent
from ..pipeline.data_types import StoredContent

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentResult:
    """Result of content enrichment process."""
    original_content: StoredContent
    enriched_content: str
    enrichment_type: str  # 'web_scrape', 'pdf_extract', 'none', 'error'
    enrichment_successful: bool
    original_length: int
    enriched_length: int
    enhancement_ratio: float  # enriched_length / original_length
    enriched_at: datetime
    enrichment_metadata: Dict[str, Any]
    error_message: Optional[str] = None


class ContentEnricher:
    """
    Content enrichment orchestrator that takes collected metadata
    and enriches it with full content through web scraping and PDF extraction.
    """
    
    def __init__(self,
                 enable_web_scraping: bool = True,
                 enable_pdf_extraction: bool = True,
                 web_scraper_config: Optional[Dict] = None,
                 pdf_extractor_config: Optional[Dict] = None):
        """
        Initialize content enricher.
        
        Args:
            enable_web_scraping: Whether to enable web scraping
            enable_pdf_extraction: Whether to enable PDF extraction
            web_scraper_config: Configuration for web scraper
            pdf_extractor_config: Configuration for PDF extractor
        """
        self.enable_web_scraping = enable_web_scraping
        self.enable_pdf_extraction = enable_pdf_extraction
        
        # Initialize components
        if self.enable_web_scraping:
            scraper_config = web_scraper_config or {}
            self.web_scraper = RespectfulWebScraper(**scraper_config)
        else:
            self.web_scraper = None
        
        if self.enable_pdf_extraction:
            pdf_config = pdf_extractor_config or {}
            self.pdf_extractor = PDFExtractor(**pdf_config)
        else:
            self.pdf_extractor = None
        
        # Enrichment statistics
        self.enrichment_stats = {
            'total_enrichments': 0,
            'successful_enrichments': 0,
            'web_scrapes': 0,
            'pdf_extractions': 0,
            'failed_enrichments': 0,
            'content_enhancement_ratios': []
        }
    
    async def enrich_content(self, content: StoredContent) -> EnrichmentResult:
        """
        Enrich a single piece of content.
        
        Args:
            content: StoredContent to enrich
            
        Returns:
            EnrichmentResult with enriched content or error info
        """
        self.enrichment_stats['total_enrichments'] += 1
        original_length = len(content.content)
        
        try:
            # Determine enrichment strategy based on content type and available URLs
            enrichment_strategy = self._determine_enrichment_strategy(content)
            
            if enrichment_strategy == 'pdf_extract':
                result = await self._enrich_with_pdf_extraction(content)
            elif enrichment_strategy == 'web_scrape':
                result = await self._enrich_with_web_scraping(content)
            else:
                # No enrichment possible or needed
                result = EnrichmentResult(
                    original_content=content,
                    enriched_content=content.content,
                    enrichment_type='none',
                    enrichment_successful=False,
                    original_length=original_length,
                    enriched_length=original_length,
                    enhancement_ratio=1.0,
                    enriched_at=datetime.now(),
                    enrichment_metadata={'reason': 'no_enrichment_strategy'},
                    error_message='No enrichment strategy available'
                )
            
            # Update statistics
            if result.enrichment_successful:
                self.enrichment_stats['successful_enrichments'] += 1
                self.enrichment_stats['content_enhancement_ratios'].append(result.enhancement_ratio)
                
                if result.enrichment_type == 'web_scrape':
                    self.enrichment_stats['web_scrapes'] += 1
                elif result.enrichment_type == 'pdf_extract':
                    self.enrichment_stats['pdf_extractions'] += 1
            else:
                self.enrichment_stats['failed_enrichments'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error enriching content {content.id}: {e}")
            self.enrichment_stats['failed_enrichments'] += 1
            
            return EnrichmentResult(
                original_content=content,
                enriched_content=content.content,
                enrichment_type='error',
                enrichment_successful=False,
                original_length=original_length,
                enriched_length=original_length,
                enhancement_ratio=1.0,
                enriched_at=datetime.now(),
                enrichment_metadata={},
                error_message=str(e)
            )
    
    async def enrich_content_batch(self, 
                                 content_list: List[StoredContent],
                                 max_concurrent: int = 5) -> List[EnrichmentResult]:
        """
        Enrich multiple pieces of content concurrently.
        
        Args:
            content_list: List of StoredContent to enrich
            max_concurrent: Maximum concurrent enrichment operations
            
        Returns:
            List of EnrichmentResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def enrich_with_semaphore(content: StoredContent) -> EnrichmentResult:
            async with semaphore:
                return await self.enrich_content(content)
        
        # Run enrichments concurrently with rate limiting
        tasks = [enrich_with_semaphore(content) for content in content_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        enrichment_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Enrichment task failed: {result}")
                enrichment_results.append(EnrichmentResult(
                    original_content=content_list[i],
                    enriched_content=content_list[i].content,
                    enrichment_type='error',
                    enrichment_successful=False,
                    original_length=len(content_list[i].content),
                    enriched_length=len(content_list[i].content),
                    enhancement_ratio=1.0,
                    enriched_at=datetime.now(),
                    enrichment_metadata={},
                    error_message=str(result)
                ))
            else:
                enrichment_results.append(result)
        
        return enrichment_results
    
    def _determine_enrichment_strategy(self, content: StoredContent) -> str:
        """Determine the best enrichment strategy for the content."""
        
        # Check for PDF URLs (ArXiv papers, etc.)
        pdf_url = self._extract_pdf_url(content)
        if pdf_url and self.enable_pdf_extraction:
            return 'pdf_extract'
        
        # Check for web URLs that can be scraped
        web_url = self._extract_web_url(content)
        if web_url and self.enable_web_scraping:
            return 'web_scrape'
        
        # No enrichment strategy available
        return 'none'
    
    def _extract_pdf_url(self, content: StoredContent) -> Optional[str]:
        """Extract PDF URL from content if available."""
        
        # Check raw_data for PDF URL
        if 'pdf_url' in content.raw_data:
            return content.raw_data['pdf_url']
        
        # Check if main URL is a PDF
        if content.url and content.url.lower().endswith('.pdf'):
            return content.url
        
        # Check for ArXiv PDF pattern
        if content.url and 'arxiv.org/abs/' in content.url:
            # Convert arxiv.org/abs/XXXX to arxiv.org/pdf/XXXX.pdf
            pdf_url = content.url.replace('/abs/', '/pdf/') + '.pdf'
            return pdf_url
        
        return None
    
    def _extract_web_url(self, content: StoredContent) -> Optional[str]:
        """Extract web URL suitable for scraping."""
        
        # Use the main URL if it's not a PDF
        if content.url and not content.url.lower().endswith('.pdf'):
            parsed = urlparse(content.url)
            
            # Skip certain domains that are not useful to scrape
            skip_domains = ['github.com', 'twitter.com', 'reddit.com']
            if any(domain in parsed.netloc.lower() for domain in skip_domains):
                return None
            
            return content.url
        
        return None
    
    async def _enrich_with_pdf_extraction(self, content: StoredContent) -> EnrichmentResult:
        """Enrich content using PDF extraction."""
        
        pdf_url = self._extract_pdf_url(content)
        if not pdf_url or not self.pdf_extractor:
            return EnrichmentResult(
                original_content=content,
                enriched_content=content.content,
                enrichment_type='pdf_extract',
                enrichment_successful=False,
                original_length=len(content.content),
                enriched_length=len(content.content),
                enhancement_ratio=1.0,
                enriched_at=datetime.now(),
                enrichment_metadata={},
                error_message='PDF extraction not available'
            )
        
        logger.info(f"Extracting PDF content from: {pdf_url}")
        pdf_result = await self.pdf_extractor.extract_from_url(pdf_url, content.title)
        
        if pdf_result.success and pdf_result.full_text:
            enriched_content = pdf_result.full_text
            enhancement_ratio = len(enriched_content) / len(content.content) if content.content else float('inf')
            
            return EnrichmentResult(
                original_content=content,
                enriched_content=enriched_content,
                enrichment_type='pdf_extract',
                enrichment_successful=True,
                original_length=len(content.content),
                enriched_length=len(enriched_content),
                enhancement_ratio=enhancement_ratio,
                enriched_at=datetime.now(),
                enrichment_metadata={
                    'pdf_url': pdf_url,
                    'page_count': pdf_result.page_count,
                    'extraction_method': pdf_result.extraction_method,
                    'word_count': pdf_result.word_count
                }
            )
        else:
            return EnrichmentResult(
                original_content=content,
                enriched_content=content.content,
                enrichment_type='pdf_extract',
                enrichment_successful=False,
                original_length=len(content.content),
                enriched_length=len(content.content),
                enhancement_ratio=1.0,
                enriched_at=datetime.now(),
                enrichment_metadata={'pdf_url': pdf_url},
                error_message=pdf_result.error_message
            )
    
    async def _enrich_with_web_scraping(self, content: StoredContent) -> EnrichmentResult:
        """Enrich content using web scraping."""
        
        web_url = self._extract_web_url(content)
        if not web_url or not self.web_scraper:
            return EnrichmentResult(
                original_content=content,
                enriched_content=content.content,
                enrichment_type='web_scrape',
                enrichment_successful=False,
                original_length=len(content.content),
                enriched_length=len(content.content),
                enhancement_ratio=1.0,
                enriched_at=datetime.now(),
                enrichment_metadata={},
                error_message='Web scraping not available'
            )
        
        logger.info(f"Scraping web content from: {web_url}")
        scrape_result = await self.web_scraper.scrape_url(web_url)
        
        if scrape_result.success and scrape_result.clean_text:
            # Combine original content with scraped content
            enriched_content = f"{content.content}\n\n--- Full Article Content ---\n\n{scrape_result.clean_text}"
            enhancement_ratio = len(enriched_content) / len(content.content) if content.content else float('inf')
            
            return EnrichmentResult(
                original_content=content,
                enriched_content=enriched_content,
                enrichment_type='web_scrape',
                enrichment_successful=True,
                original_length=len(content.content),
                enriched_length=len(enriched_content),
                enhancement_ratio=enhancement_ratio,
                enriched_at=datetime.now(),
                enrichment_metadata={
                    'scraped_url': web_url,
                    'scraped_title': scrape_result.title,
                    'scraped_author': scrape_result.author,
                    'word_count': scrape_result.word_count,
                    'content_type': scrape_result.content_type
                }
            )
        else:
            return EnrichmentResult(
                original_content=content,
                enriched_content=content.content,
                enrichment_type='web_scrape',
                enrichment_successful=False,
                original_length=len(content.content),
                enriched_length=len(content.content),
                enhancement_ratio=1.0,
                enriched_at=datetime.now(),
                enrichment_metadata={'scraped_url': web_url},
                error_message=scrape_result.error_message
            )
    
    def get_enrichment_statistics(self) -> Dict[str, Any]:
        """Get enrichment statistics."""
        stats = self.enrichment_stats.copy()
        
        # Calculate average enhancement ratio
        if stats['content_enhancement_ratios']:
            stats['average_enhancement_ratio'] = sum(stats['content_enhancement_ratios']) / len(stats['content_enhancement_ratios'])
            stats['max_enhancement_ratio'] = max(stats['content_enhancement_ratios'])
        else:
            stats['average_enhancement_ratio'] = 1.0
            stats['max_enhancement_ratio'] = 1.0
        
        # Calculate success rate
        if stats['total_enrichments'] > 0:
            stats['success_rate'] = stats['successful_enrichments'] / stats['total_enrichments']
        else:
            stats['success_rate'] = 0.0
        
        # Add component availability
        stats['web_scraping_enabled'] = self.enable_web_scraping
        stats['pdf_extraction_enabled'] = self.enable_pdf_extraction
        
        if self.pdf_extractor:
            stats['pdf_extraction_stats'] = self.pdf_extractor.get_extraction_stats()
        
        if self.web_scraper:
            stats['web_scraping_stats'] = self.web_scraper.get_domain_stats()
        
        return stats