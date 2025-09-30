"""
Respectful web scraper that follows robots.txt and web standards.

This module provides ethical web scraping capabilities for content enrichment.
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Set, List
from urllib.parse import urlparse, urljoin, urlunparse
from urllib.robotparser import RobotFileParser
import hashlib

import aiohttp
from bs4 import BeautifulSoup, Comment
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScrapedContent:
    """Represents successfully scraped web content."""
    url: str
    title: Optional[str]
    content: str
    clean_text: str
    author: Optional[str]
    published_date: Optional[datetime]
    meta_description: Optional[str]
    scraped_at: datetime
    content_type: str
    word_count: int
    success: bool = True
    error_message: Optional[str] = None


class RobotCache:
    """Cache for robots.txt files to avoid repeated requests."""
    
    def __init__(self, cache_duration_hours: int = 24):
        self.cache = {}
        self.cache_duration = timedelta(hours=cache_duration_hours)
    
    def get_robots_parser(self, base_url: str) -> Optional[RobotFileParser]:
        """Get cached robots.txt parser or None if not cached/expired."""
        if base_url in self.cache:
            parser, cached_at = self.cache[base_url]
            if datetime.now() - cached_at < self.cache_duration:
                return parser
        return None
    
    def cache_robots_parser(self, base_url: str, parser: RobotFileParser):
        """Cache a robots.txt parser."""
        self.cache[base_url] = (parser, datetime.now())


class RespectfulWebScraper:
    """
    Web scraper that respects robots.txt, rate limits, and web standards.
    
    Features:
    - Robots.txt compliance checking
    - Configurable rate limiting per domain
    - User-Agent identification
    - Content extraction with fallbacks
    - Caching to avoid repeated requests
    """
    
    def __init__(self, 
                 user_agent: str = "Stoma/1.0 (+https://github.com/stoma/bot) Research Intelligence Bot",
                 rate_limit_delay: float = 1.0,
                 max_content_size: int = 10 * 1024 * 1024,  # 10MB
                 timeout: int = 30):
        """
        Initialize the respectful web scraper.
        
        Args:
            user_agent: User-Agent string to identify our bot
            rate_limit_delay: Minimum delay between requests to same domain (seconds)
            max_content_size: Maximum content size to download (bytes)
            timeout: Request timeout in seconds
        """
        self.user_agent = user_agent
        self.rate_limit_delay = rate_limit_delay
        self.max_content_size = max_content_size
        self.timeout = timeout
        
        # Domain-specific tracking
        self.last_request_time = {}  # domain -> last request time
        self.robots_cache = RobotCache()
        self.failed_domains = set()  # Domains that consistently fail
        
        # Content extraction patterns
        self.content_selectors = [
            'article',
            '[role="main"]',
            '.post-content',
            '.article-content', 
            '.content',
            '.entry-content',
            '.post-body',
            'main',
            '#content',
            '.container'
        ]
        
        # Elements to remove (noise)
        self.noise_selectors = [
            'nav', 'header', 'footer', 'aside',
            '.nav', '.navigation', '.menu',
            '.sidebar', '.ads', '.advertisement',
            '.social-share', '.comments', '.related-posts',
            'script', 'style', 'noscript'
        ]
    
    async def scrape_url(self, url: str) -> ScrapedContent:
        """
        Scrape content from a URL with full compliance checks.
        
        Args:
            url: URL to scrape
            
        Returns:
            ScrapedContent object with extracted content or error info
        """
        try:
            # Parse URL and validate
            parsed = urlparse(url)
            if not parsed.netloc:
                return ScrapedContent(
                    url=url, title=None, content="", clean_text="",
                    author=None, published_date=None, meta_description=None,
                    scraped_at=datetime.now(), content_type="error",
                    word_count=0, success=False,
                    error_message="Invalid URL format"
                )
            
            domain = parsed.netloc.lower()
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            
            # Check if domain is blocked
            if domain in self.failed_domains:
                return ScrapedContent(
                    url=url, title=None, content="", clean_text="",
                    author=None, published_date=None, meta_description=None,
                    scraped_at=datetime.now(), content_type="blocked",
                    word_count=0, success=False,
                    error_message="Domain previously failed multiple times"
                )
            
            # Check robots.txt compliance
            if not await self._can_fetch(base_url, url):
                return ScrapedContent(
                    url=url, title=None, content="", clean_text="",
                    author=None, published_date=None, meta_description=None,
                    scraped_at=datetime.now(), content_type="blocked",
                    word_count=0, success=False,
                    error_message="Blocked by robots.txt"
                )
            
            # Rate limiting
            await self._respect_rate_limit(domain)
            
            # Perform the actual scraping
            return await self._scrape_content(url)
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ScrapedContent(
                url=url, title=None, content="", clean_text="",
                author=None, published_date=None, meta_description=None,
                scraped_at=datetime.now(), content_type="error",
                word_count=0, success=False,
                error_message=str(e)
            )
    
    async def _can_fetch(self, base_url: str, url: str) -> bool:
        """Check if we're allowed to fetch the URL according to robots.txt."""
        try:
            # Check cache first
            parser = self.robots_cache.get_robots_parser(base_url)
            
            if parser is None:
                # Fetch and parse robots.txt
                robots_url = urljoin(base_url, '/robots.txt')
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    try:
                        async with session.get(robots_url, headers={'User-Agent': self.user_agent}) as response:
                            if response.status == 200:
                                robots_content = await response.text()
                                parser = RobotFileParser()
                                parser.set_url(robots_url)
                                # Parse the content line by line
                                for line in robots_content.split('\n'):
                                    parser.read_text(line)
                                self.robots_cache.cache_robots_parser(base_url, parser)
                            else:
                                # If robots.txt doesn't exist or fails, assume allowed
                                return True
                    except:
                        # If robots.txt request fails, assume allowed
                        return True
            
            # Check if our user agent can fetch the URL
            return parser.can_fetch(self.user_agent, url)
            
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {base_url}: {e}")
            return True  # Assume allowed if check fails
    
    async def _respect_rate_limit(self, domain: str):
        """Enforce rate limiting per domain."""
        if domain in self.last_request_time:
            time_since_last = time.time() - self.last_request_time[domain]
            if time_since_last < self.rate_limit_delay:
                delay = self.rate_limit_delay - time_since_last
                logger.debug(f"Rate limiting: waiting {delay:.2f}s for {domain}")
                await asyncio.sleep(delay)
        
        self.last_request_time[domain] = time.time()
    
    async def _scrape_content(self, url: str) -> ScrapedContent:
        """Perform the actual content scraping."""
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=headers
        ) as session:
            
            try:
                async with session.get(url) as response:
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if not ('text/html' in content_type or 'application/xhtml' in content_type):
                        return ScrapedContent(
                            url=url, title=None, content="", clean_text="",
                            author=None, published_date=None, meta_description=None,
                            scraped_at=datetime.now(), content_type=content_type,
                            word_count=0, success=False,
                            error_message=f"Unsupported content type: {content_type}"
                        )
                    
                    # Check content size
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_content_size:
                        return ScrapedContent(
                            url=url, title=None, content="", clean_text="",
                            author=None, published_date=None, meta_description=None,
                            scraped_at=datetime.now(), content_type="too_large",
                            word_count=0, success=False,
                            error_message=f"Content too large: {content_length} bytes"
                        )
                    
                    # Get the HTML content
                    html_content = await response.text()
                    
                    # Parse and extract content
                    return self._extract_content(url, html_content)
                    
            except aiohttp.ClientError as e:
                return ScrapedContent(
                    url=url, title=None, content="", clean_text="",
                    author=None, published_date=None, meta_description=None,
                    scraped_at=datetime.now(), content_type="error",
                    word_count=0, success=False,
                    error_message=f"Request failed: {str(e)}"
                )
    
    def _extract_content(self, url: str, html_content: str) -> ScrapedContent:
        """Extract meaningful content from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata
            title = self._extract_title(soup)
            author = self._extract_author(soup)
            published_date = self._extract_published_date(soup)
            meta_description = self._extract_meta_description(soup)
            
            # Remove noise elements
            for selector in self.noise_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            clean_text = self._clean_text(main_content)
            
            word_count = len(clean_text.split()) if clean_text else 0
            
            return ScrapedContent(
                url=url,
                title=title,
                content=main_content,
                clean_text=clean_text,
                author=author,
                published_date=published_date,
                meta_description=meta_description,
                scraped_at=datetime.now(),
                content_type="article",
                word_count=word_count,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return ScrapedContent(
                url=url, title=None, content="", clean_text="",
                author=None, published_date=None, meta_description=None,
                scraped_at=datetime.now(), content_type="error",
                word_count=0, success=False,
                error_message=f"Content extraction failed: {str(e)}"
            )
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title."""
        # Try various title sources in order of preference
        title_selectors = [
            'h1',
            'title',
            '.title',
            '.headline',
            '[property="og:title"]',
            '[name="twitter:title"]'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text().strip() if selector not in ['[property="og:title"]', '[name="twitter:title"]'] else element.get('content', '').strip()
                if title and len(title) > 5:  # Reasonable title length
                    return title
        
        return None
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author information."""
        author_selectors = [
            '[rel="author"]',
            '.author',
            '.byline',
            '[property="article:author"]',
            '[name="author"]',
            '.writer',
            '.journalist'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                author = element.get_text().strip() if selector not in ['[property="article:author"]', '[name="author"]'] else element.get('content', '').strip()
                if author:
                    return author
        
        return None
    
    def _extract_published_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract publication date."""
        date_selectors = [
            '[property="article:published_time"]',
            '[property="datePublished"]',
            'time[datetime]',
            '.published',
            '.date',
            '.timestamp'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_str = element.get('datetime') or element.get('content') or element.get_text().strip()
                if date_str:
                    try:
                        # Try various date formats
                        for fmt in ['%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S']:
                            try:
                                return datetime.strptime(date_str, fmt)
                            except ValueError:
                                continue
                    except:
                        continue
        
        return None
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta description."""
        desc_selectors = [
            '[name="description"]',
            '[property="og:description"]',
            '[name="twitter:description"]'
        ]
        
        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                description = element.get('content', '').strip()
                if description:
                    return description
        
        return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content using content selectors."""
        # Try content selectors in order of preference
        for selector in self.content_selectors:
            elements = soup.select(selector)
            if elements:
                # Get the largest content block
                best_element = max(elements, key=lambda x: len(x.get_text()))
                if len(best_element.get_text().strip()) > 100:  # Minimum content length
                    return best_element.get_text()
        
        # Fallback: get body content
        body = soup.find('body')
        if body:
            return body.get_text()
        
        # Last resort: entire document
        return soup.get_text()
    
    def _clean_text(self, content: str) -> str:
        """Clean and normalize extracted text."""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common web artifacts
        content = re.sub(r'(Cookie|Privacy Policy|Terms of Service|Subscribe|Newsletter)', '', content, flags=re.IGNORECASE)
        
        # Remove navigation artifacts
        content = re.sub(r'(Home|About|Contact|Menu|Search)', '', content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def get_domain_stats(self) -> Dict[str, Dict]:
        """Get statistics about scraping by domain."""
        stats = {
            'total_domains_accessed': len(self.last_request_time),
            'failed_domains': len(self.failed_domains),
            'robots_cache_size': len(self.robots_cache.cache),
            'failed_domain_list': list(self.failed_domains)
        }
        return stats