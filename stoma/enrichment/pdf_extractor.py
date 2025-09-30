"""
PDF content extraction for academic papers and documents.

This module provides PDF text extraction capabilities for ArXiv papers
and other PDF documents in the pipeline.
"""

import asyncio
import logging
import tempfile
import os
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class PDFContent:
    """Represents extracted PDF content."""
    url: str
    title: Optional[str]
    full_text: str
    page_count: int
    extracted_at: datetime
    extraction_method: str
    word_count: int
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class PDFExtractor:
    """
    PDF content extractor with multiple extraction methods.
    
    Uses Apache Tika as primary method with fallbacks to other libraries.
    """
    
    def __init__(self, 
                 max_file_size: int = 50 * 1024 * 1024,  # 50MB
                 timeout: int = 60,
                 user_agent: str = "Stoma/1.0 Research Intelligence Bot"):
        """
        Initialize PDF extractor.
        
        Args:
            max_file_size: Maximum PDF file size to process (bytes)
            timeout: Processing timeout in seconds
            user_agent: User-Agent for PDF downloads
        """
        self.max_file_size = max_file_size
        self.timeout = timeout
        self.user_agent = user_agent
        
        # Check available extraction methods
        self.available_methods = self._check_available_methods()
        logger.info(f"PDF extraction methods available: {self.available_methods}")
    
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which PDF extraction methods are available."""
        methods = {}
        
        # Check for Apache Tika
        try:
            from tika import parser
            methods['tika'] = True
        except ImportError:
            methods['tika'] = False
        
        # Check for PyPDF2
        try:
            import PyPDF2
            methods['pypdf2'] = True
        except ImportError:
            methods['pypdf2'] = False
        
        # Check for pdfplumber
        try:
            import pdfplumber
            methods['pdfplumber'] = True
        except ImportError:
            methods['pdfplumber'] = False
        
        # Check for fitz (PyMuPDF)
        try:
            import fitz
            methods['pymupdf'] = True
        except ImportError:
            methods['pymupdf'] = False
        
        return methods
    
    async def extract_from_url(self, pdf_url: str, title: Optional[str] = None) -> PDFContent:
        """
        Extract text content from a PDF URL.
        
        Args:
            pdf_url: URL to the PDF file
            title: Optional title for the document
            
        Returns:
            PDFContent with extracted text or error information
        """
        try:
            # Download PDF to temporary file
            temp_file_path = await self._download_pdf(pdf_url)
            
            if not temp_file_path:
                return PDFContent(
                    url=pdf_url, title=title, full_text="", page_count=0,
                    extracted_at=datetime.now(), extraction_method="none",
                    word_count=0, success=False,
                    error_message="Failed to download PDF"
                )
            
            try:
                # Extract content using best available method
                return await self._extract_from_file(temp_file_path, pdf_url, title)
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error extracting PDF from {pdf_url}: {e}")
            return PDFContent(
                url=pdf_url, title=title, full_text="", page_count=0,
                extracted_at=datetime.now(), extraction_method="error",
                word_count=0, success=False,
                error_message=str(e)
            )
    
    async def _download_pdf(self, pdf_url: str) -> Optional[str]:
        """Download PDF to temporary file."""
        try:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/pdf,application/octet-stream,*/*'
            }
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(pdf_url) as response:
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if 'pdf' not in content_type and 'octet-stream' not in content_type:
                        logger.warning(f"Unexpected content type for PDF: {content_type}")
                    
                    # Check file size
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_file_size:
                        logger.warning(f"PDF too large: {content_length} bytes")
                        return None
                    
                    # Create temporary file
                    temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
                    
                    try:
                        # Download in chunks
                        total_size = 0
                        async with aiofiles.open(temp_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                total_size += len(chunk)
                                if total_size > self.max_file_size:
                                    logger.warning(f"PDF download exceeded size limit")
                                    return None
                                await f.write(chunk)
                        
                        logger.debug(f"Downloaded PDF: {total_size} bytes to {temp_path}")
                        return temp_path
                        
                    finally:
                        os.close(temp_fd)
                        
        except Exception as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {e}")
            return None
    
    async def _extract_from_file(self, file_path: str, pdf_url: str, title: Optional[str]) -> PDFContent:
        """Extract text from a local PDF file using best available method."""
        
        # Try extraction methods in order of preference
        extraction_methods = [
            ('tika', self._extract_with_tika),
            ('pymupdf', self._extract_with_pymupdf),
            ('pdfplumber', self._extract_with_pdfplumber),
            ('pypdf2', self._extract_with_pypdf2)
        ]
        
        for method_name, extract_func in extraction_methods:
            if self.available_methods.get(method_name, False):
                try:
                    logger.debug(f"Trying PDF extraction with {method_name}")
                    result = await extract_func(file_path, pdf_url, title)
                    if result.success and len(result.full_text.strip()) > 100:
                        logger.info(f"Successfully extracted PDF with {method_name}: {result.word_count} words")
                        return result
                except Exception as e:
                    logger.warning(f"PDF extraction failed with {method_name}: {e}")
                    continue
        
        # If all methods failed
        return PDFContent(
            url=pdf_url, title=title, full_text="", page_count=0,
            extracted_at=datetime.now(), extraction_method="failed",
            word_count=0, success=False,
            error_message="All extraction methods failed"
        )
    
    async def _extract_with_tika(self, file_path: str, pdf_url: str, title: Optional[str]) -> PDFContent:
        """Extract PDF content using Apache Tika."""
        from tika import parser
        
        # Run Tika parsing in executor to avoid blocking
        loop = asyncio.get_event_loop()
        parsed = await loop.run_in_executor(None, parser.from_file, file_path)
        
        if not parsed or 'content' not in parsed:
            raise Exception("Tika failed to extract content")
        
        content = parsed['content'] or ""
        metadata = parsed.get('metadata', {})
        
        # Extract metadata
        extracted_title = title or metadata.get('title') or metadata.get('dc:title')
        page_count = self._extract_page_count(metadata)
        
        word_count = len(content.split()) if content else 0
        
        return PDFContent(
            url=pdf_url,
            title=extracted_title,
            full_text=content.strip(),
            page_count=page_count,
            extracted_at=datetime.now(),
            extraction_method="tika",
            word_count=word_count,
            metadata=metadata
        )
    
    async def _extract_with_pymupdf(self, file_path: str, pdf_url: str, title: Optional[str]) -> PDFContent:
        """Extract PDF content using PyMuPDF (fitz)."""
        import fitz
        
        loop = asyncio.get_event_loop()
        
        def extract():
            doc = fitz.open(file_path)
            full_text = ""
            
            for page in doc:
                full_text += page.get_text()
            
            metadata = doc.metadata
            page_count = len(doc)
            doc.close()
            
            return full_text, metadata, page_count
        
        full_text, metadata, page_count = await loop.run_in_executor(None, extract)
        
        extracted_title = title or metadata.get('title')
        word_count = len(full_text.split()) if full_text else 0
        
        return PDFContent(
            url=pdf_url,
            title=extracted_title,
            full_text=full_text.strip(),
            page_count=page_count,
            extracted_at=datetime.now(),
            extraction_method="pymupdf",
            word_count=word_count,
            metadata=metadata
        )
    
    async def _extract_with_pdfplumber(self, file_path: str, pdf_url: str, title: Optional[str]) -> PDFContent:
        """Extract PDF content using pdfplumber."""
        import pdfplumber
        
        loop = asyncio.get_event_loop()
        
        def extract():
            full_text = ""
            page_count = 0
            
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
            
            return full_text, page_count
        
        full_text, page_count = await loop.run_in_executor(None, extract)
        
        word_count = len(full_text.split()) if full_text else 0
        
        return PDFContent(
            url=pdf_url,
            title=title,
            full_text=full_text.strip(),
            page_count=page_count,
            extracted_at=datetime.now(),
            extraction_method="pdfplumber",
            word_count=word_count
        )
    
    async def _extract_with_pypdf2(self, file_path: str, pdf_url: str, title: Optional[str]) -> PDFContent:
        """Extract PDF content using PyPDF2."""
        import PyPDF2
        
        loop = asyncio.get_event_loop()
        
        def extract():
            full_text = ""
            page_count = 0
            
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page_count = len(reader.pages)
                
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
            
            return full_text, page_count
        
        full_text, page_count = await loop.run_in_executor(None, extract)
        
        word_count = len(full_text.split()) if full_text else 0
        
        return PDFContent(
            url=pdf_url,
            title=title,
            full_text=full_text.strip(),
            page_count=page_count,
            extracted_at=datetime.now(),
            extraction_method="pypdf2",
            word_count=word_count
        )
    
    def _extract_page_count(self, metadata: Dict) -> int:
        """Extract page count from PDF metadata."""
        page_fields = ['xmpTPg:NPages', 'meta:page-count', 'Page-Count']
        
        for field in page_fields:
            if field in metadata:
                try:
                    return int(metadata[field])
                except (ValueError, TypeError):
                    continue
        
        return 0
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about available extraction methods."""
        return {
            'available_methods': self.available_methods,
            'primary_method': next((method for method, available in 
                                   [('tika', self.available_methods.get('tika', False)),
                                    ('pymupdf', self.available_methods.get('pymupdf', False)),
                                    ('pdfplumber', self.available_methods.get('pdfplumber', False)),
                                    ('pypdf2', self.available_methods.get('pypdf2', False))]
                                   if available), 'none'),
            'max_file_size_mb': self.max_file_size / (1024 * 1024),
            'timeout_seconds': self.timeout
        }