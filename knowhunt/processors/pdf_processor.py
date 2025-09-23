"""PDF download and processing using Apache Tika."""

import aiohttp
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ..parsers.tika_parser import TikaParser


class PDFProcessor:
    """Downloads and processes PDFs with Tika."""
    
    def __init__(self, download_dir: str = "./data/pdfs", tika_url: str = "http://localhost:9998"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.tika_parser = TikaParser(tika_url)
    
    async def download_and_process_pdf(self, pdf_url: str, paper_id: str) -> Dict[str, Any]:
        """Download PDF and extract text/metadata."""
        try:
            # Download PDF
            pdf_path = await self._download_pdf(pdf_url, paper_id)
            if not pdf_path:
                return {"error": "Failed to download PDF"}
            
            # Process with Tika
            try:
                text_content = await self.tika_parser.parse_document(str(pdf_path))
                metadata = await self.tika_parser.get_metadata(str(pdf_path))
                
                return {
                    "success": True,
                    "pdf_path": str(pdf_path),
                    "text_content": text_content,
                    "metadata": metadata,
                    "processed_at": datetime.now().isoformat()
                }
            except Exception as tika_error:
                return {
                    "success": False,
                    "pdf_path": str(pdf_path),
                    "error": f"Tika processing failed: {tika_error}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"PDF processing failed: {e}"
            }
    
    async def _download_pdf(self, pdf_url: str, paper_id: str) -> Optional[Path]:
        """Download PDF to local storage."""
        try:
            # Sanitize paper_id for filename
            safe_id = "".join(c for c in paper_id if c.isalnum() or c in ('-', '_'))
            pdf_path = self.download_dir / f"{safe_id}.pdf"
            
            # Skip if already exists
            if pdf_path.exists():
                return pdf_path
            
            # Download
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        with open(pdf_path, 'wb') as f:
                            f.write(content)
                        return pdf_path
                    else:
                        print(f"Failed to download PDF: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            print(f"Error downloading PDF {pdf_url}: {e}")
            return None
    
    def get_pdf_path(self, paper_id: str) -> Path:
        """Get the expected path for a paper's PDF."""
        safe_id = "".join(c for c in paper_id if c.isalnum() or c in ('-', '_'))
        return self.download_dir / f"{safe_id}.pdf"
    
    def pdf_exists(self, paper_id: str) -> bool:
        """Check if PDF already downloaded."""
        return self.get_pdf_path(paper_id).exists()
    
    async def process_existing_pdf(self, paper_id: str) -> Dict[str, Any]:
        """Process an already downloaded PDF."""
        pdf_path = self.get_pdf_path(paper_id)
        if not pdf_path.exists():
            return {"error": "PDF not found"}
        
        try:
            text_content = await self.tika_parser.parse_document(str(pdf_path))
            metadata = await self.tika_parser.get_metadata(str(pdf_path))
            
            return {
                "success": True,
                "pdf_path": str(pdf_path),
                "text_content": text_content,
                "metadata": metadata,
                "processed_at": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "pdf_path": str(pdf_path),
                "error": f"Processing failed: {e}"
            }