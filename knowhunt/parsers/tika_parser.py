"""Apache Tika parser for universal document processing."""

import requests
import asyncio
import aiohttp
from typing import Dict, Any, Optional
from pathlib import Path


class TikaParser:
    """Universal document parser using Apache Tika."""
    
    def __init__(self, tika_url: str = "http://localhost:9998"):
        self.tika_url = tika_url
    
    async def parse_document(self, file_path: str) -> str:
        """Extract text from any document format."""
        try:
            with open(file_path, 'rb') as f:
                async with aiohttp.ClientSession() as session:
                    async with session.put(
                        f"{self.tika_url}/tika",
                        data=f,
                        headers={"Accept": "text/plain"}
                    ) as response:
                        if response.status == 200:
                            return await response.text()
                        else:
                            raise Exception(f"Tika parsing failed: {response.status}")
        except Exception as e:
            raise Exception(f"Failed to parse document {file_path}: {e}")
    
    async def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document."""
        try:
            with open(file_path, 'rb') as f:
                async with aiohttp.ClientSession() as session:
                    async with session.put(
                        f"{self.tika_url}/meta",
                        data=f,
                        headers={"Accept": "application/json"}
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            raise Exception(f"Tika metadata extraction failed: {response.status}")
        except Exception as e:
            raise Exception(f"Failed to extract metadata from {file_path}: {e}")
    
    def parse_document_sync(self, file_path: str) -> str:
        """Synchronous version for simple use cases."""
        try:
            with open(file_path, 'rb') as f:
                response = requests.put(
                    f"{self.tika_url}/tika",
                    data=f,
                    headers={"Accept": "text/plain"}
                )
                if response.status_code == 200:
                    return response.text
                else:
                    raise Exception(f"Tika parsing failed: {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to parse document {file_path}: {e}")
    
    def get_metadata_sync(self, file_path: str) -> Dict[str, Any]:
        """Synchronous version for metadata extraction."""
        try:
            with open(file_path, 'rb') as f:
                response = requests.put(
                    f"{self.tika_url}/meta",
                    data=f,
                    headers={"Accept": "application/json"}
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"Tika metadata extraction failed: {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to extract metadata from {file_path}: {e}")
    
    async def health_check(self) -> bool:
        """Check if Tika server is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.tika_url}/version") as response:
                    return response.status == 200
        except:
            return False
    
    def health_check_sync(self) -> bool:
        """Synchronous health check."""
        try:
            response = requests.get(f"{self.tika_url}/version")
            return response.status_code == 200
        except:
            return False