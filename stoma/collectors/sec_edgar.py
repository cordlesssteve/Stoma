"""SEC EDGAR collector for corporate filings and financial data."""

import asyncio
import re
from datetime import datetime, timedelta
from typing import AsyncIterator, Dict, Any, Optional, List
import aiohttp
from bs4 import BeautifulSoup

from .base import APICollector, CollectionResult, SourceType


class SECEdgarCollector(APICollector):
    """Collector for SEC EDGAR filings and corporate data."""
    
    def __init__(self, config: Dict[str, Any]):
        if "base_url" not in config:
            config["base_url"] = "https://www.sec.gov"
        
        super().__init__(config)
        
        # SEC specific configuration
        self.edgar_api_url = "https://www.sec.gov/edgar"
        self.company_search_url = f"{self.base_url}/cgi-bin/browse-edgar"
        self.filing_search_url = f"{self.base_url}/Archives/edgar/daily-index"
        
        # SEC requires specific User-Agent
        self.headers["User-Agent"] = config.get(
            "user_agent", 
            "Stoma Research Intelligence System admin@stoma.com"
        )
        
        # Supported filing types
        self.filing_types = {
            "10-K": "Annual Report",
            "10-Q": "Quarterly Report", 
            "8-K": "Current Report",
            "10-K/A": "Annual Report Amendment",
            "10-Q/A": "Quarterly Report Amendment",
            "8-K/A": "Current Report Amendment",
            "DEF 14A": "Proxy Statement",
            "S-1": "Registration Statement",
            "S-3": "Registration Statement",
            "4": "Insider Trading Report",
            "3": "Initial Statement of Ownership",
            "5": "Annual Statement of Changes in Ownership"
        }
    
    def _get_source_type(self) -> SourceType:
        return SourceType.PUBLIC_DOCS
    
    async def collect(self, **kwargs) -> AsyncIterator[CollectionResult]:
        """Default collect method - delegates to recent filings."""
        async for result in self.collect_recent_filings(**kwargs):
            yield result
    
    async def collect_recent_filings(self, 
                                   filing_type: str = None,
                                   days_back: int = 1,
                                   **kwargs) -> AsyncIterator[CollectionResult]:
        """
        Collect recent SEC filings.
        
        Args:
            filing_type: Specific filing type (e.g., "10-K", "8-K")
            days_back: Number of days back to search
        """
        
        # Get the target date
        target_date = datetime.now() - timedelta(days=days_back)
        date_str = target_date.strftime("%Y%m%d")
        
        # Build the daily index URL
        index_url = f"{self.filing_search_url}/{target_date.year}/QTR{(target_date.month-1)//3 + 1}/master.{date_str}.idx"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(index_url, headers=self.headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        async for result in self._parse_daily_index(content, filing_type):
                            yield result
                            # Rate limiting for SEC
                            await asyncio.sleep(0.1)
                    else:
                        yield self._create_error_result(
                            f"sec_daily_index_error_{date_str}",
                            f"Failed to fetch daily index: HTTP {response.status}"
                        )
                        
            except Exception as e:
                yield self._create_error_result(
                    f"sec_daily_index_exception_{date_str}",
                    f"Exception while collecting daily index: {e}"
                )
    
    async def collect_company_filings(self,
                                    company_ticker: str = None,
                                    cik: str = None,
                                    filing_type: str = None,
                                    count: int = 10) -> AsyncIterator[CollectionResult]:
        """
        Collect filings for a specific company.
        
        Args:
            company_ticker: Company stock ticker symbol
            cik: Central Index Key (CIK) number
            filing_type: Specific filing type filter
            count: Number of filings to retrieve
        """
        
        if not company_ticker and not cik:
            yield self._create_error_result(
                "sec_company_missing_identifier",
                "Either company_ticker or cik must be provided"
            )
            return
        
        # Convert ticker to CIK if needed
        if company_ticker and not cik:
            cik = await self._get_cik_from_ticker(company_ticker)
            if not cik:
                yield self._create_error_result(
                    f"sec_ticker_not_found_{company_ticker}",
                    f"Could not find CIK for ticker: {company_ticker}"
                )
                return
        
        # Build search URL
        params = {
            "action": "getcompany",
            "CIK": cik,
            "type": filing_type or "",
            "dateb": "",
            "owner": "exclude",
            "count": count,
            "output": "atom"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    self.company_search_url,
                    headers=self.headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        async for result in self._parse_company_filings_atom(content, company_ticker or cik):
                            yield result
                    else:
                        yield self._create_error_result(
                            f"sec_company_filings_error_{cik}",
                            f"Failed to fetch company filings: HTTP {response.status}"
                        )
                        
            except Exception as e:
                yield self._create_error_result(
                    f"sec_company_filings_exception_{cik}",
                    f"Exception while collecting company filings: {e}"
                )
    
    async def collect_insider_trading(self,
                                    company_ticker: str = None,
                                    cik: str = None,
                                    days_back: int = 7) -> AsyncIterator[CollectionResult]:
        """Collect insider trading reports (Forms 3, 4, 5)."""
        
        insider_forms = ["3", "4", "5"]
        
        for form_type in insider_forms:
            async for result in self.collect_company_filings(
                company_ticker=company_ticker,
                cik=cik,
                filing_type=form_type,
                count=20
            ):
                # Filter by date
                if result.success:
                    filing_date = result.data.get("filing_date")
                    if filing_date:
                        try:
                            filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
                            cutoff_date = datetime.now() - timedelta(days=days_back)
                            if filing_dt >= cutoff_date:
                                yield result
                        except ValueError:
                            # If date parsing fails, include the result anyway
                            yield result
                else:
                    yield result
    
    async def _parse_daily_index(self, 
                                content: str, 
                                filing_type_filter: str = None) -> AsyncIterator[CollectionResult]:
        """Parse SEC daily index file."""
        
        lines = content.split('\n')
        
        # Skip header lines (usually first 10-11 lines)
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('CIK|Company Name|Form Type|Date Filed|Filename'):
                data_start = i + 1
                break
        
        for line in lines[data_start:]:
            if not line.strip():
                continue
                
            try:
                parts = line.split('|')
                if len(parts) >= 5:
                    cik, company_name, form_type, date_filed, filename = parts[:5]
                    
                    # Apply filing type filter
                    if filing_type_filter and form_type.strip() != filing_type_filter:
                        continue
                    
                    filing_data = {
                        "cik": cik.strip(),
                        "company_name": company_name.strip(),
                        "form_type": form_type.strip(),
                        "filing_date": date_filed.strip(),
                        "filename": filename.strip(),
                        "filing_url": f"{self.base_url}/Archives/{filename.strip()}",
                        "form_description": self.filing_types.get(form_type.strip(), "Other Filing")
                    }
                    
                    yield CollectionResult(
                        source_id=f"sec_filing_{cik.strip()}_{form_type.strip()}_{date_filed.strip()}",
                        source_type=self.source_type,
                        collected_at=datetime.now(),
                        data=filing_data,
                        metadata={
                            "source": "sec_edgar",
                            "type": "daily_filing",
                            "form_type": form_type.strip()
                        }
                    )
                    
            except Exception as e:
                # Skip malformed lines
                continue
    
    async def _parse_company_filings_atom(self, 
                                        atom_content: str,
                                        identifier: str) -> AsyncIterator[CollectionResult]:
        """Parse SEC company filings Atom feed."""
        
        try:
            soup = BeautifulSoup(atom_content, 'xml')
            entries = soup.find_all('entry')
            
            for entry in entries:
                try:
                    filing_data = await self._parse_filing_entry(entry)
                    if filing_data:
                        yield CollectionResult(
                            source_id=f"sec_company_filing_{identifier}_{filing_data.get('form_type', '')}_{filing_data.get('filing_date', '')}",
                            source_type=self.source_type,
                            collected_at=datetime.now(),
                            data=filing_data,
                            metadata={
                                "source": "sec_edgar",
                                "type": "company_filing",
                                "company_identifier": identifier
                            }
                        )
                except Exception as e:
                    # Skip malformed entries
                    continue
                    
        except Exception as e:
            # If parsing fails completely, return error
            pass
    
    async def _parse_filing_entry(self, entry) -> Optional[Dict[str, Any]]:
        """Parse individual filing entry from Atom feed."""
        
        try:
            title = entry.find('title').get_text() if entry.find('title') else ""
            link = entry.find('link')['href'] if entry.find('link') else ""
            updated = entry.find('updated').get_text() if entry.find('updated') else ""
            summary = entry.find('summary').get_text() if entry.find('summary') else ""
            
            # Extract form type and other details from title
            form_type = ""
            company_name = ""
            
            # Title format is usually: "Form TYPE by COMPANY (CIK)"
            title_match = re.match(r'Form\s+([^\s]+)\s+.*?by\s+(.+?)\s*\(', title)
            if title_match:
                form_type = title_match.group(1)
                company_name = title_match.group(2)
            
            # Extract filing date from updated field
            filing_date = ""
            if updated:
                try:
                    dt = datetime.fromisoformat(updated.replace('Z', '+00:00'))
                    filing_date = dt.strftime("%Y-%m-%d")
                except:
                    filing_date = updated[:10]  # Try to get just the date part
            
            # Extract CIK from link or summary
            cik = ""
            cik_match = re.search(r'CIK=(\d+)', link) or re.search(r'\((\d+)\)', title)
            if cik_match:
                cik = cik_match.group(1)
            
            return {
                "title": title,
                "form_type": form_type,
                "company_name": company_name,
                "cik": cik,
                "filing_date": filing_date,
                "filing_url": link,
                "summary": summary,
                "form_description": self.filing_types.get(form_type, "Other Filing"),
                "updated": updated
            }
            
        except Exception as e:
            return None
    
    async def _get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """Convert stock ticker to CIK number."""
        
        # This is a simplified implementation
        # In practice, you might want to use a more comprehensive ticker-to-CIK mapping
        search_url = f"{self.company_search_url}"
        params = {
            "action": "getcompany",
            "company": ticker,
            "type": "",
            "dateb": "",
            "owner": "exclude",
            "count": "1"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    search_url,
                    headers=self.headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Extract CIK from the response
                        cik_match = re.search(r'CIK=(\d+)', content)
                        if cik_match:
                            return cik_match.group(1).zfill(10)  # Pad with zeros
            except Exception:
                pass
        
        return None
    
    def _create_error_result(self, source_id: str, error_message: str) -> CollectionResult:
        """Create an error result."""
        return CollectionResult(
            source_id=source_id,
            source_type=self.source_type,
            collected_at=datetime.now(),
            data={},
            metadata={"error": error_message},
            success=False,
            error_message=error_message
        )
    
    async def health_check(self) -> bool:
        """Check if SEC EDGAR is accessible."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/edgar",
                    headers=self.headers
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    
    def get_supported_filing_types(self) -> Dict[str, str]:
        """Get dictionary of supported filing types and descriptions."""
        return self.filing_types.copy()