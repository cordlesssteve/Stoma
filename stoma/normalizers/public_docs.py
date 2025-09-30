"""Normalizer for public documents, government filings, and corporate reports."""

from datetime import datetime
from typing import List, Dict, Any, Optional

from .base import BaseNormalizer, NormalizedData
from ..collectors.base import CollectionResult, SourceType


class PublicDocsNormalizer(BaseNormalizer):
    """Normalizer for public documents, SEC filings, and government data."""
    
    def _get_supported_source_types(self) -> List[SourceType]:
        return [SourceType.PUBLIC_DOCS]
    
    async def normalize(self, result: CollectionResult) -> NormalizedData:
        """Normalize public document data based on source and type."""
        
        metadata_source = result.metadata.get("source", "")
        
        if metadata_source == "sec_edgar":
            return await self._normalize_sec_filing(result)
        else:
            # Default normalization for other public documents
            return await self._normalize_generic_public_doc(result)
    
    async def _normalize_sec_filing(self, result: CollectionResult) -> NormalizedData:
        """Normalize SEC EDGAR filing data."""
        data = result.data
        
        # Extract key information
        form_type = data.get("form_type", "")
        company_name = data.get("company_name", "")
        cik = data.get("cik", "")
        
        # Create title
        title = f"{form_type}: {company_name}" if company_name else f"SEC Filing {form_type}"
        if cik:
            title += f" (CIK: {cik})"
        
        # Create content from available fields
        content_parts = []
        
        if data.get("form_description"):
            content_parts.append(f"Filing Type: {data['form_description']}")
        
        if data.get("summary"):
            content_parts.append(f"Summary: {data['summary']}")
        
        if company_name:
            content_parts.append(f"Company: {company_name}")
        
        if cik:
            content_parts.append(f"Central Index Key (CIK): {cik}")
        
        content = " | ".join(content_parts)
        
        # Generate summary
        summary = data.get("summary", "")
        if not summary:
            summary = f"{data.get('form_description', form_type)} filed by {company_name or 'Unknown Company'}"
        
        # Extract categories and keywords
        categories = ["sec_filing", "corporate"]
        keywords = [form_type]
        
        if company_name:
            # Extract company keywords (simple approach)
            keywords.extend(company_name.split()[:3])  # First 3 words of company name
        
        # Add filing type categories
        categories.extend(self._categorize_filing_type(form_type))
        
        # Extract metrics
        metrics = {
            "form_type": form_type,
            "cik": cik,
            "is_amendment": "/A" in form_type
        }
        
        # Add filing urgency/importance score
        metrics["importance_score"] = self._calculate_filing_importance(form_type)
        
        return NormalizedData(
            id=self._generate_sec_filing_id(result),
            source_type=result.source_type,
            source_id=result.source_id,
            title=title,
            content=content,
            summary=summary,
            authors=[company_name] if company_name else [],
            published_date=self._parse_filing_date(data.get("filing_date")),
            collected_date=result.collected_at,
            url=data.get("filing_url"),
            keywords=keywords,
            categories=categories,
            tags=self._generate_sec_tags(data),
            metrics=metrics,
            raw_data=data
        )
    
    async def _normalize_generic_public_doc(self, result: CollectionResult) -> NormalizedData:
        """Normalize generic public document data."""
        data = result.data
        
        return NormalizedData(
            id=self._generate_generic_id(result),
            source_type=result.source_type,
            source_id=result.source_id,
            title=data.get("title", "Public Document"),
            content=data.get("content", data.get("summary", "")),
            summary=data.get("summary", ""),
            authors=data.get("authors", []),
            published_date=self._parse_date(data.get("published_date")),
            collected_date=result.collected_at,
            url=data.get("url"),
            keywords=data.get("keywords", []),
            categories=["public_document"],
            tags=data.get("tags", []),
            metrics=data.get("metrics", {}),
            raw_data=data
        )
    
    def _generate_sec_filing_id(self, result: CollectionResult) -> str:
        """Generate unique ID for SEC filings."""
        data = result.data
        cik = data.get("cik", "unknown")
        form_type = data.get("form_type", "unknown")
        filing_date = data.get("filing_date", "unknown")
        
        return f"sec_{cik}_{form_type}_{filing_date}".replace("/", "_").replace("-", "_")
    
    def _generate_generic_id(self, result: CollectionResult) -> str:
        """Generate unique ID for generic public documents."""
        import hashlib
        
        content = f"{result.source_id}_{result.data.get('title', '')}_{result.collected_at.isoformat()}"
        return f"public_doc_{hashlib.md5(content.encode()).hexdigest()}"
    
    def _categorize_filing_type(self, form_type: str) -> List[str]:
        """Categorize SEC filing types."""
        categories = []
        
        if form_type in ["10-K", "10-K/A"]:
            categories.extend(["annual_report", "financial_statement"])
        elif form_type in ["10-Q", "10-Q/A"]:
            categories.extend(["quarterly_report", "financial_statement"])
        elif form_type in ["8-K", "8-K/A"]:
            categories.extend(["current_report", "material_event"])
        elif form_type == "DEF 14A":
            categories.extend(["proxy_statement", "shareholder_communication"])
        elif form_type in ["S-1", "S-3"]:
            categories.extend(["registration_statement", "securities_offering"])
        elif form_type in ["3", "4", "5"]:
            categories.extend(["insider_trading", "ownership_change"])
        elif form_type.startswith("SC "):
            categories.append("tender_offer")
        elif form_type.startswith("N-"):
            categories.append("investment_company")
        
        return categories
    
    def _calculate_filing_importance(self, form_type: str) -> int:
        """Calculate importance score (1-10) for different filing types."""
        
        # High importance filings (8-10)
        if form_type in ["8-K", "8-K/A"]:  # Material events
            return 9
        elif form_type in ["10-K", "10-K/A"]:  # Annual reports
            return 8
        elif form_type in ["4"]:  # Insider trading
            return 8
        
        # Medium importance filings (5-7)
        elif form_type in ["10-Q", "10-Q/A"]:  # Quarterly reports
            return 7
        elif form_type in ["DEF 14A"]:  # Proxy statements
            return 6
        elif form_type in ["S-1", "S-3"]:  # Registration statements
            return 6
        
        # Lower importance filings (3-4)
        elif form_type in ["3", "5"]:  # Other ownership reports
            return 4
        
        # Default importance
        else:
            return 3
    
    def _generate_sec_tags(self, data: Dict[str, Any]) -> List[str]:
        """Generate tags for SEC filings."""
        tags = ["sec_filing"]
        
        form_type = data.get("form_type", "")
        
        # Add form type tag
        if form_type:
            tags.append(f"form_{form_type.lower().replace('/', '_').replace('-', '_')}")
        
        # Add amendment tag
        if "/A" in form_type:
            tags.append("amendment")
        
        # Add urgency tags based on form type
        if form_type in ["8-K", "8-K/A"]:
            tags.append("urgent")
        elif form_type in ["4"]:
            tags.append("insider_activity")
        elif form_type in ["10-K", "10-Q"]:
            tags.append("financial_report")
        
        # Add timing tags based on filing date
        filing_date = data.get("filing_date")
        if filing_date:
            try:
                filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
                days_old = (datetime.now() - filing_dt).days
                
                if days_old <= 1:
                    tags.append("recent")
                elif days_old <= 7:
                    tags.append("this_week")
                elif days_old <= 30:
                    tags.append("this_month")
            except ValueError:
                pass
        
        return tags
    
    def _parse_filing_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse various SEC date formats."""
        if not date_str:
            return None
        
        # Common SEC date formats
        date_formats = [
            "%Y-%m-%d",      # 2023-12-31
            "%m/%d/%Y",      # 12/31/2023
            "%m-%d-%Y",      # 12-31-2023
            "%Y%m%d",        # 20231231
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        # If none of the formats work, try dateutil
        try:
            import dateutil.parser
            return dateutil.parser.parse(date_str)
        except Exception:
            return None
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse generic date formats."""
        return self._parse_filing_date(date_str)