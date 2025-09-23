"""Research-specific report implementations."""

from typing import Dict, Any
from .base_generator import ReportTemplate
from datetime import datetime


def create_research_landscape_template() -> ReportTemplate:
    """Create research landscape report template."""
    return ReportTemplate(
        template_id="research_landscape",
        name="Research Landscape Report",
        description="Comprehensive overview of research developments and trends",
        category="research",
        sections=[
            {
                "section_id": "executive_summary",
                "title": "Executive Summary",
                "section_type": "executive_summary",
                "order": 1
            },
            {
                "section_id": "research_overview",
                "title": "Research Landscape Overview", 
                "section_type": "research_landscape",
                "order": 2
            },
            {
                "section_id": "trend_analysis",
                "title": "Trending Research Areas",
                "section_type": "trend_analysis", 
                "order": 3
            },
            {
                "section_id": "correlation_insights",
                "title": "Research Correlations",
                "section_type": "correlation_analysis",
                "order": 4
            },
            {
                "section_id": "keyword_analysis",
                "title": "Key Research Terms",
                "section_type": "keyword_insights",
                "order": 5
            },
            {
                "section_id": "recommendations",
                "title": "Research Recommendations", 
                "section_type": "recommendations",
                "order": 6
            }
        ],
        parameters={
            "required": ["timeframe_days"],
            "optional": ["focus_domains", "min_papers"]
        }
    )


def create_technology_impact_template() -> ReportTemplate:
    """Create technology impact assessment template."""
    return ReportTemplate(
        template_id="technology_impact",
        name="Technology Impact Assessment",
        description="Analysis of emerging technologies and their potential impact",
        category="technology",
        sections=[
            {
                "section_id": "executive_summary",
                "title": "Executive Summary",
                "section_type": "executive_summary",
                "order": 1
            },
            {
                "section_id": "technology_overview",
                "title": "Technology Impact Overview",
                "section_type": "technology_impact",
                "order": 2
            },
            {
                "section_id": "emerging_trends",
                "title": "Emerging Technology Trends",
                "section_type": "trend_analysis",
                "order": 3
            },
            {
                "section_id": "correlation_patterns",
                "title": "Technology Convergence",
                "section_type": "correlation_analysis", 
                "order": 4
            },
            {
                "section_id": "strategic_recommendations",
                "title": "Strategic Recommendations",
                "section_type": "recommendations",
                "order": 5
            }
        ],
        parameters={
            "required": ["timeframe_days"],
            "optional": ["technology_focus", "impact_threshold"]
        }
    )


class ResearchLandscapeReport:
    """Research landscape report implementation."""
    
    @staticmethod
    def get_template() -> ReportTemplate:
        """Get the research landscape template."""
        return create_research_landscape_template()


class TechnologyImpactReport:
    """Technology impact assessment report implementation."""
    
    @staticmethod
    def get_template() -> ReportTemplate:
        """Get the technology impact template."""
        return create_technology_impact_template()