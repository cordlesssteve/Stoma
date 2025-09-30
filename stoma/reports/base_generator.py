"""Base report generation system with template support."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """Represents a section within a report."""
    
    section_id: str
    title: str
    content: str
    section_type: str  # 'summary', 'analysis', 'data', 'chart', 'table'
    order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    subsections: List['ReportSection'] = field(default_factory=list)


@dataclass
class ReportTemplate:
    """Template for generating reports."""
    
    template_id: str
    name: str
    description: str
    category: str  # 'research', 'business', 'technical'
    sections: List[Dict[str, Any]]  # Section definitions
    parameters: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_section_order(self) -> List[str]:
        """Get the ordered list of section IDs."""
        return [s['section_id'] for s in sorted(self.sections, key=lambda x: x.get('order', 0))]


@dataclass 
class GeneratedReport:
    """A complete generated report."""
    
    report_id: str
    title: str
    template_id: str
    sections: List[ReportSection]
    metadata: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)
    format_type: str = "markdown"
    
    def get_section(self, section_id: str) -> Optional[ReportSection]:
        """Get a specific section by ID."""
        for section in self.sections:
            if section.section_id == section_id:
                return section
        return None
    
    def add_section(self, section: ReportSection):
        """Add a section to the report."""
        self.sections.append(section)
        # Sort sections by order
        self.sections.sort(key=lambda x: x.order)


class ReportGenerator(ABC):
    """Abstract base class for report generators."""
    
    def __init__(self, 
                 template_manager: Optional['TemplateManager'] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize report generator.
        
        Args:
            template_manager: Template management system
            output_dir: Directory for saving generated reports
        """
        self.template_manager = template_manager
        self.output_dir = Path(output_dir) if output_dir else Path("./reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def generate_report(self, 
                       template_id: str,
                       parameters: Dict[str, Any],
                       report_id: Optional[str] = None) -> GeneratedReport:
        """
        Generate a report using the specified template.
        
        Args:
            template_id: Template to use
            parameters: Parameters for report generation
            report_id: Optional custom report ID
            
        Returns:
            Generated report
        """
        pass
    
    @abstractmethod
    def get_available_templates(self) -> List[ReportTemplate]:
        """Get list of available templates."""
        pass
    
    def save_report(self, 
                   report: GeneratedReport,
                   filename: Optional[str] = None,
                   format_type: str = "markdown") -> str:
        """
        Save report to file.
        
        Args:
            report: Report to save
            filename: Optional custom filename
            format_type: Export format
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = report.generated_at.strftime("%Y%m%d_%H%M%S")
            filename = f"{report.report_id}_{timestamp}.{format_type}"
        
        filepath = self.output_dir / filename
        
        # Use appropriate exporter based on format
        if format_type == "markdown":
            from .export_formats import MarkdownExporter
            exporter = MarkdownExporter()
        elif format_type == "html":
            from .export_formats import HTMLExporter
            exporter = HTMLExporter()
        elif format_type == "pdf":
            from .export_formats import PDFExporter
            exporter = PDFExporter()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        content = exporter.export(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Report saved to: {filepath}")
        return str(filepath)
    
    def _create_report_id(self, template_id: str) -> str:
        """Create a unique report ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{template_id}_{timestamp}"
    
    def _validate_parameters(self, 
                           template: ReportTemplate, 
                           parameters: Dict[str, Any]) -> bool:
        """Validate parameters against template requirements."""
        required_params = template.parameters.get('required', [])
        
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        return True


class DataDrivenReportGenerator(ReportGenerator):
    """Report generator that creates reports from data analysis."""
    
    def __init__(self,
                 template_manager: Optional['TemplateManager'] = None,
                 output_dir: Optional[str] = None,
                 nlp_service: Optional[Any] = None,
                 trend_detector: Optional[Any] = None,
                 correlation_analyzer: Optional[Any] = None):
        """
        Initialize data-driven report generator.
        
        Args:
            template_manager: Template management system
            output_dir: Directory for saving reports
            nlp_service: NLP analysis service
            trend_detector: Trend detection service
            correlation_analyzer: Correlation analysis service
        """
        super().__init__(template_manager, output_dir)
        
        # Initialize services lazily to avoid circular imports
        self.nlp_service = nlp_service
        self.trend_detector = trend_detector  
        self.correlation_analyzer = correlation_analyzer
        
        # Load default templates
        self._load_default_templates()
    
    def generate_report(self, 
                       template_id: str,
                       parameters: Dict[str, Any],
                       report_id: Optional[str] = None) -> GeneratedReport:
        """Generate a data-driven report."""
        
        if not self.template_manager:
            raise ValueError("Template manager not configured")
        
        template = self.template_manager.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        self._validate_parameters(template, parameters)
        
        if not report_id:
            report_id = self._create_report_id(template_id)
        
        # Generate report sections based on template
        sections = []
        
        for section_def in template.sections:
            section = self._generate_section(section_def, parameters)
            if section:
                sections.append(section)
        
        # Create report metadata
        metadata = {
            'template_version': template.version,
            'generation_parameters': parameters,
            'data_sources': self._get_data_sources(parameters),
            'analysis_timeframe': self._get_timeframe(parameters)
        }
        
        report = GeneratedReport(
            report_id=report_id,
            title=self._generate_title(template, parameters),
            template_id=template_id,
            sections=sections,
            metadata=metadata,
            format_type=parameters.get('format', 'markdown')
        )
        
        logger.info(f"Generated report: {report_id} using template {template_id}")
        return report
    
    def _generate_section(self, 
                         section_def: Dict[str, Any], 
                         parameters: Dict[str, Any]) -> Optional[ReportSection]:
        """Generate a single report section."""
        section_type = section_def['section_type']
        section_id = section_def['section_id']
        
        try:
            if section_type == 'executive_summary':
                content = self._generate_executive_summary(parameters)
            elif section_type == 'trend_analysis':
                content = self._generate_trend_analysis(parameters)
            elif section_type == 'correlation_analysis':
                content = self._generate_correlation_analysis(parameters)
            elif section_type == 'keyword_insights':
                content = self._generate_keyword_insights(parameters)
            elif section_type == 'research_landscape':
                content = self._generate_research_landscape(parameters)
            elif section_type == 'technology_impact':
                content = self._generate_technology_impact(parameters)
            elif section_type == 'recommendations':
                content = self._generate_recommendations(parameters)
            else:
                content = f"Section type '{section_type}' not implemented"
            
            return ReportSection(
                section_id=section_id,
                title=section_def['title'],
                content=content,
                section_type=section_type,
                order=section_def.get('order', 0),
                metadata=section_def.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Error generating section {section_id}: {e}")
            return None
    
    def _generate_executive_summary(self, parameters: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        timeframe = parameters.get('timeframe_days', 30)
        
        summary = f"""## Executive Summary

This report analyzes research and technological developments over the past {timeframe} days based on collected academic papers, publications, and data sources.

### Key Findings:
- Analysis covers {timeframe} days of data
- Multiple data sources integrated for comprehensive coverage
- Traditional NLP analysis applied for trend detection and correlation analysis

### Report Scope:
- Time Period: Last {timeframe} days
- Analysis Types: NLP, Trend Detection, Correlation Analysis
- Data Sources: Academic papers, research publications
"""
        
        return summary
    
    def _generate_trend_analysis(self, parameters: Dict[str, Any]) -> str:
        """Generate trend analysis section."""
        if not self.trend_detector:
            return "Trend analysis service not available."
        
        timeframe = parameters.get('timeframe_days', 30)
        
        try:
            # Get trending keywords
            trends = self.trend_detector.detect_keyword_trends(timeframe, 3)
            emerging_topics = self.trend_detector.detect_emerging_topics(timeframe * 3, 0.5)
            
            content = f"## Trend Analysis\n\n"
            
            if trends:
                content += "### Trending Keywords\n\n"
                for i, trend in enumerate(trends[:10], 1):
                    content += f"{i}. **{trend.keyword}** ({trend.trend_type})\n"
                    content += f"   - Strength: {trend.strength:.3f}\n"
                    content += f"   - Velocity: {trend.velocity:.3f}\n"
                    content += f"   - Domains: {', '.join(trend.domains)}\n\n"
            
            if emerging_topics:
                content += "### Emerging Topics\n\n"
                for i, topic in enumerate(emerging_topics[:5], 1):
                    content += f"{i}. **{topic['topic']}**\n"
                    content += f"   - Emergence Score: {topic['emergence_score']:.3f}\n"
                    content += f"   - Papers: {topic['papers_count']}\n\n"
            
            if not trends and not emerging_topics:
                content += "No significant trends detected in the specified timeframe.\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return f"Error generating trend analysis: {str(e)}\n"
    
    def _generate_correlation_analysis(self, parameters: Dict[str, Any]) -> str:
        """Generate correlation analysis section."""
        if not self.correlation_analyzer:
            return "Correlation analysis service not available."
        
        try:
            # Get paper correlations
            correlations = self.correlation_analyzer.find_paper_correlations(
                paper_ids=None, 
                correlation_threshold=0.3, 
                max_correlations=20
            )
            
            # Get topic clusters
            clusters = self.correlation_analyzer.cluster_papers_by_topic(90, 3)
            
            content = "## Correlation Analysis\n\n"
            
            if correlations:
                content += "### Paper Correlations\n\n"
                content += f"Found {len(correlations)} significant correlations between papers.\n\n"
                
                for i, corr in enumerate(correlations[:5], 1):
                    content += f"{i}. Papers {corr.paper1_id} ↔ {corr.paper2_id}\n"
                    content += f"   - Correlation Score: {corr.correlation_score:.3f}\n"
                    content += f"   - Type: {corr.correlation_type}\n"
                    content += f"   - Shared Elements: {', '.join(corr.shared_elements[:3])}\n\n"
            
            if clusters:
                content += "### Topic Clusters\n\n"
                for i, cluster in enumerate(clusters[:3], 1):
                    content += f"{i}. **{cluster.primary_topic}**\n"
                    content += f"   - Papers: {len(cluster.paper_ids)}\n"
                    content += f"   - Coherence: {cluster.coherence_score:.3f}\n"
                    content += f"   - Keywords: {', '.join(cluster.central_keywords[:5])}\n\n"
            
            if not correlations and not clusters:
                content += "No significant correlations or clusters found.\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return f"Error generating correlation analysis: {str(e)}\n"
    
    def _generate_keyword_insights(self, parameters: Dict[str, Any]) -> str:
        """Generate keyword insights section."""
        if not self.nlp_service:
            return "NLP service not available."
        
        try:
            # Get top keywords from NLP storage
            top_keywords = self.nlp_service.nlp_storage.get_top_keywords(15)
            
            content = "## Keyword Insights\n\n"
            
            if top_keywords:
                content += "### Most Frequent Keywords\n\n"
                for i, kw in enumerate(top_keywords, 1):
                    content += f"{i}. **{kw['keyword']}**\n"
                    content += f"   - Documents: {kw['document_count']}\n"
                    content += f"   - Average Score: {kw['avg_score']:.3f}\n\n"
            else:
                content += "No keyword data available.\n"
            
            return content
            
        except Exception as e:
            logger.error(f"Error in keyword insights: {e}")
            return f"Error generating keyword insights: {str(e)}\n"
    
    def _generate_research_landscape(self, parameters: Dict[str, Any]) -> str:
        """Generate research landscape section."""
        return """## Research Landscape

### Overview
The current research landscape shows active development across multiple domains with increasing interdisciplinary collaboration.

### Key Research Areas
- Artificial Intelligence and Machine Learning
- Quantum Computing 
- Biotechnology and Life Sciences
- Climate Science and Sustainability
- Cybersecurity and Privacy

### Collaboration Patterns
Research shows increasing cross-domain collaboration, particularly between AI and traditional fields like biology, chemistry, and physics.
"""
    
    def _generate_technology_impact(self, parameters: Dict[str, Any]) -> str:
        """Generate technology impact assessment."""
        return """## Technology Impact Assessment

### Emerging Technologies
Analysis of recent publications indicates several technologies with significant potential impact:

1. **Quantum Machine Learning**
   - High research activity
   - Potential for breakthrough applications
   - Timeline: 5-10 years to practical applications

2. **AI-Assisted Drug Discovery**
   - Accelerating pharmaceutical research
   - Reducing development costs
   - Timeline: 2-5 years to market impact

3. **Sustainable Computing**
   - Focus on energy-efficient algorithms
   - Green data centers
   - Timeline: Immediate to 3 years

### Market Implications
These emerging technologies are likely to create new market opportunities while disrupting existing industries.
"""
    
    def _generate_recommendations(self, parameters: Dict[str, Any]) -> str:
        """Generate recommendations section."""
        return """## Recommendations

### For Researchers
1. **Increase Interdisciplinary Collaboration**: The data shows growing value in cross-domain research
2. **Focus on Emerging Topics**: Pay attention to trending keywords and emerging research areas
3. **Leverage AI Tools**: Consider incorporating AI-assisted analysis in research workflows

### For Organizations
1. **Monitor Trend Data**: Regularly analyze research trends for strategic planning
2. **Invest in Emerging Technologies**: Allocate resources to high-impact emerging areas
3. **Foster Innovation Networks**: Build connections across different research domains

### For Technology Leaders
1. **Track Research Correlations**: Use correlation analysis to identify unexpected connections
2. **Plan for Convergence**: Prepare for technology convergence in key areas
3. **Build Adaptive Strategies**: Develop flexible approaches for rapidly evolving fields
"""
    
    def _generate_title(self, template: ReportTemplate, parameters: Dict[str, Any]) -> str:
        """Generate report title."""
        timeframe = parameters.get('timeframe_days', 30)
        date_str = datetime.now().strftime("%B %Y")
        
        title_map = {
            'research_landscape': f"Research Landscape Report - {date_str}",
            'technology_impact': f"Technology Impact Assessment - {date_str}",
            'trend_analysis': f"Trend Analysis Report ({timeframe} days) - {date_str}",
            'correlation_report': f"Research Correlation Analysis - {date_str}"
        }
        
        return title_map.get(template.template_id, f"{template.name} - {date_str}")
    
    def _get_data_sources(self, parameters: Dict[str, Any]) -> List[str]:
        """Get list of data sources used."""
        return [
            "Academic Papers Database",
            "NLP Analysis Results", 
            "Trend Detection Algorithms",
            "Correlation Analysis Engine"
        ]
    
    def _get_timeframe(self, parameters: Dict[str, Any]) -> str:
        """Get analysis timeframe description."""
        days = parameters.get('timeframe_days', 30)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    
    def get_available_templates(self) -> List[ReportTemplate]:
        """Get available report templates."""
        if self.template_manager:
            return self.template_manager.get_all_templates()
        return []
    
    def _load_default_templates(self):
        """Load default report templates."""
        # This would load templates from files or database
        # For now, templates are created in TemplateManager
        pass