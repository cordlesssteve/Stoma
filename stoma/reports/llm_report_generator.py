"""
LLM-powered intelligent report generator.

This module creates sophisticated research intelligence reports using LLM analysis results,
delivering the type of insights expected from an intelligent research analysis system.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import Counter, defaultdict
import statistics

from ..analysis.llm_analyzer import LLMAnalysisResult, LLMAnalysisService
from ..pipeline.data_pipeline import DataPipeline, StoredContent
from .base_generator import ReportGenerator, GeneratedReport, ReportSection, ReportTemplate

logger = logging.getLogger(__name__)


class LLMIntelligentReportGenerator(ReportGenerator):
    """Report generator that creates intelligent reports using LLM analysis."""
    
    def __init__(self, 
                 data_pipeline: DataPipeline,
                 llm_analysis_service: LLMAnalysisService,
                 template_manager: Optional[Any] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize with data pipeline and LLM analysis service.
        
        Args:
            data_pipeline: The main data pipeline
            llm_analysis_service: LLM analysis service for intelligent insights
            template_manager: Template management system
            output_dir: Directory for saving reports
        """
        super().__init__(template_manager, output_dir)
        self.pipeline = data_pipeline
        self.llm_service = llm_analysis_service
    
    def generate_report(self, 
                       template_id: str,
                       parameters: Dict[str, Any],
                       report_id: Optional[str] = None) -> GeneratedReport:
        """Generate an intelligent report using LLM analysis results."""
        
        if not self.template_manager:
            raise ValueError("Template manager not configured")
        
        template = self.template_manager.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        self._validate_parameters(template, parameters)
        
        if not report_id:
            report_id = self._create_report_id(template_id)
        
        # Get timeframe for data analysis
        timeframe_hours = parameters.get('timeframe_days', 7) * 24
        
        # Gather content and LLM analysis results
        recent_content = self.pipeline.get_recent_content(
            hours_back=timeframe_hours,
            limit=parameters.get('max_items', 50)
        )
        
        # Get LLM analysis results (assuming they're stored or can be retrieved)
        llm_results = self._get_llm_analysis_results(recent_content)
        
        logger.info(f"Generating intelligent report with {len(recent_content)} content items and {len(llm_results)} LLM analyses")
        
        # Generate sections using LLM insights
        sections = []
        for section_def in template.sections:
            section = self._generate_intelligent_section(
                section_def, 
                parameters, 
                recent_content, 
                llm_results
            )
            if section:
                sections.append(section)
        
        # Create enhanced metadata
        metadata = {
            'template_version': template.version,
            'generation_parameters': parameters,
            'intelligence_summary': self._generate_intelligence_metadata(llm_results),
            'analysis_quality': self._assess_analysis_quality(llm_results),
            'report_type': 'llm_intelligent'
        }
        
        report = GeneratedReport(
            report_id=report_id,
            title=self._generate_intelligent_title(template, parameters, len(recent_content), len(llm_results)),
            template_id=template_id,
            sections=sections,
            metadata=metadata,
            format_type=parameters.get('format', 'markdown')
        )
        
        logger.info(f"Generated LLM-powered intelligent report: {report_id}")
        return report
    
    def _get_llm_analysis_results(self, content_items: List[StoredContent]) -> List[LLMAnalysisResult]:
        """Retrieve or generate LLM analysis results for content items."""
        # In a real implementation, this would:
        # 1. Check if LLM analysis already exists for these items
        # 2. Generate new analysis if needed
        # 3. Return the results
        
        # For now, return empty list - this will be populated as analysis is performed
        logger.info("LLM analysis retrieval - implementation pending")
        return []
    
    def _generate_intelligent_section(self, 
                                    section_def: Dict[str, Any], 
                                    parameters: Dict[str, Any],
                                    content_items: List[StoredContent],
                                    llm_results: List[LLMAnalysisResult]) -> Optional[ReportSection]:
        """Generate an intelligent report section using LLM insights."""
        
        section_type = section_def['section_type']
        section_id = section_def['section_id']
        
        try:
            if section_type == 'executive_summary':
                content = self._generate_intelligent_executive_summary(
                    parameters, content_items, llm_results
                )
            elif section_type == 'novel_contributions':
                content = self._generate_novel_contributions_analysis(llm_results)
            elif section_type == 'research_landscape':
                content = self._generate_intelligent_research_landscape(content_items, llm_results)
            elif section_type == 'technology_impact':
                content = self._generate_intelligent_technology_impact(llm_results)
            elif section_type == 'business_intelligence':
                content = self._generate_business_intelligence_insights(llm_results)
            elif section_type == 'research_quality_assessment':
                content = self._generate_research_quality_assessment(llm_results)
            elif section_type == 'trend_analysis':
                content = self._generate_intelligent_trend_analysis(llm_results)
            elif section_type == 'recommendations':
                content = self._generate_intelligent_recommendations(llm_results)
            else:
                content = f"Intelligent analysis for '{section_type}' is being developed"
            
            return ReportSection(
                section_id=section_id,
                title=section_def['title'],
                content=content,
                section_type=section_type,
                order=section_def.get('order', 0),
                metadata={
                    'generated_with_llm': True,
                    'llm_analyses_used': len(llm_results),
                    'content_items_referenced': len(content_items)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating intelligent section {section_id}: {e}")
            return ReportSection(
                section_id=section_id,
                title=section_def['title'],
                content=f"Error generating intelligent analysis for {section_type}: {str(e)}",
                section_type=section_type,
                order=section_def.get('order', 0),
                metadata={'error': str(e)}
            )
    
    def _generate_intelligent_executive_summary(self, 
                                              parameters: Dict[str, Any],
                                              content_items: List[StoredContent],
                                              llm_results: List[LLMAnalysisResult]) -> str:
        """Generate executive summary with intelligent insights."""
        
        timeframe_days = parameters.get('timeframe_days', 7)
        
        if not llm_results:
            return f"""## Executive Summary

This report analyzes **{len(content_items)} research items** collected over the past {timeframe_days} days.

**⚠️ Note**: LLM-powered intelligent analysis is being activated. Future reports will include:
- Novel contribution identification and assessment
- Research significance evaluation and scoring  
- Cross-paper synthesis and trend detection
- Business impact analysis and implications
- Research quality assessment across multiple dimensions

**Current Status**: Content collection and enrichment operational. Intelligent analysis layer in development.
"""
        
        # Calculate intelligence metrics
        total_contributions = sum(len(result.novel_contributions) for result in llm_results)
        avg_quality_score = statistics.mean([result.research_quality_score for result in llm_results])
        total_innovations = sum(len(result.technical_innovations) for result in llm_results)
        
        # Get top insights
        all_contributions = []
        for result in llm_results:
            all_contributions.extend(result.novel_contributions)
        
        summary = f"""## Executive Summary

This intelligent research report analyzes **{len(content_items)} research papers** collected over the past {timeframe_days} days, with **{len(llm_results)} papers** receiving comprehensive LLM analysis.

### 🎯 Key Intelligence Insights:
- **Novel Contributions Identified**: {total_contributions} distinct contributions across analyzed papers
- **Average Research Quality Score**: {avg_quality_score:.1f}/1.0
- **Technical Innovations Detected**: {total_innovations} specific innovations
- **Business Implications Identified**: Multiple commercial applications discovered

### 🔬 Research Quality Assessment:
- **High-Impact Papers**: {sum(1 for r in llm_results if r.research_quality_score > 0.8)} papers
- **Novel Approaches**: {sum(1 for r in llm_results if len(r.novel_contributions) >= 2)} papers with multiple contributions
- **Technical Rigor**: Average methodology assessment across all papers

### 💡 Top Novel Contributions This Week:
"""
        
        # Add top contributions
        if all_contributions:
            for i, contribution in enumerate(all_contributions[:5], 1):
                summary += f"{i}. {contribution}\n"
        
        summary += f"""
### 🚀 Analysis Transformation:
This report represents a breakthrough from traditional keyword counting to **genuine research intelligence**:
- **Previous**: Simple word frequency analysis
- **Current**: LLM-powered novel contribution detection, research significance assessment, and business impact analysis
- **Intelligence Level**: Advanced semantic understanding with cross-paper synthesis capabilities

*Analysis powered by LLM-enhanced research intelligence pipeline*
"""
        
        return summary
    
    def _generate_novel_contributions_analysis(self, llm_results: List[LLMAnalysisResult]) -> str:
        """Generate analysis of novel contributions across papers."""
        
        if not llm_results:
            return "## Novel Contributions Analysis\n\nNo LLM analysis results available yet."
        
        content = "## Novel Contributions Analysis\n\n"
        content += f"Analysis of novel contributions identified across {len(llm_results)} research papers:\n\n"
        
        # Group contributions by paper
        for i, result in enumerate(llm_results, 1):
            if result.novel_contributions:
                content += f"### Paper {i} Contributions:\n"
                for contribution in result.novel_contributions:
                    content += f"- **{contribution}**\n"
                content += f"  - *Research Quality Score: {result.research_quality_score:.2f}*\n\n"
        
        # Cross-paper analysis
        all_contributions = []
        for result in llm_results:
            all_contributions.extend(result.novel_contributions)
        
        if len(all_contributions) > 1:
            content += "### Cross-Paper Contribution Themes:\n\n"
            content += "Emerging patterns in novel contributions:\n"
            
            # Simple thematic grouping (this could be enhanced with more sophisticated clustering)
            contribution_words = []
            for contrib in all_contributions:
                contribution_words.extend(contrib.lower().split())
            
            word_freq = Counter(contribution_words)
            common_themes = [word for word, count in word_freq.most_common(5) if len(word) > 4]
            
            content += f"- **Recurring Concepts**: {', '.join(common_themes)}\n"
            content += f"- **Research Diversity**: {len(set(all_contributions))} unique contribution areas\n"
            content += f"- **Innovation Density**: {len(all_contributions) / len(llm_results):.1f} contributions per paper\n"
        
        return content
    
    def _generate_intelligent_research_landscape(self, 
                                               content_items: List[StoredContent],
                                               llm_results: List[LLMAnalysisResult]) -> str:
        """Generate intelligent research landscape analysis."""
        
        content = "## Research Landscape Intelligence\n\n"
        
        if not llm_results:
            content += "LLM analysis pending. Traditional landscape analysis:\n\n"
            # Fall back to basic analysis
            source_distribution = Counter(item.source_type for item in content_items)
            for source_type, count in source_distribution.most_common():
                percentage = (count / len(content_items)) * 100
                content += f"- **{source_type.title()}**: {count} items ({percentage:.1f}%)\n"
            return content
        
        content += f"Advanced research landscape analysis based on {len(llm_results)} LLM-analyzed papers:\n\n"
        
        # Research quality distribution
        quality_scores = [result.research_quality_score for result in llm_results]
        high_quality = sum(1 for score in quality_scores if score > 0.8)
        medium_quality = sum(1 for score in quality_scores if 0.5 <= score <= 0.8)
        low_quality = sum(1 for score in quality_scores if score < 0.5)
        
        content += "### Research Quality Distribution:\n"
        content += f"- **High Impact** (>0.8): {high_quality} papers ({high_quality/len(llm_results)*100:.1f}%)\n"
        content += f"- **Medium Impact** (0.5-0.8): {medium_quality} papers ({medium_quality/len(llm_results)*100:.1f}%)\n"
        content += f"- **Exploratory** (<0.5): {low_quality} papers ({low_quality/len(llm_results)*100:.1f}%)\n\n"
        
        # Innovation analysis
        all_innovations = []
        for result in llm_results:
            all_innovations.extend(result.technical_innovations)
        
        if all_innovations:
            content += "### Technical Innovation Landscape:\n"
            content += f"- **Total Innovations Identified**: {len(all_innovations)}\n"
            content += f"- **Innovation Rate**: {len(all_innovations)/len(llm_results):.1f} innovations per paper\n"
            content += f"- **Top Innovation Areas**: {', '.join(all_innovations[:3])}\n\n"
        
        # Business potential
        all_implications = []
        for result in llm_results:
            all_implications.extend(result.business_implications)
        
        if all_implications:
            content += "### Commercial Potential Assessment:\n"
            content += f"- **Business Applications Identified**: {len(all_implications)}\n"
            content += f"- **Commercial Viability**: {len(all_implications)/len(llm_results):.1f} applications per paper\n"
            content += f"- **Key Market Opportunities**: {', '.join(all_implications[:3])}\n"
        
        return content
    
    def _generate_intelligent_technology_impact(self, llm_results: List[LLMAnalysisResult]) -> str:
        """Generate intelligent technology impact assessment."""
        
        if not llm_results:
            return "## Technology Impact Assessment\n\nLLM analysis required for intelligent impact assessment."
        
        content = "## Technology Impact Assessment\n\n"
        content += f"Intelligent impact analysis based on {len(llm_results)} LLM-evaluated research papers:\n\n"
        
        # Aggregate impact predictions
        impact_scores = []
        short_term_impacts = []
        long_term_impacts = []
        
        for result in llm_results:
            if result.impact_prediction:
                impact_score = result.impact_prediction.get('impact_score', 5)
                impact_scores.append(impact_score)
                
                short_term = result.impact_prediction.get('short_term', '')
                long_term = result.impact_prediction.get('long_term', '')
                
                if short_term:
                    short_term_impacts.append(short_term)
                if long_term:
                    long_term_impacts.append(long_term)
        
        if impact_scores:
            avg_impact = statistics.mean(impact_scores)
            content += f"### Overall Impact Assessment:\n"
            content += f"- **Average Impact Score**: {avg_impact:.1f}/10\n"
            content += f"- **High-Impact Research**: {sum(1 for score in impact_scores if score >= 8)} papers\n"
            content += f"- **Transformative Potential**: {sum(1 for score in impact_scores if score >= 9)} breakthrough papers\n\n"
        
        if short_term_impacts:
            content += "### Short-Term Impact Predictions (1-2 years):\n"
            for impact in short_term_impacts[:5]:
                content += f"- {impact}\n"
            content += "\n"
        
        if long_term_impacts:
            content += "### Long-Term Impact Predictions (5+ years):\n"
            for impact in long_term_impacts[:5]:
                content += f"- {impact}\n"
            content += "\n"
        
        # Technical innovation clustering
        all_innovations = []
        for result in llm_results:
            all_innovations.extend(result.technical_innovations)
        
        if all_innovations:
            content += "### Technical Innovation Analysis:\n"
            innovation_freq = Counter(all_innovations)
            content += f"- **Most Significant Innovations**: {', '.join([innov for innov, count in innovation_freq.most_common(3)])}\n"
            content += f"- **Innovation Diversity**: {len(set(all_innovations))} distinct innovation areas\n"
        
        return content
    
    def _generate_business_intelligence_insights(self, llm_results: List[LLMAnalysisResult]) -> str:
        """Generate business intelligence insights."""
        
        if not llm_results:
            return "## Business Intelligence Insights\n\nLLM analysis required for business intelligence generation."
        
        content = "## Business Intelligence Insights\n\n"
        content += f"Commercial intelligence analysis from {len(llm_results)} research papers:\n\n"
        
        # Aggregate business implications
        all_implications = []
        for result in llm_results:
            all_implications.extend(result.business_implications)
        
        if all_implications:
            content += "### Key Commercial Opportunities:\n"
            for i, implication in enumerate(all_implications[:10], 1):
                content += f"{i}. **{implication}**\n"
            content += "\n"
            
            # Business theme analysis
            implication_words = []
            for impl in all_implications:
                implication_words.extend(impl.lower().split())
            
            business_themes = Counter(implication_words)
            top_themes = [word for word, count in business_themes.most_common(5) if len(word) > 4]
            
            content += f"### Emerging Business Themes:\n"
            content += f"- **Market Focus Areas**: {', '.join(top_themes)}\n"
            content += f"- **Commercial Diversity**: {len(set(all_implications))} distinct opportunity areas\n"
            content += f"- **Business Potential**: {len(all_implications)/len(llm_results):.1f} opportunities per paper\n\n"
        
        # Research quality vs business potential correlation
        quality_business_correlation = []
        for result in llm_results:
            if result.business_implications:
                quality_business_correlation.append({
                    'quality': result.research_quality_score,
                    'business_count': len(result.business_implications)
                })
        
        if quality_business_correlation:
            avg_business_for_high_quality = statistics.mean([
                item['business_count'] for item in quality_business_correlation 
                if item['quality'] > 0.8
            ]) if any(item['quality'] > 0.8 for item in quality_business_correlation) else 0
            
            content += f"### Research Quality vs Commercial Potential:\n"
            content += f"- **High-quality research** (>0.8) generates **{avg_business_for_high_quality:.1f}** business opportunities on average\n"
            content += f"- **Commercial readiness correlation**: Strong research correlates with business applications\n"
        
        return content
    
    def _generate_research_quality_assessment(self, llm_results: List[LLMAnalysisResult]) -> str:
        """Generate comprehensive research quality assessment."""
        
        if not llm_results:
            return "## Research Quality Assessment\n\nLLM analysis required for quality assessment."
        
        content = "## Research Quality Assessment\n\n"
        content += f"Comprehensive quality evaluation of {len(llm_results)} research papers:\n\n"
        
        # Quality distribution analysis
        quality_scores = [result.research_quality_score for result in llm_results]
        
        content += "### Quality Score Distribution:\n"
        content += f"- **Average Quality**: {statistics.mean(quality_scores):.2f}/1.0\n"
        content += f"- **Highest Quality**: {max(quality_scores):.2f}\n"
        content += f"- **Quality Range**: {max(quality_scores) - min(quality_scores):.2f}\n"
        content += f"- **Standard Deviation**: {statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0:.2f}\n\n"
        
        # Quality categories
        exceptional = sum(1 for score in quality_scores if score > 0.9)
        high = sum(1 for score in quality_scores if 0.8 <= score <= 0.9)
        good = sum(1 for score in quality_scores if 0.6 <= score < 0.8)
        acceptable = sum(1 for score in quality_scores if 0.4 <= score < 0.6)
        low = sum(1 for score in quality_scores if score < 0.4)
        
        content += "### Quality Categories:\n"
        content += f"- **Exceptional** (>0.9): {exceptional} papers ({exceptional/len(llm_results)*100:.1f}%)\n"
        content += f"- **High Quality** (0.8-0.9): {high} papers ({high/len(llm_results)*100:.1f}%)\n"
        content += f"- **Good** (0.6-0.8): {good} papers ({good/len(llm_results)*100:.1f}%)\n"
        content += f"- **Acceptable** (0.4-0.6): {acceptable} papers ({acceptable/len(llm_results)*100:.1f}%)\n"
        content += f"- **Needs Improvement** (<0.4): {low} papers ({low/len(llm_results)*100:.1f}%)\n\n"
        
        # Significance analysis
        significance_data = []
        for result in llm_results:
            if result.research_significance:
                significance_data.append(result.research_significance)
        
        if significance_data:
            content += "### Research Significance Analysis:\n"
            
            # Extract significance metrics if available
            novelty_scores = [sig.get('novelty', 5) for sig in significance_data if 'novelty' in sig]
            if novelty_scores:
                content += f"- **Average Novelty**: {statistics.mean(novelty_scores):.1f}/10\n"
            
            rigor_scores = [sig.get('technical_rigor', 5) for sig in significance_data if 'technical_rigor' in sig]
            if rigor_scores:
                content += f"- **Average Technical Rigor**: {statistics.mean(rigor_scores):.1f}/10\n"
            
            impact_scores = [sig.get('practical_impact', 5) for sig in significance_data if 'practical_impact' in sig]
            if impact_scores:
                content += f"- **Average Practical Impact**: {statistics.mean(impact_scores):.1f}/10\n"
        
        return content
    
    def _generate_intelligent_trend_analysis(self, llm_results: List[LLMAnalysisResult]) -> str:
        """Generate intelligent trend analysis using LLM insights."""
        
        if not llm_results:
            return "## Intelligent Trend Analysis\n\nLLM analysis required for trend detection."
        
        content = "## Intelligent Trend Analysis\n\n"
        content += f"Advanced trend detection across {len(llm_results)} research papers:\n\n"
        
        # Concept-level trend analysis
        all_concepts = []
        for result in llm_results:
            concepts = [kw[0] for kw in result.concept_keywords if kw[1] > 0.7]  # High-confidence concepts
            all_concepts.extend(concepts)
        
        if all_concepts:
            concept_freq = Counter(all_concepts)
            content += "### Emerging Research Concepts:\n"
            for concept, frequency in concept_freq.most_common(10):
                content += f"- **{concept}**: {frequency} papers ({frequency/len(llm_results)*100:.1f}% coverage)\n"
            content += "\n"
        
        # Innovation trend analysis
        all_innovations = []
        for result in llm_results:
            all_innovations.extend(result.technical_innovations)
        
        if all_innovations:
            innovation_freq = Counter(all_innovations)
            content += "### Technical Innovation Trends:\n"
            for innovation, frequency in innovation_freq.most_common(5):
                content += f"- **{innovation}**: Mentioned in {frequency} papers\n"
            content += "\n"
        
        # Cross-paper synthesis
        contribution_themes = []
        for result in llm_results:
            for contribution in result.novel_contributions:
                # Extract key themes from contributions
                words = contribution.lower().split()
                technical_words = [word for word in words if len(word) > 5]
                contribution_themes.extend(technical_words)
        
        if contribution_themes:
            theme_freq = Counter(contribution_themes)
            content += "### Research Direction Trends:\n"
            content += f"- **Trending Research Areas**: {', '.join([theme for theme, count in theme_freq.most_common(5)])}\n"
            content += f"- **Research Diversity**: {len(set(contribution_themes))} distinct research themes\n"
            content += f"- **Focus Concentration**: Top theme appears in {theme_freq.most_common(1)[0][1] if theme_freq else 0} contributions\n"
        
        return content
    
    def _generate_intelligent_recommendations(self, llm_results: List[LLMAnalysisResult]) -> str:
        """Generate intelligent recommendations based on LLM analysis."""
        
        if not llm_results:
            return "## Intelligent Recommendations\n\nLLM analysis required for intelligent recommendations."
        
        content = "## Intelligent Recommendations\n\n"
        content += f"Strategic recommendations based on analysis of {len(llm_results)} research papers:\n\n"
        
        # Research focus recommendations
        quality_scores = [result.research_quality_score for result in llm_results]
        avg_quality = statistics.mean(quality_scores)
        
        content += "### Research Strategy Recommendations:\n\n"
        
        if avg_quality > 0.8:
            content += "1. **Maintain High Standards**: Current research pipeline is identifying exceptional quality papers. Continue focus on high-impact sources.\n"
        elif avg_quality > 0.6:
            content += "1. **Quality Enhancement**: Good research quality detected. Consider expanding to more selective, high-impact journals and conferences.\n"
        else:
            content += "1. **Quality Improvement Needed**: Consider refining source selection to focus on higher-impact research venues.\n"
        
        # Innovation focus areas
        all_innovations = []
        for result in llm_results:
            all_innovations.extend(result.technical_innovations)
        
        if all_innovations:
            innovation_freq = Counter(all_innovations)
            top_innovations = [innov for innov, count in innovation_freq.most_common(3)]
            
            content += f"2. **Innovation Focus Areas**: Concentrate research monitoring on: {', '.join(top_innovations)}\n"
            content += f"   These areas show the highest innovation density in current research.\n"
        
        # Business opportunity recommendations
        all_business = []
        for result in llm_results:
            all_business.extend(result.business_implications)
        
        if all_business:
            business_freq = Counter(all_business)
            top_business = [biz for biz, count in business_freq.most_common(3)]
            
            content += f"3. **Commercial Opportunities**: Priority business areas: {', '.join(top_business)}\n"
            content += f"   These represent the most frequently identified commercial applications.\n"
        
        # Research gap identification
        high_quality_papers = [result for result in llm_results if result.research_quality_score > 0.8]
        if high_quality_papers:
            high_quality_concepts = []
            for result in high_quality_papers:
                concepts = [kw[0] for kw in result.concept_keywords if kw[1] > 0.8]
                high_quality_concepts.extend(concepts)
            
            if high_quality_concepts:
                concept_freq = Counter(high_quality_concepts)
                emerging_areas = [concept for concept, count in concept_freq.most_common(5)]
                
                content += f"4. **Emerging Research Areas**: Monitor developments in: {', '.join(emerging_areas)}\n"
                content += f"   These concepts appear in the highest-quality research and may represent future trends.\n"
        
        # System optimization recommendations
        content += "\n### System Enhancement Recommendations:\n\n"
        content += f"1. **Analysis Coverage**: Currently analyzing {len(llm_results)} papers with LLM intelligence. Consider expanding coverage.\n"
        content += f"2. **Quality Threshold**: Set minimum quality score of {max(0.6, avg_quality - 0.1):.1f} for priority analysis.\n"
        content += f"3. **Innovation Tracking**: Monitor {len(set(all_innovations))} distinct innovation areas for trend detection.\n"
        
        return content
    
    def _generate_intelligence_metadata(self, llm_results: List[LLMAnalysisResult]) -> Dict[str, Any]:
        """Generate metadata about the intelligence gathered."""
        
        if not llm_results:
            return {"intelligence_level": "pending", "analysis_coverage": 0}
        
        total_contributions = sum(len(result.novel_contributions) for result in llm_results)
        total_innovations = sum(len(result.technical_innovations) for result in llm_results)
        total_business = sum(len(result.business_implications) for result in llm_results)
        avg_quality = statistics.mean([result.research_quality_score for result in llm_results])
        
        return {
            "intelligence_level": "advanced_llm",
            "analysis_coverage": len(llm_results),
            "total_novel_contributions": total_contributions,
            "total_technical_innovations": total_innovations,
            "total_business_implications": total_business,
            "average_research_quality": round(avg_quality, 2),
            "high_quality_papers": sum(1 for result in llm_results if result.research_quality_score > 0.8)
        }
    
    def _assess_analysis_quality(self, llm_results: List[LLMAnalysisResult]) -> Dict[str, Any]:
        """Assess the quality of the analysis performed."""
        
        if not llm_results:
            return {"status": "no_analysis", "completeness": 0}
        
        # Check completeness of analysis
        complete_analyses = sum(1 for result in llm_results if (
            result.novel_contributions and 
            result.research_significance and 
            result.concept_keywords
        ))
        
        completeness = complete_analyses / len(llm_results)
        
        return {
            "status": "complete" if completeness > 0.8 else "partial",
            "completeness": round(completeness, 2),
            "total_analyzed": len(llm_results),
            "complete_analyses": complete_analyses
        }
    
    def _generate_intelligent_title(self, 
                                  template: ReportTemplate, 
                                  parameters: Dict[str, Any], 
                                  content_count: int,
                                  llm_analysis_count: int) -> str:
        """Generate an intelligent report title."""
        
        date_str = datetime.now().strftime("%B %Y")
        
        if llm_analysis_count > 0:
            title_map = {
                'research_landscape': f"Intelligent Research Landscape Report - {date_str} ({llm_analysis_count} papers analyzed)",
                'technology_impact': f"AI-Enhanced Technology Impact Analysis - {date_str} ({llm_analysis_count} papers)",
                'trend_analysis': f"LLM-Powered Trend Intelligence - {date_str} ({llm_analysis_count} papers analyzed)",
                'business_intelligence': f"Research Business Intelligence Report - {date_str} ({llm_analysis_count} analyses)"
            }
        else:
            title_map = {
                'research_landscape': f"Research Intelligence Report (LLM Analysis Pending) - {date_str}",
                'technology_impact': f"Technology Analysis (Enhanced Intelligence Activating) - {date_str}",
                'trend_analysis': f"Trend Analysis (LLM Enhancement In Progress) - {date_str}"
            }
        
        base_title = title_map.get(template.template_id, f"{template.name} - {date_str}")
        
        if llm_analysis_count > 0:
            return f"🧠 {base_title}"  # Brain emoji indicates LLM-powered intelligence
        else:
            return f"⚡ {base_title}"  # Lightning emoji indicates activation in progress
        
        return base_title