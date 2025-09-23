"""
Real data-driven report generator that uses actual collected and analyzed data.

This replaces the mock report generator with one that produces reports
based on real data from the pipeline.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import Counter, defaultdict
import statistics

from ..pipeline.data_pipeline import DataPipeline, StoredContent, AnalyzedContent
from .base_generator import ReportGenerator, GeneratedReport, ReportSection, ReportTemplate

logger = logging.getLogger(__name__)


class RealDataDrivenReportGenerator(ReportGenerator):
    """Report generator that creates reports from actual pipeline data."""
    
    def __init__(self, 
                 data_pipeline: DataPipeline,
                 template_manager: Optional[Any] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize with data pipeline.
        
        Args:
            data_pipeline: The main data pipeline containing collected/analyzed data
            template_manager: Template management system
            output_dir: Directory for saving reports
        """
        super().__init__(template_manager, output_dir)
        self.pipeline = data_pipeline
    
    def generate_report(self, 
                       template_id: str,
                       parameters: Dict[str, Any],
                       report_id: Optional[str] = None) -> GeneratedReport:
        """Generate a report using actual data from the pipeline."""
        
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
        
        # Gather actual data from pipeline
        recent_content = self.pipeline.get_recent_content(
            hours_back=timeframe_hours,
            limit=parameters.get('max_items', 100)
        )
        
        analyzed_content = self.pipeline.get_analyzed_content(
            hours_back=timeframe_hours,
            limit=parameters.get('max_items', 100)
        )
        
        pipeline_stats = self.pipeline.get_pipeline_statistics()
        
        logger.info(f"Generating report with {len(recent_content)} content items and {len(analyzed_content)} analyses")
        
        # Generate sections using real data
        sections = []
        for section_def in template.sections:
            section = self._generate_section_with_data(
                section_def, 
                parameters, 
                recent_content, 
                analyzed_content, 
                pipeline_stats
            )
            if section:
                sections.append(section)
        
        # Create report metadata
        metadata = {
            'template_version': template.version,
            'generation_parameters': parameters,
            'data_summary': {
                'content_items_analyzed': len(recent_content),
                'nlp_analyses_performed': len(analyzed_content),
                'timeframe_hours': timeframe_hours,
                'source_types_covered': list(set(item.source_type for item in recent_content)),
            },
            'pipeline_statistics': pipeline_stats
        }
        
        report = GeneratedReport(
            report_id=report_id,
            title=self._generate_title(template, parameters, len(recent_content)),
            template_id=template_id,
            sections=sections,
            metadata=metadata,
            format_type=parameters.get('format', 'markdown')
        )
        
        logger.info(f"Generated real data-driven report: {report_id}")
        return report
    
    def _generate_section_with_data(self, 
                                  section_def: Dict[str, Any], 
                                  parameters: Dict[str, Any],
                                  content_items: List[StoredContent],
                                  analyzed_items: List[AnalyzedContent],
                                  pipeline_stats: Dict[str, Any]) -> Optional[ReportSection]:
        """Generate a report section using real data."""
        
        section_type = section_def['section_type']
        section_id = section_def['section_id']
        
        try:
            if section_type == 'executive_summary':
                content = self._generate_real_executive_summary(
                    parameters, content_items, analyzed_items, pipeline_stats
                )
            elif section_type == 'trend_analysis':
                content = self._generate_real_trend_analysis(analyzed_items)
            elif section_type == 'correlation_analysis':
                content = self._generate_real_correlation_analysis(analyzed_items)
            elif section_type == 'keyword_insights':
                content = self._generate_real_keyword_insights(analyzed_items)
            elif section_type == 'research_landscape':
                content = self._generate_real_research_landscape(content_items, analyzed_items)
            elif section_type == 'technology_impact':
                content = self._generate_real_technology_impact(content_items, analyzed_items)
            elif section_type == 'recommendations':
                content = self._generate_real_recommendations(analyzed_items, pipeline_stats)
            else:
                content = f"Section type '{section_type}' analysis not yet implemented with real data"
            
            return ReportSection(
                section_id=section_id,
                title=section_def['title'],
                content=content,
                section_type=section_type,
                order=section_def.get('order', 0),
                metadata={
                    'generated_from_real_data': True,
                    'data_items_used': len(content_items),
                    'analyses_used': len(analyzed_items)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating section {section_id}: {e}")
            return ReportSection(
                section_id=section_id,
                title=section_def['title'],
                content=f"Error generating {section_type}: {str(e)}",
                section_type=section_type,
                order=section_def.get('order', 0),
                metadata={'error': str(e)}
            )
    
    def _generate_real_executive_summary(self, 
                                       parameters: Dict[str, Any],
                                       content_items: List[StoredContent],
                                       analyzed_items: List[AnalyzedContent],
                                       pipeline_stats: Dict[str, Any]) -> str:
        """Generate executive summary from real data."""
        
        timeframe_days = parameters.get('timeframe_days', 7)
        
        # Calculate real statistics
        total_content = len(content_items)
        total_analyses = len(analyzed_items)
        source_types = list(set(item.source_type for item in content_items))
        
        # Content volume analysis
        avg_content_length = statistics.mean([len(item.content) for item in content_items]) if content_items else 0
        
        # Time distribution
        if content_items:
            latest_content = max(item.collected_at for item in content_items)
            earliest_content = min(item.collected_at for item in content_items)
            time_span = (latest_content - earliest_content).days
        else:
            time_span = 0
        
        summary = f"""## Executive Summary

This report analyzes **{total_content} pieces of content** collected over the past {timeframe_days} days from {len(source_types)} different data sources, with **{total_analyses} items** processed through NLP analysis.

### Key Data Points:
- **Content Volume**: {total_content:,} items collected
- **Analysis Coverage**: {total_analyses:,} items analyzed ({(total_analyses/total_content*100):.1f}% coverage)
- **Source Diversity**: {', '.join(source_types)}
- **Average Content Length**: {avg_content_length:.0f} characters
- **Time Span**: Content spans {time_span} days

### Collection Summary:
- **Data Sources Active**: {len(source_types)}
- **Pipeline Status**: {'Active' if pipeline_stats.get('analyzer_available') else 'Partial'}
- **Recent Activity**: {pipeline_stats.get('recent_activity', {}).get('collections_24h', 0)} collections in last 24h

### Analysis Scope:
- **NLP Processing**: Keyword extraction, sentiment analysis, entity recognition
- **Content Types**: {', '.join(source_types)}
- **Real-time Data**: All statistics based on actual collected content
"""
        
        return summary
    
    def _generate_real_trend_analysis(self, analyzed_items: List[AnalyzedContent]) -> str:
        """Generate trend analysis from real analyzed data."""
        
        if not analyzed_items:
            return "## Trend Analysis\n\nNo analyzed content available for trend detection."
        
        # Extract all keywords with their frequencies
        all_keywords = []
        keyword_by_time = defaultdict(list)
        
        for item in analyzed_items:
            item_time = item.analyzed_at.date()
            for keyword, score in item.analysis_result.keywords:
                all_keywords.append((keyword, score))
                keyword_by_time[item_time].append((keyword, score))
        
        # Find most frequent keywords
        keyword_counts = Counter([kw for kw, score in all_keywords])
        top_keywords = keyword_counts.most_common(10)
        
        # Calculate average scores for top keywords
        keyword_scores = defaultdict(list)
        for keyword, score in all_keywords:
            keyword_scores[keyword].append(score)
        
        content = "## Trend Analysis\n\n"
        
        if top_keywords:
            content += "### Most Frequent Keywords\n\n"
            content += "Based on analysis of actual collected content:\n\n"
            
            for i, (keyword, count) in enumerate(top_keywords, 1):
                avg_score = statistics.mean(keyword_scores[keyword])
                content += f"{i}. **{keyword}** - {count} occurrences (avg score: {avg_score:.3f})\n"
            
            content += f"\n*Analysis based on {len(analyzed_items)} real content items*\n\n"
        
        # Time-based trending (simple version)
        if len(keyword_by_time) > 1:
            content += "### Temporal Patterns\n\n"
            
            dates = sorted(keyword_by_time.keys())
            if len(dates) >= 2:
                recent_date = dates[-1]
                previous_date = dates[-2] if len(dates) > 1 else dates[0]
                
                recent_keywords = set([kw for kw, score in keyword_by_time[recent_date]])
                previous_keywords = set([kw for kw, score in keyword_by_time[previous_date]])
                
                emerging = recent_keywords - previous_keywords
                if emerging:
                    content += f"**Emerging topics** (appeared on {recent_date}):\n"
                    for keyword in list(emerging)[:5]:
                        content += f"- {keyword}\n"
                    content += "\n"
        
        if not top_keywords:
            content += "No significant keyword trends detected in analyzed content.\n"
        
        return content
    
    def _generate_real_correlation_analysis(self, analyzed_items: List[AnalyzedContent]) -> str:
        """Generate correlation analysis from real data."""
        
        if len(analyzed_items) < 2:
            return "## Correlation Analysis\n\nInsufficient data for correlation analysis (need at least 2 analyzed items)."
        
        content = "## Correlation Analysis\n\n"
        
        # Group by source type for cross-source analysis
        by_source = defaultdict(list)
        for item in analyzed_items:
            source_type = item.analysis_metadata.get('source_type', 'unknown')
            by_source[source_type].append(item)
        
        content += f"### Cross-Source Analysis\n\n"
        content += f"Analyzed content from {len(by_source)} different source types:\n\n"
        
        for source_type, items in by_source.items():
            # Get common keywords for this source
            source_keywords = []
            for item in items:
                source_keywords.extend([kw for kw, score in item.analysis_result.keywords])
            
            keyword_counts = Counter(source_keywords)
            top_source_keywords = keyword_counts.most_common(3)
            
            content += f"**{source_type.title()}** ({len(items)} items):\n"
            if top_source_keywords:
                content += f"- Common themes: {', '.join([kw for kw, count in top_source_keywords])}\n"
            content += f"- Average content length: {statistics.mean([len(item.analysis_result.summary) for item in items]):.0f} chars\n"
            content += "\n"
        
        # Find keywords that appear across multiple sources
        if len(by_source) > 1:
            content += "### Cross-Source Keywords\n\n"
            
            source_keyword_sets = {}
            for source_type, items in by_source.items():
                keywords = set()
                for item in items:
                    keywords.update([kw for kw, score in item.analysis_result.keywords])
                source_keyword_sets[source_type] = keywords
            
            # Find intersections
            source_types = list(source_keyword_sets.keys())
            for i, source1 in enumerate(source_types):
                for source2 in source_types[i+1:]:
                    intersection = source_keyword_sets[source1] & source_keyword_sets[source2]
                    if intersection:
                        content += f"**{source1} ↔ {source2}**: {', '.join(list(intersection)[:5])}\n"
        
        content += f"\n*Analysis based on {len(analyzed_items)} real analyzed items*\n"
        
        return content
    
    def _generate_real_keyword_insights(self, analyzed_items: List[AnalyzedContent]) -> str:
        """Generate keyword insights from real analyzed data."""
        
        if not analyzed_items:
            return "## Key Research Terms\n\nNo analyzed content available for keyword analysis."
        
        # Collect all keywords with scores and metadata
        keyword_data = []
        for item in analyzed_items:
            source_type = item.analysis_metadata.get('source_type', 'unknown')
            for keyword, score in item.analysis_result.keywords:
                keyword_data.append({
                    'keyword': keyword,
                    'score': score,
                    'source_type': source_type,
                    'content_id': item.content_id
                })
        
        # Aggregate keyword statistics
        keyword_stats = defaultdict(lambda: {'scores': [], 'sources': set(), 'count': 0})
        
        for kw_data in keyword_data:
            keyword = kw_data['keyword']
            keyword_stats[keyword]['scores'].append(kw_data['score'])
            keyword_stats[keyword]['sources'].add(kw_data['source_type'])
            keyword_stats[keyword]['count'] += 1
        
        # Calculate final statistics and sort
        keyword_analysis = []
        for keyword, stats in keyword_stats.items():
            keyword_analysis.append({
                'keyword': keyword,
                'frequency': stats['count'],
                'avg_score': statistics.mean(stats['scores']),
                'max_score': max(stats['scores']),
                'source_count': len(stats['sources']),
                'sources': list(stats['sources'])
            })
        
        # Sort by frequency first, then by average score
        keyword_analysis.sort(key=lambda x: (x['frequency'], x['avg_score']), reverse=True)
        
        content = "## Key Research Terms\n\n"
        content += f"Analysis of keywords extracted from {len(analyzed_items)} real content items:\n\n"
        
        # Top keywords by frequency
        content += "### Most Frequent Terms\n\n"
        for i, kw_data in enumerate(keyword_analysis[:15], 1):
            sources_str = ', '.join(kw_data['sources'])
            content += f"{i}. **{kw_data['keyword']}**\n"
            content += f"   - Frequency: {kw_data['frequency']} occurrences\n"
            content += f"   - Average relevance: {kw_data['avg_score']:.3f}\n"
            content += f"   - Sources: {sources_str}\n\n"
        
        # Cross-source keywords
        cross_source_keywords = [kw for kw in keyword_analysis if kw['source_count'] > 1]
        if cross_source_keywords:
            content += "### Cross-Source Terms\n\n"
            content += "Keywords appearing across multiple data sources:\n\n"
            
            for kw_data in cross_source_keywords[:10]:
                content += f"- **{kw_data['keyword']}** (in {kw_data['source_count']} source types)\n"
        
        # Source-specific insights
        source_keywords = defaultdict(list)
        for kw_data in keyword_analysis:
            for source in kw_data['sources']:
                source_keywords[source].append(kw_data['keyword'])
        
        if len(source_keywords) > 1:
            content += "\n### Source-Specific Insights\n\n"
            for source_type, keywords in source_keywords.items():
                content += f"**{source_type.title()}**: {', '.join(keywords[:5])}\n"
        
        return content
    
    def _generate_real_research_landscape(self, 
                                        content_items: List[StoredContent],
                                        analyzed_items: List[AnalyzedContent]) -> str:
        """Generate research landscape from real data."""
        
        content = "## Research Landscape\n\n"
        
        if not content_items:
            return content + "No content data available for landscape analysis."
        
        # Analyze content distribution
        source_distribution = Counter(item.source_type for item in content_items)
        
        content += "### Current Data Landscape\n\n"
        content += f"Analysis based on {len(content_items)} real content items:\n\n"
        
        for source_type, count in source_distribution.most_common():
            percentage = (count / len(content_items)) * 100
            content += f"- **{source_type.title()}**: {count} items ({percentage:.1f}%)\n"
        
        # Recent activity analysis
        if content_items:
            recent_cutoff = datetime.now() - timedelta(days=1)
            recent_items = [item for item in content_items if item.collected_at >= recent_cutoff]
            
            content += f"\n### Recent Activity (Last 24 Hours)\n\n"
            content += f"- New content items: {len(recent_items)}\n"
            
            if recent_items:
                recent_sources = Counter(item.source_type for item in recent_items)
                content += f"- Most active sources: {', '.join([src for src, count in recent_sources.most_common(3)])}\n"
        
        # Content characteristics
        if analyzed_items:
            content += f"\n### Content Characteristics\n\n"
            
            # Average metrics
            avg_word_count = statistics.mean([item.analysis_result.word_count for item in analyzed_items])
            avg_keywords = statistics.mean([len(item.analysis_result.keywords) for item in analyzed_items])
            
            content += f"- Average word count: {avg_word_count:.0f} words\n"
            content += f"- Average keywords per item: {avg_keywords:.1f}\n"
            
            # Sentiment distribution
            sentiments = [item.analysis_result.sentiment.get('sentiment_label', 'neutral') for item in analyzed_items]
            sentiment_dist = Counter(sentiments)
            
            content += f"- Sentiment distribution: "
            content += ", ".join([f"{sent} ({count})" for sent, count in sentiment_dist.most_common()])
            content += "\n"
        
        return content
    
    def _generate_real_technology_impact(self, 
                                       content_items: List[StoredContent],
                                       analyzed_items: List[AnalyzedContent]) -> str:
        """Generate technology impact assessment from real data."""
        
        content = "## Technology Impact Assessment\n\n"
        
        if not analyzed_items:
            return content + "No analyzed content available for impact assessment."
        
        # Look for technology-related keywords
        tech_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
            'neural network', 'algorithm', 'automation', 'robotics', 'quantum',
            'blockchain', 'cryptocurrency', 'cloud', 'api', 'software', 'hardware',
            'data science', 'analytics', 'python', 'javascript', 'react', 'tensorflow'
        ]
        
        # Find mentions of technology terms
        tech_mentions = defaultdict(int)
        tech_contexts = defaultdict(list)
        
        for item in analyzed_items:
            item_title = item.analysis_metadata.get('title', '')
            for keyword, score in item.analysis_result.keywords:
                if keyword.lower() in tech_keywords:
                    tech_mentions[keyword.lower()] += 1
                    tech_contexts[keyword.lower()].append({
                        'title': item_title[:50] + '...' if len(item_title) > 50 else item_title,
                        'score': score,
                        'source': item.analysis_metadata.get('source_type', 'unknown')
                    })
        
        if tech_mentions:
            content += "### Technology Themes in Real Data\n\n"
            content += f"Analysis of technology mentions in {len(analyzed_items)} analyzed items:\n\n"
            
            # Sort by frequency
            sorted_tech = sorted(tech_mentions.items(), key=lambda x: x[1], reverse=True)
            
            for tech_term, count in sorted_tech[:10]:
                content += f"**{tech_term.title()}**: {count} mentions\n"
                
                # Show sample contexts
                contexts = tech_contexts[tech_term][:3]
                for ctx in contexts:
                    content += f"  - \"{ctx['title']}\" ({ctx['source']}, relevance: {ctx['score']:.3f})\n"
                content += "\n"
        
        # Impact categories based on source types
        source_impact = defaultdict(list)
        for item in analyzed_items:
            source_type = item.analysis_metadata.get('source_type', 'unknown')
            tech_keywords_in_item = [kw for kw, score in item.analysis_result.keywords if kw.lower() in tech_keywords]
            if tech_keywords_in_item:
                source_impact[source_type].extend(tech_keywords_in_item)
        
        if source_impact:
            content += "### Impact by Information Source\n\n"
            
            for source_type, keywords in source_impact.items():
                keyword_counts = Counter(keywords)
                top_keywords = keyword_counts.most_common(5)
                
                content += f"**{source_type.title()}**: "
                content += f"{', '.join([kw for kw, count in top_keywords])}\n"
        
        if not tech_mentions:
            content += "No significant technology keywords detected in current dataset.\n"
        
        return content
    
    def _generate_real_recommendations(self, 
                                     analyzed_items: List[AnalyzedContent],
                                     pipeline_stats: Dict[str, Any]) -> str:
        """Generate recommendations based on real data analysis."""
        
        content = "## Data-Driven Recommendations\n\n"
        
        # Pipeline health recommendations
        content += "### System Optimization\n\n"
        
        total_collected = pipeline_stats.get('content_statistics', {}).get('total_collected', 0)
        total_analyzed = pipeline_stats.get('content_statistics', {}).get('total_analyzed', 0)
        
        if total_collected > 0:
            analysis_coverage = (total_analyzed / total_collected) * 100
            
            if analysis_coverage < 80:
                content += f"1. **Increase Analysis Coverage**: Currently {analysis_coverage:.1f}% of collected content is analyzed. Consider increasing analysis frequency.\n"
            else:
                content += f"1. **Analysis Coverage Good**: {analysis_coverage:.1f}% coverage is healthy.\n"
        
        recent_collections = pipeline_stats.get('recent_activity', {}).get('collections_24h', 0)
        if recent_collections == 0:
            content += "2. **Collection Activity**: No recent collections detected. Check data source connectivity.\n"
        elif recent_collections < 5:
            content += f"3. **Low Collection Volume**: Only {recent_collections} items collected in 24h. Consider expanding sources.\n"
        
        # Content-based recommendations
        if analyzed_items:
            content += "\n### Content Strategy\n\n"
            
            # Source diversity analysis
            source_types = set(item.analysis_metadata.get('source_type', 'unknown') for item in analyzed_items)
            
            if len(source_types) < 3:
                content += f"1. **Diversify Sources**: Currently analyzing {len(source_types)} source types. Consider adding more data sources.\n"
            
            # Keyword insights for recommendations
            all_keywords = []
            for item in analyzed_items:
                all_keywords.extend([kw for kw, score in item.analysis_result.keywords])
            
            if all_keywords:
                keyword_counts = Counter(all_keywords)
                top_keywords = keyword_counts.most_common(5)
                
                content += f"2. **Focus Areas**: Top themes in your data: {', '.join([kw for kw, count in top_keywords])}\n"
                content += "   Consider expanding collection in these areas for deeper insights.\n"
        
        # Technical recommendations
        content += "\n### Technical Improvements\n\n"
        
        active_collectors = len(pipeline_stats.get('registered_collectors', []))
        if active_collectors < 3:
            content += f"1. **Expand Collection**: {active_collectors} collectors active. Consider adding Reddit, HackerNews, or other sources.\n"
        
        if not pipeline_stats.get('analyzer_available'):
            content += "2. **Analysis Engine**: NLP analyzer not fully connected. Ensure analysis pipeline is operational.\n"
        
        content += f"\n*Recommendations based on analysis of {len(analyzed_items)} real data points*\n"
        
        return content
    
    def _generate_title(self, template: ReportTemplate, parameters: Dict[str, Any], content_count: int) -> str:
        """Generate report title with real data context."""
        timeframe = parameters.get('timeframe_days', 7)
        date_str = datetime.now().strftime("%B %Y")
        
        title_map = {
            'research_landscape': f"Research Intelligence Report - {date_str} ({content_count} items analyzed)",
            'technology_impact': f"Technology Impact Analysis - {date_str} ({content_count} items)",
            'trend_analysis': f"Trend Analysis Report ({timeframe} days, {content_count} items) - {date_str}",
            'correlation_report': f"Correlation Analysis - {date_str} ({content_count} items)"
        }
        
        return title_map.get(template.template_id, f"{template.name} - {date_str} ({content_count} items)")
    
    def get_available_templates(self) -> List[ReportTemplate]:
        """Get available report templates."""
        if self.template_manager:
            return self.template_manager.get_all_templates()
        return []