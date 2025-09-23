#!/usr/bin/env python3
"""
Test the enhanced KnowHunt pipeline with content enrichment.

This tests: Collection → Enrichment → Analysis → Reporting
where enrichment downloads full content instead of just metadata.
"""

import asyncio
import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def test_enhanced_pipeline():
    """Test the enhanced pipeline with content enrichment."""
    
    print("🚀 Testing Enhanced KnowHunt Pipeline with Content Enrichment")
    print("=" * 70)
    
    # Create temporary directory for this test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # Import all components
            from knowhunt.pipeline.data_pipeline import DataPipeline
            from knowhunt.collectors.hackernews import HackerNewsCollector
            from knowhunt.collectors.arxiv import ArXivCollector
            from knowhunt.analysis.nlp_analyzer import NLPAnalyzer
            from knowhunt.enrichment.content_enricher import ContentEnricher
            from knowhunt.reports.data_driven_generator import RealDataDrivenReportGenerator
            from knowhunt.reports.template_manager import TemplateManager
            
            print("✅ All enhanced components imported successfully")
            
            # Step 1: Initialize enhanced pipeline
            print("\n📊 Step 1: Initialize Enhanced Data Pipeline")
            pipeline = DataPipeline(storage_dir=str(temp_path / "enhanced_pipeline"))
            
            # Step 2: Register collectors
            print("\n🔍 Step 2: Register Data Collectors")
            
            # Focus on ArXiv for PDF enrichment testing
            arxiv_collector = ArXivCollector({'max_results': 2})
            pipeline.register_collector('arxiv', arxiv_collector)
            
            # HackerNews for web scraping testing  
            hn_collector = HackerNewsCollector({
                'max_stories': 2,
                'min_score': 5,
                'include_comments': False,
                'rate_limit': 1.0
            })
            pipeline.register_collector('hackernews', hn_collector)
            
            # Step 3: Register content enricher
            print("\n🔧 Step 3: Register Content Enricher")
            
            # Check available enrichment methods
            enricher = ContentEnricher(
                enable_web_scraping=True,
                enable_pdf_extraction=True,
                web_scraper_config={
                    'rate_limit_delay': 2.0,  # Be respectful
                    'timeout': 30
                },
                pdf_extractor_config={
                    'max_file_size': 20 * 1024 * 1024,  # 20MB limit
                    'timeout': 45
                }
            )
            
            pipeline.set_content_enricher(enricher)
            
            # Show enrichment capabilities
            if enricher.pdf_extractor:
                pdf_stats = enricher.pdf_extractor.get_extraction_stats()
                print(f"   📄 PDF extraction available: {pdf_stats['primary_method']}")
            
            print(f"   🌐 Web scraping enabled: {enricher.enable_web_scraping}")
            
            # Step 4: Register analyzer
            print("\n🧠 Step 4: Register NLP Analyzer")
            analyzer = NLPAnalyzer()
            pipeline.set_analyzer(analyzer)
            
            # Step 5: Run collection cycle
            print("\n📥 Step 5: Run Data Collection")
            collected_count = await pipeline.run_collection_cycle({
                'arxiv': {'search_query': 'machine learning'},
                'hackernews': {}
            })
            
            print(f"   ✅ Collected {collected_count} items")
            
            if collected_count == 0:
                print("   ⚠️  No content collected - may be connectivity issues")
                return False
            
            # Show what we collected (before enrichment)
            print("\n📋 Pre-Enrichment Content Preview:")
            recent_content = pipeline.get_recent_content(hours_back=1, limit=3)
            for i, content in enumerate(recent_content, 1):
                print(f"   {i}. [{content.source_type}] {content.title[:50]}...")
                print(f"      Original length: {len(content.content)} characters")
                if content.url:
                    print(f"      URL: {content.url}")
            
            # Step 6: Run enrichment cycle (THE KEY NEW STEP)
            print("\n🔧 Step 6: Run Content Enrichment")
            enriched_count = await pipeline.run_enrichment_cycle()
            
            print(f"   ✅ Enriched {enriched_count} items")
            
            # Show enrichment results
            if enriched_count > 0:
                print("\n📈 Enrichment Results:")
                enrichment_stats = enricher.get_enrichment_statistics()
                print(f"   Success rate: {enrichment_stats['success_rate']:.1%}")
                print(f"   Average enhancement ratio: {enrichment_stats.get('average_enhancement_ratio', 1.0):.1f}x")
                print(f"   PDF extractions: {enrichment_stats['pdf_extractions']}")
                print(f"   Web scrapes: {enrichment_stats['web_scrapes']}")
                
                # Show specific enrichment results
                for i, enrichment in enumerate(pipeline.state.enriched_items, 1):
                    if enrichment.enrichment_successful:
                        print(f"   {i}. {enrichment.enrichment_type.upper()}: {enrichment.original_length} → {enrichment.enriched_length} chars ({enrichment.enhancement_ratio:.1f}x)")
                        if enrichment.enrichment_metadata:
                            if 'extraction_method' in enrichment.enrichment_metadata:
                                print(f"      PDF method: {enrichment.enrichment_metadata['extraction_method']}")
                            if 'scraped_title' in enrichment.enrichment_metadata:
                                print(f"      Scraped: {enrichment.enrichment_metadata['scraped_title'][:40]}...")
            
            # Step 7: Run analysis on enriched content
            print("\n🔬 Step 7: Run NLP Analysis (on enriched content)")
            analyzed_count = await pipeline.run_analysis_cycle()
            
            print(f"   ✅ Analyzed {analyzed_count} items")
            
            # Step 8: Generate report with enriched data
            print("\n📋 Step 8: Generate Enhanced Report")
            
            template_manager = TemplateManager()
            report_generator = RealDataDrivenReportGenerator(
                data_pipeline=pipeline,
                template_manager=template_manager,
                output_dir=str(temp_path / "reports")
            )
            
            report = report_generator.generate_report(
                template_id='research_landscape',
                parameters={'timeframe_days': 1, 'max_items': 20}
            )
            
            print(f"   ✅ Generated enhanced report: {report.title}")
            print(f"   📄 Sections: {len(report.sections)}")
            
            # Step 9: Save and analyze the enhanced report
            print("\n💾 Step 9: Save Enhanced Report")
            
            md_path = report_generator.save_report(report, 'enhanced_report.md', 'markdown')
            print(f"   ✅ Saved enhanced report: {md_path}")
            
            # Step 10: Compare content quality
            print("\n🔍 Step 10: Content Quality Analysis")
            
            # Show content length improvements
            total_original_length = sum(len(item.content) for item in pipeline.state.collected_items)
            
            enriched_content_lengths = []
            for enrichment in pipeline.state.enriched_items:
                if enrichment.enrichment_successful:
                    enriched_content_lengths.append(enrichment.enriched_length)
                else:
                    enriched_content_lengths.append(enrichment.original_length)
            
            total_enriched_length = sum(enriched_content_lengths) if enriched_content_lengths else total_original_length
            
            improvement_ratio = total_enriched_length / total_original_length if total_original_length > 0 else 1.0
            
            print(f"   📊 Content volume improvement: {improvement_ratio:.1f}x")
            print(f"   📏 Original total: {total_original_length:,} characters")
            print(f"   📏 Enriched total: {total_enriched_length:,} characters")
            
            # Step 11: Pipeline statistics
            print("\n📈 Step 11: Enhanced Pipeline Statistics")
            
            stats = pipeline.get_pipeline_statistics()
            print(f"   📦 Total collected: {stats['content_statistics']['total_collected']}")
            print(f"   🔧 Total enriched: {len(pipeline.state.enriched_items)}")
            print(f"   🧠 Total analyzed: {stats['content_statistics']['total_analyzed']}")
            print(f"   📊 Analysis coverage: {stats['content_statistics']['analysis_coverage_percent']}%")
            
            # Success criteria
            success = (
                collected_count > 0 and
                analyzed_count > 0 and 
                improvement_ratio > 1.0  # Must have some content improvement
            )
            
            if success:
                print("\n" + "=" * 70)
                print("🎉 ENHANCED PIPELINE TEST SUCCESSFUL!")
                print("=" * 70)
                print("✅ Collection: Real data collected from live APIs")
                print("✅ Enrichment: Content enhanced with full-text extraction")
                print("✅ Analysis: Real NLP on enriched content")
                print("✅ Reporting: Enhanced reports with substantial content")
                print("✅ Integration: Complete enriched pipeline working")
                
                return True
            else:
                print("\n❌ Enhanced pipeline test failed - insufficient improvement")
                return False
                
        except Exception as e:
            print(f"\n❌ Enhanced pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Run the enhanced pipeline test."""
    success = await test_enhanced_pipeline()
    
    if success:
        print("\n🚀 KnowHunt enhanced pipeline is fully operational!")
        print("The collection → enrichment → analysis → reporting flow is working.")
        print("Reports now contain substantial, meaningful content.")
    else:
        print("\n❌ Enhanced pipeline test failed - check logs for issues")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())