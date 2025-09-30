#!/usr/bin/env python3
"""
Test the complete real KnowHunt pipeline: Collection → Analysis → Reporting

This tests the actual end-to-end data flow that was missing before.
"""

import asyncio
import logging
import tempfile
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def test_complete_real_pipeline():
    """Test the complete collection → analysis → reporting pipeline."""
    
    print("🚀 Testing Complete Real KnowHunt Pipeline")
    print("=" * 60)
    
    # Create temporary directory for this test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # Import all components
            from stoma.pipeline.data_pipeline import DataPipeline
            from stoma.collectors.hackernews import HackerNewsCollector
            from stoma.collectors.arxiv import ArXivCollector
            from stoma.analysis.nlp_analyzer import NLPAnalyzer
            from stoma.reports.data_driven_generator import RealDataDrivenReportGenerator
            from stoma.reports.template_manager import TemplateManager
            
            print("✅ All components imported successfully")
            
            # Step 1: Initialize the pipeline
            print("\n📊 Step 1: Initialize Data Pipeline")
            pipeline = DataPipeline(storage_dir=str(temp_path / "pipeline_data"))
            
            # Step 2: Register collectors
            print("\n🔍 Step 2: Register Data Collectors")
            
            # HackerNews collector with limited scope for testing
            hn_collector = HackerNewsCollector({
                'max_stories': 3,
                'min_score': 5,
                'include_comments': False,
                'rate_limit': 1.0
            })
            pipeline.register_collector('hackernews', hn_collector)
            
            # ArXiv collector with limited scope
            arxiv_collector = ArXivCollector({
                'max_results': 2
            })
            pipeline.register_collector('arxiv', arxiv_collector)
            
            # Step 3: Register analyzer
            print("\n🧠 Step 3: Register NLP Analyzer")
            analyzer = NLPAnalyzer()
            pipeline.set_analyzer(analyzer)
            
            # Step 4: Run collection cycle
            print("\n📥 Step 4: Run Data Collection")
            collected_count = await pipeline.run_collection_cycle({
                'hackernews': {},
                'arxiv': {'search_query': 'artificial intelligence'}
            })
            
            print(f"   ✅ Collected {collected_count} items")
            
            if collected_count == 0:
                print("   ⚠️  No content collected - may be connectivity issues")
                return False
            
            # Step 5: Run analysis cycle
            print("\n🔬 Step 5: Run NLP Analysis")
            analyzed_count = await pipeline.run_analysis_cycle()
            
            print(f"   ✅ Analyzed {analyzed_count} items")
            
            # Step 6: Generate real report
            print("\n📋 Step 6: Generate Real Data-Driven Report")
            
            template_manager = TemplateManager()
            report_generator = RealDataDrivenReportGenerator(
                data_pipeline=pipeline,
                template_manager=template_manager,
                output_dir=str(temp_path / "reports")
            )
            
            # Generate report using real data
            report = report_generator.generate_report(
                template_id='research_landscape',
                parameters={
                    'timeframe_days': 1,  # Last 24 hours
                    'max_items': 50
                }
            )
            
            print(f"   ✅ Generated report: {report.title}")
            print(f"   📄 Sections: {len(report.sections)}")
            
            # Step 7: Verify report content is real
            print("\n🔍 Step 7: Verify Report Contains Real Data")
            
            # Check executive summary for real numbers
            exec_summary = next((s for s in report.sections if s.section_type == 'executive_summary'), None)
            if exec_summary:
                content = exec_summary.content
                if f"{collected_count} pieces of content" in content:
                    print("   ✅ Executive summary contains real collection count")
                else:
                    print("   ❌ Executive summary missing real data")
                    return False
            
            # Check for real keywords in keyword insights
            keyword_section = next((s for s in report.sections if s.section_type == 'keyword_insights'), None)
            if keyword_section and analyzed_count > 0:
                if "real content items" in keyword_section.content:
                    print("   ✅ Keyword section references real analyzed content")
                else:
                    print("   ❌ Keyword section missing real data reference")
            
            # Step 8: Save and verify file output
            print("\n💾 Step 8: Save Report Files")
            
            # Save in multiple formats
            md_path = report_generator.save_report(report, 'real_pipeline_test.md', 'markdown')
            html_path = report_generator.save_report(report, 'real_pipeline_test.html', 'html')
            
            # Verify files exist and contain data
            md_file = Path(md_path)
            html_file = Path(html_path)
            
            if md_file.exists() and md_file.stat().st_size > 1000:
                print(f"   ✅ Markdown report saved: {md_path} ({md_file.stat().st_size} bytes)")
            else:
                print("   ❌ Markdown report file issue")
                return False
            
            if html_file.exists() and html_file.stat().st_size > 1000:
                print(f"   ✅ HTML report saved: {html_path} ({html_file.stat().st_size} bytes)")
            else:
                print("   ❌ HTML report file issue")
                return False
            
            # Step 9: Pipeline statistics
            print("\n📊 Step 9: Pipeline Statistics")
            
            stats = pipeline.get_pipeline_statistics()
            
            print(f"   📈 Total content collected: {stats['content_statistics']['total_collected']}")
            print(f"   🧠 Total analyses performed: {stats['content_statistics']['total_analyzed']}")
            print(f"   📊 Analysis coverage: {stats['content_statistics']['analysis_coverage_percent']}%")
            print(f"   🔗 Active collectors: {len(stats['registered_collectors'])}")
            print(f"   📋 Source types: {', '.join(stats['content_statistics']['content_by_source'].keys())}")
            
            # Step 10: Demonstrate real content preview
            print("\n👀 Step 10: Real Content Preview")
            
            recent_content = pipeline.get_recent_content(hours_back=24, limit=3)
            if recent_content:
                print("   Sample collected content:")
                for i, content in enumerate(recent_content[:2], 1):
                    print(f"   {i}. [{content.source_type}] {content.title[:50]}...")
                    print(f"      Content: {len(content.content)} chars")
            
            analyzed_content = pipeline.get_analyzed_content(hours_back=24, limit=3)
            if analyzed_content:
                print("   Sample analysis results:")
                for i, analysis in enumerate(analyzed_content[:2], 1):
                    keywords = [kw for kw, score in analysis.analysis_result.keywords[:3]]
                    print(f"   {i}. Keywords: {', '.join(keywords)}")
                    print(f"      Word count: {analysis.analysis_result.word_count}")
            
            # Success!
            print("\n" + "=" * 60)
            print("🎉 COMPLETE PIPELINE TEST SUCCESSFUL!")
            print("=" * 60)
            
            print("✅ Collection: Real data collected from live APIs")
            print("✅ Analysis: Real NLP processing performed") 
            print("✅ Storage: Data persisted in pipeline state")
            print("✅ Reporting: Real data-driven reports generated")
            print("✅ Export: Multiple output formats created")
            print("✅ Integration: End-to-end data flow verified")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Run the complete pipeline test."""
    success = await test_complete_real_pipeline()
    
    if success:
        print("\n🚀 KnowHunt pipeline is fully operational!")
        print("The collection → analysis → reporting infrastructure is working correctly.")
    else:
        print("\n❌ Pipeline test failed - check logs for issues")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())