#!/usr/bin/env python3
"""
Integration tests for Phase 3: Enhanced Collection
Tests the new Reddit and HackerNews collectors with the NLP pipeline.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_reddit_collector():
    """Test Reddit collector functionality."""
    print("\n🔍 Testing Reddit Collector...")
    print("=" * 50)
    
    try:
        from stoma.collectors.reddit import RedditCollector
        
        # Initialize with test configuration
        config = {
            "subreddits": ["MachineLearning", "technology"],  # Smaller list for testing
            "max_posts": 5,  # Limit for testing
            "include_comments": True,
            "max_comments": 3,
            "rate_limit": 0.5  # Slower rate for testing
        }
        
        collector = RedditCollector(config)
        
        # Test 1: Health check
        print("1. Testing Reddit API health check...")
        is_healthy = await collector.health_check()
        print(f"   ✓ Health check: {'PASS' if is_healthy else 'FAIL'}")
        
        if not is_healthy:
            print("   ⚠️  Reddit API not accessible - skipping collection test")
            return False
        
        # Test 2: Basic collection
        print("2. Testing Reddit collection...")
        results = []
        
        async for result in collector.collect():
            results.append(result)
            print(f"   📄 Collected: {result.data.get('title', 'N/A')[:60]}...")
            
            # Limit results for testing
            if len(results) >= 5:
                break
        
        print(f"   ✓ Collected {len(results)} items from Reddit")
        
        # Test 3: Analyze collected data structure
        if results:
            sample_result = results[0]
            print("3. Analyzing Reddit data structure...")
            print(f"   - Source ID: {sample_result.source_id}")
            print(f"   - Source Type: {sample_result.source_type}")
            print(f"   - Data Keys: {list(sample_result.data.keys())}")
            print(f"   - Has Keywords: {'keywords' in sample_result.data}")
            print(f"   - Content Length: {len(sample_result.data.get('content', ''))}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"   ✗ Reddit collector test failed: {e}")
        return False


async def test_hackernews_collector():
    """Test HackerNews collector functionality."""
    print("\n🔍 Testing HackerNews Collector...")
    print("=" * 50)
    
    try:
        from stoma.collectors.hackernews import HackerNewsCollector
        
        # Initialize with test configuration
        config = {
            "story_types": ["topstories"],  # Just top stories for testing
            "max_stories": 5,  # Limit for testing
            "include_comments": True,
            "max_comments_per_story": 2,
            "min_score": 5,  # Lower threshold for testing
            "rate_limit": 0.5  # Slower rate for testing
        }
        
        collector = HackerNewsCollector(config)
        
        # Test 1: Health check
        print("1. Testing HackerNews API health check...")
        is_healthy = await collector.health_check()
        print(f"   ✓ Health check: {'PASS' if is_healthy else 'FAIL'}")
        
        if not is_healthy:
            print("   ⚠️  HackerNews API not accessible - skipping collection test")
            return False
        
        # Test 2: Basic collection
        print("2. Testing HackerNews collection...")
        results = []
        
        async for result in collector.collect():
            results.append(result)
            print(f"   📄 Collected: {result.data.get('title', 'N/A')[:60]}...")
            
            # Limit results for testing
            if len(results) >= 5:
                break
        
        print(f"   ✓ Collected {len(results)} items from HackerNews")
        
        # Test 3: Analyze collected data structure
        if results:
            sample_result = results[0]
            print("3. Analyzing HackerNews data structure...")
            print(f"   - Source ID: {sample_result.source_id}")
            print(f"   - Source Type: {sample_result.source_type}")
            print(f"   - Data Keys: {list(sample_result.data.keys())}")
            print(f"   - Has Keywords: {'keywords' in sample_result.data}")
            print(f"   - Content Length: {len(sample_result.data.get('content', ''))}")
            print(f"   - Story Type: {sample_result.data.get('story_type', 'N/A')}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"   ✗ HackerNews collector test failed: {e}")
        return False


async def test_nlp_integration():
    """Test integration of new collectors with NLP pipeline."""
    print("\n🔍 Testing NLP Pipeline Integration...")
    print("=" * 50)
    
    try:
        from stoma.analysis.nlp_analyzer import NLPAnalyzer
        from stoma.collectors.reddit import RedditCollector
        from stoma.collectors.hackernews import HackerNewsCollector
        
        # Initialize NLP analyzer
        analyzer = NLPAnalyzer()
        print("1. NLP Analyzer initialized ✓")
        
        # Test with Reddit data
        print("2. Testing Reddit → NLP integration...")
        reddit_config = {
            "subreddits": ["technology"],
            "max_posts": 2,
            "include_comments": False,
            "rate_limit": 0.5
        }
        
        reddit_collector = RedditCollector(reddit_config)
        reddit_results = []
        
        # Collect a few Reddit posts
        async for result in reddit_collector.collect():
            reddit_results.append(result)
            if len(reddit_results) >= 2:
                break
        
        # Analyze Reddit content with NLP
        reddit_analyses = []
        for result in reddit_results:
            if result.success and result.data.get('content'):
                content = result.data['content']
                analysis = analyzer.analyze(content)
                reddit_analyses.append({
                    'source': 'reddit',
                    'title': result.data.get('title', 'N/A')[:50],
                    'analysis': analysis
                })
                print(f"   ✓ Analyzed Reddit post: {len(analysis.keywords)} keywords, {analysis.word_count} words")
        
        # Test with HackerNews data
        print("3. Testing HackerNews → NLP integration...")
        hn_config = {
            "story_types": ["topstories"],
            "max_stories": 2,
            "include_comments": False,
            "min_score": 1,
            "rate_limit": 0.5
        }
        
        hn_collector = HackerNewsCollector(hn_config)
        hn_results = []
        
        # Collect a few HN stories
        async for result in hn_collector.collect():
            hn_results.append(result)
            if len(hn_results) >= 2:
                break
        
        # Analyze HN content with NLP
        hn_analyses = []
        for result in hn_results:
            if result.success and result.data.get('content'):
                content = result.data['content']
                analysis = analyzer.analyze(content)
                hn_analyses.append({
                    'source': 'hackernews',
                    'title': result.data.get('title', 'N/A')[:50],
                    'analysis': analysis
                })
                print(f"   ✓ Analyzed HN story: {len(analysis.keywords)} keywords, {analysis.word_count} words")
        
        # Summary
        total_analyses = len(reddit_analyses) + len(hn_analyses)
        print(f"\n4. Integration Summary:")
        print(f"   - Reddit posts analyzed: {len(reddit_analyses)}")
        print(f"   - HackerNews stories analyzed: {len(hn_analyses)}")
        print(f"   - Total NLP analyses: {total_analyses}")
        
        # Show sample analysis results
        if reddit_analyses:
            sample = reddit_analyses[0]
            print(f"   - Sample Reddit keywords: {[k[0] for k in sample['analysis'].keywords[:5]]}")
        
        if hn_analyses:
            sample = hn_analyses[0]
            print(f"   - Sample HN keywords: {[k[0] for k in sample['analysis'].keywords[:5]]}")
        
        return total_analyses > 0
        
    except Exception as e:
        print(f"   ✗ NLP integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_report_generation():
    """Test report generation with new data sources."""
    print("\n🔍 Testing Report Generation with New Sources...")
    print("=" * 50)
    
    try:
        from stoma.reports.base_generator import DataDrivenReportGenerator
        from stoma.reports.template_manager import TemplateManager
        
        # Initialize report generator
        template_manager = TemplateManager()
        generator = DataDrivenReportGenerator(template_manager=template_manager)
        
        print("1. Report generator initialized ✓")
        
        # Generate a sample report
        parameters = {
            'timeframe_days': 7,
            'format': 'markdown'
        }
        
        templates = generator.get_available_templates()
        if templates:
            template_id = templates[0].template_id
            report = generator.generate_report(template_id, parameters)
            
            print(f"2. Generated report: {report.title}")
            print(f"   - Sections: {len(report.sections)}")
            print(f"   - Template: {template_id}")
            
            # Test export
            from stoma.reports.export_formats import MarkdownExporter
            exporter = MarkdownExporter()
            content = exporter.export(report)
            
            print(f"3. Exported report: {len(content)} characters")
            
            return True
        else:
            print("   ✗ No templates available")
            return False
            
    except Exception as e:
        print(f"   ✗ Report generation test failed: {e}")
        return False


async def test_source_diversity():
    """Test that we can collect from multiple diverse sources."""
    print("\n🔍 Testing Multi-Source Collection...")
    print("=" * 50)
    
    sources_tested = {
        'reddit': False,
        'hackernews': False,
        'existing_sources': False
    }
    
    try:
        # Test Reddit
        from stoma.collectors.reddit import RedditCollector
        reddit_config = {"subreddits": ["technology"], "max_posts": 1, "rate_limit": 0.5}
        reddit_collector = RedditCollector(reddit_config)
        
        reddit_healthy = await reddit_collector.health_check()
        if reddit_healthy:
            async for result in reddit_collector.collect():
                if result.success:
                    sources_tested['reddit'] = True
                    print(f"1. ✓ Reddit source active - {result.source_type.value}")
                break
        
        # Test HackerNews
        from stoma.collectors.hackernews import HackerNewsCollector
        hn_config = {"max_stories": 1, "rate_limit": 0.5}
        hn_collector = HackerNewsCollector(hn_config)
        
        hn_healthy = await hn_collector.health_check()
        if hn_healthy:
            async for result in hn_collector.collect():
                if result.success:
                    sources_tested['hackernews'] = True
                    print(f"2. ✓ HackerNews source active - {result.source_type.value}")
                break
        
        # Test existing source (ArXiv)
        from stoma.collectors.arxiv import ArXivCollector
        arxiv_config = {"max_results": 1}
        arxiv_collector = ArXivCollector(arxiv_config)
        
        arxiv_healthy = await arxiv_collector.health_check()
        if arxiv_healthy:
            async for result in arxiv_collector.collect():
                if result.success:
                    sources_tested['existing_sources'] = True
                    print(f"3. ✓ ArXiv source active - {result.source_type.value}")
                break
        
        active_sources = sum(sources_tested.values())
        print(f"\nSource Diversity Summary:")
        print(f"   - Active sources: {active_sources}/3")
        print(f"   - Reddit: {'✓' if sources_tested['reddit'] else '✗'}")
        print(f"   - HackerNews: {'✓' if sources_tested['hackernews'] else '✗'}")
        print(f"   - Existing (ArXiv): {'✓' if sources_tested['existing_sources'] else '✗'}")
        
        return active_sources >= 2  # At least 2 sources should work
        
    except Exception as e:
        print(f"   ✗ Multi-source test failed: {e}")
        return False


async def main():
    """Run all Phase 3 integration tests."""
    print("🚀 KnowHunt Phase 3: Enhanced Collection Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Reddit Collector", test_reddit_collector),
        ("HackerNews Collector", test_hackernews_collector),
        ("NLP Integration", test_nlp_integration),
        ("Report Generation", test_report_generation),
        ("Source Diversity", test_source_diversity)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🏆 PHASE 3 INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Phase 3 Enhanced Collection is ready!")
        return True
    elif passed_tests >= total_tests * 0.8:  # 80% threshold
        print("⚠️  Most tests passed - Phase 3 is mostly functional with some issues")
        return True
    else:
        print("❌ Multiple test failures - Phase 3 needs more work")
        return False


if __name__ == "__main__":
    asyncio.run(main())