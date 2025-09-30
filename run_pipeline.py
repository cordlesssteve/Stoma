#!/usr/bin/env python3
"""
Simplified KnowHunt pipeline runner that bypasses CLI dependency issues.
"""

import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add the project to Python path
sys.path.insert(0, '/home/cordlesssteve/projects/Utility/RESEARCH/Stoma')

try:
    from stoma.collectors.arxiv import ArXivCollector
    from stoma.analysis.llm_analyzer import LLMAnalyzer
    from stoma.storage.report_manager import ReportStorageManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Some components may not be available")
    sys.exit(1)


async def run_pipeline(query: str, max_results: int = 10, model: str = "gemma2:2b"):
    """Run the complete pipeline: collect, analyze, store."""

    print(f"🚀 KnowHunt Pipeline: {query}")
    print(f"📊 Papers: {max_results} | Model: {model}")
    print("=" * 50)

    # Step 1: Collect from ArXiv
    print("🔍 Collecting papers from ArXiv...")

    arxiv_config = {
        "max_results": max_results,
        "rate_limit": 1.0
    }

    collector = ArXivCollector(arxiv_config)
    collected_papers = []

    try:
        async for result in collector.collect(search_query=query):
            if result.success:
                collected_papers.append(result)
                print(f"  ✓ {result.data.get('title', 'No title')[:80]}...")
            else:
                print(f"  ✗ Collection error: {result.error_message}")

        print(f"\n📚 Collected {len(collected_papers)} papers successfully")

    except Exception as e:
        print(f"❌ ArXiv collection failed: {e}")
        return

    if not collected_papers:
        print("❌ No papers collected - aborting pipeline")
        return

    # Step 2: Prepare content for LLM analysis
    print("\n🧠 Preparing for LLM analysis...")

    # Combine paper abstracts and titles for analysis
    combined_content = []
    for paper in collected_papers[:5]:  # Analyze top 5 papers
        title = paper.data.get('title', '')
        abstract = paper.data.get('summary', '')
        combined_content.append(f"Title: {title}\nAbstract: {abstract}")

    analysis_text = "\n\n---\n\n".join(combined_content)

    # Step 3: LLM Analysis
    print(f"🤖 Running LLM analysis with {model}...")

    try:
        analyzer = LLMAnalyzer(
            provider="ollama",
            model=model,
            max_tokens=1500,
            temperature=0.1
        )

        document_id = f"pipeline_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result = await analyzer.analyze_research_paper(
            text=analysis_text,
            title=f"Research Analysis: {query}",
            document_id=document_id
        )

        print(f"✅ Analysis complete - Quality Score: {result.research_quality_score:.2f}/10")

        # Display key results
        if result.novel_contributions:
            print(f"\n🔬 Novel Contributions Found: {len(result.novel_contributions)}")
            for i, contribution in enumerate(result.novel_contributions[:2], 1):
                print(f"  {i}. {contribution[:100]}...")

        if result.technical_innovations:
            print(f"\n⚡ Technical Innovations: {len(result.technical_innovations)}")

        if result.business_implications:
            print(f"\n💼 Business Implications: {len(result.business_implications)}")

    except Exception as e:
        print(f"❌ LLM analysis failed: {e}")
        print("This might be due to Ollama not running or model not available")
        return

    # Step 4: Store Results
    print(f"\n💾 Storing analysis results...")

    try:
        # Get usage statistics
        stats = analyzer.get_usage_statistics()

        # Create report data
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "input_text": analysis_text[:1000] + "..." if len(analysis_text) > 1000 else analysis_text,
            "provider": "ollama",
            "model": model,
            "query": query,
            "papers_analyzed": len(collected_papers),
            "analysis": {
                "research_quality_score": result.research_quality_score,
                "novel_contributions": result.novel_contributions,
                "technical_innovations": result.technical_innovations,
                "business_implications": result.business_implications,
                "research_significance": result.research_significance,
                "methodology_assessment": result.methodology_assessment,
                "impact_prediction": result.impact_prediction,
                "research_gaps_identified": result.research_gaps_identified,
                "related_work_connections": result.related_work_connections,
                "concept_keywords": result.concept_keywords,
                "document_id": result.document_id,
                "metadata": result.metadata
            },
            "usage_statistics": stats,
            "collected_papers": [
                {
                    "title": p.data.get('title', ''),
                    "authors": p.data.get('authors', []),
                    "published": str(p.data.get('published', '')),
                    "arxiv_id": p.data.get('id', ''),
                    "url": p.data.get('link', '')
                }
                for p in collected_papers
            ]
        }

        # Store with report manager
        report_manager = ReportStorageManager(use_postgresql=False)  # Use SQLite fallback
        file_path = report_manager.save_analysis_report(report_data, auto_path=True)

        print(f"✅ Report saved: {file_path.name}")
        print(f"📈 Tokens used: {stats.get('total_tokens', 'N/A')}")
        print(f"📊 Papers collected: {len(collected_papers)}")

        # Show search tip
        print(f"\n🔍 To find this report later:")
        print(f"   kh-search \"{query}\"")
        print(f"   kh-view {document_id}")

    except Exception as e:
        print(f"❌ Storage failed: {e}")
        # Still show the analysis results even if storage fails
        print(f"\n📊 Analysis Results (Quality: {result.research_quality_score:.2f}/10):")
        print(f"Novel Contributions: {len(result.novel_contributions)}")
        print(f"Technical Innovations: {len(result.technical_innovations)}")
        print(f"Business Implications: {len(result.business_implications)}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 run_pipeline.py \"<query>\" [max_results] [model]")
        print("Example: python3 run_pipeline.py \"AI agents\" 10 \"gemma2:2b\"")
        sys.exit(1)

    query = sys.argv[1]
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    model = sys.argv[3] if len(sys.argv) > 3 else "gemma2:2b"

    # Run the pipeline
    asyncio.run(run_pipeline(query, max_results, model))


if __name__ == "__main__":
    main()