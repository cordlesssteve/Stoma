#!/usr/bin/env python3
"""
Test reinforcement learning analysis with ArXiv paper citations.
Combines Stoma's ArXiv collection with Ollama analysis.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from stoma.collectors.arxiv import ArXivCollector
from stoma.normalizers.base import AcademicNormalizer
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


async def analyze_rl_with_citations():
    """Analyze RL architectures with real ArXiv paper citations."""

    print("🚀 Reinforcement Learning Analysis with ArXiv Citations")
    print("=" * 70)

    # Step 1: Collect recent RL papers from ArXiv
    print("📚 Step 1: Collecting recent RL papers from ArXiv...")

    arxiv_config = {
        "max_results": 5,  # Get 5 recent papers
        "rate_limit": 1.0
    }

    collector = ArXivCollector(arxiv_config)
    normalizer = AcademicNormalizer({})

    papers = []
    search_queries = [
        "reinforcement learning architecture",
        "deep reinforcement learning",
        "actor critic architecture"
    ]

    for query in search_queries[:2]:  # Use first 2 queries to limit papers
        print(f"   Searching: {query}")
        async for result in collector.collect(
            search_query=query,
            category="cs.LG",  # Machine Learning category
            start=0
        ):
            if result.success and len(papers) < 5:
                normalized = await normalizer.normalize(result)
                papers.append(normalized)
                print(f"   ✅ Found: {normalized.title[:60]}...")

            if len(papers) >= 5:
                break

        if len(papers) >= 5:
            break

    print(f"✅ Collected {len(papers)} papers")

    # Step 2: Create citations bibliography
    citations = []
    for i, paper in enumerate(papers, 1):
        citation = {
            "id": i,
            "title": paper.title,
            "authors": paper.authors,
            "url": paper.url,
            "published": paper.published_date.strftime("%Y-%m-%d") if paper.published_date else "Unknown",
            "summary": paper.content[:300] + "..." if len(paper.content) > 300 else paper.content
        }
        citations.append(citation)

    # Step 3: Create enhanced prompt with citations
    citations_text = "\n\n".join([
        f"[{c['id']}] {c['title']}\n"
        f"Authors: {', '.join(c['authors']) if c['authors'] else 'Unknown'}\n"
        f"Published: {c['published']}\n"
        f"URL: {c['url']}\n"
        f"Summary: {c['summary']}"
        for c in citations
    ])

    enhanced_prompt = f"""You are a research expert analyzing reinforcement learning architectures. Using the recent research papers provided below, create a comprehensive analysis.

RECENT RESEARCH PAPERS:
{citations_text}

Based on these papers and your knowledge, provide a comprehensive analysis of reinforcement learning architectures that includes:

1. **Current State of the Field** (referencing the provided papers)
2. **Key Architectural Approaches** with specific citations to the papers
3. **Recent Innovations** highlighted in the research papers
4. **Technical Analysis** of approaches mentioned in the papers
5. **Future Directions** suggested by recent research
6. **Citations and References** - Include proper citations using [1], [2], etc. format

IMPORTANT:
- Reference the provided papers using [1], [2], [3], [4], [5] format throughout your analysis
- Include specific details from the papers where relevant
- Maintain academic rigor with proper attribution
- Create a bibliography section at the end listing all references

Provide a detailed, well-cited analysis that demonstrates how recent research is advancing the field."""

    # Step 4: Run analysis with Ollama
    print("\n🧠 Step 2: Running analysis with Ollama + Citations...")

    try:
        model = init_chat_model(
            model="ollama:llama3.1:latest",
            max_tokens=6000,  # Increased for detailed analysis
            temperature=0.1
        )
        print("✅ Model initialized: llama3.1:latest")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

    start_time = datetime.now()

    try:
        # Run the enhanced analysis
        response = await model.ainvoke([HumanMessage(content=enhanced_prompt)])

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"✅ Analysis completed in {duration.total_seconds():.1f} seconds")
        print("=" * 70)
        print("📋 REINFORCEMENT LEARNING ARCHITECTURES - WITH CITATIONS")
        print("=" * 70)
        print(response.content)
        print("=" * 70)

        # Step 5: Save comprehensive results
        result = {
            "topic": "reinforcement learning architectures",
            "analysis_with_citations": response.content,
            "model": "llama3.1:latest",
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration.total_seconds(),
            "citations_used": citations,
            "papers_collected": len(papers),
            "search_queries": search_queries[:2],
            "methodology": "ArXiv collection + Ollama analysis",
            "success": True
        }

        output_file = f"rl_analysis_with_citations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"💾 Complete analysis with citations saved to: {output_file}")

        # Show citation summary
        print(f"\n📖 Citations Summary:")
        for citation in citations:
            print(f"[{citation['id']}] {citation['title']}")

        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(analyze_rl_with_citations())
    if success:
        print("\n🎉 Enhanced analysis with citations completed successfully!")
        print("Your research now includes proper academic references!")
    else:
        print("\n💥 Analysis failed. Check the error messages above.")