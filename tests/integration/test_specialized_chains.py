#!/usr/bin/env python3
"""
Test specialized prompt chains with real paper data
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

from stoma.analysis.specialized_llm_analyzer import SpecializedLLMAnalyzer


async def test_real_paper():
    """Test specialized chains with real research paper."""

    # Load real paper data
    with open('reports/minimal_pipeline/analysis_AI_agents_20250925_174251.json', 'r') as f:
        data = json.load(f)

    # Use the EmbeddingGemma paper (first one)
    paper = data['papers'][0]

    # Create comprehensive content for analysis
    content = f"""
RESEARCH PAPER ANALYSIS

Title: {paper['title']}

Authors: {', '.join(paper['authors'][:5])} et al. ({len(paper['authors'])} total authors)

Abstract: {paper['abstract']}

Published: {paper['published']}
ArXiv ID: {paper['arxiv_id']}
"""

    print("🧪 Testing Specialized Prompt Chains")
    print("=" * 60)
    print(f"Paper: {paper['title']}")
    print(f"Authors: {len(paper['authors'])} researchers")
    print("=" * 60)

    # Test the specialized analyzer
    analyzer = SpecializedLLMAnalyzer(model='codellama:13b-instruct')

    result = await analyzer.analyze_with_specialized_chains(content, paper['title'])

    if result:
        print("\n🎉 SPECIALIZED CHAINS SUCCESS!")
        print("\n" + "=" * 60)
        print(f"🎯 Research Quality Score: {result['research_quality_score']}/10")
        print(f"🔧 Analysis Method: {result['analysis_method']}")
        print(f"🔢 Tokens Used: {result['tokens_used']}")

        print("\n🔬 NOVEL CONTRIBUTIONS:")
        for i, contrib in enumerate(result['novel_contributions'], 1):
            print(f"{i}. {contrib}")

        print("\n⚡ TECHNICAL INNOVATIONS:")
        for i, innovation in enumerate(result['technical_innovations'], 1):
            print(f"{i}. {innovation}")

        print("\n💼 BUSINESS IMPLICATIONS:")
        for i, implication in enumerate(result['business_implications'], 1):
            print(f"{i}. {implication}")

        print("\n" + "=" * 60)
        print("✅ Analysis complete - genuine insights extracted!")

    else:
        print("\n❌ SPECIALIZED CHAINS FAILED")
        print("Check reports/unanalyzed/ for failure details")

        # Show unanalyzed tally
        tally_file = Path("reports/unanalyzed/unanalyzed_tally.json")
        if tally_file.exists():
            with open(tally_file, 'r') as f:
                tally = json.load(f)
            print(f"Total failures: {tally['total_failures']}")
            print(f"By type: {tally['by_type']}")


if __name__ == "__main__":
    asyncio.run(test_real_paper())