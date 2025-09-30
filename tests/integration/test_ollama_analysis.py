#!/usr/bin/env python3
"""
Direct test of OpenDeepResearch with Ollama models.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add OpenDeepResearch to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "open_deep_research" / "src"))

from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.state import AgentInputState
from langchain_core.messages import HumanMessage


async def test_ollama_analysis():
    """Test OpenDeepResearch directly with Ollama models."""

    print("🚀 Testing OpenDeepResearch with Ollama models")

    # Research topic
    topic = "reinforcement learning architectures"

    # Create research message
    research_message = f"""
Please provide a comprehensive analysis of {topic}, including:

1. Current state of the field
2. Key architectural approaches and innovations
3. Strengths and limitations of different approaches
4. Recent developments and trends
5. Future research directions
6. Practical applications and implications

Focus on technical depth and analytical insights.
"""

    # Configuration for Ollama models
    config = {
        "configurable": {
            # Use Ollama models
            "research_model": "ollama:llama3.1:latest",
            "final_report_model": "ollama:llama3.1:latest",
            "summarization_model": "ollama:phi3.5:latest",
            "compression_model": "ollama:phi3.5:latest",

            # Reduce complexity for local models
            "max_researcher_iterations": 2,
            "max_concurrent_research_units": 1,
            "max_react_tool_calls": 5,
            "allow_clarification": False,

            # No external search API
            "search_api": "none",

            # Token limits
            "research_model_max_tokens": 4096,
            "final_report_model_max_tokens": 4096,
            "summarization_model_max_tokens": 2048,
            "compression_model_max_tokens": 2048,
        }
    }

    # Create input state
    input_state = AgentInputState(messages=[HumanMessage(content=research_message)])

    print(f"📊 Analyzing: {topic}")
    print("🔧 Using models:")
    print(f"   Research: {config['configurable']['research_model']}")
    print(f"   Final Report: {config['configurable']['final_report_model']}")
    print("⏱️  Starting analysis...")

    try:
        # Run the deep research workflow
        result = await deep_researcher.ainvoke(input_state, config)

        print("\n✅ Analysis Complete!")
        print("=" * 60)

        # Extract and display the final report
        final_report = result.get("final_report", "No final report generated")
        print("📋 Final Report:")
        print("-" * 40)
        print(final_report)

        # Show research findings if available
        notes = result.get("notes", [])
        if notes:
            print(f"\n🔍 Research Findings ({len(notes)} items):")
            print("-" * 40)
            for i, note in enumerate(notes[:3], 1):  # Show first 3
                print(f"{i}. {note[:200]}...")

        # Save results
        output_file = "ollama_rl_analysis.json"
        with open(output_file, 'w') as f:
            json.dump({
                "topic": topic,
                "final_report": final_report,
                "research_findings": notes,
                "model_config": config["configurable"],
                "analysis_complete": True
            }, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        return True

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_ollama_analysis())
    print(f"\n🎯 Test {'✅ PASSED' if success else '❌ FAILED'}")