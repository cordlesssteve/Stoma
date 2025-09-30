#!/usr/bin/env python3
"""
Debug OpenDeepResearch integration with proper Ollama configuration.
"""

import asyncio
import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Add OpenDeepResearch to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "open_deep_research" / "src"))

# Set up environment for local models
os.environ['OPENAI_API_KEY'] = 'not-needed'  # Ollama doesn't need this but ODR might check
os.environ['TAVILY_API_KEY'] = 'not-needed'  # We'll use search_api=none

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.state import AgentInputState
from langchain_core.messages import HumanMessage


async def test_odr_with_ollama():
    """Test OpenDeepResearch with proper Ollama configuration."""

    print("🔧 Testing OpenDeepResearch with Ollama Configuration")
    print("=" * 60)

    # Create proper configuration
    config = Configuration(
        # Use Ollama models
        summarization_model="ollama:llama3.1:latest",
        research_model="ollama:llama3.1:latest",
        compression_model="ollama:llama3.1:latest",
        final_report_model="ollama:llama3.1:latest",

        # Reduce complexity for local models
        max_researcher_iterations=2,
        max_concurrent_research_units=1,
        max_react_tool_calls=5,

        # Disable external dependencies
        allow_clarification=False,
        search_api=SearchAPI.NONE,  # No external search

        # Conservative token limits
        summarization_model_max_tokens=2048,
        research_model_max_tokens=4096,
        compression_model_max_tokens=2048,
        final_report_model_max_tokens=4096,
        max_content_length=10000
    )

    print(f"✅ Configuration created:")
    print(f"   Research Model: {config.research_model}")
    print(f"   Search API: {config.search_api.value}")
    print(f"   Max Iterations: {config.max_researcher_iterations}")
    print(f"   Max Concurrent: {config.max_concurrent_research_units}")

    # Create research input
    topic = "reinforcement learning architectures"
    research_message = f"""
Please provide a comprehensive analysis of {topic}, focusing on:

1. Current state of the field
2. Key architectural approaches (value-based, policy-based, actor-critic)
3. Recent innovations and trends
4. Technical strengths and limitations
5. Future research directions
6. Practical applications

Provide detailed technical analysis without requiring external web search.
"""

    input_state = AgentInputState(messages=[HumanMessage(content=research_message)])

    # Convert to runnable config format
    runnable_config = {
        "configurable": {
            "summarization_model": config.summarization_model,
            "research_model": config.research_model,
            "compression_model": config.compression_model,
            "final_report_model": config.final_report_model,
            "max_researcher_iterations": config.max_researcher_iterations,
            "max_concurrent_research_units": config.max_concurrent_research_units,
            "max_react_tool_calls": config.max_react_tool_calls,
            "allow_clarification": config.allow_clarification,
            "search_api": config.search_api.value,
            "summarization_model_max_tokens": config.summarization_model_max_tokens,
            "research_model_max_tokens": config.research_model_max_tokens,
            "compression_model_max_tokens": config.compression_model_max_tokens,
            "final_report_model_max_tokens": config.final_report_model_max_tokens,
            "max_content_length": config.max_content_length
        }
    }

    print(f"\n🚀 Starting OpenDeepResearch analysis...")
    print(f"Topic: {topic}")

    try:
        start_time = datetime.now()

        # Run the deep research workflow
        result = await deep_researcher.ainvoke(input_state, runnable_config)

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n✅ Analysis completed in {duration.total_seconds():.1f} seconds")
        print("=" * 60)

        # Display results
        final_report = result.get("final_report", "No final report generated")
        research_findings = result.get("notes", [])

        print("📋 FINAL REPORT:")
        print("-" * 40)
        print(final_report)

        if research_findings:
            print(f"\n🔍 RESEARCH FINDINGS ({len(research_findings)} items):")
            print("-" * 40)
            for i, finding in enumerate(research_findings[:3], 1):
                print(f"{i}. {finding[:200]}...")

        # Save results
        output_data = {
            "success": True,
            "topic": topic,
            "final_report": final_report,
            "research_findings": research_findings,
            "configuration": {
                "models": {
                    "research": config.research_model,
                    "summarization": config.summarization_model,
                    "compression": config.compression_model,
                    "final_report": config.final_report_model
                },
                "search_api": config.search_api.value,
                "max_iterations": config.max_researcher_iterations,
                "max_concurrent": config.max_concurrent_research_units
            },
            "duration_seconds": duration.total_seconds(),
            "timestamp": datetime.now().isoformat()
        }

        output_file = f"odr_debug_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")
        print("\n🎉 OpenDeepResearch + Ollama integration successful!")

        return True

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        print("\n🔍 Full error traceback:")
        traceback.print_exc()

        # Save error details
        error_data = {
            "success": False,
            "topic": topic,
            "error": str(e),
            "error_type": type(e).__name__,
            "configuration": {
                "research_model": config.research_model,
                "search_api": config.search_api.value
            },
            "timestamp": datetime.now().isoformat()
        }

        error_file = f"odr_debug_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w') as f:
            json.dump(error_data, f, indent=2)

        print(f"\n💾 Error details saved to: {error_file}")

        return False


if __name__ == "__main__":
    success = asyncio.run(test_odr_with_ollama())
    print(f"\n🎯 Integration test: {'✅ PASSED' if success else '❌ FAILED'}")