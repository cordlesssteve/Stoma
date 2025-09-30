#!/usr/bin/env python3
"""
Test the legacy OpenDeepResearch multi-agent implementation with Ollama models.
This should be more compatible than the main LangGraph workflow.
"""

import asyncio
import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Add OpenDeepResearch legacy to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "open_deep_research" / "src"))

# Set up environment for local models
os.environ['OPENAI_API_KEY'] = 'not-needed'  # Ollama doesn't need this
os.environ['TAVILY_API_KEY'] = 'not-needed'  # We'll use search_api=none

print("🔍 Testing Legacy Multi-Agent OpenDeepResearch with Ollama...")
print("=" * 60)

# Test basic imports
try:
    from legacy.multi_agent import graph
    from legacy.configuration import MultiAgentConfiguration, SearchAPI
    from langchain_core.messages import HumanMessage
    print("✅ Legacy multi-agent import successful")
except Exception as e:
    print(f"❌ Legacy multi-agent import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

async def test_legacy_multiagent_with_ollama():
    """Test legacy multi-agent implementation with Ollama configuration."""

    print("\n🚀 Starting Legacy Multi-Agent Test")
    print("=" * 40)

    # Configure for Ollama with no search API
    runnable_config = {
        "configurable": {
            # Use Ollama models
            "supervisor_model": "ollama:llama3.1:latest",
            "researcher_model": "ollama:llama3.1:latest",

            # Disable external search for testing
            "search_api": "none",

            # Minimal queries to reduce complexity
            "number_of_queries": 1,

            # No clarification needed
            "ask_for_clarification": False,

            # Include source string for evaluation
            "include_source_str": False,
        }
    }

    print(f"✅ Configuration created:")
    print(f"   Supervisor Model: ollama:llama3.1:latest")
    print(f"   Researcher Model: ollama:llama3.1:latest")
    print(f"   Search API: none")
    print(f"   Number of Queries: 1")

    # Create simple research request
    topic = "reinforcement learning basics"
    research_message = f"""
Please create a comprehensive research report on {topic}.

Focus on:
1. What is reinforcement learning
2. Key algorithms and approaches
3. Applications and use cases

Since we don't have web search, use your knowledge to create sections.
"""

    # Create input state
    input_state = {
        "messages": [HumanMessage(content=research_message)]
    }

    print(f"\n📝 Research Topic: {topic}")
    print("⏱️  Starting legacy multi-agent analysis...")

    try:
        start_time = datetime.now()

        # Set timeout for this test (5 minutes)
        result = await asyncio.wait_for(
            graph.ainvoke(input_state, runnable_config),
            timeout=300  # 5 minute timeout
        )

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n✅ Analysis completed in {duration.total_seconds():.1f} seconds!")
        print("=" * 60)

        # Display results
        final_report = result.get("final_report", "No final report generated")

        print("📋 FINAL REPORT:")
        print("-" * 40)
        print(final_report)

        # Save results
        output_data = {
            "success": True,
            "implementation": "legacy_multi_agent",
            "topic": topic,
            "final_report": final_report,
            "configuration": {
                "supervisor_model": "ollama:llama3.1:latest",
                "researcher_model": "ollama:llama3.1:latest",
                "search_api": "none",
                "number_of_queries": 1
            },
            "duration_seconds": duration.total_seconds(),
            "timestamp": datetime.now().isoformat()
        }

        output_file = f"legacy_multiagent_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")
        print("✅ Legacy Multi-Agent + Ollama integration successful!")

        return True

    except asyncio.TimeoutError:
        print("⏰ Test timed out after 5 minutes")
        return False

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        print("\n🔍 Full error traceback:")
        traceback.print_exc()

        # Save error details
        error_data = {
            "success": False,
            "implementation": "legacy_multi_agent",
            "topic": topic,
            "error": str(e),
            "error_type": type(e).__name__,
            "configuration": {
                "supervisor_model": "ollama:llama3.1:latest",
                "researcher_model": "ollama:llama3.1:latest",
                "search_api": "none"
            },
            "timestamp": datetime.now().isoformat()
        }

        error_file = f"legacy_multiagent_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w') as f:
            json.dump(error_data, f, indent=2)

        print(f"\n💾 Error details saved to: {error_file}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_legacy_multiagent_with_ollama())
    print(f"\n🎯 Legacy Multi-Agent test: {'✅ PASSED' if success else '❌ FAILED'}")