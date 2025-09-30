#!/usr/bin/env python3
"""
Minimal OpenDeepResearch test to isolate issues.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add OpenDeepResearch to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "open_deep_research" / "src"))

# Test basic imports first
print("🔍 Testing OpenDeepResearch imports...")

try:
    from open_deep_research.configuration import Configuration, SearchAPI
    print("✅ Configuration import successful")
except Exception as e:
    print(f"❌ Configuration import failed: {e}")
    sys.exit(1)

try:
    from open_deep_research.deep_researcher import deep_researcher
    print("✅ DeepResearcher import successful")
except Exception as e:
    print(f"❌ DeepResearcher import failed: {e}")
    sys.exit(1)

try:
    from open_deep_research.state import AgentInputState
    from langchain_core.messages import HumanMessage
    print("✅ State classes import successful")
except Exception as e:
    print(f"❌ State classes import failed: {e}")
    sys.exit(1)

# Test model initialization
print("\n🔧 Testing model initialization...")

try:
    from langchain.chat_models import init_chat_model

    model = init_chat_model("ollama:llama3.1:latest", max_tokens=100, temperature=0.1)
    print("✅ Ollama model init successful")

except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    import traceback
    traceback.print_exc()


# Test configuration creation
print("\n⚙️ Testing configuration...")

try:
    config = Configuration(
        summarization_model="ollama:llama3.1:latest",
        research_model="ollama:llama3.1:latest",
        compression_model="ollama:llama3.1:latest",
        final_report_model="ollama:llama3.1:latest",
        max_researcher_iterations=1,  # Minimal
        max_concurrent_research_units=1,
        max_react_tool_calls=2,
        allow_clarification=False,
        search_api=SearchAPI.NONE,
        summarization_model_max_tokens=512,
        research_model_max_tokens=1024,
        compression_model_max_tokens=512,
        final_report_model_max_tokens=1024
    )
    print("✅ Configuration creation successful")
    print(f"   Search API: {config.search_api.value}")
    print(f"   Models: All using llama3.1:latest")

except Exception as e:
    print(f"❌ Configuration creation failed: {e}")
    import traceback
    traceback.print_exc()


async def main():
    print("\n🚀 Minimal OpenDeepResearch Test")
    print("=" * 40)

    # Test very simple input
    simple_input = AgentInputState(
        messages=[HumanMessage(content="Explain what reinforcement learning is in one paragraph.")]
    )

    # Minimal configuration
    runnable_config = {
        "configurable": {
            "summarization_model": "ollama:llama3.1:latest",
            "research_model": "ollama:llama3.1:latest",
            "compression_model": "ollama:llama3.1:latest",
            "final_report_model": "ollama:llama3.1:latest",
            "max_researcher_iterations": 1,
            "max_concurrent_research_units": 1,
            "max_react_tool_calls": 1,
            "allow_clarification": False,
            "search_api": "none",
            "summarization_model_max_tokens": 512,
            "research_model_max_tokens": 1024,
            "compression_model_max_tokens": 512,
            "final_report_model_max_tokens": 1024
        }
    }

    print("⏱️  Starting minimal deep research...")
    print("   (This should be much faster)")

    try:
        import time
        start_time = time.time()

        # Set timeout for this test
        result = await asyncio.wait_for(
            deep_researcher.ainvoke(simple_input, runnable_config),
            timeout=120  # 2 minute timeout
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ Completed in {duration:.1f} seconds!")
        print(f"📋 Result keys: {list(result.keys())}")

        final_report = result.get("final_report", "No report")
        print(f"📄 Final report (first 200 chars): {final_report[:200]}...")

        return True

    except asyncio.TimeoutError:
        print("⏰ Test timed out after 2 minutes")
        return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\n🎯 Minimal test: {'✅ PASSED' if success else '❌ FAILED'}")