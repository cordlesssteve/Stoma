#!/usr/bin/env python3
"""
Test Ollama tool calling capabilities with proper langchain-ollama integration.
"""

import asyncio
import json
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# Define test tools
class ResearchSection(BaseModel):
    """A section of a research report."""
    name: str = Field(description="Name of the section")
    content: str = Field(description="Detailed content of the section")

@tool
def create_section(name: str, content: str) -> ResearchSection:
    """Create a research section with the given name and content."""
    return ResearchSection(name=name, content=content)


async def test_ollama_tool_calling():
    """Test if Ollama models can properly handle tool calling."""

    print("🔍 Testing Ollama Tool Calling Capabilities")
    print("=" * 50)

    # Initialize Ollama model with proper configuration
    llm = ChatOllama(
        model="llama3.1:latest",
        temperature=0.1,
        num_predict=2048,
        # Explicitly enable function calling
        format="json"  # This might help with structured output
    )

    # Define our tools
    tools = [create_section]

    print("✅ Ollama model initialized")
    print(f"   Model: llama3.1:latest")
    print(f"   Temperature: 0.1")

    try:
        # Test 1: Simple tool calling
        print("\n🔧 Test 1: Basic Tool Calling")

        llm_with_tools = llm.bind_tools(tools)

        messages = [
            HumanMessage(content="""
Create a research section about reinforcement learning basics.

Use the create_section tool with:
- name: "Introduction to Reinforcement Learning"
- content: A comprehensive overview of what reinforcement learning is, its key concepts, and basic principles.
""")
        ]

        print("⏱️  Invoking model with tools...")
        start_time = datetime.now()

        result = await llm_with_tools.ainvoke(messages)

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"✅ Tool calling completed in {duration.total_seconds():.1f} seconds")

        # Check if tool calls were made
        if hasattr(result, 'tool_calls') and result.tool_calls:
            print(f"🔧 Found {len(result.tool_calls)} tool call(s)")

            for i, tool_call in enumerate(result.tool_calls, 1):
                print(f"   Tool Call {i}:")
                print(f"     Name: {tool_call['name']}")
                print(f"     Args: {tool_call['args']}")

                # Execute the tool
                if tool_call['name'] == 'create_section':
                    section = create_section.invoke(tool_call['args'])
                    print(f"     Result: {section}")

                    # Save successful result
                    output_data = {
                        "success": True,
                        "test": "ollama_tool_calling",
                        "model": "llama3.1:latest",
                        "tool_calls": result.tool_calls,
                        "section_created": {
                            "name": section.name,
                            "content": section.content
                        },
                        "duration_seconds": duration.total_seconds(),
                        "timestamp": datetime.now().isoformat()
                    }

                    output_file = f"ollama_tools_success_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(output_file, 'w') as f:
                        json.dump(output_data, f, indent=2)

                    print(f"\n💾 Success results saved to: {output_file}")
                    print("🎉 Ollama tool calling works!")

                    return True
        else:
            print("⚠️  No tool calls found in result")
            print(f"   Result type: {type(result)}")
            print(f"   Result content: {result.content}")

            return False

    except NotImplementedError as e:
        print(f"❌ Tool calling not implemented: {e}")
        return False

    except Exception as e:
        print(f"❌ Tool calling failed: {e}")
        import traceback
        traceback.print_exc()

        # Save error details
        error_data = {
            "success": False,
            "test": "ollama_tool_calling",
            "model": "llama3.1:latest",
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }

        error_file = f"ollama_tools_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w') as f:
            json.dump(error_data, f, indent=2)

        print(f"\n💾 Error details saved to: {error_file}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_ollama_tool_calling())
    print(f"\n🎯 Ollama tool calling test: {'✅ PASSED' if success else '❌ FAILED'}")