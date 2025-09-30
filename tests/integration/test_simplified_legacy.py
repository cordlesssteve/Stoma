#!/usr/bin/env python3
"""
Test a simplified version of the legacy OpenDeepResearch multi-agent implementation
that bypasses optional search API dependencies we don't need for testing.
"""

import asyncio
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Annotated, TypedDict, Literal, cast
from pydantic import BaseModel, Field
import operator

# Add OpenDeepResearch legacy to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "open_deep_research" / "src"))

# Set up environment for local models
os.environ['OPENAI_API_KEY'] = 'not-needed'
os.environ['TAVILY_API_KEY'] = 'not-needed'

print("🔍 Testing Simplified Legacy Multi-Agent with Ollama...")
print("=" * 60)

# Test basic imports that should work
try:
    from langchain.chat_models import init_chat_model
    from langchain_core.tools import tool, BaseTool
    from langchain_core.runnables import RunnableConfig
    from langchain_core.messages import HumanMessage
    from langgraph.graph import MessagesState, START, END, StateGraph
    from langgraph.types import Command, Send
    print("✅ Basic LangGraph imports successful")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Define our own simplified configuration
class SimplifiedConfig:
    def __init__(self):
        self.supervisor_model = "ollama:llama3.1:latest"
        self.researcher_model = "ollama:llama3.1:latest"
        self.search_api = "none"
        self.number_of_queries = 1
        self.ask_for_clarification = False

# Define the Pydantic models we need
class Section(BaseModel):
    """Section of the report."""
    name: str = Field(description="Name for this section of the report.")
    description: str = Field(description="Research scope for this section of the report.")
    content: str = Field(description="The content of the section.")

class Sections(BaseModel):
    """List of section titles of the report."""
    sections: List[str] = Field(description="Sections of the report.")

class Introduction(BaseModel):
    """Introduction to the report."""
    name: str = Field(description="Name for the report.")
    content: str = Field(description="The content of the introduction, giving an overview of the report.")

class Conclusion(BaseModel):
    """Conclusion to the report."""
    name: str = Field(description="Name for the conclusion of the report.")
    content: str = Field(description="The content of the conclusion, summarizing the report.")

class FinishResearch(BaseModel):
    """Finish the research."""
    pass

class FinishReport(BaseModel):
    """Finish the report."""
    pass

# State definitions
class ReportStateOutput(MessagesState):
    final_report: str
    source_str: str

class ReportState(MessagesState):
    sections: list[str]
    completed_sections: Annotated[list[Section], operator.add]
    final_report: str
    source_str: Annotated[str, operator.add]

class SectionState(MessagesState):
    section: str
    completed_sections: list[Section]
    source_str: str

class SectionOutputState(TypedDict):
    completed_sections: list[Section]
    source_str: str

# Simplified prompts
SUPERVISOR_INSTRUCTIONS = """You are a research supervisor coordinating a comprehensive report.

Your job is to:
1. First, use the Sections tool to break the research topic into logical sections
2. Once research sections are complete, use the Introduction tool to write an introduction
3. Finally, use the Conclusion tool to write a conclusion
4. Use FinishReport when the report is complete

Create sections that cover the key aspects of the topic comprehensively.
Today is {today}."""

RESEARCH_INSTRUCTIONS = """You are a research specialist assigned to research and write one specific section of a report.

Your section: {section_description}

Since web search is disabled, use your knowledge to write a comprehensive section on this topic.

Your job is to:
1. Research the topic thoroughly using your knowledge
2. Use the Section tool to write a well-structured section with detailed content
3. Use FinishResearch when you're done

Write a substantial section with multiple paragraphs covering the key aspects of your assigned topic.
Include specific details, examples, and comprehensive coverage.
Today is {today}."""

def get_today_str():
    """Get today's date as a string."""
    return datetime.now().strftime("%Y-%m-%d")

# Simplified supervisor function
async def supervisor(state: ReportState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""
    messages = state["messages"]

    # Initialize the model
    llm = init_chat_model(model="ollama:llama3.1:latest")

    # If sections have been completed, but we don't yet have the final report, then we need to initiate writing the introduction and conclusion
    if state.get("completed_sections") and not state.get("final_report"):
        research_complete_message = {"role": "user", "content": "Research is complete. Now write the introduction and conclusion for the report. Here are the completed main body sections: \n\n" + "\n\n".join([s.content for s in state["completed_sections"]])}
        messages = messages + [research_complete_message]

    # Define supervisor tools
    supervisor_tools = [tool(Sections), tool(Introduction), tool(Conclusion), tool(FinishReport)]

    llm_with_tools = llm.bind_tools(
        supervisor_tools,
        parallel_tool_calls=False,
        tool_choice="any"
    )

    # Get system prompt
    system_prompt = SUPERVISOR_INSTRUCTIONS.format(today=get_today_str())

    # Invoke
    return {
        "messages": [
            await llm_with_tools.ainvoke(
                [{"role": "system", "content": system_prompt}] + messages
            )
        ]
    }

async def supervisor_tools(state: ReportState, config: RunnableConfig) -> Command[Literal["supervisor", "research_team", "__end__"]]:
    """Performs the tool call and sends to the research agent"""
    result = []
    sections_list = []
    intro_content = None
    conclusion_content = None

    # Define supervisor tools
    supervisor_tool_list = [tool(Sections), tool(Introduction), tool(Conclusion), tool(FinishReport)]
    supervisor_tools_by_name = {t.name: t for t in supervisor_tool_list}

    # Process all tool calls
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool
        tool = supervisor_tools_by_name[tool_call["name"]]
        # Perform the tool call
        observation = tool.invoke(tool_call["args"], config)

        # Append to messages
        result.append({"role": "tool",
                       "content": observation,
                       "name": tool_call["name"],
                       "tool_call_id": tool_call["id"]})

        # Store special tool results
        if tool_call["name"] == "FinishReport":
            result.append({"role": "user", "content": "Report is finished"})
            return Command(goto=END, update={"messages": result})
        elif tool_call["name"] == "Sections":
            sections_list = cast(Sections, observation).sections
        elif tool_call["name"] == "Introduction":
            observation = cast(Introduction, observation)
            if not observation.content.startswith("# "):
                intro_content = f"# {observation.name}\n\n{observation.content}"
            else:
                intro_content = observation.content
        elif tool_call["name"] == "Conclusion":
            observation = cast(Conclusion, observation)
            if not observation.content.startswith("## "):
                conclusion_content = f"## {observation.name}\n\n{observation.content}"
            else:
                conclusion_content = observation.content

    # After processing all tool calls, decide what to do next
    if sections_list:
        # Send the sections to the research agents
        return Command(goto=[Send("research_team", {"section": s}) for s in sections_list], update={"messages": result})
    elif intro_content:
        # Store introduction while waiting for conclusion
        result.append({"role": "user", "content": "Introduction written. Now write a conclusion section."})
        state_update = {
            "final_report": intro_content,
            "messages": result,
        }
    elif conclusion_content:
        # Get all sections and combine in proper order: Introduction, Body Sections, Conclusion
        intro = state.get("final_report", "")
        body_sections = "\n\n".join([s.content for s in state["completed_sections"]])

        # Assemble final report
        complete_report = f"{intro}\n\n{body_sections}\n\n{conclusion_content}"

        result.append({"role": "user", "content": "Report is now complete with introduction, body sections, and conclusion."})
        state_update = {
            "final_report": complete_report,
            "messages": result,
        }
    else:
        state_update = {"messages": result}

    return Command(goto="supervisor", update=state_update)

async def supervisor_should_continue(state: ReportState) -> str:
    """Decide if we should continue the loop or stop"""
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    return "supervisor_tools"

async def research_agent(state: SectionState, config: RunnableConfig):
    """Research agent that handles one section"""
    llm = init_chat_model(model="ollama:llama3.1:latest")

    # Define research tools
    research_tools = [tool(Section), tool(FinishResearch)]

    system_prompt = RESEARCH_INSTRUCTIONS.format(
        section_description=state["section"],
        today=get_today_str(),
    )

    # Ensure we have at least one user message
    messages = state.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": f"Please research and write the section: {state['section']}"}]

    return {
        "messages": [
            await llm.bind_tools(research_tools, parallel_tool_calls=False, tool_choice="any").ainvoke(
                [{"role": "system", "content": system_prompt}] + messages
            )
        ]
    }

async def research_agent_tools(state: SectionState, config: RunnableConfig):
    """Performs the tool call and route to supervisor or continue the research loop"""
    result = []
    completed_section = None

    # Define research tools
    research_tool_list = [tool(Section), tool(FinishResearch)]
    research_tools_by_name = {t.name: t for t in research_tool_list}

    # Process all tool calls
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool
        tool = research_tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"], config)

        # Append to messages
        result.append({"role": "tool",
                       "content": observation,
                       "name": tool_call["name"],
                       "tool_call_id": tool_call["id"]})

        # Store the section if a Section tool was called
        if tool_call["name"] == "Section":
            completed_section = cast(Section, observation)

    # After processing all tools, decide what to do next
    state_update = {"messages": result}
    if completed_section:
        # Write the completed section to state and return to the supervisor
        state_update["completed_sections"] = [completed_section]

    return state_update

async def research_agent_should_continue(state: SectionState) -> str:
    """Decide if we should continue the loop or stop"""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls[0]["name"] == "FinishResearch":
        return END
    else:
        return "research_agent_tools"

# Build the graphs
def build_simplified_multiagent_graph():
    """Build the simplified multi-agent workflow"""

    # Research agent workflow
    research_builder = StateGraph(SectionState, output=SectionOutputState)
    research_builder.add_node("research_agent", research_agent)
    research_builder.add_node("research_agent_tools", research_agent_tools)
    research_builder.add_edge(START, "research_agent")
    research_builder.add_conditional_edges(
        "research_agent",
        research_agent_should_continue,
        ["research_agent_tools", END]
    )
    research_builder.add_edge("research_agent_tools", "research_agent")

    # Supervisor workflow
    supervisor_builder = StateGraph(ReportState, input=MessagesState, output=ReportStateOutput)
    supervisor_builder.add_node("supervisor", supervisor)
    supervisor_builder.add_node("supervisor_tools", supervisor_tools)
    supervisor_builder.add_node("research_team", research_builder.compile())

    # Flow of the supervisor agent
    supervisor_builder.add_edge(START, "supervisor")
    supervisor_builder.add_conditional_edges(
        "supervisor",
        supervisor_should_continue,
        ["supervisor_tools", END]
    )
    supervisor_builder.add_edge("research_team", "supervisor")

    return supervisor_builder.compile()

async def test_simplified_legacy():
    """Test the simplified legacy multi-agent implementation."""

    print("\n🚀 Starting Simplified Legacy Multi-Agent Test")
    print("=" * 40)

    # Build the graph
    graph = build_simplified_multiagent_graph()

    # Create simple research request
    topic = "reinforcement learning fundamentals"
    research_message = f"""
Please create a comprehensive research report on {topic}.

Focus on:
1. Introduction to reinforcement learning concepts
2. Key algorithms and methodologies
3. Applications and real-world use cases
4. Future directions in the field

Use your knowledge to create detailed sections since web search is disabled.
"""

    # Create input state
    input_state = {
        "messages": [HumanMessage(content=research_message)]
    }

    print(f"📝 Research Topic: {topic}")
    print("⏱️  Starting simplified legacy analysis...")

    try:
        start_time = datetime.now()

        # Set timeout for this test (10 minutes)
        result = await asyncio.wait_for(
            graph.ainvoke(input_state, {}),
            timeout=600  # 10 minute timeout
        )

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n✅ Analysis completed in {duration.total_seconds():.1f} seconds!")
        print("=" * 60)

        # Display results
        final_report = result.get("final_report", "No final report generated")

        print("📋 FINAL REPORT:")
        print("-" * 40)
        print(final_report[:2000] + "..." if len(final_report) > 2000 else final_report)

        # Save results
        output_data = {
            "success": True,
            "implementation": "simplified_legacy_multi_agent",
            "topic": topic,
            "final_report": final_report,
            "configuration": {
                "supervisor_model": "ollama:llama3.1:latest",
                "researcher_model": "ollama:llama3.1:latest",
                "search_api": "none"
            },
            "duration_seconds": duration.total_seconds(),
            "timestamp": datetime.now().isoformat()
        }

        output_file = f"simplified_legacy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")
        print("🎉 Simplified Legacy Multi-Agent + Ollama integration successful!")

        return True

    except asyncio.TimeoutError:
        print("⏰ Test timed out after 10 minutes")
        return False

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        print("\n🔍 Full error traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_simplified_legacy())
    print(f"\n🎯 Simplified Legacy test: {'✅ PASSED' if success else '❌ FAILED'}")