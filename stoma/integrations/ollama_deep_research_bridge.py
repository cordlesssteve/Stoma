#!/usr/bin/env python3
"""
Working OpenDeepResearch Bridge using legacy multi-agent implementation with Ollama.

This bridge provides the integration between Stoma and OpenDeepResearch using
the proven working approach with proper Ollama tool calling support.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Annotated, TypedDict, Literal, cast, Optional, Dict, Any
from pydantic import BaseModel, Field
import operator

# Add OpenDeepResearch legacy to path
ODR_PATH = Path(__file__).parent.parent.parent / "external" / "open_deep_research" / "src"
if str(ODR_PATH) not in sys.path:
    sys.path.insert(0, str(ODR_PATH))

try:
    from langchain_ollama import ChatOllama
    from langchain_core.tools import tool, BaseTool
    from langchain_core.runnables import RunnableConfig
    from langchain_core.messages import HumanMessage
    from langgraph.graph import MessagesState, START, END, StateGraph
    from langgraph.types import Command, Send
except ImportError as e:
    raise ImportError(f"Missing required dependencies for OpenDeepResearch integration: {e}")

# Import Stoma components
from stoma.storage.report_manager import ReportStorageManager


class ResearchSection(BaseModel):
    """Section of the report."""
    name: str = Field(description="Name for this section of the report.")
    description: str = Field(description="Research scope for this section of the report.")
    content: str = Field(description="The content of the section.")


class ResearchSections(BaseModel):
    """List of section titles of the report."""
    sections: List[str] = Field(description="Sections of the report.")


class ResearchIntroduction(BaseModel):
    """Introduction to the report."""
    name: str = Field(description="Name for the report.")
    content: str = Field(description="The content of the introduction, giving an overview of the report.")


class ResearchConclusion(BaseModel):
    """Conclusion to the report."""
    name: str = Field(description="Name for the conclusion of the report.")
    content: str = Field(description="The content of the conclusion, summarizing the report.")


class FinishResearch(BaseModel):
    """Finish the research."""
    pass


class FinishReport(BaseModel):
    """Finish the report."""
    pass


# State definitions for the multi-agent workflow
class ReportStateOutput(MessagesState):
    final_report: str
    source_str: str


class ReportState(MessagesState):
    sections: list[str]
    completed_sections: Annotated[list[ResearchSection], operator.add]
    final_report: str
    source_str: Annotated[str, operator.add]


class SectionState(MessagesState):
    section: str
    completed_sections: list[ResearchSection]
    source_str: str


class SectionOutputState(TypedDict):
    completed_sections: list[ResearchSection]
    source_str: str


class OllamaDeepResearchBridge:
    """
    Bridge between Stoma and OpenDeepResearch using the working legacy multi-agent approach.
    """

    def __init__(self, model_name: str = "llama3.1:latest", storage_manager: Optional[ReportStorageManager] = None):
        """
        Initialize the bridge with Ollama model configuration.

        Args:
            model_name: Name of the Ollama model to use
            storage_manager: Optional storage manager for saving reports
        """
        self.model_name = model_name
        self.storage_manager = storage_manager or ReportStorageManager()

        # Set up environment
        os.environ['OPENAI_API_KEY'] = 'not-needed'
        os.environ['TAVILY_API_KEY'] = 'not-needed'

        print(f"🔧 Initialized OllamaDeepResearchBridge with model: {model_name}")

    def create_ollama_model(self) -> ChatOllama:
        """Create a ChatOllama model with proper configuration."""
        return ChatOllama(
            model=self.model_name,
            temperature=0.1,
            num_predict=3000,  # Allow longer responses
        )

    def get_today_str(self) -> str:
        """Get today's date as a string."""
        return datetime.now().strftime("%Y-%m-%d")

    def get_supervisor_prompt(self) -> str:
        """Get the supervisor instruction prompt."""
        return f"""You are a research supervisor coordinating a comprehensive research report.

Your job is to:
1. First, use the ResearchSections tool to break the research topic into logical sections (3-5 sections work well)
2. Once research sections are complete, use the ResearchIntroduction tool to write an introduction
3. Finally, use the ResearchConclusion tool to write a conclusion that summarizes the key findings
4. Use FinishReport when the report is complete

Create sections that cover the key aspects of the topic comprehensively.
Each section should be distinct and cover a different aspect of the topic.
Focus on providing detailed, technical, and well-researched content.
Today is {self.get_today_str()}.

IMPORTANT: You must use the tools provided. Always call a tool to perform your task."""

    def get_researcher_prompt(self, section_description: str) -> str:
        """Get the researcher instruction prompt for a specific section."""
        return f"""You are a research specialist assigned to research and write one specific section of a report.

Your section: {section_description}

Your job is to:
1. Research the topic thoroughly using your knowledge (no web search available)
2. Use the ResearchSection tool to write a well-structured section with detailed content
3. Include specific examples, technical details, and comprehensive coverage
4. Make your content informative, accurate, and well-researched

Write a substantial section with multiple paragraphs covering the key aspects of your assigned topic.
Provide detailed technical information, real-world applications, and specific examples where relevant.
Today is {self.get_today_str()}.

IMPORTANT: You must use the tools provided. Always call a tool to perform your task."""

    async def supervisor(self, state: ReportState, config: RunnableConfig):
        """Supervisor agent that coordinates the research workflow."""
        messages = state["messages"]
        llm = self.create_ollama_model()

        # If sections have been completed, initiate introduction and conclusion writing
        if state.get("completed_sections") and not state.get("final_report"):
            research_complete_message = {
                "role": "user",
                "content": "Research is complete. Now write the introduction and conclusion for the report. Here are the completed main body sections: \n\n" +
                          "\n\n".join([s.content for s in state["completed_sections"]])
            }
            messages = messages + [research_complete_message]

        # Define supervisor tools
        supervisor_tools = [
            tool(ResearchSections),
            tool(ResearchIntroduction),
            tool(ResearchConclusion),
            tool(FinishReport)
        ]

        llm_with_tools = llm.bind_tools(supervisor_tools)
        system_prompt = self.get_supervisor_prompt()

        return {
            "messages": [
                await llm_with_tools.ainvoke(
                    [{"role": "system", "content": system_prompt}] + messages
                )
            ]
        }

    async def supervisor_tools(self, state: ReportState, config: RunnableConfig) -> Command[Literal["supervisor", "research_team", "__end__"]]:
        """Process supervisor tool calls."""
        result = []
        sections_list = []
        intro_content = None
        conclusion_content = None

        # Define supervisor tools
        supervisor_tool_list = [
            tool(ResearchSections),
            tool(ResearchIntroduction),
            tool(ResearchConclusion),
            tool(FinishReport)
        ]
        supervisor_tools_by_name = {t.name: t for t in supervisor_tool_list}

        # Process all tool calls
        for tool_call in state["messages"][-1].tool_calls:
            tool_instance = supervisor_tools_by_name[tool_call["name"]]
            observation = tool_instance.invoke(tool_call["args"], config)

            result.append({
                "role": "tool",
                "content": str(observation),
                "name": tool_call["name"],
                "tool_call_id": tool_call["id"]
            })

            # Handle specific tool results
            if tool_call["name"] == "FinishReport":
                result.append({"role": "user", "content": "Report is finished"})
                return Command(goto=END, update={"messages": result})
            elif tool_call["name"] == "ResearchSections":
                sections_list = cast(ResearchSections, observation).sections
            elif tool_call["name"] == "ResearchIntroduction":
                observation = cast(ResearchIntroduction, observation)
                if not observation.content.startswith("# "):
                    intro_content = f"# {observation.name}\n\n{observation.content}"
                else:
                    intro_content = observation.content
            elif tool_call["name"] == "ResearchConclusion":
                observation = cast(ResearchConclusion, observation)
                if not observation.content.startswith("## "):
                    conclusion_content = f"## {observation.name}\n\n{observation.content}"
                else:
                    conclusion_content = observation.content

        # Route based on tool results
        if sections_list:
            print(f"📋 Created {len(sections_list)} sections: {sections_list}")
            return Command(goto=[Send("research_team", {"section": s}) for s in sections_list], update={"messages": result})
        elif intro_content:
            result.append({"role": "user", "content": "Introduction written. Now write a conclusion section."})
            state_update = {"final_report": intro_content, "messages": result}
            print("📝 Introduction completed")
        elif conclusion_content:
            # Assemble final report
            intro = state.get("final_report", "")
            body_sections = "\n\n".join([s.content for s in state["completed_sections"]])
            complete_report = f"{intro}\n\n{body_sections}\n\n{conclusion_content}"

            result.append({"role": "user", "content": "Report is now complete with introduction, body sections, and conclusion."})
            state_update = {"final_report": complete_report, "messages": result}
            print("🎯 Final report assembled")
        else:
            state_update = {"messages": result}

        return Command(goto="supervisor", update=state_update)

    async def supervisor_should_continue(self, state: ReportState) -> str:
        """Decide if supervisor should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return END
        return "supervisor_tools"

    async def research_agent(self, state: SectionState, config: RunnableConfig):
        """Research agent that handles one section."""
        llm = self.create_ollama_model()
        research_tools = [tool(ResearchSection), tool(FinishResearch)]

        system_prompt = self.get_researcher_prompt(state["section"])

        messages = state.get("messages", [])
        if not messages:
            messages = [{"role": "user", "content": f"Please research and write the section: {state['section']}"}]

        return {
            "messages": [
                await llm.bind_tools(research_tools).ainvoke(
                    [{"role": "system", "content": system_prompt}] + messages
                )
            ]
        }

    async def research_agent_tools(self, state: SectionState, config: RunnableConfig):
        """Process research agent tool calls."""
        result = []
        completed_section = None

        research_tool_list = [tool(ResearchSection), tool(FinishResearch)]
        research_tools_by_name = {t.name: t for t in research_tool_list}

        for tool_call in state["messages"][-1].tool_calls:
            tool_instance = research_tools_by_name[tool_call["name"]]
            observation = tool_instance.invoke(tool_call["args"], config)

            result.append({
                "role": "tool",
                "content": str(observation),
                "name": tool_call["name"],
                "tool_call_id": tool_call["id"]
            })

            if tool_call["name"] == "ResearchSection":
                completed_section = cast(ResearchSection, observation)
                print(f"✅ Section completed: {completed_section.name}")

        state_update = {"messages": result}
        if completed_section:
            state_update["completed_sections"] = [completed_section]

        return state_update

    async def research_agent_should_continue(self, state: SectionState) -> str:
        """Decide if research agent should continue."""
        if "completed_sections" in state and state["completed_sections"]:
            return END

        messages = state["messages"]
        last_message = messages[-1]
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return "research_agent_tools"

        return "research_agent_tools"

    def build_research_graph(self):
        """Build the multi-agent research workflow graph."""
        # Research agent workflow
        research_builder = StateGraph(SectionState, output_schema=SectionOutputState)
        research_builder.add_node("research_agent", self.research_agent)
        research_builder.add_node("research_agent_tools", self.research_agent_tools)
        research_builder.add_edge(START, "research_agent")
        research_builder.add_conditional_edges(
            "research_agent",
            self.research_agent_should_continue,
            ["research_agent_tools", END]
        )
        research_builder.add_edge("research_agent_tools", "research_agent")

        # Supervisor workflow
        supervisor_builder = StateGraph(ReportState, input_schema=MessagesState, output_schema=ReportStateOutput)
        supervisor_builder.add_node("supervisor", self.supervisor)
        supervisor_builder.add_node("supervisor_tools", self.supervisor_tools)
        supervisor_builder.add_node("research_team", research_builder.compile())

        supervisor_builder.add_edge(START, "supervisor")
        supervisor_builder.add_conditional_edges(
            "supervisor",
            self.supervisor_should_continue,
            ["supervisor_tools", END]
        )
        supervisor_builder.add_edge("research_team", "supervisor")

        return supervisor_builder.compile()

    async def analyze_topic(self, topic: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a research topic using the OpenDeepResearch multi-agent workflow.

        Args:
            topic: Research topic or query
            context: Optional additional context

        Returns:
            Dictionary containing the analysis results
        """
        print(f"🚀 Starting OpenDeepResearch analysis for topic: {topic}")

        # Build the research graph
        graph = self.build_research_graph()

        # Create research message
        research_message = f"Please create a comprehensive research report on {topic}."
        if context:
            research_message += f"\n\nAdditional context: {context}"

        research_message += """

Please provide a detailed analysis covering:
1. Current state and background
2. Key concepts and approaches
3. Recent developments and trends
4. Technical aspects and methodologies
5. Applications and use cases
6. Future directions and implications

Create a well-structured report with multiple sections, an introduction, and a conclusion.
Use your knowledge to provide detailed, technical content with specific examples where relevant.
"""

        input_state = {"messages": [HumanMessage(content=research_message)]}

        try:
            start_time = datetime.now()

            # Execute the research workflow
            result = await asyncio.wait_for(
                graph.ainvoke(input_state, {}),
                timeout=1800  # 30 minute timeout
            )

            end_time = datetime.now()
            duration = end_time - start_time

            final_report = result.get("final_report", "No final report generated")

            print(f"✅ Analysis completed in {duration.total_seconds():.1f} seconds")

            # Save the report using Stoma's storage system
            report_data = {
                "success": True,
                "topic": topic,
                "implementation": "ollama_deep_research_bridge",
                "model": self.model_name,
                "final_report": final_report,
                "duration_seconds": duration.total_seconds(),
                "timestamp": datetime.now().isoformat(),
                "sections_count": len(result.get("completed_sections", [])),
                "content_length": len(final_report)
            }

            # Store using Stoma's report manager
            report_id = f"deep_research_{topic.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Prepare report data in the expected format
            report_data_for_storage = {
                "timestamp": datetime.now().isoformat(),
                "provider": "ollama_deep_research",
                "model": self.model_name,
                "analysis": {
                    "final_report": final_report,
                    "topic": topic,
                    "duration_seconds": duration.total_seconds(),
                    "sections_count": len(result.get("completed_sections", [])),
                    "analysis_type": "comprehensive_research"
                }
            }

            saved_path = self.storage_manager.save_analysis_report(report_data_for_storage)
            print(f"💾 Report saved to: {saved_path}")

            return {
                "report_id": report_id,
                "final_report": final_report,
                "duration_seconds": duration.total_seconds(),
                "success": True,
                **report_data
            }

        except asyncio.TimeoutError:
            print("⏰ Analysis timed out after 30 minutes")
            return {
                "success": False,
                "error": "Analysis timed out after 30 minutes",
                "topic": topic
            }
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "topic": topic
            }

    async def analyze_documents(self, documents: List[Dict], research_question: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a collection of documents with optional research question.

        Args:
            documents: List of document dictionaries with content
            research_question: Optional specific research question to focus on

        Returns:
            Dictionary containing the analysis results
        """
        # Create a topic based on documents and research question
        if research_question:
            topic = f"Analysis of provided documents focusing on: {research_question}"
        else:
            topic = "Analysis of provided documents"

        # Create context from documents
        doc_summaries = []
        for i, doc in enumerate(documents[:5], 1):  # Limit to first 5 docs for context
            title = doc.get('title', f'Document {i}')
            content_preview = doc.get('content', '')[:200] + "..." if doc.get('content') else "No content"
            doc_summaries.append(f"Document {i} - {title}: {content_preview}")

        context = f"Base your analysis on the following {len(documents)} documents:\n\n" + "\n\n".join(doc_summaries)

        return await self.analyze_topic(topic, context)

    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = [line.split()[0] for line in lines if line.strip()]
                return models
            return ["llama3.1:latest"]  # Default fallback
        except Exception:
            return ["llama3.1:latest"]  # Default fallback