"""
Bridge integration between Stoma and OpenDeepResearch systems.

This module provides seamless integration between Stoma's content enrichment
pipeline and OpenDeepResearch's sophisticated LangGraph-based research workflows.
"""

import sys
import os
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from pathlib import Path

# Add external submodule to path
sys.path.append(str(Path(__file__).parent.parent.parent / "external" / "open_deep_research" / "src"))

try:
    from open_deep_research.configuration import Configuration as ODRConfiguration
    from open_deep_research.deep_researcher import deep_researcher
    from open_deep_research.state import AgentInputState
    from langchain_core.messages import HumanMessage
    OPEN_DEEP_RESEARCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"OpenDeepResearch not available: {e}")
    OPEN_DEEP_RESEARCH_AVAILABLE = False

# Stoma imports
from ..analysis.llm_analyzer import LLMAnalysisResult
from ..pipeline.data_types import NormalizedDocument
from ..storage.report_manager import ReportStorageManager

logger = logging.getLogger(__name__)


@dataclass
class DeepResearchConfig:
    """Configuration for deep research integration."""

    # Model Configuration
    summarization_model: str = "openai:gpt-4.1-mini"
    research_model: str = "openai:gpt-4.1"
    compression_model: str = "openai:gpt-4.1"
    final_report_model: str = "openai:gpt-4.1"

    # Research Configuration
    max_researcher_iterations: int = 6
    max_concurrent_research_units: int = 5
    max_react_tool_calls: int = 10
    allow_clarification: bool = False  # Disable for automated workflows

    # Search Configuration
    search_api: str = "tavily"  # or "openai", "anthropic", "none"

    # Output Configuration
    max_tokens: int = 8192
    temperature: float = 0.1


@dataclass
class DeepResearchResult:
    """Result from deep research analysis."""

    document_id: str
    research_question: str
    final_report: str
    research_findings: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime
    model_used: str
    token_usage: Optional[Dict[str, int]] = None


class DeepResearchBridge:
    """Bridge between Stoma and OpenDeepResearch systems."""

    def __init__(self,
                 config: Optional[DeepResearchConfig] = None,
                 storage_manager: Optional[ReportStorageManager] = None):
        """
        Initialize the bridge.

        Args:
            config: Deep research configuration
            storage_manager: Stoma report storage manager
        """
        if not OPEN_DEEP_RESEARCH_AVAILABLE:
            raise ImportError(
                "OpenDeepResearch is not available. "
                "Please ensure the submodule is properly initialized."
            )

        self.config = config or DeepResearchConfig()
        self.storage_manager = storage_manager

        # Convert to OpenDeepResearch configuration
        self.odr_config = self._build_odr_configuration()

        # Usage tracking
        self.usage_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "total_tokens_used": 0
        }

    def _build_odr_configuration(self) -> Dict[str, Any]:
        """Convert DeepResearchConfig to OpenDeepResearch configuration format."""
        return {
            "configurable": {
                "summarization_model": self.config.summarization_model,
                "research_model": self.config.research_model,
                "compression_model": self.config.compression_model,
                "final_report_model": self.config.final_report_model,
                "max_researcher_iterations": self.config.max_researcher_iterations,
                "max_concurrent_research_units": self.config.max_concurrent_research_units,
                "max_react_tool_calls": self.config.max_react_tool_calls,
                "allow_clarification": self.config.allow_clarification,
                "search_api": self.config.search_api,
                "summarization_model_max_tokens": self.config.max_tokens,
                "research_model_max_tokens": self.config.max_tokens,
                "compression_model_max_tokens": self.config.max_tokens,
                "final_report_model_max_tokens": self.config.max_tokens,
            }
        }

    async def analyze_document(self,
                             document: NormalizedDocument,
                             research_question: Optional[str] = None) -> DeepResearchResult:
        """
        Perform deep research analysis on a document.

        Args:
            document: Stoma normalized document
            research_question: Optional custom research question

        Returns:
            DeepResearchResult with comprehensive analysis
        """
        self.usage_stats["total_analyses"] += 1

        try:
            # Generate research question if not provided
            if not research_question:
                research_question = self._generate_research_question(document)

            # Prepare input for OpenDeepResearch
            input_messages = [
                HumanMessage(content=f"""
Analyze the following research paper/document:

Title: {document.title}
Authors: {', '.join(document.authors) if document.authors else 'Unknown'}
Published: {document.published_date}
URL: {document.url}

Content:
{document.content[:10000]}...  # Truncate very long content

Research Question: {research_question}

Please provide a comprehensive analysis including:
1. Novel contributions and innovations
2. Research significance and impact
3. Methodology assessment
4. Business and technical implications
5. Identified research gaps
6. Connections to related work
""")
            ]

            # Create input state
            input_state = AgentInputState(messages=input_messages)

            # Run OpenDeepResearch workflow
            logger.info(f"Starting deep research analysis for document: {document.id}")
            result = await deep_researcher.ainvoke(input_state, self.odr_config)

            # Extract results
            final_report = result.get("final_report", "Analysis failed")
            research_findings = result.get("notes", [])

            # Create result object
            deep_result = DeepResearchResult(
                document_id=document.id,
                research_question=research_question,
                final_report=final_report,
                research_findings=research_findings,
                metadata={
                    "document_title": document.title,
                    "document_url": document.url,
                    "document_categories": document.categories,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "config_used": asdict(self.config)
                },
                timestamp=datetime.now(),
                model_used=self.config.research_model
            )

            # Store result if storage manager available
            if self.storage_manager:
                await self._store_result(deep_result)

            self.usage_stats["successful_analyses"] += 1
            logger.info(f"Deep research analysis completed for document: {document.id}")

            return deep_result

        except Exception as e:
            self.usage_stats["failed_analyses"] += 1
            logger.error(f"Deep research analysis failed for document {document.id}: {e}")

            # Return error result
            return DeepResearchResult(
                document_id=document.id,
                research_question=research_question or "Analysis failed",
                final_report=f"Deep research analysis failed: {str(e)}",
                research_findings=[],
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                model_used=self.config.research_model
            )

    async def batch_analyze_documents(self,
                                    documents: List[NormalizedDocument],
                                    research_questions: Optional[List[str]] = None) -> List[DeepResearchResult]:
        """
        Perform batch deep research analysis on multiple documents.

        Args:
            documents: List of Stoma normalized documents
            research_questions: Optional list of custom research questions

        Returns:
            List of DeepResearchResult objects
        """
        if research_questions and len(research_questions) != len(documents):
            raise ValueError("Number of research questions must match number of documents")

        # Create analysis tasks
        tasks = []
        for i, document in enumerate(documents):
            question = research_questions[i] if research_questions else None
            task = self.analyze_document(document, question)
            tasks.append(task)

        # Execute with limited concurrency to avoid rate limits
        results = []
        batch_size = min(3, len(tasks))  # Process 3 at a time

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch analysis error: {result}")
                    # Create error result
                    error_result = DeepResearchResult(
                        document_id="unknown",
                        research_question="Analysis failed",
                        final_report=f"Batch analysis error: {str(result)}",
                        research_findings=[],
                        metadata={"error": str(result)},
                        timestamp=datetime.now(),
                        model_used=self.config.research_model
                    )
                    results.append(error_result)
                else:
                    results.append(result)

        return results

    def _generate_research_question(self, document: NormalizedDocument) -> str:
        """Generate a research question based on document content."""
        if document.categories:
            categories_str = ", ".join(document.categories)
            return f"What are the key contributions and implications of this research in {categories_str}?"
        else:
            return f"What are the key contributions, methodology, and implications of the research presented in '{document.title}'?"

    async def _store_result(self, result: DeepResearchResult) -> None:
        """Store deep research result using Stoma's storage system."""
        if not self.storage_manager:
            return

        try:
            # Convert to storage format
            report_data = {
                "analysis_type": "deep_research",
                "document_id": result.document_id,
                "research_question": result.research_question,
                "final_report": result.final_report,
                "research_findings": result.research_findings,
                "metadata": result.metadata,
                "model_used": result.model_used,
                "timestamp": result.timestamp.isoformat(),
                "quality_score": self._calculate_quality_score(result)
            }

            # Store in Stoma's report system
            report_id = f"deep_research_{result.document_id}_{int(result.timestamp.timestamp())}"
            await self.storage_manager.store_analysis_report(
                report_id=report_id,
                report_data=report_data,
                provider="deep_research_bridge"
            )

        except Exception as e:
            logger.error(f"Failed to store deep research result: {e}")

    def _calculate_quality_score(self, result: DeepResearchResult) -> float:
        """Calculate quality score for the analysis result."""
        score = 1.0  # Base score

        # Increase score based on content length and depth
        if len(result.final_report) > 1000:
            score += 1.0
        if len(result.research_findings) > 3:
            score += 1.0
        if "methodology" in result.final_report.lower():
            score += 0.5
        if "implications" in result.final_report.lower():
            score += 0.5

        return min(score, 5.0)  # Cap at 5.0

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for the bridge."""
        success_rate = 0.0
        if self.usage_stats["total_analyses"] > 0:
            success_rate = self.usage_stats["successful_analyses"] / self.usage_stats["total_analyses"]

        return {
            **self.usage_stats,
            "success_rate": success_rate,
            "config": asdict(self.config)
        }


# Utility functions for easy integration

async def analyze_papers_with_deep_research(papers: List[NormalizedDocument],
                                          config: Optional[DeepResearchConfig] = None,
                                          storage_manager: Optional[ReportStorageManager] = None) -> List[DeepResearchResult]:
    """
    Convenience function to analyze papers using deep research.

    Args:
        papers: List of normalized documents from Stoma
        config: Optional configuration
        storage_manager: Optional storage manager

    Returns:
        List of deep research results
    """
    bridge = DeepResearchBridge(config, storage_manager)
    return await bridge.batch_analyze_documents(papers)


def create_default_deep_research_config() -> DeepResearchConfig:
    """Create a default configuration for deep research."""
    return DeepResearchConfig(
        # Use smaller, faster models for development
        summarization_model="openai:gpt-4.1-mini",
        research_model="openai:gpt-4.1",
        compression_model="openai:gpt-4.1-mini",
        final_report_model="openai:gpt-4.1",
        max_researcher_iterations=4,  # Reduce for faster processing
        max_concurrent_research_units=3,  # Reduce to avoid rate limits
        allow_clarification=False,  # Disable for automated workflows
        search_api="tavily"
    )