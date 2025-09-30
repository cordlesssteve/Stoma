#!/usr/bin/env python3
"""
High Quality Analysis Script for Stoma

This script runs comprehensive analysis using the highest quality models available,
with enhanced prompting and detailed output for research-grade insights.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from minimal_pipeline import MinimalArXivCollector, MinimalLLMAnalyzer, MinimalHealthChecker
from advanced_model_manager import AdvancedModelManager

logger = logging.getLogger(__name__)


class HighQualityAnalyzer:
    """Enhanced analyzer using highest quality models with sophisticated prompting."""

    def __init__(self):
        self.model_manager = AdvancedModelManager()
        self.health_checker = MinimalHealthChecker(auto_repair=True)
        self.results_path = Path("./reports/high_quality")
        self.results_path.mkdir(parents=True, exist_ok=True)

    async def run_comprehensive_analysis(self, query: str, max_papers: int = 10) -> Dict[str, Any]:
        """Run comprehensive analysis with highest quality model."""
        analysis_start = datetime.now()
        print(f"🎯 High Quality Analysis: {query}")
        print(f"📊 Papers: {max_papers} | Started: {analysis_start.strftime('%H:%M:%S')}")
        print("=" * 70)

        # Health check
        print("🔍 System health check...")
        health_status = await self.health_checker.check_all_systems()
        if not health_status['arxiv'] or not health_status['ollama']:
            print("❌ System not ready for analysis")
            return {"error": "System health check failed"}

        # Select highest quality model
        available_models = health_status['models']
        print(f"📋 Available models: {len(available_models)}")

        # Choose the largest/best model available
        quality_preference_order = [
            "codellama:13b-instruct",  # Largest, best for reasoning
            "mistral:7b-instruct",     # Good balance
            "qwen2.5-coder:3b",       # Fallback
            "phi3.5:latest",          # Alternative
            "phi3:latest"             # Last resort
        ]

        selected_model = None
        for preferred_model in quality_preference_order:
            if preferred_model in available_models:
                selected_model = preferred_model
                break

        if not selected_model:
            selected_model = available_models[0] if available_models else "qwen2.5-coder:3b"

        print(f"🧠 Selected high-quality model: {selected_model}")
        model_info = self.model_manager.get_model_info(selected_model)
        if model_info:
            print(f"   Performance tier: {model_info.performance_tier}")
            print(f"   Size: {model_info.size_gb:.1f}GB")
            print(f"   Specialty: {model_info.specialty}")

        # Collect papers
        print(f"\n📚 Collecting papers for: '{query}'...")
        collector = MinimalArXivCollector(max_papers)
        papers = await collector.collect(query)

        if not papers:
            print("❌ No papers collected")
            return {"error": "No papers found"}

        print(f"✅ Collected {len(papers)} papers")

        # Enhanced analysis with high-quality prompting
        analyzer = HighQualityLLMAnalyzer(selected_model)

        print(f"\n🔬 Running enhanced analysis...")
        enhanced_results = await self._run_enhanced_analysis(analyzer, papers, query)

        # Save comprehensive results
        report_path = await self._save_comprehensive_report(
            enhanced_results, papers, query, selected_model, analysis_start
        )

        analysis_duration = (datetime.now() - analysis_start).total_seconds()
        print(f"\n✅ Analysis complete in {analysis_duration:.1f} seconds")
        print(f"📁 Comprehensive report saved to: {report_path}")

        return enhanced_results

    async def _run_enhanced_analysis(self, analyzer, papers: List[Dict], query: str) -> Dict[str, Any]:
        """Run enhanced analysis with sophisticated prompting."""

        # Prepare enhanced content
        papers_content = []
        for i, paper in enumerate(papers[:8], 1):  # Analyze top 8 papers for quality
            content = f"""Paper {i}:
Title: {paper['title']}
Authors: {', '.join(paper.get('authors', [])[:5])}
Abstract: {paper['abstract']}
ArXiv ID: {paper.get('arxiv_id', 'N/A')}
Published: {paper.get('published', 'N/A')}
---"""
            papers_content.append(content)

        combined_content = "\n".join(papers_content)

        # Enhanced analysis
        print("   🧠 Running deep semantic analysis...")
        analysis_result = await analyzer.analyze_comprehensive(combined_content, query)

        return analysis_result

    async def _save_comprehensive_report(self, analysis: Dict, papers: List,
                                       query: str, model: str, start_time: datetime) -> Path:
        """Save comprehensive high-quality analysis report."""

        timestamp = start_time.strftime('%Y%m%d_%H%M%S')
        filename = f"high_quality_analysis_{query.replace(' ', '_')}_{timestamp}.json"
        report_path = self.results_path / filename

        comprehensive_report = {
            "analysis_metadata": {
                "query": query,
                "model_used": model,
                "papers_analyzed": len(papers),
                "analysis_timestamp": start_time.isoformat(),
                "analysis_duration_seconds": (datetime.now() - start_time).total_seconds(),
                "analysis_type": "high_quality_comprehensive"
            },
            "enhanced_analysis": analysis,
            "paper_details": [
                {
                    "title": paper['title'],
                    "authors": paper.get('authors', []),
                    "arxiv_id": paper.get('arxiv_id'),
                    "published": paper.get('published'),
                    "url": paper.get('url'),
                    "abstract_length": len(paper.get('abstract', ''))
                }
                for paper in papers
            ],
            "model_information": {
                "model_name": model,
                "model_info": self.model_manager.get_model_info(model).__dict__ if self.model_manager.get_model_info(model) else None
            }
        }

        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)

        return report_path


class HighQualityLLMAnalyzer:
    """Enhanced LLM analyzer with sophisticated prompting for research-grade analysis."""

    def __init__(self, model: str):
        self.model = model
        self.base_analyzer = MinimalLLMAnalyzer(model)

    async def analyze_comprehensive(self, content: str, query: str) -> Dict[str, Any]:
        """Run comprehensive analysis with enhanced prompting."""

        enhanced_prompt = f"""
You are an expert AI researcher conducting a comprehensive analysis of cutting-edge research papers in {query}.

Your task is to provide a DEEP, NUANCED analysis that goes beyond surface-level observations. Focus on:

1. NOVEL THEORETICAL CONTRIBUTIONS - What genuinely new theoretical insights do these papers provide?
2. METHODOLOGICAL INNOVATIONS - What specific technical innovations advance the state-of-the-art?
3. EMPIRICAL BREAKTHROUGHS - What experimental results or performance improvements are significant?
4. INTERDISCIPLINARY CONNECTIONS - How do these works connect different research areas?
5. FUTURE RESEARCH DIRECTIONS - What important open problems and research gaps are identified?
6. PRACTICAL IMPLICATIONS - What real-world applications and business opportunities emerge?
7. LIMITATIONS AND CRITIQUES - What are the genuine limitations and potential criticisms?

Research Papers to Analyze:
{content[:8000]}

Please provide your analysis in JSON format with SPECIFIC, DETAILED insights:

{{
  "theoretical_contributions": [
    "Specific theoretical advance 1 with detailed explanation",
    "Specific theoretical advance 2 with detailed explanation",
    "Specific theoretical advance 3 with detailed explanation"
  ],
  "methodological_innovations": [
    "Detailed technical innovation 1 with implementation specifics",
    "Detailed technical innovation 2 with implementation specifics",
    "Detailed technical innovation 3 with implementation specifics"
  ],
  "empirical_breakthroughs": [
    "Specific performance improvement with metrics",
    "Experimental validation with concrete results",
    "Benchmark achievements with quantitative details"
  ],
  "interdisciplinary_connections": [
    "Connection to field X with specific examples",
    "Integration with domain Y showing concrete benefits",
    "Cross-pollination with area Z demonstrating novel applications"
  ],
  "future_research_directions": [
    "Specific open problem 1 with research approach suggestions",
    "Important research gap 2 with potential solution directions",
    "Emerging opportunity 3 with development pathway"
  ],
  "practical_implications": [
    "Real-world application 1 with deployment considerations",
    "Business opportunity 2 with market impact analysis",
    "Societal benefit 3 with implementation challenges"
  ],
  "limitations_and_critiques": [
    "Methodological limitation 1 with potential remedies",
    "Scope restriction 2 with expansion possibilities",
    "Validation concern 3 with verification approaches"
  ],
  "research_quality_assessment": {{
    "overall_quality_score": 0.0,
    "novelty_score": 0.0,
    "rigor_score": 0.0,
    "impact_potential": 0.0,
    "clarity_score": 0.0
  }},
  "synthesis_insights": [
    "Cross-paper pattern 1 with evidence",
    "Emerging trend 2 with supporting papers",
    "Collective impact 3 with field-level implications"
  ]
}}

CRITICAL: Provide SPECIFIC, DETAILED analysis, not generic statements. Include concrete examples, metrics, and evidence from the papers.
"""

        try:
            # Use the base analyzer but with enhanced prompt
            result = await self.base_analyzer.analyze(enhanced_prompt, f"Comprehensive Analysis: {query}")

            if result:
                return result
            else:
                # Fallback analysis if enhanced fails
                fallback_result = await self.base_analyzer.analyze(content[:4000], f"Analysis: {query}")
                return fallback_result or {"error": "Analysis failed"}

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {"error": f"Analysis failed: {e}"}


async def main():
    """Main entry point for high quality analysis."""
    import sys

    if len(sys.argv) < 2:
        print("High Quality Analysis for Stoma")
        print("Usage: python3 high_quality_analysis.py <query> [max_papers]")
        print("Examples:")
        print("  python3 high_quality_analysis.py 'semantic reasoning AI' 10")
        print("  python3 high_quality_analysis.py 'knowledge graphs' 15")
        return

    query = sys.argv[1]
    max_papers = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    analyzer = HighQualityAnalyzer()
    results = await analyzer.run_comprehensive_analysis(query, max_papers)

    if "error" not in results:
        print(f"\n🎉 High Quality Analysis Results:")

        # Display key insights if available
        if "theoretical_contributions" in results:
            print(f"\n🔬 Key Theoretical Contributions:")
            for contrib in results["theoretical_contributions"][:3]:
                print(f"  • {contrib}")

        if "methodological_innovations" in results:
            print(f"\n⚙️ Methodological Innovations:")
            for innovation in results["methodological_innovations"][:3]:
                print(f"  • {innovation}")

        if "research_quality_assessment" in results:
            quality = results["research_quality_assessment"]
            print(f"\n📊 Quality Assessment:")
            for metric, score in quality.items():
                if isinstance(score, (int, float)):
                    print(f"  {metric}: {score:.2f}")
                else:
                    print(f"  {metric}: {score}")

    else:
        print(f"\n❌ Analysis failed: {results['error']}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    asyncio.run(main())