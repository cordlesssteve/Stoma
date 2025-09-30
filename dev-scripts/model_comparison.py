#!/usr/bin/env python3
"""
Model Comparison Script for KnowHunt

Compare analysis quality across different models on the same papers.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import aiohttp

logger = logging.getLogger(__name__)

class ModelComparator:
    """Compare analysis quality across different models."""

    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.results_path = Path("./reports/model_comparison")
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Test papers for comparison
        self.test_papers = [
            {
                "title": "SciReasoner: Laying the Scientific Reasoning Ground Across Disciplines",
                "abstract": "We present a scientific reasoning foundation model that aligns natural language with heterogeneous scientific representations. The model is pretrained on a 206B-token corpus spanning scientific text, pure sequences, and sequence-text pairs, then aligned via SFT on 40M instructions, annealed cold-start bootstrapping to elicit long-form chain-of-thought, and reinforcement learning with task-specific reward shaping, which instills deliberate scientific reasoning."
            },
            {
                "title": "SAGE: A Realistic Benchmark for Semantic Understanding",
                "abstract": "As large language models (LLMs) achieve strong performance on traditional benchmarks, there is an urgent need for more challenging evaluation frameworks that probe deeper aspects of semantic understanding. We introduce SAGE (Semantic Alignment & Generalization Evaluation), a rigorous benchmark designed to assess both embedding models and similarity metrics across five categories."
            }
        ]

    async def compare_models(self, models_to_test: List[str]) -> Dict[str, Any]:
        """Compare analysis quality across multiple models."""
        print("🔍 Model Quality Comparison")
        print("=" * 40)

        comparison_results = {
            "comparison_timestamp": datetime.now().isoformat(),
            "models_tested": models_to_test,
            "test_papers_count": len(self.test_papers),
            "model_results": {}
        }

        # Test each model
        for model in models_to_test:
            print(f"\n🧠 Testing model: {model}")

            try:
                result = await self._test_model(model)
                comparison_results["model_results"][model] = result

                # Display quick summary
                if result["success"]:
                    print(f"   ✅ Analysis successful")
                    print(f"   📊 Quality score: {result.get('quality_score', 'N/A')}")
                    print(f"   ⏱️  Response time: {result.get('response_time_seconds', 'N/A'):.1f}s")
                    print(f"   🔢 Tokens: {result.get('tokens_used', 'N/A')}")
                else:
                    print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                print(f"   💥 Error testing {model}: {e}")
                comparison_results["model_results"][model] = {
                    "success": False,
                    "error": str(e)
                }

        # Save comparison report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.results_path / f"model_comparison_{timestamp}.json"

        with open(report_path, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)

        print(f"\n📁 Comparison report saved: {report_path}")

        # Display summary
        self._display_comparison_summary(comparison_results)

        return comparison_results

    async def _test_model(self, model_name: str) -> Dict[str, Any]:
        """Test a single model on the test papers."""
        start_time = datetime.now()

        # Prepare test content
        test_content = "\n\n".join([
            f"Paper: {paper['title']}\nAbstract: {paper['abstract']}"
            for paper in self.test_papers
        ])

        # Simple but effective prompt
        prompt = f"""Analyze these research papers and provide insights in JSON format:

{test_content}

Please provide:
{{
  "key_innovations": ["innovation 1", "innovation 2", "innovation 3"],
  "research_significance": ["significance 1", "significance 2"],
  "business_potential": ["potential 1", "potential 2"],
  "quality_score": 7.5
}}"""

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 800
                    }
                }

                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_text = result.get("response", "")

                        response_time = (datetime.now() - start_time).total_seconds()
                        tokens_used = len(analysis_text.split())

                        # Try to parse JSON response
                        parsed_analysis = self._parse_analysis_response(analysis_text)

                        return {
                            "success": True,
                            "response_time_seconds": response_time,
                            "tokens_used": tokens_used,
                            "raw_response": analysis_text,
                            "parsed_analysis": parsed_analysis,
                            "quality_score": parsed_analysis.get("quality_score", 0) if parsed_analysis else 0,
                            "has_structured_output": parsed_analysis is not None
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}"
                        }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time_seconds": (datetime.now() - start_time).total_seconds()
            }

    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Try to parse JSON from model response."""
        try:
            # Look for JSON in the response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1

            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
        except:
            pass

        return None

    def _display_comparison_summary(self, results: Dict[str, Any]):
        """Display comparison summary."""
        print("\n📊 Model Comparison Summary")
        print("=" * 40)

        successful_models = []
        failed_models = []

        for model, result in results["model_results"].items():
            if result["success"]:
                successful_models.append((model, result))
            else:
                failed_models.append((model, result))

        if successful_models:
            print("\n✅ Successful Models:")
            # Sort by quality score
            successful_models.sort(key=lambda x: x[1].get("quality_score", 0), reverse=True)

            for model, result in successful_models:
                quality = result.get("quality_score", 0)
                time_taken = result.get("response_time_seconds", 0)
                structured = "✓" if result.get("has_structured_output") else "✗"

                print(f"   🥇 {model}")
                print(f"      Quality: {quality}/10, Time: {time_taken:.1f}s, Structured: {structured}")

        if failed_models:
            print("\n❌ Failed Models:")
            for model, result in failed_models:
                print(f"   💥 {model}: {result.get('error', 'Unknown error')}")

        if successful_models:
            best_model, best_result = successful_models[0]
            print(f"\n🏆 Best Performing Model: {best_model}")
            print(f"   Quality Score: {best_result.get('quality_score', 'N/A')}")
            print(f"   Response Time: {best_result.get('response_time_seconds', 0):.1f} seconds")

async def main():
    """Main entry point for model comparison."""

    # Available models to test (in order of preference)
    models_to_test = [
        "mistral:7b-instruct",    # 7B parameter model
        "qwen2.5-coder:3b",      # 3B parameter model
        "phi3.5:latest",         # Small efficient model
        # "codellama:13b-instruct" # Skip for now due to issues
    ]

    print("🚀 KnowHunt Model Quality Comparison")
    print(f"🧪 Testing {len(models_to_test)} models on semantic reasoning analysis")

    comparator = ModelComparator()
    results = await comparator.compare_models(models_to_test)

    print(f"\n🎉 Model comparison complete!")
    print(f"📈 Results will help optimize model selection for higher quality analysis")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(main())