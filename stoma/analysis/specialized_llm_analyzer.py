#!/usr/bin/env python3
"""
Specialized LLM Analyzer with Chain-of-Thought Prompting
Replaces generic single prompt with specialized analysis chains.
"""

import json
import aiohttp
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class SpecializedLLMAnalyzer:
    """LLM analyzer using specialized prompt chains for each insight type."""

    def __init__(self, model="codellama:13b-instruct", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.tokens_used = 0

        # Track analysis failures for unanalyzed reports
        self.unanalyzed_dir = Path("reports/unanalyzed")
        self.unanalyzed_dir.mkdir(parents=True, exist_ok=True)

    async def analyze_with_specialized_chains(self, text: str, title: str = "Research Analysis") -> Optional[Dict]:
        """Run specialized analysis chains for each insight type."""

        print(f"🔗 Running specialized analysis chains...")

        try:
            # Chain 1: Novel Contributions Analysis
            novel_contributions = await self._analyze_novel_contributions(text)
            if novel_contributions is None:
                return await self._handle_analysis_failure(text, title, "novel_contributions")

            # Chain 2: Technical Innovations Analysis
            technical_innovations = await self._analyze_technical_innovations(text)
            if technical_innovations is None:
                return await self._handle_analysis_failure(text, title, "technical_innovations")

            # Chain 3: Business Implications Analysis
            business_implications = await self._analyze_business_implications(text)
            if business_implications is None:
                return await self._handle_analysis_failure(text, title, "business_implications")

            # Chain 4: Research Quality Scoring
            quality_score = await self._analyze_research_quality(text)
            if quality_score is None:
                quality_score = 5  # Default fallback

            print("✅ All specialized chains completed successfully")

            return {
                "document_id": f"specialized_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "title": title,
                "research_quality_score": quality_score,
                "novel_contributions": novel_contributions,
                "technical_innovations": technical_innovations,
                "business_implications": business_implications,
                "analysis_method": "specialized_chains",
                "tokens_used": self.tokens_used
            }

        except Exception as e:
            print(f"❌ Specialized chain analysis failed: {e}")
            return await self._handle_analysis_failure(text, title, "complete_failure")

    async def _analyze_novel_contributions(self, text: str) -> Optional[List[str]]:
        """Specialized chain for novel contributions analysis."""

        prompt = f"""
You are a research analyst specialized in identifying NOVEL CONTRIBUTIONS.

Focus ONLY on what is genuinely NEW in this research:

{text[:3000]}

Task: Identify 3-5 specific novel contributions. For each, explain:
1. What exactly is new/novel about it?
2. How does it differ from existing work?
3. What problem does this novelty solve?

Format as a JSON list of strings. Each string should be a complete insight, not just a title.

Example format:
[
  "First quantum error correction method that works on current NISQ hardware with 95% fidelity - previous methods required fault-tolerant quantum computers",
  "Novel attention mechanism that reduces O(n²) complexity to O(n log n) while maintaining 99% accuracy - enables processing of documents 10x longer",
  "Breakthrough in few-shot learning requiring only 5 examples vs 1000+ in previous approaches - democratizes AI for small datasets"
]

Return only the JSON array, no other text:"""

        return await self._get_llm_response_as_list(prompt, "novel_contributions")

    async def _analyze_technical_innovations(self, text: str) -> Optional[List[str]]:
        """Specialized chain for technical innovations analysis."""

        prompt = f"""
You are a technical specialist analyzing SPECIFIC TECHNICAL INNOVATIONS.

Focus on concrete technical methods, algorithms, architectures:

{text[:3000]}

Task: Identify 3-5 specific technical innovations. For each, include:
1. The specific technical method/algorithm/architecture
2. Quantitative improvements (speed, accuracy, efficiency)
3. Implementation details that make it work

Format as JSON list. Focus on HOW things work technically, not just what they do.

Example format:
[
  "Sparse attention patterns using locality-sensitive hashing reduce memory from 16GB to 2GB while maintaining 99.2% accuracy on GLUE benchmark",
  "Novel gradient accumulation strategy with momentum correction achieves 40% faster training than Adam optimizer on transformer models",
  "Hybrid CPU-GPU memory management automatically switches between compute types, reducing inference latency by 60% on edge devices"
]

Return only the JSON array:"""

        return await self._get_llm_response_as_list(prompt, "technical_innovations")

    async def _analyze_business_implications(self, text: str) -> Optional[List[str]]:
        """Specialized chain for business implications analysis."""

        prompt = f"""
You are a business analyst identifying COMMERCIAL IMPLICATIONS.

Focus on real-world business value and market opportunities:

{text[:3000]}

Task: Identify 3-5 specific business implications. For each, include:
1. Concrete business value (cost savings, new revenue, efficiency)
2. Target market/industry applications
3. Competitive advantages or market positioning

Format as JSON list. Focus on BUSINESS VALUE, not just technical capabilities.

Example format:
[
  "40% reduction in cloud computing costs for enterprises running large language models - potential $2B market opportunity in Fortune 500",
  "Enables real-time language translation on smartphones without internet - opens offline markets in developing countries worth $500M",
  "Reduces drug discovery time from 5 years to 18 months - could accelerate $50B pharmaceutical pipeline by 3.5x"
]

Return only the JSON array:"""

        return await self._get_llm_response_as_list(prompt, "business_implications")

    async def _analyze_research_quality(self, text: str) -> Optional[int]:
        """Specialized chain for research quality scoring."""

        prompt = f"""
You are a research quality evaluator. Rate this research 0-10 based on:

{text[:2000]}

Scoring criteria:
- Novelty (0-3): How new/innovative are the ideas?
- Methodology (0-3): How rigorous is the experimental approach?
- Impact (0-2): How significant are the results/implications?
- Clarity (0-2): How well-written and clear is the research?

Return ONLY a single integer 0-10, no explanation:"""

        try:
            result = await self._get_llm_response(prompt)
            if result:
                # Extract number from response
                import re
                match = re.search(r'\b([0-9]|10)\b', result)
                if match:
                    return int(match.group(1))
            return None
        except:
            return None

    async def _get_llm_response_as_list(self, prompt: str, analysis_type: str) -> Optional[List[str]]:
        """Get LLM response and parse as JSON list."""

        result = await self._get_llm_response(prompt)
        if not result:
            return None

        try:
            # Try to extract JSON array from response
            start = result.find('[')
            end = result.rfind(']') + 1
            if start != -1 and end > start:
                json_str = result[start:end]
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    print(f"✅ {analysis_type}: extracted {len(parsed)} insights")
                    return parsed

            print(f"⚠️  {analysis_type}: could not parse JSON from LLM response")
            return None

        except json.JSONDecodeError as e:
            print(f"⚠️  {analysis_type}: JSON parse error - {e}")
            return None

    async def _get_llm_response(self, prompt: str) -> Optional[str]:
        """Get response from Ollama API."""

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1000
                    }
                }

                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_text = result.get("response", "")
                        self.tokens_used += len(analysis_text.split())
                        return analysis_text
                    else:
                        print(f"❌ Ollama API error: HTTP {response.status}")
                        return None

        except aiohttp.ClientConnectionError:
            print("❌ Cannot connect to Ollama")
            return None
        except Exception as e:
            print(f"❌ LLM request failed: {e}")
            return None

    async def _handle_analysis_failure(self, text: str, title: str, failure_type: str) -> None:
        """Handle analysis failure by tracking unanalyzed content."""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        failure_file = self.unanalyzed_dir / f"unanalyzed_{failure_type}_{timestamp}.json"

        failure_data = {
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "failure_type": failure_type,
            "model_used": self.model,
            "content_preview": text[:500] + "..." if len(text) > 500 else text,
            "status": "pending_reanalysis"
        }

        with open(failure_file, 'w') as f:
            json.dump(failure_data, f, indent=2)

        print(f"📝 Analysis failure logged: {failure_file}")

        # Update unanalyzed tally
        await self._update_unanalyzed_tally(failure_type)

        return None

    async def _update_unanalyzed_tally(self, failure_type: str):
        """Update running tally of unanalyzed reports."""

        tally_file = self.unanalyzed_dir / "unanalyzed_tally.json"

        # Load existing tally
        if tally_file.exists():
            with open(tally_file, 'r') as f:
                tally = json.load(f)
        else:
            tally = {
                "total_failures": 0,
                "by_type": {},
                "last_updated": None
            }

        # Update counts
        tally["total_failures"] += 1
        tally["by_type"][failure_type] = tally["by_type"].get(failure_type, 0) + 1
        tally["last_updated"] = datetime.now().isoformat()

        # Save updated tally
        with open(tally_file, 'w') as f:
            json.dump(tally, f, indent=2)

        print(f"📊 Unanalyzed tally updated: {tally['total_failures']} total failures")


def main():
    """Test the specialized analyzer."""

    async def test():
        analyzer = SpecializedLLMAnalyzer()

        # Test with sample text
        test_text = """
        We introduce EmbeddingGemma, a new lightweight, open text embedding model
        based on the Gemma 3 language model family. Our innovative training recipe
        strategically captures knowledge from larger models via encoder-decoder
        initialization and geometric embedding distillation. EmbeddingGemma (300M)
        achieves state-of-the-art results and outperforms prior top models with
        fewer than 500M parameters.
        """

        result = await analyzer.analyze_with_specialized_chains(test_text, "EmbeddingGemma Test")

        if result:
            print("🎉 Specialized analysis successful!")
            for key, value in result.items():
                print(f"{key}: {value}")
        else:
            print("❌ Analysis failed - check unanalyzed directory")

    asyncio.run(test())


if __name__ == "__main__":
    main()