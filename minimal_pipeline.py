#!/usr/bin/env python3
"""
Minimal KnowHunt pipeline runner - no dependencies on problematic imports.
"""

import sys
import asyncio
import json
import aiohttp
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import hashlib
import sqlite3
import os
import subprocess
import time
import re


class MinimalArXivCollector:
    """Simplified ArXiv collector."""

    def __init__(self, max_results=10):
        self.max_results = max_results
        self.base_url = "http://export.arxiv.org/api/query"

    async def collect(self, query):
        """Collect papers from ArXiv API."""
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        papers = []

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        papers = self._parse_arxiv_xml(content)
                        print(f"✅ Successfully collected {len(papers)} papers")
                    else:
                        print(f"❌ ArXiv API error: HTTP {response.status}")
                        return []
            except Exception as e:
                print(f"❌ Collection failed: {e}")
                return []

        return papers

    def _parse_arxiv_xml(self, xml_content):
        """Parse ArXiv XML response."""
        papers = []

        try:
            root = ET.fromstring(xml_content)

            # ArXiv uses Atom namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            entries = root.findall('atom:entry', ns)

            for entry in entries:
                paper = {}

                # Title
                title_elem = entry.find('atom:title', ns)
                paper['title'] = title_elem.text.strip() if title_elem is not None else "No title"

                # Summary/Abstract
                summary_elem = entry.find('atom:summary', ns)
                paper['abstract'] = summary_elem.text.strip() if summary_elem is not None else ""

                # Authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None:
                        authors.append(name_elem.text.strip())
                paper['authors'] = authors

                # Published date
                published_elem = entry.find('atom:published', ns)
                paper['published'] = published_elem.text.strip() if published_elem is not None else ""

                # ArXiv ID and link
                id_elem = entry.find('atom:id', ns)
                if id_elem is not None:
                    arxiv_url = id_elem.text.strip()
                    paper['url'] = arxiv_url
                    # Extract ID from URL like http://arxiv.org/abs/2401.12345v1
                    if 'abs/' in arxiv_url:
                        paper['arxiv_id'] = arxiv_url.split('abs/')[-1]

                papers.append(paper)

        except ET.ParseError as e:
            print(f"❌ XML parsing error: {e}")

        return papers


class MinimalHealthChecker:
    """System health checker with automatic repair capabilities."""

    def __init__(self, auto_repair=True):
        self.ollama_url = "http://localhost:11434"
        self.arxiv_url = "http://export.arxiv.org/api/query"
        self.auto_repair = auto_repair

    async def check_all_systems(self):
        """Check all system dependencies with automatic repair."""
        print("🔍 KnowHunt System Health Check")
        if self.auto_repair:
            print("🔧 Auto-repair mode: ENABLED")
        print("=" * 50)

        health_status = {}

        # Check ArXiv API
        print("📚 Checking ArXiv API...")
        health_status['arxiv'] = await self._check_arxiv()

        # Check Ollama service with auto-repair
        print("🤖 Checking Ollama service...")
        health_status['ollama'] = await self._check_and_repair_ollama()

        # Check available models with auto-install
        if health_status['ollama']:
            print("📋 Checking available models...")
            health_status['models'] = await self._check_and_install_models()
        else:
            health_status['models'] = []

        # Summary
        self._print_health_summary(health_status)
        return health_status

    async def _check_arxiv(self):
        """Check ArXiv API accessibility."""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "search_query": "all:test",
                    "start": 0,
                    "max_results": 1
                }

                async with session.get(self.arxiv_url, params=params, timeout=10) as response:
                    if response.status == 200:
                        print("   ✅ ArXiv API: Online")
                        return True
                    else:
                        print(f"   ❌ ArXiv API: HTTP {response.status}")
                        return False

        except asyncio.TimeoutError:
            print("   ❌ ArXiv API: Timeout")
            return False
        except Exception as e:
            print(f"   ❌ ArXiv API: {str(e)}")
            return False

    async def _check_and_repair_ollama(self):
        """Check Ollama service availability with auto-repair."""
        # First, check if Ollama is running
        if await self._is_ollama_running():
            print("   ✅ Ollama: Running")
            return True

        print("   ❌ Ollama: Not running")

        if not self.auto_repair:
            print("   💡 Start with: ollama serve")
            return False

        # Auto-repair: Try to start Ollama
        print("   🔧 Auto-repair: Attempting to start Ollama...")

        if await self._start_ollama():
            print("   ✅ Ollama: Successfully started")
            return True
        else:
            print("   ❌ Ollama: Failed to auto-start")
            print("   💡 Manual start: ollama serve")
            return False

    async def _is_ollama_running(self):
        """Check if Ollama is currently running."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_url}/api/tags", timeout=3) as response:
                    return response.status == 200
        except:
            return False

    async def _start_ollama(self):
        """Attempt to start Ollama service."""
        try:
            # Check if ollama command is available
            result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
            if result.returncode != 0:
                print("   ❌ Ollama not installed - cannot auto-start")
                print("   💡 Install: curl -fsSL https://ollama.ai/install.sh | sh")
                return False

            print("   🚀 Starting Ollama in background...")

            # Start Ollama serve in background
            process = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Detach from parent
            )

            # Wait a bit for startup
            await asyncio.sleep(3)

            # Check if it started successfully
            for attempt in range(5):
                if await self._is_ollama_running():
                    return True
                await asyncio.sleep(1)

            return False

        except Exception as e:
            print(f"   ❌ Failed to start Ollama: {e}")
            return False

    async def _check_and_install_models(self):
        """Check available models and auto-install if needed."""
        models = await self._get_ollama_models()

        if models:
            print(f"   ✅ Available models: {len(models)}")
            for model in models[:3]:  # Show first 3
                print(f"      - {model}")
            if len(models) > 3:
                print(f"      ... and {len(models) - 3} more")
            return models

        print("   ⚠️  No models installed")

        if not self.auto_repair:
            print("   💡 Install with: ollama pull gemma2:2b")
            return []

        # Auto-repair: Install a default lightweight model
        print("   🔧 Auto-repair: Installing default model (gemma2:2b)...")

        if await self._install_model("gemma2:2b"):
            print("   ✅ Successfully installed gemma2:2b")
            return ["gemma2:2b"]
        else:
            print("   ❌ Failed to install default model")
            print("   💡 Manual install: ollama pull gemma2:2b")
            return []

    async def _get_ollama_models(self):
        """Get list of available Ollama models."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model['name'] for model in data.get('models', [])]
                    else:
                        return []
        except Exception:
            return []

    async def _install_model(self, model_name):
        """Install a specific Ollama model."""
        try:
            print(f"   📥 Downloading {model_name} (this may take a few minutes)...")

            # Start the pull command
            process = subprocess.Popen(
                ['ollama', 'pull', model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for completion (with timeout)
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout

                if process.returncode == 0:
                    return True
                else:
                    print(f"   ❌ Model install failed: {stderr}")
                    return False

            except subprocess.TimeoutExpired:
                process.kill()
                print("   ❌ Model install timed out (5 minutes)")
                return False

        except Exception as e:
            print(f"   ❌ Failed to install model: {e}")
            return False

    def _print_health_summary(self, health_status):
        """Print health check summary."""
        print("\n📊 Health Summary")
        print("-" * 20)

        arxiv_status = "🟢 Online" if health_status['arxiv'] else "🔴 Offline"
        ollama_status = "🟢 Running" if health_status['ollama'] else "🔴 Down"
        models_count = len(health_status['models'])

        print(f"ArXiv API:     {arxiv_status}")
        print(f"Ollama:        {ollama_status}")
        print(f"Models:        {models_count} available")

        # Recommendations
        if not health_status['arxiv']:
            print("\n⚠️  ArXiv offline - paper collection will fail")

        if not health_status['ollama']:
            print("\n⚠️  Ollama not running - analysis will fail")
            print("   Start with: ollama serve")

        if health_status['ollama'] and not health_status['models']:
            print("\n⚠️  No models installed")
            print("   Quick start: ollama pull gemma2:2b")

        # Overall status
        all_good = health_status['arxiv'] and health_status['ollama'] and len(health_status['models']) > 0

        if all_good:
            print("\n✅ All systems operational - pipeline ready!")
        else:
            if self.auto_repair:
                print("\n⚠️  Some issues remain after auto-repair attempts")
            else:
                print("\n⚠️  Some issues detected - pipeline may have limited functionality")


class MinimalLLMAnalyzer:
    """Simplified LLM analyzer using Ollama."""

    def __init__(self, model="gemma2:2b"):
        self.model = model
        self.base_url = "http://localhost:11434"
        self.tokens_used = 0

    async def analyze(self, text, title="Research Analysis"):
        """Analyze text using Ollama."""

        prompt = f"""
You are an expert research analyst. Analyze these research papers and extract SPECIFIC, ACTIONABLE insights.

{text[:4000]}

Provide detailed analysis in JSON format. For each category, give SPECIFIC insights, NOT just paper titles:

1. NOVEL CONTRIBUTIONS - What are the actual new ideas, methods, or discoveries? Be specific about what makes them novel.
2. TECHNICAL INNOVATIONS - What specific algorithms, architectures, or technical approaches are introduced? Include performance improvements.
3. BUSINESS IMPLICATIONS - What commercial opportunities, cost savings, or market impacts do these enable? Be concrete about business value.
4. RESEARCH QUALITY SCORE - Rate overall quality 0-10 based on novelty, methodology, and impact.

CRITICAL: Do NOT just list paper titles. Extract the actual insights and innovations from within the papers.

Example good response:
{{
  "novel_contributions": [
    "New attention mechanism reduces computational complexity by 40% while maintaining accuracy",
    "First successful application of quantum error correction to practical NLP tasks",
    "Novel training approach that works with 10x less labeled data"
  ],
  "technical_innovations": [
    "Sparse attention patterns that scale O(n log n) instead of O(n²)",
    "Hybrid quantum-classical architecture achieving 95% gate fidelity",
    "Self-supervised pre-training with contrastive learning on unlabeled text"
  ],
  "business_implications": [
    "40% reduction in training costs for enterprise language models",
    "Enables quantum NLP on today's NISQ hardware, opening new market",
    "Reduces annotation costs by 90%, making AI accessible to smaller companies"
  ],
  "research_quality_score": 8
}}

Now analyze the provided research and return actual insights, not paper titles:"""

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1500
                    }
                }

                print(f"🤖 Sending request to Ollama ({self.model})...")

                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_text = result.get("response", "")
                        self.tokens_used += len(analysis_text.split())

                        print("✅ LLM analysis complete")
                        return self._parse_analysis(analysis_text, title)
                    else:
                        error_text = await response.text()
                        print(f"❌ Ollama API error: HTTP {response.status}")
                        print(f"Response: {error_text}")
                        return None

        except aiohttp.ClientConnectionError:
            print("❌ Cannot connect to Ollama - is it running on localhost:11434?")
            print("Try: ollama serve")
            return None
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return None

    def _parse_analysis(self, analysis_text, title):
        """Parse LLM response into structured format."""

        # Handle markdown code blocks if present
        if '```json' in analysis_text:
            start_marker = analysis_text.find('```json') + 7
            end_marker = analysis_text.find('```', start_marker)
            if end_marker != -1:
                analysis_text = analysis_text[start_marker:end_marker].strip()

        # Extract JSON boundaries
        start = analysis_text.find('{')
        end = analysis_text.rfind('}') + 1

        if start == -1 or end <= start:
            return {
                "document_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "title": title,
                "error": "No JSON found in LLM response",
                "raw_analysis": analysis_text,
                "tokens_used": self.tokens_used
            }

        json_str = analysis_text[start:end].strip()

        # Fix the common issue: comments in JSON values like "8 (Paper 1)"
        json_str = re.sub(r'(\d+(?:\.\d+)?)\s*\([^)]+\)', r'\1', json_str)

        try:
            parsed = json.loads(json_str)

            return {
                "document_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "title": title,
                "research_quality_score": parsed.get("research_quality_score", 5),
                "novel_contributions": parsed.get("novel_contributions", []),
                "technical_innovations": parsed.get("technical_innovations", []),
                "business_implications": parsed.get("business_implications", []),
                "raw_analysis": analysis_text,
                "tokens_used": self.tokens_used
            }

        except json.JSONDecodeError as e:
            return {
                "document_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "title": title,
                "error": f"JSON parsing failed: {e.msg} at position {e.pos}",
                "raw_analysis": analysis_text,
                "tokens_used": self.tokens_used
            }



class MinimalReportStorage:
    """Simplified report storage."""

    def __init__(self):
        self.reports_dir = Path("/home/cordlesssteve/projects/Utility/KnowHunt/reports/minimal_pipeline")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save_report(self, analysis, papers, query, model):
        """Save analysis report."""

        timestamp = datetime.now()
        filename = f"analysis_{query.replace(' ', '_')}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "timestamp": timestamp.isoformat(),
            "query": query,
            "model": model,
            "papers_count": len(papers),
            "analysis": analysis,
            "papers": papers[:5],  # Store first 5 papers
            "tokens_used": analysis.get("tokens_used", 0)
        }

        report_path = self.reports_dir / filename

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"💾 Report saved: {filename}")
        return report_path


async def run_minimal_pipeline(query, max_results=10, model="gemma2:2b"):
    """Run the minimal pipeline."""

    print(f"🚀 Minimal KnowHunt Pipeline")
    print(f"🔍 Query: {query}")
    print(f"📊 Papers: {max_results} | Model: {model}")
    print("=" * 60)

    # Step 0: Health Check with Auto-Repair
    print("🔍 Running system health check with auto-repair...")
    health_checker = MinimalHealthChecker(auto_repair=True)
    health_status = await health_checker.check_all_systems()

    # Decide whether to proceed based on health
    if not health_status['arxiv']:
        print("\n❌ ArXiv is offline - cannot collect papers")
        return

    if not health_status['ollama']:
        print("\n⚠️  Ollama is not running - continuing with collection only")
        proceed_with_analysis = False
    elif model not in health_status['models']:
        print(f"\n⚠️  Model '{model}' not found - continuing with collection only")
        print(f"   Available models: {', '.join(health_status['models'][:3])}")
        proceed_with_analysis = False
    else:
        print(f"\n✅ Model '{model}' is available - full pipeline enabled")
        proceed_with_analysis = True

    print("=" * 60)

    # Step 1: Collect papers
    print("📚 Collecting from ArXiv...")
    collector = MinimalArXivCollector(max_results)
    papers = await collector.collect(query)

    if not papers:
        print("❌ No papers collected - stopping pipeline")
        return

    # Display collected papers
    print(f"\n📖 Papers found:")
    for i, paper in enumerate(papers, 1):
        print(f"  {i}. {paper['title'][:80]}...")
        if paper['authors']:
            authors_str = ", ".join(paper['authors'][:3])
            print(f"     Authors: {authors_str}")

    # Step 2: Prepare content for analysis
    print(f"\n🧠 Preparing content for analysis...")

    # Combine abstracts from top papers
    analysis_content = []
    for i, paper in enumerate(papers[:3], 1):  # Analyze top 3 papers
        content = f"Paper {i}:\nTitle: {paper['title']}\nAbstract: {paper['abstract']}\n"
        analysis_content.append(content)

    combined_text = "\n".join(analysis_content)

    # Step 3: LLM Analysis (if available)
    if proceed_with_analysis:
        print(f"🤖 Running LLM analysis...")
        analyzer = MinimalLLMAnalyzer(model)
        analysis = await analyzer.analyze(combined_text, f"Research Analysis: {query}")

        if not analysis:
            print("❌ Analysis failed - but papers were collected successfully")
            # Still save the papers
            storage = MinimalReportStorage()
            storage.save_report({"error": "LLM analysis failed"}, papers, query, model)
            return
    else:
        print("⚠️  Skipping LLM analysis (service unavailable)")
        # Create basic analysis placeholder
        analysis = {
            "document_id": f"collection_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": f"Paper Collection: {query}",
            "research_quality_score": 0,
            "novel_contributions": ["LLM analysis not available"],
            "technical_innovations": ["LLM analysis not available"],
            "business_implications": ["LLM analysis not available"],
            "raw_analysis": "Collection only - LLM analysis skipped due to service unavailability",
            "tokens_used": 0
        }

    # Step 4: Display results
    print(f"\n📊 Analysis Results:")
    if 'error' in analysis:
        print(f"   ❌ Error: {analysis['error']}")
        print(f"   Raw response preserved for inspection")
    else:
        print(f"   Quality Score: {analysis.get('research_quality_score', 'N/A')}/10")
        print(f"   Novel Contributions: {len(analysis.get('novel_contributions', []))}")
        print(f"   Technical Innovations: {len(analysis.get('technical_innovations', []))}")
        print(f"   Business Implications: {len(analysis.get('business_implications', []))}")
    print(f"   Tokens Used: {analysis.get('tokens_used', 0)}")

    # Show some results
    if 'error' not in analysis and analysis.get('novel_contributions'):
        print(f"\n🔬 Top Novel Contributions:")
        for i, contrib in enumerate(analysis['novel_contributions'][:2], 1):
            print(f"  {i}. {contrib}")

    if 'error' not in analysis and analysis.get('business_implications'):
        print(f"\n💼 Business Implications:")
        for i, impl in enumerate(analysis['business_implications'][:2], 1):
            print(f"  {i}. {impl}")

    # Step 5: Save report
    print(f"\n💾 Saving report...")
    storage = MinimalReportStorage()
    report_path = storage.save_report(analysis, papers, query, model)

    print(f"\n✅ Pipeline Complete!")
    print(f"📁 Report saved to: {report_path}")
    if 'error' not in analysis:
        print(f"📊 Summary: {len(papers)} papers → Quality {analysis.get('research_quality_score', 'N/A')}/10")
    else:
        print(f"📊 Summary: {len(papers)} papers → Analysis failed (see raw_analysis in report)")


async def run_health_check_only():
    """Run only the health check with auto-repair."""
    health_checker = MinimalHealthChecker(auto_repair=True)
    await health_checker.check_all_systems()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 minimal_pipeline.py \"<query>\" [max_results] [model]")
        print("       python3 minimal_pipeline.py health")
        print("Example: python3 minimal_pipeline.py \"AI agents\" 10 \"gemma2:2b\"")
        print("         python3 minimal_pipeline.py health")
        sys.exit(1)

    # Special case for health check only
    if sys.argv[1].lower() == "health":
        asyncio.run(run_health_check_only())
        return

    query = sys.argv[1]
    max_results = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    model = sys.argv[3] if len(sys.argv) > 3 else "gemma2:2b"

    # Run the minimal pipeline
    asyncio.run(run_minimal_pipeline(query, max_results, model))


if __name__ == "__main__":
    main()