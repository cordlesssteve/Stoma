#!/usr/bin/env python3
"""
Production Pipeline Integration for KnowHunt

This module integrates LLM analysis into the main collection pipeline,
providing a complete end-to-end system that automatically collects,
enriches, analyzes, and reports on research papers.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time

# Import all our components
from minimal_pipeline import (
    MinimalArXivCollector,
    MinimalLLMAnalyzer,
    MinimalHealthChecker,
    MinimalReportStorage
)
from advanced_model_manager import AdvancedModelManager
from cross_paper_synthesizer import CrossPaperSynthesizer, ResearchPaper
from batch_processor import BatchProcessor, BatchProcessingConfig

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Configuration for production pipeline."""
    # Collection settings
    queries: List[str]
    papers_per_query: int = 20
    collection_interval_hours: int = 12

    # Analysis settings
    enable_llm_analysis: bool = True
    enable_synthesis: bool = True
    synthesis_interval_hours: int = 24

    # Model settings
    preferred_model: Optional[str] = None
    fallback_model: str = "qwen2.5-coder:3b"
    auto_model_selection: bool = True

    # Storage settings
    storage_path: str = "./reports/production"
    backup_enabled: bool = True

    # Performance settings
    max_concurrent_jobs: int = 3
    batch_size: int = 50
    enable_overnight_processing: bool = True

    # Monitoring settings
    health_check_interval_minutes: int = 30
    alert_on_failure: bool = True
    max_retry_attempts: int = 3


class ProductionPipelineManager:
    """Manages the production pipeline with automated scheduling."""

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.storage_path = Path(config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.health_checker = MinimalHealthChecker(auto_repair=True)
        self.model_manager = AdvancedModelManager()
        self.batch_processor = BatchProcessor(BatchProcessingConfig(
            max_concurrent_jobs=config.max_concurrent_jobs,
            storage_path=str(self.storage_path / "batch_jobs")
        ))

        # State tracking
        self.is_running = False
        self.last_collection = None
        self.last_synthesis = None
        self.failed_attempts = 0

        # Statistics
        self.stats = {
            "collections_run": 0,
            "papers_analyzed": 0,
            "synthesis_runs": 0,
            "total_reports": 0,
            "last_health_check": None,
            "system_status": "unknown"
        }

    async def initialize_system(self) -> bool:
        """Initialize and validate the production system."""
        logger.info("Initializing production pipeline system...")

        # Run comprehensive health check
        health_status = await self.health_checker.check_all_systems()

        if not health_status['arxiv']:
            logger.error("ArXiv API is not available - cannot run production pipeline")
            return False

        if not health_status['ollama'] or not health_status['models']:
            logger.error("Ollama or models not available")

            if self.config.auto_model_selection:
                logger.info("Attempting to set up recommended models...")
                installed_models = await self.model_manager.setup_recommended_models(
                    max_models=2, include_advanced=False
                )
                if installed_models:
                    logger.info(f"Successfully installed {len(installed_models)} models")
                else:
                    logger.error("Failed to install any models")
                    return False
            else:
                return False

        # Update stats
        self.stats["last_health_check"] = datetime.now().isoformat()
        self.stats["system_status"] = "operational"

        logger.info("Production pipeline system initialized successfully")
        return True

    async def run_collection_cycle(self) -> Dict[str, Any]:
        """Run a complete collection and analysis cycle."""
        cycle_start = datetime.now()
        logger.info(f"Starting collection cycle at {cycle_start}")

        results = {
            "cycle_id": f"collection_{cycle_start.strftime('%Y%m%d_%H%M%S')}",
            "started_at": cycle_start.isoformat(),
            "queries_processed": 0,
            "papers_collected": 0,
            "papers_analyzed": 0,
            "reports_generated": 0,
            "errors": [],
            "completed_at": None
        }

        all_papers = []

        # Collect papers for each query
        collector = MinimalArXivCollector(max_results=self.config.papers_per_query)

        for query in self.config.queries:
            try:
                logger.info(f"Collecting papers for query: {query}")
                papers = await collector.collect(query)

                if papers:
                    # Convert to ResearchPaper objects for analysis
                    query_papers = []
                    for i, paper_data in enumerate(papers):
                        paper = ResearchPaper(
                            paper_id=f"{query.replace(' ', '_')}_{i}_{int(time.time())}",
                            title=paper_data['title'],
                            abstract=paper_data.get('abstract', ''),
                            authors=paper_data.get('authors', []),
                            published_date=paper_data.get('published', ''),
                            arxiv_id=paper_data.get('arxiv_id', ''),
                            url=paper_data.get('url', ''),
                            domain=query.replace(' ', '_'),
                            keywords=[],
                            analysis_result=None
                        )
                        query_papers.append(paper)

                    all_papers.extend(query_papers)
                    results["papers_collected"] += len(papers)
                    logger.info(f"Collected {len(papers)} papers for '{query}'")

                results["queries_processed"] += 1

            except Exception as e:
                error_msg = f"Failed to collect papers for query '{query}': {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        # Run LLM analysis on collected papers if enabled
        if self.config.enable_llm_analysis and all_papers:
            logger.info("Running LLM analysis on collected papers...")

            try:
                analyzed_papers = await self._analyze_papers_batch(all_papers)
                results["papers_analyzed"] = len(analyzed_papers)

                # Generate individual reports
                for paper in analyzed_papers:
                    if paper.analysis_result:
                        report_path = await self._save_paper_report(paper, cycle_start)
                        if report_path:
                            results["reports_generated"] += 1

                # Update all_papers with analysis results
                all_papers = analyzed_papers

            except Exception as e:
                error_msg = f"LLM analysis failed: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        # Save collection summary
        summary_path = self.storage_path / f"collection_summary_{cycle_start.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "cycle_results": results,
                "collected_papers": [
                    {
                        "paper_id": p.paper_id,
                        "title": p.title,
                        "domain": p.domain,
                        "has_analysis": p.analysis_result is not None
                    }
                    for p in all_papers
                ],
                "system_stats": self.stats
            }, f, indent=2, default=str)

        results["completed_at"] = datetime.now().isoformat()
        self.last_collection = datetime.now()

        # Update global stats
        self.stats["collections_run"] += 1
        self.stats["papers_analyzed"] += results["papers_analyzed"]
        self.stats["total_reports"] += results["reports_generated"]

        logger.info(f"Collection cycle complete: {results['papers_collected']} collected, {results['papers_analyzed']} analyzed")
        return results

    async def run_synthesis_cycle(self) -> Dict[str, Any]:
        """Run cross-paper synthesis on recent collections."""
        synthesis_start = datetime.now()
        logger.info(f"Starting synthesis cycle at {synthesis_start}")

        # Find recent collection reports
        cutoff_time = synthesis_start - timedelta(hours=self.config.synthesis_interval_hours * 2)
        recent_papers = await self._load_recent_papers(cutoff_time)

        if len(recent_papers) < 5:
            logger.warning(f"Not enough recent papers for synthesis: {len(recent_papers)}")
            return {
                "synthesis_id": f"synthesis_{synthesis_start.strftime('%Y%m%d_%H%M%S')}",
                "status": "skipped",
                "reason": "insufficient_papers",
                "papers_found": len(recent_papers)
            }

        # Select optimal model for synthesis
        optimal_model = self.model_manager.select_model_for_task(
            task_type="synthesis",
            content_length=sum(len(p.abstract) for p in recent_papers),
            performance_preference="quality"
        )

        logger.info(f"Running synthesis on {len(recent_papers)} papers with model {optimal_model}")

        try:
            # Run synthesis
            synthesizer = CrossPaperSynthesizer(model=optimal_model)
            synthesis_result = await synthesizer.synthesize_papers(recent_papers)

            # Save synthesis report
            synthesis_path = await self._save_synthesis_report(synthesis_result, synthesis_start)

            result = {
                "synthesis_id": synthesis_result.synthesis_id,
                "status": "completed",
                "started_at": synthesis_start.isoformat(),
                "completed_at": datetime.now().isoformat(),
                "papers_analyzed": len(recent_papers),
                "trends_found": len(synthesis_result.cross_domain_trends),
                "quality_score": synthesis_result.synthesis_quality_score,
                "model_used": optimal_model,
                "report_path": str(synthesis_path)
            }

            self.last_synthesis = datetime.now()
            self.stats["synthesis_runs"] += 1

            logger.info(f"Synthesis complete: {result['trends_found']} trends, quality {result['quality_score']:.2f}")
            return result

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                "synthesis_id": f"synthesis_{synthesis_start.strftime('%Y%m%d_%H%M%S')}",
                "status": "failed",
                "error": str(e),
                "papers_analyzed": len(recent_papers)
            }

    async def _analyze_papers_batch(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Analyze papers in batches using LLM."""
        analyzed_papers = []
        batch_size = 5  # Analyze 5 papers at a time to avoid overwhelming the model

        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]

            # Select optimal model for this batch
            total_content = sum(len(p.title) + len(p.abstract) for p in batch)
            optimal_model = self.model_manager.select_model_for_task(
                task_type="research_analysis",
                content_length=total_content,
                performance_preference="balanced"
            )

            logger.info(f"Analyzing batch {i//batch_size + 1} with model {optimal_model}")

            analyzer = MinimalLLMAnalyzer(optimal_model)

            for paper in batch:
                try:
                    # Prepare analysis content
                    analysis_content = f"Title: {paper.title}\nAbstract: {paper.abstract}"

                    # Run analysis
                    analysis_result = await analyzer.analyze(
                        analysis_content,
                        f"Paper Analysis: {paper.paper_id}"
                    )

                    if analysis_result:
                        paper.analysis_result = analysis_result
                        logger.debug(f"Successfully analyzed paper: {paper.paper_id}")
                    else:
                        logger.warning(f"Analysis failed for paper: {paper.paper_id}")

                    analyzed_papers.append(paper)

                    # Small delay between papers to avoid overwhelming the model
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error analyzing paper {paper.paper_id}: {e}")
                    analyzed_papers.append(paper)  # Add without analysis

            # Delay between batches
            if i + batch_size < len(papers):
                logger.info(f"Completed batch {i//batch_size + 1}, waiting before next batch...")
                await asyncio.sleep(2)

        return analyzed_papers

    async def _save_paper_report(self, paper: ResearchPaper, cycle_time: datetime) -> Optional[Path]:
        """Save individual paper analysis report."""
        if not paper.analysis_result:
            return None

        try:
            report_path = self.storage_path / "paper_reports" / f"{paper.paper_id}_{cycle_time.strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)

            report_data = {
                "paper_info": {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "published_date": paper.published_date,
                    "arxiv_id": paper.arxiv_id,
                    "url": paper.url,
                    "domain": paper.domain
                },
                "analysis_result": paper.analysis_result,
                "analysis_timestamp": cycle_time.isoformat(),
                "metadata": {
                    "collection_cycle": cycle_time.strftime('%Y%m%d_%H%M%S'),
                    "abstract_length": len(paper.abstract),
                    "title_length": len(paper.title)
                }
            }

            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            return report_path

        except Exception as e:
            logger.error(f"Failed to save paper report for {paper.paper_id}: {e}")
            return None

    async def _save_synthesis_report(self, synthesis_result, synthesis_time: datetime) -> Path:
        """Save synthesis analysis report."""
        report_path = self.storage_path / "synthesis_reports" / f"{synthesis_result.synthesis_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert synthesis result to JSON-serializable format
        report_data = {
            "synthesis_info": {
                "synthesis_id": synthesis_result.synthesis_id,
                "query_domains": synthesis_result.query_domains,
                "papers_analyzed": synthesis_result.papers_analyzed,
                "synthesis_timestamp": synthesis_result.synthesis_timestamp.isoformat(),
                "synthesis_quality_score": synthesis_result.synthesis_quality_score
            },
            "analysis_results": {
                "cross_domain_trends": [
                    {
                        "trend_name": trend.trend_name,
                        "description": trend.description,
                        "confidence_score": trend.confidence_score,
                        "trend_type": trend.trend_type,
                        "supporting_papers": len(trend.supporting_papers)
                    }
                    for trend in synthesis_result.cross_domain_trends
                ],
                "common_techniques": synthesis_result.common_techniques,
                "emerging_themes": synthesis_result.emerging_themes,
                "research_gaps": synthesis_result.research_gaps,
                "recommendations": synthesis_result.recommendations
            },
            "metadata": {
                "limitations": synthesis_result.limitations,
                "confidence_intervals": synthesis_result.confidence_intervals,
                "collaboration_patterns": synthesis_result.collaboration_patterns
            }
        }

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        return report_path

    async def _load_recent_papers(self, cutoff_time: datetime) -> List[ResearchPaper]:
        """Load recent papers for synthesis."""
        papers = []
        reports_dir = self.storage_path / "paper_reports"

        if not reports_dir.exists():
            return papers

        for report_file in reports_dir.glob("*.json"):
            try:
                # Check file modification time
                if datetime.fromtimestamp(report_file.stat().st_mtime) < cutoff_time:
                    continue

                with open(report_file) as f:
                    report_data = json.load(f)

                paper_info = report_data.get("paper_info", {})
                analysis_result = report_data.get("analysis_result")

                if paper_info and analysis_result:
                    paper = ResearchPaper(
                        paper_id=paper_info["paper_id"],
                        title=paper_info["title"],
                        abstract="",  # Not saved in report
                        authors=paper_info.get("authors", []),
                        published_date=paper_info.get("published_date", ""),
                        arxiv_id=paper_info.get("arxiv_id", ""),
                        url=paper_info.get("url", ""),
                        domain=paper_info.get("domain", ""),
                        keywords=[],
                        analysis_result=analysis_result
                    )
                    papers.append(paper)

            except Exception as e:
                logger.warning(f"Failed to load report {report_file}: {e}")

        return papers

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "pipeline_status": {
                "is_running": self.is_running,
                "last_collection": self.last_collection.isoformat() if self.last_collection else None,
                "last_synthesis": self.last_synthesis.isoformat() if self.last_synthesis else None,
                "failed_attempts": self.failed_attempts
            },
            "configuration": {
                "queries": self.config.queries,
                "papers_per_query": self.config.papers_per_query,
                "llm_analysis_enabled": self.config.enable_llm_analysis,
                "synthesis_enabled": self.config.enable_synthesis,
                "auto_model_selection": self.config.auto_model_selection
            },
            "statistics": self.stats,
            "storage": {
                "storage_path": str(self.storage_path),
                "disk_usage_gb": sum(f.stat().st_size for f in self.storage_path.rglob('*') if f.is_file()) / (1024**3)
            }
        }

    async def run_continuous_pipeline(self):
        """Run the pipeline continuously with scheduled intervals."""
        logger.info("Starting continuous production pipeline...")
        self.is_running = True

        # Initialize system
        if not await self.initialize_system():
            logger.error("Failed to initialize system - stopping pipeline")
            return

        try:
            while self.is_running:
                # Run collection cycle
                try:
                    await self.run_collection_cycle()
                    self.failed_attempts = 0  # Reset on success

                except Exception as e:
                    self.failed_attempts += 1
                    logger.error(f"Collection cycle failed (attempt {self.failed_attempts}): {e}")

                    if self.failed_attempts >= self.config.max_retry_attempts:
                        logger.error("Max retry attempts reached - stopping pipeline")
                        break

                # Run synthesis if enough time has passed
                if (self.config.enable_synthesis and
                    (not self.last_synthesis or
                     datetime.now() - self.last_synthesis > timedelta(hours=self.config.synthesis_interval_hours))):

                    try:
                        await self.run_synthesis_cycle()
                    except Exception as e:
                        logger.error(f"Synthesis cycle failed: {e}")

                # Wait for next collection cycle
                logger.info(f"Waiting {self.config.collection_interval_hours} hours until next collection cycle...")
                await asyncio.sleep(self.config.collection_interval_hours * 3600)

        finally:
            self.is_running = False
            logger.info("Production pipeline stopped")


# Default production configuration
DEFAULT_PRODUCTION_QUERIES = [
    "large language models",
    "computer vision",
    "reinforcement learning",
    "neural architecture search",
    "federated learning",
    "graph neural networks",
    "transformer architectures",
    "multimodal learning",
    "AI safety alignment",
    "quantum machine learning"
]


async def main():
    """Main entry point for production pipeline."""
    import sys

    if len(sys.argv) < 2:
        print("KnowHunt Production Pipeline")
        print("Usage: python3 production_pipeline.py <command> [options]")
        print("\nCommands:")
        print("  start           - Start continuous production pipeline")
        print("  collect         - Run single collection cycle")
        print("  synthesize      - Run single synthesis cycle")
        print("  status          - Show system status")
        print("  init            - Initialize system and check health")
        print("\nExamples:")
        print("  python3 production_pipeline.py start")
        print("  python3 production_pipeline.py collect")
        return

    # Create production configuration
    config = ProductionConfig(
        queries=DEFAULT_PRODUCTION_QUERIES,
        papers_per_query=15,
        collection_interval_hours=12,
        enable_llm_analysis=True,
        enable_synthesis=True,
        synthesis_interval_hours=24,
        auto_model_selection=True,
        storage_path=os.getenv("PRODUCTION_STORAGE_PATH", "./reports/production"),
        max_concurrent_jobs=int(os.getenv("MAX_CONCURRENT_JOBS", "3")),
        enable_overnight_processing=True
    )

    pipeline = ProductionPipelineManager(config)
    command = sys.argv[1].lower()

    if command == "init":
        print("🔧 Initializing production system...")
        success = await pipeline.initialize_system()
        if success:
            print("✅ System initialization complete")
            pipeline.model_manager.print_system_analysis()
        else:
            print("❌ System initialization failed")

    elif command == "collect":
        print("📚 Running single collection cycle...")
        results = await pipeline.run_collection_cycle()
        print(f"✅ Collection complete: {results['papers_collected']} papers collected, {results['papers_analyzed']} analyzed")

    elif command == "synthesize":
        print("🔬 Running single synthesis cycle...")
        results = await pipeline.run_synthesis_cycle()
        print(f"✅ Synthesis complete: Status = {results['status']}")
        if results['status'] == 'completed':
            print(f"   Trends found: {results['trends_found']}")
            print(f"   Quality score: {results['quality_score']:.2f}")

    elif command == "status":
        print("📊 System Status:")
        status = pipeline.get_system_status()
        print(json.dumps(status, indent=2, default=str))

    elif command == "start":
        print("🚀 Starting continuous production pipeline...")
        print(f"   Queries: {len(config.queries)} research areas")
        print(f"   Collection interval: {config.collection_interval_hours} hours")
        print(f"   LLM analysis: {'Enabled' if config.enable_llm_analysis else 'Disabled'}")
        print(f"   Synthesis: {'Enabled' if config.enable_synthesis else 'Disabled'}")

        try:
            await pipeline.run_continuous_pipeline()
        except KeyboardInterrupt:
            print("\n🛑 Pipeline stopped by user")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./production_pipeline.log'),
            logging.StreamHandler()
        ]
    )

    asyncio.run(main())